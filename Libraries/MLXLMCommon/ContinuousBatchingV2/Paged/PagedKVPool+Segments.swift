import Foundation
import MLX

extension PagedKVPool {
    func bindAdmission(_ admission: AdmissionV2) {
        precondition(segmentGrant != nil && physicalLease == nil)
        memoryAdmission = admission
        physicalLease = admission.bindBackendPhysicalFloor(initialBytes: bytesMaterialized)
    }

    struct SegmentGrowth {
        let group: PagedKVGroup
        let plan: PagedKVGroup.GrowthPlan
    }

    /// Build complete group plans using one aggregate grant. This stage may
    /// allocate host metadata, but creates no GPU buffers or visible pages.
    private func planSegments(
        additional needs: [PagedKVGroupKey: Int] = [:], eager: Bool = false,
        grant: PagedKVGrant.Snapshot
    ) throws -> [SegmentGrowth] {
        var logical = bytesReserved
        let reuseCeiling = max(grant.bytes, bytesMaterialized)
        for (key, pages) in needs {
            guard pages >= 0 else {
                throw CBv2KVError.backendIneligible(reason: "negative paged reservation")
            }
            let (extra, multiplyOverflow) = pages.multipliedReportingOverflow(
                by: group(key).pageBytes)
            let (total, addOverflow) = logical.addingReportingOverflow(extra)
            guard !multiplyOverflow, !addOverflow, total <= reuseCeiling else {
                throw CBv2KVError.capacityExhausted(
                    needed: multiplyOverflow || addOverflow ? Int.max : extra,
                    available: max(0, reuseCeiling - logical))
            }
            logical = total
        }
        var result: [SegmentGrowth] = []
        for key in groupKeys {
            let g = group(key)
            var target = g.pagesReserved + needs[key, default: 0]
            if eager {
                let product = UInt64(grant.bytes).multipliedFullWidth(
                    by: UInt64(groupDemandBytes[key]!))
                let bytes = UInt64(totalDemandBytes).dividingFullWidth(product).quotient
                let pages = Int(bytes) / g.pageBytes
                target = max(target, g.segmentLayout!.usablePages(fittingPhysicalPages: pages))
            }
            result.append(SegmentGrowth(group: g, plan: try g.planGrowth(usablePages: target)))
        }
        return result
    }

    private func physicalBytes(_ plans: [SegmentGrowth]) throws -> Int {
        var sum = 0
        for item in plans {
            let (next, overflow) = sum.addingReportingOverflow(item.plan.physicalBytes)
            guard !overflow else {
                throw CBv2KVError.backendIneligible(reason: "paged physical byte overflow")
            }
            sum = next
        }
        return sum
    }

    private func requestedPages(
        tokens: some Sequence<Int>, layerKinds: [CBv2LayerKind]
    ) -> [PagedKVGroupKey: Int]? {
        let residency = CBv2PagedKVResidency(config: config)
        var requested: [PagedKVGroupKey: Int] = [:]
        for count in tokens where count > 0 {
            for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
                guard let rows = residency.residentRows(layer: kind, tokens: count) else {
                    return nil
                }
                let key = groupKey(forLayer: index)
                let (total, overflow) = requested[key, default: 0].addingReportingOverflow(
                    rows / config.pageSize)
                guard !overflow else { return nil }
                requested[key] = total
            }
        }
        return requested
    }

    /// Any-thread submit probe over immutable configuration. A request that
    /// fits nominal pages but cannot fit its minimum poison overhead must not
    /// enter an endless allocate/preempt loop. This allocates no page map.
    func minimumSegmentedOverhead(tokens: Int, layerKinds: [CBv2LayerKind]) -> Int? {
        guard segmentGrant != nil,
            let pages = requestedPages(tokens: [tokens], layerKinds: layerKinds)
        else { return nil }
        var overhead = 0
        for (key, count) in pages {
            // Geometry was checked at construction; the token-dependent
            // products below still use checked arithmetic.
            guard let tokenBytes = try? key.bytesPerToken(),
                let pageBytes = try? PagedKVQuantizationConfig.multiply(
                    tokenBytes, config.pageSize),
                let layout = try? PagedKVSegmentLayout(
                    pageBytes: pageBytes,
                    targetBytes: config.segmentSizeBytes ?? PagedKVSegmentLayout.defaultTargetBytes,
                    maximumBufferBytes: config.maxBufferLength,
                    maximumAddressPages: Int(Int32.max) / config.pageSize),
                let physical = layout.allocationBytes(addingUsablePages: count)
            else { return nil }
            let (nominal, multiplyOverflow) = count.multipliedReportingOverflow(by: pageBytes)
            guard !multiplyOverflow, physical >= nominal else { return nil }
            let (next, addOverflow) = overhead.addingReportingOverflow(physical - nominal)
            guard !addOverflow else { return nil }
            overhead = next
        }
        return overhead
    }

    /// Engine-queue-only deadline probe. Charge projected row promises and
    /// retain existing backing; a projected release does not promise that free
    /// segments have retired. No GPU work or grant-sized metadata allocation.
    func projectedPhysicalBytes(
        reservedTokens: [CBv2RequestID: Int], layerKinds: [CBv2LayerKind]
    ) -> Int? {
        guard segmentGrant != nil,
            let requested = requestedPages(tokens: reservedTokens.values, layerKinds: layerKinds)
        else { return nil }
        var physical = bytesMaterialized
        for (key, pages) in requested {
            let group = group(key)
            let extra = max(0, pages - group.committedUsablePages)
            guard let additional = group.segmentLayout?.allocationBytes(addingUsablePages: extra)
            else { return nil }
            let (next, overflow) = physical.addingReportingOverflow(additional)
            guard !overflow else { return nil }
            physical = next
        }
        // Existing debt can be reused; new physical growth must fit the grant.
        return physical <= max(segmentGrant!.snapshot().bytes, bytesMaterialized) ? physical : nil
    }

    /// Reserve logical pages against the exact physical growth plan. A later
    /// commitment rechecks the grant before publishing any native buffers.
    func reserveSegments(_ needs: [PagedKVGroupKey: Int]) throws {
        // Pure planning creates private host layout maps but no native data.
        // Keep its real allowance through the inner scope's last map alias.
        let hostMetadata = try nativePrefixHostMetadataPreparation(additional: needs)
        try withExtendedLifetime(hostMetadata) {
            let grant = segmentGrant!.snapshot()
            let plans: [SegmentGrowth]
            do { plans = try planSegments(additional: needs, grant: grant) } catch {
                if let error = error as? CBv2KVError, case .capacityExhausted = error {
                    PagedKVStorageTelemetry.increment(&storageTelemetry.grantRefusals)
                }
                throw error
            }
            let physical = try physicalBytes(plans)
            let accepted = segmentGrant!.publish(
                expected: grant, physicalBytes: physical, existingPhysicalBytes: bytesMaterialized
            ) {
                for (key, pages) in needs { group(key).pagesReserved += pages }
            }
            storageTelemetry.record(accepted)
            guard accepted == .installed else {
                throw CBv2KVError.capacityExhausted(
                    needed: physical, available: segmentGrant!.snapshot().bytes)
            }
        }
    }

    func materializeSegments(all: Bool) throws {
        if let binding = nativeModelBinding {
            try materializeIssuedNativeSegments(all: all, binding: binding)
            return
        }
        let grant = segmentGrant!.snapshot()
        let plans = try planSegments(eager: all, grant: grant)
        // Existing promises survive a shrink. With no allocation to publish,
        // their committed backing remains valid even above the new grant.
        guard plans.contains(where: { !$0.plan.segmentIDs.isEmpty }) else { return }
        let physical = try physicalBytes(plans)
        guard physical <= grant.bytes else {
            PagedKVStorageTelemetry.increment(&storageTelemetry.grantRefusals)
            throw CBv2KVError.capacityExhausted(needed: physical, available: grant.bytes)
        }
        let previousPhysicalBytes = bytesMaterialized
        // The plan contains all old backing plus every private new segment.
        // Admission charges the peak before the first native allocation.
        do { try physicalLease?.resize(to: physical) } catch {
            PagedKVStorageTelemetry.increment(&storageTelemetry.admissionRefusals)
            throw error
        }
        var prepared: [(PagedKVGroup, PagedKVGroup.PreparedGrowth)] = []
        var preparing = true
        do {
            for item in plans where !item.plan.segmentIDs.isEmpty {
                prepared.append(
                    (
                        item.group,
                        try item.group.prepareGrowth(
                            item.plan, evaluate: slabEval, admission: memoryAdmission)
                    ))
            }
            preparing = false
            let actual = prepared.reduce(bytesMaterialized) { total, entry in
                total - entry.0.committedSegmentBytes
                    + entry.1.segments.values.reduce(0) { $0 + $1.allocatedBytes }
            }
            let accepted = segmentGrant!.publish(expected: grant, physicalBytes: actual) {
                for (group, replacement) in prepared { group.installGrowth(replacement) }
            }
            storageTelemetry.record(accepted)
            guard accepted == .installed else {
                throw CBv2KVError.capacityExhausted(
                    needed: actual, available: segmentGrant!.snapshot().bytes)
            }
            physicalLease?.release(to: actual)
            storageTelemetry.recordSettlement(bound: physical, actual: actual)
        } catch {
            if preparing { PagedKVStorageTelemetry.increment(&storageTelemetry.allocationFailures) }
            // Failed preparation has already destroyed its local arrays. Drop
            // all earlier groups before refunding the private-allocation peak.
            prepared.removeAll()
            physicalLease?.release(to: previousPhysicalBytes)
            throw error
        }
    }

    /// Capture on the engine queue, then hand the immutable value to gauges.
    var segmentStorageSnapshot: PagedKVStorageSnapshot? {
        guard let segmentGrant else { return nil }
        let capturedAt = storageTelemetry.capture()
        let accounting = physicalLease?.accountingSnapshot
        let committed = groups.values.reduce(0) { $0 + $1.committedSegmentBytes }
        // Native recent generations have separate transient owners and do not
        // occupy these segments. Keep every page gauge inside the same backing.
        let livePages = groups.values.reduce(0) { $0 + $1.pagesInUse * $1.pageBytes }
        let poison = groups.values.reduce(0) { $0 + $1.segments.count * $1.pageBytes }
        let logical = groups.values.reduce(0) { $0 + $1.committedLogicalBytes }
        return PagedKVStorageSnapshot(
            generation: storageTelemetry.generation,
            captureSequence: storageTelemetry.captureSequence,
            capturedUptimeNanoseconds: capturedAt,
            grantBytes: segmentGrant.snapshot().bytes, committedBytes: committed,
            reservedPageBytes: bytesReserved, livePageBytes: livePages,
            poisonBytes: poison, slackBytes: max(0, logical - poison - bytesReserved),
            allocatorPaddingBytes: max(0, committed - logical),
            lastAllocationAllowanceBytes: storageTelemetry.lastAllocationAllowanceBytes,
            segmentCount: groups.values.reduce(0) { $0 + $1.segments.count },
            addressPages: groups.values.reduce(0) { $0 + $1.pageCount },
            nominalKVBytes: accounting?.nominalKVBytes,
            physicalFloorOverheadBytes: accounting?.physicalFloorOverheadBytes,
            allocationFailures: storageTelemetry.allocationFailures,
            admissionRefusals: storageTelemetry.admissionRefusals,
            grantRefusals: storageTelemetry.grantRefusals,
            grantEpochRetries: storageTelemetry.grantEpochRetries)
    }
}

extension PagedKVPool {
    /// Checked scalar upper bound before any replacement map is constructed.
    /// A fresh segment has at most one poison page per usable page. Reused
    /// address ranges only reduce this bound. Soft grant changes are NOT a
    /// new immutable capacity limit, and shrink never refunds retained maps.
    func nativePrefixHostMetadataPreparation(additional: [PagedKVGroupKey: Int] = [:])
        throws -> CBv2NativePagedHostMetadataGeneration?
    {
        guard let binding = nativeModelBinding, binding.completePrefixIdentity != nil else {
            return nil
        }
        var addresses = 0
        var grows = false
        for group in groups.values {
            guard additional[group.key, default: 0] >= 0,
                let promised = CBv2KVGeometry.add(
                    group.pagesReserved, additional[group.key, default: 0])
            else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            let missing = max(0, promised - group.committedUsablePages)
            grows = grows || missing > 0
            guard let extra = CBv2KVGeometry.multiply(missing, 2),
                let total = CBv2KVGeometry.add(group.pageCount, extra),
                let next = CBv2KVGeometry.add(addresses, total)
            else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            addresses = next
        }
        return grows ? try binding.reserveHostMetadata(addressPages: addresses) : nil
    }

    private func materializeIssuedNativeSegments(
        all: Bool,
        binding: CBv2NativePagedModelBinding
    ) throws {
        try binding.requireEngineQueue()
        guard !all else {
            throw CBv2KVError.backendIneligible(
                reason: "native paged profile does not authorize eager slabs")
        }
        let operation = try binding.beginWork()
        let previous = bytesMaterialized
        var grewFloor = false
        var hostMetadata: CBv2NativePagedHostMetadataGeneration?
        var retiringHostMetadata: CBv2NativePagedHostMetadataGeneration?
        defer { withExtendedLifetime(retiringHostMetadata) {} }
        do {
            hostMetadata = try nativePrefixHostMetadataPreparation()
            if let hostMetadata { operation.retain(owner: hostMetadata) }
            let grant = segmentGrant!.snapshot()
            let plans = try planSegments(eager: false, grant: grant)
            guard plans.contains(where: { !$0.plan.segmentIDs.isEmpty }) else {
                if hostMetadata != nil { try operation.requiredDrain() }
                operation.finish(unstarted: hostMetadata == nil)
                return
            }
            let physical = try physicalBytes(plans)
            guard physical <= grant.bytes else {
                throw CBv2KVError.capacityExhausted(needed: physical, available: grant.bytes)
            }
            try physicalLease?.resize(to: physical)
            grewFloor = true
            var prepared: [(PagedKVGroup, PagedKVGroup.PreparedGrowth)] = []
            try operation.withConstruction {
                for item in plans where !item.plan.segmentIDs.isEmpty {
                    prepared.append(
                        (
                            item.group,
                            try item.group.prepareGrowth(
                                item.plan, evaluate: slabEval, admission: memoryAdmission)
                        ))
                }
            }
            // Slab eval/actual constructor fences and this captured-stream
            // completion are outside the native metadata commit.
            try operation.requiredDrain()
            let actual = prepared.reduce(bytesMaterialized) { total, entry in
                total - entry.0.committedSegmentBytes
                    + entry.1.segments.values.reduce(0) { $0 + $1.allocatedBytes }
            }
            guard
                try operation.tracking.commitIfHealthyThrowing({
                    let accepted = segmentGrant!.publish(expected: grant, physicalBytes: actual) {
                        for (group, replacement) in prepared { group.installGrowth(replacement) }
                        if hostMetadata != nil {
                            retiringHostMetadata = binding.installHostMetadata(hostMetadata)
                        }
                    }
                    storageTelemetry.record(accepted)
                    guard accepted == .installed else {
                        throw CBv2KVError.capacityExhausted(needed: actual, available: grant.bytes)
                    }
                })
            else { throw CBv2NativeShutdownError.operationClosed }
            operation.finish { physicalLease?.release(to: actual) }
        } catch {
            if operation.hasArrays || operation.failed || !operation.tracking.mayExecute {
                // Even a late grant/cancellation failure retains actual
                // partial buffers/backing AND the full previously paid peak.
                operation.fail()
            } else {
                // Actual typed cold scope: no array was created/submitted.
                // The old live pool has not been changed.
                // Host-only planning owners still need an honest completion
                // boundary before the operation can detach them.
                if hostMetadata != nil {
                    do { try operation.requiredDrain() } catch {
                        operation.fail()
                        throw error
                    }
                }
                operation.finish(unstarted: hostMetadata == nil) {
                    if grewFloor { physicalLease?.release(to: previous) }
                }
            }
            throw error
        }
    }
}
