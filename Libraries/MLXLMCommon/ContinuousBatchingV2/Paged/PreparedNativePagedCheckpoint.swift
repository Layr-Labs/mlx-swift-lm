// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// One actual stage + suffix preparation, not a ledger or a capability marker.
/// Creation, preparation and publication stay in ONE engine-queue turn.
/// Only metadata publication runs under the native outcome commit.
final class CBv2PreparedNativePagedCheckpoint {
    private struct Group {
        let value: PagedKVGroup
        let plan: PagedKVGroup.ImportPlan
        let prepared: PagedKVGroup.PreparedImport
    }
    private let backend: PagedKVBackend
    private let binding: CBv2NativePagedModelBinding
    private let work: CBv2NativeCompletePrefixWork
    private let operation: CBv2NativePagedOperation
    let requestID: CBv2RequestID
    let streamGeneration: UInt64
    let maximumTokens: Int
    private let promptLength: Int
    private var owner: CBv2PagedCheckpointOwner?
    private var groups: [Group] = []
    private var rows: [PagedSequenceKV] = []
    private var tables: [[Int32]] = []
    private var grant: PagedKVGrant.Snapshot?
    private var previousPhysicalBytes = 0
    private var actualPhysicalBytes = 0
    private var ready = false
    private var adopted = false
    private var retired = false
    private var nativeFinished = false
    private var registered = false
    private var hostMetadata: CBv2NativePagedHostMetadataGeneration?
    private var retiringHostMetadata: CBv2NativePagedHostMetadataGeneration?
    private(set) var addedHostBytes = 0
    private(set) var addedTargetBound = 0

    init(backend: PagedKVBackend, codec: CBv2CompleteCheckpointCodec,
         work: CBv2NativeCompletePrefixWork, request: CBv2Request,
         streamGeneration: UInt64, maximumTokens: Int) throws {
        let maximum = request.promptTokens.count.addingReportingOverflow(max(1, request.maxTokens))
        guard let binding = backend.nativeModelBinding,
              codec.nativePagedBinding === binding, codec.isNativePagedHistorical,
              work.purpose == .importing, work.codecIdentity == codec.identity,
              codec.layerKinds == backend.layerKinds,
              codec.admission === backend.pool.memoryAdmission,
              codec.admission.hasProcessMemoryOwner,
              request.prefixCacheEnabled, request.multimodal == nil, request.positionState == nil,
              request.prefixCacheReceiptID != nil, streamGeneration > 0,
              !maximum.overflow, maximumTokens == maximum.partialValue,
              maximumTokens > request.promptTokens.count,
              maximumTokens <= binding.maximumContextTokens else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try binding.requireEngineQueue()
        self.backend = backend; self.binding = binding; self.work = work
        self.requestID = request.id; self.streamGeneration = streamGeneration
        self.maximumTokens = maximumTokens
        promptLength = request.promptTokens.count
        operation = try binding.beginWork(requests: [request.id])
    }

    /// The containing import owner attaches this object BEFORE invoking this.
    /// No prepared page table or suffix allocation exists before its real
    /// upward extension of the original typed stage.
    func prepare(frame: CBv2PagedCheckpointFrame, admission: AdmissionV2) throws {
        try binding.requireEngineQueue()
        guard owner == nil, !ready, !adopted, !retired else {
            throw CBv2CompleteCheckpointError.closed
        }
        let owner = try frame.consume()
        self.owner = owner
        let pool = backend.pool, storage = owner.storage, checkpoint = owner.storage.plan
        guard let segmentGrant = pool.segmentGrant, let physical = pool.physicalLease,
              admission === owner.lease.admission, admission === pool.memoryAdmission,
              physical.bytes == pool.bytesMaterialized,
              checkpoint.pageSize == pool.config.pageSize,
              checkpoint.position < promptLength,
              checkpoint.ownerMap == Array(backend.layerKinds.indices),
              checkpoint.layers.map(\.modelIndex) == Array(backend.layerKinds.indices),
              checkpoint.groups.count == pool.groups.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var needs: [PagedKVGroupKey: Int] = [:]
        var rowNeeds: [Int] = []
        for layer in checkpoint.layers {
            let kind = backend.layerKinds[layer.modelIndex]
            guard kind.sharesKVWithLayer == nil, layer.key == pool.groupKey(forLayer: layer.modelIndex),
                  layer.key == PagedKVGroupKey(kind, dtype: pool.layerDTypes[layer.modelIndex], separateWindow: true),
                  layer.tokenStart == (layer.key.windowSize.map { max(0, checkpoint.position - $0) } ?? 0) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let count = PagedKVPool.pageDemand(kind: kind, maxLength: maximumTokens, config: pool.config)
            guard layer.pageCount <= count else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            needs[layer.key] = try CBv2CheckpointAllocationFootprint.add(needs[layer.key, default: 0], count)
            rowNeeds.append(count)
        }
        let snapshot = segmentGrant.snapshot()
        var growthBound = 0, addressBound = 0
        // Scalar bound: creates no replacement dictionary, page-address map
        // or native array. Existing immutable stage metadata remains charged.
        for key in pool.groupKeys {
            let group = pool.group(key)
            guard let source = storage.groups[key], let layout = group.segmentLayout,
                  source.segments.values.allSatisfy({ $0.backing.belongs(to: admission) }) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let demand = try CBv2CheckpointAllocationFootprint.add(group.pagesReserved, needs[key, default: 0])
            let available = try CBv2CheckpointAllocationFootprint.add(group.committedUsablePages, source.pages.count)
            let missing = max(0, demand - available)
            guard let bound = layout.allocationBytes(addingUsablePages: missing),
                  let growthAddresses = CBv2KVGeometry.multiply(missing, 2) else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            growthBound = try CBv2CheckpointAllocationFootprint.add(growthBound, bound)
            addressBound = try CBv2CheckpointAllocationFootprint.add(addressBound,
                CBv2CheckpointAllocationFootprint.add(group.pageCount,
                    CBv2CheckpointAllocationFootprint.add(source.layout.pageCount, growthAddresses)))
        }
        let totalBound = try CBv2CheckpointAllocationFootprint.add(pool.bytesMaterialized,
            CBv2CheckpointAllocationFootprint.add(storage.allocatedBytes, growthBound))
        guard totalBound <= snapshot.bytes else {
            throw CBv2KVError.capacityExhausted(needed: totalBound, available: snapshot.bytes)
        }
        // A real map-generation owner moves with the published pool maps.
        // The stage now prices only this helper's private control/row wrappers,
        // not a duplicate reservation for those same replacement maps.
        hostMetadata = try binding.reserveHostMetadata(addressPages: addressBound)
        guard let hostMetadata else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        try work.retain(owners: [hostMetadata])
        operation.retain(owner: hostMetadata)
        let host = try CBv2CheckpointAllocationFootprint.add(64 << 10, checkpoint.layers.count * 512)
        try owner.lease.extendForNativePagedPreparation(targetBytes: growthBound, auxiliaryBytes: host)
        addedTargetBound = growthBound; addedHostBytes = host
        previousPhysicalBytes = pool.bytesMaterialized
        grant = snapshot
        try work.captureCurrentStreams()
        do {
            try operation.withConstruction {
                for key in pool.groupKeys {
                    let group = pool.group(key), source = storage.groups[key]!
                    let plan = try group.planImport(source, additionalReservedPages: needs[key, default: 0])
                    let prepared = try group.prepareImport(plan, source: source,
                        evaluate: pool.slabEval, admission: admission)
                    groups.append(.init(value: group, plan: plan, prepared: prepared))
                }
            }
            // The same complete private plans are used at commit. Do not build
            // a second host page map or mutate any live group here.
            for (index, layer) in checkpoint.layers.enumerated() {
                guard let group = groups.first(where: { $0.value.key == layer.key }),
                      layer.firstPage <= group.plan.pages.count,
                      layer.pageCount <= group.plan.pages.count - layer.firstPage else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                tables.append(Array(group.plan.pages[layer.firstPage..<layer.firstPage + layer.pageCount]))
                rows.append(PagedSequenceKV(pool: pool, kind: backend.layerKinds[layer.modelIndex],
                    groupKey: layer.key, maxLength: maximumTokens, reservedPages: rowNeeds[index]))
            }
            actualPhysicalBytes = try groups.reduce(0) { total, group in
                try group.prepared.growth.segments.values.reduce(total) {
                    try CBv2CheckpointAllocationFootprint.add($0, $1.allocatedBytes)
                }
            }
            guard actualPhysicalBytes >= previousPhysicalBytes,
                  actualPhysicalBytes <= totalBound else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            try operation.requiredDrain()
            try work.fenceForProtectedPromotion()
            try owner.lease.settleDestinationAfterEvaluation(
                targetBytes: actualPhysicalBytes - previousPhysicalBytes,
                auxiliaryBytes: owner.lease.auxiliaryBytes)
            ready = true
        } catch {
            // An actual partial suffix allocation/evaluation is a required
            // native failure. The two real loans keep all roots and stage C.
            if operation.hasArrays || error is MLXError {
                operation.fail(); work.requiredCompletionFailed()
            }
            throw error
        }
    }

    var scalarWitness: (ready: Bool, rows: Int, targetBound: Int, hostBound: Int) {
        (ready && operation.completed && work.hasProtectedPromotionCompletion,
         rows.count, addedTargetBound, addedHostBytes)
    }

    func authorizesRegistration(_ candidate: [CBv2SequenceKV?], backend expected: PagedKVBackend) -> Bool {
        backend === expected && backend.nativeModelBinding === binding && ready && !adopted && !retired
            && operation.completed && !operation.failed && work.hasProtectedPromotionCompletion
            && candidate.count == rows.count
            && zip(candidate, rows).allSatisfy { $0.0 === $0.1 }
    }

    /// Native outcome lock is held by enqueue. Every fallible assistant check
    /// precedes this method. No array operation, evaluation or fence occurs.
    func publish(admission: AdmissionV2, streamGeneration: UInt64) throws -> [CBv2SequenceKV?] {
        try binding.requireEngineQueue()
        guard ready, !adopted, !retired, !registered, operation.completed, !operation.failed,
              work.hasProtectedPromotionCompletion, self.streamGeneration == streamGeneration,
              let owner, let grant, let segmentGrant = backend.pool.segmentGrant,
              let physical = backend.pool.physicalLease, owner.lease.admission === admission,
              physical.bytes == previousPhysicalBytes,
              backend.pool.bytesMaterialized == previousPhysicalBytes else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let state: [CBv2SequenceKV?] = rows.map { $0 }
        try backend.registerPreparedNativeCheckpoint(state, preparation: self)
        registered = true
        let reservation: CBv2CheckpointAdoptionReservation
        do {
            reservation = try physical.transferCheckpoint(to: actualPhysicalBytes, admission: admission) { previous in
                try admission.transferCheckpointStage(owner.lease, requestID: requestID,
                    maximumTokens: maximumTokens, previousPhysicalBytes: previous,
                    physicalBytes: actualPhysicalBytes, retainingNativeAuxiliaryStage: true)
            }
        } catch {
            do { try backend.rollbackPreparedNativeCheckpointRegistration(state, preparation: self) }
            catch { operation.fail(); work.requiredCompletionFailed(); throw error }
            registered = false
            throw error
        }
        let result = segmentGrant.publish(expected: grant, physicalBytes: actualPhysicalBytes) {
            for group in groups { group.value.installImport(group.prepared) }
            retiringHostMetadata = binding.installHostMetadata(hostMetadata)
            for (index, row) in rows.enumerated() {
                let layer = owner.storage.plan.layers[index]
                if layer.ringPages != nil {
                    row.adoptHistoricalWindowPages(tables[index], retainedStart: layer.tokenStart,
                                                   storedThrough: owner.storage.plan.position)
                } else {
                    row.adoptExclusiveCheckpointPages(tables[index], storedThrough: owner.storage.plan.position)
                }
            }
        }
        backend.pool.storageTelemetry.record(result)
        guard result == .installed else {
            // Same engine-queue turn: no ordinary mutation intervened. Target
            // bytes return to the private stage, NEVER to free capacity while
            // the actual immutable buffers/loan are still retained.
            reservation.rollbackAfterDroppingOwners()
            do { try backend.rollbackPreparedNativeCheckpointRegistration(state, preparation: self) }
            catch { operation.fail(); work.requiredCompletionFailed(); throw error }
            registered = false
            throw CBv2KVError.capacityExhausted(needed: actualPhysicalBytes, available: segmentGrant.snapshot().bytes)
        }
        reservation.commit()
        adopted = true
        backend.pool.storageTelemetry.recordSettlement(
            bound: previousPhysicalBytes + owner.lease.targetBytes, actual: actualPhysicalBytes)
        rows.removeAll(); tables.removeAll()
        return state
    }

    /// Executed by the containing import owner's real post-fence callback on
    /// this engine queue. A failed operation cannot be mistaken for completion.
    func retireAfterCompletion() throws {
        try binding.requireEngineQueue()
        if nativeFinished { return }
        // A refused host-only preparation may own map allowances but have
        // submitted no suffix array. Fence its actual streams; do not mint an
        // unstarted success while a retained owner still exists.
        if !operation.completed && !operation.failed && !operation.hasArrays {
            do { try operation.requiredDrain() }
            catch { work.requiredCompletionFailed(); throw error }
        }
        guard !operation.failed,
              operation.finish(unstarted: !operation.hasArrays) else {
            throw CBv2NativeShutdownError.operationClosed
        }
        nativeFinished = true
    }

    /// All duplicate imported arrays/checkpoint aliases are dropped first by
    /// the containing owner. Only then may this stage release its remaining
    /// auxiliary/host/scratch charge. Active pool/request ownership is separate.
    func closeAfterNativeCompletion() {
        precondition(nativeFinished && !retired)
        retired = true
        if !adopted {
            precondition(!registered)
            for row in rows { row.discardUninstalledCheckpointRow() }
        }
        rows.removeAll(); tables.removeAll(); groups.removeAll()
        hostMetadata = nil
        retiringHostMetadata = nil
        owner?.close()
        owner = nil
    }

    deinit {
        assert(retired, "native paged prefix preparation bypassed actual retirement")
    }
}
