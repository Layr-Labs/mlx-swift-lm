// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Package-owned target declaration, independent of the MiMo native issuer.
package protocol GLM5NextPagedModelProviding: AnyObject {
    var glm5NextPagedKinds: [CBv2LayerKind] { get }
    var glm5NextPagedMaximumContext: Int { get }
    var glm5NextPagedIndexHeads: Int { get }
    var glm5NextPagedIndexDimension: Int { get }
    var glm5NextPagedSelectionWidth: Int { get }
}

/// One actual target, main backend and cache-bank layout. It owns no requests,
/// creates no AdmissionV2, and grants neither MTP nor complete-prefix reuse.
package final class GLM5NextPagedModelBinding: @unchecked Sendable {
    private weak var target: (any GLM5NextPagedModelProviding)?
    let targetIdentity: ObjectIdentifier
    let kinds: [CBv2LayerKind]
    let limits: CBv2PagedGatheredAttentionLimits
    private let indexHeads: Int
    private let indexDimension: Int
    let selectionWidth: Int
    private weak var backend: PagedKVBackend?
    private var stepResources: [UInt64: GLM5NextPagedStepResources] = [:]
    package private(set) var observedSelectedRows = 0
    package private(set) var observedMetadataRows = 0

    func recordSelectedRows(_ count: Int) { observedSelectedRows += count }
    func recordMetadataRows(_ count: Int) { observedMetadataRows += count }

    package init(target: any GLM5NextPagedModelProviding, config: PagedKVPoolConfig) throws {
        let kinds = target.glm5NextPagedKinds
        guard let limits = config.gatheredAttention,
            limits.admissionMode == .stepOwned(.pinnedMetal), config.segmentSizeBytes != nil,
            config.quantization == nil, config.layerDTypes == Array(repeating: .float32, count: kinds.count),
            limits.maximumContextTokens <= target.glm5NextPagedMaximumContext,
            !kinds.isEmpty, kinds.count <= 64,
            kinds.allSatisfy({ $0.attention == .full && $0.sharesKVWithLayer == nil
                && $0.kvHeads == 1 && $0.headDim > 0 && $0.headDim <= 512
                && $0.headDim != $0.valueHeadDim
                && $0.valueHeadDim == 2 * target.glm5NextPagedIndexDimension + 1 }),
            target.glm5NextPagedIndexHeads > 0,
            target.glm5NextPagedIndexDimension > 0,
            (1...2051).contains(target.glm5NextPagedSelectionWidth)
        else { throw CBv2KVError.backendIneligible(reason: "invalid loaded GLM paged payload declaration") }
        self.target = target
        targetIdentity = ObjectIdentifier(target)
        self.kinds = kinds
        self.limits = limits
        indexHeads = target.glm5NextPagedIndexHeads
        indexDimension = target.glm5NextPagedIndexDimension
        selectionWidth = target.glm5NextPagedSelectionWidth
    }

    func attach(_ backend: PagedKVBackend) throws {
        guard self.backend == nil, backend.layerKinds == kinds,
            backend.pool.layerDTypes == Array(repeating: .float32, count: kinds.count),
            backend.pool.usesStepOwnedAttention, backend.pool.bytesReserved == 0,
            backend.pool.bytesMaterialized == 0, backend.residentPrefixIndex == nil else {
            throw CBv2KVError.backendIneligible(reason: "GLM payload binding requires its fresh main pool")
        }
        self.backend = backend
    }

    package func matches(_ identity: ObjectIdentifier) -> Bool {
        guard let target, let backend else { return false }
        return identity == targetIdentity && ObjectIdentifier(target) == targetIdentity
            && target.glm5NextPagedKinds == kinds && backend.layerKinds == kinds
    }

    func validateContext(_ maximum: Int) throws {
        guard matches(targetIdentity), maximum > 0, maximum <= limits.maximumContextTokens,
            let pool = backend?.pool, pool.memoryAdmission?.hasProcessMemoryOwner == true,
            pool.attentionWorkEnginePrepared, pool.attentionWorkEngineRefusal == nil else {
            throw CBv2KVError.backendIneligible(reason: "GLM paged context lacks the actual engine admission/binding")
        }
    }

    func prepareWork(
        backend: PagedKVBackend, assignments: [(id: CBv2RequestID, range: Range<Int>)],
        states: [CBv2RequestID: [CBv2SequenceKV?]], layerForRow: (UInt64) -> Int?
    ) throws -> CBv2PagedAttentionStepOwner {
        let pool = backend.pool
        guard self.backend === backend, matches(targetIdentity), pool.activeAttentionWork == nil,
            pool.attentionWorkEnginePrepared, pool.attentionWorkEngineRefusal == nil,
            let admission = pool.memoryAdmission, admission.hasProcessMemoryOwner,
            let policy = Memory.allocationFootprintPolicy(),
            StreamOrDevice.default.stream == Stream.gpu,
            assignments.count > 0, assignments.count <= limits.maximumBatchSize,
            Set(assignments.map(\.id)).count == assignments.count,
            pool.attentionWorkOwners.values.filter({ !$0.completed }).count < limits.maximumInFlightGraphs
        else { throw CBv2KVError.backendIneligible(reason: "GLM paged work is not bound to the active engine") }
        let environment = CBv2PagedAttentionWorkEnvironment.capture()
        try environment.requireSupported()
        var descriptors: [CBv2PagedAttentionTicketKey: CBv2PagedAttentionRowDescriptor] = [:]
        var allocations: [CBv2PagedAttentionAllocation] = []
        var copies: [CBv2PagedAttentionTicketKey: [PagedSequenceKV.ImmutablePageCopy]] = [:]
        var copyPages: [PagedKVGroupKey: Int] = [:]
        var seen = Set<UInt64>()
        // The model constructs rectangular FP32 transport buffers before row
        // slicing. Ragged assignments conservatively use B * max(T), not sum(T).
        for assignment in assignments {
            guard assignment.range.lowerBound >= 0,
                assignment.range.upperBound > assignment.range.lowerBound,
                assignment.range.upperBound <= limits.maximumContextTokens,
                assignment.range.count <= limits.maximumQueryTokens else {
                throw CBv2KVError.backendIneligible(reason: "GLM paged work range")
            }
        }
        let maximumQuery = assignments.map { $0.range.count }.max()!
        for (index, kind) in kinds.enumerated() {
            for (role, width) in [("K", kind.headDim), ("V", kind.valueHeadDim)] {
                let logical = try CBv2PagedWorkMath.product([
                    assignments.count, maximumQuery, kind.kvHeads, width, DType.float32.size,
                ])
                guard logical <= pool.config.maxBufferLength else {
                    throw CBv2KVError.backendIneligible(
                        reason: "GLM batched \(role) transport at layer \(index) exceeds native maximum buffer length")
                }
                guard let bound = policy.upperBound(byteCount: logical), bound >= logical else {
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
                allocations.append(.init(
                    role: "write.batch\(role).\(index)", logicalBytes: logical, allocationCount: 1,
                    upperBoundBytes: bound, maximumLogicalBufferBytes: logical))
            }
        }
        for assignment in assignments {
            guard let rows = states[assignment.id], rows.count == kinds.count,
                !assignment.range.isEmpty, assignment.range.lowerBound >= 0,
                assignment.range.upperBound <= limits.maximumContextTokens,
                assignment.range.count <= limits.maximumQueryTokens else {
                throw CBv2KVError.backendIneligible(reason: "GLM paged work range")
            }
            for (index, kind) in kinds.enumerated() {
                guard let row = rows[index] as? PagedSequenceKV, row.pool === pool, !row.isReleased,
                    layerForRow(row.serial) == index, seen.insert(row.serial).inserted,
                    row.absoluteOffset == assignment.range.lowerBound,
                    row.maxLength >= assignment.range.upperBound, row.windowSize == nil,
                    (row.speculativeBase == nil
                        || CBv2NativePagedMTPWork.current?.permitsPlannedColumn(
                            row, range: assignment.range) == true),
                    row.baseOffset == 0, row.frozenHighWater == 0,
                    row.groupKey == pool.groupKey(forLayer: index)
                else { throw CBv2KVError.backendIneligible(reason: "GLM paged row identity/frontier") }
                let descriptor = CBv2PagedAttentionRowDescriptor(
                    requestID: assignment.id, row: .init(row), rowSerial: row.serial,
                    layerIndex: index, ownerLayerIndex: index, range: assignment.range,
                    baseOffset: 0, frozenHighWater: 0, gatheredRange: 0..<assignment.range.upperBound,
                    keyStart: 0, keyCount: assignment.range.upperBound, blocks: [], softcap: false)
                let key = CBv2PagedAttentionTicketKey(layer: index, serial: row.serial)
                let rowCopies = try row.immutableCopiesForWrite(count: assignment.range.count)
                descriptors[key] = descriptor
                copies[key] = rowCopies
                if !rowCopies.isEmpty {
                    copyPages[row.groupKey] = try CBv2PagedWorkMath.add(copyPages[row.groupKey, default: 0], rowCopies.count)
                }
                allocations += try workAllocations(kind: kind, range: assignment.range,
                    pageSize: pool.config.pageSize, copyPageCount: rowCopies.count, policy: policy)
            }
        }
        let bytes = try allocations.reduce(64 << 10) { try CBv2PagedWorkMath.add($0, $1.upperBoundBytes) }
        guard try CBv2PagedWorkMath.add(pool.attentionWorkBytesReserved, bytes) <= limits.maximumScratchBytes,
            allocations.allSatisfy({ $0.maximumLogicalBufferBytes <= pool.config.maxBufferLength }) else {
            throw CBv2KVError.capacityExhausted(needed: bytes, available: max(0, limits.maximumScratchBytes - pool.attentionWorkBytesReserved))
        }
        let (generation, overflow) = pool.attentionWorkGeneration.addingReportingOverflow(1)
        guard !overflow else { throw CBv2KVError.backendIneligible(reason: "GLM paged generation exhausted") }
        let reservation = try admission.reserveTransient(bytes: bytes)
        // Price and commit every extra live destination before publishing the
        // ticket. Old row references survive until this step's fence, so they
        // cannot fund the clones, even when a row was rewound across pages.
        if !copyPages.isEmpty {
            do {
                try pool.reserve(copyPages)
                do { try pool.materializeReservedSegments() } catch {
                    pool.unreserve(copyPages)
                    throw error
                }
            } catch {
                reservation.release()
                throw error
            }
        }
        let resources = GLM5NextPagedStepResources(pool: pool)
        resources.rows = descriptors.values.compactMap { $0.row.value }
        resources.copies = copies
        resources.prepaidPages = copyPages
        stepResources[generation] = resources
        let scoped = CBv2CheckpointReservation { [weak self, resources] in
            resources.releaseAfterFence()
            self?.stepResources.removeValue(forKey: generation)
            reservation.release()
        }
        let work = CBv2PagedAttentionStepOwner(generation: generation, pool: pool, environment: environment,
            descriptors: descriptors, allocations: allocations, bytes: bytes, reservation: scoped,
            allocationPolicy: policy)
        pool.attentionWorkGeneration = generation
        pool.attentionWorkOwners[generation] = work
        pool.publishAttentionWorkCharge()
        pool.activeAttentionWork = work
        return work
    }

    func clonePage(_ copy: PagedSequenceKV.ImmutablePageCopy, row: PagedSequenceKV,
                   work: CBv2PagedAttentionStepOwner) throws -> Int32 {
        guard let backend, backend.pool.activeAttentionWork === work,
            let layer = backend.pool.activeAttentionLayer,
            row.pool === backend.pool, !row.isReleased,
            copy.source.group == backend.pool.groupKey(forLayer: layer),
            (backend.pool.refCount(for: copy.source) ?? 0) > 1,
            let resources = stepResources[work.generation], backend.pool.isValid(copy.source),
            let planned = resources.copies[.init(layer: layer, serial: row.serial)],
            let index = planned.firstIndex(where: {
                $0.logicalPage == copy.logicalPage && $0.source == copy.source
                    && $0.initializedTokens == copy.initializedTokens
            }),
            row.table.indices.contains(copy.logicalPage), row.table[copy.logicalPage] == copy.source.page,
            (0...backend.pool.config.pageSize).contains(copy.initializedTokens) else {
            throw PagedImmutableSnapshotError.stale
        }
        resources.copies[.init(layer: layer, serial: row.serial)]!.remove(at: index)
        let pool = backend.pool
        let source = copy.source
        let page = pool.allocatePage(group: source.group)
        if copy.initializedTokens > 0 {
            let copied = PagedAsymmetricTransfers.gatherSegmented(
                group: pool.group(source.group), pages: [source.page], firstSlot: 0,
                count: copy.initializedTokens, work: work)
            let slots = (0..<copy.initializedTokens).map { page * Int32(pool.config.pageSize) + Int32($0) }
            pool.writeTokens(group: source.group, slots: slots,
                             keys: copied.keys.squeezed(axis: 0), values: copied.values.squeezed(axis: 0))
        }
        // Transfer the old row reference, not an uncounted duplicate. It is
        // released only when the existing engine step drops its reservation.
        resources.retired.append(source)
        return page
    }

    private func workAllocations(kind: CBv2LayerKind, range: Range<Int>, pageSize: Int, copyPageCount: Int,
                                 policy: AllocationFootprintPolicy) throws -> [CBv2PagedAttentionAllocation] {
        let q = range.count, n = range.upperBound
        let pages = (n - 1) / pageSize + 1
        guard let extra = policy.maximumExtraBytes else { throw CBv2CompleteCheckpointError.allocationFailed }
        var result: [CBv2PagedAttentionAllocation] = []
        func buffers(_ role: String, _ factors: [Int], bytes: Int = 4, copies: Int = 1) throws {
            let logical = try CBv2PagedWorkMath.product(factors + [bytes])
            guard let bound = policy.upperBound(byteCount: logical) else { throw CBv2CompleteCheckpointError.allocationFailed }
            result.append(.init(role: role, logicalBytes: try CBv2PagedWorkMath.product([logical, copies]),
                allocationCount: copies, upperBoundBytes: try CBv2PagedWorkMath.product([bound, copies]),
                maximumLogicalBufferBytes: logical))
        }
        // Possible per-segment partitions are priced separately, including the
        // minimum 24-word records and each allocation's allocator overhead.
        for (role, count) in [("write", q), ("indexRead", n)] {
            let logical = try CBv2PagedWorkMath.add(
                CBv2PagedWorkMath.product([count, 3, 4]), CBv2PagedWorkMath.product([pages, 24, 4]))
            result.append(.init(role: role + ".records", logicalBytes: logical, allocationCount: pages,
                upperBoundBytes: try CBv2PagedWorkMath.add(logical, CBv2PagedWorkMath.product([pages, extra])), maximumLogicalBufferBytes: logical))
            try buffers(role + ".fences", [1], copies: pages + 1)
            // Host slots (8/token), growing triples (36/token), padded
            // assembly (24/token), then per-bucket padding/map/page metadata.
            try buffers(role + ".hostRecords", [count, 68], bytes: 1)
            try buffers(role + ".hostBuckets", [pages, 456], bytes: 1)
            try buffers(role + ".shapes", [3], copies: pages)
        }
        try buffers("write.contiguousK", [q, kind.headDim], copies: pages)
        try buffers("write.contiguousV", [q, kind.valueHeadDim], copies: pages)
        try buffers("index.V", [n, kind.valueHeadDim])
        try buffers("index.castAndPool", [n, indexDimension], copies: 16)
        try buffers("index.score", [n, indexHeads], copies: CBv2PagedWorkMath.product([q, 4]))
        try buffers("index.validity", [n], copies: 4)
        try buffers("selected.K", [selectionWidth, kind.headDim], copies: q)
        try buffers("selected.zeroAndProjectionInputs", [selectionWidth, kind.headDim], copies: q)
        try buffers("selected.tables", [pages], copies: q + 1)
        try buffers("selected.fences", [1], copies: CBv2PagedWorkMath.product([q, pages + 1]))
        try buffers("selected.indices", [selectionWidth], copies: CBv2PagedWorkMath.product([q, pages]))
        try buffers("selected.host", [pages, 128], bytes: 1, copies: q + 1)
        let copyCount = max(1, copyPageCount)
        try buffers("cow.payload", [pageSize, kind.headDim + kind.valueHeadDim],
                    copies: CBv2PagedWorkMath.product([4, copyCount]))
        try buffers("cow.records", [max(24, pageSize * 3)], copies: CBv2PagedWorkMath.product([2, copyCount]))
        try buffers("cow.fences", [1], copies: CBv2PagedWorkMath.product([4, copyCount]))
        try buffers("cow.hostPlan", [pageSize, 128], bytes: 1, copies: copyCount)
        return result
    }
}

private final class GLM5NextPagedStepResources: @unchecked Sendable {
    let pool: PagedKVPool
    var rows: [PagedSequenceKV] = []
    var copies: [CBv2PagedAttentionTicketKey: [PagedSequenceKV.ImmutablePageCopy]] = [:]
    var prepaidPages: [PagedKVGroupKey: Int] = [:]
    var retired: [PagedKVPageHandle] = []
    init(pool: PagedKVPool) { self.pool = pool }
    func releaseAfterFence() {
        for handle in retired { pool.freePages(group: handle.group, pages: [handle.page]) }
        retired.removeAll()
        rows.removeAll()
        copies.removeAll()
        if !prepaidPages.isEmpty { pool.unreserve(prepaidPages) }
        prepaidPages.removeAll()
    }
}

extension PagedKVPool {
    var glm5NextPayloadBinding: GLM5NextPagedModelBinding? {
        attentionWorkCaches.values.compactMap { $0.value?.glm5NextModelBinding }.first
    }
}
