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
    /// Byte bound of one query block's FP32 `[1,Qb,H,P]` index scores. The
    /// target's selector blocks queries by this bound; step pricing mirrors it.
    var glm5NextPagedSelectionScoreBlockBytes: Int { get }
    /// Byte bound of one dense-history query block's FP32 `[H*Tb,N]` scores.
    var glm5NextPagedAttentionBlockBytes: Int { get }
    /// Whether queries at raw positions below `index_topk` attend their
    /// dense history prefix (one prefix read) instead of per-query reads.
    var glm5NextPagedDenseHistoryPrefix: Bool { get }
    /// Largest sparse query count whose selection pools in kernel memory
    /// (no pooled-key planes or per-head scores); 0 when that path is off.
    var glm5NextPagedNativeSparseDecodeQueries: Int { get }
    /// Smallest sparse query count of one row's step that may read the row's
    /// whole latent history once instead of per-query selection rows; 0 never.
    var glm5NextPagedLatentHistoryMinimumQueries: Int { get }
    /// Whether sparse attention is absorbed (gathers bounded query blocks
    /// over a history read) rather than the expanded per-query reference.
    var glm5NextPagedAbsorbedAttention: Bool { get }
    /// The target's load-time choice; every backend bound to it uses this form.
    var glm5NextPagedMetadataReads: GLM5NextPagedMetadataReads { get }
}

/// One actual target, main backend and cache-bank layout. It owns no requests,
/// creates no AdmissionV2, and grants neither MTP nor complete-prefix reuse.
package final class GLM5NextPagedModelBinding: @unchecked Sendable {
    private weak var target: (any GLM5NextPagedModelProviding)?
    let targetIdentity: ObjectIdentifier
    let kinds: [CBv2LayerKind]
    let limits: CBv2PagedGatheredAttentionLimits
    let indexHeads: Int
    let indexDimension: Int
    let selectionWidth: Int
    let scoreBlockBytes: Int
    let attentionBlockBytes: Int
    let denseHistoryPrefix: Bool
    let nativeSparseDecodeQueries: Int
    let latentHistoryMinimumQueries: Int
    let absorbedAttention: Bool
    let metadataReads: GLM5NextPagedMetadataReads
    private(set) weak var backend: PagedKVBackend?
    private var stepResources: [UInt64: GLM5NextPagedStepResources] = [:]
    package private(set) var observedSelectedRows = 0
    package private(set) var observedMetadataRows = 0
    package private(set) var observedHistoryRows = 0

    func recordSelectedRows(_ count: Int) { observedSelectedRows += count }
    func recordMetadataRows(_ count: Int) { observedMetadataRows += count }
    func recordHistoryRows(_ count: Int) { observedHistoryRows += count }

    /// One history read replaces the per-query selection rows only for
    /// prefill-sized steps, and only when they would read at least the
    /// history, so a row's retained latent reads never exceed
    /// `max(N, (minimum - 1) * selectionWidth)` rows however long the step is.
    func readsLatentHistory(sparseQueries: Int, historyLength: Int) -> Bool {
        latentHistoryMinimumQueries > 0 && sparseQueries >= latentHistoryMinimumQueries
            && sparseQueries * selectionWidth >= historyLength
    }

    /// The latent reads a row-layer's step is priced for.
    enum ReadPlan: Equatable {
        /// One whole-history read and no selection rows.
        case history
        /// At most this many selection-row reads (the dense prefix's and one
        /// per sparse query) and no history read.
        case selectedRows(Int)
    }

    func readPlan(range: Range<Int>) -> ReadPlan {
        let dense = denseQueryCount(range: range)
        let sparse = range.count - dense
        if sparse > 0, readsLatentHistory(sparseQueries: sparse, historyLength: range.upperBound) {
            return .history
        }
        return .selectedRows(sparse + (dense > 0 ? 1 : 0))
    }

    /// Leading queries below `index_topk` that attend the dense history prefix.
    private func denseQueryCount(range: Range<Int>) -> Int {
        denseHistoryPrefix ? max(0, min(range.count, selectionWidth - 3 - range.lowerBound)) : 0
    }

    /// The plan this step priced for one row-layer; nil outside its step.
    func readPlan(work: CBv2PagedAttentionStepOwner, layer: Int, serial: UInt64) -> ReadPlan? {
        stepResources[work.generation]?.readPlans[.init(layer: layer, serial: serial)]
    }

    package init(target: any GLM5NextPagedModelProviding, config: PagedKVPoolConfig) throws {
        let kinds = target.glm5NextPagedKinds
        guard let limits = config.gatheredAttention,
            limits.admissionMode == .stepOwned(.pinnedMetal), config.segmentSizeBytes != nil,
            config.quantization == nil,
            config.layerDTypes == Array(repeating: .float32, count: kinds.count),
            limits.maximumContextTokens <= target.glm5NextPagedMaximumContext,
            !kinds.isEmpty, kinds.count <= 64,
            kinds.allSatisfy({
                $0.attention == .full && $0.sharesKVWithLayer == nil
                    && $0.kvHeads == 1 && $0.headDim > 0 && $0.headDim <= 512
                    && $0.headDim != $0.valueHeadDim
                    && $0.valueHeadDim == 2 * target.glm5NextPagedIndexDimension + 1
            }),
            target.glm5NextPagedIndexHeads > 0,
            target.glm5NextPagedIndexDimension > 0,
            (1 ... 2051).contains(target.glm5NextPagedSelectionWidth),
            target.glm5NextPagedSelectionScoreBlockBytes > 0,
            target.glm5NextPagedAttentionBlockBytes > 0
        else {
            throw CBv2KVError.backendIneligible(
                reason: "invalid loaded GLM paged payload declaration")
        }
        self.target = target
        targetIdentity = ObjectIdentifier(target)
        self.kinds = kinds
        self.limits = limits
        indexHeads = target.glm5NextPagedIndexHeads
        indexDimension = target.glm5NextPagedIndexDimension
        selectionWidth = target.glm5NextPagedSelectionWidth
        scoreBlockBytes = target.glm5NextPagedSelectionScoreBlockBytes
        attentionBlockBytes = target.glm5NextPagedAttentionBlockBytes
        denseHistoryPrefix = target.glm5NextPagedDenseHistoryPrefix
        nativeSparseDecodeQueries = max(0, target.glm5NextPagedNativeSparseDecodeQueries)
        latentHistoryMinimumQueries = max(0, target.glm5NextPagedLatentHistoryMinimumQueries)
        absorbedAttention = target.glm5NextPagedAbsorbedAttention
        metadataReads = target.glm5NextPagedMetadataReads
    }

    func attach(_ backend: PagedKVBackend) throws {
        guard self.backend == nil, backend.layerKinds == kinds,
            backend.pool.layerDTypes == Array(repeating: .float32, count: kinds.count),
            backend.pool.usesStepOwnedAttention, backend.pool.bytesReserved == 0,
            backend.pool.bytesMaterialized == 0, backend.residentPrefixIndex == nil
        else {
            throw CBv2KVError.backendIneligible(
                reason: "GLM payload binding requires its fresh main pool")
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
            pool.attentionWorkEnginePrepared, pool.attentionWorkEngineRefusal == nil
        else {
            throw CBv2KVError.backendIneligible(
                reason: "GLM paged context lacks the actual engine admission/binding")
        }
    }

    /// The exact tickets, allocations and copy-on-write destinations of one
    /// step's assignments. Metadata only: nothing is reserved or published.
    private struct PlannedWork {
        var descriptors: [CBv2PagedAttentionTicketKey: CBv2PagedAttentionRowDescriptor] = [:]
        var allocations: [CBv2PagedAttentionAllocation] = []
        var copies: [CBv2PagedAttentionTicketKey: [PagedSequenceKV.ImmutablePageCopy]] = [:]
        var readPlans: [CBv2PagedAttentionTicketKey: ReadPlan] = [:]
        var copyPages: [PagedKVGroupKey: Int] = [:]
        var bytes = 0
    }

    private func planWork(
        pool: PagedKVPool, assignments: [(id: CBv2RequestID, range: Range<Int>)],
        states: [CBv2RequestID: [CBv2SequenceKV?]], layerForRow: (UInt64) -> Int?,
        policy: AllocationFootprintPolicy
    ) throws -> PlannedWork {
        var plan = PlannedWork()
        var seen = Set<UInt64>()
        // The model constructs rectangular FP32 transport buffers before row
        // slicing. Ragged assignments conservatively use B * max(T), not sum(T).
        for assignment in assignments {
            guard assignment.range.lowerBound >= 0,
                assignment.range.upperBound > assignment.range.lowerBound,
                assignment.range.upperBound <= limits.maximumContextTokens,
                assignment.range.count <= limits.maximumQueryTokens
            else {
                throw CBv2KVError.backendIneligible(reason: "GLM paged work range")
            }
        }
        let maximumQuery = assignments.map { $0.range.count }.max()!
        plan.allocations += try transportAllocations(
            rows: assignments.count, maximumQuery: maximumQuery,
            maximumBufferLength: pool.config.maxBufferLength, policy: policy)
        var stepBytes = GLM5NextStepBytes(layers: kinds.count)
        try stepBytes.add(plan.allocations, layer: nil)
        for assignment in assignments {
            guard let rows = states[assignment.id], rows.count == kinds.count,
                !assignment.range.isEmpty, assignment.range.lowerBound >= 0,
                assignment.range.upperBound <= limits.maximumContextTokens,
                assignment.range.count <= limits.maximumQueryTokens
            else {
                throw CBv2KVError.backendIneligible(reason: "GLM paged work range")
            }
            for (index, kind) in kinds.enumerated() {
                guard let row = rows[index] as? PagedSequenceKV, row.pool === pool, !row.isReleased,
                    layerForRow(row.serial) == index, seen.insert(row.serial).inserted,
                    row.absoluteOffset == assignment.range.lowerBound,
                    row.maxLength >= assignment.range.upperBound, row.windowSize == nil,
                    row.speculativeBase == nil
                        || CBv2NativePagedMTPWork.current?.permitsPlannedColumn(
                            row, range: assignment.range) == true,
                    row.baseOffset == 0, row.frozenHighWater == 0,
                    row.groupKey == pool.groupKey(forLayer: index)
                else {
                    throw CBv2KVError.backendIneligible(reason: "GLM paged row identity/frontier")
                }
                let descriptor = CBv2PagedAttentionRowDescriptor(
                    requestID: assignment.id, row: .init(row), rowSerial: row.serial,
                    layerIndex: index, ownerLayerIndex: index, range: assignment.range,
                    baseOffset: 0, frozenHighWater: 0,
                    gatheredRange: 0 ..< assignment.range.upperBound,
                    keyStart: 0, keyCount: assignment.range.upperBound, blocks: [], softcap: false)
                let key = CBv2PagedAttentionTicketKey(layer: index, serial: row.serial)
                let rowCopies = try row.immutableCopiesForWrite(count: assignment.range.count)
                plan.descriptors[key] = descriptor
                plan.readPlans[key] = readPlan(range: assignment.range)
                plan.copies[key] = rowCopies
                if !rowCopies.isEmpty {
                    plan.copyPages[row.groupKey] = try CBv2PagedWorkMath.add(
                        plan.copyPages[row.groupKey, default: 0], rowCopies.count)
                }
                let rowAllocations = try workAllocations(
                    kind: kind, range: assignment.range, pageSize: pool.config.pageSize,
                    segments: segmentBound(row: row, through: assignment.range.upperBound),
                    copyPageCount: rowCopies.count, policy: policy)
                plan.allocations += rowAllocations
                try stepBytes.add(rowAllocations, layer: index)
            }
        }
        plan.bytes = try stepBytes.total()
        return plan
    }

    /// The scratch bytes `prepareWork` would reserve for these assignments
    /// now. Metadata only: no reservation, ticket, page or row changes.
    func priceWork(
        backend: PagedKVBackend, assignments: [(id: CBv2RequestID, range: Range<Int>)],
        states: [CBv2RequestID: [CBv2SequenceKV?]], layerForRow: (UInt64) -> Int?
    ) throws -> Int {
        guard self.backend === backend, matches(targetIdentity), !assignments.isEmpty,
            assignments.count <= limits.maximumBatchSize,
            Set(assignments.map(\.id)).count == assignments.count,
            let policy = Memory.allocationFootprintPolicy()
        else {
            throw CBv2KVError.backendIneligible(
                reason: "GLM paged work is not bound to the active engine")
        }
        return try planWork(
            pool: backend.pool, assignments: assignments, states: states,
            layerForRow: layerForRow, policy: policy
        ).bytes
    }

    /// `prepaid`, when it holds exactly this step's price, is spent instead of
    /// a new reservation (`CBv2NativePagedMTPWork` prices its window early).
    func prepareWork(
        backend: PagedKVBackend, assignments: [(id: CBv2RequestID, range: Range<Int>)],
        states: [CBv2RequestID: [CBv2SequenceKV?]], layerForRow: (UInt64) -> Int?,
        prepaid: CBv2StepScratchPrepayment? = nil
    ) throws -> CBv2PagedAttentionStepOwner {
        let pool = backend.pool
        guard self.backend === backend, matches(targetIdentity), pool.activeAttentionWork == nil,
            pool.attentionWorkEnginePrepared, pool.attentionWorkEngineRefusal == nil,
            let admission = pool.memoryAdmission, admission.hasProcessMemoryOwner,
            let policy = Memory.allocationFootprintPolicy(),
            StreamOrDevice.default.stream == Stream.gpu,
            assignments.count > 0, assignments.count <= limits.maximumBatchSize,
            Set(assignments.map(\.id)).count == assignments.count,
            pool.attentionWorkOwners.values.filter({ !$0.completed }).count
                < limits.maximumInFlightGraphs
        else {
            throw CBv2KVError.backendIneligible(
                reason: "GLM paged work is not bound to the active engine")
        }
        let environment = CBv2PagedAttentionWorkEnvironment.capture()
        try environment.requireSupported()
        let plan = try planWork(
            pool: pool, assignments: assignments, states: states, layerForRow: layerForRow,
            policy: policy)
        let descriptors = plan.descriptors
        let allocations = plan.allocations
        let copyPages = plan.copyPages
        let bytes = plan.bytes
        func available() -> Int {
            max(0, limits.maximumScratchBytes - pool.attentionWorkBytesReserved)
        }
        // A native shape limit never fits later; only the shared scratch
        // ceiling is a deferrable, pre-mutation refusal.
        guard
            allocations.allSatisfy({
                $0.maximumLogicalBufferBytes <= pool.config.maxBufferLength
            })
        else {
            throw CBv2KVError.capacityExhausted(needed: bytes, available: available())
        }
        guard
            try CBv2PagedWorkMath.add(pool.attentionWorkBytesReserved, bytes)
                <= limits.maximumScratchBytes
        else {
            throw CBv2StepScratchRefusal(needed: bytes, available: available())
        }
        let (generation, overflow) = pool.attentionWorkGeneration.addingReportingOverflow(1)
        guard !overflow else {
            throw CBv2KVError.backendIneligible(reason: "GLM paged generation exhausted")
        }
        let reservation: CBv2CheckpointReservation
        if let prepaid, prepaid.bytes == bytes {
            reservation = prepaid.reservation
        } else {
            prepaid?.reservation.release()
            reservation = try pool.reserveStepScratch(
                bytes: bytes, requests: Set(assignments.map(\.id)), admission: admission)
        }
        // Price and commit every extra live destination before publishing the
        // ticket. Old row references survive until this step's fence, so they
        // cannot fund the clones, even when a row was rewound across pages.
        if !copyPages.isEmpty {
            do {
                try CBv2StepScratchRefusal.typing { try pool.reserve(copyPages) }
                do {
                    try CBv2StepScratchRefusal.typing { try pool.materializeReservedSegments() }
                } catch {
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
        resources.copies = plan.copies
        resources.readPlans = plan.readPlans
        resources.prepaidPages = copyPages
        stepResources[generation] = resources
        let scoped = CBv2CheckpointReservation { [weak self, resources] in
            resources.releaseAfterFence()
            self?.stepResources.removeValue(forKey: generation)
            reservation.release()
        }
        let work = CBv2PagedAttentionStepOwner(
            generation: generation, pool: pool, environment: environment,
            descriptors: descriptors, allocations: allocations, bytes: bytes, reservation: scoped,
            allocationPolicy: policy)
        pool.attentionWorkGeneration = generation
        pool.attentionWorkOwners[generation] = work
        pool.publishAttentionWorkCharge()
        pool.activeAttentionWork = work
        return work
    }

    func clonePage(
        _ copy: PagedSequenceKV.ImmutablePageCopy, row: PagedSequenceKV,
        work: CBv2PagedAttentionStepOwner
    ) throws -> Int32 {
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
            row.table.indices.contains(copy.logicalPage),
            row.table[copy.logicalPage] == copy.source.page,
            (0 ... backend.pool.config.pageSize).contains(copy.initializedTokens)
        else {
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
            let slots = (0 ..< copy.initializedTokens).map {
                page * Int32(pool.config.pageSize) + Int32($0)
            }
            pool.writeTokens(
                group: source.group, slots: slots,
                keys: copied.keys.squeezed(axis: 0), values: copied.values.squeezed(axis: 0))
        }
        // Transfer the old row reference, not an uncounted duplicate. It is
        // released only when the existing engine step drops its reservation.
        resources.retired.append(source)
        return page
    }

    /// Worst-case step price, checked once when the engine is assembled so
    /// an unservable envelope is refused before any request is admitted
    /// rather than failing mid-stream. `maximumRowTokens` bounds one row's
    /// query in a step (its prefill chunk, solo stripe or MTP verification
    /// width); `maximumStepTokens` bounds the whole step. The price is the
    /// split `stepScratch` setup bytes plus its per-token rate for the most
    /// history the step's rows can hold: every row at the context limit, or
    /// the pool's whole token capacity when that is smaller (each token of a
    /// row's history occupies a page in every layer). Per-row segment spread
    /// assumes the pool's size-class layout; a more fragmented live layout,
    /// or a pool grown after setup, is still priced exactly (and refused) per
    /// step. The native maximum-buffer shape limit is not checked here: each
    /// step still refuses an oversized rectangular transport.
    @discardableResult
    package func validateStepEnvelope(
        maximumRequests: Int, maximumStepTokens: Int, maximumRowTokens: Int,
        admissibleBytes: Int?
    ) throws -> Int {
        guard let pool = backend?.pool else {
            throw CBv2KVError.backendIneligible(reason: "GLM step envelope lacks its bound pool")
        }
        let scratch = try stepScratch(
            maximumRequests: maximumRequests, maximumStepTokens: maximumStepTokens,
            maximumRowTokens: maximumRowTokens)
        let tokenBytes = try kinds.indices.reduce(0) { total, index in
            try CBv2PagedWorkMath.add(
                total,
                CBv2PagedWorkMath.product([
                    kinds[index].headDim + kinds[index].valueHeadDim, pool.layerDTypes[index].size,
                ]))
        }
        let rows = min(maximumRequests, limits.maximumBatchSize)
        let history = min(
            pool.bytesCapacity / max(1, tokenBytes),
            try CBv2PagedWorkMath.product([rows, limits.maximumContextTokens]))
        // Both bound every step; the all-rows-at-context price is the smaller
        // one when a short context leaves allocator overhead dominant.
        let required = min(
            try CBv2PagedWorkMath.add(
                scratch.setupBytes,
                CBv2PagedWorkMath.product([scratch.bytesPerContextToken, history])),
            try worstCaseStepBytes(
                maximumRequests: maximumRequests, maximumStepTokens: maximumStepTokens,
                maximumRowTokens: maximumRowTokens))
        let ceiling = min(limits.maximumScratchBytes, admissibleBytes ?? Int.max)
        guard required <= ceiling else {
            throw CBv2KVError.capacityExhausted(needed: required, available: ceiling)
        }
        return required
    }

    package func worstCaseStepBytes(
        maximumRequests: Int, maximumStepTokens: Int, maximumRowTokens: Int
    ) throws -> Int {
        guard let pool = backend?.pool, maximumRequests > 0, maximumStepTokens > 0,
            maximumRowTokens > 0, let policy = Memory.allocationFootprintPolicy()
        else {
            throw CBv2KVError.backendIneligible(reason: "GLM step envelope lacks its bound pool")
        }
        let context = limits.maximumContextTokens
        let shapes = try stepShapes(
            maximumRequests: maximumRequests, maximumStepTokens: maximumStepTokens,
            maximumRowTokens: maximumRowTokens)
        let pageSize = pool.config.pageSize
        let pages = (context - 1) / pageSize + 1
        func step(_ queries: [Int]) throws -> GLM5NextStepBytes {
            var bytes = GLM5NextStepBytes(layers: kinds.count)
            try bytes.add(
                transportAllocations(
                    rows: queries.count, maximumQuery: queries.max()!,
                    maximumBufferLength: nil, policy: policy),
                layer: nil)
            for (index, kind) in kinds.enumerated() {
                // Full-size segments, one per smaller binary size class, and
                // one partial segment at each end of the row.
                let usable = max(
                    1,
                    pool.group(pool.groupKey(forLayer: index)).segmentLayout?.maximumUsablePages
                        ?? pages)
                let classes = Int.bitWidth - usable.leadingZeroBitCount
                let segments = min(pages, pages / usable + classes + 1)
                for q in queries {
                    try bytes.add(
                        workAllocations(
                            kind: kind, range: (context - q) ..< context, pageSize: pageSize,
                            segments: segments, copyPageCount: 0, policy: policy),
                        layer: index)
                }
            }
            return bytes
        }
        // Each shape beside its chained decode successor.
        let decode = try step(shapes.decode)
        return try [shapes.decode, shapes.packed, shapes.even].map {
            try step($0).merged(with: decode).total()
        }.max()!
    }

    func transportAllocations(
        rows: Int, maximumQuery: Int, maximumBufferLength: Int?, policy: AllocationFootprintPolicy
    ) throws -> [CBv2PagedAttentionAllocation] {
        var allocations: [CBv2PagedAttentionAllocation] = []
        for (index, kind) in kinds.enumerated() {
            for (role, width) in [("K", kind.headDim), ("V", kind.valueHeadDim)] {
                let logical = try CBv2PagedWorkMath.product([
                    rows, maximumQuery, kind.kvHeads, width, DType.float32.size,
                ])
                guard logical <= maximumBufferLength ?? logical else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "GLM batched \(role) transport at layer \(index) exceeds native maximum buffer length"
                    )
                }
                guard let bound = policy.upperBound(byteCount: logical), bound >= logical else {
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
                allocations.append(
                    .init(
                        role: "write.batch\(role).\(index)", logicalBytes: logical,
                        allocationCount: 1,
                        upperBoundBytes: bound, maximumLogicalBufferBytes: logical))
            }
        }
        return allocations
    }

    /// Distinct segments a row's reads and writes can touch this step: those
    /// of its current pages plus one per page it still has to acquire.
    private func segmentBound(row: PagedSequenceKV, through end: Int) -> Int {
        let pageSize = row.pool.config.pageSize
        let pages = (end - 1) / pageSize + 1
        guard let layout = row.pool.group(row.groupKey).segmentLayout else { return pages }
        let held = Set(row.table.map { layout.segmentIndex(page: $0) }).count
        return min(pages, held + max(0, pages - row.table.count))
    }

    /// Prices the buffers one row's DSA layer actually allocates. `n` is the
    /// history after this step's writes and `q` its query tokens. Retained
    /// latent reads are bounded by `2n` rows (`readsLatentHistory`) or by the
    /// selection rows of a decode-sized step; index scores scale with one
    /// query block, never `n*q`, but the pooled index planes scale with `n`.
    func workAllocations(
        kind: CBv2LayerKind, range: Range<Int>, pageSize: Int, segments: Int, copyPageCount: Int,
        policy: AllocationFootprintPolicy
    ) throws -> [CBv2PagedAttentionAllocation] {
        let q = range.count
        let n = range.upperBound
        let pages = (n - 1) / pageSize + 1
        let segments = max(1, min(segments, pages))
        let writePages = (range.lowerBound % pageSize + q - 1) / pageSize + 1
        let writeSegments = min(segments, writePages)
        guard let extra = policy.maximumExtraBytes else {
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        var result: [CBv2PagedAttentionAllocation] = []
        func buffers(_ role: String, _ factors: [Int], bytes: Int = 4, copies: Int = 1) throws {
            let logical = try CBv2PagedWorkMath.product(factors + [bytes])
            guard let bound = policy.upperBound(byteCount: logical) else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            result.append(
                .init(
                    role: role, logicalBytes: try CBv2PagedWorkMath.product([logical, copies]),
                    allocationCount: copies,
                    upperBoundBytes: try CBv2PagedWorkMath.product([bound, copies]),
                    maximumLogicalBufferBytes: logical))
        }
        // Writes and history reads are partitioned per touched segment,
        // including the minimum 24-word records and each allocation's
        // allocator overhead.
        for (role, count, partitions) in [("write", q, writeSegments), ("indexRead", n, segments)] {
            let logical = try CBv2PagedWorkMath.add(
                CBv2PagedWorkMath.product([count, 3, 4]),
                CBv2PagedWorkMath.product([partitions, 24, 4]))
            result.append(
                .init(
                    role: role + ".records", logicalBytes: logical, allocationCount: partitions,
                    upperBoundBytes: try CBv2PagedWorkMath.add(
                        logical, CBv2PagedWorkMath.product([partitions, extra])),
                    maximumLogicalBufferBytes: logical))
            try buffers(role + ".fences", [1], copies: partitions + 1)
            // Host slots (8/token), growing triples (36/token), padded
            // assembly (24/token), then per-bucket padding/map/page metadata.
            try buffers(role + ".hostRecords", [count, 68], bytes: 1)
            try buffers(role + ".hostBuckets", [pages, 456], bytes: 1)
            try buffers(role + ".shapes", [3], copies: partitions)
        }
        try buffers("write.contiguousK", [q, kind.headDim], copies: writeSegments)
        try buffers("write.contiguousV", [q, kind.valueHeadDim], copies: writeSegments)
        try buffers("index.V", [n, kind.valueHeadDim])
        // Validity, first-key, pool offset/index/clip and grouped-validity planes.
        try buffers("index.validity", [n], copies: 4)
        try buffers("index.poolMetadata", [n], copies: 6)
        let budget = selectionWidth - 3
        let heads = kind.queryHeads
        // Absorbed `[H,T,C]` queries and their FP32 copy for the whole step.
        try buffers("attention.queryLatents", [q, heads, kind.headDim], copies: 2)
        // Leading queries below `index_topk` attend the dense history prefix:
        // one retained selection-width read, then per block of at most the
        // attention bound an FP32 score and an equal probability block.
        let denseQueries = denseQueryCount(range: range)
        let sparse = q - denseQueries
        let plan = readPlan(range: range)
        let history = plan == .history
        if denseQueries > 0 {
            let keys = min(n, budget)
            let block = max(1, min(denseQueries, attentionBlockBytes / (heads * keys * 4)))
            // A history read serves the prefix as a slice.
            if !history { try buffers("dense.prefixRead", [selectionWidth, kind.headDim]) }
            try buffers("dense.scores", [block, heads, keys], copies: 2)
        }
        let reads: Int
        switch plan {
        case .history: reads = 1
        case .selectedRows(let rows): reads = rows
        }
        let poolCount = (n + 3) / 4
        if sparse > 0 {
            if budget / 4 >= poolCount {
                // Exact dense shortcut: every complete pool is selected, so no
                // index key is pooled and no score is formed. Per-query causal
                // eligibility `[Q,P]` and the cumulative visibility for tails.
                try buffers("index.denseEligibility", [sparse, poolCount], bytes: 1, copies: 2)
                try buffers("index.cumulative", [n])
            } else if sparse <= nativeSparseDecodeQueries {
                // Native decode selection pools in threadgroup memory: FP32
                // `[Q,P]` scores, complete-pool masks, the selector's scratch
                // rows and the cumulative visibility for tails.
                try buffers("index.decodeScores", [sparse, poolCount])
                try buffers("index.decodeCompletePools", [poolCount], bytes: 1, copies: 2)
                try buffers("index.decodeSelector", [sparse, min(poolCount, 64 * 512 - 1)])
                try buffers("index.cumulative", [n])
            } else {
                // Restored-dtype key/gate planes, their pool gathers, FP32 gate
                // logits/softmax/weights and the pooled-key product. MLX frees
                // each consumed intermediate, so at most this many coexist.
                try buffers("index.castAndPool", [n, indexDimension], copies: 6)
                // Scores for one query block `[Qb,H,P]` (GEMM output and the
                // weighted product), its reduced/masked `[Qb,P]` selector rows,
                // and per-query causal visibility over the history.
                let block = max(
                    1, min(sparse, scoreBlockBytes / max(1, indexHeads * poolCount * 4)))
                try buffers("index.score", [block, indexHeads, poolCount], copies: 2)
                try buffers("index.poolScores", [block, poolCount], copies: 3)
                try buffers("index.visibility", [block, n], bytes: 5)
            }
            // Selection rows (masked members, padding, tail concat, final
            // mask) and the tail-overlap comparison against each row.
            try buffers("selection.tokens", [sparse, selectionWidth], copies: 4)
            try buffers("selection.tailMatch", [sparse, 3, selectionWidth], bytes: 1)
            if history {
                // The step owner retains the one history read (in the latent
                // dtype, at most FP32), which the native kernels index in
                // place. Absorbed attention gathers bounded query blocks
                // `[Tb,W,C]` (and their FP32 copy) with `[Tb,H,W]` score, mask,
                // exp and probability planes, two blocks' worth overlapping;
                // the expanded reference gathers one query's row at a time.
                // Both write `[T,H,C]` latents.
                try buffers("selected.history", [n, kind.headDim])
                if absorbedAttention {
                    let block = max(
                        1, min(sparse, attentionBlockBytes / (selectionWidth * kind.headDim * 4)))
                    let blocks = min(2, (sparse + block - 1) / block)
                    try buffers(
                        "selected.blockLatents", [block, selectionWidth, kind.headDim],
                        copies: 2 * blocks)
                    try buffers(
                        "selected.blockScores", [block, heads, selectionWidth], copies: 4 * blocks)
                } else {
                    let overlap = min(sparse, 2)
                    try buffers(
                        "selected.attentionScores", [heads, selectionWidth], copies: 4 * overlap)
                    try buffers(
                        "selected.projectionInputs", [selectionWidth, kind.headDim],
                        copies: overlap)
                }
                try buffers("selected.attended", [sparse, heads, kind.headDim])
            } else {
                // Every sparse query's selected-latent read output is retained by
                // the step owner until its fence (the read writes the zero-filled
                // output in place, so that is one buffer per query).
                try buffers("selected.K", [selectionWidth, kind.headDim], copies: sparse)
                // Absorbed attention consumes each read in turn: FP32 score, mask,
                // exponentials and probabilities `[H,W]`, and one latent copy for
                // a dtype restore. MLX frees each query's intermediates before the
                // next, so two queries' worth bound the overlap.
                let overlap = min(sparse, 2)
                try buffers(
                    "selected.attentionScores", [heads, selectionWidth], copies: 4 * overlap)
                try buffers(
                    "selected.projectionInputs", [selectionWidth, kind.headDim], copies: overlap)
            }
        }
        // One flattened index row per read and one fence per touched segment
        // plus completion. The row's table and host plan are built once; each
        // table carries the history length as one extra entry.
        try buffers("selected.tables", [pages + 1])
        try buffers(
            "selected.fences", [1], copies: CBv2PagedWorkMath.product([reads, segments + 1]))
        try buffers("selected.indices", [selectionWidth], copies: max(1, reads))
        try buffers("selected.host", [pages, 128], bytes: 1)
        let copyCount = max(1, copyPageCount)
        try buffers(
            "cow.payload", [pageSize, kind.headDim + kind.valueHeadDim],
            copies: CBv2PagedWorkMath.product([4, copyCount]))
        try buffers(
            "cow.records", [max(24, pageSize * 3)],
            copies: CBv2PagedWorkMath.product([2, copyCount]))
        try buffers("cow.fences", [1], copies: CBv2PagedWorkMath.product([4, copyCount]))
        try buffers("cow.hostPlan", [pageSize, 128], bytes: 1, copies: copyCount)
        return result
    }
}

private final class GLM5NextPagedStepResources: @unchecked Sendable {
    let pool: PagedKVPool
    var rows: [PagedSequenceKV] = []
    var copies: [CBv2PagedAttentionTicketKey: [PagedSequenceKV.ImmutablePageCopy]] = [:]
    var readPlans: [CBv2PagedAttentionTicketKey: GLM5NextPagedModelBinding.ReadPlan] = [:]
    var prepaidPages: [PagedKVGroupKey: Int] = [:]
    var retired: [PagedKVPageHandle] = []
    init(pool: PagedKVPool) { self.pool = pool }
    func releaseAfterFence() {
        for handle in retired { pool.freePages(group: handle.group, pages: [handle.page]) }
        retired.removeAll()
        rows.removeAll()
        copies.removeAll()
        readPlans.removeAll()
        if !prepaidPages.isEmpty { pool.unreserve(prepaidPages) }
        prepaidPages.removeAll()
    }
}

extension PagedKVPool {
    var glm5NextPayloadBinding: GLM5NextPagedModelBinding? {
        attentionWorkCaches.values.compactMap { $0.value?.glm5NextModelBinding }.first
    }
}
