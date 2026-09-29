import Foundation
import MLX

/// Opt-in source envelope, not a measured peak or a paging capability claim.
/// The implementation fingerprint must be repinned with any native-core change.
public enum CBv2PagedAttentionWorkProfile: String, Sendable {
    case pinnedMetal
}

/// One independently allocated buffer, or a partition family whose bound is
/// sum(logical bytes) + allocationCount * allocator.maximumExtraBytes.
/// The latter is necessary for page-to-segment bucket partitions not yet chosen.
struct CBv2PagedAttentionAllocation: Equatable {
    let role: String
    let logicalBytes: Int
    let allocationCount: Int
    let upperBoundBytes: Int
    let maximumLogicalBufferBytes: Int
}

enum CBv2PagedWorkMath {
    static func add(_ a: Int, _ b: Int) throws -> Int {
        guard a >= 0, b >= 0, let value = CBv2KVGeometry.add(a, b) else {
            throw CBv2KVError.backendIneligible(reason: "paged attention work addition overflow")
        }
        return value
    }
    static func product(_ factors: [Int]) throws -> Int {
        try factors.reduce(1) { a, b in
            guard b >= 0, let value = CBv2KVGeometry.multiply(a, b) else {
                throw CBv2KVError.backendIneligible(reason: "paged attention work product overflow")
            }
            return value
        }
    }
}

struct CBv2PagedAttentionBlock: Equatable {
    let queryOffset: Int
    let queryCount: Int
    let visibleStart: Int
    let visibleEnd: Int
}

final class CBv2PagedAttentionWeakRow {
    weak var value: PagedSequenceKV?
    init(_ value: PagedSequenceKV?) { self.value = value }
}

struct CBv2PagedAttentionRowDescriptor {
    let requestID: CBv2RequestID
    let row: CBv2PagedAttentionWeakRow
    let rowSerial: UInt64
    let layerIndex: Int
    let ownerLayerIndex: Int
    let range: Range<Int>
    let baseOffset: Int
    let frozenHighWater: Int
    let gatheredRange: Range<Int>?
    let keyStart: Int
    let keyCount: Int
    let blocks: [CBv2PagedAttentionBlock]
    let softcap: Bool
    var writes: Bool { layerIndex == ownerLayerIndex }
}

/// Immutable execution inputs. Environment overrides are not silently clamped:
/// the first profile covers the pinned backend defaults only.
struct CBv2PagedAttentionWorkEnvironment: Equatable {
    let queryBlockSize: Int
    let sdpaBlocks: String?

    static func capture() -> Self {
        .init(
            queryBlockSize: CBv2AttentionV1.queryBlockSize,
            sdpaBlocks: ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"])
    }

    func requireSupported() throws {
        guard queryBlockSize >= 0, sdpaBlocks == nil else {
            throw CBv2KVError.backendIneligible(
                reason: "step-owned paging requires the pinned default MLX_SDPA_BLOCKS profile")
        }
    }
}

/// Source-derived upper bounds for one actual layer/row call. Sum every block
/// and both possible native dispatch families; no max-across-layers discount.
enum CBv2PagedAttentionWorkProjection {
    static func allocations(
        kind: CBv2LayerKind, dtype: DType, descriptor d: CBv2PagedAttentionRowDescriptor,
        pageSize: Int, policy: AllocationFootprintPolicy
    ) throws -> [CBv2PagedAttentionAllocation] {
        guard pageSize > 0, let extra = policy.maximumExtraBytes,
            kind.queryHeads > 0, kind.kvHeads > 0,
            kind.queryHeads % kind.kvHeads == 0,
            [.float16, .bfloat16, .float32].contains(dtype)
        else {
            throw CBv2KVError.backendIneligible(reason: "unbound paged work allocator/geometry")
        }
        let h = kind.queryHeads
        let hk = kind.kvHeads
        let dk = kind.headDim
        let dv = kind.valueHeadDim
        let q = d.range.count
        var result: [CBv2PagedAttentionAllocation] = []
        func buffer(_ role: String, _ factors: [Int], bytes: Int = 4, copies: Int = 1) throws {
            let logical = try CBv2PagedWorkMath.product(factors + [bytes])
            guard copies > 0, let bound = policy.upperBound(byteCount: logical) else {
                throw CBv2KVError.backendIneligible(reason: "paged work allocation bound overflow")
            }
            // Keep the independent allocation multiplicity, not one B-sized pad.
            result.append(
                .init(
                    role: role,
                    logicalBytes: try CBv2PagedWorkMath.product([logical, copies]),
                    allocationCount: copies,
                    upperBoundBytes: try CBv2PagedWorkMath.product([bound, copies]),
                    maximumLogicalBufferBytes: logical))
        }
        func batchPartition(_ role: String, _ factors: [Int], bytes: Int) throws {
            let logical = try CBv2PagedWorkMath.product(factors + [bytes])
            // This row can belong to one larger batch allocation. For any
            // grouping B(total)<=sum(logical)+extra; charging extra per row
            // does not assume subadditivity of allocator rounding.
            result.append(
                .init(
                    role: role, logicalBytes: logical, allocationCount: 1,
                    upperBoundBytes: try CBv2PagedWorkMath.add(logical, extra),
                    maximumLogicalBufferBytes: logical))
        }
        func transfers(_ role: String, count: Int, start: Int, writes: Bool) throws {
            guard count > 0 else { return }
            let span = try CBv2PagedWorkMath.add(start % pageSize, count)
            let pages = try CBv2PagedWorkMath.add(span, pageSize - 1) / pageSize
            // A bucket owns <= all tokens; every bucket has >=24 Int32 entries.
            // For any partition, sum max(24,3*n_i) <= 3*N + 24*bucketCount.
            // Each independent native record allocation pays its own extra.
            let logical = try CBv2PagedWorkMath.add(
                CBv2PagedWorkMath.product([count, 3, 4]),
                CBv2PagedWorkMath.product([pages, 24, 4]))
            result.append(
                .init(
                    role: role + ".records", logicalBytes: logical,
                    allocationCount: pages,
                    upperBoundBytes: try CBv2PagedWorkMath.add(
                        logical,
                        CBv2PagedWorkMath.product([pages, extra])),
                    maximumLogicalBufferBytes: logical))
            try buffer(
                role + ".fences", [1], copies: try CBv2PagedWorkMath.add(pages, writes ? 0 : 1))
            if writes {
                // ensureRowContiguous may copy EACH bucket's projected inputs.
                // Broadcast syntax does not prove these native copies absent.
                try buffer(role + ".keyContiguous", [hk, q, dk], bytes: dtype.size, copies: pages)
                try buffer(
                    role + ".valueContiguous", [hk, q, dv], bytes: dtype.size, copies: pages)
            }
        }
        if let gathered = d.gatheredRange {
            try buffer("gather.keys", [hk, gathered.count, dk], bytes: dtype.size)
            try buffer("gather.values", [hk, gathered.count, dv], bytes: dtype.size)
            try transfers(
                "gather", count: gathered.count, start: gathered.lowerBound, writes: false)
        }
        if d.writes && d.range.lowerBound >= d.frozenHighWater {
            try transfers("write", count: q, start: d.range.lowerBound, writes: true)
        }
        if q > 1 && d.writes && d.frozenHighWater <= d.range.lowerBound {
            // Both history-concat and no-history retained projection paths get
            // compact owner outputs. Their parents never escape the step.
            try buffer("prefill.keys", [hk, d.keyCount, dk], bytes: dtype.size)
            try buffer("prefill.values", [hk, d.keyCount, dv], bytes: dtype.size)
        }
        try batchPartition("position.offsetBind", [1], bytes: 4)
        try batchPartition("position.offsetAdvance", [1], bytes: 4)
        try buffer("control.rowScalars", [1], copies: 8)
        if q > 1 {
            try buffer("position.queries", [q])
            try buffer("position.keys", [d.keyCount])
        }
        if kind.hasSinks { try buffer("sinks", [h], copies: 2) }

        for (index, block) in d.blocks.enumerated() {
            let n = block.visibleEnd - block.visibleStart
            let bq = block.queryCount
            let p = "block\(index)"
            // scale, where floor, optional cap divide/multiply/window offset,
            // and cast/fill control inputs. Independent scalar buffers too.
            try buffer(p + ".scalars", [1], copies: 8)
            let withSink = try CBv2PagedWorkMath.add(n, kind.hasSinks ? 1 : 0)
            let collapsedRows = try CBv2PagedWorkMath.product([h, bq])
            let columns = max(withSink, max(dk, dv))
            let rowTiles = try CBv2PagedWorkMath.add(collapsedRows, 15) / 16
            let columnTiles = try CBv2PagedWorkMath.add(columns, 15) / 16
            // Matmul's branch-selection scalars are Int32 even when tensor
            // byte sizes are UInt64: bound its worst batch collapse, 3*max,
            // and tile product. Do NOT cap tensor elements to Int32.
            guard collapsedRows <= Int(Int32.max) / 3, columns <= Int(Int32.max) / 3,
                try CBv2PagedWorkMath.product([bq, columns]) <= Int(Int32.max),
                try CBv2PagedWorkMath.product([rowTiles, columnTiles]) <= Int(Int32.max)
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "actual attention shape overflows pinned native dispatch scalars")
            }
            if q > 1 {
                try buffer(p + ".mask.causal", [bq, n], bytes: 1)
                if case .slidingWindow = kind.attention {
                    try buffer(p + ".mask.window", [bq, n], bytes: 1, copies: 2)
                    try buffer(p + ".mask.queryShift", [bq])
                }
            }
            // Ordinary fast.cpp fallback: grouped GQA is views, NOT repeat.
            // Softcap's Swift composed route really casts and repeats K/V.
            try buffer(p + ".scaledQuery", [h, bq, dk], copies: d.softcap ? 2 : 1)
            if d.softcap {
                try buffer(p + ".castKeys", [hk, n, dk])
                try buffer(p + ".castValues", [hk, n, dv])
                if h > hk {
                    try buffer(p + ".repeatKeys", [h, n, dk])
                    try buffer(p + ".repeatValues", [h, n, dv])
                }
            }
            try buffer(p + ".scores", [h, bq, n], copies: d.softcap ? 4 : 1)
            if q > 1 { try buffer(p + ".maskedScores", [h, bq, n]) }
            if kind.hasSinks { try buffer(p + ".augmentedScores", [h, bq, withSink]) }
            // softmax.cpp either allocates or copies one output; do not assume donation.
            try buffer(p + ".probabilities", [h, bq, withSink])
            try buffer(p + ".output", [h, bq, dv], copies: d.softcap ? 2 : 1)

            // matmul.cpp check_transpose can copy each *broadcasted* operand.
            // These are backend contingencies, not fictitious source repeat ops.
            try buffer(p + ".qk.copyQ", [h, bq, dk])
            try buffer(p + ".qk.copyK", [h, n, dk])
            try buffer(p + ".pv.copyP", [h, bq, n])
            try buffer(p + ".pv.copyV", [h, n, dv])
            // Split-K is only reachable with batch_size_out==1. If heads
            // collapse at all, Matmul collapses the whole batch into M=Hq*q.
            // Price both hardware alternatives ONLY when their pinned branch
            // predicates can hold; a small DK cannot split a huge QK matrix.
            for (name, inner, columns) in [("qk", dk, n), ("pv", n, dv)] {
                let maximum = max(collapsedRows, columns)
                let tiles = try CBv2PagedWorkMath.product([
                    CBv2PagedWorkMath.add(collapsedRows, 15) / 16,
                    CBv2PagedWorkMath.add(columns, 15) / 16,
                ])
                if min(collapsedRows, columns) > 1 && tiles <= 2048 && inner / 16 >= 8
                    && inner >= maximum
                {
                    try buffer(p + "." + name + ".simdSplitK", [collapsedRows, columns, 32])
                }
                let triple = try CBv2PagedWorkMath.product([maximum, 3])
                let double = try CBv2PagedWorkMath.product([maximum, 2])
                if min(collapsedRows, columns) > 1
                    && (inner >= triple || (maximum <= 1024 && inner > double))
                {
                    let stride =
                        inner <= 1024
                        ? inner / 2 : (inner <= 2048 ? 1024 : (inner <= 4096 ? 2048 : 4096))
                    guard stride > 0 else {
                        throw CBv2KVError.backendIneligible(reason: "invalid native split-K stride")
                    }
                    let partitions = try CBv2PagedWorkMath.add(inner, stride - 1) / stride
                    try buffer(p + "." + name + ".naxSplitK", [collapsedRows, columns, partitions])
                }
            }
            if h == 1 && bq == 1 && (n == 1 || dv == 1) {
                let inner = max(n == 1 ? dk : 0, dv == 1 ? n : 0)
                let partials = try CBv2PagedWorkMath.add(inner, 16383) / 16384
                try buffer(p + ".dotPartials", [partials], copies: 2)
                try buffer(p + ".dotResult", [1], copies: 2)
            }

            if !d.softcap && dk == 192 && dv == 128 && bq <= 8
                && bq <= 32 / (h / hk) && bq <= n
            {
                // Pinned SDPA vector dispatch defaults choose at most1024 blocks.
                // A nonnil MLX_SDPA_BLOCKS was refused/revalidated separately.
                // Price all copy candidates and all three FP32 temporaries,
                // for shapes which can actually select this asymmetric vector route.
                try buffer(p + ".fused.copyQ", [h, bq, dk], bytes: dtype.size)
                try buffer(p + ".fused.copyK", [hk, n, dk], bytes: dtype.size)
                try buffer(p + ".fused.copyV", [hk, n, dv], bytes: dtype.size)
                if q > 1 { try buffer(p + ".fused.copyMask", [bq, n], bytes: 1) }
                if n >= 1024 {
                    let partitions = n <= 8192 ? 128 : (n < 16384 ? 256 : (n < 65536 ? 512 : 1024))
                    try buffer(p + ".fused.partials", [h, bq, partitions, dv])
                    try buffer(p + ".fused.sumsMaxs", [h, bq, partitions], copies: 2)
                }
            }
        }
        if d.blocks.count > 1 { try buffer("output.blockConcat", [h, q, dv], bytes: dtype.size) }
        // The final batch concat exists only for packed rows, but pricing one
        // per row is a partition-safe upper bound, including allocator padding.
        try batchPartition("output.batchConcat", [h, q, dv], bytes: dtype.size)
        return result
    }
}

struct CBv2PagedAttentionTicketKey: Hashable {
    let layer: Int
    let serial: UInt64
}

/// This owner has C only. BufferInfo is validation, never an allocation ID/M credit.
/// Mutable state is confined to the engine queue; it is intentionally not Sendable.
final class CBv2PagedAttentionStepOwner {
    let generation: UInt64
    let poolIdentity: ObjectIdentifier
    let environment: CBv2PagedAttentionWorkEnvironment
    let descriptors: [CBv2PagedAttentionTicketKey: CBv2PagedAttentionRowDescriptor]
    let allocations: [CBv2PagedAttentionAllocation]
    let reservedBytes: Int
    let constructionStream: MLX.Stream
    private let allocationPolicy: AllocationFootprintPolicy
    private var reservation: CBv2CheckpointReservation?
    private var nativeOperation: CBv2NativePagedOperation?
    private var nativeRetirementQueued = false
    private var roots: [MLXArray] = []
    private var compactRoots: [MLXArray] = []
    private var consumed: Set<CBv2PagedAttentionTicketKey> = []
    private var writes: Set<CBv2PagedAttentionTicketKey> = []
    private var reads: Set<CBv2PagedAttentionTicketKey> = []
    private var loans = 0
    private var graphClosed = false
    private(set) var completionFailed = false
    private(set) var completed = false
    private(set) var released = false
    private(set) var published = false
    private var sealed = false
    weak var pool: PagedKVPool?
    /// Per-owner failure-order witness, nil in production; never substitutes a
    /// successful completion receipt or grants permission to refund.
    var beforeCompletionDrainForTesting: (() throws -> Void)?

    init(
        generation: UInt64, pool: PagedKVPool,
        environment: CBv2PagedAttentionWorkEnvironment,
        descriptors: [CBv2PagedAttentionTicketKey: CBv2PagedAttentionRowDescriptor],
        allocations: [CBv2PagedAttentionAllocation], bytes: Int,
        reservation: CBv2CheckpointReservation, allocationPolicy: AllocationFootprintPolicy,
        nativeOperation: CBv2NativePagedOperation? = nil
    ) {
        self.generation = generation
        self.pool = pool
        poolIdentity = ObjectIdentifier(pool)
        self.environment = environment
        self.descriptors = descriptors
        self.allocations = allocations
        reservedBytes = bytes
        self.reservation = reservation
        constructionStream = StreamOrDevice.default.stream
        self.allocationPolicy = allocationPolicy
        self.nativeOperation = nativeOperation
    }

    func consume(layer: Int, rows: [PagedSequenceKV], queries: Int, softcap: Bool) -> Bool {
        guard !sealed, !graphClosed, environment == .capture(),
            constructionStream == StreamOrDevice.default.stream,
            !rows.isEmpty
        else { return false }
        let keys = rows.map { CBv2PagedAttentionTicketKey(layer: layer, serial: $0.serial) }
        guard Set(keys).count == keys.count else { return false }
        for (row, key) in zip(rows, keys) {
            guard let d = descriptors[key], !consumed.contains(key),
                d.row.value === row, !row.isReleased,
                ObjectIdentifier(row.pool) == poolIdentity,
                d.range.count == queries, d.softcap == softcap,
                d.baseOffset == row.baseOffset, d.frozenHighWater == row.frozenHighWater,
                row.absoluteOffset == (d.writes ? d.range.lowerBound : d.range.upperBound)
            else { return false }
        }
        consumed.formUnion(keys)
        return true
    }

    func authorizeWrite(layer: Int, row: PagedSequenceKV, count: Int) -> Bool {
        let key = CBv2PagedAttentionTicketKey(layer: layer, serial: row.serial)
        guard !published, !graphClosed, environment == .capture(),
            let d = descriptors[key], d.writes, consumed.contains(key),
            !writes.contains(key), d.row.value === row, d.range.count == count,
            row.absoluteOffset == d.range.lowerBound
        else { return false }
        writes.insert(key)
        return true
    }

    func authorizeRead(layer: Int, row: PagedSequenceKV, start: Int, count: Int) -> Bool {
        let key = CBv2PagedAttentionTicketKey(layer: layer, serial: row.serial)
        guard !published, !graphClosed, environment == .capture(),
            let d = descriptors[key], consumed.contains(key), !reads.contains(key),
            d.row.value === row, let range = d.gatheredRange,
            range.lowerBound == start, range.count == count
        else { return false }
        reads.insert(key)
        return true
    }

    func retainRoots(_ arrays: [MLXArray]) {
        roots.append(contentsOf: arrays)
        for array in arrays { nativeOperation?.retain(array) }
    }
    func retainCompactRoots(_ arrays: [MLXArray]) {
        compactRoots.append(contentsOf: arrays)
        roots.append(contentsOf: arrays)
        for array in arrays { nativeOperation?.retain(array) }
    }
    var evaluationTargets: [MLXArray] { roots }

    func seal() throws {
        guard consumed.count == descriptors.count,
            descriptors.allSatisfy({ key, d in
                (!d.writes || writes.contains(key))
                    && (d.gatheredRange == nil || reads.contains(key))
            }), environment == .capture()
        else {
            throw CBv2KVError.backendIneligible(
                reason: "paged attention step did not consume its exact sealed work")
        }
        sealed = true
    }

    func publish() { published = true }

    func makeLoan(
        keys: MLXArray, values: MLXArray, layer: Int, serial: UInt64,
        range: Range<Int>
    ) -> CBv2PagedAttentionLoan {
        loans += 1
        return .init(
            owner: self, keys: keys, values: values,
            layer: layer, serial: serial, range: range)
    }

    /// Fence every registered transfer/output root and drain the exact stream.
    /// This ends future work, not physical ownership: callback-retained buffers
    /// remain in allocator active/cache accounting. Never issue M credit here.
    /// On error, do not drop roots, completion state, or C.
    func finishEvaluation() throws {
        guard !completionFailed else {
            throw CBv2KVError.backendIneligible(
                reason: "paged attention completion previously failed")
        }
        guard !completed else { return }
        do {
            try withError { fault in
                eval(roots)
                CBv2CoreInstrumentation.recordHostSync()
                try beforeCompletionDrainForTesting?()
                constructionStream.synchronize()
                CBv2CoreInstrumentation.recordHostSync()
                try fault.check()
            }
            try nativeOperation?.requiredDrain()
            for root in compactRoots {
                guard let info = try root.evaluatedBufferInfo(), info.dataOffset == 0,
                    info.isRowContiguous, info.dataElements == root.nbytes / root.dtype.size,
                    let bound = allocationPolicy.upperBound(byteCount: root.nbytes),
                    info.allocatedBytes <= bound
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "retained paged producer root is not a bounded compact destination")
                }
            }
            completed = true
            compactRoots.removeAll()
            roots.removeAll()
            releaseIfFinished()
        } catch {
            failCompletion()
            throw error
        }
    }

    /// Failed graph locals must already have unwound, its group fences restored,
    /// and its cache bindings dropped. NEVER evaluate the failed graph here.
    func discardAfterDrain() throws {
        guard !completionFailed else {
            throw CBv2KVError.backendIneligible(
                reason: "paged attention completion previously failed")
        }
        guard !completed else { return }
        do {
            try withError { fault in
                try beforeCompletionDrainForTesting?()
                constructionStream.synchronize()
                CBv2CoreInstrumentation.recordHostSync()
                Stream.cpu.synchronize()
                CBv2CoreInstrumentation.recordHostSync()
                try fault.check()
            }
            try nativeOperation?.requiredDrain()
            roots.removeAll()
            compactRoots.removeAll()
            completed = true
            graphClosed = true
            releaseIfFinished()
        } catch {
            failCompletion()
            throw error
        }
    }

    func failCompletion() {
        completionFailed = true
        nativeOperation?.fail()
        pool?.attentionWorkEngineRefusal = "paged attention completion failed; rebuild the engine"
    }

    func closeGraph() {
        graphClosed = true
        releaseIfFinished()
    }

    fileprivate func closeLoan() {
        precondition(loans > 0)
        loans -= 1
        releaseIfFinished()
    }

    private func releaseIfFinished() {
        guard !completionFailed, completed, graphClosed, loans == 0, !released else { return }
        if let nativeOperation {
            guard !nativeRetirementQueued else { return }
            nativeRetirementQueued = true
            // closeGraph can be called under finalize's metadata commit.
            // Queue the actual detach/refund on the SAME engine queue.
            nativeOperation.enqueueRetirement { [self, nativeOperation] in
                guard !completionFailed, completed, graphClosed, loans == 0, !released else {
                    return
                }
                nativeOperation.finish {
                    released = true
                    roots.removeAll()
                    compactRoots.removeAll()
                    let lease = reservation
                    reservation = nil
                    lease?.release()
                    pool?.forgetAttentionWork(generation)
                    self.nativeOperation = nil
                }
            }
            return
        }
        released = true
        let lease = reservation
        reservation = nil
        roots.removeAll()
        lease?.release()
        pool?.forgetAttentionWork(generation)
    }

    deinit {
        // A failed/unknown stream is not a refund signal. Keep the typed permit
        // and native roots alive even if its engine/pool is being destroyed.
        if let reservation {
            CBv2PagedAttentionQuarantine.retain(
                reservation: reservation, roots: roots + compactRoots)
        }
    }
}

/// Owns its arrays explicitly so they are disposed BEFORE releasing the loan.
/// Borrower graphs are also retained by the step until its completion fence.
final class CBv2PagedAttentionLoan {
    private var owner: CBv2PagedAttentionStepOwner?
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    let generation: UInt64
    let layer: Int
    let serial: UInt64
    let range: Range<Int>

    fileprivate init(
        owner: CBv2PagedAttentionStepOwner, keys: MLXArray, values: MLXArray,
        layer: Int, serial: UInt64, range: Range<Int>
    ) {
        self.owner = owner
        self.keys = keys
        self.values = values
        generation = owner.generation
        self.layer = layer
        self.serial = serial
        self.range = range
    }

    func close() {
        keys = nil
        values = nil
        let old = owner
        owner = nil
        old?.closeLoan()
    }

    deinit { close() }
}

/// Rare fault containment, not a timer or reusable admission credit. Unknown
/// completion is intentionally retained until process teardown. All access to
/// this global escape hatch is locked; owners themselves stay queue-confined.
private enum CBv2PagedAttentionQuarantine {
    private struct Entry {
        let reservation: CBv2CheckpointReservation
        let roots: [MLXArray]
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [Entry] = []
    static func retain(reservation: CBv2CheckpointReservation, roots: [MLXArray]) {
        lock.withLock { entries.append(.init(reservation: reservation, roots: roots)) }
    }
}

final class CBv2PagedAttentionWeakCache {
    weak var value: PagedLayerCache?
    init(_ value: PagedLayerCache) { self.value = value }
}

extension PagedKVPool {
    func forgetAttentionWork(_ generation: UInt64) {
        attentionWorkOwners.removeValue(forKey: generation)
        publishAttentionWorkCharge()
    }

    func endAttentionConstruction(_ owner: CBv2PagedAttentionStepOwner?) {
        guard let owner, activeAttentionWork === owner else { return }
        activeAttentionLayer = nil
        activeAttentionWork = nil
    }

    /// Called only after the existing failure boundary restored group fences,
    /// dropped cache aliases and unwound the failed model-call locals.
    func failUnpublishedAttentionWorkCompletion() {
        for owner in attentionWorkOwners.values where !owner.published { owner.failCompletion() }
    }

    /// A failed required drain is sticky; no later successful wait can refund it.
    func discardUnpublishedAttentionWorkAfterDrain() {
        activeAttentionLayer = nil
        activeAttentionWork = nil
        for owner in Array(attentionWorkOwners.values) where !owner.published {
            // A failed drain deliberately leaves this owner in the registry;
            // pool teardown transfers it to the fail-closed quarantine.
            try? owner.discardAfterDrain()
        }
    }

    func authorizeAttentionWrite(row: PagedSequenceKV, count: Int) -> Bool {
        guard usesStepOwnedAttention else { return true }
        guard let layer = activeAttentionLayer, let work = activeAttentionWork,
            work.authorizeWrite(layer: layer, row: row, count: count)
        else {
            return writeValidation.refuse(
                "missing/stale/duplicate step-owned paged write ticket",
                expected: row.groupKey.dtype)
        }
        return true
    }

    func authorizeAttentionRead(row: PagedSequenceKV, start: Int, count: Int) -> Bool {
        guard usesStepOwnedAttention else { return true }
        guard let layer = activeAttentionLayer, let work = activeAttentionWork,
            work.authorizeRead(layer: layer, row: row, start: start, count: count)
        else {
            return writeValidation.refuse(
                "missing/stale/duplicate step-owned paged read ticket",
                expected: row.groupKey.dtype)
        }
        return true
    }

    func refuseUnplannedAttentionMutation(_ operation: String, dtype: DType) -> Bool {
        guard usesStepOwnedAttention else { return false }
        writeValidation.refuse("step-owned paging does not permit " + operation, expected: dtype)
        return true
    }
}
