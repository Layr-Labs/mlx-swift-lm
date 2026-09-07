import MLX

/// Owned by exactly one submitted engine step, never cleared by a different
/// step's finalizer. The completion root is also in the group's ordinary fence
/// chain before asyncEval; the lease does not secretly submit work.
final class PagedQuantizedScratchLease {
    private var reservation: CBv2CheckpointReservation?
    private var roots: [MLXArray] = []
    private var completion: MLXArray?
    private var onFinish: (() -> Void)?
    let reservedBytes: Int

    init(reservation: CBv2CheckpointReservation?, bytes: Int, onFinish: (() -> Void)? = nil) {
        self.reservation = reservation
        self.reservedBytes = bytes
        self.onFinish = onFinish
    }

    var evaluationTargets: [MLXArray] { completion.map { [$0] } ?? [] }

    func retain(_ arrays: [MLXArray]) { roots = arrays }
    func retainAdditional(_ arrays: [MLXArray]) { roots.append(contentsOf: arrays) }
    func retainCompletion(_ array: MLXArray) { completion = array }

    /// Caller has completed this step's GPU work and all diagnostic readbacks.
    func finishAfterSynchronization() {
        completion = nil
        roots = []
        reservation = nil
        onFinish?()
        onFinish = nil
    }
}

/// A prefill call reuses these exact allocations for its query blocks. A
/// completion kernel after each merge fences the next overwrite. No allocation
/// contains dequantized history. Native compute/output arrays remain separate.
final class PagedQuantizedAttentionWorkspace {
    let partials: MLXArray
    let meta: MLXArray
    let partitionTokens: Int
    let maximumPartitions: Int
    let maximumQueries: Int
    let lease: PagedQuantizedScratchLease
    private let sharedArena: PagedQuantizedStepArena?

    init(pool: PagedKVPool, group: PagedKVGroup, queryCount: Int, blockSize: Int,
         queryHeads: Int, maxAttendLength: Int, maximumSegmentCount: Int? = nil,
         broadcastTopology: Bool = false, nativeOutputBytes: Int = 0,
         nativeWriteBytes: Int = 0) throws {
        let d = group.key.headDim
        let scoped = pool.quantizedScratchScope != nil
        let total = try Self.reservationBytes(
            queryCount: queryCount, blockSize: blockSize, queryHeads: queryHeads,
            headDim: d, pageSize: group.pageSize, maxAttendLength: maxAttendLength,
            maximumSegmentCount: maximumSegmentCount ?? group.segments.count,
            broadcastTopology: broadcastTopology, nativeOutputBytes: nativeOutputBytes,
            nativeWriteBytes: nativeWriteBytes, includeArena: !scoped)
        let permit = try pool.memoryAdmission?.reserveWorkspace(bytes: total)
        lease = PagedQuantizedScratchLease(reservation: permit, bytes: total)
        // The per-call permit also covers host construction before the shared
        // arena is reserved. Its independent lease is appended exactly once.
        sharedArena = try pool.quantizedScratchScope?.arena(
            pool: pool, group: group, queryHeads: queryHeads,
            blockSize: blockSize, maxAttendLength: maxAttendLength)
        if let sharedArena {
            partitionTokens = sharedArena.partitionTokens
            maximumPartitions = sharedArena.maximumPartitions
            maximumQueries = sharedArena.maximumQueries
            partials = sharedArena.partials
            meta = sharedArena.meta
        } else {
            partitionTokens = PagedSegmentDispatchPlan.boundedPartitionTokens(
                PagedAttentionKernel.partitionTokens, pageSize: group.pageSize)
            maximumPartitions = (maxAttendLength - 1) / partitionTokens + 1
            maximumQueries = blockSize
            let metaElements = max(8, try Self.product([blockSize, queryHeads, maximumPartitions, 2]))
            partials = MLXArray.zeros([blockSize, queryHeads, maximumPartitions, d], dtype: .float32)
            meta = MLXArray.zeros([metaElements], dtype: .float32)
            lease.retain([partials, meta])
        }
        pool.appendQuantizedScratch(lease)
    }

    func acquire(after previous: MLXArray) -> MLXArray {
        sharedArena?.acquire(after: previous) ?? previous
    }

    func recordCompletion(_ fence: MLXArray) {
        sharedArena?.recordCompletion(fence)
        lease.retainCompletion(fence)
    }

    static func sharedArenaBytes(blockSize: Int, queryHeads: Int, headDim: Int,
                                 pageSize: Int, maxAttendLength: Int,
                                 allocationPolicy: AllocationFootprintPolicy? = nil) throws -> Int {
        guard blockSize > 0, queryHeads > 0, headDim > 0, pageSize > 0, maxAttendLength > 0,
              let policy = allocationPolicy ?? Memory.allocationFootprintPolicy() else {
            throw CBv2KVError.backendIneligible(reason: "invalid shared packed arena bound")
        }
        let partitionTokens = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: pageSize)
        let partitions = (maxAttendLength - 1) / partitionTokens + 1
        let partialBytes = try product([blockSize, queryHeads, partitions, headDim, 4])
        let metaBytes = try product([max(8, product([blockSize, queryHeads, partitions, 2])), 4])
        guard let partial = policy.upperBound(byteCount: partialBytes),
              let meta = policy.upperBound(byteCount: metaBytes) else {
            throw CBv2KVError.backendIneligible(reason: "shared packed arena byte overflow")
        }
        return try sum([partial, meta])
    }

    static func reservationBytes(queryCount: Int, blockSize: Int, queryHeads: Int,
                                 headDim: Int, pageSize: Int, maxAttendLength: Int,
                                 maximumSegmentCount: Int,
                                 allocationPolicy: AllocationFootprintPolicy? = nil,
                                 broadcastTopology: Bool = false,
                                 nativeOutputBytes: Int = 0, nativeWriteBytes: Int = 0,
                                 includeArena: Bool = true) throws -> Int {
        guard queryCount > 0, blockSize > 0, blockSize <= queryCount, maxAttendLength > 0 else {
            throw CBv2KVError.backendIneligible(reason: "invalid packed attention workspace")
        }
        let partitionTokens = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: pageSize)
        let maximumPartitions = (maxAttendLength - 1) / partitionTokens + 1
        let d = headDim
        let partialElements = try Self.product([blockSize, queryHeads, maximumPartitions, d])
        let metaElements = max(8, try Self.product([blockSize, queryHeads, maximumPartitions, 2]))
        guard let policy = allocationPolicy ?? Memory.allocationFootprintPolicy() else {
            throw CBv2KVError.backendIneligible(reason: "allocator footprint policy unavailable")
        }
        func bound(_ bytes: Int) throws -> Int {
            guard let result = policy.upperBound(byteCount: bytes) else {
                throw CBv2KVError.backendIneligible(reason: "scratch allocator bound overflow")
            }
            return result
        }
        let partialBytes = includeArena ? try bound(Self.product([partialElements, 4])) : 0
        let metaBytes = includeArena ? try bound(Self.product([metaElements, 4])) : 0
        let blocks = (queryCount - 1) / blockSize + 1
        // A broadcast prefill has ONE physical row's topology for all query
        // columns and blocks. Only causal seqinfo changes between blocks.
        let records = try Self.product([broadcastTopology ? 1 : queryCount, maximumPartitions])
        let buckets = maximumSegmentCount <= PagedSegmentDispatchPlan.maximumBindings
            ? (broadcastTopology ? 1 : blocks) : records
        let recordBytes = try Self.product([records, PagedSegmentDispatchPlan.recordStride, 4])
        let offsetBytes = try Self.product([buckets, 17, 8])
        let metadata = try Self.sum([bound(recordBytes), bound(offsetBytes)])
        let dispatches = try Self.product([buckets, broadcastTopology ? blocks : 1])
        // Scoped users additionally join the prior arena user's merge fence.
        let fenceCount = try Self.sum([dispatches, Self.product([includeArena ? 3 : 4, blocks]), 2])
        let fences = try Self.product([fenceCount, try bound(4)])
        let seqinfo = try Self.product([blocks, try bound(Self.product([blockSize, 8, 4]))])
        // Metadata copies/host planning remain charged independently of GPU
        // arrays. Query rotation lives in threadgroup memory, not a tensor.
        // Host topology, bucket assembly, flattening and array/view objects.
        // GPU records/offsets each use one arena, priced separately above.
        let hostRecords = try Self.product([8, Self.sum([recordBytes, offsetBytes])])
        let output = try bound(nativeOutputBytes)
        // Transfer kernels may materialize noncontiguous input K/V. This bound
        // includes both copies plus records, per-bucket padding and write fences.
        let writeRecords = try Self.product([queryCount, 3, 4, 8])
        // Decode may issue one write call per row even in the same segment.
        // At most one independently padded record/fence pair per new token.
        let writes = try Self.product([queryCount, try bound(24 * 4) + bound(4)])
        var total = 64 << 10
        for bytes in [partialBytes, metaBytes, metadata, fences, seqinfo, hostRecords,
                      output, try Self.product([2, bound(nativeWriteBytes / 2)]), writeRecords, writes] {
            let (next, overflow) = total.addingReportingOverflow(bytes)
            guard !overflow else { throw CBv2KVError.backendIneligible(reason: "scratch byte overflow") }
            total = next
        }
        return total
    }

    private static func sum(_ values: [Int]) throws -> Int {
        try values.reduce(0) { result, value in
            let (sum, overflow) = result.addingReportingOverflow(value)
            guard !overflow, value >= 0 else {
                throw CBv2KVError.backendIneligible(reason: "scratch byte overflow")
            }
            return sum
        }
    }

    private static func product(_ factors: [Int]) throws -> Int {
        try factors.reduce(1) { try PagedKVQuantizationConfig.multiply($0, $1) }
    }
}

extension PagedKVPool {
    func appendQuantizedScratch(_ lease: PagedQuantizedScratchLease) {
        pendingQuantizedScratch.append(lease)
    }

    /// Transfer ownership to the step that built these graphs, even on failure.
    func takePendingQuantizedScratch() -> [PagedQuantizedScratchLease] {
        let result = pendingQuantizedScratch
        pendingQuantizedScratch = []
        quantizedScratchScope = nil
        return result
    }
}
