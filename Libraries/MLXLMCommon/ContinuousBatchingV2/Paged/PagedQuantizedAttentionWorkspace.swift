import MLX

/// One attention call's actual scratch owners. The engine drains these only
/// after the submitted step completed (including failure/cancellation drains).
final class PagedQuantizedScratchLease {
    private var reservation: CBv2CheckpointReservation?
    private var roots: [MLXArray] = []
    private var completion: MLXArray?
    private var owners: [AnyObject] = []
    let reservedBytes: Int

    init(reservation: CBv2CheckpointReservation?, bytes: Int) {
        self.reservation = reservation
        reservedBytes = bytes
    }

    var evaluationTargets: [MLXArray] { completion.map { [$0] } ?? [] }
    func retainAdditional(_ arrays: [MLXArray]) { roots.append(contentsOf: arrays) }
    func retainCompletion(_ array: MLXArray) { completion = array }
    func retainOwners(_ values: [AnyObject]) { owners.append(contentsOf: values) }

    func finishAfterSynchronization() {
        completion = nil
        roots = []
        owners = []
        reservation?.release()
        reservation = nil
    }
}

/// Query-block scratch, never a dequantized full-history allocation. Metadata
/// describes one physical row and is broadcast over the bounded query block.
final class PagedQuantizedAttentionWorkspace {
    static let queryBlockSize = 8
    let partials: MLXArray
    let meta: MLXArray
    let output: MLXArray
    let lease: PagedQuantizedScratchLease
    let partitionTokens: Int
    let maximumPartitions: Int

    init(
        pool: PagedKVPool, group: PagedKVGroup, queryShape: [Int], dtype: DType,
        attendLength: Int, additionalNativeBytes: Int = 0,
        metadataRecordCounts: [Int]? = nil
    ) throws {
        let heads = queryShape[1]
        let count = queryShape[2]
        let width = queryShape[3]
        partitionTokens = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: group.pageSize)
        guard attendLength > 0, count > 0, heads > 0, width > 0 else {
            throw CBv2KVError.backendIneligible(reason: "invalid packed attention workspace")
        }
        maximumPartitions = (attendLength - 1) / partitionTokens + 1
        let block = min(Self.queryBlockSize, count)
        let bytes = try Self.reservationBytes(
            queries: count, heads: heads, width: width, partitions: maximumPartitions,
            nativeElementBytes: dtype.size, segmentCount: group.segments.count,
            additionalNativeBytes: additionalNativeBytes,
            maximumBufferBytes: pool.config.maxBufferLength,
            metadataRecordCounts: metadataRecordCounts)
        let permit = try pool.memoryAdmission?.reserveTransient(bytes: bytes)
        lease = PagedQuantizedScratchLease(reservation: permit, bytes: bytes)
        partials = MLXArray.zeros([block, heads, maximumPartitions, width], dtype: .float32)
        meta = MLXArray.zeros([max(8, block * heads * maximumPartitions * 2)], dtype: .float32)
        output = MLXArray.zeros(queryShape, dtype: dtype)
        lease.retainAdditional([partials, meta, output])
        pool.pendingQuantizedScratch.append(lease)
    }

    static func reservationBytes(
        queries: Int, heads: Int, width: Int, partitions: Int,
        nativeElementBytes: Int, segmentCount: Int,
        additionalNativeBytes: Int,
        maximumBufferBytes: Int, metadataRecordCounts: [Int]? = nil
    ) throws -> Int {
        func product(_ values: [Int]) throws -> Int {
            try values.reduce(1) { accumulated, value in
                guard value >= 0, let result = CBv2KVGeometry.multiply(accumulated, value) else {
                    throw CBv2KVError.backendIneligible(reason: "packed workspace product overflow")
                }
                return result
            }
        }
        func sum(_ values: [Int]) throws -> Int {
            try values.reduce(0) { accumulated, value in
                guard value >= 0, let result = CBv2KVGeometry.add(accumulated, value) else {
                    throw CBv2KVError.backendIneligible(reason: "packed workspace sum overflow")
                }
                return result
            }
        }
        guard queries > 0, heads > 0, width > 0, partitions > 0,
            let policy = Memory.allocationFootprintPolicy()
        else {
            throw CBv2KVError.backendIneligible(
                reason: "packed workspace allocation policy unavailable")
        }
        let block = min(queryBlockSize, queries)
        let blocks = (queries - 1) / block + 1
        let partial = try product([block, heads, partitions, width, 4])
        let metadataElements = max(8, try product([block, heads, partitions, 2]))
        let metadata = try product([metadataElements, 4])
        let output = try product([queries, heads, width, nativeElementBytes])
        guard [partial, metadata, output].allSatisfy({ $0 <= maximumBufferBytes }),
            let counts = metadataRecordCounts, !counts.isEmpty,
            counts.allSatisfy({ $0 > 0 && $0 % PagedSegmentDispatchPlan.recordStride == 0 })
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid packed metadata allocation plan")
        }
        let recordBytes = try counts.map { try product([$0, 4]) }
        let offsetBytes = 17 * 8
        let seqinfoBytes = try product([block, 8, 4])
        guard
            ([partial, metadata, output, offsetBytes, seqinfoBytes] + recordBytes).allSatisfy({
                $0 <= maximumBufferBytes
            })
        else {
            throw CBv2KVError.backendIneligible(reason: "packed workspace exceeds buffer limit")
        }
        func bound(_ bytes: Int) throws -> Int {
            guard let result = policy.upperBound(byteCount: bytes) else {
                throw CBv2KVError.backendIneligible(reason: "packed workspace allocation overflow")
            }
            return result
        }
        // Exact host plans determine the separately padded metadata buffers.
        // Fence/seqinfo allocations are also priced one at a time, avoiding a
        // maximum-extra surcharge on every four-byte fence.
        let recordsBound = try sum(recordBytes.map { try bound($0) })
        let offsetsBound = try product([counts.count, bound(offsetBytes)])
        let nativeLayoutBound = try sum([
            bound(PagedQuantizedAttention.nativeLayoutFieldCount * MemoryLayout<Int64>.stride),
            bound(2 * MemoryLayout<Int64>.stride),
        ])
        let seqinfosBound = try product([blocks, bound(seqinfoBytes)])
        let fenceCount = try sum([product([blocks, sum([counts.count, 1])]), 1])
        let fencesBound = try product([fenceCount, bound(4)])
        let host = try product([
            8,
            sum([
                sum(recordBytes), product([counts.count, offsetBytes]),
                product([blocks, seqinfoBytes]),
            ]),
        ])
        return try sum([
            64 << 10, bound(partial), bound(metadata), bound(output), nativeLayoutBound,
            recordsBound,
            offsetsBound, seqinfosBound, fencesBound, host, bound(additionalNativeBytes),
        ])
    }
}

extension PagedKVPool {
    func completeQuantizedAttentionStep() {
        for cache in quantizedAttentionCaches.values {
            cache.value?.completeQuantizedAttentionStep()
        }
        quantizedAttentionCaches = quantizedAttentionCaches.filter { $0.value.value != nil }
    }

    func takePendingQuantizedScratch() -> [PagedQuantizedScratchLease] {
        let result = pendingQuantizedScratch
        pendingQuantizedScratch = []
        return result
    }
}
