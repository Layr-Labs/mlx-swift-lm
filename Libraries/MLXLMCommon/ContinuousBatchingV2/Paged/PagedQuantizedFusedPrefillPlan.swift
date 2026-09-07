import MLX

struct PagedQuantizedFusedPrefillGeometry: Hashable {
    let kvHeads: Int
    let queryHeads: Int
    let headDim: Int
}

/// Host-only preflight. One additive permit covers all allocations this call
/// can introduce; the direct request allowance is never used as its credit.
struct PagedQuantizedFusedPrefillPlan {
    static let maximumQueries = 128
    let geometry: PagedQuantizedFusedPrefillGeometry
    let maximumTokens: Int
    let partitionTokens: Int
    let blocks: [Range<Int>]
    let reservationBytes: Int
    let arenaBytes: Int

    init(geometry: PagedQuantizedFusedPrefillGeometry, queryCount: Int,
         maximumTokens: Int, attendLength: Int, pageSize: Int,
         outputElementBytes: Int, needsArena: Bool, maxBufferLength: Int,
         allocationPolicy: AllocationFootprintPolicy? = nil) throws {
        guard queryCount >= 9, attendLength >= queryCount, maximumTokens >= attendLength,
              maximumTokens <= Int(Int32.max), geometry.kvHeads > 0,
              geometry.queryHeads > 0, geometry.queryHeads % geometry.kvHeads == 0,
              pageSize > 0, [64, 256].contains(geometry.headDim),
              [2, 4].contains(outputElementBytes),
              let policy = allocationPolicy ?? Memory.allocationFootprintPolicy() else {
            throw CBv2KVError.backendIneligible(reason: "ineligible fused packed prefill geometry")
        }
        self.geometry = geometry
        self.maximumTokens = maximumTokens
        partitionTokens = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: pageSize)
        let blockCount = (queryCount - 1) / Self.maximumQueries + 1
        let width = queryCount / blockCount, remainder = queryCount % blockCount
        var offset = 0, ranges: [Range<Int>] = []
        for index in 0 ..< blockCount {
            let count = width + (index < remainder ? 1 : 0)
            guard count >= 9, count <= Self.maximumQueries else {
                throw CBv2KVError.backendIneligible(reason: "fused packed prefill tail would use vector SDPA")
            }
            ranges.append(offset ..< offset + count)
            offset += count
        }
        blocks = ranges
        func product(_ values: [Int]) throws -> Int {
            try values.reduce(1) { try PagedKVQuantizationConfig.multiply($0, $1) }
        }
        func sum(_ values: [Int]) throws -> Int {
            try values.reduce(0) { total, bytes in
                let (next, overflow) = total.addingReportingOverflow(bytes)
                guard bytes >= 0, !overflow else {
                    throw CBv2KVError.backendIneligible(reason: "fused prefill reservation overflow")
                }
                return next
            }
        }
        func bound(_ bytes: Int) throws -> Int {
            guard bytes <= maxBufferLength, let result = policy.upperBound(byteCount: bytes) else {
                throw CBv2KVError.backendIneligible(reason: "fused prefill allocation exceeds its buffer bound")
            }
            return result
        }
        let h = geometry.queryHeads, kh = geometry.kvHeads, d = geometry.headDim
        arenaBytes = needsArena ? try sum([
            bound(product([8, kh, d, maximumTokens])),
            bound(product([4, h, d, Self.maximumQueries])),
            bound(product([Self.maximumQueries, maximumTokens]))]) : 0
        let partitions = (attendLength - 1) / partitionTokens + 1
        let recordBytes = try product([partitions, PagedSegmentDispatchPlan.recordStride, 4])
        let offsetBytes = try product([partitions, PagedSegmentDispatchPlan.maximumBindings, 8])
        let metadata = try sum([bound(recordBytes), bound(offsetBytes),
                                product([8, sum([recordBytes, offsetBytes])])])
        let outputs = try sum(ranges.map { try bound(product([4, h, d, $0.count])) })
        let finalOutput = try bound(product([outputElementBytes, h, d, queryCount]))
        let fences = try product([sum([partitions, product([4, blockCount]), 3]), bound(4)])
        let parameters = try product([blockCount + 1, bound(8 * 4)])
        let writes = try product([queryCount, sum([bound(24 * 4), bound(4), 3 * 4 * 8])])
        reservationBytes = try sum([64 << 10, arenaBytes, metadata, outputs, finalOutput,
                                    fences, parameters, writes, bound(4 * h)])
    }
}

/// Fixed maximum geometry owned by the first successful call's extra lease.
/// Subsequent calls only reserve fresh outputs/metadata and update this fence.
final class PagedQuantizedFusedPrefillArena {
    let keysValues: MLXArray
    let queries: MLXArray
    let mask: MLXArray
    let owner: PagedQuantizedScratchLease
    var lastUse: MLXArray?

    init(plan: PagedQuantizedFusedPrefillPlan, owner: PagedQuantizedScratchLease) {
        let g = plan.geometry
        self.owner = owner
        keysValues = MLXArray.zeros([2, 1, g.kvHeads, plan.maximumTokens, g.headDim], dtype: .float32)
        queries = MLXArray.zeros([1, g.queryHeads, PagedQuantizedFusedPrefillPlan.maximumQueries, g.headDim], dtype: .float32)
        mask = MLXArray.zeros([PagedQuantizedFusedPrefillPlan.maximumQueries, plan.maximumTokens], dtype: .bool)
        owner.retain([keysValues, queries, mask])
    }

    func recordCompletion(_ fence: MLXArray, lease: PagedQuantizedScratchLease) {
        lastUse = fence
        owner.retainCompletion(fence)
        lease.retainCompletion(fence)
    }
}
