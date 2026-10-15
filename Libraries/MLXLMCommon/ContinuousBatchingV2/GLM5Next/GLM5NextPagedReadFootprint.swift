// Copyright © 2026 Eigen Labs.

import MLX

/// Allocation plan for PagedSegmentTransfers.gatherCombined followed by the
/// GLM index view's independent K/gate/validity/position copies. No tensors are
/// allocated here. Records are three Int32s PER TOKEN, not one per page.
struct GLM5NextPagedIndexReadFootprint {
    let destinationBytes: Int
    let transferRecordBytes: Int
    let transferSupportBytes: Int
    let hostBytes: Int
    let totalBytes: Int

    static func plan(
        tokens: Int, indexDimension: Int, dtype: DType, pages: Int,
        tokensPerSegment: [Int],
        bound: (Int) throws -> Int = CBv2CheckpointAllocationFootprint.bound
    ) throws -> Self {
        guard (1 ... 1_048_576).contains(tokens), (1 ... 4096).contains(indexDimension),
            [.float16, .bfloat16, .float32].contains(dtype), (1 ... tokens).contains(pages),
            !tokensPerSegment.isEmpty, tokensPerSegment.count <= pages,
            tokensPerSegment.allSatisfy({ $0 > 0 && $0 <= tokens })
        else { throw GLM5NextSparseStateError.invalidGeometry }
        func add(_ a: Int, _ b: Int) throws -> Int {
            try CBv2CheckpointAllocationFootprint.add(a, b)
        }
        func multiply(_ a: Int, _ b: Int) throws -> Int {
            let (value, overflow) = a.multipliedReportingOverflow(by: b)
            guard a >= 0, b >= 0, !overflow else { throw GLM5NextSparseStateError.invalidGeometry }
            return value
        }
        func allocation(_ bytes: Int) throws -> Int {
            let result = try bound(bytes)
            guard result >= bytes else { throw GLM5NextSparseStateError.invalidGeometry }
            return result
        }
        guard try tokensPerSegment.reduce(0, add) == tokens else {
            throw GLM5NextSparseStateError.invalidGeometry
        }
        let plane = try multiply(multiply(tokens, indexDimension), dtype.size)
        // One combined K/V gather destination and two independently copied planes.
        var destinations = try add(allocation(multiply(plane, 2)), multiply(allocation(plane), 2))
        destinations = try add(destinations, allocation(tokens * DType.bool.size))
        destinations = try add(destinations, allocation(tokens * DType.int32.size))

        var records = 0
        var support = try allocation(1)  // copy selector
        var assembledRecords = 0
        var paddingWords = 0
        for count in tokensPerSegment {
            let words = try multiply(count, 3)
            let paddedWords = max(24, words)  // PagedSegmentTransfers.records
            let bytes = try multiply(paddedWords, MemoryLayout<Int32>.stride)
            records = try add(records, allocation(bytes))
            assembledRecords = try add(assembledRecords, bytes)
            paddingWords = try add(paddingWords, paddedWords - words)
            support = try add(support, allocation(DType.int32.size))  // transfer fence
            support = try add(support, allocation(5 * DType.int32.size))  // output_shape
        }
        support = try add(support, allocation(DType.int32.size))  // completion witness

        // Host arrays coexist while the lazy transfer graph is constructed:
        // page ids, slots, bucket triples and records(triples)' padded assembly.
        // Triple append growth can retain old and replacement allocations, so
        // charge three logical spans; page/slot/assembly capacity gets two.
        var host = try multiply(pages, 2 * MemoryLayout<Int32>.stride)
        host = try add(host, multiply(tokens, 2 * MemoryLayout<Int32>.stride))
        host = try add(host, multiply(tokens, 9 * MemoryLayout<Int32>.stride))
        host = try add(host, multiply(assembledRecords, 2))
        host = try add(host, multiply(paddingWords, 2 * MemoryLayout<Int32>.stride))
        // Bucket dictionary/result entries and the fixed transfer call metadata.
        host = try add(host, multiply(tokensPerSegment.count, 256))
        host = try add(host, 64 << 10)
        return try .init(
            destinationBytes: destinations, transferRecordBytes: records,
            transferSupportBytes: support, hostBytes: host,
            totalBytes: add(add(destinations, records), add(support, host)))
    }
}
