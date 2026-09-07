/// A nonnegative polynomial upper bound for request-owned workspace. Separating
/// its geometry from allocation costs lets routing bound every split of a raw
/// token total without multiplying the longest context by the request count.
struct CBv2WorkspaceCostEnvelope: Sendable {
    var constant = 0
    var partitions = 0
    var queries = 0
    var blocks = 0
    var partitionBlocks = 0

    func bytes(
        totalTokens: Int, maximumRequests: Int, maximumQueries: Int,
        blockSize: Int, partitionTokens: Int, lookahead: Int
    ) -> Int? {
        guard totalTokens >= 0, maximumRequests >= 0, maximumQueries > 0,
            blockSize > 0, partitionTokens > 0, lookahead >= 0,
            [constant, partitions, queries, blocks, partitionBlocks].allSatisfy({ $0 >= 0 })
        else { return nil }
        if totalTokens == 0 { return 0 }
        guard maximumRequests > 0 else { return nil }
        let count = min(totalTokens, maximumRequests)
        guard let padded = Self.add(totalTokens, Self.multiply(count, lookahead)),
            let sumPartitions = Self.add(Self.ceil(padded, by: partitionTokens), count - 1),
            let queryLimit = Self.multiply(count, maximumQueries),
            let blockLimit = Self.multiply(count, Self.ceil(maximumQueries, by: blockSize))
        else { return nil }
        let sumQueries = min(totalTokens, queryLimit)
        guard let roundedBlocks = Self.add(Self.ceil(sumQueries, by: blockSize), count - 1),
            let paddedLongest = Self.add(totalTokens, lookahead)
        else { return nil }
        let sumBlocks = min(sumQueries, min(blockLimit, roundedBlocks))
        let longestBlocks = Self.ceil(min(totalTokens, maximumQueries), by: blockSize)
        let longestPartitions = Self.ceil(paddedLongest, by: partitionTokens)

        // For N_i > 0, sum ceil((N_i+s)/p) <= ceil((T+K*s)/p)+K-1.
        // Q_i=min(N_i,C), so sum Q_i <= min(T,K*C); the same ceiling
        // inequality bounds sum ceil(Q_i/b). For the product, either maximum
        // factor times the sum of the other factor is an upper bound.
        guard let blocksTimesPartitions = Self.multiply(longestBlocks, sumPartitions),
            let partitionsTimesBlocks = Self.multiply(longestPartitions, sumBlocks)
        else { return nil }
        let sumPartitionBlocks = min(blocksTimesPartitions, partitionsTimesBlocks)
        var result = 0
        for (coefficient, units) in [
            (constant, count), (partitions, sumPartitions), (queries, sumQueries),
            (blocks, sumBlocks), (partitionBlocks, sumPartitionBlocks),
        ] {
            guard let next = Self.add(result, Self.multiply(coefficient, units)) else { return nil }
            result = next
        }
        return result
    }

    func adding(_ other: Self) -> Self? {
        guard let constant = Self.add(constant, other.constant),
            let partitions = Self.add(partitions, other.partitions),
            let queries = Self.add(queries, other.queries),
            let blocks = Self.add(blocks, other.blocks),
            let partitionBlocks = Self.add(partitionBlocks, other.partitionBlocks)
        else { return nil }
        return Self(constant: constant, partitions: partitions, queries: queries,
                    blocks: blocks, partitionBlocks: partitionBlocks)
    }

    func scaled(by factor: Int) -> Self? {
        guard factor >= 0,
            let constant = Self.multiply(constant, factor),
            let partitions = Self.multiply(partitions, factor),
            let queries = Self.multiply(queries, factor),
            let blocks = Self.multiply(blocks, factor),
            let partitionBlocks = Self.multiply(partitionBlocks, factor)
        else { return nil }
        return Self(constant: constant, partitions: partitions, queries: queries,
                    blocks: blocks, partitionBlocks: partitionBlocks)
    }

    /// max(A(x), B(x)) <= coefficientMax(A, B)(x) for nonnegative x.
    func covering(_ other: Self) -> Self {
        Self(constant: max(constant, other.constant),
             partitions: max(partitions, other.partitions),
             queries: max(queries, other.queries), blocks: max(blocks, other.blocks),
             partitionBlocks: max(partitionBlocks, other.partitionBlocks))
    }

    private static func ceil(_ value: Int, by divisor: Int) -> Int {
        value == 0 ? 0 : (value - 1) / divisor + 1
    }

    private static func multiply(_ lhs: Int, _ rhs: Int) -> Int? {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : result
    }

    private static func add(_ lhs: Int, _ rhs: Int?) -> Int? {
        guard let rhs else { return nil }
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : result
    }
}
