import MLX

/// Algebraic envelope for PagedQuantizedAttentionWorkspace.reservationBytes.
/// Exact request admission still calls that shared runtime bound. These terms
/// only allow routing to sum arbitrary request lengths without K * W(total).
enum CBv2QuantizedWorkspaceCostEnvelope {
    static func layer(
        queryHeads: Int, headDim: Int, batch: Int, policy: AllocationFootprintPolicy,
        includeArena: Bool = true
    ) -> (prefill: CBv2WorkspaceCostEnvelope, decode: CBv2WorkspaceCostEnvelope)? {
        guard queryHeads > 0, headDim > 0, batch > 0,
            let extra = policy.maximumExtraBytes,
            let fence = policy.upperBound(byteCount: 4),
            let parameters = policy.upperBound(byteCount: 24 * 4),
            let blockSeqinfo = policy.upperBound(byteCount: 8 * 8 * 4),
            let vectorBytes = product([queryHeads, headDim, 4]),
            let paddedHeadDim = sum([headDim, 2]),
            let prefillPartial = product([8, queryHeads, paddedHeadDim, 4]),
            let decodePartial = product([batch, queryHeads, paddedHeadDim, 4]),
            let topology = product([9, PagedSegmentDispatchPlan.recordStride * 4 + 17 * 8]),
            let writeBucket = sum([parameters, fence]),
            let writeRecords = product([3, 4, 8]),
            let prefillConstant = sum([64 << 10, product([includeArena ? 5 : 3, extra]), product([2, fence])]),
            let prefillPartitions = sum([includeArena ? prefillPartial : 0, topology]),
            let prefillQueries = sum([vectorBytes, writeRecords, writeBucket]),
            let prefillBlocks = sum([blockSeqinfo, product([includeArena ? 3 : 4, fence])]),
            let decodePartitions = sum([includeArena ? decodePartial : 0,
                product([batch, topology]), product([batch, fence])]),
            let decodeSeqinfoRaw = product([batch, 8, 4]),
            let decodeOutputRaw = product([batch, vectorBytes]),
            let decodeSeqinfo = policy.upperBound(byteCount: decodeSeqinfoRaw),
            let decodeOutput = policy.upperBound(byteCount: decodeOutputRaw),
            let decodeConstant = sum([
                64 << 10, product([includeArena ? 4 : 2, extra]), includeArena ? 32 : 0,
                product([includeArena ? 5 : 6, fence]),
                decodeSeqinfo, decodeOutput, product([batch, writeRecords]), product([batch, writeBucket]),
            ])
        else { return nil }

        // B(n) <= n + E for variable-size allocations. Prefill has five:
        // partials, accumulators, one native output arena, and two physical
        // topology arenas shared by every query block. Their raw bytes plus
        // separately priced eightfold host construction give the factor nine.
        // Tiny parameter/fence buffers keep exact B(n), not a page-sized cost.
        let prefill = CBv2WorkspaceCostEnvelope(
            constant: prefillConstant, partitions: prefillPartitions,
            queries: prefillQueries, blocks: prefillBlocks, partitionBlocks: fence)

        // One decode's rectangular B*P partials and topology are priced at
        // maximum batch width. Partials, accumulators and two topology arenas
        // vary with context; +32 covers the minimum eight Float accumulators.
        let decode = CBv2WorkspaceCostEnvelope(
            constant: decodeConstant, partitions: decodePartitions)
        return (prefill, decode)
    }

    static func sharedArena(
        queryHeads: Int, headDim: Int, blockSize: Int, policy: AllocationFootprintPolicy
    ) -> CBv2WorkspaceCostEnvelope? {
        guard queryHeads > 0, headDim > 0, blockSize > 0,
            let extra = policy.maximumExtraBytes,
            let paddedHeadDim = sum([headDim, 2]),
            let partitions = product([blockSize, queryHeads, paddedHeadDim, 4]),
            let constant = sum([product([2, extra]), 32])
        else { return nil }
        return CBv2WorkspaceCostEnvelope(constant: constant, partitions: partitions)
    }

    private static func product(_ factors: [Int]) -> Int? {
        factors.reduce(Optional(1)) { result, factor in
            guard let result, factor >= 0 else { return nil }
            let (value, overflow) = result.multipliedReportingOverflow(by: factor)
            return overflow ? nil : value
        }
    }

    private static func sum(_ terms: [Int?]) -> Int? {
        terms.reduce(Optional(0)) { result, term in
            guard let result, let term, term >= 0 else { return nil }
            let (value, overflow) = result.addingReportingOverflow(term)
            return overflow ? nil : value
        }
    }
}
