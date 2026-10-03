// Copyright © 2026 Eigen Labs.

extension CBv2SchedulerConfig {
    /// Shared by authoritative scheduling and pure deadline projection.
    /// This bounds one existing range; it never raises a stripe, token budget,
    /// capture count, byte grant or cache-scope permission.
    func demandedShortCheckpointChunk(
        promptTokens: Int, hintTokens: Int?, computedTokens: Int, proposed: Int,
        armedSoloStripeTokens: Int?, hasPrefixReuse: Bool, hasCacheScope: Bool,
        isMultimodal: Bool
    ) -> Int {
        guard enablePrefixCache, let minimum = demandedShortCheckpointMinimumTokens,
            minimum > 0, let stripe = armedSoloStripeTokens, promptTokens < stripe,
            promptTokens > minimum, let hint = hintTokens, hint >= minimum,
            computedTokens >= 0, proposed > 0, !hasPrefixReuse, hasCacheScope,
            !isMultimodal
        else { return proposed }
        // Preserve both exact block-chain and the native query alignment.
        // A future incompatible alignment keeps the old geometry unchanged.
        let hashBlock = CBv2BlockHasher.defaultBlockSize
        let queryBlock = CBv2AttentionV1.queryBlockSize
        let alignment = max(hashBlock, queryBlock)
        guard alignment > 0, alignment.isMultiple(of: hashBlock),
            alignment.isMultiple(of: queryBlock)
        else { return proposed }
        let target = min(hint, promptTokens - 1) / alignment * alignment
        guard target >= minimum, target > computedTokens else { return proposed }
        return min(proposed, target - computedTokens)
    }
}
