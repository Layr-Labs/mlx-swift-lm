// Copyright © 2026 Eigen Labs.

extension CBv2Request {
    /// Qualification for this text-only scheduling policy, not cache-hit proof.
    /// Reuse/capture keep their existing scope, identity and allocation gates.
    var canScheduleDemandedShortCheckpoint: Bool {
        prefixCacheEnabled && !hasOutOfBandCheckpointInput && !(cacheSalt ?? "").isEmpty
    }
}

extension CBv2ScheduledRequest {
    /// Capture never re-arms a preempted or geometrically disarmed donor.
    /// Ordinary computed progress under changing chunk caps stays eligible.
    var canScheduleDemandedShortCheckpoint: Bool {
        request.canScheduleDemandedShortCheckpoint && preemptionCount == 0
            && !shortCheckpointCaptureDisarmed
    }
}

extension CBv2SchedulerConfig {
    /// Existing count-only callers retain their ordinary short geometry.
    func demandedShortCheckpointChunk(
        promptTokens: Int, hintTokens: Int?, computedTokens: Int, proposed: Int,
        armedSoloStripeTokens: Int?, hasPrefixReuse: Bool, requestAllowsCheckpoint: Bool
    ) -> Int {
        demandedCheckpointRange(
            promptTokens: promptTokens, hintTokens: hintTokens,
            computedTokens: computedTokens, proposed: proposed,
            armedSoloStripeTokens: armedSoloStripeTokens,
            hasPrefixReuse: hasPrefixReuse, requestAllowsCheckpoint: requestAllowsCheckpoint,
            continuation: nil
        ).count
    }

    /// Shared by authoritative scheduling and pure deadline projection.
    /// A long split remembers this actual proposed range's end; it never
    /// reconstructs a global grid or increases a stripe or token/byte grant.
    func demandedCheckpointRange(
        promptTokens: Int, hintTokens: Int?, computedTokens: Int, proposed: Int,
        armedSoloStripeTokens: Int?, hasPrefixReuse: Bool, requestAllowsCheckpoint: Bool,
        continuation: CBv2DemandedCheckpointContinuation?
    ) -> CBv2DemandedCheckpointRange {
        let ordinary = CBv2DemandedCheckpointRange(count: proposed, continuation: nil)
        guard enablePrefixCache, let minimum = demandedShortCheckpointMinimumTokens,
            minimum > 0, let stripe = armedSoloStripeTokens,
            promptTokens < stripe || demandedCheckpointPartitionIncludesLongPrompts,
            promptTokens > minimum, let hint = hintTokens, hint >= minimum,
            computedTokens >= 0, proposed > 0, !hasPrefixReuse, requestAllowsCheckpoint
        else { return ordinary }
        // Preserve both exact block-chain and the native query alignment.
        let hashBlock = CBv2BlockHasher.defaultBlockSize
        let queryBlock = CBv2AttentionV1.queryBlockSize
        let alignment = max(hashBlock, queryBlock)
        guard alignment > 0, alignment.isMultiple(of: hashBlock),
            alignment.isMultiple(of: queryBlock)
        else { return ordinary }
        let target = min(hint, promptTokens - 1) / alignment * alignment
        guard target >= minimum else { return ordinary }
        let isLong = promptTokens >= stripe
        if isLong, let continuation,
            continuation.target == target,
            continuation.soloStripeTokens == stripe,
            computedTokens >= continuation.start,
            computedTokens < continuation.originalEnd,
            continuation.originalEnd <= promptTokens
        {
            let end = computedTokens < target ? target : continuation.originalEnd
            return .init(count: min(proposed, end - computedTokens), continuation: continuation)
        }
        guard target > computedTokens else { return ordinary }
        let count = min(proposed, target - computedTokens)
        guard isLong, count < proposed else {
            return .init(count: count, continuation: nil)
        }
        let (originalEnd, overflow) = computedTokens.addingReportingOverflow(proposed)
        guard !overflow, originalEnd <= promptTokens else { return ordinary }
        return .init(
            count: count,
            continuation: .init(
                start: computedTokens, target: target, originalEnd: originalEnd,
                soloStripeTokens: stripe))
    }
}
