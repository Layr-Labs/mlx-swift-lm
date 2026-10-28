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
    /// A shape-bound adopter must run its forced chunk whole, so no demanded
    /// boundary may shorten its range.
    var canScheduleDemandedShortCheckpoint: Bool {
        request.canScheduleDemandedShortCheckpoint && preemptionCount == 0
            && !shortCheckpointCaptureDisarmed && prefixReusePlan?.recurrentChunkSize == nil
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
        if demandedCaptureBoundaryMinimumTokens != nil {
            return demandedCaptureBoundaryRange(
                promptTokens: promptTokens, hintTokens: hintTokens,
                computedTokens: computedTokens, proposed: proposed,
                requestAllowsCheckpoint: requestAllowsCheckpoint) ?? ordinary
        }
        let short = demandedShortCheckpointRange(
            promptTokens: promptTokens, hintTokens: hintTokens,
            computedTokens: computedTokens, proposed: proposed,
            armedSoloStripeTokens: armedSoloStripeTokens, hasPrefixReuse: hasPrefixReuse,
            requestAllowsCheckpoint: requestAllowsCheckpoint, continuation: continuation)
        return demandedLatestCheckpointRange(
            promptTokens: promptTokens, hintTokens: hintTokens,
            computedTokens: computedTokens, proposed: short.count,
            requestAllowsCheckpoint: requestAllowsCheckpoint) ?? short
    }

    /// The short-checkpoint partition (`demandedShortCheckpointMinimumTokens`).
    private func demandedShortCheckpointRange(
        promptTokens: Int, hintTokens: Int?, computedTokens: Int, proposed: Int,
        armedSoloStripeTokens: Int?, hasPrefixReuse: Bool, requestAllowsCheckpoint: Bool,
        continuation: CBv2DemandedCheckpointContinuation?
    ) -> CBv2DemandedCheckpointRange {
        let ordinary = CBv2DemandedCheckpointRange(count: proposed, continuation: nil)
        guard enablePrefixCache, let minimum = demandedShortCheckpointMinimumTokens,
            minimum > 0, let stripe = armedSoloStripeTokens,
            promptTokens < stripe || demandedCheckpointPartitionIncludesLongPrompts,
            promptTokens > minimum, let hint = hintTokens, hint >= minimum,
            computedTokens >= 0, proposed > 0, !hasPrefixReuse, requestAllowsCheckpoint,
            let alignment = Self.demandedBoundaryAlignment
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

    /// The end-of-prompt partition (`demandedLatestCheckpointMinimumTokens`).
    /// Nil without write demand: a hint below the floor marks a fleet-novel prompt.
    private func demandedLatestCheckpointRange(
        promptTokens: Int, hintTokens: Int?, computedTokens: Int, proposed: Int,
        requestAllowsCheckpoint: Bool
    ) -> CBv2DemandedCheckpointRange? {
        guard enablePrefixCache, let minimum = demandedLatestCheckpointMinimumTokens, minimum > 0,
            requestAllowsCheckpoint, computedTokens >= 0, proposed > 0,
            hintTokens.map({ $0 >= minimum }) ?? true,
            let alignment = Self.demandedBoundaryAlignment
        else { return nil }
        let (end, overflow) = computedTokens.addingReportingOverflow(proposed)
        let reach = promptTokens - 1 - Self.demandedLatestPromptEndMarginTokens
        guard !overflow, reach > 0 else { return nil }
        let latest = reach / alignment * alignment
        guard latest >= minimum, latest < end, latest - computedTokens >= minimum else {
            return nil
        }
        return .init(count: latest - computedTokens, continuation: nil)
    }

    /// Kept below the prompt end, where the next turn re-renders the
    /// generation prompt (fewer than 16 tokens for current templates).
    static let demandedLatestPromptEndMarginTokens = 64

    /// The grid of demanded boundaries: the larger of the block-hash size and
    /// the native query block, so a boundary keeps both exact block-chain and
    /// query alignment. Nil when it is not a multiple of the smaller one.
    static var demandedBoundaryAlignment: Int? {
        let hashBlock = CBv2BlockHasher.defaultBlockSize
        let queryBlock = CBv2AttentionV1.queryBlockSize
        let alignment = max(hashBlock, queryBlock)
        guard alignment > 0, alignment.isMultiple(of: hashBlock),
            alignment.isMultiple(of: queryBlock)
        else { return nil }
        return alignment
    }

    /// The capture-boundary partition (`demandedCaptureBoundaryMinimumTokens`).
    /// Its boundaries are the deepest aligned position below the prompt end
    /// (the rolling latest, which a longer next turn restores) and, with a
    /// coordinator hint, the deepest one at or below it (the fork target).
    /// A range that crosses a boundary ends on it, so recurrent capture sees
    /// an aligned range end whatever decode company shortened the chunk; an
    /// adopter captures above its restore the same way. A range that would
    /// stop less than one query block short of a boundary ends on the query
    /// grid instead, so the range that then reaches the boundary is at least
    /// a block long and keeps the ordinary prefill kernels. Nil without write
    /// demand: a hint below the floor marks a fleet-novel prompt the store
    /// will not write, so its ordinary geometry stays.
    private func demandedCaptureBoundaryRange(
        promptTokens: Int, hintTokens: Int?, computedTokens: Int, proposed: Int,
        requestAllowsCheckpoint: Bool
    ) -> CBv2DemandedCheckpointRange? {
        guard enablePrefixCache, let minimum = demandedCaptureBoundaryMinimumTokens, minimum > 0,
            requestAllowsCheckpoint, computedTokens >= 0, proposed > 0,
            hintTokens.map({ $0 >= minimum }) ?? true,
            let alignment = Self.demandedBoundaryAlignment
        else { return nil }
        let (end, overflow) = computedTokens.addingReportingOverflow(proposed)
        guard !overflow, promptTokens > 0 else { return nil }
        let latest = (promptTokens - 1) / alignment * alignment
        let target = hintTokens.map { min($0, promptTokens - 1) / alignment * alignment }
        let ahead = [latest, target].compactMap { $0 }
            .filter { $0 >= minimum && $0 > computedTokens }
        guard let next = ahead.min() else { return nil }
        if next <= end {
            return .init(count: next - computedTokens, continuation: nil)
        }
        let queryBlock = CBv2AttentionV1.queryBlockSize
        let snapped = end / queryBlock * queryBlock
        guard next - end < queryBlock, snapped - computedTokens >= queryBlock else { return nil }
        return .init(count: snapped - computedTokens, continuation: nil)
    }
}
