// CBv2MTPDepthController.swift
//
// Step-global adaptive depth selection for rectangular MTP verification.
// One controller belongs to one EngineV2, so model build, assistant revision,
// chip class, and target/assistant quantization are naturally isolated by the
// loaded engine. Within that engine, learned state is keyed by the planned
// decode-row bucket and persists across requests.

import Foundation

struct CBv2MTPDepthDecision: Equatable {
    let depth: Int
    let decodeRowBucket: Int
    let reason: String
    let isExploration: Bool
}

struct CBv2MTPControllerSnapshot {
    let selectedDepth: Int
    let decodeRowBucket: Int
    let conditionalAcceptance: [Double]
    let costInputs: [CBv2MTPCostInput]
}

/// Cost attribution attached to one launched engine step. The timestamp is
/// host-only; observing it at finalize adds no MLX synchronization.
struct CBv2MTPStepMeasurement {
    let decision: CBv2MTPDepthDecision
    let actualDepth: Int
    let costEligible: Bool
    /// True when this interval overlaps either predecessor finalization or
    /// successor construction because the step participated in a chain.
    var chained: Bool
    let seedOnly: Bool
}

final class CBv2MTPDepthController {
    private static let costAlpha = 0.3
    private static let costClampFraction = 0.25
    private static let acceptanceAlpha = 0.1
    private static let acceptanceMinSamples = 10
    private static let hysteresisFraction = 0.05
    private static let baseProbeInterval = 8
    private static let maxProbeInterval = 256
    /// Cost samples required at a depth before its goodput is trusted. One
    /// round is a noisy estimate of a round, and acting on a single sample is
    /// how a depth gets written off for the rest of a run. A statistics
    /// constant like `acceptanceMinSamples`, not a shape constant: it counts
    /// observations, never tokens or context.
    private static let minCostSamples = 3

    /// PARTICIPANT POLICY LEVER (editable; the participant contract names
    /// this controller as the adaptive policy a submission may change).
    ///
    /// This submission runs TARGET-ONLY: the controller never selects a
    /// positive draft depth, so no seed step, no verify step, and no cost
    /// probe is ever planned. The sealed verification mode for this track is
    /// `.serialTarget`, where a depth-k round costs 1+k FULL target forwards;
    /// the adaptive policy therefore converges to depth 0 on its own, but it
    /// keeps re-proving that at every probe cadence (a seed step plus a
    /// 1+k verify step) — pure loss on this arm. Pinning the policy at 0
    /// removes those rounds. Every committed token is still produced by an
    /// ordinary target decode step, so the emitted stream stays bit-identical
    /// to serial decode.
    static let speculationEnabled = false

    private struct CostState {
        var samples = 0
        var ewmaNanos = 0.0
        var totalNanos: UInt64 = 0

        mutating func observe(_ nanos: UInt64) {
            guard nanos > 0 else { return }
            let sample = Double(nanos)
            if samples == 0 {
                ewmaNanos = sample
            } else {
                let limit = SelfLimit.fraction * ewmaNanos
                let innovation = min(max(sample - ewmaNanos, -limit), limit)
                ewmaNanos += CBv2MTPDepthController.costAlpha * innovation
            }
            samples += 1
            totalNanos &+= nanos
        }

        private enum SelfLimit {
            static let fraction = CBv2MTPDepthController.costClampFraction
        }
    }

    private struct AcceptanceState {
        /// Index zero is unused so the draft-position math is 1-based.
        var rates: [Double] = [0]
        var seen: [Int] = [0]

        mutating func observe(drafted: Int, accepted: Int) {
            guard drafted > 0 else { return }
            for position in 1 ... drafted {
                if accepted < position - 1 { break }
                grow(to: position)
                let outcome = accepted >= position ? 1.0 : 0.0
                if seen[position] == 0 {
                    rates[position] = outcome
                } else {
                    rates[position] +=
                        CBv2MTPDepthController.acceptanceAlpha
                        * (outcome - rates[position])
                }
                seen[position] += 1
            }
        }

        func rate(at position: Int) -> Double {
            if position > 0, position < seen.count,
                seen[position] >= CBv2MTPDepthController.acceptanceMinSamples
            {
                return rates[position]
            }
            guard position > 1 else { return 1 }
            for prior in stride(from: position - 1, through: 1, by: -1) {
                if prior < seen.count,
                    seen[prior] >= CBv2MTPDepthController.acceptanceMinSamples
                {
                    return rates[prior]
                }
            }
            return 1
        }

        func expectedCommitted(depth: Int) -> Double {
            guard depth > 0 else { return 1 }
            var total = 1.0
            var prefixProbability = 1.0
            for position in 1 ... depth {
                prefixProbability *= rate(at: position)
                total += prefixProbability
            }
            return total
        }

        var frontier: Int {
            var result = 0
            guard seen.count > 1 else { return result }
            for position in 1 ..< seen.count {
                guard seen[position] >= CBv2MTPDepthController.acceptanceMinSamples else {
                    break
                }
                result = position
            }
            return result
        }

        var trustedRates: [Double] {
            guard frontier > 0 else { return [] }
            return (1 ... frontier).map { rates[$0] }
        }

        private mutating func grow(to position: Int) {
            while seen.count <= position {
                seen.append(0)
                rates.append(0)
            }
        }
    }

    private struct BucketState {
        var costs: [Int: CostState] = [:]
        var acceptance = AcceptanceState()
        var activeDepth = 0
        var probeInterval = CBv2MTPDepthController.baseProbeInterval
        var roundsSinceProbe = 0
        /// Rotation cursor for periodic re-exploration, so revisiting the
        /// envelope sweeps it instead of always probing the same neighbour.
        var lastProbedDepth = 0
    }

    let maxDepth: Int
    let fixedDepth: Int?
    private var buckets: [Int: BucketState] = [:]
    private var lastDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)

    init(maxDepth: Int, fixedDepth: Int?) {
        let resolvedMax = min(max(maxDepth, 0), CBv2MTPConfig.testedMaxDraftTokens)
        self.maxDepth = resolvedMax
        self.fixedDepth = fixedDepth.map { min(max($0, 0), resolvedMax) }
    }

    static func decodeRowBucket(_ rows: Int) -> Int {
        guard rows > 0 else { return 0 }
        var bucket = 1
        while bucket < rows { bucket *= 2 }
        return bucket
    }

    func preview(plannedDecodeRows: Int, canSpeculate: Bool) -> CBv2MTPDepthDecision {
        decide(plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate, mutate: false)
    }

    func select(plannedDecodeRows: Int, canSpeculate: Bool) -> CBv2MTPDepthDecision {
        decide(plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate, mutate: true)
    }

    func observeAcceptance(decodeRowBucket: Int, drafted: Int, accepted: Int) {
        guard decodeRowBucket > 0, drafted > 0 else { return }
        var state = buckets[decodeRowBucket] ?? BucketState()
        state.acceptance.observe(drafted: drafted, accepted: accepted)
        buckets[decodeRowBucket] = state
    }

    func observeCost(decodeRowBucket: Int, depth: Int, wallTimeNanos: UInt64) {
        guard decodeRowBucket > 0, depth >= 0, depth <= maxDepth, wallTimeNanos > 0 else {
            return
        }
        var state = buckets[decodeRowBucket] ?? BucketState()
        var cost = state.costs[depth] ?? CostState()
        cost.observe(wallTimeNanos)
        state.costs[depth] = cost
        buckets[decodeRowBucket] = state
    }

    /// A depth-zero baseline is only comparable with verify steps when it is
    /// finalized before another graph is constructed. One such probe is
    /// required per bucket; ordinary target-only steps may keep chaining
    /// after the baseline exists.
    func requiresNonChainedDepthZeroProbe(_ decision: CBv2MTPDepthDecision) -> Bool {
        // Target-only policy: the baseline this probe would establish is only
        // ever compared against a verify step that can never be selected.
        guard Self.speculationEnabled else { return false }
        guard decision.depth == 0, decision.decodeRowBucket > 0 else { return false }
        return buckets[decision.decodeRowBucket]?.costs[0] == nil
    }

    /// Commit one completed controller sample. Positive depths require a
    /// finalized verification at exactly the requested depth. Chained
    /// depth-zero work advances normal-round cadence but never contributes a
    /// wall-cost sample because its elapsed interval overlaps neighboring
    /// graph construction/finalization.
    @discardableResult
    func recordFinalizedStep(
        decision: CBv2MTPDepthDecision,
        actualDepth: Int,
        wallTimeNanos: UInt64,
        costEligible: Bool,
        chained: Bool,
        finalizedPlainWork: Bool,
        finalizedVerification: Bool
    ) -> Bool {
        guard decision.decodeRowBucket > 0,
            actualDepth == decision.depth,
            actualDepth >= 0,
            actualDepth <= maxDepth
        else { return false }

        if actualDepth > 0 {
            guard finalizedVerification, costEligible, wallTimeNanos > 0 else { return false }
        } else {
            guard finalizedPlainWork else { return false }
            if chained {
                var state = buckets[decision.decodeRowBucket] ?? BucketState()
                // A warmup baseline must be measured non-chained. Once it
                // exists, completed chained target work may drive the bounded
                // exploration cadence without polluting the cost curve.
                guard state.costs[0] != nil, !decision.isExploration else { return false }
                complete(decision, state: &state)
                buckets[decision.decodeRowBucket] = state
                return true
            }
            guard costEligible, wallTimeNanos > 0 else { return false }
        }

        var state = buckets[decision.decodeRowBucket] ?? BucketState()
        var cost = state.costs[actualDepth] ?? CostState()
        cost.observe(wallTimeNanos)
        state.costs[actualDepth] = cost
        complete(decision, state: &state)
        buckets[decision.decodeRowBucket] = state
        return true
    }

    func activeDepthForTesting(decodeRowBucket: Int) -> Int {
        buckets[decodeRowBucket]?.activeDepth ?? 0
    }

    func probeIntervalForTesting(decodeRowBucket: Int) -> Int {
        buckets[decodeRowBucket]?.probeInterval ?? Self.baseProbeInterval
    }

    func snapshot() -> CBv2MTPControllerSnapshot {
        var inputs: [CBv2MTPCostInput] = []
        for bucket in buckets.keys.sorted() {
            guard let state = buckets[bucket] else { continue }
            for depth in state.costs.keys.sorted() {
                guard let cost = state.costs[depth] else { continue }
                inputs.append(
                    CBv2MTPCostInput(
                        decodeRowBucket: bucket,
                        depth: depth,
                        samples: cost.samples,
                        ewmaWallTimeNanos: UInt64(max(0, cost.ewmaNanos.rounded())),
                        totalWallTimeNanos: cost.totalNanos))
            }
        }
        let state = buckets[lastDecision.decodeRowBucket]
        return CBv2MTPControllerSnapshot(
            selectedDepth: lastDecision.depth,
            decodeRowBucket: lastDecision.decodeRowBucket,
            conditionalAcceptance: state?.acceptance.trustedRates ?? [],
            costInputs: inputs)
    }

    private func decide(
        plannedDecodeRows: Int, canSpeculate: Bool, mutate: Bool
    ) -> CBv2MTPDepthDecision {
        let bucket = Self.decodeRowBucket(plannedDecodeRows)
        guard bucket > 0 else {
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: 0, reason: "no_decode_rows",
                    isExploration: false),
                mutate: mutate)
        }
        guard Self.speculationEnabled else {
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: bucket, reason: "policy_target_only",
                    isExploration: false),
                mutate: mutate)
        }
        guard canSpeculate, maxDepth > 0 else {
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: bucket,
                    reason: maxDepth == 0 ? "max_depth_zero" : "ineligible",
                    isExploration: false),
                mutate: mutate)
        }
        if let fixedDepth {
            return finish(
                CBv2MTPDepthDecision(
                    depth: fixedDepth, decodeRowBucket: bucket, reason: "fixed",
                    isExploration: false),
                mutate: mutate)
        }

        let state = buckets[bucket] ?? BucketState()
        // SELECTION stays bounded by the acceptance frontier: a depth whose
        // positions have not been observed enough times has no trustworthy
        // committed-token estimate, so it must not be chosen on one.
        let selectionLimit = min(maxDepth, state.acceptance.frontier + 1)
        // EXPLORATION is not, and that is the fix. Bounding exploration by the
        // frontier is a ratchet: the frontier only advances once a position
        // has been DRAFTED `acceptanceMinSamples` times, so refusing to
        // explore deeper is also refusing to gather the evidence that would
        // justify exploring deeper. Combined with a probe of `selected + 1`,
        // a controller sitting at depth 0 could only ever see depth 1 —
        // measured at THE TEST as `depthSelections {0: ~1002, 1: 16}` over
        // 1,018 decisions, while a fixed depth 4 was committing 4.0 tokens a
        // round at 1.40x serial. The caller's rectangular envelope already
        // bounds `maxDepth` per plan, so exploring to it is always legal.
        let explorationLimit = maxDepth
        let decision: CBv2MTPDepthDecision

        if state.costs[0] == nil {
            decision = CBv2MTPDepthDecision(
                depth: 0, decodeRowBucket: bucket, reason: "warmup_baseline",
                isExploration: true)
        } else if let undersampled = (0 ... explorationLimit).first(where: {
            (state.costs[$0]?.samples ?? 0) < Self.minCostSamples
        }) {
            decision = CBv2MTPDepthDecision(
                depth: undersampled, decodeRowBucket: bucket, reason: "explore_cost",
                isExploration: true)
        } else {
            let current = min(state.activeDepth, selectionLimit)
            let currentGoodput = goodput(depth: current, state: state)
            var best = current
            var bestGoodput = currentGoodput
            for depth in 0 ... selectionLimit {
                let candidate = goodput(depth: depth, state: state)
                if candidate > bestGoodput {
                    best = depth
                    bestGoodput = candidate
                }
            }

            var selected = current
            var reason = current == 0 ? "unprofitable" : "goodput"
            if best != current {
                if currentGoodput <= 0
                    || bestGoodput >= currentGoodput * (1 + Self.hysteresisFraction)
                {
                    selected = best
                    reason = best == 0 ? "unprofitable" : "goodput"
                } else {
                    reason = "hysteresis"
                }
            }

            var explore = false
            let nextRounds = state.roundsSinceProbe + 1
            if nextRounds >= state.probeInterval, explorationLimit > 0 {
                // Rotate through the WHOLE envelope rather than probing one
                // deeper. The optimum moves in both directions — a shape
                // change, a drafter change, thermal drift — and a depth that
                // measured badly once has to be revisitable, which a
                // monotone `selected + 1` probe never allows.
                let probe = (state.lastProbedDepth + 1) % (explorationLimit + 1)
                if probe != selected {
                    selected = probe
                    reason = "explore_rotate"
                    explore = true
                }
            }
            decision = CBv2MTPDepthDecision(
                depth: selected, decodeRowBucket: bucket, reason: reason,
                isExploration: explore)
        }

        return finish(decision, mutate: mutate)
    }

    private func complete(
        _ decision: CBv2MTPDepthDecision,
        state: inout BucketState
    ) {
        if decision.isExploration {
            state.roundsSinceProbe = 0
            state.lastProbedDepth = decision.depth
            if decision.reason == "explore_rotate" {
                state.probeInterval = min(
                    state.probeInterval * 2, Self.maxProbeInterval)
            }
            return
        }
        if decision.depth != state.activeDepth {
            state.activeDepth = decision.depth
            state.probeInterval = Self.baseProbeInterval
            state.roundsSinceProbe = 0
        } else {
            state.roundsSinceProbe += 1
        }
    }

    private func goodput(depth: Int, state: BucketState) -> Double {
        guard let cost = state.costs[depth], cost.ewmaNanos > 0 else { return 0 }
        return state.acceptance.expectedCommitted(depth: depth) / cost.ewmaNanos
    }

    private func finish(
        _ decision: CBv2MTPDepthDecision, mutate: Bool
    ) -> CBv2MTPDepthDecision {
        if mutate { lastDecision = decision }
        return decision
    }
}

/// Request-owned conditional acceptance estimates for the Qwen MTP marginal
/// depth policy. Hardware cost observations deliberately do not live here.
struct CBv2MTPRequestAcceptanceState: Equatable {
    static let maximumDepth = 4
    private static let alpha = 0.15

    private(set) var probabilities: [Double] =
        (0 ..< CBv2MTPRequestAcceptanceState.maximumDepth).map {
            0.85 * pow(0.98, Double($0))
        }

    /// Records positions whose target outcome was actually observed and, after
    /// a fully accepted round, transfers bounded optimism to the next position.
    /// A truncation (stop, token budget, or common-width clamp) passes
    /// `rejectionObserved: false` and `endedByTruncation: true`, so it never
    /// manufactures either a failure or next-position optimism.
    mutating func observe(
        draftedDepth: Int,
        acceptedDepth: Int,
        rejectionObserved: Bool,
        endedByTruncation: Bool = false
    ) {
        let drafted = min(max(draftedDepth, 0), Self.maximumDepth)
        let accepted = min(max(acceptedDepth, 0), drafted)

        for position in 0 ..< accepted {
            probabilities[position] += Self.alpha * (1.0 - probabilities[position])
        }
        if rejectionObserved, accepted < drafted {
            probabilities[accepted] += Self.alpha * (0.0 - probabilities[accepted])
        } else if !rejectionObserved, !endedByTruncation,
            drafted > 0, accepted == drafted, drafted < Self.maximumDepth,
            probabilities[drafted] < 0.95
        {
            // A fully accepted round is bounded evidence that the hot chain
            // may profitably extend one position farther.
            probabilities[drafted] += Self.alpha * (0.95 - probabilities[drafted])
        }
    }
}

/// Engine-shared raw, nonchained wall-cost estimates for the marginal policy.
/// Callers record the isolated interval itself: seed-attributed cost has no
/// input in this API and therefore cannot contaminate the inferred slope.
struct CBv2MTPRawCostEstimator {
    static let bootstrapHeadStepCostRatio = 0.18
    private static let maximumDepth = CBv2MTPRequestAcceptanceState.maximumDepth
    private static let alpha = 0.3
    private static let clampFraction = 0.25

    private struct Cost {
        var samples = 0
        var ewmaNanos = 0.0

        mutating func observe(_ sample: Double) {
            if samples == 0 {
                ewmaNanos = sample
            } else {
                let limit = CBv2MTPRawCostEstimator.clampFraction * ewmaNanos
                let innovation = min(max(sample - ewmaNanos, -limit), limit)
                ewmaNanos += CBv2MTPRawCostEstimator.alpha * innovation
            }
            samples += 1
        }
    }

    private var costs: [Int: Cost] = [:]
    private var positiveDepthWarmups: Set<Int> = []

    /// Returns whether the raw sample entered the steady-state estimate.
    /// The first isolated sample at each positive depth is a compile/JIT
    /// warm-up: remember that it occurred, but never anchor the clamped EWMA
    /// to it. Depth zero uses its first sample because the target is already
    /// compiled before the nonchained baseline probe.
    @discardableResult
    mutating func observe(
        depth: Int,
        rawWallTimeNanos: Double,
        chained: Bool = false
    ) -> Bool {
        guard depth >= 0, depth <= Self.maximumDepth,
            !chained, rawWallTimeNanos.isFinite, rawWallTimeNanos > 0
        else { return false }

        if depth > 0, positiveDepthWarmups.insert(depth).inserted {
            return false
        }
        var cost = costs[depth] ?? Cost()
        cost.observe(rawWallTimeNanos)
        costs[depth] = cost
        return true
    }

    /// Sample-count-weighted h_k = max(0, (Ck/C0 - 1)/k). Until both C0 and
    /// at least one positive-depth Ck exist, use the measured bootstrap.
    var headStepCostRatio: Double {
        guard let baseline = costs[0], baseline.ewmaNanos.isFinite,
            baseline.ewmaNanos > 0
        else { return Self.bootstrapHeadStepCostRatio }

        var weightedSlope = 0.0
        var weight = 0
        for depth in 1 ... Self.maximumDepth {
            guard let cost = costs[depth],
                cost.samples > 0, cost.ewmaNanos.isFinite, cost.ewmaNanos > 0
            else {
                continue
            }
            let normalizedIncrement =
                cost.ewmaNanos <= baseline.ewmaNanos
                ? 0
                : (cost.ewmaNanos - baseline.ewmaNanos) / baseline.ewmaNanos
            let rawSlope = normalizedIncrement / Double(depth)
            let slope = rawSlope.isFinite ? rawSlope : Double.greatestFiniteMagnitude
            let newWeight = weight + cost.samples
            let fraction = Double(cost.samples) / Double(newWeight)
            // Incremental weighting cannot overflow for nonnegative finite
            // slopes, unlike accumulating `slope * samples`.
            weightedSlope += (slope - weightedSlope) * fraction
            weight = newWeight
        }
        guard weight > 0 else { return Self.bootstrapHeadStepCostRatio }
        return weightedSlope.isFinite ? weightedSlope : Double.greatestFiniteMagnitude
    }

    /// True until a post-warm-up isolated sample exists at `depth`. The engine
    /// may satisfy this with a bounded one-token probe even when confidence
    /// or the provisional cost estimate would otherwise select zero.
    func needsSteadyStateProbe(depth: Int) -> Bool {
        guard depth > 0, depth <= Self.maximumDepth, costs[0] != nil else {
            return false
        }
        return costs[depth] == nil
    }
    func sampleCount(depth: Int) -> Int {
        costs[depth]?.samples ?? 0
    }
}

/// Pure marginal-cost selector. One request supplies its own acceptance
/// probabilities; a caller may share only `headStepCostRatio` across requests.
enum CBv2MTPMarginalDepthPolicy {
    static let maximumDepth = CBv2MTPRequestAcceptanceState.maximumDepth

    static func selectDepth(
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int,
        acceptanceProbabilities: [Double],
        previousTargetTopTwoMargin: Double?,
        headStepCostRatio: Double
    ) -> Int {
        guard offeredDepth > 0, remainingTokens > 1, verificationLimit > 0 else {
            return 0
        }
        let cap = min(
            maximumDepth,
            min(offeredDepth, min(remainingTokens - 1, verificationLimit)))
        guard cap > 0 else { return 0 }

        let h =
            headStepCostRatio.isFinite && headStepCostRatio >= 0
            ? headStepCostRatio
            : CBv2MTPRawCostEstimator.bootstrapHeadStepCostRatio
        var reach = 1.0
        var expected = 0.0
        var depth = 0
        while depth < cap {
            let rawProbability =
                depth < acceptanceProbabilities.count
                ? acceptanceProbabilities[depth]
                : 0
            let probability = cappedAcceptanceProbability(
                position: depth,
                acceptanceProbability: rawProbability,
                previousMargin: previousTargetTopTwoMargin)
            reach *= probability
            let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)
            guard reach > threshold else { break }
            expected += reach
            depth += 1
        }
        return depth
    }

    /// A cost-confirmation probe is always at most one draft and obeys every
    /// ordinary capacity limit. Whether a probe is due remains engine cadence
    /// state; this helper keeps its depth choice pure and deterministic.
    static func boundedProbeDepth(
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int
    ) -> Int {
        guard offeredDepth > 0, remainingTokens > 1, verificationLimit > 0 else {
            return 0
        }
        return min(1, min(offeredDepth, min(remainingTokens - 1, verificationLimit)))
    }

    private static func cappedAcceptanceProbability(
        position: Int,
        acceptanceProbability: Double,
        previousMargin: Double?
    ) -> Double {
        guard acceptanceProbability.isFinite else { return 0 }
        var probability = min(max(acceptanceProbability, 0), 1)
        guard (position == 0 || position == 1),
            let margin = previousMargin, margin.isFinite
        else { return probability }

        let divisor = position == 0 ? 2.0 : 3.0
        probability = min(probability, sigmoid(margin / divisor))
        return probability
    }

    private static func sigmoid(_ value: Double) -> Double {
        if value >= 0 {
            return 1.0 / (1.0 + exp(-value))
        }
        let exponential = exp(value)
        return exponential / (1.0 + exponential)
    }
}
