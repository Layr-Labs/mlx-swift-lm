import Foundation
import Testing

@testable import MLXLMCommon

/// The depth controller against synthetic cost tables whose optimum is known by
/// construction, driven through its real API (`select` / `observeAcceptance` /
/// `recordFinalizedStep`) with no test-only hooks.
///
/// `findsAnOptimumBehindALocalDip` is the one that matters. At THE TEST the
/// goodput curve rises to k=4, but the controller reached only depth 1 in 1,018
/// decisions (`depthSelections {0: 1002, 1: 16}`), because its probe was
/// `min(selected + 1, limit)`: with `selected` pinned at 0 the probe is always
/// 1, so depth 2 is unreachable however long the run. The acceptance frontier
/// compounded it -- `limit` is `frontier + 1`, and the frontier only advances
/// once a position has been DRAFTED `acceptanceMinSamples` times, so declining
/// to explore deeper was also declining to gather the evidence that would
/// justify it.
///
/// The reason a hill-climb is the wrong shape here, and not merely slow, is
/// that goodput is NOT monotone in depth. A round pays a fixed setup and verify
/// overhead that amortizes over the tokens it commits, so depth 1 can be a
/// genuine loss while depth 4 is the best arm on the board. `bestDepth` states
/// the objective independently of the controller's arithmetic, and every table
/// below is checked against it.
///
/// Acceptance is held at 1.0 in every table so `expectedCommitted(depth)` is
/// exactly `depth + 1` and the optimum is set purely by the cost column. That
/// keeps each table's answer checkable by hand.
@Suite
struct CBv2MTPDepthSweepTests {
    private static let bucket = 1

    /// Goodput the controller is trying to maximise, stated independently of
    /// its arithmetic: committed tokens per nanosecond of round.
    private func bestDepth(costNanos: [Double]) -> Int {
        var best = 0
        var bestValue = 1.0 / costNanos[0]
        for depth in costNanos.indices {
            let value = Double(depth + 1) / costNanos[depth]
            if value > bestValue {
                best = depth
                bestValue = value
            }
        }
        return best
    }

    /// One decode step in a world where a round at depth `d` costs
    /// `costNanos[d]` and every drafted token is accepted.
    ///
    /// The step must go through `recordFinalizedStep`, not `observeCost`:
    /// `observeCost` files a cost sample but never runs `complete()`, so the
    /// probe cadence never advances and the controller can only ever move on
    /// the goodput comparison. Driving the wrong entry point is how an earlier
    /// draft of this suite reported a pass for a controller that could not
    /// probe at all.
    @discardableResult
    private func step(
        _ controller: CBv2MTPDepthController, costNanos: [Double]
    ) -> Int {
        let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        let depth = min(max(decision.depth, 0), costNanos.count - 1)
        if depth > 0 {
            controller.observeAcceptance(
                decodeRowBucket: Self.bucket, drafted: depth, accepted: depth)
        }
        controller.recordFinalizedStep(
            decision: decision,
            actualDepth: depth,
            wallTimeNanos: UInt64(costNanos[depth]),
            costEligible: true,
            chained: false,
            finalizedPlainWork: depth == 0,
            finalizedVerification: depth > 0)
        return depth
    }

    /// The depth the controller serves at over the tail of a long run.
    private func settledDepth(costNanos: [Double], rounds: Int = 800) -> Int {
        let controller = CBv2MTPDepthController(
            maxDepth: costNanos.count - 1, fixedDepth: nil)
        var tail: [Int] = []
        for _ in 0 ..< rounds {
            tail.append(step(controller, costNanos: costNanos))
            if tail.count > 96 { tail.removeFirst() }
        }
        var counts: [Int: Int] = [:]
        for depth in tail { counts[depth, default: 0] += 1 }
        return counts.max { ($0.value, -$0.key) < ($1.value, -$1.key) }?.key ?? 0
    }

    @Test("speculation that never pays is declined")
    func declinesUnprofitableSpeculation() {
        // Each drafted token costs far more than the token it commits.
        let cost = [11e6, 51e6, 91e6, 131e6, 171e6]
        #expect(bestDepth(costNanos: cost) == 0)
        #expect(settledDepth(costNanos: cost) == 0)
    }

    @Test("a curve that peaks in the middle is followed to its peak")
    func followsACurveToItsPeak() {
        // goodput: .091 .118 .130 .138 .083 -> depth 3.
        let cost = [11e6, 17e6, 23e6, 29e6, 60e6]
        #expect(bestDepth(costNanos: cost) == 3)
        #expect(settledDepth(costNanos: cost) == 3)
    }

    @Test("an optimum behind a local dip at depth 1 is still found")
    func findsAnOptimumBehindALocalDip() {
        // THE TEST's shape, exaggerated. Depth 1 is genuinely WORSE than
        // target-only, so a hill-climb probing only `selected + 1` stops at 0
        // forever -- yet depth 4 is the best arm on the board.
        // goodput: .0909 .0667 .0938 .1176 .1389
        let cost = [11e6, 30e6, 32e6, 34e6, 36e6]
        #expect(bestDepth(costNanos: cost) == 4)
        #expect(
            1.0 / cost[0] > 2.0 / cost[1],
            "depth 1 must be a real dip or this test proves nothing")
        #expect(settledDepth(costNanos: cost) == 4)
    }

    /// The climb is paid for, and the price is bounded. Reaching the optimum
    /// costs a run of losing probes; what must not happen is paying them
    /// forever, or paying so many that the optimum arrives after the request
    /// has already finished.
    @Test("the dip is crossed early and the optimum then holds")
    func crossesTheDipEarlyAndHolds() {
        let cost = [11e6, 30e6, 32e6, 34e6, 36e6]
        let controller = CBv2MTPDepthController(
            maxDepth: cost.count - 1, fixedDepth: nil)
        var depths: [Int] = []
        for _ in 0 ..< 800 { depths.append(step(controller, costNanos: cost)) }

        let firstOptimum = depths.firstIndex(of: 4) ?? Int.max
        #expect(firstOptimum < 200, "reached depth 4 only at \(firstOptimum)")
        // The tail is the optimum and nothing else: exploration has stopped
        // because the envelope is fully evidenced and depth 4 wins it.
        #expect(depths.suffix(200).allSatisfy { $0 == 4 })
        #expect(controller.activeDepthForTesting(decodeRowBucket: Self.bucket) == 4)
    }

    @Test("every depth in the envelope gets sampled")
    func samplesTheWholeEnvelope() {
        // Exploration must not be bounded by the acceptance frontier, because
        // the frontier only advances by drafting at those depths.
        let cost = [11e6, 17e6, 23e6, 29e6, 34e6, 39e6]
        let maxDepth = cost.count - 1
        let controller = CBv2MTPDepthController(maxDepth: maxDepth, fixedDepth: nil)
        var seen: Set<Int> = []
        for _ in 0 ..< 400 { seen.insert(step(controller, costNanos: cost)) }
        #expect(
            seen == Set(0 ... maxDepth),
            "never sampled: \(Set(0 ... maxDepth).subtracting(seen).sorted())")
    }

    @Test("a fixed pin overrides the sweep entirely")
    func fixedPinWins() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: 4)
        for _ in 0 ..< 50 {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            #expect(decision.depth == 4)
            #expect(decision.reason == "fixed")
        }
    }
}
