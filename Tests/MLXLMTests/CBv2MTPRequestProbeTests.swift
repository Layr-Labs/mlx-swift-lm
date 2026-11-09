import Foundation
import Testing

@testable import MLXLMCommon

/// The adaptive controller's request-probe mode (`CBv2MTPConfig.requestProbeDepth`): each request opens with one probe
/// window, the depth then follows the request's own acceptance against this engine's measured round cost by depth
/// (kept across requests), and ordinary decode re-probes on a backed-off interval. Pure controller arithmetic.
@Suite("CBv2MTP request probe")
struct CBv2MTPRequestProbeTests {
    static let plainNanos: UInt64 = 10_000_000
    /// One round's cost by depth, in plain steps: a depth-d round costs about 1 + 0.5 d steps.
    static func roundNanos(_ depth: Int) -> UInt64 { plainNanos + UInt64(depth) * plainNanos / 2 }

    static func controller(maxDepth: Int = 4, probe: Int? = 2) -> CBv2MTPDepthController {
        CBv2MTPDepthController(
            maxDepth: maxDepth, fixedDepth: nil, useCommittedDecodeBaseline: true,
            requestProbeDepth: probe)
    }

    /// The isolated warmup and three chained intervals: the depth-zero baselines.
    static func calibrate(_ controller: CBv2MTPDepthController) {
        let warmup = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(warmup.reason == "warmup_baseline")
        _ = controller.recordFinalizedStep(
            decision: warmup, actualDepth: 0, wallTimeNanos: plainNanos,
            costEligible: true, chained: false, finalizedPlainWork: true,
            finalizedVerification: false)
        for _ in 0 ..< 3 {
            #expect(
                controller.select(plannedDecodeRows: 1, canSpeculate: true).reason
                    == "warmup_chained_baseline")
            controller.observeCommittedDecodeInterval(decodeRowBucket: 1, wallTimeNanos: plainNanos)
        }
    }

    /// Takes `rounds` decisions: depth zero runs one ordinary step, a positive depth one round that keeps
    /// `accepted(depth)` drafts. Returns every decision taken.
    @discardableResult
    static func run(_ controller: CBv2MTPDepthController, rounds: Int, accepted: (Int) -> Int)
        -> [CBv2MTPDepthDecision]
    {
        var decisions: [CBv2MTPDepthDecision] = []
        for _ in 0 ..< rounds {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            decisions.append(decision)
            if decision.depth == 0 {
                _ = controller.recordFinalizedStep(
                    decision: decision, actualDepth: 0, wallTimeNanos: plainNanos,
                    costEligible: true, chained: true, finalizedPlainWork: true,
                    finalizedVerification: false)
                controller.observeCommittedDecodeInterval(
                    decodeRowBucket: 1, wallTimeNanos: plainNanos)
            } else {
                let kept = min(accepted(decision.depth), decision.depth)
                controller.observeAcceptance(
                    decodeRowBucket: 1, drafted: decision.depth, accepted: kept)
                controller.recordCommittedVerification(
                    decision: decision, wallTimeNanos: roundNanos(decision.depth),
                    committedTokens: kept + 1, rowCount: 1)
            }
        }
        return decisions
    }

    @Test func eachRequestOpensWithItsProbeWindow() {
        let controller = Self.controller()
        Self.calibrate(controller)
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(probe.depth == 2 && probe.reason == "probe" && probe.isExploration)
        // The whole window runs at the probe depth.
        let window = Self.run(controller, rounds: 8, accepted: { $0 })
        #expect(window.allSatisfy { $0.depth == 2 })
    }

    /// Every draft kept: the request goes as deep as the cap once each depth's round cost is measured, and stays.
    @Test func aRequestThatAcceptsGoesDeepAndStays() {
        let controller = Self.controller()
        Self.calibrate(controller)
        let decisions = Self.run(controller, rounds: 8 * 8, accepted: { $0 })
        // Depths 1, 3 and 4 are each measured once (one window), then the deepest wins.
        #expect(Set(decisions.map(\.depth)).isSuperset(of: [1, 2, 3, 4]))
        #expect(decisions.suffix(16).allSatisfy { $0.depth == 4 && !$0.isExploration })
    }

    /// Nothing kept: ordinary decode after the probe; re-probes after 128, then 256, then every 512 tokens.
    @Test func aLosingRequestBacksItsProbesOff() {
        let controller = Self.controller()
        controller.beginWorkload(rowIDs: [CBv2RequestID(1)])
        Self.calibrate(controller)
        // An earlier request measured every depth's round cost (it survives requests).
        Self.run(controller, rounds: 8 * 4, accepted: { _ in 0 })
        controller.beginWorkload(rowIDs: [CBv2RequestID(2)])
        var gaps: [Int] = []
        var plain = 0
        for decision in Self.run(controller, rounds: 2_500, accepted: { _ in 0 }) {
            if decision.reason == "unprofitable" {
                plain += 1
            } else if plain > 0 {
                gaps.append(plain)
                plain = 0
            }
        }
        #expect(gaps.count >= 4)
        #expect(Array(gaps.prefix(4)) == [128, 256, 512, 512])
    }

    /// A winning probe resets the back-off: after losses, a request whose drafts start to be kept gets its next
    /// probe after the base interval again once it leaves the speculative depth.
    @Test func aWinningProbeResetsTheBackOff() {
        let controller = Self.controller()
        controller.beginWorkload(rowIDs: [CBv2RequestID(1)])
        Self.calibrate(controller)
        Self.run(controller, rounds: 8 * 4, accepted: { _ in 0 })
        controller.beginWorkload(rowIDs: [CBv2RequestID(2)])
        // The opening probe and two re-probes lose: the next gap would be 512.
        Self.run(controller, rounds: 3 + 8 + 128 + 8 + 256 + 8 + 16, accepted: { _ in 0 })
        // Drafts now kept: the next probe wins and the request speculates.
        let winning = Self.run(controller, rounds: 600, accepted: { $0 })
        #expect(winning.suffix(8).allSatisfy { $0.depth > 0 })
        // Drafts rejected again: it returns to ordinary decode, and the next probe comes after the base interval.
        var plain = 0
        var gap: Int?
        for decision in Self.run(controller, rounds: 400, accepted: { _ in 0 }) {
            if decision.reason == "unprofitable" {
                plain += 1
            } else if plain > 0 {
                gap = plain
                break
            }
        }
        #expect(gap == CBv2MTPDepthController.probeBaseIntervalTokens)
    }

    /// Round costs are the engine's own and survive a request; acceptance does not, and the next request probes again.
    @Test func roundCostsSurviveRequestsAndAcceptanceDoesNot() throws {
        let controller = Self.controller()
        controller.beginWorkload(rowIDs: [CBv2RequestID(1)])
        Self.calibrate(controller)
        Self.run(controller, rounds: 8 * 5, accepted: { $0 })
        let measured = controller.snapshot().costInputs.filter { $0.depth > 0 }.map(\.depth)
        #expect(Set(measured) == [1, 2, 3, 4])
        controller.beginWorkload(rowIDs: [CBv2RequestID(2)])
        #expect(
            Set(controller.snapshot().costInputs.filter { $0.depth > 0 }.map(\.depth)) == [
                1, 2, 3, 4,
            ])
        #expect(controller.snapshot().conditionalAcceptance.isEmpty)
        for _ in 0 ..< 3 {
            #expect(
                controller.select(plannedDecodeRows: 1, canSpeculate: true).reason
                    == "warmup_chained_baseline")
            controller.observeCommittedDecodeInterval(
                decodeRowBucket: 1, wallTimeNanos: Self.plainNanos)
        }
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(probe.reason == "probe" && probe.depth == 2)
        // Costs are known: a winning request goes straight to the best depth after its probe.
        let next = Self.run(controller, rounds: 8 + 1, accepted: { $0 })
        #expect(next.last?.depth == 4)
    }

    /// Without the setting, the incremental controller is unchanged: no probe, and positive-depth costs reset.
    @Test func withoutTheSettingTheIncrementalControllerIsUnchanged() {
        let controller = Self.controller(probe: nil)
        Self.calibrate(controller)
        let first = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(first.depth == 1 && first.reason == "explore_cost")
        #expect(
            CBv2MTPConfig(
                enabled: true, maxDraftTokens: 4, fixedDraftTokens: 1, requestProbeDepth: 2
            )
            .requestProbeDepth == nil)
        #expect(
            CBv2MTPConfig(enabled: true, maxDraftTokens: 4, requestProbeDepth: 9).requestProbeDepth
                == 4)
        #expect(
            CBv2MTPConfig(enabled: true, maxDraftTokens: 0, requestProbeDepth: 2).requestProbeDepth
                == nil)
    }

    /// Every draft rejected under a plan ceiling (the driver's automatic envelope: rows x (1 + depth) <= 8 on M3-M5,
    /// <= 4 on M1/M2). Returns ordinary steps, the deepest depth asked for, and the decisions' reasons.
    static func rejectedUnderCeiling(rows: Int, ceiling: Int, steps: Int)
        -> (plain: Int, deepest: Int, reasons: Set<String>)
    {
        let controller = Self.controller()
        let bucket = CBv2MTPDepthController.decodeRowBucket(rows)
        let warmup = controller.select(
            plannedDecodeRows: rows, canSpeculate: true, depthCeiling: ceiling)
        _ = controller.recordFinalizedStep(
            decision: warmup, actualDepth: 0, wallTimeNanos: plainNanos, costEligible: true,
            chained: false, finalizedPlainWork: true, finalizedVerification: false)
        for _ in 0 ..< 3 {
            _ = controller.select(
                plannedDecodeRows: rows, canSpeculate: true, depthCeiling: ceiling)
            controller.observeCommittedDecodeInterval(
                decodeRowBucket: bucket, wallTimeNanos: plainNanos)
        }
        var plain = 0
        var deepest = 0
        var reasons: Set<String> = []
        for _ in 0 ..< steps {
            let decision = controller.select(
                plannedDecodeRows: rows, canSpeculate: true, depthCeiling: ceiling)
            deepest = max(deepest, decision.depth)
            reasons.insert(decision.reason)
            if decision.depth == 0 {
                plain += 1
                _ = controller.recordFinalizedStep(
                    decision: decision, actualDepth: 0, wallTimeNanos: plainNanos,
                    costEligible: true, chained: true, finalizedPlainWork: true,
                    finalizedVerification: false)
                controller.observeCommittedDecodeInterval(
                    decodeRowBucket: bucket, wallTimeNanos: plainNanos)
            } else {
                controller.observeAcceptance(
                    decodeRowBucket: bucket, drafted: decision.depth, accepted: 0)
                controller.recordCommittedVerification(
                    decision: decision, wallTimeNanos: roundNanos(decision.depth),
                    committedTokens: rows, rowCount: rows)
            }
        }
        return (plain, deepest, reasons)
    }

    /// The envelope's ceilings: M3-M5 with 1 (none), 2 (depth 3) and 3-4 rows (depth 1); M1/M2 with 1 (depth 3) and
    /// 2 rows (depth 1). The request probes and measures within the ceiling, then decodes ordinarily.
    @Test(arguments: [(1, 4), (2, 3), (3, 1), (4, 1), (1, 3), (2, 1)])
    func aRejectedRequestReturnsToOrdinaryDecodeUnderTheCeiling(rows: Int, ceiling: Int) {
        let result = Self.rejectedUnderCeiling(rows: rows, ceiling: ceiling, steps: 2_000)
        #expect(result.deepest == ceiling)
        #expect(result.plain >= 1_850, "ordinary steps \(result.plain)")
    }

    /// A plan that cannot speculate (5+ rows on M3-M5, 3+ on M1/M2) decodes ordinarily and leaves the request's
    /// opening probe due.
    @Test func aCeilingOfZeroDecodesOrdinarilyAndKeepsTheProbeDue() {
        let result = Self.rejectedUnderCeiling(rows: 6, ceiling: 0, steps: 200)
        #expect(result.plain == 200 && result.reasons == ["automatic_envelope_plain"])
        let controller = Self.controller()
        Self.calibrate(controller)
        let blocked = controller.select(plannedDecodeRows: 1, canSpeculate: true, depthCeiling: 0)
        #expect(blocked.depth == 0 && blocked.reason == "automatic_envelope_plain")
        _ = controller.recordFinalizedStep(
            decision: blocked, actualDepth: 0, wallTimeNanos: Self.plainNanos, costEligible: true,
            chained: true, finalizedPlainWork: true, finalizedVerification: false)
        let opened = controller.select(plannedDecodeRows: 1, canSpeculate: true, depthCeiling: 4)
        #expect(opened.reason == "probe" && opened.depth == 2)
    }

    /// Probe outcomes are counted by completed probe window: a probe followed by a depth with no measured round cost
    /// is `measuring` (neither a win nor a loss), one followed by a speculative depth is `won`, one followed by
    /// ordinary decode is `lost`.
    /// A later request's three chained intervals (the isolated warmup survives requests).
    static func rebaseline(_ controller: CBv2MTPDepthController) {
        for _ in 0 ..< 3 {
            #expect(
                controller.select(plannedDecodeRows: 1, canSpeculate: true).reason
                    == "warmup_chained_baseline")
            controller.observeCommittedDecodeInterval(decodeRowBucket: 1, wallTimeNanos: plainNanos)
        }
    }

    @Test func probeOutcomesAreCountedByWindow() {
        let controller = Self.controller()
        controller.beginWorkload(rowIDs: [CBv2RequestID(1)])
        Self.calibrate(controller)
        Self.run(controller, rounds: 8, accepted: { $0 })
        #expect(controller.probeOutcomes == ["measuring": 1])
        Self.run(controller, rounds: 8 * 3, accepted: { $0 })  // depths 1, 3 and 4 measured
        controller.beginWorkload(rowIDs: [CBv2RequestID(2)])
        Self.rebaseline(controller)
        Self.run(controller, rounds: 8, accepted: { $0 })
        #expect(controller.probeOutcomes == ["measuring": 1, "won": 1])
        controller.beginWorkload(rowIDs: [CBv2RequestID(3)])
        Self.rebaseline(controller)
        Self.run(controller, rounds: 8, accepted: { _ in 0 })
        #expect(controller.probeOutcomes == ["measuring": 1, "won": 1, "lost": 1])
    }
}
