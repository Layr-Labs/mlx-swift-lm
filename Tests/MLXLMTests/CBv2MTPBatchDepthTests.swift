// CBv2MTPBatchDepthTests.swift
//
// The rectangular plan's depth for drafters whose rows commit their own
// accepted prefix (`CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost`), and
// the relaxation that lets a row that chose depth zero draft again
// (`CBv2MTPRequestAcceptanceState.relaxTowardPrior`). Depth never changes
// tokens: every depth verifies exactly.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPBatchDepth")
struct CBv2MTPBatchDepthTests {
    private let strong = [Double](repeating: 0.9, count: 7)
    private let weak = [Double](repeating: 0.1, count: 7)
    private let flatCosts: [Double] = [1, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]

    private func row(
        _ probabilities: [Double], remaining: Int = 100, margin: Double? = nil
    ) -> CBv2MTPMarginalDepthPolicy.BatchRow {
        .init(
            remainingTokens: remaining, acceptanceProbabilities: probabilities,
            previousTargetTopTwoMargin: margin)
    }

    /// One row and no fixed rows is exactly the per-row cost-curve policy.
    @Test func oneRowMatchesThePerRowPolicy() {
        let acceptances: [[Double]] = [
            strong,
            weak,
            [0.85, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05],
            CBv2MTPRequestAcceptanceState.prior,
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0.5, .nan, 0.5],
        ]
        let costs: [[Double]] = [
            flatCosts, [1, 2, 3, 4, 5, 6, 7, 8], [1, 1.4, 1.5, 1.6, 1.7], [1, 1, 1, 1, 1, 1, 1, 1],
            [1, .infinity, 1.2], [1],
        ]
        for acceptance in acceptances {
            for ratios in costs {
                for margin in [nil, 0.0, 0.5, 3.0, -2.0] as [Double?] {
                    for remaining in [0, 1, 2, 3, 8, 100] {
                        for offered in [0, 1, 4, 7] {
                            for limit in [0, 2, 7] {
                                let single = CBv2MTPMarginalDepthPolicy.selectDepthByCost(
                                    offeredDepth: offered, remainingTokens: remaining,
                                    verificationLimit: limit,
                                    acceptanceProbabilities: acceptance,
                                    previousTargetTopTwoMargin: margin, costRatios: ratios)
                                let batch = CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost(
                                    offeredDepth: offered, verificationLimit: limit,
                                    rows: [row(acceptance, remaining: remaining, margin: margin)],
                                    fixedTokenRows: 0, costRatios: ratios)
                                #expect(
                                    batch == single,
                                    "acceptance \(acceptance) ratios \(ratios) margin \(String(describing: margin)) remaining \(remaining) offered \(offered) limit \(limit)"
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    /// The minimum of the rows' own choices lets one weak row stop a strong
    /// row from drafting; the rectangle's own goodput keeps it drafting.
    @Test func oneWeakRowNoLongerStopsTheBatch() {
        func own(_ p: [Double]) -> Int {
            CBv2MTPMarginalDepthPolicy.selectDepthByCost(
                offeredDepth: 7, remainingTokens: 100, verificationLimit: 7,
                acceptanceProbabilities: p, previousTargetTopTwoMargin: nil, costRatios: flatCosts)
        }
        #expect(own(weak) == 0)
        #expect(own(strong) == 7)
        let minimum = min(own(weak), own(strong))
        let batch = CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost(
            offeredDepth: 7, verificationLimit: 7, rows: [row(strong), row(weak)],
            fixedTokenRows: 0, costRatios: flatCosts)
        #expect(minimum == 0)
        #expect(batch == 7)
        // A low margin on the weak row's carry caps only that row.
        let capped = CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost(
            offeredDepth: 7, verificationLimit: 7, rows: [row(strong), row(strong, margin: -6)],
            fixedTokenRows: 0, costRatios: flatCosts)
        #expect(capped > 0)
    }

    /// Rows that cannot draft emit one token at every depth but share the
    /// round's cost, so they lower the depth at which drafting pays.
    @Test func fixedRowsCountOneTokenAtEveryDepth() {
        let steep: [Double] = [1, 1.5, 1.9, 2.3, 2.7, 3.1, 3.5, 3.9]
        let modest = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        let alone = CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost(
            offeredDepth: 7, verificationLimit: 7, rows: [row(modest), row(modest)],
            fixedTokenRows: 0, costRatios: steep)
        let beside = CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost(
            offeredDepth: 7, verificationLimit: 7, rows: [row(modest), row(modest)],
            fixedTokenRows: 4, costRatios: steep)
        #expect(alone == 1)
        #expect(beside == 0)
        // Negative counts are treated as none.
        #expect(
            CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost(
                offeredDepth: 7, verificationLimit: 7, rows: [row(modest), row(modest)],
                fixedTokenRows: -3, costRatios: steep) == alone)
    }

    /// Every ordinary limit bounds the depth of the whole rectangle.
    @Test func everyLimitBoundsTheRectangle() {
        func depth(
            offered: Int = 7, limit: Int = 7, rows: [CBv2MTPMarginalDepthPolicy.BatchRow],
            ratios: [Double]? = nil
        ) -> Int {
            CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost(
                offeredDepth: offered, verificationLimit: limit, rows: rows, fixedTokenRows: 0,
                costRatios: ratios ?? flatCosts)
        }
        #expect(depth(rows: [row(strong), row(strong)]) == 7)
        #expect(depth(rows: [row(strong), row(strong, remaining: 3)]) == 2)
        #expect(depth(rows: [row(strong), row(strong, remaining: 1)]) == 0)
        #expect(depth(offered: 3, rows: [row(strong), row(strong)]) == 3)
        #expect(depth(limit: 2, rows: [row(strong), row(strong)]) == 2)
        #expect(depth(limit: 0, rows: [row(strong), row(strong)]) == 0)
        #expect(depth(offered: 0, rows: [row(strong), row(strong)]) == 0)
        #expect(depth(rows: [row(strong), row(strong)], ratios: [1, 1.1, 1.2]) == 2)
        #expect(depth(rows: []) == 0)
        // An invalid ratio ends the search at the last valid depth.
        #expect(depth(rows: [row(strong), row(strong)], ratios: [1, 1.1, .nan, 1.2]) == 1)
    }

    /// Relaxation moves each estimate toward the prior and leaves the prior
    /// itself, the per-request isolation and zero steps unchanged.
    @Test func relaxationReturnsEstimatesToThePrior() {
        let prior = CBv2MTPRequestAcceptanceState.prior
        for position in prior.indices {
            #expect(abs(prior[position] - 0.85 * pow(0.98, Double(position))) < 1e-12)
        }
        var fresh = CBv2MTPRequestAcceptanceState()
        fresh.relaxTowardPrior(steps: 500)
        #expect(fresh.probabilities == prior)

        var rejected = CBv2MTPRequestAcceptanceState()
        for _ in 0 ..< 40 {
            rejected.observe(draftedDepth: 7, acceptedDepth: 0, rejectionObserved: true)
        }
        let low = rejected.probabilities
        #expect(low[0] < 0.01)
        var unchanged = rejected
        unchanged.relaxTowardPrior(steps: 0)
        #expect(unchanged.probabilities == low)

        var once = rejected
        once.relaxTowardPrior()
        let rate = CBv2MTPRequestAcceptanceState.relaxationPerStep
        let expected = prior[0] + (low[0] - prior[0]) * (1 - rate)
        #expect(abs(once.probabilities[0] - expected) < 1e-12)

        var stepwise = rejected
        for _ in 0 ..< 10 { stepwise.relaxTowardPrior() }
        var batched = rejected
        batched.relaxTowardPrior(steps: 10)
        for position in prior.indices {
            #expect(abs(stepwise.probabilities[position] - batched.probabilities[position]) < 1e-12)
        }
    }

    /// A row whose drafts were all rejected chooses depth zero, and after
    /// enough target-only steps it is offered drafts again.
    @Test func aRejectedRowDraftsAgainAfterTargetOnlySteps() {
        let ratios = [1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        var state = CBv2MTPRequestAcceptanceState()
        for _ in 0 ..< 40 {
            state.observe(draftedDepth: 7, acceptedDepth: 0, rejectionObserved: true)
        }
        func choice(_ s: CBv2MTPRequestAcceptanceState) -> Int {
            CBv2MTPMarginalDepthPolicy.selectDepthByCost(
                offeredDepth: 7, remainingTokens: 1000, verificationLimit: 7,
                acceptanceProbabilities: s.probabilities, previousTargetTopTwoMargin: nil,
                costRatios: ratios)
        }
        #expect(choice(state) == 0)
        var steps = 0
        while choice(state) == 0, steps < 1000 {
            state.relaxTowardPrior()
            steps += 1
        }
        #expect(choice(state) > 0)
        #expect(steps > 8 && steps < 200, "re-offered after \(steps) target-only steps")
    }

    /// Only a per-row-commit drafter on the marginal policy gets the batch
    /// depth and the relaxation. A common-width drafter on the marginal
    /// policy (Qwen3.5's inline assistant) keeps the old cadence.
    @Test func batchDepthAndRelaxationNeedPerRowCommit() {
        #expect(
            CBv2MTPRoundDriver.batchDepthApplies(
                enabled: true, marginalPolicy: true, perRowCommit: true))
        #expect(
            !CBv2MTPRoundDriver.batchDepthApplies(
                enabled: true, marginalPolicy: true, perRowCommit: false))
        #expect(
            !CBv2MTPRoundDriver.batchDepthApplies(
                enabled: true, marginalPolicy: false, perRowCommit: true))
        #expect(
            !CBv2MTPRoundDriver.batchDepthApplies(
                enabled: false, marginalPolicy: true, perRowCommit: true))
    }

    /// The process switch is on unless set off.
    @Test func batchDepthSwitchDefaultsOn() {
        if ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_BATCH_DEPTH"] == nil {
            #expect(CBv2MTPRoundDriver.envBatchDepth)
        }
    }
}
