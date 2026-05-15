// Tests for Sampling.swift and repetition-penalty helpers — §3-4 of ContinuousBatchingTestPlan.md

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - §3  Sampling

final class CBSamplingTests: XCTestCase {

    // Helper: [1, vocab] logprob row.
    private func logprobs(_ values: [Float]) -> MLXArray {
        MLXArray(values)[.newAxis, .ellipsis]
    }

    func testApplyTopPFiltersLowProbTokens() {
        // One dominant token at index 2; others ~0.
        let lp = logprobs([-10, -10, 0, -10, -10])
        let filtered = applyTopP(lp, topP: 0.9)
        // All tokens other than token 2 should be masked to -inf.
        let flat = filtered.reshaped(-1).asArray(Float.self)
        let finite = flat.filter { $0.isFinite }
        XCTAssertEqual(finite.count, 1)
        XCTAssertTrue(flat[2].isFinite)
    }

    func testApplyMinPFiltersTokensBelowThreshold() {
        // token 0: log(0.8), token 1: log(0.1), token 2: log(0.1)
        // minP = 0.5 → threshold = 0.5 * 0.8 = 0.4 → keep only token 0
        let lp = logprobs([log(0.8), log(0.1), log(0.1)])
        let filtered = applyMinP(lp, minP: 0.5)
        let flat = filtered.reshaped(-1).asArray(Float.self)
        let finite = flat.filter { $0.isFinite }
        XCTAssertEqual(finite.count, 1, "only the dominant token survives minP filter")
        XCTAssertTrue(flat[0].isFinite)
    }

    func testTemperatureZeroIsGreedy() {
        let sampler = makeRowSampler(temperature: 0)
        // [1, 5] logit row; token 3 has highest logit.
        let logits = MLXArray([0.1, 0.2, 0.1, 5.0, 0.1] as [Float])[.newAxis, .ellipsis]
        for _ in 0 ..< 5 {
            XCTAssertEqual(sampler(logits).item(Int.self), 3)
        }
    }

    func testTopKOneAlwaysSelectsBestToken() {
        let sampler = makeRowSampler(temperature: 1.0, topK: 1)
        let logits = MLXArray([0.1, 5.0, 0.2, 0.1] as [Float])[.newAxis, .ellipsis]
        for _ in 0 ..< 5 {
            XCTAssertEqual(sampler(logits).item(Int.self), 1)
        }
    }
}

// MARK: - §4  Repetition penalty

final class CBRepetitionPenaltyTests: XCTestCase {

    private func baseGreedy() -> RowSampler { { argMax($0, axis: -1) } }

    // Helper: run sampler on a [1, vocab] logit tensor.
    private func sample(_ sampler: RowSampler, _ logits: [Float]) -> Int {
        sampler(MLXArray(logits)[.newAxis, .ellipsis]).item(Int.self)
    }

    func testRepetitionSamplerPassesThroughWithNoHistory() {
        let history = TokenHistoryHolder(tokens: [])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 2.0, presencePenalty: 0.5, frequencyPenalty: 0.5
        )
        // With no history the output must equal the base (greedy) sampler.
        let logits: [Float] = [0.1, 0.2, 3.0, 0.0]
        XCTAssertEqual(sample(sampler, logits), 2)
    }

    func testRepetitionSamplerReducesPositiveLogit() {
        // Token 0 has positive logit 2.0 and is in history → should be halved to 1.0.
        // Token 1 has logit 1.5 and is NOT in history → stays 1.5.
        // After penalty, token 1 should be chosen (1.5 > 1.0).
        let history = TokenHistoryHolder(tokens: [0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 2.0, presencePenalty: 0.0, frequencyPenalty: 0.0
        )
        // token 0 → 2.0/2 = 1.0; token 1 → 1.5 (unchanged)
        XCTAssertEqual(sample(sampler, [2.0, 1.5, 0.0, 0.0]), 1)
    }

    func testRepetitionSamplerAmplifiesNegativeLogit() {
        // Token 0 has negative logit -1.0 and is in history → multiplied by 2 = -2.0.
        // Token 1 has logit -0.5 → stays -0.5.
        // After penalty, token 1 should win (-0.5 > -2.0).
        let history = TokenHistoryHolder(tokens: [0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 2.0, presencePenalty: 0.0, frequencyPenalty: 0.0
        )
        XCTAssertEqual(sample(sampler, [-1.0, -0.5, -2.0, -3.0]), 1)
    }

    func testPresencePenaltyAppliesFlatDecrease() {
        // Token 0 at logit 2.0; presencePenalty = 1.0 → 2.0 - 1.0 = 1.0.
        // Token 1 at logit 1.5 (no history) → 1.5; token 1 wins.
        let history = TokenHistoryHolder(tokens: [0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 1.0, presencePenalty: 1.0, frequencyPenalty: 0.0
        )
        XCTAssertEqual(sample(sampler, [2.0, 1.5, 0.0, 0.0]), 1)
    }

    func testFrequencyPenaltyScalesWithCount() {
        // Token 0 appears 3 times in history with frequencyPenalty = 0.5.
        // Penalty = 0.5 * 3 = 1.5.  logit[0] = 3.0 - 1.5 = 1.5.
        // Token 1 at logit 1.6 (no history) → wins.
        let history = TokenHistoryHolder(tokens: [0, 0, 0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 1.0, presencePenalty: 0.0, frequencyPenalty: 0.5
        )
        XCTAssertEqual(sample(sampler, [3.0, 1.6, 0.0, 0.0]), 1)
    }
}
