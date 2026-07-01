// Copyright © 2026 Eigen Labs.
//
// Tests for the continuous-batching tail-loop detector
// (`detectLoopContinuation` / `makeLoopDetectionSampler` in
// `LoopDetectionSampler.swift`).

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - Pure pattern-matching logic

final class CBLoopDetectionPatternTests: XCTestCase {

    func testReturnsNilForNonRepeatingHistory() {
        XCTAssertNil(detectLoopContinuation(tokens: [1, 2, 3, 4, 5, 6], maxPatternSize: 8, minCount: 2))
    }

    func testReturnsNilForEmptyHistory() {
        XCTAssertNil(detectLoopContinuation(tokens: [], maxPatternSize: 8, minCount: 2))
    }

    func testDetectsSimplePeriod2Cycle() {
        // "3 4" repeated 3 times → continuation is tokens[n-2] == 3.
        let tokens = [9, 3, 4, 3, 4, 3, 4]
        XCTAssertEqual(detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3), 3)
    }

    func testDetectsSingleTokenRepeatAtPeriodOne() {
        let tokens = [1, 2, 7, 7, 7]
        XCTAssertEqual(detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3), 7)
    }

    func testRequiresExactlyMinCountRepeats() {
        // Only 2 repeats of "3 4" — minCount 3 must not trigger yet.
        XCTAssertNil(detectLoopContinuation(tokens: [9, 3, 4, 3, 4], maxPatternSize: 8, minCount: 3))
        // A 3rd repeat completes it.
        XCTAssertEqual(
            detectLoopContinuation(tokens: [9, 3, 4, 3, 4, 3, 4], maxPatternSize: 8, minCount: 3), 3)
    }

    func testRespectsMaxPatternSize() {
        // Period-5 cycle repeated 3 times; capping maxPatternSize below 5
        // must suppress detection.
        let tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        XCTAssertNil(detectLoopContinuation(tokens: tokens, maxPatternSize: 4, minCount: 3))
        XCTAssertEqual(detectLoopContinuation(tokens: tokens, maxPatternSize: 5, minCount: 3), 1)
    }

    func testPrefersSmallestFundamentalPeriod() {
        // True period is 2 ("1 2") repeated 4 times; must report continuation
        // 1 (period 2), not 2 (which a period-4 view would suggest).
        let tokens = [1, 2, 1, 2, 1, 2, 1, 2]
        XCTAssertEqual(detectLoopContinuation(tokens: tokens, maxPatternSize: 16, minCount: 2), 1)
    }

    func testMinCountBelowTwoIsRejected() {
        XCTAssertNil(detectLoopContinuation(tokens: [1, 1, 1, 1], maxPatternSize: 8, minCount: 1))
    }

    func testShortHistoryBelowMinCountReturnsNil() {
        XCTAssertNil(detectLoopContinuation(tokens: [1], maxPatternSize: 8, minCount: 2))
    }
}

// MARK: - On-device sampler wrapper

final class CBLoopDetectionSamplerTests: XCTestCase {

    private func baseGreedy() -> RowSampler { { argMax($0, axis: -1) } }

    private func sample(_ sampler: RowSampler, _ logits: [Float]) -> Int {
        sampler(MLXArray(logits)[.newAxis, .ellipsis]).item(Int.self)
    }

    func testPassesThroughWhenNoLoopDetected() {
        let history = TokenHistoryHolder(tokens: [1, 2, 3, 4, 5])
        let sampler = makeLoopDetectionSampler(
            base: baseGreedy(), history: history, maxPatternSize: 8, minCount: 3)
        // token 2 has the highest logit and is not banned.
        XCTAssertEqual(sample(sampler, [0.0, 0.1, 9.0, 0.2]), 2)
    }

    func testBansLoopContinuationToken() {
        // History "3 4 3 4 3 4" → continuation token 3 would be banned.
        // Token 3 starts with the highest logit (9.0); after the ban, token 1
        // (next highest) should win instead.
        let history = TokenHistoryHolder(tokens: [3, 4, 3, 4, 3, 4])
        let sampler = makeLoopDetectionSampler(
            base: baseGreedy(), history: history, maxPatternSize: 8, minCount: 3)
        XCTAssertEqual(sample(sampler, [0.0, 5.0, 1.0, 9.0]), 1)
    }

    func testOutOfRangeBannedTokenPassesThroughUnharmed() {
        // Loop continuation token (99) is outside the logits' vocab (4) —
        // must fall back to `base` untouched rather than crash.
        let history = TokenHistoryHolder(tokens: [99, 1, 99, 1, 99, 1])
        let sampler = makeLoopDetectionSampler(
            base: baseGreedy(), history: history, maxPatternSize: 8, minCount: 3)
        XCTAssertEqual(sample(sampler, [0.0, 9.0, 1.0, 2.0]), 1)
    }
}
