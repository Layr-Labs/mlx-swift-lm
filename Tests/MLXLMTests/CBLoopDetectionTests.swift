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

    // MARK: - lookahead

    func testLookaheadShiftsThePredictionByOnePosition() {
        // Period-2 cycle "3 4" repeated 3 times.
        let tokens = [3, 4, 3, 4, 3, 4]
        // lookahead 0 (default): "immediate next" prediction.
        XCTAssertEqual(detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3), 3)
        // lookahead 1: one position further -- what makeLoopDetectionSampler
        // needs, since it's deciding the token AFTER the one `tokens` doesn't
        // know about yet (see makeLoopDetectionSampler's doc comment).
        XCTAssertEqual(
            detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3, lookahead: 1), 4)
    }

    func testLookaheadWithPeriodOneIsUnaffected() {
        // Period-1 cycles ("7 7 7") have the same value at every position, so
        // shifting the prediction forward by one doesn't change the answer.
        let tokens = [1, 2, 7, 7, 7]
        XCTAssertEqual(detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3), 7)
        XCTAssertEqual(
            detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3, lookahead: 1), 7)
    }

    func testLookaheadWithPeriodThreeShiftsByOnePosition() {
        // Period-3 cycle "1 2 3" repeated 3 times: immediate-next is 1
        // (tokens[n-3]); one-ahead is 2 (tokens[n-2]).
        let tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        XCTAssertEqual(detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3), 1)
        XCTAssertEqual(
            detectLoopContinuation(tokens: tokens, maxPatternSize: 8, minCount: 3, lookahead: 1), 2)
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

    func testBansTheCorrectlyShiftedLoopContinuationToken() {
        // History "3 4 3 4 3 4" (period 2). The naive "immediate next"
        // prediction is token 3, but `makeLoopDetectionSampler` needs
        // `lookahead: 1` (see its doc comment for why) -- the actually
        // correct continuation for THIS sampler's own token is 4. Token 4
        // has the highest logit; after the (correct) ban, token 5 wins.
        //
        // This is a regression test: before the lookahead fix, this sampler
        // would have banned token 3 instead -- a token that had already been
        // produced and couldn't be changed -- leaving token 4 free to win and
        // never breaking the cycle.
        let history = TokenHistoryHolder(tokens: [3, 4, 3, 4, 3, 4])
        let sampler = makeLoopDetectionSampler(
            base: baseGreedy(), history: history, maxPatternSize: 8, minCount: 3)
        XCTAssertEqual(sample(sampler, [0.0, 1.0, 2.0, 3.0, 9.0, 5.0]), 5)
    }

    func testOutOfRangeBannedTokenPassesThroughUnharmed() {
        // Period-1 cycle ("99 99 99") -- lookahead doesn't change the
        // predicted token for period 1 (see testLookaheadWithPeriodOneIsUnaffected),
        // so this stays a clean out-of-range case regardless of the shift:
        // continuation token 99 is outside the logits' vocab (4) and must
        // fall back to `base` untouched rather than crash.
        let history = TokenHistoryHolder(tokens: [1, 2, 99, 99, 99])
        let sampler = makeLoopDetectionSampler(
            base: baseGreedy(), history: history, maxPatternSize: 8, minCount: 3)
        XCTAssertEqual(sample(sampler, [0.0, 9.0, 1.0, 2.0]), 1)
    }
}
