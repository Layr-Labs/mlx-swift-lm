// Copyright © 2026 Eigen Labs.
//
// Tests for the single-stream tail-loop detector (`TailLoopDetector` in
// `Evaluate.swift` / `LoopDetection.swift`) and its `GenerateParameters` /
// `PenaltyProcessor` wiring.

import MLX
import MLXLMCommon
import XCTest

public class LoopDetectionTests: XCTestCase {

    private let vocab = 100

    private func flatLogits(_ value: Float = 4.0) -> MLXArray {
        MLXArray.ones([1, vocab], type: Float32.self) * value
    }

    private func tok(_ id: Int) -> MLXArray {
        MLXArray([Int32(id)])
    }

    /// Drives `detector` through a sequence of "didSample" calls, simulating
    /// a generation history, without ever calling `prompt(_:)` (an empty
    /// prompt — the bug class this guards against is loop detection
    /// triggering on too little history, not an empty one).
    private func feed(_ detector: inout TailLoopDetector, _ tokens: [Int]) {
        detector.prompt(MLXArray.zeros([0], type: Int32.self))
        for t in tokens {
            detector.didSample(token: tok(t))
        }
    }

    // MARK: - Pass-through

    func testNonRepeatingHistoryPassesThrough() {
        var detector = TailLoopDetector(maxPatternSize: 8, minCount: 2)
        feed(&detector, [1, 2, 3, 4, 5, 6, 7])
        let out = detector.process(logits: flatLogits())
        for i in 0 ..< vocab {
            XCTAssertEqual(out[0, i].item(Float.self), 4.0, accuracy: 1e-4, "token \(i) should be untouched")
        }
    }

    func testEmptyHistoryPassesThrough() {
        var detector = TailLoopDetector(maxPatternSize: 8, minCount: 2)
        feed(&detector, [])
        let out = detector.process(logits: flatLogits())
        XCTAssertEqual(out[0, 0].item(Float.self), 4.0, accuracy: 1e-4)
    }

    func testTooFewRepeatsDoesNotTrigger() {
        // period 2, only 1 full repeat ([3,4]) before minCount=3's threshold.
        var detector = TailLoopDetector(maxPatternSize: 8, minCount: 3)
        feed(&detector, [9, 3, 4, 3, 4])
        let out = detector.process(logits: flatLogits())
        for i in 0 ..< vocab {
            XCTAssertEqual(out[0, i].item(Float.self), 4.0, accuracy: 1e-4, "token \(i) should be untouched")
        }
    }

    // MARK: - Detection

    func testDetectsSimpleCycleAndBansContinuation() {
        // period 2 ("3 4"), repeated 3 times → minCount=3 satisfied.
        // Next token to continue the cycle would be 3 (tokens[n-2]).
        var detector = TailLoopDetector(maxPatternSize: 8, minCount: 3)
        feed(&detector, [3, 4, 3, 4, 3, 4])
        let out = detector.process(logits: flatLogits())
        XCTAssertTrue(out[0, 3].item(Float.self).isInfinite)
        XCTAssertLessThan(out[0, 3].item(Float.self), 0)
        // unrelated tokens stay untouched.
        XCTAssertEqual(out[0, 4].item(Float.self), 4.0, accuracy: 1e-4)
        XCTAssertEqual(out[0, 50].item(Float.self), 4.0, accuracy: 1e-4)
    }

    func testDetectsSingleTokenRepeatAtPeriodOne() {
        // period 1 ("x"), repeated 3 times.
        var detector = TailLoopDetector(maxPatternSize: 8, minCount: 3)
        feed(&detector, [1, 2, 7, 7, 7])
        let out = detector.process(logits: flatLogits())
        XCTAssertTrue(out[0, 7].item(Float.self).isInfinite)
    }

    func testRespectsMaxPatternSize() {
        // period 5 cycle repeated 3 times needs maxPatternSize >= 5; capping
        // it at 4 must suppress detection entirely.
        var detector = TailLoopDetector(maxPatternSize: 4, minCount: 3)
        feed(&detector, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        let out = detector.process(logits: flatLogits())
        for i in 0 ..< vocab {
            XCTAssertEqual(out[0, i].item(Float.self), 4.0, accuracy: 1e-4, "token \(i) should be untouched")
        }
    }

    func testPrefersSmallestFundamentalPeriod() {
        // True period is 2 ("1 2"), repeated 4 times — also trivially
        // satisfies a "period 4" repeat-of-repeats check, but the smallest
        // matching period (2) is what should be reported: continuation == 1.
        var detector = TailLoopDetector(maxPatternSize: 16, minCount: 2)
        feed(&detector, [1, 2, 1, 2, 1, 2, 1, 2])
        let out = detector.process(logits: flatLogits())
        XCTAssertTrue(out[0, 1].item(Float.self).isInfinite)
        // token 2 (period-4-style false continuation) must NOT be banned.
        XCTAssertEqual(out[0, 2].item(Float.self), 4.0, accuracy: 1e-4)
    }

    /// A larger period can independently satisfy the repeat test with a
    /// *different* banned-token candidate than the true smallest period: here
    /// period 1 ("1") bans token 1, and period 3 ("0 1 1") independently also
    /// matches and would ban token 0. Only the smallest period's ban (token 1)
    /// must apply — token 0 must stay untouched.
    func testLargerPeriodMatchDoesNotAddExtraBan() {
        var detector = TailLoopDetector(maxPatternSize: 8, minCount: 2)
        feed(&detector, [0, 1, 1, 0, 1, 1])
        let out = detector.process(logits: flatLogits())
        XCTAssertTrue(out[0, 1].item(Float.self).isInfinite)
        XCTAssertEqual(out[0, 0].item(Float.self), 4.0, accuracy: 1e-4, "larger-period match must not add a ban")
    }

    // MARK: - 2-D prompt (VLM-shaped) safety

    /// VLM requests hand the processor a 2-D `[1, seq]` prompt token array
    /// (see `PenaltyLongPromptTests` for the historical `TokenRing` crash this
    /// pattern exposed). `TailLoopDetector` must handle it without crashing
    /// or mis-sizing its window.
    func test2DPromptThenSampleDoesNotCrash() {
        var detector = TailLoopDetector(maxPatternSize: 8, minCount: 3)
        // Flattened: 0,1,2,0,1,2,0,1,2,0 — a period-3 "0 1 2" cycle whose
        // next token (continuing the pattern at position 10 mod 3) is 1.
        let prompt2D = MLXArray((0 ..< 10).map { Int32($0 % 3) }).reshaped(1, 10)
        detector.prompt(prompt2D)
        let out = detector.process(logits: flatLogits())
        XCTAssertTrue(out[0, 1].item(Float.self).isInfinite)
        detector.didSample(token: tok(1))
        // should still process without crashing after appending.
        _ = detector.process(logits: flatLogits())
    }

    // MARK: - GenerateParameters / PenaltyProcessor wiring

    func testDisabledByDefault() {
        XCTAssertNil(GenerateParameters().processor())
    }

    func testLoopDetectionAloneBuildsAProcessor() {
        XCTAssertNotNil(GenerateParameters(loopDetectionMaxPatternSize: 8).processor())
    }

    func testZeroMaxPatternSizeDoesNotEnableLoopDetection() {
        // nil (default) and an explicit non-positive value both disable it;
        // with no other penalty set, processor() must be nil.
        XCTAssertNil(GenerateParameters(loopDetectionMaxPatternSize: nil).processor())
    }

    func testComposesWithRepetitionPenaltyAndBansLoopContinuation() {
        guard var processor = GenerateParameters(
            repetitionPenalty: 1.3, loopDetectionMaxPatternSize: 8, loopDetectionMinCount: 3
        ).processor() else {
            XCTFail("expected a composed processor")
            return
        }
        processor.prompt(MLXArray.zeros([0], type: Int32.self))
        for t in [3, 4, 3, 4, 3, 4] {
            processor.didSample(token: tok(t))
        }
        let out = processor.process(logits: flatLogits())
        XCTAssertTrue(out[0, 3].item(Float.self).isInfinite)
    }
}
