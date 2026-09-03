// CPU-only tests for the batch-1 adaptations of the Gemma 4 mlxfast port.
//
// Every test here exercises a PURE policy or index/tiling function — no MLX
// array is constructed, so the suite runs without a Metal device and never
// contends for the GPU.

import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Gemma4SinglePromptAdaptationTests: XCTestCase {

    // MARK: - B1-LADDER: decode async-eval submission boundaries

    /// The scored cohort geometry keeps its exact boundary set in both modes.
    func testLadderCohortGeometryUnchanged() {
        for anyBatch in [false, true] {
            for layer in 0 ..< 32 {
                let expected = (0 ... 3).contains(layer)
                XCTAssertEqual(
                    gemma4ShouldSubmitDecodeAsyncEvalLadder(
                        enabled: true, schedulePrefill: false, isCBv2: true,
                        batchSize: 8, inputLength: 1, layerIndex: layer,
                        anyBatch: anyBatch),
                    expected,
                    "layer \(layer) anyBatch=\(anyBatch)")
            }
        }
    }

    /// Batch one takes the same boundaries when admitted, and none when the
    /// kill switch restores the `batchSize == 8` pin.
    func testLadderBatchOneAdmission() {
        for layer in 0 ..< 32 {
            XCTAssertEqual(
                gemma4ShouldSubmitDecodeAsyncEvalLadder(
                    enabled: true, schedulePrefill: false, isCBv2: true,
                    batchSize: 1, inputLength: 1, layerIndex: layer,
                    anyBatch: true),
                (0 ... 3).contains(layer))
            XCTAssertFalse(
                gemma4ShouldSubmitDecodeAsyncEvalLadder(
                    enabled: true, schedulePrefill: false, isCBv2: true,
                    batchSize: 1, inputLength: 1, layerIndex: layer,
                    anyBatch: false))
        }
    }

    /// Every non-batch precondition still fails closed at batch one.
    func testLadderBatchOneStillFailsClosed() {
        let cases: [(Bool, Bool, Bool, Int)] = [
            (false, false, true, 1),  // master switch off
            (true, true, true, 1),  // prefill scheduled
            (true, false, false, 1),  // not CBv2
            (true, false, true, 2),  // multi-token input
        ]
        for (enabled, prefill, isCBv2, inputLength) in cases {
            XCTAssertFalse(
                gemma4ShouldSubmitDecodeAsyncEvalLadder(
                    enabled: enabled, schedulePrefill: prefill, isCBv2: isCBv2,
                    batchSize: 1, inputLength: inputLength, layerIndex: 0,
                    anyBatch: true))
        }
        XCTAssertFalse(
            gemma4ShouldSubmitDecodeAsyncEvalLadder(
                enabled: true, schedulePrefill: false, isCBv2: true,
                batchSize: 0, inputLength: 1, layerIndex: 0, anyBatch: true))
    }

    func testLadderAnyBatchSwitchParsing() {
        XCTAssertTrue(resolveGemma4DecodeLadderAnyBatchEnabled(nil))
        XCTAssertTrue(resolveGemma4DecodeLadderAnyBatchEnabled("1"))
        for off in ["0", "false", "no", "off", "OFF", "False"] {
            XCTAssertFalse(
                resolveGemma4DecodeLadderAnyBatchEnabled(off), "value \(off)")
        }
    }
}
