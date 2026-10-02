import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class MiMoV26DecodeRowsTests: XCTestCase {
    func testPlanUsesEveryScalarReferenceBoundary() {
        XCTAssertNil(MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 1024, deviceClass: "d"))
        XCTAssertEqual(MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 1026, deviceClass: "d"), 128)
        XCTAssertNil(MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 16384, deviceClass: "d"))
        XCTAssertEqual(MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 16386, deviceClass: "d"), 512)
        XCTAssertNil(MiMoV26DecodeRows.serialBlocks(rows: 4, keys: 65536, deviceClass: "d"))
        XCTAssertEqual(MiMoV26DecodeRows.serialBlocks(rows: 4, keys: 65539, deviceClass: "d"), 1024)
        XCTAssertNil(MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 8193, deviceClass: "s"))
        XCTAssertNil(MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 32769, deviceClass: "s"))
        XCTAssertEqual(
            MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 8193, deviceClass: "s", override: 65), 96)
        XCTAssertNil(
            MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 8193, deviceClass: "s", override: Int.max)
        )
        XCTAssertNil(MiMoV26DecodeRows.serialBlocks(rows: 1, keys: 8193, deviceClass: "d"))
        XCTAssertNil(MiMoV26DecodeRows.serialBlocks(rows: 5, keys: 8193, deviceClass: "d"))
        XCTAssertNil(
            MiMoV26DecodeRows.serialBlocks(rows: 3, keys: 8193, kvHeads: 8, deviceClass: "d"))
    }

    func testFullCausalRowsMatchActualScalarFP32PartialAttention() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_TEST_MIMO_DECODE_ROWS"] == "1",
            MiMoV26NAXGatherQMM.gpuStream(.default)
        else {
            throw XCTSkip("Requires exclusive native GPU qualification")
        }
        let name = GPU.deviceInfo().architecture
        guard name.hasPrefix("applegpu_"), let deviceClass = name.last else {
            throw XCTSkip("Unknown native Metal block policy")
        }
        let override = Int(ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"] ?? "") ?? 0
        for dtype in [DType.bfloat16, .float16] {
            for rows in 2 ... 4 {
                for keys in [1100, 4099, 16400, 65540] {
                    let blocks = try XCTUnwrap(
                        MiMoV26DecodeRows.serialBlocks(
                            rows: rows, keys: keys, deviceClass: deviceClass, override: override))
                    let q = (MLXRandom.normal([1, 64, rows, 192], key: MLXRandom.key(301)) * 0.2)
                        .asType(dtype)
                    let k = (MLXRandom.normal([1, 4, keys, 192], key: MLXRandom.key(302)) * 0.2)
                        .asType(dtype)
                    let v = (MLXRandom.normal([1, 4, keys, 128], key: MLXRandom.key(303)) * 0.2)
                        .asType(dtype)
                    for hasSinks in [false, true] {
                        let sinks =
                            hasSinks
                            ? MLXArray(Array(repeating: Float(-1.75), count: 64)).asType(dtype)
                            : nil
                        let actual = MiMoV26DecodeRows.launch(
                            queries: q, keys: k, values: v,
                            scale: 0.07216878, sinks: sinks, blocks: blocks)
                        let expected = concatenated(
                            (0 ..< rows).map { row in
                                MLXFast.scaledDotProductAttention(
                                    queries: q[0..., 0..., row ..< row + 1, 0...],
                                    keys: k[0..., 0..., ..<(keys - rows + row + 1), 0...],
                                    values: v[0..., 0..., ..<(keys - rows + row + 1), 0...],
                                    scale: 0.07216878, mask: .none, sinks: sinks)
                            }, axis: 2)
                        eval(actual, expected)
                        XCTAssertEqual(actual.shape, expected.shape)
                        XCTAssertEqual(
                            actual.asData(access: .copy).data, expected.asData(access: .copy).data,
                            "scalar parity dtype=\(dtype) rows=\(rows) keys=\(keys) sinks=\(hasSinks)"
                        )
                    }
                }
            }
        }
    }
}
