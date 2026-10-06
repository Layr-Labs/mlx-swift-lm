import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// Run alone in the owned GPU lane. The original ordered Steel path is the
/// independent oracle; do not let the new default select both sides of a test.
final class Qwen4ParallelFullKVParityTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST"] == "1",
            "Requires the exclusive native GPU lane")
        for flag in [Qwen4ExpParallelQSA.fullKVFlag, Qwen4ExpParallelQSA.valuePartitionsFlag] {
            XCTAssertNil(ProcessInfo.processInfo.environment[flag], "Test the actual unset defaults")
        }
    }

    func testDefaultsAndExplicitPartitionsPreserveOrderedSteelBits() throws {
        XCTAssertTrue(Qwen4ExpNativeSparseGQA.steelEnabled())
        XCTAssertEqual(Qwen4ExpNativeSparseGQA.resolvedSteelTiles().keyTile, 64)
        XCTAssertEqual(Qwen4ExpNativeSparseGQA.resolvedSteelTiles().dimensionTile, 64)
        let initial = Qwen4ExpParallelQSAInvocation.snapshot()
        var cells = 0
        for dtype in [DType.bfloat16, .float16] {
            for keyTokens in [2053, 4103, 16389] {
                for width in 1...6 {
                    for capacityView in [false, true] {
                        MLXRandom.seed(UInt64(7117 + keyTokens + width))
                        let queries = capacityView
                            ? MLXRandom.normal([1, width, 24, 256]).asType(dtype).transposed(0, 2, 1, 3)
                            : MLXRandom.normal([1, 24, width, 256]).asType(dtype)
                        let capacity = keyTokens + (capacityView ? 16 : 0)
                        let keys = MLXRandom.normal([1, 2, capacity, 256]).asType(dtype)[0..., 0..., 0..<keyTokens, 0...]
                        let values = MLXRandom.normal([1, 2, capacity, 256]).asType(dtype)[0..., 0..., 0..<keyTokens, 0...]
                        let offset = keyTokens - width
                        let selected = (0..<width).flatMap { column -> [Int32] in
                            let completeBlocks = (offset + column + 1) / 4
                            return (0..<512).map { Int32($0 * completeBlocks / 512) }
                        }
                        let blocks = MLXArray(selected, [1, width, 512])
                        eval(queries, keys, values, blocks)
                        let reference = try XCTUnwrap(Qwen4ExpNativeSparseGQA.attend(
                            queries: queries, keys: keys, values: values, selectedBlocks: blocks,
                            qOffset: offset, outputPartitions: 1, preserveKVStrides: false,
                            parallelScores: false, parallelFullKV: false))
                        eval(reference)
                        let bits = reference.asType(.float32).asArray(Float.self).map(\.bitPattern)
                        // nil exercises both the unset full-KV switch and its
                        // new partition default. The other six are overrides.
                        for partitions: Int? in [nil, 1, 2, 4, 8, 16, 32] {
                            let candidate = try XCTUnwrap(Qwen4ExpNativeSparseGQA.attend(
                                queries: queries, keys: keys, values: values, selectedBlocks: blocks,
                                qOffset: offset, outputPartitions: 1, preserveKVStrides: false,
                                parallelScores: false, parallelValuePartitions: partitions,
                                parallelFullKV: partitions == nil ? nil : true))
                            eval(candidate)
                            XCTAssertEqual(candidate.shape, reference.shape)
                            XCTAssertEqual(candidate.dtype, reference.dtype)
                            XCTAssertEqual(candidate.asType(.float32).asArray(Float.self).map(\.bitPattern), bits,
                                "Output bits changed: dtype=\(dtype), keys=\(keyTokens), width=\(width), view=\(capacityView), partitions=\(String(describing: partitions))")
                            cells += 1
                        }
                    }
                }
            }
        }
        XCTAssertEqual(cells, 504)
        XCTAssertEqual(Qwen4ExpParallelQSAInvocation.snapshot() - initial, cells,
            "A reference/reference comparison must not silently pass")
        print("[qwen4-default-full-kv] bit_exact_cells=\(cells) actual_dispatches=\(cells)")
    }

    func testWiderPrefillRetainsOriginalDispatchAndOutput() throws {
        let initial = Qwen4ExpParallelQSAInvocation.snapshot()
        for width in [7, 16, 64] {
            MLXRandom.seed(UInt64(9331 + width))
            let queries = MLXRandom.normal([1, 24, width, 256]).asType(.bfloat16)
            let keys = MLXRandom.normal([1, 2, 2053, 256]).asType(.bfloat16)
            let values = MLXRandom.normal([1, 2, 2053, 256]).asType(.bfloat16)
            let blocks = broadcast(MLXArray((0..<512).map(Int32.init)).reshaped(1, 1, 512), to: [1, width, 512])
            let reference = try XCTUnwrap(Qwen4ExpNativeSparseGQA.attend(
                queries: queries, keys: keys, values: values, selectedBlocks: blocks,
                qOffset: 2053 - width, parallelFullKV: false))
            let candidate = try XCTUnwrap(Qwen4ExpNativeSparseGQA.attend(
                queries: queries, keys: keys, values: values, selectedBlocks: blocks,
                qOffset: 2053 - width))
            eval(reference, candidate)
            XCTAssertEqual(candidate.asType(.float32).asArray(Float.self).map(\.bitPattern),
                           reference.asType(.float32).asArray(Float.self).map(\.bitPattern))
        }
        XCTAssertEqual(Qwen4ExpParallelQSAInvocation.snapshot(), initial)
    }
}
