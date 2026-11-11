import Cmlx
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite("Gemma 4 dense GELU product", .serialized)
    struct Gemma4GeluProductTests {
        // The shared unary GELU has the same formula, constants and compile
        // gate as Gemma's original file-private unary implementation. Keep the
        // final multiply outside it to retain the pre-change rounding boundary.
        private func original(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
            safeGeluApproximate(gate) * up
        }

        private func expectExact(_ actual: MLXArray, _ expected: MLXArray) {
            eval(actual, expected)
            #expect(actual.shape == expected.shape)
            #expect(actual.dtype == expected.dtype)
            #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
        }

        private func graph(_ output: MLXArray) throws -> String {
            let namer = mlx_node_namer_new()
            defer { mlx_node_namer_free(namer) }
            let handles = [output.ctx]
            let roots = handles.withUnsafeBufferPointer {
                mlx_vector_array_new_data($0.baseAddress, $0.count)
            }
            defer { mlx_vector_array_free(roots) }
            let path = NSTemporaryDirectory() + "gemma4-gelu-product-\(UUID().uuidString).dot"
            defer { try? FileManager.default.removeItem(atPath: path) }
            let file = try #require(fopen(path, "w"))
            mlx_export_to_dot(file, namer, roots)
            fclose(file)
            return try String(contentsOfFile: path, encoding: .utf8)
        }

        @Test func everyFiniteLowPrecisionGatePreservesNativeBytes() {
            for dtype in [DType.float16, .bfloat16] {
                let exponentMask: UInt16 = dtype == .float16 ? 0x7c00 : 0x7f80
                let bits = (0 ... 65535).map { UInt16($0) }.filter {
                    $0 & exponentMask != exponentMask
                }
                let gate = MLXArray(bits).view(dtype: dtype)
                let variedUp = MLXArray(Array(bits.reversed())).view(dtype: dtype)
                eval(gate, variedUp)
                expectExact(gemma4GeluProduct(gate, variedUp), original(gate, variedUp))
                for value: Float in [-100, -1.00390625, -1, -0.5, -0.0, 0, 0.5, 1, 1.00390625, 100]
                {
                    let up = MLXArray([value]).asType(dtype)
                    eval(up)
                    expectExact(gemma4GeluProduct(gate, up), original(gate, up))
                }
            }
        }

        @Test func denseShapesAndStridedInputsPreserveNativeBytes() {
            for dtype in [DType.float16, .bfloat16, .float32] {
                for shape in [
                    [1, 1, 2112], [2, 1, 2112], [4, 1, 2112],
                    [1, 7, 2112], [1, 8, 2112], [1, 64, 2112], [1, 1, 4224],
                ] {
                    let gate = MLXRandom.normal(shape, key: MLXRandom.key(9101)).asType(dtype)
                    let up = MLXRandom.normal(shape, key: MLXRandom.key(9102)).asType(dtype)
                    eval(gate, up)
                    expectExact(gemma4GeluProduct(gate, up), original(gate, up))
                    let stridedGate = gate[.ellipsis, .stride(by: 2)]
                    let stridedUp = up[.ellipsis, .stride(by: 2)]
                    expectExact(
                        gemma4GeluProduct(stridedGate, stridedUp), original(stridedGate, stridedUp))
                    expectExact(
                        gemma4GeluProduct(gate.transposed(), up.transposed()),
                        original(gate.transposed(), up.transposed()))
                }
            }
        }

        @Test func exceptionalValuesAndMixedDtypesRetainOriginalPath() {
            let values: [Float] = [
                -.infinity, -.greatestFiniteMagnitude, -100, -1, -Float.leastNormalMagnitude,
                -Float.leastNonzeroMagnitude, -0.0, 0, Float.leastNonzeroMagnitude,
                Float.leastNormalMagnitude, 1, 100, .greatestFiniteMagnitude, .infinity, .nan,
            ]
            for gateType in [DType.float16, .bfloat16, .float32] {
                for upType in [DType.float16, .bfloat16, .float32] {
                    let gate = MLXArray(values).asType(gateType)
                    let up = MLXArray(Array(values.reversed())).asType(upType)
                    eval(gate, up)
                    expectExact(gemma4GeluProduct(gate, up), original(gate, up))
                }
            }
        }

        @Test func compiledCallersKeepShapeChangesAndGradients() {
            let expected = compile(shapeless: true) { (gate: MLXArray, up: MLXArray) in
                safeGeluApproximate(gate) * up
            }
            let actual = compile(shapeless: true) { (gate: MLXArray, up: MLXArray) in
                gemma4GeluProduct(gate, up)
            }
            for dtype in [DType.float16, .bfloat16, .float32] {
                for rows in [1, 2, 4, 7, 8, 4, 2, 1] {
                    let gate = MLXRandom.normal([rows, 2112], key: MLXRandom.key(9201)).asType(
                        dtype)
                    let up = MLXRandom.normal([rows, 2112], key: MLXRandom.key(9202)).asType(dtype)
                    eval(gate, up)
                    expectExact(actual(gate, up), expected(gate, up))
                }
                let gate = linspace(Float(-3), Float(3), count: 64).asType(dtype)
                let up = linspace(Float(-1), Float(1), count: 64).asType(dtype)
                eval(gate, up)
                expectExact(
                    grad({ sum(gemma4GeluProduct($0, up)) })(gate),
                    grad({ sum(safeGeluApproximate($0) * up) })(gate))
                expectExact(
                    grad({ sum(gemma4GeluProduct(gate, $0)) })(up),
                    grad({ sum(safeGeluApproximate(gate) * $0) })(up))
            }
        }

        @Test func cpuFallbackRetainsNativeBytes() {
            Stream.withNewDefaultStream(device: .cpu) {
                for dtype in [DType.float16, .bfloat16, .float32] {
                    let gate = linspace(Float(-4), Float(4), count: 128).asType(dtype)
                    let up = linspace(Float(1), Float(-1), count: 128).asType(dtype)
                    eval(gate, up)
                    expectExact(gemma4GeluProduct(gate, up), original(gate, up))
                }
            }
        }

        @Test func ordinaryForwardRemovesTheSeparateMultiply() throws {
            let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] ?? "<unset>"
            print(
                "[Gemma4GeluProductTests] MLX_COMPILED_DECODE=\(raw) effective=\(MLXHardwareInfo.isCompiledDecodeSupported)"
            )
            let gate = MLXArray.ones([1, 1, 2112], dtype: .bfloat16)
            let up = MLXArray.ones([1, 1, 2112], dtype: .bfloat16)
            eval(gate, up)
            let before = try graph(original(gate, up))
            let after = try graph(gemma4GeluProduct(gate, up))
            #expect(before.contains("[label =\"Multiply\""))
            if MLXHardwareInfo.isCompiledDecodeSupported {
                #expect(!after.contains("[label =\"Multiply\""))
                #expect(after.contains("Compiled"))
            } else {
                #expect(after.contains("[label =\"Multiply\""))
                #expect(!after.contains("Compiled"))
            }
        }
    }
}
