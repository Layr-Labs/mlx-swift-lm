import Foundation
import MLX
import Testing

@_spi(DiffusionGemmaDiagnostics) @testable import MLXLLM

@Suite("DiffusionGemma soft-conditioning projection", .serialized)
struct DiffusionGemmaSoftEmbeddingTests {
    @Test func diagnosticsCountOnlyWhileExplicitlyArmed() {
        DiffusionGemmaSoftEmbeddingDiagnostics.clearAndArm()
        #expect(DiffusionGemmaSoftEmbeddingDiagnostics.snapshot().calls == 0)
        #expect(DiffusionGemmaSoftEmbeddingDiagnostics.snapshot().armed)
        DiffusionGemmaSoftEmbeddingDiagnostics.recordDispatch()
        DiffusionGemmaSoftEmbeddingDiagnostics.recordDispatch()
        let recorded = DiffusionGemmaSoftEmbeddingDiagnostics.snapshotAndDisarm()
        #expect(!recorded.armed && recorded.calls == 2)
        DiffusionGemmaSoftEmbeddingDiagnostics.recordDispatch()
        #expect(DiffusionGemmaSoftEmbeddingDiagnostics.snapshot() == recorded)
    }

    @Test func prototypeAndRollbackAreExplicit() {
        #expect(!DiffusionGemmaSoftEmbedding.enabled(from: nil))
        for value in ["1", "TRUE", "yes", "on"] {
            #expect(DiffusionGemmaSoftEmbedding.enabled(from: value))
        }
        for value in ["0", "false", "off", "", "unexpected"] {
            #expect(!DiffusionGemmaSoftEmbedding.enabled(from: value))
        }
    }

    @Test func geometryAndExecutionGuardsDoNotEvaluateLargeFixtures() {
        let x = MLXArray.zeros([1, 256, 262144], dtype: .bfloat16)
        let w = MLXArray.zeros([262144, 704], dtype: .uint32)
        let s = MLXArray.zeros([262144, 44], dtype: .bfloat16)
        func eligible(
            input: MLXArray? = nil, weight: MLXArray? = nil, scales: MLXArray? = nil,
            biases: MLXArray? = nil, missingBias: Bool = false, group: Int = 64,
            bits: Int = 8, mode: QuantizationMode = .affine, inference: Bool = true,
            enabled: Bool = true
        ) -> Bool {
            DiffusionGemmaSoftEmbedding.eligible(
                input ?? x, weight: weight ?? w, scales: scales ?? s,
                biases: missingBias ? nil : (biases ?? s), groupSize: group, bits: bits,
                mode: mode, inference: inference, enabled: enabled)
        }
        Device.withDefaultDevice(.gpu) {
            #expect(eligible())
            #expect(!eligible(enabled: false))
            #expect(!eligible(inference: false))
            #expect(!eligible(missingBias: true))
            #expect(!eligible(group: 32))
            #expect(!eligible(bits: 4))
            #expect(!eligible(mode: .mxfp4))
            for shape in [[256, 262144], [2, 256, 262144], [1, 128, 262144], [1, 256, 262080]] {
                #expect(!eligible(input: .zeros(shape, dtype: .bfloat16)))
            }
            for dtype in [DType.float16, .float32] {
                #expect(!eligible(input: x.asType(dtype)))
                #expect(!eligible(scales: s.asType(dtype)))
                #expect(!eligible(biases: s.asType(dtype)))
            }
            #expect(!eligible(weight: .zeros([262144, 703], dtype: .uint32)))
            #expect(!eligible(weight: w.asType(.int32)))
            #expect(!eligible(scales: .zeros([262144, 43], dtype: .bfloat16)))
            #expect(!eligible(biases: .zeros([262144, 43], dtype: .bfloat16)))
            Stream.withNewDefaultStream(device: .cpu) { #expect(!eligible()) }
            Stream.withNewDefaultStream(device: .gpu) { #expect(!eligible()) }
        }
        Device.withDefaultDevice(.cpu) {
            #expect(!eligible())
            Stream.withNewDefaultStream(device: .gpu) { #expect(!eligible()) }
        }
    }

    @Test func transformationsDeclineBeforeCreatingACustomPrimitive() {
        Device.withDefaultDevice(.cpu) {
            let input = MLXArray([Float(1), 2, 3, 4])
            var seen = Set<String>()
            func observed(_ x: MLXArray, _ name: String) -> MLXArray {
                #expect(!DiffusionGemmaSoftEmbedding.outsideTransform([x]))
                seen.insert(name)
                return x * x
            }
            #expect(DiffusionGemmaSoftEmbedding.outsideTransform([input]))
            eval(compile { (x: MLXArray) in observed(x, "compile") }(input))
            eval(grad { (x: MLXArray) in observed(x, "grad").sum() }(input))
            eval(vmap { (x: MLXArray) in observed(x, "vmap") }(input.reshaped(2, 2)))
            let (_, tangent) = jvp(
                { [observed($0[0], "jvp")] }, primals: [input], tangents: [.ones(like: input)])
            eval(tangent)
            #expect(seen == ["compile", "grad", "vmap", "jvp"])
            #expect(DiffusionGemmaSoftEmbedding.outsideTransform([input]))
        }
    }

    @Test func ineligibleProjectionPreservesTheOriginalGraph() {
        let x = MLXArray((0 ..< 192).map { Float($0 % 19 - 9) / 32 })
            .reshaped(1, 3, 64).asType(.bfloat16)
        let w = MLXArray((0 ..< 2048).map { UInt32($0 * 37 + 1) }).reshaped(64, 32)
        let s = MLXArray.full([64, 2], values: MLXArray(Float(0.125)), dtype: .bfloat16)
        let b = MLXArray.full([64, 2], values: MLXArray(Float(-0.5)), dtype: .bfloat16)
        let reference = quantizedMM(x, w, scales: s, biases: b, transpose: false, groupSize: 64, bits: 8)
        for (inference, enabled) in [(true, true), (true, false), (false, true)] {
            let actual = DiffusionGemmaSoftEmbedding.project(
                x, weight: w, scales: s, biases: b, groupSize: 64, bits: 8, mode: .affine,
                inference: inference, enabled: enabled)
            eval(reference, actual)
            #expect(actual.shape == reference.shape && actual.dtype == reference.dtype)
            #expect(actual.asData(access: .copy).data == reference.asData(access: .copy).data)
        }
    }
}
