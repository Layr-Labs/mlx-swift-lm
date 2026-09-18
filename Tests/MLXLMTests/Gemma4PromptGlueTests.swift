// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import Testing
@testable import MLXLMCommon

/// Native-only gates, authored but not executed during source-only work.
/// Compare the two actual baseline expression shapes: dense unary GELU then
/// multiply, and compiled expert GeGLU, including strided packed views.
@Suite("Gemma4 prompt GeGLU", .serialized)
struct Gemma4PromptGlueTests {
    private func context() -> Gemma4PrefillGluePolicy.Context {
        Gemma4PrefillGluePolicy(environment: ["DARKBLOOM_GEMMA4_PREFILL_GLUE": "1",
            "DARKBLOOM_GEMMA4_PROMPT_GLUE": "1"])
            .context(scheduledPrefill: true, eligibleModel: true)!
    }

    private func input(_ shape: [Int], salt: Int) -> MLXArray {
        MLXArray((0..<shape.reduce(1, *)).map { Float(($0 * 13 + salt) % 257 - 128) / 32 }, shape)
            .asType(.bfloat16)
    }

    private func identical(_ actual: MLXArray, _ reference: MLXArray) {
        #expect(actual.shape == reference.shape && actual.dtype == reference.dtype)
        #expect(actual.asData(access: .copy).data == reference.asData(access: .copy).data)
    }

    @Test func splitAndPackedMatchCurrentBaselineExpressions() throws {
        let unary = compile(shapeless: true) { (x: MLXArray) in
            0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
        }
        let product = compile(shapeless: true) { (gate: MLXArray, up: MLXArray) in
            (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
        }
        for shape in [[1, 1024, 2112], [8192, 1, 704]] {
            for ready in [false, true] {
                let gate = input(shape, salt: 7), up = input(shape, salt: 11)
                if ready { eval(gate, up) }
                #expect(Gemma4PrefillGlueV1.alignedActivations([gate, up]) == ready)
                let split = try #require(Gemma4PromptGlueV1.geluProduct(
                    gate: gate, up: up, context: context(), compiledBaseline: true))
                identical(split, unary(gate) * up)
                identical(split, product(gate, up))
                let width = shape.last!
                let packed = concatenated([gate, up], axis: -1)
                if ready { eval(packed) }
                let actual = try #require(Gemma4PromptGlueV1.geluProductFusedPlane(
                    packed, hidden: width, context: context(), compiledBaseline: true))
                identical(actual, product(packed[.ellipsis, ..<width], packed[.ellipsis, width...]))
            }
        }
    }

    @Test func allBF16PatternsRetainByteEquality() throws {
        let product = compile(shapeless: true) { (gate: MLXArray, up: MLXArray) in
            (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
        }
        // 1024 * 704 is exactly eleven copies of all 65536 BF16 words.
        let gate = MLXArray((0..<(1024 * 704)).map { UInt16(truncatingIfNeeded: $0) }, [1024, 704])
            .view(dtype: .bfloat16)
        let up = MLXArray.ones([1024, 704], dtype: .bfloat16)
        let actual = try #require(Gemma4PromptGlueV1.geluProduct(
            gate: gate, up: up, context: context(), compiledBaseline: true))
        identical(actual, product(gate, up))
        // Includes non-finite payloads; a mismatch is reported, not hidden by
        // a relaxed tolerance or a challenge-specific token allowance.
    }

    @Test func unalignedAndDisabledCases() throws {
        let product = compile(shapeless: true) { (gate: MLXArray, up: MLXArray) in
            (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
        }
        let base = input([1024 * 704 + 1], salt: 17)
        let gate = base[1...].reshaped([1024, 704])
        let up = input([1024, 704], salt: 19)
        eval(gate, up)
        #expect(!Gemma4PrefillGlueV1.alignedActivations([gate, up]))
        let actual = try #require(Gemma4PromptGlueV1.geluProduct(
            gate: gate, up: up, context: context(), compiledBaseline: true))
        identical(actual, product(gate, up))
        #expect(Gemma4PromptGlueV1.geluProduct(gate: gate, up: up,
            context: context(), compiledBaseline: false) == nil)
        #expect(Gemma4PromptGlueV1.geluProduct(gate: gate, up: up,
            context: context(), compiledBaseline: true, stream: .cpu) == nil)
        #expect(Gemma4PromptGlueV1.geluProduct(gate: gate.asType(.float32), up: up,
            context: context(), compiledBaseline: true) == nil)
    }
}
