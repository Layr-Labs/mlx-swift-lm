// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM

@Suite("Qwen3 forwardForDFlash")
struct QwenDFlashForwardTests {
    private func tinyQwen3Config(hiddenLayers: Int = 3) throws -> Qwen3Configuration {
        let json = """
        {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": \(hiddenLayers),
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128
        }
        """
        return try JSONDecoder.json5().decode(Qwen3Configuration.self, from: Data(json.utf8))
    }

    @Test func logitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen3Model(try tinyQwen3Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 2]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test func capturesRequestedHiddenStatesInRequestedOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen3Model(try tinyQwen3Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [2, 0]
            )

            #expect(forward.hiddenStates.count == 2)
            #expect(forward.hiddenStates[0].shape == [1, 3, 16])
            #expect(forward.hiddenStates[1].shape == [1, 3, 16])
            #expect(forward.targetHidden.shape == [1, 3, 32])
        }
    }

    @Test func rejectsInvalidLayerIds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen3Model(try tinyQwen3Config())
            let tokens = MLXArray([Int32(1), 2])[.newAxis, .ellipsis]

            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens,
                    cache: model.newCache(parameters: nil),
                    targetLayerIds: []
                )
            }

            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens,
                    cache: model.newCache(parameters: nil),
                    targetLayerIds: [0, 0]
                )
            }

            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens,
                    cache: model.newCache(parameters: nil),
                    targetLayerIds: [3]
                )
            }
        }
    }
}
