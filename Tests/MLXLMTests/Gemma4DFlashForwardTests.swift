// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4TextModel forwardForDFlash")
struct Gemma4DFlashForwardTests {
    private func tinyGemma4Config(
        hiddenLayers: Int = 4,
        sharedLayers: Int = 0
    ) throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 16,
            "num_hidden_layers": \(hiddenLayers),
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "global_partial_rotary_factor": 1.0,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": \(sharedLayers),
            "hidden_size_per_layer_input": 0,
            "sliding_window": 16,
            "sliding_window_pattern": 2,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "attention_k_eq_v": false,
            "final_logit_softcapping": 30.0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyGemma4WrapperConfig() throws -> Gemma4Configuration {
        let json = """
        {
            "model_type": "gemma4",
            "vocab_size": 32,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 8,
                "global_partial_rotary_factor": 1.0,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 16,
                "sliding_window_pattern": 2,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128,
                "attention_k_eq_v": false,
                "final_logit_softcapping": 30.0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true
            }
        }
        """
        return try JSONDecoder.json5().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }

    @Test func logitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test func capturesRequestedHiddenStatesInRequestedOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [3, 1]
            )

            #expect(forward.hiddenStates.count == 2)
            #expect(forward.hiddenStates[0].shape == [1, 3, 16])
            #expect(forward.hiddenStates[1].shape == [1, 3, 16])
            #expect(forward.targetHidden.shape == [1, 3, 32])
        }
    }

    @Test func greedyTokensMatchSoftcappedLogitsArgmax() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )
            let greedy = try model.forwardGreedyTokensForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )

            #expect((greedy.tokens .== forward.logits.argMax(axis: -1)).all().item(Bool.self))
        }
    }

    @Test func capturesSharedKVLayerHiddenStates() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(
                try tinyGemma4Config(hiddenLayers: 4, sharedLayers: 2))
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [1, 3]
            )

            #expect(forward.logits.shape == [1, 3, 32])
            #expect(forward.targetHidden.shape == [1, 3, 32])
        }
    }

    @Test func drafterEmbeddingUsesGemmaInputScale() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2])[.newAxis, .ellipsis]

            let dFlashEmbedding = model.embedTokensForDFlash(tokens)
            let drafterEmbedding = model.embedTokensForDrafter(tokens)

            #expect(
                allClose(dFlashEmbedding, drafterEmbedding, rtol: 1e-5, atol: 1e-5)
                    .item(Bool.self))
        }
    }

    @Test func dFlashHiddenLogitsAreRawHeadOutput() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let hidden = MLXArray.ones([1, 2, 16]) * 1_000_000

            let rawLogits = model.logitsForDFlashHidden(hidden)

            #expect(MLX.abs(rawLogits).max().item(Float.self) > 30.0)
        }
    }

    @Test func wrapperDelegatesDFlashTargetSurfaceToTextModel() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4Model(try tinyGemma4WrapperConfig())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(model.dFlashVocabularySize == 32)
            #expect(model.dFlashHiddenSize == 16)
            #expect(model.dFlashLayerCount == 4)
            #expect(forward.targetHidden.shape == [1, 3, 32])
            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test func rejectsInvalidLayerIds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
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
                    targetLayerIds: [4]
                )
            }
        }
    }
}
