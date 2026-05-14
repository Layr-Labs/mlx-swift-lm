// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM

@Suite("Hybrid Qwen forwardForDFlash")
struct QwenHybridDFlashForwardTests {
    private func tinyQwen3NextConfig(hiddenLayers: Int = 4) throws -> Qwen3NextConfiguration {
        let json = """
        {
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "num_hidden_layers": \(hiddenLayers),
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 2,
            "num_experts": 0,
            "num_experts_per_tok": 1,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 0,
            "mlp_only_layers": [],
            "moe_intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "num_key_value_heads": 1,
            "rope_theta": 1000000,
            "partial_rotary_factor": 1.0,
            "max_position_embeddings": 128,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "head_dim": 32,
            "full_attention_interval": 2
        }
        """
        return try JSONDecoder.json5().decode(
            Qwen3NextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyQwen35Config(hiddenLayers: Int = 4) throws -> Qwen35Configuration {
        let json = """
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "num_hidden_layers": \(hiddenLayers),
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "rope_theta": 1000000,
                "partial_rotary_factor": 1.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "head_dim": 32,
                "full_attention_interval": 2,
                "num_experts": 0,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 0,
                "moe_intermediate_size": 16
            }
        }
        """
        return try JSONDecoder.json5().decode(Qwen35Configuration.self, from: Data(json.utf8))
    }

    @Test func qwen3NextLogitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = Qwen3NextModel(try tinyQwen3NextConfig())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [1, 3]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(forward.targetHidden.shape == [1, 3, 128])
        }
    }

    @Test func qwen35TopLevelLogitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = Qwen35Model(try tinyQwen35Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [1, 3]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(forward.hiddenStates.count == 2)
            #expect(forward.targetHidden.shape == [1, 3, 128])
        }
    }

    @Test func qwen35MoEInheritsDFlashTargetSurface() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = Qwen35MoEModel(try tinyQwen35Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [1, 3]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(forward.targetHidden.shape == [1, 3, 128])
        }
    }

    @Test func qwen3NextRejectsInvalidLayerIds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen3NextModel(try tinyQwen3NextConfig())
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
                    targetLayerIds: [4]
                )
            }
        }
    }
}
