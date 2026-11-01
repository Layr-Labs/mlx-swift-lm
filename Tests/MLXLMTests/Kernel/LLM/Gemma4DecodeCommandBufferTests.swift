// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    @Suite("Gemma 4 decode command buffers")
    struct Gemma4DecodeCommandBufferTests {
        private func gemma4() throws -> Gemma4TextModel {
            let json = """
                {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 4,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 8,
                    "global_head_dim": 16,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention",
                                    "sliding_attention", "full_attention"],
                    "sliding_window": 16,
                    "final_logit_softcapping": 30.0,
                    "hidden_size_per_layer_input": 0,
                    "use_double_wide_mlp": false,
                    "tie_word_embeddings": true,
                    "vocab_size": 64,
                    "vocab_size_per_layer_input": 64,
                    "rms_norm_eps": 1e-6
                }
                """
            let model = Gemma4TextModel(
                try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: Data(json.utf8)))
            eval(model)
            return model
        }

        @Test func declaresDecodeLimitsOnlyThroughTheSteppableAdapter() throws {
            let adapter = CBv2SteppableLanguageModelAdapter(try gemma4())
            #expect(adapter.decodeCommandBufferLimits == Gemma4DecodeCommandBuffers.limits)
            // MTP rounds keep MLX's defaults.
            #expect(adapter.speculativeCommandBufferLimits == nil)
            #expect(Gemma4DecodeCommandBuffers.limits.operations == 100)
            #expect(Gemma4DecodeCommandBuffers.limits.megabytes == 1 << 20)
        }

        @Test func decodeLimitsKeepExactBytes() throws {
            let model = try gemma4()
            let tokens = MLXArray((0 ..< 24).map { Int32($0 % 64) }).reshaped(1, -1)
            let expected = model(tokens, cache: nil)
            eval(expected)
            let actual = GPU.withCommandBufferLimits(Gemma4DecodeCommandBuffers.limits) {
                let logits = model(tokens, cache: nil)
                eval(logits)
                return logits
            }
            #expect(expected.asData(access: .copy).data == actual.asData(access: .copy).data)
        }
    }
}
