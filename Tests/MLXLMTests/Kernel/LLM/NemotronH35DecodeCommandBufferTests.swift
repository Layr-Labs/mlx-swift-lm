// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    @Suite("Nemotron 3.5 decode command buffers")
    struct NemotronH35DecodeCommandBufferTests {
        /// The miniature Lightning topology of `NemotronHTests`: two Mamba layers separated by
        /// MoE, and one native attention layer.
        private func nemotron() throws -> NemotronH35Model {
            let json = """
                {"model_type":"nemotron_h", "vocab_size":100, "hidden_size":64,
                 "num_hidden_layers":4, "num_attention_heads":4, "num_key_value_heads":2,
                 "head_dim":64, "mamba_num_heads":4, "mamba_head_dim":16,
                 "ssm_state_size":16, "conv_kernel":4, "n_groups":2,
                 "intermediate_size":128, "moe_intermediate_size":64,
                 "moe_shared_expert_intermediate_size":64, "n_routed_experts":4,
                 "num_experts_per_tok":2, "layers_block_type":["mamba","moe","mamba","attention"],
                 "norm_eps":0.00001, "mamba_ssm_cache_dtype":"float32"}
                """
            let model = NemotronH35Model(
                try JSONDecoder().decode(NemotronH35Configuration.self, from: Data(json.utf8)))
            eval(model)
            return model
        }

        @Test func declaresDecodeLimitsOnlyThroughTheSteppableAdapter() throws {
            let adapter = CBv2SteppableLanguageModelAdapter(try nemotron())
            #expect(adapter.decodeCommandBufferLimits == NemotronH35DecodeCommandBuffers.limits)
            // MTP rounds keep MLX's defaults.
            #expect(adapter.speculativeCommandBufferLimits == nil)
            #expect(NemotronH35DecodeCommandBuffers.limits.operations == 100)
            #expect(NemotronH35DecodeCommandBuffers.limits.megabytes == 1 << 20)
        }

        @Test func decodeLimitsKeepExactBytes() throws {
            let model = try nemotron()
            let tokens = MLXArray((0 ..< 24).map { Int32($0 % 100) }).reshaped(1, -1)
            let expected = model(tokens, cache: nil)
            eval(expected)
            let actual = GPU.withCommandBufferLimits(NemotronH35DecodeCommandBuffers.limits) {
                let logits = model(tokens, cache: nil)
                eval(logits)
                return logits
            }
            #expect(expected.asData(access: .copy).data == actual.asData(access: .copy).data)
        }
    }
}
