// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    @Suite("GPT-OSS decode command buffers")
    struct GPTOSSDecodeCommandBufferTests {
        private func gptOSS() throws -> GPTOSSModel {
            let json = """
                {
                    "model_type": "gpt_oss",
                    "num_hidden_layers": 4,
                    "num_local_experts": 4,
                    "num_experts_per_tok": 2,
                    "vocab_size": 64,
                    "rms_norm_eps": 1e-5,
                    "hidden_size": 32,
                    "intermediate_size": 32,
                    "head_dim": 8,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "sliding_window": 8
                }
                """
            let model = GPTOSSModel(
                try JSONDecoder.json5().decode(GPTOSSConfiguration.self, from: Data(json.utf8)))
            eval(model)
            return model
        }

        @Test func declaresDecodeLimitsOnlyThroughTheSteppableAdapter() throws {
            let adapter = CBv2SteppableLanguageModelAdapter(try gptOSS())
            #expect(adapter.decodeCommandBufferLimits == GPTOSSDecodeCommandBuffers.limits)
            // GPT-OSS has no MTP head, so it declares no speculative limits.
            #expect(adapter.speculativeCommandBufferLimits == nil)
            #expect(GPTOSSDecodeCommandBuffers.limits.operations == 100)
            #expect(GPTOSSDecodeCommandBuffers.limits.megabytes == 1 << 20)
        }

        @Test func decodeLimitsKeepExactBytes() throws {
            let model = try gptOSS()
            let tokens = MLXArray((0 ..< 24).map { Int32($0 % 64) }).reshaped(1, -1)
            let expected = model(tokens, cache: nil)
            eval(expected)
            let actual = GPU.withCommandBufferLimits(GPTOSSDecodeCommandBuffers.limits) {
                let logits = model(tokens, cache: nil)
                eval(logits)
                return logits
            }
            #expect(expected.asData(access: .copy).data == actual.asData(access: .copy).data)
        }
    }
}
