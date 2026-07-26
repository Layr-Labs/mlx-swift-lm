import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("LagunaModel")
struct LagunaModelTests {

    static let tinyConfigJSON = """
        {
          "model_type": "laguna",
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "max_position_embeddings": 4096,
          "rms_norm_eps": 1e-6,
          "attention_bias": false,
          "qkv_bias": false,
          "tie_word_embeddings": false,
          "rope_theta": 500000.0,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
          "gating": "per-head",
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 8,
          "shared_expert_intermediate_size": 8,
          "moe_routed_scaling_factor": 1.0,
          "norm_topk_prob": true,
          "decoder_sparse_step": 1,
          "moe_router_logit_softcapping": 0.0
        }
        """

    private func tinyConfig() throws -> LagunaConfiguration {
        try JSONDecoder.json5().decode(
            LagunaConfiguration.self, from: Data(Self.tinyConfigJSON.utf8))
    }

    @Test func decodesTinyConfig() throws {
        let config = try tinyConfig()
        #expect(config.numHiddenLayers == 4)
        #expect(config.gatingEnabled)
        #expect(config.gatePerHead)
        // mlp_only_layers defaults to [0]: layer 0 dense, the rest sparse.
        #expect(!config.isSparse(layer: 0))
        #expect(config.isSparse(layer: 1))
    }

    @Test func forwardProducesLogitsAndFillsCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LagunaModel(try tinyConfig())
            eval(model)
            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 4)
            #expect(cache[0] is RotatingKVCache)
            #expect(cache[1] is KVCacheSimple)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let logits = model(tokens, cache: cache)
            eval(logits)
            #expect(logits.shape == [1, 3, 32])
            #expect(cache[0].offset == 3)
        }
    }

    @Test func factoryRegistersLaguna() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.tinyConfigJSON.utf8), modelType: "laguna")
        #expect(model is LagunaModel)
    }

    @Test func sanitizeDropsRotaryTables() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LagunaModel(try tinyConfig())
            let cleaned = model.sanitize(weights: [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2])
            ])
            #expect(cleaned.isEmpty)
        }
    }
}
