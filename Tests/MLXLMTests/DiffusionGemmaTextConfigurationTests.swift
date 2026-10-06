import Foundation
import MLXLLM
import Testing

@Suite("DiffusionGemma native text geometry")
struct DiffusionGemmaTextConfigurationTests {
    private func fixture() throws -> [String: Any] {
        let url = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        return try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    private func decode(_ values: [String: Any]) throws -> DiffusionGemmaTextConfiguration {
        try JSONDecoder().decode(
            DiffusionGemmaTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
    }

    @Test func publishedGeometryAndAllSemanticFieldsRoundTrip() throws {
        let config = try decode(fixture())
        #expect(config.modelType == "diffusion_gemma_text")
        #expect(config.layerCount == 30 && config.hiddenSize == 2816)
        #expect(config.maxPositionEmbeddings == 262144)
        #expect(config.keyValueHeads == 8 && config.globalKeyValueHeads == 2)
        #expect(config.headDimension == 256 && config.globalHeadDimension == 512)
        #expect(config.expertCount == 128 && config.topKExperts == 8)
        #expect(config.layerTypes.filter { $0 == "full_attention" }.count == 5)
        #expect(config.ropeParameters["full_attention"]?.partialRotaryFactor == 0.25)
        #expect(config.eosTokenIds == [1])
        #expect(
            try JSONDecoder().decode(
                DiffusionGemmaTextConfiguration.self,
                from: JSONEncoder().encode(config)) == config)
    }

    @Test func nondefaultsArePreservedRatherThanMerelyDecodable() throws {
        var values = try fixture()
        values["hidden_size"] = 64
        values["intermediate_size"] = 128
        values["moe_intermediate_size"] = 32
        values["num_hidden_layers"] = 2
        values["num_attention_heads"] = 4
        values["num_key_value_heads"] = 2
        values["num_global_key_value_heads"] = 1
        values["head_dim"] = 16
        values["global_head_dim"] = 32
        values["vocab_size"] = 128
        values["num_experts"] = 8
        values["top_k_experts"] = 2
        values["sliding_window"] = 33
        values["max_position_embeddings"] = 9123
        values["attention_bias"] = true
        values["attention_dropout"] = 0.1
        values["rms_norm_eps"] = 0.00002
        values["final_logit_softcapping"] = 21.5
        values["layer_types"] = ["sliding_attention", "full_attention"]
        values["eos_token_id"] = [0, 7]
        values["bos_token_id"] = 3
        values["pad_token_id"] = 4
        values["rope_parameters"] = [
            "sliding_attention": ["rope_type": "default", "rope_theta": 7777],
            "full_attention": [
                "rope_type": "proportional", "rope_theta": 20000, "partial_rotary_factor": 0.5,
            ],
        ]
        let config = try decode(values)
        let result = try JSONDecoder().decode(
            DiffusionGemmaTextConfiguration.self,
            from: JSONEncoder().encode(config))
        #expect(result == config)
        #expect(result.eosTokenIds == [0, 7] && result.slidingWindow == 33)
        #expect(result.maxPositionEmbeddings == 9123 && result.attentionBias)
        #expect(result.ropeParameters["full_attention"]?.theta == 20000)
    }

    @Test(arguments: [
        "model_type", "layer_types", "num_global_key_value_heads", "top_k_experts",
        "hidden_activation", "tie_word_embeddings", "use_bidirectional_attention",
        "per_layer_config", "rms_norm_eps", "eos_token_id",
    ])
    func rejectsIncompatibleGeometryAndUnimplementedModes(_ key: String) throws {
        var values = try fixture()
        let invalid: [String: Any] = [
            "model_type": "gemma4_text", "layer_types": ["full_attention"],
            "num_global_key_value_heads": 3, "top_k_experts": 129,
            "hidden_activation": "silu", "tie_word_embeddings": false,
            "use_bidirectional_attention": "all", "per_layer_config": ["5": ["head_dim": 64]],
            "rms_norm_eps": 0, "eos_token_id": 262144,
        ]
        values[key] = invalid[key]
        #expect(throws: (any Error).self) { try decode(values) }
    }
}
