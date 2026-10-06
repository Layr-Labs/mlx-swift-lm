import Foundation
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 DFlash2 configuration")
struct Qwen38DFlash2ConfigurationTests {
    @Test("decodes the pinned DFlash2 physical contract")
    func decodesPinnedContract() throws {
        let configuration = try JSONDecoder().decode(
            DFlash2Configuration.self,
            from: Data(Self.pinnedJSON.utf8))

        #expect(configuration.architectures == ["DFlash2DraftModel"])
        #expect(configuration.hiddenSize == 5_120)
        #expect(configuration.hiddenLayers == 5)
        #expect(configuration.attentionHeads == 32)
        #expect(configuration.keyValueHeads == 8)
        #expect(configuration.headDimension == 128)
        #expect(configuration.slidingWindow == 2_048)
        #expect(configuration.blockSize == 8)
        #expect(configuration.targetLayerIDs == [5, 19, 33, 47, 61])
        #expect(configuration.maskTokenID == 248_070)
        #expect(configuration.convKernelSize == 2)
        #expect(configuration.convGroupSize == 16)
        #expect(configuration.selectorRank == 256)
        #expect(configuration.selectorTopK == 16)
        try configuration.validatePinnedContract()
    }

    @Test("one-field geometry mutations fail before model construction")
    func rejectsGeometryMutation() throws {
        let mutated = Self.pinnedJSON.replacingOccurrences(
            of: "\"block_size\": 8", with: "\"block_size\": 7")
        let configuration = try JSONDecoder().decode(
            DFlash2Configuration.self,
            from: Data(mutated.utf8))

        #expect(throws: DFlash2ConfigurationError.self) {
            try configuration.validatePinnedContract()
        }
    }

    private static let pinnedJSON = """
        {
          "architectures": ["DFlash2DraftModel"],
          "attention_bias": false,
          "is_causal": false,
          "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]
          },
          "dtype": "bfloat16",
          "head_dim": 128,
          "hidden_size": 5120,
          "intermediate_size": 17408,
          "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention"
          ],
          "max_position_embeddings": 262144,
          "model_type": "qwen3",
          "num_attention_heads": 32,
          "num_hidden_layers": 5,
          "num_key_value_heads": 8,
          "num_target_layers": 64,
          "rms_norm_eps": 0.000001,
          "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
          "sliding_window": 2048,
          "tie_word_embeddings": false,
          "vocab_size": 248320
        }
        """
}
