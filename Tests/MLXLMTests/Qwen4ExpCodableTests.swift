import Foundation
import XCTest

@testable import MLXLLM

/// Value-contract tests only: no model construction, weights or tensor execution.
final class Qwen4ExpCodableTests: XCTestCase {
    static let nonDefaultJSON = #"""
    {
      "model_type": "qwen4_exp",
      "hidden_size": 128, "num_hidden_layers": 4,
      "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 16,
      "linear_num_value_heads": 8, "linear_num_key_heads": 4,
      "linear_key_head_dim": 16, "linear_value_head_dim": 24,
      "linear_conv_kernel_dim": 3, "rms_norm_eps": 0.000002,
      "vocab_size": 64, "max_position_embeddings": 4096,
      "tie_word_embeddings": true, "attention_bias": true,
      "full_attention_interval": 2,
      "layer_types": ["linear_attention", "qwen_sparse_attention", "linear_attention", "qwen_sparse_attention"],
      "hc_count": 2, "hc_lowrank": 16,
      "ple_layer_ids": [1, 3], "ple_embed_dim": 32, "ple_conv_kernel_size": 5,
      "ngram_size": 4, "heads_per_ngram": 2, "ngram_vocab_size_base": 97,
      "make_ngram_vocab_size_divisible_by": 32, "split_ngram_parts": 4,
      "indexer_n_heads": 2, "indexer_kv_heads": 2, "indexer_head_dim": 16,
      "indexer_budget": 16, "indexer_compress_ratio": 2,
      "output_gate_type": "identity", "num_experts": 4, "num_experts_per_tok": 2,
      "shared_expert_intermediate_size": 32, "moe_intermediate_size": 48,
      "norm_topk_prob": false,
      "rope_parameters": {"rope_theta": 32768, "partial_rotary_factor": 0.5, "mrope_section": [2, 3, 7]},
      "eos_token_id": [0, 9], "mtp_num_hidden_layers": 1, "seed": 99
    }
    """#

    private func values(_ config: Qwen4ExpTextConfiguration) -> [String: String] {
        Dictionary(uniqueKeysWithValues: Mirror(reflecting: config).children.compactMap { child in
            child.label.map { ($0, String(reflecting: child.value)) }
        })
    }

    private func assertSameValues(
        _ expected: Qwen4ExpTextConfiguration, _ actual: Qwen4ExpTextConfiguration,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let before = values(expected), after = values(actual)
        XCTAssertEqual(Set(before.keys), Set(after.keys), file: file, line: line)
        for key in before.keys.sorted() {
            XCTAssertEqual(before[key], after[key], "Semantic field: \(key)", file: file, line: line)
        }
    }

    func testAllNondefaultSemanticFieldsRoundTrip() throws {
        let original = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data(Self.nonDefaultJSON.utf8))
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: data)
        assertSameValues(original, decoded)
        let supplied = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(Self.nonDefaultJSON.utf8)) as? [String: Any])
        let encoded = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(supplied.keys), Set(encoded.keys))
        let rope = try XCTUnwrap(encoded["rope_parameters"] as? [String: Any])
        XCTAssertEqual(rope["rope_theta"] as? Float, original.ropeTheta)
        XCTAssertEqual(rope["partial_rotary_factor"] as? Float, original.partialRotaryFactor)
        XCTAssertEqual(rope["mrope_section"] as? [Int], original.mropeSection)
    }

    func testScalarAndMultipleEOSPreserveZero() throws {
        for eos in ["0", "[0]", "[0, 9]", "[]"] {
            let data = Data("{\"eos_token_id\":\(eos),\"mtp_num_hidden_layers\":1,\"indexer_budget\":16}".utf8)
            let original = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: data)
            let restored = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: JSONEncoder().encode(original))
            assertSameValues(original, restored)
            XCTAssertEqual(restored.mtpNumHiddenLayers, 1)
            XCTAssertEqual(restored.indexerBudget, 16)
        }
    }

    func testOuterConfigurationPreservesTextAndMediaIdentifiers() throws {
        let text = try JSONSerialization.jsonObject(with: Data(Self.nonDefaultJSON.utf8))
        let data = try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen4_exp", "text_config": text,
            "image_token_id": 61, "video_token_id": 62,
        ])
        let original = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data)
        let restored = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(restored.modelType, original.modelType)
        XCTAssertEqual(restored.imageTokenId, 61)
        XCTAssertEqual(restored.videoTokenId, 62)
        assertSameValues(original.textConfig, restored.textConfig)
    }

    func testDecodedDependentDefaultsAndNormalizedLayerKindsRoundTrip() throws {
        let data = Data(#"{"hidden_size":128,"num_hidden_layers":2,"layer_types":["full_attention","linear_attention"]}"#.utf8)
        let original = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: data)
        XCTAssertEqual(original.pleEmbedDim, 128)
        XCTAssertEqual(original.layerTypes, ["qwen_sparse_attention", "linear_attention"])
        let restored = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: JSONEncoder().encode(original))
        assertSameValues(original, restored)
    }

    func testFlashNextEOSAndMTPContractRoundTrip() throws {
        let data = Data(#"{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","eos_token_id":[248044],"mtp_num_hidden_layers":1,"ple_layer_ids":[2]}}"#.utf8)
        let original = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data)
        let restored = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(restored.textConfig.eosTokenId, [248_044])
        XCTAssertEqual(restored.textConfig.mtpNumHiddenLayers, 1)
        XCTAssertEqual(restored.textConfig.pleLayerIds, [2])
        assertSameValues(original.textConfig, restored.textConfig)
    }
}
