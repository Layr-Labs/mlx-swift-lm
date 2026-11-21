// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The family gate of `CBv2PrefillRowInvariance` and what a dense Qwen 3.5 model computes
/// inside MLX's prefill-row scope.
final class CBv2PrefillRowInvarianceTests: XCTestCase {

    private func configuration(moe: Bool) -> Data {
        Data(
            """
            {
                "model_type": "qwen3_5", "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_5_text", "hidden_size": 64,
                    "num_hidden_layers": 2, "intermediate_size": 128,
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                    "vocab_size": 512, "full_attention_interval": 2,
                    "tie_word_embeddings": false, "linear_num_value_heads": 2,
                    "linear_num_key_heads": 1, "linear_key_head_dim": 32,
                    "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
                    "num_experts": \(moe ? 4 : 0), "num_experts_per_tok": 2,
                    "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
                    "max_position_embeddings": 4096,
                    "rope_parameters": {"mrope_section": [8,4,4], "rope_theta": 100000,
                        "partial_rotary_factor": 1.0, "type": "default"}
                }
            }
            """.utf8)
    }

    private func model(moe: Bool = false) throws -> Qwen35Model {
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self, from: configuration(moe: moe))
        return withRandomState(MLXRandom.RandomState(seed: 17)) { Qwen35Model(config) }
    }

    func testOnlyQualifiedFamiliesTurnOn() {
        XCTAssertTrue(CBv2PrefillRowInvariance.qualified.contains(.qwen35Dense))
        XCTAssertTrue(
            CBv2PrefillRowInvariance.defaultOn.isSubset(of: CBv2PrefillRowInvariance.qualified))
        let family = CBv2PrefillRowInvariance.Family.qwen35Dense
        for raw in ["1", "on", "TRUE", " yes "] {
            XCTAssertTrue(CBv2PrefillRowInvariance.enabled(family, environment: raw), raw)
            XCTAssertFalse(CBv2PrefillRowInvariance.enabled(nil, environment: raw), raw)
        }
        for raw in ["0", "off", "False", "no"] {
            XCTAssertFalse(CBv2PrefillRowInvariance.enabled(family, environment: raw), raw)
        }
        let byDefault = CBv2PrefillRowInvariance.defaultOn.contains(family)
        for raw in [nil, "", "2", "auto"] {
            XCTAssertEqual(
                CBv2PrefillRowInvariance.enabled(family, environment: raw), byDefault,
                raw ?? "nil")
        }
    }

    func testReferenceRowsCoverTheLongestChunk() {
        typealias Gate = CBv2PrefillRowInvariance
        XCTAssertEqual(
            Gate.referenceRows(prefillChunkSize: 512, soloPrefillStripeTokens: nil), 4096)
        XCTAssertEqual(
            Gate.referenceRows(prefillChunkSize: 512, soloPrefillStripeTokens: 4096), 4096)
        XCTAssertEqual(
            Gate.referenceRows(prefillChunkSize: 512, soloPrefillStripeTokens: 16384), 16384)
        XCTAssertEqual(
            Gate.referenceRows(prefillChunkSize: 8192, soloPrefillStripeTokens: nil), 8192)
    }

    func testDenseQwen35IsTheQualifiedFamily() throws {
        let dense = try model()
        let moe = try model(moe: true)
        XCTAssertEqual(dense.prefillRowInvarianceFamily, .qwen35Dense)
        XCTAssertEqual(dense.languageModel.prefillRowInvarianceFamily, .qwen35Dense)
        XCTAssertNil(moe.prefillRowInvarianceFamily)

        typealias Gate = CBv2PrefillRowInvariance
        let adapter = CBv2SteppableLanguageModelAdapter(dense)
        let moeAdapter = CBv2SteppableLanguageModelAdapter(moe)
        XCTAssertEqual(adapter.prefillRowInvarianceFamily, .qwen35Dense)
        XCTAssertNil(moeAdapter.prefillRowInvarianceFamily)
        let config = CBv2SchedulerConfig(prefillChunkSize: 512, soloPrefillStripeTokens: 16384)
        XCTAssertEqual(Gate.engineRows(model: adapter, config: config, environment: "1"), 16384)
        XCTAssertEqual(
            Gate.engineRows(model: adapter, config: CBv2SchedulerConfig(), environment: "1"), 4096)
        XCTAssertEqual(Gate.engineRows(model: adapter, config: config, environment: "0"), 0)
        XCTAssertEqual(Gate.engineRows(model: moeAdapter, config: config, environment: "1"), 0)
        XCTAssertEqual(
            Gate.engineRows(model: adapter, config: config, environment: nil),
            Gate.defaultOn.contains(.qwen35Dense) ? 16384 : 0)
    }

    /// A 4-bit dense model (a gated delta layer and a full-attention layer) prefilled in
    /// chunks cut on multiples of 128 inside the scope: each chunk's logits have the bits of
    /// the same rows of the whole prompt in one call.
    func testChunkedPrefillInTheScopeKeepsTheWholePromptsBits() throws {
        let model = try model()
        quantize(model: model, groupSize: 32, bits: 4)
        model.apply { $0.dtype == .float32 ? $0.asType(.bfloat16) : $0 }
        eval(model)
        let length = 384
        let tokens = MLXArray((0 ..< length).map { ($0 * 13 + 7) % 480 }).reshaped(1, length)

        func prefill(_ cuts: [Int]) throws -> [MLXArray] {
            let cache = try model.newCache(parameters: nil)
            var start = 0
            var chunks: [MLXArray] = []
            for end in cuts + [length] {
                let logits = GPU.withPrefillRowInvariance(rows: 4096) {
                    model(
                        LMInput.Text(tokens: tokens[0..., start ..< end]), cache: cache,
                        state: nil
                    ).logits
                }
                eval(logits, cache)
                chunks.append(logits)
                start = end
            }
            return chunks
        }
        func sameBits(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape
                && arrayEqual(a.view(dtype: .uint16), b.view(dtype: .uint16)).item(Bool.self)
        }

        let whole = try prefill([])[0]
        XCTAssertEqual(whole.dtype, .bfloat16)
        for cuts in [[128], [256], [128, 256]] {
            let chunks = try prefill(cuts)
            var start = 0
            for (chunk, end) in zip(chunks, cuts + [length]) {
                XCTAssertTrue(
                    sameBits(chunk, whole[0..., start ..< end, 0...]),
                    "cuts \(cuts), rows \(start)..<\(end)")
                start = end
            }
        }
    }
}
