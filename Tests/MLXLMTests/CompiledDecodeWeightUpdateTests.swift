// Copyright © 2026 Apple Inc.
//
// A compiled decode path must read the weights a model has now, not the ones
// its traces were built with. Each test warms a model until its traces are
// compiled, changes every weight, and compares it against an identical model
// whose traces are built after the change. Any weight left out of a trace's
// declared compile state shows up as a mismatch.

//
// Fork: MLXLLM `Qwen35TextModel` keeps the fork's decode path, which has no
// compiled decode segments (fork.yaml section "Qwen 3.5 / 3.6 / 3.8 (hybrid
// DeltaNet MoE, MTP)"). The fork's compiled Qwen 3.5 decode is MLXVLM
// `Qwen35` (`Qwen35Language.Model`), so the Qwen 3.5 cases run on it, in
// fp16, the compiled decode dtype.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class CompiledDecodeWeightUpdateTests: XCTestCase {

    private func logits(
        _ model: some LanguageModel, tokens: [Int32], cache: [KVCache]
    ) -> [Float] {
        var result: [Float] = []
        for token in tokens {
            let output = model(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(output)
            result += output.asArray(Float.self)
        }
        return result
    }

    /// Scales every parameter of both models to the same new values.
    private func perturb(_ warm: some Module, _ reference: some Module) {
        let updated = warm.parameters().mapValues { $0 * 1.5 }
        warm.update(parameters: updated)
        reference.update(parameters: updated)
    }

    // MARK: - Qwen 3.5

    /// Tiny MLXVLM Qwen 3.5: 4 decoder layers (GDN, attention, GDN, attention)
    /// and a 2-layer vision tower that text-only decode never runs.
    private func qwen35Configuration(numExperts: Int) throws -> MLXVLM.Qwen35Configuration {
        let moe =
            numExperts > 0
            ? """
            "num_experts": \(numExperts), "num_experts_per_tok": 2,
            "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
            """ : ""
        let json = """
            {
                "model_type": "qwen3_5_vl",
                "image_token_id": 500, "video_token_id": 501,
                "vision_start_token_id": 502, "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_5",
                    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                    "vocab_size": 512, "full_attention_interval": 2,
                    "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                    "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096,
                    \(moe)
                    "rope_parameters": {
                        "type": "default", "mrope_section": [8, 4, 4],
                        "rope_theta": 100000.0, "partial_rotary_factor": 1.0
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                    "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "num_position_embeddings": 64
                }
            }
            """
        return try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self, from: Data(json.utf8))
    }

    /// An fp16 MLXVLM Qwen 3.5 in evaluation mode, the compiled decode setting.
    private func qwen35Model(_ configuration: MLXVLM.Qwen35Configuration) -> MLXVLM.Qwen35 {
        let model = MLXVLM.Qwen35(configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        model.train(false)
        return model
    }

    private func assertQwen35TracksWeights(
        numExperts: Int,
        usesSegments: Bool = true,
        cache makeCache: (MLXVLM.Qwen35) throws -> [KVCache] = {
            try $0.newCache(parameters: nil)
        }
    ) throws {
        let configuration = try qwen35Configuration(numExperts: numExperts)
        let (warm, reference) = withRandomState(MLXRandom.RandomState(seed: 17)) {
            let warm = qwen35Model(configuration)
            let reference = qwen35Model(configuration)
            reference.update(parameters: warm.parameters())
            return (warm, reference)
        }

        // Compile the traces against the original weights.
        _ = logits(warm, tokens: [1, 2, 3], cache: try makeCache(warm))
        if usesSegments {
            XCTAssertGreaterThan(warm.languageModel.model.compiledDecodeSegmentCount, 0)
        } else {
            XCTAssertEqual(warm.languageModel.model.compiledDecodeSegmentCount, 0)
            XCTAssertGreaterThan(warm.languageModel.model.compiledGDNTraceCount, 0)
        }

        perturb(warm, reference)

        let updated = logits(warm, tokens: [4, 5, 6], cache: try makeCache(warm))
        let expected = logits(reference, tokens: [4, 5, 6], cache: try makeCache(reference))
        XCTAssertEqual(updated, expected, "compiled decode used stale weights")
    }

    func testQwen35CompiledDecodeTracksWeightUpdates() throws {
        try assertQwen35TracksWeights(numExperts: 0)
    }

    func testQwen35MoECompiledDecodeTracksWeightUpdates() throws {
        try assertQwen35TracksWeights(numExperts: 4)
    }

    /// A quantized KV cache falls back to the per-layer traces, which are the
    /// other place Qwen 3.5 compiles module weights.
    func testQwen35PerLayerTracesTrackWeightUpdates() throws {
        try assertQwen35TracksWeights(numExperts: 4, usesSegments: false) { model in
            try model.newCache(parameters: nil).map { cache in
                cache is MambaCache ? cache : QuantizedKVCache(groupSize: 32, bits: 8)
            }
        }
    }

    // MARK: - Qwen 3 Next

    func testQwen3NextCompiledDecodeTracksWeightUpdates() throws {
        let json = """
            {
                "hidden_size": 16, "num_hidden_layers": 4, "intermediate_size": 32,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                "linear_num_value_heads": 2, "linear_num_key_heads": 2,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 16, "mlp_only_layers": [],
                "moe_intermediate_size": 16, "rms_norm_eps": 1e-6, "vocab_size": 32,
                "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                "max_position_embeddings": 128, "norm_topk_prob": true,
                "tie_word_embeddings": true, "attention_bias": false,
                "full_attention_interval": 2
            }
            """
        let configuration = try JSONDecoder().decode(
            Qwen3NextConfiguration.self, from: Data(json.utf8))

        // The compiled decode path is fp16 only.
        let (warm, reference) = withRandomState(MLXRandom.RandomState(seed: 19)) {
            let warm = Qwen3NextModel(configuration)
            warm.update(parameters: warm.parameters().mapValues { $0.asType(.float16) })
            let reference = Qwen3NextModel(configuration)
            reference.update(parameters: warm.parameters())
            return (warm, reference)
        }

        _ = logits(warm, tokens: [1, 2, 3], cache: try warm.newCache(parameters: nil))
        XCTAssertGreaterThan(warm.model.compiledDecodeSegmentCount, 0)

        perturb(warm, reference)

        let updated = logits(warm, tokens: [4, 5, 6], cache: try warm.newCache(parameters: nil))
        let expected = logits(
            reference, tokens: [4, 5, 6], cache: try reference.newCache(parameters: nil))
        XCTAssertEqual(updated, expected, "compiled decode used stale weights")
    }

    // MARK: - Adapters

    /// Loading an adapter into a model that has already run replaces modules
    /// under the traces. `LoRAContainer.load` invalidates them.
    func testLoRALoadAfterCompiledDecodeChangesOutput() throws {
        let configuration = try qwen35Configuration(numExperts: 0)
        let model = withRandomState(MLXRandom.RandomState(seed: 23)) {
            qwen35Model(configuration)
        }
        let cache = try model.newCache(parameters: nil)
        _ = logits(model, tokens: [1, 2, 3], cache: cache)
        XCTAssertGreaterThan(model.languageModel.model.compiledDecodeSegmentCount, 0)

        let baseline = logits(model, tokens: [4], cache: try model.newCache(parameters: nil))

        let rank = 4
        let adapter = LoRAContainer(
            configuration: .init(
                numLayers: 4,
                loraParameters: .init(rank: rank, scale: 1.0, keys: ["self_attn.q_proj"])),
            // q_proj maps hidden 64 to 4 heads x 32 x 2 (query and gate) = 256.
            parameters: ModuleParameters.unflattened([
                "language_model.model.layers.1.self_attn.q_proj.lora_a": MLXArray.full(
                    [64, rank], values: MLXArray(Float16(0.05))),
                "language_model.model.layers.1.self_attn.q_proj.lora_b": MLXArray.full(
                    [rank, 256], values: MLXArray(Float16(0.05))),
                "language_model.model.layers.3.self_attn.q_proj.lora_a": MLXArray.full(
                    [64, rank], values: MLXArray(Float16(0.05))),
                "language_model.model.layers.3.self_attn.q_proj.lora_b": MLXArray.full(
                    [rank, 256], values: MLXArray(Float16(0.05))),
            ]))
        try adapter.load(into: model)

        let adapted = logits(model, tokens: [4], cache: try model.newCache(parameters: nil))
        XCTAssertNotEqual(adapted, baseline, "compiled decode ignored the loaded adapter")

        adapter.unload(from: model)
        let restored = logits(model, tokens: [4], cache: try model.newCache(parameters: nil))
        XCTAssertEqual(restored, baseline, "compiled decode kept the unloaded adapter")
    }

    /// The training entry point replaces the same layers as `load(into:)`.
    /// Fresh adapters are zero-initialized, so the trace has to survive until
    /// the first optimizer step to be caught here.
    func testLoRAConversionAfterCompiledDecodeSeesTrainedAdapter() throws {
        let configuration = try qwen35Configuration(numExperts: 0)
        let model = withRandomState(MLXRandom.RandomState(seed: 29)) {
            qwen35Model(configuration)
        }
        _ = logits(model, tokens: [1, 2, 3], cache: try model.newCache(parameters: nil))
        XCTAssertGreaterThan(model.languageModel.model.compiledDecodeSegmentCount, 0)

        let baseline = logits(model, tokens: [4], cache: try model.newCache(parameters: nil))

        _ = try LoRAContainer.from(
            model: model,
            configuration: .init(
                numLayers: 4,
                loraParameters: .init(rank: 4, scale: 1.0, keys: ["self_attn.q_proj"])))

        let trainable = model.trainableParameters()
        XCTAssertFalse(trainable.isEmpty, "conversion produced no trainable adapter parameters")
        model.update(parameters: trainable.mapValues { $0 + 0.05 })

        let trained = logits(model, tokens: [4], cache: try model.newCache(parameters: nil))
        XCTAssertNotEqual(
            trained, baseline, "compiled decode ignored the adapter it was trained on")
    }
}
