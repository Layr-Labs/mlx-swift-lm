import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression tests for the FP8 loader of DeepSeek V4 (issue #194, defect 4).
///
/// An FP8 checkpoint stores `weight` with a `weight_scale_inv` for each
/// 128 x 128 block. `sanitize(weights:)` multiplies them out and must then
/// drop the `weight_scale_inv` keys, as mlx-lm `deepseek_v3.py` does.
/// Otherwise the strict update of the loader rejects the unused keys.
///
/// The test `loaderDequantizesBlockScaledWeights` is copied from
/// `DeepseekV4ForwardPassTests` of PR #184, without its `withKnownIssue`
/// block. `originalCheckpoint` and `load` are copied from the same test file
/// and from `SyntheticModel` of PR #183.
@Suite
struct DeepseekV4SanitizeTests {

    typealias Tiny = DeepseekV4TinyModel

    static func flatParameters(_ model: Module) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    }

    /// Writes `weights` to a `.safetensors` file in a new temporary folder and
    /// loads them into `model` with `loadWeights(modelDirectory:model:)`. The
    /// loader calls `sanitize(weights:)` and then updates the model with
    /// `verify: [.all]`, so the load fails when a key is not used.
    static func load(_ weights: [String: MLXArray], into model: DeepseekV4Model) throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("deepseek-v4-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try MLX.save(arrays: weights, url: folder.appendingPathComponent("model.safetensors"))
        try loadWeights(modelDirectory: folder, model: model)
    }

    /// The original DeepSeek checkpoint layout: top-level `embed`, `head` and
    /// `hc_head_*` names, `layers.N` without `model.`, flat `hc_attn_fn`
    /// names, the router bias as `gate.bias`, `w1`/`w2`/`w3` expert names, one
    /// tensor per expert and a 2-D `wo_a`.
    static func originalCheckpoint(_ model: DeepseekV4Model) -> [String: MLXArray] {
        var checkpoint: [String: MLXArray] = [:]
        let top = [
            "model.embed_tokens.weight": "embed.weight", "model.norm.weight": "norm.weight",
            "lm_head.weight": "head.weight", "model.hc_head.fn": "hc_head_fn",
            "model.hc_head.base": "hc_head_base", "model.hc_head.scale": "hc_head_scale",
        ]
        let expertNames = ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"]
        for (key, value) in flatParameters(model) {
            if let name = top[key] {
                checkpoint[name] = value
                continue
            }
            var name = String(key.dropFirst("model.".count))
            for sub in ["attn", "ffn"] {
                for param in ["fn", "base", "scale"] {
                    name = name.replacingOccurrences(
                        of: ".\(sub)_hc.\(param)", with: ".hc_\(sub)_\(param)")
                }
            }
            name = name.replacingOccurrences(
                of: ".ffn.gate.e_score_correction_bias", with: ".ffn.gate.bias")
            for (new, old) in expertNames {
                name = name.replacingOccurrences(
                    of: ".shared_experts.\(new).", with: ".shared_experts.\(old).")
            }
            if name.contains(".switch_mlp.") {
                for (new, old) in expertNames where name.contains(".\(new).") {
                    for expert in 0 ..< 4 {
                        let expertName = name.replacingOccurrences(
                            of: ".switch_mlp.\(new).", with: ".experts.\(expert).\(old).")
                        checkpoint[expertName] = value[expert]
                    }
                }
                continue
            }
            if name.hasSuffix("attn.wo_a.weight") {
                checkpoint[name] = value.reshaped(-1, value.dim(-1))
                continue
            }
            checkpoint[name] = value
        }
        return checkpoint
    }

    /// An FP8 checkpoint stores `weight` with a `weight_scale_inv` for each
    /// 128 x 128 block. `sanitize(weights:)` multiplies them out.
    @Test func loaderDequantizesBlockScaledWeights() throws {
        let reference = try Tiny.make(seed: 5)
        var checkpoint = Self.originalCheckpoint(reference)
        let key = "layers.0.attn.wq_a.weight"
        checkpoint[key] = checkpoint[key]! / 4
        checkpoint["layers.0.attn.wq_a.weight_scale_inv"] = MLXArray([4] as [Float])
            .reshaped(1, 1)

        let loaded = try Tiny.make(seed: 6)
        let sanitized = loaded.sanitize(weights: checkpoint)
        // Tolerance 1e-6: dividing by 4 and multiplying by 4 is exact in
        // float32, so the margin is for rounding only.
        #expect(
            Tiny.maxAbsDifference(
                sanitized["model.layers.0.attn.wq_a.weight"]!,
                Self.flatParameters(reference)["model.layers.0.attn.wq_a.weight"]!)
                <= 1e-6)
        #expect(
            !sanitized.keys.contains { $0.contains("weight_scale_inv") },
            "weight_scale_inv kept")
        try Self.load(checkpoint, into: loaded)

        // The loaded model gives the logits of the reference model.
        let rows = [Tiny.row(3, count: 9)]
        // Tolerance 1e-5: both models have the same weight values, so the
        // logits differ only in float32 rounding.
        #expect(
            Tiny.maxAbsDifference(Tiny.logits(reference, rows), Tiny.logits(loaded, rows))
                <= 1e-5)
    }
}
