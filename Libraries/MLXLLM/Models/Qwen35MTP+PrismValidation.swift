// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon

/// Artifact-specific validation for the F16 and affine Q4 Prism 9B sidecars.
/// These checks establish compatibility; runtime qualification is separate.
extension Qwen35InlineMTPAssistant {
    /// Pure metadata checks keep unsupported Prism sizes out before allocation.
    static func validatePrism9BMetadata(
        _ metadata: Qwen35InlineMTPMetadata, against target: Qwen35TextConfiguration
    ) throws {
        try validatePrism9BQuantization(metadata.quantization)
        guard metadata.prefix.isEmpty, metadata.blockSize == 2,
            metadata.textConfiguration.mtpNumHiddenLayers == 1, target.mtpNumHiddenLayers == 0,
            target.modelType == "qwen3_5_text", target.hiddenSize == 4096,
            target.hiddenLayers == 32, target.intermediateSize == 12288,
            target.attentionHeads == 16, target.kvHeads == 4, target.headDim == 256,
            target.vocabularySize == 248320, target.fullAttentionInterval == 4,
            target.linearNumValueHeads == 32, target.linearNumKeyHeads == 16,
            target.linearKeyHeadDim == 128, target.linearValueHeadDim == 128,
            target.linearConvKernelDim == 4, target.maxPositionEmbeddings == 262144,
            target.ropeTheta == 10_000_000, target.partialRotaryFactor == 0.25,
            target.mropeSection == [11, 11, 10], target.rmsNormEps == 1e-6,
            !target.tieWordEmbeddings, !target.attentionBias,
            target.numExperts == 0, target.numExpertsPerTok == 0
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "only the one-layer Prism 9B sidecar with block_size=2 is supported")
        }
        var artifact = metadata.textConfiguration
        artifact.mtpNumHiddenLayers = target.mtpNumHiddenLayers
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard try encoder.encode(artifact) == encoder.encode(target),
            artifact.mropeSection == target.mropeSection
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "Prism sidecar text configuration differs from target")
        }
    }

    static func validatePrism9BQuantization(
        _ quantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        guard let quantization else { return }
        guard quantization.perLayerQuantization.isEmpty,
            let global = quantization.quantization,
            global.bits == 4, global.groupSize == 64, global.mode == .affine
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "Prism 9B sidecar requires F16 or uniform affine 4-bit/group-64 matrices")
        }
    }

    /// Include fields the native Codable configuration does not interpret.
    /// The sole permitted correction is the sidecar's proven one-layer count.
    static func validatePrism9BTextConfiguration(artifactData: Data, targetData: Data) throws {
        guard let artifact = try JSONSerialization.jsonObject(with: artifactData) as? [String: Any],
            let target = try JSONSerialization.jsonObject(with: targetData) as? [String: Any],
            var artifactText = artifact["text_config"] as? [String: Any],
            let targetText = target["text_config"] as? [String: Any],
            target["model_type"] as? String == "prism_hadamard_qwen35",
            artifactText["mtp_num_hidden_layers"] as? Int == 1,
            targetText["mtp_num_hidden_layers"] as? Int == 0,
            targetText["mtp_use_dedicated_embeddings"] as? Bool == false,
            targetText["attn_output_gate"] as? Bool == true,
            targetText["hidden_act"] as? String == "silu"
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "invalid Prism 9B sidecar/base text declaration")
        }
        artifactText["mtp_num_hidden_layers"] = 0
        guard NSDictionary(dictionary: artifactText).isEqual(to: targetText) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "Prism sidecar must retain the entire target text_config except mtp_num_hidden_layers"
            )
        }
        for key in ["quantization", "quantization_config"] {
            guard let raw = artifact[key], !(raw is NSNull) else { continue }
            guard let declaration = raw as? [String: Any],
                Set(declaration.keys).isSubset(of: ["group_size", "bits", "mode"]),
                declaration["group_size"] as? Int == 64,
                declaration["bits"] as? Int == 4,
                declaration["mode"] == nil || declaration["mode"] as? String == "affine"
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "Prism 9B sidecar quantization declarations must agree on affine 4-bit/group-64"
                )
            }
        }
    }

    static let prism9BWeightShapes: [String: [Int]] = [
        "fc.weight": [4096, 8192],
        "pre_fc_norm_hidden.weight": [4096],
        "pre_fc_norm_embedding.weight": [4096],
        "norm.weight": [4096],
        "layers.0.input_layernorm.weight": [4096],
        "layers.0.post_attention_layernorm.weight": [4096],
        "layers.0.self_attn.q_proj.weight": [8192, 4096],
        "layers.0.self_attn.k_proj.weight": [1024, 4096],
        "layers.0.self_attn.v_proj.weight": [1024, 4096],
        "layers.0.self_attn.o_proj.weight": [4096, 4096],
        "layers.0.self_attn.q_norm.weight": [256],
        "layers.0.self_attn.k_norm.weight": [256],
        "layers.0.mlp.gate_proj.weight": [12288, 4096],
        "layers.0.mlp.up_proj.weight": [12288, 4096],
        "layers.0.mlp.down_proj.weight": [4096, 12288],
    ]

    static func validatePrism9BWeights(
        _ weights: [String: (shape: [Int], dtype: DType)],
        quantization: BaseConfiguration.PerLayerQuantization? = nil
    ) throws {
        try validatePrism9BQuantization(quantization)
        var expected = prism9BWeightShapes.mapValues { (shape: $0, dtype: DType.float16) }
        if quantization != nil {
            for (key, shape) in prism9BWeightShapes where shape.count == 2 {
                let path = String(key.dropLast(".weight".count))
                expected[key] = ([shape[0], shape[1] / 8], .uint32)
                expected[path + ".scales"] = ([shape[0], shape[1] / 64], .float16)
                expected[path + ".biases"] = ([shape[0], shape[1] / 64], .float16)
            }
        }
        guard Set(weights.keys) == Set(expected.keys),
            weights.allSatisfy({ key, value in
                value.shape == expected[key]?.shape && value.dtype == expected[key]?.dtype
            })
        else {
            throw Qwen35InlineMTPError.invalidWeightIndex(
                "Prism 9B sidecar requires the exact published F16 or affine Q4 tensor inventory")
        }
    }
}
