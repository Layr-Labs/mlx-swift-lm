// Image composition semantics derived from NVIDIA modeling_nemotron_h_omni.py.
// Copyright (c) 2026 NVIDIA CORPORATION. Licensed under Apache License 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
// Distributed AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
// Reference revision: 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c.
import Foundation
import MLX
import MLXNN

/// Validates the pinned image-side architecture without decoding away unknown
/// language-model fields. This is not a model loader or a serving capability.
public struct NemotronSuperImageConfiguration {
    public let imageTokenID: Int32 = 18
    public let languageWidth = 4096
    public let languageConfigurationJSON: Data

    public enum Failure: Error { case invalidConfiguration(String) }

    public init(data: Data) throws {
        guard data.count <= 4 * 1024 * 1024,
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let vision = root["vision_config"] as? [String: Any],
            let language = root["llm_config"] as? [String: Any]
        else { throw Failure.invalidConfiguration("missing composite configuration") }
        func require(_ ok: Bool, _ field: String) throws {
            guard ok else { throw Failure.invalidConfiguration(field) }
        }
        func number(_ object: [String: Any], _ field: String, _ expected: Double) throws {
            guard let value = object[field] as? NSNumber,
                String(cString: value.objCType) != "c", value.doubleValue == expected
            else { throw Failure.invalidConfiguration(field) }
        }
        try require(root["model_type"] as? String == "nemotron_h_omni", "model_type")
        try require(
            root["architectures"] as? [String] == ["NemotronH_Omni_Reasoning_V3"], "architectures")
        try require(root["ps_version"] as? String == "v2", "ps_version")
        try require(root["img_context_token"] as? String == "<image>", "img_context_token")
        try require(root["sound_config"] is NSNull, "sound_config")
        for (key, value) in [
            "downsample_ratio": 0.5, "vit_hidden_size": 1280,
            "projector_hidden_size": 20480, "patch_size": 16, "img_context_token_id": 18,
        ] {
            try number(root, key, value)
        }
        try require(vision["dtype"] as? String == "bfloat16", "vision_config.dtype")
        try require(vision["model_type"] as? String == "radio", "vision_config.model_type")
        try require(vision["hidden_act"] as? String == "gelu", "vision_config.hidden_act")
        try require(vision["use_swiglu_ffn"] as? Bool == false, "vision_config.use_swiglu_ffn")
        try require(vision["qkv_bias"] as? Bool == true, "vision_config.qkv_bias")
        for (key, value) in [
            "hidden_size": 1280.0, "num_hidden_layers": 32,
            "num_attention_heads": 16, "num_channels": 3, "patch_size": 16,
            "max_img_size": 2048, "num_cls_tokens": 3, "num_registers": 7,
            "mlp_ratio": 4, "layer_norm_eps": 1e-6, "layerscale_value": 1,
            "attention_probs_dropout_prob": 0, "hidden_dropout_prob": 0, "drop_path_rate": 0,
        ] {
            try number(vision, key, value)
        }
        try require(language["model_type"] as? String == "nemotron_h", "llm_config.model_type")
        try number(language, "hidden_size", 4096)
        try number(language, "num_nextn_predict_layers", 1)
        // Retain the complete language config, including tagged Infinity and
        // moe_latent_size; passing it through the old decoder would lose semantics.
        languageConfigurationJSON = try JSONSerialization.data(
            withJSONObject: language, options: [.sortedKeys])
    }
}

/// Unregistered image encoder and placeholder substitution. The language trunk
/// remains caller-owned until its latent-MoE and mixed-precision loader are ready.
public final class NemotronSuperImageComposition: Module {
    @ModuleInfo(key: "vision_model") var tower: NemotronRadio
    @ModuleInfo(key: "vision_projector") var projector: NemotronSuperVisionProjector
    let visionDType: DType
    let width: Int
    let imageTokenID: Int32

    public init(configuration: NemotronSuperImageConfiguration) {
        self._tower.wrappedValue = NemotronRadio()
        self._projector.wrappedValue = NemotronSuperVisionProjector()
        self.visionDType = .bfloat16
        self.width = configuration.languageWidth
        self.imageTokenID = configuration.imageTokenID
    }

    init(
        tower: NemotronRadio, projector: NemotronSuperVisionProjector, width: Int,
        imageTokenID: Int32, visionDType: DType = .float32
    ) {
        self._tower.wrappedValue = tower
        self._projector.wrappedValue = projector
        self.visionDType = visionDType
        self.width = width
        self.imageTokenID = imageTokenID
    }

    public enum Failure: Error {
        case invalidPixels, invalidPrompt, invalidFeatures, placeholderCount
    }

    /// Validates the complete image tower and projector before mutating either.
    /// Other checkpoint components remain caller-owned. No model allocation or
    /// GPU evaluation occurs here; tensor payload/hash validation is upstream.
    /// Caller must serialize loading against inference on this module instance.
    public func loadImageWeights(from checkpoint: [String: MLXArray]) throws {
        let vision = try tower.validatedImageWeights(from: checkpoint)
        let projection = try projector.validatedProjectorWeights(from: checkpoint)
        let combined =
            vision.map { ("vision_model." + $0.key, $0.value) }
            + projection.map { ("vision_projector." + $0.key, $0.value) }
        try update(parameters: ModuleParameters.unflattened(combined), verify: [.all])
    }

    /// Each image is evaluated independently by the caller; this function only
    /// constructs its lazy graph. Caller owns memory admission and MLX fault checks.
    public func imageFeatures(normalizedPixels pixels: MLXArray) throws -> MLXArray {
        let d = tower.dimensions
        guard pixels.ndim == 4, pixels.dim(0) == 1, pixels.dim(1) == d.channels,
            pixels.dim(2) > 0, pixels.dim(3) > 0,
            pixels.dim(2) % (2 * d.patch) == 0, pixels.dim(3) % (2 * d.patch) == 0,
            [.float32, .float16, .bfloat16].contains(pixels.dtype)
        else { throw Failure.invalidPixels }
        let rows = pixels.dim(2) / d.patch
        let columns = pixels.dim(3) / d.patch
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard !overflow, count <= 13312 else { throw Failure.invalidPixels }
        return projector(
            tower(normalizedPixels: pixels.asType(visionDType)), height: rows, width: columns)
    }

    /// Images and their placeholder occurrences must already be ordered by the
    /// processor. No padding, truncation, hidden feature reuse or mask rewriting.
    public func substitute(
        prompt: [Int32], textEmbeddings: MLXArray,
        imageFeatures: [MLXArray]
    ) throws -> MLXArray {
        guard !prompt.isEmpty, textEmbeddings.shape == [1, prompt.count, width],
            [.float32, .float16, .bfloat16].contains(textEmbeddings.dtype)
        else { throw Failure.invalidPrompt }
        var count = 0
        for features in imageFeatures {
            guard features.ndim == 3, features.dim(0) == 1, features.dim(2) == width,
                features.dim(1) > 0, [.float32, .float16, .bfloat16].contains(features.dtype)
            else { throw Failure.invalidFeatures }
            let (next, overflow) = count.addingReportingOverflow(features.dim(1))
            guard !overflow, next <= prompt.count else { throw Failure.placeholderCount }
            count = next
        }
        guard prompt.filter({ $0 == imageTokenID }).count == count else {
            throw Failure.placeholderCount
        }
        guard count > 0 else { return textEmbeddings }
        let visual = concatenated(imageFeatures, axis: 1).asType(textEmbeddings.dtype)
        var cursor = 0
        var start = 0
        var pieces: [MLXArray] = []
        while start < prompt.count {
            let isImage = prompt[start] == imageTokenID
            var end = start + 1
            while end < prompt.count && (prompt[end] == imageTokenID) == isImage { end += 1 }
            if isImage {
                pieces.append(visual[0..., cursor ..< (cursor + end - start), 0...])
                cursor += end - start
            } else {
                pieces.append(textEmbeddings[0..., start ..< end, 0...])
            }
            start = end
        }
        return concatenated(pieces, axis: 1)
    }
}
