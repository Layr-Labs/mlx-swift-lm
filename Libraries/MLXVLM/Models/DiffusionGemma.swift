// Native shared encoder + block denoiser for diffusion_gemma.
// Architecture references and license attribution: docs/diffusiongemma/implementation-references.md.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct DiffusionGemmaConfiguration: Codable, Sendable {
    public let textConfig: DiffusionGemmaTextConfiguration
    public let visionConfig: Gemma4VisionConfig?
    public let canvasLength: Int
    public let imageTokenId: Int
    public let videoTokenId: Int?
    public let beginImageTokenId: Int
    public let endImageTokenId: Int
    public let visionSoftTokensPerImage: Int
    public let tieWordEmbeddings: Bool
    public let dtype: String?
    public let initializerRange: Float
    public let base: BaseConfiguration
    public var modelType: String { base.modelType }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case canvasLength = "canvas_length"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case beginImageTokenId = "boi_token_id"
        case endImageTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case tieWordEmbeddings = "tie_word_embeddings"
        case dtype
        case initializerRange = "initializer_range"
    }
    enum VisionExtraKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
        case modelType = "model_type"
        case activation = "hidden_activation"
        case attentionBias = "attention_bias"
        case globalHeadDimension = "global_head_dim"
    }

    public init(from decoder: Decoder) throws {
        base = try BaseConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(DiffusionGemmaTextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decodeIfPresent(Gemma4VisionConfig.self, forKey: .visionConfig)
        canvasLength = try c.decodeIfPresent(Int.self, forKey: .canvasLength) ?? 256
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId)
        beginImageTokenId = try c.decodeIfPresent(Int.self, forKey: .beginImageTokenId) ?? 255999
        endImageTokenId = try c.decodeIfPresent(Int.self, forKey: .endImageTokenId) ?? 258882
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage)
            ?? visionConfig?.defaultOutputLength ?? 280
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype)
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        guard base.modelType == "diffusion_gemma", tieWordEmbeddings, canvasLength > 0,
            canvasLength <= textConfig.maxPositionEmbeddings,
            initializerRange.isFinite, initializerRange >= 0
        else { throw DiffusionGemmaModelError.invalidInput("root architecture") }
        if let visionConfig {
            let extras = try c.superDecoder(forKey: .visionConfig).container(
                keyedBy: VisionExtraKeys.self)
            let type =
                try extras.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_vision"
            let activation =
                try extras.decodeIfPresent(String.self, forKey: .activation) ?? "gelu_pytorch_tanh"
            let bias = try extras.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            let globalHead =
                try extras.decodeIfPresent(Int.self, forKey: .globalHeadDimension)
                ?? visionConfig.headDim
            let rope =
                try extras.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
                ?? [:]
            guard type == "gemma4_vision", activation == "gelu_pytorch_tanh", !bias,
                globalHead == visionConfig.headDim,
                rope.keys.allSatisfy({ ["rope_type", "rope_theta"].contains($0) }),
                rope["rope_type"] == nil || rope["rope_type"] == .string("default")
            else { throw DiffusionGemmaModelError.invalidInput("unsupported vision architecture") }
            guard !visionConfig.useClippedLinears, visionConfig.rmsNormEps == 1e-6,
                visionConfig.patchSize > 0, visionConfig.poolingKernelSize > 0,
                visionConfig.hiddenSize > 0, visionConfig.numHiddenLayers > 0,
                visionConfig.numAttentionHeads > 0, visionConfig.numKeyValueHeads > 0,
                visionConfig.headDim > 0, visionConfig.headDim.isMultiple(of: 4),
                visionConfig.numAttentionHeads.isMultiple(of: visionConfig.numKeyValueHeads),
                visionConfig.positionEmbeddingSize > 0, visionConfig.ropeTheta.isFinite,
                visionConfig.ropeTheta > 0,
                visionConfig.defaultOutputLength > 0,
                visionSoftTokensPerImage > 0, imageTokenId >= 0,
                imageTokenId < textConfig.vocabularySize
            else { throw DiffusionGemmaModelError.invalidInput("vision configuration") }
        }
    }

    public func encode(to encoder: Encoder) throws {
        // Preserve the entire per-module quantization map and EOS through the
        // shared metadata contract, not just the default bit depth.
        try base.encode(to: encoder)
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(textConfig, forKey: .textConfig)
        if let visionConfig {
            let visionEncoder = c.superEncoder(forKey: .visionConfig)
            try visionConfig.encode(to: visionEncoder)
            var extra = visionEncoder.container(keyedBy: VisionExtraKeys.self)
            try extra.encode(
                [
                    "rope_theta": StringOrNumber.float(visionConfig.ropeTheta),
                    "rope_type": .string("default"),
                ], forKey: .ropeParameters)
            try extra.encode("gemma4_vision", forKey: .modelType)
            try extra.encode("gelu_pytorch_tanh", forKey: .activation)
            try extra.encode(false, forKey: .attentionBias)
            try extra.encode(visionConfig.headDim, forKey: .globalHeadDimension)
        }
        try c.encode(canvasLength, forKey: .canvasLength)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encodeIfPresent(videoTokenId, forKey: .videoTokenId)
        try c.encode(beginImageTokenId, forKey: .beginImageTokenId)
        try c.encode(endImageTokenId, forKey: .endImageTokenId)
        try c.encode(visionSoftTokensPerImage, forKey: .visionSoftTokensPerImage)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encodeIfPresent(dtype, forKey: .dtype)
        try c.encode(initializerRange, forKey: .initializerRange)
    }
}

/// DiffusionGemma normalizes vision features BEFORE the learned projection.
/// Do not substitute the existing AR wrapper's post-projection normalization.
final class DiffusionGemmaVisionProjection: Module {
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    let epsilon: Float
    init(vision: Gemma4VisionConfig, hiddenSize: Int) {
        epsilon = vision.rmsNormEps
        _projection.wrappedValue = Linear(vision.hiddenSize, hiddenSize, bias: false)
    }
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        projection(MLXFast.rmsNorm(features, weight: .mlxNone, eps: epsilon))
    }
}

public final class DiffusionGemmaEncoder: Module {
    @ModuleInfo(key: "language_model") public var languageModel: DiffusionGemmaEncoderTextParameters
    @ModuleInfo(key: "vision_tower") var visionTower: Gemma4VisionTower?
    @ModuleInfo(key: "embed_vision") var embedVision: DiffusionGemmaVisionProjection?
    init(_ config: DiffusionGemmaConfiguration) {
        _languageModel.wrappedValue = DiffusionGemmaEncoderTextParameters(
            layerCount: config.textConfig.layerCount)
        if let vision = config.visionConfig {
            _visionTower.wrappedValue = Gemma4VisionTower(vision, contract: .diffusionGemma)
            _embedVision.wrappedValue = DiffusionGemmaVisionProjection(
                vision: vision, hiddenSize: config.textConfig.hiddenSize)
        }
    }
}

public final class DiffusionGemmaBackbone: Module {
    @ModuleInfo public var encoder: DiffusionGemmaEncoder
    @ModuleInfo public var decoder: DiffusionGemmaTextDecoder
    init(_ config: DiffusionGemmaConfiguration) {
        _encoder.wrappedValue = DiffusionGemmaEncoder(config)
        _decoder.wrappedValue = DiffusionGemmaTextDecoder(config.textConfig)
    }
}

/// Native model surface. Generic autoregressive registration is intentionally
/// separate: a loaded diffusion model must never fall through TokenIterator.
public final class DiffusionGemma: Module, BaseLanguageModel, QuantizationPathAliasing {
    public let configuration: DiffusionGemmaConfiguration
    @ModuleInfo public var model: DiffusionGemmaBackbone
    public init(_ configuration: DiffusionGemmaConfiguration) {
        self.configuration = configuration
        _model.wrappedValue = DiffusionGemmaBackbone(configuration)
    }
    public func makeCache(expectedPromptLength: Int, maximumSequenceLength: Int? = nil,
                          pagedBackend: PagedKVBackend? = nil) throws -> DiffusionGemmaRequestCache {
        try DiffusionGemmaRequestCache(
            configuration: configuration.textConfig,
            expectedPromptLength: expectedPromptLength,
            maximumSequenceLength: maximumSequenceLength, pagedBackend: pagedBackend)
    }

    public func encode(
        tokenIds: MLXArray, cache: DiffusionGemmaRequestCache,
        pixelValues: MLXArray? = nil, visualOutputLengths: [Int]? = nil,
        visualBlockIds: MLXArray? = nil
    ) throws -> MLXArray {
        guard tokenIds.ndim == 2, tokenIds.dim(0) == 1, tokenIds.dim(1) > 0,
            cache.configuration == configuration.textConfig,
            tokenIds.dim(1) <= configuration.textConfig.maxPositionEmbeddings - cache.position
        else { throw DiffusionGemmaModelError.invalidInput("encoder input/capacity") }
        // Same placeholder substitution as the native reference. Real media
        // preparation/HTTP validation owns presence and provenance checks.
        var imageMask = tokenIds .== configuration.imageTokenId
        if let video = configuration.videoTokenId {
            imageMask = logicalOr(imageMask, tokenIds .== video)
        }
        // The released config may omit video_token_id. Its processor still
        // supplies video positions through native media types (resolved here
        // to contiguous visual block IDs). Those positions must select both
        // feature substitution AND attention, not just the attention overlay.
        if let visualBlockIds {
            guard visualBlockIds.shape == tokenIds.shape, visualBlockIds.dtype == .int32,
                (visualBlockIds .>= -1).all().item(Bool.self)
            else { throw DiffusionGemmaModelError.invalidInput("visual block identity") }
            imageMask = logicalOr(imageMask, visualBlockIds .>= 0)
        }
        let ids = which(
            imageMask, MLXArray(Int32(configuration.textConfig.padTokenId ?? 0)), tokenIds)
        var embeddings = try model.decoder.scaledEmbeddings(ids)
        var blockIds = visualBlockIds
        if let pixelValues {
            guard let tower = model.encoder.visionTower, let projector = model.encoder.embedVision,
                let vision = configuration.visionConfig,
                pixelValues.ndim == 4, pixelValues.dtype.isFloatingPoint, pixelValues.dim(0) > 0,
                pixelValues.dim(1) == 3,
                pixelValues.dim(2) > 0, pixelValues.dim(3) > 0,
                pixelValues.dim(2).isMultiple(of: vision.patchSize * vision.poolingKernelSize),
                pixelValues.dim(3).isMultiple(of: vision.patchSize * vision.poolingKernelSize)
            else { throw DiffusionGemmaModelError.invalidInput("vision inputs") }
            let actualLength =
                (pixelValues.dim(2) / (vision.patchSize * vision.poolingKernelSize))
                * (pixelValues.dim(3) / (vision.patchSize * vision.poolingKernelSize))
            let lengths =
                visualOutputLengths
                ?? Array(
                    repeating: actualLength,
                    count: pixelValues.dim(0))
            guard lengths.count == pixelValues.dim(0),
                actualLength > 0, actualLength <= 1120,
                lengths.allSatisfy({ $0 == actualLength })
            else {
                throw DiffusionGemmaModelError.invalidInput("vision output budget")
            }
            let flags = imageMask.asArray(Bool.self)
            let positions = flags.enumerated().compactMap { $0.element ? Int32($0.offset) : nil }
            guard positions.count == lengths.reduce(0, +) else {
                throw DiffusionGemmaModelError.invalidInput("vision placeholder count")
            }
            var projected = [MLXArray]()
            for index in 0 ..< pixelValues.dim(0) {
                let image = pixelValues[index ..< (index + 1), 0..., 0..., 0...]
                let maxPatches =
                    lengths[index] * vision.poolingKernelSize * vision.poolingKernelSize
                guard image.dim(3) / vision.patchSize <= maxPatches else {
                    throw DiffusionGemmaModelError.invalidInput("vision grid width")
                }
                projected.append(projector(tower(image, outputLength: lengths[index])))
            }
            let features = concatenated(projected, axis: 1).asType(embeddings.dtype)
            let flat = embeddings.reshaped(-1, configuration.textConfig.hiddenSize)
            flat[MLXArray(positions)] = features.reshaped(-1, configuration.textConfig.hiddenSize)
            embeddings = flat.reshaped(embeddings.shape)
            if blockIds == nil {
                var next: Int32 = -1
                var previous = false
                let ids: [Int32] = flags.map { isVisual in
                    if isVisual && !previous { next += 1 }
                    previous = isVisual
                    return isVisual ? next : -1
                }
                blockIds = MLXArray(ids).reshaped(tokenIds.shape)
            }
        }
        return try model.decoder.encode(
            tokenIds: tokenIds, cache: cache,
            encoderParameters: model.encoder.languageModel, preparedEmbeddings: embeddings,
            visualBlockIds: blockIds)
    }

    public func denoise(
        canvasIds: MLXArray, cache: DiffusionGemmaRequestCache,
        selfConditioningLogits: MLXArray? = nil
    ) throws -> MLXArray {
        try model.decoder.denoise(
            canvasIds: canvasIds, cache: cache,
            selfConditioningLogits: selfConditioningLogits)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (original, value) in weights {
            var name = original
            if name.hasSuffix(".experts.down_proj") || name.hasSuffix(".experts.gate_up_proj") {
                name += ".weight"
            }
            if name.hasPrefix("model.encoder.vision_tower.") {
                name = name.replacingOccurrences(of: ".linear.", with: ".")
            }
            // Preserve an alias collision as an unexpected input key so strict
            // weight loading rejects it, rather than choosing one payload.
            guard result[name] == nil else { return weights }
            result[name] = value
        }
        return result
    }

    public func quantizationPathAliases(for path: String) -> [String] {
        path.hasPrefix("model.encoder.vision_tower.") ? [path + ".linear"] : []
    }
}
