// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension Qwen35Configuration {
    /// Pure metadata used by request preparation before any model allocation.
    public var visionSeamConfiguration: Qwen35VisionSeamConfiguration {
        .init(
            imagePlaceholderTokenId: imageTokenIndex, videoPlaceholderTokenId: videoTokenIndex,
            imagePositionTokenId: imageTokenId, videoPositionTokenId: videoTokenId,
            visionStartTokenId: visionStartTokenId, visionEndTokenId: visionEndTokenId,
            spatialMergeSize: visionConfiguration.spatialMergeSize,
            temporalPatchSize: visionConfiguration.temporalPatchSize, attention: .causal)
    }
}

/// The exact Qwen3.5 visual tower, without a second language model. Native
/// serving owns verified loading and memory admission before construction or
/// evaluation. Features retain the ordinary wrapper's text-space conversion.
/// Each invocation accepts one image or one complete video, bounding the
/// quadratic attention workspace without splitting a clip's temporal grid.
public final class Qwen35VisionTower: Module, QwenVisionSeamModel {
    public let configuration: Qwen35Configuration
    public let textDType: DType
    @ModuleInfo(key: "vision_tower") private var tower: Qwen3VLVision.VisionModel

    public init(configuration: Qwen35Configuration, textDType: DType) throws {
        try Self.validate(configuration: configuration, textDType: textDType)
        let vision = configuration.visionConfiguration
        self.configuration = configuration
        self.textDType = textDType
        _tower.wrappedValue = Qwen3VLVision.VisionModel(vision)
        super.init()
    }

    static func validate(configuration: Qwen35Configuration, textDType: DType) throws {
        let vision = configuration.visionConfiguration
        let gridSide = Int(Double(max(0, vision.numPositionEmbeddings)).squareRoot())
        let (gridSquare, gridOverflow) = gridSide.multipliedReportingOverflow(by: gridSide)
        guard
            ["qwen3_5", "qwen3_5_moe", "prism_hadamard_qwen35"].contains(configuration.modelType),
            (1 ... 1024).contains(vision.depth), vision.hiddenSize > 0, vision.intermediateSize > 0,
            vision.outHiddenSize == configuration.textConfiguration.hiddenSize,
            vision.numHeads > 0, vision.hiddenSize % vision.numHeads == 0,
            (vision.hiddenSize / vision.numHeads) % 4 == 0,
            vision.patchSize > 0, vision.temporalPatchSize > 0, vision.spatialMergeSize > 0,
            vision.inChannels == 3, vision.numPositionEmbeddings > 0,
            !gridOverflow, gridSquare == vision.numPositionEmbeddings,
            vision.deepstackVisualIndexes.isEmpty,
            [.float16, .bfloat16, .float32].contains(textDType)
        else {
            throw VLMError.processing("Unsupported independent Qwen vision tower configuration")
        }
        _ = try Qwen4ExpMediaGeometry.product(vision.patchSize, vision.spatialMergeSize)
        _ = try Qwen4ExpMediaGeometry.product(
            vision.patchSize, vision.patchSize,
            vision.temporalPatchSize, vision.inChannels)
    }

    public var visionSeamConfiguration: Qwen35VisionSeamConfiguration {
        configuration.visionSeamConfiguration
    }
    public var imagePlaceholderTokenId: Int { configuration.imageTokenIndex }
    public var videoPlaceholderTokenId: Int { configuration.videoTokenIndex }
    public var visionTowerAttentionGeometry: (hiddenSize: Int, numHeads: Int) {
        (configuration.visionConfiguration.hiddenSize, configuration.visionConfiguration.numHeads)
    }
    public func positionResult(
        tokens: MLXArray, imageGrids: [THW]?, videoGrids: [THW]?,
        attentionMask: MLXArray?
    ) throws -> Qwen35PositionResult {
        try QwenVisionSeamSupport.positionResult(
            tokens: tokens, imageGrids: imageGrids,
            videoGrids: videoGrids, attentionMask: attentionMask, seam: visionSeamConfiguration)
    }
    public func visionFeatures(
        imagePixels: MLXArray?, imageGrids: [THW]?,
        videoPixels: MLXArray?, videoGrids: [THW]?
    ) throws -> Qwen35VisionFeatures {
        guard (imageGrids?.count ?? 0) + (videoGrids?.count ?? 0) == 1 else {
            throw VLMError.processing(
                "Independent Qwen vision evaluation requires one image or video")
        }
        let vision = configuration.visionConfiguration
        let patchWidth = try Qwen4ExpMediaGeometry.product(
            vision.patchSize, vision.patchSize,
            vision.temporalPatchSize, vision.inChannels)
        for pixels in [imagePixels, videoPixels].compactMap({ $0 }) {
            guard pixels.ndim == 2, pixels.dim(1) == patchWidth else {
                throw VLMError.processing("Qwen vision patch width differs from the loaded tower")
            }
        }
        return try QwenVisionSeamSupport.visionFeatures(
            imagePixels: imagePixels,
            imageGrids: imageGrids, videoPixels: videoPixels, videoGrids: videoGrids,
            tower: tower, spatialMergeSize: vision.spatialMergeSize,
            textHiddenSize: configuration.textConfiguration.hiddenSize, textDType: textDType)
    }

    /// Source-name normalization shared by descriptor admission and serial
    /// reads. Text tensors cannot enter the vision-only module.
    public static func canonicalWeightName(_ name: String) -> String? {
        if name.hasPrefix("vision_tower.") { return name }
        if name.hasPrefix("model.visual.") {
            return "vision_tower." + name.dropFirst("model.visual.".count)
        }
        return nil
    }

    /// The unchanged Qwen patch-embedding layout conversion. Packed projection
    /// installation, strict parameter coverage and dtype checks stay with the
    /// verified loader; this method neither quantizes nor evaluates weights.
    public func sanitizeVisionWeights(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var canonical: [String: MLXArray] = [:]
        for (name, value) in weights {
            guard let key = Self.canonicalWeightName(name), canonical[key] == nil,
                !key.contains("position_ids")
            else {
                throw VLMError.processing("Unexpected or duplicate Qwen vision source tensor")
            }
            if key.hasSuffix("patch_embed.proj.weight") {
                guard value.ndim == 5 else {
                    throw VLMError.processing("Qwen patch embedding must have rank five")
                }
            }
            canonical[key] = value
        }
        return tower.sanitize(weights: canonical)
    }
}
