// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Shared serving seam for Qwen VLM wrappers whose visual placeholder tokens
// stay on the ordinary causal path and whose positions are Qwen M-RoPE:
// `Qwen35` (Qwen3.5 / Qwen3.8 dense + MoE) and `Qwen4Exp` (Flash-Next).
//
// Fusion `mlx_vlm/models/qwen4_exp/qwen4_exp.py` subclasses `Qwen3_5Model` for
// exactly this reason: the Qwen4 wrapper reuses Qwen3.5's `get_input_embeddings`,
// `merge_input_ids_with_image_features` and `get_rope_index`, while the vision
// tower is the published Qwen3-VL ViT with `deepstack_visual_indexes = []`.
// The helpers below are that shared surface, expressed once so the CBv2 vision
// prefill path can drive either wrapper through one protocol.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One Qwen VLM wrapper the CBv2 vision prefill path can drive: request-owned
/// M-RoPE positions plus per-image / per-video tower features in text space.
public protocol QwenVisionSeamModel: AnyObject {
    var visionSeamConfiguration: Qwen35VisionSeamConfiguration { get }
    var imagePlaceholderTokenId: Int { get }
    var videoPlaceholderTokenId: Int { get }

    /// Vision tower attention geometry (`hidden_size`, `num_heads`) for the
    /// N² attention-buffer admission rule.
    var visionTowerAttentionGeometry: (hiddenSize: Int, numHeads: Int) { get }

    func positionResult(
        tokens: MLXArray,
        imageGrids: [THW]?,
        videoGrids: [THW]?,
        attentionMask: MLXArray?
    ) throws -> Qwen35PositionResult

    func visionFeatures(
        imagePixels: MLXArray?,
        imageGrids: [THW]?,
        videoPixels: MLXArray?,
        videoGrids: [THW]?
    ) throws -> Qwen35VisionFeatures
}

extension Qwen35: QwenVisionSeamModel {
    public var visionTowerAttentionGeometry: (hiddenSize: Int, numHeads: Int) {
        (config.visionConfiguration.hiddenSize, config.visionConfiguration.numHeads)
    }
}

/// Pure implementations behind `QwenVisionSeamModel`, parameterised on the
/// wrapper's seam configuration, tower and text embedding geometry.
public enum QwenVisionSeamSupport {

    /// Compute all prompt and decode positions without touching any
    /// model-global mutable state.
    public static func positionResult(
        tokens: MLXArray,
        imageGrids: [THW]?,
        videoGrids: [THW]?,
        attentionMask: MLXArray?,
        seam: Qwen35VisionSeamConfiguration
    ) throws -> Qwen35PositionResult {
        guard tokens.ndim == 1 || tokens.ndim == 2 else {
            throw Qwen35PositionSeamError.invalidInputRank(tokens.ndim)
        }
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens

        var mask = attentionMask
        if let attentionMask {
            guard attentionMask.ndim == 1 || attentionMask.ndim == 2 else {
                throw Qwen35PositionSeamError.invalidAttentionMaskRank(attentionMask.ndim)
            }
            mask = attentionMask.ndim == 1
                ? attentionMask.expandedDimensions(axis: 0) : attentionMask
            guard mask?.shape == tokens.shape else {
                throw Qwen35PositionSeamError.attentionMaskShapeMismatch
            }
        }

        let imageGrids = imageGrids.flatMap { $0.isEmpty ? nil : $0 }
        let videoGrids = videoGrids.flatMap { $0.isEmpty ? nil : $0 }
        if (imageGrids != nil || videoGrids != nil), tokens.dim(0) != 1 {
            throw Qwen35PositionSeamError.multimodalBatchUnsupported(tokens.dim(0))
        }

        let merge = seam.spatialMergeSize
        for (kind, grids) in [("image", imageGrids ?? []), ("video", videoGrids ?? [])] {
            for (index, grid) in grids.enumerated()
            where grid.t <= 0 || grid.h <= 0 || grid.w <= 0
                || grid.h % merge != 0 || grid.w % merge != 0
            {
                throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
            }
        }
        try validateVisualTokenRuns(
            tokens: tokens, attentionMask: mask,
            imageGrids: imageGrids ?? [], videoGrids: videoGrids ?? [],
            merge: merge, seam: seam)

        let (positionIds, delta) = Qwen3VLLanguage.getRopeIndex(
            inputIds: tokens,
            imageGridTHW: imageGrids,
            videoGridTHW: videoGrids,
            spatialMergeSize: merge,
            imageTokenId: seam.imagePositionTokenId,
            videoTokenId: seam.videoPositionTokenId,
            visionStartTokenId: seam.visionStartTokenId,
            attentionMask: mask)

        return Qwen35PositionResult(
            promptPositionIds: positionIds,
            decodeState: Qwen35PositionState(deltas: delta.asArray(Int32.self)),
            promptLength: tokens.dim(1))
    }

    static func validateVisualTokenRuns(
        tokens: MLXArray,
        attentionMask: MLXArray?,
        imageGrids: [THW],
        videoGrids: [THW],
        merge: Int,
        seam: Qwen35VisionSeamConfiguration
    ) throws {
        guard tokens.dim(0) == 1 else { return }
        let values = tokens[0, 0...].asArray(Int32.self).map(Int.init)
        let maskValues = attentionMask?.asType(.int32)[0, 0...].asArray(Int32.self)
        let imageToken = seam.imagePositionTokenId
        let videoToken = seam.videoPositionTokenId
        var imageIndex = 0
        var videoIndex = 0
        var cursor = 0

        while cursor < values.count {
            let visible = maskValues.map { $0[cursor] == 1 } ?? true
            let token = values[cursor]
            guard visible, token == imageToken || token == videoToken else {
                cursor += 1
                continue
            }

            let kind = token == imageToken ? "image" : "video"
            let gridIndex = kind == "image" ? imageIndex : videoIndex
            let grids = kind == "image" ? imageGrids : videoGrids
            var end = cursor + 1
            while end < values.count,
                (maskValues.map { $0[end] == 1 } ?? true), values[end] == token
            {
                end += 1
            }
            let actual = end - cursor
            guard gridIndex < grids.count else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: 0, actual: actual)
            }
            let expected = try mergedTokenCount(
                grids[gridIndex], merge: merge, kind: kind, index: gridIndex)
            guard actual == expected else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: expected, actual: actual)
            }
            if kind == "image" { imageIndex += 1 } else { videoIndex += 1 }
            cursor = end
        }

        if imageIndex < imageGrids.count {
            let expected = try mergedTokenCount(
                imageGrids[imageIndex], merge: merge, kind: "image", index: imageIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "image", gridIndex: imageIndex, expected: expected, actual: 0)
        }
        if videoIndex < videoGrids.count {
            let expected = try mergedTokenCount(
                videoGrids[videoIndex], merge: merge, kind: "video", index: videoIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "video", gridIndex: videoIndex, expected: expected, actual: 0)
        }
    }

    static func mergedTokenCount(
        _ grid: THW, merge: Int, kind: String, index: Int
    ) throws -> Int {
        let (spatial, spatialOverflow) = (grid.h / merge).multipliedReportingOverflow(
            by: grid.w / merge)
        let (count, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
        guard !spatialOverflow, !temporalOverflow else {
            throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
        }
        return count
    }

    /// Run the shared Qwen3-VL tower once over the packed image+video pixels,
    /// cast to the text embedding dtype, and return stable ordered slices:
    /// images first in grid order, then each video split into temporal frames.
    ///
    /// Qwen visual tokens remain ordinary causal language tokens; this API
    /// supplies no Gemma-style bidirectional span mask. DeepStack outputs (if
    /// the tower has any) are intentionally not part of this seam.
    static func visionFeatures(
        imagePixels: MLXArray?,
        imageGrids: [THW]?,
        videoPixels: MLXArray?,
        videoGrids: [THW]?,
        tower: Qwen3VLVision.VisionModel,
        spatialMergeSize merge: Int,
        textHiddenSize: Int,
        textDType: DType
    ) throws -> Qwen35VisionFeatures {
        if imagePixels != nil, imageGrids?.isEmpty != false {
            throw Qwen35VisionSeamError.missingImageGrids
        }
        if imagePixels == nil, imageGrids?.isEmpty == false {
            throw Qwen35VisionSeamError.missingImagePixels
        }
        if videoPixels != nil, videoGrids?.isEmpty != false {
            throw Qwen35VisionSeamError.missingVideoGrids
        }
        if videoPixels == nil, videoGrids?.isEmpty == false {
            throw Qwen35VisionSeamError.missingVideoPixels
        }

        let imageGrids = imageGrids ?? []
        let videoGrids = videoGrids ?? []

        func validate(grids: [THW], kind: String) throws {
            for (index, grid) in grids.enumerated()
            where grid.t <= 0 || grid.h <= 0 || grid.w <= 0
                || grid.h % merge != 0 || grid.w % merge != 0
            {
                throw Qwen35VisionSeamError.invalidGrid(kind: kind, index: index)
            }
        }
        try validate(grids: imageGrids, kind: "image")
        try validate(grids: videoGrids, kind: "video")

        var pixelParts: [MLXArray] = []
        if let imagePixels {
            guard imagePixels.ndim == 2 else {
                throw Qwen35VisionSeamError.invalidPixelRank(
                    kind: "image", actual: imagePixels.ndim)
            }
            pixelParts.append(imagePixels)
        }
        if let videoPixels {
            guard videoPixels.ndim == 2 else {
                throw Qwen35VisionSeamError.invalidPixelRank(
                    kind: "video", actual: videoPixels.ndim)
            }
            pixelParts.append(videoPixels)
        }

        func validatePixels(
            _ pixels: MLXArray?, grids: [THW], kind: String
        ) throws {
            var expected = 0
            for (index, grid) in grids.enumerated() {
                let (spatial, spatialOverflow) = grid.h.multipliedReportingOverflow(by: grid.w)
                let (count, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
                let (total, totalOverflow) = expected.addingReportingOverflow(count)
                guard !spatialOverflow, !temporalOverflow, !totalOverflow else {
                    throw Qwen35VisionSeamError.invalidGrid(kind: kind, index: index)
                }
                expected = total
            }
            let actual = pixels?.dim(0) ?? 0
            guard expected == actual else {
                throw Qwen35VisionSeamError.pixelCountMismatch(
                    kind: kind, expected: expected, actual: actual)
            }
        }
        try validatePixels(imagePixels, grids: imageGrids, kind: "image")
        try validatePixels(videoPixels, grids: videoGrids, kind: "video")

        let grids = imageGrids + videoGrids
        guard !pixelParts.isEmpty else {
            return Qwen35VisionFeatures(
                ordered: [],
                flattenedFeatures: MLXArray.zeros([0, textHiddenSize], dtype: textDType))
        }

        let pixels = (pixelParts.count == 1 ? pixelParts[0] : concatenated(pixelParts))
            .asType(tower.patchEmbed.proj.weight.dtype)
        let (visionHidden, _) = tower(pixels, gridTHW: grids)
        let flattened = visionHidden.asType(textDType)
        let expectedFeatures = grids.reduce(0) { $0 + $1.product / (merge * merge) }
        guard flattened.dim(0) == expectedFeatures else {
            throw Qwen35VisionSeamError.featureCountMismatch(
                expected: expectedFeatures, actual: flattened.dim(0))
        }

        var ordered: [Qwen35VisionFeature] = []
        ordered.reserveCapacity(imageGrids.count + videoGrids.reduce(0) { $0 + $1.t })
        var cursor = 0

        for (index, grid) in imageGrids.enumerated() {
            let count = grid.product / (merge * merge)
            let slice = flattened[cursor ..< cursor + count, 0...]
                .expandedDimensions(axis: 0)
            ordered.append(.init(kind: .image(index: index), features: slice))
            cursor += count
        }
        for (videoIndex, grid) in videoGrids.enumerated() {
            let count = (grid.h / merge) * (grid.w / merge)
            for frameIndex in 0 ..< grid.t {
                let slice = flattened[cursor ..< cursor + count, 0...]
                    .expandedDimensions(axis: 0)
                ordered.append(
                    .init(
                        kind: .videoFrame(
                            videoIndex: videoIndex, frameIndex: frameIndex),
                        features: slice))
                cursor += count
            }
        }

        return Qwen35VisionFeatures(ordered: ordered, flattenedFeatures: flattened)
    }
}
