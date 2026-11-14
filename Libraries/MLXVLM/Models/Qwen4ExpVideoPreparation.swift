// Copyright © 2026 Eigen Labs.
import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// The Transformers Qwen3-VL video path: reference frame sampling, one resize per
/// clip under a pixel budget, and a timestamped vision block per temporal group.
/// `Qwen4ExpProcessor` and `Qwen3VLProcessor` share it.
enum Qwen4ExpVideoPreparation {
    /// `minPixels` and `maxPixels` bound each clip's frames × height × width.
    /// `maxFramePixels`, when set, also bounds one frame's height × width.
    static func prepare(
        input: UserInput, imageInput: LMInput, tokenizer: any Tokenizer,
        config: Qwen4ExpVideoConfiguration, minPixels: Int, maxPixels: Int,
        maxFramePixels: Int? = nil
    ) async throws -> LMInput {
        var pixels: [MLXArray] = [], grids: [THW] = [], timestamps: [[Double]] = []
        let mean = (config.imageMean[0], config.imageMean[1], config.imageMean[2])
        let std = (config.imageStd[0], config.imageStd[1], config.imageStd[2])
        let factor = try Qwen4ExpMediaGeometry.product(config.patchSize, config.mergeSize)
        for video in input.videos {
            try Task.checkCancellation()
            var selectedSize: Qwen4ExpMediaGeometry.Size?
            let sampled = try await Qwen4ExpVideoSampler.sample(video, config: config) { frame, count in
                let source = MediaProcessing.apply(frame, processing: input.processing)
                let extent = source.extent.size
                guard extent.width.isFinite, extent.height.isFinite, extent.width > 0, extent.height > 0,
                      extent.width < CGFloat(Int.max), extent.height < CGFloat(Int.max) else {
                    throw Qwen4ExpMediaGeometry.Failure.invalidGeometry
                }
                var lower = minPixels, upper = maxPixels
                if let maxFramePixels {
                    let ceiling = try Qwen4ExpMediaGeometry.product(count, maxFramePixels)
                    if ceiling < upper { (lower, upper) = (min(lower, ceiling), ceiling) }
                }
                let size = try Qwen4ExpMediaGeometry.video(frames: count,
                    height: Int(extent.height), width: Int(extent.width),
                    temporalFactor: config.temporalPatchSize, factor: factor,
                    minPixels: lower, maxPixels: upper)
                if let previous = selectedSize, previous != size {
                    throw VLMError.processing("Qwen video changes frame geometry within a clip")
                }
                selectedSize = size
                return source.toSRGB()
                    .resampled(to: CGSize(width: size.width, height: size.height), method: .bicubic)
                    .normalized(mean: mean, std: std)
            }
            let (array, grid) = try QwenVL.patchify(images: sampled.frames,
                mergeSize: config.mergeSize, patchSize: config.patchSize, temporalPatchSize: config.temporalPatchSize)
            let times = try Qwen4ExpMediaGeometry.timestamps(indices: sampled.indices,
                sourceFPS: sampled.sourceFPS, temporalFactor: config.temporalPatchSize)
            guard times.count == grid.t else { throw VLMError.processing("Qwen video temporal groups disagree") }
            pixels.append(array); grids.append(grid); timestamps.append(times)
        }
        let tokens = try Qwen4ExpVideoPrompt.expandedTokens(imageInput.text.tokens.asArray(Int.self),
            tokenizer: tokenizer, grids: grids, timestamps: timestamps, merge: config.mergeSize)
        let array = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(text: .init(tokens: array, mask: ones(like: array).asType(.int8)),
            image: imageInput.image, video: .init(pixels: concatenated(pixels), frames: grids))
    }
}
