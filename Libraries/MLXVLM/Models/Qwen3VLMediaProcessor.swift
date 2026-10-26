// Copyright © 2026 Eigen Labs.
import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// The existing Qwen image/video processor without a tokenizer or a model.
/// Serving and a native cluster rank use the same pixel operations. The caller
/// owns encoded-media admission, decoding, byte/pixel budgets and cancellation.
public struct Qwen3VLMediaProcessor {
    public static let policyID = "qwen35-media-preparation-v1"
    public let configuration: Qwen3VLProcessorConfiguration
    private var config: Qwen3VLProcessorConfiguration { configuration }
    private let checkpointImageBounds: (min: Int, max: Int)?

    init(
        _ configuration: Qwen3VLProcessorConfiguration,
        checkpointImageBounds: (min: Int, max: Int)? = nil
    ) {
        self.configuration = configuration
        self.checkpointImageBounds = checkpointImageBounds
    }

    /// The same checked Prism9 image-policy selection used by the VLM factory.
    /// Model/processor byte identity and the registry grant remain caller-owned.
    public init(modelConfiguration: Data, processorConfiguration: Data) throws {
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: modelConfiguration)
        guard ["qwen3_5", "qwen3_5_moe", "prism_hadamard_qwen35"].contains(base.modelType) else {
            throw VLMError.processing("The model has no Qwen3.5 media processor")
        }
        if base.modelType == "prism_hadamard_qwen35",
            try PrismHadamardImageProfile.resolve(configurationData: modelConfiguration)
                == .checkpoint9B
        {
            let image = try JSONDecoder.json5().decode(
                QwenCheckpointImageConfiguration.self, from: processorConfiguration)
            let factor = try Qwen4ExpMediaGeometry.product(
                image.base.patchSize, image.base.mergeSize)
            let maximum = Self.defaultVisionTokenBudgetPixels(
                factor: factor, ceiling: image.maxPixels)
            guard image.minPixels <= maximum else {
                throw VLMError.processing(
                    "Prism image minimum exceeds the default serving pixel budget")
            }
            self.init(image.base, checkpointImageBounds: (image.minPixels, maximum))
        } else {
            let configuration = try JSONDecoder.json5().decode(
                Qwen3VLProcessorConfiguration.self, from: processorConfiguration)
            guard configuration.imageMean.count == 3, configuration.imageStd.count == 3,
                configuration.imageMean.allSatisfy(\.isFinite),
                configuration.imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
                configuration.patchSize > 0, configuration.mergeSize > 0,
                configuration.temporalPatchSize > 0
            else {
                throw VLMError.processing("Invalid Qwen media processor geometry or normalization")
            }
            _ = try Qwen4ExpMediaGeometry.product(configuration.patchSize, configuration.mergeSize)
            self.init(configuration)
        }
    }

    /// Build the ordinary prompt processor with exactly the same checkpoint
    /// image policy used by media-only native preparation.
    public func makeUserInputProcessor(tokenizer: any Tokenizer) -> Qwen3VLProcessor {
        if let checkpointImageBounds {
            return Qwen3VLProcessor(
                configuration, tokenizer: tokenizer,
                checkpointImageBounds: checkpointImageBounds)
        }
        return Qwen3VLProcessor(configuration, tokenizer: tokenizer)
    }

    /// Exact standalone top-level merge: preprocessor values win, including
    /// nulls and nested objects; processor values fill missing keys only.
    public static func resolvedConfiguration(preprocessor: Data?, processor: Data?) throws -> Data {
        if let preprocessor, let processor {
            return try fillingMissingProcessorKeys(of: preprocessor, from: processor)
        }
        guard let data = preprocessor ?? processor else {
            throw VLMError.processing("Missing Qwen media processor configuration")
        }
        return data
    }

    private func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    /// Pixels for the default 1,280 vision-token budget (`1280 * factor²`),
    /// clamped to `ceiling`. Overflow-safe: a pathological config (an absurd
    /// `patch_size`/`merge_size`) yields `ceiling` instead of trapping on the
    /// unchecked multiply, so bad model metadata degrades to the config ceiling
    /// rather than crashing preprocess.
    static func defaultVisionTokenBudgetPixels(factor: Int, ceiling: Int) -> Int {
        let (squared, squaredOverflow) = factor.multipliedReportingOverflow(by: factor)
        guard !squaredOverflow else { return ceiling }
        let (budget, budgetOverflow) = squared.multipliedReportingOverflow(by: 1280)
        guard !budgetOverflow else { return ceiling }
        return min(ceiling, budget)
    }

    /// Exact pre-raster image patch geometry for the default processor policy.
    /// `height` and `width` are after any caller-requested resize. A still image
    /// pads to one temporal group; the language token count is h*w/merge².
    public func imageGrid(
        height: Int, width: Int,
        processing: UserInput.Processing? = nil
    ) throws -> THW {
        guard height > 0, width > 0 else {
            throw VLMError.imageProcessingFailure("Qwen image has invalid dimensions")
        }
        // The qwen3_5 ViT runs *global* O(patches²) attention with no windowing,
        // so an uncapped full-resolution image can allocate a tens-of-GB score
        // matrix, and published configs ship a ~12.8 MP max_pixels that never
        // clamps. Default each image to a 1,280 vision-token budget
        // (1280 * factor² pixels ⇒ 1,280 tokens after the spatial merge, since
        // tokens = pixels / factor²), matching the video path's philosophy and
        // the sibling Qwen2.5-VL. An explicit `processing.maxPixels` overrides it.
        let factor = config.patchSize * config.mergeSize
        let maxPixels =
            processing?.maxPixels
            ?? checkpointImageBounds?.max
            ?? Self.defaultVisionTokenBudgetPixels(factor: factor, ceiling: config.size.maxPixels)
        try requireSafeTargetSize(
            height: height, width: width, factor: factor,
            minimumPixels: processing?.minPixels ?? checkpointImageBounds?.min
                ?? min(config.size.minPixels, maxPixels),
            maximumPixels: maxPixels)
        let resizedHeight: Int
        let resizedWidth: Int
        if let bounds = checkpointImageBounds {
            let size = try Qwen4ExpMediaGeometry.image(
                height: height, width: width,
                factor: factor, minPixels: processing?.minPixels ?? bounds.min, maxPixels: maxPixels
            )
            (resizedHeight, resizedWidth) = (size.height, size.width)
        } else {
            (resizedHeight, resizedWidth) = try QwenVL.targetSize(
                height: height, width: width, factor: factor,
                minPixels: processing?.minPixels ?? min(config.size.minPixels, maxPixels),
                maxPixels: maxPixels)
        }

        return THW(1, resizedHeight / config.patchSize, resizedWidth / config.patchSize)
    }

    /// The exact per-frame video resize. At most eight sampled source frames
    /// form at most four temporal pairs under the registered temporal size2.
    public func videoFrameGrid(
        height: Int, width: Int,
        processing: UserInput.Processing = .init()
    ) throws -> THW {
        guard height > 0, width > 0 else {
            throw VLMError.imageProcessingFailure("Qwen video has invalid dimensions")
        }
        // An explicit `processing.maxPixels` narrows the frame budget but
        // never exceeds the video-tower bound; `processing.minPixels` replaces
        // the configured minimum.
        let maxPixels = min(
            processing.maxPixels ?? config.maxPixels,
            Qwen3VLProcessor.maxVideoFramePixels)
        let minPixels = min(processing.minPixels ?? config.minPixels, maxPixels)
        try requireSafeTargetSize(
            height: height, width: width,
            factor: config.patchSize * config.mergeSize,
            minimumPixels: minPixels, maximumPixels: maxPixels)
        let resized = try QwenVL.targetSize(
            height: height, width: width,
            factor: config.patchSize * config.mergeSize,
            minPixels: minPixels, maxPixels: maxPixels)
        return THW(1, resized.0 / config.patchSize, resized.1 / config.patchSize)
    }

    /// A pre-decode upper bound, including the legacy minimum-pixel upscale.
    /// The resize rejects aspect ratios >=201. In real arithmetic, rounding
    /// either resized side up by `factor` bounds its area by
    /// `2*maxPixels + 202*factor²`; eight times `(maxPixels + 201*factor²)`
    /// leaves ample slack for the original Float operations. This intentionally
    /// prices all eight sampled frames, padded to complete temporal groups.
    /// It does not use the eventual prompt length: patches exist before the
    /// tokenizer can reject an oversized expanded prompt.
    public func maximumVideoPatchCount(processing: UserInput.Processing = .init()) throws -> Int {
        let maximum = min(
            processing.maxPixels ?? config.maxPixels,
            Qwen3VLProcessor.maxVideoFramePixels)
        let factor = try Qwen4ExpMediaGeometry.product(config.patchSize, config.mergeSize)
        let aspectSlack = try Qwen4ExpMediaGeometry.product(201, factor, factor)
        let sum = maximum.addingReportingOverflow(aspectSlack)
        guard maximum > 0, !sum.overflow else {
            throw VLMError.processing("Qwen video patch bound exceeds integer limits")
        }
        let pixels = try Qwen4ExpMediaGeometry.product(8, sum.partialValue)
        let patchArea = try Qwen4ExpMediaGeometry.product(config.patchSize, config.patchSize)
        let patches = (pixels - 1) / patchArea + 1
        let groups = (Qwen3VLProcessor.maxVideoFrames - 1) / config.temporalPatchSize + 1
        return try Qwen4ExpMediaGeometry.product(patches, groups)
    }

    // The legacy resize uses Float rounding and Int products. Validate ample
    // integer headroom before calling it; ordinary admitted image dimensions
    // and budgets are far below this bound, so their arithmetic is unchanged.
    private func requireSafeTargetSize(
        height: Int, width: Int, factor: Int,
        minimumPixels: Int, maximumPixels: Int
    ) throws {
        let pixels = height.multipliedReportingOverflow(by: width)
        let square = factor.multipliedReportingOverflow(by: factor)
        guard height > 0, width > 0, factor > 0, !pixels.overflow, !square.overflow,
            pixels.partialValue <= Int.max / 1024, square.partialValue <= Int.max / 1024,
            minimumPixels > 0, minimumPixels <= maximumPixels,
            maximumPixels <= Int.max / 1024
        else {
            throw VLMError.imageProcessingFailure(
                "Qwen resize geometry exceeds safe integer bounds")
        }
    }

    private func checkedSize(_ extent: CGSize) throws -> (height: Int, width: Int) {
        guard extent.height.isFinite, extent.width.isFinite,
            extent.height > 0, extent.width > 0,
            extent.height < CGFloat(Int.max), extent.width < CGFloat(Int.max)
        else {
            throw VLMError.imageProcessingFailure("Qwen media has invalid dimensions")
        }
        return (Int(extent.height), Int(extent.width))
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processed = images.map { MediaProcessing.apply($0, processing: processing) }

        guard let first = processed.first else {
            throw VLMError.imageProcessingFailure("No image provided")
        }

        let extent = first.extent.size
        let size = try checkedSize(extent)
        let grid = try imageGrid(height: size.height, width: size.width, processing: processing)
        let resizedHeight = grid.h * config.patchSize
        let resizedWidth = grid.w * config.patchSize

        let targetSize = CGSize(width: resizedWidth, height: resizedHeight)

        // The sRGB tone-curve step must precede resample/normalize: CoreImage
        // filters run in a *linear-light* working space and `asMLXArray`
        // renders with a nil colorSpace (raw working-space values), so without
        // it the model receives linearized values instead of the gamma-encoded
        // bytes the HF reference preprocesses — crushing dark-region contrast
        // ~12× (near-black text becomes invisible to the model). The sibling
        // Qwen25VL/Qwen2VL image paths and this file's own video path apply
        // the same step.
        let resampled = processed.map {
            MediaProcessing.resampleBicubic(
                MediaProcessing.inSRGBToneCurveSpace($0), to: targetSize)
        }

        let normalized =
            resampled
            .map {
                MediaProcessing.normalize(
                    $0,
                    mean: config.imageMeanTuple,
                    std: config.imageStdTuple)
            }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: normalized,
            mergeSize: config.mergeSize,
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    /// Exact full-clip sampling/preprocessing used by `Qwen3VLProcessor`.
    /// One complete T×H×W grid is retained; temporal attention must not become
    /// unrelated still-image invocations.
    func videoFrames(_ video: UserInput.Video, processing: UserInput.Processing) async throws
        -> [MLXArray]
    {
        var resizedSize: CGSize = .zero
        let sequence = try await MediaProcessing.asProcessedSequence(
            video,
            processing: processing.video,
            targetFPS: { _ in Double(2) },
            maxFrames: Qwen3VLProcessor.maxVideoFrames,
            frameProcessing: { frame in
                let processed = MediaProcessing.apply(
                    try frame.image.asCIImage(), processing: processing)
                if resizedSize == .zero {
                    let size = processed.extent.size
                    let checked = try checkedSize(size)
                    let grid = try videoFrameGrid(
                        height: checked.height, width: checked.width, processing: processing)
                    resizedSize = CGSize(
                        width: grid.w * config.patchSize, height: grid.h * config.patchSize)
                }
                let finalImage = preprocess(image: processed, resizedSize: resizedSize)
                return VideoFrame(image: .ciImage(finalImage), timeStamp: frame.timeStamp)
            })
        return sequence.frames
    }

    public func preprocess(video: UserInput.Video, processing: UserInput.Processing = .init())
        async throws -> (MLXArray, THW)
    {
        let frames = try await videoFrames(video, processing: processing)
        return try QwenVL.patchify(
            images: frames, mergeSize: config.mergeSize,
            patchSize: config.patchSize, temporalPatchSize: config.temporalPatchSize)
    }
}
