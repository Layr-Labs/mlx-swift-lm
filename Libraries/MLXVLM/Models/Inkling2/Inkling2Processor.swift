// Copyright © 2026 Eigen Labs.
//
// Inkling 2 processor: chat template + image patches. The template renders
// each image part as `<|content_image|><|unused_200054|><|end_message|>`; the
// processor then repeats the placeholder once per 40x40 image token
// (transformers `Inkling2Processor.replace_image_token`, vLLM
// `_get_prompt_updates`). Inkling 2 accepts no video or audio input.

import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import MLX
import MLXLMCommon

/// `processor_config.json` of an Inkling 2 checkpoint.
public struct Inkling2ProcessorConfiguration: Codable, Sendable {
    public struct ImageProcessor: Codable, Sendable {
        public var imageMean: [Float]
        public var imageStd: [Float]
        public var rescaleFactor: Double
        public var tokenHeight: Int
        public var tokenWidth: Int
        public var doRescale: Bool
        public var doNormalize: Bool

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case rescaleFactor = "rescale_factor"
            case size
            case doRescale = "do_rescale"
            case doNormalize = "do_normalize"
        }

        struct Size: Codable {
            var height: Int
            var width: Int
        }

        public init(
            imageMean: [Float] = [0.5, 0.5, 0.5], imageStd: [Float] = [0.5, 0.5, 0.5],
            rescaleFactor: Double = 1 / 255, tokenHeight: Int = 40, tokenWidth: Int = 40,
            doRescale: Bool = true, doNormalize: Bool = true
        ) {
            self.imageMean = imageMean
            self.imageStd = imageStd
            self.rescaleFactor = rescaleFactor
            self.tokenHeight = tokenHeight
            self.tokenWidth = tokenWidth
            self.doRescale = doRescale
            self.doNormalize = doNormalize
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let defaults = ImageProcessor()
            imageMean =
                try c.decodeIfPresent([Float].self, forKey: .imageMean) ?? defaults.imageMean
            imageStd = try c.decodeIfPresent([Float].self, forKey: .imageStd) ?? defaults.imageStd
            rescaleFactor =
                try c.decodeIfPresent(Double.self, forKey: .rescaleFactor) ?? defaults.rescaleFactor
            let size = try c.decodeIfPresent(Size.self, forKey: .size)
            tokenHeight = size?.height ?? defaults.tokenHeight
            tokenWidth = size?.width ?? defaults.tokenWidth
            doRescale = try c.decodeIfPresent(Bool.self, forKey: .doRescale) ?? true
            doNormalize = try c.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? true
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(imageMean, forKey: .imageMean)
            try c.encode(imageStd, forKey: .imageStd)
            try c.encode(rescaleFactor, forKey: .rescaleFactor)
            try c.encode(Size(height: tokenHeight, width: tokenWidth), forKey: .size)
            try c.encode(doRescale, forKey: .doRescale)
            try c.encode(doNormalize, forKey: .doNormalize)
        }
    }

    public var imageProcessor: ImageProcessor
    public var imageToken: String

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageToken = "image_token"
    }

    public init(
        imageProcessor: ImageProcessor = ImageProcessor(), imageToken: String = "<|unused_200054|>"
    ) {
        self.imageProcessor = imageProcessor
        self.imageToken = imageToken
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageProcessor =
            try c.decodeIfPresent(ImageProcessor.self, forKey: .imageProcessor) ?? ImageProcessor()
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<|unused_200054|>"
    }

    /// The image processor carries no resampling rule, so only the reference
    /// geometry and normalization are servable.
    public func validate() throws {
        let image = imageProcessor
        guard image.tokenHeight == Inkling2ImageLayout.tokenSide,
            image.tokenWidth == Inkling2ImageLayout.tokenSide,
            image.imageMean.count == 3, image.imageStd.count == 3,
            image.imageStd.allSatisfy({ $0 > 0 }), image.rescaleFactor > 0,
            image.doRescale, image.doNormalize
        else {
            throw VLMError.processing("Unsupported Inkling 2 image processor configuration")
        }
    }
}

extension Inkling2ProcessorConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws { try validate() }
}

/// Renders image parts first, then the text; a message without images keeps
/// plain string content, as the text-only path renders it.
public struct Inkling2MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        var dictionary: Message = ["role": message.role.rawValue]
        if message.images.isEmpty {
            dictionary["content"] = message.content
        } else {
            var parts = contentParts(for: message, layout: .imagesThenText)
            // An image-only turn renders no empty text part, as the reference.
            if message.content.isEmpty, parts.last?["type"] == "text" {
                parts.removeLast()
            }
            dictionary["content"] = parts
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

/// Test hook of the red control: the error type, never thrown.
public struct Inkling2ImageLimitError: LocalizedError, Equatable, Sendable {
    public let images: Int
    public let limit: Int

    public init(images: Int, limit: Int) {
        self.images = images
        self.limit = limit
    }

    public var errorDescription: String? {
        "Inkling 2 takes at most \(limit) images per request; this request has \(images)"
    }
}

public struct Inkling2Processor: UserInputProcessor {
    /// `<|unused_200054|>`, the published placeholder id.
    public static let defaultImageTokenId = 200_054
    /// Test hook of the red control: the cap, not enforced.
    public static let maximumImagesPerRequest = 64

    private let configuration: Inkling2ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let table: [[Float]]

    public init(_ configuration: Inkling2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.configuration = configuration
        self.tokenizer = tokenizer
        let image = configuration.imageProcessor
        table = Inkling2ImagePatches.normalizationTable(
            rescaleFactor: image.rescaleFactor, mean: image.imageMean, std: image.imageStd)
    }

    public var imageTokenId: Int {
        tokenizer.convertTokenToId(configuration.imageToken) ?? Self.defaultImageTokenId
    }

    /// Patches and grid of one image. `processing.resize` bounds the source
    /// (aspect kept) for a caller trading resolution for latency; sides above
    /// 4096 px are always scaled down to 4096.
    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        try configuration.validate()
        var image = image
        let extent = image.extent.integral
        let width = Int(extent.width)
        let height = Int(extent.height)
        guard width > 0, height > 0 else {
            throw VLMError.imageProcessingFailure("Inkling 2 cannot read an empty image")
        }
        let bound = processing?.resize.flatMap { size -> (width: Int, height: Int)? in
            guard size.width >= 1, size.height >= 1 else { return nil }
            return (Int(size.width.rounded()), Int(size.height.rounded()))
        }
        let target = Inkling2ImageLayout.processedSize(width: width, height: height, bound: bound)
        if target.width != width || target.height != height {
            // Edge-clamped, so the border keeps full alpha after the filter.
            let filter = CIFilter.lanczosScaleTransform()
            filter.inputImage = image.transformed(
                by: .init(translationX: -extent.minX, y: -extent.minY)
            ).clampedToExtent()
            filter.scale = Float(target.height) / Float(height)
            filter.aspectRatio =
                (Float(target.width) / Float(width)) / (Float(target.height) / Float(height))
            guard let scaled = filter.outputImage else {
                throw VLMError.imageProcessingFailure("Inkling 2 could not scale the image")
            }
            image = scaled.cropped(
                to: CGRect(x: 0, y: 0, width: target.width, height: target.height))
        }
        return Inkling2ImagePatches.patches(try Inkling2RGBImage(image), table: table)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.processing("Inkling 2 accepts image input only; video is not supported")
        }
        guard input.audios.isEmpty else {
            throw VLMError.processing("Inkling 2 does not accept audio input")
        }
        let messages = Inkling2MessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        guard !input.images.isEmpty else {
            return LMInput(text: .init(tokens: MLXArray(promptTokens).expandedDimensions(axis: 0)))
        }
        let processed = try input.images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let grids = processed.map(\.1)
        let expanded = try Self.expandPlaceholders(
            promptTokens, imageTokenId: imageTokenId, tokenCounts: grids.map(\.product))
        return LMInput(
            text: .init(tokens: MLXArray(expanded).expandedDimensions(axis: 0)),
            image: .init(pixels: concatenated(processed.map(\.0), axis: 0), frames: grids))
    }

    /// Repeats the `i`-th placeholder `tokenCounts[i]` times. Throws unless the
    /// prompt holds exactly one placeholder per image.
    public static func expandPlaceholders(
        _ tokens: [Int], imageTokenId: Int, tokenCounts: [Int]
    ) throws -> [Int] {
        var expanded: [Int] = []
        expanded.reserveCapacity(tokens.count + tokenCounts.reduce(0, +))
        var image = 0
        for token in tokens {
            guard token == imageTokenId else {
                expanded.append(token)
                continue
            }
            guard image < tokenCounts.count else {
                throw VLMError.processing("More image placeholders in the prompt than images")
            }
            expanded.append(contentsOf: repeatElement(imageTokenId, count: tokenCounts[image]))
            image += 1
        }
        guard image == tokenCounts.count else {
            throw VLMError.processing(
                "Prompt has \(image) image placeholders but \(tokenCounts.count) images")
        }
        return expanded
    }
}
