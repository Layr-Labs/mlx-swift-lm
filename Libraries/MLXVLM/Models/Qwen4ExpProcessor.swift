// Copyright © 2026 Eigen Labs.
// Qwen4's image settings follow its checkpoint and Transformers v5.8.0,
// not the older Darkbloom Qwen3VL serving-resolution policy.

import Foundation
import MLXLMCommon

public struct Qwen4ExpProcessorConfiguration: Codable, Sendable {
    struct Size: Codable, Sendable {
        let shortestEdge: Int
        let longestEdge: Int
        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge", longestEdge = "longest_edge"
        }
    }

    let base: Qwen3VLProcessorConfiguration
    let minPixels: Int
    let maxPixels: Int
    let video: Qwen4ExpVideoConfiguration?

    enum CodingKeys: String, CodingKey {
        case size, minPixels = "min_pixels", maxPixels = "max_pixels", video = "video_processor"
    }

    public init(from decoder: Decoder) throws {
        base = try Qwen3VLProcessorConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let size = try container.decodeIfPresent(Size.self, forKey: .size)
        // Explicit legacy min/max override size in the official processor.
        minPixels = try container.decodeIfPresent(Int.self, forKey: .minPixels)
            ?? size?.shortestEdge ?? (56 * 56)
        maxPixels = try container.decodeIfPresent(Int.self, forKey: .maxPixels)
            ?? size?.longestEdge ?? (28 * 28 * 1280)
        video = try container.decodeIfPresent(Qwen4ExpVideoConfiguration.self, forKey: .video)
        // Missing sizes use Qwen2VLImageProcessor's actual v5.8 defaults,
        // not the different defaults of Darkbloom's shared Swift processor.
        guard minPixels > 0, maxPixels >= minPixels,
              ["Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"].contains(base.imageProcessorType),
              base.imageMean.count == 3, base.imageStd.count == 3,
              base.imageMean.allSatisfy({ $0.isFinite }),
              base.imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
              base.patchSize > 0, base.mergeSize > 0, base.temporalPatchSize > 0 else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                debugDescription: "Invalid Qwen4 image preprocessing geometry or normalization"))
        }
        _ = try Qwen4ExpMediaGeometry.product(base.patchSize, base.mergeSize)
        if let video {
            guard video.patchSize == base.patchSize, video.mergeSize == base.mergeSize,
                  video.temporalPatchSize == base.temporalPatchSize else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                    debugDescription: "Qwen4 image and video patch geometry disagree"))
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        try base.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(Size(shortestEdge: minPixels, longestEdge: maxPixels), forKey: .size)
        try container.encodeIfPresent(video, forKey: .video)
    }
}

public struct Qwen4ExpProcessor: UserInputProcessor {
    private let imageProcessor: Qwen3VLProcessor
    private let videoConfig: Qwen4ExpVideoConfiguration?
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen4ExpProcessorConfiguration, tokenizer: any Tokenizer) {
        imageProcessor = Qwen3VLProcessor(config.base, tokenizer: tokenizer,
            checkpointImageBounds: (config.minPixels, config.maxPixels))
        videoConfig = config.video
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard !input.videos.isEmpty else { return try await imageProcessor.prepare(input: input) }
        guard let videoConfig else { throw VLMError.processing("Qwen4 video processor metadata is unavailable") }
        // Retain the original rendered video placeholders while processing
        // only images in the common path; videos use their own reference path.
        var imagesOnly = input
        imagesOnly.videos = []
        let imageInput = try await imageProcessor.prepare(input: imagesOnly)
        return try await Qwen4ExpVideoPreparation.prepare(input: input, imageInput: imageInput,
                                                         tokenizer: tokenizer, config: videoConfig)
    }
}
