// Copyright © 2026 Eigen Labs.
// Qwen4's image settings follow its checkpoint and Transformers v5.8.0,
// not the older Darkbloom Qwen3VL serving-resolution policy.

import Foundation
import MLXLMCommon

public struct Qwen4ExpProcessorConfiguration: Codable, Sendable {
    typealias Size = QwenCheckpointImageConfiguration.Size
    private let image: QwenCheckpointImageConfiguration
    var base: Qwen3VLProcessorConfiguration { image.base }
    var minPixels: Int { image.minPixels }
    var maxPixels: Int { image.maxPixels }
    let video: Qwen4ExpVideoConfiguration?

    enum CodingKeys: String, CodingKey {
        case video = "video_processor"
    }

    public init(from decoder: Decoder) throws {
        image = try QwenCheckpointImageConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        video = try container.decodeIfPresent(Qwen4ExpVideoConfiguration.self, forKey: .video)
        if let video {
            guard video.patchSize == base.patchSize, video.mergeSize == base.mergeSize,
                  video.temporalPatchSize == base.temporalPatchSize else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                    debugDescription: "Qwen4 image and video patch geometry disagree"))
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        try image.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
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
