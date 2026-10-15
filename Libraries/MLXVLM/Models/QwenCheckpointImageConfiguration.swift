// Copyright © 2026 Eigen Labs.
// Qwen image bounds as declared by the checkpoint's image processor.

import Foundation

struct QwenCheckpointImageConfiguration: Codable, Sendable {
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

    private enum CodingKeys: String, CodingKey {
        case size, minPixels = "min_pixels", maxPixels = "max_pixels"
    }

    init(from decoder: Decoder) throws {
        base = try Qwen3VLProcessorConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let size = try container.decodeIfPresent(Size.self, forKey: .size)
        // Explicit legacy min/max override size in the reference processor.
        minPixels = try container.decodeIfPresent(Int.self, forKey: .minPixels)
            ?? size?.shortestEdge ?? (56 * 56)
        maxPixels = try container.decodeIfPresent(Int.self, forKey: .maxPixels)
            ?? size?.longestEdge ?? (28 * 28 * 1280)
        guard minPixels > 0, maxPixels >= minPixels,
              ["Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"].contains(base.imageProcessorType),
              base.imageMean.count == 3, base.imageStd.count == 3,
              base.imageMean.allSatisfy({ $0.isFinite }),
              base.imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
              base.patchSize > 0, base.mergeSize > 0, base.temporalPatchSize > 0 else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                debugDescription: "Invalid Qwen image preprocessing geometry or normalization"))
        }
        _ = try Qwen4ExpMediaGeometry.product(base.patchSize, base.mergeSize)
    }

    func encode(to encoder: Encoder) throws {
        try base.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(Size(shortestEdge: minPixels, longestEdge: maxPixels), forKey: .size)
    }
}
