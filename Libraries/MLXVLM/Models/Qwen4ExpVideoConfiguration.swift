// Copyright © 2026 Eigen Labs.
// Configuration follows Transformers v5.8.0; see docs/qwen4/preprocessing.md.
import Foundation

struct Qwen4ExpVideoConfiguration: Codable, Sendable {
    let size: Qwen4ExpProcessorConfiguration.Size
    let patchSize: Int
    let mergeSize: Int
    let temporalPatchSize: Int
    let imageMean: [CGFloat]
    let imageStd: [CGFloat]
    let fps: Double
    let minFrames: Int
    let maxFrames: Int

    enum CodingKeys: String, CodingKey {
        case size, fps
        case patchSize = "patch_size", mergeSize = "merge_size", temporalPatchSize = "temporal_patch_size"
        case imageMean = "image_mean", imageStd = "image_std"
        case minFrames = "min_frames", maxFrames = "max_frames"
        case processorType = "video_processor_type"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        size = try c.decodeIfPresent(Qwen4ExpProcessorConfiguration.Size.self, forKey: .size)
            ?? .init(shortestEdge: 128 * 32 * 32, longestEdge: 32 * 32 * 768)
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        fps = try c.decodeIfPresent(Double.self, forKey: .fps) ?? 2
        minFrames = try c.decodeIfPresent(Int.self, forKey: .minFrames) ?? 4
        maxFrames = try c.decodeIfPresent(Int.self, forKey: .maxFrames) ?? 768
        let type = try c.decodeIfPresent(String.self, forKey: .processorType)
        guard type == nil || type == "Qwen3VLVideoProcessor",
              size.shortestEdge > 0, size.longestEdge >= size.shortestEdge,
              patchSize > 0, mergeSize > 0, temporalPatchSize > 0,
              imageMean.count == 3, imageStd.count == 3,
              imageMean.allSatisfy({ $0.isFinite }),
              imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
              fps.isFinite, fps > 0, minFrames > 0, maxFrames >= minFrames else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                debugDescription: "Invalid Qwen4 video preprocessing configuration"))
        }
        _ = try Qwen4ExpMediaGeometry.product(patchSize, mergeSize)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(size, forKey: .size)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(mergeSize, forKey: .mergeSize)
        try c.encode(temporalPatchSize, forKey: .temporalPatchSize)
        try c.encode(imageMean, forKey: .imageMean)
        try c.encode(imageStd, forKey: .imageStd)
        try c.encode(fps, forKey: .fps)
        try c.encode(minFrames, forKey: .minFrames)
        try c.encode(maxFrames, forKey: .maxFrames)
        try c.encode("Qwen3VLVideoProcessor", forKey: .processorType)
    }
}

/// The reference prefers nested processor metadata, then the separate video
/// config, then image metadata. Never fetch or rewrite model files at runtime.
enum Qwen4ExpProcessorFiles {
    static func combined(directory: URL, fallbackImage: Data) throws -> Data {
        let fm = FileManager.default
        func object(_ data: Data) throws -> [String: Any] {
            guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw Qwen4ExpMediaGeometry.Failure.invalidGeometry
            }
            return value
        }
        let processorURL = directory.appendingPathComponent("processor_config.json")
        let root = fm.fileExists(atPath: processorURL.path)
            ? try object(Data(contentsOf: processorURL)) : [:]
        var image = try object(fallbackImage)
        if let nested = root["image_processor"] {
            guard let value = nested as? [String: Any] else { throw Qwen4ExpMediaGeometry.Failure.invalidGeometry }
            image = value
        }
        let videoURL = directory.appendingPathComponent("video_preprocessor_config.json")
        let video: [String: Any]?
        if let nested = root["video_processor"] {
            guard let value = nested as? [String: Any] else { throw Qwen4ExpMediaGeometry.Failure.invalidGeometry }
            video = value
        } else if fm.fileExists(atPath: videoURL.path) {
            video = try object(Data(contentsOf: videoURL))
        } else {
            let imageURL = directory.appendingPathComponent("preprocessor_config.json")
            video = fm.fileExists(atPath: imageURL.path) ? try object(Data(contentsOf: imageURL)) : nil
        }
        image["video_processor"] = video ?? NSNull()
        return try JSONSerialization.data(withJSONObject: image, options: [.sortedKeys])
    }
}
