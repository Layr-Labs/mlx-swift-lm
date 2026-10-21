import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Gated full-checkpoint golden for the video path: the pinned checkpoint's
/// tower, video embedder and projector on the first two tubelets of a 256 x 256
/// clip, against NVIDIA's own modeling code run in float32 on CPU (the BF16
/// weights upcast exactly). Needs `NEMOTRON_SUPER_REFERENCE_DIR` (the checkpoint)
/// and `NEMOTRON_SUPER_VIDEO_GOLDEN_DIR` with `square_tubelets01_pixels.npy`
/// ([4, 3, 512, 512] float32 normalized frames) and `square_features.npy`
/// ([2, 256, 4096] float32 projected features).
@Suite(
    .tags(.integration),
    .enabled(
        if: ProcessInfo.processInfo.environment["NEMOTRON_SUPER_REFERENCE_DIR"] != nil
            && ProcessInfo.processInfo.environment["NEMOTRON_SUPER_VIDEO_GOLDEN_DIR"] != nil))
struct NemotronSuperVideoReferenceTests {
    private static let environment = ProcessInfo.processInfo.environment

    @Test func videoFeaturesMatchTheUpstreamReference() throws {
        let checkpoint = URL(
            fileURLWithPath: try #require(Self.environment["NEMOTRON_SUPER_REFERENCE_DIR"]))
        let golden = URL(
            fileURLWithPath: try #require(Self.environment["NEMOTRON_SUPER_VIDEO_GOLDEN_DIR"]))
        // The goldens are checked before the checkpoint's tower is read.
        let pixels = try Self.npyFloat32(
            golden.appendingPathComponent("square_tubelets01_pixels.npy"))
        let expected = try Self.npyFloat32(golden.appendingPathComponent("square_features.npy"))
        try #require(pixels.shape == [4, 3, 512, 512] && expected.shape == [2, 256, 4096])

        let index = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: checkpoint.appendingPathComponent("model.safetensors.index.json")))
                as? [String: Any])
        let map = try #require(index["weight_map"] as? [String: String])
        let wanted = map.filter {
            $0.key.hasPrefix("vision_model.") || $0.key.hasPrefix("mlp1.")
                || $0.key.hasPrefix("vision_projector.")
        }
        var tensors: [String: MLXArray] = [:]
        for shard in Set(wanted.values) {
            for (name, tensor) in try loadArrays(url: checkpoint.appendingPathComponent(shard))
            where wanted[name] != nil {
                tensors[name] = tensor
            }
        }
        let configuration = try NemotronSuperImageConfiguration(
            data: Data(contentsOf: checkpoint.appendingPathComponent("config.json")))
        let composition = NemotronSuperImageComposition(configuration: configuration)
        try composition.loadImageWeights(from: tensors)

        let features = try composition.videoFeatures(normalizedFrames: pixels).asType(.float32)
        eval(features)
        try #require(features.shape == expected.shape)

        // Per-token cosine against the float32 reference. BF16 serving of the same
        // code measured a minimum of 0.993 (mean 0.99988) against this golden.
        let dot = (features * expected).sum(axis: -1)
        let norms =
            sqrt((features * features).sum(axis: -1)) * sqrt((expected * expected).sum(axis: -1))
        let cosine = dot / maximum(norms, MLXArray(Float(1e-12)))
        let minimumCosine = cosine.min().item(Float.self)
        let meanCosine = cosine.mean().item(Float.self)
        print("SUPER_VIDEO_GOLDEN min_cos=\(minimumCosine) mean_cos=\(meanCosine)")
        #expect(minimumCosine > 0.98, "minimum per-token cosine \(minimumCosine)")
        #expect(meanCosine > 0.999, "mean per-token cosine \(meanCosine)")
    }

    /// Minimal reader for C-order little-endian float32 `.npy` files (versions 1-3).
    /// A short or malformed file fails the test instead of reading out of bounds.
    private static func npyFloat32(_ url: URL) throws -> MLXArray {
        let name = url.lastPathComponent
        let bytes = [UInt8](try Data(contentsOf: url))
        try #require(
            bytes.count >= 10 && bytes[0] == 0x93 && Array(bytes[1 ... 5]) == Array("NUMPY".utf8),
            "\(name): not an .npy file")
        let major = bytes[6]
        try #require((1 ... 3).contains(major), "\(name): version \(major)")
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(bytes[8]) | Int(bytes[9]) << 8
            headerStart = 10
        } else {
            try #require(bytes.count >= 12, "\(name): truncated preamble")
            headerLength =
                Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16 | Int(bytes[11]) << 24
            headerStart = 12
        }
        let payloadStart = headerStart + headerLength
        try #require(payloadStart <= bytes.count, "\(name): truncated header")
        let header = try #require(
            String(bytes: bytes[headerStart ..< payloadStart], encoding: .ascii))
        try #require(
            header.contains("'descr': '<f4'") && header.contains("'fortran_order': False"),
            "\(name): \(header)")
        let shapeText = try #require(
            header.range(of: "\\(([0-9, ]*)\\)", options: .regularExpression).map {
                String(header[$0]).dropFirst().dropLast()
            })
        let shape = shapeText.split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        var count = 1
        for dimension in shape {
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            try #require(dimension > 0 && !overflow, "\(name): shape \(shape)")
            count = next
        }
        let payloadBytes = bytes.count - payloadStart
        try #require(
            !shape.isEmpty && payloadBytes % 4 == 0 && payloadBytes / 4 == count,
            "\(name): shape \(shape) with \(payloadBytes) payload bytes")
        var values = [Float](repeating: 0, count: count)
        values.withUnsafeMutableBytes { $0.copyBytes(from: bytes[payloadStart...]) }
        return MLXArray(values, shape)
    }
}
