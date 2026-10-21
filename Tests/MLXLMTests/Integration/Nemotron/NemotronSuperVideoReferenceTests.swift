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

        let pixels = try Self.npyFloat32(
            golden.appendingPathComponent("square_tubelets01_pixels.npy"))
        let expected = try Self.npyFloat32(golden.appendingPathComponent("square_features.npy"))
        #expect(pixels.shape == [4, 3, 512, 512] && expected.shape == [2, 256, 4096])
        let features = try composition.videoFeatures(normalizedFrames: pixels).asType(.float32)
        eval(features)
        #expect(features.shape == expected.shape)

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
    private static func npyFloat32(_ url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        let bytes = [UInt8](data)
        #expect(bytes.count > 10 && Array(bytes[1 ... 5]) == Array("NUMPY".utf8))
        let major = bytes[6]
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(bytes[8]) | Int(bytes[9]) << 8
            headerStart = 10
        } else {
            headerLength =
                Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16 | Int(bytes[11]) << 24
            headerStart = 12
        }
        let header = try #require(
            String(bytes: bytes[headerStart ..< headerStart + headerLength], encoding: .ascii))
        #expect(header.contains("'descr': '<f4'") && header.contains("'fortran_order': False"))
        let shapeText = try #require(
            header.range(of: "\\(([0-9, ]*)\\)", options: .regularExpression).map {
                String(header[$0]).dropFirst().dropLast()
            })
        let shape = shapeText.split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        let payload = data.subdata(in: (headerStart + headerLength) ..< data.count)
        let values = payload.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(values.count == shape.reduce(1, *))
        return MLXArray(values, shape)
    }
}
