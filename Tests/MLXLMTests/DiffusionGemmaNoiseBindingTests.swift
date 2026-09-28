import CryptoKit
import Foundation
import MLX
import Testing

private final class DiffusionNoiseBundleAnchor: NSObject {}

/// Prove actual random tensors at the selected artifact's production canvas and
/// vocabulary, not equivalence inferred from the same language-level seed.
@Suite("DiffusionGemma cross-runtime noise witness", .serialized)
struct DiffusionGemmaNoiseBindingTests {
    private func digest<T>(_ values: [T]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_NOISE_LIVE"] == "1"))
    func recordActualNativeDrawsForIndependentReferenceBinding() throws {
        let count = try #require(Int(ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_NOISE_DRAWS"] ?? "32"))
        try #require((1...128).contains(count))
        let seed: UInt64 = 7419
        let vocabulary = 262144, canvas = 256
        _ = Bundle(for: DiffusionNoiseBundleAnchor.self).bundleURL
        let oldCacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 1 << 30
        defer { Memory.clearCache(); Memory.cacheLimit = oldCacheLimit }
        var key = MLXRandom.key(seed)
        for index in 0..<count {
            try Task.checkCancellation()
            // Exactly the request-local key progression in GenerationSession.
            let split = MLXRandom.split(key: key)
            key = split.0
            let draw = split.1
            let integers = MLXRandom.randInt(Int32(0)..<Int32(vocabulary), [1, canvas], key: draw)
            let gumbel = MLXRandom.gumbel([1, canvas, vocabulary], dtype: .float32, key: draw)
            let categorical = MLXRandom.categorical(MLXArray.zeros([1, canvas, vocabulary]), key: draw).asType(.int32)
            let direct = gumbel.argMax(axis: -1).asType(.int32)
            eval(key, draw, integers, gumbel, categorical, direct)
            #expect(categorical.asArray(Int32.self) == direct.asArray(Int32.self))
            let record: [String: Any] = [
                "index": index, "seed": seed, "canvas": canvas, "vocabulary": vocabulary,
                "key": draw.asArray(UInt32.self), "dtype": "float32",
                "integerSHA256": digest(integers.asArray(Int32.self)),
                "gumbelSHA256": digest(gumbel.asArray(Float.self)),
                "categoricalSHA256": digest(categorical.asArray(Int32.self)),
                "scope": "noise witness only; not generation or quality qualification",
            ]
            print("DIFFUSION_NATIVE_NOISE " + String(decoding: try JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]), as: UTF8.self))
        }
    }
}
