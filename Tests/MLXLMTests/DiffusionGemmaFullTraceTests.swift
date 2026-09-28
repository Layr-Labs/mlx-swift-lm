import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

private final class DiffusionFullTraceAnchor: NSObject {}

/// Export the actual selected model's first canvas and retained encoder state.
/// This is a bounded diagnostic, not an accepted numerical tolerance or speed run.
@Suite("DiffusionGemma full-model first-step trace", .serialized)
struct DiffusionGemmaFullTraceTests {
    private struct Input: Decodable {
        let thinking: Bool
        let promptTokenIds: [Int32]
    }
    private struct Loader: TokenizerLoader {
        let tokenizer: any MLXLMCommon.Tokenizer
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { tokenizer }
    }
    private func fileHash(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let data = try file.read(upToCount: 1 << 20), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_FULL_TRACE"] == "1"))
    func exportFirstStepAtExactProviderInputs() async throws {
        let environment = ProcessInfo.processInfo.environment
        let modelPath = environment["DARKBLOOM_DIFFUSION_MODEL_DIR"]
        let inputPath = environment["DARKBLOOM_DIFFUSION_NATIVE_INPUT_LOG"]
        let outputPath = environment["DARKBLOOM_DIFFUSION_TRACE_DIRECTORY"]
        let directory = URL(fileURLWithPath: try #require(modelPath))
        let input = URL(fileURLWithPath: try #require(inputPath))
        let output = URL(fileURLWithPath: try #require(outputPath), isDirectory: true)
        try #require(!FileManager.default.fileExists(atPath: output.path), "Never overwrite a trace")
        try #require(try fileHash(directory.appendingPathComponent("config.json"))
            == "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5")
        let inputs = try String(contentsOf: input, encoding: .utf8).components(separatedBy: .newlines)
            .filter { $0.hasPrefix("DIFFUSION_RAW_TOOL ") }
            .map { try JSONDecoder().decode(Input.self, from: Data($0.dropFirst("DIFFUSION_RAW_TOOL ".count).utf8)) }
        try #require(inputs.count == 2 && inputs.map(\.thinking) == [false, true])
        _ = Bundle(for: DiffusionFullTraceAnchor.self).bundleURL
        let oldCacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 8 << 30
        defer { Memory.clearCache(); Memory.cacheLimit = oldCacheLimit }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(tokenizer: #adaptHuggingFaceTokenizer(tokenizer)))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        for item in inputs {
            let name = item.thinking ? "on" : "off"
            let ids = MLXArray(item.promptTokenIds).reshaped(1, item.promptTokenIds.count)
            let cache = try context.model.makeCache(expectedPromptLength: item.promptTokenIds.count)
            // Both actual inputs are below the native512-token prefill quantum.
            try #require(item.promptTokenIds.count < 512)
            _ = try context.model.encode(tokenIds: ids, cache: cache)
            eval(cache.stateArrays())
            let key = MLXRandom.split(key: MLXRandom.key(7419)).1
            let canvas = MLXRandom.randInt(Int32(0)..<Int32(context.model.configuration.textConfig.vocabularySize),
                [1, context.model.configuration.canvasLength], key: key)
            let logits = try context.model.denoise(canvasIds: canvas, cache: cache)
            eval(logits)
            var arrays = ["prompt": ids, "canvas": canvas, "logits0": logits]
            let fullIndex = try #require(context.model.configuration.textConfig.layerTypes.firstIndex(of: "full_attention"))
            let fullRoPE = try #require(context.model.model.decoder.layers[fullIndex].attention.rope as? ProportionalRoPE)
            arrays["full_rope_frequencies"] = try #require(fullRoPE._freqs)
            let snapshots = cache.snapshots()
            try #require(snapshots.count == context.model.configuration.textConfig.layerCount)
            for (index, state) in snapshots.enumerated() {
                try #require(state.offset == item.promptTokenIds.count)
                arrays["layer\(index).keys"] = state.keys
                arrays["layer\(index).values"] = state.values
            }
            eval(Array(arrays.values))
            let target = output.appendingPathComponent(name + ".safetensors")
            try save(arrays: arrays, url: target)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: target.path)
            let record: [String: Any] = ["case": name, "arrays": arrays.count,
                "promptTokens": item.promptTokenIds.count, "logitsDType": String(describing: logits.dtype),
                "sha256": try fileHash(target), "inputLogSHA256": try fileHash(input),
                "numericalQualification": false]
            print("DIFFUSION_FULL_TRACE " + String(decoding: try JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]), as: UTF8.self))
        }
    }
}
