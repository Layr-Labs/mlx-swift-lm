import CoreImage
import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import MLXVLM

@Suite("DiffusionGemma full-model resident media cache", .serialized)
struct DiffusionGemmaMediaPrefixLiveTests {
    private struct Loader: TokenizerLoader {
        let tokenizer: any MLXLMCommon.Tokenizer
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { tokenizer }
    }
    private func run(_ engine: CBv2NativeBlockEngine, _ request: CBv2Request) async throws -> ([Int], String, CBv2Usage) {
        var ids = [Int](), text = "", result: CBv2Usage?
        for await event in try engine.submit(request) {
            switch event {
            case .delta(let value, let tokens, _): ids += tokens; text += value
            case .finished(let reason, let usage):
                #expect(reason == .stop && result == nil)
                result = usage
            }
        }
        return (ids, text, try #require(result))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_MEDIA_PREFIX_LIVE"] == "1"))
    func actualImagesRepeatAppendAndChangedPixelsRemainNative() async throws {
        let selected = ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_MODEL_DIR"]
        let directory = URL(fileURLWithPath: try #require(selected))
        let config = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        try #require(SHA256.hash(data: config).map { String(format: "%02x", $0) }.joined()
            == "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5")
        _ = try #require(Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        let oldCache = Memory.cacheLimit
        Memory.cacheLimit = 8 << 30
        defer { Memory.clearCache(); Memory.cacheLimit = oldCache }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(tokenizer: #adaptHuggingFaceTokenizer(tokenizer)))
        let processor = try #require(context.processor)
        let prefix = try DiffusionGemmaResidentPrefixConfiguration(maximumBytes: 1 << 30,
            artifactIdentity: "2ad9d4a10fe791e9e74a6475298c048e9da2d3df9e049eeb37a9d98055fcc2ce",
            templateIdentity: "selected-native-template", numericalProfile: "source-matched-native")
        let owner = NativePageTestProcessOwner(maximum: 8 << 30)
        let cached = try context.makeNativeEngine(kvBytesCapacity: 4 << 30, prefillChunkSize: 512,
            prefixCache: prefix, pagedConfiguration: .init(capacityBytes: 4 << 30, dtype: .bfloat16,
                maxPrefillChunk: 1120, nominalMaxSequenceLength: 4096, segmentSizeBytes: 16 << 20),
            processMemoryOwner: owner)
        let control = try context.makeNativeEngine(kvBytesCapacity: 4 << 30, prefillChunkSize: 512)
        let notes = (1...50).map { "Record \($0): Keep each request independent and ground the answer in the actual image." }.joined(separator: "\n")
        func prepared(_ color: CIColor, extra: String = "") async throws -> CBv2Request {
            let image = CIImage(color: color).cropped(to: .init(x: 0, y: 0, width: 64, height: 64))
            let input = UserInput(messages: [["role": "user", "content": [
                ["type": "image"], ["type": "text", "text": notes + extra
                    + "\nName the single dominant color of this image. Reply using only the color name."],
            ]]], images: [.ciImage(image)], additionalContext: ["enable_thinking": false])
            let media = try await processor.prepare(input: input)
            let binding = try #require(try context.model.prepareVision(media))
            let features = try binding.embeddings()
            var hash = SHA256()
            for (span, feature) in zip(binding.spans, features) {
                hash.update(data: Data("image:\(span.tokenOffset):\(span.length):\(feature.shape):\(feature.dtype)".utf8))
                hash.update(data: feature.asData(access: .copy).data)
            }
            return .init(id: .init(1), promptTokens: media.tokens.map(Int.init),
                sampling: .init(seed: 8132), maxTokens: 64, cacheSalt: "fixture-tenant",
                multimodal: .init(spans: binding.spans, embeddings: { features }),
                hybridPrefixIdentity: try .init(digest: Data(hash.finalize())))
        }
        do {
            let blue = try await prepared(.blue)
            try #require(blue.promptTokens.count > 1024)
            let reference = try await run(control, blue)
            #expect(reference.1.lowercased().contains("blue"))
            let cold = try await run(cached, blue)
            let warm = try await run(cached, blue)
            #expect(cold.0 == reference.0 && warm.0 == reference.0)
            #expect(cold.2.prefixCachePrefillTokensSaved == 0)
            #expect(warm.2.prefixCachePrefillTokensSaved == blue.promptTokens.count)
            let appended = try await prepared(.blue, extra: "\nAdditional note: keep reporting actual image content.")
            let appendedReference = try await run(control, appended)
            let appendedHit = try await run(cached, appended)
            #expect(appendedHit.0 == appendedReference.0 && appendedHit.2.prefixCachePrefillTokensSaved > 0)
            let red = try await prepared(.red)
            #expect(red.promptTokens == blue.promptTokens)
            let redReference = try await run(control, red)
            let redResult = try await run(cached, red)
            #expect(redResult.0 == redReference.0 && redResult.1.lowercased().contains("red"))
            #expect(redResult.2.prefixCachePrefillTokensSaved == 0)
            print("DIFFUSION_MEDIA_PREFIX prompt=\(blue.promptTokens.count) exactHit=\(warm.2.prefixCachePrefillTokensSaved) appendedHit=\(appendedHit.2.prefixCachePrefillTokensSaved) changedPixelsHit=\(redResult.2.prefixCachePrefillTokensSaved) scope=native-sdk-resident-only")
        } catch { await cached.shutdown(); await control.shutdown(); throw error }
        await cached.shutdown()
        await control.shutdown()
        #expect(owner.bytes == 0 && owner.coverage == 0)
    }
}
