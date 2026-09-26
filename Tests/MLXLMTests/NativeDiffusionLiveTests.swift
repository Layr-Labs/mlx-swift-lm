import CryptoKit
import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
@testable import MLXVLM
import Testing
import Tokenizers

/// Explicit full-artifact opt-in, isolated from ordinary component filters.
/// This is native SDK qualification, not authenticated provider/hosted coverage.
@Suite("Native diffusion full-artifact qualification", .serialized)
struct NativeDiffusionLiveTests {
    private struct Loader: TokenizerLoader {
        let tokenizer: any MLXLMCommon.Tokenizer
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { tokenizer }
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_MEDIA_LIVE"] == "1"))
    func selectedArtifactPixelsSteerKnownAnswers() async throws {
        let directory = URL(fileURLWithPath: try #require(
            ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_MODEL_DIR"]))
        let configHash = SHA256.hash(data: try Data(contentsOf: directory.appendingPathComponent("config.json")))
            .map { String(format: "%02x", $0) }.joined()
        try #require(configHash == "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5")
        _ = try #require(Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(tokenizer: #adaptHuggingFaceTokenizer(tokenizer)))
        let config = context.model.configuration
        let vision = try #require(config.visionConfig)
        let geometry = try DiffusionGemmaMediaGeometry.resized(width: 64, height: 64,
            patchSize: vision.patchSize, poolingSize: vision.poolingKernelSize,
            maxSoftTokens: config.visionSoftTokensPerImage)
        let rendered = try context.renderTokens(messages: [["role": "user", "content": [
            ["type": "image"],
            ["type": "text", "text": "Name the single dominant color of this image. Reply using only the color name."],
        ]]], additionalContext: ["enable_thinking": false])
        try #require(rendered.filter { $0 == Int32(config.imageTokenId) }.count == 1)
        let tokens = rendered.flatMap { token -> [Int32] in
            token == Int32(config.imageTokenId)
                ? [Int32(config.beginImageTokenId)] + Array(repeating: token, count: geometry.softTokens) + [Int32(config.endImageTokenId)]
                : [token]
        }
        let base = context.generationConfiguration
        let generation = try DiffusionGemmaGenerationConfiguration(maxNewTokens: 64,
            maxDenoisingSteps: base.maxDenoisingSteps, sampler: base.sampler,
            minimumTemperature: base.minimumTemperature, maximumTemperature: base.maximumTemperature,
            stabilityThreshold: base.stabilityThreshold, confidenceThreshold: base.confidenceThreshold,
            bosTokenId: base.bosTokenId, padTokenId: base.padTokenId, eosTokenIds: base.eosTokenIds)
        let processor = try #require(context.processor)
        let engine = try context.makeNativeEngine(kvBytesCapacity: 16 * 1024 * 1024 * 1024)
        for (name, channels) in [("red", [Float(1), 0, 0]), ("blue", [Float(0), 0, 1])] {
            let pixels = broadcast(MLXArray(channels).reshaped(1, 3, 1, 1),
                to: [1, 3, geometry.height, geometry.width])
            let result = try context.model.generateNative(
                promptTokenIds: MLXArray(tokens).reshaped(1, tokens.count), generation: generation,
                seed: 8132, pixelValues: pixels, visualOutputLengths: [geometry.softTokens])
            let text = tokenizer.decode(tokens: result.tokenIds.map(Int.init), skipSpecialTokens: false)
            #expect(result.finishReason == "stop")
            #expect(text.lowercased().contains(name), "Actual pixels must steer the color answer: \(text)")
            #expect(!text.lowercased().contains(name == "red" ? "blue" : "red"))
            let image = CIImage(color: name == "red" ? .red : .blue).cropped(to: .init(x: 0, y: 0, width: 64, height: 64))
            let prepared = try await processor.prepare(input: UserInput(messages: [["role": "user", "content": [
                ["type": "image"],
                ["type": "text", "text": "Name the single dominant color of this image. Reply using only the color name."],
            ]]], images: [.ciImage(image)], additionalContext: ["enable_thinking": false]))
            #expect(prepared.tokens == tokens)
            let media = try #require(try context.model.prepareVision(prepared))
            let stream = try engine.submit(.init(id: .init(name == "red" ? 301 : 302),
                promptTokens: prepared.tokens.map(Int.init), sampling: .init(seed: 8132), maxTokens: 64,
                multimodal: media))
            var streamed = "", terminalCount = 0
            var rawTokens = [Int]()
            for await event in stream {
                switch event {
                case .delta(let value, let committed, _): streamed += value; rawTokens += committed
                case .finished(let reason, let usage):
                    #expect(reason == .stop && usage.completionTokens == result.generatedTokenCount)
                    terminalCount += 1
                }
            }
            #expect(streamed == text && terminalCount == 1)
            #expect(rawTokens.prefix(result.tokenIds.count).map(Int32.init) == result.tokenIds)
            #expect(engine.capacity().kvBytesReserved == 0)
            let row: [String: Any] = ["color": name, "rawText": text, "promptTokens": tokens.count,
                "softTokens": geometry.softTokens, "steps": result.denoisingSteps,
                "activeBytes": Memory.activeMemory, "cacheBytes": Memory.cacheMemory,
                "nativeEngineTokenExact": rawTokens.prefix(result.tokenIds.count).map(Int32.init) == result.tokenIds,
                "scope": "native SDK full quant + image processor + native engine; no HTTP claim"]
            print("DIFFUSION_MEDIA_LIVE " + String(decoding: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), as: UTF8.self))
            Memory.clearCache()
        }
        await engine.shutdown()
        let textTokens = try context.renderTokens(messages: [["role": "user", "content": "What is 2+2? Reply with the digit only."]],
            additionalContext: ["enable_thinking": false])
        let result = try context.model.generateNative(promptTokenIds: MLXArray(textTokens).reshaped(1, textTokens.count),
            generation: generation, seed: 8132)
        #expect(tokenizer.decode(tokens: result.tokenIds.map(Int.init), skipSpecialTokens: false).contains("4"))
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_LIVE"] == "1"))
    func selectedArtifactLoadsAndGenerates() async throws {
        let env = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(env["DARKBLOOM_DIFFUSION_MODEL_DIR"]))
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configHash = SHA256.hash(data: configData).map { String(format: "%02x", $0) }.joined()
        // Immutable selected quant, not a similarly named architecture/checkpoint.
        #expect(configHash == "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5")
        guard configHash == "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5"
        else {
            throw DiffusionGemmaModelError.invalidInput("unqualified live artifact")
        }
        let generationData = try Data(
            contentsOf: directory.appendingPathComponent("generation_config.json"))
        let generation = try JSONDecoder().decode(
            DiffusionGemmaGenerationConfiguration.self, from: generationData)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let template = try String(
            contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let prompts = [
            "Explain why leaves are green in three clear sentences.",
            "Write a short Python function that adds two integers. Then explain it in one sentence.",
        ]
        // Preflight template/tokenizer before any heavyweight allocation.
        let tokens = try prompts.map { prompt in
            try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]],
                chatTemplate: .literal(SwiftJinjaSyntaxCompatibility.normalize(template)),
                addGenerationPrompt: true, truncation: false,
                maxLength: nil, tools: nil, additionalContext: ["enable_thinking": false])
        }
        for (index, ids) in tokens.enumerated() {
            let diagnostic: [String: Any] = [
                "event": "template_preflight", "index": index,
                "tokenIds": ids, "decoded": tokenizer.decode(tokens: ids, skipSpecialTokens: false),
            ]
            print(
                "DIFFUSION_LIVE "
                    + String(
                        decoding: try JSONSerialization.data(
                            withJSONObject: diagnostic, options: [.sortedKeys]), as: UTF8.self))
        }
        try #require(
            tokens.map(\.count) == [19, 26],
            "Template/tokenizer parity must pass before loading weights")
        // Register the owned XCTest resource bundle before any device/memory
        // API. The headless test helper otherwise has no main-app bundle.
        _ = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        func emit(_ row: [String: Any]) throws {
            let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            print("DIFFUSION_LIVE " + String(decoding: data, as: UTF8.self))
        }
        Memory.peakMemory = 0
        let started = ProcessInfo.processInfo.systemUptime
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(tokenizer: #adaptHuggingFaceTokenizer(tokenizer)))
        let model = context.model
        #expect(context.generationConfiguration == generation)
        for (index, prompt) in prompts.enumerated() {
            let nativeTokens = try context.renderTokens(
                messages: [["role": "user", "content": prompt]],
                additionalContext: ["enable_thinking": false])
            #expect(nativeTokens.map(Int.init) == tokens[index])
        }
        #expect(model.parameters().flattened().count == 1647)
        try emit([
            "event": "loaded", "configSHA256": configHash,
            "loadSeconds": ProcessInfo.processInfo.systemUptime - started,
            "parameterTensors": model.parameters().flattened().count,
            "activeBytes": Memory.activeMemory, "cacheBytes": Memory.cacheMemory,
            "peakBytes": Memory.peakMemory, "scope": "native SDK only",
        ])
        let engine = try context.makeNativeEngine(kvBytesCapacity: 16 * 1024 * 1024 * 1024)
        for (index, input) in tokens.enumerated() {
            Memory.peakMemory = 0
            var blocks = [[Int32]]()
            let result = try model.generateNative(
                promptTokenIds: MLXArray(input.map(Int32.init)).reshaped(1, input.count),
                generation: generation, seed: UInt64(4100 + index),
                onCommittedBlock: { blocks.append($0) })
            #expect(!result.tokenIds.isEmpty)
            #expect(blocks.flatMap { $0 } == result.tokenIds)
            #expect(result.generatedTokenCount <= 256)
            let text = tokenizer.decode(
                tokens: result.tokenIds.map(Int.init), skipSpecialTokens: false)
            #expect(!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            try emit([
                "event": "request", "index": index, "promptTokens": input.count,
                "committedTokens": result.tokenIds.count,
                "generatedIncludingEOS": result.generatedTokenCount,
                "finishReason": result.finishReason, "denoisingSteps": result.denoisingSteps,
                "canvasCount": result.committedCanvasCount, "workTokens": result.workTokenCount,
                "prefillSeconds": result.prefillSeconds, "totalSeconds": result.totalSeconds,
                "firstCommittedOutputSeconds": result.firstCommittedOutputSeconds ?? -1,
                "activeBytes": Memory.activeMemory, "cacheBytes": Memory.cacheMemory,
                "peakBytes": Memory.peakMemory, "rawText": text, "tokenIds": result.tokenIds,
                "qualityQualification": false, "providerQualification": false,
            ])
            let cache = try model.makeCache(expectedPromptLength: input.count)
            let promptArray = MLXArray(input.map(Int32.init)).reshaped(1, input.count)
            _ = try model.encode(tokenIds: promptArray, cache: cache)
            let prefixIdentity = try DiffusionGemmaPrefixIdentity(
                tenantScope: "local-live-fixture", artifact: configHash,
                template: "pinned-mlx4-template", media: "text-only",
                numericalProfile: "canonical-strict-metal", epoch: "current-factory-load")
            let checkpoint = try model.model.decoder.checkpoint(
                cache: cache, identity: prefixIdentity)
            let warm = try model.generateNative(
                promptTokenIds: promptArray, generation: generation, seed: UInt64(4100 + index),
                prefixCheckpoint: checkpoint, prefixIdentity: prefixIdentity)
            #expect(warm.tokenIds == result.tokenIds)
            #expect(warm.generatedTokenCount == result.generatedTokenCount)
            #expect(warm.denoisingSteps == result.denoisingSteps)
            #expect(warm.reusedPromptTokenCount == input.count)
            try emit([
                "event": "prefix-reuse", "index": index,
                "reusedPromptTokens": warm.reusedPromptTokenCount,
                "tokenExact": warm.tokenIds == result.tokenIds,
                "stepsExact": warm.denoisingSteps == result.denoisingSteps,
                "prefillSeconds": warm.prefillSeconds, "totalSeconds": warm.totalSeconds,
                "scope": "native in-process committed text prefix only",
                "providerQualification": false,
            ])
            let stream = try engine.submit(
                .init(
                    id: .init(UInt64(index + 1)), promptTokens: input,
                    sampling: .init(seed: UInt64(4100 + index)), maxTokens: generation.maxNewTokens)
            )
            var streamText = ""
            var raw = [Int]()
            var usage: CBv2Usage?
            var terminal: CBv2FinishReason?
            for await event in stream {
                switch event {
                case .delta(let text, let tokens, _):
                    streamText += text
                    raw += tokens
                case .finished(let reason, let counts):
                    terminal = reason
                    usage = counts
                }
            }
            #expect(raw.prefix(result.tokenIds.count).map(Int32.init) == result.tokenIds)
            #expect(raw.count == result.generatedTokenCount)
            #expect(streamText == text)
            #expect(usage?.completionTokens == result.generatedTokenCount)
            #expect(terminal == (result.finishReason == "stop" ? .stop : .length))
            #expect(engine.capacity().kvBytesReserved == 0)
            try emit([
                "event": "native-engine", "index": index,
                "rawCommittedTokens": raw.count, "completionTokens": usage?.completionTokens ?? -1,
                "textExact": streamText == text,
                "tokenExact": raw.prefix(result.tokenIds.count).map(Int32.init) == result.tokenIds,
                "batchRowsMax": usage?.timing.batchRowsMax ?? 0,
                "nativeSteps": engine.capacity().stepsExecuted,
                "providerQualification": false,
            ])
            Memory.clearCache()
        }
        await engine.shutdown()
    }
}
