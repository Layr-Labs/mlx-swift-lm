import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

@Suite("DiffusionGemma full-weight page-backed compatibility", .serialized)
struct DiffusionGemmaPagedLiveTests {
    private struct Loader: TokenizerLoader {
        let tokenizer: any MLXLMCommon.Tokenizer
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { tokenizer }
    }
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_PAGED_STATE_LIVE"] == "1"))
    func nativeGenerationAndReusedPrefixMatchContiguousExactly() async throws {
        let directory = URL(
            fileURLWithPath: try #require(
                ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_MODEL_DIR"]))
        let hash = SHA256.hash(
            data: try Data(contentsOf: directory.appendingPathComponent("config.json"))
        ).map { String(format: "%02x", $0) }.joined()
        try #require(hash == "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5")
        _ = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory,
            using: Loader(tokenizer: #adaptHuggingFaceTokenizer(tokenizer)))
        let config = context.model.configuration.textConfig
        let pool = try PagedKVBackend(
            layerKinds: config.diffusionPagedLayerKinds,
            config: .init(
                capacityBytes: 8 << 30, dtype: .bfloat16, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 4096, segmentSizeBytes: 8 << 20,
                layerDTypes: Array(repeating: .bfloat16, count: config.layerCount)))
        let notes = (1 ... 64).map {
            "Record \($0): Keep the input order, independent request state and exact arithmetic. Cache only committed values. Never expose an unfinished canvas."
        }.joined(separator: "\n")
        let prompts = [
            "Explain why leaves are green in three clear sentences.",
            notes + "\nWhat is 17 times 19? Reply with the number only.",
            "Write a detailed technical tutorial of at least 700 words about how a database executes a SQL query. Explain parsing, planning, indexes, joins, memory, transactions and recovery in separate sections. Begin with the tutorial and keep explaining until all seven topics are covered.",
        ]
        let base = context.generationConfiguration
        let recipe = try DiffusionGemmaGenerationConfiguration(
            maxNewTokens: 512, maxDenoisingSteps: base.maxDenoisingSteps,
            sampler: base.sampler, minimumTemperature: base.minimumTemperature,
            maximumTemperature: base.maximumTemperature,
            stabilityThreshold: base.stabilityThreshold,
            confidenceThreshold: base.confidenceThreshold,
            bosTokenId: base.bosTokenId, padTokenId: base.padTokenId, eosTokenIds: base.eosTokenIds)
        for (index, prompt) in prompts.enumerated() {
            let tokens = try context.renderTokens(
                messages: [["role": "user", "content": prompt]],
                additionalContext: ["enable_thinking": false])
            if index == 1 { try #require(tokens.count > config.slidingWindow) }
            let ids = MLXArray(tokens).reshaped(1, tokens.count)
            let identity = try DiffusionGemmaPrefixIdentity(
                tenantScope: "full-paged-fixture", artifact: hash,
                template: "unchanged-loaded", media: "text-only",
                numericalProfile: "canonical-native-sdpa", epoch: "current")
            var checkpoint: DiffusionGemmaPrefixCheckpoint?
            let cold = try context.model.generateNative(
                promptTokenIds: ids, generation: recipe, seed: 8132, prefillChunkSize: 512)
            let paged = try context.model.generateNative(
                promptTokenIds: ids, generation: recipe, seed: 8132,
                prefillChunkSize: 512, pagedBackend: pool, prefixIdentity: identity,
                onPromptCheckpoint: { checkpoint = $0 })
            try #require(
                paged.tokenIds == cold.tokenIds && paged.denoisingSteps == cold.denoisingSteps)
            try #require(
                paged.generatedTokenCount == cold.generatedTokenCount
                    && paged.finishReason == cold.finishReason)
            if index == 2 {
                try #require(
                    paged.committedCanvasCount >= 2
                        && paged.generatedTokenCount > context.model.configuration.canvasLength)
            }
            #expect(pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0)
            let reused = try context.model.generateNative(
                promptTokenIds: ids, generation: recipe, seed: 8132,
                prefillChunkSize: 512, pagedBackend: pool,
                prefixCheckpoint: try #require(checkpoint), prefixIdentity: identity)
            try #require(
                reused.tokenIds == cold.tokenIds && reused.denoisingSteps == cold.denoisingSteps)
            #expect(reused.reusedPromptTokenCount == tokens.count)
            #expect(pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0)
            let text = context.tokenizer.decode(
                tokenIds: paged.tokenIds.map(Int.init), skipSpecialTokens: false)
            try #require(!text.isEmpty)
            if index == 1 { try #require(text.contains("323")) }
            let row: [String: Any] = [
                "index": index, "promptTokens": tokens.count,
                "outputTokens": paged.generatedTokenCount, "denoisingSteps": paged.denoisingSteps,
                "committedCanvases": paged.committedCanvasCount,
                "tokenExact": true, "prefixRestored": reused.reusedPromptTokenCount,
                "coldTotalSeconds": cold.totalSeconds, "pagedTotalSeconds": paged.totalSeconds,
                "coldPrefillSeconds": cold.prefillSeconds,
                "pagedPrefillSeconds": paged.prefillSeconds,
                "mlxActiveBytes": Memory.activeMemory, "mlxCacheBytes": Memory.cacheMemory,
                "scope":
                    "native SDK page-backed gather + unchanged SDPA; not direct paged kernel, HTTP or matched benchmark",
            ]
            print(
                "DIFFUSION_PAGED_STATE_LIVE "
                    + String(
                        decoding: try JSONSerialization.data(
                            withJSONObject: row, options: [.sortedKeys]), as: UTF8.self))
            checkpoint = nil
            Memory.clearCache()
        }
    }
}
