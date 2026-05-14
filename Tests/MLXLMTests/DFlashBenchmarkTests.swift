// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@Suite("DFlashBenchmark", .serialized)
struct DFlashBenchmarkTests {
    @Test func benchmarkHelpersRunAndProduceFiniteTimings() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let prompt = MLXArray([Int32(1), 2, 3])
            let baseline = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: prompt,
                maxTokens: 4
            )
            let dflash = try measureDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: prompt,
                maxTokens: 4,
                blockSize: 3,
                collectPhaseTimings: true
            )
            let phases = try #require(dflash.phaseTimings)

            #expect(baseline.generatedTokens == 4)
            #expect(dflash.generatedTokens == 4)
            #expect(baseline.prefillSeconds.isFinite && baseline.prefillSeconds >= 0)
            #expect(baseline.generationSeconds.isFinite && baseline.generationSeconds >= 0)
            #expect(dflash.prefillSeconds.isFinite && dflash.prefillSeconds >= 0)
            #expect(dflash.generationSeconds.isFinite && dflash.generationSeconds >= 0)
            #expect(phases.rounds > 0)
            #expect(phases.roundSeconds.isFinite && phases.roundSeconds >= 0)
            #expect(phases.verifyAndWaitSeconds.isFinite && phases.verifyAndWaitSeconds >= 0)

            let stopAfterTwo: DFlashStopPredicate = { generated in
                generated.count >= 2 ? 2 : nil
            }
            let stoppedBaseline = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: prompt,
                maxTokens: 4,
                stopAfterGeneratedTokenCount: stopAfterTwo
            )
            let stoppedDFlash = try measureDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: prompt,
                maxTokens: 4,
                blockSize: 3,
                stopAfterGeneratedTokenCount: stopAfterTwo
            )
            #expect(stoppedBaseline.generatedTokens == 2)
            #expect(stoppedBaseline.generatedTokenIds.count == 2)
            #expect(stoppedDFlash.generatedTokens == 2)
            #expect(stoppedDFlash.generatedTokenIds.count == 2)
        }
    }

    @Test func batchedBenchmarkHelpersRunAndProduceFiniteTimings() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let prompts: [[Int32]] = [
                [1, 2, 3],
                [1, 2, 3],
            ]
            let baseline = try measureBatchedDFlashBaselineThroughput(
                target: target,
                promptTokens: prompts,
                maxTokens: 4
            )
            let dflash = try measureBatchedDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: prompts,
                maxTokens: 4,
                blockSize: 3
            )

            #expect(baseline.batchSize == 2)
            #expect(dflash.batchSize == 2)
            #expect(baseline.generatedTokensPerRow == [4, 4])
            #expect(dflash.generatedTokensPerRow == [4, 4])
            #expect(baseline.totalGeneratedTokens == 8)
            #expect(dflash.totalGeneratedTokens == 8)
            #expect(baseline.generatedTokenIds.count == 2)
            #expect(dflash.generatedTokenIds.count == 2)
            #expect(baseline.generatedTokenIds.allSatisfy { $0.count == 4 })
            #expect(dflash.generatedTokenIds.allSatisfy { $0.count == 4 })
            #expect(dflash.generatedTokenIds == baseline.generatedTokenIds)
            #expect(baseline.prefillSeconds.isFinite && baseline.prefillSeconds >= 0)
            #expect(baseline.generationSeconds.isFinite && baseline.generationSeconds >= 0)
            #expect(dflash.prefillSeconds.isFinite && dflash.prefillSeconds >= 0)
            #expect(dflash.generationSeconds.isFinite && dflash.generationSeconds >= 0)
        }
    }

    @Test func batchedBenchmarkKeepsDFlashForHeterogeneousPrompts() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let prompts: [[Int32]] = [
                [1, 2, 3],
                [2, 4, 6, 8],
                [3, 5],
            ]
            let baseline = try measureBatchedDFlashBaselineThroughput(
                target: target,
                promptTokens: prompts,
                maxTokens: 5
            )
            let dflash = try measureBatchedDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: prompts,
                maxTokens: 5,
                blockSize: 3
            )

            #expect(dflash.batchSize == prompts.count)
            #expect(dflash.generatedTokensPerRow == [5, 5, 5])
            #expect(dflash.acceptLengths != nil)
            #expect(dflash.generatedTokenIds == baseline.generatedTokenIds)
        }
    }

    @Test func batchedEffectiveBlockSizeUsesSingleRowCap() {
        #expect(dFlashBatchedEffectiveBlockSize(requestedBlockSize: 16, activeBatchSize: 1) == 4)
        #expect(
            dFlashBatchedEffectiveBlockSize(
                requestedBlockSize: 16,
                activeBatchSize: 1,
                totalLiveRequestCount: 4) == 4)
        #expect(dFlashBatchedEffectiveBlockSize(requestedBlockSize: 6, activeBatchSize: 1) == 4)
        #expect(dFlashBatchedEffectiveBlockSize(requestedBlockSize: 16, activeBatchSize: 2) == 4)
        #expect(dFlashBatchedEffectiveBlockSize(requestedBlockSize: 16, activeBatchSize: 4) == 4)
    }

    @Test func batchedTokenGeneratorSupportsStaggeredConcurrentRequests() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let prompts: [[Int32]] = [
                [1, 2, 3],
                [1, 2, 3],
                [2, 4, 6, 8],
            ]
            let maxTokens = 5
            let generator = try DFlashBatchedTokenGenerator(
                target: target,
                drafter: drafter,
                blockSize: 3,
                parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0)
            )

            try generator.insert(
                prompts: Array(prompts.prefix(2)),
                uids: [10, 11],
                maxTokens: [maxTokens, maxTokens])

            var generatedByUID: [Int: [Int]] = [:]
            for response in try generator.next() {
                generatedByUID[response.uid, default: []].append(response.token)
            }

            try generator.insert(
                prompts: [prompts[2]],
                uids: [12],
                maxTokens: [maxTokens])

            var finishReasons: [Int: String] = [:]
            while !generator.isEmpty {
                for response in try generator.next() {
                    generatedByUID[response.uid, default: []].append(response.token)
                    if let finishReason = response.finishReason {
                        finishReasons[response.uid] = finishReason
                    }
                }
            }

            for (uid, prompt) in zip([10, 11, 12], prompts) {
                let expected = measureDFlashBaselineThroughput(
                    target: target,
                    promptTokens: MLXArray(prompt),
                    maxTokens: maxTokens
                )
                #expect(generatedByUID[uid] == expected.generatedTokenIds)
                #expect(finishReasons[uid] == "length")
            }
        }
    }

    @Test func dFlashBatchedEngineRunsConcurrentRequestsOnOneGenerator() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let tokenizer = DeterministicDFlashTokenizer()
            let context = ModelContext(
                configuration: ModelConfiguration(id: "test"),
                model: target,
                processor: StandInUserInputProcessor(),
                tokenizer: tokenizer)
            let engine = try DFlashBatchedEngine(
                context: context,
                drafter: drafter,
                blockSize: 3)
            await engine.start()
            defer {
                Task { await engine.stop() }
            }

            async let first = engine.generateWithResult(
                prompt: "alpha",
                samplingParams: SamplingParams(maxTokens: 5, temperature: 0))
            async let second = engine.generateWithResult(
                prompt: "alpha",
                samplingParams: SamplingParams(maxTokens: 5, temperature: 0))
            async let third = engine.generateWithResult(
                prompt: "beta",
                samplingParams: SamplingParams(maxTokens: 5, temperature: 0))

            let outputs = try await [first, second, third]
            let expectedAlpha = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: MLXArray(tokenizer.promptTokens["alpha"]!),
                maxTokens: 5)
            let expectedBeta = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: MLXArray(tokenizer.promptTokens["beta"]!),
                maxTokens: 5)

            #expect(outputs[0].outputTokenIds == expectedAlpha.generatedTokenIds)
            #expect(outputs[1].outputTokenIds == expectedAlpha.generatedTokenIds)
            #expect(outputs[2].outputTokenIds == expectedBeta.generatedTokenIds)
            #expect(outputs.allSatisfy { $0.finishReason == "length" })
        }
    }

    /// Real-checkpoint benchmark. Skipped unless local target and DFlash
    /// drafter directories are provided.
    ///
    /// Required env vars:
    ///   - `DFLASH_BENCH_TARGET_DIR`, or `MLX_SWIFT_LM_DFLASH_TARGET_DIR`
    ///   - `DFLASH_BENCH_DRAFTER_DIR`, or `MLX_SWIFT_LM_DFLASH_DRAFTER_DIR`
    ///
    /// Optional env vars:
    ///   - `DFLASH_BENCH_PROMPTS`: JSON file containing `[[Int]]`
    ///   - `DFLASH_BENCH_PROMPT_TOKENS`: comma-separated token ids
    ///   - `DFLASH_BENCH_MAX_TOKENS`: default `128`
    ///   - `DFLASH_BENCH_WARMUP_TOKENS`: default `min(16, maxTokens)`
    ///   - `DFLASH_BENCH_BLOCK_SIZES`: comma-separated block sizes
    @Test func realCheckpointThroughputBenchmark() async throws {
        let env = ProcessInfo.processInfo.environment
        guard
            let targetPath = env["DFLASH_BENCH_TARGET_DIR"]
                ?? env["MLX_SWIFT_LM_DFLASH_TARGET_DIR"],
            let drafterPath = env["DFLASH_BENCH_DRAFTER_DIR"]
                ?? env["MLX_SWIFT_LM_DFLASH_DRAFTER_DIR"]
        else {
            print("Skipping: DFLASH_BENCH_TARGET_DIR/DFLASH_BENCH_DRAFTER_DIR not set")
            return
        }

        let targetDirectory = URL(fileURLWithPath: targetPath)
        let drafterDirectory = URL(fileURLWithPath: drafterPath)
        let targetContext = try await LLMModelFactory.shared.load(
            from: targetDirectory,
            using: DFlashBenchmarkTokenizerLoader()
        )
        let targetModel = try #require(targetContext.model as? any DFlashTargetModel)
        let drafter = try await DFlashDraftModel.load(
            from: drafterDirectory,
            bindTo: targetModel
        )
        eval(targetContext.model, drafter)

        let prompts = try loadBenchmarkPrompts(env: env)
        let maxTokens = envInt("DFLASH_BENCH_MAX_TOKENS", env: env, default: 128)
        let warmupTokens = envInt(
            "DFLASH_BENCH_WARMUP_TOKENS",
            env: env,
            default: Swift.min(16, maxTokens)
        )
        let blockSizes = envIntList(
            "DFLASH_BENCH_BLOCK_SIZES",
            env: env,
            default: [drafter.config.blockSize]
        )

        print("\n\n=== DFlash real-checkpoint benchmark ===")
        print("target=\(targetDirectory.path)")
        print("drafter=\(drafterDirectory.path)")
        print("prompts=\(prompts.count), max_tokens=\(maxTokens), warmup=\(warmupTokens)")
        print("K  base tok/s  dflash tok/s  speedup  generated")

        let warmupPrompt = MLXArray(prompts[0])
        _ = measureDFlashBaselineThroughput(
            target: targetModel,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens
        )
        _ = try measureDFlashThroughput(
            target: targetModel,
            drafter: drafter,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens,
            blockSize: blockSizes.max()
        )
        MLX.Memory.clearCache()

        var baseRates: [Double] = []
        for promptInts in prompts {
            let baseline = measureDFlashBaselineThroughput(
                target: targetModel,
                promptTokens: MLXArray(promptInts),
                maxTokens: maxTokens
            )
            baseRates.append(baseline.tokensPerSecond)
            MLX.Memory.clearCache()
        }
        let baseAverage = average(baseRates)

        for blockSize in blockSizes {
            var dflashRates: [Double] = []
            var generatedCounts: [Int] = []
            for promptInts in prompts {
                let dflash = try measureDFlashThroughput(
                    target: targetModel,
                    drafter: drafter,
                    promptTokens: MLXArray(promptInts),
                    maxTokens: maxTokens,
                    blockSize: blockSize
                )
                dflashRates.append(dflash.tokensPerSecond)
                generatedCounts.append(dflash.generatedTokens)
                MLX.Memory.clearCache()
            }

            let dflashAverage = average(dflashRates)
            let speedup = dflashAverage / max(baseAverage, 1e-9)
            let generated = generatedCounts.map(String.init).joined(separator: ",")
            print(
                "\(blockSize)  "
                    + "\(String(format: "%10.1f", baseAverage)) "
                    + "\(String(format: "%12.1f", dflashAverage))   "
                    + "\(String(format: "%.2fx", speedup))   "
                    + "\(generated)"
            )
        }
    }

    private func tinyQwen3Config() throws -> Qwen3Configuration {
        let json = """
        {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128
        }
        """
        return try JSONDecoder.json5().decode(Qwen3Configuration.self, from: Data(json.utf8))
    }

    private func dflashConfig() throws -> DFlashConfiguration {
        let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 128,
            "block_size": 4,
            "num_target_layers": 3,
            "layer_types": ["full_attention", "full_attention"],
            "tie_word_embeddings": true,
            "dflash_config": {
                "target_layer_ids": [0, 1],
                "mask_token_id": 4
            }
        }
        """
        return try JSONDecoder.json5().decode(DFlashConfiguration.self, from: Data(json.utf8))
    }

    private func loadBenchmarkPrompts(env: [String: String]) throws -> [[Int32]] {
        if let fixturePath = env["DFLASH_BENCH_PROMPTS"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
            let decoded = try JSONDecoder().decode([[Int]].self, from: data)
            let prompts = decoded.map { $0.map(Int32.init) }.filter { !$0.isEmpty }
            if !prompts.isEmpty { return prompts }
        }

        if let tokenString = env["DFLASH_BENCH_PROMPT_TOKENS"]
            ?? env["MLX_SWIFT_LM_DFLASH_PROMPT_TOKENS"]
        {
            let tokens = tokenString
                .split(separator: ",")
                .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            if !tokens.isEmpty { return [tokens] }
        }

        return [(0 ..< 16).map { Int32($0 * 211 + 7) }]
    }

    private func envInt(
        _ name: String,
        env: [String: String],
        default defaultValue: Int
    ) -> Int {
        guard let value = env[name], let parsed = Int(value), parsed > 0 else {
            return defaultValue
        }
        return parsed
    }

    private func envIntList(
        _ name: String,
        env: [String: String],
        default defaultValue: [Int]
    ) -> [Int] {
        guard let value = env[name] else { return defaultValue }
        let parsed = value
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            .filter { $0 > 1 }
        return parsed.isEmpty ? defaultValue : parsed
    }

    private func average(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }
}

private struct DFlashBenchmarkTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        TestTokenizer(vocabularySize: 300_000)
    }
}

private struct DeterministicDFlashTokenizer: Tokenizer {
    let promptTokens: [String: [Int32]] = [
        "alpha": [1, 2, 3],
        "beta": [2, 4, 6, 8],
    ]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        (promptTokens[text] ?? [1]).map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(text: messages.last?["content"] as? String ?? "", addSpecialTokens: true)
    }
}
