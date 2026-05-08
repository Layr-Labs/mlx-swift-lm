// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXSpeculative
import Testing

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
