// Copyright © 2026 Apple Inc.
//
// Smoke tests for the Gemma 4 MTP benchmark primitives. Verifies the
// measurement harness runs end-to-end, produces finite timings, and
// returns a token count matching `maxTokens`. Actual throughput numbers
// are hardware-dependent and are not asserted; a separate driver
// (outside the test target) collects them against real models.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import MLXSpeculative
import Testing

@Suite("Gemma4MTPBenchmark")
struct Gemma4MTPBenchmarkTests {

    /// Tiny target for smoke tests — matches the E2B-shaped config used
    /// in the parity suite so we know the harness runs cleanly.
    private func tinyTargetConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 6,
            "sliding_window": 128,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyDrafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": false,
            "num_centroids": 32,
            "centroid_intermediate_top_k": 4,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "sliding_window": 64,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    }

    @Test func baselineHarnessRunsAndProducesFiniteTimings() async throws {
        MLXRandom.seed(42)
        let target = Gemma4TextModel(try tinyTargetConfig())
        eval(target)
        let prompt = MLXArray([Int32](repeating: 7, count: 8))

        let r = measureBaselineThroughput(
            target: target, promptTokens: prompt, maxTokens: 8)

        #expect(r.generatedTokens == 8)
        #expect(r.prefillSeconds.isFinite && r.prefillSeconds >= 0)
        #expect(r.generationSeconds.isFinite && r.generationSeconds >= 0)
        #expect(r.acceptLengths == nil, "baseline has no accept histogram")
    }

    @Test func mtpHarnessRunsAndProducesAcceptLengths() async throws {
        MLXRandom.seed(42)
        let target = Gemma4TextModel(try tinyTargetConfig())
        let drafter = Gemma4AssistantDraftModel(config: try tinyDrafterConfig())
        eval(target, drafter)
        let prompt = MLXArray([Int32](repeating: 7, count: 8))

        let r = try await measureMTPThroughput(
            target: target, drafter: drafter,
            promptTokens: prompt, maxTokens: 12, blockSize: 3)

        #expect(r.generatedTokens == 12)
        #expect(r.prefillSeconds.isFinite && r.prefillSeconds >= 0)
        #expect(r.generationSeconds.isFinite && r.generationSeconds >= 0)
    }

    // MARK: - Real-model benchmark (opt-in, gated on MTP_BENCH_DATA_DIR)

    /// Real-model benchmark. Measures tokens/sec for baseline vs MTP on
    /// locally-resident models. Skipped unless `MTP_BENCH_DATA_DIR` env var
    /// points to a directory containing:
    ///   - gemma-4-e2b-it-bf16/
    ///   - gemma-4-E2B-it-assistant-bf16/
    /// (and optionally the e4b / 26b-a4b-4bit pairs).
    ///
    /// Produces a printed table. No assertions on throughput — numbers are
    /// hardware-dependent. Used to populate the README performance section.
    @Test func realModelThroughputBenchmark() async throws {
        guard let dataDir = ProcessInfo.processInfo.environment["MTP_BENCH_DATA_DIR"]
        else {
            print("Skipping: MTP_BENCH_DATA_DIR env var not set")
            return
        }

        struct Pair {
            let label: String
            let targetDir: String
            let drafterDir: String
        }
        let allPairs: [Pair] = [
            Pair(label: "E2B-bf16",
                 targetDir: "gemma-4-e2b-it-bf16",
                 drafterDir: "gemma-4-E2B-it-assistant-bf16"),
            Pair(label: "E4B-bf16",
                 targetDir: "gemma-4-e4b-it-bf16",
                 drafterDir: "gemma-4-E4B-it-assistant-bf16"),
            Pair(label: "26B-A4B-4bit",
                 targetDir: "gemma-4-26b-a4b-it-4bit",
                 drafterDir: "gemma-4-26B-A4B-it-assistant-bf16"),
        ]
        let pairs = allPairs.filter {
            FileManager.default.fileExists(atPath: "\(dataDir)/\($0.targetDir)")
                && FileManager.default.fileExists(atPath: "\(dataDir)/\($0.drafterDir)")
        }
        #expect(!pairs.isEmpty, "No model pairs found in \(dataDir)")

        // Load real chat-templated prompt tokens from a JSON fixture
        // (produced via Python's transformers tokenizer — path pointed to
        // by MTP_BENCH_PROMPTS env var). Using garbage token IDs produces
        // misleading "slower than baseline" results because the drafter
        // has no signal to predict from. Falls back to a single arbitrary
        // prompt if the fixture isn't provided.
        let promptsData: [[Int32]]
        if let fixturePath = ProcessInfo.processInfo.environment["MTP_BENCH_PROMPTS"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
            let decoded = try JSONDecoder().decode([[Int]].self, from: data)
            promptsData = decoded.map { $0.map { Int32($0) } }
        } else {
            // Degenerate fallback. Use MTP_BENCH_PROMPTS for real numbers.
            promptsData = [(0 ..< 16).map { Int32($0 * 211 + 7) }]
        }
        let maxTokens = 64
        let warmupTokens = 16
        let blockSizes = [2, 3, 4, 5]

        print("\n\n=== Gemma 4 MTP real-model benchmark ===")
        print("prompts=\(promptsData.count), max_tokens=\(maxTokens), warmup=\(warmupTokens)")
        print("model              K  base tok/s  mtp tok/s  speedup  accept_avg")

        for pair in pairs {
            let targetURL = URL(fileURLWithPath: "\(dataDir)/\(pair.targetDir)")
            let drafterURL = URL(fileURLWithPath: "\(dataDir)/\(pair.drafterDir)")
            print("--- loading \(pair.label) ---")

            let (target, _) = try loadRealTargetAsTextModel(from: targetURL)
            let drafter = try await Gemma4AssistantDraftModel.load(from: drafterURL)
            eval(target, drafter)

            // Warmup on first prompt with largest block size to exercise
            // the full drafter path.
            let warmupPrompt = MLXArray(promptsData[0])
            _ = measureBaselineThroughput(
                target: target, promptTokens: warmupPrompt, maxTokens: warmupTokens)
            _ = try measureMTPThroughput(
                target: target, drafter: drafter,
                promptTokens: warmupPrompt, maxTokens: warmupTokens,
                blockSize: blockSizes.max()!)
            MLX.Memory.clearCache()

            // Baseline once per prompt (block-size invariant).
            var baseRates: [Double] = []
            for promptInts in promptsData {
                let prompt = MLXArray(promptInts)
                let base = measureBaselineThroughput(
                    target: target, promptTokens: prompt, maxTokens: maxTokens)
                MLX.Memory.clearCache()
                baseRates.append(base.tokensPerSecond)
            }
            let baseAvg = baseRates.reduce(0, +) / Double(baseRates.count)

            for K in blockSizes {
                var mtpRates: [Double] = []
                var allAccepts: [Int] = []
                for promptInts in promptsData {
                    let prompt = MLXArray(promptInts)
                    let mtp = try measureMTPThroughput(
                        target: target, drafter: drafter,
                        promptTokens: prompt, maxTokens: maxTokens,
                        blockSize: K)
                    MLX.Memory.clearCache()
                    mtpRates.append(mtp.tokensPerSecond)
                    allAccepts.append(contentsOf: mtp.acceptLengths ?? [])
                }
                let mtpAvg = mtpRates.reduce(0, +) / Double(mtpRates.count)
                let accAvg = allAccepts.isEmpty
                    ? 0.0
                    : Double(allAccepts.reduce(0, +)) / Double(allAccepts.count)
                let speedup = mtpAvg / max(baseAvg, 1e-9)
                let baseStr = String(format: "%10.1f", baseAvg)
                let mtpStr = String(format: "%10.1f", mtpAvg)
                let spdStr = String(format: "%.2fx", speedup)
                let accStr = String(format: "%.2f", accAvg)
                let label = pair.label.padding(toLength: 18, withPad: " ", startingAt: 0)
                print("\(label) \(K)  \(baseStr) \(mtpStr)   \(spdStr)   \(accStr)/\(K-1)")
            }

            // Batched (B=4) — the apples-to-apples config vs Python's
            // headline numbers which use batch=4.
            let B4 = Array(repeating: promptsData[0], count: 4)
            let blockForBatch = 4
            _ = measureBatchedBaselineThroughput(
                target: target, promptTokens: B4, maxTokens: warmupTokens)
            _ = try await measureBatchedMTPThroughput(
                target: target, drafter: drafter,
                promptTokens: B4, maxTokens: warmupTokens,
                blockSize: blockForBatch)
            MLX.Memory.clearCache()

            let batchBase = measureBatchedBaselineThroughput(
                target: target, promptTokens: B4, maxTokens: maxTokens)
            MLX.Memory.clearCache()
            let batchMtp = try await measureBatchedMTPThroughput(
                target: target, drafter: drafter,
                promptTokens: B4, maxTokens: maxTokens,
                blockSize: blockForBatch)
            MLX.Memory.clearCache()
            let batchSpeedup = batchMtp.tokensPerSecond
                             / max(batchBase.tokensPerSecond, 1e-9)
            let bbStr = String(format: "%10.1f", batchBase.tokensPerSecond)
            let bmStr = String(format: "%10.1f", batchMtp.tokensPerSecond)
            let bsStr = String(format: "%.2fx", batchSpeedup)
            let labelB4 = pair.label.padding(toLength: 18, withPad: " ", startingAt: 0)
            print("\(labelB4) 4(B=4) \(bbStr) \(bmStr)   \(bsStr)   -- (aggregate, uniform budget)")

            // Continuous-batching benchmark: staggered budgets at B ∈
            // {4, 8, 16}. Each row gets a different maxTokens cap, so
            // rows finish at different times and compaction can fire.
            //
            // For each B: run WITH compaction (maxTokensPerRow set) vs
            // WITHOUT compaction (uniform maxTokens = max of budgets,
            // pad all rows). Compare wall time for the SAME total
            // emitted tokens.
            //
            // B=16 only runs when MTP_BENCH_LARGE_B=1 is set (requires
            // >64 GB to be safe on 26B-A4B).
            let batchSizes: [Int] = {
                if ProcessInfo.processInfo.environment["MTP_BENCH_LARGE_B"] != nil {
                    return [4, 8, 16]
                }
                return [4]
            }()
            print("\(labelB4)   --- continuous-batching comparison ---")
            print("\(labelB4)   B   budgets            compact_s   pad_s    win")
            for B in batchSizes {
                // Build staggered budgets — half the rows at max, a
                // quarter at max/2, a quarter at max/4 + max/8.
                var budgets: [Int] = []
                for i in 0 ..< B {
                    switch i % 4 {
                    case 0: budgets.append(Swift.max(maxTokens / 8, 1))
                    case 1: budgets.append(Swift.max(maxTokens / 4, 1))
                    case 2: budgets.append(maxTokens / 2)
                    default: budgets.append(maxTokens)
                    }
                }
                let budgetMax = budgets.max() ?? maxTokens
                let budgetTotal = budgets.reduce(0, +)

                let prompts = Array(repeating: promptsData[0], count: B)

                // Warmup (small budget) to clear kernel-compile noise.
                _ = try await measureBatchedMTPThroughputStaggered(
                    target: target, drafter: drafter,
                    promptTokens: prompts,
                    maxTokensPerRow: Array(repeating: 4, count: B),
                    blockSize: blockForBatch)
                MLX.Memory.clearCache()

                // Compacted run (true continuous batching).
                let compactRun = try await measureBatchedMTPThroughputStaggered(
                    target: target, drafter: drafter,
                    promptTokens: prompts, maxTokensPerRow: budgets,
                    blockSize: blockForBatch)
                MLX.Memory.clearCache()

                // Non-compacted run: all rows run to budgetMax (pad).
                // Reports per-token time, so the win is compact_per_tok /
                // pad_per_tok invariant (>1 means compact is faster per
                // emitted token, regardless of total token count).
                let padRun = try await measureBatchedMTPThroughput(
                    target: target, drafter: drafter,
                    promptTokens: prompts,
                    maxTokens: budgetMax,
                    blockSize: blockForBatch)
                MLX.Memory.clearCache()

                let compactPerTok =
                    compactRun.generationSeconds
                    / Double(max(compactRun.totalGenerated, 1))
                let padPerTok =
                    padRun.generationSeconds
                    / Double(max(padRun.totalGenerated, 1))
                let win = padPerTok / compactPerTok
                _ = budgetTotal  // kept for clarity above
                let budgetStr = budgets.map(String.init).joined(separator: ",")
                let compStr = String(
                    format: "%.1f tok/s", 1.0 / compactPerTok)
                let padStr = String(
                    format: "%.1f tok/s", 1.0 / padPerTok)
                let winStr = String(format: "%.2fx", win)
                print(
                    "\(labelB4)   \(B)   "
                        + "\(budgetStr.padding(toLength: 18, withPad: " ", startingAt: 0))"
                        + " compact=\(compStr)  pad=\(padStr)  win=\(winStr)")
            }
        }
    }

    /// Load a HF Gemma 4 target (VLM-format checkpoint) into a bare
    /// `Gemma4TextModel`. The VLM wrapper is instantiated briefly so its
    /// sanitize() can strip vision/audio weights + remap key names; we
    /// then reach in and return the inner text model for MTP use.
    private func loadRealTargetAsTextModel(
        from modelDir: URL
    ) throws -> (Gemma4TextModel, Gemma4Configuration) {
        let configData = try Data(
            contentsOf: modelDir.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        let fullConfig = try decoder.decode(Gemma4Configuration.self, from: configData)
        let baseConfig = try? decoder.decode(BaseConfiguration.self, from: configData)

        let wrapper = Gemma4Model(fullConfig)
        try loadWeights(
            modelDirectory: modelDir,
            model: wrapper,
            quantization: nil,
            perLayerQuantization: baseConfig?.perLayerQuantization
        )
        return (wrapper.textModel, fullConfig)
    }
}
