// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXSpeculative
import Testing

/// Swift-internal MTP parity. The invariant:
///
///    target_greedy(prompt) == MTP_greedy(prompt, drafter)
///
/// holds regardless of drafter weight quality — the MTP algorithm's
/// verify-and-correct path guarantees the drafter's presence doesn't
/// change the emitted token sequence. A divergence is an implementation
/// bug in the target forward, the drafter forward, the round loop, or
/// the cache rollback.
@Suite("Gemma4 MTP parity vs baseline (Swift-internal)")
struct Gemma4MTPParityTests {

    // MARK: - Config fixtures

    /// E2B-shaped target (reduced for test speed): 12 layers, 6 kv-shared,
    /// hidden=256, vocab=1024. Reduced vocab (2^10) keeps tests fast — the
    /// algorithm is vocab-size-agnostic.
    private func e2bStyleTargetConfig() throws -> Gemma4TextConfiguration {
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
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    /// E2B-shaped drafter. Matches target hidden_size + vocab_size for
    /// compat validation; 4 layers all kv-shared; configurable centroid head.
    private func e2bStyleDrafterConfig(
        useOrderedEmbeddings: Bool = false
    ) throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": \(useOrderedEmbeddings),
            "num_centroids": 32,
            "centroid_intermediate_top_k": 4,
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
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4AssistantConfiguration.self, from: data)
    }

    // MARK: - Baseline (no-drafter greedy)

    /// Run target-only greedy generation: prefill with promptTokens, then
    /// sample one token at a time for up to `maxTokens` steps. Returns the
    /// generated token IDs (excluding the prompt).
    private func runBaselineGreedy(
        target: Gemma4TextModel,
        promptTokens: [Int32],
        maxTokens: Int
    ) -> [Int] {
        let cache = target.newCache(parameters: nil)
        // Prefill.
        let prompt = MLXArray(promptTokens)[.newAxis, .ellipsis]
        var logits = target(prompt, cache: cache)  // [1, L, vocab]
        var tok = logits[0..., -1, 0...].argMax(axis: -1)  // [1]
        eval(tok)

        var out: [Int] = [Int(tok.item(Int32.self))]
        for _ in 1 ..< maxTokens {
            // Feed last token back; cache advances.
            let input = tok[.newAxis, .ellipsis]  // [1, 1]
            logits = target(input, cache: cache)
            tok = logits[0..., -1, 0...].argMax(axis: -1)
            eval(tok)
            out.append(Int(tok.item(Int32.self)))
        }
        return out
    }

    // MARK: - MTP greedy (via runGemma4MTPRounds)

    private func runMTPGreedy(
        target: Gemma4TextModel,
        drafter: Gemma4AssistantDraftModel,
        promptTokens: [Int32],
        maxTokens: Int,
        blockSize: Int
    ) async throws -> [Int] {
        // Prefill to produce firstBonus / firstHidden / firstSharedKV.
        let prompt = MLXArray(promptTokens)[.newAxis, .ellipsis]
        let cache = target.newCache(parameters: nil)
        let prefillOut = target.forwardForMTP(prompt, cache: cache)
        let lastLogits = prefillOut.logits[0..., -1, 0...]
        let firstBonus = Int(lastLogits.argMax(axis: -1).item(Int32.self))
        let firstHidden = prefillOut.lastHidden[
            0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
        let firstSharedKV = prefillOut.capturedSharedKV

        let stream = try runGemma4MTPRounds(
            target: target,
            drafter: drafter,
            targetCache: cache,
            firstBonus: firstBonus,
            firstHidden: firstHidden,
            firstSharedKV: firstSharedKV,
            maxTokens: maxTokens,
            blockSize: blockSize
        )
        var tokens: [Int] = []
        for await gen in stream {
            if case .chunk(let s) = gen, let tok = Int(s) {
                tokens.append(tok)
            }
        }
        return tokens
    }

    // MARK: - Parity tests (parameterized)

    @Test(arguments: [
        (blockSize: 2, promptLen: 4, maxTokens: 12),
        (blockSize: 3, promptLen: 4, maxTokens: 12),
        (blockSize: 4, promptLen: 4, maxTokens: 12),
        (blockSize: 2, promptLen: 16, maxTokens: 24),
        (blockSize: 3, promptLen: 16, maxTokens: 24),
        (blockSize: 4, promptLen: 16, maxTokens: 24),
        (blockSize: 2, promptLen: 64, maxTokens: 32),
        (blockSize: 3, promptLen: 64, maxTokens: 32),
        (blockSize: 4, promptLen: 64, maxTokens: 32),
    ])
    func dense_drafter_parity(
        config: (blockSize: Int, promptLen: Int, maxTokens: Int)
    ) async throws {
        let targetCfg = try e2bStyleTargetConfig()
        let target = Gemma4TextModel(targetCfg)
        let drafterCfg = try e2bStyleDrafterConfig(useOrderedEmbeddings: false)
        let drafter = Gemma4AssistantDraftModel(config: drafterCfg)
        eval(target, drafter)

        // Deterministic random prompt — we only care about parity, not content.
        var rng = SystemRandomNumberGenerator()
        let promptTokens: [Int32] = (0 ..< config.promptLen).map { _ in
            Int32.random(in: 0 ..< 1024, using: &rng)
        }

        let baseline = runBaselineGreedy(
            target: target,
            promptTokens: promptTokens,
            maxTokens: config.maxTokens
        )
        let mtp = try await runMTPGreedy(
            target: target,
            drafter: drafter,
            promptTokens: promptTokens,
            maxTokens: config.maxTokens,
            blockSize: config.blockSize
        )

        #expect(
            baseline == mtp,
            """
            Baseline vs MTP token divergence
              blockSize=\(config.blockSize)
              promptLen=\(config.promptLen)
              maxTokens=\(config.maxTokens)
              prompt=\(promptTokens)
              baseline=\(baseline)
              mtp=\(mtp)
            """
        )
    }

    @Test(arguments: [
        (blockSize: 2, promptLen: 8, maxTokens: 12),
        (blockSize: 3, promptLen: 8, maxTokens: 12),
        (blockSize: 4, promptLen: 8, maxTokens: 12),
    ])
    func centroid_drafter_parity(
        config: (blockSize: Int, promptLen: Int, maxTokens: Int)
    ) async throws {
        // Same invariant, but exercising the MaskedEmbedder path.
        let targetCfg = try e2bStyleTargetConfig()
        let target = Gemma4TextModel(targetCfg)
        let drafterCfg = try e2bStyleDrafterConfig(useOrderedEmbeddings: true)
        let drafter = Gemma4AssistantDraftModel(config: drafterCfg)
        eval(target, drafter)

        var rng = SystemRandomNumberGenerator()
        let promptTokens: [Int32] = (0 ..< config.promptLen).map { _ in
            Int32.random(in: 0 ..< 1024, using: &rng)
        }

        let baseline = runBaselineGreedy(
            target: target,
            promptTokens: promptTokens,
            maxTokens: config.maxTokens
        )
        let mtp = try await runMTPGreedy(
            target: target,
            drafter: drafter,
            promptTokens: promptTokens,
            maxTokens: config.maxTokens,
            blockSize: config.blockSize
        )

        #expect(
            baseline == mtp,
            """
            Centroid-head baseline vs MTP divergence
              blockSize=\(config.blockSize)
              promptLen=\(config.promptLen)
              maxTokens=\(config.maxTokens)
              prompt=\(promptTokens)
              baseline=\(baseline)
              mtp=\(mtp)
            """
        )
    }
}
