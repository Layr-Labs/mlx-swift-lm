import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// `CBv2MTPConfig.maxDraftPromptTokens` through the real EngineV2 with a
/// miniature Gemma 4 target and a scripted drafter. The limit is a plan-time
/// gate: a plan whose rows are all over it runs at depth zero on the chained
/// path, and a row over it beside a row under it drafts with the cohort.
/// Every row's tokens equal the MTP-off engine's. Synthetic weights, no
/// model load.
@Suite("MTP draft prompt-length limit", .tags(.integration), .serialized)
struct CBv2MTPDraftPromptLimitTests {
    private let vocabSize = 256
    private let k = 2
    private let limit = 8
    private let maxTokens = 24

    private func makeTarget() throws -> Gemma4TextModel {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "full_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 16,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": false,
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(0x7E9A)
        let target = Gemma4TextModel(config)
        stabilizeCBv2MTPGreedyCycleTarget(target)
        eval(target)
        return target
    }

    /// A prompt of `count` tokens ending in token 23, so every prompt has the
    /// same greedy continuation and the same oracle drafts.
    private func prompt(_ count: Int) -> [Int] {
        (0 ..< count - 1).map { ($0 * 7 + 3) % vocabSize } + [23]
    }

    private struct Served {
        var tokens: [[Int]]
        var rounds: Int
        var proposedTokens: Int
        var promptLimitClamps: Int
        var chainedSteps: Int
    }

    /// Serve `prompts` together on one engine. `drafts` nil is the MTP-off
    /// engine; otherwise an oracle drafter (the target's own continuation).
    private func serve(
        target: Gemma4TextModel, prompts: [[Int]], drafts: Bool, limit: Int?,
        mode: CBv2MTPVerificationMode = .rectangular
    ) async throws -> Served {
        let kinds = target.cbv2LayerKinds
        let script = cbv2MTPExpectedGreedyCycle(
            after: 23, count: maxTokens + 2 * k, vocabularySize: vocabSize)
        let drafter =
            drafts
            ? CBv2TypicalScriptedDrafter(
                script: script, promptLength: prompts[0].count, offset: 0,
                vocabSize: vocabSize, target: target) : nil
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 41),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: k, maxSpeculativeBatch: 4,
                fixedDraftTokens: k, verificationMode: mode,
                maxAutomaticRectangularTokens: 8, acceptance: .exact,
                maxDraftPromptTokens: limit))
        let streams = try prompts.enumerated().map { index, prompt in
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(UInt64(index + 1)), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: maxTokens))
        }
        var tokens: [[Int]] = []
        for stream in streams {
            tokens.append(await cbv2SchedCollect(stream).tokens)
        }
        let metrics = engine.mtpMetricsSnapshot()
        let chainedSteps = engine.chainedStepCount
        await engine.shutdown()
        return Served(
            tokens: tokens, rounds: metrics?.rounds ?? 0,
            proposedTokens: metrics?.proposedTokens ?? 0,
            promptLimitClamps: metrics?.controllerFallbacks["prompt_limit", default: 0] ?? 0,
            chainedSteps: chainedSteps)
    }

    /// A lone row whose prompt holds the limit drafts; one token more and it
    /// decodes without a round. Both equal the MTP-off engine's tokens.
    @Test(arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func aPromptOverTheLimitDecodesWithoutDrafting(_ mode: CBv2MTPVerificationMode)
        async throws
    {
        let target = try makeTarget()
        for count in [limit, limit + 1] {
            let prompts = [prompt(count)]
            let off = try await serve(target: target, prompts: prompts, drafts: false, limit: nil)
            try #require(off.tokens[0].count == maxTokens)
            let on = try await serve(
                target: target, prompts: prompts, drafts: true, limit: limit, mode: mode)
            #expect(on.tokens == off.tokens, "\(count)-token prompt, \(mode)")
            if count > limit {
                #expect(on.rounds == 0, "\(count)-token prompt drafts no round, \(mode)")
            } else {
                #expect(on.rounds > 0, "\(count)-token prompt drafts, \(mode)")
            }
        }
    }

    /// Without a limit, or with one below one (the integrator's off state),
    /// the gate never acts: the longer prompt drafts alone and beside a
    /// shorter one, and no plan is clamped for the prompt limit.
    @Test(arguments: [nil, 0, -1] as [Int?])
    func noLimitLetsEveryPromptDraft(_ configured: Int?) async throws {
        let target = try makeTarget()
        for prompts in [[prompt(limit + 1)], [prompt(limit + 1), prompt(limit + 4)]] {
            let off = try await serve(target: target, prompts: prompts, drafts: false, limit: nil)
            let on = try await serve(
                target: target, prompts: prompts, drafts: true, limit: configured)
            #expect(on.tokens == off.tokens, "\(prompts.count) rows")
            #expect(on.rounds > 0, "\(prompts.count) rows")
            #expect(on.promptLimitClamps == 0, "\(prompts.count) rows")
        }
    }

    /// A row over the limit decoding beside a row under it: each row's tokens
    /// equal the same rows on the MTP-off engine.
    @Test func rowsOverAndUnderTheLimitTogetherMatchTheMTPOffEngine() async throws {
        let target = try makeTarget()
        let prompts = [prompt(limit), prompt(limit + 4)]
        let off = try await serve(target: target, prompts: prompts, drafts: false, limit: nil)
        let on = try await serve(target: target, prompts: prompts, drafts: true, limit: limit)
        #expect(on.tokens == off.tokens)
    }

    /// A lone row over the limit: the plan is held at depth zero for the
    /// prompt limit (not excluded from MTP for the request's lifetime), the
    /// step stays on the chained decode path, and the tokens are the MTP-off
    /// engine's.
    @Test(arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func aLoneRowOverTheLimitHoldsThePlanAtDepthZero(_ mode: CBv2MTPVerificationMode)
        async throws
    {
        let target = try makeTarget()
        let prompts = [prompt(limit + 1)]
        let off = try await serve(target: target, prompts: prompts, drafts: false, limit: nil)
        let on = try await serve(
            target: target, prompts: prompts, drafts: true, limit: limit, mode: mode)
        #expect(on.tokens == off.tokens, "\(mode)")
        #expect(on.rounds == 0, "\(mode)")
        #expect(on.promptLimitClamps > 0, "\(mode)")
        #expect(on.chainedSteps > 0, "\(mode)")
    }

    /// Two rows, both over the limit: the plan is held at depth zero and
    /// neither row drafts.
    @Test func rowsAllOverTheLimitHoldThePlanAtDepthZero() async throws {
        let target = try makeTarget()
        let prompts = [prompt(limit + 1), prompt(limit + 4)]
        let off = try await serve(target: target, prompts: prompts, drafts: false, limit: nil)
        let on = try await serve(target: target, prompts: prompts, drafts: true, limit: limit)
        #expect(on.tokens == off.tokens)
        #expect(on.rounds == 0)
        #expect(on.promptLimitClamps > 0)
    }

    /// A row over the limit beside a row under it follows the cohort's depth
    /// (this table drafts at two rows), so together they propose more drafts
    /// than the row under the limit does alone. Each row's tokens equal the
    /// MTP-off engine's.
    @Test func aRowOverTheLimitDraftsBesideARowUnderIt() async throws {
        let target = try makeTarget()
        let short = [prompt(limit)]
        let pair = [prompt(limit), prompt(limit + 4)]
        let shortOff = try await serve(target: target, prompts: short, drafts: false, limit: nil)
        let shortOn = try await serve(target: target, prompts: short, drafts: true, limit: limit)
        let pairOff = try await serve(target: target, prompts: pair, drafts: false, limit: nil)
        let pairOn = try await serve(target: target, prompts: pair, drafts: true, limit: limit)
        #expect(shortOn.tokens == shortOff.tokens)
        #expect(pairOn.tokens == pairOff.tokens)
        try #require(shortOn.proposedTokens > 0)
        #expect(pairOn.proposedTokens > shortOn.proposedTokens)
    }
}
