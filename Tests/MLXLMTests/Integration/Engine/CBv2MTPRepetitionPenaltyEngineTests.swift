import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Repetition-penalty rows through the real EngineV2 with MTP: the committed
/// tokens equal the MTP-off engine's (T 0 and seeded T > 0, alone and with
/// presence/frequency penalties), the rows actually speculate, a rejected
/// draft never enters a later position's window, typical acceptance does not
/// apply to them, and without the opt-in they stay target-only. Miniature
/// Gemma 4 target with a scripted drafter; synthetic weights, no model load.
@Suite("MTP repetition-penalty rows", .tags(.integration), .serialized)
struct CBv2MTPRepetitionPenaltyEngineTests {
    private let vocabSize = 16
    private let hiddenSize = 64
    private let k = 2

    private func makeTarget(seed: UInt64 = 0x7E9A) throws -> Gemma4TextModel {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
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
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(config)
        stabilizeCBv2MTPGreedyCycleTarget(target)
        eval(target)
        return target
    }

    /// History-independent logits with a strict order (identical embeddings,
    /// head row t = (1 - t / vocab) · ones): unpenalized greedy repeats one
    /// token, so a repetition penalty of 2.5 or more changes every answer
    /// whichever sign the logits take.
    private func makeRankedTarget() throws -> Gemma4TextModel {
        let target = try makeTarget()
        let parameters = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        let embedding = try #require(parameters["model.embed_tokens.weight"])
        let head = try #require(parameters["lm_head.weight"])
        let headRows = (0 ..< vocabSize).flatMap { token in
            [Float](repeating: 1 - Float(token) / Float(vocabSize), count: hiddenSize)
        }
        target.update(
            parameters: ModuleParameters.unflattened([
                "model.embed_tokens.weight": MLXArray.ones(embedding.shape, dtype: embedding.dtype),
                "lm_head.weight": MLXArray(headRows, head.shape).asType(head.dtype),
            ]))
        eval(target)
        return target
    }

    private func run(
        target: Gemma4TextModel, script: [Int]?, prompt: [Int],
        acceptance: CBv2MTPAcceptance = .exact,
        verificationMode: CBv2MTPVerificationMode = .serialTarget,
        penaltySpeculation: Bool = true, repetitionPenaltySpeculation: Bool = true,
        requests: [CBv2Request]
    ) async throws -> (tokens: [[Int]], metrics: CBv2MTPMetrics?) {
        let drafter = script.map {
            CBv2TypicalScriptedDrafter(
                script: $0, promptLength: prompt.count, offset: 0, vocabSize: vocabSize,
                target: target)
        }
        let kinds = target.cbv2LayerKinds
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
                fixedDraftTokens: k, verificationMode: verificationMode,
                maxAutomaticRectangularTokens: 8, acceptance: acceptance,
                penaltySpeculation: penaltySpeculation,
                repetitionPenaltySpeculation: repetitionPenaltySpeculation))
        var tokens: [[Int]] = []
        for request in requests {
            tokens.append(await cbv2SchedCollect(try engine.submit(request)).tokens)
        }
        let metrics = engine.mtpMetricsSnapshot()
        await engine.shutdown()
        return (tokens, metrics)
    }

    private func request(
        _ id: UInt64, _ prompt: [Int], _ maxTokens: Int, _ sampling: CBv2SamplingParams
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt, sampling: sampling, maxTokens: maxTokens)
    }

    /// Repetition penalties that reorder the ranked target's tokens, alone
    /// and with presence/frequency penalties, windows shorter and longer than
    /// one round.
    private let samplings: [(String, CBv2SamplingParams)] = [
        ("t0-repetition", .init(temperature: 0, repetitionPenalty: 3)),
        ("t0-window3", .init(temperature: 0, repetitionPenalty: 2.5, repetitionContextSize: 3)),
        (
            "t0-repetition-presence-frequency",
            .init(
                temperature: 0, repetitionPenalty: 3, repetitionContextSize: 6,
                frequencyPenalty: 0.7, presencePenalty: 0.2)
        ),
        (
            "t1-repetition-card",
            .init(
                temperature: 1, topP: 0.95, topK: 20, repetitionPenalty: 2.5,
                presencePenalty: 1.5, seed: 1234)
        ),
        (
            "t07-window2",
            .init(temperature: 0.7, repetitionPenalty: 3, repetitionContextSize: 2, seed: 77)
        ),
    ]

    /// The MTP engine commits the MTP-off engine's tokens for repetition rows
    /// with an oracle draft (kept) and a wrong draft (rejected and rolled
    /// back), in serial and rectangular verification, and the rows speculate.
    @Test func repetitionRowsMatchTheMTPOffEngine() async throws {
        let target = try makeRankedTarget()
        let prompt = [3, 7, 11, 9, 13]
        let maxTokens = 24
        for (name, sampling) in samplings {
            let req = request(1, prompt, maxTokens, sampling)
            let expected = try await run(
                target: target, script: nil, prompt: prompt, requests: [req]
            ).tokens[0]
            try #require(expected.count == maxTokens, "\(name)")
            if sampling.temperature == 0, !CBv2MTPVerifyPenalties.applies(sampling) {
                var plain = sampling
                plain.repetitionPenalty = 1
                let unpenalized = try await run(
                    target: target, script: nil, prompt: prompt,
                    requests: [request(1, prompt, maxTokens, plain)])
                #expect(unpenalized.tokens[0] != expected, "\(name): the penalty must matter")
            }
            for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
                let oracle = try await run(
                    target: target, script: expected + [Int](repeating: 0, count: 2 * k),
                    prompt: prompt, verificationMode: mode, requests: [req])
                #expect(oracle.tokens[0] == expected, "\(name) \(mode) oracle")
                let oracleMetrics = try #require(oracle.metrics)
                #expect(oracleMetrics.rounds > 0, "\(name) \(mode): the row must speculate")
                #expect(oracleMetrics.acceptedTokens > 0, "\(name) \(mode)")
                if sampling.temperature == 0 {
                    #expect(
                        oracleMetrics.acceptedTokens == oracleMetrics.draftedTokens,
                        "\(name) \(mode): greedy oracle drafts are all kept")
                }

                let wrong = try await run(
                    target: target,
                    script: (expected + [Int](repeating: 0, count: 2 * k)).map {
                        ($0 + 1) % vocabSize
                    },
                    prompt: prompt, verificationMode: mode, requests: [req])
                #expect(wrong.tokens[0] == expected, "\(name) \(mode) wrong drafts")
                #expect(try #require(wrong.metrics).rounds > 0)
            }
        }
    }

    /// Rejections mid-round (the oracle stream with every third draft
    /// changed): the rejected tokens are rolled back and never enter a later
    /// position's repetition window, so the stream is the MTP-off one.
    @Test func rejectedDraftsLeaveNoTraceInTheWindow() async throws {
        let target = try makeRankedTarget()
        let prompt = [2, 6, 10, 14]
        let maxTokens = 30
        for (name, sampling) in samplings {
            let req = request(3, prompt, maxTokens, sampling)
            let expected = try await run(
                target: target, script: nil, prompt: prompt, requests: [req]
            ).tokens[0]
            let script = (expected + [Int](repeating: 0, count: 2 * k)).enumerated().map {
                $0.offset % 3 == 1 ? ($0.element + 5) % vocabSize : $0.element
            }
            for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
                let mixed = try await run(
                    target: target, script: script, prompt: prompt, verificationMode: mode,
                    requests: [req])
                #expect(mixed.tokens[0] == expected, "\(name) \(mode)")
                let metrics = try #require(mixed.metrics)
                #expect(metrics.rounds > 0, "\(name) \(mode)")
                #expect(metrics.acceptedTokens < metrics.draftedTokens, "\(name) \(mode)")
            }
        }
    }

    /// Repetition, plain and presence rows interleaved through one engine,
    /// including a finished request's id reused with other sampling.
    @Test func sequentialRepetitionRequestsShareOneEngineExactly() async throws {
        let target = try makeRankedTarget()
        let prompt = [2, 4, 8, 6, 12]
        let maxTokens = 24
        let reqs = [
            request(5, prompt, maxTokens, .init(temperature: 0, repetitionPenalty: 3)),
            request(6, prompt, maxTokens, .init(temperature: 0)),
            request(5, prompt, maxTokens, .init(temperature: 0, presencePenalty: 1.5)),
            request(
                7, prompt, maxTokens,
                .init(temperature: 0, repetitionPenalty: 2.5, repetitionContextSize: 4)),
        ]
        var expected: [[Int]] = []
        for req in reqs {
            expected.append(
                try await run(target: target, script: nil, prompt: prompt, requests: [req])
                    .tokens[0])
        }
        let script = expected[0] + [Int](repeating: 0, count: 2 * k)
        let mtp = try await run(target: target, script: script, prompt: prompt, requests: reqs)
        #expect(mtp.tokens == expected)
        #expect(try #require(mtp.metrics).rounds > 0)
    }

    /// Repetition rows accept exactly even when typical acceptance is
    /// configured: a likely-but-different draft is never kept for them.
    @Test func repetitionRowsIgnoreTypicalAcceptance() async throws {
        let target = try makeRankedTarget()
        let prompt = [5, 9, 13, 1]
        let maxTokens = 24
        let typical = CBv2MTPAcceptance.typical(delta: CBv2MTPAcceptance.defaultTypicalDelta)
        let req = request(
            8, prompt, maxTokens,
            .init(temperature: 1, topP: 0.95, topK: 20, repetitionPenalty: 2.5, seed: 99))
        let expected = try await run(
            target: target, script: nil, prompt: prompt, requests: [req]
        ).tokens[0]
        for offset in [0, 1, 2] {
            let script = (expected + [Int](repeating: 0, count: 2 * k)).map {
                ($0 + offset) % vocabSize
            }
            let lossy = try await run(
                target: target, script: script, prompt: prompt, acceptance: typical,
                requests: [req])
            #expect(lossy.tokens[0] == expected, "offset \(offset)")
            #expect(try #require(lossy.metrics).rounds > 0)
        }
    }

    /// Without the repetition opt-in (what
    /// `DARKBLOOM_MTP_REPETITION_PENALTY_ROWS=0` forces) repetition rows stay
    /// target-only and unchanged, as do rows that also carry presence or
    /// frequency without that opt-in; rows without penalties still speculate.
    @Test func optOutKeepsRepetitionRowsTargetOnly() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 9, 13]
        let maxTokens = 20
        let cases: [(String, CBv2SamplingParams, Bool, Bool)] = [
            ("repetition, no opt-in", .init(temperature: 0, repetitionPenalty: 3), true, false),
            (
                "repetition and presence, presence opt-in off",
                .init(temperature: 0, repetitionPenalty: 3, presencePenalty: 1.5), false, true
            ),
        ]
        for (name, sampling, penaltyOptIn, repetitionOptIn) in cases {
            let req = request(1, prompt, maxTokens, sampling)
            let expected = try await run(
                target: target, script: nil, prompt: prompt, requests: [req]
            ).tokens[0]
            let off = try await run(
                target: target, script: expected + [0, 0, 0, 0], prompt: prompt,
                penaltySpeculation: penaltyOptIn, repetitionPenaltySpeculation: repetitionOptIn,
                requests: [req])
            #expect(off.tokens[0] == expected, "\(name)")
            #expect((off.metrics?.rounds ?? 0) == 0, "\(name)")
        }

        let plain = request(2, prompt, maxTokens, .init(temperature: 0))
        let plainExpected = cbv2MTPExpectedGreedyCycle(
            after: prompt.last!, count: maxTokens, vocabularySize: vocabSize)
        let plainOff = try await run(
            target: target, script: plainExpected + [0, 0, 0, 0], prompt: prompt,
            repetitionPenaltySpeculation: false, requests: [plain])
        #expect(plainOff.tokens[0] == plainExpected)
        #expect(try #require(plainOff.metrics).rounds > 0)
    }

    /// The opt-in is per engine, off by default, and the process switch can
    /// only turn it off.
    @Test func repetitionOptInDefaultsOff() {
        #expect(!CBv2MTPConfig().repetitionPenaltySpeculation)
        #expect(
            CBv2MTPConfig(repetitionPenaltySpeculation: true).effectiveRepetitionPenaltySpeculation
                == CBv2MTPConfig.envRepetitionPenaltySpeculation)
        #expect(
            !CBv2MTPConfig(repetitionPenaltySpeculation: false)
                .effectiveRepetitionPenaltySpeculation)
        #expect(CBv2DefaultSampler().supportsMTPRepetitionPenaltyVerification)
    }
}
