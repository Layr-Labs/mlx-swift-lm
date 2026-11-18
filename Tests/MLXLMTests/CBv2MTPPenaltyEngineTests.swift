// CBv2MTPPenaltyEngineTests.swift
//
// Presence/frequency penalty rows through the real EngineV2 with MTP: the
// committed tokens equal the MTP-off engine's (T 0 and seeded T > 0), the
// rows actually speculate, typical acceptance does not apply to them, and
// the kill switch keeps them target-only.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("CBv2MTPPenaltyEngine", .serialized)
struct CBv2MTPPenaltyEngineTests {
    /// A 16-token greedy cycle repeats within one answer, so penalties bite.
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
    /// token, so presence and frequency penalties change every answer.
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

    private func makeEngine(
        target: Gemma4TextModel, drafter: (any CBv2MTPDrafter)?,
        acceptance: CBv2MTPAcceptance, verificationMode: CBv2MTPVerificationMode,
        penaltySpeculation: Bool
    ) -> EngineV2 {
        let kinds = target.cbv2LayerKinds
        return EngineV2(
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
                penaltySpeculation: penaltySpeculation))
    }

    private func run(
        target: Gemma4TextModel, script: [Int]?, prompt: [Int],
        acceptance: CBv2MTPAcceptance = .exact,
        verificationMode: CBv2MTPVerificationMode = .serialTarget,
        penaltySpeculation: Bool = true, requests: [CBv2Request]
    ) async throws -> (tokens: [[Int]], metrics: CBv2MTPMetrics?) {
        let drafter = script.map {
            CBv2TypicalScriptedDrafter(
                script: $0, promptLength: prompt.count, offset: 0, vocabSize: vocabSize,
                target: target)
        }
        let engine = makeEngine(
            target: target, drafter: drafter, acceptance: acceptance,
            verificationMode: verificationMode, penaltySpeculation: penaltySpeculation)
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

    /// Penalties that reorder the ranked target's tokens.
    private let samplings: [(String, CBv2SamplingParams)] = [
        ("t0-presence", .init(temperature: 0, presencePenalty: 1.5)),
        ("t0-frequency", .init(temperature: 0, frequencyPenalty: 0.7, presencePenalty: 0.2)),
        (
            "t1-card",
            .init(temperature: 1, topP: 0.95, topK: 20, presencePenalty: 1.5, seed: 1234)
        ),
        ("t07-frequency", .init(temperature: 0.7, frequencyPenalty: 4, seed: 77)),
    ]

    /// (a) The MTP engine commits the MTP-off engine's tokens for penalty
    /// rows, with an oracle draft (accepted) and a wrong draft (rejected),
    /// in serial and rectangular verification, and the rows speculate.
    @Test func penaltyRowsMatchTheMTPOffEngine() async throws {
        let target = try makeRankedTarget()
        let prompt = [3, 7, 11, 9, 13]
        let maxTokens = 24
        for (name, sampling) in samplings {
            let req = request(1, prompt, maxTokens, sampling)
            let baseline = try await run(
                target: target, script: nil, prompt: prompt, requests: [req])
            let expected = baseline.tokens[0]
            try #require(expected.count == maxTokens, "\(name)")
            if sampling.temperature == 0 {
                var plain = sampling
                plain.presencePenalty = 0
                plain.frequencyPenalty = 0
                let unpenalized = try await run(
                    target: target, script: nil, prompt: prompt,
                    requests: [request(1, prompt, maxTokens, plain)])
                #expect(unpenalized.tokens[0] != expected, "\(name): penalties must matter")
            }
            for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
                // Oracle drafts: the MTP-off stream itself.
                let oracle = try await run(
                    target: target, script: expected + [Int](repeating: 0, count: 2 * k),
                    prompt: prompt, verificationMode: mode, requests: [req])
                #expect(oracle.tokens[0] == expected, "\(name) \(mode) oracle")
                let oracleMetrics = try #require(oracle.metrics)
                #expect(oracleMetrics.rounds > 0, "\(name) \(mode): penalty row must speculate")
                #expect(oracleMetrics.acceptedTokens > 0, "\(name) \(mode)")
                if sampling.temperature == 0 {
                    #expect(
                        oracleMetrics.acceptedTokens == oracleMetrics.draftedTokens,
                        "\(name) \(mode): greedy oracle drafts are all kept")
                }

                // Wrong drafts: every draft is rejected, stream unchanged.
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

    /// Two penalty rows and a plain row interleaved through one engine: the
    /// per-request count cache and row reconfiguration stay exact.
    @Test func sequentialPenaltyRequestsShareOneEngineExactly() async throws {
        let target = try makeRankedTarget()
        let prompt = [2, 4, 8, 6, 12]
        let maxTokens = 24
        let reqs = [
            request(5, prompt, maxTokens, .init(temperature: 0, presencePenalty: 1.5)),
            request(6, prompt, maxTokens, .init(temperature: 0)),
            // The id of a finished request reused with different sampling.
            request(5, prompt, maxTokens, .init(temperature: 0, frequencyPenalty: 0.7)),
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

    /// (c) Penalty rows accept exactly even when typical acceptance is
    /// configured: a likely-but-different draft is never kept for them.
    @Test func penaltyRowsIgnoreTypicalAcceptance() async throws {
        let target = try makeRankedTarget()
        let prompt = [5, 9, 13, 1]
        let maxTokens = 24
        let typical = CBv2MTPAcceptance.typical(delta: CBv2MTPAcceptance.defaultTypicalDelta)
        let req = request(
            8, prompt, maxTokens,
            .init(temperature: 1, topP: 0.95, topK: 20, presencePenalty: 1.5, seed: 99))
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

    /// (c) Without the per-engine opt-in (what `DARKBLOOM_MTP_PENALTY_ROWS=0`
    /// forces through `effectivePenaltySpeculation`) penalty rows stay
    /// target-only and unchanged; rows without penalties still speculate.
    @Test func optOutKeepsPenaltyRowsTargetOnly() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 9, 13]
        let maxTokens = 20
        let penalized = request(1, prompt, maxTokens, .init(temperature: 0, presencePenalty: 100))
        let expected = try await run(
            target: target, script: nil, prompt: prompt, requests: [penalized]
        ).tokens[0]
        let off = try await run(
            target: target, script: expected + [0, 0, 0, 0], prompt: prompt,
            penaltySpeculation: false, requests: [penalized])
        #expect(off.tokens[0] == expected)
        #expect((off.metrics?.rounds ?? 0) == 0)

        let plain = request(2, prompt, maxTokens, .init(temperature: 0))
        let plainExpected = cbv2MTPExpectedGreedyCycle(
            after: prompt.last!, count: maxTokens, vocabularySize: vocabSize)
        let plainOff = try await run(
            target: target, script: plainExpected + [0, 0, 0, 0], prompt: prompt,
            penaltySpeculation: false, requests: [plain])
        #expect(plainOff.tokens[0] == plainExpected)
        #expect(try #require(plainOff.metrics).rounds > 0)
    }

    /// (c) The gate's truth table: penalty rows need the switch, target-prefix
    /// support, penalty-aware verification, no constraint and finite values.
    @Test func penaltyGateTruthTable() {
        for switchOn in [false, true] {
            for targetPrefix in [false, true] {
                for verification in [false, true] {
                    for constrained in [false, true] {
                        for finite in [false, true] {
                            let admitted = EngineLoopV2.mtpPenaltyRowEligible(
                                switchOn: switchOn, targetPrefix: targetPrefix,
                                penaltyVerification: verification, constrained: constrained,
                                finite: finite)
                            #expect(
                                admitted
                                    == (switchOn && targetPrefix && verification && !constrained
                                        && finite))
                        }
                    }
                }
            }
        }
        #expect(CBv2DefaultSampler().supportsMTPPenaltyVerification)
        #expect(!CBv2GreedySampler().supportsMTPPenaltyVerification)
        #expect(!CBv2MTPConfig().penaltySpeculation, "opt-in per engine")
        #expect(
            CBv2MTPConfig(penaltySpeculation: true).effectivePenaltySpeculation
                == CBv2MTPConfig.envPenaltySpeculation)
    }
}
