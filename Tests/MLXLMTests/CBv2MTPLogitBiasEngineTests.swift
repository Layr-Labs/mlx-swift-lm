// CBv2MTPLogitBiasEngineTests.swift
//
// Logit-bias rows through the real EngineV2 with MTP: with the opt-in the
// rows speculate and commit the MTP-off engine's tokens (T 0 and seeded
// T > 0, oracle and wrong drafts, serial and rectangular verification);
// without it they stay target-only, unchanged.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("CBv2MTPLogitBiasEngine", .serialized)
struct CBv2MTPLogitBiasEngineTests {
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

    private func makeEngine(
        target: Gemma4TextModel, drafter: (any CBv2MTPDrafter)?,
        verificationMode: CBv2MTPVerificationMode, logitBiasSpeculation: Bool
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
                maxAutomaticRectangularTokens: 8,
                logitBiasSpeculation: logitBiasSpeculation))
    }

    private func run(
        target: Gemma4TextModel, script: [Int]?, prompt: [Int],
        verificationMode: CBv2MTPVerificationMode = .serialTarget,
        logitBiasSpeculation: Bool = true, requests: [CBv2Request]
    ) async throws -> (tokens: [[Int]], metrics: CBv2MTPMetrics?) {
        let drafter = script.map {
            CBv2TypicalScriptedDrafter(
                script: $0, promptLength: prompt.count, offset: 0, vocabSize: vocabSize,
                target: target)
        }
        let engine = makeEngine(
            target: target, drafter: drafter, verificationMode: verificationMode,
            logitBiasSpeculation: logitBiasSpeculation)
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

    /// The greedy cycle's first token after the prompt, banned so the bias bites.
    private func biasedSamplings(prompt: [Int]) -> [(String, CBv2SamplingParams)] {
        let cycle = cbv2MTPExpectedGreedyCycle(
            after: prompt.last!, count: 4, vocabularySize: vocabSize)
        return [
            ("t0-ban", .init(temperature: 0, logitBias: [cycle[0]: -100])),
            (
                "t0-boost",
                .init(temperature: 0, logitBias: [cycle[1]: -100, (cycle[2] + 3) % vocabSize: 4])
            ),
            (
                "t1-ban",
                .init(temperature: 1, topP: 0.95, topK: 12, seed: 1234, logitBias: [cycle[0]: -100])
            ),
            ("t07-zero", .init(temperature: 0.7, seed: 77, logitBias: [cycle[0]: 0])),
        ]
    }

    /// The MTP engine commits the MTP-off engine's tokens for bias rows, with
    /// oracle (accepted) and wrong (rejected) drafts, in serial and
    /// rectangular verification, and the rows speculate.
    @Test func biasRowsMatchTheMTPOffEngine() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 9, 13]
        let maxTokens = 24
        for (name, sampling) in biasedSamplings(prompt: prompt) {
            let req = request(1, prompt, maxTokens, sampling)
            let baseline = try await run(
                target: target, script: nil, prompt: prompt, requests: [req])
            let expected = baseline.tokens[0]
            try #require(expected.count == maxTokens, "\(name)")
            if sampling.temperature == 0 {
                let unbiased = cbv2MTPExpectedGreedyCycle(
                    after: prompt.last!, count: maxTokens, vocabularySize: vocabSize)
                #expect(unbiased != expected, "\(name): the bias must matter")
            }
            for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
                let oracle = try await run(
                    target: target, script: expected + [Int](repeating: 0, count: 2 * k),
                    prompt: prompt, verificationMode: mode, requests: [req])
                #expect(oracle.tokens[0] == expected, "\(name) \(mode) oracle")
                let oracleMetrics = try #require(oracle.metrics)
                #expect(oracleMetrics.rounds > 0, "\(name) \(mode): bias row must speculate")
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

    /// A biased row and a plain row interleaved through one engine stay exact.
    @Test func biasAndPlainRequestsShareOneEngineExactly() async throws {
        let target = try makeTarget()
        let prompt = [2, 4, 8, 6, 12]
        let maxTokens = 24
        let samplings = biasedSamplings(prompt: prompt)
        let reqs = [
            request(5, prompt, maxTokens, samplings[0].1),
            request(6, prompt, maxTokens, .init(temperature: 0)),
            request(5, prompt, maxTokens, samplings[2].1),
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

    /// Without the per-engine opt-in (what `DARKBLOOM_MTP_LOGIT_BIAS_ROWS=0`
    /// forces through `effectiveLogitBiasSpeculation`) bias rows stay
    /// target-only and unchanged; rows without a bias still speculate.
    @Test func optOutKeepsBiasRowsTargetOnly() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 9, 13]
        let maxTokens = 20
        let biased = request(1, prompt, maxTokens, biasedSamplings(prompt: prompt)[0].1)
        let expected = try await run(
            target: target, script: nil, prompt: prompt, requests: [biased]
        ).tokens[0]
        let off = try await run(
            target: target, script: expected + [0, 0, 0, 0], prompt: prompt,
            logitBiasSpeculation: false, requests: [biased])
        #expect(off.tokens[0] == expected)
        #expect((off.metrics?.rounds ?? 0) == 0)

        let plain = request(2, prompt, maxTokens, .init(temperature: 0))
        let plainExpected = cbv2MTPExpectedGreedyCycle(
            after: prompt.last!, count: maxTokens, vocabularySize: vocabSize)
        let plainOff = try await run(
            target: target, script: plainExpected + [0, 0, 0, 0], prompt: prompt,
            logitBiasSpeculation: false, requests: [plain])
        #expect(plainOff.tokens[0] == plainExpected)
        #expect(try #require(plainOff.metrics).rounds > 0)
    }

    /// The gate's truth table.
    @Test func biasGateTruthTable() {
        for switchOn in [false, true] {
            for targetPrefix in [false, true] {
                for verification in [false, true] {
                    for constrained in [false, true] {
                        for finite in [false, true] {
                            let admitted = EngineLoopV2.mtpLogitBiasRowEligible(
                                switchOn: switchOn, targetPrefix: targetPrefix,
                                biasVerification: verification, constrained: constrained,
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
        #expect(!CBv2MTPConfig().logitBiasSpeculation, "opt-in per engine")
        #expect(
            CBv2MTPConfig(logitBiasSpeculation: true).effectiveLogitBiasSpeculation
                == CBv2MTPConfig.envLogitBiasSpeculation)
    }
}
