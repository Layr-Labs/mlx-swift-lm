// CBv2MTPStopStringEngineTests.swift
//
// Rows with stop strings through the real EngineV2 with MTP: under the
// stop-string opt-in they draft, and their text, tokens and finish equal
// the MTP-off engine's (the row without speculation, truncated at the first
// match); without the opt-in they stay target-only.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// "<id>" per token: distinct ids never render as substrings of each other.
private struct CBv2MTPStopTokenizer: Tokenizer {
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "<\($0)>" }.joined()
    }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}

private func rendered(_ tokens: [Int]) -> String {
    tokens.map { "<\($0)>" }.joined()
}

@Suite("CBv2MTPStopStringEngine", .serialized)
struct CBv2MTPStopStringEngineTests {
    private let vocabSize = 256
    private let hiddenSize = 64
    private let k = 2

    private func makeTarget() throws -> Gemma4TextModel {
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
        MLXRandom.seed(0x7E9A)
        let target = Gemma4TextModel(config)
        stabilizeCBv2MTPGreedyCycleTarget(target)
        eval(target)
        return target
    }

    private struct Served {
        var tokens: [Int] = []
        var text = ""
        var finish: CBv2FinishReason?
        var metrics: CBv2MTPMetrics?
    }

    private func serve(
        target: Gemma4TextModel, script: [Int]?, prompt: [Int], request: CBv2Request,
        verificationMode: CBv2MTPVerificationMode = .serialTarget,
        stopStringSpeculation: Bool = true, penaltySpeculation: Bool = true
    ) async throws -> Served {
        let kinds = target.cbv2LayerKinds
        let drafter = script.map {
            CBv2TypicalScriptedDrafter(
                script: $0, promptLength: prompt.count, offset: 0, vocabSize: vocabSize,
                target: target)
        }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 41),
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2MTPStopTokenizer()),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: k, maxSpeculativeBatch: 4,
                fixedDraftTokens: k, verificationMode: verificationMode,
                maxAutomaticRectangularTokens: 8, acceptance: .exact,
                penaltySpeculation: penaltySpeculation,
                stopStringSpeculation: stopStringSpeculation))
        let collected = await cbv2SchedCollect(try engine.submit(request))
        let metrics = engine.mtpMetricsSnapshot()
        await engine.shutdown()
        return Served(
            tokens: collected.tokens, text: collected.text, finish: collected.finishReason,
            metrics: metrics)
    }

    private func request(
        _ id: UInt64, _ prompt: [Int], _ maxTokens: Int, _ sampling: CBv2SamplingParams,
        stops: [String]
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt, sampling: sampling,
            maxTokens: maxTokens, stopStrings: stops)
    }

    /// Parity: a stop-string row under the opt-in drafts, and its text,
    /// tokens and finish equal the MTP-off engine's; the text is the row
    /// without a stop string, truncated at the first match. Single-token and
    /// two-token (round-spanning) matches, a never-matching stop string,
    /// greedy and the Gemma card's sampled shape with a presence penalty;
    /// serial and rectangular verification.
    @Test func stopStringRowsMatchTheMTPOffEngine() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 19, 23]
        let maxTokens = 24
        let cycle = cbv2MTPExpectedGreedyCycle(
            after: prompt.last!, count: maxTokens + 2 * k, vocabularySize: vocabSize)
        let samplings: [(String, CBv2SamplingParams)] = [
            ("t0", .init(temperature: 0)),
            (
                "gemma-card+presence",
                .init(temperature: 1, topP: 0.95, topK: 64, presencePenalty: 1.5, seed: 77)
            ),
        ]
        for (name, sampling) in samplings {
            let plain = try await serve(
                target: target, script: nil, prompt: prompt,
                request: request(1, prompt, maxTokens, sampling, stops: []))
            try #require(plain.tokens.count == maxTokens, "\(name)")
            let cases: [(String, [String], Int?)] = [
                ("one-token", [rendered([plain.tokens[9]])], 9),
                ("two-token", [rendered(Array(plain.tokens[12 ... 13]))], 12),
                ("never", ["<never>"], nil),
            ]
            // The same request id everywhere: keyed sampling is (seed, id, step).
            for (label, stops, matchAt) in cases {
                let req = request(1, prompt, maxTokens, sampling, stops: stops)
                let off = try await serve(
                    target: target, script: nil, prompt: prompt, request: req)
                if let matchAt {
                    #expect(off.finish == .stop, "\(name) \(label)")
                    #expect(
                        off.text == rendered(Array(plain.tokens[..<matchAt])), "\(name) \(label)")
                } else {
                    #expect(off.tokens == plain.tokens && off.finish == .length)
                }
                for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
                    // Oracle drafts (accepted) and wrong drafts (rejected).
                    for script in [
                        plain.tokens + [0, 0, 0, 0], cycle.map { ($0 + 1) % vocabSize },
                    ] {
                        let on = try await serve(
                            target: target, script: script, prompt: prompt, request: req,
                            verificationMode: mode)
                        let tag = "\(name) \(label) \(mode)"
                        #expect(on.text == off.text, "\(tag) text")
                        #expect(on.tokens == off.tokens, "\(tag) tokens")
                        #expect(on.finish == off.finish, "\(tag) finish")
                        #expect((on.metrics?.rounds ?? 0) > 0, "\(tag): the row speculates")
                    }
                }
            }
        }
    }

    /// Without the opt-in (what `DARKBLOOM_MTP_STOP_STRING_ROWS=0` forces)
    /// a stop-string row stays target-only, with unchanged output.
    @Test func optOutKeepsStopStringRowsTargetOnly() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 19, 23]
        let maxTokens = 16
        let plain = try await serve(
            target: target, script: nil, prompt: prompt,
            request: request(1, prompt, maxTokens, .init(temperature: 0), stops: []))
        let req = request(
            1, prompt, maxTokens, .init(temperature: 0), stops: [rendered([plain.tokens[7]])])
        let off = try await serve(target: target, script: nil, prompt: prompt, request: req)
        let optOut = try await serve(
            target: target, script: plain.tokens + [0, 0, 0, 0], prompt: prompt, request: req,
            stopStringSpeculation: false)
        #expect(optOut.text == off.text && optOut.tokens == off.tokens)
        #expect(optOut.finish == .stop)
        #expect((optOut.metrics?.rounds ?? 0) == 0)
    }

    @Test func stopStringOptInIsOffByDefaultAndKillSwitched() {
        #expect(!CBv2MTPConfig().stopStringSpeculation)
        #expect(
            CBv2MTPConfig(stopStringSpeculation: true).effectiveStopStringSpeculation
                == CBv2MTPConfig.envStopStringSpeculation)
        #expect(!CBv2MTPConfig(stopStringSpeculation: false).effectiveStopStringSpeculation)
    }
}
