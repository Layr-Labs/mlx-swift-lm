// CBv2MTPPenaltyLongRunTests.swift
//
// Presence/frequency penalty rows over long MTP generations: at every draw
// and every verify call, the counts the sampler applies equal a recount of
// the row's emitted tokens (prompt excluded), plus, inside a verify window,
// the drafts before each position. Drafts are wrong at varied positions so
// rounds keep rolling back; the runs cover sampled rows (correction draws),
// stop-string holdback and a stop, a stop token and length caps that end
// inside a window. The committed stream must also equal the MTP-off engine's.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// "<id>" per token, so stop strings are token sequences.
private struct CBv2PenaltyLongRunTokenizer: Tokenizer {
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

@Suite("CBv2MTPPenaltyLongRun", .serialized)
struct CBv2MTPPenaltyLongRunTests {
    private let vocabSize = 32
    private let hiddenSize = 64

    /// History-independent ranked logits (as in `CBv2MTPPenaltyEngineTests`):
    /// only the penalties move the answer, so a count error changes tokens.
    private func makeRankedTarget() throws -> Gemma4TextModel {
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

    private struct Served {
        var tokens: [Int] = []
        var finish: CBv2FinishReason?
        var metrics: CBv2MTPMetrics?
        var observations: [CBv2PenaltyCountObservation] = []
    }

    private func serve(
        target: Gemma4TextModel, script: [Int]?, request: CBv2Request, k: Int,
        verificationMode: CBv2MTPVerificationMode = .serialTarget
    ) async throws -> Served {
        let kinds = target.cbv2LayerKinds
        let drafter = script.map {
            CBv2TypicalScriptedDrafter(
                script: $0, promptLength: request.promptTokens.count, offset: 0,
                vocabSize: vocabSize, target: target)
        }
        final class Log: @unchecked Sendable {
            private let lock = NSLock()
            private var items: [CBv2PenaltyCountObservation] = []
            func append(_ item: CBv2PenaltyCountObservation) {
                lock.withLock { items.append(item) }
            }
            var all: [CBv2PenaltyCountObservation] { lock.withLock { items } }
        }
        let sampler = CBv2DefaultSampler(fallbackSeed: 41)
        let log = Log()
        sampler.penaltyCountObserverForTesting = { log.append($0) }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: sampler,
            detokenizerFactory: CBv2TextDetokenizerFactory(
                tokenizer: CBv2PenaltyLongRunTokenizer()),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: k, maxSpeculativeBatch: 4,
                fixedDraftTokens: k, verificationMode: verificationMode,
                maxAutomaticRectangularTokens: 8, acceptance: .exact,
                penaltySpeculation: true, stopStringSpeculation: true))
        let collected = await cbv2SchedCollect(try engine.submit(request), timeoutSeconds: 900)
        let metrics = engine.mtpMetricsSnapshot()
        await engine.shutdown()
        return Served(
            tokens: collected.tokens, finish: collected.finishReason, metrics: metrics,
            observations: log.all)
    }

    private func recount(_ tokens: some Sequence<Int>) -> [Float] {
        var counts = [Float](repeating: 0, count: vocabSize)
        for token in tokens where token >= 0 && token < vocabSize { counts[token] += 1 }
        return counts
    }

    /// Every observed count vector equals a recount of the emitted stream
    /// (`emitted`: the unstopped MTP-off stream, of which a stopped run is a
    /// prefix): a draw uses the first `n` emitted tokens (n = the counts'
    /// total, never decreasing); a verify window position j uses the engine's
    /// confirmed output, which must be a prefix of the stream, plus the
    /// drafts before j.
    private func expectExactCounts(
        _ served: Served, id: CBv2RequestID, emitted: [Int], _ name: String
    ) -> (draws: Int, verifies: Int) {
        var draws = 0
        var verifies = 0
        var lastDrawn = 0
        var lastConfirmed = 0
        for observation in served.observations where observation.id == id {
            switch observation.kind {
            case .sample:
                draws += 1
                let counts = observation.counts[0]
                let n = Int(counts.reduce(0, +))
                guard n >= lastDrawn, n <= emitted.count, counts == recount(emitted.prefix(n))
                else {
                    Issue.record(
                        "\(name): draw \(draws) counts \(n) do not recount the first emitted tokens"
                    )
                    return (draws, verifies)
                }
                lastDrawn = n
            case .verify:
                verifies += 1
                let confirmed = observation.outputTokens
                guard confirmed.count >= lastConfirmed, confirmed.count <= emitted.count,
                    Array(emitted.prefix(confirmed.count)) == confirmed
                else {
                    Issue.record(
                        "\(name): verify \(verifies) confirmed output is not the emitted prefix")
                    return (draws, verifies)
                }
                lastConfirmed = confirmed.count
                let base = recount(confirmed)
                for (j, counts) in observation.counts.enumerated() {
                    let preceding = observation.drafts.prefix(observation.columnOffset + j)
                    let expected = zip(base, recount(preceding)).map { $0 + $1 }
                    guard counts == expected else {
                        Issue.record("\(name): verify \(verifies) position \(j) counts drift")
                        return (draws, verifies)
                    }
                }
            }
        }
        return (draws, verifies)
    }

    /// The MTP-off stream with ~40% of drafts wrong at pseudo-random positions
    /// (runs of wrong drafts too), so rounds reject at every window offset.
    private func script(from expected: [Int], seed: UInt64, k: Int) -> [Int] {
        var state = seed
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        return (expected + [Int](repeating: 0, count: 2 * k)).map { token in
            next() % 5 < 2 ? (token + 1 + Int(next() % UInt64(vocabSize - 1))) % vocabSize : token
        }
    }

    private struct Case {
        let name: String
        let sampling: CBv2SamplingParams
        let maxTokens: Int
        var k = 2
        var mode = CBv2MTPVerificationMode.serialTarget
        var stops: [String] = []
        var lateStop = false
        var lateStopToken = false
    }

    private let cases: [Case] = [
        Case(
            name: "t0-presence-frequency",
            sampling: .init(temperature: 0, frequencyPenalty: 0.3, presencePenalty: 0.6),
            maxTokens: 2500),
        Case(
            name: "t0-rectangular-k3",
            sampling: .init(temperature: 0, frequencyPenalty: 0.5, presencePenalty: 1.5),
            maxTokens: 2001, k: 3, mode: .rectangular),
        Case(
            name: "t1-card",
            sampling: .init(
                temperature: 1, topP: 0.95, topK: 20, frequencyPenalty: 0.2,
                presencePenalty: 1.5, seed: 1234),
            maxTokens: 2000),
        Case(
            name: "t07-holdback",
            sampling: .init(
                temperature: 0.7, frequencyPenalty: 0.4, presencePenalty: 0.5, seed: 77),
            maxTokens: 2003, stops: ["<3><31><31><31>"]),
        Case(
            name: "t0-late-stop-string",
            sampling: .init(temperature: 0, frequencyPenalty: 0.25, presencePenalty: 0.5),
            maxTokens: 2000, lateStop: true),
        Case(
            name: "t1-late-stop-token",
            sampling: .init(
                temperature: 1, topP: 0.9, frequencyPenalty: 0.3,
                presencePenalty: 0.8, seed: 9),
            maxTokens: 2000, k: 3, lateStopToken: true),
    ]

    /// Serves one entry of `cases`. Its position in `cases` sets the request
    /// ID and the draft script's seed. The target is rebuilt from its fixed
    /// seed, so it is the same for every case.
    private func check(_ name: String) async throws {
        let index = try #require(cases.firstIndex { $0.name == name }, "\(name) is a case")
        let c = cases[index]
        let target = try makeRankedTarget()
        let prompt = [3, 7, 11, 9, 13, 3, 3]
        let id = CBv2RequestID(UInt64(100 + index))
        var req = CBv2Request(
            id: id, promptTokens: prompt, sampling: c.sampling, maxTokens: c.maxTokens,
            stopStrings: c.stops)
        let unstopped = try await serve(target: target, script: nil, request: req, k: c.k)
        try #require(unstopped.finish != nil && unstopped.tokens.count > 1000, "\(c.name)")
        if c.lateStop || c.lateStopToken {
            // Stop on a late first occurrence, so the stop lands deep in the run.
            let stream = unstopped.tokens
            if c.lateStop {
                let grams = (0 ..< stream.count - 2).map { Array(stream[$0 ..< $0 + 3]) }
                var first: [[Int]: Int] = [:]
                for (i, g) in grams.enumerated() where first[g] == nil { first[g] = i }
                let (gram, at) = try #require(first.max { $0.value < $1.value })
                try #require(at > 200, "\(c.name): no late trigram")
                req.stopStrings = [gram.map { "<\($0)>" }.joined()]
            } else {
                var first: [Int: Int] = [:]
                for (i, t) in stream.enumerated() where first[t] == nil { first[t] = i }
                let (token, at) = try #require(first.max { $0.value < $1.value })
                try #require(at > 20, "\(c.name): no late token")
                req.stopTokens = [token]
            }
        }
        let baseline = try await serve(target: target, script: nil, request: req, k: c.k)
        let mtp = try await serve(
            target: target,
            script: script(from: unstopped.tokens, seed: UInt64(index + 1), k: c.k),
            request: req, k: c.k, verificationMode: c.mode)
        #expect(
            mtp.tokens == baseline.tokens, "\(c.name): MTP stream equals the MTP-off stream")
        #expect(mtp.finish == baseline.finish, "\(c.name)")
        let metrics = try #require(mtp.metrics)
        #expect(
            metrics.rounds > baseline.tokens.count / 8, "\(c.name): rows speculate throughout")
        #expect(
            metrics.acceptedTokens < metrics.draftedTokens, "\(c.name): drafts were rejected")
        #expect(
            Array(unstopped.tokens.prefix(baseline.tokens.count)) == baseline.tokens,
            "\(c.name)")
        let checked = expectExactCounts(mtp, id: id, emitted: unstopped.tokens, c.name)
        #expect(checked.verifies > baseline.tokens.count / 8, "\(c.name): \(checked)")
        _ = expectExactCounts(baseline, id: id, emitted: unstopped.tokens, "\(c.name) MTP-off")
    }

    // One suite per case, so the Fork Tests shard map can place each case on
    // its own shard (it assigns whole suites).

    @Suite struct T0PresenceFrequency {
        @Test func penaltyCountsRecountTheEmittedTokensEveryRound() async throws {
            try await CBv2MTPPenaltyLongRunTests().check("t0-presence-frequency")
        }
    }

    @Suite struct T0RectangularK3 {
        @Test func penaltyCountsRecountTheEmittedTokensEveryRound() async throws {
            try await CBv2MTPPenaltyLongRunTests().check("t0-rectangular-k3")
        }
    }

    @Suite struct T1Card {
        @Test func penaltyCountsRecountTheEmittedTokensEveryRound() async throws {
            try await CBv2MTPPenaltyLongRunTests().check("t1-card")
        }
    }

    @Suite struct T07Holdback {
        @Test func penaltyCountsRecountTheEmittedTokensEveryRound() async throws {
            try await CBv2MTPPenaltyLongRunTests().check("t07-holdback")
        }
    }

    @Suite struct T0LateStopString {
        @Test func penaltyCountsRecountTheEmittedTokensEveryRound() async throws {
            try await CBv2MTPPenaltyLongRunTests().check("t0-late-stop-string")
        }
    }

    @Suite struct T1LateStopToken {
        @Test func penaltyCountsRecountTheEmittedTokensEveryRound() async throws {
            try await CBv2MTPPenaltyLongRunTests().check("t1-late-stop-token")
        }
    }
}
