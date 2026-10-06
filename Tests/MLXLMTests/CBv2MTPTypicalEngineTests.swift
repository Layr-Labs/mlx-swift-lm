// CBv2MTPTypicalEngineTests.swift
//
// Typical acceptance through the real EngineV2: greedy rows are untouched,
// a confident target still rejects a wrong draft and commits its own token,
// and an oracle draft is kept without touching the keyed sample stream.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Position-keyed scripted drafter that opts into target-prefix acceptance,
/// so stochastic rows reach the verify path (the parity drafter does not
/// opt in and keeps sampled rows target-only). offset 0 proposes the
/// stabilized target's own continuation; offset 1 always disagrees.
final class CBv2TypicalScriptedDrafter: CBv2MTPDrafter {
    private let script: [Int]
    private let promptLength: Int
    private let offset: Int
    private let vocabSize: Int
    let mtpTargetIdentity: ObjectIdentifier?
    var supportsTargetPrefixAcceptance: Bool { true }

    init(script: [Int], promptLength: Int, offset: Int, vocabSize: Int, target: Gemma4TextModel) {
        self.script = script
        self.promptLength = promptLength
        self.offset = offset
        self.vocabSize = vocabSize
        self.mtpTargetIdentity = ObjectIdentifier(target)
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        CBv2ParityScriptCursor(baseIndices: rows.map { $0.anchor - promptLength + 1 })
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let cursor = prepared as! CBv2ParityScriptCursor
        defer { cursor.step += 1 }
        let ids = cursor.baseIndices.map { base -> Int32 in
            let index = base + cursor.step
            let value = index < script.count ? script[index] : 0
            return Int32((value + offset) % vocabSize)
        }
        return (MLXArray(ids), hidden)
    }
}

@Suite("CBv2MTPTypicalEngine", .serialized)
struct CBv2MTPTypicalEngineTests {
    private let vocabSize = 256
    private let hiddenSize = 64
    private let slidingWindow = 16
    private let k = 2
    private let typical = CBv2MTPAcceptance.typical(delta: CBv2MTPAcceptance.defaultTypicalDelta)

    private func targetConfig() throws -> Gemma4TextConfiguration {
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
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": false,
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func makeTarget(seed: UInt64 = 0x7E9A) throws -> Gemma4TextModel {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(try targetConfig())
        stabilizeCBv2MTPGreedyCycleTarget(target)
        eval(target)
        return target
    }

    private func makeEngine(
        target: Gemma4TextModel, drafter: (any CBv2MTPDrafter)?,
        acceptance: CBv2MTPAcceptance,
        verificationMode: CBv2MTPVerificationMode = .serialTarget
    ) -> EngineV2 {
        let kinds = target.cbv2LayerKinds
        let mtpConfig = CBv2MTPConfig(
            enabled: drafter != nil, maxDraftTokens: k,
            maxSpeculativeBatch: 4,
            fixedDraftTokens: k,
            verificationMode: verificationMode,
            maxAutomaticRectangularTokens: 8,
            acceptance: acceptance)
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
            mtpConfig: mtpConfig)
    }

    private func scriptedDrafter(
        target: Gemma4TextModel, prompt: [Int], maxTokens: Int, offset: Int
    ) -> CBv2TypicalScriptedDrafter {
        let script = cbv2MTPExpectedGreedyCycle(
            after: prompt.last!, count: maxTokens + 2 * k, vocabularySize: vocabSize)
        return CBv2TypicalScriptedDrafter(
            script: script, promptLength: prompt.count, offset: offset,
            vocabSize: vocabSize, target: target)
    }

    private func request(
        id: UInt64, prompt: [Int], maxTokens: Int, temperature: Float, seed: UInt64? = nil
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: temperature, seed: seed),
            maxTokens: maxTokens)
    }

    private func run(
        target: Gemma4TextModel, drafter: (any CBv2MTPDrafter)?,
        acceptance: CBv2MTPAcceptance, request: CBv2Request
    ) async throws -> (tokens: [Int], metrics: CBv2MTPMetrics?) {
        let engine = makeEngine(target: target, drafter: drafter, acceptance: acceptance)
        let collected = await cbv2SchedCollect(try engine.submit(request))
        let metrics = engine.mtpMetricsSnapshot()
        await engine.shutdown()
        return (collected.tokens, metrics)
    }

    @Test func configReportsTheInstalledAcceptance() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 19, 23]
        let drafter = scriptedDrafter(target: target, prompt: prompt, maxTokens: 8, offset: 0)
        let result = try await run(
            target: target, drafter: drafter, acceptance: typical,
            request: request(id: 1, prompt: prompt, maxTokens: 8, temperature: 0))
        let metrics = try #require(result.metrics)
        #expect(metrics.acceptance == typical)
        #expect(metrics.acceptance.name == "typical")
        #expect(CBv2MTPConfig().acceptance == .exact)
    }

    /// Greedy rows never reach the typical rule: the packet and the walk are
    /// the exact ones, so output equals exact mode bit for bit.
    @Test func greedyRowsAreUnchangedUnderTypical() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 19, 23]
        let maxTokens = 24
        for offset in [0, 1] {
            let exact = try await run(
                target: target,
                drafter: scriptedDrafter(
                    target: target, prompt: prompt, maxTokens: maxTokens, offset: offset),
                acceptance: .exact,
                request: request(id: 1, prompt: prompt, maxTokens: maxTokens, temperature: 0))
            let lossy = try await run(
                target: target,
                drafter: scriptedDrafter(
                    target: target, prompt: prompt, maxTokens: maxTokens, offset: offset),
                acceptance: typical,
                request: request(id: 1, prompt: prompt, maxTokens: maxTokens, temperature: 0))
            let expected = cbv2MTPExpectedGreedyCycle(
                after: prompt.last!, count: maxTokens, vocabularySize: vocabSize)
            #expect(exact.tokens == expected, "offset \(offset)")
            #expect(lossy.tokens == exact.tokens, "offset \(offset)")
            let exactMetrics = try #require(exact.metrics)
            let lossyMetrics = try #require(lossy.metrics)
            #expect(lossyMetrics.rounds > 0)
            #expect(lossyMetrics.acceptedTokens == exactMetrics.acceptedTokens, "offset \(offset)")
            #expect(lossyMetrics.controllerFallbacks["typical_acceptance_unsupported"] == nil)
        }
    }

    /// A confident target (the stabilized greedy-cycle codebook) at a
    /// sampling temperature: the oracle drafter's proposals clear the floor
    /// and are committed as proposed; the adversarial drafter's never do, and
    /// each round commits the target's own token instead of the draft.
    @Test func confidentTargetKeepsOracleDraftsAndRejectsWrongOnes() async throws {
        let target = try makeTarget()
        let prompt = [5, 9, 13, 17]
        let maxTokens = 20
        let expected = cbv2MTPExpectedGreedyCycle(
            after: prompt.last!, count: maxTokens, vocabularySize: vocabSize)

        let oracle = try await run(
            target: target,
            drafter: scriptedDrafter(
                target: target, prompt: prompt, maxTokens: maxTokens, offset: 0),
            acceptance: typical,
            request: request(
                id: 7, prompt: prompt, maxTokens: maxTokens, temperature: 0.5, seed: 99))
        let oracleMetrics = try #require(oracle.metrics)
        #expect(oracle.tokens == expected)
        #expect(oracleMetrics.rounds > 0)
        #expect(oracleMetrics.acceptedTokens == oracleMetrics.draftedTokens)
        #expect(oracleMetrics.controllerFallbacks["typical_acceptance_unsupported"] == nil)

        let adversarial = try await run(
            target: target,
            drafter: scriptedDrafter(
                target: target, prompt: prompt, maxTokens: maxTokens, offset: 1),
            acceptance: typical,
            request: request(
                id: 8, prompt: prompt, maxTokens: maxTokens, temperature: 0.5, seed: 99))
        let adversarialMetrics = try #require(adversarial.metrics)
        #expect(adversarial.tokens == expected)
        #expect(adversarialMetrics.rounds > 0)
        #expect(adversarialMetrics.acceptedTokens == 0)
        #expect(adversarialMetrics.emittedTokens == adversarialMetrics.rounds)
    }

    /// The committed token at the first rejection and at the bonus position
    /// is the keyed target sample, so a stochastic request under typical
    /// mode commits exactly what exact mode commits whenever every draft is
    /// decided the same way. The adversarial drafter guarantees that: both
    /// rules reject every draft, and every committed token is the keyed
    /// sample. (An oracle draft can be kept by typical and rejected by exact
    /// when the keyed sample differs from it, so the streams may diverge
    /// there by design.)
    @Test func correctionAndBonusTokensMatchExactMode() async throws {
        let target = try makeTarget()
        let prompt = [2, 4, 8, 16, 32]
        let maxTokens = 16
        let adversarialOffset = 1
        var tokens: [[Int]] = []
        for acceptance in [CBv2MTPAcceptance.exact, typical] {
            let result = try await run(
                target: target,
                drafter: scriptedDrafter(
                    target: target, prompt: prompt, maxTokens: maxTokens, offset: adversarialOffset),
                acceptance: acceptance,
                request: request(
                    id: 3, prompt: prompt, maxTokens: maxTokens, temperature: 0.9, seed: 1234))
            tokens.append(result.tokens)
            #expect(try #require(result.metrics).rounds > 0, "acceptance \(acceptance.name)")
            #expect(result.tokens.count == maxTokens)
            for token in result.tokens { #expect(token >= 0 && token < vocabSize) }
        }
        #expect(tokens[0] == tokens[1])
    }
}
