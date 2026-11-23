// CBv2MTPIneligibilityEngineTests.swift
//
// The whole-request MTP gates a row fails (`CBv2MTPIneligibility`) on a real
// engine loop, with the eligibility expression from before the set as the
// oracle, and the set stamped once on the row's exported timing through
// EngineV2 (absent without a drafter).

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// "<id>" per token, so a stop string is matched through a real detokenizer.
private struct CBv2IneligibilityTokenizer: Tokenizer {
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

@Suite("CBv2MTPIneligibilityEngine", .serialized, .tags(.integration))
struct CBv2MTPIneligibilityEngineTests {
    typealias Gate = CBv2MTPIneligibility

    private let vocabSize = 256
    private let hiddenSize = 64

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

    private func makeEngine(
        target: Gemma4TextModel, script: [Int]?, promptLength: Int,
        sampler: any CBv2StepSampler = CBv2DefaultSampler(fallbackSeed: 41),
        stopStringSpeculation: Bool = false, penaltySpeculation: Bool = false,
        constrainedSpeculation: Bool = false
    ) -> EngineV2 {
        let kinds = target.cbv2LayerKinds
        let drafter = script.map {
            CBv2TypicalScriptedDrafter(
                script: $0, promptLength: promptLength, offset: 0, vocabSize: vocabSize,
                target: target)
        }
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: sampler,
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2IneligibilityTokenizer()),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: 2, maxSpeculativeBatch: 4,
                fixedDraftTokens: 2, verificationMode: .serialTarget, acceptance: .exact,
                constrainedSpeculation: constrainedSpeculation, freeSpanSpeculation: false,
                penaltySpeculation: penaltySpeculation,
                stopStringSpeculation: stopStringSpeculation))
    }

    /// `mtpHistoryEligible` as it read before the reason set, without its
    /// media registration: the oracle for the set's emptiness.
    private static func earlierHistoryEligible(
        _ loop: EngineLoopV2, _ rec: CBv2ScheduledRequest
    ) -> Bool {
        if loop.mtp?.requiresTargetOnlyMedia(rec.request) == true { return false }
        let sampling = rec.request.sampling
        let samplingEligible =
            sampling.temperature == 0
            || (loop.mtp?.targetPrefixAcceptance == true && loop.sampler.supportsMTPTargetPrefix)
        let constraintEligible =
            rec.request.tokenConstraint == nil || loop.grammarMTPEnabled
            || loop.mtpSpanConstraint(rec) != nil
        guard rec.request.reasoningBudget == nil,
            constraintEligible,
            samplingEligible,
            sampling.topLogprobs == 0,
            sampling.logitBias.isEmpty,
            sampling.repetitionPenalty == 1,
            loop.mtpPenaltiesEligible(rec),
            rec.request.stopStrings.isEmpty
                || loop.mtp?.config.effectiveStopStringSpeculation == true
        else { return false }
        let hasMultimodalSpans = loop.multimodalByID[rec.id]?.spans.isEmpty == false
        return EngineLoopV2.mtpMultimodalHistoryEligible(
            hasSpans: hasMultimodalSpans,
            tracksPersistentHistory: loop.mtp?.tracksPersistentHistory == true)
    }

    /// On a real engine loop, over 256 requests (temperature, logprobs, logit
    /// bias, repetition and presence penalties, stop strings, reasoning budget,
    /// token constraint) and 16 engine settings (a sampler with and without
    /// target-prefix verification, the stop-string, penalty and constrained
    /// opt-ins): the row is eligible exactly when the earlier expression says
    /// so and the set is empty, and each bit follows its own field.
    @Test func reasonsMatchTheEarlierEligibilityExpression() async throws {
        let target = try makeTarget()
        let budget = try CBv2ReasoningBudget(
            maximumTokens: 4, openTokenID: 5, closeTokenID: 6, startsInsideThought: false)
        for targetPrefixSampler in [true, false] {
            for stops in [false, true] {
                for penalties in [false, true] {
                    for constrained in [false, true] {
                        let sampler: any CBv2StepSampler
                        if targetPrefixSampler {
                            sampler = CBv2DefaultSampler(fallbackSeed: 41)
                        } else {
                            sampler = CBv2GreedySampler()
                        }
                        let engine = makeEngine(
                            target: target, script: [], promptLength: 3, sampler: sampler,
                            stopStringSpeculation: stops, penaltySpeculation: penalties,
                            constrainedSpeculation: constrained)
                        let loop = engine.loopForTesting
                        #expect(loop.mtp != nil)
                        for mask in 0 ..< (1 << 8) {
                            var request = CBv2Request(
                                id: .init(UInt64(mask + 1)), promptTokens: [3, 7, 11],
                                sampling: .init(temperature: 0), maxTokens: 8)
                            if mask & 1 != 0 { request.sampling.temperature = 0.7 }
                            if mask & 2 != 0 { request.sampling.topLogprobs = 2 }
                            if mask & 4 != 0 { request.sampling.logitBias = [1: 0.5] }
                            if mask & 8 != 0 { request.sampling.repetitionPenalty = 1.1 }
                            if mask & 16 != 0 { request.sampling.presencePenalty = 1.5 }
                            if mask & 32 != 0 { request.stopStrings = ["<9>"] }
                            if mask & 64 != 0 { request.reasoningBudget = budget }
                            if mask & 128 != 0 {
                                CBv2NonSpeculatingControl.tokenConstraint.apply(to: &request)
                            }
                            let (reasons, eligible, earlier) = loop.onEngineQueueSync {
                                let rec = CBv2ScheduledRequest(
                                    request: request, arrivalSeq: 1, submittedAt: Date())
                                return (
                                    loop.mtpIneligibility(rec), loop.mtpHistoryEligible(rec),
                                    Self.earlierHistoryEligible(loop, rec)
                                )
                            }
                            let label =
                                "mask \(mask) sampler \(targetPrefixSampler) stops \(stops) "
                                + "penalties \(penalties) constrained \(constrained)"
                            #expect(eligible == earlier, "\(label)")
                            #expect(reasons.isEmpty == earlier, "\(label)")
                            var expected: Gate = []
                            if mask & 1 != 0, !targetPrefixSampler { expected.insert(.sampling) }
                            if mask & 2 != 0 { expected.insert(.logprobs) }
                            if mask & 4 != 0 { expected.insert(.logitBias) }
                            if mask & 8 != 0 { expected.insert(.repetitionPenalty) }
                            if mask & 16 != 0,
                                !(penalties && CBv2MTPConfig.envPenaltySpeculation
                                    && targetPrefixSampler && mask & 128 == 0)
                            {
                                expected.insert(.penalties)
                            }
                            if mask & 32 != 0, !(stops && CBv2MTPConfig.envStopStringSpeculation) {
                                expected.insert(.stopStrings)
                            }
                            if mask & 64 != 0 { expected.insert(.reasoningBudget) }
                            if mask & 128 != 0, !constrained { expected.insert(.grammar) }
                            #expect(reasons == expected, "\(label)")
                        }
                        await engine.shutdown()
                    }
                }
            }
        }
    }

    private struct Served {
        var tokens: [Int] = []
        var ineligibility: Gate?
        var rounds = 0
    }

    private func serve(
        _ target: Gemma4TextModel, script: [Int]?, _ request: CBv2Request,
        stopStringSpeculation: Bool = false, penaltySpeculation: Bool = false
    ) async throws -> Served {
        let engine = makeEngine(
            target: target, script: script, promptLength: request.promptTokens.count,
            stopStringSpeculation: stopStringSpeculation, penaltySpeculation: penaltySpeculation)
        let collected = await cbv2SchedCollect(try engine.submit(request))
        let rounds = engine.mtpMetricsSnapshot()?.rounds ?? 0
        await engine.shutdown()
        return Served(
            tokens: collected.tokens, ineligibility: collected.usage?.timing.mtpIneligibility,
            rounds: rounds)
    }

    /// Through the engine: a plain row's exported timing carries an empty set
    /// and the row drafts; a logprobs row carries exactly the logprobs bit and
    /// never drafts; a row failing four gates carries all four; without a
    /// drafter nothing is evaluated.
    @Test func aLogprobsRowCarriesTheLogprobsBitAndAPlainRowNone() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 19, 23]
        let plain = CBv2Request(
            id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: 12)
        let noDrafter = try await serve(target, script: nil, plain)
        #expect(noDrafter.ineligibility == nil)
        let script = noDrafter.tokens + [0, 0, 0, 0]

        let drafted = try await serve(target, script: script, plain)
        #expect(drafted.ineligibility == [])
        #expect(drafted.rounds > 0)

        var logprobs = plain
        logprobs.sampling.topLogprobs = 2
        let logprobsRow = try await serve(target, script: script, logprobs)
        #expect(logprobsRow.ineligibility == [.logprobs])
        #expect(logprobsRow.rounds == 0)

        var four = logprobs
        four.sampling.logitBias = [5: -1]
        four.sampling.repetitionPenalty = 1.1
        four.stopStrings = ["\u{1}never matches\u{1}"]
        let fourRow = try await serve(target, script: script, four)
        #expect(fourRow.ineligibility == [.logprobs, .logitBias, .repetitionPenalty, .stopStrings])
        #expect(fourRow.rounds == 0)
    }

    /// The stop-string and penalty opt-ins (the Gemma 4 provider sets both)
    /// clear their bits and the rows draft; without them the bit is set and
    /// the row stays target-only.
    @Test func optInsClearTheStopStringAndPenaltyBits() async throws {
        let target = try makeTarget()
        let prompt = [3, 7, 11, 19, 23]
        var stopRow = CBv2Request(
            id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: 12)
        stopRow.stopStrings = ["\u{1}never matches\u{1}"]
        var penaltyRow = CBv2Request(
            id: .init(2), promptTokens: prompt,
            sampling: .init(temperature: 1, topP: 0.95, topK: 64, seed: 7), maxTokens: 12)
        penaltyRow.sampling.presencePenalty = 1.5
        let script = [Int](repeating: 0, count: 16)
        for optIn in [false, true] {
            let stop = try await serve(
                target, script: script, stopRow, stopStringSpeculation: optIn)
            let stopCleared = optIn && CBv2MTPConfig.envStopStringSpeculation
            #expect(stop.ineligibility == (stopCleared ? [] : [.stopStrings]), "opt-in \(optIn)")
            #expect((stop.rounds > 0) == stopCleared, "opt-in \(optIn)")

            let penalty = try await serve(
                target, script: script, penaltyRow, penaltySpeculation: optIn)
            let penaltyCleared = optIn && CBv2MTPConfig.envPenaltySpeculation
            #expect(
                penalty.ineligibility == (penaltyCleared ? [] : [.penalties]), "opt-in \(optIn)")
            #expect((penalty.rounds > 0) == penaltyCleared, "opt-in \(optIn)")
        }
    }
}
