// CBv2SuffixTreeDrafterTests.swift
//
// Unit tests for `OnlineSuffixAutomaton` + `CBv2SuffixTreeDrafter`, and real
// `EngineV2` (`PagedKVBackend` + synthetic `Gemma4TextModel`) bit-exact parity
// tests at B = 1, B = 4, and B = 12 (greedy, stochastic, stop strings,
// unmatchable random text, and compacted Block-SDCM rows).

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private struct CBv2SuffixStopTokenizer: Tokenizer {
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

@Suite("CBv2 Suffix-Automaton Token-Only Drafter", .serialized)
struct CBv2SuffixTreeDrafterTests {
    private let vocabSize = 256
    private let hiddenSize = 64

    // MARK: - OnlineSuffixAutomaton Unit Tests

    @Test("Exact longest suffix match and adaptive draft horizon tiers")
    func exactLongestSuffixMatchAndAdaptiveHorizons() {
        // 1. Short match (matchLen == 2) -> adaptive k capped at 3.
        do {
            var sam = OnlineSuffixAutomaton()
            let seq = [10, 11, 20, 21, 22, 23, 24, 25, 26, 99, 10, 11]
            sam.append(seq)
            let result = sam.proposeAdaptive(tokens: seq, maxDraft: 7, minMatchLen: 2)
            #expect(result.matchLen == 2)
            #expect(result.draft == [20, 21, 22])
        }

        // 2. Medium match (matchLen == 4) -> adaptive k capped at 5.
        do {
            var sam = OnlineSuffixAutomaton()
            let seq = [1, 2, 3, 4, 30, 31, 32, 33, 34, 35, 36, 99, 1, 2, 3, 4]
            sam.append(seq)
            let result = sam.proposeAdaptive(tokens: seq, maxDraft: 7, minMatchLen: 2)
            #expect(result.matchLen == 4)
            #expect(result.draft == [30, 31, 32, 33, 34])
        }

        // 3. Long match (matchLen >= 6) -> full maxDraft (7).
        do {
            var sam = OnlineSuffixAutomaton()
            let seq = [
                1, 2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 47, 99, 1, 2, 3, 4, 5, 6,
            ]
            sam.append(seq)
            let result = sam.proposeAdaptive(tokens: seq, maxDraft: 7, minMatchLen: 2)
            #expect(result.matchLen == 6)
            #expect(result.draft == [40, 41, 42, 43, 44, 45, 46])
        }

        // 4. Periodic sequence continues smoothly across suffix links.
        do {
            var sam = OnlineSuffixAutomaton()
            let seq = [7, 8, 9, 7, 8, 9, 7, 8, 9]
            sam.append(seq)
            let result = sam.proposeAdaptive(tokens: seq, maxDraft: 7, minMatchLen: 2)
            #expect(result.matchLen == 6)
            #expect(result.draft == [7, 8, 9, 7, 8, 9, 7])
        }

        // 5. Unique non-repeating tokens yield no match.
        do {
            var sam = OnlineSuffixAutomaton()
            let seq = Array(0 ..< 64)
            sam.append(seq)
            let result = sam.proposeAdaptive(tokens: seq, maxDraft: 7, minMatchLen: 2)
            #expect(result.matchLen == 0)
            #expect(result.draft.isEmpty)
        }
    }

    @Test("Multi-branch scoring ranks by frequency and recency deterministically")
    func multiBranchRecencyAndFrequencyRanking() {
        // Equal frequency (1 each): more recent branch wins.
        do {
            var sam = OnlineSuffixAutomaton()
            let seq = [
                1, 2, 3, 100, 50, 51, 52, 53, 54,
                1, 2, 3, 200, 60, 61, 62,
                1, 2, 3,
            ]
            sam.append(seq)
            let result = sam.proposeAdaptive(tokens: seq, maxDraft: 7, minMatchLen: 2)
            #expect(result.matchLen == 3)
            #expect(result.draft.first == 200)
        }

        // High frequency (5 occurrences) outweighs single slightly newer branch.
        do {
            var sam = OnlineSuffixAutomaton()
            var seq: [Int] = []
            for i in 0 ..< 5 {
                seq.append(contentsOf: [1, 2, 3, 100, 101, 102, 150 + i])
            }
            seq.append(contentsOf: [1, 2, 3, 200, 201, 202, 250])
            seq.append(contentsOf: [1, 2, 3])
            sam.append(seq)
            let result = sam.proposeAdaptive(tokens: seq, maxDraft: 7, minMatchLen: 2)
            #expect(result.matchLen == 3)
            #expect(result.draft == [100, 101, 102])
        }
    }

    @Test("Incremental append matches full rebuild and handles shrink safely")
    func incrementalAppendEquivalence() {
        let base = (0 ..< 64).map { $0 % 19 }
        var incremental = OnlineSuffixAutomaton()
        for end in 1 ... base.count {
            incremental.append(Array(base.prefix(end)))
            #expect(incremental.consumedCount == end)
        }
        var batch = OnlineSuffixAutomaton()
        batch.append(base)

        let incResult = incremental.proposeAdaptive(tokens: base, maxDraft: 7, minMatchLen: 2)
        let batchResult = batch.proposeAdaptive(tokens: base, maxDraft: 7, minMatchLen: 2)
        #expect(incResult.matchLen == batchResult.matchLen)
        #expect(incResult.draft == batchResult.draft)

        // Shrinking resets cleanly without out-of-bounds slicing.
        let shorter = [5, 6, 7, 8, 9, 5, 6, 7]
        incremental.append(shorter)
        #expect(incremental.consumedCount == shorter.count)
        let afterReset = incremental.proposeAdaptive(tokens: shorter, maxDraft: 7, minMatchLen: 2)
        #expect(afterReset.matchLen == 3)
        #expect(afterReset.draft == [8, 9, 5])
    }

    @Test("Incremental step append and proposeAdaptive stays under 0.5 ms on 8K-16K tokens")
    func latencyUnderHalfMillisecondAt8KAnd16K() {
        for length in [8_192, 16_384] {
            var tokens = [Int]()
            tokens.reserveCapacity(length + 32)
            // Mix unique regions and repeated code-like motifs.
            for i in 0 ..< length {
                if (i % 128) < 32 {
                    tokens.append(1000 + (i % 32))
                } else {
                    tokens.append(2000 + ((i * 2_654_435_761) & 0x7FFF))
                }
            }
            // End with a matching motif from earlier in the stream.
            tokens.append(contentsOf: 1000 ..< 1008)

            var sam = OnlineSuffixAutomaton()
            sam.append(tokens)

            // Warm up 1 incremental step, then measure 100 incremental steps.
            tokens.append(1008)
            sam.append(tokens)
            _ = sam.proposeAdaptive(tokens: tokens, maxDraft: 7, minMatchLen: 2)

            let iterations = 100
            let start = DispatchTime.now().uptimeNanoseconds
            var lastDraftCount = 0
            for step in 0 ..< iterations {
                tokens.append(1000 + ((9 + step) % 32))
                sam.append(tokens)
                let proposal = sam.proposeAdaptive(tokens: tokens, maxDraft: 7, minMatchLen: 2)
                lastDraftCount = proposal.draft.count
            }
            let elapsedNanos = DispatchTime.now().uptimeNanoseconds - start
            let meanMillis = (Double(elapsedNanos) / Double(iterations)) / 1_000_000.0
            #expect(lastDraftCount == 7)
            #expect(
                meanMillis < 0.5,
                "SAM incremental step at \(length) tokens took \(meanMillis) ms (limit 0.5 ms)")
        }
    }

    @Test("CBv2SuffixTreeDrafter profitability gate, zero-padding, and memory cleanup")
    func drafterProfitabilityGateAndCleanup() {
        let drafter = CBv2SuffixTreeDrafter()
        #expect(drafter.mtpTargetIdentity == nil)
        #expect(drafter.supportsTargetPrefixAcceptance)
        #expect(drafter.commitsAcceptedPrefixPerRow)
        #expect(drafter.prefersBatchedRectangularAttention)
        #expect(drafter.maximumSpeculativeBatch == 16)

        let matching = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        let nonMatching = Array(100 ..< 132)

        // B = 1 matching passes; B = 1 non-matching stays off.
        #expect(drafter.hasCandidateMatch(for: [(CBv2RequestID(1), matching)]))
        #expect(!drafter.hasCandidateMatch(for: [(CBv2RequestID(2), nonMatching)]))

        // B = 4 with only 1 matching row (< 50% of batch): gate suppresses round so
        // non-repetitive continuous batches never stall on false-positive verification.
        let sparseB4: [(id: CBv2RequestID, tokens: [Int])] = [
            (CBv2RequestID(1), matching),
            (CBv2RequestID(2), nonMatching),
            (CBv2RequestID(3), nonMatching),
            (CBv2RequestID(4), nonMatching),
        ]
        #expect(!drafter.hasCandidateMatch(for: sparseB4))

        // B = 4 with 2 strong-matching rows (>= 50%, matchLen >= 4, post-cap mean >= 0.5):
        // passes and caps B >= 3 depth at 1 (width 2) for compiled verification, monotonically
        // across maxDraftTokens = 1..7.
        let mixedB4: [(id: CBv2RequestID, tokens: [Int])] = [
            (CBv2RequestID(1), matching),
            (CBv2RequestID(2), matching),
            (CBv2RequestID(3), nonMatching),
            (CBv2RequestID(4), nonMatching),
        ]
        #expect(drafter.hasCandidateMatch(for: mixedB4, maxDraftTokens: 7))
        #expect(drafter.hasCandidateMatch(for: mixedB4, maxDraftTokens: 2))
        let proposedB4 = drafter.proposeDrafts(for: mixedB4, maxDraftTokens: 7)
        #expect(proposedB4.depth == 1)
        #expect(proposedB4.draftsByRequest[CBv2RequestID(1)] == [7])
        #expect(proposedB4.draftsByRequest[CBv2RequestID(2)] == [7])
        #expect(proposedB4.draftsByRequest[CBv2RequestID(3)] == [0])

        // B = 2 with 2 strong-matching rows caps depth at 2 (width 3, 2*3*8=48 < 64 compiled).
        let pairB2: [(id: CBv2RequestID, tokens: [Int])] = [
            (CBv2RequestID(1), matching),
            (CBv2RequestID(2), matching),
        ]
        let proposedB2 = drafter.proposeDrafts(for: pairB2, maxDraftTokens: 7)
        #expect(proposedB2.depth == 2)
        #expect(proposedB2.draftsByRequest[CBv2RequestID(1)] == [7, 8])

        // B = 12 with only 1 matching row (< 50% threshold == 6): gate suppresses round.
        var sparseB12: [(id: CBv2RequestID, tokens: [Int])] = [(CBv2RequestID(1), matching)]
        for i in 2 ... 12 {
            sparseB12.append((CBv2RequestID(UInt64(i)), Array((100 + i * 20) ..< (120 + i * 20))))
        }
        #expect(!drafter.hasCandidateMatch(for: sparseB12))
        let suppressed = drafter.proposeDrafts(for: sparseB12, maxDraftTokens: 4)
        #expect(suppressed.depth == 0)
        #expect(suppressed.draftsByRequest.isEmpty)

        // Cleanup frees per-request automata completely.
        #expect(drafter.trackedRequestCountForTesting == 12)
        drafter.requestDidFinish(CBv2RequestID(1))
        #expect(drafter.trackedRequestCountForTesting == 11)
        drafter.removeAllRequestState()
        #expect(drafter.trackedRequestCountForTesting == 0)
    }

    // MARK: - Synthetic Gemma 4 Fixtures (PagedKVBackend)

    private func makePagedDenseTarget(seed: UInt64 = 0x5A11) throws -> Gemma4TextModel {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 64,
                "global_head_dim": 64,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 32,
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

    private func makePagedMoEBlockSDCMTarget(seed: UInt64 = 0x5DC1) throws -> Gemma4TextModel {
        let layerTypes = (0 ..< 12).map {
            ($0 + 1) % 6 == 0 ? "\"full_attention\"" : "\"sliding_attention\""
        }.joined(separator: ",")
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 12,
                "intermediate_size": 96,
                "num_attention_heads": 4,
                "head_dim": 64,
                "global_head_dim": 128,
                "vocab_size": \(vocabSize),
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 2,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 48,
                "attention_k_eq_v": true,
                "layer_types": [\(layerTypes)],
                "tie_word_embeddings": false,
                "enable_moe_block": true,
                "num_experts": 8,
                "top_k_experts": 4,
                "moe_intermediate_size": 32,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                    "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(config)
        stabilizeMoECycleWithLiveAttention(target)
        target.blockSDCMConfig = Gemma4BlockSDCMConfig(
            mode: .fast,
            minPromptTokens: 96,
            keepSchedule: [6: 0.50],
            earlyMoEK: 2,
            lateMoEK: 4,
            earlyMoEUntilLayer: 6,
            blockSize: 16,
            prefixKeep: 16,
            tailKeep: 16,
            forwardHalo: 0.85,
            backwardHalo: 0.50,
            strataCount: 2,
            twoHopBlockPeaks: 4,
            slidingChunkSize: 32)
        eval(target)
        return target
    }

    /// Install the deterministic `(tok + 1) % vocabSize` wide-margin codebook
    /// while scaling attention/MLP/MoE residual projections by `0.01` (non-zero)
    /// so Block-SDCM computes genuine compaction scores and compacted KVs stay
    /// active without near-tie logit flips.
    private func stabilizeMoECycleWithLiveAttention(_ target: Gemma4TextModel) {
        let vocabularySize = target.vocabularySize
        let h = target.configuration.hiddenSize
        let bitCount = max(1, Int(ceil(log2(Double(vocabularySize)))))
        var codebook = [Float](repeating: 0, count: vocabularySize * h)
        for token in 0 ..< vocabularySize {
            for col in 0 ..< h {
                let bit = col % bitCount
                codebook[token * h + col] = (token & (1 << bit)) == 0 ? -0.25 : 0.25
            }
        }
        var cycleHead = [Float](repeating: 0, count: codebook.count)
        for outputToken in 0 ..< vocabularySize {
            let inputToken = (outputToken - 1 + vocabularySize) % vocabularySize
            for col in 0 ..< h {
                cycleHead[outputToken * h + col] = codebook[inputToken * h + col]
            }
        }

        let parameters = target.parameters().flattened()
        var replacements: [String: MLXArray] = [:]
        for (key, value) in parameters {
            if key == "model.embed_tokens.weight" {
                replacements[key] = MLXArray(codebook, [vocabularySize, h]).asType(value.dtype)
            } else if key == "lm_head.weight" {
                replacements[key] = MLXArray(cycleHead, [vocabularySize, h]).asType(value.dtype)
            } else if key.hasSuffix(".o_proj.weight") || key.hasSuffix(".down_proj.weight") {
                replacements[key] = value * Float(0.01)
            } else if key.hasSuffix(".layer_scalar") || key == "model.norm.weight" {
                replacements[key] = MLXArray.ones(value.shape, dtype: value.dtype)
            }
        }
        target.update(parameters: ModuleParameters.unflattened(replacements))
    }

    private func makePagedEngine(
        target: Gemma4TextModel,
        useSuffixDrafter: Bool,
        maxConcurrent: Int = 16,
        prefillChunkSize: Int = 64,
        maxBatchedTokensPerStep: Int = 256,
        stopStringSpeculation: Bool = false,
        detokenizerFactory: CBv2DetokenizerFactory = CBv2NullDetokenizerFactory()
    ) throws -> EngineV2 {
        let kinds = target.cbv2LayerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                pageSize: 16,
                capacityBytes: 64 << 20,
                dtype: .float32,
                maxPrefillChunk: max(prefillChunkSize, 64),
                nominalMaxSequenceLength: 512))
        let drafter: (any CBv2MTPDrafter)? = useSuffixDrafter ? CBv2SuffixTreeDrafter() : nil
        let mtpConfig = CBv2MTPConfig(
            enabled: useSuffixDrafter,
            maxDraftTokens: 7,
            maxSpeculativeBatch: 16,
            verificationMode: .automatic,
            maxAutomaticRectangularTokens: 64,
            stopStringSpeculation: stopStringSpeculation)
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2DefaultSampler(fallbackSeed: 991),
            detokenizerFactory: detokenizerFactory,
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent,
                maxBatchedTokensPerStep: maxBatchedTokensPerStep,
                prefillChunkSize: prefillChunkSize,
                maxWaiting: max(16, maxConcurrent)),
            mtpDrafter: drafter,
            mtpConfig: mtpConfig)
    }

    private func runBatch(
        _ engine: EngineV2,
        requests: [CBv2Request]
    ) async throws -> [CBv2RequestID: CBv2SchedCollected] {
        var streams: [(CBv2RequestID, AsyncStream<CBv2Event>)] = []
        streams.reserveCapacity(requests.count)
        for req in requests {
            streams.append((req.id, try engine.submit(req)))
        }
        return await withTaskGroup(of: (CBv2RequestID, CBv2SchedCollected).self) { group in
            for (id, stream) in streams {
                group.addTask {
                    (id, await cbv2SchedCollect(stream, timeoutSeconds: 30))
                }
            }
            var results: [CBv2RequestID: CBv2SchedCollected] = [:]
            for await (id, collected) in group {
                results[id] = collected
            }
            return results
        }
    }

    /// Prompt containing a `cycleLength`-token ascending cycle followed by a
    /// repeated prefix of that cycle so the cycle target's continuation matches
    /// deeply in SAM (`length > cycleLength` and `length % cycleLength != 0`).
    private func repetitivePrompt(base: Int, length: Int = 72, cycleLength: Int = 64) -> [Int] {
        let cycle = (0 ..< cycleLength).map { (base + $0) % vocabSize }
        var prompt: [Int] = []
        prompt.reserveCapacity(length)
        while prompt.count < length {
            prompt.append(contentsOf: cycle.prefix(length - prompt.count))
        }
        return prompt
    }

    /// Non-repeating prompt spaced far apart so `(tok + 1) % vocab` continuations
    /// never form a 2-token repeat in the prompt or generated output.
    private func unmatchablePrompt(start: Int, count: Int = 16) -> [Int] {
        (0 ..< count).map { (start + $0 * 11) % vocabSize }
    }

    // MARK: - EngineV2 Parity Tests (B = 1, B = 4, B = 12)

    @Test("B = 1 greedy and stochastic bit-exact parity with acceptedTokens > 0")
    func engineParityB1GreedyAndStochastic() async throws {
        let target = try makePagedDenseTarget()
        let prompt = repetitivePrompt(base: 10, length: 72)
        let samplingModes: [(label: String, params: CBv2SamplingParams)] = [
            ("greedy", CBv2SamplingParams(temperature: 0)),
            (
                "stochastic",
                CBv2SamplingParams(temperature: 0.7, topP: 0.9, topK: 16, seed: 42)
            ),
        ]

        for (label, sampling) in samplingModes {
            let req = CBv2Request(
                id: CBv2RequestID(1), promptTokens: prompt, sampling: sampling, maxTokens: 28)

            let offEngine = try makePagedEngine(target: target, useSuffixDrafter: false)
            let baseline = try await runBatch(offEngine, requests: [req])
            await offEngine.shutdown()

            let onEngine = try makePagedEngine(target: target, useSuffixDrafter: true)
            let speculated = try await runBatch(onEngine, requests: [req])
            let metrics = try #require(onEngine.mtpMetricsSnapshot())
            await onEngine.shutdown()

            let expected = try #require(baseline[req.id])
            let actual = try #require(speculated[req.id])
            #expect(actual.tokens == expected.tokens, "B=1 \(label) token parity mismatch")
            #expect(actual.finishReason == expected.finishReason)
            #expect(metrics.seedSteps == 0, "token-only SAM must run zero seed steps")
            #expect(metrics.rounds > 0, "B=1 \(label) expected speculative rounds")
            #expect(metrics.acceptedTokens > 0, "B=1 \(label) expected acceptedTokens > 0")
        }
    }

    @Test("B = 4 greedy and stochastic parity including mixed matching and non-matching rows")
    func engineParityB4GreedyAndStochastic() async throws {
        let target = try makePagedDenseTarget(seed: 0x5A44)
        let prompts: [[Int]] = [
            repetitivePrompt(base: 0, length: 72),
            repetitivePrompt(base: 40, length: 72),
            repetitivePrompt(base: 80, length: 72),
            unmatchablePrompt(start: 200, count: 16),
        ]
        let samplingModes: [(label: String, params: CBv2SamplingParams)] = [
            ("greedy", CBv2SamplingParams(temperature: 0)),
            (
                "stochastic",
                CBv2SamplingParams(temperature: 0.7, topP: 0.9, topK: 16, seed: 77)
            ),
        ]

        for (label, sampling) in samplingModes {
            let requests = prompts.enumerated().map { idx, prompt in
                CBv2Request(
                    id: CBv2RequestID(UInt64(idx + 1)), promptTokens: prompt,
                    sampling: sampling, maxTokens: 24)
            }

            let offEngine = try makePagedEngine(
                target: target, useSuffixDrafter: false, maxConcurrent: 4,
                prefillChunkSize: 128, maxBatchedTokensPerStep: 512)
            let baseline = try await runBatch(offEngine, requests: requests)
            await offEngine.shutdown()

            let onEngine = try makePagedEngine(
                target: target, useSuffixDrafter: true, maxConcurrent: 4,
                prefillChunkSize: 128, maxBatchedTokensPerStep: 512)
            let speculated = try await runBatch(onEngine, requests: requests)
            let metrics = try #require(onEngine.mtpMetricsSnapshot())
            await onEngine.shutdown()

            for req in requests {
                #expect(
                    speculated[req.id]?.tokens == baseline[req.id]?.tokens,
                    "B=4 \(label) row \(req.id) token parity mismatch")
                #expect(speculated[req.id]?.finishReason == baseline[req.id]?.finishReason)
            }
            #expect(metrics.seedSteps == 0)
            #expect(metrics.rounds > 0)
            #expect(metrics.acceptedTokens > 0)
        }
    }

    @Test("B = 12 (above neural batch cap 8) greedy and stochastic bit-exact parity")
    func engineParityB12GreedyAndStochastic() async throws {
        let target = try makePagedDenseTarget(seed: 0x5A12)
        // 8 matching rows + 4 non-matching rows at B = 12 -> passes batch gate
        // (>= 3 matching rows and meanProposed >= 1.5) and exercises per-row
        // prefix commits + zero-padded dummy drafts at B > testedMaxSpeculativeBatch.
        let prompts: [[Int]] = (0 ..< 12).map { idx in
            idx < 8
                ? repetitivePrompt(base: (idx * 17) % vocabSize, length: 72)
                : unmatchablePrompt(start: 180 + idx * 5, count: 12)
        }
        let samplingModes: [(label: String, params: CBv2SamplingParams)] = [
            ("greedy", CBv2SamplingParams(temperature: 0)),
            (
                "stochastic",
                CBv2SamplingParams(temperature: 0.6, topP: 0.9, topK: 16, seed: 123)
            ),
        ]

        for (label, sampling) in samplingModes {
            let requests = prompts.enumerated().map { idx, prompt in
                CBv2Request(
                    id: CBv2RequestID(UInt64(idx + 1)), promptTokens: prompt,
                    sampling: sampling, maxTokens: 20)
            }

            let offEngine = try makePagedEngine(
                target: target, useSuffixDrafter: false, maxConcurrent: 16,
                prefillChunkSize: 128, maxBatchedTokensPerStep: 1024)
            let baseline = try await runBatch(offEngine, requests: requests)
            await offEngine.shutdown()

            let onEngine = try makePagedEngine(
                target: target, useSuffixDrafter: true, maxConcurrent: 16,
                prefillChunkSize: 128, maxBatchedTokensPerStep: 1024)
            let speculated = try await runBatch(onEngine, requests: requests)
            let metrics = try #require(onEngine.mtpMetricsSnapshot())
            await onEngine.shutdown()

            for req in requests {
                #expect(
                    speculated[req.id]?.tokens == baseline[req.id]?.tokens,
                    "B=12 \(label) row \(req.id) token parity mismatch")
                #expect(speculated[req.id]?.finishReason == baseline[req.id]?.finishReason)
            }
            #expect(metrics.seedSteps == 0)
            #expect(metrics.rounds > 0)
            #expect(metrics.acceptedTokens > 0)
        }
    }

    @Test("Unmatchable non-repeating stream executes zero speculative rounds")
    func unmatchableRandomStreamRunsZeroRounds() async throws {
        let target = try makePagedDenseTarget(seed: 0x5A99)
        let prompt = unmatchablePrompt(start: 1, count: 10)
        let req = CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: 16)

        let offEngine = try makePagedEngine(target: target, useSuffixDrafter: false)
        let baseline = try await runBatch(offEngine, requests: [req])
        await offEngine.shutdown()

        let onEngine = try makePagedEngine(target: target, useSuffixDrafter: true)
        let actual = try await runBatch(onEngine, requests: [req])
        let metrics = try #require(onEngine.mtpMetricsSnapshot())
        await onEngine.shutdown()

        #expect(actual[req.id]?.tokens == baseline[req.id]?.tokens)
        #expect(actual[req.id]?.finishReason == baseline[req.id]?.finishReason)
        #expect(metrics.rounds == 0)
        #expect(metrics.acceptedTokens == 0)
        #expect(metrics.seedSteps == 0)
    }

    @Test("Stop strings with token-only SAM rollback match MTP-off bit-identically")
    func stopStringsParityWithRollback() async throws {
        let target = try makePagedDenseTarget(seed: 0x5A57)
        let prompt = repetitivePrompt(base: 10, length: 72)
        // Last prompt token is (10 + 7) = 17; output cycles 18, 19, 20, 21, 22, ...
        let req = CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: 32,
            stopStrings: ["<25>"])

        let offEngine = try makePagedEngine(
            target: target, useSuffixDrafter: false,
            stopStringSpeculation: true,
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2SuffixStopTokenizer()))
        let baseline = try await runBatch(offEngine, requests: [req])
        await offEngine.shutdown()

        let onEngine = try makePagedEngine(
            target: target, useSuffixDrafter: true,
            stopStringSpeculation: true,
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2SuffixStopTokenizer()))
        let actual = try await runBatch(onEngine, requests: [req])
        let metrics = try #require(onEngine.mtpMetricsSnapshot())
        await onEngine.shutdown()

        #expect(actual[req.id]?.tokens == baseline[req.id]?.tokens)
        #expect(actual[req.id]?.text == baseline[req.id]?.text)
        #expect(actual[req.id]?.finishReason == .stop)
        #expect(actual[req.id]?.finishReason == baseline[req.id]?.finishReason)
        #expect(metrics.rounds > 0)
        #expect(metrics.acceptedTokens > 0)
        #expect(metrics.seedSteps == 0)
    }

    @Test("Compacted Block-SDCM paged rows allow token-only SAM with bit-exact parity")
    func compactedBlockSDCMRowsParity() async throws {
        let target = try makePagedMoEBlockSDCMTarget()
        let prompt = repetitivePrompt(base: 0, length: 128, cycleLength: 96)

        // 1. Direct contract check: compacted PagedSequenceKV keeps
        //    supportsSpeculativeWrites == false (preserving Gemma4BlockSDCMTests)
        //    while overriding supportsTokenOnlySpeculativeWrites == true.
        do {
            let kinds = target.cbv2LayerKinds
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 << 20,
                    dtype: .float32, maxPrefillChunk: 128, nominalMaxSequenceLength: 256))
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: prompt.count, maxLength: 256)
            defer { backend.release(state) }
            let rows = state.map { $0 as! PagedSequenceKV }
            let pagedCaches = backend.makeLayerCaches()
            let caches: [KVCache] = pagedCaches.enumerated().map { idx, cache in
                cache.setRows([rows[idx]])
                return cache
            }
            let prefillInput = MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0)
            let prefillOut = target.cbv2Prefill(
                prefillInput, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(prefillOut)
            #expect(rows[11].isCompacted)
            #expect(!rows[11].supportsSpeculativeWrites)
            #expect(rows[11].supportsTokenOnlySpeculativeWrites)
            #expect(EngineLoopV2.mtpStorageEligible(state, tokenOnly: false) == false)
            #expect(EngineLoopV2.mtpStorageEligible(state, tokenOnly: true) == true)
        }

        // 2. End-to-end EngineV2 parity on compacted Block-SDCM rows at B = 1 and B = 4.
        for batchSize in [1, 4] {
            let requests = (0 ..< batchSize).map { idx in
                CBv2Request(
                    id: CBv2RequestID(UInt64(idx + 1)),
                    promptTokens: repetitivePrompt(
                        base: (idx * 23) % vocabSize, length: 128, cycleLength: 96),
                    sampling: CBv2SamplingParams(temperature: 0),
                    maxTokens: 20)
            }

            let offEngine = try makePagedEngine(
                target: target, useSuffixDrafter: false, maxConcurrent: batchSize,
                prefillChunkSize: 128, maxBatchedTokensPerStep: 512)
            let baseline = try await runBatch(offEngine, requests: requests)
            await offEngine.shutdown()

            let onEngine = try makePagedEngine(
                target: target, useSuffixDrafter: true, maxConcurrent: batchSize,
                prefillChunkSize: 128, maxBatchedTokensPerStep: 512)
            let speculated = try await runBatch(onEngine, requests: requests)
            let metrics = try #require(onEngine.mtpMetricsSnapshot())
            await onEngine.shutdown()

            for req in requests {
                #expect(
                    speculated[req.id]?.tokens == baseline[req.id]?.tokens,
                    "Block-SDCM B=\(batchSize) row \(req.id) token parity mismatch")
                #expect(speculated[req.id]?.finishReason == baseline[req.id]?.finishReason)
            }
            #expect(metrics.seedSteps == 0)
            #expect(metrics.rounds > 0, "Block-SDCM compacted rows must speculate via SAM")
            #expect(metrics.acceptedTokens > 0)
            #expect((metrics.skippedRows["kv_unsupported"] ?? 0) == 0)
        }
    }
}
