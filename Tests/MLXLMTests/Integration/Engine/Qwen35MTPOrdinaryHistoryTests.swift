import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A real tiny Qwen3.5 target with the production-loaded inline assistant on
/// paged storage: batch-gated rows chain as ordinary decode while feeding the
/// assistant's persistent history, and prompts seed the first carry. These are
/// mechanics tests, not numerical or latency evidence.
@Suite("Qwen3.5 inline MTP ordinary-decode history", .serialized, .tags(.integration))
struct Qwen35MTPOrdinaryHistoryTests {
    private func fixture() throws -> (Qwen35TextModel, Qwen35InlineMTPAssistant) {
        let json = Data(
            """
            {
              "model_type": "qwen3_5_moe", "mtplx_mtp": {
                "included": true, "prefix": "mtp.", "block_size": 3
              }, "mtplx_mtp_quantization": {},
              "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                "num_hidden_layers": 4, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                "linear_num_value_heads": 1, "linear_num_key_heads": 1,
                "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                "vocab_size": 64, "num_experts": 0, "num_experts_per_tok": 0,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "norm_topk_prob": true, "mtp_num_hidden_layers": 1
              }
            }
            """.utf8)
        let object = try JSONSerialization.jsonObject(with: json) as! [String: Any]
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: object["text_config"]!))
        MLXRandom.seed(7031)
        let model = Qwen35TextModel(configuration)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        quantize(model: model, groupSize: 32, bits: 4) { _, module in module is Embedding }
        let draft = Qwen35MTPModule(configuration)
        let weights = Dictionary(
            uniqueKeysWithValues: draft.parameters().flattened().map {
                ("mtp." + $0.0, $0.1.asType(.bfloat16))
            })
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-mtp-history-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try json.write(to: directory.appendingPathComponent("config.json"))
        let shard = "model-00001-of-00001.safetensors"
        try save(arrays: weights, url: directory.appendingPathComponent(shard))
        let index = [
            "weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shard) })
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        eval(model)
        let assistant = try Qwen35InlineMTPAssistant.load(
            from: directory, target: model, verificationMode: .serialTarget)
        return (model, assistant)
    }

    private func engine(
        _ target: (Qwen35TextModel, Qwen35InlineMTPAssistant),
        drafter: (any CBv2MTPDrafter)? = nil,
        mtpConfig: CBv2MTPConfig
    ) throws -> (EngineV2, PagedKVBackend) {
        let (model, assistant) = target
        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let observed = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 64 << 20, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 512, segmentSizeBytes: 64 << 10,
                layerDTypes: observed.layerDTypes))
        let storage = backend.makeLayerCaches()
        let indices = Dictionary(
            uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
            })
        let engine = EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: model.newCacheV2 { index, _ in storage[indices[index]!] }),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 32, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter ?? assistant, mtpConfig: mtpConfig)
        return (engine, backend)
    }

    /// Admit both requests atomically so they decode together from step two.
    private func runTogether(
        _ engine: EngineV2, _ requests: [CBv2Request]
    ) async throws -> [[Int]] {
        let streams = try engine.loopForTesting.onEngineQueueSync {
            try requests.map { try engine.submit($0) }
        }
        var tokens: [[Int]] = []
        for stream in streams { tokens.append(await cbv2SchedCollect(stream).tokens) }
        return tokens
    }

    /// Two prompts decode above the assistant's speculative batch of one;
    /// the shorter finishes and the longer resumes rounds. Serial
    /// verification keeps MTP-on tokens equal to target-only decode. Every
    /// stored carry (prefill, seed, verify or chained finalize) must find the
    /// production assistant's history holding exactly the transitions before
    /// the carry token: no gap and no duplicate.
    @Test("batch-gated rows chain with exact tokens and contiguous history")
    func batchGatedRowsChainWithContiguousHistory() async throws {
        final class CarryAudit: @unchecked Sendable {
            var carries = 0
            var firstOffsets: [CBv2RequestID: Int] = [:]
            var mismatches: [String] = []
        }
        let requests = [
            CBv2Request(
                id: .init(931), promptTokens: (1 ... 17).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 20),
            CBv2Request(
                id: .init(932), promptTokens: (3 ... 15).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 5),
        ]
        // One target for every engine: parallel suites share the global
        // random stream, so a per-engine build would not reproduce weights.
        let target = try fixture()
        let (reference, _) = try engine(target, mtpConfig: .init(enabled: false))
        let expected = try await runTogether(reference, requests)
        await reference.shutdown()

        for (ordinary, prefillCarry) in [(true, true), (false, false)] {
            let (candidate, backend) = try engine(
                target,
                mtpConfig: .init(
                    enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                    verificationMode: .serialTarget, ordinaryDecodeHistory: ordinary,
                    prefillCarry: prefillCarry))
            let audit = CarryAudit()
            candidate.loopForTesting.onEngineQueueSync {
                let driver = candidate.loopForTesting.mtp
                driver?.carryStoredObserverForTesting = { [weak driver] carry in
                    audit.carries += 1
                    if audit.firstOffsets[carry.id] == nil {
                        audit.firstOffsets[carry.id] = carry.kvOffset
                    }
                    let committed = driver?.assistantStateCountsForTesting(carry.id)?.committed
                    if committed != carry.kvOffset - 1 {
                        audit.mismatches.append(
                            "\(carry.id.raw)@\(carry.kvOffset): \(String(describing: committed))")
                    }
                }
            }
            let actual = try await runTogether(candidate, requests)
            let metrics = try #require(candidate.mtpMetricsSnapshot())
            let chained = candidate.chainedStepCount
            let released = candidate.loopForTesting.onEngineQueueSync {
                (
                    candidate.loopForTesting.mtp?.requestStateCountForTesting,
                    candidate.loopForTesting.recurrentStates.isEmpty,
                    backend.bytesReserved
                )
            }
            await candidate.shutdown()

            #expect(actual == expected)
            #expect(metrics.rounds > 0)
            #expect(audit.carries > metrics.rounds)
            #expect(audit.mismatches.isEmpty, "\(audit.mismatches)")
            #expect(released.0 == 0 && released.1 && released.2 == 0)
            if ordinary {
                // Prefill carries anchor at the prompt lengths; batch-gated
                // steps chain; rounds resume from a finalized carry. Only the
                // lone long row's output-budget tail may still seed.
                #expect(audit.firstOffsets == [.init(931): 17, .init(932): 13])
                #expect(chained > 0)
                #expect(metrics.controllerFallbacks["synchronized_seed"] == nil)
                #expect(metrics.seedSteps <= 1)
            } else {
                #expect(audit.firstOffsets == [.init(931): 18, .init(932): 14])
                #expect(chained == 0)
                #expect(metrics.seedSteps > 0)
            }
        }
    }

    /// Releasing finished steps' events only drops fired event handles. With
    /// the release off (the old retention), a lone drafting row and a row that
    /// drafts alone after a batch-gated stretch (consuming that stretch's
    /// history) give the same tokens, drafts and acceptance.
    @Test("history event release leaves drafting bit-identical", arguments: [true, false])
    func historyEventReleaseKeepsDraftingIdentical(ordinary: Bool) async throws {
        let lone = [
            CBv2Request(
                id: .init(951), promptTokens: (2 ... 20).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 24)
        ]
        let gated = [
            CBv2Request(
                id: .init(952), promptTokens: (1 ... 17).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 24),
            CBv2Request(
                id: .init(953), promptTokens: (3 ... 15).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 8),
        ]
        let target = try fixture()
        let (reference, _) = try engine(target, mtpConfig: .init(enabled: false))
        let expected = [
            try await runTogether(reference, lone), try await runTogether(reference, gated),
        ]
        await reference.shutdown()

        for (requests, expectedTokens) in zip([lone, gated], expected) {
            var runs: [(tokens: [[Int]], metrics: CBv2MTPMetrics)] = []
            for disabled in [true, false] {
                let (candidate, _) = try engine(
                    target,
                    mtpConfig: .init(
                        enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                        verificationMode: .serialTarget, ordinaryDecodeHistory: ordinary,
                        prefillCarry: ordinary))
                candidate.loopForTesting.onEngineQueueSync {
                    candidate.loopForTesting.mtpHistoryEventReleaseDisabledForTesting = disabled
                }
                let tokens = try await runTogether(candidate, requests)
                let metrics = try #require(candidate.mtpMetricsSnapshot())
                await candidate.shutdown()
                runs.append((tokens, metrics))
            }
            let (old, new) = (runs[0], runs[1])
            #expect(new.tokens == old.tokens)
            #expect(new.tokens == expectedTokens)
            #expect(new.metrics.rounds > 0 && new.metrics.draftedTokens > 0)
            #expect(new.metrics.rounds == old.metrics.rounds)
            #expect(new.metrics.seedSteps == old.metrics.seedSteps)
            #expect(new.metrics.draftedTokens == old.metrics.draftedTokens)
            #expect(new.metrics.acceptedTokens == old.metrics.acceptedTokens)
            #expect(new.metrics.perPositionAccepted == old.metrics.perPositionAccepted)
        }
    }

    /// Depth-zero plans chain as ordinary decode only for a drafter that opts
    /// in. A forwarding wrapper of the production assistant takes
    /// ordinary-decode history but does not opt in: with every plan forced to
    /// depth zero it runs exactly as with the process switch
    /// `DARKBLOOM_CBV2_MTP_DEPTH0_CHAINED=0`. The production assistant still
    /// chains those plans. Tokens equal target-only decode in every run.
    @Test("depth-zero plans chain only for a drafter that opts in")
    func depthZeroChainingNeedsTheDrafterOptIn() async throws {
        struct Run {
            var tokens: [[Int]]
            var depthZeroChaining: Bool?
            var feedsHistory: Bool?
            var chained: Int
            var seedSteps: Int?
        }
        let request = CBv2Request(
            id: .init(961), promptTokens: (1 ... 17).map { $0 },
            sampling: .init(temperature: 0), maxTokens: 16)
        let target = try fixture()
        let (reference, _) = try engine(target, mtpConfig: .init(enabled: false))
        let expected = try await runTogether(reference, [request])
        await reference.shutdown()

        func run(_ drafter: any CBv2MTPDrafter, switchedOff: Bool) async throws -> Run {
            let (candidate, _) = try engine(
                target, drafter: drafter,
                mtpConfig: .init(
                    enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                    verificationMode: .serialTarget, ordinaryDecodeHistory: true,
                    prefillCarry: true))
            let (chaining, feeds) = candidate.loopForTesting.onEngineQueueSync {
                () -> (Bool?, Bool?) in
                let driver = candidate.loopForTesting.mtp
                let chaining = driver?.depthZeroChaining
                if switchedOff { driver?.depthZeroChaining = false }
                driver?.planDepthLimitForTesting = { _ in 0 }
                return (chaining, driver?.feedsHistoryFromOrdinaryDecode)
            }
            let tokens = try await runTogether(candidate, [request])
            let metrics = candidate.mtpMetricsSnapshot()
            let chained = candidate.chainedStepCount
            await candidate.shutdown()
            return Run(
                tokens: tokens, depthZeroChaining: chaining, feedsHistory: feeds,
                chained: chained, seedSteps: metrics?.seedSteps)
        }

        let qwen = try await run(target.1, switchedOff: false)
        let qwenOff = try await run(target.1, switchedOff: true)
        let wrapped = try await run(
            NoDepthZeroOptInQwenAssistant(target.1), switchedOff: false)
        let wrappedOff = try await run(
            NoDepthZeroOptInQwenAssistant(target.1), switchedOff: true)

        for result in [qwen, qwenOff, wrapped, wrappedOff] {
            #expect(result.tokens == expected)
            #expect(result.feedsHistory == true)
        }
        #expect(wrapped.depthZeroChaining == false)
        #expect(wrapped.chained == wrappedOff.chained)
        #expect(wrapped.seedSteps == wrappedOff.seedSteps)
        #expect(qwen.depthZeroChaining == CBv2MTPRoundDriver.envDepthZeroChaining)
        if CBv2MTPRoundDriver.envDepthZeroChaining {
            #expect(qwen.chained > qwenOff.chained)
            #expect((qwen.seedSteps ?? 0) < (qwenOff.seedSteps ?? 0))
        }
    }

    #if canImport(Metal) && canImport(ObjectiveC)
    /// Each eval's completion event (an `MTLSharedEvent`, a finite
    /// system-wide resource) stays on the arrays the eval computed until
    /// they are waited on or read again. History keeps views of every
    /// committed step's token and hidden arrays while a batch-gated row
    /// cannot draft. Two rows decode together between the two holds, on
    /// chained ordinary history and on unchained MTP-step observation (the
    /// path of the other request-stateful assistants): the live count must
    /// not grow with the steps.
    @Test("batch-gated rows keep no Metal event per decode step", arguments: [true, false])
    func batchGatedHistoryKeepsNoEventPerStep(ordinary: Bool) async throws {
        _ = try #require(MetalSharedEventCounter.live, "needs a Metal device")
        let (candidate, _) = try engine(
            try fixture(),
            mtpConfig: .init(
                enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                verificationMode: .serialTarget, ordinaryDecodeHistory: ordinary,
                prefillCarry: ordinary))
        let requests = [
            CBv2Request(
                id: .init(941), promptTokens: (1 ... 17).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 48),
            CBv2Request(
                id: .init(942), promptTokens: (3 ... 15).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 48),
        ]
        let steps = 12 ... 36
        let sample = try #require(
            await MetalSharedEventCounter.sampleBatchGatedStretch(
                candidate, requests: requests, steps: steps))

        #expect(sample.outputs.map(\.count) == [48, 48])
        #expect((sample.metrics?.skippedRows["batch_gate"] ?? 0) > 0)
        // The old retention kept about one event per step between the holds.
        #expect(
            sample.late - sample.early <= 4,
            "live Metal shared events \(sample.early) -> \(sample.late) over \(steps)")
    }
    #endif
}

/// Forwards every call to the production Qwen 3.5 inline assistant, ordinary-
/// decode history included, but keeps the default depth-zero chaining (off).
private final class NoDepthZeroOptInQwenAssistant: CBv2MTPRequestStatefulDrafter {
    let native: Qwen35InlineMTPAssistant

    init(_ native: Qwen35InlineMTPAssistant) { self.native = native }
    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { native.requiredVerificationMode }
    var maximumDraftTokens: Int? { native.maximumDraftTokens }
    var maximumSpeculativeBatch: Int? { native.maximumSpeculativeBatch }
    var supportsTargetPrefixAcceptance: Bool { native.supportsTargetPrefixAcceptance }
    var supportsEarlyDraftSubmission: Bool { native.supportsEarlyDraftSubmission }
    var prefersBatchedRectangularAttention: Bool { native.prefersBatchedRectangularAttention }
    var commitsAcceptedPrefixPerRow: Bool { native.commitsAcceptedPrefixPerRow }
    var supportsOrdinaryDecodeHistory: Bool { native.supportsOrdinaryDecodeHistory }
    var requiresCommittedObservationFence: Bool { native.requiresCommittedObservationFence }
    var draftShortlistSize: Int? { native.draftShortlistSize }
    var submitsEveryDraftStepEarly: Bool { native.submitsEveryDraftStepEarly }
    var requestStateBytesPerToken: Int { native.requestStateBytesPerToken }
    var requestStateTokenGranularity: Int { native.requestStateTokenGranularity }
    var requestStateTokenAllocationPadding: Int { native.requestStateTokenAllocationPadding }
    var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        native.requestStateAllocationSpecs
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        native.prepare(rows: rows)
    }

    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        native.draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
    }

    func makeRequestState() -> any CBv2MTPRequestState { native.makeRequestState() }
    func configureRequestState(_ state: any CBv2MTPRequestState, maximumSequenceLength: Int)
        throws
    {
        try native.configureRequestState(state, maximumSequenceLength: maximumSequenceLength)
    }

    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, requestState: any CBv2MTPRequestState
    ) {
        native.observeCommittedTarget(observation, requestState: requestState)
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        native.draftStep(
            tokens: tokens, hidden: hidden, shortlist: shortlist, requestState: requestState)
    }

    func evaluationTargets(for state: any CBv2MTPRequestState) -> [MLXArray] {
        native.evaluationTargets(for: state)
    }

    func requestStateDidFinishEvaluation(_ state: any CBv2MTPRequestState) throws {
        try native.requestStateDidFinishEvaluation(state)
    }

    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        native.finalizeRound(
            requestState: requestState, confirmedInputTokens: confirmedInputTokens,
            committedDraftTokens: committedDraftTokens, committedTargetHidden: committedTargetHidden
        )
    }

    func discardRound(requestState: any CBv2MTPRequestState) {
        native.discardRound(requestState: requestState)
    }

    func releaseRequestState(_ state: any CBv2MTPRequestState) {
        native.releaseRequestState(state)
    }
}
