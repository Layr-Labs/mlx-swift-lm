import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Cohort rounds that pay off: each row of a multi-row verify step commits its
/// own accepted prefix (not the cohort's shortest), and the rows' later draft
/// steps and output-head products run as shared launches. The tiny dense
/// fixture's two-row target arithmetic is row-invariant (see
/// `Qwen35MTPCohortTests`), so at temperature 0 each row must still commit
/// exactly its tokens alone with MTP off. These checks assume the runner leaves
/// `DARKBLOOM_QWEN_MTP_COHORT_DRAFTS` and `DARKBLOOM_QWEN_MTP_ROW_WIDTHS` unset.
@Suite("Qwen3.5 inline MTP cohort rows", .serialized, .tags(.integration))
struct Qwen35MTPCohortRowsTests {
    private func fixture(experts: Int = 0) throws -> (Qwen35TextModel, Qwen35InlineMTPAssistant) {
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
                "vocab_size": 64, "num_experts": \(experts), "num_experts_per_tok": \(min(experts, 2)),
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "norm_topk_prob": true, "mtp_num_hidden_layers": 1
              }
            }
            """.utf8)
        let object = try JSONSerialization.jsonObject(with: json) as! [String: Any]
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: object["text_config"]!))
        MLXRandom.seed(7032)
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
            .appendingPathComponent("qwen-mtp-cohort-rows-\(UUID().uuidString)", isDirectory: true)
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
            from: directory, target: model, verificationMode: .rectangular)
        return (model, assistant)
    }

    private func engine(
        _ target: (Qwen35TextModel, Qwen35InlineMTPAssistant),
        mtpConfig: CBv2MTPConfig, maxConcurrentRequests: Int = 4
    ) throws -> EngineV2 {
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
        return EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: model.newCacheV2 { index, _ in storage[indices[index]!] }),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: maxConcurrentRequests, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 32, maxWaiting: 8, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: assistant, mtpConfig: mtpConfig)
    }

    /// Admit the requests atomically so they decode together from step two.
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

    private static let mtpConfig = CBv2MTPConfig(
        enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2, verificationMode: .rectangular)

    /// Rows of unequal length: the cohort shrinks as rows finish.
    private static func requests(_ base: UInt64, rows: Int) -> [CBv2Request] {
        (0 ..< rows).map { row in
            CBv2Request(
                id: .init(base + UInt64(row)),
                promptTokens: ((1 + row) ... (17 + 2 * row)).map { $0 },
                sampling: .init(temperature: 0), maxTokens: [24, 9, 17, 5][row % 4])
        }
    }

    /// Reference: every row alone with MTP off.
    private func aloneReference(
        _ target: (Qwen35TextModel, Qwen35InlineMTPAssistant), _ requests: [CBv2Request]
    ) async throws -> [[Int]] {
        let reference = try engine(target, mtpConfig: .init(enabled: false))
        var tokens: [[Int]] = []
        for request in requests { tokens += try await runTogether(reference, [request]) }
        await reference.shutdown()
        return tokens
    }

    @Test("cohort rows commit their own accepted prefixes and keep exact tokens")
    func rowsCommitOwnPrefixes() async throws {
        let target = try fixture()
        let requests = Self.requests(1101, rows: 2)
        let expected = try await aloneReference(target, requests)

        target.1.speculativeBatchLimit = 2
        defer { target.1.speculativeBatchLimit = 1 }
        #expect(target.1.commitsAcceptedPrefixPerRow)
        let candidate = try engine(target, mtpConfig: Self.mtpConfig)
        let actual = try await runTogether(candidate, requests)
        let metrics = try #require(candidate.mtpMetricsSnapshot())
        let released = candidate.loopForTesting.onEngineQueueSync {
            (
                candidate.loopForTesting.mtp?.requestStateCountForTesting,
                candidate.loopForTesting.recurrentStates.isEmpty
            )
        }
        await candidate.shutdown()

        #expect(actual == expected)
        #expect(metrics.rounds > metrics.rectangularVerificationRounds)
        // Some step's rows accepted different prefixes and kept them.
        #expect(metrics.rowWidthSplitSteps > 0, "\(metrics)")
        #expect(released.0 == 0 && released.1)
    }

    @Test("a one-row limit keeps one row per round and no split widths")
    func oneRowLimitHasNoSplits() async throws {
        let target = try fixture()
        target.1.speculativeBatchLimit = 1
        #expect(!target.1.commitsAcceptedPrefixPerRow)
        let requests = Self.requests(1111, rows: 2)
        let candidate = try engine(target, mtpConfig: Self.mtpConfig)
        _ = try await runTogether(candidate, requests)
        let metrics = try #require(candidate.mtpMetricsSnapshot())
        await candidate.shutdown()
        #expect(metrics.rowWidthSplitSteps == 0)
        #expect(metrics.batchedDraftSteps == 0)
    }

    @Test("a dense cohort drafts its rows together and keeps exact tokens")
    func denseCohortDraftsTogether() async throws {
        let target = try fixture()
        let requests = Self.requests(1121, rows: 2)
        let expected = try await aloneReference(target, requests)

        target.1.speculativeBatchLimit = 2
        defer { target.1.speculativeBatchLimit = 1 }
        let candidate = try engine(target, mtpConfig: Self.mtpConfig)
        let actual = try await runTogether(candidate, requests)
        let metrics = try #require(candidate.mtpMetricsSnapshot())
        await candidate.shutdown()

        #expect(actual == expected)
        #expect(metrics.batchedDraftSteps > 0, "\(metrics)")
        #expect(metrics.rounds > metrics.rectangularVerificationRounds)
    }

    @Test("a routed-expert head drafts its cohort rows together")
    func expertHeadDraftsTogether() async throws {
        let target = try fixture(experts: 4)
        let requests = Self.requests(1131, rows: 2)
        let plain = try engine(target, mtpConfig: .init(enabled: false))
        let plainTogether = try await runTogether(plain, requests)
        await plain.shutdown()

        target.1.speculativeBatchLimit = 2
        defer { target.1.speculativeBatchLimit = 1 }
        let candidate = try engine(target, mtpConfig: Self.mtpConfig)
        let actual = try await runTogether(candidate, requests)
        let metrics = try #require(candidate.mtpMetricsSnapshot())
        let released = candidate.loopForTesting.onEngineQueueSync {
            (
                candidate.loopForTesting.mtp?.requestStateCountForTesting,
                candidate.loopForTesting.recurrentStates.isEmpty
            )
        }
        await candidate.shutdown()

        // Expert routing over a [B, 1 + k] window need not be row-invariant,
        // so lengths, the shared draft launches and state release are pinned.
        #expect(actual.map(\.count) == plainTogether.map(\.count))
        #expect(metrics.batchedDraftSteps > 0, "\(metrics)")
        #expect(released.0 == 0 && released.1)
    }

    @Test(
        "the cohort switches stay on unless set to 0, false, no or off",
        arguments: [
            (nil, true), ("", true), ("1", true), ("on", true), ("0", false), (" off ", false),
            ("FALSE", false), ("no", false), ("2", true),
        ] as [(String?, Bool)])
    func cohortSwitches(raw: String?, expected: Bool) {
        let key = "DARKBLOOM_QWEN_MTP_COHORT_DRAFTS"
        let environment = raw.map { [key: $0] } ?? [:]
        #expect(
            Qwen35InlineMTPAssistant.enabledUnlessOff(key, environment: environment) == expected)
    }

    @Test("the expert sort scope binds only inside its closure")
    func sortScopeBindsInside() {
        #expect(SwitchGLUSortScope.minimumAssignments == nil)
        SwitchGLUSortScope.$minimumAssignments.withValue(256) {
            #expect(SwitchGLUSortScope.minimumAssignments == 256)
        }
        #expect(SwitchGLUSortScope.minimumAssignments == nil)
    }

    @Test("verify windows of the inline assistant keep the unsorted expert route by default")
    func verifySortDefault() throws {
        // Assumes DARKBLOOM_QWEN_MTP_VERIFY_SORT_MIN is unset in the runner.
        let target = try fixture()
        #expect(target.1.verifySortMinimumAssignments == 256)
    }
}
