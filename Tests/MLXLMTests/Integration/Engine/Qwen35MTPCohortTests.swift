import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A real tiny Qwen3.5 target with the production-loaded inline assistant on
/// paged storage and rectangular captured verification: decode cohorts of up
/// to `speculativeBatchLimit` rows speculate together. Each row drafts against
/// its own assistant history and the cohort's rows are verified in one
/// `[B, 1 + k]` window, so at temperature 0 every row must commit exactly the
/// tokens it commits alone, with MTP off. That oracle relies on this tiny
/// dense fixture's batched target arithmetic being row-invariant at two rows
/// (the module promises distribution-exactness, not bit parity across batch
/// shapes), so four rows pin the gate, a multi-row verify step, lengths and state
/// release, not tokens.
@Suite("Qwen3.5 inline MTP cohorts", .serialized, .tags(.integration))
struct Qwen35MTPCohortTests {
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
            .appendingPathComponent("qwen-mtp-cohort-\(UUID().uuidString)", isDirectory: true)
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

    @Test("a two-row cohort speculates and each row commits its own tokens")
    func twoRowCohortSpeculatesExactly() async throws {
        let target = try fixture()
        let requests = Self.requests(961, rows: 2)
        let expected = try await aloneReference(target, requests)

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

        #expect(actual == expected)
        // Plain batched decode of the same two rows commits the same tokens.
        let plain = try engine(target, mtpConfig: .init(enabled: false))
        let plainTogether = try await runTogether(plain, requests)
        await plain.shutdown()
        #expect(actual == plainTogether)
        #expect(metrics.rounds > 0 && metrics.draftedTokens > 0)
        // `rounds` counts verified rows, `rectangularVerificationRounds` steps:
        // more rows than steps means some step verified both rows.
        #expect(metrics.rounds > metrics.rectangularVerificationRounds)
        #expect((metrics.skippedRows["batch_gate"] ?? 0) == 0, "\(metrics.skippedRows)")
        #expect(released.0 == 0 && released.1)
    }

    @Test("adaptive two-row cohorts match plain batched decode and chain depth-zero steps")
    func adaptiveTwoRowCohortMatchesPlainBatched() async throws {
        let target = try fixture()
        let requests = Self.requests(991, rows: 2)
        let plain = try engine(target, mtpConfig: .init(enabled: false))
        let expected = try await runTogether(plain, requests)
        await plain.shutdown()

        target.1.speculativeBatchLimit = 2
        defer { target.1.speculativeBatchLimit = 1 }
        let candidate = try engine(
            target,
            mtpConfig: .init(enabled: true, maxDraftTokens: 2, verificationMode: .rectangular))
        let actual = try await runTogether(candidate, requests)
        let metrics = try #require(candidate.mtpMetricsSnapshot())
        let chained = candidate.chainedStepCount
        await candidate.shutdown()

        #expect(actual == expected)
        #expect((metrics.skippedRows["batch_gate"] ?? 0) == 0, "\(metrics.skippedRows)")
        // Plans the controller or marginal policy holds at depth zero run as
        // chained ordinary decode instead of unchained MTP seed steps.
        if (metrics.depthSelections[0] ?? 0) > 1 { #expect(chained > 0) }
    }

    @Test("a four-row cohort under a limit of four speculates without a batch gate")
    func fourRowCohortSpeculates() async throws {
        let target = try fixture()
        let requests = Self.requests(971, rows: 4)
        let expected = try await aloneReference(target, requests)

        target.1.speculativeBatchLimit = 4
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

        // Four-row target arithmetic need not be row-invariant (plain B = 4
        // decode already differs from a row alone on the 27B), so only the
        // lengths and the gate are pinned here; B = 2 pins the tokens.
        #expect(actual.map(\.count) == expected.map(\.count))
        #expect(metrics.rounds > metrics.rectangularVerificationRounds)
        #expect(released.0 == 0 && released.1)
        #expect((metrics.skippedRows["batch_gate"] ?? 0) == 0, "\(metrics.skippedRows)")
    }

    @Test("a limit of one keeps cohorts target-only with exact tokens")
    func limitOfOneKeepsBatchGate() async throws {
        let target = try fixture()
        // Independent of the runner's DARKBLOOM_QWEN_MTP_MAX_BATCH.
        target.1.speculativeBatchLimit = 1
        #expect(target.1.maximumSpeculativeBatch == 1)
        let requests = Self.requests(981, rows: 2)
        let expected = try await aloneReference(target, requests)
        let candidate = try engine(target, mtpConfig: Self.mtpConfig)
        let actual = try await runTogether(candidate, requests)
        let metrics = try #require(candidate.mtpMetricsSnapshot())
        await candidate.shutdown()

        #expect(actual == expected)
        #expect((metrics.skippedRows["batch_gate"] ?? 0) > 0)
    }

    @Test(
        "only plain rectangular verification of a standard assistant takes cohorts",
        arguments: [
            (.rectangular, true, 4),
            (.rectangular, false, 1), (.rectangularExact, true, 1), (.serialTarget, true, 1),
            (.automatic, true, 1),
        ] as [(CBv2MTPVerificationMode, Bool, Int)])
    func cohortScope(mode: CBv2MTPVerificationMode, allowsCohorts: Bool, expected: Int) {
        #expect(
            Qwen35InlineMTPAssistant.speculativeBatch(
                mode: mode, allowsCohorts: allowsCohorts, limit: 4) == expected)
    }

    /// The `maximumSpeculativeBatch == 8` check (like the cohort tests) assumes
    /// `DARKBLOOM_QWEN_MTP_SERIAL` is unset in the runner.
    @Test("the limit is clamped to 1...8")
    func limitIsClamped() throws {
        let target = try fixture()
        target.1.speculativeBatchLimit = 0
        #expect(target.1.speculativeBatchLimit == 1)
        target.1.speculativeBatchLimit = 12
        #expect(target.1.speculativeBatchLimit == 8)
        #expect(target.1.maximumSpeculativeBatch == 8)
    }

    @Test(
        "DARKBLOOM_QWEN_MTP_MAX_BATCH 1...8 wins, anything else keeps the default",
        arguments: [
            (nil, 1), (nil, 2), ("", 2), ("1", 1), ("1", 2), ("2", 2), (" 4 ", 4), ("8", 8),
            ("9", 2), ("0", 2), ("-2", 1), ("four", 2), ("2.5", 1),
        ] as [(String?, Int)])
    func environmentLimit(raw: String?, defaultLimit: Int) {
        let environment =
            raw.map { [Qwen35InlineMTPAssistant.speculativeBatchEnvironmentKey: $0] } ?? [:]
        let parsed = raw.flatMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let expected = parsed.map { (1 ... 8).contains($0) ? $0 : defaultLimit } ?? defaultLimit
        #expect(
            Qwen35InlineMTPAssistant.speculativeBatchLimit(
                environment: environment, defaultLimit: defaultLimit) == expected)
    }

    @Test("sparse targets default to cohorts of two, dense targets to one row")
    func defaultsBySparsity() throws {
        #expect(Qwen35InlineMTPAssistant.defaultSpeculativeBatchLimit(sparseTarget: true) == 2)
        #expect(Qwen35InlineMTPAssistant.defaultSpeculativeBatchLimit(sparseTarget: false) == 1)
        // The tiny fixture is dense (no experts): one row unless the runner
        // sets DARKBLOOM_QWEN_MTP_MAX_BATCH.
        if ProcessInfo.processInfo.environment[
            Qwen35InlineMTPAssistant.speculativeBatchEnvironmentKey] == nil
        {
            #expect(try fixture().1.speculativeBatchLimit == 1)
        }
    }
}
