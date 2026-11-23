import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Durable prefix checkpoints on the real Inkling 2 Mini attention geometry
/// (24 layers, core 6..<18 looped 3 times = 48 KV slots, 40 of them sliding
/// with a 1,024-token window, 8 x 128 KV heads) at a tiny hidden size, BF16
/// and float32, with and without the DSpark drafter: the cold donor publishes
/// the demanded boundary, it restores into a fresh engine, and greedy tokens
/// equal the cold run and the drafter-off run. The same checks run on the
/// real weights when `INKLING2_FULL_MODEL=1` (see `realWeights`).
@Suite(.serialized, .tags(.integration))
struct Inkling2PrefixRealGeometryTests {
    static let visits = [5, 23, 41, 48]
    static let localLayers = [
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
    ]

    struct Shape {
        /// The real runs' 4 GiB pool: staged windows may hold 1/8 of it slot-wide
        /// and 1/16 per donor, and this geometry copies 40 sliding windows of up
        /// to 1,024 x 8 x 128 BF16 K/V (~4 MiB each) per boundary. The pool is
        /// segmented, so only touched pages are allocated.
        var capacityBytes = 4 << 30
        var chunk = 64
        var stepTokens = 256
        var maxSequence = 1024
        var segmentBytes = 16 << 20
        var concurrent = 2
        var maxTokens = 8
        var target = 256
        /// The store's read size; the export refuses reads above
        /// `CBv2CompleteCheckpointManifest.maximumSegmentBytes` (4 MiB).
        var storeSegmentBytes = 1 << 20
    }

    static func target(dtype: DType) throws -> Inkling2Model {
        let config: [String: Any] = [
            "model_type": "inkling2_loop",
            "text_config": [
                "hidden_size": 128, "num_hidden_layers": 24, "vocab_size": 64,
                "unpadded_vocab_size": 60, "num_attention_heads": 8, "num_key_value_heads": 8,
                "head_dim": 128, "d_rel": 4, "rel_extent": 24, "log_scaling_n_floor": 8,
                "log_scaling_alpha": 0.5, "qk_l2_clip_threshold": 8.0,
                "rel_l2_clip_threshold": 2.0, "local_layer_ids": localLayers,
                "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
                "sliding_window_size": 1024, "n_layers_per_attn_res_block": 6,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32,
                "loop_config": [
                    "core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 8,
                ],
            ] as [String: Any],
            "mtp_config": [
                "hidden_size": 128, "num_hidden_layers": 3, "num_attention_heads": 8,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 64,
                "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0, "local_layer_ids": [0, 2],
                "sliding_window_size": 1024, "block_size": 8, "markov_rank": 16,
                "markov_head_type": "rnn_hidden", "target_layer_ids": visits,
            ] as [String: Any],
        ]
        let model = Inkling2Model(
            try SyntheticModel.configuration(Inkling2Configuration.self, config))
        SyntheticModel.randomize(model, seed: 11)
        if dtype != .float32 {
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map {
                        (
                            $0.0,
                            $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                                ? $0.1.asType(dtype) : $0.1
                        )
                    }))
        }
        eval(model)
        return model
    }

    static func drafter(target: Inkling2Model, dtype: DType) throws -> Inkling2DSparkDrafter {
        #expect(target.dsparkVisits == visits)
        let configuration = Inkling2DSparkConfiguration(
            hiddenSize: 128, numHiddenLayers: 3, numAttentionHeads: 8, numKeyValueHeads: 8,
            headDim: 128, intermediateSize: 64, localLayerIds: [0, 2], slidingWindowSize: 1024,
            blockSize: 8, markovRank: 16, targetLayerIds: visits, vocabularySize: 60,
            paddedVocabularySize: 64, finalLogitSoftcapping: 30, targetLayerVisits: 48)
        try configuration.validate()
        let model = Inkling2DSparkModel(configuration)
        SyntheticModel.randomize(model, seed: 5)
        if dtype != .float32 {
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
        }
        eval(model)
        model.prepareDerivedWeights()
        return Inkling2DSparkDrafter(
            model: model, target: target, requiredVerificationMode: .rectangularExact,
            commitsAcceptedPrefixPerRow: true)
    }

    static func engine(
        model: Inkling2Model, drafter: Inkling2DSparkDrafter?, dtype: DType, shape: Shape,
        store: CompleteCheckpointFixtureStore?
    ) throws -> (EngineV2, PagedKVBackend) {
        let kinds = model.cbv2LayerKinds
        let types = Array(repeating: dtype, count: kinds.count)
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: shape.capacityBytes, dtype: dtype, maxPrefillChunk: shape.chunk,
                nominalMaxSequenceLength: shape.maxSequence,
                segmentSizeBytes: shape.segmentBytes, layerDTypes: types))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: shape.concurrent,
                maxBatchedTokensPerStep: shape.stepTokens,
                prefillChunkSize: shape.chunk, maxWaiting: 8, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            completeCheckpointKVDTypes: store == nil ? nil : types,
            mtpDrafter: drafter,
            mtpConfig: drafter == nil ? .init() : Inkling2DSparkDrafter.defaultMTPConfig())
        return (engine, paged)
    }

    static func request(
        _ id: UInt64, prompt: [Int], shape: Shape, salt: String = "real-geometry"
    ) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0, seed: 0),
            maxTokens: shape.maxTokens, cacheSalt: salt, prefixCacheReceiptID: .init(id + 1000),
            prefixCheckpointTargetTokens: shape.target)
    }

    /// Prompt ranges each request computed (engine queue writes).
    final class Ranges: @unchecked Sendable {
        private let lock = NSLock()
        private var ranges: [UInt64: [Range<Int>]] = [:]
        func observe(_ engine: EngineV2) {
            engine.loopForTesting.recurrentGeometryObserverForTesting = {
                [self] id, range, _, _, phase, _ in
                guard phase == "range" else { return }
                lock.withLock { ranges[id.raw, default: []].append(range) }
            }
        }
        func line(_ id: UInt64, promptCount: Int) -> String {
            lock.withLock { ranges[id] ?? [] }.filter { $0.lowerBound < promptCount }
                .map { "\($0.lowerBound)..<\($0.upperBound)" }.joined(separator: " ")
        }
    }

    static func eventLine(_ engine: EngineV2) -> String {
        let events = engine.completeCheckpointCapture?.events.snapshot ?? [:]
        let details = engine.completeCheckpointCapture?.events.lastDetails ?? [:]
        return events.map { "\($0.key.rawValue)=\($0.value)" }.sorted().joined(separator: " ")
            + " | "
            + details.map { "\($0.key.rawValue) \($0.value)" }.sorted().joined(
                separator: "; ")
    }

    /// Drafter off (no store) for every prompt; drafter as given, cold, on
    /// `prompts[0]`, which must publish `shape.target`; then every prompt
    /// restored from that archive into a fresh engine. Restored == cold ==
    /// drafter off, a foreign salt never stages, nothing stays reserved.
    static func check(
        _ label: String, model: Inkling2Model, drafter: Inkling2DSparkDrafter?, dtype: DType,
        shape: Shape, prompts: [[Int]]
    ) async throws {
        let (off, _) = try engine(
            model: model, drafter: nil, dtype: dtype, shape: shape, store: nil)
        var offTokens: [[Int]] = []
        for (index, prompt) in prompts.enumerated() {
            let result = try await cbv2CollectFinished(
                try off.submit(request(UInt64(10 + index), prompt: prompt, shape: shape)),
                timeoutSeconds: 1200)
            offTokens.append(result.tokens)
        }
        await off.shutdown()
        try #require(offTokens.allSatisfy { $0.count == shape.maxTokens })

        let seed = CompleteCheckpointFixtureStore(segmentBytes: shape.storeSegmentBytes)
        let (cold, coldBackend) = try engine(
            model: model, drafter: drafter, dtype: dtype, shape: shape, store: seed)
        let coldRanges = Ranges()
        coldRanges.observe(cold)
        let coldResult = try await cbv2CollectFinished(
            try cold.submit(request(1, prompt: prompts[0], shape: shape)), timeoutSeconds: 1200)
        seed.finishPublicationCallbacks(engine: cold)
        let coldEvents = eventLine(cold)
        let rounds = cold.mtpMetricsSnapshot()?.rounds ?? 0
        let accepted = cold.mtpMetricsSnapshot()?.acceptedTokens ?? 0
        await cold.shutdown()
        print(
            "INKLING2_PREFIX_FIX \(label) tokens off=\(offTokens[0]) cold=\(coldResult.tokens) accepted=\(accepted) ranges: \(coldRanges.line(1, promptCount: prompts[0].count))"
        )
        print(
            "INKLING2_PREFIX_FIX \(label) cold rounds=\(rounds) archives=\(seed.saved.map(\.manifest.position)) events: \(coldEvents) donationErrors: \(seed.donationErrors)"
        )
        #expect(seed.donationErrors.isEmpty)
        #expect(coldResult.tokens == offTokens[0])
        #expect(coldBackend.bytesReserved == 0)
        let archives = seed.saved.filter { $0.manifest.position == shape.target }
        try #require(!archives.isEmpty)
        if drafter != nil {
            #expect(rounds > 0)
            #expect(
                archives[0].manifest.backendLayout
                    == CBv2CompleteCheckpointManifest.pagedHistoricalRecurrentMTPLayout)
            #expect(archives[0].manifest.assistantCodecID?.hasPrefix("inkling2-dspark") == true)
        }

        for (index, prompt) in prompts.enumerated() {
            let restored = CompleteCheckpointFixtureStore(
                archives: archives, segmentBytes: 1 << 20)
            let (warm, warmBackend) = try engine(
                model: model, drafter: drafter, dtype: dtype, shape: shape, store: restored)
            let warmRanges = Ranges()
            warmRanges.observe(warm)
            let id = UInt64(20 + 2 * index)
            #expect(
                try !restored.stage(
                    engine: warm,
                    request: request(id + 1, prompt: prompt, shape: shape, salt: "other"))
            )
            let input = request(id, prompt: prompt, shape: shape)
            try #require(try restored.stage(engine: warm, request: input))
            let warmResult = try await cbv2CollectFinished(
                try warm.submit(input), timeoutSeconds: 1200)
            restored.finishPublicationCallbacks(engine: warm)
            let warmRounds = warm.mtpMetricsSnapshot()?.rounds ?? 0
            let warmAccepted = warm.mtpMetricsSnapshot()?.acceptedTokens ?? 0
            await warm.shutdown()
            let saved = warmResult.usage?.prefixCachePrefillTokensSaved ?? -1
            print(
                "INKLING2_PREFIX_FIX \(label) restored prompt \(index): equal=\(warmResult.tokens == offTokens[index]) saved=\(saved) rounds=\(warmRounds) accepted=\(warmAccepted) tokens=\(warmResult.tokens) off=\(offTokens[index]) ranges: \(warmRanges.line(id, promptCount: prompt.count))"
            )
            #expect(warmResult.tokens == offTokens[index])
            #expect(saved == shape.target)
            if drafter != nil { #expect(warmRounds > 0) }
            #expect(warmBackend.bytesReserved == 0 && warm.admissionForTesting.bytesReserved == 0)
        }
    }

    @Test(arguments: ["bfloat16", "float32"], [true, false])
    func restoredPrefixEqualsColdAndDrafterOff(dtypeName: String, drafting: Bool) async throws {
        let dtype: DType = dtypeName == "bfloat16" ? .bfloat16 : .float32
        let model = try Self.target(dtype: dtype)
        let drafter = drafting ? try Self.drafter(target: model, dtype: dtype) : nil
        let prompt = SyntheticModel.tokens(count: 768, vocabularySize: 60, seed: 77)
        let other =
            Array(prompt[..<256])
            + SyntheticModel.tokens(count: 200, vocabularySize: 60, seed: 78)
        var shape = Shape()
        // Same budget relative to the windows: float32 windows are twice as large.
        shape.capacityBytes = shape.capacityBytes / 2 * dtype.size
        try await Self.check(
            "tiny \(dtypeName) drafter=\(drafting)", model: model, drafter: drafter,
            dtype: dtype, shape: shape, prompts: [prompt, other])
    }

    /// The target part of a drafter-on checkpoint (windows, full-layer rows,
    /// conv state) is bit-identical to the drafter-off checkpoint at the same
    /// boundary: the drafter never changes target numerics. With the drafter
    /// on, prompt chunks run `cbv2ForwardWithHiddenForPrefill` instead of
    /// `cbv2RecurrentPrefill`; this catches any mode flag (such as
    /// `Inkling2PrefillStep`, a2ef1965) the hidden-capturing path misses.
    @Test func drafterCheckpointCarriesTheSameTargetState() async throws {
        let dtype = DType.bfloat16
        let model = try Self.target(dtype: dtype)
        let drafter = try Self.drafter(target: model, dtype: dtype)
        let shape = Shape()
        let prompt = SyntheticModel.tokens(count: 320, vocabularySize: 60, seed: 77)
        func archive(_ drafter: Inkling2DSparkDrafter?) async throws
            -> CompleteCheckpointFixtureStore.Archive
        {
            let store = CompleteCheckpointFixtureStore(segmentBytes: shape.storeSegmentBytes)
            let (engine, _) = try Self.engine(
                model: model, drafter: drafter, dtype: dtype, shape: shape, store: store)
            _ = try await cbv2CollectFinished(
                try engine.submit(Self.request(1, prompt: prompt, shape: shape)),
                timeoutSeconds: 600)
            store.finishPublicationCallbacks(engine: engine)
            await engine.shutdown()
            return try #require(store.saved.first { $0.manifest.position == shape.target })
        }
        let on = try await archive(drafter)
        let off = try await archive(nil)
        func bytes(_ archive: CompleteCheckpointFixtureStore.Archive, _ index: Int) -> Data {
            archive.chunks.filter { $0.tensor == index }.sorted { $0.offset < $1.offset }
                .reduce(into: Data()) { $0.append($1.bytes) }
        }
        let assistantRoles: Set<CBv2CheckpointTensorRole> = [
            .assistantHidden, .assistantTokens, .assistantFrontier, .assistantKeys,
            .assistantValues, .assistantCacheMetadata,
        ]
        var compared = 0
        var differing: [String] = []
        for (index, tensor) in on.manifest.tensors.enumerated()
        where !assistantRoles.contains(tensor.role) {
            guard
                let match = off.manifest.tensors.firstIndex(where: {
                    $0.role == tensor.role && $0.layer == tensor.layer && $0.shape == tensor.shape
                })
            else {
                differing.append("\(tensor.role) layer \(tensor.layer ?? -1) missing")
                continue
            }
            compared += 1
            if bytes(on, index) != bytes(off, match) {
                differing.append("\(tensor.role) layer \(tensor.layer ?? -1)")
            }
        }
        print("INKLING2_PREFIX_FIX target tensors compared=\(compared) differing=\(differing)")
        #expect(compared > 0)
        #expect(differing.isEmpty)
    }

    /// With the drafter on, prompt chunks run the hidden-capturing
    /// `cbv2ForwardWithHiddenForPrefill`; it must take the same prefill mode
    /// as `cbv2RecurrentPrefill` (a2ef1965: without `Inkling2PrefillStep` the
    /// chunks took the decode attention core and, on GPUs with neural
    /// accelerators, different bits). Hardware-independent: the step hook
    /// reports each trunk call's mode. A distinctive chunk length (96) keeps
    /// calls from suites running in parallel out of the count.
    @Test func drafterPromptChunksRunAsPrefill() async throws {
        final class Calls: @unchecked Sendable {
            private let lock = NSLock()
            private var calls: [String] = []
            func add(_ mode: String, _ length: Int) {
                guard length == 96 else { return }
                lock.withLock { calls.append(mode) }
            }
            func take() -> [String] {
                lock.withLock {
                    defer { calls = [] }
                    return calls
                }
            }
        }
        let dtype = DType.bfloat16
        let model = try Self.target(dtype: dtype)
        let drafter = try Self.drafter(target: model, dtype: dtype)
        var shape = Shape()
        shape.chunk = 96
        let prompt = SyntheticModel.tokens(count: 7 * 96 + 5, vocabularySize: 60, seed: 79)
        let calls = Calls()
        Inkling2Diagnostics.stepHook = { calls.add($0, $1) }
        defer { Inkling2Diagnostics.stepHook = nil }
        var modes: [Bool: [String]] = [:]
        var tokens: [Bool: [Int]] = [:]
        for drafting in [false, true] {
            let (engine, _) = try Self.engine(
                model: model, drafter: drafting ? drafter : nil, dtype: dtype, shape: shape,
                store: nil)
            _ = calls.take()
            let result = try await cbv2CollectFinished(
                try engine.submit(Self.request(1, prompt: prompt, shape: shape)),
                timeoutSeconds: 600)
            if drafting { #expect((engine.mtpMetricsSnapshot()?.rounds ?? 0) > 0) }
            await engine.shutdown()
            modes[drafting] = calls.take()
            tokens[drafting] = result.tokens
        }
        let on = modes[true] ?? []
        let off = modes[false] ?? []
        #expect(on.filter { $0 == "prefill" }.count >= 7)
        #expect(!on.contains("forward"))
        #expect(
            on.filter { $0 == "prefill-built" }.count == off.filter { $0 == "prefill-built" }.count)
        #expect(tokens[true] == tokens[false])
    }

    /// A refused assistant copy loses only the checkpoint. Before the fix the
    /// engine discarded the donor's whole step: its conv state rolled back
    /// while the chunk's KV and scheduler progress stood, and greedy tokens
    /// diverged from the next chunk on.
    @Test func refusedAssistantCaptureNeverChangesTheDonor() async throws {
        let dtype = DType.bfloat16
        let model = try Self.target(dtype: dtype)
        let drafter = try Self.drafter(target: model, dtype: dtype)
        let shape = Shape()
        let prompt = SyntheticModel.tokens(count: 384, vocabularySize: 60, seed: 77)
        let (off, _) = try Self.engine(
            model: model, drafter: nil, dtype: dtype, shape: shape, store: nil)
        let offResult = try await cbv2CollectFinished(
            try off.submit(Self.request(9, prompt: prompt, shape: shape)), timeoutSeconds: 600)
        await off.shutdown()

        drafter.refusesPrefixCaptureForTesting = true
        defer { drafter.refusesPrefixCaptureForTesting = false }
        let store = CompleteCheckpointFixtureStore(segmentBytes: shape.storeSegmentBytes)
        let (cold, backend) = try Self.engine(
            model: model, drafter: drafter, dtype: dtype, shape: shape, store: store)
        let result = try await cbv2CollectFinished(
            try cold.submit(Self.request(1, prompt: prompt, shape: shape)), timeoutSeconds: 600)
        store.finishPublicationCallbacks(engine: cold)
        let events = cold.completeCheckpointCapture?.events.snapshot ?? [:]
        let rounds = cold.mtpMetricsSnapshot()?.rounds ?? 0
        await cold.shutdown()
        #expect(result.tokens == offResult.tokens)
        #expect((events[.assistantCaptureRefused] ?? 0) > 0)
        #expect(store.saved.isEmpty)
        #expect(rounds > 0)
        #expect(backend.bytesReserved == 0 && cold.admissionForTesting.bytesReserved == 0)
    }

    /// A store that reads 64 MiB segments: every `readSegment` above
    /// `CBv2CompleteCheckpointManifest.maximumSegmentBytes` throws, the store
    /// drops the archive, and nothing is ever published. The engine side is
    /// fine; the store must read within the wire contract.
    @Test func storeReadsAboveTheSegmentLimitPublishNothing() async throws {
        let model = try Self.target(dtype: .bfloat16)
        var shape = Shape()
        shape.storeSegmentBytes = 64 << 20
        let prompt = SyntheticModel.tokens(count: 320, vocabularySize: 60, seed: 77)
        let store = CompleteCheckpointFixtureStore(segmentBytes: shape.storeSegmentBytes)
        let (engine, _) = try Self.engine(
            model: model, drafter: nil, dtype: .bfloat16, shape: shape, store: store)
        _ = try await cbv2CollectFinished(
            try engine.submit(Self.request(1, prompt: prompt, shape: shape)), timeoutSeconds: 600)
        store.finishPublicationCallbacks(engine: engine)
        let events = engine.completeCheckpointCapture?.events.snapshot ?? [:]
        await engine.shutdown()
        #expect((events[.exported] ?? 0) > 0)
        #expect(store.saved.isEmpty)
        #expect(!store.donationErrors.isEmpty)
        #expect(shape.storeSegmentBytes > CBv2CompleteCheckpointManifest.maximumSegmentBytes)
    }

    static let environment = ProcessInfo.processInfo.environment
    static let realEnabled =
        environment["INKLING2_FULL_MODEL"] == "1" && environment["INKLING2_MODEL_DIR"] != nil
        && environment["INKLING2_ORACLE_MODEL_DIR"] != nil

    /// Real weights (correctness only): oracle prompt c, A = c[0..<target+512],
    /// B = c[0..<target] + c[2300..<2601] (shares only the target prefix).
    /// Knobs: INKLING2_PREFIX_{TARGET,CHUNK,KVGB,STEP_TOKENS,SEGMENT_MB,NEW_TOKENS,DRAFTER,
    /// STORE_SEGMENT_KB}.
    @Test(.enabled(if: realEnabled))
    func realWeights() async throws {
        let env = Self.environment
        MLX.GPU.set(memoryLimit: Int(env["INKLING2_MLX_MEMORY_GB"] ?? "44")! << 30)
        MLX.GPU.set(cacheLimit: 8 << 30)
        var shape = Shape()
        shape.target = Int(env["INKLING2_PREFIX_TARGET"] ?? "2048")!
        shape.chunk = Int(env["INKLING2_PREFIX_CHUNK"] ?? "2048")!
        shape.capacityBytes = Int(env["INKLING2_PREFIX_KVGB"] ?? "4")! << 30
        shape.stepTokens = Int(env["INKLING2_PREFIX_STEP_TOKENS"] ?? "4096")!
        shape.segmentBytes = Int(env["INKLING2_PREFIX_SEGMENT_MB"] ?? "256")! << 20
        shape.maxTokens = Int(env["INKLING2_PREFIX_NEW_TOKENS"] ?? "24")!
        shape.storeSegmentBytes = Int(env["INKLING2_PREFIX_STORE_SEGMENT_KB"] ?? "1024")! << 10
        shape.maxSequence = 4096
        shape.concurrent = 1
        let drafting = env["INKLING2_PREFIX_DRAFTER"] != "0"
        let weights = URL(fileURLWithPath: env["INKLING2_MODEL_DIR"]!, isDirectory: true)
        let configuration = try JSONDecoder().decode(
            Inkling2Configuration.self,
            from: Data(contentsOf: weights.appendingPathComponent("config.json")))
        let model = Inkling2Model(configuration)
        try await loadWeights(modelDirectory: weights, model: model)
        let drafter =
            drafting ? try Inkling2DSparkDrafter.load(directory: weights, target: model) : nil
        let fixture = URL(fileURLWithPath: env["INKLING2_ORACLE_MODEL_DIR"]!)
            .appendingPathComponent("c/teacher_forced.safetensors")
        let ids = try loadArrays(url: fixture)["input_ids"]!.asType(.int32).asArray(Int32.self)
            .map(Int.init)
        try #require(ids.count >= 2601 && shape.target + 512 <= 2601)
        let promptA = Array(ids[0 ..< shape.target + 512])
        let promptB = Array(ids[0 ..< shape.target]) + Array(ids[2300 ..< 2601])
        try await Self.check(
            "real drafter=\(drafting) target=\(shape.target) chunk=\(shape.chunk) kv=\(shape.capacityBytes >> 30)GB",
            model: model, drafter: drafter, dtype: .bfloat16, shape: shape,
            prompts: [promptA, promptB])
    }
}
