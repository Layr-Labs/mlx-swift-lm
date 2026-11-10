import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The Qwen 3.5/3.6 builds' default request: the model card's presence
/// penalty (1.5) at temperature 1, top-p 0.95 and top-k 20, with a seed. The
/// penalty keeps the row from speculating; clients may add a stop string,
/// logprobs or a token constraint. Such a row must donate the checkpoints a
/// speculating row donates, durable and resident, and a request that restores
/// them must emit its cold tokens and logprob bits, speculating or not.
/// Miniature dense and MoE targets with the production-loaded inline
/// assistant and the production sampler; synthetic weights.
@Suite("Qwen 3.5 card-default rows donate checkpoints", .serialized, .tags(.integration))
struct Qwen35CardDefaultCheckpointTests {
    typealias Store = CompleteCheckpointFixtureStore
    typealias Served = CBv2ControlledServed

    static let chunk: Int = max(32, CBv2AttentionV1.queryBlockSize)
    static let prompt: [Int] = (0 ..< 2 * chunk + 7).map { 1 + ($0 * 7) % 61 }
    /// The durable store's archives, deepest first.
    static let positions: [Int] = [2 * chunk, chunk]
    static let maxTokens = 12
    /// One fallback seed for every engine: a request draws the same noise
    /// cold and warm.
    static let samplerSeed: UInt64 = 0x5eed_c0de
    static let acceptances: [CBv2MTPAcceptance] = [
        .exact, .typical(delta: CBv2MTPAcceptance.defaultTypicalDelta),
    ]

    enum Target: String, CaseIterable, CustomTestStringConvertible {
        /// Qwen3.5-9B's shape: a dense MLP.
        case dense
        /// The 35B-A3B builds' shape: routed quantized experts and a shared expert.
        case moe
        var testDescription: String { rawValue }
    }

    /// The card's sampling, alone or with what a client adds to it.
    enum Shape: String, CaseIterable, CustomTestStringConvertible {
        case card, cardStop, cardLogprobs, cardConstraint
        var testDescription: String { rawValue }

        var control: CBv2NonSpeculatingControl? {
            switch self {
            case .card: return nil
            case .cardStop: return .stopString
            case .cardLogprobs: return .logprobs
            case .cardConstraint: return .tokenConstraint
            }
        }
    }

    static func sampling(presencePenalty: Float) -> CBv2SamplingParams {
        .init(temperature: 1, topP: 0.95, topK: 20, presencePenalty: presencePenalty, seed: 4242)
    }

    /// The card's default request: it cannot speculate.
    static func card(
        _ id: UInt64, _ shape: Shape, prompt: [Int] = prompt, maxTokens: Int = maxTokens,
        cached: Bool = true
    ) -> CBv2Request {
        var request = CBv2Request(
            id: .init(id), promptTokens: prompt, sampling: sampling(presencePenalty: 1.5),
            maxTokens: maxTokens, cacheSalt: "tenant", prefixCacheEnabled: cached,
            prefixCacheReceiptID: cached ? .init(id + 1000) : nil)
        shape.control?.apply(to: &request)
        return request
    }

    /// The same request with the penalty sent as 0: it speculates.
    static func speculating(_ id: UInt64, cached: Bool = true) -> CBv2Request {
        CBv2Request(
            id: .init(id), promptTokens: prompt, sampling: sampling(presencePenalty: 0),
            maxTokens: maxTokens, cacheSalt: "tenant", prefixCacheEnabled: cached,
            prefixCacheReceiptID: cached ? .init(id + 1000) : nil)
    }

    /// One target and assistant for every engine of a test: parallel suites
    /// share the global random stream, so a per-engine build would not
    /// reproduce the weights.
    static func fixture(_ target: Target) throws -> (Qwen35TextModel, Qwen35InlineMTPAssistant) {
        let moe = target == .moe
        let text: [String: Any] = [
            "model_type": moe ? "qwen3_5_moe_text" : "qwen3_5_text", "hidden_size": 64,
            "num_hidden_layers": 4, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_value_heads": 1, "linear_num_key_heads": 1,
            "linear_key_head_dim": 64, "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
            "vocab_size": 64, "num_experts": moe ? 4 : 0, "num_experts_per_tok": moe ? 2 : 0,
            "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
            "norm_topk_prob": true, "mtp_num_hidden_layers": 1,
        ]
        let json = try JSONSerialization.data(withJSONObject: [
            "model_type": moe ? "qwen3_5_moe" : "qwen3_5",
            "mtplx_mtp": ["included": true, "prefix": "mtp.", "block_size": 3],
            "mtplx_mtp_quantization": [String: Any](),
            "text_config": text,
        ])
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: JSONSerialization.data(withJSONObject: text))
        MLXRandom.seed(moe ? 7043 : 7041)
        let model = Qwen35TextModel(configuration)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        quantize(model: model, groupSize: 32, bits: 4) { _, module in
            module is Embedding || module is SwitchLinear
        }
        let weights = Dictionary(
            uniqueKeysWithValues: Qwen35MTPModule(configuration).parameters().flattened().map {
                ("mtp." + $0.0, $0.1.asType(.bfloat16))
            })
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen35-card-rows-\(UUID().uuidString)", isDirectory: true)
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

    static func mtpConfig(_ acceptance: CBv2MTPAcceptance) -> CBv2MTPConfig {
        .init(
            enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
            verificationMode: .serialTarget, acceptance: acceptance)
    }

    /// The production paged backend and the durable complete-checkpoint
    /// store (`store == nil`: no prefix cache). `rows` above one admits a
    /// second request whose decode shares steps with the other's prompt.
    static func engine(
        _ target: (Qwen35TextModel, Qwen35InlineMTPAssistant), store: Store?,
        acceptance: CBv2MTPAcceptance = acceptances[1], rows: Int = 1
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
                capacityBytes: 96 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 512,
                segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
        let storage = backend.makeLayerCaches()
        let indices = Dictionary(
            uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
            })
        let engine = EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: model.newCacheV2 { index, _ in storage[indices[index]!] }),
            sampler: CBv2DefaultSampler(fallbackSeed: samplerSeed),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: rows * chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            mtpDrafter: assistant, mtpConfig: mtpConfig(acceptance))
        #expect((engine.completeCheckpointCodec != nil) == (store != nil))
        return (engine, backend)
    }

    /// Native-precision contiguous KV with the resident checkpoint bank.
    static func residentEngine(
        _ target: (Qwen35TextModel, Qwen35InlineMTPAssistant), bank: Bool
    ) throws -> (EngineV2, CBv2ContiguousKVBackend) {
        let (model, assistant) = target
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 96 << 20, kvDType: .bfloat16))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: model.cbv2LayerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }),
            sampler: CBv2DefaultSampler(fallbackSeed: samplerSeed),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: bank),
            admissionConfig: .init(watermarkFraction: 0),
            hybridPrefixCache: bank
                ? .init(
                    maximumBytes: 64 << 20, modelID: "qwen35-card-fixture",
                    promptContractID: "fixture-template", buildID: "fixture-build")
                : nil,
            mtpDrafter: assistant, mtpConfig: mtpConfig(acceptances[1]))
        #expect((engine.hybridPrefixCache != nil) == bank)
        return (engine, backend)
    }

    static func drafted(_ engine: EngineV2) -> Int {
        engine.mtpMetricsSnapshot()?.draftedTokens ?? 0
    }

    /// Every charge and every assistant request state is back.
    static func released(_ engine: EngineV2, _ backend: any CBv2KVBackend) {
        let state = engine.loopForTesting.onEngineQueueSync {
            (
                engine.admissionForTesting.bytesReserved, backend.bytesReserved,
                engine.loopForTesting.recurrentStates.isEmpty,
                engine.loopForTesting.mtp?.requestStateCountForTesting
            )
        }
        #expect(state.0 == 0 && state.1 == 0)
        #expect(state.2)
        #expect(state.3 == 0)
    }

    static func archives(_ store: Store, count: Int = positions.count) async -> [Store.Archive] {
        _ = await cbv2SchedWait(timeoutSeconds: 3) { store.saved.count >= count }
        return store.saved
    }

    static func sameTensors(_ a: Store.Archive, _ b: Store.Archive) -> Bool {
        func parts(_ archive: Store.Archive) -> [Store.Chunk] {
            archive.chunks.sorted { ($0.tensor, $0.offset) < ($1.tensor, $1.offset) }
        }
        let (left, right) = (parts(a), parts(b))
        return left.count == right.count
            && zip(left, right).allSatisfy {
                $0.tensor == $1.tensor && $0.offset == $1.offset && $0.bytes == $1.bytes
            }
    }

    /// The card's request donates C and 2C with capture on, and nothing with
    /// it off, without changing a token or a logprob bit. Its archives equal
    /// a speculating donor's. From the reopened store, the card's request
    /// and the speculating request each restore 2C and emit their cold
    /// tokens and logprob bits, under exact and typical acceptance.
    @Test(.timeLimit(.minutes(5)), arguments: Target.allCases, Shape.allCases)
    func cardRowDonatesAndRestoresExactly(_ target: Target, _ shape: Shape) async throws {
        let fixture = try Self.fixture(target)

        let (cold, coldBackend) = try Self.engine(fixture, store: nil)
        let expected = try await cbv2ServeControlled(cold, Self.card(11, shape))
        #expect(expected.finish == .length)
        #expect(expected.tokens.count == Self.maxTokens)
        #expect(Self.drafted(cold) == 0)
        Self.released(cold, coldBackend)
        await cold.shutdown()

        let store = Store(segmentBytes: 4096)
        let (donor, donorBackend) = try Self.engine(fixture, store: store)
        let on = try await cbv2ServeControlled(donor, Self.card(11, shape))
        let saved = await Self.archives(store)
        Self.released(donor, donorBackend)
        let off = try await cbv2ServeControlled(donor, Self.card(11, shape, cached: false))
        #expect(on.tokens == expected.tokens, "capture on")
        #expect(off.tokens == expected.tokens, "capture off")
        #expect(on.logprobBits == expected.logprobBits, "capture on")
        #expect(off.logprobBits == expected.logprobBits, "capture off")
        #expect(on.finish == expected.finish && off.finish == expected.finish)
        #expect(on.saved == 0 && off.saved == 0)
        #expect(Self.drafted(donor) == 0, "the card's row never drafts")
        #expect(saved.map(\.manifest.position) == Self.positions, "the card row's checkpoints")
        #expect(store.saved.count == saved.count, "capture off saves nothing")
        Self.released(donor, donorBackend)
        await donor.shutdown()

        let referenceStore = Store(segmentBytes: 4096)
        let (eligible, eligibleBackend) = try Self.engine(fixture, store: referenceStore)
        _ = try await cbv2ServeControlled(eligible, Self.speculating(12))
        let reference = await Self.archives(referenceStore)
        #expect(Self.drafted(eligible) > 0)
        #expect(reference.map(\.manifest.position) == Self.positions)
        #expect(saved.count == reference.count)
        for (ours, theirs) in zip(saved, reference) {
            #expect(ours.manifest == theirs.manifest, "position \(ours.manifest.position)")
            #expect(ours.manifest.assistantCodecID != nil)
            #expect(Self.sameTensors(ours, theirs), "position \(ours.manifest.position)")
        }
        Self.released(eligible, eligibleBackend)
        await eligible.shutdown()

        for acceptance in Self.acceptances {
            let (coldSpeculating, coldSpeculatingBackend) = try Self.engine(
                fixture, store: nil, acceptance: acceptance)
            let uncached = try await cbv2ServeControlled(coldSpeculating, Self.speculating(12))
            #expect(Self.drafted(coldSpeculating) > 0, "\(acceptance.name)")
            Self.released(coldSpeculating, coldSpeculatingBackend)
            await coldSpeculating.shutdown()

            let reopened = Store(archives: saved, segmentBytes: 4096)
            let (warm, warmBackend) = try Self.engine(
                fixture, store: reopened, acceptance: acceptance)
            for (request, reference) in [
                (Self.card(11, shape), expected), (Self.speculating(12), uncached),
            ] {
                let label = "\(acceptance.name) request \(request.id.raw)"
                let staged = try reopened.stage(engine: warm, request: request)
                #expect(staged, "\(label)")
                let served = try await cbv2ServeControlled(warm, request)
                #expect(served.tokens == reference.tokens, "\(label)")
                #expect(served.logprobBits == reference.logprobBits, "\(label)")
                #expect(served.finish == reference.finish, "\(label)")
                #expect(served.saved == 2 * Self.chunk, "\(label)")
                #expect(served.usage?.prefixCacheReplayTokens == 0, "\(label)")
                Self.released(warm, warmBackend)
            }
            #expect(Self.drafted(warm) > 0, "\(acceptance.name)")
            await warm.shutdown()
        }
    }

    /// A card request decodes while another card request's prompt prefills
    /// beside it, so that prompt's chunks share steps with the first
    /// request's decode. Both requests emit what they emit alone, with
    /// capture on and off, and the prompt donates C and 2C.
    @Test(.timeLimit(.minutes(5)), arguments: Target.allCases)
    func cardPromptBesideCardDecodeStaysExact(_ target: Target) async throws {
        let fixture = try Self.fixture(target)
        let short = Array(Self.prompt.prefix(9))
        func decoding(cached: Bool) -> CBv2Request {
            Self.card(21, .cardLogprobs, prompt: short, maxTokens: 24, cached: cached)
        }
        func prefilling(cached: Bool) -> CBv2Request {
            Self.card(22, .cardLogprobs, cached: cached)
        }

        let (alone, aloneBackend) = try Self.engine(fixture, store: nil, rows: 2)
        let decodingAlone = try await cbv2ServeControlled(alone, decoding(cached: false))
        let prefillingAlone = try await cbv2ServeControlled(alone, prefilling(cached: false))
        Self.released(alone, aloneBackend)
        await alone.shutdown()

        let store = Store(segmentBytes: 4096)
        let (engine, backend) = try Self.engine(fixture, store: store, rows: 2)
        let loop = engine.loopForTesting
        func together(cached: Bool) async throws -> (Served, Served, Int) {
            let mixed = Mixed()
            let prefillingID = prefilling(cached: cached).id
            let decodingID = decoding(cached: cached).id
            loop.onEngineQueueSync {
                loop.recurrentGeometryObserverForTesting = { [unowned loop] id, _, _, _, phase, _ in
                    // A prompt range of a step after the other request's
                    // first decode step.
                    guard phase == "range", id == prefillingID,
                        let other = loop.scheduler.record(for: decodingID),
                        other.generatedTokenCount >= 2
                    else { return }
                    mixed.count += 1
                }
            }
            // Admitted together: the long prompt prefills while the short one decodes.
            let streams = try loop.onEngineQueueSync {
                [
                    try engine.submit(decoding(cached: cached)),
                    try engine.submit(prefilling(cached: cached)),
                ]
            }
            let first = Task { await cbv2CollectControlled(streams[0]) }
            let second = await cbv2CollectControlled(streams[1])
            let firstServed = await first.value
            loop.onEngineQueueSync { loop.recurrentGeometryObserverForTesting = nil }
            return (firstServed, second, mixed.count)
        }
        let off = try await together(cached: false)
        #expect(store.saved.isEmpty)
        let on = try await together(cached: true)
        let saved = await Self.archives(store)
        for (run, name) in [(off, "off"), (on, "on")] {
            #expect(run.0.tokens == decodingAlone.tokens, "\(name)")
            #expect(run.0.logprobBits == decodingAlone.logprobBits, "\(name)")
            #expect(run.1.tokens == prefillingAlone.tokens, "\(name)")
            #expect(run.1.logprobBits == prefillingAlone.logprobBits, "\(name)")
            #expect(run.2 > 0, "\(name): the prompt shared steps with the decode")
        }
        #expect(saved.map(\.manifest.position) == Self.positions)
        #expect(Self.drafted(engine) == 0)
        Self.released(engine, backend)
        await engine.shutdown()
    }

    /// Steps in which one request decodes while the other's prompt prefills.
    final class Mixed: @unchecked Sendable {
        var count = 0
    }

    /// On the resident bank, the card's request publishes its prompt's
    /// checkpoints; the next card request and the speculating request
    /// restore 2C and emit their cold tokens and logprob bits.
    @Test(.timeLimit(.minutes(5)), arguments: Target.allCases)
    func cardRowFillsTheResidentBank(_ target: Target) async throws {
        let fixture = try Self.fixture(target)
        let (cold, coldBackend) = try Self.residentEngine(fixture, bank: false)
        let expected = try await cbv2ServeControlled(cold, Self.card(31, .cardLogprobs))
        let uncached = try await cbv2ServeControlled(cold, Self.speculating(32))
        #expect(Self.drafted(cold) > 0)
        Self.released(cold, coldBackend)
        await cold.shutdown()

        let (engine, backend) = try Self.residentEngine(fixture, bank: true)
        let bank = try #require(engine.hybridPrefixCache)
        let donor = try await cbv2ServeControlled(engine, Self.card(31, .cardLogprobs))
        #expect(donor.tokens == expected.tokens)
        #expect(donor.logprobBits == expected.logprobBits)
        #expect(donor.saved == 0)
        #expect(Self.drafted(engine) == 0)
        let published = await cbv2SchedWait(timeoutSeconds: 3) { bank.stats.entries == 1 }
        #expect(published, "the card row's prompt reaches the bank")
        for (request, reference) in [
            (Self.card(31, .cardLogprobs), expected), (Self.speculating(32), uncached),
        ] {
            let served = try await cbv2ServeControlled(engine, request)
            #expect(served.saved == 2 * Self.chunk, "request \(request.id.raw)")
            #expect(served.usage?.prefixCacheTier == .resident, "request \(request.id.raw)")
            #expect(served.tokens == reference.tokens, "request \(request.id.raw)")
            #expect(served.logprobBits == reference.logprobBits, "request \(request.id.raw)")
            #expect(served.finish == reference.finish, "request \(request.id.raw)")
        }
        #expect(Self.drafted(engine) > 0)
        let idle = engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.recurrentStates.isEmpty
                && engine.loopForTesting.mtp?.requestStateCountForTesting == 0
        }
        #expect(idle)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
    }
}
