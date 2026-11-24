import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Inkling 2 rows that cannot speculate (a stop string, a penalty, a logit
/// bias, logprobs, a token constraint) record the DSpark drafter's prompt
/// history, so they save the durable checkpoints an unconstrained row saves.
/// A later request, speculating or not, restores them and emits its cold
/// tokens and logprob bits, which equal the drafter-off engine's. The real
/// Inkling 2 Mini attention geometry at a tiny hidden size, BF16, synthetic
/// weights, the production sampler.
@Suite("Inkling 2 rows that cannot speculate save checkpoints", .serialized, .tags(.integration))
struct Inkling2DSparkNonSpeculatingRowCheckpointTests {
    typealias Control = CBv2NonSpeculatingControl
    typealias Store = CompleteCheckpointFixtureStore
    typealias Geometry = Inkling2PrefixRealGeometryTests

    static let chunk = 64
    static let segmentBytes = 1 << 20
    /// Inkling's sampled logits cover the unpadded vocabulary only.
    static let vocabulary = 60
    static var dtype: DType { .bfloat16 }

    static func engine(
        model: Inkling2Model, drafter: Inkling2DSparkDrafter?, store: Store?
    ) throws -> (EngineV2, PagedKVBackend) {
        let kinds = model.cbv2LayerKinds
        let types = Array(repeating: dtype, count: kinds.count)
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 4 << 30, dtype: dtype, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: 1024, segmentSizeBytes: 16 << 20,
                layerDTypes: types))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2DefaultSampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            completeCheckpointKVDTypes: store == nil ? nil : types,
            mtpDrafter: drafter,
            mtpConfig: drafter == nil ? .init() : Inkling2DSparkDrafter.defaultMTPConfig())
        return (engine, paged)
    }

    /// Every charge and every drafter request state is back.
    static func released(_ engine: EngineV2, _ backend: PagedKVBackend) {
        let state = engine.loopForTesting.onEngineQueueSync {
            (
                engine.admissionForTesting.bytesReserved, backend.bytesReserved,
                backend.bytesWired, engine.loopForTesting.recurrentStates.isEmpty,
                engine.loopForTesting.mtp?.requestStateCountForTesting ?? 0
            )
        }
        #expect(state.0 == 0 && state.1 == 0 && state.2 == 0)
        #expect(state.3)
        #expect(state.4 == 0)
    }

    static func request(
        _ id: UInt64, _ prompt: [Int], _ control: Control?, maxTokens: Int = 8,
        cached: Bool = true
    ) -> CBv2Request {
        var request = CBv2Request(
            id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0),
            maxTokens: maxTokens, cacheSalt: "tenant", prefixCacheEnabled: cached,
            prefixCacheReceiptID: cached ? .init(id + 1000) : nil)
        control?.apply(to: &request, vocabulary: vocabulary)
        return request
    }

    static func manifestBytes(_ manifest: CBv2CompleteCheckpointManifest) throws -> Int {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(manifest).count
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

    @Test("the DSpark drafter opts in")
    func drafterOptsIn() throws {
        let model = try Geometry.target(dtype: Self.dtype)
        let drafter = try Geometry.drafter(target: model, dtype: Self.dtype)
        #expect(drafter.checkpointsNonSpeculatingRows)
    }

    /// The shared check of every family that opts in: the controlled donor
    /// emits its cold tokens with capture on and off, saves the unconstrained
    /// donor's archives byte for byte, and a controlled and an unconstrained
    /// (drafting) request each restore them and emit their cold tokens.
    @Test(.timeLimit(.minutes(10)), arguments: Control.allCases)
    func rowSavesAndReusesCheckpoints(_ control: Control) async throws {
        let model = try Geometry.target(dtype: Self.dtype)
        let drafter = try Geometry.drafter(target: model, dtype: Self.dtype)
        try await cbv2ExpectNonSpeculatingDonorCheckpoints(
            control, chunk: Self.chunk, maxTokens: 8, segmentBytes: Self.segmentBytes,
            vocabulary: Self.vocabulary,
            make: { try Self.engine(model: model, drafter: drafter, store: $0) },
            released: Self.released)
    }

    /// Drafter on equals drafter off: the controlled donor (capture on) and
    /// both restored requests emit the tokens and logprob bits of an engine
    /// without the drafter and without a prefix cache.
    @Test(.timeLimit(.minutes(10)), arguments: Control.allCases)
    func restoredOutputEqualsDrafterOff(_ control: Control) async throws {
        let model = try Geometry.target(dtype: Self.dtype)
        let drafter = try Geometry.drafter(target: model, dtype: Self.dtype)
        let prompt = (0 ..< 2 * Self.chunk + 7).map { 1 + ($0 * 7) % 59 }

        let (off, offBackend) = try Self.engine(model: model, drafter: nil, store: nil)
        let offControlled = try await cbv2ServeControlled(
            off, Self.request(1, prompt, control, cached: false))
        let offPlain = try await cbv2ServeControlled(
            off, Self.request(2, prompt, nil, cached: false))
        Self.released(off, offBackend)
        await off.shutdown()
        #expect(offControlled.tokens.count == 8 && offPlain.tokens.count == 8)

        let store = Store(segmentBytes: Self.segmentBytes)
        let (donor, donorBackend) = try Self.engine(model: model, drafter: drafter, store: store)
        let donated = try await cbv2ServeControlled(donor, Self.request(3, prompt, control))
        _ = await cbv2SchedWait(timeoutSeconds: 3) { store.saved.count >= 2 }
        store.finishPublicationCallbacks(engine: donor)
        Self.released(donor, donorBackend)
        await donor.shutdown()
        #expect(donated.tokens == offControlled.tokens, "donor")
        #expect(donated.logprobBits == offControlled.logprobBits, "donor")
        #expect(store.donationErrors.isEmpty)
        let saved = store.saved
        #expect(saved.map(\.manifest.position) == [2 * Self.chunk, Self.chunk])
        #expect(
            saved.allSatisfy {
                $0.manifest.assistantCodecID?.hasPrefix("inkling2-dspark") == true
            })

        let reopened = Store(archives: saved, segmentBytes: Self.segmentBytes)
        let (warm, warmBackend) = try Self.engine(model: model, drafter: drafter, store: reopened)
        for (id, controlled) in [(UInt64(4), true), (UInt64(5), false)] {
            let next = Self.request(id, prompt, controlled ? control : nil)
            let reference = controlled ? offControlled : offPlain
            try #require(try reopened.stage(engine: warm, request: next), "request \(id)")
            let served = try await cbv2ServeControlled(warm, next)
            #expect(served.tokens == reference.tokens, "request \(id)")
            #expect(served.logprobBits == reference.logprobBits, "request \(id)")
            #expect(served.finish == reference.finish, "request \(id)")
            #expect(served.saved == 2 * Self.chunk, "request \(id)")
            reopened.finishPublicationCallbacks(engine: warm)
            Self.released(warm, warmBackend)
        }
        #expect((warm.mtpMetricsSnapshot()?.draftedTokens ?? 0) > 0, "the plain request drafts")
        await warm.shutdown()
    }

    /// The store's manifest bound applies to a controlled donor as to an
    /// unconstrained one: with the bound between the manifests of the first
    /// and the deepest boundary, both keep the same writable boundaries, with
    /// the same bytes, and nothing above the bound.
    @Test(.timeLimit(.minutes(10)))
    func controlledDonorKeepsTheDeepestWritableCheckpoint() async throws {
        let model = try Geometry.target(dtype: Self.dtype)
        let drafter = try Geometry.drafter(target: model, dtype: Self.dtype)
        let prompt = (0 ..< 4 * Self.chunk + 7).map { 1 + ($0 * 7) % 59 }
        let control = Control.logprobs

        func donate(_ store: Store, _ request: CBv2Request) async throws -> CBv2ControlledServed {
            let (engine, backend) = try Self.engine(model: model, drafter: drafter, store: store)
            let served = try await cbv2ServeControlled(engine, request)
            _ = await cbv2SchedWait(timeoutSeconds: 3) { store.saved.count >= 2 }
            store.finishPublicationCallbacks(engine: engine)
            Self.released(engine, backend)
            await engine.shutdown()
            return served
        }

        let reference = Store(segmentBytes: Self.segmentBytes)
        _ = try await donate(reference, Self.request(1, prompt, nil))
        let measured = try reference.saved.map {
            ($0.manifest.position, try Self.manifestBytes($0.manifest))
        }
        try #require(measured.count == 2)
        #expect(measured.map(\.0) == [4 * Self.chunk, Self.chunk])
        let (deepest, first) = (measured[0].1, measured[1].1)
        let bound = first + (deepest - first) / 2

        let plainStore = Store(segmentBytes: Self.segmentBytes, maximumManifestBytes: bound)
        let plain = try await donate(plainStore, Self.request(2, prompt, nil))
        let controlledStore = Store(segmentBytes: Self.segmentBytes, maximumManifestBytes: bound)
        let controlled = try await donate(controlledStore, Self.request(3, prompt, control))

        let (off, offBackend) = try Self.engine(model: model, drafter: nil, store: nil)
        let offControlled = try await cbv2ServeControlled(
            off, Self.request(4, prompt, control, cached: false))
        let offPlain = try await cbv2ServeControlled(
            off, Self.request(5, prompt, nil, cached: false))
        Self.released(off, offBackend)
        await off.shutdown()
        #expect(plain.tokens == offPlain.tokens)
        #expect(controlled.tokens == offControlled.tokens)
        #expect(controlled.logprobBits == offControlled.logprobBits)

        let kept = controlledStore.saved
        #expect(!kept.isEmpty)
        #expect(controlledStore.donationErrors.isEmpty && plainStore.donationErrors.isEmpty)
        #expect(kept.map(\.manifest.position) == plainStore.saved.map(\.manifest.position))
        #expect(kept.allSatisfy { $0.manifest.position < 4 * Self.chunk })
        for archive in kept {
            #expect(try Self.manifestBytes(archive.manifest) <= bound)
        }
        for (ours, theirs) in zip(kept, plainStore.saved) {
            #expect(ours.manifest == theirs.manifest, "position \(ours.manifest.position)")
            #expect(Self.sameTensors(ours, theirs), "position \(ours.manifest.position)")
        }
    }
}
