// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Image requests through the production engine on the segmented paged
/// backend, with the adapter and cache bank the provider builds. The model is
/// tiny with the production attention shape; image features are fixed random
/// rows, since the engine only splices them.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerImageEngineTests {
    static let window = 320
    static let placeholder = 7
    static let collectSeconds: TimeInterval = 180

    private func model(_ dtype: DType) throws -> MuseGlimmerTextModel {
        try MuseGlimmerFixture.model(
            MuseGlimmerFixture.pagedText(window: Self.window), dtype: dtype)
    }

    private func engine(
        _ m: MuseGlimmerTextModel, chunk: Int, concurrency: Int = 2,
        store: CompleteCheckpointFixtureStore? = nil
    ) throws -> (EngineV2, PagedKVBackend) {
        m.prefillChunkAlignment = chunk
        let backend = try PagedKVBackend(
            layerKinds: m.cbv2LayerKinds,
            config: .init(
                capacityBytes: 256 << 20, dtype: m.activationDType, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: 2048, segmentSizeBytes: 1 << 20,
                layerDTypes: m.kvStorageDTypes))
        #expect(
            EngineV2.backendCapabilityViolation(capabilities: m.cbv2Capabilities, backend: backend)
                == nil)
        let engine = EngineV2(
            model: MuseGlimmerCBv2Adapter(m), layerKinds: m.cbv2LayerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: concurrency, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, soloPrefillStripeTokens: chunk, maxWaiting: 4,
                enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store)
        return (engine, backend)
    }

    /// A prompt with placeholder ids under `spans` and one feature array per span.
    struct ImagePrompt {
        let tokens: [Int]
        let spans: [CBv2ImageSpan]
        let features: [MLXArray]

        init(length: Int, spans: [CBv2ImageSpan], hidden: Int, salt: Int, seed: UInt64) {
            var tokens = MuseGlimmerFixture.tokens(length, vocabulary: 96, salt: salt).map(Int.init)
            for span in spans {
                for index in span.tokenOffset ..< span.end {
                    tokens[index] = MuseGlimmerImageEngineTests.placeholder
                }
            }
            self.tokens = tokens
            self.spans = spans
            features = spans.enumerated().map {
                MuseGlimmerFixture.hostTensor([$1.length, hidden], seed: seed &+ UInt64($0))
            }
            eval(features)
        }

        func request(
            _ id: UInt64, maxTokens: Int, cacheSalt: String? = nil, receipt: UInt64? = nil,
            checkpointAt target: Int? = nil, identity: CBv2HybridPrefixIdentity? = nil,
            attention: CBv2MultimodalAttention = .causal
        ) -> CBv2Request {
            let features = features
            return CBv2Request(
                id: .init(id), promptTokens: tokens, sampling: .init(temperature: 0, seed: 0),
                maxTokens: maxTokens, cacheSalt: cacheSalt,
                multimodal: CBv2MultimodalInput(spans: spans, attention: attention) { features },
                hybridPrefixIdentity: identity, prefixCacheReceiptID: receipt.map { .init($0) },
                prefixCheckpointTargetTokens: target)
        }

        func text(
            _ id: UInt64, maxTokens: Int, cacheSalt: String? = nil, receipt: UInt64? = nil,
            checkpointAt target: Int? = nil
        ) -> CBv2Request {
            CBv2Request(
                id: .init(id), promptTokens: tokens, sampling: .init(temperature: 0, seed: 0),
                maxTokens: maxTokens, cacheSalt: cacheSalt,
                prefixCacheReceiptID: receipt.map { .init($0) },
                prefixCheckpointTargetTokens: target)
        }
    }

    /// The prompt's embeddings with the features spliced in, prefilled in
    /// `chunk`-token pieces on the eager caches, then greedy decode.
    static func eagerGreedy(
        _ m: MuseGlimmerTextModel, _ prompt: ImagePrompt, images: Bool = true, chunk: Int,
        steps: Int
    ) throws -> [Int] {
        let cache = try m.newCache(parameters: nil)
        let ids = MLXArray(prompt.tokens.map(Int32.init)).reshaped(1, -1)
        let embeddings = CBv2MultimodalPlan.spliceEmbeddings(
            textEmbeddings: m.scaledTokenEmbeddings(ids), chunkStart: 0,
            spans: images
                ? zip(prompt.spans, prompt.features).map { ($0, $1.expandedDimensions(axis: 0)) }
                : [])
        var logits = MLXArray.zeros([1])
        for start in stride(from: 0, to: prompt.tokens.count, by: chunk) {
            let end = min(start + chunk, prompt.tokens.count)
            logits = m(inputEmbeddings: embeddings[0..., start ..< end, 0...], cache: cache)
            eval([logits] + cache.flatMap { $0.innerState() })
        }
        var generated = [Int]()
        for _ in 0 ..< steps {
            let next = argMax(logits[0, -1].asType(.float32), axis: -1).item(Int.self)
            generated.append(next)
            logits = m(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)
            eval([logits] + cache.flatMap { $0.innerState() })
        }
        return generated
    }

    private func run(_ engine: EngineV2, _ request: CBv2Request) async throws -> CBv2SchedCollected
    {
        await cbv2SchedCollect(try engine.submit(request), timeoutSeconds: Self.collectSeconds)
    }

    // MARK: Capability

    @Test func theAdapterTakesCausalImageSpansAtCacheOffsets() throws {
        let m = try model(.bfloat16)
        let adapter = MuseGlimmerCBv2Adapter(m)
        #expect(adapter.supportsMultimodalPrefill(attention: .causal))
        #expect(!adapter.supportsMultimodalPrefill(attention: .bidirectionalSpans))
        #expect(adapter.causalPositionRequirement == .scalarCacheOffset)
        // Packed prompt cohorts would splice nothing into a causal span
        // (`chunkContext` is nil for causal media), so they must stay off.
        #expect(!((adapter as AnyObject) is any CBv2PackedPrefillSteppableModel))
        #expect(!m.cbv2Capabilities.supportsPackedPrefill)
        // No checkpoint of an image prompt is ever published or restored.
        #expect(!m.cbv2Capabilities.supportsCausalMediaCheckpointReuse)
        let ids = MLXArray([Int32(3), 7, 95]).reshaped(1, 3)
        #expect(
            MuseGlimmerFixture.sameBits(
                adapter.embedPromptTokens(ids), m.scaledTokenEmbeddings(ids)))
        // The bank the provider builds admits causal media as it is.
        m.prefillChunkAlignment = 256
        let backend = try PagedKVBackend(
            layerKinds: m.cbv2LayerKinds,
            config: .init(
                capacityBytes: 64 << 20, dtype: m.activationDType, maxPrefillChunk: 256,
                nominalMaxSequenceLength: 2048, segmentSizeBytes: 1 << 20,
                layerDTypes: m.kvStorageDTypes))
        let bank = CBv2LayerCacheBank(caches: backend.makeLayerCaches())
        #expect(bank.supportsMultimodalPrefill(attention: .causal))
    }

    @Test func bidirectionalAndMisshapenImagesAreRefusedAtSubmit() async throws {
        let m = try model(.bfloat16)
        let (engine, backend) = try engine(m, chunk: 256)
        let prompt = ImagePrompt(
            length: 300, spans: [.init(tokenOffset: 10, length: 40)], hidden: 64, salt: 1, seed: 3)
        let bidirectional = #expect(throws: CBv2MultimodalError.self) {
            try engine.submit(prompt.request(1, maxTokens: 4, attention: .bidirectionalSpans))
        }
        guard case .unsupportedModel? = bidirectional else {
            Issue.record("expected unsupportedModel, got \(String(describing: bidirectional))")
            return
        }
        let wide = ImagePrompt(
            length: 300, spans: [.init(tokenOffset: 10, length: 40)], hidden: 48, salt: 1, seed: 3)
        let misshapen = #expect(throws: CBv2MultimodalError.self) {
            try engine.submit(wide.request(2, maxTokens: 4))
        }
        guard case .embeddingMismatch? = misshapen else {
            Issue.record("expected embeddingMismatch, got \(String(describing: misshapen))")
            return
        }
        // A span past the prompt's end.
        let features = prompt.features
        let outside = #expect(throws: CBv2MultimodalError.self) {
            try engine.submit(
                CBv2Request(
                    id: .init(3), promptTokens: Array(prompt.tokens.prefix(30)), maxTokens: 4,
                    multimodal: CBv2MultimodalInput(
                        spans: prompt.spans, attention: .causal
                    ) { features }))
        }
        guard case .invalidSpans? = outside else {
            Issue.record("expected invalidSpans, got \(String(describing: outside))")
            return
        }
        let served = try await run(engine, prompt.request(4, maxTokens: 4))
        #expect(served.finishReason == .length && served.tokens.count == 4)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    // MARK: Spans and chunk cuts

    /// Production grid: one 250-row image straddles the 1,024-token cut, a
    /// second one sits right after it, and a third ends the prompt.
    @Test(.timeLimit(.minutes(6)))
    func spansAcrossThe1024GridEqualTheEagerSplice() async throws {
        let m = try model(.float32)
        let prompt = ImagePrompt(
            length: 1300,
            spans: [
                .init(tokenOffset: 900, length: 250), .init(tokenOffset: 1150, length: 6),
                .init(tokenOffset: 1290, length: 10),
            ],
            hidden: 64, salt: 4, seed: 11)
        let expected = try Self.eagerGreedy(m, prompt, chunk: 1024, steps: 24)
        let textOnly = try Self.eagerGreedy(m, prompt, images: false, chunk: 1024, steps: 24)
        #expect(expected != textOnly, "the images must change the continuation")
        #expect(Set(expected).count > 3)
        let (engine, backend) = try engine(m, chunk: 1024)
        #expect(engine.loopForTesting.scheduler.prefillChunkAlignment == 1024)
        let result = try await run(engine, prompt.request(1, maxTokens: 24))
        await engine.shutdown()
        #expect(result.finishReason == .length)
        #expect(result.tokens == expected)
        #expect(result.usage?.promptTokens == 1300)
        #expect(!backend.pool.writeValidation.isFaulted && backend.bytesReserved == 0)
    }

    /// The sliding window's left edge reaches the image during the prompt
    /// and leaves it during decode: the last prompt row sees only the image's
    /// final row, and the next rows see none of it.
    @Test(.timeLimit(.minutes(4)))
    func aSpanAtTheSlidingWindowEdgeEqualsTheEagerSplice() async throws {
        let m = try model(.float32)
        let span = CBv2ImageSpan(tokenOffset: 40, length: 260)
        let prompt = ImagePrompt(
            length: span.end + Self.window - 1, spans: [span], hidden: 64, salt: 6, seed: 13)
        let expected = try Self.eagerGreedy(m, prompt, chunk: 256, steps: 24)
        let (engine, backend) = try engine(m, chunk: 256)
        let result = try await run(engine, prompt.request(1, maxTokens: 24))
        await engine.shutdown()
        #expect(result.tokens == expected)
        #expect(!backend.pool.writeValidation.isFaulted && backend.bytesReserved == 0)
    }

    /// Image and text requests share steps; each gets its solo output.
    @Test(.timeLimit(.minutes(6)), arguments: [DType.float32, DType.bfloat16])
    func imageAndTextRequestsTogetherEqualTheirSoloRuns(dtype: DType) async throws {
        let m = try model(dtype)
        let image = ImagePrompt(
            length: 700, spans: [.init(tokenOffset: 200, length: 300)], hidden: 64, salt: 2,
            seed: 17)
        let text = ImagePrompt(length: 520, spans: [], hidden: 64, salt: 3, seed: 0)
        let inputs = [image.request(1, maxTokens: 24), text.text(2, maxTokens: 24)]
        var solo = [[Int]]()
        for input in inputs {
            let (engine, _) = try engine(m, chunk: 256, concurrency: 1)
            let result = try await run(engine, input)
            await engine.shutdown()
            #expect(result.finishReason == .length)
            solo.append(result.tokens)
        }
        if dtype == .float32 {
            #expect(solo[0] == (try Self.eagerGreedy(m, image, chunk: 256, steps: 24)))
        }
        let (engine, backend) = try engine(m, chunk: 256)
        let first = try engine.submit(inputs[0])
        let second = try engine.submit(inputs[1])
        async let a = cbv2SchedCollect(first, timeoutSeconds: Self.collectSeconds)
        async let b = cbv2SchedCollect(second, timeoutSeconds: Self.collectSeconds)
        let together = await [a, b]
        await engine.shutdown()
        #expect(together.map(\.tokens) == solo)
        #expect(!backend.pool.writeValidation.isFaulted && backend.bytesReserved == 0)
    }

    // MARK: Prefix cache

    /// An image prompt neither restores nor publishes a checkpoint, resident
    /// or durable, with or without a media identity, even when a checkpoint
    /// of the same token ids is staged under its receipt.
    @Test(.timeLimit(.minutes(8)))
    func imageRequestsNeverReadOrWriteTheCache() async throws {
        let m = try model(.bfloat16)
        let prompt = ImagePrompt(
            length: 1543, spans: [.init(tokenOffset: 100, length: 200)], hidden: 64, salt: 5,
            seed: 19)
        let (plain, _) = try engine(m, chunk: 256)
        let cold = try await run(plain, prompt.request(1, maxTokens: 8))
        let coldText = try await run(plain, prompt.text(2, maxTokens: 8))
        await plain.shutdown()
        try #require(cold.finishReason == .length && coldText.finishReason == .length)
        #expect(coldText.tokens != cold.tokens, "the image must change the continuation")

        // One engine with the cache on: the image request runs first, then
        // text with the same ids, then the image again.
        let store = CompleteCheckpointFixtureStore(segmentBytes: 4093)
        let (shared, sharedBackend) = try engine(m, chunk: 256, store: store)
        let first = try await run(
            shared,
            prompt.request(3, maxTokens: 8, cacheSalt: "muse", receipt: 1003, checkpointAt: 1024))
        store.finishPublicationCallbacks(engine: shared)
        #expect(first.tokens == cold.tokens)
        #expect((first.usage?.prefixCachePrefillTokensSaved ?? 0) == 0)
        #expect(store.saved.isEmpty, "an image request published a checkpoint")
        let text = try await run(
            shared,
            prompt.text(4, maxTokens: 8, cacheSalt: "muse", receipt: 1004, checkpointAt: 1024))
        store.finishPublicationCallbacks(engine: shared)
        #expect(text.tokens == coldText.tokens)
        #expect((text.usage?.prefixCachePrefillTokensSaved ?? 0) == 0, "text reused image state")
        let published = store.saved.count
        try #require(published > 0, "text must publish, or this test proves nothing")
        let again = try await run(
            shared,
            prompt.request(5, maxTokens: 8, cacheSalt: "muse", receipt: 1005, checkpointAt: 1024))
        store.finishPublicationCallbacks(engine: shared)
        #expect(again.tokens == cold.tokens)
        #expect(
            (again.usage?.prefixCachePrefillTokensSaved ?? 0) == 0, "an image reused text state")
        #expect(store.saved.count == published)
        await shared.shutdown()
        #expect(sharedBackend.bytesReserved == 0)

        // A fresh engine whose store holds the text checkpoint, staged under
        // the image request's receipt by a text twin.
        let identity = try CBv2HybridPrefixIdentity(digest: Data(repeating: 7, count: 32))
        for (offset, mediaIdentity) in [nil, identity].enumerated() {
            let id = UInt64(6 + offset)
            let warmStore = CompleteCheckpointFixtureStore(
                archives: store.saved, segmentBytes: 4093)
            let (warm, backend) = try engine(m, chunk: 256, store: warmStore)
            let image = prompt.request(
                id, maxTokens: 8, cacheSalt: "muse", receipt: 1000 + id, checkpointAt: 1024,
                identity: mediaIdentity)
            // The store's own import plan refuses the image request.
            let refused = (try? warmStore.stage(engine: warm, request: image)) ?? false
            #expect(!refused)
            let twin = prompt.text(
                id, maxTokens: 8, cacheSalt: "muse", receipt: 1000 + id, checkpointAt: 1024)
            try #require(try warmStore.stage(engine: warm, request: twin))
            let result = try await run(warm, image)
            warmStore.finishPublicationCallbacks(engine: warm)
            #expect(result.tokens == cold.tokens, "identity \(mediaIdentity != nil)")
            #expect((result.usage?.prefixCachePrefillTokensSaved ?? 0) == 0)
            #expect(warmStore.saved.count == published)
            warmStore.close()
            await warm.shutdown()
            #expect(backend.bytesReserved == 0)
        }
    }
}

/// The DFlash drafter with exact verify windows beside image requests: image
/// rows stay on the target alone, text rows keep drafting, and every output is
/// the drafter-off output.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerImageDrafterTests {
    typealias Scripted = MuseGlimmerDFlashScriptedDrafter
    static let chunk = max(32, CBv2AttentionV1.queryBlockSize)
    static let maxTokens = 32
    static let taps = [1, 3, 5]
    static let hidden = MuseGlimmerExactWindowTests.hidden

    private func target() throws -> MuseGlimmerTextModel {
        let model = try MuseGlimmerExactWindowTests.mixed(layers: 6, dtype: .bfloat16)
        model.mtpHiddenLayerIDs = Self.taps
        try model.setRowExactWindows(true)
        return model
    }

    private func drafter(_ target: MuseGlimmerTextModel) throws -> MuseGlimmerDFlashDrafter {
        let configuration = try MuseGlimmerDFlashConfiguration(
            hiddenSize: Self.hidden, intermediateSize: 256, hiddenLayers: 2, attentionHeads: 4,
            kvHeads: 2, headDimensions: 16, blockSize: 16, maskTokenID: 95,
            targetLayerIDs: Self.taps, slidingWindow: 32, rmsNormEps: 1e-5, ropeTheta: 500_000)
        return try MuseGlimmerDFlashDrafter(
            target: target,
            model: try MuseGlimmerDFlashFixture.model(
                configuration, seed: 5, dtype: target.activationDType),
            settings: .verifying(
                target, maximumSpeculativeRows: 2, batchesVerifyColumnsAsDecodeRows: true))
    }

    private func engine(_ m: MuseGlimmerTextModel, drafter: (any CBv2MTPDrafter)?) throws
        -> (EngineV2, PagedKVBackend)
    {
        m.prefillChunkAlignment = Self.chunk
        let backend = try PagedKVBackend(
            layerKinds: m.cbv2LayerKinds,
            config: .init(
                capacityBytes: 256 << 20, dtype: m.activationDType, maxPrefillChunk: Self.chunk,
                nominalMaxSequenceLength: 2048, segmentSizeBytes: 1 << 20,
                layerDTypes: m.kvStorageDTypes))
        let engine = EngineV2(
            model: MuseGlimmerCBv2Adapter(m), layerKinds: m.cbv2LayerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: Self.chunk,
                prefillChunkSize: Self.chunk, soloPrefillStripeTokens: Self.chunk, maxWaiting: 4,
                enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: 7, maxSpeculativeBatch: 2,
                fixedDraftTokens: 7, verificationMode: .rectangularExact,
                maxAutomaticRectangularTokens: 16))
        return (engine, backend)
    }

    private func collect(_ stream: AsyncStream<CBv2Event>) async -> CBv2SchedCollected {
        await cbv2SchedCollect(stream, timeoutSeconds: 180)
    }

    /// The coordinator's "text after an image": a text request beside an
    /// image request, then one after it on the same engine. The image
    /// request's own continuation is never drafted (media requests run on
    /// the target alone), so it is checked against the drafter-off run.
    @Test(.timeLimit(.minutes(8)))
    func textAfterAndBesideAnImageVerifiesExactly() async throws {
        let m = try target()
        typealias Prompt = MuseGlimmerImageEngineTests.ImagePrompt
        let image = Prompt(
            length: 120, spans: [.init(tokenOffset: 8, length: 60)], hidden: Self.hidden, salt: 7,
            seed: 23)
        let text = Prompt(length: 53, spans: [], hidden: Self.hidden, salt: 2, seed: 0)
        let requests = [
            image.request(1, maxTokens: Self.maxTokens), text.text(2, maxTokens: Self.maxTokens),
            text.text(3, maxTokens: Self.maxTokens), image.text(4, maxTokens: Self.maxTokens),
        ]
        var expected = [[Int]]()
        for request in requests {
            let (plain, _) = try engine(m, drafter: nil)
            let result = await collect(try plain.submit(request))
            await plain.shutdown()
            #expect(result.tokens.count == Self.maxTokens)
            expected.append(result.tokens)
        }
        #expect(expected[3] != expected[0], "the image must change the continuation")
        #expect(expected[1] == expected[2])

        for mode in [Scripted.Mode.native, .right, .wrong(at: 2)] {
            let scripted = Scripted(
                try drafter(m), mode: mode, scripts: [text.tokens + expected[1]], vocabulary: 96)
            let (engine, backend) = try engine(m, drafter: scripted)
            // Beside: the image and a text request share the engine.
            let first = try engine.submit(requests[0])
            let second = try engine.submit(requests[1])
            async let a = collect(first)
            async let b = collect(second)
            let together = await [a, b]
            #expect(together.map(\.tokens) == Array(expected[0 ... 1]), "\(mode)")
            let roundsBeside = try #require(engine.mtpMetricsSnapshot()).rounds
            // After: text on the engine that served the image.
            let late = await collect(try engine.submit(requests[2]))
            #expect(late.tokens == expected[2], "\(mode)")
            let metrics = try #require(engine.mtpMetricsSnapshot())
            #expect(metrics.verificationMode == .rectangularExact)
            #expect(metrics.serialVerificationRounds == 0)
            #expect(metrics.rounds > roundsBeside, "\(mode)")
            if mode == .right { #expect(metrics.acceptedTokens == metrics.draftedTokens) }
            // Drafter state for the two text requests only, all released.
            #expect(scripted.states.count == 2, "\(mode)")
            engine.loopForTesting.onEngineQueueSync {
                #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
            }
            #expect(scripted.states.allSatisfy { $0.released })
            await engine.shutdown()
            #expect(backend.bytesReserved == 0)
        }
    }
}
