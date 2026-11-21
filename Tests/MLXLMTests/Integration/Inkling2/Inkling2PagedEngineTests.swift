import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny Inkling 2 through the CBv2 engine on segmented paged storage: 48-style
/// slot layout (10 slots here), sliding and full paged layers, conv states as
/// request-owned recurrent state.
@Suite(.serialized, .tags(.integration))
struct Inkling2PagedEngineTests {
    static let vocabularySize = 64

    /// head_dim 64 keeps the layers eligible for the paged kernel geometry check.
    static func model(dtype: DType, seed: UInt64 = 3) throws -> Inkling2Model {
        let config: [String: Any] = [
            "model_type": "inkling2_loop",
            "text_config": [
                "hidden_size": 128, "num_hidden_layers": 6, "vocab_size": vocabularySize,
                "unpadded_vocab_size": 60, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 64, "d_rel": 4, "rel_extent": 24, "log_scaling_n_floor": 8,
                "log_scaling_alpha": 0.5, "qk_l2_clip_threshold": 8.0,
                "rel_l2_clip_threshold": 2.0, "local_layer_ids": [0, 2, 4],
                "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
                "sliding_window_size": 24, "n_layers_per_attn_res_block": 2,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32,
                "loop_config": ["core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8],
            ] as [String: Any],
        ]
        let model = Inkling2Model(
            try SyntheticModel.configuration(Inkling2Configuration.self, config))
        SyntheticModel.randomize(model, seed: seed)
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
        // The loader's lifecycle: concatenated row-exact projections.
        try model.prepare()
        eval(model)
        return model
    }

    static func engine(
        _ model: Inkling2Model, dtype: DType, chunk: Int = 16, concurrent: Int = 4,
        store: CompleteCheckpointFixtureStore? = nil, budget: Int = 256
    ) throws -> (EngineV2, PagedKVBackend) {
        let kinds = model.cbv2LayerKinds
        let types = Array(repeating: dtype, count: kinds.count)
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, dtype: dtype, maxPrefillChunk: max(chunk, 32),
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20, layerDTypes: types))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: concurrent, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, maxWaiting: 8, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            completeCheckpointKVDTypes: store == nil ? nil : types)
        return (engine, paged)
    }

    static func prompt(_ seed: Int, count: Int) -> [Int] {
        SyntheticModel.tokens(count: count, vocabularySize: 60, seed: seed)
    }

    static func generate(_ engine: EngineV2, id: UInt64, prompt: [Int], tokens: Int)
        async throws -> [Int]
    {
        let stream = try engine.submit(
            .init(
                id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0),
                maxTokens: tokens, prefixCacheEnabled: false))
        let collected = await cbv2SchedCollect(stream, timeoutSeconds: 120)
        return collected.tokens
    }

    /// Greedy continuation by full recompute of the whole sequence each step.
    static func noCacheGreedy(_ model: Inkling2Model, prompt: [Int], tokens: Int) -> [Int] {
        var sequence = prompt
        var out: [Int] = []
        for _ in 0 ..< tokens {
            let logits = model.forwardNoCache(
                MLXArray(sequence.map(Int32.init)).reshaped(1, sequence.count)
            ).logits
            let next = argMax(logits[0, -1], axis: -1).item(Int.self)
            out.append(next)
            sequence.append(next)
        }
        return out
    }

    @Test func pagedGreedyMatchesTheFullRecompute() async throws {
        let model = try Self.model(dtype: .float32)
        let prompt = Self.prompt(1, count: 45)
        let expected = Self.noCacheGreedy(model, prompt: prompt, tokens: 12)
        let (engine, backend) = try Self.engine(model, dtype: .float32)
        let tokens = try await Self.generate(engine, id: 1, prompt: prompt, tokens: 12)
        await engine.shutdown()
        #expect(tokens == expected)
        #expect(backend.bytesInUse == 0)
    }

    @Test(arguments: [8, 16, 32])
    func chunkSizesGiveTheSameTokens(chunk: Int) async throws {
        let model = try Self.model(dtype: .float32)
        let prompt = Self.prompt(2, count: 53)
        let (reference, _) = try Self.engine(model, dtype: .float32, chunk: 64)
        let expected = try await Self.generate(reference, id: 1, prompt: prompt, tokens: 10)
        await reference.shutdown()
        let (engine, _) = try Self.engine(model, dtype: .float32, chunk: chunk)
        let tokens = try await Self.generate(engine, id: 2, prompt: prompt, tokens: 10)
        await engine.shutdown()
        #expect(tokens == expected)
    }

    static func prefixRequest(_ id: UInt64, prompt: [Int], salt: String = "inkling2-fixture")
        -> CBv2Request
    {
        .init(
            id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0, seed: 0),
            maxTokens: 8, cacheSalt: salt, prefixCacheReceiptID: .init(id + 1000),
            prefixCheckpointTargetTokens: 256)
    }

    /// A checkpoint at 256 (window 24 has wrapped many times; 10 slots of conv
    /// state) restores into a fresh model and engine and continues exactly as
    /// the cold run; another tenant's salt never stages it.
    @Test func prefixRestoreMatchesCold() async throws {
        let prompt = Self.prompt(5, count: 300)
        let seed = CompleteCheckpointFixtureStore(segmentBytes: 4093)
        let (cold, coldBackend) = try Self.engine(
            try Self.model(dtype: .float32), dtype: .float32, chunk: 64, store: seed)
        let coldResult = await cbv2SchedCollect(
            try cold.submit(Self.prefixRequest(1, prompt: prompt)), timeoutSeconds: 120)
        seed.finishPublicationCallbacks(engine: cold)
        await cold.shutdown()
        try #require(coldResult.finishReason == .length && coldResult.tokens.count == 8)
        #expect(coldBackend.bytesReserved == 0)
        let archives = seed.saved.filter { $0.manifest.position == 256 }
        try #require(!archives.isEmpty)
        #expect(
            archives[0].manifest.backendLayout
                == CBv2CompleteCheckpointManifest.pagedHistoricalRecurrentLayout)

        let restored = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 4093)
        let (warm, warmBackend) = try Self.engine(
            try Self.model(dtype: .float32), dtype: .float32, chunk: 64, store: restored)
        #expect(
            try !restored.stage(
                engine: warm, request: Self.prefixRequest(2, prompt: prompt, salt: "other")))
        let input = Self.prefixRequest(3, prompt: prompt)
        try #require(try restored.stage(engine: warm, request: input))
        let warmResult = await cbv2SchedCollect(try warm.submit(input), timeoutSeconds: 120)
        restored.finishPublicationCallbacks(engine: warm)
        await warm.shutdown()
        #expect(warmResult.tokens == coldResult.tokens)
        #expect(warmResult.usage?.prefixCachePrefillTokensSaved == 256)
        #expect(warmBackend.bytesReserved == 0 && warm.admissionForTesting.bytesReserved == 0)
    }

    /// One request's paged rows, layer caches and recurrent state.
    struct DirectRequest {
        let caches: [PagedLayerCache]
        let rows: [CBv2SequenceKV]
        let state: CBv2RecurrentRequestState

        func rebind() { for (cache, row) in zip(caches, rows) { cache.setRows([row]) } }
    }

    static func directRequest(_ model: Inkling2Model, _ paged: PagedKVBackend) throws
        -> DirectRequest
    {
        let rows = try paged.makeSequenceState(
            layerKinds: model.cbv2LayerKinds, promptLength: 24, maxLength: 64
        ).map { $0! }
        let request = DirectRequest(
            caches: paged.makeLayerCaches(), rows: rows,
            state: try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec))
        request.rebind()
        return request
    }

    /// Runs `tokens` and evaluates; the caller commits or rolls back.
    static func directStep(
        _ model: Inkling2Model, _ request: DirectRequest, _ tokens: [Int], captured: Bool = false
    ) throws -> (logits: MLXArray, evaluation: CBv2RecurrentStateEvaluation) {
        let evaluation = try request.state.bind()
        let (logits, _) = model.cbv2ForwardCapturing(
            MLXArray(tokens.map { Int32($0) }).reshaped(1, -1),
            caches: request.caches.map { $0 as KVCache }, recurrentState: [evaluation],
            positionIds: nil, captureVisits: [], capturesPositions: captured)
        eval(try evaluation.evaluate() + [logits])
        return (logits, evaluation)
    }

    /// Prompt-chunk lengths of one model's engine steps (`Inkling2Diagnostics`).
    final class PrefillLengths: @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [Int] = []

        let model: Inkling2Model

        init(model: Inkling2Model) {
            self.model = model
            Inkling2Diagnostics.setStepHook(for: model) { [self] kind, length in
                guard kind == "prefill" else { return }
                lock.withLock { recorded.append(length) }
            }
        }

        func stop() { Inkling2Diagnostics.setStepHook(for: model, nil) }

        /// The lengths recorded since the last call.
        func take() -> [Int] {
            lock.withLock {
                defer { recorded = [] }
                return recorded
            }
        }
    }

    /// A prompt admitted while other rows decode yields exactly the tokens it
    /// yields alone. With fixed-row prefill products the scheduler may shorten
    /// its chunks to the budget the decoding rows leave; with MLX's own
    /// products the fixed grid cuts it at multiples of the chunk size, still
    /// progressing every step.
    @Test(arguments: [nil, 64] as [Int?])
    func promptChunksBesideDecodingRowsKeepTheirTokens(blockRows: Int?) async throws {
        let model = try Self.model(dtype: .bfloat16)
        model.invariantPrefillBlockRows = blockRows
        let long = Self.prompt(70, count: 70)
        let lengths = PrefillLengths(model: model)
        defer { lengths.stop() }
        let (aloneEngine, _) = try Self.engine(
            model, dtype: .bfloat16, chunk: 16, concurrent: 1, budget: 18)
        let alone = try await Self.generate(aloneEngine, id: 1, prompt: long, tokens: 8)
        await aloneEngine.shutdown()
        #expect(lengths.take() == [16, 16, 16, 16, 6])

        // Three 8-token rows decode while the long prompt prefills: their 3
        // decode tokens leave 15 of the 18-token budget, one short of a chunk.
        let (engine, _) = try Self.engine(
            model, dtype: .bfloat16, chunk: 16, concurrent: 4, budget: 18)
        var shorts: [AsyncStream<CBv2Event>] = []
        for i in 0 ..< 3 {
            shorts.append(
                try engine.submit(
                    .init(
                        id: .init(UInt64(10 + i)), promptTokens: Self.prompt(20 + i, count: 8),
                        sampling: .init(temperature: 0), maxTokens: 40,
                        prefixCacheEnabled: false)))
        }
        for stream in shorts {
            for await event in stream {
                if case .delta = event { break }
            }
        }
        _ = lengths.take()  // the short prompts
        let mixed = try await Self.generate(engine, id: 2, prompt: long, tokens: 8)
        await engine.shutdown()
        let chunks = lengths.take()
        #expect(mixed == alone, "chunks \(chunks)")
        if blockRows != nil {
            // The long prompt took chunks off the 16-token grid the alone run
            // used (decoding rows left 15).
            #expect(chunks.reduce(0, +) == 70, "chunks \(chunks)")
            #expect(chunks.contains { !$0.isMultiple(of: 16) && $0 != 6 }, "chunks \(chunks)")
        } else {
            // Every prompt chunk ends on the 16-token grid (70 = 4 x 16 + 6)
            // even though decoding rows left only 15 tokens.
            #expect(chunks == [16, 16, 16, 16, 6])
        }
    }

    /// Chunk-invariant prefill makes a restore at a boundary off the engine's
    /// chunk grid lossless: a checkpoint at 256 (taken by a 64-token-chunk
    /// engine) restores into an engine of 96-token chunks, whose cold prefill
    /// has boundaries at 96, 192 and 288 only, and continues exactly as that
    /// cold prefill: for the same prompt (44 tokens after the restore), a
    /// follow-up prompt extending it (84) and a prompt ending 10 tokens after it.
    @Test func prefixHitOffTheChunkGridMatchesCold() async throws {
        let prompt = Self.prompt(9, count: 300)
        let followUp = prompt + Self.prompt(10, count: 40)
        let seed = CompleteCheckpointFixtureStore(segmentBytes: 4093)
        let seedModel = try Self.model(dtype: .bfloat16)
        seedModel.invariantPrefillBlockRows = 64
        let (seeding, _) = try Self.engine(
            seedModel, dtype: .bfloat16, chunk: 64, store: seed)
        let seeded = await cbv2SchedCollect(
            try seeding.submit(Self.prefixRequest(1, prompt: prompt)), timeoutSeconds: 120)
        seed.finishPublicationCallbacks(engine: seeding)
        await seeding.shutdown()
        try #require(seeded.tokens.count == 8)
        let archives = seed.saved.filter { $0.manifest.position == 256 }
        try #require(!archives.isEmpty, "archived at \(seed.saved.map { $0.manifest.position })")

        let model = try Self.model(dtype: .bfloat16)
        model.invariantPrefillBlockRows = 64
        let lengths = PrefillLengths(model: model)
        defer { lengths.stop() }
        // A 10-token tail after the restore, as well as 44 and 84.
        let shortTail = Array(prompt[0 ..< 266])
        for (index, tokens) in [prompt, followUp, shortTail].enumerated() {
            let id = UInt64(10 + 10 * index)
            let (cold, _) = try Self.engine(model, dtype: .bfloat16, chunk: 96)
            let coldTokens = try await Self.generate(cold, id: id, prompt: tokens, tokens: 8)
            await cold.shutdown()
            var boundaries: [Int] = []
            for length in lengths.take() { boundaries.append((boundaries.last ?? 0) + length) }
            #expect(!boundaries.contains(256), "cold boundaries \(boundaries)")

            let restored = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 4093)
            let (warm, warmBackend) = try Self.engine(
                model, dtype: .bfloat16, chunk: 96, store: restored)
            let input = Self.prefixRequest(id + 1, prompt: tokens)
            try #require(try restored.stage(engine: warm, request: input))
            let warmResult = await cbv2SchedCollect(try warm.submit(input), timeoutSeconds: 120)
            restored.finishPublicationCallbacks(engine: warm)
            await warm.shutdown()
            let warmChunks = lengths.take()
            #expect(warmResult.usage?.prefixCachePrefillTokensSaved == 256)
            #expect(warmChunks.reduce(0, +) == tokens.count - 256, "warm chunks \(warmChunks)")
            #expect(warmResult.tokens == coldTokens, "prompt of \(tokens.count) tokens")
            #expect(warmBackend.bytesReserved == 0)
        }
    }

    /// Chunk-invariant prefill: a prompt prefilled in chunks of any length
    /// (1 token included) leaves exactly the state of the one-shot prefill:
    /// the same last-position logits and the same logits on every following
    /// decode step, bit for bit.
    @Test(arguments: [1, 3, 7, 16, 17, 33])
    func prefillIsChunkInvariant(chunk: Int) throws {
        let model = try Self.model(dtype: .bfloat16)
        model.invariantPrefillBlockRows = 64
        let kinds = model.cbv2LayerKinds
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, dtype: .bfloat16, maxPrefillChunk: 64,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: .bfloat16, count: kinds.count)))
        let prompt = Self.prompt(31, count: 50)
        func run(_ size: Int) throws -> [MLXArray] {
            let rows = try paged.makeSequenceState(
                layerKinds: kinds, promptLength: prompt.count, maxLength: prompt.count + 8
            ).map { $0! }
            let caches = paged.makeLayerCaches()
            for (cache, row) in zip(caches, rows) { cache.setRows([row]) }
            let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
            var outputs: [MLXArray] = []
            var start = 0
            while start < prompt.count {
                let end = min(prompt.count, start + size)
                let evaluation = try state.bind()
                let out = model.cbv2RecurrentPrefill(
                    MLXArray(prompt[start ..< end].map { Int32($0) }).reshaped(1, -1),
                    inputEmbedding: nil, cache: caches.map { $0 as KVCache },
                    recurrentState: [evaluation], positionIds: nil,
                    requirement: end == prompt.count ? .lastPositionLogits : .evaluationOnly)
                eval(try evaluation.evaluate() + [out])
                try evaluation.commit()
                if end == prompt.count { outputs.append(out) }
                start = end
            }
            var token = argMax(outputs[0], axis: -1).reshaped(1, 1).asType(.int32)
            for _ in 0 ..< 4 {
                let evaluation = try state.bind()
                let logits = model.cbv2Forward(
                    token, caches: caches.map { $0 as KVCache }, recurrentState: [evaluation],
                    positionIds: nil)
                eval(try evaluation.evaluate() + [logits])
                try evaluation.commit()
                outputs.append(logits[0, -1])
                token = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
            }
            paged.release(rows.map { Optional($0) })
            try state.release()
            return outputs
        }
        let oneShot = try run(prompt.count)
        let chunked = try run(chunk)
        for (i, (a, b)) in zip(oneShot, chunked).enumerated() {
            #expect(arrayEqual(a, b).item(Bool.self), "output \(i) at chunk \(chunk)")
        }
    }

    /// Records each attention call's `isPrefill` and forwards to the model's core.
    final class FlagRecorder: Inkling2PagedAttentionCore {
        let inner: any Inkling2PagedAttentionCore
        var flags: [Bool] = []
        init(_ inner: any Inkling2PagedAttentionCore) { self.inner = inner }
        func attend(
            slot: Int, cache: any CBv2AttendingLayerCache, rowOffsets: [Int],
            _ inputs: Inkling2AttentionInputs
        ) -> MLXArray {
            flags.append(inputs.isPrefill)
            return inner.attend(slot: slot, cache: cache, rowOffsets: rowOffsets, inputs)
        }
    }

    /// Prompt chunks of any length report `isPrefill` (so the kernel core
    /// picks its prefill arithmetic); decode steps do not.
    @Test func promptChunksOfAnyLengthAreMarkedPrefill() throws {
        let model = try Self.model(dtype: .bfloat16)
        let kinds = model.cbv2LayerKinds
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, dtype: .bfloat16, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: .bfloat16, count: kinds.count)))
        let recorder = FlagRecorder(model.pagedAttentionCore)
        let saved = model.pagedAttentionCore
        model.pagedAttentionCore = recorder
        defer { model.pagedAttentionCore = saved }
        let request = try Self.directRequest(model, paged)
        for chunk in [Self.prompt(5, count: 8), Self.prompt(6, count: 16)] {
            let evaluation = try request.state.bind()
            let out = model.cbv2RecurrentPrefill(
                MLXArray(chunk.map { Int32($0) }).reshaped(1, -1), inputEmbedding: nil,
                cache: request.caches.map { $0 as KVCache }, recurrentState: [evaluation],
                positionIds: nil, requirement: .lastPositionLogits)
            eval(try evaluation.evaluate() + [out])
            try evaluation.commit()
            request.rebind()
        }
        #expect(!recorder.flags.isEmpty && recorder.flags.allSatisfy { $0 })
        recorder.flags = []
        let (_, evaluation) = try Self.directStep(model, request, [7])
        try evaluation.commit()
        #expect(!recorder.flags.isEmpty && recorder.flags.allSatisfy { !$0 })
        #expect(
            Inkling2KernelAttentionCore.kernel(
                queries: 8, prefill: false, writes: false, dtype: .bfloat16, headDim: 64,
                relativeWidth: 4, window: nil) == .automatic)
        if Inkling2PagedAttention.neuralAcceleratorsAvailable {
            #expect(
                Inkling2KernelAttentionCore.kernel(
                    queries: 8, prefill: true, writes: false, dtype: .bfloat16, headDim: 64,
                    relativeWidth: 4, window: nil) == .prefill)
        }
        paged.release(request.rows.map { Optional($0) })
        try request.state.release()
    }

    /// A captured verify window committed at its accepted prefix (pages of the
    /// rejected tail rolled back) continues exactly like plain decode steps of
    /// that prefix, and each verified position equals the decode step at it.
    @Test func capturedVerifyWindowKeepsTheAcceptedPrefix() throws {
        let model = try Self.model(dtype: .bfloat16)
        let kinds = model.cbv2LayerKinds
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, dtype: .bfloat16, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: .bfloat16, count: kinds.count)))
        let prompt = Self.prompt(9, count: 22)
        let window = Self.prompt(10, count: 5)
        let verified = try Self.directRequest(model, paged)
        let decoded = try Self.directRequest(model, paged)
        for request in [verified, decoded] {
            try Self.directStep(model, request, prompt).evaluation.commit()
            request.rebind()
        }

        let (windowLogits, evaluation) = try Self.directStep(
            model, verified, window, captured: true)
        try evaluation.commit(keepPositions: 3)
        for row in verified.rows { row.rollback(2) }
        verified.rebind()

        for (i, token) in window.prefix(3).enumerated() {
            let (logits, evaluation) = try Self.directStep(model, decoded, [token])
            try evaluation.commit()
            decoded.rebind()
            #expect(arrayEqual(logits[0, 0], windowLogits[0, i]).item(Bool.self))
        }
        let next = [window[4]]
        let (a, ea) = try Self.directStep(model, verified, next)
        let (b, eb) = try Self.directStep(model, decoded, next)
        try ea.commit()
        try eb.commit()
        #expect(arrayEqual(a, b).item(Bool.self))
        for slot in 0 ..< kinds.count {
            let x = verified.state.state(modelLayerIndex: slot)!
            let y = decoded.state.state(modelLayerIndex: slot)!
            #expect(arrayEqual(x.conv!, y.conv!).item(Bool.self))
            #expect(arrayEqual(x.ssm!, y.ssm!).item(Bool.self))
        }
        for request in [verified, decoded] {
            paged.release(request.rows.map { Optional($0) })
            try request.state.release()
        }
    }

    /// Compiled elementwise chains change no bits of paged prefill or decode steps.
    @Test(arguments: [DType.bfloat16, .float32])
    func fusedDecodeMatchesPlainDecodeBitForBit(dtype: DType) throws {
        let model = try Self.model(dtype: dtype)
        let kinds = model.cbv2LayerKinds
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, dtype: dtype, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: dtype, count: kinds.count)))
        let requests = [try Self.directRequest(model, paged), try Self.directRequest(model, paged)]
        let modes = [false, true]
        var tokens = [Self.prompt(11, count: 21), Self.prompt(11, count: 21)]
        for step in 0 ..< 6 {
            var logits: [MLXArray] = []
            for (request, fused) in zip(requests, modes) {
                let (out, evaluation) = try Inkling2Fused.$override.withValue(fused) {
                    try Self.directStep(model, request, tokens[0])
                }
                try evaluation.commit()
                request.rebind()
                logits.append(out)
            }
            #expect(arrayEqual(logits[0], logits[1]).item(Bool.self), "step \(step)")
            let next = argMax(logits[0][0, -1], axis: -1).item(Int.self)
            tokens = [[next], [next]]
        }
        for request in requests {
            paged.release(request.rows.map { Optional($0) })
            try request.state.release()
        }
    }

    /// Rows decoded together are bit-identical to the same rows decoded alone.
    @Test(arguments: [2, 4])
    func batchedRowsMatchSingleRows(rows: Int) async throws {
        let model = try Self.model(dtype: .bfloat16)
        let prompts = (0 ..< rows).map { Self.prompt(10 + $0, count: 9 + 11 * $0) }
        var alone: [[Int]] = []
        for (i, prompt) in prompts.enumerated() {
            let (engine, _) = try Self.engine(model, dtype: .bfloat16, concurrent: 1)
            alone.append(
                try await Self.generate(engine, id: UInt64(i + 1), prompt: prompt, tokens: 16))
            await engine.shutdown()
        }
        let (engine, backend) = try Self.engine(model, dtype: .bfloat16, concurrent: rows)
        let streams = try prompts.enumerated().map { i, prompt in
            try engine.submit(
                .init(
                    id: .init(UInt64(100 + i)), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: 16, prefixCacheEnabled: false))
        }
        let together = await withTaskGroup(of: (Int, [Int]).self) { group in
            for (i, stream) in streams.enumerated() {
                group.addTask { (i, await cbv2SchedCollect(stream, timeoutSeconds: 120).tokens) }
            }
            var result = [[Int]](repeating: [], count: rows)
            for await (i, tokens) in group { result[i] = tokens }
            return result
        }
        await engine.shutdown()
        #expect(together == alone)
        #expect(backend.bytesInUse == 0)
    }
}
