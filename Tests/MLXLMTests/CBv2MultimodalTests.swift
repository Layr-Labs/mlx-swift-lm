// CBv2MultimodalTests.swift — WS-G: vision (multimodal) prefill for the
// ContinuousBatchingV2 engine.
//
// A vision request carries precomputed image EMBEDDINGS (the vision tower +
// projector output the provider computes) plus the placeholder-token spans
// they replace. The engine splices those embeddings into the text-token
// embeddings and prefills the span-containing chunks under a NEW pinned
// attention path: causal(∧window) PLUS bidirectional within each image
// block, matching MLXVLM Gemma4's `gemma4BidirectionalVisionMask` overlay
// exactly. Decode after prefill is unchanged (image tokens are history in
// KV by then).
//
// This suite proves:
//  (a) TOKEN-EXACT: the v2 chunked+batched path == an independent
//      full-sequence non-chunked reference (mimicking the MLXVLM wrapper
//      semantics), for spans at chunk-boundary-adjacent positions, spans
//      that would straddle a naive boundary (snapped), multiple spans, a
//      span at position 0, and the KV-sharing model shape;
//  (b) BATCH INVARIANCE: a vision request batched with text neighbors is
//      token-exact vs its solo run, and the neighbors are unaffected;
//  (c) TEXT REGRESSION: text-only requests are byte-identical with the
//      feature compiled in (a no-multimodal request takes the untouched
//      path);
//  (d) PREFIX-CACHE EXCLUSION: vision requests neither adopt nor donate;
//  (e) SUBMIT-TIME REJECTIONS: bad spans, oversized block, unsupported
//      model, unsupported backend, embedding shape mismatch.
//
// The reference is INDEPENDENT of the engine's span-mask code: it rebuilds
// the block-bidirectional mask from span coordinates and runs full-sequence
// attention through the fixture model's own weights.
//
// No model downloads; weights + synthetic image embeddings are seeded.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

// MARK: - Independent full-sequence vision reference

/// Full-sequence, non-chunked greedy reference that mimics the MLXVLM Gemma4
/// wrapper: splice image embeddings at placeholder spans, attend with a
/// single causal(∧window) mask overlaid with block-bidirectional attention,
/// recompute the whole sequence per decode step. Uses TinyTestModel's own
/// weights but NONE of the engine's span-mask/splice code.
enum CBv2VisionReference {

    /// One image span's synthetic embeddings ([1, length, hidden]).
    struct Image {
        let span: CBv2ImageSpan
        let embedding: MLXArray
    }

    /// Boolean [1,1,T,T] mask: causal, ∧ window for sliding layers, ∨
    /// bidirectional within each block. Independent reimplementation of
    /// `gemma4BidirectionalVisionMask` from block coordinates.
    static func mask(length T: Int, window: Int?, blocks: [CBv2ImageSpan]) -> MLXArray {
        let q = MLXArray(Int32(0) ..< Int32(T)).expandedDimensions(axis: 1)  // [T,1]
        let k = MLXArray(Int32(0) ..< Int32(T)).expandedDimensions(axis: 0)  // [1,T]
        var m = k .<= q
        if let window { m = m .&& (k .> (q - Int32(window))) }
        for block in blocks {
            let lo = Int32(block.tokenOffset)
            let hi = Int32(block.end)
            let qIn = (q .>= lo) .&& (q .< hi)
            let kIn = (k .>= lo) .&& (k .< hi)
            m = m .|| (qIn .&& kIn)
        }
        return m.reshaped(1, 1, T, T)
    }

    /// Splice image embeddings over the token embeddings ([1,T,hidden]).
    static func splice(_ emb: MLXArray, images: [Image]) -> MLXArray {
        var out = emb
        for image in images {
            let lo = image.span.tokenOffset
            let hi = image.span.end
            let pre = out[0..., ..<lo, 0...]
            let post = out[0..., hi..., 0...]
            out = concatenated([pre, image.embedding.asType(out.dtype), post], axis: 1)
        }
        return out
    }

    /// Last-position logits over the full token sequence.
    static func logits(model: TinyTestModel, tokens: [Int], images: [Image]) -> MLXArray {
        let T = tokens.count
        let blocks = CBv2MultimodalPlan.coalescedBlocks(spans: images.map(\.span))
        var h = model.embed(MLXArray(tokens.map(Int32.init)).reshaped(1, T))
        h = splice(h, images: images)

        var kvByLayer = [(keys: MLXArray, values: MLXArray)?](repeating: nil, count: model.blocks.count)
        for (i, block) in model.blocks.enumerated() {
            let kind = model.layerKinds[i]
            let window: Int?
            if case .slidingWindow(let w) = kind.attention { window = w } else { window = nil }
            let m = mask(length: T, window: window, blocks: blocks)
            let shared = kind.sharesKVWithLayer.flatMap { kvByLayer[$0] }
            let (hOut, keys, values) = block.referenceForward(h, maskArray: m, sharedKV: shared)
            h = hOut
            kvByLayer[i] = (keys, values)
        }
        return model.lmHead(model.finalNorm(h))[0..., -1, 0...]
    }

    /// Greedy decode: prefill `prompt`, then extend `decodeSteps` tokens.
    /// The first sampled token is the argmax at the last prompt position —
    /// same greedy discipline as the v2 driver (prefill prompt[0..<n-1], the
    /// last prompt token is the first decode input).
    static func greedy(
        model: TinyTestModel, prompt: [Int], images: [Image], decodeSteps: Int
    ) -> [Int] {
        var sequence = prompt
        var generated: [Int] = []
        for _ in 0 ..< decodeSteps {
            let last = logits(model: model, tokens: sequence, images: images)
            let next = Int(argMax(last, axis: -1).asArray(Int32.self)[0])
            generated.append(next)
            sequence.append(next)
        }
        return generated
    }
}

// MARK: - Synthetic image fixtures

enum CBv2VisionFixtures {
    static let placeholderTokenID = 7  // any vocab id; its embedding is replaced

    /// A prompt of `length` tokens with placeholder ids at `spans`, and
    /// deterministic synthetic image embeddings for each span.
    static func make(
        model: TinyTestModel, length: Int, spans: [CBv2ImageSpan], seed: UInt64
    ) -> (prompt: [Int], images: [CBv2VisionReference.Image]) {
        var prompt = makePromptTokens(length: length, seed: seed)
        for span in spans {
            for p in span.tokenOffset ..< span.end { prompt[p] = placeholderTokenID }
        }
        MLXRandom.seed(seed ^ 0xA11CE)
        let hidden = model.config.hiddenSize
        var images: [CBv2VisionReference.Image] = []
        for span in spans {
            let embedding = MLXRandom.normal([1, span.length, hidden]) * 0.7
            eval(embedding)
            images.append(.init(span: span, embedding: embedding))
        }
        return (prompt, images)
    }

    static func input(
        _ images: [CBv2VisionReference.Image]
    ) -> CBv2MultimodalInput {
        CBv2MultimodalInput(spans: images.map(\.span)) { images.map(\.embedding) }
    }
}

// MARK: - Synchronous v2 driver (real CBv2LayerCache + contiguous backend)

/// A deterministic greedy driver over the PRODUCTION contiguous layer cache
/// (`CBv2LayerCache`, which honors span-mask binding) and KV backend, with
/// explicit control of prefill chunk sizes and batch composition. Chunk
/// planning uses the real `CBv2ScheduledRequest.snappedChunkTokens`, so span
/// blocks are never split — the same logic the engine's scheduler runs.
final class CBv2VisionDriver {
    struct Req {
        let record: CBv2ScheduledRequest
        var tokens: [Int]
        var kv: [CBv2SequenceKV?]
        let multimodal: CBv2ResolvedMultimodal?
        var last: Int
        var generated: [Int] = []
    }

    let model: TinyTestModel
    let layerKinds: [CBv2LayerKind]
    let backend: CBv2ContiguousKVBackend
    let caches: [CBv2LayerCache]
    private(set) var active: [Req] = []
    private var nextID: UInt64 = 0

    init(model: TinyTestModel) {
        self.model = model
        self.layerKinds = model.layerKinds
        self.backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27))
        self.caches = layerKinds.enumerated().map { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind)
        }
    }

    private func rebuildRows() {
        for (i, cache) in caches.enumerated() where cache.kind.sharesKVWithLayer == nil {
            cache.setRows(active.map { $0.kv[i]! })
        }
    }

    /// Admit one request; chunked prefill of prompt[0..<n-1] with span
    /// snapping + span-mask binding, then join the decode batch.
    @discardableResult
    func admit(
        prompt: [Int], maxTokens: Int, chunkSize: Int,
        multimodal: CBv2ResolvedMultimodal? = nil
    ) throws -> CBv2RequestID {
        let id = CBv2RequestID(nextID)
        nextID += 1
        var request = CBv2Request(id: id, promptTokens: prompt, maxTokens: maxTokens)
        if let multimodal {
            request.multimodal = CBv2MultimodalInput(spans: multimodal.spans) { [] }
        }
        let record = CBv2ScheduledRequest(
            request: request, arrivalSeq: nextID, submittedAt: Date(), deadline: nil)

        let kv = try backend.makeSequenceState(
            layerKinds: layerKinds, promptLength: prompt.count,
            maxLength: prompt.count + maxTokens + 1)
        let soloCaches = layerKinds.enumerated().map { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind, rows: kind.sharesKVWithLayer == nil ? [kv[index]!] : [])
        }

        let prefillLen = prompt.count - 1
        var start = 0
        while start < prefillLen {
            let proposed = min(chunkSize, prefillLen - start)
            let n = record.snappedChunkTokens(start: start, proposed: proposed, budget: 1 << 20)
            XCTAssertGreaterThan(n, 0, "snapped chunk must make progress with a large budget")
            let slice = Array(prompt[start ..< start + n])
            let inputs = MLXArray(slice.map(Int32.init)).reshaped(1, n)
            if let multimodal, let context = multimodal.chunkContext(start: start, count: n) {
                let textEmb = model.embedPromptTokens(inputs)
                let spliced = CBv2MultimodalPlan.spliceEmbeddings(
                    textEmbeddings: textEmb, chunkStart: start,
                    spans: multimodal.spansInChunk(start: start, count: n))
                // Mirror the engine: a 1-token chunk splices its embedding but
                // is not span-mask-bound (a length-1 block is self-attention).
                let bind = n > 1
                if bind { for cache in soloCaches { cache.bindSpanContext(context) } }
                _ = model.forward(tokens: inputs, inputEmbeddings: spliced, caches: soloCaches)
                if bind { for cache in soloCaches { cache.bindSpanContext(nil) } }
            } else {
                _ = model.forward(tokens: inputs, caches: soloCaches)
            }
            start += n
        }

        active.append(
            Req(record: record, tokens: prompt, kv: kv, multimodal: multimodal, last: prompt.last!))
        rebuildRows()
        return id
    }

    /// One batched [B,1] greedy decode step. No span context (decode is
    /// always mask-free — image tokens are history in KV).
    @discardableResult
    func decodeStep() -> [Int] {
        let inputs = MLXArray(active.map { Int32($0.last) }).reshaped(active.count, 1)
        let logits = model.forward(tokens: inputs, caches: caches)
        let sampled = argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self).map(Int.init)
        for (i, token) in sampled.enumerated() {
            active[i].generated.append(token)
            active[i].last = token
        }
        return sampled
    }

    func generated(for id: CBv2RequestID) -> [Int] {
        active.first { $0.record.id == id }?.generated ?? []
    }

    func run(steps: Int) { for _ in 0 ..< steps { decodeStep() } }
}

// MARK: - Resolution helper (bypasses the async engine for driver tests)

extension CBv2MultimodalTests {
    func resolve(
        _ images: [CBv2VisionReference.Image], model: TinyTestModel,
        promptCount: Int, maxBatchedTokens: Int = 4096
    ) throws -> CBv2ResolvedMultimodal {
        try CBv2MultimodalPlan.resolve(
            CBv2VisionFixtures.input(images),
            promptTokenCount: promptCount,
            model: model,
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            maxBatchedTokensPerStep: maxBatchedTokens)
    }
}

// MARK: - Suite

final class CBv2MultimodalTests: XCTestCase {

    // MARK: (a) Token-exact chunked+batched vs full-sequence reference

    private func assertVisionTokenExact(
        model: TinyTestModel, prompt: [Int], images: [CBv2VisionReference.Image],
        chunkSize: Int, decodeSteps: Int, _ message: String
    ) throws {
        let reference = CBv2VisionReference.greedy(
            model: model, prompt: prompt, images: images, decodeSteps: decodeSteps)
        let resolved = try resolve(images, model: model, promptCount: prompt.count)
        let driver = CBv2VisionDriver(model: model)
        let id = try driver.admit(
            prompt: prompt, maxTokens: decodeSteps, chunkSize: chunkSize, multimodal: resolved)
        driver.run(steps: decodeSteps)
        XCTAssertEqual(driver.generated(for: id), reference, message)
    }

    func testSingleSpanMidPrompt() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let spans = [CBv2ImageSpan(tokenOffset: 4, length: 6)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 24, spans: spans, seed: 100)
        // chunkSize 8 puts a naive boundary at 8 — inside the span [4,10) —
        // so the scheduler must snap: chunk 0 = [0,4), chunk 1 = [4,10).
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 8, decodeSteps: 10,
            "single mid-prompt span, chunked prefill must match full-sequence reference")
    }

    func testSpanAtPositionZero() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let spans = [CBv2ImageSpan(tokenOffset: 0, length: 5)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 20, spans: spans, seed: 101)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 4, decodeSteps: 8,
            "span at position 0 must match reference")
    }

    func testSpanAtChunkEdge() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        // Span begins exactly at a chunk boundary (offset 8, chunkSize 8):
        // chunk 1 starts at the span and must EXTEND to cover it whole.
        let spans = [CBv2ImageSpan(tokenOffset: 8, length: 7)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 26, spans: spans, seed: 102)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 8, decodeSteps: 8,
            "span at a chunk edge must ride one chunk and match reference")
    }

    func testSpanStraddlingWouldBeBoundarySnapped() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        // Span [10,20) straddles naive boundaries at 12, 16 (chunkSize 6:
        // 0,6,12,18). The scheduler snaps so the whole span rides one chunk.
        let spans = [CBv2ImageSpan(tokenOffset: 10, length: 10)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 30, spans: spans, seed: 103)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 6, decodeSteps: 8,
            "a span that would straddle a boundary must be snapped whole")
    }

    func testMultipleSpans() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let spans = [
            CBv2ImageSpan(tokenOffset: 2, length: 4),
            CBv2ImageSpan(tokenOffset: 12, length: 5),
        ]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 32, spans: spans, seed: 104)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 8, decodeSteps: 10,
            "multiple spans must match reference")
    }

    func testAdjacentSpansCoalesceIntoOneBlock() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        // Two ADJACENT spans (no separator) attend bidirectionally as ONE
        // block — the reference coalesces them; the engine must too.
        let spans = [
            CBv2ImageSpan(tokenOffset: 5, length: 4),
            CBv2ImageSpan(tokenOffset: 9, length: 4),
        ]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 28, spans: spans, seed: 105)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 6, decodeSteps: 8,
            "adjacent spans must coalesce and attend as one bidirectional block")
    }

    func testKVSharingModelWithSpan() throws {
        // KV-sharing shape (Gemma-4 style): a windowed borrowing layer over a
        // windowed source, plus a downstream full layer whose KV caches the
        // borrowed outputs — the span mask must be honored on the borrowing
        // path too.
        let model = TinyTestModel.make(seed: 0xC0FFEE, withKVSharing: true)
        let spans = [CBv2ImageSpan(tokenOffset: 6, length: 8)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 34, spans: spans, seed: 106)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 10, decodeSteps: 8,
            "KV-sharing model with a vision span must match reference")
    }

    func testLengthOneSpanInTrailingOneTokenChunk() throws {
        // Regression: a length-1 image span positioned so the prefill splits
        // it into a trailing 1-token chunk must NOT trip the span-mask
        // precondition — the engine splices its embedding but skips the
        // (no-op) bidirectional bind. prompt 10, span [8,9), chunkSize 8 ⇒
        // chunk0 [0,8), chunk1 [8,9) L=1 carrying the span.
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let spans = [CBv2ImageSpan(tokenOffset: 8, length: 1)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 10, spans: spans, seed: 108)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 8, decodeSteps: 6,
            "length-1 span in a trailing 1-token chunk must match reference without trapping")
    }

    func testLengthOneSpanChunkSizeOne() throws {
        // Pathological but valid: prefillChunkSize 1 makes EVERY prefill
        // chunk 1 token, so a length-1 span is always isolated. Must still
        // splice + match reference.
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let spans = [CBv2ImageSpan(tokenOffset: 3, length: 1)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 12, spans: spans, seed: 109)
        try assertVisionTokenExact(
            model: model, prompt: prompt, images: images, chunkSize: 1, decodeSteps: 6,
            "length-1 span with chunkSize 1 must splice and match reference")
    }

    func testUnchunkedEqualsChunkedForVision() throws {
        // Same request, two chunk sizes: chunking must not change the output.
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let spans = [CBv2ImageSpan(tokenOffset: 7, length: 6)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 30, spans: spans, seed: 107)

        let resolvedA = try resolve(images, model: model, promptCount: prompt.count)
        let driverA = CBv2VisionDriver(model: model)
        let idA = try driverA.admit(
            prompt: prompt, maxTokens: 10, chunkSize: 64, multimodal: resolvedA)
        driverA.run(steps: 10)

        let resolvedB = try resolve(images, model: model, promptCount: prompt.count)
        let driverB = CBv2VisionDriver(model: model)
        let idB = try driverB.admit(
            prompt: prompt, maxTokens: 10, chunkSize: 5, multimodal: resolvedB)
        driverB.run(steps: 10)

        XCTAssertEqual(driverA.generated(for: idA), driverB.generated(for: idB))
    }

    // MARK: (b) Batch invariance — vision + text neighbors

    func testBatchInvarianceVisionWithTextNeighbors() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let spans = [CBv2ImageSpan(tokenOffset: 5, length: 6)]
        let (visionPrompt, images) = CBv2VisionFixtures.make(
            model: model, length: 26, spans: spans, seed: 200)
        let textA = makePromptTokens(length: 14, seed: 201)
        let textB = makePromptTokens(length: 33, seed: 202)
        let decodeSteps = 12

        // Solo references.
        let visionSolo = CBv2VisionReference.greedy(
            model: model, prompt: visionPrompt, images: images, decodeSteps: decodeSteps)
        let textASolo = soloText(model: model, prompt: textA, decodeSteps: decodeSteps)
        let textBSolo = soloText(model: model, prompt: textB, decodeSteps: decodeSteps)

        // Batched: vision + two text neighbors, admitted together.
        let resolved = try resolve(images, model: model, promptCount: visionPrompt.count)
        let driver = CBv2VisionDriver(model: model)
        let idV = try driver.admit(
            prompt: visionPrompt, maxTokens: decodeSteps, chunkSize: 8, multimodal: resolved)
        let idA = try driver.admit(prompt: textA, maxTokens: decodeSteps, chunkSize: 8)
        let idB = try driver.admit(prompt: textB, maxTokens: decodeSteps, chunkSize: 8)
        driver.run(steps: decodeSteps)

        XCTAssertEqual(
            driver.generated(for: idV), visionSolo,
            "vision request must be token-exact under batching with text neighbors")
        XCTAssertEqual(
            driver.generated(for: idA), textASolo, "text neighbor A must be unaffected")
        XCTAssertEqual(
            driver.generated(for: idB), textBSolo, "text neighbor B must be unaffected")
    }

    /// Text-only solo run through the same driver (no multimodal) — the
    /// batch-invariance baseline for the text neighbors.
    private func soloText(model: TinyTestModel, prompt: [Int], decodeSteps: Int) -> [Int] {
        let driver = CBv2VisionDriver(model: model)
        let id = try! driver.admit(prompt: prompt, maxTokens: decodeSteps, chunkSize: 8)
        driver.run(steps: decodeSteps)
        return driver.generated(for: id)
    }

    // MARK: (c) Text regression — identical with the feature compiled in

    func testTextOnlyUnaffectedByCompiledFeature() throws {
        // A text request routed through the SAME driver that supports vision
        // must be byte-identical to a run that never touches multimodal code
        // (CBv2HarnessEngine, the pre-feature harness).
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let prompt = makePromptTokens(length: 30, seed: 300)
        let decodeSteps = 16

        let harness = try CBv2HarnessEngine.runSolo(
            model: model, prompt: prompt, maxTokens: decodeSteps, prefillChunkSize: 8)

        let driver = CBv2VisionDriver(model: model)
        let id = try driver.admit(prompt: prompt, maxTokens: decodeSteps, chunkSize: 8)
        driver.run(steps: decodeSteps)

        XCTAssertEqual(
            driver.generated(for: id), harness,
            "text-only output must be byte-identical with the vision feature compiled in")
    }

    // MARK: (d) Prefix-cache exclusion — both directions, through EngineV2

    /// A full-attention model (no windowed layers ⇒ zero trailing recompute,
    /// so a text prefix actually adopts) is used so the positive controls
    /// register real cache hits; the exclusion assertions then isolate the
    /// vision-specific skip. Returns the cache so tests can synchronize on the
    /// ASYNC donation queue (donation lands after the stream finishes).
    private func makeMultimodalEngine(_ model: TinyTestModel) -> (EngineV2, PrefixCacheV2) {
        let cache = PrefixCacheV2(config: .init(blockSize: 8, modelName: "cbv2-mm"))
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 5),
            schedulerConfig: CBv2SchedulerConfig(
                maxBatchedTokensPerStep: 256, prefillChunkSize: 8, maxWaiting: 16,
                enablePrefixCache: true),
            prefixCache: cache)
        return (engine, cache)
    }

    private func textRequest(_ id: UInt64, prompt: [Int]) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: .init(temperature: 0), maxTokens: 4)
    }
    private func visionRequest(
        _ id: UInt64, prompt: [Int], images: [CBv2VisionReference.Image]
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: .init(temperature: 0), maxTokens: 4,
            multimodal: CBv2VisionFixtures.input(images))
    }

    func testPrefixCacheExcludesVisionAdoption() async throws {
        // Token ids match a cached TEXT prefix, but the vision request must
        // NOT adopt it — the ids are identical for every image, so a hit
        // would silently reuse the wrong KV.
        let model = TinyTestModel.make(seed: 0xC0FFEE, fullAttentionOnly: true)
        let spans = [CBv2ImageSpan(tokenOffset: 4, length: 6)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 24, spans: spans, seed: 400)
        let (engine, cache) = makeMultimodalEngine(model)

        let text1 = await cbv2SchedCollect(try engine.submit(textRequest(1, prompt: prompt)))
        XCTAssertEqual(text1.usage?.prefixCacheHitTokens, 0, "cold text: no hit")
        // Donation is async — wait for text1's prefix to be indexed.
        let donated = await cbv2SchedWait { cache.stats().entryCount >= 1 }
        XCTAssertTrue(donated, "text must donate its prefix")
        let text2 = await cbv2SchedCollect(try engine.submit(textRequest(2, prompt: prompt)))
        XCTAssertGreaterThan(
            text2.usage?.prefixCacheHitTokens ?? 0, 0, "control: text DOES adopt the cached prefix")
        let visionHit = await cbv2SchedCollect(
            try engine.submit(visionRequest(3, prompt: prompt, images: images)))
        XCTAssertEqual(
            visionHit.usage?.prefixCacheHitTokens, 0,
            "vision must NOT adopt a token-id-matching text prefix")
        await engine.shutdown()
    }

    func testPrefixCacheExcludesVisionDonation() async throws {
        // A finished vision request must not donate; a later text request with
        // the identical token prompt therefore gets no hit from the vision KV.
        let model = TinyTestModel.make(seed: 0xC0FFEE, fullAttentionOnly: true)
        let spans = [CBv2ImageSpan(tokenOffset: 4, length: 6)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: model, length: 24, spans: spans, seed: 401)
        let (engine, cache) = makeMultimodalEngine(model)

        let vision = await cbv2SchedCollect(
            try engine.submit(visionRequest(1, prompt: prompt, images: images)))
        XCTAssertEqual(vision.finishReason, .length)
        // Vision never donates, so nothing is ever indexed for it. A text
        // request with the SAME token ids therefore cannot hit — this is the
        // donation-exclusion assertion (text shares vision's placeholder ids).
        let text1 = await cbv2SchedCollect(try engine.submit(textRequest(2, prompt: prompt)))
        XCTAssertEqual(
            text1.usage?.prefixCacheHitTokens, 0,
            "no vision KV may be donated for a later same-token text request to hit")
        // Positive control: text DOES donate, so a second identical text hits —
        // proving the cache is live and only vision was excluded.
        let donated = await cbv2SchedWait { cache.stats().entryCount >= 1 }
        XCTAssertTrue(donated, "text must donate its prefix")
        let text2 = await cbv2SchedCollect(try engine.submit(textRequest(3, prompt: prompt)))
        XCTAssertGreaterThan(
            text2.usage?.prefixCacheHitTokens ?? 0, 0,
            "control: text donation is live — only vision is excluded")
        await engine.shutdown()
    }

    // MARK: - Splice + mask unit checks

    func testSpliceReplacesExactlyTheSpanRows() {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let hidden = model.config.hiddenSize
        let text = MLXArray(0 ..< Int32(6 * hidden)).asType(.float32).reshaped(1, 6, hidden)
        let embedding = MLXArray.full([1, 2, hidden], values: MLXArray(Float(-1)))
        let span = CBv2ImageSpan(tokenOffset: 2, length: 2)
        let spliced = CBv2MultimodalPlan.spliceEmbeddings(
            textEmbeddings: text, chunkStart: 0, spans: [(span: span, embedding: embedding)])
        eval(spliced)
        // Rows 0,1,4,5 unchanged; rows 2,3 == -1.
        XCTAssertTrue(allClose(spliced[0..., 0 ..< 2, 0...], text[0..., 0 ..< 2, 0...]).item(Bool.self))
        XCTAssertTrue(allClose(spliced[0..., 4 ..< 6, 0...], text[0..., 4 ..< 6, 0...]).item(Bool.self))
        XCTAssertTrue(
            allClose(spliced[0..., 2 ..< 4, 0...], embedding).item(Bool.self),
            "span rows must be replaced verbatim")
    }

    func testSpanChunkMaskMatchesReferenceOverlay() {
        // The engine's span mask (relative coords) must equal the reference's
        // full-sequence overlay restricted to the chunk's rows.
        let T = 12
        let window = 6
        let block = CBv2ImageSpan(tokenOffset: 3, length: 4)
        let full = CBv2VisionReference.mask(length: T, window: window, blocks: [block])
        // Chunk = whole sequence [0,T): kL == L == T, chunkEnd == T.
        let context = CBv2SpanChunkContext(chunkEnd: T, blocks: [block])
        let engineMask = CBv2AttentionV1.spanChunkMask(
            L: T, kL: T, window: window, context: context)
        eval(full, engineMask)
        XCTAssertTrue(
            allClose(engineMask.asType(.int32), full.reshaped(T, T).asType(.int32)).item(Bool.self),
            "engine span mask must equal the independent reference overlay")
    }

    // MARK: - Coalescing + snapping (pure scheduler logic)

    func testCoalescedBlocks() {
        let merged = CBv2MultimodalPlan.coalescedBlocks(spans: [
            CBv2ImageSpan(tokenOffset: 2, length: 3),
            CBv2ImageSpan(tokenOffset: 5, length: 4),  // adjacent to previous
            CBv2ImageSpan(tokenOffset: 12, length: 2),  // gap ⇒ separate
        ])
        XCTAssertEqual(merged, [
            CBv2ImageSpan(tokenOffset: 2, length: 7),
            CBv2ImageSpan(tokenOffset: 12, length: 2),
        ])
    }

    func testSnappedChunkTokens() {
        let span = CBv2ImageSpan(tokenOffset: 4, length: 6)  // [4,10)
        let record = makeRecord(spans: [span])
        // Boundary inside the span shrinks to the span start.
        XCTAssertEqual(record.snappedChunkTokens(start: 0, proposed: 8, budget: 100), 4)
        // Chunk starting at the span extends to cover it whole.
        XCTAssertEqual(record.snappedChunkTokens(start: 4, proposed: 3, budget: 100), 6)
        // A chunk fully past the span is unchanged.
        XCTAssertEqual(record.snappedChunkTokens(start: 10, proposed: 5, budget: 100), 5)
        // A chunk ending exactly at the span start is unchanged (no split).
        XCTAssertEqual(record.snappedChunkTokens(start: 0, proposed: 4, budget: 100), 4)
        // A block that cannot fit the remaining budget yields 0 (wait).
        XCTAssertEqual(record.snappedChunkTokens(start: 4, proposed: 3, budget: 5), 0)
    }

    func testTextRecordSnappingIsIdentity() {
        let record = makeRecord(spans: [])
        XCTAssertEqual(record.snappedChunkTokens(start: 0, proposed: 7, budget: 100), 7)
    }

    private func makeRecord(spans: [CBv2ImageSpan]) -> CBv2ScheduledRequest {
        var request = CBv2Request(id: CBv2RequestID(1), promptTokens: [Int](repeating: 0, count: 64), maxTokens: 8)
        if !spans.isEmpty {
            request.multimodal = CBv2MultimodalInput(spans: spans) { [] }
        }
        return CBv2ScheduledRequest(
            request: request, arrivalSeq: 0, submittedAt: Date(), deadline: nil)
    }

    // MARK: - (e) Submit-time rejections

    private func makeCacheProvider(_ model: TinyTestModel) -> CBv2LayerCacheBank {
        CBv2LayerCacheBank(layerKinds: model.layerKinds)
    }

    func testRejectInvalidSpans() {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let bank = makeCacheProvider(model)
        // Out of bounds.
        let oob = CBv2MultimodalInput(spans: [CBv2ImageSpan(tokenOffset: 8, length: 5)]) {
            [MLXArray.zeros([5, model.config.hiddenSize])]
        }
        assertResolveThrows(oob, model: model, bank: bank, promptCount: 10) {
            if case .invalidSpans = $0 { return true }; return false
        }
        // Overlapping / out of order.
        let overlap = CBv2MultimodalInput(spans: [
            CBv2ImageSpan(tokenOffset: 4, length: 4),
            CBv2ImageSpan(tokenOffset: 6, length: 2),
        ]) { [MLXArray.zeros([4, 1]), MLXArray.zeros([2, 1])] }
        assertResolveThrows(overlap, model: model, bank: bank, promptCount: 20) {
            if case .invalidSpans = $0 { return true }; return false
        }
        // Empty spans.
        let empty = CBv2MultimodalInput(spans: []) { [] }
        assertResolveThrows(empty, model: model, bank: bank, promptCount: 20) {
            if case .invalidSpans = $0 { return true }; return false
        }
    }

    func testRejectOversizedBlock() {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let bank = makeCacheProvider(model)
        let input = CBv2MultimodalInput(spans: [CBv2ImageSpan(tokenOffset: 0, length: 40)]) {
            [MLXArray.zeros([40, model.config.hiddenSize])]
        }
        assertResolveThrows(input, model: model, bank: bank, promptCount: 64, maxBatched: 16) {
            if case .spanTooLong(let block, let budget) = $0 {
                return block == 40 && budget == 16
            }
            return false
        }
    }

    func testRejectUnsupportedModel() {
        // A steppable model that is NOT CBv2MultimodalSteppableModel.
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let plainModel = CBv2PlainSteppable(model)
        let bank = makeCacheProvider(model)
        let input = CBv2VisionFixtures.input([
            .init(span: CBv2ImageSpan(tokenOffset: 0, length: 2),
                embedding: MLXArray.zeros([1, 2, model.config.hiddenSize]))
        ])
        XCTAssertThrowsError(
            try CBv2MultimodalPlan.resolve(
                input, promptTokenCount: 10, model: plainModel, cacheProvider: bank,
                maxBatchedTokensPerStep: 4096)
        ) { error in
            guard case CBv2MultimodalError.unsupportedModel = error else {
                return XCTFail("expected unsupportedModel, got \(error)")
            }
        }
    }

    func testRejectUnsupportedBackend() {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let input = CBv2VisionFixtures.input([
            .init(span: CBv2ImageSpan(tokenOffset: 0, length: 2),
                embedding: MLXArray.zeros([1, 2, model.config.hiddenSize]))
        ])
        // A provider that does not vend span-mask-capable caches.
        let opaque = CBv2NonSpanCacheProvider()
        XCTAssertThrowsError(
            try CBv2MultimodalPlan.resolve(
                input, promptTokenCount: 10, model: model, cacheProvider: opaque,
                maxBatchedTokensPerStep: 4096)
        ) { error in
            guard case CBv2MultimodalError.unsupportedBackend = error else {
                return XCTFail("expected unsupportedBackend, got \(error)")
            }
        }
    }

    func testRejectEmbeddingMismatch() {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let bank = makeCacheProvider(model)
        let hidden = model.config.hiddenSize
        // Wrong count.
        let wrongCount = CBv2MultimodalInput(spans: [CBv2ImageSpan(tokenOffset: 0, length: 3)]) {
            []
        }
        assertResolveThrows(wrongCount, model: model, bank: bank, promptCount: 10) {
            if case .embeddingMismatch = $0 { return true }; return false
        }
        // Wrong length.
        let wrongLen = CBv2MultimodalInput(spans: [CBv2ImageSpan(tokenOffset: 0, length: 3)]) {
            [MLXArray.zeros([2, hidden])]
        }
        assertResolveThrows(wrongLen, model: model, bank: bank, promptCount: 10) {
            if case .embeddingMismatch = $0 { return true }; return false
        }
        // Wrong hidden dim.
        let wrongHidden = CBv2MultimodalInput(spans: [CBv2ImageSpan(tokenOffset: 0, length: 3)]) {
            [MLXArray.zeros([3, hidden + 1])]
        }
        assertResolveThrows(wrongHidden, model: model, bank: bank, promptCount: 10) {
            if case .embeddingMismatch = $0 { return true }; return false
        }
    }

    private func assertResolveThrows(
        _ input: CBv2MultimodalInput, model: TinyTestModel, bank: CBv2LayerCacheBank,
        promptCount: Int, maxBatched: Int = 4096,
        _ predicate: @escaping (CBv2MultimodalError) -> Bool,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try CBv2MultimodalPlan.resolve(
                input, promptTokenCount: promptCount, model: model, cacheProvider: bank,
                maxBatchedTokensPerStep: maxBatched),
            file: file, line: line
        ) { error in
            guard let mm = error as? CBv2MultimodalError, predicate(mm) else {
                return XCTFail("unexpected error \(error)", file: file, line: line)
            }
        }
    }
}

// MARK: - Test-only steppable wrappers

/// A `CBv2SteppableModel` that is deliberately NOT multimodal-capable.
private final class CBv2PlainSteppable: CBv2SteppableModel {
    private let model: TinyTestModel
    init(_ model: TinyTestModel) { self.model = model }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        model.forward(tokens: tokens, caches: caches)
    }
}

/// A layer-cache provider whose caches cannot bind span contexts.
private final class CBv2NonSpanCacheProvider: CBv2LayerCacheProvider {
    var uniformAttentionSoftcap: Float?? { .some(nil) }
    // supportsMultimodalSpans defaults to false.
    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] { [] }
}
