// CBv2PagedKernelTests.swift
//
// WS-C kernel correctness: paged-attention decode kernel parity vs the
// composed fp32 reference (rel-err <= 1e-2 fp16), batch-composition
// invariance (bitwise), prefill fallback parity, KV-shared borrowing, and
// a 200-step greedy decode token-match. No model weights required.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedKernel", .serialized)
struct CBv2PagedKernelTests {

    // MARK: - Config helper

    /// A paged config whose `maxPrefillChunk` the pool will actually accept
    /// for `kind`.
    ///
    /// WS-1.2 shrank the windowed ring from `ceil((window + maxPrefillChunk) /
    /// pageSize) + 1` pages to `ceil(window / pageSize) + ceil(
    /// maxSpeculativeSpan / pageSize)`, and `PagedKVPool` now REFUSES a config
    /// whose `maxPrefillChunk` exceeds the whole ring — a chunk that laps the
    /// ring would overwrite the tokens it just wrote. Before the shrink the
    /// ring was sized as `window + maxPrefillChunk`, so any chunk fitted by
    /// construction and these fixtures could hard-code 64.
    ///
    /// Clamping here rather than hard-coding per shape keeps the shape matrix
    /// declarative: each case asks for the largest chunk its window admits.
    /// Full-attention layers have no ring and are unclamped.
    static func pagedConfig(
        for kind: CBv2LayerKind, capacityBytes: Int, desiredPrefillChunk: Int = 64,
        nominalMaxSequenceLength: Int = 4096
    ) -> PagedKVPoolConfig {
        var config = PagedKVPoolConfig(
            capacityBytes: capacityBytes,
            maxPrefillChunk: desiredPrefillChunk,
            nominalMaxSequenceLength: nominalMaxSequenceLength)
        if case .slidingWindow(let window) = kind.attention {
            let ringTokens =
                PagedKVPool.ringPageCount(window: window, config: config) * config.pageSize
            config.maxPrefillChunk = min(desiredPrefillChunk, ringTokens)
        }
        return config
    }

    /// Largest chunk that may be handed to `PagedSequenceKV.write` DIRECTLY —
    /// i.e. bypassing `PagedLayerCache`, as the fixtures do when they seed a
    /// row's history.
    ///
    /// Distinct from `config.maxPrefillChunk`, and the difference is the whole
    /// of WS-1.2's trade. After a write of `n` tokens a windowed row reports
    /// `retainedCount == min(written, window - 1 + n)` and `attendableViews()`
    /// gathers exactly that span, which `gatherRange` refuses (process abort)
    /// once it exceeds the ring. `PagedLayerCache` is allowed a chunk as large
    /// as the whole ring because it gathers BEFORE it writes and concatenates
    /// the chunk it already holds (see PagedLayerCache.updateAndAttend,
    /// WS-1.2 comment) — it never asks the ring for more than `window - 1`.
    /// A direct writer has no such trick, so it is bounded by
    /// `ringTokens - window + 1`.
    static func rowSafeWriteChunk(for kind: CBv2LayerKind, desired: Int = 64) -> Int {
        guard case .slidingWindow(let window) = kind.attention else { return desired }
        let config = pagedConfig(
            for: kind, capacityBytes: 1 << 20, desiredPrefillChunk: desired)
        let ringTokens =
            PagedKVPool.ringPageCount(window: window, config: config) * config.pageSize
        return max(1, min(config.maxPrefillChunk, ringTokens - window + 1))
    }

    // MARK: - Fixtures

    /// One attention layer + its paged plumbing plus an independent
    /// contiguous "mirror" of everything written, used to compute
    /// references without touching the pool.
    final class Fixture {
        let kind: CBv2LayerKind
        let backend: PagedKVBackend
        let cache: PagedLayerCache
        var states: [[CBv2SequenceKV?]] = []
        var rows: [PagedSequenceKV] = []
        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        /// Largest chunk `row.write` accepts — see `pagedConfig(for:...)`.
        let writeChunk: Int

        init(
            kind: CBv2LayerKind, maxPrefillChunk: Int = 64,
            attentionSoftcap: Float? = nil
        ) throws {
            self.kind = kind
            let config = CBv2PagedKernelTests.pagedConfig(
                for: kind, capacityBytes: 64 << 20,
                desiredPrefillChunk: maxPrefillChunk)
            self.writeChunk = CBv2PagedKernelTests.rowSafeWriteChunk(
                for: kind, desired: config.maxPrefillChunk)
            self.backend = try PagedKVBackend(layerKinds: [kind], config: config)
            self.cache = backend.makeLayerCaches(attentionSoftcap: attentionSoftcap)[0]
        }

        deinit {
            states.forEach { backend.release($0) }
        }

        @discardableResult
        func addRow(tokens: Int, maxLength: Int = 2048) throws -> Int {
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: tokens, maxLength: maxLength)
            let row = state[0] as! PagedSequenceKV
            states.append(state)
            rows.append(row)
            var k = MLXArray.zeros([kind.kvHeads, 0, kind.headDim], dtype: .float16)
            var v = k
            var remaining = tokens
            while remaining > 0 {
                let n = min(remaining, writeChunk)
                let ck = MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16)
                let cv = MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16)
                row.write(keys: ck, values: cv)
                k = concatenated([k, ck], axis: 1)
                v = concatenated([v, cv], axis: 1)
                remaining -= n
            }
            mirrorK.append(k)
            mirrorV.append(v)
            cache.setRows(rows)
            return rows.count - 1
        }

        /// Reference decode for row `i` given this step's (q, k, v) —
        /// window clamping recomputed independently from the mirror.
        func referenceDecode(
            rowIndex i: Int, q: MLXArray, newK: MLXArray, newV: MLXArray,
            sinks: MLXArray?, scale: Float, softcap: Float? = nil
        ) -> MLXArray {
            mirrorK[i] = concatenated([mirrorK[i], newK], axis: 1)
            mirrorV[i] = concatenated([mirrorV[i], newV], axis: 1)
            let t = mirrorK[i].dim(1)
            var start = 0
            if case .slidingWindow(let w) = kind.attention {
                start = max(0, t - w)
            }
            let k = mirrorK[i][0..., start ..< t, 0...].expandedDimensions(axis: 0)
            let v = mirrorV[i][0..., start ..< t, 0...].expandedDimensions(axis: 0)
            return PagedAttentionReference.composedAttention(
                queries: q.expandedDimensions(axis: 0), keys: k, values: v,
                scale: scale, sinks: sinks, softcap: softcap)
        }
    }

    private func assertClose(
        _ got: MLXArray, _ want: MLXArray, rtol: Float = 1e-2, atol: Float = 2e-3,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        let ok = allClose(
            got.asType(.float32), want.asType(.float32), rtol: Double(rtol), atol: Double(atol)
        ).item(Bool.self)
        if !ok {
            let diff = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
            Issue.record(
                "arrays differ, max abs err \(diff)", sourceLocation: sourceLocation)
        }
    }

    // MARK: - Kernel parity vs composed reference

    @Test(arguments: [
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: true),
        (headDim: 64, kvHeads: 2, queryHeads: 16, window: Int?(20), sinks: true),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?(33), sinks: false),
        (headDim: 256, kvHeads: 2, queryHeads: 4, window: Int?.none, sinks: false),
        (headDim: 512, kvHeads: 2, queryHeads: 4, window: Int?.none, sinks: false),
        // Gemma-4-26B global-layer shape (d512, GQA 8) — exercises the
        // head split (HPT=2, 4 threadgroups per kv head).
        (headDim: 512, kvHeads: 2, queryHeads: 16, window: Int?.none, sinks: false),
        (headDim: 512, kvHeads: 1, queryHeads: 8, window: Int?(24), sinks: false),
    ])
    func decodeKernelParity(
        _ shape: (headDim: Int, kvHeads: Int, queryHeads: Int, window: Int?, sinks: Bool)
    ) throws {
        MLXRandom.seed(42)
        let attention: CBv2LayerKind.Attention =
            shape.window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, hasSinks: shape.sinks, headDim: shape.headDim,
            kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)
        let fixture = try Fixture(kind: kind)

        // Mixed row lengths, including page-boundary and sub-page cases.
        for tokens in [4, 16, 33, 100] {
            try fixture.addRow(tokens: tokens)
        }
        let b = fixture.rows.count
        let scale = Float(1.0 / Double(shape.headDim).squareRoot())
        let sinks: MLXArray? =
            shape.sinks
            ? MLXRandom.normal([shape.queryHeads], dtype: .float32) : nil

        // A few decode steps so positions advance past page boundaries.
        for _ in 0 ..< 3 {
            let q = MLXRandom.normal(
                [b, shape.queryHeads, 1, shape.headDim], dtype: .float16)
            let k = MLXRandom.normal(
                [b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let v = MLXRandom.normal(
                [b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let out = fixture.cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            #expect(out.shape == [b, shape.queryHeads, 1, shape.headDim])
            for i in 0 ..< b {
                let ref = fixture.referenceDecode(
                    rowIndex: i, q: q[i], newK: k[i], newV: v[i],
                    sinks: sinks, scale: scale)
                assertClose(out[i].expandedDimensions(axis: 0), ref)
            }
        }
    }

    @Test func decodeKernelSoftcapParity() throws {
        MLXRandom.seed(7)
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind, attentionSoftcap: 30.0)
        try fixture.addRow(tokens: 50)
        let scale: Float = 0.125

        let q = MLXRandom.normal([1, 4, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        let ref = fixture.referenceDecode(
            rowIndex: 0, q: q[0], newK: k[0], newV: v[0],
            sinks: nil, scale: scale, softcap: 30.0)
        assertClose(out[0].expandedDimensions(axis: 0), ref)
    }

    // MARK: - Batch-composition invariance

    @Test func decodeBatchCompositionInvariance() throws {
        MLXRandom.seed(11)
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 8)
        let scale: Float = 0.125

        // Two fixtures: the probe row solo, and the SAME content batched
        // with two batchmates. Identical content => bitwise-identical
        // outputs, or the backend leaks batch composition. The 600-token
        // batchmate spans multiple flash-decoding partitions, so the
        // batched dispatch launches MORE partition threadgroups than the
        // solo one — the probe row's math must not notice.
        let solo = try Fixture(kind: kind)
        let batched = try Fixture(kind: kind)

        MLXRandom.seed(100)
        try solo.addRow(tokens: 37)
        MLXRandom.seed(100)
        try batched.addRow(tokens: 37)
        try batched.addRow(tokens: 600)
        try batched.addRow(tokens: 90)

        MLXRandom.seed(200)
        let q = MLXRandom.normal([1, 8, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let qB = concatenated(
            [q, MLXRandom.normal([2, 8, 1, 64], dtype: .float16)], axis: 0)
        let kB = concatenated(
            [k, MLXRandom.normal([2, 2, 1, 64], dtype: .float16)], axis: 0)
        let vB = concatenated(
            [v, MLXRandom.normal([2, 2, 1, 64], dtype: .float16)], axis: 0)

        let outSolo = solo.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        let outBatched = batched.cache.updateAndAttend(
            queries: qB, keys: kB, values: vB, scale: scale, sinks: nil)

        #expect(
            arrayEqual(outSolo[0], outBatched[0]).item(Bool.self),
            "row output depends on batchmates — invariance violation")
    }

    // MARK: - Prefill fallback

    @Test(arguments: [Int?.none, Int?(24)])
    func prefillChunkParity(window: Int?) throws {
        MLXRandom.seed(3)
        let attention: CBv2LayerKind.Attention =
            window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind, maxPrefillChunk: 32)
        try fixture.addRow(tokens: 0)
        let row = fixture.rows[0]
        let scale: Float = 0.125

        var mirrorK = MLXArray.zeros([1, 2, 0, 64], dtype: .float16)
        var mirrorV = mirrorK
        for chunk in [9, 32, 16] {
            let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
            let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let out = fixture.cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: nil)
            mirrorK = concatenated([mirrorK, k], axis: 2)
            mirrorV = concatenated([mirrorV, v], axis: 2)

            // Reference: full mirror + explicit causal/window bool mask.
            let t = mirrorK.dim(2)
            let qpos = MLXArray(Int32(t - chunk) ..< Int32(t)).expandedDimensions(axis: 1)
            let kpos = MLXArray(Int32(0) ..< Int32(t)).expandedDimensions(axis: 0)
            var mask = kpos .<= qpos
            if let window {
                mask = mask & (kpos .> (qpos - Int32(window)))
            }
            let ref = PagedAttentionReference.composedAttention(
                queries: q, keys: mirrorK, values: mirrorV, scale: scale,
                boolMask: mask, sinks: nil)
            assertClose(out, ref)
            #expect(row.absoluteOffset == t)
        }

        // A decode step right after chunked prefill must agree too
        // (prefill -> decode transition over the same ring).
        let q = MLXRandom.normal([1, 4, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        mirrorK = concatenated([mirrorK, k], axis: 2)
        mirrorV = concatenated([mirrorV, v], axis: 2)
        let t = mirrorK.dim(2)
        var start = 0
        if let window { start = max(0, t - window) }
        let ref = PagedAttentionReference.composedAttention(
            queries: q,
            keys: mirrorK[0..., 0..., start ..< t, 0...],
            values: mirrorV[0..., 0..., start ..< t, 0...],
            scale: scale, sinks: nil)
        assertClose(out, ref)
    }

    @Test func prefillWithSinksParity() throws {
        MLXRandom.seed(13)
        let kind = CBv2LayerKind(
            attention: .full, hasSinks: true, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 0)
        let sinks = MLXRandom.normal([4], dtype: .float32)
        let scale: Float = 0.125

        let chunk = 12
        let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: sinks)

        let qpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 0)
        let ref = PagedAttentionReference.composedAttention(
            queries: q, keys: k, values: v, scale: scale,
            boolMask: kpos .<= qpos, sinks: sinks)
        assertClose(out, ref)
    }

    // MARK: - KV-shared borrowing (Gemma-style)

    @Test func borrowingMatchesSourceStorage() throws {
        MLXRandom.seed(21)
        let owner = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 8)
        var shared = owner
        shared.sharesKVWithLayer = 0
        let kinds = [owner, shared]
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(capacityBytes: 32 << 20))
        let caches = backend.makeLayerCaches()
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        let row = state[0] as! PagedSequenceKV
        caches[0].setRows([row])
        let scale: Float = 0.125

        var mirrorK = MLXArray.zeros([2, 0, 64], dtype: .float16)
        var mirrorV = mirrorK
        // Prefill through the owner, then decode; borrow at each decode.
        let pk = MLXRandom.normal([1, 2, 30, 64], dtype: .float16)
        let pv = MLXRandom.normal([1, 2, 30, 64], dtype: .float16)
        _ = caches[0].updateAndAttend(
            queries: MLXRandom.normal([1, 8, 30, 64], dtype: .float16),
            keys: pk, values: pv, scale: scale, sinks: nil)
        mirrorK = concatenated([mirrorK, pk.squeezed(axis: 0)], axis: 1)
        mirrorV = concatenated([mirrorV, pv.squeezed(axis: 0)], axis: 1)

        for _ in 0 ..< 2 {
            let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
            _ = caches[0].updateAndAttend(
                queries: MLXRandom.normal([1, 8, 1, 64], dtype: .float16),
                keys: k, values: v, scale: scale, sinks: nil)
            mirrorK = concatenated([mirrorK, k.squeezed(axis: 0)], axis: 1)
            mirrorV = concatenated([mirrorV, v.squeezed(axis: 0)], axis: 1)

            // The shared layer attends the owner's K/V with its own queries.
            let qShared = MLXRandom.normal([1, 8, 1, 64], dtype: .float16)
            let out = caches[1].attendBorrowing(
                source: caches[0], queries: qShared, scale: scale, sinks: nil)
            let ref = PagedAttentionReference.composedAttention(
                queries: qShared,
                keys: mirrorK.expandedDimensions(axis: 0),
                values: mirrorV.expandedDimensions(axis: 0),
                scale: scale, sinks: nil)
            assertClose(out, ref)
        }
        backend.release(state)
    }

    // MARK: - Position offsets

    @Test func positionOffsetsMatchAbsolutePositions() throws {
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 12)
        try fixture.addRow(tokens: 40)
        let offsets = fixture.cache.positionOffsets
        #expect(offsets.shape == [2])
        #expect(offsets[0].item(Int32.self) == 12)
        #expect(offsets[1].item(Int32.self) == 40)
    }

    // MARK: - End-to-end greedy decode token match

    /// 200 greedy decode steps on a tiny random attention "model": the
    /// paged kernel path and the composed contiguous reference each feed
    /// their own argmax back. Correctness bar: >= 99.5% token match.
    @Test func greedyDecodeTokenMatch200Steps() throws {
        MLXRandom.seed(1234)
        let vocab = 97
        let headDim = 64
        let kvHeads = 2
        let queryHeads = 4

        let embedQ = MLXRandom.normal([vocab, queryHeads * headDim], dtype: .float16)
        let embedK = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let embedV = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let unembed = MLXRandom.normal([queryHeads * headDim, vocab], dtype: .float16)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        let kind = CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 0, maxLength: 512)

        func project(_ token: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
            let q = embedQ[token].reshaped([1, queryHeads, 1, headDim])
            let k = embedK[token].reshaped([1, kvHeads, 1, headDim])
            let v = embedV[token].reshaped([1, kvHeads, 1, headDim])
            return (q, k, v)
        }
        func logits(_ attnOut: MLXArray) -> MLXArray {
            matmul(attnOut.reshaped([1, queryHeads * headDim]).asType(.float32),
                   unembed.asType(.float32))
        }

        var pagedToken = 1
        var refToken = 1
        var mirrorK = MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16)
        var mirrorV = mirrorK
        var matches = 0
        let steps = 200

        for _ in 0 ..< steps {
            // Paged path.
            let p = project(pagedToken)
            let outPaged = fixture.cache.updateAndAttend(
                queries: p.q, keys: p.k, values: p.v, scale: scale, sinks: nil)
            let nextPaged = argMax(logits(outPaged), axis: -1).item(Int32.self)

            // Composed contiguous reference path.
            let r = project(refToken)
            mirrorK = concatenated([mirrorK, r.k], axis: 2)
            mirrorV = concatenated([mirrorV, r.v], axis: 2)
            let outRef = PagedAttentionReference.composedAttention(
                queries: r.q, keys: mirrorK, values: mirrorV, scale: scale, sinks: nil)
            let nextRef = argMax(logits(outRef), axis: -1).item(Int32.self)

            if nextPaged == nextRef { matches += 1 }
            pagedToken = Int(nextPaged)
            refToken = Int(nextRef)
        }
        #expect(
            Double(matches) >= 0.995 * Double(steps),
            "greedy token match \(matches)/\(steps) below 99.5%")
    }

    // MARK: - Cross-backend differential: paged vs the CONTIGUOUS engine
    //
    // Everything above this mark compares the paged pool against an in-file
    // fp32 recomputation over a MIRROR of the same writes. That is a strong
    // oracle for the ATTENTION math and a weak one for STORAGE: a bug in the
    // shared write path — a mis-sized ring, a slipped page bound, a pad entry
    // that lands somewhere live — corrupts the pool while the mirror stays
    // clean only if the mirror is an independent STORAGE ENGINE. It is not; it
    // is a Swift array of the tensors the test itself generated, so a write
    // that never reached the slab and a read that never happened cancel out.
    //
    // These tests replace the mirror with `CBv2ContiguousKVBackend` +
    // `CBv2LayerCache`: a second, independently implemented store (contiguous
    // per-row buffers read by MLXFast SDPA) fed byte-identical inputs. The two
    // engines are compared against EACH OTHER, so agreement requires both to
    // have stored, evicted and re-read the same values.

    private enum StorageArm {
        case paged
        case contiguous
    }

    /// One layer of one storage engine, driven through the shared
    /// `CBv2AttendingLayerCache` surface. History is appended in identical
    /// chunk sizes on both arms so they observe identical `lastUpdateTokens`
    /// trajectories (which is what governs a windowed row's retained span).
    private final class Arm {
        let cache: CBv2AttendingLayerCache
        private let release: () -> Void

        init(
            _ arm: StorageArm, kind: CBv2LayerKind, histories: [[MLXArray]],
            maxLength: Int
        ) throws {
            let kinds = [kind]
            var rows: [CBv2SequenceKV] = []
            switch arm {
            case .paged:
                let backend = try PagedKVBackend(
                    layerKinds: kinds,
                    config: CBv2PagedKernelTests.pagedConfig(
                        for: kind, capacityBytes: 128 << 20))
                cache = backend.makeLayerCaches()[0]
                var states: [[CBv2SequenceKV?]] = []
                for chunks in histories {
                    let state = try backend.makeSequenceState(
                        layerKinds: kinds, promptLength: 0, maxLength: maxLength)
                    let row = state[0] as! PagedSequenceKV
                    for pair in stride(from: 0, to: chunks.count, by: 2) {
                        row.write(keys: chunks[pair], values: chunks[pair + 1])
                    }
                    states.append(state)
                    rows.append(row)
                }
                release = { states.forEach { backend.release($0) } }
            case .contiguous:
                let backend = CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 1 << 29))
                cache = CBv2LayerCache(layerIndex: 0, kind: kind)
                var states: [[CBv2SequenceKV?]] = []
                for chunks in histories {
                    let state = try backend.makeSequenceState(
                        layerKinds: kinds, promptLength: 0, maxLength: maxLength)
                    let row = state[0]!
                    for pair in stride(from: 0, to: chunks.count, by: 2) {
                        _ = row.update(
                            keys: chunks[pair].expandedDimensions(axis: 0),
                            values: chunks[pair + 1].expandedDimensions(axis: 0))
                    }
                    states.append(state)
                    rows.append(row)
                }
                release = { states.forEach { backend.release($0) } }
            }
            cache.setRows(rows)
        }

        deinit { release() }

        func step(
            queries: MLXArray, keys: MLXArray, values: MLXArray,
            scale: Float, sinks: MLXArray?
        ) -> MLXArray {
            cache.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: sinks)
        }
    }

    /// `[kvHeads, n, headDim]` history chunks per row, generated ONCE and
    /// handed to both arms so the two engines store byte-identical bytes AND
    /// observe identical `lastUpdateTokens` trajectories.
    ///
    /// Chunk width is whatever the paged pool will accept for `kind` (see
    /// `pagedConfig(for:...)`); the contiguous arm has no such bound but must
    /// use the same widths or the two rows' retained spans diverge for reasons
    /// that have nothing to do with storage.
    private func makeHistories(kind: CBv2LayerKind, lengths: [Int]) -> [[MLXArray]] {
        let chunk = CBv2PagedKernelTests.rowSafeWriteChunk(for: kind)
        return lengths.map { tokens in
            var chunks: [MLXArray] = []
            var remaining = tokens
            while remaining > 0 {
                let n = min(remaining, chunk)
                chunks.append(MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16))
                chunks.append(MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16))
                remaining -= n
            }
            return chunks
        }
    }

    /// Decode parity between the two STORAGE ENGINES across the shape matrix
    /// `decodeKernelParity` covers with a mirror, plus both gemma-4-26b layer
    /// shapes (sliding: window 1024 / d256 / 8 kv heads; full: d512 / 2 kv
    /// heads).
    ///
    /// Tolerance: both arms hold fp16 pages read by different kernels (Metal
    /// paged flash-decoding with fp32 accumulation vs `MLXFast.scaledDot
    /// ProductAttention`), so this is a numerical-agreement bar, not a bitwise
    /// one — the same `1e-2 / 2e-3` bar the fp32-reference tests already hold
    /// each arm to individually.
    @Test(arguments: [
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: true),
        (headDim: 64, kvHeads: 2, queryHeads: 16, window: Int?(20), sinks: true),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?(33), sinks: false),
        (headDim: 256, kvHeads: 8, queryHeads: 16, window: Int?(1024), sinks: false),
        (headDim: 512, kvHeads: 2, queryHeads: 16, window: Int?.none, sinks: false),
    ])
    func decodeMatchesContiguousBackend(
        _ shape: (headDim: Int, kvHeads: Int, queryHeads: Int, window: Int?, sinks: Bool)
    ) throws {
        MLXRandom.seed(0x5EED_0A11)
        let attention: CBv2LayerKind.Attention =
            shape.window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, hasSinks: shape.sinks, headDim: shape.headDim,
            kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)

        // Row lengths straddle a page boundary (16), a whole page, and — for
        // the window-20/33 kinds — the window itself, so the ring has already
        // wrapped on at least one row before the first differential step.
        let lengths = [4, 16, 33, 100]
        let histories = makeHistories(kind: kind, lengths: lengths)
        let b = lengths.count
        let scale = Float(1.0 / Double(shape.headDim).squareRoot())
        // Sinks in the QUERY dtype: both arms must hold bit-identical inputs
        // for a storage differential to mean anything, and the paged decode
        // kernel re-widens to fp32 in `preparedSinks` regardless.
        //
        // This line used to be load-bearing for a different reason, worth one
        // note so nobody "simplifies" it back. On its first run this test
        // aborted the whole bundle: `PagedLayerCache` coerced sinks to the
        // query dtype, `CBv2AttentionV1` did not, and fp16 queries with fp32
        // sinks reached MLXFast and raised "[scaled_dot_product_attention]
        // Type of sinks must promote to output type float16" — a `fatalError`,
        // i.e. a daemon abort, on the CONTIGUOUS backend only. Latent in
        // production only because gpt-oss-20b's `sinks` parameter happens to
        // load in the activation dtype. Fixed 2026-07-25 (A2): AttentionV1 now
        // coerces at both MLXFast sites, and cross-backend sink-dtype
        // acceptance has its own cover in `CBv2AttentionSinkDtypeTests`.
        // Storage parity — this test — is not the place to re-assert it.
        let sinks: MLXArray? =
            shape.sinks ? MLXRandom.normal([shape.queryHeads], dtype: .float16) : nil

        let paged = try Arm(.paged, kind: kind, histories: histories, maxLength: 2048)
        let contiguous = try Arm(
            .contiguous, kind: kind, histories: histories, maxLength: 2048)

        for _ in 0 ..< 4 {
            let q = MLXRandom.normal([b, shape.queryHeads, 1, shape.headDim], dtype: .float16)
            let k = MLXRandom.normal([b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let v = MLXRandom.normal([b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let outPaged = paged.step(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            let outContiguous = contiguous.step(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            #expect(outPaged.shape == [b, shape.queryHeads, 1, shape.headDim])
            for row in 0 ..< b {
                assertClose(
                    outPaged[row].expandedDimensions(axis: 0),
                    outContiguous[row].expandedDimensions(axis: 0))
            }
        }
    }

    /// The chunked-prefill fallback compared against the contiguous engine
    /// rather than a mirror. Prefill is per-request `[1, chunk]` on BOTH
    /// backends, so one call sequence drives both.
    @Test(arguments: [Int?.none, Int?(24)])
    func prefillMatchesContiguousBackend(window: Int?) throws {
        MLXRandom.seed(0x5EED_0A12)
        let attention: CBv2LayerKind.Attention =
            window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, headDim: 64, kvHeads: 2, queryHeads: 4)
        let scale: Float = 0.125

        let paged = try Arm(.paged, kind: kind, histories: [[]], maxLength: 512)
        let contiguous = try Arm(.contiguous, kind: kind, histories: [[]], maxLength: 512)

        // Chunks that cross the window (24) and page boundaries (16), then a
        // decode step over the same ring — the prefill -> decode transition is
        // where a mis-sized ring first shows up.
        for chunk in [9, 32, 16, 1] {
            let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
            let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            assertClose(
                paged.step(queries: q, keys: k, values: v, scale: scale, sinks: nil),
                contiguous.step(queries: q, keys: k, values: v, scale: scale, sinks: nil))
        }
    }

    /// The 200-step greedy trajectory with the reference arm backed by a real
    /// contiguous store instead of an in-file recomputation. Divergence
    /// compounds: one wrong key at step 3 changes every later token, so a
    /// token-match bar over 200 steps is a far sharper storage oracle than any
    /// single-step comparison. Windowed(16): the ring wraps ~12 times, so
    /// eviction arithmetic is under test, not just append arithmetic.
    @Test func greedyDecodeMatchesContiguousBackend200Steps() throws {
        MLXRandom.seed(0x5EED_0A13)
        let vocab = 97
        let headDim = 64
        let kvHeads = 2
        let queryHeads = 4

        let embedQ = MLXRandom.normal([vocab, queryHeads * headDim], dtype: .float16)
        let embedK = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let embedV = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let unembed = MLXRandom.normal([queryHeads * headDim, vocab], dtype: .float16)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        let kind = CBv2LayerKind(
            attention: .slidingWindow(16), headDim: headDim,
            kvHeads: kvHeads, queryHeads: queryHeads)
        let paged = try Arm(.paged, kind: kind, histories: [[]], maxLength: 512)
        let contiguous = try Arm(.contiguous, kind: kind, histories: [[]], maxLength: 512)

        func project(_ token: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
            (
                embedQ[token].reshaped([1, queryHeads, 1, headDim]),
                embedK[token].reshaped([1, kvHeads, 1, headDim]),
                embedV[token].reshaped([1, kvHeads, 1, headDim])
            )
        }
        func logits(_ attnOut: MLXArray) -> MLXArray {
            matmul(
                attnOut.reshaped([1, queryHeads * headDim]).asType(.float32),
                unembed.asType(.float32))
        }

        var pagedToken = 1
        var contiguousToken = 1
        var matches = 0
        let steps = 200
        for _ in 0 ..< steps {
            let p = project(pagedToken)
            let nextPaged = argMax(
                logits(
                    paged.step(queries: p.q, keys: p.k, values: p.v, scale: scale, sinks: nil)),
                axis: -1
            ).item(Int32.self)

            let c = project(contiguousToken)
            let nextContiguous = argMax(
                logits(
                    contiguous.step(
                        queries: c.q, keys: c.k, values: c.v, scale: scale, sinks: nil)),
                axis: -1
            ).item(Int32.self)

            if nextPaged == nextContiguous { matches += 1 }
            pagedToken = Int(nextPaged)
            contiguousToken = Int(nextContiguous)
        }
        #expect(
            Double(matches) >= 0.995 * Double(steps),
            "paged vs contiguous greedy token match \(matches)/\(steps) below 99.5%")
    }
}
