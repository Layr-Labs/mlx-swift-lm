// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Inkling 2 paged attention against a plain-MLX FP32 reference of
/// `eager_attention_forward` (pages gathered into a dense view here only),
/// plus bit-equality across batch composition, row count, chunking and kernel.
final class Inkling2PagedAttentionTests: XCTestCase {
    typealias Attention = Inkling2PagedAttention

    static let qh = 24
    static let kvh = 8
    static let d = 128
    static let s = 16
    static let scale = Float(1.0 / Double(128).squareRoot())

    override func setUp() {
        super.setUp()
        MLX.GPU.set(cacheLimit: 512 << 20)
        MLX.GPU.set(memoryLimit: 8 << 30)
    }

    // MARK: - Fixture

    /// One sequence of `total` tokens whose first `history` positions sit in
    /// shuffled pages (a ring for windowed rows when `ringPages` is set).
    struct Sequence {
        var keys: MLXArray  // [KVH, total, D] float32 (rounded to the slab dtype)
        var values: MLXArray
        var queries: MLXArray  // [QH, total, D]
        var states: MLXArray  // [QH, total, 16]
        var sinks: MLXArray  // [QH, total]
    }

    struct Pool {
        var kSlab: MLXArray
        var vSlab: MLXArray
        var pages: Int
    }

    static func makeSequence(total: Int, seed: UInt64, dtype: DType, sinkScale: Float = 1)
        -> Sequence
    {
        MLXRandom.seed(seed)
        func r(_ shape: [Int], _ scale: Float) -> MLXArray {
            (MLXRandom.normal(shape) * scale).asType(dtype).asType(.float32)
        }
        return Sequence(
            keys: r([kvh, total, d], 0.6), values: r([kvh, total, d], 1),
            queries: r([qh, total, d], 0.6), states: r([qh, total, 16], 0.5),
            sinks: MLXRandom.normal([qh, total]) * sinkScale)
    }

    static func makeProj(seed: UInt64, dtype: DType) -> MLXArray {
        MLXRandom.seed(seed)
        return (MLXRandom.normal([16, 1024]) * 0.5).asType(dtype)
    }

    /// Lays out `history` tokens of each sequence into pages of one shared
    /// pool. Returns tables `[B, maxPages]` and per-row table lengths.
    static func layout(
        _ seqs: [Sequence], history: [Int], ringPages: Int?, dtype: DType, seed: UInt64
    ) -> (pool: Pool, tables: MLXArray, tableLengths: [Int]) {
        var rng = SplitMix(seed: seed)
        let need = history.map { h -> Int in
            let pages = (h + s - 1) / s
            return ringPages.map { min(pages, $0) } ?? pages
        }
        let total = 1 + need.reduce(0, +)
        var ids = Array(1 ..< total).map { Int32($0) }
        ids.shuffle(using: &rng)
        // MLXArray is a reference type: K and V need separate arrays.
        var k = MLXArray.zeros([total * s, kvh, d], dtype: dtype)
        var v = MLXArray.zeros([total * s, kvh, d], dtype: dtype)
        var tables: [[Int32]] = []
        var cursor = 0
        for (row, seq) in seqs.enumerated() {
            let table = Array(ids[cursor ..< cursor + need[row]])
            cursor += need[row]
            tables.append(table)
            guard history[row] > 0 else { continue }
            // A ring keeps only its last `ringPages` pages of positions.
            let first = ringPages.map { max(0, (history[row] - 1) / s - $0 + 1) * s } ?? 0
            let slots = (first ..< history[row]).map { pos -> Int32 in
                let lp = pos / s
                return table[ringPages.map { lp % $0 } ?? lp] * Int32(s) + Int32(pos % s)
            }
            let index = MLXArray(slots)
            k[index] = seq.keys[0..., first ..< history[row]].transposed(1, 0, 2).asType(dtype)
            v[index] = seq.values[0..., first ..< history[row]].transposed(1, 0, 2).asType(dtype)
        }
        k = k.reshaped(total, s, kvh, d).transposed(0, 2, 1, 3).contiguous()
        v = v.reshaped(total, s, kvh, d).transposed(0, 2, 1, 3).contiguous()
        eval(k, v)
        let width = max(8, tables.map(\.count).max() ?? 0, ringPages ?? 0)
        var flat = [Int32](repeating: 0, count: seqs.count * width)
        for (row, table) in tables.enumerated() {
            flat.replaceSubrange(row * width ..< row * width + table.count, with: table)
        }
        let lengths = zip(need, history).map { n, _ in ringPages ?? max(1, n) }
        return (
            Pool(kSlab: k, vSlab: v, pages: total), MLXArray(flat, [seqs.count, width]), lengths
        )
    }

    struct Call {
        var queries: MLXArray
        var newKeys: MLXArray
        var newValues: MLXArray
        var states: MLXArray
        var sinks: MLXArray
        var rows: [Attention.Row]
    }

    /// Queries `[start, start + l)` of every sequence.
    static func call(
        _ seqs: [Sequence], start: [Int], l: Int, tableLengths: [Int], base: [Int]? = nil,
        dtype: DType
    ) -> Call {
        func cut(_ x: MLXArray, _ row: Int) -> MLXArray {
            x[0..., start[row] ..< start[row] + l]
        }
        return Call(
            queries: stacked(seqs.indices.map { cut(seqs[$0].queries, $0) }).asType(dtype),
            newKeys: stacked(seqs.indices.map { cut(seqs[$0].keys, $0) }).asType(dtype),
            newValues: stacked(seqs.indices.map { cut(seqs[$0].values, $0) }).asType(dtype),
            states: stacked(seqs.indices.map { cut(seqs[$0].states, $0) }).asType(dtype),
            sinks: stacked(seqs.indices.map { seqs[$0].sinks[0..., start[$0] ..< start[$0] + l] }),
            rows: seqs.indices.map {
                Attention.Row(
                    queryStart: start[$0], base: base?[$0] ?? 0, tableLength: tableLengths[$0])
            })
    }

    static func run(
        _ c: Call, pool: Pool, tables: MLXArray, proj: MLXArray, window: Int?,
        kernel: Attention.Kernel = .automatic, subPartitions: Int? = nil,
        write: Bool = false, fence: MLXArray = MLXArray.zeros([1], dtype: .int32)
    ) -> Attention.Output {
        Attention.attend(
            queries: c.queries, newKeys: c.newKeys, newValues: c.newValues,
            relativeStates: c.states, proj: proj, sinks: c.sinks,
            cache: .init(keys: pool.kSlab, values: pool.vSlab, tables: tables, writeFence: fence),
            rows: c.rows, window: window, scale: scale, write: write, kernel: kernel,
            subPartitions: subPartitions)
    }

    /// FP32 `eager_attention_forward` on the dense sequence: `[L, QH * D]`.
    static func reference(
        _ seq: Sequence, start: Int, l: Int, base: Int, proj: MLXArray, window: Int?
    ) -> MLXArray {
        let end = start + l
        let lowest = window.map { max(base, start - $0 + 1) } ?? base
        let keys = seq.keys[0..., lowest ..< end]
        let values = seq.values[0..., lowest ..< end]
        let qpos = MLXArray(Int32(start) ..< Int32(end)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(lowest) ..< Int32(end)).expandedDimensions(axis: 0)
        let dist = qpos - kpos
        let inWindow = window.map { dist .< Int32($0) } ?? MLXArray(true)
        let visible = (dist .>= 0) .&& (kpos .>= Int32(base)) .&& inWindow
        let band = (dist .>= 0) .&& (dist .< 1024)
        let index = clip(dist, min: 0, max: 1023)
        let projF = proj.asType(.float32)
        var heads: [MLXArray] = []
        for h in 0 ..< qh {
            let q = seq.queries[h, start ..< end]
            let k = keys[h / 3]
            let v = values[h / 3]
            let relative = matmul(seq.states[h, start ..< end], projF)
            let bias = which(band, takeAlong(relative, index, axis: 1), MLXArray(Float(0)))
            var logits = matmul(q, k.transposed()) * scale + bias
            logits = which(visible, logits, MLXArray(-Float.infinity))
            let sink = seq.sinks[h, start ..< end].expandedDimensions(axis: 1)
            let probs = softmax(concatenated([logits, sink], axis: 1), axis: 1, precise: true)
            heads.append(matmul(probs[0..., 0 ..< logits.dim(1)], v))
        }
        return concatenated(heads, axis: 1)
    }

    struct ErrorStats {
        var maxAbs: Float = 0
        var maxRel: Float = 0
        mutating func add(_ actual: MLXArray, _ expected: MLXArray) {
            let diff = abs(actual.asType(.float32) - expected).max().item(Float.self)
            let scale = abs(expected).max().item(Float.self)
            maxAbs = max(maxAbs, diff)
            maxRel = max(maxRel, diff / max(scale, 1e-30))
        }
    }

    static func bytes(_ a: MLXArray) -> Data {
        eval(a)
        return a.asData(access: .copy).data
    }

    // MARK: - Accuracy

    func testMatchesFP32ReferenceAcrossContexts() throws {
        var lines: [String] = []
        var worst = ErrorStats()
        let proj = Self.makeProj(seed: 11, dtype: .float32)
        for history in [1, 7, 1023, 1024, 1025, 3000, 9000] {
            for window in [nil, 1024] as [Int?] {
                for l in [1, 2, 8, 16, 512] {
                    let seq = Self.makeSequence(
                        total: history + l, seed: UInt64(history * 31 + l), dtype: .float32)
                    let ring = window.map { _ in 65 }
                    let (pool, tables, lengths) = Self.layout(
                        [seq], history: [history], ringPages: ring, dtype: .float32,
                        seed: UInt64(history))
                    let c = Self.call(
                        [seq], start: [history], l: l, tableLengths: lengths, dtype: .float32)
                    let expected = Self.reference(
                        seq, start: history, l: l, base: 0, proj: proj, window: window)
                    let parts = Attention.partitionCount(
                        row: .init(queryStart: history, tableLength: 1), queries: l,
                        window: window ?? 0)
                    for kernel in [Attention.Kernel.fused, .split, .sequential]
                    where (l <= 16 || kernel == .sequential)
                        && (kernel != .fused || parts <= Attention.fusedMaximumPartitions)
                    {
                        let out = Self.run(
                            c, pool: pool, tables: tables, proj: proj, window: window,
                            kernel: kernel
                        ).output
                        var stats = ErrorStats()
                        stats.add(out[0], expected)
                        worst.add(out[0], expected)
                        lines.append(
                            "history=\(history) L=\(l) window=\(window ?? 0) kernel=\(kernel) "
                                + "maxAbs=\(stats.maxAbs) maxRel=\(stats.maxRel)")
                        XCTAssertLessThan(stats.maxAbs, 2e-5, lines.last!)
                    }
                }
            }
        }
        print("[Inkling2PagedAttention] FP32 reference\n" + lines.joined(separator: "\n"))
        print("[Inkling2PagedAttention] FP32 worst maxAbs=\(worst.maxAbs) maxRel=\(worst.maxRel)")
    }

    func testBFloat16MatchesReferenceWithinOutputRounding() throws {
        var worst = ErrorStats()
        let proj = Self.makeProj(seed: 12, dtype: .bfloat16)
        for (history, l, window) in [
            (1, 1, nil), (1023, 8, 1024), (1025, 16, 1024), (3000, 16, nil), (9000, 1, nil),
            (1500, 512, 1024),
        ] as [(Int, Int, Int?)] {
            let seq = Self.makeSequence(
                total: history + l, seed: UInt64(history + l), dtype: .bfloat16)
            let (pool, tables, lengths) = Self.layout(
                [seq], history: [history], ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                seed: 5)
            let c = Self.call(
                [seq], start: [history], l: l, tableLengths: lengths, dtype: .bfloat16)
            let out = Self.run(c, pool: pool, tables: tables, proj: proj, window: window).output
            XCTAssertEqual(out.dtype, .bfloat16)
            let expected = Self.reference(
                seq, start: history, l: l, base: 0, proj: proj, window: window)
            var stats = ErrorStats()
            stats.add(out[0], expected)
            worst.add(out[0], expected)
            // BF16 output rounding is 2^-9 relative to each element.
            XCTAssertLessThan(stats.maxAbs, 4e-3 * max(1, abs(expected).max().item(Float.self)))
        }
        print("[Inkling2PagedAttention] BF16 worst maxAbs=\(worst.maxAbs) maxRel=\(worst.maxRel)")
    }

    func testSinkLargeAndSmall() throws {
        let proj = Self.makeProj(seed: 13, dtype: .float32)
        for sinkScale: Float in [40, -40, 8] {
            let seq = Self.makeSequence(total: 300, seed: 9, dtype: .float32, sinkScale: 1)
            var shifted = seq
            shifted.sinks = MLXArray.ones([Self.qh, 300]) * sinkScale + seq.sinks
            let (pool, tables, lengths) = Self.layout(
                [shifted], history: [290], ringPages: nil, dtype: .float32, seed: 3)
            let c = Self.call(
                [shifted], start: [290], l: 10, tableLengths: lengths, dtype: .float32)
            let out = Self.run(c, pool: pool, tables: tables, proj: proj, window: 1024).output
            let expected = Self.reference(
                shifted, start: 290, l: 10, base: 0, proj: proj, window: 1024)
            var stats = ErrorStats()
            stats.add(out[0], expected)
            XCTAssertLessThan(stats.maxAbs, 2e-5, "sink shift \(sinkScale)")
            if sinkScale > 30 {
                XCTAssertLessThan(abs(out).max().item(Float.self), 1e-6)
            }
        }
    }

    func testPageBoundariesRingWrapAndAdoptedBase() throws {
        let proj = Self.makeProj(seed: 14, dtype: .float32)
        // Histories on both sides of page (16), partition (128), window (1024)
        // and ring (65 pages = 1040 tokens) boundaries.
        for history in [15, 16, 17, 127, 128, 129, 1039, 1040, 1041, 2079, 2080, 5000] {
            for l in [1, 9, 16] {
                let seq = Self.makeSequence(
                    total: history + l, seed: UInt64(history), dtype: .float32)
                let (pool, tables, lengths) = Self.layout(
                    [seq], history: [history], ringPages: 65, dtype: .float32, seed: 8)
                let c = Self.call(
                    [seq], start: [history], l: l, tableLengths: lengths, dtype: .float32)
                let out = Self.run(c, pool: pool, tables: tables, proj: proj, window: 1024).output
                let expected = Self.reference(
                    seq, start: history, l: l, base: 0, proj: proj, window: 1024)
                var stats = ErrorStats()
                stats.add(out[0], expected)
                XCTAssertLessThan(stats.maxAbs, 2e-5, "history \(history) L \(l)")
            }
        }
        // A prefix-adopted windowed row starts writing at `base`; nothing older exists.
        for (base, history) in [(1000, 1010), (2048, 2100), (512, 1700)] {
            let seq = Self.makeSequence(total: history + 4, seed: 77, dtype: .float32)
            for window in [1024, nil] as [Int?] {
                // Sliding rows keep a 65-page ring; full rows keep every page.
                let (pool, tables, lengths) = Self.layout(
                    [seq], history: [history], ringPages: window.map { _ in 65 },
                    dtype: .float32, seed: 9)
                let c = Self.call(
                    [seq], start: [history], l: 4, tableLengths: lengths, base: [base],
                    dtype: .float32)
                let out = Self.run(c, pool: pool, tables: tables, proj: proj, window: window).output
                let expected = Self.reference(
                    seq, start: history, l: 4, base: base, proj: proj, window: window)
                var stats = ErrorStats()
                stats.add(out[0], expected)
                XCTAssertLessThan(stats.maxAbs, 2e-5, "base \(base) window \(window ?? 0)")
            }
        }
    }

    func testThreadgroupMemoryBudgetFitsEveryVariant() {
        for split in [true, false] {
            for elementSize in [2, 4] {
                for headDim in [64, 128] {
                    for gqa in 1 ... 8 {
                        for l in [1, 2, 3, 9, 16, 512] {
                            // Two simdgroups per block only where the split
                            // family's smallest launch fits them.
                            let psub =
                                split
                                ? Attention.subPartitionCount(
                                    requested: 2, b: 1, kvHeads: 1, queries: l, gqa: gqa,
                                    maxPartitions: 1, headDim: headDim, elementSize: elementSize)
                                : 1
                            let qpt = Attention.queriesPerThreadgroup(
                                split: split, queries: l, gqa: gqa, headDim: headDim,
                                elementSize: elementSize, simdgroupsPerBlock: psub)
                            let bytes = Attention.threadgroupBytes(
                                split: split, blocks: (qpt * gqa + 7) / 8, headDim: headDim,
                                elementSize: elementSize, simdgroupsPerBlock: psub)
                            XCTAssertLessThanOrEqual(
                                bytes, Attention.threadgroupMemoryLimit,
                                "split \(split) es \(elementSize) d \(headDim) gqa \(gqa) L \(l)")
                        }
                    }
                }
            }
        }
        // Inkling 2 BF16: 5 queries (two blocks, each with its first-half hold)
        // per split threadgroup on one simdgroup per block, 2 (one block on two
        // simdgroups) on two, 16 (six simdgroups) per sequential one.
        XCTAssertEqual(
            Attention.queriesPerThreadgroup(
                split: true, queries: 9, gqa: 3, headDim: 128, elementSize: 2), 5)
        XCTAssertEqual(
            Attention.queriesPerThreadgroup(
                split: true, queries: 9, gqa: 3, headDim: 128, elementSize: 2,
                simdgroupsPerBlock: 2), 2)
        XCTAssertEqual(
            Attention.queriesPerThreadgroup(
                split: false, queries: 2048, gqa: 3, headDim: 128, elementSize: 2), 16)
    }

    // MARK: - Prefill on the neural accelerators

    /// The runner's GPU facts, logged on every run, and the prefill kernel's
    /// gate never claims neural accelerators MLX's own NAX check denies (the
    /// kernel uses MLX's NAX fragment layout).
    func testNeuralAcceleratorGateFollowsMLX() {
        let mlx = GPU.gemma4ExpertQMMDiagnostics().naxAvailable
        let facts =
            "gpu \(GPU.deviceInfo().architecture) "
            + "os \(ProcessInfo.processInfo.operatingSystemVersionString) mlx_nax \(mlx) "
            + "prefill_gate \(Attention.neuralAcceleratorsAvailable)"
        print("[Inkling2Attn] \(facts)")
        XCTAssertTrue(!Attention.neuralAcceleratorsAvailable || mlx, facts)
    }

    /// The prefill kernel's own arithmetic: within BF16 output rounding of the
    /// reference, and bit-identical however the prompt is chunked, whatever
    /// rows share the batch, for slabs and segments, and for both layouts.
    func testPrefillKernelAccuracyAndInvariance() throws {
        guard Attention.neuralAcceleratorsAvailable else {
            throw XCTSkip("the prefill kernel needs the M5 neural accelerators")
        }
        let proj = Self.makeProj(seed: 21, dtype: .bfloat16)
        for (history, l, window) in [
            (0, 1, nil), (0, 40, 1024), (5, 17, nil), (1023, 64, 1024), (1025, 33, 1024),
            (3000, 100, nil), (1500, 512, 1024),
        ] as [(Int, Int, Int?)] {
            let seq = Self.makeSequence(
                total: history + l, seed: UInt64(7 * history + l), dtype: .bfloat16)
            let (pool, tables, lengths) = Self.layout(
                [seq], history: [history], ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                seed: 6)
            let c = Self.call(
                [seq], start: [history], l: l, tableLengths: lengths, dtype: .bfloat16)
            let out = Self.run(
                c, pool: pool, tables: tables, proj: proj, window: window, kernel: .prefill
            ).output
            let expected = Self.reference(
                seq, start: history, l: l, base: 0, proj: proj, window: window)
            var stats = ErrorStats()
            stats.add(out[0], expected)
            XCTAssertLessThan(
                stats.maxAbs, 4e-3 * max(1, abs(expected).max().item(Float.self)),
                "history \(history) L \(l) window \(window ?? 0)")
        }

        // Page, tile, window and ring boundaries, and a prefix-adopted base.
        for (history, base, window) in [
            (15, 0, 1024), (16, 0, 1024), (31, 0, 1024), (32, 0, nil), (1039, 0, 1024),
            (1040, 0, 1024), (1041, 0, 1024), (2080, 0, 1024), (1700, 512, 1024), (1700, 512, nil),
            (2100, 2048, nil),
        ] as [(Int, Int, Int?)] {
            let l = 33
            let seq = Self.makeSequence(
                total: history + l, seed: UInt64(history + base), dtype: .bfloat16)
            let (pool, tables, lengths) = Self.layout(
                [seq], history: [history], ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                seed: 10)
            let c = Self.call(
                [seq], start: [history], l: l, tableLengths: lengths, base: [base],
                dtype: .bfloat16)
            let out = Self.run(
                c, pool: pool, tables: tables, proj: proj, window: window, kernel: .prefill
            ).output
            let expected = Self.reference(
                seq, start: history, l: l, base: base, proj: proj, window: window)
            var stats = ErrorStats()
            stats.add(out[0], expected)
            XCTAssertLessThan(
                stats.maxAbs, 4e-3 * max(1, abs(expected).max().item(Float.self)),
                "history \(history) base \(base) window \(window ?? 0)")
        }

        // Chunking: the whole prompt in one call equals every chunking of it.
        let total = 1300
        for window in [nil, 1024] as [Int?] {
            let seq = Self.makeSequence(total: total, seed: 43, dtype: .bfloat16)
            let (pool, tables, lengths) = Self.layout(
                [seq], history: [total], ringPages: nil, dtype: .bfloat16, seed: 8)
            func attend(_ start: Int, _ l: Int) -> MLXArray {
                let c = Self.call(
                    [seq], start: [start], l: l, tableLengths: lengths, dtype: .bfloat16)
                return Self.run(
                    c, pool: pool, tables: tables, proj: proj, window: window, kernel: .prefill
                ).output[0]
            }
            let whole = attend(0, total)
            for cuts in [
                [0, 512, 1024, total], [0, 333, 1033, total], [0, 1, 2, 130, total],
                [0, 31, 32, 33, 1299, total],
            ] {
                let pieces = zip(cuts, cuts.dropFirst()).map { attend($0, $1 - $0) }
                XCTAssertEqual(
                    Self.bytes(concatenated(pieces, axis: 0)), Self.bytes(whole),
                    "window \(window ?? 0) chunks \(cuts)")
            }
        }

        // Batch: a row alone equals the same row among others.
        let histories = [5, 900, 4000, 17, 2049]
        for window in [nil, 1024] as [Int?] {
            let l = 40
            let seqs = histories.enumerated().map {
                Self.makeSequence(total: $1 + l, seed: UInt64(200 + $0), dtype: .bfloat16)
            }
            let (pool, tables, lengths) = Self.layout(
                seqs, history: histories, ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                seed: 22)
            let all = Self.call(
                seqs, start: histories, l: l, tableLengths: lengths, dtype: .bfloat16)
            let batched = Self.run(
                all, pool: pool, tables: tables, proj: proj, window: window, kernel: .prefill
            ).output
            for row in histories.indices {
                let one = Self.call(
                    [seqs[row]], start: [histories[row]], l: l, tableLengths: [lengths[row]],
                    dtype: .bfloat16)
                let alone = Self.run(
                    one, pool: pool, tables: tables[row ..< row + 1], proj: proj, window: window,
                    kernel: .prefill
                ).output
                XCTAssertEqual(
                    Self.bytes(alone[0]), Self.bytes(batched[row]),
                    "row \(row) window \(window ?? 0)")
            }
        }
    }

    // MARK: - Bit equality

    func testBatchInvarianceRowAloneVersusSevenOthers() throws {
        let proj = Self.makeProj(seed: 15, dtype: .bfloat16)
        let histories = [5, 900, 4000, 1500, 17, 2049, 3333, 128]
        for window in [nil, 1024] as [Int?] {
            for l in [1, 9, 16, 40] {
                let seqs = histories.enumerated().map {
                    Self.makeSequence(total: $1 + l, seed: UInt64(100 + $0), dtype: .bfloat16)
                }
                let (pool, tables, lengths) = Self.layout(
                    seqs, history: histories, ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                    seed: 21)
                let all = Self.call(
                    seqs, start: histories, l: l, tableLengths: lengths, dtype: .bfloat16)
                let batched = Self.run(all, pool: pool, tables: tables, proj: proj, window: window)
                    .output
                for row in histories.indices {
                    let one = Self.call(
                        [seqs[row]], start: [histories[row]], l: l, tableLengths: [lengths[row]],
                        dtype: .bfloat16)
                    let alone = Self.run(
                        one, pool: pool, tables: tables[row ..< row + 1], proj: proj, window: window
                    ).output
                    XCTAssertEqual(
                        Self.bytes(alone[0]), Self.bytes(batched[row]),
                        "row \(row) L \(l) window \(window ?? 0)")
                }
            }
        }
    }

    /// One or two simdgroups per partition (each half partition its own
    /// online softmax, the halves merged in order): the same bits for split
    /// and fused decode and verification, with and without fused writes, and
    /// for a row alone or among others.
    func testSubPartitionsGiveTheSameBits() throws {
        let proj = Self.makeProj(seed: 17, dtype: .bfloat16)
        let histories = [5, 900, 2000, 63, 64, 65, 127, 128, 129, 191, 192, 1500]
        for window in [nil, 1024] as [Int?] {
            for l in [1, 2, 5, 16] {
                let seqs = histories.enumerated().map {
                    Self.makeSequence(total: $1 + l, seed: UInt64(300 + $0), dtype: .bfloat16)
                }
                let (pool, tables, lengths) = Self.layout(
                    seqs, history: histories, ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                    seed: 31)
                let all = Self.call(
                    seqs, start: histories, l: l, tableLengths: lengths, dtype: .bfloat16)
                var outputs: [Data] = []
                for kernel in [Attention.Kernel.split, .sequential] {
                    for psub in [1, 2] {
                        outputs.append(
                            Self.bytes(
                                Self.run(
                                    all, pool: pool, tables: tables, proj: proj, window: window,
                                    kernel: kernel, subPartitions: psub
                                ).output))
                    }
                }
                for (index, data) in outputs.enumerated() {
                    XCTAssertEqual(data, outputs[0], "window \(window ?? 0) L \(l) run \(index)")
                }
                // Fused (at most 8 rows, 16 partitions) for the short rows.
                let short = [0, 3, 4, 5, 6, 7]
                let some = Self.call(
                    short.map { seqs[$0] }, start: short.map { histories[$0] }, l: l,
                    tableLengths: short.map { lengths[$0] }, dtype: .bfloat16)
                let someTables = concatenated(short.map { tables[$0 ..< $0 + 1] }, axis: 0)
                let reference = Self.bytes(
                    Self.run(
                        some, pool: pool, tables: someTables, proj: proj, window: window,
                        kernel: .split, subPartitions: 1
                    ).output)
                for psub in [1, 2] {
                    XCTAssertEqual(
                        Self.bytes(
                            Self.run(
                                some, pool: pool, tables: someTables, proj: proj, window: window,
                                kernel: .fused, subPartitions: psub
                            ).output), reference, "fused psub \(psub) window \(window ?? 0) L \(l)")
                }
            }
        }
    }

    func testDecodeVerifyAndPrefillChunksAreBitIdentical() throws {
        let proj = Self.makeProj(seed: 16, dtype: .bfloat16)
        let total = 1300
        for window in [nil, 1024] as [Int?] {
            let seq = Self.makeSequence(total: total, seed: 42, dtype: .bfloat16)
            // Every position paged: a call at `start` reads [0, start) from pages.
            let (pool, tables, lengths) = Self.layout(
                [seq], history: [total], ringPages: nil, dtype: .bfloat16, seed: 4)
            func attend(_ start: Int, _ l: Int, _ kernel: Attention.Kernel) -> MLXArray {
                let c = Self.call(
                    [seq], start: [start], l: l, tableLengths: lengths, dtype: .bfloat16)
                return Self.run(
                    c, pool: pool, tables: tables, proj: proj, window: window, kernel: kernel
                )
                .output[0]
            }
            let whole = attend(0, total, .sequential)
            for cuts in [[0, 512, 1024, total], [0, 333, 1033, total], [0, 1, 2, 130, total]] {
                let pieces = zip(cuts, cuts.dropFirst()).map { attend($0, $1 - $0, .sequential) }
                XCTAssertEqual(
                    Self.bytes(concatenated(pieces, axis: 0)), Self.bytes(whole), "chunks \(cuts)")
            }
            for start in [0, 127, 128, 1000, 1023, 1024, 1291] {
                let l = min(9, total - start)
                let verify = attend(start, l, .split)
                XCTAssertEqual(
                    Self.bytes(verify), Self.bytes(whole[start ..< start + l]), "verify at \(start)"
                )
                XCTAssertEqual(
                    Self.bytes(attend(start, l, .sequential)), Self.bytes(verify),
                    "sequential verify at \(start)")
                XCTAssertEqual(
                    Self.bytes(attend(start, l, .fused)), Self.bytes(verify),
                    "fused verify at \(start)")
                for j in 0 ..< l {
                    for kernel in [Attention.Kernel.split, .fused] {
                        XCTAssertEqual(
                            Self.bytes(attend(start + j, 1, kernel)),
                            Self.bytes(whole[(start + j) ..< (start + j + 1)]),
                            "\(kernel) decode at \(start + j)")
                    }
                }
            }
            for kernel in [Attention.Kernel.split, .fused] {
                let wide = attend(600, 16, kernel)
                XCTAssertEqual(Self.bytes(wide), Self.bytes(whole[600 ..< 616]), "\(kernel) wide")
            }
        }
    }

    func testFusedWriteStoresTokensAndKeepsOutput() throws {
        let proj = Self.makeProj(seed: 17, dtype: .bfloat16)
        for window in [nil, 1024] as [Int?] {
            for l in [1, 9] {
                let histories = [100, 1500]
                let seqs = histories.map {
                    Self.makeSequence(total: $0 + l, seed: UInt64($0 + l), dtype: .bfloat16)
                }
                // Lay out the full sequences, then clear the new positions' slots.
                let full = histories.map { $0 + l }
                let (pool, tables, lengths) = Self.layout(
                    seqs, history: full, ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                    seed: 6)
                let tableRows = tables.asArray(Int32.self)
                let width = tables.dim(1)
                var flatSlots: [Int32] = []
                for (row, h) in histories.enumerated() {
                    for i in 0 ..< l {
                        let pos = h + i
                        let lp = pos / Self.s
                        let page = tableRows[row * width + (window != nil ? lp % 65 : lp)]
                        flatSlots.append(page * Int32(Self.s) + Int32(pos % Self.s))
                    }
                }
                let expectedK = Self.bytes(pool.kSlab)
                let expectedV = Self.bytes(pool.vSlab)
                let flat = MLXArray(flatSlots)
                func cleared(_ slab: MLXArray) -> MLXArray {
                    let rows = slab.transposed(0, 2, 1, 3).reshaped(-1, Self.kvh, Self.d)
                    rows[flat] = MLXArray.zeros([flat.dim(0), Self.kvh, Self.d], dtype: slab.dtype)
                    return rows.reshaped(-1, Self.s, Self.kvh, Self.d).transposed(0, 2, 1, 3)
                        .contiguous()
                }
                let writable = Pool(
                    kSlab: cleared(pool.kSlab), vSlab: cleared(pool.vSlab), pages: pool.pages)
                eval(writable.kSlab, writable.vSlab)
                let c = Self.call(
                    seqs, start: histories, l: l, tableLengths: lengths, dtype: .bfloat16)
                let plain = Self.run(c, pool: pool, tables: tables, proj: proj, window: window)
                let fence = MLXArray([Int32(41)])
                let fused = Self.run(
                    c, pool: writable, tables: tables, proj: proj, window: window,
                    write: true, fence: fence)
                eval(fused.output, fused.writeFence)
                XCTAssertEqual(fused.writeFence.item(Int32.self), 42)
                XCTAssertEqual(Self.bytes(fused.output), Self.bytes(plain.output))
                XCTAssertEqual(
                    Self.bytes(writable.kSlab), expectedK, "window \(window ?? 0) L \(l)")
                XCTAssertEqual(Self.bytes(writable.vSlab), expectedV)
            }
        }
    }

    /// The same pages split over segments (`[2, pages, KVH, 16, D]` each,
    /// local page 0 inert) give the same bits as the slab, read and fused write.
    /// The pool's pages split at random into `count` segments (local page 0 of
    /// each copies the slab's zero poison page) and the matching tables.
    static func segmentize(
        _ pool: Pool, tables: MLXArray, count: Int, seed: UInt64
    ) -> (segments: [MLXArray], tables: MLXArray) {
        var rng = SplitMix(seed: seed)
        var binding = [0]
        var local = [0]
        var counts = [Int](repeating: 1, count: count)
        for _ in 1 ..< pool.pages {
            let sIndex = Int(rng.next() % UInt64(count))
            binding.append(sIndex)
            local.append(counts[sIndex])
            counts[sIndex] += 1
        }
        let segments = (0 ..< count).map { sIndex -> MLXArray in
            let members = (1 ..< pool.pages).filter { binding[$0] == sIndex }
            let order = MLXArray([Int32(0)] + members.map { Int32($0) })
            return stacked([pool.kSlab[order], pool.vSlab[order]])
        }
        eval(segments)
        let entries = tables.asArray(Int32.self).map { page -> Int32 in
            page == 0 ? 0 : Int32(binding[Int(page)] << 24 | local[Int(page)])
        }
        return (segments, MLXArray(entries, tables.shape))
    }

    /// Calls over 1 ... 16 segments compile nothing after a warm-up with 1, 8
    /// and 16: every pass binds one of two fixed counts, so no kernel depends
    /// on the pool's segment count. Outputs equal slabs throughout.
    func testKernelsDoNotDependOnTheSegmentCount() throws {
        let proj = Self.makeProj(seed: 23, dtype: .bfloat16)
        // Fused decode (one sliding row), split decode and verification (three
        // rows, one past the fused partition limit), and the sequential kernel.
        let cases: [(histories: [Int], l: Int, window: Int?, kernel: Attention.Kernel)] = [
            ([700], 1, 1024, .automatic), ([700, 40, 2500], 1, nil, .automatic),
            ([700, 40, 2500], 9, nil, .automatic), ([700], 40, nil, .sequential),
        ]
        func run(segments count: Int) {
            for (index, c) in cases.enumerated() {
                let seqs = c.histories.enumerated().map {
                    Self.makeSequence(
                        total: $1 + c.l, seed: UInt64(500 + 10 * index + $0), dtype: .bfloat16)
                }
                let (pool, tables, lengths) = Self.layout(
                    seqs, history: c.histories, ringPages: c.window.map { _ in 65 },
                    dtype: .bfloat16, seed: UInt64(41 + index))
                let call = Self.call(
                    seqs, start: c.histories, l: c.l, tableLengths: lengths, dtype: .bfloat16)
                let slab = Self.run(
                    call, pool: pool, tables: tables, proj: proj, window: c.window,
                    kernel: c.kernel)
                let (segments, segmentTables) = Self.segmentize(
                    pool, tables: tables, count: count, seed: UInt64(count))
                let segmented = Attention.attend(
                    queries: call.queries, newKeys: call.newKeys, newValues: call.newValues,
                    relativeStates: call.states, proj: proj, sinks: call.sinks,
                    cache: .init(
                        segments: segments, tables: segmentTables,
                        writeFence: MLXArray.zeros([1], dtype: .int32)),
                    rows: call.rows, window: c.window, scale: Self.scale, kernel: c.kernel)
                XCTAssertEqual(
                    Self.bytes(segmented.output), Self.bytes(slab.output),
                    "\(count) segments, case \(index)")
            }
        }
        for count in [1, 8, 16] { run(segments: count) }
        let compiled = Attention.kernels.count
        for count in 1 ... 16 {
            run(segments: count)
            XCTAssertEqual(Attention.kernels.count, compiled, "\(count) segments compiled more")
        }
    }

    func testSegmentedStorageMatchesSlabBitwise() throws {
        let proj = Self.makeProj(seed: 19, dtype: .bfloat16)
        let histories = [700, 2500, 40]
        for window in [nil, 1024] as [Int?] {
            for (l, kernel) in [
                (1, Attention.Kernel.split), (9, .split), (1, .fused), (9, .automatic),
                (300, .sequential), (300, .prefill),
            ]
            // Full rows of 2,500 tokens span more partitions than the fused kernel takes.
            where (kernel != .fused || window != nil)
                && (kernel != .prefill || Attention.neuralAcceleratorsAvailable)
            {
                let seqs = histories.enumerated().map {
                    Self.makeSequence(total: $1 + l, seed: UInt64(300 + $0 + l), dtype: .bfloat16)
                }
                // Pages hold the new positions too, so the fused write has slots.
                let (pool, tables, lengths) = Self.layout(
                    seqs, history: histories.map { $0 + l }, ringPages: window.map { _ in 65 },
                    dtype: .bfloat16, seed: 31)
                // Three segments for decode; twelve (16 bindings, every kernel's
                // largest buffer list) for verification and prefill.
                let segmentCount = l == 1 ? 3 : 12
                var rng = SplitMix(seed: UInt64(l))
                var binding = [0]
                var local = [0]
                var counts = [Int](repeating: 1, count: segmentCount)
                for _ in 1 ..< pool.pages {
                    let sIndex = Int(rng.next() % UInt64(segmentCount))
                    binding.append(sIndex)
                    local.append(counts[sIndex])
                    counts[sIndex] += 1
                }
                var segments: [MLXArray] = []
                for sIndex in 0 ..< segmentCount {
                    let members = (1 ..< pool.pages).filter { binding[$0] == sIndex }
                    // Local page 0 copies the slab's poison page (zeros): inert.
                    let order = MLXArray([Int32(0)] + members.map { Int32($0) })
                    segments.append(stacked([pool.kSlab[order], pool.vSlab[order]]))
                }
                eval(segments)
                let tableRows = tables.asArray(Int32.self)
                let width = tables.dim(1)
                let entries = tableRows.map { page -> Int32 in
                    page == 0 ? 0 : Int32(binding[Int(page)] << 24 | local[Int(page)])
                }
                let segmentTables = MLXArray(entries, tables.shape)
                let c = Self.call(
                    seqs, start: histories, l: l, tableLengths: lengths, dtype: .bfloat16)
                let slab = Self.run(
                    c, pool: pool, tables: tables, proj: proj, window: window, kernel: kernel)
                func attendSegments(_ segs: [MLXArray], write: Bool) -> Attention.Output {
                    Attention.attend(
                        queries: c.queries, newKeys: c.newKeys, newValues: c.newValues,
                        relativeStates: c.states, proj: proj, sinks: c.sinks,
                        cache: .init(
                            segments: segs, tables: segmentTables,
                            writeFence: MLXArray.zeros([1], dtype: .int32)),
                        rows: c.rows, window: window, scale: Self.scale, write: write,
                        kernel: kernel)
                }
                let segmented = attendSegments(segments, write: false)
                XCTAssertEqual(
                    Self.bytes(segmented.output), Self.bytes(slab.output),
                    "window \(window ?? 0) L \(l)")
                guard l <= 9 else { continue }
                // Fused write into copies whose new-token slots are cleared: the
                // tokens land in their segment pages and the output is unchanged.
                var targets: [(segment: Int, local: Int, slot: Int, row: Int, i: Int)] = []
                for (row, h) in histories.enumerated() {
                    for i in 0 ..< l {
                        let pos = h + i
                        let lp = pos / Self.s
                        let page = Int(tableRows[row * width + (window != nil ? lp % 65 : lp)])
                        targets.append((binding[page], local[page], pos % Self.s, row, i))
                    }
                }
                var writable = segments.map { $0 + MLXArray.zeros($0.shape, dtype: $0.dtype) }
                for target in targets {
                    writable[target.segment][0..., target.local, 0..., target.slot] =
                        MLXArray.zeros([2, Self.kvh, Self.d], dtype: .bfloat16)
                }
                writable = writable.map { $0.contiguous() }
                eval(writable)
                let fused = attendSegments(writable, write: true)
                eval(fused.output, fused.writeFence)
                XCTAssertEqual(Self.bytes(fused.output), Self.bytes(slab.output))
                for target in targets {
                    let segment = writable[target.segment]
                    XCTAssertEqual(
                        Self.bytes(segment[0, target.local, 0..., target.slot]),
                        Self.bytes(c.newKeys[target.row, 0..., target.i]), "key \(target)")
                    XCTAssertEqual(
                        Self.bytes(segment[1, target.local, 0..., target.slot]),
                        Self.bytes(c.newValues[target.row, 0..., target.i]), "value \(target)")
                }
            }
        }
    }

    /// Token-major inputs (`[B, L, heads, ...]`, the projections' own order)
    /// give the same bits as head-major ones, for both kernels and the merge.
    func testTokenMajorLayoutMatchesHeadMajor() throws {
        let proj = Self.makeProj(seed: 20, dtype: .bfloat16)
        let histories = [700, 2500, 40]
        for window in [nil, 1024] as [Int?] {
            for (l, kernel) in [
                (1, Attention.Kernel.split), (9, .split), (1, .fused), (9, .fused),
                (300, .sequential), (300, .prefill),
            ]
            // Full rows of 2,500 tokens span more partitions than the fused kernel takes.
            where (kernel != .fused || window != nil)
                && (kernel != .prefill || Attention.neuralAcceleratorsAvailable)
            {
                let seqs = histories.enumerated().map {
                    Self.makeSequence(total: $1 + l, seed: UInt64(500 + $0 + l), dtype: .bfloat16)
                }
                let (pool, tables, lengths) = Self.layout(
                    seqs, history: histories, ringPages: window.map { _ in 65 }, dtype: .bfloat16,
                    seed: 41)
                let c = Self.call(
                    seqs, start: histories, l: l, tableLengths: lengths, dtype: .bfloat16)
                let headMajor = Self.run(
                    c, pool: pool, tables: tables, proj: proj, window: window, kernel: kernel)
                func tokenMajor(_ x: MLXArray) -> MLXArray {
                    (x.ndim == 4 ? x.transposed(0, 2, 1, 3) : x.transposed(0, 2, 1)).contiguous()
                }
                let swapped = Attention.attend(
                    queries: tokenMajor(c.queries), newKeys: tokenMajor(c.newKeys),
                    newValues: tokenMajor(c.newValues), relativeStates: tokenMajor(c.states),
                    proj: proj, sinks: tokenMajor(c.sinks),
                    cache: .init(
                        keys: pool.kSlab, values: pool.vSlab, tables: tables,
                        writeFence: MLXArray.zeros([1], dtype: .int32)),
                    rows: c.rows, window: window, scale: Self.scale, layout: .tokenMajor,
                    kernel: kernel)
                XCTAssertEqual(
                    Self.bytes(swapped.output), Self.bytes(headMajor.output),
                    "window \(window ?? 0) L \(l)")
            }
        }
    }

    // MARK: - Engine pages

    /// Pages written by the engine's own pool (`PagedSequenceKV.write`, fixed
    /// slab, sliding ring and full table) read back exactly.
    func testReadsEnginePoolPagesForFullAndRingRows() throws {
        let proj = Self.makeProj(seed: 18, dtype: .bfloat16)
        let kinds = [
            CBv2LayerKind(
                attention: .slidingWindow(1024), headDim: 128, kvHeads: 8, queryHeads: 24),
            CBv2LayerKind(attention: .full, headDim: 128, kvHeads: 8, queryHeads: 24),
        ]
        let history = 2600
        let l = 9
        let seq = Self.makeSequence(total: history + l, seed: 99, dtype: .bfloat16)
        let expected = [1024, nil].map {
            Self.reference(seq, start: history, l: l, base: 0, proj: proj, window: $0)
        }
        // Fixed slabs, then the serving layout's segments (4 MiB each).
        var outputs: [[Data]] = []
        for segmentBytes in [nil, 4 << 20] as [Int?] {
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: 512 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
                    nominalMaxSequenceLength: 4096, segmentSizeBytes: segmentBytes))
            let states = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: history, maxLength: 4096)
            defer { backend.release(states) }
            var layoutOutputs: [Data] = []
            for (index, window) in [(0, 1024), (1, nil)] as [(Int, Int?)] {
                let row = try XCTUnwrap(states[index] as? PagedSequenceKV)
                for start in stride(from: 0, to: history, by: 512) {
                    let end = min(history, start + 512)
                    row.write(
                        keys: seq.keys[0..., start ..< end].asType(.bfloat16),
                        values: seq.values[0..., start ..< end].asType(.bfloat16))
                }
                let group = backend.pool.group(row.groupKey)
                eval(group.writeFence)
                let cache: Attention.PagedKV
                if let layout = group.segmentLayout {
                    let ids = Array(Set(row.table.map { layout.segmentIndex(page: $0) })).sorted()
                    XCTAssertLessThanOrEqual(ids.count, Attention.maximumSegments)
                    let bindingOf = Dictionary(
                        uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
                    var entries = row.table.map {
                        Int32(
                            bindingOf[layout.segmentIndex(page: $0)]! << 24 | layout.localPage($0))
                    }
                    while entries.count < 8 { entries.append(0) }
                    cache = .init(
                        segments: ids.map { group.segments[$0]!.storage },
                        tables: MLXArray(entries, [1, entries.count]),
                        writeFence: group.writeFence)
                } else {
                    var table = row.table
                    while table.count < 8 { table.append(0) }
                    cache = .init(
                        keys: group.kSlab, values: group.vSlab,
                        tables: MLXArray(table, [1, table.count]), writeFence: group.writeFence)
                }
                let c = Self.call(
                    [seq], start: [history], l: l, tableLengths: [row.decodeTableLength],
                    dtype: .bfloat16)
                let out = Attention.attend(
                    queries: c.queries, newKeys: c.newKeys, newValues: c.newValues,
                    relativeStates: c.states, proj: proj, sinks: c.sinks, cache: cache,
                    rows: c.rows, window: window, scale: Self.scale)
                var stats = ErrorStats()
                stats.add(out.output[0], expected[index])
                XCTAssertLessThan(
                    stats.maxAbs, 4e-3 * max(1, abs(expected[index]).max().item(Float.self)),
                    "engine pages window \(window ?? 0) segments \(segmentBytes ?? 0)")
                layoutOutputs.append(Self.bytes(out.output))
            }
            outputs.append(layoutOutputs)
        }
        XCTAssertEqual(outputs[0], outputs[1], "slab and segmented engine pages differ")
    }
}

/// Deterministic shuffles for page layouts.
struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
