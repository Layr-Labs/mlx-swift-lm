// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The kernel attention core through `PagedLayerCache.attendWithModelKernel`:
/// fixed slabs, segments bound directly, and segments past the binding limit
/// (decode and verification in segment buckets, prefill through a gathered
/// copy) give the same outputs and the same stored K/V for decode,
/// verification and prefill steps.
final class Inkling2PagedAttentionCoreTests: XCTestCase {
    static let qh = 24
    static let kvh = 8
    static let d = 128

    struct Step {
        let queries: MLXArray  // [1, L, QH, D] token-major
        let keys: MLXArray  // [1, L, KVH, D]
        let values: MLXArray
        let states: MLXArray  // [1, L, QH, 16]
        let sinks: MLXArray  // [1, L, QH]
    }

    static func step(_ l: Int, seed: UInt64) -> Step {
        MLXRandom.seed(seed)
        func r(_ shape: [Int], _ scale: Float) -> MLXArray {
            (MLXRandom.normal(shape) * scale).asType(.bfloat16)
        }
        return Step(
            queries: r([1, l, qh, d], 0.6), keys: r([1, l, kvh, d], 0.6),
            values: r([1, l, kvh, d], 1), states: r([1, l, qh, 16], 0.5),
            sinks: MLXRandom.normal([1, l, qh]))
    }

    /// Runs `history` tokens of writes, then `steps` calls (L each) through the
    /// core; returns each step's output bytes and the stored K/V bytes.
    /// `afterRelease`: another row first writes and is released, so a segmented
    /// pool has retired every segment when the measured row starts.
    func run(
        window: Int?, segmentBytes: Int?, expectGathered: Bool, history: Int = 600,
        steps: [Int] = [1, 4, 40], afterRelease: Bool = false
    ) throws -> [Data] {
        let kind = CBv2LayerKind(
            attention: window.map { .slidingWindow($0) } ?? .full, headDim: Self.d,
            kvHeads: Self.kvh, queryHeads: Self.qh)
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 256 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 4096, segmentSizeBytes: segmentBytes))
        if afterRelease {
            let other = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 48, maxLength: 4096)
            let warm = Self.step(48, seed: 3)
            try XCTUnwrap(other[0] as? PagedSequenceKV).write(
                keys: warm.keys[0].transposed(1, 0, 2), values: warm.values[0].transposed(1, 0, 2))
            backend.release(other)
        }
        // After a release, a small reservation takes a new (smaller) segment
        // size class: the retired segment 0 is not recommitted.
        let maxLength = afterRelease ? history + steps.reduce(0, +) + 16 : 4096
        let states = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: history, maxLength: maxLength)
        defer { backend.release(states) }
        let row = try XCTUnwrap(states[0] as? PagedSequenceKV)
        let prefix = Self.step(history, seed: 7)
        // Chunks within the pool's prefill chunk bound.
        for first in stride(from: 0, to: history, by: 300) {
            let range = first ..< min(history, first + 300)
            row.write(
                keys: prefix.keys[0, range].transposed(1, 0, 2),
                values: prefix.values[0, range].transposed(1, 0, 2))
        }
        let cache = try XCTUnwrap(backend.makeLayerCaches()[0] as? PagedLayerCache)
        cache.setRows([row])
        let group = backend.pool.group(row.groupKey)
        if afterRelease && history == 0, let layout = group.segmentLayout {
            // The released row's segment is retired: the poison page's segment
            // (the old seam's binding for a row without pages) is absent.
            XCTAssertNil(
                group.segments[layout.segmentIndex(page: PagedKVGroup.poisonPage)],
                "segments: \(group.segments.keys.sorted())")
        }
        if let layout = group.segmentLayout {
            let touched = Set(row.table.map { layout.segmentIndex(page: $0) }).count
            XCTAssertEqual(
                touched > PagedLayerCache.modelAttentionMaximumBindings, expectGathered,
                "segments touched: \(touched)")
        }
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let core = Inkling2KernelAttentionCore()
        var out: [Data] = []
        var written = [prefix.keys]
        var writtenValues = [prefix.values]
        for (index, l) in steps.enumerated() {
            let s = Self.step(l, seed: UInt64(100 + index))
            written.append(s.keys)
            writtenValues.append(s.values)
            let start = row.absoluteOffset
            let inputs = Inkling2AttentionInputs(
                queries: s.queries.transposed(0, 2, 1, 3),
                keys: s.keys.transposed(0, 2, 1, 3), values: s.values.transposed(0, 2, 1, 3),
                relativeLogits: MLXArray.zeros([1, Self.qh, l, 1024], dtype: .bfloat16),
                relativeStates: s.states, relativeProjection: proj,
                sinks: s.sinks.transposed(0, 2, 1),
                positions: MLXArray((start ..< start + l).map { Int32($0) }, [1, l]),
                window: window, scale: Float(1.0 / Double(Self.d).squareRoot()))
            let counters = PagedLayerCache.modelAttentionCounters
            counters.reset()
            let o = core.attend(slot: 0, cache: cache, rowOffsets: [start], inputs)
            eval(o)
            if expectGathered {
                // Past the binding limit: decode and verification calls run in
                // segment buckets, prefill chunks read a gathered copy.
                let snapshot = counters.snapshot
                XCTAssertEqual(snapshot.calls, 1)
                XCTAssertEqual(snapshot.gathered, l > 16 ? 1 : 0, "step \(index) L \(l)")
                if l <= 16 { XCTAssertGreaterThan(snapshot.buckets, 1, "step \(index)") }
            }
            XCTAssertEqual(row.absoluteOffset, start + l)
            XCTAssertEqual(o.shape, [1, l, Self.qh * Self.d])
            let magnitude = abs(o.asType(.float32)).max().item(Float.self)
            XCTAssertTrue(magnitude.isFinite && magnitude > 0, "step \(index): \(magnitude)")
            out.append(o.asData(access: .copy).data)
        }
        try backend.pool.writeValidation.check()
        let end = row.absoluteOffset
        let first = window.map { max(0, end - $0) } ?? 0
        let kv = row.gatherRange(start: first, count: end - first)
        // The stored K/V are the written tokens, whichever path wrote them.
        let expectedKeys = concatenated(written, axis: 1)[0, first ..< end].transposed(1, 0, 2)
        let expectedValues = concatenated(writtenValues, axis: 1)[0, first ..< end]
            .transposed(1, 0, 2)
        XCTAssertTrue(arrayEqual(kv.keys[0], expectedKeys).item(Bool.self), "stored keys")
        XCTAssertTrue(arrayEqual(kv.values[0], expectedValues).item(Bool.self), "stored values")
        eval(kv.keys, kv.values)
        out.append(kv.keys.asData(access: .copy).data)
        out.append(kv.values.asData(access: .copy).data)
        return out
    }

    /// The seam on a fresh segmented pool and across segment boundaries: a
    /// fresh prompt's first chunk (no pages yet, the pool holds no segment),
    /// a fresh row's first decode steps (fused writes into its first page), and
    /// decode steps whose writes land in a new segment while the history sits
    /// in the previous ones (three usable pages per segment). Same bits and
    /// stored K/V as fixed slabs.
    func testFreshRowsAndWritesIntoNewSegments() throws {
        let pageBytes = 2 * Self.kvh * 16 * Self.d * 2
        for window in [nil, 1024] as [Int?] {
            for (history, steps, segmentBytes) in [
                (0, [40, 1, 4], 4 << 20), (0, [1, 1, 4, 40], 4 << 20),
                (48, Array(repeating: 1, count: 20) + [4, 40], 4 * pageBytes),
                (40, [1, 4, 4, 4, 4], 4 * pageBytes),
            ] as [(Int, [Int], Int)] {
                let slab = try run(
                    window: window, segmentBytes: nil, expectGathered: false, history: history,
                    steps: steps)
                for afterRelease in [false, true] {
                    let segments = try run(
                        window: window, segmentBytes: segmentBytes, expectGathered: false,
                        history: history, steps: steps, afterRelease: afterRelease)
                    XCTAssertEqual(
                        segments, slab,
                        "window \(window ?? 0) history \(history) steps \(steps) "
                            + "after release \(afterRelease)")
                }
            }
        }
    }

    /// One core serves every engine on a loaded model, and engines build
    /// steps on their own threads. Six engines (each its own pool, the same
    /// rows and inputs, so the same row geometry) step a sliding and a full
    /// layer through one shared core from six threads at once: every engine
    /// gets, step for step, the bits one engine gets alone, round after round.
    /// The steps run the one-dispatch kernel (1 and 4 tokens) and the split
    /// kernel (16 tokens); engines never share a plan or its fused workspace.
    func testOneCoreServesEnginesOnManyThreads() throws {
        let kinds = [
            CBv2LayerKind(
                attention: .slidingWindow(1024), headDim: Self.d, kvHeads: Self.kvh,
                queryHeads: Self.qh),
            CBv2LayerKind(
                attention: .full, headDim: Self.d, kvHeads: Self.kvh, queryHeads: Self.qh),
        ]
        let history = 600
        let lengths = [1, 1, 4, 1, 16, 1, 1, 4]
        let prefix = Self.step(history, seed: 7)
        let steps = lengths.enumerated().map { Self.step($1, seed: UInt64(200 + $0)) }
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        eval(proj)
        struct Engine {
            let backend: PagedKVBackend
            let rows: [PagedSequenceKV]
            let caches: [PagedLayerCache]
        }
        func engine() throws -> Engine {
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: 256 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
                    nominalMaxSequenceLength: 4096, segmentSizeBytes: 4 << 20))
            let states = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: history, maxLength: 4096)
            let rows = try states.map { try XCTUnwrap($0 as? PagedSequenceKV) }
            for row in rows {
                for first in stride(from: 0, to: history, by: 300) {
                    let range = first ..< min(history, first + 300)
                    row.write(
                        keys: prefix.keys[0, range].transposed(1, 0, 2),
                        values: prefix.values[0, range].transposed(1, 0, 2))
                }
            }
            let caches = try backend.makeLayerCaches().map { try XCTUnwrap($0 as? PagedLayerCache) }
            for (cache, row) in zip(caches, rows) { cache.setRows([row]) }
            return Engine(backend: backend, rows: rows, caches: caches)
        }
        /// Every step's output of both layers, as bytes.
        func run(_ engine: Engine, core: Inkling2KernelAttentionCore) -> [Data] {
            var out: [Data] = []
            for (index, s) in steps.enumerated() {
                let l = lengths[index]
                for (layer, kind) in kinds.enumerated() {
                    let start = engine.rows[layer].absoluteOffset
                    let window: Int? = kind.attention == .full ? nil : 1024
                    let inputs = Inkling2AttentionInputs(
                        queries: s.queries.transposed(0, 2, 1, 3),
                        keys: s.keys.transposed(0, 2, 1, 3),
                        values: s.values.transposed(0, 2, 1, 3),
                        relativeLogits: MLXArray.zeros([1, Self.qh, l, 1024], dtype: .bfloat16),
                        relativeStates: s.states, relativeProjection: proj,
                        sinks: s.sinks.transposed(0, 2, 1),
                        positions: MLXArray((start ..< start + l).map { Int32($0) }, [1, l]),
                        window: window, scale: Float(1.0 / Double(Self.d).squareRoot()))
                    let o = core.attend(
                        slot: layer, cache: engine.caches[layer], rowOffsets: [start], inputs)
                    eval(o)
                    out.append(o.asData(access: .copy).data)
                }
            }
            return out
        }
        let alone = try engine()
        let expected = run(alone, core: Inkling2KernelAttentionCore())
        alone.backend.release(alone.rows.map { $0 as CBv2SequenceKV? })
        let threads = 6
        for round in 0 ..< 5 {
            let engines = try (0 ..< threads).map { _ in try engine() }
            let core = Inkling2KernelAttentionCore()
            let results = ResultSlots(count: threads)
            DispatchQueue.concurrentPerform(iterations: threads) { index in
                results.set(index, run(engines[index], core: core))
            }
            for index in 0 ..< threads {
                XCTAssertEqual(results.get(index), expected, "round \(round) engine \(index)")
                try engines[index].backend.pool.writeValidation.check()
                engines[index].backend.release(engines[index].rows.map { $0 as CBv2SequenceKV? })
            }
        }
    }

    func testSlabsSegmentsAndGatheredSegmentsGiveTheSameBits() throws {
        for window in [nil, 1024] as [Int?] {
            let slab = try run(window: window, segmentBytes: nil, expectGathered: false)
            let segments = try run(window: window, segmentBytes: 4 << 20, expectGathered: false)
            // One usable 64 KiB page per segment: the row spans more segments
            // than one dispatch binds.
            let gathered = try run(window: window, segmentBytes: 128 << 10, expectGathered: true)
            XCTAssertEqual(segments, slab, "window \(window ?? 0): segments")
            XCTAssertEqual(gathered, slab, "window \(window ?? 0): bucketed / gathered segments")
        }
    }

    /// Several rows decoding together over many segments: steps of 1-16
    /// tokens run in segment buckets (whole rows per bucket with four usable
    /// pages per segment, single partitions with one), a 40-token chunk reads a
    /// gathered copy, and so does an 8-token prompt chunk (`isPrefill`: the
    /// prefill kernel, on GPUs that have it, reads each row in one
    /// threadgroup). Same outputs and stored K/V as fixed slabs, including a
    /// sliding row whose ring has wrapped.
    func testSegmentBucketsGiveTheSameBitsAsSlabs() throws {
        let pageBytes = 2 * Self.kvh * 16 * Self.d * 2
        let steps = [1, 1, 4, 1, 16, 1, 40, 8, 1, 3]
        for (window, histories) in [
            (nil, [600, 37, 300, 129]), (1024, [600, 37, 1500, 129]),
        ] as [(Int?, [Int])] {
            let slab = try runRows(
                window: window, segmentBytes: nil, histories: histories, steps: steps,
                path: .inPlace)
            var compiled: Int?
            for (usable, path) in [(4, .buckets), (1, .buckets), (1, .gathered)] as [(Int, Path)] {
                let segments = try runRows(
                    window: window, segmentBytes: (usable + 1) * pageBytes,
                    histories: histories, steps: steps, path: path)
                let label = "window \(window ?? 0) usable \(usable) \(path)"
                // Buckets of other segment counts run the kernels already built.
                if path == .buckets {
                    if let compiled {
                        XCTAssertEqual(Inkling2PagedAttention.kernels.count, compiled, label)
                    }
                    compiled = Inkling2PagedAttention.kernels.count
                }
                XCTAssertEqual(segments.count, slab.count, label)
                for (index, (a, b)) in zip(segments, slab).enumerated() {
                    XCTAssertEqual(a, b, "\(label) output \(index)")
                }
            }
        }
    }

    /// How the seam is expected to bind the pages.
    enum Path {
        /// Slabs or at most 16 segments.
        case inPlace
        /// Past the binding limit with bucketing: buckets for calls of at
        /// most 16 tokens, a gathered copy for longer ones.
        case buckets
        /// Past the binding limit without bucketing: always a gathered copy.
        case gathered
    }

    /// `histories.count` rows prefilled one after another, then decoding
    /// `steps` together; returns each step's output bytes and each row's
    /// stored K/V bytes.
    func runRows(
        window: Int?, segmentBytes: Int?, histories: [Int], steps: [Int], path: Path,
        prefillSteps: Set<Int> = [7]
    ) throws -> [Data] {
        let kind = CBv2LayerKind(
            attention: window.map { .slidingWindow($0) } ?? .full, headDim: Self.d,
            kvHeads: Self.kvh, queryHeads: Self.qh)
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 512 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 4096, segmentSizeBytes: segmentBytes))
        var rows: [PagedSequenceKV] = []
        var held: [[CBv2SequenceKV?]] = []
        var written: [[MLXArray]] = []
        var writtenValues: [[MLXArray]] = []
        defer { for state in held { backend.release(state) } }
        for (index, history) in histories.enumerated() {
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: history, maxLength: 4096)
            held.append(state)
            let row = try XCTUnwrap(state[0] as? PagedSequenceKV)
            let prefix = Self.step(history, seed: UInt64(7 + index))
            for first in stride(from: 0, to: history, by: 300) {
                let range = first ..< min(history, first + 300)
                row.write(
                    keys: prefix.keys[0, range].transposed(1, 0, 2),
                    values: prefix.values[0, range].transposed(1, 0, 2))
            }
            rows.append(row)
            written.append([prefix.keys[0]])
            writtenValues.append([prefix.values[0]])
        }
        let cache = try XCTUnwrap(backend.makeLayerCaches()[0] as? PagedLayerCache)
        cache.setRows(rows)
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let core = Inkling2KernelAttentionCore()
        if path == .gathered { core.bucketing = nil }
        let counters = PagedLayerCache.modelAttentionCounters
        var out: [Data] = []
        for (index, l) in steps.enumerated() {
            let parts = rows.indices.map { Self.step(l, seed: UInt64(100 + 10 * index + $0)) }
            func joined(_ x: (Step) -> MLXArray) -> MLXArray {
                concatenated(parts.map(x), axis: 0)
            }
            for (r, part) in parts.enumerated() {
                written[r].append(part.keys[0])
                writtenValues[r].append(part.values[0])
            }
            let starts = rows.map(\.absoluteOffset)
            let inputs = Inkling2AttentionInputs(
                queries: joined(\.queries).transposed(0, 2, 1, 3),
                keys: joined(\.keys).transposed(0, 2, 1, 3),
                values: joined(\.values).transposed(0, 2, 1, 3),
                relativeLogits: MLXArray.zeros([rows.count, Self.qh, l, 1024], dtype: .bfloat16),
                relativeStates: joined(\.states), relativeProjection: proj,
                sinks: joined(\.sinks).transposed(0, 2, 1),
                positions: MLXArray(
                    starts.flatMap { start in (start ..< start + l).map { Int32($0) } },
                    [rows.count, l]),
                window: window, scale: Float(1.0 / Double(Self.d).squareRoot()),
                isPrefill: prefillSteps.contains(index))
            counters.reset()
            let o = core.attend(slot: 0, cache: cache, rowOffsets: starts, inputs)
            eval(o)
            let snapshot = counters.snapshot
            XCTAssertEqual(snapshot.calls, 1)
            // Calls on the prefill kernel take no buckets.
            let prefillKernel =
                l > 16
                || (prefillSteps.contains(index)
                    && Inkling2PagedAttention.neuralAcceleratorsAvailable)
            switch path {
            case .inPlace:
                XCTAssertEqual(snapshot.gathered, 0, "step \(index) L \(l)")
            case .buckets:
                XCTAssertEqual(snapshot.gathered, prefillKernel ? 1 : 0, "step \(index) L \(l)")
                if !prefillKernel { XCTAssertGreaterThan(snapshot.buckets, 1, "step \(index)") }
            case .gathered:
                XCTAssertEqual(snapshot.gathered, 1, "step \(index) L \(l)")
            }
            XCTAssertEqual(rows.map(\.absoluteOffset), starts.map { $0 + l })
            let magnitude = abs(o.asType(.float32)).max().item(Float.self)
            XCTAssertTrue(magnitude.isFinite && magnitude > 0, "step \(index): \(magnitude)")
            out.append(o.asData(access: .copy).data)
        }
        try backend.pool.writeValidation.check()
        for (r, row) in rows.enumerated() {
            let end = row.absoluteOffset
            let first = window.map { max(0, end - $0) } ?? 0
            let kv = row.gatherRange(start: first, count: end - first)
            let keys = concatenated(written[r], axis: 0)[first ..< end].transposed(1, 0, 2)
            let values = concatenated(writtenValues[r], axis: 0)[first ..< end].transposed(1, 0, 2)
            XCTAssertTrue(arrayEqual(kv.keys[0], keys).item(Bool.self), "row \(r) stored keys")
            XCTAssertTrue(
                arrayEqual(kv.values[0], values).item(Bool.self), "row \(r) stored values")
            eval(kv.keys, kv.values)
            out.append(kv.keys.asData(access: .copy).data)
            out.append(kv.values.asData(access: .copy).data)
        }
        return out
    }
}

/// Per-thread results of a concurrent test run.
private final class ResultSlots: @unchecked Sendable {
    private let lock = NSLock()
    private var slots: [[Data]]

    init(count: Int) { slots = Array(repeating: [], count: count) }

    func set(_ index: Int, _ value: [Data]) { lock.withLock { slots[index] = value } }
    func get(_ index: Int) -> [Data] { lock.withLock { slots[index] } }
}
