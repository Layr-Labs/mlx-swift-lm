// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

/// `mtpBatchesRectangularColumnsAsRows`: a verify round attended as `B * W`
/// rows of one paged decode dispatch gives each column the output bytes, and
/// leaves the K/V bytes, of its request's one-token steps.
@Suite(.serialized)
struct PagedColumnsAsDecodeRowsTests {
    static let window = 2048

    struct Shape: Sendable {
        let headDim: Int
        let kvHeads: Int
        let queryHeads: Int
    }

    /// Muse Glimmer's attention, and a small GQA shape.
    static let shapes = [
        Shape(headDim: 128, kvHeads: 2, queryHeads: 32),
        Shape(headDim: 64, kvHeads: 2, queryHeads: 4),
    ]

    /// One layer. Each request has two rows with the same prompt: `rounds`
    /// attends one, `steps` the other, one token at a time.
    final class Fixture {
        let backend: PagedKVBackend
        let shape: Shape
        let dtype: DType
        let rounds: PagedLayerCache
        let steps: PagedLayerCache
        private var states: [[CBv2SequenceKV?]] = []
        private(set) var roundRows: [PagedSequenceKV] = []
        private(set) var stepRows: [PagedSequenceKV] = []
        private var seed: UInt64

        init(sliding: Bool, shape: Shape, dtype: DType, lengths: [Int], extra: Int = 64)
            throws
        {
            let layer = CBv2LayerKind(
                attention: sliding ? .slidingWindow(PagedColumnsAsDecodeRowsTests.window) : .full,
                headDim: shape.headDim, kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)
            self.shape = shape
            self.dtype = dtype
            backend = try PagedKVBackend(
                layerKinds: [layer],
                config: .init(
                    capacityBytes: 512 << 20, dtype: dtype, maxPrefillChunk: 512,
                    nominalMaxSequenceLength: 8192, segmentSizeBytes: 4 << 20))
            rounds = backend.makeLayerCaches()[0]
            steps = backend.makeLayerCaches()[0]
            seed = UInt64(lengths.count * 7919 + (lengths.first ?? 0) + (sliding ? 1 : 0))
            for length in lengths {
                let a = try backend.makeSequenceState(
                    layerKinds: [layer], promptLength: length, maxLength: length + extra)
                let b = try backend.makeSequenceState(
                    layerKinds: [layer], promptLength: length, maxLength: length + extra)
                states += [a, b]
                let ar = try #require(a[0] as? PagedSequenceKV)
                let br = try #require(b[0] as? PagedSequenceKV)
                for start in stride(from: 0, to: length, by: 512) {
                    let n = min(512, length - start)
                    let k = random([shape.kvHeads, n, shape.headDim])
                    let v = random([shape.kvHeads, n, shape.headDim])
                    ar.write(keys: k, values: v)
                    br.write(keys: k, values: v)
                }
                roundRows.append(ar)
                stepRows.append(br)
            }
        }

        deinit {
            rounds.setRows([])
            steps.setRows([])
            for state in states { backend.release(state) }
        }

        func random(_ shape: [Int]) -> MLXArray {
            seed += 1
            return MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(dtype)
        }

        /// Queries `[B, QH, W, D]` and new K/V `[B, KVH, W, D]`.
        func inputs(_ width: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
            let b = roundRows.count
            let q = random([b, shape.queryHeads, width, shape.headDim])
            let k = random([b, shape.kvHeads, width, shape.headDim])
            let v = random([b, shape.kvHeads, width, shape.headDim])
            eval(q, k, v)
            return (q, k, v)
        }

        var scale: Float { 1 / Float(shape.headDim).squareRoot() }

        /// One rectangular call over every round row, as an MTP round makes it.
        func round(
            _ x: (q: MLXArray, k: MLXArray, v: MLXArray), columnsAsRows: Bool = true
        ) -> (output: MLXArray, calls: Int) {
            rounds.setRows(roundRows)
            rounds.mtpSerializesRectangularAttention = true
            rounds.mtpBatchesRectangularColumnsAsRows = columnsAsRows
            defer {
                rounds.mtpSerializesRectangularAttention = false
                rounds.mtpBatchesRectangularColumnsAsRows = false
            }
            let before = rounds.mtpColumnsAsRowsCalls
            let output = rounds.updateAndAttend(
                queries: x.q, keys: x.k, values: x.v, scale: scale, sinks: nil)
            eval(output)
            return (output, rounds.mtpColumnsAsRowsCalls - before)
        }

        /// The first `columns[i]` columns of request `i`, each a one-row step.
        func stepped(_ x: (q: MLXArray, k: MLXArray, v: MLXArray), columns: [Int]) -> [MLXArray] {
            zip(stepRows, columns).enumerated().map { i, pair in
                steps.setRows([pair.0])
                let pieces = (0 ..< pair.1).map { t in
                    steps.updateAndAttend(
                        queries: x.q[i ..< i + 1, 0..., t ..< t + 1, 0...],
                        keys: x.k[i ..< i + 1, 0..., t ..< t + 1, 0...],
                        values: x.v[i ..< i + 1, 0..., t ..< t + 1, 0...],
                        scale: scale, sinks: nil)
                }
                let row = concatenated(pieces, axis: 2)
                eval(row)
                return row
            }
        }

        /// Both rows of every request at the same position with the same
        /// attendable K/V bytes.
        var sameKV: Bool {
            zip(roundRows, stepRows).allSatisfy { a, b in
                let (x, y) = (a.snapshot(), b.snapshot())
                eval(x.keys, x.values, y.keys, y.values)
                return x.offset == y.offset
                    && PagedColumnsAsDecodeRowsTests.same(x.keys, y.keys)
                    && PagedColumnsAsDecodeRowsTests.same(x.values, y.values)
            }
        }

        var faulted: Bool { backend.pool.writeValidation.isFaulted }
    }

    static func bytes(_ a: MLXArray) -> Data { a.asData(access: .copy).data }

    static func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.dtype == b.dtype && a.shape == b.shape && bytes(a) == bytes(b)
    }

    /// Rounds `widths` in order; each must equal the requests' steps.
    static func expectRoundsEqualSteps(
        _ f: Fixture, widths: [Int], columnsAsRows: Bool = true, _ label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        for width in widths {
            let x = f.inputs(width)
            let (output, calls) = f.round(x, columnsAsRows: columnsAsRows)
            let expected = f.stepped(x, columns: Array(repeating: width, count: f.roundRows.count))
            #expect(output.dtype == f.dtype, sourceLocation: sourceLocation)
            #expect(
                !isNaN(output).any().item(Bool.self)
                    && abs(output).max().asType(.float32).item(Float.self) > 0,
                sourceLocation: sourceLocation)
            #expect(
                calls == (columnsAsRows && width > 1 ? 1 : 0), "\(label) width \(width)",
                sourceLocation: sourceLocation)
            #expect(
                same(output, concatenated(expected, axis: 0)), "\(label) width \(width)",
                sourceLocation: sourceLocation)
            #expect(f.sameKV, "\(label) width \(width) K/V", sourceLocation: sourceLocation)
            #expect(!f.faulted, sourceLocation: sourceLocation)
        }
    }

    /// Prompts that start before, at and past the 2,048 window edge, and one
    /// whose ring has wrapped; rounds of 1 to 8 columns follow each other.
    @Test(arguments: [false, true], [DType.bfloat16, .float32])
    func oneRequestRoundsEqualSteps(sliding: Bool, dtype: DType) throws {
        for shape in Self.shapes {
            for length in [15, 255, 511, 2030, 2045, 4100] {
                let f = try Fixture(sliding: sliding, shape: shape, dtype: dtype, lengths: [length])
                Self.expectRoundsEqualSteps(
                    f, widths: Array(1 ... 8),
                    "sliding \(sliding) \(dtype) d\(shape.headDim) prompt \(length)")
            }
        }
    }

    /// 2, 8 and 16 requests of mixed lengths share a round: up to 128 rows in
    /// one dispatch, each equal to its own request's steps.
    @Test(arguments: [false, true], [DType.bfloat16, .float32])
    func batchedRequestsOfMixedLengthsEqualTheirSteps(sliding: Bool, dtype: DType) throws {
        let lengths = [
            2045, 15, 2060, 255, 4100, 2047, 511, 2049, 3, 2040, 1000, 2063, 31, 2100, 64, 2048,
        ]
        for shape in Self.shapes {
            for requests in [2, 8, 16] {
                let f = try Fixture(
                    sliding: sliding, shape: shape, dtype: dtype,
                    lengths: Array(lengths.prefix(requests)))
                Self.expectRoundsEqualSteps(
                    f, widths: [8, 2, 5, 1, 8, 7],
                    "sliding \(sliding) \(dtype) d\(shape.headDim) requests \(requests)")
            }
        }
    }

    /// Rolled-back columns leave nothing a later round or step can see.
    @Test(arguments: [false, true], [DType.bfloat16, .float32])
    func rejectedColumnsLeaveNoTrace(sliding: Bool, dtype: DType) throws {
        let shape = Self.shapes[0]
        let f = try Fixture(
            sliding: sliding, shape: shape, dtype: dtype, lengths: [2045, 2058, 15], extra: 96)
        for keep in [[1, 8, 3], [8, 1, 5], [2, 2, 8], [1, 1, 1]] {
            let x = f.inputs(8)
            for row in f.roundRows { row.beginSpeculativeWrite() }
            let (output, calls) = f.round(x)
            for (row, kept) in zip(f.roundRows, keep) {
                row.rollback(8 - kept)
                row.commitSpeculativeWrite()
            }
            let expected = f.stepped(x, columns: keep)
            #expect(calls == 1)
            for (i, kept) in keep.enumerated() {
                #expect(
                    Self.same(output[i ..< i + 1, 0..., 0 ..< kept, 0...], expected[i]),
                    "keep \(keep) row \(i)")
            }
            #expect(f.sameKV, "keep \(keep)")
            Self.expectRoundsEqualSteps(f, widths: [8], "after keep \(keep)")
        }
        #expect(!f.faulted)
    }

    // MARK: Default off

    /// Off, a round takes the column loop: one dispatch per column, the
    /// bytes it always had. A round wider than the speculative span falls
    /// back with the flag on.
    @Test(arguments: [false, true])
    func offByDefaultAndIneligibleRoundsTakeTheColumnLoop(sliding: Bool) throws {
        let shape = Self.shapes[0]
        let fresh = try Fixture(sliding: sliding, shape: shape, dtype: .bfloat16, lengths: [10])
        #expect(!fresh.rounds.mtpBatchesRectangularColumnsAsRows)
        let f = try Fixture(
            sliding: sliding, shape: shape, dtype: .bfloat16, lengths: [2045, 300])
        Self.expectRoundsEqualSteps(f, widths: [8, 3], columnsAsRows: false, "off")
        let wide = f.inputs(CBv2PagedSpeculation.maxSpeculativeSpan + 1)
        let (output, calls) = f.round(wide)
        #expect(calls == 0)
        let expected = f.stepped(wide, columns: [wide.q.dim(2), wide.q.dim(2)])
        #expect(Self.same(output, concatenated(expected, axis: 0)))
        #expect(f.sameKV && !f.faulted)
    }

    /// The bank sets the flag only when asked, and clears it after the body.
    @Test func theBankSetsTheFlagOnlyWhenAsked() throws {
        let f = try Fixture(sliding: true, shape: Self.shapes[1], dtype: .bfloat16, lengths: [4])
        let bank = CBv2LayerCacheBank(caches: [f.rounds])
        bank.withMTPRectangularVerification {
            #expect(f.rounds.mtpSerializesRectangularAttention)
            #expect(!f.rounds.mtpBatchesRectangularColumnsAsRows)
        }
        bank.withMTPRectangularVerification(columnsAsRows: true) {
            #expect(f.rounds.mtpBatchesRectangularColumnsAsRows)
        }
        #expect(!f.rounds.mtpSerializesRectangularAttention)
        #expect(!f.rounds.mtpBatchesRectangularColumnsAsRows)
    }
}
