// Exact paged prefill regression: the reference reconstructs every absolute
// Boolean mask and preserves the original ordered, array-masked SDPA calls.
// Flint geometry follows ThirdPartyNotices/Flint.md: Prince Canuma/mlx-vlm
// (MIT) and Recursive rec (Apache-2.0). No checkpoint or downloaded data.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 paged sliding mask reuse", .serialized)
struct CBv2PagedSlidingMaskReuseTests {
    private let window = 513
    private let blockSize = 128
    private let headDim = 128
    private let queryHeads = 4
    private let kvHeads = 1
    private var scale: Float { 1 / Float(headDim).squareRoot() }
    private var kind: CBv2LayerKind {
        .init(
            attention: .slidingWindow(window), hasSinks: true,
            headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func makeBackend(_ dtype: DType) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 64 << 20, dtype: dtype, maxPrefillChunk: 2048,
                nominalMaxSequenceLength: 8192, segmentSizeBytes: 1 << 20,
                layerDTypes: [dtype]))
    }

    private func addRow(
        _ backend: PagedKVBackend, states: inout [[CBv2SequenceKV?]]
    ) throws -> PagedSequenceKV {
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 8192, maxLength: 8192)
        states.append(state)  // The caller's defer owns cleanup even if the cast fails.
        return try #require(state[0] as? PagedSequenceKV)
    }

    // Host fixtures avoid global random state shared with other GPU suites.
    private func tensor(_ shape: [Int], dtype: DType, seed: UInt64) -> MLXArray {
        var state = seed
        let values: [Float] = (0 ..< shape.reduce(1, *)).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1
            return Float(Int((state >> 32) & 0xffff) - 32_768) / 32_768
        }
        return MLXArray(values).reshaped(shape).asType(dtype)
    }

    private func inputs(count: Int, offset: Int, tag: Int, dtype: DType)
        -> (q: MLXArray, k: MLXArray, v: MLXArray)
    {
        let seed = UInt64(offset + count + tag * 10_007)
        let q = tensor([1, queryHeads, count, headDim], dtype: dtype, seed: seed + 31)
        let k = tensor([1, kvHeads, count, headDim], dtype: dtype, seed: seed + 47)
        let v = tensor([1, kvHeads, count, headDim], dtype: dtype, seed: seed + 59)
        eval(q, k, v)
        return (q, k, v)
    }

    private func prime(_ rows: [PagedSequenceKV], count: Int, tag: Int, dtype: DType) {
        guard count > 0 else { return }
        let k = tensor([kvHeads, count, headDim], dtype: dtype, seed: UInt64(17 + tag))
        let v = tensor([kvHeads, count, headDim], dtype: dtype, seed: UInt64(29 + tag))
        eval(k, v)
        for row in rows {
            row.write(keys: k, values: v)
            let snapshot = row.snapshot()
            eval(snapshot.keys, snapshot.values)
        }
    }

    private func expectExact(_ actual: MLXArray, _ expected: MLXArray, _ context: String) {
        let note = Comment(rawValue: context)
        #expect(actual.shape == expected.shape, note)
        #expect(actual.dtype == expected.dtype, note)
        // Native bytes detect BF16 changes and signed zero without widening.
        // Do not include large fixture payloads in failure diagnostics.
        let bytesMatch = actual.asData(access: .copy).data == expected.asData(access: .copy).data
        #expect(bytesMatch, note)
    }

    private func expectSameCache(
        _ actual: PagedSequenceKV, _ expected: PagedSequenceKV, offset: Int
    ) {
        #expect(actual.absoluteOffset == offset && expected.absoluteOffset == offset)
        #expect(actual.writtenHighWater == offset && expected.writtenHighWater == offset)
        #expect(actual.baseOffset == expected.baseOffset)
        #expect(actual.retainedCount == min(window, offset))
        #expect(actual.retainedCount == expected.retainedCount)
        #expect(actual.ringPages == expected.ringPages)
        #expect(actual.reservedPages == expected.reservedPages)
        #expect(actual.table.count == expected.table.count)
        let a = actual.snapshot()
        let b = expected.snapshot()
        #expect(a.offset == b.offset)
        expectExact(a.keys, b.keys, "retained keys at \(offset)")
        expectExact(a.values, b.values, "retained values at \(offset)")
    }

    /// Same absolute positions, slices, mask expression, SDPA order and
    /// concatenation as the original paged loop; no mask cache or production
    /// mask helper. Even the one-query tail uses .array, never .none.
    private func rebuiltBlockReference(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        queryStart: Int, sinks: MLXArray
    ) -> MLXArray {
        let length = queries.dim(2)
        let history = keys.dim(2) - length
        let keyStart = queryStart - history
        let qpos = MLXArray(Int32(queryStart) ..< Int32(queryStart + length))
            .expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(keyStart) ..< Int32(keyStart + keys.dim(2)))
            .expandedDimensions(axis: 0)
        var outputs: [MLXArray] = []
        for offset in stride(from: 0, to: length, by: blockSize) {
            let count = min(blockSize, length - offset)
            let end = history + offset + count
            let start = max(0, history + offset + 1 - window)
            let qp = qpos[offset ..< (offset + count), 0...]
            let kp = kpos[0..., start ..< end]
            var mask = kp .<= qp
            mask = mask & (kp .> (qp - Int32(window)))
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., offset ..< (offset + count), 0...],
                    keys: keys[0..., 0..., start ..< end, 0...],
                    values: values[0..., 0..., start ..< end, 0...],
                    scale: scale, mask: .array(mask), sinks: sinks))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }

    private func runCase(dtype: DType, history: Int, chunks: [Int]) throws {
        let backend = try makeBackend(dtype)
        let reservedBefore = backend.bytesReserved
        let actualCache = backend.makeLayerCaches()[0]
        let referenceCache = backend.makeLayerCaches()[0]
        var states: [[CBv2SequenceKV?]] = []
        defer {
            eval(actualCache.innerState() + referenceCache.innerState())
            actualCache.setRows([])
            referenceCache.setRows([])
            for state in states { backend.release(state) }
            #expect(backend.bytesReserved == reservedBefore)
        }
        let actualRow = try addRow(backend, states: &states)
        let referenceRow = try addRow(backend, states: &states)
        prime([actualRow, referenceRow], count: history, tag: 0, dtype: dtype)
        actualCache.setRows([actualRow])
        referenceCache.setRows([referenceRow])
        let sinks = MLXArray([Float(0), 0.25, -0.5, 1]).asType(dtype)
        eval(sinks)
        var offset = history
        for count in chunks + [1] {
            let input = inputs(count: count, offset: offset, tag: 0, dtype: dtype)
            let expected: MLXArray
            if count == 1 {
                // Decode must compare against the same native PAGED kernel,
                // not against a numerically different SDPA decode oracle.
                referenceCache.setRows([referenceRow])
                expected = referenceCache.updateAndAttend(
                    queries: input.q, keys: input.k, values: input.v, scale: scale, sinks: sinks)
            } else {
                // PagedSequenceKV.update gathers history BEFORE writing the
                // ring. Native fixture dtype equals the pool dtype.
                let (k, v) = referenceRow.update(keys: input.k, values: input.v)
                expected = rebuiltBlockReference(
                    queries: input.q, keys: k, values: v, queryStart: offset, sinks: sinks)
                referenceCache.setRows([referenceRow])
            }
            eval([expected] + referenceCache.innerState())
            let actual = actualCache.updateAndAttend(
                queries: input.q, keys: input.k, values: input.v, scale: scale, sinks: sinks)
            eval([actual] + actualCache.innerState())
            // A write-validation fault returns queries; report it directly.
            #expect(!backend.pool.writeValidation.isFaulted)
            #expect(isFinite(actual).all().item(Bool.self))
            expectExact(actual, expected, "\(dtype), start=\(offset), count=\(count)")
            offset += count
            #expect(actualCache.positionOffsets.asArray(Int32.self) == [Int32(offset)])
            expectSameCache(actualRow, referenceRow, offset: offset)
        }
    }

    @Test(arguments: [false, true])
    func pagedOutputsAndKVMatchOrderedRebuild(useBF16: Bool) throws {
        try #require(CBv2AttentionV1.queryBlockSize == blockSize)
        let dtype: DType = useBF16 ? .bfloat16 : .float32
        for history in [0, 1543] {
            // Consecutive full chunks, a seven-query tail, a one-query tail
            // after saturation, and unblocked/window-boundary controls.
            // With history 1543, [641, 2048] gathers history 1672..2183,
            // across the 2048-token physical ring seam, during prefill.
            for chunks in [[2048, 2048], [1031], [641], [128, 513], [641, 2048]] {
                try runCase(dtype: dtype, history: history, chunks: chunks)
            }
        }
    }

    @Test(arguments: [false, true])
    func reorderedPagedRowsMatchIndependentRebuilds(useBF16: Bool) throws {
        try #require(CBv2AttentionV1.queryBlockSize == blockSize)
        let dtype: DType = useBF16 ? .bfloat16 : .float32
        let backend = try makeBackend(dtype)
        let reservedBefore = backend.bytesReserved
        let packedCache = backend.makeLayerCaches()[0]
        let referenceCache = backend.makeLayerCaches()[0]
        var states: [[CBv2SequenceKV?]] = []
        defer {
            eval(packedCache.innerState() + referenceCache.innerState())
            packedCache.setRows([])
            referenceCache.setRows([])
            for state in states { backend.release(state) }
            #expect(backend.bytesReserved == reservedBefore)
        }
        let histories = [0, 1543]
        var actualRows: [PagedSequenceKV] = []
        var referenceRows: [PagedSequenceKV] = []
        for (tag, history) in histories.enumerated() {
            let a = try addRow(backend, states: &states)
            let b = try addRow(backend, states: &states)
            prime([a, b], count: history, tag: tag, dtype: dtype)
            actualRows.append(a)
            referenceRows.append(b)
        }
        let sinks = MLXArray([Float(0), 0.25, -0.5, 1]).asType(dtype)
        eval(sinks)
        for (count, order) in [(1031, [0, 1]), (641, [1, 0])] {
            var queries: [MLXArray] = []
            var keys: [MLXArray] = []
            var values: [MLXArray] = []
            var expected: [MLXArray] = []
            for tag in order {
                let row = referenceRows[tag]
                let start = row.absoluteOffset
                let input = inputs(count: count, offset: start, tag: tag, dtype: dtype)
                queries.append(input.q)
                keys.append(input.k)
                values.append(input.v)
                let (k, v) = row.update(keys: input.k, values: input.v)
                let output = rebuiltBlockReference(
                    queries: input.q, keys: k, values: v, queryStart: start, sinks: sinks)
                referenceCache.setRows([row])
                eval([output] + referenceCache.innerState())
                expected.append(output)
            }
            packedCache.setRows(order.map { actualRows[$0] })
            let actual = packedCache.updateAndAttend(
                queries: concatenated(queries, axis: 0), keys: concatenated(keys, axis: 0),
                values: concatenated(values, axis: 0), scale: scale, sinks: sinks)
            eval([actual] + packedCache.innerState())
            #expect(isFinite(actual).all().item(Bool.self))
            #expect(
                packedCache.positionOffsets.asArray(Int32.self)
                    == order.map { Int32(actualRows[$0].absoluteOffset) })
            for (lane, tag) in order.enumerated() {
                expectExact(actual[lane ..< (lane + 1)], expected[lane], "packed row \(tag)")
                expectSameCache(
                    actualRows[tag], referenceRows[tag], offset: referenceRows[tag].absoluteOffset)
            }
        }
    }
}
