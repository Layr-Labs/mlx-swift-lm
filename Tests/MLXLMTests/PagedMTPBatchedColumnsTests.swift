import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class PagedMTPBatchedColumnsTests: XCTestCase {
    func testEveryColumnMatchesSerialAcrossPageAndPartitionBoundaries() throws {
        guard PagedMTPBatchedColumns.enabled else {
            throw XCTSkip("Requires explicit experimental batched-attention flag")
        }
        for dtype: DType in [.bfloat16, .float32] {
            for prime in [15, 255, 511] {
                for columns in 2 ... 8 {
                    let layer = CBv2LayerKind(
                        attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
                    let backend = try PagedKVBackend(
                        layerKinds: [layer],
                        config: .init(
                            capacityBytes: 64 << 20, dtype: dtype, maxPrefillChunk: 512,
                            nominalMaxSequenceLength: 1024, segmentSizeBytes: 1 << 20))
                    let a = try backend.makeSequenceState(
                        layerKinds: [layer], promptLength: prime, maxLength: 1024)
                    let b = try backend.makeSequenceState(
                        layerKinds: [layer], promptLength: prime, maxLength: 1024)
                    defer {
                        backend.release(a)
                        backend.release(b)
                    }
                    let ar = try XCTUnwrap(a[0] as? PagedSequenceKV)
                    let br = try XCTUnwrap(b[0] as? PagedSequenceKV)
                    MLXRandom.seed(UInt64(35_003_518 + prime * 4 + columns))
                    let pk = MLXRandom.normal([2, prime, 64]).asType(dtype)
                    let pv = MLXRandom.normal([2, prime, 64]).asType(dtype)
                    ar.write(keys: pk, values: pv)
                    br.write(keys: pk, values: pv)
                    let q = MLXRandom.normal([1, 4, columns, 64]).asType(dtype)
                    let k = MLXRandom.normal([1, 2, columns, 64]).asType(dtype)
                    let v = MLXRandom.normal([1, 2, columns, 64]).asType(dtype)
                    let rectangular = backend.makeLayerCaches()[0]
                    rectangular.setRows([ar])
                    rectangular.mtpSerializesRectangularAttention = true
                    rectangular.mtpBatchesRectangularAttention = true
                    let actual = rectangular.updateAndAttend(
                        queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
                    XCTAssertEqual(rectangular.mtpBatchedAttentionCalls, 1)
                    let serial = backend.makeLayerCaches()[0]
                    serial.setRows([br])
                    let pieces = (0 ..< columns).map { t in
                        serial.updateAndAttend(
                            queries: q[0..., 0..., t ..< (t + 1), 0...],
                            keys: k[0..., 0..., t ..< (t + 1), 0...],
                            values: v[0..., 0..., t ..< (t + 1), 0...], scale: 0.125, sinks: nil)
                    }
                    let expected = concatenated(pieces, axis: 2)
                    eval(actual, expected)
                    XCTAssertEqual(actual.dtype, dtype)
                    XCTAssertEqual(
                        actual.asData(access: .copy).data, expected.asData(access: .copy).data,
                        "dtype=\(dtype) prime=\(prime) width=\(columns)")
                    XCTAssertEqual(ar.absoluteOffset, br.absoluteOffset)
                    XCTAssertEqual(ar.absoluteOffset, prime + columns)
                    let akv = ar.gatherRange(start: 0, count: prime + columns)
                    let bkv = br.gatherRange(start: 0, count: prime + columns)
                    eval(akv.0, akv.1, bkv.0, bkv.1)
                    XCTAssertEqual(
                        akv.0.asData(access: .copy).data, bkv.0.asData(access: .copy).data)
                    XCTAssertEqual(
                        akv.1.asData(access: .copy).data, bkv.1.asData(access: .copy).data)
                }
            }
        }
    }

    func testExtendedBlock16AndSlidingWindowBatchedColumnsMatchSerialByteExact() throws {
        guard PagedMTPBatchedColumns.enabled else {
            throw XCTSkip("Requires explicit experimental batched-attention flag")
        }
        // Covers both full-attention 16-column multi-query and sliding-window ring wraparound
        // at B=1 and B=4 (columns=16, ring headroom=16 for window=64, pageSize=16).
        let configs: [(CBv2LayerKind, Int, Int, Int)] = [
            (CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4), 1, 95, 16),
            (
                CBv2LayerKind(
                    attention: .slidingWindow(64), headDim: 64, kvHeads: 2, queryHeads: 4), 1, 70,
                16
            ),
            (
                CBv2LayerKind(
                    attention: .slidingWindow(64), headDim: 64, kvHeads: 2, queryHeads: 4), 4, 70, 8
            ),
            (CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4), 4, 48, 8),
        ]
        for (layer, batchSize, prime, columns) in configs {
            let backend = try PagedKVBackend(
                layerKinds: [layer],
                config: .init(
                    capacityBytes: 64 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
                    nominalMaxSequenceLength: 512, segmentSizeBytes: 1 << 20))
            let statesA = try (0 ..< batchSize).map { _ in
                try backend.makeSequenceState(
                    layerKinds: [layer], promptLength: prime, maxLength: 512)
            }
            let statesB = try (0 ..< batchSize).map { _ in
                try backend.makeSequenceState(
                    layerKinds: [layer], promptLength: prime, maxLength: 512)
            }
            defer {
                for s in statesA { backend.release(s) }
                for s in statesB { backend.release(s) }
            }
            let rowsA = try statesA.map { try XCTUnwrap($0[0] as? PagedSequenceKV) }
            let rowsB = try statesB.map { try XCTUnwrap($0[0] as? PagedSequenceKV) }
            MLXRandom.seed(UInt64(991100 + batchSize * 100 + columns))
            for b in 0 ..< batchSize {
                let pk = MLXRandom.normal([2, prime + b * 3, 64]).asType(.bfloat16)
                let pv = MLXRandom.normal([2, prime + b * 3, 64]).asType(.bfloat16)
                rowsA[b].write(keys: pk, values: pv)
                rowsB[b].write(keys: pk, values: pv)
            }
            let q = MLXRandom.normal([batchSize, 4, columns, 64]).asType(.bfloat16)
            let k = MLXRandom.normal([batchSize, 2, columns, 64]).asType(.bfloat16)
            let v = MLXRandom.normal([batchSize, 2, columns, 64]).asType(.bfloat16)
            let rectangular = backend.makeLayerCaches()[0]
            rectangular.setRows(rowsA)
            rectangular.mtpSerializesRectangularAttention = true
            rectangular.mtpBatchesRectangularAttention = true
            let actual = rectangular.updateAndAttend(
                queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
            XCTAssertEqual(rectangular.mtpBatchedAttentionCalls, 1)

            let serial = backend.makeLayerCaches()[0]
            serial.setRows(rowsB)
            let pieces = (0 ..< columns).map { t in
                serial.updateAndAttend(
                    queries: q[0..., 0..., t ..< (t + 1), 0...],
                    keys: k[0..., 0..., t ..< (t + 1), 0...],
                    values: v[0..., 0..., t ..< (t + 1), 0...], scale: 0.125, sinks: nil)
            }
            let expected = concatenated(pieces, axis: 2)
            eval(actual, expected)
            XCTAssertEqual(
                actual.asData(access: .copy).data, expected.asData(access: .copy).data,
                "layer=\(layer.attention) B=\(batchSize) cols=\(columns)")
        }
    }
}
