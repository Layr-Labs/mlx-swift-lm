import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Real QSA head geometry over the native paged backend, but no model weights.
/// Compare the attention seam independently of projections or draft quality.
final class Qwen4ExpQSACrossoverTests: XCTestCase {
    func testRectangularAttentionMatchesCanonicalColumnsAtCrossover() throws {
        let windows = [(2044, 4), (2045, 4), (2047, 2), (2048, 4), (2049, 5),
                       (2050, 2), (2050, 5), (2051, 4), (2080, 4)]
        let cases = [DType.float16, .bfloat16].flatMap { dtype in
            windows.map { ($0.0, $0.1, dtype) }
        }
        for (offset, width, dtype) in cases {
            let length = offset + width
            func seeded(_ shape: [Int], _ seed: UInt64) -> MLXArray {
                MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(dtype)
            }
            let queries = seeded([1, 24, width, 256], 451)
            let keys = seeded([1, 2, length, 256], 452)
            let values = seeded([1, 2, length, 256], 453)
            let indexKeys = seeded([1, length, 128], 454)
            let indexQueries = seeded([1, width, 4, 128], 455)
            let positions = MLXArray(0..<length).asType(.int32).reshaped([1, length])
            eval(queries, keys, values, indexKeys, indexQueries, positions)

            func run(rectangular: Bool) throws -> MLXArray {
                let kind = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 24,
                    qwen4IndexerCompressRatio: 4)
                let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
                    capacityBytes: 64 << 20, maxPrefillChunk: 4096,
                    nominalMaxSequenceLength: 4096, segmentSizeBytes: 4 << 20,
                    layerDTypes: [dtype]))
                let state = try backend.makeSequenceState(
                    layerKinds: [kind], promptLength: length, maxLength: 4096)
                defer { backend.release(state) }
                let row = try XCTUnwrap(state[0] as? PagedSequenceKV)
                row.write(keys: keys[0, 0..., 0..<offset, 0...], values: values[0, 0..., 0..<offset, 0...])
                let cache = backend.makeLayerCaches()[0]
                cache.setRows([row])
                let gathered: CBv2Qwen4GatheredCache = cache
                gathered.qwen4IndexKeys = indexKeys
                gathered.qwen4IndexPositionIds = positions
                func pooled(_ count: Int) -> MLXArray {
                    Qwen4ExpPooledIndex.reuseOrCompute(
                        cache: gathered, compressRatio: 4, logicalTokens: count,
                        indexKeyNorm: { $0 }, applyIndexRope: { x, _ in x })
                }
                let output: MLXArray
                if rectangular && length <= 2048 {
                    cache.mtpSerializesRectangularAttention = true
                    defer { cache.mtpSerializesRectangularAttention = false }
                    output = cache.updateAndAttend(
                        queries: queries, keys: keys[0..., 0..., offset..<length, 0...],
                        values: values[0..., 0..., offset..<length, 0...], scale: 1.0 / 16.0, sinks: nil)
                        .transposed(0, 2, 1, 3)
                } else if rectangular && Qwen4ExpGatheredQSA.serialCrossoverEligible(
                    offset: offset, length: width, compressRatio: 4, tokenBudget: 2048) {
                    XCTAssertFalse(cache.qwen4SerializesRectangularAttention)
                    cache.mtpSerializesRectangularAttention = true
                    defer { cache.mtpSerializesRectangularAttention = false }
                    XCTAssertTrue(cache.qwen4SerializesRectangularAttention)
                    output = Qwen4ExpGatheredQSA.attendCrossover(
                        cache: cache, queries: queries,
                        keys: keys[0..., 0..., offset..<length, 0...],
                        values: values[0..., 0..., offset..<length, 0...],
                        indexQueries: indexQueries, offset: offset,
                        queryHeads: 24, kvHeads: 2, headDim: 256, indexerHeadDim: 128,
                        compressRatio: 4, tokenBudget: 2048, scale: 1.0 / 16.0,
                        indexKeyNorm: { $0 }, applyIndexRope: { x, _ in x })
                } else if rectangular {
                    let views = gathered.updateKVAndAdvanceOffsets(
                        keys: keys[0..., 0..., offset..<length, 0...],
                        values: values[0..., 0..., offset..<length, 0...])
                    output = Qwen4ExpGatheredQSA.attend(
                        queries: queries, keys: views[0].keys, values: views[0].values,
                        indexQueries: indexQueries, indexKeys: indexKeys, indexPositionIds: positions,
                        queryHeads: 24, kvHeads: 2, headDim: 256, indexerHeadDim: 128,
                        compressRatio: 4, tokenBudget: 2048,
                        indexKeyNorm: { $0 }, applyIndexRope: { x, _ in x },
                        pooledIndexKeys: pooled(length))
                } else {
                    var columns: [MLXArray] = []
                    for column in 0..<width {
                        let q = queries[0..., 0..., column..<(column + 1), 0...]
                        let k = keys[0..., 0..., (offset + column)..<(offset + column + 1), 0...]
                        let v = values[0..., 0..., (offset + column)..<(offset + column + 1), 0...]
                        if Qwen4ExpGatheredQSA.decodeCrossesBudget(
                            offset: offset + column, compressRatio: 4, tokenBudget: 2048) {
                            let view = gathered.updateKVAndAdvanceOffsets(keys: k, values: v)[0]
                            columns.append(Qwen4ExpGatheredQSA.attendDecode(
                                queries: q, keys: view.keys, values: view.values,
                                indexQueries: indexQueries[0..., column..<(column + 1), 0..., 0...],
                                pooledIndexKeys: pooled(offset + column + 1),
                                queryHeads: 24, kvHeads: 2, headDim: 256, indexerHeadDim: 128,
                                compressRatio: 4, tokenBudget: 2048))
                        } else {
                            columns.append(cache.updateAndAttend(
                                queries: q, keys: k, values: v, scale: 1.0 / 16.0, sinks: nil)
                                .transposed(0, 2, 1, 3))
                        }
                    }
                    output = concatenated(columns, axis: 1)
                }
                eval(output)
                XCTAssertEqual(row.absoluteOffset, length)
                return output
            }
            let reference = try run(rectangular: false)
            let candidate = try run(rectangular: true)
            for column in 0..<width {
                let error = abs(reference[0, column].asType(.float32) - candidate[0, column].asType(.float32))
                    .max().item(Float.self)
                XCTAssertEqual(error, 0, "dtype=\(dtype) offset=\(offset) column=\(column): verify must use canonical attention")
            }
        }
    }

    func testCrossoverGeometryAndContiguousMarker() {
        for offset in [2047, 2048, 2049, 2050] {
            XCTAssertTrue(Qwen4ExpGatheredQSA.serialCrossoverEligible(
                offset: offset, length: 4, compressRatio: 4, tokenBudget: 2048))
        }
        for offset in [0, 2044, 2051, 10_000, Int.max] {
            XCTAssertFalse(Qwen4ExpGatheredQSA.serialCrossoverEligible(
                offset: offset, length: 4, compressRatio: 4, tokenBudget: 2048))
        }
        XCTAssertFalse(Qwen4ExpGatheredQSA.serialCrossoverEligible(
            offset: 2048, length: 1, compressRatio: 4, tokenBudget: 2048))
        XCTAssertFalse(Qwen4ExpGatheredQSA.serialCrossoverEligible(
            offset: 2048, length: 4, compressRatio: 0, tokenBudget: 2048))
        let cache = CBv2LayerCache(layerIndex: 0,
            kind: .init(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 24))
        XCTAssertFalse(cache.qwen4SerializesRectangularAttention)
        cache.mtpSerializesRectangularAttention = true
        XCTAssertTrue(cache.qwen4SerializesRectangularAttention)
        cache.mtpSerializesRectangularAttention = false
        XCTAssertFalse(cache.qwen4SerializesRectangularAttention)
    }
}
