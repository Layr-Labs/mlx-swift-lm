import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Qwen4 packed selected-page copies", .serialized)
struct Qwen4QuantizedSelectedPageCopiesTests {
    private func fixture(
        dtype: DType = .bfloat16, dimension: Int = 64,
        heads: Int = 2, maximum: Int = 16_400,
        segmentBytes: Int = 32 << 10,
        format: PagedKVQuantizationConfig = .init()
    ) throws
        -> (backend: PagedKVBackend, row: PagedSequenceKV, kind: CBv2LayerKind)
    {
        let kind = CBv2LayerKind(
            attention: .full, headDim: dimension,
            kvHeads: heads, queryHeads: heads * 12)
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 64 << 20, dtype: dtype, maxPrefillChunk: 512,
                nominalMaxSequenceLength: maximum, maxBufferLength: 64 << 20,
                segmentSizeBytes: segmentBytes, quantization: format))
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0,
            maxLength: maximum)
        return (backend, try #require(state[0] as? PagedSequenceKV), kind)
    }

    private func tensor(
        start: Int, count: Int, heads: Int, dimension: Int,
        dtype: DType, tag: Float = 0
    ) -> MLXArray {
        let tokens = arange(start, start + count, dtype: .float32).reshaped([1, count, 1])
        let channels = arange(dimension, dtype: .float32).reshaped([1, 1, dimension])
        let head = arange(heads, dtype: .float32).reshaped([heads, 1, 1])
        return
            (sin(tokens * Float(0.071) + channels * Float(0.031) + head * Float(0.25) + tag)
            + cos(channels * Float(0.037) - tokens * Float(0.019)) * Float(0.23)).asType(dtype)
    }

    private func writeHistory(_ row: PagedSequenceKV, count: Int) throws {
        let key = row.groupKey
        for start in stride(from: 0, to: count, by: row.pool.config.maxPrefillChunk) {
            let size = min(row.pool.config.maxPrefillChunk, count - start)
            row.write(
                keys: tensor(
                    start: start, count: size, heads: key.kvHeads,
                    dimension: key.headDim, dtype: key.dtype),
                values: tensor(
                    start: start, count: size, heads: key.kvHeads,
                    dimension: key.valueHeadDim, dtype: key.dtype, tag: 0.8))
            try row.pool.writeValidation.check()
            eval([row.pool.group(key).writeFence] + row.nativeRecentEvaluationRoots)
            StreamOrDevice.default.stream.synchronize()
            try row.pool.finishQuantizedStorageStep()
        }
    }

    private func assertSelectedMatchesNativeBasis(_ row: PagedSequenceKV, indices: MLXArray) {
        let snapshot = row.snapshot()
        let actual = row.gatherSelected(indices)
        let clipped = minimum(
            maximum(indices, MLXArray(Int32(0))),
            MLXArray(Int32(row.absoluteOffset - 1)))
        let valid = ((indices .>= Int32(0)) .&& (indices .< Int32(row.absoluteOffset)))
            .reshaped([1, 1, indices.size, 1])
        let zero = MLXArray(Float(0)).asType(row.groupKey.dtype)
        let expectedK = MLX.where(valid, take(snapshot.keys, clipped, axis: 2), zero)
        let expectedV = MLX.where(valid, take(snapshot.values, clipped, axis: 2), zero)
        eval(actual.keys, actual.values, expectedK, expectedV)
        #expect(
            actual.keys.asData(access: .copy).data == expectedK.asData(access: .copy).data,
            "Selected old keys must decode in the native basis; recent/pending keys remain exact")
        #expect(actual.values.asData(access: .copy).data == expectedV.asData(access: .copy).data)
    }

    private func codedRecentBytes(_ row: PagedSequenceKV) -> [Data] {
        let size = row.pool.config.pageSize
        let pages = (384 / size ..< 512 / size).map { row.table[$0] }
        let coded = PagedQuantizedTransfers.gatherPacked(
            group: row.pool.group(row.groupKey),
            pages: pages, firstSlot: 0, count: 128)
        return [coded.keys, coded.values].map { $0.asData(access: .copy).data }
    }

    @Test(
        arguments: [DType.float16, .bfloat16, .float32],
        [
            PagedKVQuantizationConfig(),
            PagedKVQuantizationConfig(keyBits: 8, valueBits: 4),
            PagedKVQuantizationConfig(keyBits: 8, valueBits: 8),
        ])
    func codedHistoryAndOriginalRecentRowsCrossBindingGroups(
        dtype: DType, format: PagedKVQuantizationConfig
    ) throws {
        let f = try fixture(dtype: dtype, format: format)
        defer { f.backend.release([f.row]) }
        try writeHistory(f.row, count: 8_192)
        f.row.write(
            keys: tensor(start: 8_192, count: 6, heads: 2, dimension: 64, dtype: dtype),
            values: tensor(start: 8_192, count: 6, heads: 2, dimension: 64, dtype: dtype, tag: 0.8))
        try f.backend.pool.writeValidation.check()
        let group = f.backend.pool.group(f.row.groupKey)
        #expect(
            group.segments.count > PagedSelectedGather.maximumBindings,
            "Exercise actual old packed rows across multiple Metal binding groups")
        #expect(f.row.nativeRecentStart == 8_064)
        let selected: [Int32] = [
            0, 15, 16, 31, 32, 2_048, 4_096, 8_063, 8_064,
            8_191, 8_192, 8_197, 0, 8_197, -1, Int32.min, 8_198, Int32.max,
        ]
        // Preserve an on-device selector view's stride without forcing a copy.
        let interleaved = MLXArray(selected.flatMap { [$0, Int32(-777)] })
        let indices = interleaved.reshaped([selected.count, 2])[0..., 0]
        assertSelectedMatchesNativeBasis(f.row, indices: indices)
        try f.backend.pool.writeValidation.check()
        #expect(f.row.absoluteOffset == 8_198)
        try f.backend.pool.finishQuantizedStorageStep()
        #expect(f.backend.pool.pendingQuantizedScratch.isEmpty)
    }

    @Test func rejectedSuffixAndRewrittenPendingRowsNeverAlterConfirmedSelectedKV() throws {
        let f = try fixture(dtype: .float32, maximum: 1_024)
        defer { f.backend.release([f.row]) }
        try writeHistory(f.row, count: 512)
        let oldKeys = f.row.gatherSelected(MLXArray([Int32(0), 383, 400, 511])).keys
            .asData(access: .copy).data
        let oldCodes = codedRecentBytes(f.row)
        try f.backend.pool.finishQuantizedStorageStep()
        f.row.beginSpeculativeWrite()
        f.row.write(
            keys: tensor(start: 512, count: 6, heads: 2, dimension: 64, dtype: .float32),
            values: tensor(start: 512, count: 6, heads: 2, dimension: 64, dtype: .float32, tag: 0.8)
        )
        assertSelectedMatchesNativeBasis(
            f.row,
            indices: MLXArray([Int32(0), 383, 384, 511, 512, 514, 515, 517, 518, -1]))
        StreamOrDevice.default.stream.synchronize()
        f.row.rollback(3)
        f.row.commitSpeculativeWrite()
        assertSelectedMatchesNativeBasis(
            f.row,
            indices: MLXArray([Int32(0), 383, 384, 511, 512, 514, 515, 517, -1]))
        #expect(f.row.absoluteOffset == 515)
        #expect(
            f.row.gatherSelected(MLXArray([Int32(0), 383, 400, 511])).keys
                .asData(access: .copy).data == oldKeys)
        try f.backend.pool.finishQuantizedStorageStep()
        #expect(f.row.nativeRecentStart == 387)
        f.row.write(
            keys: tensor(
                start: 515, count: 3, heads: 2, dimension: 64,
                dtype: .float32, tag: 2.3),
            values: tensor(
                start: 515, count: 3, heads: 2, dimension: 64,
                dtype: .float32, tag: 3.1))
        assertSelectedMatchesNativeBasis(
            f.row,
            indices: MLXArray([Int32(0), 383, 384, 511, 512, 514, 515, 517, 518, -1]))
        #expect(
            f.row.gatherSelected(MLXArray([Int32(0), 383, 400, 511])).keys
                .asData(access: .copy).data == oldKeys)
        #expect(
            codedRecentBytes(f.row) == oldCodes,
            "Rejected and rewritten suffixes never requantize confirmed rows as the native band ages"
        )
        try f.backend.pool.finishQuantizedStorageStep()
        #expect(f.backend.pool.pendingQuantizedScratch.isEmpty)
        #expect(f.row.retiredNativeRecentOwners.isEmpty)
    }

    @Test(arguments: [1, 6])
    func compactQSAAboveTheBudgetMatchesTheExistingOrderedSteelReader(width: Int) throws {
        let count = 16_389
        let offset = count - width
        let f = try fixture(dimension: 256, segmentBytes: 256 << 10)
        let cache = try #require(f.backend.makeLayerCaches()[0] as? PagedLayerCache)
        defer {
            cache.setRows([])
            f.backend.release([f.row])
        }
        cache.setRows([f.row])
        cache.setRetainsChunkForBorrowers(false)
        try writeHistory(f.row, count: offset)
        #expect(
            Qwen4ExpCompactQSA.isSparsityWin(offset: offset, width: width),
            "The actual default compact reader is eligible beyond its QSA budget")
        let keys = tensor(
            start: offset, count: width, heads: 2, dimension: 256,
            dtype: .bfloat16
        ).expandedDimensions(axis: 0)
        let values = tensor(
            start: offset, count: width, heads: 2, dimension: 256,
            dtype: .bfloat16, tag: 0.8
        ).expandedDimensions(axis: 0)
        #expect(cache.qwen4CanGatherSelectedKV(keys: keys, values: values))
        let selected = (0 ..< width).flatMap { column -> [Int32] in
            let completeBlocks = (offset + column + 1) / 4
            return (0 ..< 512).map { Int32($0 * completeBlocks / 512) }
        }
        let blocks = MLXArray(selected, [1, width, 512])
        let indices = Qwen4ExpCompactQSA.tokenIndices(
            selected: blocks, offset: offset, keyTokens: count)
        let compact = cache.qwen4UpdateAndGatherSelectedKV(
            keys: keys, values: values,
            tokenIndices: indices)
        try f.backend.pool.writeValidation.check()
        let snapshot = f.row.snapshot()
        let queries = MLXRandom.normal(
            [1, 24, width, 256],
            key: MLXRandom.key(UInt64(321 + width))
        ).asType(.bfloat16)
        let reference = try #require(
            Qwen4ExpNativeSparseGQA.attend(
                queries: queries, keys: snapshot.keys, values: snapshot.values,
                selectedBlocks: blocks, qOffset: offset, parallelScores: false,
                parallelFullKV: false))
        let actual = try #require(
            Qwen4ExpNativeSparseGQA.attend(
                queries: queries, keys: compact.keys, values: compact.values,
                selectedBlocks: blocks, qOffset: offset, compactLogicalKeyTokens: count,
                parallelScores: true, parallelValuePartitions: 32, parallelFullKV: false))
        eval(reference, actual)
        #expect(actual.shape == reference.shape && actual.dtype == reference.dtype)
        #expect(
            actual.asData(access: .copy).data == reference.asData(access: .copy).data,
            "Packed selected addressing preserves Qwen4's existing ordered sparse attention arithmetic"
        )
        try f.backend.pool.finishQuantizedStorageStep()
        #expect(f.row.absoluteOffset == count)
    }

    @Test func selectedScratchRefusesBeforeSubmittingOrChangingTheReadFence() throws {
        let f = try fixture(dtype: .float32, heads: 1, maximum: 512)
        defer { f.backend.release([f.row]) }
        let admission = AdmissionV2(
            layerKinds: [f.kind], bytesCapacity: 8 << 20,
            config: .init(watermarkFraction: 0))
        f.backend.pool.memoryAdmission = admission
        try writeHistory(f.row, count: 256)
        let before = admission.transientBytesReserved
        let group = f.backend.pool.group(f.row.groupKey)
        let fence = ObjectIdentifier(group.writeFence)
        admission.updateBytesCapacity(before + 1_024)
        #expect(throws: CBv2KVError.self) {
            try PagedQuantizedSelectedGather.gather(
                row: f.row,
                indices: MLXArray((0 ..< 1_024).map { Int32($0 % 256) }))
        }
        #expect(admission.transientBytesReserved == before)
        #expect(f.backend.pool.pendingQuantizedScratch.isEmpty)
        #expect(ObjectIdentifier(group.writeFence) == fence)
        #expect(f.row.absoluteOffset == 256)
    }
}
