import MLX
import Testing

@testable import MLXLMCommon

@Suite("Qwen4 selected-page copies", .serialized)
struct Qwen4SelectedPageCopiesTests {
    @Test(arguments: [DType.bfloat16, .float32])
    func selectedRowsAcrossBindingGroupsMatchDenseReference(dtype: DType) throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 2 << 20, dtype: dtype, maxPrefillChunk: 640,
            nominalMaxSequenceLength: 672, segmentSizeBytes: 2 * 16 * 64 * dtype.size * 3))
        let states = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 672)
        defer { backend.release(states) }
        let row = try #require(states[0] as? PagedSequenceKV)
        let keys = (arange(640 * 64, dtype: .float32) * Float(0.125))
            .reshaped([1, 640, 64]).asType(dtype)
        let values = (-keys - Float(4)).asType(dtype)
        eval(keys, values)
        row.write(keys: keys, values: values)
        eval(backend.pool.group(row.groupKey).writeFence)
        let plan = PagedSelectedGather.prepare(group: backend.pool.group(row.groupKey), pages: row.table)
        #expect(plan.segmentIDs.count > PagedSelectedGather.maximumBindings,
                "The test must cross an actual argument-binding group")
        let indices = MLXArray([Int32(0), 16, 31, 32, 300, 639, 0, 639, -1, 640, 645])
        for length in [640, 635] {
            if length == 635 { row.rollback(5) }
            let actual = row.gatherSelected(indices)
            let clipped = minimum(maximum(indices, MLXArray(Int32(0))), MLXArray(Int32(length - 1)))
            let valid = ((indices .>= Int32(0)) .&& (indices .< Int32(length)))
                .reshaped([1, 1, indices.size, 1])
            let expectedK = MLX.where(valid, take(keys, clipped, axis: 1).expandedDimensions(axis: 0),
                                     MLXArray(Float(0)).asType(dtype))
            let expectedV = MLX.where(valid, take(values, clipped, axis: 1).expandedDimensions(axis: 0),
                                     MLXArray(Float(0)).asType(dtype))
            eval(actual.keys, actual.values, expectedK, expectedV)
            #expect(actual.keys.asData(access: .copy).data == expectedK.asData(access: .copy).data)
            #expect(actual.values.asData(access: .copy).data == expectedV.asData(access: .copy).data)
        }
    }

    @Test func sixColumnVerifyIsEligibleButUnsupportedWidthsNeverAdvance() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 1 << 20, dtype: .bfloat16, maxPrefillChunk: 16,
            nominalMaxSequenceLength: 64, segmentSizeBytes: 32_768))
        let states = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let cache = try #require(backend.makeLayerCaches()[0] as? PagedLayerCache)
        defer { cache.setRows([]); backend.release(states) }
        let row = try #require(states[0] as? PagedSequenceKV)
        cache.setRows([row])
        for width in [1, 5, 6, 7] {
            let keys = MLXArray.zeros([1, 1, width, 64], dtype: .bfloat16)
            #expect(cache.qwen4CanGatherSelectedKV(keys: keys, values: keys) == (width <= 6))
            #expect(row.absoluteOffset == 0, "Eligibility must not mutate the row")
        }
    }
}
