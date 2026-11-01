import MLX
import MLXFast
import Testing

@testable import MLXLMCommon

@Suite("Packed attention with distinct native K/V layouts", .serialized)
struct CBv2QuantizedColdWindowLayoutTests {
    private let heads = 8
    private let dimension = 256
    private let tokens = 17

    private func inputs(dtype: DType) -> (
        keys: MLXArray, values: MLXArray, sequenceValues: MLXArray
    ) {
        // Post-RoPE K is contiguous in head/sequence order. Gemma V stays
        // transposed from sequence/head order after its independent norm.
        let keys = MLXArray(
            (0 ..< heads * tokens * dimension).map { Float($0 % 19) * 0.013 },
            [1, heads, tokens, dimension]
        ).asType(dtype)
        let sequenceValues = MLXArray(
            (0 ..< tokens * heads * dimension).map { index -> Float in
                let token = index / (heads * dimension)
                let head = (index / dimension) % heads
                let tokenPart = Float(token) * 0.019
                let headPart = Float(head) * 0.071
                let channelPart = Float(index % dimension % 7) * 0.009
                return 0.05 + tokenPart + headPart + channelPart
            }, [1, tokens, heads, dimension]
        ).asType(dtype)
        return (keys, sequenceValues.transposed(0, 2, 1, 3), sequenceValues)
    }

    private func fixture(dtype: DType, sharing: Bool) throws
        -> (
            backend: PagedKVBackend, row: PagedSequenceKV, caches: [PagedLayerCache],
            admission: AdmissionV2
        )
    {
        let kind = CBv2LayerKind(
            attention: .slidingWindow(1_024), headDim: dimension,
            kvHeads: heads, queryHeads: heads * 2)
        var kinds = [kind]
        if sharing {
            kinds.append(
                .init(
                    attention: kind.attention, sharesKVWithLayer: 0,
                    headDim: dimension, kvHeads: heads, queryHeads: heads * 2))
        }
        let config = PagedKVPoolConfig(
            capacityBytes: 128 << 20, dtype: dtype,
            maxPrefillChunk: 64, nominalMaxSequenceLength: 1_024,
            segmentSizeBytes: 256 << 10, quantization: .init())
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: config.capacityBytes,
            config: try backend.pool.admissionStorageConfig(
                .init(watermarkFraction: 0, elementBytes: dtype.size)),
            residency: backend.kvResidency)
        backend.pool.bindAdmission(admission)
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let state = try backend.makeSequenceState(layerKinds: kinds, promptLength: 0, maxLength: 64)
        let row = try #require(state[0] as? PagedSequenceKV)
        let caches = backend.makeLayerCaches().map { $0 as! PagedLayerCache }
        caches[0].setRows([row])
        return (backend, row, caches, admission)
    }

    private func expected(sequenceValues: MLXArray, causal: Bool, masked: Bool = false) -> [Float] {
        let values = sequenceValues.asType(.float32).asArray(Float.self)
        var result = [Float](repeating: 0, count: heads * 2 * tokens * dimension)
        for head in 0 ..< heads * 2 {
            for query in 0 ..< tokens {
                let visible = (0 ..< (causal ? query + 1 : tokens)).filter {
                    !masked || ($0 + query) % 3 != 1
                }
                for channel in 0 ..< dimension {
                    let sum = visible.reduce(Float(0)) { total, token in
                        total + values[(token * heads + head / 2) * dimension + channel]
                    }
                    result[(head * tokens + query) * dimension + channel] =
                        sum / Float(visible.count)
                }
            }
        }
        return result
    }

    private func assertMatches(_ output: MLXArray, expected: [Float], queryDType: DType) {
        let actual = output.asType(.float32).asArray(Float.self)
        let tolerance: Float =
            queryDType == .bfloat16 ? 0.003 : (queryDType == .float16 ? 0.0005 : 0.00001)
        #expect(output.dtype == queryDType)
        #expect(actual.count == expected.count)
        #expect(
            zip(actual, expected).map { abs($0.0 - $0.1) }.max()! < tolerance,
            "Each query head must average its own original V rows, not K's physical layout")
    }

    @Test(arguments: [DType.float16, .bfloat16, .float32], [DType.float16, .bfloat16, .float32])
    func coldWindowProducerAndBorrowerUseIndependentEvaluatedStrides(
        nativeDType: DType, queryDType: DType
    ) throws {
        let f = try fixture(dtype: nativeDType, sharing: true)
        defer {
            for cache in f.caches { cache.setRows([]) }
            f.backend.release([f.row])
            f.admission.releaseAll(id: .init(1))
        }
        let input = inputs(dtype: nativeDType)
        eval(input.keys, input.values)
        #expect(
            input.keys.strides != input.values.strides,
            "The fixture must reproduce contiguous RoPE keys and transposed values")
        let queries = MLXArray.zeros([1, heads * 2, tokens, dimension], dtype: queryDType)
        let owner = f.caches[0].updateAndAttend(
            queries: queries, keys: input.keys,
            values: input.values, scale: 1 / Float(dimension).squareRoot(), sinks: nil)
        let borrower = f.caches[1].attendBorrowing(
            source: f.caches[0], queries: queries,
            scale: 1 / Float(dimension).squareRoot(), sinks: nil)
        eval([owner, borrower] + f.caches.flatMap { $0.innerState() })
        let reference = expected(sequenceValues: input.sequenceValues, causal: true)
        assertMatches(owner, expected: reference, queryDType: queryDType)
        assertMatches(borrower, expected: reference, queryDType: queryDType)
        #expect(f.row.absoluteOffset == tokens)
        #expect(f.row.nativeRecentStart == 0)
        try f.backend.pool.writeValidation.check()
        StreamOrDevice.default.stream.synchronize()
        try f.backend.pool.finishQuantizedStorageStep()
        #expect(f.backend.pool.pendingQuantizedScratch.isEmpty)
    }

    @Test(arguments: [DType.float16, .bfloat16, .float32], [DType.float16, .bfloat16, .float32])
    func coldReadOnlyCanvasUsesValueStridesWithoutAppending(
        nativeDType: DType, queryDType: DType
    ) throws {
        let f = try fixture(dtype: nativeDType, sharing: false)
        defer {
            f.caches[0].setRows([])
            f.backend.release([f.row])
            f.admission.releaseAll(id: .init(1))
        }
        let input = inputs(dtype: nativeDType)
        let queries = MLXArray.zeros([1, heads * 2, tokens, dimension], dtype: queryDType)
        for masked in [false, true] {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                masked
                ? .array(
                    MLXArray(
                        (0 ..< tokens * tokens).map {
                            (($0 / tokens) + ($0 % tokens)) % 3 != 1
                        }, [1, 1, tokens, tokens])) : .none
            let candidate = try f.row.attendQuantizedReadOnly(
                queries: queries,
                currentKeys: input.keys, currentValues: input.values,
                scale: 1 / Float(dimension).squareRoot(), mask: mask, prefixCount: 0)
            let output = try #require(candidate)
            eval(output)
            assertMatches(
                output,
                expected: expected(
                    sequenceValues: input.sequenceValues,
                    causal: false, masked: masked), queryDType: queryDType)
            #expect(
                f.row.absoluteOffset == 0 && f.row.table.isEmpty,
                "An ephemeral canvas must not become committed KV")
            StreamOrDevice.default.stream.synchronize()
            try f.backend.pool.finishQuantizedStorageStep()
        }
        #expect(f.backend.pool.pendingQuantizedScratch.isEmpty)
    }
}
