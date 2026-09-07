import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("Packed attention explicit tensor strides", .serialized)
struct CBv2QuantizedStridedInputTests {
    enum Layout: CaseIterable, Sendable {
        case featureTwo, broadcastRows, broadcastFeatures, reversed
    }

    private func tensor(heads: Int, tokens: Int, dtype: DType,
                        layout: Layout, query: Bool, seed: UInt64) -> MLXArray {
        let shape = [2, heads, tokens, 64]
        func random(_ shape: [Int]) -> MLXArray {
            MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(dtype)
        }
        switch layout {
        case .featureTwo:
            return asStrided(random([2, heads, tokens, 128]), shape,
                             strides: [heads * tokens * 128, tokens * 128, 128, 2])
        case .broadcastRows:
            // Q varies across features; K/V vary across tokens and batch, so
            // reading the wrong broadcast head/row changes actual attention.
            return broadcast(random(query ? [1, 1, 1, 64] : [2, 1, tokens, 64]), to: shape)
        case .broadcastFeatures:
            return broadcast(random([2, heads, tokens, 1]), to: shape)
        case .reversed:
            return asStrided(random(shape), shape,
                             strides: [-heads * tokens * 64, -tokens * 64, -64, -1],
                             offset: 2 * heads * tokens * 64 - 1)
        }
    }

    @Test(arguments: [DType.float16, .bfloat16, .float32], Layout.allCases)
    func prefillAndBatchedDecodeRespectAllStrides(dtype: DType, layout: Layout) throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        func backend() throws -> PagedKVBackend {
            try PagedKVBackend(layerKinds: [kind], config: .init(
                capacityBytes: 8 << 20, dtype: dtype, maxPrefillChunk: 64,
                nominalMaxSequenceLength: 64, maxBufferLength: 8 << 20,
                segmentSizeBytes: 32768, quantization: .init()))
        }
        let actual = try backend(), reference = try backend()
        let actualStates = try (0 ..< 2).map { _ in
            try actual.makeSequenceState(layerKinds: [kind], promptLength: 33, maxLength: 64)
        }
        let referenceStates = try (0 ..< 2).map { _ in
            try reference.makeSequenceState(layerKinds: [kind], promptLength: 33, maxLength: 64)
        }
        defer {
            for state in actualStates { actual.release(state) }
            for state in referenceStates { reference.release(state) }
        }
        let cache = actual.makeLayerCaches()[0], baseline = reference.makeLayerCaches()[0]
        cache.setRows(actualStates.map { $0[0]! })
        baseline.setRows(referenceStates.map { $0[0]! })
        for tokens in [33, 1] {
            let q = tensor(heads: 4, tokens: tokens, dtype: dtype, layout: layout, query: true, seed: UInt64(810 + tokens))
            let k = tensor(heads: 2, tokens: tokens, dtype: dtype, layout: layout, query: false, seed: UInt64(820 + tokens))
            let v = tensor(heads: 2, tokens: tokens, dtype: dtype, layout: layout, query: false, seed: UInt64(830 + tokens))
            eval(q, k, v)
            let strides = q.asData(access: .noCopy).strides
            switch layout {
            case .featureTwo: #expect(strides[3] == 2)
            case .broadcastRows: #expect(strides[0] == 0 && strides[1] == 0 && (tokens == 1 || strides[2] == 0))
            case .broadcastFeatures: #expect(strides[3] == 0)
            case .reversed: #expect(strides[0] < 0 && strides[1] < 0 && strides[3] == -1)
            }
            func copy(_ value: MLXArray) -> MLXArray {
                MLXArray(value.asArray(Float.self), value.shape).asType(dtype)
            }
            try actual.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 64)
            let want = baseline.updateAndAttend(queries: copy(q), keys: copy(k), values: copy(v), scale: 0.125, sinks: nil)
            let got = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
            let leases = actual.pool.takePendingQuantizedScratch() + reference.pool.takePendingQuantizedScratch()
            eval([got, want] + leases.flatMap(\.evaluationTargets))
            Stream.gpu.synchronize()
            Stream.cpu.synchronize()
            #expect(got.asData(access: .copy).data == want.asData(access: .copy).data)
            for lease in leases { lease.finishAfterSynchronization() }
        }
        // The writers must also encode the same bytes for strided K/V.
        for (gotState, wantState) in zip(actualStates, referenceStates) {
            let got = gotState[0]!.snapshot(), want = wantState[0]!.snapshot()
            #expect(got.keys.asData(access: .copy).data == want.keys.asData(access: .copy).data)
            #expect(got.values.asData(access: .copy).data == want.values.asData(access: .copy).data)
        }
        Stream.gpu.synchronize()
        Stream.cpu.synchronize()
    }
}
