import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("Packed attention step arena ownership", .serialized)
struct CBv2QuantizedStepScratchTests {
    private func backend() throws -> (PagedKVBackend, [CBv2LayerKind]) {
        // Distinct KV groups share the same query/accumulator geometry.
        let kinds = [1, 2].map {
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: $0, queryHeads: 4)
        }
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 16 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
            nominalMaxSequenceLength: 512, maxBufferLength: 16 << 20,
            segmentSizeBytes: 32768, quantization: .init()))
        return (backend, kinds)
    }

    @Test func groupsAndSerialColumnsShareOneArenaWithoutChangingOutputs() throws {
        let (actual, kinds) = try backend(), (reference, _) = try backend()
        let a = try actual.makeSequenceState(layerKinds: kinds, promptLength: 273, maxLength: 512)
        let b = try reference.makeSequenceState(layerKinds: kinds, promptLength: 273, maxLength: 512)
        defer { actual.release(a); reference.release(b) }
        let caches = actual.makeLayerCaches(), referenceCaches = reference.makeLayerCaches()
        for index in kinds.indices {
            caches[index].setRows([a[index]!])
            referenceCaches[index].setRows([b[index]!])
        }
        try actual.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 512)
        var outputs: [MLXArray] = [], expected: [MLXArray] = []
        func forward(tokens: Int, seed: UInt64) {
            for index in kinds.indices {
                let q = MLXRandom.normal([1, tokens, 4, 64], key: MLXRandom.key(seed + UInt64(index)))
                    .asType(.bfloat16).transposed(0, 2, 1, 3)
                let k = MLXRandom.normal([1, tokens, kinds[index].kvHeads, 64], key: MLXRandom.key(seed + 10 + UInt64(index)))
                    .asType(.bfloat16).transposed(0, 2, 1, 3)
                let v = MLXRandom.normal([1, tokens, kinds[index].kvHeads, 64], key: MLXRandom.key(seed + 20 + UInt64(index)))
                    .asType(.bfloat16).transposed(0, 2, 1, 3)
                expected.append(referenceCaches[index].updateAndAttend(
                    queries: q, keys: k, values: v, scale: 0.125, sinks: nil))
                outputs.append(caches[index].updateAndAttend(
                    queries: q, keys: k, values: v, scale: 0.125, sinks: nil))
            }
        }
        // No evaluation between layers or columns: the dependency graph must
        // prevent a later layer/column from overwriting an earlier merge's input.
        forward(tokens: 273, seed: 301)
        for column in 0 ..< 3 { forward(tokens: 1, seed: 401 + UInt64(column * 30)) }
        let firstScope = try #require(actual.pool.quantizedScratchScope)
        #expect(firstScope.arenas.count == 1)
        let firstArena = try #require(firstScope.arenas.values.first)
        let first = actual.pool.takePendingQuantizedScratch()
        #expect(first.count == outputs.count + 1, "one arena owner plus one nonarena lease per call")
        #expect(actual.pool.quantizedScratchScope == nil)

        // An overlapping successor must own a fresh arena. Finalizing its
        // predecessor must never clear this scope or release its lease.
        try actual.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 512)
        forward(tokens: 1, seed: 701)
        let secondScope = try #require(actual.pool.quantizedScratchScope)
        let secondArena = try #require(secondScope.arenas.values.first)
        #expect(firstArena !== secondArena)
        #expect(firstArena.partials !== secondArena.partials)
        let referenceLeases = reference.pool.takePendingQuantizedScratch()
        let second = actual.pool.takePendingQuantizedScratch()
        #expect(first.reduce(0) { $0 + $1.reservedBytes } + second.reduce(0) { $0 + $1.reservedBytes }
            < referenceLeases.reduce(0) { $0 + $1.reservedBytes })
        eval(outputs + expected + (first + second + referenceLeases).flatMap(\.evaluationTargets))
        Stream.gpu.synchronize()
        Stream.cpu.synchronize()
        for lease in first { lease.finishAfterSynchronization() }
        #expect(!secondArena.lease.evaluationTargets.isEmpty)
        for (got, want) in zip(outputs, expected) {
            #expect(got.dtype == .bfloat16)
            #expect(got.asData(access: .copy).data == want.asData(access: .copy).data)
        }
        for lease in second + referenceLeases { lease.finishAfterSynchronization() }
    }

    @Test func scopeCannotGrowOrBeReplacedBeforeDetach() throws {
        let (backend, kinds) = try backend()
        let states = try backend.makeSequenceState(layerKinds: kinds, promptLength: 273, maxLength: 512)
        defer { backend.release(states) }
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([states[0]!])
        try backend.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 256)
        #expect(throws: CBv2KVError.self) {
            try backend.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 512)
        }
        _ = cache.updateAndAttend(
            queries: MLXArray.ones([1, 4, 273, 64], dtype: .bfloat16),
            keys: MLXArray.ones([1, 1, 273, 64], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 273, 64], dtype: .bfloat16), scale: 0.125, sinks: nil)
        #expect(backend.pool.writeValidation.isFaulted)
        #expect(backend.pool.quantizedScratchScope?.arenas.isEmpty == true)
        #expect(backend.pool.takePendingQuantizedScratch().isEmpty)
        #expect(backend.pool.quantizedScratchScope == nil)
    }
}
