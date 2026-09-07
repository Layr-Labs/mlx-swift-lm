import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("Opportunistic fused packed prefill", .serialized)
struct CBv2QuantizedFusedPrefillTests {
    private func makeBackend(dim: Int, dtype: DType, mode: PagedQuantizedPrefillMode,
                             quantization: PagedKVQuantizationConfig = .init(),
                             productionGeometry: Bool = false) throws -> (PagedKVBackend, CBv2LayerKind) {
        let kh = productionGeometry && dim == 64 ? 8 : 2
        let qh = productionGeometry ? (dim == 64 ? 64 : 16) : 4
        let kind = CBv2LayerKind(attention: .full, hasSinks: true, headDim: dim, kvHeads: kh, queryHeads: qh)
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 16 << 20, dtype: dtype, maxPrefillChunk: 512,
            nominalMaxSequenceLength: 512, maxBufferLength: 16 << 20,
            segmentSizeBytes: 32768, quantization: quantization, quantizedPrefillMode: mode))
        return (backend, kind)
    }

    @Test(arguments: [DType.float16, .bfloat16, .float32], [64, 256])
    func coldAndWarmPrefillMatchDirectWithBalancedBlocks(dtype: DType, dim: Int) throws {
        let (backend, kind) = try makeBackend(dim: dim, dtype: dtype, mode: .opportunisticSDPA, productionGeometry: true)
        let (reference, _) = try makeBackend(dim: dim, dtype: dtype, mode: .direct, productionGeometry: true)
        let state = try backend.makeSequenceState(layerKinds: [kind], promptLength: 9, maxLength: 512)
        let refState = try reference.makeSequenceState(layerKinds: [kind], promptLength: 9, maxLength: 512)
        defer { backend.release(state); reference.release(refState) }
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 64 << 20, config: .init(watermarkFraction: 0))
        backend.pool.memoryAdmission = admission
        let cache = backend.makeLayerCaches()[0], baseline = reference.makeLayerCaches()[0]
        cache.setRows([state[0]!]); baseline.setRows([refState[0]!])
        let qh = kind.queryHeads, kh = kind.kvHeads
        let sinks = MLXArray((0 ..< qh).map { [Float(0.2), -0.1, 0.4, -0.3][$0 % 4] })
        try backend.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 512)
        var outputs: [MLXArray] = [], expected: [MLXArray] = []
        for (index, count) in [9, 129, 273].enumerated() {
            // Feature-strided Q and negative K/V strides exercise the native
            // input-to-FP32 arena transfer, not a hidden contiguous copy.
            let q = asStrided(MLXRandom.normal([1, qh, count, dim * 2], key: MLXRandom.key(UInt64(901 + index))).asType(dtype),
                              [1, qh, count, dim], strides: [qh * count * dim * 2, count * dim * 2, dim * 2, 2])
            func kv(_ seed: UInt64) -> MLXArray {
                asStrided(MLXRandom.normal([1, kh, count, dim], key: MLXRandom.key(seed)).asType(dtype),
                          [1, kh, count, dim], strides: [kh * count * dim, -count * dim, -dim, -1],
                          offset: kh * count * dim - 1)
            }
            let k = kv(UInt64(911 + index)), v = kv(UInt64(921 + index))
            expected.append(baseline.updateAndAttend(queries: q, keys: k, values: v, scale: 1 / Float(dim).squareRoot(), sinks: sinks))
            outputs.append(cache.updateAndAttend(queries: q, keys: k, values: v, scale: 1 / Float(dim).squareRoot(), sinks: sinks))
        }
        #expect(backend.pool.quantizedScratchScope?.fusedPrefillArenas.count == 1)
        let leases = backend.pool.takePendingQuantizedScratch()
        #expect(leases.count == 3)
        let before = backend.pool.quantizedPrefillStatistics
        #expect(before.fusedCallCount == 3 && before.directCallCount == 0)
        #expect(before.fusedQueryTokenCount == 411 && before.currentAdditionalWorkspaceBytes > 0)
        #expect(admission.bytesReserved == before.currentAdditionalWorkspaceBytes)
        eval(outputs + expected + leases.flatMap(\.evaluationTargets))
        Stream.gpu.synchronize(); Stream.cpu.synchronize()
        let tolerance: Double = dtype == .bfloat16 ? 0.02 : 0.003
        for (got, want) in zip(outputs, expected) {
            #expect(got.dtype == dtype)
            #expect(allClose(got.asType(.float32), want.asType(.float32), rtol: tolerance, atol: tolerance).item(Bool.self))
        }
        for lease in leases { lease.finishAfterSynchronization() }
        for lease in reference.pool.takePendingQuantizedScratch() { lease.finishAfterSynchronization() }
        #expect(admission.bytesReserved == 0)
        #expect(backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == 0)
        let actualKV = state[0]!.snapshot(), expectedKV = refState[0]!.snapshot()
        #expect(actualKV.keys.asData(access: .copy).data == expectedKV.keys.asData(access: .copy).data)
        #expect(actualKV.values.asData(access: .copy).data == expectedKV.values.asData(access: .copy).data)
        Stream.gpu.synchronize(); Stream.cpu.synchronize()
    }

    @Test func refusedFastCandidatePreservesDirectPrepaymentAndWritesExactlyOnce() throws {
        let (backend, kind) = try makeBackend(dim: 64, dtype: .bfloat16, mode: .opportunisticSDPA)
        let state = try backend.makeSequenceState(layerKinds: [kind], promptLength: 33, maxLength: 64)
        defer { backend.release(state) }
        var config = AdmissionV2.Config(watermarkFraction: 0,
                                       layerBytesPerToken: [try backend.pool.groupKey(forLayer: 0).bytesPerToken()])
        config.workspaceProjection = .quantizedPaged(layerKinds: [kind], config: backend.pool.config,
            maximumChunk: 512, overlapPolicy: .engineSerialPrefill, sharesStepArenas: true)
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 64 << 20, config: config)
        let needed = admission.allocatedBytes(forTokens: 64)
        admission.updateBytesCapacity(needed)
        try admission.reserve(id: .init(401), additionalTokens: 64)
        backend.pool.memoryAdmission = admission
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([state[0]!])
        try backend.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 64)
        let out = cache.updateAndAttend(queries: MLXArray.ones([1, 4, 33, 64], dtype: .bfloat16),
            keys: MLXArray.ones([1, 2, 33, 64], dtype: .bfloat16),
            values: 3 * MLXArray.ones([1, 2, 33, 64], dtype: .bfloat16), scale: 0.125, sinks: nil)
        #expect(backend.pool.quantizedScratchScope?.fusedPrefillArenas.isEmpty == true)
        #expect(backend.pool.quantizedPrefillStatistics.budgetFallbackCount == 1)
        #expect(backend.pool.quantizedPrefillStatistics.directCallCount == 1)
        #expect(backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == 0)
        #expect(!backend.pool.writeValidation.isFaulted)
        #expect(state[0]!.absoluteOffset == 33)
        let leases = backend.pool.takePendingQuantizedScratch()
        eval([out] + leases.flatMap(\.evaluationTargets))
        Stream.gpu.synchronize(); Stream.cpu.synchronize()
        #expect(allClose(out, MLXArray(Float(3))).item(Bool.self))
        for lease in leases { lease.finishAfterSynchronization() }
        #expect(admission.bytesReserved == needed)
        admission.releaseAll(id: .init(401))
        #expect(admission.bytesReserved == 0)
    }

    @Test func balancedBlockAndAllocationPreflightRejectUnsupportedGeometry() throws {
        for count in [9, 127, 128, 129, 136, 257, 273, 2048] {
            let plan = try PagedQuantizedFusedPrefillPlan(
                geometry: .init(kvHeads: 2, queryHeads: 16, headDim: 256), queryCount: count,
                maximumTokens: 32768, attendLength: 32768, pageSize: 16,
                outputElementBytes: 2, needsArena: true, maxBufferLength: 1 << 30)
            #expect(plan.blocks.allSatisfy { (9 ... 128).contains($0.count) })
            #expect(plan.blocks.first?.lowerBound == 0 && plan.blocks.last?.upperBound == count)
            #expect(plan.blocks.reduce(0) { $0 + $1.count } == count)
            #expect(plan.reservationBytes >= plan.arenaBytes + count * 16 * 256 * 6)
        }
        #expect(throws: CBv2KVError.self) {
            try PagedQuantizedFusedPrefillPlan(geometry: .init(kvHeads: 2, queryHeads: 16, headDim: 512),
                queryCount: 33, maximumTokens: 64, attendLength: 33, pageSize: 16,
                outputElementBytes: 2, needsArena: true, maxBufferLength: 1 << 30)
        }
    }
}
