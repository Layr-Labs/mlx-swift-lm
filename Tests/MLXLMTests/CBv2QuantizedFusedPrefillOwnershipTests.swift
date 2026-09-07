import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("Fused packed prefill extra ownership", .serialized)
struct CBv2QuantizedFusedPrefillOwnershipTests {
    @Test func mixedNativeGroupsReuseArenaAndRejectedNewGeometryKeepsItIntact() throws {
        let kinds = [4, 4, 8].map { CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: $0) }
        let dtypes: [DType] = [.float16, .bfloat16, .float16]
        func backend(_ mode: PagedQuantizedPrefillMode) throws -> PagedKVBackend {
            try PagedKVBackend(layerKinds: kinds, config: .init(
                capacityBytes: 8 << 20, maxPrefillChunk: 64, nominalMaxSequenceLength: 64,
                maxBufferLength: 8 << 20, segmentSizeBytes: 32768, layerDTypes: dtypes,
                quantization: .init(), quantizedPrefillMode: mode))
        }
        let actual = try backend(.opportunisticSDPA), reference = try backend(.direct)
        let a = try actual.makeSequenceState(layerKinds: kinds, promptLength: 33, maxLength: 64)
        let b = try reference.makeSequenceState(layerKinds: kinds, promptLength: 33, maxLength: 64)
        defer { actual.release(a); reference.release(b) }
        var admissionConfig = AdmissionV2.Config(watermarkFraction: 0)
        admissionConfig.workspaceProjection = .init { _ in 1 << 20 }
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 64 << 20, config: admissionConfig)
        try admission.reserve(id: .init(601), additionalTokens: 64)
        actual.pool.memoryAdmission = admission
        let caches = actual.makeLayerCaches(), baselines = reference.makeLayerCaches()
        try actual.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 64)
        var outputs: [MLXArray] = [], expected: [MLXArray] = []
        var reservationDeltas: [Int] = []
        for index in kinds.indices {
            caches[index].setRows([a[index]!]); baselines[index].setRows([b[index]!])
            if index == 2 { admission.updateBytesCapacity(admission.bytesReserved) }
            let q = MLXRandom.normal([1, kinds[index].queryHeads, 33, 64], key: MLXRandom.key(UInt64(1011 + index))).asType(dtypes[index])
            let k = MLXRandom.normal([1, 2, 33, 64], key: MLXRandom.key(UInt64(1021 + index))).asType(dtypes[index])
            let v = MLXRandom.normal([1, 2, 33, 64], key: MLXRandom.key(UInt64(1031 + index))).asType(dtypes[index])
            expected.append(baselines[index].updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil))
            let before = actual.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes
            outputs.append(caches[index].updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil))
            reservationDeltas.append(actual.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes - before)
        }
        let firstPlan = try PagedQuantizedFusedPrefillPlan(
            geometry: .init(kvHeads: 2, queryHeads: 4, headDim: 64), queryCount: 33,
            maximumTokens: 64, attendLength: 33, pageSize: 16, outputElementBytes: 2,
            needsArena: true, maxBufferLength: 8 << 20)
        #expect(reservationDeltas[0] - reservationDeltas[1] == firstPlan.arenaBytes)
        #expect(reservationDeltas[2] == 0)
        #expect(actual.pool.groupKey(forLayer: 0) != actual.pool.groupKey(forLayer: 1))
        #expect(actual.pool.quantizedScratchScope?.fusedPrefillArenas.count == 1)
        let statistics = actual.pool.quantizedPrefillStatistics
        #expect(statistics.fusedCallCount == 2 && statistics.directCallCount == 1)
        #expect(statistics.budgetFallbackCount == 1 && !actual.pool.writeValidation.isFaulted)
        let leases = actual.pool.takePendingQuantizedScratch() + reference.pool.takePendingQuantizedScratch()
        eval(outputs + expected + leases.flatMap(\.evaluationTargets))
        Stream.gpu.synchronize(); Stream.cpu.synchronize()
        for (got, want) in zip(outputs, expected) {
            #expect(allClose(got.asType(.float32), want.asType(.float32), rtol: 0.02, atol: 0.02).item(Bool.self))
        }
        for lease in leases { lease.finishAfterSynchronization() }
        #expect(actual.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == 0)
        admission.releaseAll(id: .init(601))
        #expect(admission.bytesReserved == 0)
    }

    @Test func cancelledRequestAndReusedIDKeepDistinctFastStepPermits() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 8 << 20, dtype: .bfloat16, maxPrefillChunk: 64,
            nominalMaxSequenceLength: 64, maxBufferLength: 8 << 20,
            segmentSizeBytes: 32768, quantization: .init(), quantizedPrefillMode: .opportunisticSDPA))
        let oldState = try backend.makeSequenceState(layerKinds: [kind], promptLength: 33, maxLength: 64)
        let newState = try backend.makeSequenceState(layerKinds: [kind], promptLength: 33, maxLength: 64)
        defer { backend.release(oldState); backend.release(newState) }
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 64 << 20, config: .init(watermarkFraction: 0))
        backend.pool.memoryAdmission = admission
        let id = CBv2RequestID(701), cache = backend.makeLayerCaches()[0]
        let q = MLXRandom.normal([1, 4, 33, 64], key: MLXRandom.key(1201)).asType(.bfloat16)
        let k = MLXRandom.normal([1, 2, 33, 64], key: MLXRandom.key(1202)).asType(.bfloat16)
        let v = MLXRandom.normal([1, 2, 33, 64], key: MLXRandom.key(1203)).asType(.bfloat16)
        try admission.reserve(id: id, additionalTokens: 64)
        cache.setRows([oldState[0]!])
        try backend.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 64)
        let oldOutput = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
        let oldArrayID = ObjectIdentifier(try #require(backend.pool.quantizedScratchScope?.fusedPrefillArenas.values.first?.keysValues))
        let oldLeases = backend.pool.takePendingQuantizedScratch()
        let oldBytes = backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes
        admission.releaseAll(id: id)
        #expect(oldBytes > 0 && admission.bytesReserved == oldBytes)
        try admission.reserve(id: id, additionalTokens: 64)
        let newRequestBytes = admission.bytesReserved - oldBytes
        cache.setRows([newState[0]!])
        try backend.pool.beginQuantizedScratch(maximumQueries: 8, maximumAttendLength: 64)
        let newOutput = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
        let newArrayID = ObjectIdentifier(try #require(backend.pool.quantizedScratchScope?.fusedPrefillArenas.values.first?.keysValues))
        #expect(oldArrayID != newArrayID)
        let newLeases = backend.pool.takePendingQuantizedScratch()
        let newBytes = backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes - oldBytes
        eval([oldOutput, newOutput] + (oldLeases + newLeases).flatMap(\.evaluationTargets))
        Stream.gpu.synchronize(); Stream.cpu.synchronize()
        for lease in oldLeases { lease.finishAfterSynchronization() }
        #expect(newBytes > 0 && backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == newBytes)
        #expect(admission.bytesReserved == newRequestBytes + newBytes)
        #expect(oldOutput.asData(access: .copy).data == newOutput.asData(access: .copy).data)
        for lease in newLeases { lease.finishAfterSynchronization() }
        #expect(backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == 0)
        #expect(admission.bytesReserved == newRequestBytes)
        admission.releaseAll(id: id)
        #expect(admission.bytesReserved == 0)
    }

    @Test func singleBufferLimitRefusesBeforeAnyPermitMutation() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 16)
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 1 << 30, config: .init(watermarkFraction: 0))
        try admission.reserve(id: .init(801), additionalTokens: 64)
        let before = admission.bytesReserved
        #expect(throws: CBv2KVError.self) {
            try PagedQuantizedFusedPrefillPlan(geometry: .init(kvHeads: 2, queryHeads: 16, headDim: 256),
                queryCount: 128, maximumTokens: 32768, attendLength: 32768, pageSize: 16,
                outputElementBytes: 2, needsArena: true, maxBufferLength: 64 << 20)
        }
        #expect(admission.bytesReserved == before)
        admission.releaseAll(id: .init(801))
    }
}
