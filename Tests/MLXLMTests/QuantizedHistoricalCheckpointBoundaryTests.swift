import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Quantized historical checkpoint boundaries", .serialized)
struct QuantizedHistoricalCheckpointBoundaryTests {
    private let donorID = CBv2RequestID(9001)

    private struct Fixture {
        let backend: PagedKVBackend
        let admission: AdmissionV2
        let codec: CBv2CompleteCheckpointCodec
        let capture: CBv2CompleteCheckpointCapture
        let kinds: [CBv2LayerKind]
        let request: CBv2Request
    }

    private func fixture(
        dtype: DType = .float16, quantization: PagedKVQuantizationConfig? = .init(),
        nativeLayers: Set<Int> = [], windowOnly: Bool = false
    ) throws -> Fixture {
        let first: CBv2LayerKind.Attention = windowOnly ? .slidingWindow(64) : .full
        let kinds = [
            CBv2LayerKind(attention: first, headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: .slidingWindow(windowOnly ? 64 : 256),
                headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: first, sharesKVWithLayer: 0,
                headDim: 64, kvHeads: 2, queryHeads: 4),
        ]
        let config = PagedKVPoolConfig(
            capacityBytes: 256 << 20, maxPrefillChunk: 512,
            segmentSizeBytes: 64 << 10, layerDTypes: Array(repeating: dtype, count: kinds.count),
            quantization: quantization, nativeLayerIndices: nativeLayers)
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: config.capacityBytes,
            config: try backend.pool.admissionStorageConfig(
                .init(watermarkFraction: 0, elementBytes: dtype.size)),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        let store = CompleteCheckpointFixtureStore()
        let codec = CBv2CompleteCheckpointCodec(
            identity: store.identity, layerKinds: kinds, recurrentSpec: nil,
            kvDTypes: Array(repeating: dtype, count: kinds.count), assistant: nil,
            admission: admission, pagedConfig: config)
        let capture = CBv2CompleteCheckpointCapture(codec: codec, store: store)
        capture.historicalCheckpointStrideTokens = 128
        return .init(
            backend: backend, admission: admission, codec: codec, capture: capture, kinds: kinds,
            request: .init(
                id: .init(1), promptTokens: Array(repeating: 1, count: 769),
                maxTokens: 256, cacheSalt: "tenant"))
    }

    private func donor(_ fixture: Fixture) throws -> [CBv2SequenceKV?] {
        try fixture.admission.reserve(id: donorID, additionalTokens: 1025)
        return try fixture.backend.makeSequenceState(
            layerKinds: fixture.kinds, promptLength: 769, maxLength: 1025)
    }

    private func write(
        _ state: [CBv2SequenceKV?], fixture: Fixture, start: Int, count: Int,
        layer: Int? = nil
    ) throws {
        for (index, entry) in state.enumerated() where layer == nil || layer == index {
            guard let row = entry as? PagedSequenceKV else { continue }
            let data = (0 ..< 2 * count * 64).map {
                Float(sin(Double($0 + start * 64 + index * 31) * 0.017))
            }
            let keys = MLXArray(data, [2, count, 64]).asType(row.groupKey.dtype)
            row.write(keys: keys, values: keys * 0.75)
            eval(
                [fixture.backend.pool.group(row.groupKey).writeFence]
                    + row.nativeRecentEvaluationRoots)
        }
        StreamOrDevice.default.stream.synchronize()
    }

    private func bytes(_ array: MLXArray) throws -> Data {
        eval(array)
        let pointer = try #require(mlx_array_data_uint8(array.ctx))
        return Data(bytes: pointer, count: array.nbytes)
    }

    private func read(_ source: CBv2CompleteCheckpointExport) throws -> [Data] {
        try source.manifest.tensors.indices.map { index in
            var data = Data()
            while data.count < source.manifest.tensors[index].byteCount {
                data.append(
                    try source.readSegment(
                        tensorIndex: index, byteOffset: data.count, maximumBytes: 257))
            }
            return data
        }
    }

    private func codedRecent(_ row: PagedSequenceKV, fixture: Fixture) throws -> [Data] {
        let size = fixture.backend.pool.config.pageSize
        let pages = (384 / size ..< 512 / size).map { row.table[$0] }
        let copied = PagedQuantizedTransfers.gatherPacked(
            group: fixture.backend.pool.group(row.groupKey), pages: pages, firstSlot: 0, count: 128)
        return try [copied.keys, copied.values].map(bytes)
    }

    @Test(
        "Interior cuts do not copy, charge or displace a completed packed frontier",
        arguments: [DType.float16, .bfloat16, .float32])
    func interiorRefusal(dtype: DType) throws {
        let f = try fixture(dtype: dtype)
        let state = try donor(f)
        defer {
            f.backend.release(state)
            f.admission.releaseAll(id: donorID)
            f.capture.close()
        }
        try write(state, fixture: f, start: 0, count: 256)
        let first = try #require(
            try f.capture.prepareHistorical(position: 256, chunkSize: 128, state: state))
        try first.finishEvaluation()
        f.capture.commitHistorical(first, requestID: donorID)
        defer {
            let done = DispatchSemaphore(value: 0)
            let dropped = f.capture.drop(requestID: donorID, completion: { done.signal() })
            #expect(dropped)
            #expect(done.wait(timeout: .now() + 10) == .success)
            f.capture.queue.sync {}
        }
        try f.backend.pool.finishQuantizedStorageStep()
        try write(state, fixture: f, start: 256, count: 256)
        var copies = 0
        f.capture.makeHistoricalWindow = { row, position, admission in
            copies += 1
            return try .init(row: row, position: position, admission: admission)
        }
        let before = f.admission.bytesReserved
        let retention = f.capture.retention(
            requestID: donorID, stride: 128, hintTokens: 385, resumedAt: 0)
        let refused = try f.capture.prepareHistorical(
            positions: [384], retention: retention, state: state, requestID: donorID)
        #expect(refused.isEmpty)
        #expect(copies == 0)
        #expect(f.admission.bytesReserved == before)
        #expect(f.capture.inFlightHistoricalBytes == 0)
        #expect(f.capture.staged[donorID]?.compactMap(\.position) == [256])
        refused.forEach(f.capture.discardHistorical)
        let frontier = try f.capture.prepareHistorical(
            positions: [384, 512], retention: retention, state: state, requestID: donorID)
        defer { frontier.forEach(f.capture.discardHistorical) }
        #expect(frontier.compactMap(\.position) == [512])
    }

    @Test(
        "Completed packed frontiers preserve exact native bytes, codes and later aging",
        arguments: [DType.float16, .bfloat16, .float32])
    func frontierRoundTrip(dtype: DType) throws {
        let f = try fixture(dtype: dtype)
        let state = try donor(f)
        defer {
            f.backend.release(state)
            f.admission.releaseAll(id: donorID)
            f.capture.close()
        }
        try write(state, fixture: f, start: 0, count: 512)
        let candidate = try #require(
            try f.capture.prepareHistorical(position: 512, chunkSize: 128, state: state))
        defer { f.capture.discardHistorical(candidate) }
        try candidate.finishEvaluation()
        let checkpoint = try #require(candidate.historical)
        let recent = try #require(checkpoint.quantizedRecent[0])
        let expectedNative = try [#require(recent.keys), #require(recent.values)].map(bytes)
        let source = try f.codec.exportHistorical(
            checkpoint: checkpoint, state: state,
            tokens: f.request.promptTokens, cacheSalt: f.request.cacheSalt)
        defer { source.close() }
        let original = try read(source)
        #expect(source.manifest.position == 512)
        #expect(
            source.manifest.backendLayout
                == CBv2CompleteCheckpointManifest.quantizedHistoricalLayout)
        #expect(source.manifest.tensors.allSatisfy { $0.dtype == .uint8 })
        try f.backend.pool.finishQuantizedStorageStep()
        try write(state, fixture: f, start: 512, count: 256)
        try f.backend.pool.finishQuantizedStorageStep()
        #expect(try read(source) == original)
        let plan = try f.codec.plan(manifest: source.manifest, request: f.request)
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        for (index, data) in original.enumerated() {
            var offset = 0
            while offset < data.count {
                let end = min(offset + 257, data.count)
                try sink.appendSegment(
                    tensorIndex: index, byteOffset: offset, data: data.subdata(in: offset ..< end))
                offset = end
            }
        }
        let staged = try sink.finish()
        defer { staged.close() }
        let restored = try staged.consumePreparedState { prepared in
            let frame = try #require(prepared.pagedFrame)
            prepared.pagedFrame = nil
            let adoption = try f.backend.pool.importCheckpoint(
                frame, admission: f.admission, requestID: f.request.id,
                layerKinds: f.kinds, maximumTokens: plan.maximumSequenceLength)
            return try adoption.moveToActiveRequest { #expect($0.isEmpty) }
        }
        defer {
            f.backend.release(restored)
            f.admission.releaseAll(id: f.request.id)
        }
        let full = try #require(restored[0] as? PagedSequenceKV)
        #expect(full.nativeRecentStart == 384)
        #expect(
            try [#require(full.nativeRecentKeys), #require(full.nativeRecentValues)].map(bytes)
                == expectedNative)
        let restoredCandidate = try #require(
            try f.capture.prepareHistorical(position: 512, chunkSize: 128, state: restored))
        defer { f.capture.discardHistorical(restoredCandidate) }
        try restoredCandidate.finishEvaluation()
        let reexport = try f.codec.exportHistorical(
            checkpoint: #require(restoredCandidate.historical), state: restored,
            tokens: f.request.promptTokens, cacheSalt: f.request.cacheSalt)
        defer { reexport.close() }
        #expect(try read(reexport) == original)
        let codes = try codedRecent(#require(state[0] as? PagedSequenceKV), fixture: f)
        #expect(try codedRecent(full, fixture: f) == codes)
        try write(restored, fixture: f, start: 512, count: 129)
        try f.backend.pool.finishQuantizedStorageStep()
        #expect(full.nativeRecentStart == 513)
        #expect(try codedRecent(full, fixture: f) == codes)
    }

    @Test("Native owners retain interior cuts", arguments: [false, true])
    func nativeInterior(defaultQuantization: Bool) throws {
        let f = try fixture(
            quantization: defaultQuantization ? .init() : nil,
            nativeLayers: defaultQuantization ? [0, 1] : [])
        try checkNativeInterior(f)
    }

    @Test("Small native windows remain eligible with quantization configured")
    func smallNativeWindow() throws {
        try checkNativeInterior(fixture(windowOnly: true))
    }

    private func checkNativeInterior(_ f: Fixture) throws {
        let state = try donor(f)
        defer {
            f.backend.release(state)
            f.admission.releaseAll(id: donorID)
            f.capture.close()
        }
        #expect(
            state.compactMap { $0 as? PagedSequenceKV }.allSatisfy {
                $0.groupKey.quantization == nil
            })
        try write(state, fixture: f, start: 0, count: 512)
        let candidate = try #require(
            try f.capture.prepareHistorical(position: 384, chunkSize: 128, state: state))
        defer { f.capture.discardHistorical(candidate) }
        try candidate.finishEvaluation()
        #expect(candidate.position == 384)
    }

    @Test("Only actual packed owners require the frontier", arguments: [0, 1])
    func mixedOwners(nativeLayer: Int) throws {
        let f = try fixture(nativeLayers: [nativeLayer])
        let state = try donor(f)
        defer {
            f.backend.release(state)
            f.admission.releaseAll(id: donorID)
            f.capture.close()
        }
        #expect((state[nativeLayer] as? PagedSequenceKV)?.groupKey.quantization == nil)
        #expect((state[1 - nativeLayer] as? PagedSequenceKV)?.groupKey.quantization != nil)
        #expect(state[2] == nil)
        try write(state, fixture: f, start: 0, count: 512)
        try write(state, fixture: f, start: 512, count: 128, layer: nativeLayer)
        let refused = try f.capture.prepareHistorical(position: 384, chunkSize: 128, state: state)
        #expect(refused == nil)
        if let refused { f.capture.discardHistorical(refused) }
        let candidate = try #require(
            try f.capture.prepareHistorical(position: 512, chunkSize: 128, state: state))
        defer { f.capture.discardHistorical(candidate) }
        try candidate.finishEvaluation()
        #expect(candidate.position == 512)
    }
}
