import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Native paged inference with quantized checkpoints", .serialized)
struct NativeQuantizedPagedCheckpointTests {
    private let position = 513
    private let donorID = CBv2RequestID(9001)

    private struct Fixture {
        let backend: PagedKVBackend
        let admission: AdmissionV2
        let codec: CBv2CompleteCheckpointCodec
        let request: CBv2Request
        let kinds: [CBv2LayerKind]
        let dtype: DType
    }

    private func fixture(_ dtype: DType = .float16) throws -> Fixture {
        let kinds: [CBv2LayerKind] = [
            .init(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4),
            .init(attention: .slidingWindow(257), headDim: 64, kvHeads: 1, queryHeads: 2),
            .init(attention: .slidingWindow(64), headDim: 64, kvHeads: 1, queryHeads: 2),
            .init(attention: .full, sharesKVWithLayer: 0, headDim: 64, kvHeads: 2, queryHeads: 4),
        ]
        let config = PagedKVPoolConfig(
            capacityBytes: 256 << 20, maxPrefillChunk: 1024, segmentSizeBytes: 32 << 10,
            layerDTypes: Array(repeating: dtype, count: kinds.count))
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: config.capacityBytes,
            config: .init(watermarkFraction: 0, elementBytes: dtype.size),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        let identity = CBv2CompleteCheckpointIdentity(
            modelAggregateHash: "native-page-model", promptContractID: "causal",
            buildID: "test-build", numericsFingerprint: "native-checkpoint-\(dtype)")
        let codec = CBv2CompleteCheckpointCodec(
            identity: identity, layerKinds: kinds, recurrentSpec: nil,
            kvDTypes: Array(repeating: dtype, count: kinds.count), assistant: nil,
            admission: admission, pagedConfig: config, checkpointQuantization: .init())
        return .init(
            backend: backend, admission: admission, codec: codec,
            request: .init(
                id: .init(1), promptTokens: Array(repeating: 1, count: 769),
                maxTokens: 256, cacheSalt: "tenant"), kinds: kinds, dtype: dtype)
    }

    private func values(_ f: Fixture, layer: Int, start: Int, count: Int, value: Bool) -> MLXArray {
        let heads = f.kinds[layer].kvHeads
        let floats: [Float] = (0 ..< heads * count * 64).map { i in
            let head = i / (count * 64)
            let token = start + (i / 64) % count
            let feature = i % 64
            if feature == 0 && token % 7 == 0 { return -Float.zero }
            return Float(sin(Double(head * 301 + layer * 131 + token * 17 + feature) * 0.019))
                * (value ? 0.75 : 1)
        }
        return MLXArray(floats, [heads, count, 64]).asType(f.dtype)
    }

    private func write(_ state: [CBv2SequenceKV?], _ f: Fixture, start: Int, count: Int) throws {
        for (index, entry) in state.enumerated() {
            guard let row = entry as? PagedSequenceKV else { continue }
            row.write(keys: values(f, layer: index, start: start, count: count, value: false),
                values: values(f, layer: index, start: start, count: count, value: true))
            eval(f.backend.pool.group(row.groupKey).writeFence)
        }
        StreamOrDevice.default.stream.synchronize()
    }

    private func source(_ state: [CBv2SequenceKV?], _ f: Fixture) throws
        -> CBv2CompleteCheckpointExport
    {
        var windows: [Int: CBv2HistoricalWindow] = [:]
        for index in [1, 2] {
            windows[index] = try .init(
                row: #require(state[index] as? PagedSequenceKV), position: position,
                admission: f.admission)
        }
        let checkpoint = CBv2HistoricalCompleteCheckpoint(
            position: position, chunkSize: 171, windows: windows)
        try checkpoint.finishEvaluation()
        return try f.codec.exportHistorical(
            checkpoint: checkpoint, state: state, tokens: f.request.promptTokens,
            cacheSalt: f.request.cacheSalt)
    }

    private func transfer(_ source: CBv2CompleteCheckpointExport, sink: CBv2CompleteCheckpointImport)
        throws
    {
        for (index, descriptor) in source.manifest.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let bytes = try source.readSegment(tensorIndex: index, byteOffset: offset,
                    maximumBytes: 257)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: bytes)
                offset += bytes.count
            }
        }
    }

    private func checkReference(_ row: PagedSequenceKV, _ f: Fixture, layer: Int) throws {
        let start: Int
        switch f.kinds[layer].attention {
        case .full: start = 0
        case .slidingWindow(let size): start = max(0, position - size)
        }
        let count = position - start
        let snapshot = row.gatherRange(start: start, count: count)
        for (isValue, actual) in [(false, snapshot.keys), (true, snapshot.values)] {
            let original = values(f, layer: layer, start: start, count: count, value: isValue)
                .asType(.float32).asArray(Float.self)
            var expected = original
            let profile = try #require(f.codec.checkpointQuantization)
            if count > profile.recentTokenCount {
                for head in 0 ..< f.kinds[layer].kvHeads {
                    for token in 0 ..< count - profile.recentTokenCount {
                        let offset = (head * count + token) * 64
                        expected.replaceSubrange(offset ..< offset + 64,
                            with: try PagedKVQuantizationReference.roundTrip(
                                Array(original[offset ..< offset + 64]), config: profile,
                                isKey: !isValue))
                    }
                }
            }
            let expectedNative = MLXArray(expected).asType(f.dtype).asType(.float32)
                .asArray(Float.self)
            let result = actual.asType(.float32).asArray(Float.self)
            #expect(result.count == expectedNative.count)
            let error = zip(result, expectedNative).reduce(Float.zero) { max($0, abs($1.0 - $1.1)) }
            #expect(error <= (f.dtype == .float32 ? 0.00001 : 0.01))
            for head in 0 ..< f.kinds[layer].kvHeads {
                let recentStart = (head * count + max(0, count - profile.recentTokenCount)) * 64
                let end = (head + 1) * count * 64
                #expect(result[recentStart ..< end].map(\.bitPattern)
                    == original[recentStart ..< end].map(\.bitPattern))
            }
        }
    }

    @Test("Interior native cuts keep ring mapping, exact recent rows and native serving pages",
        arguments: [DType.float16, .bfloat16, .float32])
    func roundTrip(dtype: DType) throws {
        let f = try fixture(dtype)
        try f.admission.reserve(id: donorID, additionalTokens: 1025)
        let donor = try f.backend.makeSequenceState(
            layerKinds: f.kinds, promptLength: 769, maxLength: 1025)
        defer { f.backend.release(donor); f.admission.releaseAll(id: donorID) }
        try write(donor, f, start: 0, count: 768)
        let exported = try source(donor, f)
        defer { exported.close() }
        #expect(exported.manifest.backendLayout == CBv2CompleteCheckpointManifest.nativeQuantizedHistoricalLayout)
        #expect(exported.manifest.tensors.map(\.dtype) == [.uint8, .uint8, .uint8, .uint8,
            CBv2CheckpointDType(dtype)!, CBv2CheckpointDType(dtype)!])
        let wireBytes = try exported.manifest.validateStructure()
        let nativeBytes = try f.codec.nativeTargetDescriptors(position: position).reduce(0) { $0 + $1.byteCount }
        #expect(wireBytes < nativeBytes)
        let plan = try f.codec.plan(manifest: exported.manifest, request: f.request)
        #expect(plan.scratchBytes >= CBv2NativeCheckpointRowCodec.scratchBytes)
        #expect(plan.pagedStoragePlan?.groups.allSatisfy { $0.key.quantization == nil } == true)
        #expect(plan.nativeTargetBytes > nativeBytes)
        let physicalBefore = f.backend.pool.groupKeys
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        try transfer(exported, sink: sink)
        let staged = try sink.finish()
        defer { staged.close() }
        let restored = try staged.consumePreparedState { prepared in
            let frame = try #require(prepared.pagedFrame)
            prepared.pagedFrame = nil
            let adopted = try f.backend.pool.importCheckpoint(
                frame, admission: f.admission, requestID: f.request.id,
                layerKinds: f.kinds, maximumTokens: plan.maximumSequenceLength)
            return try adopted.moveToActiveRequest { #expect($0.isEmpty) }
        }
        defer { f.backend.release(restored); f.admission.releaseAll(id: f.request.id) }
        #expect(f.backend.pool.groupKeys == physicalBefore)
        #expect(!f.backend.usesQuantizedStorage && f.backend.supportsOrdinaryDecodeChaining)
        #expect(restored[3] == nil)
        for index in [0, 1, 2] {
            let row = try #require(restored[index] as? PagedSequenceKV)
            #expect(row.groupKey.quantization == nil && row.nativeRecentKeys == nil)
            try checkReference(row, f, layer: index)
        }
        let rate = try f.codec.historicalReusePlan(position: position, maximumSequenceLength: 1025)
        #expect(rate.fullKVBytesPerToken == 2 * 2 * 64 * dtype.size)
        try write(restored, f, start: position, count: 400)
        let window = try #require(restored[1] as? PagedSequenceKV)
        let latest = window.gatherRange(start: position + 400 - 257, count: 257)
        let expected = values(f, layer: 1, start: position + 400 - 257, count: 257, value: true)
        #expect(latest.values.asType(.float32).asArray(Float.self)
            == expected.asType(.float32).asArray(Float.self))
    }

    @Test("Profile, native dtype, tenant and malformed metadata refuse before page mutation")
    func incompatibleMetadata() throws {
        let f = try fixture()
        try f.admission.reserve(id: donorID, additionalTokens: 1025)
        let donor = try f.backend.makeSequenceState(layerKinds: f.kinds, promptLength: 769, maxLength: 1025)
        defer { f.backend.release(donor); f.admission.releaseAll(id: donorID) }
        try write(donor, f, start: 0, count: 768)
        let exported = try source(donor, f)
        defer { exported.close() }
        let manifest = exported.manifest
        func altered(profile: PagedKVQuantizationConfig? = .init(),
            types: [CBv2CheckpointDType]? = nil, identity: CBv2CompleteCheckpointIdentity? = nil,
            tensors: [CBv2CheckpointTensorDescriptor]? = nil) -> CBv2CompleteCheckpointManifest
        {
            .init(identity: identity ?? manifest.identity, position: position, chunkSize: 171,
                prefixTokens: manifest.prefixTokens, cacheSalt: manifest.cacheSalt, assistantCodecID: nil,
                tensors: tensors ?? manifest.tensors, backendLayout: manifest.backendLayout,
                attentionLayers: manifest.attentionLayers, checkpointQuantization: profile,
                checkpointNativeDTypes: types ?? manifest.checkpointNativeDTypes)
        }
        let before = f.admission.bytesReserved
        let wired = f.backend.bytesWired
        let invalid = [
            altered(profile: .init(keyBits: 8)), altered(profile: nil),
            altered(types: [.float32]), altered(tensors: Array(manifest.tensors.dropLast())),
            altered(identity: .init(modelAggregateHash: "other", promptContractID: "causal",
                buildID: "test-build", numericsFingerprint: manifest.identity.numericsFingerprint)),
        ]
        for value in invalid {
            #expect(throws: CBv2CompleteCheckpointError.self) { try f.codec.plan(manifest: value, request: f.request) }
        }
        let foreign = CBv2Request(id: .init(2), promptTokens: f.request.promptTokens,
            maxTokens: f.request.maxTokens, cacheSalt: "other-tenant")
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try f.codec.plan(manifest: manifest, request: foreign)
        }
        #expect(f.admission.bytesReserved == before && f.backend.bytesWired == wired)
    }

    @Test("Native destination and row scratch admission precedes allocation and partial cancel releases")
    func refusalAndPartialCancel() throws {
        let f = try fixture()
        try f.admission.reserve(id: donorID, additionalTokens: 1025)
        let donor = try f.backend.makeSequenceState(layerKinds: f.kinds, promptLength: 769, maxLength: 1025)
        defer { f.backend.release(donor); f.admission.releaseAll(id: donorID) }
        try write(donor, f, start: 0, count: 768)
        let exported = try source(donor, f)
        defer { exported.close() }
        let plan = try f.codec.plan(manifest: exported.manifest, request: f.request)
        let baseline = f.admission.bytesReserved
        let full = plan.nativeDestinationBytes + plan.scratchBytes
            + CBv2CompleteCheckpointManifest.maximumProviderScratchBytes
        f.admission.updateBytesCapacity(baseline + full - 1)
        var evaluated = 0
        plan.evaluateDestinations = { _ in evaluated += 1 }
        let releases = CheckpointReleaseCounter()
        #expect(throws: CBv2KVError.self) { try plan.allocate { releases.increment() } }
        #expect(evaluated == 0 && releases.value == 1)
        #expect(f.admission.bytesReserved == baseline)
        f.admission.updateBytesCapacity(256 << 20)
        plan.evaluateDestinations = { arrays in try withError { eval(arrays) } }
        let sink = try plan.allocate { releases.increment() }
        let fragment = try exported.readSegment(tensorIndex: 0, byteOffset: 0, maximumBytes: 1)
        try sink.appendSegment(tensorIndex: 0, byteOffset: 0, data: fragment)
        #expect(throws: CBv2CompleteCheckpointError.incompleteTransfer) { try sink.finish() }
        sink.close()
        #expect(releases.value == 2 && f.admission.bytesReserved == baseline)
    }

    @Test("Recurrent auxiliary payloads preserve raw native bits beside lossy target pages")
    func nativeAuxiliaryState() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1,
            queryHeads: 2, modelLayerIndex: 1)]
        let config = PagedKVPoolConfig(capacityBytes: 64 << 20, maxPrefillChunk: 64,
            segmentSizeBytes: 32 << 10, layerDTypes: [.float32])
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let spec = CBv2RecurrentStateSpec(layers: [.init(modelLayerIndex: 0,
            convShape: [1, 2], convDType: .float32, ssmShape: [1, 1, 2], ssmDType: .float32)])
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: config.capacityBytes,
            config: .init(watermarkFraction: 0, elementBytes: 4,
                fixedBytesPerRequest: try spec.fixedBytesPerRequest()),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: "recurrent",
            promptContractID: "causal", buildID: "test", numericsFingerprint: "native-checkpoint")
        let codec = CBv2CompleteCheckpointCodec(identity: identity, layerKinds: kinds,
            recurrentSpec: spec, kvDTypes: [.float32], assistant: nil,
            admission: admission, pagedConfig: config,
            checkpointQuantization: .init(recentTokenCount: 8))
        let request = CBv2Request(id: .init(1), promptTokens: Array(repeating: 1, count: 65),
            maxTokens: 31, cacheSalt: "tenant")
        try admission.reserve(id: donorID, additionalTokens: 96)
        let donor = try backend.makeSequenceState(layerKinds: kinds, promptLength: 65, maxLength: 96)
        defer { backend.release(donor); admission.releaseAll(id: donorID) }
        let row = try #require(donor[0] as? PagedSequenceKV)
        let kv = (MLXArray(0 ..< 64 * 64).asType(.float32) / 1024).reshaped([1, 64, 64])
        row.write(keys: kv, values: kv * 0.5)
        eval(backend.pool.group(row.groupKey).writeFence)
        let conv = MLXArray([UInt32(0x8000_0000), 0x7fc0_1234], [1, 2]).view(dtype: .float32)
        let ssm = MLXArray([UInt32(0x3f80_0000), 0xff80_0000], [1, 1, 2]).view(dtype: .float32)
        let exported = try codec.export(checkpoint: .init(position: 64, chunkSize: 32,
            layers: [0: .init(conv: conv, ssm: ssm)], byteCount: 16), state: donor,
            tokens: request.promptTokens, cacheSalt: request.cacheSalt)
        defer { exported.close() }
        #expect(exported.manifest.backendLayout == CBv2CompleteCheckpointManifest.nativeQuantizedPagedLayout)
        #expect(exported.manifest.tensors.map(\.dtype) == [.uint8, .uint8, .float32, .float32])
        let expected = try [2, 3].map {
            try exported.readSegment(tensorIndex: $0, byteOffset: 0, maximumBytes: 64)
        }
        let plan = try codec.plan(manifest: exported.manifest, request: request)
        #expect(plan.nativeAuxiliaryBytes >= 16)
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        try transfer(exported, sink: sink)
        let stage = try sink.finish()
        defer { stage.close() }
        let restored = try stage.consumePreparedState { prepared in
            let frame = try #require(prepared.pagedFrame)
            prepared.pagedFrame = nil
            let adopted = try backend.pool.importCheckpoint(frame, admission: admission,
                requestID: request.id, layerKinds: kinds, maximumTokens: plan.maximumSequenceLength)
            return try adopted.moveToActiveRequest { auxiliary in
                #expect(auxiliary.map { $0.asData(access: .copy).data } == expected)
                let checkpoint = try codec.recurrentCheckpoint(manifest: exported.manifest, auxiliary: auxiliary)
                #expect(checkpoint.layers[0]?.conv?.dtype == .float32)
                #expect(checkpoint.layers[0]?.ssm?.dtype == .float32)
            }
        }
        defer { backend.release(restored); admission.releaseAll(id: request.id) }
        #expect(!backend.usesQuantizedStorage && backend.supportsOrdinaryDecodeChaining)
    }
}

private final class CheckpointReleaseCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.withLock { count } }
    func increment() { lock.withLock { count += 1 } }
}
