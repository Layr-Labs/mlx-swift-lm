import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Quantized complete checkpoint ownership", .serialized)
struct QuantizedCompleteCheckpointTests {
    @Test(
        "Packed history and exact recent band survive later ring writes and authenticated import",
        arguments: [DType.float16, .bfloat16, .float32],
        [
            PagedKVQuantizationConfig(),
            PagedKVQuantizationConfig(keyBits: 8, valueBits: 4),
            PagedKVQuantizationConfig(keyBits: 8, valueBits: 8),
        ])
    func exactRoundTrip(dtype: DType, profile: PagedKVQuantizationConfig) throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(256), headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 0, headDim: 64, kvHeads: 2, queryHeads: 4),
        ]
        let config = PagedKVPoolConfig(
            capacityBytes: 256 << 20, maxPrefillChunk: 512,
            segmentSizeBytes: 64 << 10, layerDTypes: Array(repeating: dtype, count: kinds.count),
            quantization: profile, nativeLayerIndices: [2])
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: config.capacityBytes,
            config: try backend.pool.admissionStorageConfig(
                .init(watermarkFraction: 0, elementBytes: dtype.size)),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        let identity = CBv2CompleteCheckpointIdentity(
            modelAggregateHash: "packed-model",
            promptContractID: "causal", buildID: "test", numericsFingerprint: "packed-\(dtype)")
        let codec = CBv2CompleteCheckpointCodec(
            identity: identity, layerKinds: kinds,
            recurrentSpec: nil, kvDTypes: Array(repeating: dtype, count: kinds.count),
            assistant: nil,
            admission: admission, pagedConfig: config)
        let request = CBv2Request(
            id: .init(1), promptTokens: Array(repeating: 1, count: 769),
            maxTokens: 256, cacheSalt: "tenant")
        try admission.reserve(id: .init(9001), additionalTokens: 1025)
        let donor = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 769, maxLength: 1025)
        defer {
            backend.release(donor)
            admission.releaseAll(id: .init(9001))
        }
        func write(_ state: [CBv2SequenceKV?], start: Int, count: Int) throws {
            for (index, entry) in state.enumerated() {
                if kinds[index].sharesKVWithLayer != nil {
                    #expect(entry == nil)
                    continue
                }
                let row = try #require(entry as? PagedSequenceKV)
                let data = (0 ..< 2 * count * 64).map {
                    Float(sin(Double($0 + start * 64 + index * 31) * 0.017))
                }
                let keys = MLXArray(data, [2, count, 64]).asType(dtype)
                row.write(keys: keys, values: keys * 0.75)
                eval(
                    [backend.pool.group(row.groupKey).writeFence] + row.nativeRecentEvaluationRoots)
            }
            StreamOrDevice.default.stream.synchronize()
        }
        try write(donor, start: 0, count: 512)
        let windowRow = try #require(donor[1] as? PagedSequenceKV)
        let window = try CBv2HistoricalWindow(row: windowRow, position: 512, admission: admission)
        let recent = try codec.captureQuantizedRecent(
            state: donor, position: 512, includeWindows: false)
        let checkpoint = CBv2HistoricalCompleteCheckpoint(
            position: 512, chunkSize: 512,
            windows: [1: window], quantizedRecent: recent)
        try checkpoint.finishEvaluation()
        func nativeBytes(_ array: MLXArray) throws -> Data {
            eval(array)
            let pointer = try #require(mlx_array_data_uint8(array.ctx))
            return Data(bytes: pointer, count: array.nbytes)
        }
        let expectedNative = try nativeBytes(#require(recent[0]?.keys))
        let source = try codec.exportHistorical(
            checkpoint: checkpoint, state: donor,
            tokens: request.promptTokens, cacheSalt: request.cacheSalt)
        defer { source.close() }
        #expect(
            source.manifest.backendLayout
                == CBv2CompleteCheckpointManifest.quantizedHistoricalLayout)
        #expect(source.manifest.tensors.prefix(4).allSatisfy { $0.dtype == .uint8 })
        #expect(
            source.manifest.tensors.suffix(2).allSatisfy { $0.dtype == CBv2CheckpointDType(dtype) })
        let topologies = try #require(source.manifest.tokenByteTopologies)
        #expect(topologies.count == 6 && topologies.filter(\.nativeExempt).count == 2)
        for topology in topologies {
            let validated = try source.manifest.validatedTokenByteTopology(
                tensorIndex: topology.tensorIndex)
            let checked = try #require(validated)
            #expect(checked == topology)
            #expect(checked.isFullAttentionHistory == (checked.layer != 1))
            var coverage = 0
            for head in 0 ..< checked.headCount {
                for component in try checked.components {
                    let span = try checked.byteSpan(
                        head: head, component: component.kind,
                        absoluteTokenStart: component.absoluteTokenStart,
                        tokenCount: component.tokenCount)
                    #expect(span.byteOffset == coverage)
                    coverage += span.byteCount
                }
            }
            #expect(coverage == source.manifest.tensors[checked.tensorIndex].byteCount)
        }
        let packed = try source.manifest.validateStructure()
        #expect(packed < 2 * 2 * (512 + 256 + 512) * 64 * dtype.size)
        var originalBytes: [Data] = []
        for index in source.manifest.tensors.indices {
            var bytes = Data()
            while bytes.count < source.manifest.tensors[index].byteCount {
                bytes.append(
                    try source.readSegment(
                        tensorIndex: index, byteOffset: bytes.count, maximumBytes: 257))
            }
            originalBytes.append(bytes)
        }
        try backend.pool.finishQuantizedStorageStep()
        try write(donor, start: 512, count: 256)
        try backend.pool.finishQuantizedStorageStep()
        let plan = try codec.plan(manifest: source.manifest, request: request)
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        for (index, bytes) in originalBytes.enumerated() {
            var offset = 0
            while offset < bytes.count {
                let alignment = source.manifest.tensors[index].dtype.mlxDType.size
                let fragment = 257 - 257 % alignment
                let end = min(offset + fragment, bytes.count)
                try sink.appendSegment(
                    tensorIndex: index, byteOffset: offset, data: bytes.subdata(in: offset ..< end))
                offset = end
            }
        }
        let staged = try sink.finish()
        defer { staged.close() }
        let restored = try staged.consumePreparedState { prepared in
            let frame = try #require(prepared.pagedFrame)
            prepared.pagedFrame = nil
            let adoption = try backend.pool.importCheckpoint(
                frame, admission: admission,
                requestID: request.id, layerKinds: kinds, maximumTokens: plan.maximumSequenceLength)
            return try adoption.moveToActiveRequest { #expect($0.isEmpty) }
        }
        defer {
            backend.release(restored)
            admission.releaseAll(id: request.id)
        }
        let restoredFull = try #require(restored[0] as? PagedSequenceKV)
        #expect(restoredFull.nativeRecentStart == 384)
        #expect(try nativeBytes(#require(restoredFull.nativeRecentKeys)) == expectedNative)
        let restoredWindow = try CBv2HistoricalWindow(
            row: #require(restored[1] as? PagedSequenceKV),
            position: 512, admission: admission)
        let restoredCheckpoint = CBv2HistoricalCompleteCheckpoint(
            position: 512, chunkSize: 512,
            windows: [1: restoredWindow],
            quantizedRecent: try codec.captureQuantizedRecent(
                state: restored, position: 512, includeWindows: false))
        try restoredCheckpoint.finishEvaluation()
        let reexport = try codec.exportHistorical(
            checkpoint: restoredCheckpoint, state: restored,
            tokens: request.promptTokens, cacheSalt: request.cacheSalt)
        defer { reexport.close() }
        #expect(reexport.manifest.tokenByteTopologies == source.manifest.tokenByteTopologies)
        for (index, expected) in originalBytes.enumerated() {
            var bytes = Data()
            while bytes.count < expected.count {
                bytes.append(
                    try reexport.readSegment(
                        tensorIndex: index, byteOffset: bytes.count, maximumBytes: 259))
            }
            #expect(
                bytes == expected, "No dequantization or requantization of older checkpoint rows")
        }
        func codedRecent(_ row: PagedSequenceKV) throws -> [Data] {
            let size = backend.pool.config.pageSize
            let pages = (384 / size ..< 512 / size).map { row.table[$0] }
            let copied = PagedQuantizedTransfers.gatherPacked(
                group: backend.pool.group(row.groupKey), pages: pages, firstSlot: 0, count: 128)
            return try [copied.keys, copied.values].map(nativeBytes)
        }
        let donorCodes = try codedRecent(#require(donor[0] as? PagedSequenceKV))
        let restoredCodes = try codedRecent(restoredFull)
        #expect(
            restoredCodes == donorCodes,
            "The checkpoint also preserves codes behind its exact native band")
        try write(restored, start: 512, count: 129)
        try backend.pool.finishQuantizedStorageStep()
        #expect(restoredFull.nativeRecentStart == 513)
        #expect(
            try codedRecent(restoredFull) == donorCodes,
            "All 128 formerly native rows age without a second quantization")
    }
}
