import MLX
import Testing

@testable import MLXLMCommon

@Suite("Elastic window complete checkpoint round trip", .tags(.integration), .serialized)
struct CBv2ElasticWindowCheckpointTests {
    private func checked<T>(_ phase: String, _ operation: () throws -> T) throws -> T {
        do { return try operation() } catch {
            throw MLXError.caught("elastic checkpoint \(phase): \(error)")
        }
    }
    @Test func shortAndWrappedRowsRestoreNativeBitsAndContinueExactly() throws {
        for window in [1_024, 17] {
            let (admission, backend) = try exercise(window: window)
            #expect(admission.bytesReserved == 0)
            #expect(backend.bytesReserved == 0)
        }
    }

    private func exercise(window: Int) throws -> (AdmissionV2, CBv2ContiguousKVBackend) {
        let position = 32
        let kind = CBv2LayerKind(
            attention: .slidingWindow(window), headDim: 64, valueHeadDim: 32,
            kvHeads: 1, queryHeads: 2)
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .bfloat16, elasticWindowStorage: true))
        let admission = AdmissionV2(
            layerKinds: [kind], bytesCapacity: 64 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2), residency: backend.kvResidency)
        let codec = CBv2CompleteCheckpointCodec(
            identity: .init(
                modelAggregateHash: "fixture", promptContractID: "template",
                buildID: "native", numericsFingerprint: "bf16"),
            layerKinds: [kind], recurrentSpec: nil, kvDTypes: [.bfloat16],
            assistant: nil, admission: admission)
        _ = try #require(codec.contiguousLayout)
        let sourceRow = CBv2WindowedSequenceKV(
            window: window, kvHeads: 1, headDim: 64, valueHeadDim: 32, elasticStorage: true)
        func tensor(_ start: Int, _ count: Int, _ width: Int) -> MLXArray {
            sin(MLXArray(start * width ..< (start + count) * width).asType(.float32) * 0.17)
                .reshaped([1, 1, count, width]).asType(.bfloat16)
        }
        _ = sourceRow.update(keys: tensor(0, position, 64), values: tensor(0, position, 32))
        eval(sourceRow.cbv2InnerState())
        let request = CBv2Request(
            id: .init(10), promptTokens: Array(repeating: 7, count: position + 16),
            maxTokens: 64, cacheSalt: "scope")
        let source = try checked("export") {
            try codec.export(
                checkpoint: .init(position: position, chunkSize: 32, layers: [:], byteCount: 0),
                kv: [sourceRow.snapshot()], tokens: request.promptTokens,
                cacheSalt: request.cacheSalt)
        }
        defer { source.close() }
        #expect(source.manifest.tensors[0].shape[2] == min(window, position))
        let plan = try checked("plan") {
            try codec.plan(manifest: source.manifest, request: request)
        }
        #expect(plan.destinationShapes[0][2] == window)
        let sink = try checked("allocate") { try plan.allocate {} }
        defer { sink.close() }
        for (index, descriptor) in source.manifest.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let segment = try source.readSegment(
                    tensorIndex: index, byteOffset: offset, maximumBytes: 20)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: segment)
                offset += segment.count
            }
        }
        let staged = try checked("finish") { try sink.finish() }
        defer { staged.close() }
        let restored = try checked("adopt") {
            try staged.consumePreparedState { prepared in
                try backend.adoptContiguousHistoricalState(
                    prepared.state, codec: codec, layerKinds: [kind], position: position,
                    requestID: request.id,
                    maximumSequenceLength: request.promptTokens.count + request.maxTokens)
                return prepared.state
            }
        }
        defer {
            backend.release(restored)
            admission.releaseAll(id: request.id)
        }
        let restoredRow = try #require(restored[0] as? CBv2WindowedSequenceKV)
        #expect(restoredRow.cbv2InnerState()[0].dim(2) == window)
        var offset = position
        for count in [0, 1, 7, 32] {
            if count > 0 {
                let keys = tensor(offset, count, 64)
                let values = tensor(offset, count, 32)
                let sourceViews = sourceRow.update(keys: keys, values: values)
                let restoredViews = restoredRow.update(keys: keys, values: values)
                #expect(
                    sourceViews.0.asData(access: .copy).data
                        == restoredViews.0.asData(access: .copy).data)
                #expect(
                    sourceViews.1.asData(access: .copy).data
                        == restoredViews.1.asData(access: .copy).data)
            }
            let a = sourceRow.snapshot()
            let b = restoredRow.snapshot()
            #expect(a.offset == b.offset)
            #expect(a.keys.asData(access: .copy).data == b.keys.asData(access: .copy).data)
            #expect(a.values.asData(access: .copy).data == b.values.asData(access: .copy).data)
            offset += count
        }
        return (admission, backend)
    }
}
