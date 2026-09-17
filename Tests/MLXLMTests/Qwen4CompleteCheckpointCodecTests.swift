import Foundation
import MLX
import XCTest

@testable import MLXLMCommon
@testable import MLXLLM

final class Qwen4CompleteCheckpointCodecTests: XCTestCase {
    func testContiguousAndPagedTransferRestoreAllQSAStateWithoutConversion() throws {
        for paged in [false, true] {
            for pooled in [false, true] {
                let admission = try exercise(paged: paged, pooled: pooled)
                XCTAssertEqual(admission.bytesReserved, 0, "all imported/exported metadata and array owners retired")
            }
        }
    }

    func testIdentifiedMediaRestoresNativeStateWithoutAssistantOnMTPEnabledSlot() throws {
        for paged in [false, true] {
            for pooled in [false, true] {
                let admission = try exercise(paged: paged, pooled: pooled, media: true)
                XCTAssertEqual(admission.bytesReserved, 0,
                    "media imports/exported owners return every existing native reservation")
            }
        }
    }

    private func exercise(paged: Bool, pooled: Bool, media: Bool = false) throws -> AdmissionV2 {
        let chunk = max(32, CBv2AttentionV1.queryBlockSize)
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1,
            queryHeads: 1, modelLayerIndex: 3, qwen4IndexerCompressRatio: 4)]
        let geometry = try CBv2Qwen4CheckpointGeometry(layer: 3, headDim: 8,
            compressRatio: 4, keyDType: .bfloat16, pooledDType: .bfloat16)
        let spec = CBv2RecurrentStateSpec(layers: [.init(modelLayerIndex: 0,
            convShape: [1, 2], convDType: .float32, ssmShape: [1, 2], ssmDType: .float32)])
        let pagedConfig = PagedKVPoolConfig(capacityBytes: 128 << 20, maxPrefillChunk: chunk,
            segmentSizeBytes: 64 << 10, layerDTypes: [.bfloat16])
        let pagedBackend = try paged ? PagedKVBackend(layerKinds: kinds, config: pagedConfig) : nil
        let residency: any CBv2KVResidencyPolicy
        if paged { residency = CBv2PagedKVResidency(config: pagedConfig) }
        else { residency = CBv2ContiguousKVResidency() }
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 128 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2, layerElementBytes: nil,
                fixedBytesPerRequest: 16, auxiliaryBytesPerToken: 512),
            residency: residency)
        pagedBackend?.pool.bindAdmission(admission)
        let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: "synthetic-qwen4",
            promptContractID: "template", buildID: "qsa-codec-test", numericsFingerprint: "native-bf16")
        let assistant = try media ? Qwen4ExpMTPPrefixCheckpointTests().fixture() : nil
        let donor = CBv2CompleteCheckpointCodec(identity: identity, layerKinds: kinds,
            recurrentSpec: spec, kvDTypes: [.bfloat16], assistant: assistant, admission: admission,
            qwen4Geometries: [geometry])
        let codec = CBv2CompleteCheckpointCodec(identity: identity, layerKinds: kinds,
            recurrentSpec: spec, kvDTypes: [.bfloat16], assistant: assistant, admission: admission,
            pagedConfig: paged ? pagedConfig : nil, qwen4Geometries: [geometry])
        let positions = MLXArray((0..<(3 * chunk)).map { Int64(Int32.max) + Int64($0 + 1) }, [3, 1, chunk])
        let side = CBv2Qwen4IndexerSnapshot(tokenCount: chunk,
            indexKeys: MLXArray((0..<(chunk * 8)).map { Float($0) / 9 }, [1, chunk, 8]).asType(.bfloat16),
            positionIds: positions,
            pooledIndexKeys: pooled ? MLXArray.ones([1, chunk / 4, 8], dtype: .bfloat16) : nil,
            pooledIndexBlocks: pooled ? chunk / 4 : 0)
        let mediaIdentity = try media ? CBv2HybridPrefixIdentity(digest: Data(repeating: 0x71, count: 32)) : nil
        let checkpoint = CBv2RecurrentCheckpoint(position: chunk, chunkSize: chunk,
            layers: [0: .init(conv: MLXArray([Float(1), 2]).reshaped([1, 2]),
                             ssm: MLXArray([Float(3), 4]).reshaped([1, 2]))],
            byteCount: 16 + side.arrays.reduce(0) { $0 + $1.nbytes }, qwen4: [3: side],
            mediaIdentity: mediaIdentity, mediaTargetOnly: media)
        let kv = MLXArray.ones([1, 1, chunk + 3, 64], dtype: .bfloat16)
        eval(checkpoint.evaluationRoots + [kv])
        var request = CBv2Request(id: .init(11), promptTokens: Array(repeating: 1, count: chunk + 3),
            maxTokens: 4, cacheSalt: "tenant-a")
        if media {
            request.hybridPrefixIdentity = mediaIdentity
            var fullPositions: [Int64] = []
            for axis in 0..<3 {
                for column in 0..<(chunk + 3) {
                    fullPositions.append(Int64(Int32.max) + Int64(axis * chunk + column + 1))
                }
            }
            let fullPositionTensor = MLXArray(fullPositions, [3, 1, chunk + 3])
            request.positionState = .init(
                promptPositionIds: fullPositionTensor,
                decodeDeltas: [2])
            let embeddings = MLXArray.ones([1, 2, 64], dtype: .bfloat16)
            request.multimodal = .init(spans: [.init(tokenOffset: 0, length: 2)],
                                      attention: .causal, positionState: request.positionState, embeddings: { [embeddings] })
            eval(embeddings, request.positionState!.promptPositionIds)
        }
        let source = try donor.export(checkpoint: checkpoint, kv: [(kv, kv, chunk + 3)],
            tokens: request.promptTokens, cacheSalt: request.checkpointCacheSalt)
        let manifest = CBv2CompleteCheckpointManifest(identity: identity, position: chunk, chunkSize: chunk,
            prefixTokens: Array(request.promptTokens.prefix(chunk)), cacheSalt: request.checkpointCacheSalt,
            assistantCodecID: nil, tensors: source.manifest.tensors, backendLayout: codec.backendLayout,
            mediaIdentity: mediaIdentity, mediaTargetOnly: media)
        if media {
            XCTAssertNotNil(codec.assistant, "the loaded slot genuinely has a persistent assistant")
            XCTAssertNil(manifest.assistantCodecID, "media checkpoint contains target state only")
            XCTAssertFalse(manifest.tensors.contains { [.assistantHidden, .assistantTokens, .assistantFrontier].contains($0.role) })
            let decoded = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONEncoder().encode(manifest))
            XCTAssertEqual(decoded, manifest)
            for variant in 0..<5 {
                var changed = request
                switch variant {
                case 0: changed.hybridPrefixIdentity = nil
                case 1: changed.hybridPrefixIdentity = try .init(digest: Data(repeating: 0x72, count: 32))
                case 2: changed.cacheSalt = "tenant-b"
                case 3: changed.multimodal = nil; changed.positionState = nil
                default: changed.prefixCacheEnabled = false
                }
                XCTAssertThrowsError(try codec.plan(manifest: decoded, request: changed,
                    minimumChunkSize: chunk, maximumChunkSize: chunk), "media identity negative \(variant)")
            }
        }
        XCTAssertThrowsError(try codec.tensorDescriptors(position: chunk),
            "QSA checkpoints must never silently omit indexer side-state")
        let otherTenant = CBv2Request(id: .init(12), promptTokens: request.promptTokens,
            maxTokens: 4, cacheSalt: "tenant-b")
        XCTAssertThrowsError(try codec.plan(manifest: manifest, request: otherTenant,
            minimumChunkSize: chunk, maximumChunkSize: chunk))
        let incomplete = CBv2CompleteCheckpointManifest(identity: identity, position: chunk, chunkSize: chunk,
            prefixTokens: manifest.prefixTokens, cacheSalt: request.checkpointCacheSalt, assistantCodecID: nil,
            tensors: manifest.tensors.filter { !$0.role.isQwen4Indexer }, backendLayout: codec.backendLayout,
            mediaIdentity: mediaIdentity, mediaTargetOnly: media)
        XCTAssertThrowsError(try codec.plan(manifest: incomplete, request: request,
            minimumChunkSize: chunk, maximumChunkSize: chunk))
        let plan = try codec.plan(manifest: manifest, request: request,
            minimumChunkSize: chunk, maximumChunkSize: chunk)
        let sink = try plan.allocate {}
        for (index, descriptor) in manifest.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let data = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: 256)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: data)
                offset += data.count
            }
        }
        let staged = try sink.finish()
        sink.close()
        var rows: [CBv2SequenceKV?] = try staged.consumePreparedState { prepared in
            if let pagedBackend {
                let frame = try XCTUnwrap(prepared.pagedFrame)
                prepared.pagedFrame = nil
                let adoption = try pagedBackend.pool.importCheckpoint(frame, admission: admission,
                    requestID: request.id, layerKinds: kinds, maximumTokens: plan.maximumSequenceLength)
                return try adoption.moveToActiveRequest { auxiliary in
                    let restored = try codec.recurrentCheckpoint(manifest: manifest, auxiliary: auxiliary)
                    XCTAssertEqual(restored.mediaIdentity, mediaIdentity)
                    XCTAssertEqual(restored.mediaTargetOnly, media)
                    XCTAssertNil(restored.assistant)
                    XCTAssertEqual(restored.layers[0]?.ssm?.asData().data, checkpoint.layers[0]?.ssm?.asData().data)
                    try codec.restoreQwen4(restored, rows: adoption.rows)
                }
            }
            XCTAssertEqual(prepared.checkpoint?.mediaIdentity, mediaIdentity)
            XCTAssertEqual(prepared.checkpoint?.mediaTargetOnly, media)
            XCTAssertNil(prepared.checkpoint?.assistant)
            XCTAssertEqual(prepared.checkpoint?.layers[0]?.conv?.asData().data, checkpoint.layers[0]?.conv?.asData().data)
            return prepared.state
        }
        func checkRows() throws {
            let row = try XCTUnwrap(rows[0] as? any CBv2Qwen4IndexerRow)
            let actual = try row.snapshotQwen4Indexer()
            XCTAssertEqual(actual.pooledIndexBlocks, side.pooledIndexBlocks)
            XCTAssertEqual(actual.arrays.count, side.arrays.count)
            for (expected, restored) in zip(side.arrays, actual.arrays) {
                XCTAssertEqual(expected.dtype, restored.dtype)
                XCTAssertEqual(expected.asData().data, restored.asData().data)
            }
        }
        try checkRows()
        if let pagedBackend { pagedBackend.release(rows) }
        rows.removeAll()
        admission.releaseAll(id: request.id)
        staged.close()
        source.close()
        return admission
    }
}
