import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

private final class AsymmetricReleaseCount: @unchecked Sendable {
    private let lock = NSLock(); private var count = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    func add() { lock.lock(); count += 1; lock.unlock() }
}

final class CBv2AsymmetricCompleteCheckpointTests: XCTestCase {
    /// A real call boundary ends every export/plan/sink/stage manifest alias.
    /// Returning Void cannot smuggle a metadata owner past the final-zero check.
    @inline(never)
    private func withManifestScope(_ body: () throws -> Void) rethrows { try body() }

    private func metadataBytes(_ source: CBv2CompleteCheckpointExport,
                               codec: CBv2CompleteCheckpointCodec) throws -> Int {
        // Independent expansion of the documented metadata envelope. On the
        // failing run: Int=8, descriptor=48, attention=80, position=128 ->
        // 2,361,344 bytes; two independently exported manifests -> 4,722,688.
        let position = source.manifest.position
        let expected = 2 * position * MemoryLayout<Int>.stride
            + 2 * 4096 * (MemoryLayout<CBv2CheckpointTensorDescriptor>.stride + 8 * MemoryLayout<Int>.stride + 128)
            + 2 * 2048 * MemoryLayout<CBv2CheckpointAttentionLayer>.stride + (64 << 10)
        XCTAssertEqual(try CBv2CheckpointManifestMemory.reservationBytes(position: position), expected)
        let permit = try XCTUnwrap(source.manifest.metadata.permit)
        XCTAssertEqual(permit.bytes, expected)
        XCTAssertEqual(permit.admission, ObjectIdentifier(codec.admission))
        XCTAssertNil(permit.nativeOwner)
        return expected
    }
    private let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: "native", promptContractID: "template",
        buildID: "build", numericsFingerprint: "native-types")
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }
    private func kinds(_ key: Int = 192, _ value: Int = 128, window: Int = 17) -> [CBv2LayerKind] {
        [.init(attention: .full, headDim: key, valueHeadDim: value, kvHeads: 2, queryHeads: 2, modelLayerIndex: 3),
         .init(attention: .slidingWindow(window), headDim: key, valueHeadDim: value, kvHeads: 1, queryHeads: 2, modelLayerIndex: 7),
         .init(attention: .full, sharesKVWithLayer: 0, headDim: key, valueHeadDim: value, kvHeads: 2, queryHeads: 4, modelLayerIndex: 9)]
    }
    private func codec(_ kinds: [CBv2LayerKind], dtype: DType = .float32) -> CBv2CompleteCheckpointCodec {
        .init(identity: identity, layerKinds: kinds, recurrentSpec: nil, kvDTypes: Array(repeating: dtype, count: kinds.count),
              assistant: nil, admission: .init(layerKinds: kinds, bytesCapacity: 128 << 20,
                                               config: .init(watermarkFraction: 0, elementBytes: dtype.size)))
    }
    private func tensor(heads: Int, start: Int, count: Int, width: Int, dtype: DType, bias: Float) -> MLXArray {
        let values = (0..<heads * count * width).map { i in
            bias + Float((i / (count * width)) * 1000 + start * 7 + (i / width) % count * 7 + i % width) / 32
        }
        return MLXArray(values).reshaped([1, heads, count, width]).asType(dtype)
    }
    private func rows(_ kinds: [CBv2LayerKind], position: Int, dtype: DType = .float32) throws -> [CBv2SequenceKV?] {
        kinds.map { k in
            if k.sharesKVWithLayer != nil { return nil }
            let row: CBv2SequenceKV
            switch k.attention {
            case .full: row = CBv2FullSequenceKV(promptLength: position, maxLength: position + 20,
                kvHeads: k.kvHeads, headDim: k.headDim, valueHeadDim: k.valueHeadDim)
            case .slidingWindow(let w): row = CBv2WindowedSequenceKV(window: w, kvHeads: k.kvHeads,
                headDim: k.headDim, valueHeadDim: k.valueHeadDim)
            }
            let pair = row.update(keys: tensor(heads: k.kvHeads, start: 0, count: position, width: k.headDim, dtype: dtype, bias: 0),
                                 values: tensor(heads: k.kvHeads, start: 0, count: position, width: k.valueHeadDim, dtype: dtype, bias: -8))
            eval(pair.0, pair.1)
            let s = row.snapshot(); eval(s.keys, s.values)
            return row
        }
    }
    private func request(_ position: Int) -> CBv2Request {
        .init(id: .init(1), promptTokens: Array(repeating: 1, count: position + 5), maxTokens: 7, cacheSalt: "tenant")
    }
    private func export(_ codec: CBv2CompleteCheckpointCodec, _ rows: [CBv2SequenceKV?], _ position: Int) throws -> CBv2CompleteCheckpointExport {
        try codec.export(checkpoint: .init(position: position, chunkSize: chunk, layers: [:], byteCount: 0),
                         state: rows, tokens: request(position).promptTokens, cacheSalt: "tenant")
    }
    private func transfer(_ source: CBv2CompleteCheckpointExport, _ sink: CBv2CompleteCheckpointImport) throws {
        for (i, descriptor) in source.manifest.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let data = try source.readSegment(tensorIndex: i, byteOffset: offset, maximumBytes: 258)
                try sink.appendSegment(tensorIndex: i, byteOffset: offset, data: data); offset += data.count
            }
        }
    }

    func testLegacyUniformCanonicalEncodingIsByteIdentical() throws {
        let layers = try CBv2CheckpointAttentionLayer.resolve(layerKinds: [.init(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 2)], dtypes: [.float32])
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let expected = #"[{"dtype":"float32","hasSinks":false,"headDim":64,"kvHeads":2,"modelLayer":0,"owner":0,"queryHeads":2}]"#
        XCTAssertEqual(try encoder.encode(layers), Data(expected.utf8))
        let explicit = CBv2CheckpointAttentionLayer(modelLayer: 0, owner: 0, window: nil, kvHeads: 2, headDim: 64,
            valueHeadDim: 64, queryHeads: 2, hasSinks: false, dtype: .float32)
        XCTAssertEqual(try encoder.encode([explicit]), Data(expected.utf8))
        XCTAssertEqual(try JSONDecoder().decode([CBv2CheckpointAttentionLayer].self, from: Data(expected.utf8)), layers)
        let descriptors = try [CBv2CheckpointTensorRole.keys, .values].map {
            try CBv2CheckpointTensorDescriptor(role: $0, layer: 0, shape: [1, 2, 32, 64], dtype: .float32)
        }
        let manifest = CBv2CompleteCheckpointManifest(identity: identity, position: 32, chunkSize: 32,
            prefixTokens: Array(repeating: 1, count: 32), cacheSalt: nil, assistantCodecID: nil, tensors: descriptors)
        let data = try encoder.encode(manifest)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(object.keys), Set(["schemaVersion", "identity", "backendLayout", "position", "chunkSize", "prefixTokens", "tensors"]))
        XCTAssertEqual(object["backendLayout"] as? String, "native-contiguous-full-recurrent-v1")
        // Frozen pre-v2 complete-manifest oracle, not an encoder round trip.
        let legacy = #"{"backendLayout":"native-contiguous-full-recurrent-v1","chunkSize":32,"identity":{"buildID":"build","modelAggregateHash":"native","numericsFingerprint":"native-types","promptContractID":"template"},"position":32,"prefixTokens":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"schemaVersion":1,"tensors":[{"byteCount":16384,"dtype":"float32","layer":0,"role":"keys","shape":[1,2,32,64]},{"byteCount":16384,"dtype":"float32","layer":0,"role":"values","shape":[1,2,32,64]}]}"#
        XCTAssertEqual(data, Data(legacy.utf8))
        XCTAssertEqual(try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: Data(legacy.utf8)), manifest)
        XCTAssertEqual(try encoder.encode(JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: data)), data)
    }

    func testNativeWidthsDTypesRingRestoreAdoptionAndContinuation() throws {
        for (key, value) in [(192,128), (64,128)] {
            for dtype: DType in [.float16, .bfloat16, .float32] {
                let k = kinds(key,value), c = codec(k,dtype: dtype)
                try withManifestScope {
                    let original = try rows(k,position: chunk,dtype: dtype)
                    let source = try export(c,original,chunk); defer { source.close() }
                    let metadata = try metadataBytes(source, codec: c)
                    XCTAssertEqual(source.manifest.backendLayout, CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout)
                    XCTAssertEqual(source.manifest.tensors.map { $0.shape[3] }, [key,value,key,value])
                    let plan = try c.plan(manifest: source.manifest, request: request(chunk), minimumChunkSize: chunk, maximumChunkSize: chunk)
                    XCTAssertTrue(plan.manifest.metadata === source.manifest.metadata)
                    XCTAssertEqual(plan.destinationShapes.map { $0[2] }, [chunk+12,chunk+12,17,17])
                    let expectedBound = try plan.destinationShapes.reduce(0) { total, shape in
                        total + (try Memory.allocationFootprintUpperBound(byteCount: shape.reduce(dtype.size,*)))
                    }
                    XCTAssertEqual(plan.nativeDestinationBytes, expectedBound)
                    let release = AsymmetricReleaseCount(), sink = try plan.allocate { release.add() }
                    try transfer(source,sink)
                    let staged = try sink.finish(); sink.close()
                    let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 128 << 20, kvDType: dtype))
                    var restored = try staged.consumePreparedState { prepared in
                        try backend.adoptPreparedCheckpoint(prepared.state, codec: c, position: chunk,
                            requestID: request(chunk).id, maximumSequenceLength: plan.maximumSequenceLength)
                        XCTAssertThrowsError(try backend.adoptPreparedCheckpoint(prepared.state, codec: c, position: chunk,
                            requestID: request(chunk).id, maximumSequenceLength: plan.maximumSequenceLength))
                        return prepared.state
                    }
                    XCTAssertEqual(release.value,1); XCTAssertNil(restored[2])
                    XCTAssertEqual(backend.bytesReserved, expectedBound)
                    XCTAssertGreaterThanOrEqual(c.admission.bytesReserved - metadata, staged.nativeDestinationBytes)
                    for i in [0,1] {
                        XCTAssertEqual(restored[i]!.absoluteOffset,chunk)
                        XCTAssertEqual(restored[i]!.retainedCount,i == 0 ? chunk : 17)
                        for count in [1,3] {
                            let start = original[i]!.absoluteOffset
                            let keys = tensor(heads: k[i].kvHeads,start: start,count: count,width: key,dtype: dtype,bias: 0)
                            let values = tensor(heads: k[i].kvHeads,start: start,count: count,width: value,dtype: dtype,bias: -8)
                            let expected = original[i]!.update(keys: keys, values: values)
                            let actual = restored[i]!.update(keys: keys, values: values)
                            eval(expected.0,expected.1,actual.0,actual.1)
                            XCTAssertEqual(actual.0.asType(.float32).asArray(Float.self),expected.0.asType(.float32).asArray(Float.self))
                            XCTAssertEqual(actual.1.asType(.float32).asArray(Float.self),expected.1.asType(.float32).asArray(Float.self))
                        }
                    }
                    backend.release(restored); XCTAssertEqual(backend.bytesReserved,0)
                    c.admission.releaseAll(id: request(chunk).id)
                    XCTAssertGreaterThanOrEqual(c.admission.bytesReserved - metadata, expectedBound, "caller row aliases still retain imported backing")
                    restored.removeAll()
                    withExtendedLifetime((source, plan, sink, staged)) {
                        XCTAssertTrue(staged.manifest.metadata === source.manifest.metadata)
                        XCTAssertEqual(c.admission.bytesReserved, metadata, "only the independently live manifest owner remains")
                    }
                }
                XCTAssertEqual(c.admission.bytesReserved,0)
            }
        }
    }

    func testAtomicAdoptionUsesOneAllocatorBoundAtTightCapacity() throws {
        let k = kinds(), c = codec(k), released = AsymmetricReleaseCount()
        try withManifestScope {
            let source = try export(c, rows(k, position: chunk), chunk)
            defer { source.close() }
            let metadata = try metadataBytes(source, codec: c)
            let plan = try c.plan(manifest: source.manifest, request: request(chunk), minimumChunkSize: chunk, maximumChunkSize: chunk)
            XCTAssertTrue(plan.manifest.metadata === source.manifest.metadata)
            let scratch = plan.scratchBytes + CBv2CompleteCheckpointManifest.maximumProviderScratchBytes
            let final = max(plan.nativeDestinationBytes, c.admission.allocatedBytes(forTokens: plan.maximumSequenceLength))
            // Metadata is independently alive. At the transfer boundary scratch
            // is also still held: either destination fits, but double charge cannot.
            let capacity = metadata + max(plan.nativeDestinationBytes, final) + scratch
            XCTAssertLessThanOrEqual(metadata + plan.nativeDestinationBytes + scratch, capacity)
            XCTAssertLessThanOrEqual(metadata + final + scratch, capacity)
            XCTAssertGreaterThan(metadata + plan.nativeDestinationBytes + final + scratch, capacity)
            c.admission.updateBytesCapacity(capacity)
            let sink = try plan.allocate { released.add() }
            try transfer(source, sink)
            let stage = try sink.finish(); sink.close()
            XCTAssertEqual(c.admission.bytesReserved, metadata + plan.nativeDestinationBytes + scratch)
            let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: plan.nativeDestinationBytes, kvDType: .float32))
            var restored = try stage.consumePreparedState { prepared in
                let owner = try XCTUnwrap((prepared.state[0] as? CBv2FullSequenceKV)?.checkpointBacking)
                XCTAssertEqual(owner.measuredBytesByLayer.compactMap { $0 }.reduce(0,+), stage.nativeDestinationBytes)
                XCTAssertEqual(owner.allocationBoundsByLayer.compactMap { $0 }.reduce(0,+), plan.nativeDestinationBytes)
                XCTAssertGreaterThanOrEqual(plan.nativeDestinationBytes, stage.nativeDestinationBytes)
                try backend.adoptPreparedCheckpoint(prepared.state, codec: c, position: chunk,
                    requestID: request(chunk).id, maximumSequenceLength: plan.maximumSequenceLength)
                return prepared.state
            }
            XCTAssertEqual(released.value, 1)
            XCTAssertEqual(c.admission.bytesReserved - metadata, final)
            XCTAssertEqual(c.admission.transientBytesReserved - metadata, 0)
            XCTAssertEqual(backend.bytesReserved, plan.nativeDestinationBytes)
            c.admission.unreserve(id: request(chunk).id, tokens: plan.maximumSequenceLength, bytes: Int.max)
            XCTAssertEqual(c.admission.bytesReserved - metadata, final, "logical rollback cannot refund live native capacity")
            backend.release(restored); c.admission.releaseAll(id: request(chunk).id)
            XCTAssertEqual(c.admission.bytesReserved - metadata, final)
            restored.removeAll(); stage.close()
            withExtendedLifetime((source, plan, sink, stage)) {
                XCTAssertTrue(stage.manifest.metadata === source.manifest.metadata)
                XCTAssertEqual(c.admission.bytesReserved, metadata)
                XCTAssertEqual(c.admission.transientBytesReserved, metadata)
            }
        }
        XCTAssertEqual(c.admission.bytesReserved, 0); XCTAssertEqual(released.value, 1)
        XCTAssertEqual(c.admission.transientBytesReserved, 0)
    }

    func testAdmissionRefusalAndPostTransferFailureDoNotPublishOrRefundAliases() throws {
        for failAfterTransfer in [false, true] {
            let k = kinds(), c = codec(k)
            try withManifestScope {
                let source = try export(c, rows(k, position: chunk), chunk)
                defer { source.close() }
                let metadata = try metadataBytes(source, codec: c)
                let plan = try c.plan(manifest: source.manifest, request: request(chunk), minimumChunkSize: chunk, maximumChunkSize: chunk)
                let release = AsymmetricReleaseCount(), sink = try plan.allocate { release.add() }
                try transfer(source, sink)
                let stage = try sink.finish(); sink.close()
                let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 128 << 20, kvDType: .float32))
                var aliases: [CBv2SequenceKV?] = []
                var oldGeneration: UUID?
                if failAfterTransfer {
                    backend.checkpointBeforeRegistration = { throw MLXError.caught("intentional pre-publication failure") }
                } else {
                    c.admission.updateBytesCapacity(0)
                }
                XCTAssertThrowsError(try stage.consumePreparedState { prepared in
                    aliases = prepared.state
                    oldGeneration = (prepared.state[0] as? CBv2FullSequenceKV)?.checkpointBacking?.lease.identity
                    try backend.adoptPreparedCheckpoint(prepared.state, codec: c, position: chunk,
                        requestID: request(chunk).id, maximumSequenceLength: plan.maximumSequenceLength)
                })
                stage.close(); stage.close()
                XCTAssertEqual(release.value, 1); XCTAssertEqual(backend.bytesReserved, 0)
                XCTAssertEqual(backend.bytesInUse, 0)
                XCTAssertGreaterThanOrEqual(c.admission.bytesReserved - metadata, plan.nativeDestinationBytes)
                c.admission.updateBytesCapacity(128 << 20)
                try c.admission.reserve(id: request(chunk).id, additionalTokens: 1)
                let replacement = c.admission.allocatedBytes(forTokens: 1)
                // A failed generation must neither own nor refund the reused ID.
                c.admission.releaseCheckpointRequest(id: request(chunk).id, ownerIdentity: try XCTUnwrap(oldGeneration))
                aliases.removeAll()
                XCTAssertEqual(c.admission.bytesReserved - metadata, replacement)
                c.admission.releaseAll(id: request(chunk).id)
                withExtendedLifetime((source, plan, sink, stage)) {
                    XCTAssertTrue(plan.manifest.metadata === source.manifest.metadata)
                    XCTAssertTrue(stage.manifest.metadata === source.manifest.metadata)
                    XCTAssertEqual(c.admission.bytesReserved, metadata)
                }
            }
            XCTAssertEqual(c.admission.bytesReserved, 0)
        }
    }

    func testImportedBackingWaitsForBothRetirementLeaseAndExportAliases() throws {
        for releaseLeaseFirst in [false, true] {
            let k = kinds(), c = codec(k)
            try withManifestScope {
                let source = try export(c, rows(k, position: chunk), chunk)
                defer { source.close() }
                let metadata = try metadataBytes(source, codec: c)
                let plan = try c.plan(manifest: source.manifest, request: request(chunk), minimumChunkSize: chunk, maximumChunkSize: chunk)
                let sink = try plan.allocate {}; try transfer(source, sink)
                let stage = try sink.finish(); sink.close()
                let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 128 << 20, kvDType: .float32))
                var restored = try stage.consumePreparedState { prepared in
                    try backend.adoptPreparedCheckpoint(prepared.state, codec: c, position: chunk,
                        requestID: request(chunk).id, maximumSequenceLength: plan.maximumSequenceLength)
                    return prepared.state
                }
                let outgoing = try export(c, restored, chunk)
                let outgoingMetadata = try metadataBytes(outgoing, codec: c)
                XCTAssertEqual(outgoingMetadata, metadata)
                XCTAssertFalse(outgoing.manifest.metadata === source.manifest.metadata)
                let charged = c.admission.bytesReserved
                let retirement = c.admission.detachReservation(id: request(chunk).id)
                backend.release(restored); restored.removeAll()
                XCTAssertEqual(backend.bytesReserved, 0); XCTAssertEqual(c.admission.bytesReserved, charged)
                if releaseLeaseFirst { retirement.release() } else { outgoing.close() }
                XCTAssertEqual(c.admission.bytesReserved, charged, "both owners, in either destruction order, must retire")
                if releaseLeaseFirst { outgoing.close() } else { retirement.release() }
                outgoing.close(); retirement.release(); stage.close()
                withExtendedLifetime((source, plan, sink, stage, outgoing)) {
                    XCTAssertTrue(plan.manifest.metadata === source.manifest.metadata)
                    XCTAssertTrue(stage.manifest.metadata === source.manifest.metadata)
                    XCTAssertEqual(c.admission.bytesReserved, metadata + outgoingMetadata,
                        "both manifests remain valid after tensor/export retirement")
                }
            }
            XCTAssertEqual(c.admission.bytesReserved, 0)
        }
    }

    func testCompactWindowPhysicalProofAndExportOwnCopyCharge() throws {
        for dtype: DType in [.float16, .bfloat16, .float32] {
            for window in [17, chunk + 3] {
                let k = kinds(window: window), c = codec(k, dtype: dtype)
                try withManifestScope {
                    let state = try rows(k, position: chunk, dtype: dtype)
                    let capture = try CBv2ContiguousHistoricalCheckpoint(codec: c, position: chunk, chunkSize: chunk, state: state)
                    try capture.finishEvaluation()
                    let proof = try XCTUnwrap(capture.compactAllocationEvidence)
                    XCTAssertGreaterThan(proof.actual, 0); XCTAssertGreaterThanOrEqual(proof.bound, proof.actual)
                    let measured = try capture.evaluationRoots.reduce(0) { total, array in
                        let info = try XCTUnwrap(array.evaluatedBufferInfo())
                        XCTAssertTrue(info.isUnique); XCTAssertTrue(info.isRowContiguous)
                        XCTAssertEqual(info.dataOffset, 0); XCTAssertEqual(info.dataElements, array.size)
                        return total + info.allocatedBytes
                    }
                    XCTAssertEqual(proof.actual, measured)
                    let source = try capture.export(codec: c, state: state, tokens: request(chunk).promptTokens, cacheSalt: "tenant")
                    let metadata = try metadataBytes(source, codec: c)
                    let charged = c.admission.bytesReserved
                    capture.close(); capture.close()
                    XCTAssertEqual(c.admission.bytesReserved, charged, "source still owns the immutable copies")
                    XCTAssertFalse(try source.readSegment(tensorIndex: 3, byteOffset: 0, maximumBytes: 256).isEmpty)
                    source.close(); source.close()
                    withExtendedLifetime((source, capture)) {
                        XCTAssertEqual(c.admission.bytesReserved, metadata, "closed export still owns its public manifest")
                    }
                }
                XCTAssertEqual(c.admission.bytesReserved, 0)
            }
        }
    }

    func testCompactProofRejectsDonorAliasBeforePublication() throws {
        let k = kinds(), c = codec(k), state = try rows(k, position: chunk)
        let capture = try CBv2ContiguousHistoricalCheckpoint(codec: c, position: chunk, chunkSize: chunk, state: state)
        capture.evaluate = { arrays in
            let donor = state[0]!.snapshot(), end = arrays[0].dim(2) + 1
            // Deliberate fault injection through MLX's exposed internal
            // reference replacement, not a production cache operation.
            arrays[0]._updateInternal(donor.keys[0..<1, 0..<1, 1..<end, 0...])
            arrays[1]._updateInternal(donor.values[0..<1, 0..<1, 1..<end, 0...])
            try withError { eval(arrays) }
        }
        XCTAssertThrowsError(try capture.finishEvaluation())
        XCTAssertNil(capture.compactAllocationEvidence)
        XCTAssertThrowsError(try capture.export(codec: c, state: state, tokens: request(chunk).promptTokens, cacheSalt: "tenant"))
        capture.close(); XCTAssertEqual(c.admission.bytesReserved, 0)
    }

    func testManifestTamperingRefusedBeforeReservationOrNativeAllocation() throws {
        let c = codec(kinds()), state = try rows(kinds(),position: chunk), source = try export(c,state,chunk)
        defer { source.close() }
        let data = try JSONEncoder().encode(source.manifest)
        let mutations: [(inout [String: Any]) -> Void] = [
            { $0["backendLayout"] = CBv2CompleteCheckpointManifest.layout },
            { $0["backendLayout"] = CBv2CompleteCheckpointManifest.pagedLayout },
            { $0.removeValue(forKey: "attentionLayers") },
            { $0["assistantCodecID"] = "mtp" }, { $0["cacheSalt"] = "other" },
            { x in var a=x["attentionLayers"] as! [[String:Any]];a[0].removeValue(forKey:"valueHeadDim");x["attentionLayers"]=a },
            { x in var a=x["attentionLayers"] as! [[String:Any]];a[0]["valueHeadDim"]=192;x["attentionLayers"]=a },
            { x in var a=x["attentionLayers"] as! [[String:Any]];a[0]["dtype"]="float16";x["attentionLayers"]=a },
            { x in var a=x["attentionLayers"] as! [[String:Any]];for i in a.indices {a[i]["dtype"]="float16"};x["attentionLayers"]=a
                var t=x["tensors"] as! [[String:Any]];for i in t.indices {t[i]["dtype"]="float16";t[i]["byteCount"]=(t[i]["byteCount"] as! Int)/2};x["tensors"]=t },
            { x in var a=x["attentionLayers"] as! [[String:Any]];a[2]["owner"]=1;x["attentionLayers"]=a },
            { x in var a=x["tensors"] as! [[String:Any]];a[1]["byteCount"]=1;x["tensors"]=a },
            { x in var a=x["tensors"] as! [[String:Any]];a.removeLast();x["tensors"]=a },
            { x in let a=x["tensors"] as! [[String:Any]];x["tensors"]=a+[a[0]] },
        ]
        let reservation = c.admission.bytesReserved
        StreamOrDevice.default.stream.synchronize()
        let resources = Memory.numResources
        for mutate in mutations {
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String:Any]); mutate(&object)
            do {
                let manifest = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONSerialization.data(withJSONObject: object))
                XCTAssertThrowsError(try c.plan(manifest: manifest, request: request(chunk), minimumChunkSize: chunk, maximumChunkSize: chunk))
            } catch { /* Decoding malformed structure is also a preallocation refusal. */ }
            XCTAssertEqual(c.admission.bytesReserved,reservation)
        }
        XCTAssertLessThanOrEqual(Memory.numResources,resources)
    }

    func testUnwrappedWindowRestoresPartialRingThenWraps() throws {
        let k=kinds(window: chunk+3), c=codec(k), original=try rows(k,position: chunk)
        let source=try export(c,original,chunk); defer {source.close()}
        let plan=try c.plan(manifest: source.manifest,request: request(chunk),minimumChunkSize: chunk,maximumChunkSize: chunk)
        XCTAssertEqual(plan.destinationShapes[2],[1,1,chunk+3,192])
        XCTAssertEqual(plan.destinationShapes[3],[1,1,chunk+3,128])
        let sink=try plan.allocate {}; defer {sink.close()};try transfer(source,sink)
        let stage=try sink.finish();defer{stage.close()}
        try stage.consumePreparedState { prepared in
            let restored=try XCTUnwrap(prepared.state[1])
            XCTAssertEqual(restored.retainedCount,chunk)
            let keys=tensor(heads: 1,start: chunk,count: 5,width: 192,dtype: .float32,bias: 0)
            let values=tensor(heads: 1,start: chunk,count: 5,width: 128,dtype: .float32,bias: -8)
            let expected=original[1]!.update(keys: keys,values: values),actual=restored.update(keys: keys,values: values)
            XCTAssertEqual(actual.0.asArray(Float.self),expected.0.asArray(Float.self))
            XCTAssertEqual(actual.1.asArray(Float.self),expected.1.asArray(Float.self))
            XCTAssertEqual(restored.retainedCount,chunk+3);XCTAssertEqual(restored.absoluteOffset,chunk+5)
        }
    }

    func testHistoricalCopyPreservesFloatBitsAfterDonorOverwrite() throws {
        for window in [17,chunk+3] {
            let k=kinds(window: window),c=codec(k),state=try rows(k,position: chunk)
            let pattern: [Float] = [0,-0.0,.infinity,-.infinity,Float(bitPattern: 0x7fc01234),0.125]
            let v=MLXArray((0..<chunk*128).map { pattern[$0%pattern.count] }).reshaped([1,1,chunk,128])
            let special=CBv2WindowedSequenceKV(window: window,kvHeads: 1,headDim: 192,valueHeadDim: 128)
            _=special.update(keys: tensor(heads: 1,start: 0,count: chunk,width: 192,dtype: .float32,bias: 0),values: v)
            var donor=state;donor[1]=special
            let expected=special.snapshot().values.asArray(Float.self).map(\.bitPattern)
            let capture=try CBv2ContiguousHistoricalCheckpoint(codec: c,position: chunk,chunkSize: chunk,state: donor)
            try capture.finishEvaluation()
            _=special.update(keys: tensor(heads: 1,start: chunk,count: window,width: 192,dtype: .float32,bias: 900),
                             values: tensor(heads: 1,start: chunk,count: window,width: 128,dtype: .float32,bias: 900))
            let source=try capture.export(codec: c,state: donor,tokens: request(chunk).promptTokens,cacheSalt: "tenant")
            let data=try source.readSegment(tensorIndex: 3,byteOffset: 0,maximumBytes: source.manifest.tensors[3].byteCount)
            let actual=data.withUnsafeBytes { bytes in (0..<expected.count).map { bytes.loadUnaligned(fromByteOffset: $0*4,as: UInt32.self) } }
            XCTAssertEqual(actual,expected)
            source.close()
            // Keep the original no-rollback oracle above. A chained successor
            // may have overwritten the current ring before retirement rolls
            // its unconfirmed token back; that cannot invalidate this capture.
            let evidence=try XCTUnwrap(capture.compactAllocationEvidence)
            XCTAssertEqual(special.retainedCount,window)
            XCTAssertEqual(special.absoluteOffset,chunk+window)
            for (rollback,remaining) in [(1,window-1),(window-1,0)] {
                special.rollback(rollback) // ordinary update above, NOT speculative staging
                XCTAssertEqual(special.retainedCount,remaining)
                XCTAssertEqual(special.absoluteOffset,chunk+remaining)
                let before=c.admission.bytesReserved
                XCTAssertThrowsError(try c.validateContiguousRows(donor,position: chunk,exactWindow: false),
                    "The old historical guard rejects the shrunken current ring")
                XCTAssertThrowsError(try export(c,donor,chunk),
                    "Ordinary export must not pretend current history was restored")
                XCTAssertEqual(c.admission.bytesReserved,before)
                if remaining == 0 {
                    let empty=special.snapshot()
                    XCTAssertEqual(empty.values.shape,[1,1,0,128])
                    XCTAssertEqual(empty.values.dtype,.float16,
                        "Empty live placeholders do not describe the captured FP32 window")
                }
                let rolled=try capture.export(codec: c,state: donor,tokens: request(chunk).promptTokens,cacheSalt: "tenant")
                let bytes=try rolled.readSegment(tensorIndex: 3,byteOffset: 0,maximumBytes: rolled.manifest.tensors[3].byteCount)
                let bits=bytes.withUnsafeBytes { raw in (0..<expected.count).map { raw.loadUnaligned(fromByteOffset: $0*4,as: UInt32.self) } }
                XCTAssertEqual(bits,expected)
                XCTAssertEqual(rolled.manifest.position,chunk)
                XCTAssertEqual(rolled.manifest.tensors[3].shape,[1,1,min(chunk,window),128])
                XCTAssertEqual(capture.compactAllocationEvidence?.bound,evidence.bound)
                XCTAssertEqual(capture.compactAllocationEvidence?.actual,evidence.actual)
                rolled.close()
            }
            capture.close()
        }
    }

    func testInvalidRowsForeignCaptureOwnerAndPagedRefuseWithoutCharge() throws {
        for learnedIndexer in [false,true] {
            var unsupported=kinds()
            if learnedIndexer {unsupported[0].qwen4IndexerCompressRatio=8}
            else {unsupported[0].extraStorageBytesPerToken=4}
            let blocked=codec(unsupported)
            XCTAssertNil(blocked.contiguousLayout)
            XCTAssertThrowsError(try blocked.tensorDescriptors(position: chunk))
            XCTAssertEqual(blocked.admission.bytesReserved,0)
        }
        let c = codec(kinds()), state = try rows(kinds(),position: chunk)
        var invalid = state; invalid[2] = state[0]
        XCTAssertThrowsError(try export(c,invalid,chunk)); XCTAssertEqual(c.admission.bytesReserved,0)
        invalid = state; invalid[1] = state[0]
        XCTAssertThrowsError(try export(c,invalid,chunk)); XCTAssertEqual(c.admission.bytesReserved,0)
        let wrong = try rows(kinds(),position: chunk,dtype: .float16)
        XCTAssertThrowsError(try export(c,wrong,chunk)); XCTAssertEqual(c.admission.bytesReserved,0)
        let checkpoint = CBv2RecurrentCheckpoint(position: chunk,chunkSize: chunk,layers: [:],byteCount: 0)
        XCTAssertThrowsError(try c.exportPaged(checkpoint: checkpoint,state: state,tokens: request(chunk).promptTokens,cacheSalt: "tenant"))
        XCTAssertEqual(c.admission.bytesReserved,0)
        let capture = try CBv2ContiguousHistoricalCheckpoint(codec: c,position: chunk,chunkSize: chunk,state: state)
        try capture.finishEvaluation()
        let foreign = try rows(kinds(),position: chunk)
        let before = c.admission.bytesReserved
        XCTAssertThrowsError(try capture.export(codec: c,state: foreign,tokens: request(chunk).promptTokens,cacheSalt: "tenant"))
        XCTAssertEqual(c.admission.bytesReserved,before)
        let foreignCodec=codec(kinds())
        XCTAssertEqual(foreignCodec.identity,c.identity)
        XCTAssertThrowsError(try capture.export(codec: foreignCodec,state: state,tokens: request(chunk).promptTokens,cacheSalt: "tenant"),
            "Even matching metadata cannot move a capture to another loaded codec/Admission owner")
        XCTAssertEqual(foreignCodec.admission.bytesReserved,0)
        invalid=state;invalid[2]=state[0]
        XCTAssertThrowsError(try capture.export(codec: c,state: invalid,tokens: request(chunk).promptTokens,cacheSalt: "tenant"))
        XCTAssertThrowsError(try capture.export(codec: c,state: Array(state.dropLast()),tokens: request(chunk).promptTokens,cacheSalt: "tenant"))
        state[0]!.rollback(1)
        XCTAssertEqual(state[0]!.absoluteOffset,chunk-1)
        XCTAssertThrowsError(try capture.export(codec: c,state: state,tokens: request(chunk).promptTokens,cacheSalt: "tenant"),
            "An immutable SWA copy does not authorize an unavailable full-attention prefix")
        XCTAssertEqual(c.admission.bytesReserved,before)
        capture.close(); XCTAssertEqual(c.admission.bytesReserved,0)
    }

    func testCancellationAndAllocationFailureReleaseExactlyOnce() throws {
        let c = codec(kinds()), source = try export(c,rows(kinds(),position: chunk),chunk)
        defer { source.close() }
        let plan = try c.plan(manifest: source.manifest,request: request(chunk),minimumChunkSize: chunk,maximumChunkSize: chunk)
        let before = c.admission.bytesReserved, releases = AsymmetricReleaseCount()
        var sink: CBv2CompleteCheckpointImport? = try plan.allocate { releases.add() }
        XCTAssertThrowsError(try sink!.finish())
        sink!.close(); sink!.close(); sink=nil
        XCTAssertEqual(releases.value,1); XCTAssertEqual(c.admission.bytesReserved,before)
        plan.evaluateDestinations = { _ in throw MLXError.caught("intentional allocation refusal") }
        XCTAssertThrowsError(try plan.allocate { releases.add() })
        XCTAssertEqual(releases.value,2); XCTAssertEqual(c.admission.bytesReserved,before)
    }

    func testRestoredFullRowSupportsFrozenReplayAndUnequalAppend() throws {
        let k = kinds(), c = codec(k), original = try rows(k,position: chunk), source = try export(c,original,chunk)
        defer { source.close() }
        let plan = try c.plan(manifest: source.manifest,request: request(chunk),minimumChunkSize: chunk,maximumChunkSize: chunk)
        let sink = try plan.allocate {}; defer { sink.close() }; try transfer(source,sink)
        let stage = try sink.finish(); defer { stage.close() }
        // Native snapshots borrow their row's lifetime; keep that owner while
        // the frozen replay aliases its buffers, just like ordinary caches.
        let restored = try stage.consumePreparedState { $0.state }
        defer { withExtendedLifetime(restored) {} }
        let snapshot = try XCTUnwrap(restored[0]).snapshot()
        let frozen = CBv2FrozenReplayFullSequenceKV(snapshot: snapshot,replayStart: chunk-2,maxLength: chunk+12,
            kvHeads: 2,headDim: 192,valueHeadDim: 128)
        _ = frozen.update(keys: tensor(heads: 2,start: chunk-2,count: 2,width: 192,dtype: .float32,bias: 100),
                          values: tensor(heads: 2,start: chunk-2,count: 2,width: 128,dtype: .float32,bias: 100))
        XCTAssertEqual(frozen.snapshot().values.asArray(Float.self),snapshot.values.asArray(Float.self))
        let key = tensor(heads: 2,start: chunk,count: 1,width: 192,dtype: .float32,bias: 0)
        let value = tensor(heads: 2,start: chunk,count: 1,width: 128,dtype: .float32,bias: -8)
        let actual=frozen.update(keys: key,values: value), expected=original[0]!.update(keys: key,values: value)
        XCTAssertEqual(actual.1.shape,[1,2,chunk+1,128]); XCTAssertEqual(actual.1.asArray(Float.self),expected.1.asArray(Float.self))
    }
}
