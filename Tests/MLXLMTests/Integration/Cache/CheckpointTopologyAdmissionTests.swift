import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Checkpoint topology admission", .tags(.integration), .serialized)
struct CheckpointTopologyAdmissionTests {
    @Test("Auxiliary-only exports fit their original metadata budget", arguments: [false, true])
    func auxiliaryOnlyExport(paged: Bool) throws {
        let position = 32
        let metadataBytes = try CBv2CheckpointManifestMemory.reservationBytes(position: position)
        let admission = AdmissionV2(
            layerKinds: [], bytesCapacity: metadataBytes,
            config: .init(watermarkFraction: 0))
        let spec = CBv2RecurrentStateSpec(layers: [
            .init(
                modelLayerIndex: 0, convShape: [1, 2, 2], convDType: .float32,
                ssmShape: [1, 1, 2, 2], ssmDType: .float32)
        ])
        let pagedConfig: PagedKVPoolConfig? =
            paged
            ? .init(capacityBytes: 1 << 20, segmentSizeBytes: 64 << 10, layerDTypes: [])
            : nil
        let codec = CBv2CompleteCheckpointCodec(
            identity: .init(
                modelAggregateHash: "recurrent-only", promptContractID: "template",
                buildID: "native", numericsFingerprint: "float32"),
            layerKinds: [], recurrentSpec: spec, kvDTypes: [],
            assistant: nil, admission: admission, pagedConfig: pagedConfig)
        let conv = MLXArray([Float(1), 2, 3, 4]).reshaped([1, 2, 2])
        let ssm = MLXArray([Float(5), 6, 7, 8]).reshaped([1, 1, 2, 2])
        try withError { eval(conv, ssm) }
        let checkpoint = CBv2RecurrentCheckpoint(
            position: position, chunkSize: position,
            layers: [0: .init(conv: conv, ssm: ssm)], byteCount: 32)
        let tokens = Array(repeating: 7, count: position + 1)
        let expected = [conv, ssm].map { array in
            array.asArray(Float.self).withUnsafeBytes { Data($0) }
        }

        func exercise() throws {
            // The public export chooses the real contiguous/paged producer.
            // Both failed here when an absent topology still reserved its envelope.
            let source = try codec.export(
                checkpoint: checkpoint, state: [], tokens: tokens, cacheSalt: "tenant")
            defer { source.close() }
            #expect(source.manifest.tokenByteTopologies == nil)
            #expect(source.manifest.tensors.map(\.role) == [.convolution, .recurrent])
            #expect(try #require(source.manifest.metadata.permit).bytes == metadataBytes)
            #expect(admission.bytesReserved == metadataBytes)
            #expect(throws: CBv2KVError.self) { try admission.reserveTransient(bytes: 1) }
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(source.manifest))
                    as? [String: Any])
            #expect(encoded["tokenByteTopologies"] == nil)
            for index in expected.indices {
                #expect(
                    try source.readSegment(
                        tensorIndex: index, byteOffset: 0, maximumBytes: expected[index].count)
                        == expected[index])
            }
        }
        try exercise()
        #expect(admission.bytesReserved == 0)
    }
}
