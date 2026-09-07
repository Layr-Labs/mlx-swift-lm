import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension PagedCompleteCheckpointCodecTests {
    @Test("Packed codes and FP32 metadata survive complete stage/adopt/export without requantization",
          arguments: [PagedKVQuantizationConfig(),
                      .init(keyBits: 8, valueBits: 4),
                      .init(keyBits: 8, valueBits: 8),
                      .init(rotationBlockSize: 0)])
    func quantizedRoundTrip(quantization: PagedKVQuantizationConfig) throws {
        for dtype in [DType.bfloat16, .float32] {
            try checkRoundTrip(dtype: dtype, quantization: quantization)
        }
    }

    @Test func quantizedManifestBindsFormatBeforeAllocation() throws {
        let quantized = try fixture(quantization: .init(keyBits: 8, valueBits: 4))
        let manifest = quantized.manifest
        #expect(manifest.backendLayout == CBv2CompleteCheckpointManifest.quantizedPagedLayout)
        #expect(manifest.tensors[0].dtype == .uint8 && manifest.tensors[1].dtype == .uint8)
        #expect(manifest.tensors[0].shape[3] > manifest.tensors[1].shape[3])
        #expect(manifest.tensors[2].dtype == .bfloat16 && manifest.tensors[3].dtype == .float32)
        let encoded = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: encoded)
        #expect(decoded == manifest)
        #expect(try decoded.validateStructure() > 0)

        for other in [try fixture(), try fixture(quantization: .init()),
                      try fixture(quantization: .init(keyBits: 8, valueBits: 4, rotationBlockSize: 0))] {
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try other.codec.plan(manifest: manifest, request: other.request,
                    minimumChunkSize: manifest.chunkSize, maximumChunkSize: manifest.chunkSize)
            }
            #expect(other.admission.bytesReserved == 0 && other.backend.bytesWired == 0)
        }
        let native = try fixture()
        let nativeJSON = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(native.manifest))
            as? [String: Any])
        #expect(nativeJSON["kvQuantization"] == nil, "native manifest encoding remains backward compatible")
    }

    @Test func packedMetadataCannotBeRelabeledNative() throws {
        let fixture = try fixture(quantization: .init())
        let m = fixture.manifest
        for (layout, quantization) in [
            (CBv2CompleteCheckpointManifest.pagedLayout, m.kvQuantization),
            (CBv2CompleteCheckpointManifest.quantizedPagedLayout, nil),
            (CBv2CompleteCheckpointManifest.pagedLayout, nil),
        ] {
            let invalid = CBv2CompleteCheckpointManifest(identity: m.identity, position: m.position,
                chunkSize: m.chunkSize, prefixTokens: m.prefixTokens, cacheSalt: m.cacheSalt,
                assistantCodecID: nil, tensors: m.tensors, backendLayout: layout,
                kvQuantization: quantization)
            #expect(throws: CBv2CompleteCheckpointError.invalidManifest) { try invalid.validateStructure() }
        }
    }
}
