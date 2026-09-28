import Foundation
import Testing
@testable import MLXLMCommon

@Suite("Native block checkpoint metadata")
struct NativeBlockCheckpointMetadataTests {
    func manifest(tokens: [Int], layout: String = CBv2CompleteCheckpointManifest.diffusionBlockLayout) throws -> CBv2CompleteCheckpointManifest {
        .init(identity: .init(modelAggregateHash: "model", promptContractID: "template", buildID: "build", numericsFingerprint: "numerics"),
              position: tokens.count, chunkSize: 2, prefixTokens: tokens, cacheSalt: "scope", assistantCodecID: nil,
              tensors: try (0..<30).flatMap { layer in
                try [CBv2CheckpointTensorRole.keys, .values].map {
                    try CBv2CheckpointTensorDescriptor(role: $0, layer: layer, shape: [1, 2, tokens.count, 512], dtype: .bfloat16)
                }
              }, backendLayout: layout,
              nativeBlockState: layout == CBv2CompleteCheckpointManifest.diffusionBlockLayout
                ? .init(windowPhysicalLength: 1024, windowCursor: 1) : nil)
    }

    @Test func fullNativeContextFitsExistingEncryptedEnvelopeLosslessly() throws {
        // Highest token needs19 bits; native capacity remains262144.
        let tokens = (0..<262144).map { ($0 * 11) % 262208 }
        let source = try manifest(tokens: tokens)
        _ = try source.validateStructure()
        let encoded = try JSONEncoder().encode(source)
        #expect(encoded.count < CBv2CompleteCheckpointManifest.maximumEncodedBytes)
        #expect(try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: encoded) == source)
        let json = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(json["prefixTokens"] == nil && json["packedPrefixTokens"] != nil)
    }

    @Test func legacyLayoutStillUsesItsOriginalTokenEncoding() throws {
        let source = try manifest(tokens: [1, 262207], layout: CBv2CompleteCheckpointManifest.layout)
        let encoded = try JSONEncoder().encode(source)
        let json = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(json["prefixTokens"] as? [Int] == source.prefixTokens)
        #expect(json["packedPrefixTokens"] == nil && json["nativeBlockState"] == nil)
        #expect(try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: encoded) == source)
    }

    @Test func rejectsMalformedOrAmbiguousPackedTokensAndWrongLayoutState() throws {
        let encoded = try JSONEncoder().encode(manifest(tokens: [0, 3]))
        let root = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let mutations: [[String: Any]] = [
            ["bitWidth": 0, "bytes": "AA=="], ["bitWidth": 32, "bytes": "AA=="],
            ["bitWidth": 2, "bytes": ""], ["bitWidth": 2, "bytes": "/A=="],
            ["bitWidth": 3, "bytes": "GA=="], // non-minimal representation
        ]
        for malformed in mutations {
            var value = root; value["packedPrefixTokens"] = malformed
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONSerialization.data(withJSONObject: value))
            }
        }
        var ambiguous = root; ambiguous["prefixTokens"] = [0, 3]
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONSerialization.data(withJSONObject: ambiguous))
        }
        var wrong = root
        wrong["backendLayout"] = CBv2CompleteCheckpointManifest.layout
        wrong.removeValue(forKey: "packedPrefixTokens"); wrong["prefixTokens"] = [0, 3]
        let legacy = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONSerialization.data(withJSONObject: wrong))
        #expect(throws: CBv2CompleteCheckpointError.invalidManifest) { try legacy.validateStructure() }
    }

    @Test func everyWidthAndTailRoundTripsExactly() throws {
        for width in 1...31 {
            for count in 2...19 {
                let tokens = (0..<count).map { index in index % 2 == 0 ? (1 << (width - 1)) : 0 }
                let packed = try CBv2NativeBlockPackedTokens(tokens: tokens)
                #expect(packed.bitWidth == width)
                #expect(try packed.unpack(count: count) == tokens)
            }
        }
    }

    @Test func nativeMetadataRebindsCapacityForEachDistinctEngineOwner() async throws {
        func engine() throws -> CBv2NativeBlockEngine {
            try .init(tokenizer: TestTokenizer(vocabularySize: 128), kvBytesCapacity: 32 << 20,
                      reservationForRequest: { _ in 1024 },
                      makeSession: { _, _ in throw CBv2NativeBlockError.unsupportedRequest("unused") })
        }
        let first = try engine(), second = try engine()
        let original = try manifest(tokens: [0, 3])
        var a: CBv2CompleteCheckpointManifest? = try original.owningNativeMetadata(engine: first)
        var alias: CBv2CompleteCheckpointManifest? = try a!.owningNativeMetadata(engine: first)
        var b: CBv2CompleteCheckpointManifest? = try a!.owningNativeMetadata(engine: second)
        #expect(a!.metadata === alias!.metadata)
        #expect(a!.metadata !== b!.metadata)
        #expect(a!.metadata.permit!.nativeOwner != b!.metadata.permit!.nativeOwner)
        #expect(first.capacity().kvBytesReserved == second.capacity().kvBytesReserved)
        a = nil
        #expect(first.capacity().kvBytesReserved > 0)
        alias = nil
        #expect(first.capacity().kvBytesReserved == 0 && second.capacity().kvBytesReserved > 0)
        b = nil
        #expect(second.capacity().kvBytesReserved == 0)
        await first.shutdown(); await second.shutdown()
    }
}
