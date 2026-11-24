import Foundation
import Testing

@testable import MLXLMCommon

/// Manifests of long text prompts: their prefix tokens must fit the 1 MiB
/// encrypted manifest bound the store applies.
@Suite("Complete checkpoint long prefixes")
struct CompleteCheckpointLongPrefixTests {
    /// Inkling 2 Mini's historical recurrent layout, its 48 cache slots and the
    /// DSpark drafter's history, at `tokens.count`.
    func manifest(tokens: [Int]) throws -> CBv2CompleteCheckpointManifest {
        let position = tokens.count
        var tensors: [CBv2CheckpointTensorDescriptor] = []
        for layer in 0 ..< 48 {
            let rows = layer % 6 == 5 ? position : 1024
            for role in [CBv2CheckpointTensorRole.keys, .values] {
                tensors.append(
                    try .init(role: role, layer: layer, shape: [1, 8, rows, 128], dtype: .bfloat16))
            }
            tensors.append(
                try .init(role: .convolution, layer: layer, shape: [1, 3, 2048], dtype: .bfloat16))
            tensors.append(
                try .init(role: .recurrent, layer: layer, shape: [1, 3, 4608], dtype: .bfloat16))
        }
        return .init(
            identity: .init(
                modelAggregateHash: "inkling", promptContractID: "template", buildID: "build",
                numericsFingerprint: "native"),
            position: position, chunkSize: 2048, prefixTokens: tokens, cacheSalt: "tenant-scope",
            assistantCodecID: "inkling2-dspark-ctx-kv-l3-kv8x128-w1024-f9216-bfloat16-v1",
            tensors: tensors,
            backendLayout: CBv2CompleteCheckpointManifest.pagedHistoricalRecurrentMTPLayout)
    }

    func storageEncoding(_ manifest: CBv2CompleteCheckpointManifest) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(manifest)
    }

    /// Token ids spread over a 201,024-entry vocabulary, as text tokens are.
    func tokens(_ count: Int) -> [Int] {
        (0 ..< count).map { Int((UInt64($0) &* 2_654_435_761) % 201_024) }
    }

    @Test("a 260,096-token prefix fits the manifest bound and round-trips")
    func longPrefixFitsTheBound() throws {
        let source = try manifest(tokens: tokens(260_096))
        let encoded = try storageEncoding(source)
        #expect(encoded.count <= CBv2CompleteCheckpointManifest.maximumEncodedBytes)
        #expect(
            try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: encoded) == source)
    }

    @Test("prefixes up to 65,536 tokens keep the JSON array; longer ones are packed")
    func encodingSwitchesAboveTheThreshold() throws {
        for (count, packed) in [(2048, false), (65_536, false), (65_538, true), (131_072, true)] {
            let source = try manifest(tokens: tokens(count))
            let encoded = try storageEncoding(source)
            let json = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            #expect((json["packedPrefixTokens"] != nil) == packed, "\(count)")
            #expect((json["prefixTokens"] != nil) == !packed, "\(count)")
            #expect(
                try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: encoded)
                    == source,
                "\(count)")
        }
    }

    @Test("each prefix length decodes from its one encoding only")
    func decodingRefusesTheOtherEncoding() throws {
        let long = try manifest(tokens: tokens(65_538))
        var root = try #require(
            JSONSerialization.jsonObject(with: storageEncoding(long)) as? [String: Any])
        root.removeValue(forKey: "packedPrefixTokens")
        root["prefixTokens"] = long.prefixTokens
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(
                CBv2CompleteCheckpointManifest.self,
                from: JSONSerialization.data(withJSONObject: root))
        }

        let shortTokens = tokens(2048)
        var short = try #require(
            JSONSerialization.jsonObject(with: storageEncoding(manifest(tokens: shortTokens)))
                as? [String: Any])
        short.removeValue(forKey: "prefixTokens")
        short["packedPrefixTokens"] = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(CBv2NativeBlockPackedTokens(tokens: shortTokens)))
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(
                CBv2CompleteCheckpointManifest.self,
                from: JSONSerialization.data(withJSONObject: short))
        }
    }
}
