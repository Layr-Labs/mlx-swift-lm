import Foundation
import Testing

@testable import Qwen38FastCore

@Suite("Qwen 3.8 receipt provenance")
struct Qwen38ReceiptProvenanceTests {
    @Test("artifact provenance encodes identity without a local path")
    func artifactIdentityIsPortable() throws {
        let artifact = Qwen38ReceiptArtifact(
            repository: "owner/model",
            revision: "0123456789abcdef",
            configSHA256: String(repeating: "a", count: 64))
        let encoded = try JSONEncoder().encode(artifact)
        let text = String(decoding: encoded, as: UTF8.self)
        let decoded = try JSONDecoder().decode(Qwen38ReceiptArtifact.self, from: encoded)

        #expect(decoded.repository == "owner/model")
        #expect(decoded.revision == "0123456789abcdef")
        #expect(!text.contains("/Users/example"))
    }

    @Test("token digest is stable and order-sensitive")
    func tokenDigest() {
        #expect(
            Qwen38TokenDigest.sha256([1, 2, 3])
                == "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d")
        #expect(Qwen38TokenDigest.sha256([1, 2, 3]) != Qwen38TokenDigest.sha256([3, 2, 1]))
    }
}
