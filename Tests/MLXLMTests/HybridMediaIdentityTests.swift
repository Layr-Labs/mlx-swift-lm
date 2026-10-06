import Foundation
import XCTest
@testable import MLXLMCommon

final class HybridMediaIdentityTests: XCTestCase {
    func testFixedDigestWidthCodableAndTextScopeCompatibility() throws {
        for width in [0, 1, 31, 33, 4096] {
            XCTAssertThrowsError(try CBv2HybridPrefixIdentity(digest: Data(repeating: 0, count: width)))
            let bad = try JSONSerialization.data(withJSONObject: ["digest": Data(repeating: 0, count: width).base64EncodedString()])
            XCTAssertThrowsError(try JSONDecoder().decode(CBv2HybridPrefixIdentity.self, from: bad))
        }
        let media = try CBv2HybridPrefixIdentity(digest: Data(repeating: 0x31, count: 32))
        XCTAssertEqual(try JSONDecoder().decode(CBv2HybridPrefixIdentity.self, from: JSONEncoder().encode(media)), media)
        for salt: String? in [nil, "", "tenant", "tenant|☃"] {
            var request = CBv2Request(id: .init(1), promptTokens: [1, 2], maxTokens: 1, cacheSalt: salt)
            XCTAssertEqual(request.checkpointCacheSalt, salt, "nil identity preserves text bytes")
            request.hybridPrefixIdentity = media
            XCTAssertEqual(request.cacheSalt, salt, "do not overwrite the original authenticated scope")
            XCTAssertNotEqual(request.checkpointCacheSalt, salt)
        }
    }

    func testMediaTenantNilEmptyAndChangedIdentityRemainPartitioned() throws {
        let first = try CBv2HybridPrefixIdentity(digest: Data(repeating: 0x41, count: 32))
        let changed = try CBv2HybridPrefixIdentity(digest: Data(repeating: 0x42, count: 32))
        var scopes = Set<String>()
        for identity in [first, changed] {
            for salt: String? in [nil, "", "tenant-a", "tenant-b"] {
                let request = CBv2Request(id: .init(2), promptTokens: [1, 2], maxTokens: 1,
                    cacheSalt: salt, hybridPrefixIdentity: identity)
                XCTAssertTrue(scopes.insert(try XCTUnwrap(request.checkpointCacheSalt)).inserted)
            }
        }
        XCTAssertEqual(scopes.count, 8)
    }
}
