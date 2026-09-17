import CryptoKit
import Foundation
import XCTest

@testable import MLXLMCommon

/// Packaging evidence only: these tests do not initialize or execute a GPU.
final class Qwen4ExpMetalHeaderTests: XCTestCase {
    func testPackagedPreamblesMatchPinnedMLXBytes() {
        let resources = [
            (Qwen4ExpMetalHeaders.gemm, "cc2597fad25939505da77537ff15e78eba2fccbdf14a0e636fb2d2ac0bb4bb6c"),
            (Qwen4ExpMetalHeaders.quantizedUtils, "36893d020956fec4b63f37855035dfebc317bd06a966de14c6372c9a2971ef28"),
            (Qwen4ExpMetalHeaders.quantized, "d39dae2a27d0352facbe0e07c97af1ba0782a923698ce3931a92417d9e527162"),
        ]
        for (source, expected) in resources {
            let digest = SHA256.hash(data: Data(source.utf8))
                .map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(digest, expected)
        }
    }
}
