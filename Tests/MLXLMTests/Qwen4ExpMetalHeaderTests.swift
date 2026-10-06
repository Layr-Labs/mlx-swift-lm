import CryptoKit
import Foundation
import XCTest

@testable import MLXLMCommon

/// Packaging evidence only: these tests do not initialize or execute a GPU.
final class Qwen4ExpMetalHeaderTests: XCTestCase {
    func testPackagedLookupCannotUseADeveloperFallback() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = root.appendingPathComponent("Probe.app/Contents/Resources")
        let developer = root.appendingPathComponent("developer")
        let relative = "mlx-swift-lm_MLXLMCommon.bundle/Qwen4Metal/gemm.metal"
        for directory in [resources, developer] {
            let file = directory.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try "same kernel".write(to: file, atomically: true, encoding: .utf8)
        }
        let executable = root.appendingPathComponent("Probe.app/Contents/MacOS/probe")
        XCTAssertEqual(try Qwen4ExpMetalResources.load("gemm", executableURL: executable,
            developmentSearchRoots: [developer]), "same kernel")
        try FileManager.default.removeItem(at: resources.appendingPathComponent(relative))
        XCTAssertThrowsError(try Qwen4ExpMetalResources.load("gemm", executableURL: executable,
            developmentSearchRoots: [developer]))
    }

    func testDevelopmentCopiesMustAgree() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let directories = [root.appendingPathComponent("a"), root.appendingPathComponent("b")]
        for (index, directory) in directories.enumerated() {
            let file = directory.appendingPathComponent("mlx-swift-lm_MLXLMCommon.bundle/Qwen4Metal/gemm.metal")
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try "kernel \(index)".write(to: file, atomically: true, encoding: .utf8)
        }
        XCTAssertThrowsError(try Qwen4ExpMetalResources.load("gemm", executableURL: nil,
            developmentSearchRoots: directories))
    }

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
