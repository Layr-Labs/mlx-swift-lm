import Foundation
import XCTest
@testable import MLXLMCommon

final class CBv2HiddenResourceTests: XCTestCase {
    private let source = "namespace cbv2 { void paged_attention_part_impl() {} void paged_kv_write_impl() {} }"

    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("hidden-paged-resource-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }

    private func write(_ text: String, root: URL, name: String) throws -> URL {
        let bundle = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try text.write(to: bundle.appendingPathComponent("pagedattention.metal"), atomically: true, encoding: .utf8)
        return bundle
    }

    func testDotNamedResourceBundleIsDiscoverable() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try write(source, root: root, name: ".generated_Target.bundle")
        XCTAssertEqual(try PagedAttentionResources.loadSource(roots: [root]), source)
    }

    #if os(macOS)
    func testFilesystemHiddenSwiftPMBundleIsDiscoverable() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        var bundle = try write(source, root: root, name: "generated_Target.bundle")
        var values = URLResourceValues()
        values.isHidden = true
        try bundle.setResourceValues(values)
        XCTAssertEqual(try bundle.resourceValues(forKeys: [.isHiddenKey]).isHidden, true)
        XCTAssertEqual(try PagedAttentionResources.loadSource(roots: [root]), source)
    }
    #endif

    func testHiddenConflictingResourceStillFailsClosed() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try write(source, root: root, name: "visible_Target.bundle")
        _ = try write(source + "\n// different bytes", root: root, name: ".hidden_Target.bundle")
        XCTAssertThrowsError(try PagedAttentionResources.loadSource(roots: [root])) { error in
            guard case PagedAttentionResourceError.ambiguous = error else {
                return XCTFail("Expected the original ambiguity refusal, got \(error)")
            }
        }
    }

    func testIdenticalHiddenAndVisibleResourcesDeduplicate() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try write(source, root: root, name: "visible_Target.bundle")
        _ = try write(source, root: root, name: ".hidden_Target.bundle")
        XCTAssertEqual(try PagedAttentionResources.loadSource(roots: [root]), source)
    }
}
