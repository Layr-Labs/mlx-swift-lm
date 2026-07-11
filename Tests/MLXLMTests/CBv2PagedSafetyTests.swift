// CBv2PagedSafetyTests.swift
//
// Regression gates for failures that must be caught before paged serving:
// missing/corrupt SwiftPM resources, Metal maxBufferLength violations, and
// hostile size arithmetic.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2 paged safety", .serialized)
struct CBv2PagedSafetyTests {
    private func kind(
        attention: CBv2LayerKind.Attention = .full
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: attention,
            headDim: 64,
            kvHeads: 2,
            queryHeads: 4)
    }

    @Test("package resource is readable and its minimal Metal kernel executes")
    func packagedResourceKernelSmoke() throws {
        try PagedAttentionKernel.validateRuntimeResources()
        try PagedAttentionKernel.runtimeSmoke()
    }

    @Test("missing package layout throws instead of trapping in Bundle.module")
    func missingBundleIsCatchable() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-resource-missing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(throws: PagedAttentionResourceError.self) {
            _ = try PagedAttentionResources.loadSource(roots: [root])
        }
    }

    @Test("signed-app Resources bundle layout is discovered without a bundle-name special case")
    func signedAppResourceLayout() throws {
        let app = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-resource-layout-\(UUID().uuidString)", isDirectory: true)
        let root = app.appendingPathComponent("Contents/Resources", isDirectory: true)
        let bundle = root.appendingPathComponent("any-package_Target.bundle", isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: app) }
        let source = """
            namespace cbv2 {
            inline void paged_attention_part_impl() {}
            inline void paged_kv_write_impl() {}
            }
            """
        try source.write(
            to: bundle.appendingPathComponent("pagedattention.metal"),
            atomically: true,
            encoding: .utf8)

        #expect(try PagedAttentionResources.loadSource(roots: [root]) == source)
    }

    @Test("each slab is rejected before allocation when it exceeds Metal maxBufferLength")
    func maxBufferLengthPreflight() {
        let pageBytes = 2 * 2 * 16 * 64 * 2
        let pageCount = 16
        let oneSlabBytes = pageBytes * pageCount / 2
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVPool(
                layerKinds: [kind()],
                config: PagedKVPoolConfig(
                    capacityBytes: pageBytes * pageCount,
                    nominalMaxSequenceLength: 256,
                    maxBufferLength: oneSlabBytes - 1))
        }
    }

    @Test("overflowing ring and demand sizes fail catchably before MLX allocation")
    func hostileSizesFailCatchably() {
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVPool(
                layerKinds: [kind(attention: .slidingWindow(1))],
                config: PagedKVPoolConfig(
                    capacityBytes: 1 << 20,
                    maxPrefillChunk: Int.max,
                    nominalMaxSequenceLength: 1024,
                    maxBufferLength: Int.max))
        }
    }
}
