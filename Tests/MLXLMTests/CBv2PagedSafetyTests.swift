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

    private let validSource = """
        namespace cbv2 {
        inline void paged_attention_part_impl() {}
        inline void paged_kv_write_impl() {}
        }
        """

    private func writeResource(root: URL, bundleName: String = "any-package_Target.bundle") throws {
        let bundle = root.appendingPathComponent(bundleName, isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try validSource.write(
            to: bundle.appendingPathComponent("pagedattention.metal"),
            atomically: true,
            encoding: .utf8)
    }

    @Test("package resource pre-JITs GPT part, merge, and write kernels")
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
            try PagedAttentionKernel.runtimeSmokeForTesting(searchRoots: [root])
        }
    }

    @Test("signed-app Resources bundle layout is discovered without a bundle-name special case")
    func signedAppResourceLayout() throws {
        let app = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-resource-layout-\(UUID().uuidString)", isDirectory: true)
        let root = app.appendingPathComponent("Contents/Resources", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: app) }
        try writeResource(root: root)

        #expect(try PagedAttentionResources.loadSource(roots: [root]) == validSource)
    }

    @Test("packaged lookup ignores an unsigned external bundle")
    func packagedLookupCannotEscapeApp() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-resource-adversarial-\(UUID().uuidString)", isDirectory: true)
        let app = base.appendingPathComponent("Darkbloom.app", isDirectory: true)
        let executable = app.appendingPathComponent("Contents/MacOS/darkbloom")
        let sealed = app.appendingPathComponent("Contents/Resources", isDirectory: true)
        let external = base.appendingPathComponent("unsigned-external", isDirectory: true)
        try FileManager.default.createDirectory(
            at: executable.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: sealed, withIntermediateDirectories: true)
        try writeResource(root: external)
        defer { try? FileManager.default.removeItem(at: base) }

        #expect(throws: PagedAttentionResourceError.self) {
            _ = try PagedAttentionResources.loadSourceForCurrentProcess(
                executableURL: executable,
                developmentSearchRoots: [external])
        }

        try writeResource(root: sealed, bundleName: "mlx-swift-lm_MLXLMCommon.bundle")
        #expect(
            try PagedAttentionResources.loadSourceForCurrentProcess(
                executableURL: executable,
                developmentSearchRoots: [external]) == validSource)
    }

    @Test("model-specific smoke covers fused, borrowing, sink, and large-head variants")
    func modelSpecificKernelVariants() throws {
        try PagedAttentionKernel.runtimeSmoke(shapes: [
            .init(
                headDim: 64, kvHeads: 8, queryHeads: 64,
                hasSinks: true, hasWrite: true),
            .init(
                headDim: 512, kvHeads: 2, queryHeads: 16,
                hasSinks: false, hasWrite: true),
            .init(
                headDim: 512, kvHeads: 2, queryHeads: 16,
                hasSinks: false, hasWrite: false),
        ])
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
