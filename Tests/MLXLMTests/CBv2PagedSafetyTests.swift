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

    @Test("symlinked invocation still resolves the sealed app resources")
    func symlinkedInvocationFindsSealedResource() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-resource-symlink-\(UUID().uuidString)", isDirectory: true)
        let app = base.appendingPathComponent("Darkbloom.app", isDirectory: true)
        let executable = app.appendingPathComponent("Contents/MacOS/darkbloom")
        let sealed = app.appendingPathComponent("Contents/Resources", isDirectory: true)
        let binDir = base.appendingPathComponent("bin", isDirectory: true)
        let symlink = binDir.appendingPathComponent("darkbloom")
        try FileManager.default.createDirectory(
            at: executable.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: binDir, withIntermediateDirectories: true)
        // The target must exist on disk for symlink resolution to apply.
        try Data().write(to: executable)
        // Same literal relative destination the installer writes with
        // `ln -sfn` into ~/.darkbloom/bin.
        try FileManager.default.createSymbolicLink(
            atPath: symlink.path,
            withDestinationPath: "../Darkbloom.app/Contents/MacOS/darkbloom")
        try writeResource(root: sealed, bundleName: "mlx-swift-lm_MLXLMCommon.bundle")
        defer { try? FileManager.default.removeItem(at: base) }

        let sealedRoot = PagedAttentionResources.packagedAppResourcesURL(
            executableURL: symlink)
        #expect(
            sealedRoot?.resolvingSymlinksInPath().standardizedFileURL.path
                == sealed.resolvingSymlinksInPath().standardizedFileURL.path)
        // End-to-end: the sealed resource is found and an unsigned external
        // bundle cannot hijack a symlinked launch either.
        let external = base.appendingPathComponent("unsigned-external", isDirectory: true)
        try writeResource(root: external)
        #expect(
            try PagedAttentionResources.loadSourceForCurrentProcess(
                executableURL: symlink,
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

    /// The pre-JIT exists so that no production request pays a Metal JIT on
    /// its first decode. PTOK is a kernel TEMPLATE parameter, so every rung
    /// of `partitionTokenLadder` the sizer can select is a SEPARATE compiled
    /// variant with its own name and its own compile: a smoke that exercises
    /// one rung leaves the rest to be paid by whichever live request first
    /// lands in their context bucket — a TTFT spike, which is exactly what
    /// the smoke is for.
    ///
    /// The expectation is NOT a snapshot of three integers. It is an
    /// independent oracle: every distinct value `partitionTokensForDispatch`
    /// returns over a dense sweep of attended lengths for the shape. Adding
    /// a ladder rung therefore tightens this test with no edit here, and the
    /// `DARKBLOOM_CBV2_PAGED_PTOK_TARGET=0` kill switch — which pins every
    /// dispatch to `partitionTokens` — narrows both sides together instead
    /// of failing.
    @Test("runtime smoke pre-JITs every partition rung the sizer can select")
    func runtimeSmokeCoversEveryPartitionRung() throws {
        // The regression this guards: the smoke used to dispatch one decode
        // at `maxAttendLength: 1`, and the sizer floors that onto the
        // smallest rung — one compiled variant out of the ladder's three.
        #expect(
            PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: 1, batch: 1, kvHeads: 8, headSplits: 1,
                pageSize: CBv2PagedDefaults.pageSize)
                == PagedAttentionKernel.minPartitionTokens)

        let shapes = [
            PagedAttentionKernelSmokeShape(
                headDim: 64, kvHeads: 8, queryHeads: 64,
                hasSinks: true, hasWrite: true),
            // d512/GQA 8 splits each kv head across 4 threadgroups, so the
            // sizer sees a different per-partition threadgroup count.
            PagedAttentionKernelSmokeShape(
                headDim: 512, kvHeads: 2, queryHeads: 16,
                hasSinks: false, hasWrite: true),
            PagedAttentionKernelSmokeShape(
                headDim: 512, kvHeads: 2, queryHeads: 16,
                hasSinks: false, hasWrite: false),
        ]
        let coverage = try PagedAttentionKernel.runtimeSmoke(shapes: shapes)
        #expect(Set(coverage.keys) == Set(shapes))

        for shape in shapes {
            let reachable = reachablePartitionRungs(for: shape)
            #expect(
                reachable.count > 1
                    || PagedAttentionKernel.partitionTargetThreadgroups == 0,
                """
                oracle reached only \(reachable.sorted()) for \
                \(shape.argumentValue) — the coverage assertion is vacuous
                """)
            #expect(
                coverage[shape] == reachable,
                """
                shape \(shape.argumentValue): the smoke compiled PTOK \
                \((coverage[shape] ?? []).sorted()) but the sizer can dispatch \
                \(reachable.sorted()) — every missing rung JITs on a live request
                """)
        }
    }

    /// Every PTOK `partitionTokensForDispatch` can return for a B == 1
    /// decode of `shape`, found by sweeping attended lengths rather than by
    /// reading the ladder: a rung the sizer can never select is not a hole
    /// in the smoke's coverage, and a rung added to the ladder shows up here
    /// on its own.
    private func reachablePartitionRungs(
        for shape: PagedAttentionKernelSmokeShape
    ) -> Set<Int> {
        let gqa = shape.queryHeads / shape.kvHeads
        let splits =
            gqa
            / PagedAttentionKernel.headsPerThreadgroup(
                headDim: shape.headDim, gqa: gqa)
        // A partition never holds more than `partitionTokens` tokens and the
        // sizer never wants more than `partitionTargetThreadgroups` of them,
        // so no attended length past their product can reach a new rung. The
        // target is operator-settable; clamp it the way the smoke's own probe
        // sweep does so both sides bound the search identically.
        let bound =
            PagedAttentionKernel.partitionTokens
            * min(max(1, PagedAttentionKernel.partitionTargetThreadgroups), 1 << 16)
        var rungs: Set<Int> = []
        for length in 1 ... bound {
            rungs.insert(
                PagedAttentionKernel.partitionTokensForDispatch(
                    maxAttendLength: length, batch: 1, kvHeads: shape.kvHeads,
                    headSplits: splits, pageSize: CBv2PagedDefaults.pageSize))
        }
        return rungs
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
