import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// Tests of the process-wide cache of the paged-attention Metal source.
    @Suite
    struct PagedAttentionResourcesCacheTests {

        private let source = """
            namespace cbv2 {
            inline void paged_attention_part_impl() {}
            inline void paged_kv_write_impl() {}
            }
            """

        /// The second lookup with the same inputs does not search: the
        /// resource is deleted after the first lookup, and a search would
        /// then throw `.missing`.
        @Test func secondLookupDoesNotSearchAgain() throws {
            let base = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "paged-resource-cache-\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: base) }
            let executable = base.appendingPathComponent("bin/bench-worker")
            let root = base.appendingPathComponent("root", isDirectory: true)
            let bundle = root.appendingPathComponent(
                "mlx-swift-lm_MLXLMCommon.bundle", isDirectory: true)
            try FileManager.default.createDirectory(
                at: bundle, withIntermediateDirectories: true)
            try source.write(
                to: bundle.appendingPathComponent("pagedattention.metal"),
                atomically: true,
                encoding: .utf8)

            #expect(
                try PagedAttentionResources.loadSourceForCurrentProcess(
                    executableURL: executable,
                    developmentSearchRoots: [root]) == source)

            try FileManager.default.removeItem(at: root)
            #expect(throws: PagedAttentionResourceError.self) {
                try PagedAttentionResources.loadSource(roots: [root])
            }
            #expect(
                try PagedAttentionResources.loadSourceForCurrentProcess(
                    executableURL: executable,
                    developmentSearchRoots: [root]) == source)
        }
    }
}
