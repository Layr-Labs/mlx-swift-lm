// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MLX compiles one Metal library per distinct custom-kernel template tuple.
/// GLM decode kernels must therefore read history extents at run time; these
/// tests pin that and compare against the former history-specialized kernels.
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextKernelSpecializationTests {
        private static let histories = [
            1, 2, 3, 4, 5, 7, 8, 9, 31, 127, 128, 129, 2047, 2048, 2049,
        ]

        /// The tuple MLX renders into `custom_kernel_<name>_<values>`.
        private static func rendered(_ template: [(String, any KernelTemplateArg)]) -> String {
            template.map { "\($0.0)=\($0.1)" }.joined(separator: ",")
        }

        private func fence(_ arrays: [MLXArray]) throws {
            try MLX.withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        @Test func selectorTemplatesAndLayoutsAreHistoryIndependent() throws {
            for budget in [4, 64, 2048] {
                let requested = budget / 4
                let reference = Self.rendered(
                    GLM5NextSparseSelection.referenceTemplate(requestedCount: requested))
                #expect(reference == "Requested=\(requested)")
                #expect(
                    Self.rendered(
                        GLM5NextSparseSelection.parallelTemplate(requestedCount: requested))
                        == "Count=\(requested),Threads=\(GLM5NextSparseSelection.parallelThreads)")
                for history in Self.histories {
                    let pools = (history + 3) / 4
                    // selectTokens now passes the budget; the former call
                    // passed min(budget/4, pools). Both give one layout.
                    for batch in [1, 3] {
                        let current = try GLM5NextSparseSelection.Layout(
                            shape: [batch, pools], requestedCount: requested)
                        let former = try GLM5NextSparseSelection.Layout(
                            shape: [batch, pools], requestedCount: min(requested, pools))
                        #expect(current.count == former.count)
                        #expect(current.scratchWidth == former.scratchWidth)
                        #expect(current.scratchBytes == former.scratchBytes)
                    }
                }
            }
            #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
                let source = GLM5NextSparseSelection.referenceKernelSource
                #expect(source.hasSuffix(GLM5NextSparseSelection.referenceSource))
                for extent in [
                    "const int Rows = scores_shape[0];", "const int Pools = scores_shape[1];",
                    "const int Count = metal::min(Requested, Pools);",
                    "const int Scratch = metal::min(Pools, 64 * Count - 1);",
                ] {
                    #expect(source.contains(extent))
                }
            #endif
        }

        @Test func pagedReadTemplatesCarryNoHistoryLength() {
            let first = Self.rendered(
                GLM5NextPagedTransfers.selectedLatentTemplate(
                    dtype: .float32, headDim: 512, pageSize: 16, pages: 0 ..< 64))
            #expect(first == "T=float32,D=512,S=16,FIRST=0,END=64")
            let names = GLM5NextPagedTransfers.selectedLatentTemplate(
                dtype: .float32, headDim: 512, pageSize: 16, pages: 64 ..< 96
            ).map(\.0)
            #expect(names == ["T", "D", "S", "FIRST", "END"])
        }

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            /// The former selector: the verbatim body specialized on every extent.
            private static let historySpecializedSelector = MLXFast.metalKernel(
                name: "glm5_next_reference_sparse_selection_history_specialized",
                inputNames: ["scores", "complete"], outputNames: ["selection"],
                source: GLM5NextSparseSelection.referenceSource,
                header: GLM5NextSparseSelection.referenceHeader)

            /// The former `.hostRecords` selected-latent read, specialized on N.
            private static let historySpecializedLatentRead = MLXFast.metalKernel(
                name: "glm5_next_read_selected_latents_history_specialized",
                inputNames: ["storage", "table", "indices", "output", "previous"],
                outputNames: ["fence"],
                source: """
                    const int d = int(thread_position_in_grid.x);
                    const int r = int(thread_position_in_grid.y);
                    const int token = indices[r];
                    if (token >= 0 && token < N) {
                        const int page = table[token / S];
                        if (page > FIRST && page < END) {
                            const size_t source = ((size_t)(page - FIRST) * S + token % S) * D;
                            device T* dst = output;
                            dst[(size_t)r * D + d] = storage[source + d];
                        }
                    }
                    if (d == 0 && r == 0) fence[0] = previous[0] + 1;
                    """, ensureRowContiguous: true, mutableInputs: ["output"])

            private struct Generator {
                var state: UInt64
                mutating func next() -> UInt64 {
                    state &+= 0x9E37_79B9_7F4A_7C15
                    var z = state
                    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                    return z ^ (z >> 31)
                }
                mutating func int(_ bound: Int) -> Int { Int(next() % UInt64(bound)) }
                mutating func score() -> Float {
                    switch int(6) {
                    case 0: return [0, -0.0, .nan, .infinity, -.infinity][int(5)]
                    case 1: return -Float.greatestFiniteMagnitude
                    case 2: return Float(int(4))
                    default:
                        return Float(bitPattern: UInt32(truncatingIfNeeded: next()) & 0xBF7F_FFFF)
                    }
                }
            }

            /// Raw selector output (incl. order) equals the former specialized
            /// kernel for growing histories, batches, budgets and tie patterns.
            @Test func referenceSelectorMatchesHistorySpecializedKernelBitwise() throws {
                try Device.withDefaultDevice(Device(.gpu)) { () throws -> Void in
                    var generator = Generator(state: 0x5EED_0001)
                    let pools = [
                        1, 2, 3, 5, 7, 8, 9, 31, 127, 128, 129, 511, 512, 513, 2047, 8191, 32768,
                    ]
                    for budget in [4, 8, 64, 2048] {
                        for pool in pools {
                            let batch = 1 + generator.int(3)
                            let values = (0 ..< batch * pool).map { _ in generator.score() }
                            let complete = (0 ..< batch * pool).map { _ in generator.int(8) != 0 }
                            let scores = MLXArray(values, [batch, pool])
                            let mask = MLXArray(complete, [batch, pool])
                            let current = try GLM5NextSparseSelection.select(
                                scores: scores, completePools: mask, requestedCount: budget / 4)
                            let layout = try GLM5NextSparseSelection.Layout(
                                shape: [batch, pool], requestedCount: min(budget / 4, pool))
                            let former = Self.historySpecializedSelector(
                                [scores, mask],
                                // The shared body reads the completeness mask's row
                                // count (one row per selection row here).
                                template: [
                                    ("Rows", layout.batch), ("MaskRows", layout.batch),
                                    ("Pools", layout.pools), ("Count", layout.count),
                                    ("Scratch", layout.scratchWidth),
                                ],
                                grid: (1, layout.batch, 1), threadGroup: (1, 1, 1),
                                outputShapes: [[layout.batch, layout.scratchWidth]],
                                outputDTypes: [.int32])[0][0..., ..<layout.count]
                            try fence([current, former])
                            #expect(
                                current.shape == former.shape, "B=\(batch) P=\(pool) K=\(budget)")
                            #expect(
                                current.asArray(Int32.self) == former.asArray(Int32.self),
                                "B=\(batch) P=\(pool) K=\(budget)")
                        }
                    }
                }
            }

            /// The run-time history length in the table's last entry gates
            /// exactly the ids the former `N` template gated.
            @Test(arguments: [DType.float32, .float16])
            func selectedLatentReadMatchesHistorySpecializedKernelBitwise(dtype: DType) throws {
                try Device.withDefaultDevice(Device(.gpu)) { () throws -> Void in
                    let pageSize = 4
                    let width = 8
                    let segment = 3 ..< 9
                    let storage = MLXArray(
                        (0 ..< segment.count * pageSize * width).map { Float($0) * 0.37 - 11 }
                    ).asType(dtype)
                    // In-segment, poison (FIRST), out-of-segment and END pages.
                    let cycle: [Int32] = [4, 5, 3, 6, 10, 7, 2, 8, 9, 5]
                    for history in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 25, 28, 29, 40] {
                        let pages = (0 ..< (history + pageSize - 1) / pageSize).map {
                            cycle[$0 % cycle.count]
                        }
                        let template = GLM5NextPagedTransfers.selectedLatentTemplate(
                            dtype: dtype, headDim: width, pageSize: pageSize, pages: segment)
                        let n = Int32(history)
                        for ids in [
                            [-1, 0, n - 1, n, n + 17, n / 2, 2 * n, -5, 1, n - 2, 3],
                            [n - 1, 0, n, -1, n / 3, 1, n + 1],
                        ] {
                            let indices = MLXArray(ids)
                            let previous = MLXArray([Int32(41)])
                            let currentOutput = MLXArray.zeros([1, ids.count, width], dtype: dtype)
                            let formerOutput = MLXArray.zeros([1, ids.count, width], dtype: dtype)
                            let currentFence = GLM5NextPagedTransfers.selectedLatents(
                                storage: storage, table: MLXArray(pages + [n]), indices: indices,
                                output: currentOutput, previous: previous, template: template)
                            let formerFence = Self.historySpecializedLatentRead(
                                [storage, MLXArray(pages), indices, formerOutput, previous],
                                template: [
                                    ("T", dtype), ("D", width), ("S", pageSize), ("N", history),
                                    ("FIRST", segment.lowerBound), ("END", segment.upperBound),
                                ],
                                grid: (width, ids.count, 1), threadGroup: (width, 1, 1),
                                outputShapes: [[1]], outputDTypes: [.int32])[0]
                            let current = depends(
                                input: currentOutput, dependencies: [currentFence])
                            let former = depends(input: formerOutput, dependencies: [formerFence])
                            try fence([current, former, currentFence, formerFence])
                            #expect(
                                currentFence.asArray(Int32.self) == formerFence.asArray(Int32.self))
                            #expect(
                                current.asType(.float32).asArray(Float.self).map(\.bitPattern)
                                    == former.asType(.float32).asArray(Float.self).map(\.bitPattern),
                                "N=\(history) W=\(ids.count) \(dtype)")
                        }
                    }
                }
            }
        #endif
    }
}
