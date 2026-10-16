// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// `GLM5NextSparsePrefillKernel.attend` against the gathered absorbed
    /// reference (`GLM5NextDSAAttention.absorbedAttention`'s block math) on
    /// deterministic BF16-valued synthetic inputs at real DSA geometry. Not
    /// real-checkpoint evidence; the benchmark is env-gated
    /// (`GLM53_SPARSE_PREFILL_BENCH=1`).
    @Suite(.serialized)
    struct GLM5NextSparsePrefillKernelTests {
        private static let width = 2051

        /// `[B,T,W]` ids shaped like `selectTokens` rows: ascending members,
        /// -1 static padding, a 3-slot tail. Row 1 of each batch has
        /// `leading` invalid slots (whole empty tiles), one query is fully
        /// masked and every 9th slot is -1.
        private static func indices(
            batch: Int, queries: Int, history: Int, width: Int, members: Int, leading: Int,
            seed: UInt64
        ) -> MLXArray {
            var state = seed
            func next() -> Int {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Int(state >> 33)
            }
            var ids = [Int32]()
            ids.reserveCapacity(batch * queries * width)
            for b in 0 ..< batch {
                for t in 0 ..< queries {
                    var row = [Int32](repeating: -1, count: width)
                    let fullyMasked = t == queries / 2 && (b == 0 || queries == 1)
                    let start = t == 1 ? min(leading, members) : 0
                    for slot in start ..< members where !fullyMasked && slot % 9 != 4 {
                        row[slot] = Int32(next() % history)
                    }
                    for slot in (width - 3) ..< width where !fullyMasked && slot != width - 1 {
                        row[slot] = Int32(next() % history)
                    }
                    ids += row
                }
            }
            return MLXArray(ids, [batch, queries, width])
        }

        /// BF16-valued normal samples.
        private static func normal(_ shape: [Int], scale: Float, seed: UInt64) -> MLXArray {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(.bfloat16)
        }

        /// `absorbedAttention`'s gathered block math: FP32 latents and scores,
        /// masked softmax, probabilities rounded to `roundingType`, FP32 PV,
        /// one rounding to `outputType`. Returns `[B,H,T,C]`.
        static func gatheredAttention(
            queryLatents: MLXArray, latents: MLXArray, indices: MLXArray, scale: Float,
            roundingType: DType, outputType: DType, blockBytes: Int = 64 << 20
        ) -> MLXArray {
            let (batch, count, width) = (indices.dim(0), indices.dim(1), indices.dim(2))
            let dimension = latents.dim(2)
            let blockQueries = max(1, blockBytes / (batch * width * dimension * 4))
            var blocks: [MLXArray] = []
            for start in stride(from: 0, to: count, by: blockQueries) {
                let queries = start ..< min(start + blockQueries, count)
                let blockIds = indices[0..., queries]
                let selected = takeAlong(
                    latents, clip(blockIds, min: 0, max: latents.dim(1) - 1).reshaped(batch, -1, 1),
                    axis: 1
                ).reshaped(blockIds.shape + [dimension]).asType(.float32)
                let score =
                    matmul(
                        queryLatents[0..., 0..., queries].transposed(0, 2, 1, 3).asType(.float32),
                        selected.transposed(0, 1, 3, 2)) * scale
                let mask = (blockIds .>= 0).expandedDimensions(axis: 2)
                let masked = MLX.where(mask, score, -Float.infinity)
                let maximum = MLX.where(
                    mask.any(axis: -1, keepDims: true), masked.max(axis: -1, keepDims: true), 0)
                let exps = MLX.where(mask, exp(masked - maximum), 0)
                let total = exps.sum(axis: -1, keepDims: true)
                let probability = (exps / MLX.where(total .> 0, total, 1)).asType(roundingType)
                blocks.append(matmul(probability.asType(.float32), selected).asType(outputType))
            }
            let latentsOut = blocks.count == 1 ? blocks[0] : concatenated(blocks, axis: 1)
            return latentsOut.transposed(0, 2, 1, 3)
        }

        private struct Case: CustomStringConvertible {
            let batch: Int
            let heads: Int
            let dimension: Int
            let queries: Int
            let history: Int
            let width: Int
            let members: Int
            let leading: Int
            let latentType: DType
            var description: String {
                "B=\(batch) H=\(heads) C=\(dimension) T=\(queries) N=\(history) W=\(width) "
                    + "members=\(members) leading=\(leading) latents=\(latentType)"
            }
        }

        private func check(_ c: Case, tensorUnits: Bool) throws {
            let seed = UInt64(c.history * 131 + c.queries * 7 + c.heads)
            let queryLatents = Self.normal(
                [c.batch, c.heads, c.queries, c.dimension], scale: 2, seed: seed)
            let stored = Self.normal([c.batch, c.history, c.dimension], scale: 1, seed: seed + 1)
            let latents = stored.asType(c.latentType)
            let indices = Self.indices(
                batch: c.batch, queries: c.queries, history: c.history, width: c.width,
                members: c.members, leading: c.leading, seed: seed + 2)
            let scale: Float = 1 / 16
            let actual = try GLM5NextSparsePrefillKernel.attend(
                queryLatents: queryLatents, latents: latents, indices: indices, scale: scale,
                outputType: .bfloat16, tensorUnits: tensorUnits)
            let expected = Self.gatheredAttention(
                queryLatents: queryLatents, latents: latents, indices: indices, scale: scale,
                roundingType: .bfloat16, outputType: .bfloat16)
            #expect(actual.shape == expected.shape && actual.dtype == .bfloat16, "\(c)")
            let a = actual.asType(.float32)
            let e = expected.asType(.float32)
            let worst = abs(a - e).max().item(Float.self)
            let peak = abs(e).max().item(Float.self)
            let finite = ((a - a) .== 0).all().item(Bool.self)
            let path =
                tensorUnits
                && GLM5NextSparsePrefillKernel.usesTensorUnits(
                    queryLatents: queryLatents, latents: latents)
                ? "tensor" : "portable"
            print(
                "GLM53_SPARSE_PREFILL attention \(c) path=\(path) max_abs_error=\(worst) peak=\(peak)"
            )
            #expect(finite, "\(c)")
            // Two BF16 output ULPs at the peak, as the split-K decode bound.
            #expect(worst <= peak / 128, "\(c): \(worst) > \(peak / 128)")
            // The fully masked query attends to nothing.
            let masked = actual[0, 0..., c.queries / 2].asType(.float32)
            #expect(abs(masked).max().item(Float.self) == 0, "\(c): masked row")
        }

        @Test func tensorKernelMatchesGatheredReferenceAtRealGeometry() throws {
            try Device.withDefaultDevice(.gpu) {
                for c in [
                    Case(
                        batch: 1, heads: 64, dimension: 512, queries: 512, history: 10000,
                        width: Self.width, members: 2048, leading: 1000, latentType: .bfloat16),
                    Case(
                        batch: 2, heads: 64, dimension: 512, queries: 37, history: 4099,
                        width: Self.width, members: 1700, leading: 2051, latentType: .float16),
                    Case(
                        batch: 1, heads: 32, dimension: 512, queries: 9, history: 300,
                        width: 131, members: 100, leading: 128, latentType: .bfloat16),
                ] {
                    try check(c, tensorUnits: true)
                }
            }
        }

        @Test func portablePathMatchesGatheredReference() throws {
            try Device.withDefaultDevice(.gpu) {
                for c in [
                    Case(
                        batch: 1, heads: 64, dimension: 512, queries: 40, history: 5000,
                        width: Self.width, members: 2048, leading: 700, latentType: .bfloat16),
                    // FP32 latents and geometry outside the tensor kernel run the
                    // portable path.
                    Case(
                        batch: 2, heads: 64, dimension: 512, queries: 11, history: 3000,
                        width: Self.width, members: 2048, leading: 300, latentType: .float32),
                    Case(
                        batch: 2, heads: 3, dimension: 40, queries: 12, history: 90,
                        width: 37, members: 30, leading: 5, latentType: .bfloat16),
                ] {
                    try check(c, tensorUnits: false)
                    try check(c, tensorUnits: true)
                }
            }
        }

        /// Inputs of one blocked-selection score problem: `[B,Q,H,D]` queries,
        /// `[B,Q,H]` weights, `[B,P,D]` pooled keys, a `[B,P]` complete mask
        /// with `padding[b]` leading incomplete pools and every 13th pool of
        /// row 0 incomplete, and pool ends anchored at `4 * padding[b]`.
        private struct ScoreProblem {
            let indexQueries: MLXArray
            let headWeights: MLXArray
            let pooledKeys: MLXArray
            let complete: MLXArray
            let poolEnds: MLXArray
            let queryStart: Int

            init(batch: Int, queries: Int, pools: Int, heads: Int = 32, padding: [Int], seed: UInt64)
            {
                indexQueries = GLM5NextSparsePrefillKernelTests.normal(
                    [batch, queries, heads, 128], scale: 1, seed: seed)
                headWeights = GLM5NextSparsePrefillKernelTests.normal(
                    [batch, queries, heads], scale: 1, seed: seed + 1)
                pooledKeys = GLM5NextSparsePrefillKernelTests.normal(
                    [batch, pools, 128], scale: 1, seed: seed + 2)
                var complete = [Bool]()
                var ends = [Int32]()
                for b in 0 ..< batch {
                    for p in 0 ..< pools {
                        complete.append(p >= padding[b] && !(b == 0 && p % 13 == 7))
                        ends.append(Int32(4 * padding[b] + 4 * p + 3))
                    }
                }
                self.complete = MLXArray(complete, [batch, pools])
                poolEnds = MLXArray(ends, [batch, pools])
                queryStart = 4 * pools - queries
            }

            /// `blockedSelection`'s FP32 score graph.
            func reference() -> MLXArray {
                let (batch, queries, heads) = (
                    indexQueries.dim(0), indexQueries.dim(1), indexQueries.dim(2)
                )
                let pools = pooledKeys.dim(1)
                let q = indexQueries.asType(.float32).reshaped([batch, queries * heads, 128])
                let perHead = maximum(
                    matmul(q, pooledKeys.asType(.float32).transposed(0, 2, 1)) * (1 / sqrt(128)),
                    0
                ).reshaped([batch, queries, heads, pools])
                let weights = headWeights.asType(.float32) * (1 / sqrt(Float(heads)))
                let scores = (perHead * weights.expandedDimensions(axis: -1)).sum(axis: 2)
                let positions = (MLXArray.arange(queries, dtype: .int32) + Int32(queryStart))
                    .reshaped([1, queries, 1])
                let eligible =
                    complete.expandedDimensions(axis: 1)
                    .&& (poolEnds.expandedDimensions(axis: 1) .<= positions)
                return MLX.where(eligible, scores, -Float.greatestFiniteMagnitude)
            }

            func native() throws -> MLXArray {
                try GLM5NextPrefillIndexScoreKernel.scores(
                    indexQueries: indexQueries, headWeights: headWeights, pooledKeys: pooledKeys,
                    completePools: complete, poolEnds: poolEnds, queryStart: queryStart)
            }
        }

        @Test func indexScoresMatchBlockedScoreGraph() throws {
            try Device.withDefaultDevice(.gpu) {
                for (batch, queries, pools, padding) in [
                    (1, 300, 2600, [0]), (2, 77, 1031, [0, 40]), (1, 9, 33, [3]),
                ] {
                    let problem = ScoreProblem(
                        batch: batch, queries: queries, pools: pools, padding: padding,
                        seed: UInt64(pools))
                    #expect(
                        GLM5NextPrefillIndexScoreKernel.supports(
                            indexQueries: problem.indexQueries, pooledKeys: problem.pooledKeys))
                    let actual = try problem.native()
                    let expected = problem.reference()
                    let masked = expected .== -Float.greatestFiniteMagnitude
                    #expect(
                        (masked .== (actual .== -Float.greatestFiniteMagnitude)).all().item(
                            Bool.self), "B=\(batch) Q=\(queries) P=\(pools): mask")
                    let live = MLX.where(masked, 0, expected)
                    let peak = abs(live).max().item(Float.self)
                    let worst = abs(MLX.where(masked, 0, actual) - live).max().item(Float.self)
                    print(
                        "GLM53_SPARSE_PREFILL index_scores B=\(batch) Q=\(queries) P=\(pools) "
                            + "max_abs_error=\(worst) peak=\(peak)")
                    #expect(peak > 0)
                    #expect(worst <= 1e-5 * peak, "B=\(batch) Q=\(queries) P=\(pools)")
                }
            }
        }

        /// Blocked selection with the fused scores against the score graph:
        /// identical tails and padding; pool membership may differ only at
        /// near ties of the row's K-th score.
        @Test func nativeScoreSelectionMatchesGraphUpToNearTies() throws {
            try Device.withDefaultDevice(.gpu) {
                let (queries, history, heads) = (40, 10000, 32)
                let indexQueries = Self.normal([1, queries, heads, 128], scale: 1, seed: 21)
                let indexKeys = Self.normal([1, history, 128], scale: 1, seed: 22)
                let poolGates = Self.normal([1, history, 128], scale: 2, seed: 23)
                let positionBias = Self.normal([4, 128], scale: 0.5, seed: 24)
                let headWeights = Self.normal([1, queries, heads], scale: 1, seed: 25)
                let keyValidity = MLXArray((0 ..< history).map { $0 >= 5 }, [1, history])
                let queryValidity = MLXArray.ones([1, queries], dtype: .bool)
                func select(_ native: Bool) throws -> [Int32] {
                    try GLM5NextSparseAttention.selectTokens(
                        indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
                        poolPositionBias: positionBias, headWeights: headWeights,
                        keyValidity: keyValidity, queryValidity: queryValidity,
                        queryStart: history - queries, tokenBudget: 2048,
                        execution: .init(
                            selection: .parallelThreshold, deviceRanges: true, queryBlocks: true),
                        nativePrefillScores: native
                    ).indices.asArray(Int32.self)
                }
                let actual = try select(true)
                let expected = try select(false)
                let width = 2051
                var differing = 0
                for row in 0 ..< queries {
                    let a = Array(actual[(row * width) ..< ((row + 1) * width)])
                    let e = Array(expected[(row * width) ..< ((row + 1) * width)])
                    #expect(Array(a[2048...]) == Array(e[2048...]), "row \(row): tail")
                    let pools = Set(a[..<2048].filter { $0 >= 0 }).symmetricDifference(
                        Set(e[..<2048].filter { $0 >= 0 }))
                    differing = max(differing, pools.count / 4)
                }
                print("GLM53_SPARSE_PREFILL selection max_differing_pools=\(differing)")
                #expect(differing <= 4)
            }
        }

        @Test func cpuStreamIsRefused() {
            Device.withDefaultDevice(.cpu) {
                #expect(throws: GLM5NextSparsePrefillKernel.Failure.unsupportedDeviceOrStream) {
                    try GLM5NextSparsePrefillKernel.attend(
                        queryLatents: MLXArray.zeros([1, 32, 16, 512], dtype: .bfloat16),
                        latents: MLXArray.zeros([1, 64, 512], dtype: .bfloat16),
                        indices: MLXArray.zeros([1, 16, 128], dtype: .int32), scale: 1,
                        outputType: .bfloat16)
                }
            }
        }

        @Test func invalidGeometryIsRefused() {
            Device.withDefaultDevice(.gpu) {
                #expect(throws: GLM5NextSparsePrefillKernel.Failure.invalidGeometry) {
                    try GLM5NextSparsePrefillKernel.attend(
                        queryLatents: MLXArray.zeros([1, 32, 16, 512], dtype: .bfloat16),
                        latents: MLXArray.zeros([1, 64, 256], dtype: .bfloat16),
                        indices: MLXArray.zeros([1, 16, 128], dtype: .int32), scale: 1,
                        outputType: .bfloat16)
                }
                #expect(throws: GLM5NextSparsePrefillKernel.Failure.invalidGeometry) {
                    try GLM5NextSparsePrefillKernel.attend(
                        queryLatents: MLXArray.zeros([1, 32, 16, 512], dtype: .bfloat16),
                        latents: MLXArray.zeros([1, 64, 512], dtype: .bfloat16),
                        indices: MLXArray.zeros([1, 16, 128], dtype: .uint32), scale: 1,
                        outputType: .bfloat16)
                }
            }
        }

        /// Env-gated GPU microbenchmark at the DSA prefill shape of an
        /// 8192-token chunk's sparse queries: T 6144, N 8192, W 2051, H 64.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_SPARSE_PREFILL_BENCH"] == "1"))
        func benchmarkAgainstGatheredPath() throws {
            try Device.withDefaultDevice(.gpu) {
                let (heads, queries, history) = (64, 6144, 8192)
                let queryLatents = Self.normal([1, heads, queries, 512], scale: 2, seed: 11)
                let latents = Self.normal([1, history, 512], scale: 1, seed: 12)
                let indices = Self.indices(
                    batch: 1, queries: queries, history: history, width: Self.width,
                    members: 2048, leading: 0, seed: 13)
                eval(queryLatents, latents, indices)
                func time(_ label: String, repeats: Int, _ body: () throws -> MLXArray) rethrows
                    -> Double
                {
                    eval(try body())
                    var best = Double.infinity
                    for _ in 0 ..< repeats {
                        let start = Date()
                        let result = try body()
                        eval(result)
                        best = min(best, Date().timeIntervalSince(start) * 1000)
                    }
                    print("GLM53_SPARSE_PREFILL bench \(label) T=\(queries) N=\(history) "
                        + "W=\(Self.width) H=\(heads) best_ms=\(best)")
                    return best
                }
                let kernel = try time("tensor_kernel", repeats: 5) {
                    try GLM5NextSparsePrefillKernel.attend(
                        queryLatents: queryLatents, latents: latents, indices: indices,
                        scale: 1 / 16, outputType: .bfloat16)
                }
                let portable = try time("portable_split_k", repeats: 2) {
                    try GLM5NextSparsePrefillKernel.attend(
                        queryLatents: queryLatents, latents: latents, indices: indices,
                        scale: 1 / 16, outputType: .bfloat16, tensorUnits: false)
                }
                let gathered = time("gathered_reference", repeats: 2) {
                    Self.gatheredAttention(
                        queryLatents: queryLatents, latents: latents.asType(.float32),
                        indices: indices, scale: 1 / 16, roundingType: .bfloat16,
                        outputType: .bfloat16)
                }
                print(
                    "GLM53_SPARSE_PREFILL bench speedup tensor=\(gathered / kernel) "
                        + "portable=\(gathered / portable)")
                // Blocked selection of an 8192-token chunk's last queries at
                // growing history: fused index scores against the score graph.
                for (queries, history) in [(6144, 8192), (8192, 32768)] {
                    let indexQueries = Self.normal([1, queries, 32, 128], scale: 1, seed: 31)
                    let indexKeys = Self.normal([1, history, 128], scale: 1, seed: 32)
                    let poolGates = Self.normal([1, history, 128], scale: 2, seed: 33)
                    let positionBias = Self.normal([4, 128], scale: 0.5, seed: 34)
                    let headWeights = Self.normal([1, queries, 32], scale: 1, seed: 35)
                    let keyValidity = MLXArray.ones([1, history], dtype: .bool)
                    let queryValidity = MLXArray.ones([1, queries], dtype: .bool)
                    eval(indexQueries, indexKeys, poolGates, positionBias, headWeights)
                    for native in [true, false] {
                        _ = try time(
                            "selection native_scores=\(native) Q=\(queries) N=\(history)",
                            repeats: 2
                        ) {
                            try GLM5NextSparseAttention.selectTokens(
                                indexQueries: indexQueries, indexKeys: indexKeys,
                                poolGates: poolGates, poolPositionBias: positionBias,
                                headWeights: headWeights, keyValidity: keyValidity,
                                queryValidity: queryValidity, queryStart: history - queries,
                                tokenBudget: 2048,
                                execution: .init(
                                    selection: .parallelThreshold, deviceRanges: true,
                                    queryBlocks: true),
                                nativePrefillScores: native
                            ).indices
                        }
                    }
                }
            }
        }
    }
}
