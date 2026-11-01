// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// `SparseDecode.nativeKernels` primitives against the reference graphs of
    /// `GLM5NextSparseAttention.selectTokens` and the absorbed gathered-latent
    /// attention, on deterministic synthetic inputs at real index/latent
    /// geometry. Not real-checkpoint or speed evidence.
    @Suite(.serialized)
    struct GLM5NextSparseDecodeKernelTests {
        private struct Inputs {
            var state: UInt64
            mutating func next() -> UInt64 {
                state &+= 0x9E37_79B9_7F4A_7C15
                var z = state
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                return z ^ (z >> 31)
            }
            mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
            mutating func normal(_ shape: [Int], scale: Float = 1, dtype: DType = .bfloat16)
                -> MLXArray
            {
                let count = shape.reduce(1, *)
                return MLXArray((0 ..< count).map { _ in (unit() * 2 - 1) * scale }, shape)
                    .asType(dtype)
            }
        }

        /// One selection problem. `padding[b]` leading invalid keys per row;
        /// `holes` adds interior invalid keys (unusual masks, tail overlap).
        private struct Problem: CustomStringConvertible {
            let batch: Int
            let queries: Int
            let history: Int
            let budget: Int
            let heads: Int
            let dimension: Int
            let padding: [Int]
            let holes: Bool
            let keyType: DType
            let indexQueries: MLXArray
            let indexKeys: MLXArray
            let poolGates: MLXArray
            let positionBias: MLXArray
            let headWeights: MLXArray
            let keyValidity: MLXArray
            let queryValidity: MLXArray
            var queryStart: Int { history - queries }
            var description: String {
                "B=\(batch) Q=\(queries) N=\(history) K=\(budget) pad=\(padding) holes=\(holes) \(keyType)"
            }

            init(
                batch: Int, queries: Int, history: Int, budget: Int = 2048, heads: Int = 32,
                dimension: Int = 128, padding: [Int]? = nil, holes: Bool = false,
                keyType: DType = .bfloat16, seed: UInt64 = 1
            ) {
                self.batch = batch
                self.queries = queries
                self.history = history
                self.budget = budget
                self.heads = heads
                self.dimension = dimension
                self.padding = padding ?? (0 ..< batch).map { $0 == 0 ? 0 : 3 + 2 * $0 }
                self.holes = holes
                self.keyType = keyType
                var random = Inputs(state: seed &* 7919 &+ UInt64(history))
                indexQueries = random.normal([batch, queries, heads, dimension], dtype: keyType)
                indexKeys = random.normal([batch, history, dimension], dtype: keyType)
                poolGates = random.normal([batch, history, dimension], scale: 2, dtype: keyType)
                positionBias = random.normal([4, dimension], scale: 0.5, dtype: keyType)
                headWeights = random.normal([batch, queries, heads], dtype: keyType)
                let pads = self.padding
                var valid = [Bool]()
                for row in 0 ..< batch {
                    for n in 0 ..< history {
                        valid.append(n >= pads[row] && !(holes && row == 0 && n % 37 == 5))
                    }
                }
                keyValidity = MLXArray(valid, [batch, history])
                queryValidity = MLXArray(
                    (0 ..< batch * queries).map { !($0 == batch * queries - 1 && batch > 1) },
                    [batch, queries])
            }

            func native(_ strategy: GLM5NextSparseAttention.SelectionStrategy = .parallelThreshold)
                throws -> GLM5NextSparseAttention.Selection
            {
                try GLM5NextSparseAttention.selectDecodeTokens(
                    indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
                    poolPositionBias: positionBias, headWeights: headWeights,
                    keyValidity: keyValidity, queryValidity: queryValidity,
                    queryStart: queryStart, tokenBudget: budget, strategy: strategy)
            }

            func reference() throws -> GLM5NextSparseAttention.Selection {
                try GLM5NextSparseAttention.selectTokens(
                    indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
                    poolPositionBias: positionBias, headWeights: headWeights,
                    keyValidity: keyValidity, queryValidity: queryValidity,
                    queryStart: queryStart, tokenBudget: budget,
                    execution: .init(
                        selection: .parallelThreshold, deviceRanges: true, queryBlocks: true))
            }

            /// `selectTokens`' pooling graph with elementwise FP32 scoring
            /// (no GEMM, so no TF32 rounding): `[B,Q,P]` masked scores.
            func referenceScores() -> MLXArray {
                let pools = (history + 3) / 4
                let firstKey = MLX.where(
                    keyValidity.any(axis: 1),
                    keyValidity.asType(.int32).argMax(axis: 1).asType(.int32), history)
                let poolIndices =
                    firstKey.reshaped([batch, 1, 1])
                    + MLXArray(0 ..< (pools * 4)).reshaped([1, pools, 4])
                let safe = clip(poolIndices, min: 0, max: history - 1).reshaped([batch, pools * 4])
                let groupedValid =
                    takeAlong(keyValidity, safe, axis: 1).reshaped([batch, pools, 4])
                    .&& (poolIndices .< history)
                let complete = groupedValid.all(axis: 2)
                let groupedKeys = takeAlong(indexKeys, safe.expandedDimensions(axis: -1), axis: 1)
                    .reshaped([batch, pools, 4, dimension])
                let groupedGates = takeAlong(poolGates, safe.expandedDimensions(axis: -1), axis: 1)
                    .reshaped([batch, pools, 4, dimension])
                let logits = groupedGates.asType(.float32) + positionBias.asType(.float32)
                let probabilities = softmax(
                    MLX.where(groupedValid.expandedDimensions(axis: -1), logits, -Float.infinity),
                    axis: 2, precise: true)
                let weights = MLX.where(
                    groupedValid.expandedDimensions(axis: -1), probabilities, 0
                ).asType(indexKeys.dtype)
                let pooled =
                    (weights
                    * MLX.where(groupedValid.expandedDimensions(axis: -1), groupedKeys, 0)).sum(
                        axis: 2)
                // [B,Q,H,1,D] * [B,1,1,P,D] summed over D.
                let products =
                    indexQueries.asType(.float32).expandedDimensions(axis: 3)
                    * pooled.asType(.float32).reshaped([batch, 1, 1, pools, dimension])
                let perHead = maximum(products.sum(axis: -1) * (1 / sqrt(Float(dimension))), 0)
                let scores =
                    (perHead
                    * (headWeights.asType(.float32) * (1 / sqrt(Float(heads))))
                    .expandedDimensions(axis: -1)).sum(axis: 2)
                let positions =
                    (MLXArray(0 ..< queries).asType(.int32) + Int32(queryStart))
                    .reshaped([1, queries, 1])
                let eligible =
                    complete.expandedDimensions(axis: 1)
                    .&& (poolIndices[.ellipsis, 3].expandedDimensions(axis: 1) .<= positions)
                return MLX.where(eligible, scores, -Float.greatestFiniteMagnitude)
            }
        }

        @Test func poolScoresMatchReferencePoolingGraph() throws {
            try Device.withDefaultDevice(.gpu) {
                for problem in [
                    Problem(batch: 1, queries: 1, history: 2049),
                    Problem(batch: 2, queries: 3, history: 2311, holes: true),
                    Problem(batch: 1, queries: 8, history: 2050, keyType: .float32),
                    Problem(batch: 2, queries: 2, history: 403, heads: 3, dimension: 5),
                ] {
                    let native = try GLM5NextSparseDecodeKernel.poolScores(
                        indexQueries: problem.indexQueries, headWeights: problem.headWeights,
                        indexKeys: problem.indexKeys, poolGates: problem.poolGates,
                        poolPositionBias: problem.positionBias, keyValidity: problem.keyValidity,
                        queryStart: problem.queryStart, poolingType: problem.keyType)
                    let actual = native.scores.asArray(Float.self)
                    let expected = problem.referenceScores().asArray(Float.self)
                    let masked = -Float.greatestFiniteMagnitude
                    let peak = expected.filter { $0 != masked }.map(abs).max() ?? 0
                    #expect(peak > 0, "\(problem)")
                    var worst: Float = 0
                    for (a, e) in zip(actual, expected) {
                        if e == masked || a == masked {
                            #expect(a == e, "\(problem): mask mismatch")
                        } else {
                            worst = max(worst, abs(a - e))
                        }
                    }
                    // GLM serves only on M5 (NAX), where the kernel's fast exp matches
                    // the reference; elsewhere one pooled key can round one bfloat16
                    // step differently.
                    let tolerance: Float = glm5NextNAXAvailable ? 1e-5 : 1.0 / 256
                    #expect(
                        worst <= tolerance * peak, "\(problem): max error \(worst), peak \(peak)")
                }
            }
        }

        /// Selection against the blocked `selectTokens` graph. That graph's
        /// FP32 GEMM may run at TF32, so membership may differ only for pools
        /// whose score is within the TF32 error of the row's K-th score.
        @Test func selectionMatchesSelectTokensUpToNearTies() throws {
            try Device.withDefaultDevice(.gpu) {
                var problems: [Problem] = []
                for (history, queries) in [(2049, 1), (2051, 8), (4099, 3), (10000, 1), (7001, 5)] {
                    problems.append(Problem(batch: 1, queries: queries, history: history))
                }
                problems.append(Problem(batch: 2, queries: 4, history: 3001))
                problems.append(
                    Problem(batch: 3, queries: 2, history: 2600, padding: [0, 9, 2050]))
                problems.append(Problem(batch: 1, queries: 2, history: 2500, keyType: .float32))
                problems.append(Problem(batch: 2, queries: 8, history: 5003, holes: true))
                for problem in problems {
                    for strategy in [
                        GLM5NextSparseAttention.SelectionStrategy.parallelThreshold, .reference,
                    ] where strategy == .parallelThreshold || problem.history < 3000 {
                        try compareSelection(problem, strategy: strategy)
                    }
                }
            }
        }

        private func compareSelection(
            _ problem: Problem, strategy: GLM5NextSparseAttention.SelectionStrategy
        ) throws {
            let native = try problem.native(strategy)
            let reference = try problem.reference()
            #expect(native.historyLength == reference.historyLength)
            #expect(native.staticPaddingColumns == reference.staticPaddingColumns)
            #expect(native.indices.shape == reference.indices.shape, "\(problem)")
            #expect(native.indices.dtype == .int32)
            let width = problem.budget + 3
            let actual = native.indices.asArray(Int32.self)
            let expected = reference.indices.asArray(Int32.self)
            let scores = problem.referenceScores().asArray(Float.self)
            let pools = (problem.history + 3) / 4
            for row in 0 ..< problem.batch * problem.queries {
                let a = Array(actual[(row * width) ..< ((row + 1) * width)])
                let e = Array(expected[(row * width) ..< ((row + 1) * width)])
                if a == e { continue }
                // Tail and padding columns never depend on scores.
                #expect(
                    Array(a[problem.budget...]) == Array(e[problem.budget...]),
                    "\(problem) row \(row): tail")
                let rowScores = Array(scores[(row * pools) ..< ((row + 1) * pools)])
                let ranked = rowScores.filter { $0 > -Float.greatestFiniteMagnitude }.sorted(
                    by: >)
                let kth = ranked[min(problem.budget / 4, ranked.count) - 1]
                let peak = ranked.map(abs).max() ?? 0
                // Pools are anchored at the row's first valid key.
                let firstValid = problem.padding[row / problem.queries]
                func poolSet(_ tokens: [Int32]) -> Set<Int> {
                    Set(
                        tokens[..<problem.budget].filter { $0 >= 0 }.map {
                            (Int($0) - firstValid) / 4
                        })
                }
                let difference = poolSet(a).symmetricDifference(poolSet(e))
                #expect(difference.count <= 8, "\(problem) row \(row): \(difference.count) pools")
                for pool in difference {
                    #expect(
                        abs(rowScores[pool] - kth) <= 2e-3 * peak,
                        "\(problem) row \(row): pool \(pool) is not a near tie")
                }
            }
        }

        /// The expansion is exact: same chosen pools, same token row, including
        /// interior holes whose tail overlaps a selected pool.
        @Test func expansionMatchesScalarOracle() throws {
            try Device.withDefaultDevice(.gpu) {
                for problem in [
                    Problem(batch: 2, queries: 8, history: 61, budget: 8, holes: true),
                    Problem(batch: 2, queries: 3, history: 2049, holes: true),
                    Problem(batch: 1, queries: 5, history: 30, budget: 2048),
                    Problem(batch: 3, queries: 2, history: 75, budget: 12, padding: [0, 75, 70]),
                ] {
                    let pooled = try GLM5NextSparseDecodeKernel.poolScores(
                        indexQueries: problem.indexQueries, headWeights: problem.headWeights,
                        indexKeys: problem.indexKeys, poolGates: problem.poolGates,
                        poolPositionBias: problem.positionBias, keyValidity: problem.keyValidity,
                        queryStart: problem.queryStart, poolingType: problem.keyType)
                    let pools = (problem.history + 3) / 4
                    let chosen = try GLM5NextSparseSelection.select(
                        scores: pooled.scores.reshaped([problem.batch * problem.queries, pools]),
                        completePools: pooled.complete.any(axis: 0, keepDims: true),
                        requestedCount: problem.budget / 4, strategy: .reference)
                    let tokens = try GLM5NextSparseDecodeKernel.expand(
                        chosen: chosen, keyValidity: problem.keyValidity,
                        queryValidity: problem.queryValidity, queryStart: problem.queryStart,
                        tokenBudget: problem.budget)
                    #expect(
                        tokens.asArray(Int32.self)
                            == expansionOracle(problem, chosen: chosen.asArray(Int32.self)),
                        "\(problem)")
                }
            }
        }

        private func expansionOracle(_ problem: Problem, chosen: [Int32]) -> [Int32] {
            let n = problem.history
            let pools = (n + 3) / 4
            let count = min(problem.budget / 4, pools)
            let valid = problem.keyValidity.asArray(Bool.self)
            let queryValid = problem.queryValidity.asArray(Bool.self)
            var result: [Int32] = []
            for b in 0 ..< problem.batch {
                let row = Array(valid[(b * n) ..< ((b + 1) * n)])
                let first = row.firstIndex(of: true) ?? n
                func complete(_ p: Int) -> Bool {
                    (0 ..< 4).allSatisfy { first + 4 * p + $0 < n && row[first + 4 * p + $0] }
                }
                for q in 0 ..< problem.queries {
                    let pos = problem.queryStart + q
                    let selected = chosen[
                        ((b * problem.queries + q) * count)
                            ..< ((b * problem.queries + q + 1) * count)
                    ]
                    .map(Int.init).sorted()
                    var tokens: [Int32] = []
                    for p in selected {
                        let ok = p < pools && complete(p) && first + 4 * p + 3 <= pos
                        for j in 0 ..< 4 { tokens.append(ok ? Int32(first + 4 * p + j) : -1) }
                    }
                    tokens += Array(repeating: -1, count: problem.budget - tokens.count)
                    let visible = row[0 ... pos].filter { $0 }.count
                    let tailCount = visible % 4
                    for j in 0 ..< 3 {
                        let t = first + visible - tailCount + j
                        let clipped = min(max(t, 0), n - 1)
                        let ok =
                            j < tailCount && t < n && row[clipped] && clipped <= pos
                            && !tokens.contains(Int32(t))
                        tokens.append(ok ? Int32(t) : -1)
                    }
                    if !queryValid[b * problem.queries + q] {
                        tokens = Array(repeating: -1, count: tokens.count)
                    }
                    result += tokens
                }
            }
            return result
        }

        /// The gathered absorbed reference: FP32 scores, masked softmax,
        /// probabilities rounded to `roundingType`, FP32 PV, one rounding.
        private func gatheredAttention(
            queryLatents: MLXArray, latents: MLXArray, indices: MLXArray, scale: Float,
            roundingType: DType
        ) -> MLXArray {
            let (batch, count, width) = (indices.dim(0), indices.dim(1), indices.dim(2))
            let selected = takeAlong(
                latents, clip(indices, min: 0, max: latents.dim(1) - 1).reshaped(batch, -1, 1),
                axis: 1
            ).reshaped([batch, count, width, latents.dim(2)]).asType(.float32)
            let score =
                matmul(
                    queryLatents.transposed(0, 2, 1, 3).asType(.float32),
                    selected.transposed(0, 1, 3, 2)) * scale
            let mask = (indices .>= 0).expandedDimensions(axis: 2)
            let masked = MLX.where(mask, score, -Float.infinity)
            let maximum = MLX.where(
                mask.any(axis: -1, keepDims: true), masked.max(axis: -1, keepDims: true), 0)
            let exps = MLX.where(mask, exp(masked - maximum), 0)
            let total = exps.sum(axis: -1, keepDims: true)
            let probability = (exps / MLX.where(total .> 0, total, 1)).asType(roundingType)
            return matmul(probability.asType(.float32), selected).asType(queryLatents.dtype)
                .transposed(0, 2, 1, 3)
        }

        /// Split-K attention vs the gathered path. The reference rounds each
        /// probability to the activation dtype before PV, split-K keeps FP32;
        /// both round the output once. The measured gap is reported; for 16-bit
        /// types it is bounded by two output ULPs at the peak (2^-7).
        @Test(arguments: [1, 3, 8])
        func splitKAttentionMatchesGatheredReference(queries: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                for (batch, heads, dimension, history, dtype) in [
                    (1, 64, 512, 4099, DType.bfloat16), (2, 64, 512, 2600, .bfloat16),
                    (2, 3, 5, 300, .float32), (1, 40, 72, 700, .float16),
                ] {
                    var random = Inputs(state: UInt64(history * 31 + queries))
                    let queryLatents = random.normal(
                        [batch, heads, queries, dimension], scale: 2, dtype: dtype)
                    let latents = random.normal([batch, history, dimension], dtype: dtype)
                    let width = 2051
                    var ids = [Int32]()
                    for row in 0 ..< batch * queries {
                        for slot in 0 ..< width {
                            let invalid = (slot + row) % 7 == 3 || (row == 1 && slot < 900)
                            ids.append(invalid ? -1 : Int32(Int(random.next() % UInt64(history))))
                        }
                    }
                    // One fully masked row when there are several.
                    if batch * queries > 2 {
                        for slot in 0 ..< width { ids[2 * width + slot] = -1 }
                    }
                    let indices = MLXArray(ids, [batch, queries, width])
                    let scale: Float = 1 / 16
                    let actual = try GLM5NextSparseDecodeKernel.attend(
                        queryLatents: queryLatents, latents: latents, indices: indices,
                        scale: scale, outputType: dtype)
                    let expected = gatheredAttention(
                        queryLatents: queryLatents, latents: latents, indices: indices,
                        scale: scale, roundingType: dtype)
                    #expect(actual.shape == expected.shape && actual.dtype == dtype)
                    let a = actual.asType(.float32).asArray(Float.self)
                    let e = expected.asType(.float32).asArray(Float.self)
                    let peak = e.map(abs).max() ?? 0
                    let worst = zip(a, e).map { abs($0 - $1) }.max() ?? 0
                    let bound = dtype == .float32 ? 1e-5 * peak : peak / 128
                    print(
                        "GLM53_SPARSE_DECODE attention B=\(batch) H=\(heads) C=\(dimension) "
                            + "Q=\(queries) \(dtype) max_abs_error=\(worst) peak=\(peak)")
                    #expect(!a.contains { !$0.isFinite })
                    #expect(worst <= bound, "H=\(heads) C=\(dimension): \(worst) > \(bound)")
                    if batch * queries > 2 {
                        // [B,H,T,C]: row 2 is b = 2 / T, t = 2 % T.
                        let (b, t) = (2 / queries, 2 % queries)
                        #expect(
                            actual[b, 0..., t].asType(.float32).asArray(Float.self).allSatisfy {
                                $0 == 0
                            })
                    }
                }
            }
        }

        /// Latent history views (strided batch rows of a larger buffer) are
        /// read in place and give the same result as a contiguous copy.
        @Test func attentionReadsStridedHistoryViews() throws {
            try Device.withDefaultDevice(.gpu) {
                var random = Inputs(state: 5)
                let storage = random.normal([2, 3000, 512])
                let history = storage[0..., ..<2500]
                let queryLatents = random.normal([2, 64, 2, 512])
                let indices = MLXArray(
                    (0 ..< 2 * 2 * 2051).map { (i: Int) -> Int32 in Int32((i * 7919) % 2600) - 100
                    }, [2, 2, 2051])
                let viewed = try GLM5NextSparseDecodeKernel.attend(
                    queryLatents: queryLatents, latents: history, indices: indices,
                    scale: 1 / 16, outputType: .bfloat16)
                let copied = try GLM5NextSparseDecodeKernel.attend(
                    queryLatents: queryLatents, latents: contiguous(history) + 0,
                    indices: indices, scale: 1 / 16, outputType: .bfloat16)
                #expect(
                    viewed.asType(.float32).asArray(Float.self)
                        == copied.asType(.float32).asArray(Float.self))
            }
        }

        /// Opt-in (`GLM53_DSA_DECODE_BENCH=1`) GPU time of each DSA decode
        /// component per history length at real geometry: amortized over 16
        /// independent copies per evaluation, median of 7. Asserts nothing.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_DSA_DECODE_BENCH"] == "1"))
        func decodeComponentTaper() throws {
            try Device.withDefaultDevice(.gpu) {
                func time(_ body: () throws -> [MLXArray]) rethrows -> Double {
                    let copies = 16
                    eval(try body())
                    var samples: [Double] = []
                    for _ in 0 ..< 7 {
                        let start = ContinuousClock.now
                        var outputs: [MLXArray] = []
                        for _ in 0 ..< copies { outputs += try body() }
                        eval(outputs)
                        let d = start.duration(to: .now)
                        samples.append(
                            (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
                                * 1e6 / Double(copies))
                    }
                    return samples.sorted()[samples.count / 2]
                }
                var random = Inputs(state: 3)
                let queryLatents = random.normal([1, 64, 1, 512], scale: 2)
                let indexQueries = random.normal([1, 1, 32, 128])
                let headWeights = random.normal([1, 1, 32])
                let bias = random.normal([4, 128], scale: 0.5)
                for n in [32, 2047, 2049, 8192, 32768, 65536] {
                    let keys = random.normal([1, n, 128])
                    let gates = random.normal([1, n, 128], scale: 2)
                    let latents = random.normal([1, n, 512])
                    let validity = MLXArray.ones([1, n], dtype: .bool)
                    let queryValidity = MLXArray.ones([1, 1], dtype: .bool)
                    eval(keys, gates, latents, validity)
                    let pools = (n + 3) / 4
                    var line = "GLM53_DSA_BENCH n=\(n)"
                    if n <= 2048 {
                        // denseHistoryAttention's core at the last dense position.
                        let dense = try time {
                            let history = latents.asType(.float32)
                            let rows = queryLatents.asType(.float32).reshaped(1, 64, 512)
                            let score = matmul(rows, history.transposed(0, 2, 1)).reshaped(
                                1, 64, 1, n)
                            let probability = try GLM5NextDenseSelectionKernel.maskedSoftmax(
                                score: score, keyValidity: validity, queryValidity: queryValidity,
                                queryStart: n - 1, scale: 1 / 16, roundingType: .bfloat16)
                            return [
                                matmul(probability.reshaped(1, 64, n), history).asType(.bfloat16)
                            ]
                        }
                        // The decode-sized dense path: every-pool expansion + split-K.
                        let pools = (n + 3) / 4
                        let everyPool = broadcast(
                            MLXArray.arange(pools, dtype: .int32).reshaped(1, pools),
                            to: [1, pools])
                        let splitK = try time {
                            let tokens = try GLM5NextSparseDecodeKernel.expand(
                                chosen: everyPool, keyValidity: validity,
                                queryValidity: queryValidity, queryStart: n - 1,
                                tokenBudget: 4 * pools)
                            return [
                                try GLM5NextSparseDecodeKernel.attend(
                                    queryLatents: queryLatents, latents: latents, indices: tokens,
                                    scale: 1 / 16, outputType: .bfloat16)
                            ]
                        }
                        line += String(format: " dense_us=%.1f dense_splitk_us=%.1f", dense, splitK)
                    }
                    guard n > 2048 else {
                        print(line)
                        continue
                    }
                    let pooled = try GLM5NextSparseDecodeKernel.poolScores(
                        indexQueries: indexQueries, headWeights: headWeights, indexKeys: keys,
                        poolGates: gates, poolPositionBias: bias, keyValidity: validity,
                        queryStart: n - 1, poolingType: .bfloat16)
                    let chosen = try GLM5NextSparseSelection.select(
                        scores: pooled.scores.reshaped([1, pools]), completePools: pooled.complete,
                        requestedCount: min(512, pools), strategy: .parallelThreshold)
                    let tokens = try GLM5NextSparseDecodeKernel.expand(
                        chosen: chosen, keyValidity: validity, queryValidity: queryValidity,
                        queryStart: n - 1, tokenBudget: 2048)
                    eval(pooled.scores, pooled.complete, chosen, tokens)
                    let pool = try time {
                        let r = try GLM5NextSparseDecodeKernel.poolScores(
                            indexQueries: indexQueries, headWeights: headWeights, indexKeys: keys,
                            poolGates: gates, poolPositionBias: bias, keyValidity: validity,
                            queryStart: n - 1, poolingType: .bfloat16)
                        return [r.scores, r.complete]
                    }
                    let select = try time {
                        [
                            try GLM5NextSparseSelection.select(
                                scores: pooled.scores.reshaped([1, pools]),
                                completePools: pooled.complete, requestedCount: min(512, pools),
                                strategy: .parallelThreshold)
                        ]
                    }
                    let expand = try time {
                        [
                            try GLM5NextSparseDecodeKernel.expand(
                                chosen: chosen, keyValidity: validity, queryValidity: queryValidity,
                                queryStart: n - 1, tokenBudget: 2048)
                        ]
                    }
                    let attend = try time {
                        [
                            try GLM5NextSparseDecodeKernel.attend(
                                queryLatents: queryLatents, latents: latents, indices: tokens,
                                scale: 1 / 16, outputType: .bfloat16)
                        ]
                    }
                    // One decode append to a text cache already holding n columns.
                    let cache = GLM5NextDSACache()
                    try cache.append(latents: latents, keys: keys, gates: gates, validity: validity)
                    eval(cache.state)
                    let one = (
                        random.normal([1, 1, 512]), random.normal([1, 1, 128]),
                        random.normal([1, 1, 128]), MLXArray.ones([1, 1], dtype: .bool)
                    )
                    var appends: [Double] = []
                    for _ in 0 ..< 9 {
                        let start = ContinuousClock.now
                        try cache.append(latents: one.0, keys: one.1, gates: one.2, validity: one.3)
                        eval(cache.state)
                        let d = start.duration(to: .now)
                        appends.append(
                            (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
                                * 1e6)
                    }
                    // The text forward's copy, append and adopt of the same cache.
                    var adopted: [Double] = []
                    for _ in 0 ..< 9 {
                        let start = ContinuousClock.now
                        let working = cache.copy() as! GLM5NextDSACache
                        try working.append(
                            latents: one.0, keys: one.1, gates: one.2, validity: one.3)
                        cache.adopt(working)
                        eval(cache.state)
                        let d = start.duration(to: .now)
                        adopted.append(
                            (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
                                * 1e6)
                    }
                    line += String(
                        format:
                            " pool_us=%.1f select_us=%.1f expand_us=%.1f attend_us=%.1f append_us=%.1f adopt_append_us=%.1f",
                        pool, select, expand, attend, appends.sorted()[4], adopted.sorted()[4])
                    print(line)
                }
            }
        }

        @Test func cpuStreamIsRefused() {
            Device.withDefaultDevice(.cpu) {
                let problem = Problem(batch: 1, queries: 1, history: 2049)
                #expect(throws: GLM5NextSparseDecodeKernel.Failure.unsupportedDeviceOrStream) {
                    try problem.native()
                }
                #expect(throws: GLM5NextSparseDecodeKernel.Failure.unsupportedDeviceOrStream) {
                    try GLM5NextSparseDecodeKernel.attend(
                        queryLatents: MLXArray.zeros([1, 2, 1, 8]),
                        latents: MLXArray.zeros([1, 4, 8]),
                        indices: MLXArray.zeros([1, 1, 4], dtype: .int32), scale: 1,
                        outputType: .float32)
                }
            }
        }

        @Test func invalidGeometryIsRefused() {
            Device.withDefaultDevice(.gpu) {
                // Nine queries exceed the decode bound.
                let problem = Problem(batch: 1, queries: 9, history: 2049)
                #expect(throws: GLM5NextSparseDecodeKernel.Failure.invalidGeometry) {
                    try problem.native()
                }
                let wide = Problem(batch: 1, queries: 1, history: 64, heads: 2, dimension: 129)
                #expect(throws: GLM5NextSparseDecodeKernel.Failure.invalidGeometry) {
                    try wide.native()
                }
                #expect(throws: GLM5NextSparseAttentionError.self) {
                    try GLM5NextSparseAttention.selectDecodeTokens(
                        indexQueries: problem.indexQueries, indexKeys: problem.indexKeys,
                        poolGates: problem.poolGates, poolPositionBias: problem.positionBias,
                        headWeights: problem.headWeights, keyValidity: problem.keyValidity,
                        queryValidity: problem.queryValidity, queryStart: 0, tokenBudget: 6)
                }
            }
        }
    }
}
