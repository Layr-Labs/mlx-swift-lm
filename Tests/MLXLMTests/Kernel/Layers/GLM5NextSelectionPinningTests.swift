// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// GPU: DSA prefill selection under exact score ties at the budget
    /// boundary. Index keys come from eight patterns (every pool's four keys
    /// equal, uniform gates) and queries from {-1, 0, 1}, so each row's pool
    /// scores take a handful of exact values and the 512th is a tie. The
    /// selector breaks that tie by its walk over the whole candidate row, so
    /// the same rows scored in a call of another width choose differently;
    /// with the product step's `selectionHistories` they choose exactly as
    /// the product step does.
    @Suite(.serialized)
    struct GLM5NextSelectionPinningTests {
        @Test func pinnedSelectionMatchesTheProductStep() throws {
            try Device.withDefaultDevice(.gpu) {
                // (keys before the call, product step rows, other cut rows)
                for (offset, productRows, otherRows) in [(4096, 1024, 2048), (6144, 1024, 512), (4096, 512, 1536)] {
                    let fixture = Self.fixture(offset: offset, rows: max(productRows, otherRows))
                    let product = try Self.select(fixture, offset: offset, rows: productRows, histories: nil)
                    let other = try Self.select(fixture, offset: offset, rows: otherRows, histories: nil)
                    let pinned = try Self.select(
                        fixture, offset: offset, rows: otherRows,
                        histories: Self.histories(productRows: productRows, otherRows: otherRows, offset: offset))
                    let shared = min(productRows, otherRows)
                    let unpinnedRows = Self.differingRows(product[0..., ..<shared], other[0..., ..<shared])
                    let pinnedRows = Self.differingRows(product[0..., ..<shared], pinned[0..., ..<shared])
                    print("GLM53_SELECTION_PINNING offset=\(offset) rows=\(productRows)/\(otherRows) unpinned_rows_differing=\(unpinnedRows) pinned_rows_differing=\(pinnedRows)")
                    // The fixture must exercise the tie, or the test proves nothing.
                    #expect(unpinnedRows > 0)
                    #expect(pinnedRows == 0)
                    if otherRows < productRows {
                        // The product step's later rows, as the other cut's next call (pinned) chooses them.
                        let next = try Self.select(
                            fixture, offset: offset + otherRows, rows: productRows - otherRows,
                            histories: [(rowEnd: productRows - otherRows, keys: offset + productRows)])
                        #expect(Self.differingRows(product[0..., otherRows...], next) == 0)
                    }
                }
            }
        }

        /// The other cut's rows grouped by the product steps they belong to.
        static func histories(productRows: Int, otherRows: Int, offset: Int) -> [(rowEnd: Int, keys: Int)] {
            var pieces: [(rowEnd: Int, keys: Int)] = []
            var end = productRows
            while true {
                pieces.append((rowEnd: min(end, otherRows), keys: offset + end))
                if end >= otherRows { break }
                end += productRows
            }
            return pieces
        }

        struct Fixture {
            let indexKeys: MLXArray, gates: MLXArray, bias: MLXArray, queries: MLXArray, weights: MLXArray
        }

        static func fixture(offset: Int, rows: Int) -> Fixture {
            let heads = 32, dimension = 128
            let total = offset + rows
            let random = MLXRandom.split(key: MLXRandom.key(UInt64(total)), into: 4)
            let patterns = MLXRandom.randInt(low: -1, high: 2, [8, dimension], key: random[0]).asType(.float32)
            let choice = MLXRandom.randInt(low: 0, high: 8, [(total + 3) / 4], key: random[1])
            let pooled = take(patterns, choice, axis: 0)
            let indexKeys = repeated(pooled, count: 4, axis: 0)[..<total].reshaped([1, total, dimension])
                .asType(.bfloat16)
            let gates = MLXArray.zeros([1, total, dimension], dtype: .bfloat16)
            let bias = MLXArray.zeros([4, dimension], dtype: .bfloat16)
            let queries = MLXRandom.randInt(low: -1, high: 2, [1, rows, heads, dimension], key: random[2])
                .asType(.bfloat16)
            let weights = (MLXRandom.randInt(low: 1, high: 4, [1, rows, heads], key: random[3]).asType(.float32) * 0.25)
                .asType(.bfloat16)
            eval(indexKeys, gates, bias, queries, weights)
            return Fixture(indexKeys: indexKeys, gates: gates, bias: bias, queries: queries, weights: weights)
        }

        /// One call: query rows `[offset - base, ...)` of the fixture against
        /// keys through `offset + rows`; the fixture's queries start at its
        /// first query row, which sits at `base` = the fixture's offset.
        static func select(
            _ fixture: Fixture, offset: Int, rows: Int, histories: [(rowEnd: Int, keys: Int)]?
        ) throws -> MLXArray {
            let base = fixture.indexKeys.dim(1) - fixture.queries.dim(1)
            let first = offset - base
            let n = offset + rows
            let selection = try GLM5NextSparseAttention.selectTokens(
                indexQueries: fixture.queries[0..., first ..< first + rows],
                indexKeys: fixture.indexKeys[0..., ..<n], poolGates: fixture.gates[0..., ..<n],
                poolPositionBias: fixture.bias, headWeights: fixture.weights[0..., first ..< first + rows],
                keyValidity: MLXArray.ones([1, n], dtype: .bool),
                queryValidity: MLXArray.ones([1, rows], dtype: .bool), queryStart: offset, tokenBudget: 2048,
                execution: GLM5NextExecutionPolicy.serving.sparseSelection, nativePrefillScores: true,
                selectionHistories: histories)
            eval(selection.indices)
            return selection.indices
        }

        static func differingRows(_ a: MLXArray, _ b: MLXArray) -> Int {
            Int((a .!= b).any(axis: 2).asType(.int32).sum().item(Int32.self))
        }
    }
}

extension KernelTests {
    /// Opt-in (`GLM53_PINNED_STAGE=1`, GPU, the row-invariance suite's
    /// rank-shaped serving stage with paged DSA histories): a chunk after
    /// 4,096 tokens of history, cut otherwise than the product's micro-batches
    /// with those as `selectionRows`, computes the product's hidden rows,
    /// logits and cache bit for bit, where the same cut without them does
    /// not (the synthetic indexer ties too). TF32 as served: with
    /// `MLX_ENABLE_TF32=0` an FP32 GEMM elsewhere is not row-count invariant
    /// and the cuts differ in hidden rows and cache whatever the selections.
    @Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["GLM53_PINNED_STAGE"] == "1"))
    struct GLM5NextPinnedSelectionStageTests {
        static func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        static func same(_ a: [MLXArray], _ b: [MLXArray]) throws -> Bool {
            try fence(a + b)
            func bytes(_ x: MLXArray) -> Data { x.asData(access: .copy).data }
            return a.count == b.count
                && zip(a, b).allSatisfy { $0.shape == $1.shape && $0.dtype == $1.dtype && bytes($0) == bytes($1) }
        }

        static func run(
            _ stage: GLM5NextTextStage, _ tokens: MLXArray, history: GLM5NextTextStageCache,
            _ segment: GLM5NextPrefillSegment
        ) throws -> (hidden: [MLXArray], logits: MLXArray, state: [MLXArray]) {
            let cache = history.copy()
            let candidate = try stage.prepareForward(
                tokens: tokens, cache: cache, segments: [segment], readout: .lastToken)
            try candidate.commit(afterEvaluating: fence)
            guard case .final(let output) = candidate.output else { throw GLM5NextLoadError.invalid("no logits") }
            return (candidate.segmentHidden, output.logits, cache.layers.flatMap(\.state))
        }

        @Test func pinnedCutsAreTheProductCut() throws {
            try Device.withDefaultDevice(.gpu) {
                let stage = try GLM5NextBatchInvarianceTests.makeStage(layers: 5)
                let c = stage.plan.configuration
                print("GLM53_PINNED_STAGE layers=\(c.layerTypes.prefix(5).map { $0 == .linearAttention ? "kda" : "dsa" }) index_topk=\(c.indexTopK)")
                let pool = try stage.makeDSAPagePool(pageSize: 64)
                let account = try pool.reserve(pages: 8192)
                stage.dsaPagePool = pool
                let total = 4096 + 8192
                let values = MLXArray((0 ..< total).map { Int32((($0 + 1) * 7919) % c.vocabularySize) }, [1, total])
                let history = stage.newCache(account: account)
                try stage.prepareForward(
                    tokens: values[0..., ..<4096], cache: history, segments: [.init(microBatchRows: [2048, 2048])]
                ).commit(afterEvaluating: Self.fence)
                let chunk = values[0..., 4096...]
                for (product, other) in [([4096, 4096], [8192]), ([4096, 4096], [2048, 6144]), ([8192], [4096, 4096])] {
                    let a = try Self.run(stage, chunk, history: history, .init(microBatchRows: product))
                    let b = try Self.run(
                        stage, chunk, history: history, .init(microBatchRows: other, selectionRows: product))
                    let u = try Self.run(stage, chunk, history: history, .init(microBatchRows: other))
                    let label = "product \(product) other \(other)"
                    let unpinned = try Self.same(u.hidden + [u.logits] + u.state, a.hidden + [a.logits] + a.state)
                    print("GLM53_PINNED_STAGE \(label) unpinned_same=\(unpinned ? 1 : 0)")
                    #expect(try Self.same(b.hidden, a.hidden), "\(label)")
                    #expect(try Self.same([b.logits], [a.logits]), "\(label)")
                    #expect(try Self.same(b.state, a.state), "\(label)")
                }
            }
        }
    }
}
