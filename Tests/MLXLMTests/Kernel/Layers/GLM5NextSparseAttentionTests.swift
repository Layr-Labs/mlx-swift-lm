// Copyright © 2026 Eigen Labs.
// Independent scalar oracles for the Transformers GLM5Next equations cited by
// GLM5NextSparseAttention. All MLX work in these tests is explicitly CPU-only.

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextSparseAttentionTests {
        @Test func testThreeFourFiveTokenBoundariesHaveNoFutureOrDuplicateMembers() throws {
            try Device.withDefaultDevice(.cpu) {
                for length in [3, 4, 5] {
                    let fixture = IndexFixture(batch: 1, history: length, queries: length, start: 0)
                    let selection = try fixture.select(budget: 8)
                    let indices = selection.indices.asArray(Int32.self)
                    #expect(selection.indices.shape == [1, length, 11])
                    #expect(indices == fixture.oracle(budget: 8))
                    for query in 0 ..< length {
                        let row = Array(indices[query * 11 ..< (query + 1) * 11])
                        let valid = row.filter { $0 >= 0 }
                        #expect(Set(valid) == Set((0 ... query).map(Int32.init)))
                        #expect(valid.count == Set(valid).count)
                        if query == 2 {
                            #expect(Array(row[8...]) == [0, 1, 2] as [Int32])
                        } else if query == 3 {
                            #expect(Array(row[0 ..< 4]) == [0, 1, 2, 3] as [Int32])
                            #expect(Array(row[8...]) == [-1, -1, -1] as [Int32])
                        } else if query == 4 {
                            #expect(Array(row[8...]) == [4, -1, -1] as [Int32])
                        }
                    }
                }
            }
        }

        @Test func testLearnedChannelPoolingUsesBothGateAndPositionBias() throws {
            try Device.withDefaultDevice(.cpu) {
                var fixture = IndexFixture(batch: 1, history: 9, queries: 1, start: 8)
                fixture.heads = 1
                fixture.queries = [1, 1]
                fixture.weights = [1]
                let poolOneKeys: [Float] = Array(repeating: [Float(4), 4], count: 4).flatMap {
                    (pair: [Float]) -> [Float] in pair
                }
                fixture.keys =
                    ([10, 0, 0, 10, 0, 0, 0, 0] as [Float]) + poolOneKeys + ([0, 0] as [Float])
                fixture.gates = Array(repeating: 0, count: 18)
                fixture.gates[0] = 0.4
                fixture.gates[3] = 0.4
                fixture.bias = Array(repeating: 0, count: 8)
                let gateOnly = try fixture.select(budget: 4).indices.asArray(Int32.self)
                #expect(Array(gateOnly.prefix(4)) == [4, 5, 6, 7] as [Int32])

                // Neither 0.4 preference alone beats pool 1. Together they select
                // different members for each channel and lift pool 0 above pool 1.
                fixture.bias[0] = 0.4
                fixture.bias[3] = 0.4
                let combined = try fixture.select(budget: 4).indices.asArray(Int32.self)
                #expect(combined == fixture.oracle(budget: 4))
                #expect(Array(combined.prefix(4)) == [0, 1, 2, 3] as [Int32])
                #expect(Array(combined.suffix(3)) == [8, -1, -1] as [Int32])
                fixture.gates = Array(repeating: 0, count: 18)
                let biasOnly = try fixture.select(budget: 4).indices.asArray(Int32.self)
                #expect(Array(biasOnly.prefix(4)) == [4, 5, 6, 7] as [Int32])
            }
        }

        @Test func testWeightedReLUIsPerHeadBeforeReduction() throws {
            try Device.withDefaultDevice(.cpu) {
                var fixture = IndexFixture(batch: 1, history: 9, queries: 1, start: 8)
                fixture.queries = [1, 0, 0, 1]
                let poolZeroKeys: [Float] = Array(repeating: [Float(4), -4], count: 4).flatMap {
                    (pair: [Float]) -> [Float] in pair
                }
                let poolOneKeys: [Float] = Array(repeating: [Float(1), 1], count: 4).flatMap {
                    (pair: [Float]) -> [Float] in pair
                }
                fixture.keys = poolZeroKeys + poolOneKeys + ([0, 0] as [Float])
                fixture.gates = Array(repeating: 0, count: 18)
                fixture.bias = Array(repeating: 0, count: 8)
                fixture.weights = [1, 1]
                let rectified = try fixture.select(budget: 4).indices.asArray(Int32.self)
                #expect(Array(rectified.prefix(4)) == [0, 1, 2, 3] as [Int32])
                // ReLU after summing heads would choose pool 1 instead.
                fixture.weights = [0.1, 1]
                let weighted = try fixture.select(budget: 4).indices.asArray(Int32.self)
                #expect(Array(weighted.prefix(4)) == [4, 5, 6, 7] as [Int32])
                fixture.weights = [1, -2]
                #expect(
                    try fixture.select(budget: 4).indices.asArray(Int32.self)
                        == fixture.oracle(budget: 4))
            }
        }

        @Test func testLeftPaddingAndAllPaddingStayInvalidUntilRealTokensExist() throws {
            try Device.withDefaultDevice(.cpu) {
                var fixture = IndexFixture(batch: 2, history: 9, queries: 9, start: 0)
                fixture.keyValid = (0 ..< 18).map { (2 ..< 9).contains($0) }
                fixture.queryValid = fixture.keyValid
                let selection = try fixture.select(budget: 8)
                let indices = selection.indices.asArray(Int32.self)
                #expect(indices == fixture.oracle(budget: 8))
                for query in 0 ..< 9 {
                    let actual = indices[query * 11 ..< (query + 1) * 11].filter { $0 >= 0 }
                    let expected = query < 2 ? [] : (2 ... query).map(Int32.init)
                    #expect(Set(actual) == Set(expected))
                }
                #expect(indices[99...].allSatisfy { $0 == -1 })
                let out = try GLM5NextSparseAttention.attend(
                    queries: MLXArray.ones([2, 9, 1, 2]),
                    latents: MLXArray.ones([2, 9, 3]), selection: selection,
                    keyProjection: MLXArray.ones([1, 2, 3]),
                    valueProjection: MLXArray.ones([1, 2, 3]))
                let actual = out.asArray(Float.self)
                #expect(actual.allSatisfy { $0.isFinite })
                #expect(actual.prefix(4).allSatisfy { $0 == 0 })
                #expect(actual[4 ..< 18].allSatisfy { abs($0 - 3) < 0.00001 })
                #expect(actual[18...].allSatisfy { $0 == 0 })
            }
        }

        @Test func testTailOverlapFromAnInteriorInvalidSlotDoesNotDuplicateAttentionMass() throws {
            try Device.withDefaultDevice(.cpu) {
                var fixture = IndexFixture(batch: 1, history: 12, queries: 1, start: 11)
                fixture.keyValid[4] = false
                let selection = try fixture.select(budget: 8)
                let indices = selection.indices.asArray(Int32.self)
                // Both complete pools are retained. The reference's count-based
                // tail points into pool 2; its scatter mask treats these as a set.
                #expect(indices == [0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1])
                #expect(indices == fixture.oracle(budget: 8))
            }
        }

        @Test func testNontrivialPoolSelectionMatchesScalarOracleAcrossBatchRows() throws {
            try Device.withDefaultDevice(.cpu) {
                var fixture = IndexFixture(batch: 2, history: 17, queries: 10, start: 7)
                fixture.keyValid[0] = false
                fixture.keyValid[1] = false
                fixture.keyValid[17] = false
                fixture.queryValid[13] = false
                let selection = try fixture.select(budget: 4)
                #expect(selection.indices.asArray(Int32.self) == fixture.oracle(budget: 4))
            }
        }

        @Test func testSelectedLatentAttentionMatchesIndependentExpandedKVOracle() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = IndexFixture(batch: 1, history: 13, queries: 5, start: 8)
                let selection = try fixture.select(budget: 4)
                let expectedIndices = fixture.oracle(budget: 4)
                #expect(selection.indices.asArray(Int32.self) == expectedIndices)
                let q = (0 ..< 30).map { (index: Int) -> Float in Float(index % 7 - 3) * 0.1 }
                let latent = (0 ..< 26).map { (index: Int) -> Float in
                    Float(index / 2) * 0.2 + Float(index % 2) * 0.3
                }
                let wk: [Float] = [0.4, -0.2, 0.1, 0.3, -0.5, 0.2, 0.2, 0.1, -0.1, 0.4, 0.3, 0.2]
                let wv: [Float] = [0.5, -0.2, 0.1, 0.7, -0.3, 0.4, 0.6, 0.2]
                let output = try GLM5NextSparseAttention.attend(
                    queries: MLXArray(q, [1, 5, 2, 3]), latents: MLXArray(latent, [1, 13, 2]),
                    selection: selection, keyProjection: MLXArray(wk, [2, 3, 2]),
                    valueProjection: MLXArray(wv, [2, 2, 2]))
                let expected = attentionOracle(
                    queries: q, latents: latent, indices: expectedIndices, keyWeight: wk,
                    valueWeight: wv,
                    batch: 1, queryCount: 5, history: 13, heads: 2,
                    queryDimension: 3, latentDimension: 2, valueDimension: 2, width: 7)
                expectClose(output, expected)

                // This is not an ordinary full-history attention fallback.
                var denseIndices = [Int32]()
                for position in 8 ... 12 {
                    denseIndices +=
                        (0 ... position).map(Int32.init)
                        + Array(repeating: -1, count: 12 - position)
                }
                let dense = attentionOracle(
                    queries: q, latents: latent, indices: denseIndices, keyWeight: wk,
                    valueWeight: wv,
                    batch: 1, queryCount: 5, history: 13, heads: 2,
                    queryDimension: 3, latentDimension: 2, valueDimension: 2, width: 13)
                #expect((zip(expected, dense).map { abs($0.0 - $0.1) }.max() ?? 0) > 0.01)
            }
        }

        @Test func testReal256DimensionScaleIsOneSixteenthNotLatentScale() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = IndexFixture(batch: 1, history: 2, queries: 1, start: 1)
                let selection = try fixture.select(budget: 4)
                var query = [Float](repeating: 0, count: 256)
                query[0] = 1
                var keyWeight = [Float](repeating: 0, count: 512)
                keyWeight[0] = 1
                for dtype in [DType.float32, .float16, .bfloat16] {
                    let result = try GLM5NextSparseAttention.attend(
                        queries: MLXArray(query, [1, 1, 1, 256]).asType(dtype),
                        latents: MLXArray([Float(0), 0, 16, 0], [1, 2, 2]).asType(dtype),
                        selection: selection,
                        keyProjection: MLXArray(keyWeight, [1, 256, 2]).asType(dtype),
                        valueProjection: MLXArray([Float(1), 0], [1, 1, 2]).asType(dtype))
                    #expect(result.dtype == dtype)
                    #expect(result.shape == [1, 1, 1, 1])
                    let expected = 16 * Foundation.exp(1.0) / (1 + Foundation.exp(1.0))
                    let rounded = MLXArray([Float(expected)]).asType(dtype).asType(.float32).item(
                        Float.self)
                    #expect(abs(result.asType(.float32).item(Float.self) - rounded) <= 0.00001)
                }
            }
        }

        @Test func testExplicitHistoryContinuationMatchesWholePrefillAcrossPoolBoundaries() throws {
            try Device.withDefaultDevice(.cpu) {
                let whole = IndexFixture(batch: 1, history: 13, queries: 13, start: 0)
                let fullSelection = try whole.select(budget: 8)
                let full = fullSelection.indices.asArray(Int32.self)
                let mainQueries = (0 ..< 26).map { Float($0 % 3 + 1) * 0.1 }
                let wk = MLXArray([Float(0.3), 0.4, -0.2, 0.7], [1, 2, 2])
                let wv = MLXArray([Float(0.8), -0.1], [1, 1, 2])
                let fullOutput = try GLM5NextSparseAttention.attend(
                    queries: MLXArray(mainQueries, [1, 13, 1, 2]),
                    latents: MLXArray(whole.keys, [1, 13, 2]), selection: fullSelection,
                    keyProjection: wk, valueProjection: wv
                ).asArray(Float.self)
                var previous = 0
                var joined = [Int32]()
                var joinedOutput = [Float]()
                for end in [3, 4, 5, 8, 13] {
                    var chunk = whole
                    chunk.history = end
                    chunk.queryCount = end - previous
                    chunk.start = previous
                    chunk.keys = Array(whole.keys.prefix(end * 2))
                    chunk.gates = Array(whole.gates.prefix(end * 2))
                    chunk.keyValid = Array(whole.keyValid.prefix(end))
                    chunk.queries = Array(whole.queries[previous * 4 ..< end * 4])
                    chunk.weights = Array(whole.weights[previous * 2 ..< end * 2])
                    chunk.queryValid = Array(whole.queryValid[previous ..< end])
                    let selected = try chunk.select(budget: 8)
                    let result = selected.indices.asArray(Int32.self)
                    #expect(result == chunk.oracle(budget: 8))
                    joined += result
                    let output = try GLM5NextSparseAttention.attend(
                        queries: MLXArray(
                            Array(mainQueries[previous * 2 ..< end * 2]), [1, end - previous, 1, 2]),
                        latents: MLXArray(chunk.keys, [1, end, 2]), selection: selected,
                        keyProjection: wk, valueProjection: wv)
                    joinedOutput += output.asArray(Float.self)
                    previous = end
                }
                #expect(joined == full)
                for (actual, expected) in zip(joinedOutput, fullOutput) {
                    #expect(abs(actual - expected) <= 0.00001)
                }
            }
        }

        @Test func testMaximumBudgetExpands512PoolsAndThreeTailTokens() throws {
            try Device.withDefaultDevice(.cpu) {
                var fixture = IndexFixture(batch: 1, history: 2055, queries: 1, start: 2054)
                fixture.queries = [1, 0, 1, 0]
                fixture.keys = (0 ..< 2055).flatMap { [Float($0), 0] }
                fixture.gates = Array(repeating: 0, count: 4110)
                fixture.bias = Array(repeating: 0, count: 8)
                fixture.weights = [1, 1]
                let selection = try fixture.select(budget: 2048)
                let actual = selection.indices.asArray(Int32.self)
                #expect(selection.indices.shape == [1, 1, 2051])
                let expected: [Int32] = (4 ..< 2055).map(Int32.init)
                #expect(actual == expected)
                #expect(Set(actual).count == 2051)
            }
        }

        @Test func testInvalidGeometryFailsBeforeNativeOperations() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = IndexFixture(batch: 1, history: 5, queries: 1, start: 4)
                for budget in [0, 3, 5, 2052, Int.max] {
                    #expect(throws: GLM5NextSparseAttentionError.invalidInput("raw token budget")) {
                        try fixture.select(budget: budget)
                    }
                }
                var future = fixture
                future.start = 5
                #expect(throws: GLM5NextSparseAttentionError.invalidInput("query positions")) {
                    try future.select(budget: 4)
                }
                var empty = fixture
                empty.history = 0
                empty.keys = []
                empty.gates = []
                empty.keyValid = []
                #expect(throws: GLM5NextSparseAttentionError.invalidInput("index history length")) {
                    try empty.select(budget: 4)
                }
                let selection = try fixture.select(budget: 4)
                #expect(throws: GLM5NextSparseAttentionError.invalidInput("latent history shape")) {
                    try GLM5NextSparseAttention.attend(
                        queries: MLXArray.ones([1, 1, 2, 3]), latents: MLXArray.ones([1, 4, 2]),
                        selection: selection, keyProjection: MLXArray.ones([2, 3, 2]),
                        valueProjection: MLXArray.ones([2, 2, 2]))
                }
                #expect(throws: GLM5NextSparseAttentionError.invalidInput("key projection shape")) {
                    try GLM5NextSparseAttention.attend(
                        queries: MLXArray.ones([1, 1, 2, 3]), latents: MLXArray.ones([1, 5, 2]),
                        selection: selection, keyProjection: MLXArray.ones([2, 2, 2]),
                        valueProjection: MLXArray.ones([2, 2, 2]))
                }
            }
        }

        private struct IndexFixture {
            var batch: Int
            var history: Int
            var queryCount: Int
            var start: Int
            var heads = 2
            let dimension = 2
            var queries: [Float]
            var keys: [Float]
            var gates: [Float]
            var bias: [Float]
            var weights: [Float]
            var keyValid: [Bool]
            var queryValid: [Bool]

            init(batch: Int, history: Int, queries: Int, start: Int) {
                self.batch = batch
                self.history = history
                self.queryCount = queries
                self.start = start
                self.queries = (0 ..< batch * queries * 4).map { Float(sin(Double($0 + 1) * 0.37)) }
                keys = (0 ..< batch * history * 2).map { Float(cos(Double($0 + 1) * 0.23)) }
                gates = (0 ..< batch * history * 2).map { Float(sin(Double($0 + 1) * 0.61)) }
                bias = (0 ..< 8).map { Float($0 % 3 - 1) * 0.4 }
                weights = (0 ..< batch * queries * 2).map { 1 + Float($0 % 5) * 0.1 }
                keyValid = Array(repeating: true, count: batch * history)
                queryValid = Array(repeating: true, count: batch * queries)
            }

            func select(budget: Int) throws -> GLM5NextSparseAttention.Selection {
                try GLM5NextSparseAttention.selectTokens(
                    indexQueries: MLXArray(queries, [batch, queryCount, heads, dimension]),
                    indexKeys: MLXArray(keys, [batch, history, dimension]),
                    poolGates: MLXArray(gates, [batch, history, dimension]),
                    poolPositionBias: MLXArray(bias, [4, dimension]),
                    headWeights: MLXArray(weights, [batch, queryCount, heads]),
                    keyValidity: MLXArray(keyValid, [batch, history]),
                    queryValidity: MLXArray(queryValid, [batch, queryCount]),
                    queryStart: start, tokenBudget: budget)
            }

            // Scalar Double implementation: enumerate causal complete pools, compute
            // channel softmax independently, rank weighted ReLU scores, then append
            // the incomplete visible tail. No MLX or production helper is called.
            func oracle(budget: Int) -> [Int32] {
                let width = budget + 3
                var output = [Int32](repeating: -1, count: batch * queryCount * width)
                for b in 0 ..< batch {
                    let first = (0 ..< history).first { keyValid[b * history + $0] } ?? history
                    for q in 0 ..< queryCount where queryValid[b * queryCount + q] {
                        let position = start + q
                        var scored = [(start: Int, score: Double)]()
                        for pool in 0 ..< (history + 3) / 4 {
                            let base = first + pool * 4
                            guard base + 3 <= position, base + 3 < history,
                                (base ..< base + 4).allSatisfy({ keyValid[b * history + $0] })
                            else { continue }
                            var pooled = [Double](repeating: 0, count: dimension)
                            for d in 0 ..< dimension {
                                let logits = (0 ..< 4).map {
                                    Double(gates[(b * history + base + $0) * dimension + d])
                                        + Double(bias[$0 * dimension + d])
                                }
                                let maximum = logits.max()!
                                let exps = logits.map { Foundation.exp($0 - maximum) }
                                let total = exps.reduce(0, +)
                                for j in 0 ..< 4 {
                                    pooled[d] +=
                                        exps[j] / total
                                        * Double(keys[(b * history + base + j) * dimension + d])
                                }
                            }
                            var score = 0.0
                            for h in 0 ..< heads {
                                var dot = 0.0
                                for d in 0 ..< dimension {
                                    dot +=
                                        Double(
                                            queries[
                                                ((b * queryCount + q) * heads + h) * dimension + d])
                                        * pooled[d]
                                }
                                score +=
                                    max(dot / sqrt(Double(dimension)), 0)
                                    * Double(weights[(b * queryCount + q) * heads + h])
                                    / sqrt(Double(heads))
                            }
                            scored.append((base, score))
                        }
                        scored.sort {
                            $0.score == $1.score ? $0.start < $1.start : $0.score > $1.score
                        }
                        let chosen = scored.prefix(budget / 4).sorted { $0.start < $1.start }
                        var members = [Int]()
                        for pool in chosen { members += Array(pool.start ..< pool.start + 4) }
                        let offset = (b * queryCount + q) * width
                        for (slot, member) in members.enumerated() {
                            output[offset + slot] = Int32(member)
                        }
                        let visible = (0 ... position).filter { keyValid[b * history + $0] }
                        let tailCount = visible.count % 4
                        let tailStart = first + visible.count - tailCount
                        for j in 0 ..< tailCount {
                            let member = tailStart + j
                            if member <= position && member < history
                                && keyValid[b * history + member]
                                && !members.contains(member)
                            {
                                output[offset + budget + j] = Int32(member)
                            }
                        }
                    }
                }
                return output
            }
        }

        private func attentionOracle(
            queries: [Float], latents: [Float], indices: [Int32], keyWeight: [Float],
            valueWeight: [Float],
            batch: Int, queryCount: Int, history: Int, heads: Int,
            queryDimension: Int, latentDimension: Int, valueDimension: Int, width: Int
        ) -> [Double] {
            var output = [Double]()
            for b in 0 ..< batch {
                for q in 0 ..< queryCount {
                    let selected = indices[
                        (b * queryCount + q) * width ..< (b * queryCount + q + 1) * width
                    ]
                    .filter { $0 >= 0 }.map(Int.init)
                    for h in 0 ..< heads {
                        var scores = [Double]()
                        var values = [[Double]]()
                        for token in selected {
                            // Expand the selected key and value BEFORE the dot
                            // product, independent of production's absorbed algebra.
                            var key = [Double](repeating: 0, count: queryDimension)
                            var value = [Double](repeating: 0, count: valueDimension)
                            for c in 0 ..< latentDimension {
                                let latent = Double(
                                    latents[(b * history + token) * latentDimension + c])
                                for d in 0 ..< queryDimension {
                                    key[d] +=
                                        latent
                                        * Double(
                                            keyWeight[
                                                (h * queryDimension + d) * latentDimension + c])
                                }
                                for d in 0 ..< valueDimension {
                                    value[d] +=
                                        latent
                                        * Double(
                                            valueWeight[
                                                (h * valueDimension + d) * latentDimension + c])
                                }
                            }
                            var score = 0.0
                            for d in 0 ..< queryDimension {
                                score +=
                                    key[d]
                                    * Double(
                                        queries[
                                            ((b * queryCount + q) * heads + h) * queryDimension + d]
                                    )
                            }
                            scores.append(score / sqrt(Double(queryDimension)))
                            values.append(value)
                        }
                        guard let maximum = scores.max() else {
                            output += Array(repeating: 0, count: valueDimension)
                            continue
                        }
                        let exponentials = scores.map { Foundation.exp($0 - maximum) }
                        let denominator = exponentials.reduce(0, +)
                        for d in 0 ..< valueDimension {
                            output.append(
                                zip(exponentials, values).reduce(0) { $0 + $1.0 * $1.1[d] }
                                    / denominator)
                        }
                    }
                }
            }
            return output
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: [Double],
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let values = actual.asType(.float32).asArray(Float.self)
            #expect(values.count == expected.count, sourceLocation: sourceLocation)
            for (value, reference) in zip(values, expected) {
                #expect(abs(Double(value) - reference) <= 0.00001, sourceLocation: sourceLocation)
            }
        }
    }
}
