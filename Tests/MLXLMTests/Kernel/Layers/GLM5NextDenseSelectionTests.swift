// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The every-pool shortcut must return exactly the per-query selector's
    /// indices whenever the budget covers every pool.
    @Suite(.serialized)
    struct GLM5NextDenseSelectionTests {
        private struct Case {
            let batch: Int
            let history: Int
            let queries: Int
            let leftPadding: [Int]
            let tokenBudget: Int
            let seed: UInt64
        }

        private func compare(_ value: Case) throws {
            try compare(value, reference: .reference, candidate: .reference, dense: true)
        }

        /// `candidate` (with the dense shortcut as given) against the
        /// per-query reference graph without it.
        private func compare(
            _ value: Case, reference: GLM5NextSparseAttention.Execution,
            candidate: GLM5NextSparseAttention.Execution, dense: Bool, blockBytes: Int? = nil
        ) throws {
            let key = MLXRandom.key(value.seed)
            let keys = MLXRandom.split(key: key, into: 5)
            let heads = 2
            let dimension = 8
            let queries = MLXRandom.normal(
                [value.batch, value.queries, heads, dimension], key: keys[0])
            let indexKeys = MLXRandom.normal([value.batch, value.history, dimension], key: keys[1])
            let gates = MLXRandom.normal([value.batch, value.history, dimension], key: keys[2])
            let bias = MLXRandom.normal([4, dimension], key: keys[3])
            let weights = MLXRandom.normal([value.batch, value.queries, heads], key: keys[4])
            let positions = MLXArray(0 ..< value.history).reshaped([1, value.history])
            let padding = MLXArray(value.leftPadding.map(Int32.init)).reshaped([value.batch, 1])
            let keyValidity = positions .>= padding
            let queryStart = value.history - value.queries
            let queryPositions = MLXArray(queryStart ..< value.history).reshaped([1, value.queries])
            let queryValidity = queryPositions .>= padding
            func select(
                _ dense: Bool, _ execution: GLM5NextSparseAttention.Execution,
                blockBytes: Int = GLM5NextSparseAttention.scoreBlockBytes
            ) throws -> [Int32] {
                let selection = try GLM5NextSparseAttention.selectTokens(
                    indexQueries: queries, indexKeys: indexKeys, poolGates: gates,
                    poolPositionBias: bias, headWeights: weights, keyValidity: keyValidity,
                    queryValidity: queryValidity, queryStart: queryStart,
                    tokenBudget: value.tokenBudget, execution: execution,
                    allowsDenseShortcut: dense, scoreBlockBytes: blockBytes)
                eval(selection.indices)
                #expect(selection.indices.dtype == .int32)
                #expect(
                    selection.indices.shape == [value.batch, value.queries, value.tokenBudget + 3])
                return selection.indices.asArray(Int32.self)
            }
            #expect(
                try select(
                    dense, candidate,
                    blockBytes: blockBytes ?? GLM5NextSparseAttention.scoreBlockBytes)
                    == select(false, reference),
                "case \(value) \(candidate)")
        }

        @Test(arguments: [DeviceType.cpu, .gpu])
        func everyPoolShortcutMatchesPerQuerySelection(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                let cases = [
                    Case(
                        batch: 1, history: 1, queries: 1, leftPadding: [0], tokenBudget: 4, seed: 1),
                    Case(
                        batch: 1, history: 7, queries: 7, leftPadding: [0], tokenBudget: 8, seed: 2),
                    Case(
                        batch: 1, history: 17, queries: 17, leftPadding: [0], tokenBudget: 20,
                        seed: 3),
                    Case(
                        batch: 1, history: 33, queries: 1, leftPadding: [0], tokenBudget: 36,
                        seed: 4),
                    Case(
                        batch: 2, history: 23, queries: 9, leftPadding: [0, 5], tokenBudget: 24,
                        seed: 5),
                    Case(
                        batch: 3, history: 40, queries: 40, leftPadding: [3, 0, 11],
                        tokenBudget: 64, seed: 6),
                    Case(
                        batch: 2, history: 12, queries: 4, leftPadding: [12, 2], tokenBudget: 12,
                        seed: 7),
                ]
                for value in cases { try compare(value) }
            }
        }

        /// The in-kernel membership of the dense-history softmax is exactly the
        /// selector's set (including validity holes, padded queries and batch
        /// rows), and its probabilities follow the reference masked softmax.
        @Test(arguments: [DeviceType.cpu, .gpu])
        func denseSelectionSoftmaxMatchesSelectorMembership(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                for (batch, history, queries, seed) in [
                    (1, 1, 1, 21), (2, 9, 9, 22), (3, 37, 5, 23), (2, 64, 64, 24), (1, 2048, 2, 25),
                ] {
                    let key = MLXRandom.key(UInt64(seed))
                    let keys = MLXRandom.split(key: key, into: 3)
                    // Left padding plus random holes; one row fully valid.
                    var validity = [Bool]()
                    for row in 0 ..< batch {
                        let padding = row == 0 ? 0 : (seed * 7 + row * 3) % max(1, history / 2)
                        for n in 0 ..< history {
                            let hole = row == 2 && (n * 13 + seed) % 7 == 0
                            validity.append(n >= padding && !hole)
                        }
                    }
                    let keyValidity = MLXArray(validity, [batch, history])
                    let queryStart = history - queries
                    let queryValidity = keyValidity[0..., queryStart...]
                    let selection = try GLM5NextSparseAttention.selectTokens(
                        indexQueries: MLXRandom.normal([batch, queries, 2, 8], key: keys[0]),
                        indexKeys: MLXArray.zeros([batch, history, 8]),
                        poolGates: MLXArray.zeros([batch, history, 8]),
                        poolPositionBias: MLXArray.zeros([4, 8]),
                        headWeights: MLXArray.ones([batch, queries, 2]), keyValidity: keyValidity,
                        queryValidity: queryValidity, queryStart: queryStart, tokenBudget: 2048)
                    let ids = selection.indices
                    let member = putAlong(
                        MLXArray.zeros([batch, queries, history + 1], dtype: .bool),
                        MLX.where(ids .>= 0, ids, Int32(history)), values: MLXArray(true),
                        axis: -1)[.ellipsis, ..<history]
                    let heads = 3
                    let score =
                        MLXRandom.normal([batch, heads, queries, history], key: keys[1]) * 4
                    let scale: Float = 0.0625
                    let mask = member.expandedDimensions(axis: 1)
                    let masked = MLX.where(mask, score * scale, -Float.infinity)
                    let maximum = MLX.where(
                        mask.any(axis: -1, keepDims: true), masked.max(axis: -1, keepDims: true), 0)
                    let exps = MLX.where(mask, exp(masked - maximum), 0)
                    let total = exps.sum(axis: -1, keepDims: true)
                    let expected = (exps / MLX.where(total .> 0, total, 1)).asType(.bfloat16)
                        .asType(.float32)
                    let actual = try GLM5NextDenseSelectionKernel.maskedSoftmax(
                        score: score, keyValidity: keyValidity, queryValidity: queryValidity,
                        queryStart: queryStart, scale: scale, roundingType: .bfloat16)
                    let operations = GLM5NextDenseSelectionKernel.membership(
                        keyValidity: keyValidity, queryValidity: queryValidity,
                        queryStart: queryStart)
                    eval(expected, actual, operations)
                    let label = "B\(batch) N\(history) Q\(queries) \(device)"
                    #expect(
                        operations.asArray(Bool.self) == member.asArray(Bool.self),
                        "\(label) operation membership")
                    #expect(
                        (actual .> 0).asArray(Bool.self)
                            == broadcast(mask, to: actual.shape).asArray(Bool.self),
                        "\(label) membership")
                    let error = abs(actual - expected).max().item(Float.self)
                    #expect(error <= 1.0 / 128, "\(label) probability \(error)")
                }
            }
        }

        /// Query blocks select the per-query membership whenever no tie straddles
        /// the boundary (random scores), past the every-pool regime, for both
        /// selector strategies and with forced multi-block schedules.
        @Test(arguments: [DeviceType.cpu, .gpu])
        func queryBlocksMatchPerQuerySelection(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                let cases = [
                    Case(
                        batch: 1, history: 40, queries: 40, leftPadding: [0], tokenBudget: 8,
                        seed: 11),
                    Case(
                        batch: 2, history: 57, queries: 19, leftPadding: [0, 6], tokenBudget: 12,
                        seed: 12),
                    Case(
                        batch: 3, history: 64, queries: 1, leftPadding: [2, 0, 9], tokenBudget: 16,
                        seed: 13),
                    Case(
                        batch: 2, history: 33, queries: 33, leftPadding: [33, 1], tokenBudget: 4,
                        seed: 14),
                ]
                let strategies: [GLM5NextSparseAttention.SelectionStrategy] =
                    device == .gpu ? [.reference, .parallelThreshold] : [.reference]
                for value in cases {
                    for strategy in strategies {
                        let reference = GLM5NextSparseAttention.Execution(selection: strategy)
                        let blocked = GLM5NextSparseAttention.Execution(
                            selection: strategy, deviceRanges: true, queryBlocks: true)
                        try compare(value, reference: reference, candidate: blocked, dense: false)
                        try compare(value, reference: reference, candidate: blocked, dense: true)
                        // One byte forces one query per block; 2 KiB a few per block.
                        for bytes in [1, 2048] {
                            try compare(
                                value, reference: reference, candidate: blocked, dense: false,
                                blockBytes: bytes)
                        }
                    }
                }
            }
        }
    }
}
