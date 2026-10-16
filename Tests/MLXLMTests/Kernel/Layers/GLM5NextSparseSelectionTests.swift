// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextSparseSelectionTests {
        private static var devices: [DeviceType] {
            #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
                [.cpu, .gpu]
            #else
                [.cpu]
            #endif
        }

        private func fence(_ arrays: [MLXArray]) throws {
            try MLX.withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        private func selected(
            _ rows: [[Float]], complete: [Bool]? = nil, count: Int,
            dtype: DType = .float32, strided: Bool = false
        ) throws -> [[Int32]] {
            let batch = rows.count
            let pools = rows[0].count
            let values = rows.flatMap { $0 }
            let mask = complete ?? Array(repeating: true, count: batch * pools)
            let originalScores = MLXArray(values, [batch, pools]).asType(dtype)
            let originalMask = MLXArray(mask, [batch, pools])
            let scores =
                strided
                ? stacked([originalScores, originalScores], axis: -1)
                    .reshaped(batch, 2 * pools)[0..., .stride(by: 2)]
                : originalScores
            let completeness =
                strided
                ? stacked([originalMask, originalMask], axis: -1)
                    .reshaped(batch, 2 * pools)[0..., .stride(by: 2)]
                : originalMask
            let result = try GLM5NextSparseSelection.select(
                scores: scores, completePools: completeness, requestedCount: count)
            try fence([result, scores, completeness])
            #expect(result.dtype == .int32)
            #expect(result.shape == [batch, min(count, pools)])
            #expect(
                scores.asType(.float32).asArray(Float.self).map(\.bitPattern)
                    == originalScores.asType(.float32).asArray(Float.self).map(\.bitPattern))
            #expect(completeness.asArray(Bool.self) == mask)
            let ids = result.asArray(Int32.self).map { $0 == Int32(pools) ? Int32(-1) : $0 }
            let width = result.dim(1)
            return (0 ..< batch).map { Array(ids[($0 * width) ..< (($0 + 1) * width)]).sorted() }
        }

        @Test(arguments: devices)
        func pinnedScoreOnlyTieMembership(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                let minimum = -Float.greatestFiniteMagnitude
                let cases: [([Float], [Int32])] = [
                    ([0, 0, 0, 1, 0, minimum], [1, 3]),
                    ([-2, 1, -3, 0, 0, -1, minimum, minimum], [1, 4]),
                    ([0, -3, -2, -1, -2, 0, 0, minimum], [0, 6]),
                    ([0, -3, -2, -1, -2, 0, 0, 0], [0, 6]),
                    // These two rows rule out a static index-priority tie fix.
                    ([0, -1, 0, 1, -1, -1], [2, 3]),
                    ([0, 1, 0, -1, -1, -1], [0, 1]),
                ]
                for (scores, expected) in cases {
                    #expect(try selected([scores], count: 2) == [expected])
                }
            }
        }

        @Test(arguments: devices)
        func trimsOnlyGloballyIncompletePoolsAndPreservesOriginalIDs(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                let minimum = -Float.greatestFiniteMagnitude
                let scores: [Float] = [-2, 1, -3, 0, 0, -1, minimum, minimum]
                #expect(try selected([scores], count: 2) == [[1, 4]])
                #expect(
                    try selected(
                        [scores], complete: [true, true, true, true, true, true, true, false],
                        count: 2) == [[1, 3]])
                // Row 1 keeps the future columns in row 0's candidate axis.
                let mask =
                    [true, true, true, true, true, true, false, false]
                    + [false, false, false, false, false, false, true, true]
                let other: [Float] = [minimum, minimum, minimum, minimum, minimum, minimum, 1, 2]
                #expect(
                    try selected([scores, other], complete: mask, count: 2) == [[1, 4], [6, 7]])
                #expect(
                    try selected(
                        [[9, 8, 7, 6]], complete: [false, true, false, true],
                        count: 2) == [[1, 3]])
                #expect(try selected([[1, 2]], complete: [false, false], count: 2) == [[-1, -1]])
                #expect(try selected([[1, 2]], complete: [false, true], count: 2) == [[-1, 1]])
            }
        }

        @Test(arguments: devices)
        func algorithmBoundariesMaximumCountAndShortAxes(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                for pools in [1, 2, 3, 7, 8, 127, 128, 129] {
                    let count = min(pools, 2)
                    #expect(
                        try selected([Array(repeating: 0, count: pools)], count: count)
                            == [Array(0 ..< Int32(count))])
                    let ascending = (0 ..< pools).map(Float.init)
                    #expect(
                        try selected([ascending], count: count)
                            == [Array(Int32(pools - count) ..< Int32(pools))])
                }
                // 64*K is a semantic branch: exercise both sides at full K.
                for pools in [32767, 32768] {
                    #expect(
                        try selected([Array(repeating: 0, count: pools)], count: 512)
                            == [Array(0 ..< Int32(512))])
                    #expect(
                        try selected([(0 ..< pools).map(Float.init)], count: 512)
                            == [Array(Int32(pools - 512) ..< Int32(pools))])
                }
                // Equality at the partial-sort branch retains the initial heap.
                let lateWinner = Array(repeating: Float(0), count: 127) + [1]
                #expect(try selected([lateWinner], count: 2) == [[0, 127]])
                var retainedBoundary = Array(repeating: Float(0), count: 128)
                retainedBoundary[126] = 1
                #expect(try selected([retainedBoundary], count: 2) == [[0, 126]])
                #expect(
                    try selected(
                        [retainedBoundary],
                        complete: Array(repeating: true, count: 127) + [false], count: 2)
                        == [[1, 126]])
            }
        }

        @Test(arguments: devices)
        func floatingDtypesStridesAndIndependentCalls(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                let scores: [[Float]] = [[0, 0, 0, 1, 0, -8], [0, -3, -2, -1, -2, 0]]
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    let first = try selected(scores, count: 2, dtype: dtype, strided: true)
                    _ = try selected([[3, 2, 1, 0, -1, -2]], count: 4, dtype: dtype)
                    #expect(first == [[1, 3], [0, 5]])
                    #expect(try selected(scores, count: 2, dtype: dtype, strided: true) == first)
                    #expect(
                        try selected([[0, -0.0, 0, -0.0]], count: 2, dtype: dtype) == [[0, 1]])
                    #expect(
                        try selected([[0, -.infinity, .infinity, .nan]], count: 2, dtype: dtype)
                            == [[2, 3]])
                }
            }
        }

        @Test(arguments: devices)
        func generalPartitionAndHeapMembership(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                for pools in [9, 31, 64, 257] {
                    let rows = (0 ..< 3).map { row in
                        (0 ..< pools).map { Float(($0 * 7 + row * 11) % pools) }
                    }
                    for count in [1, 2, 3, 8] {
                        let expected = rows.map { scores in
                            scores.indices.sorted { scores[$0] > scores[$1] }
                                .prefix(count).map(Int32.init).sorted()
                        }
                        #expect(try selected(rows, count: count) == expected)
                    }
                }
            }
        }

        @Test func taskLocalCPUStreamControlsDiagnosticExecution() throws {
            // The default device alone would choose GPU here. All operations
            // must instead follow the actual task-local CPU stream.
            try Device.withDefaultDevice(.gpu) { () throws -> Void in
                try Stream.withNewDefaultStream(device: .cpu) { () throws -> Void in
                    #expect(try selected([[0, 0, 0, 1, 0, -8]], count: 2) == [[1, 3]])
                }
            }
        }

        @Test(arguments: devices)
        func actualSparseSelectionKeepsFuturePoolsAndTail(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                // Six complete pools and one incomplete pool. Query 22 cannot
                // see pool 5, but it must remain in the six-column top-k input.
                let keys = (0 ..< 26).map { Float($0 / 4 == 3 ? 1 : 0) }
                let result = try GLM5NextSparseAttention.selectTokens(
                    indexQueries: .ones([1, 1, 1, 1]),
                    indexKeys: MLXArray(keys, [1, 26, 1]), poolGates: .zeros([1, 26, 1]),
                    poolPositionBias: .zeros([4, 1]), headWeights: .ones([1, 1, 1]),
                    keyValidity: .ones([1, 26], dtype: .bool),
                    queryValidity: .ones([1, 1], dtype: .bool), queryStart: 22, tokenBudget: 8)
                try fence([result.indices])
                #expect(
                    result.indices.asArray(Int32.self)
                        == [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22])

                // Eight complete pools, only six visible: removing the two
                // future columns would incorrectly select pools 1 and 3.
                let poolScores: [Float] = [-2, 1, -3, 0, 0, -1, 0, 0]
                let signedKeys = (0 ..< 32).flatMap { position -> [Float] in
                    let score = poolScores[position / 4]
                    return [max(score, 0), max(-score, 0)]
                }
                let future = try GLM5NextSparseAttention.selectTokens(
                    indexQueries: MLXArray([Float(1), 0, 0, 1], [1, 1, 2, 2]),
                    indexKeys: MLXArray(signedKeys, [1, 32, 2]), poolGates: .zeros([1, 32, 2]),
                    poolPositionBias: .zeros([4, 2]),
                    headWeights: MLXArray([Float(1), -1], [1, 1, 2]),
                    keyValidity: .ones([1, 32], dtype: .bool),
                    queryValidity: .ones([1, 1], dtype: .bool), queryStart: 25, tokenBudget: 8)
                try fence([future.indices])
                #expect(
                    future.indices.asArray(Int32.self)
                        == [4, 5, 6, 7, 16, 17, 18, 19, 24, 25, -1])
            }
        }

        @Test func validatesWorkspaceAndRefusesMalformedInputs() throws {
            for shape in [[1], [0, 8], [1, 0], [Int.max, 8], [65536, 65536]] {
                #expect(throws: GLM5NextSparseSelection.Failure.invalidGeometry) {
                    try GLM5NextSparseSelection.Layout(shape: shape, requestedCount: 2)
                }
            }
            for count in [0, -1, 513, Int.max] {
                #expect(throws: GLM5NextSparseSelection.Failure.invalidGeometry) {
                    try GLM5NextSparseSelection.Layout(shape: [1, 8], requestedCount: count)
                }
            }
            for pools in [1, 7, 8, 32767, 32768, 1_000_000] {
                let layout = try GLM5NextSparseSelection.Layout(
                    shape: [3, pools], requestedCount: 512)
                #expect(layout.scratchWidth <= 32767)
                #expect(layout.scratchBytes <= layout.previousIndexBytes)
                #expect(layout.scratchBytes == 3 * layout.scratchWidth * 4)
            }
            try Device.withDefaultDevice(.cpu) {
                #expect(throws: GLM5NextSparseSelection.Failure.invalidTensor) {
                    try GLM5NextSparseSelection.select(
                        scores: .ones([1, 8], dtype: .int32),
                        completePools: .ones([1, 8], dtype: .bool), requestedCount: 2)
                }
                #expect(throws: GLM5NextSparseSelection.Failure.invalidTensor) {
                    try GLM5NextSparseSelection.select(
                        scores: .ones([1, 8]),
                        completePools: .ones([2, 8], dtype: .bool), requestedCount: 2)
                }
            }
        }
    }
}
