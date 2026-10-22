// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Opt-in (`GLM53_SELECTION_WIDTH=1`): the DSA selector's choice for one
    /// row whose 512th score is an exact tie, with the same eligible scores
    /// and different counts of trailing masked (ineligible) pools.
    @Suite(.serialized)
    struct GLM5NextSelectionWidthTests {
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_SELECTION_WIDTH"] == "1"))
        func tiedBoundaryAgainstWidth() throws {
            try Device.withDefaultDevice(.gpu) {
                let eligible = 4705
                let pattern = ProcessInfo.processInfo.environment["GLM53_SELECTION_PATTERN"] ?? "zeros"
                var values = [Float](repeating: 0, count: eligible)
                if pattern == "zeros" {
                    // 300 distinct positive scores, the rest exactly 0 (a tie straddling 512).
                    for i in 0 ..< 300 { values[(i * 7919) % eligible] = Float(i + 1) * 0.001 }
                } else {
                    // Coarse values: many exact ties among positive scores.
                    var x: UInt64 = 88172645463325252
                    for i in 0 ..< eligible {
                        x ^= x << 13; x ^= x >> 7; x ^= x << 17
                        values[i] = Float(x % 64) * 0.125
                    }
                }
                func choose(_ width: Int, _ strategy: GLM5NextSparseAttention.SelectionStrategy) throws -> [Int32] {
                    let row = values + [Float](repeating: -Float.greatestFiniteMagnitude, count: width - eligible)
                    let scores = MLXArray(row, [1, width])
                    let complete = MLXArray.ones([1, width], dtype: .bool)
                    let chosen = try GLM5NextSparseSelection.select(
                        scores: scores, completePools: complete, requestedCount: 512, strategy: strategy)
                    return sorted(chosen, axis: 1).asArray(Int32.self)
                }
                for strategy in [GLM5NextSparseAttention.SelectionStrategy.parallelThreshold, .reference] {
                    let widths = [4705, 5120, 6144, 6656, 7168]
                    let picks = try widths.map { try choose($0, strategy) }
                    for (width, pick) in zip(widths, picks) {
                        let same = pick == picks[1]
                        print("GLM53_SELECTION_WIDTH strategy=\(strategy) width=\(width) same_as_5120=\(same ? 1 : 0) first=\(pick.prefix(3)) last=\(pick.suffix(3))")
                    }
                }
            }
        }
    }
}
