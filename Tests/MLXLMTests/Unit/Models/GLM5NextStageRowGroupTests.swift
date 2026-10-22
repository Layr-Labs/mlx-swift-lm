// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Testing

@testable import MLXLLM

extension UnitTests {
    @Suite struct GLM5NextStageRowGroupTests {
        @Test func groupsShareOneWindowLengthWithinTheRowBound() {
            #expect(GLM5NextTextStage.rowGroups(lengths: [3, 3, 3, 3, 3], maximumRows: 8) == [
                [0, 1], [2, 3], [4],
            ])
            #expect(GLM5NextTextStage.rowGroups(lengths: [3, 2, 3, 1, 1, 2], maximumRows: 8) == [
                [0, 2], [1, 5], [3], [4],
            ])
            #expect(GLM5NextTextStage.rowGroups(lengths: [2, 2, 2, 2, 2], maximumRows: 8) == [
                [0, 1, 2, 3], [4],
            ])
            #expect(GLM5NextTextStage.rowGroups(lengths: [8, 5, 5], maximumRows: 8) == [[0], [1], [2]])
            #expect(GLM5NextTextStage.rowGroups(lengths: [3, 3], maximumRows: 5) == [[0], [1]])
            #expect(GLM5NextTextStage.rowGroups(lengths: [], maximumRows: 8).isEmpty)
        }

        /// Routed-expert launches of a shared-site rows step: a one-token lane
        /// alone, runs of multi-token groups in balanced chunks of 2...8 rows.
        @Test func expertChunksKeepOneTokenLanesAloneAndBalanceRuns() {
            typealias Group = (rows: Range<Int>, shape: [Int])
            func chunks(_ groups: [Group]) -> [Range<Int>] { GLM5NextTextLayer.expertChunks(groups) }
            #expect(chunks([(0 ..< 6, [2, 3]), (6 ..< 9, [1, 3])]) == [0 ..< 4, 4 ..< 9])
            #expect(chunks([(0 ..< 6, [2, 3]), (6 ..< 7, [1, 1]), (7 ..< 13, [2, 3])]) == [
                0 ..< 6, 6 ..< 7, 7 ..< 13,
            ])
            #expect(chunks([(0 ..< 3, [1, 3])]) == [0 ..< 3])
            #expect(chunks([(0 ..< 1, [1, 1]), (1 ..< 2, [1, 1])]) == [0 ..< 1, 1 ..< 2])
            let wide = chunks((0 ..< 8).map { ($0 * 3 ..< $0 * 3 + 3, [1, 3]) })
            #expect(wide.count == 3 && wide.allSatisfy { (2 ... 8).contains($0.count) })
            #expect(wide.first?.lowerBound == 0 && wide.last?.upperBound == 24)
        }
    }
}
