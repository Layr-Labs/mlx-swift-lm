import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Tests of the compact selected-KV planner in `Qwen4ExpCompactQSA.swift`.
    ///
    /// `tokenIndices` turns the selected 4-token blocks of each query into
    /// the 2051 physical KV rows that the Steel attention loop reads:
    /// 512 blocks of 4 rows, then the 3 newest rows after the last full
    /// block. Rows that the query may not read are -1. The expected rows
    /// come from the same rules, computed here with plain Swift integers.
    ///
    /// `attend` needs a selected-KV cache with more than 2051 tokens and
    /// the native sparse attention kernel; the CBv2 Qwen4 engine tests
    /// cover it.
    @Suite
    struct Qwen4ExpCompactQSATests {

        static func expectedRows(selected: [[Int32]], offset: Int, keyTokens: Int) -> [Int32] {
            var rows: [Int32] = []
            for (query, blocks) in selected.enumerated() {
                let position = offset + query
                let count = (position + 1) / 4
                for (slot, block) in blocks.enumerated() {
                    for lane in 0 ..< 4 {
                        let row = Int(block) * 4 + lane
                        let valid =
                            slot < min(count, 512) && row >= 0 && row < keyTokens
                            && row <= position
                        rows.append(valid ? Int32(row) : -1)
                    }
                }
                for lane in 0 ..< 3 {
                    let row = count * 4 + lane
                    rows.append(row < keyTokens && row <= position ? Int32(row) : -1)
                }
            }
            return rows
        }

        /// Two queries at positions 9 and 10 of an 11-token cache. Each has
        /// two complete 4-token blocks, so only the first two slots can be
        /// valid. The blocks include a negative id, an id past the cache
        /// and an id that ends after the query position.
        @Test func tokenIndicesFollowTheSlotRules() {
            let offset = 9
            let keyTokens = 11
            var selected: [[Int32]] = []
            for query in 0 ..< 2 {
                var blocks = [Int32](repeating: 0, count: 512)
                blocks[0] = query == 0 ? 1 : 2
                blocks[1] = query == 0 ? -1 : 0
                blocks[2] = 1
                blocks[3] = 7
                selected.append(blocks)
            }
            let array = MLXArray(selected.flatMap { $0 }, [1, 2, 512])
            let actual = Qwen4ExpCompactQSA.tokenIndices(
                selected: array, offset: offset, keyTokens: keyTokens)
            #expect(actual.shape == [2 * Qwen4ExpCompactQSA.slotsPerQuery])
            let expected = Self.expectedRows(
                selected: selected, offset: offset, keyTokens: keyTokens)
            #expect(actual.asArray(Int32.self) == expected)
            // Spot checks of the rules for query 1 (position 10).
            let base = Qwen4ExpCompactQSA.slotsPerQuery
            let values = actual.asArray(Int32.self)
            #expect(Array(values[base ..< base + 8]) == [8, 9, 10, -1, 0, 1, 2, 3])
            #expect(Array(values[(base + 2048) ..< (base + 2051)]) == [8, 9, 10])
        }

        /// A long context: every one of the 512 slots is valid when the
        /// query has seen at least 2048 complete rows.
        @Test func tokenIndicesForALongContext() {
            let offset = 4100
            let keyTokens = 4101
            let blocks = (0 ..< 512).map { Int32(($0 * 7) % 1025) }
            let actual = Qwen4ExpCompactQSA.tokenIndices(
                selected: MLXArray(blocks, [1, 1, 512]), offset: offset, keyTokens: keyTokens)
            let expected = Self.expectedRows(
                selected: [blocks], offset: offset, keyTokens: keyTokens)
            #expect(actual.asArray(Int32.self) == expected)
            #expect(!expected[0 ..< 2048].contains(-1))
        }

        @Test func eligibilityAndKillSwitch() {
            #expect(Qwen4ExpCompactQSA.slotsPerQuery == 2051)
            #expect(Qwen4ExpCompactQSA.enabled(environment: [:]))
            #expect(Qwen4ExpCompactQSA.enabled(environment: [Qwen4ExpCompactQSA.envFlag: " yes "]))
            #expect(!Qwen4ExpCompactQSA.enabled(environment: [Qwen4ExpCompactQSA.envFlag: "0"]))
            // The decode crosses the 2048-token budget when
            // (offset + 1) / 4 > 512.
            #expect(!Qwen4ExpCompactQSA.eligible(offset: 2050, width: 1))
            #expect(Qwen4ExpCompactQSA.eligible(offset: 2051, width: 1))
            #expect(Qwen4ExpCompactQSA.eligible(offset: 2051, width: 6))
            #expect(!Qwen4ExpCompactQSA.eligible(offset: 2051, width: 7))
            #expect(!Qwen4ExpCompactQSA.eligible(offset: 2051, width: 0))
            // A sparsity win needs width * 2051 < offset + width.
            #expect(!Qwen4ExpCompactQSA.isSparsityWin(offset: 2050, width: 1))
            #expect(Qwen4ExpCompactQSA.isSparsityWin(offset: 2051, width: 1))
            #expect(!Qwen4ExpCompactQSA.isSparsityWin(offset: 10250, width: 5))
            #expect(Qwen4ExpCompactQSA.isSparsityWin(offset: 10251, width: 5))
        }
    }
}
