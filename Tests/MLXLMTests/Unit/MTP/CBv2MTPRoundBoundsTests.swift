import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// The owner-wide MTP round bounds. The depth policy counts every
    /// planned decode row, target-only media rows included, so a plan can
    /// be wider than the speculative batch while at most the speculative
    /// batch drafts.
    struct CBv2MTPRoundBoundsTests {
        static func config(speculative: Int, depth: Int, table: [Int]?) -> CBv2MTPConfig {
            CBv2MTPConfig(
                enabled: true, maxDraftTokens: depth, maxSpeculativeBatch: speculative,
                fixedDraftTokens: depth, fixedDraftTokensByDecodeRows: table,
                verificationMode: .rectangularExact)
        }

        /// The bounds over widths up to the speculative batch only, every
        /// planned row drafting (`CBv2MTPConfig.fixedDraftTokensByDecodeRows`).
        static func batchOnly(_ config: CBv2MTPConfig, perRequest: Int)
            -> (positions: Int, steps: Int, generations: Int)
        {
            var bounds = (positions: 0, steps: 0, generations: 0)
            for rows in 1 ... config.maxSpeculativeBatch {
                var depth = config.fixedDraftTokens ?? config.maxDraftTokens
                if config.fixedDraftTokens != nil, let table = config.fixedDraftTokensByDecodeRows {
                    depth = min(depth, table[min(rows, table.count) - 1])
                }
                bounds.positions = max(bounds.positions, rows * (depth + 1))
                bounds.steps = max(bounds.steps, rows * depth)
                bounds.generations = max(
                    bounds.generations, 2 * rows * (depth + 1) - rows * perRequest)
            }
            return bounds
        }

        /// Two speculating rows beside one media row plan three decode rows,
        /// so `[1, 1, 3]` gives depth 3 and the two drafting rows verify 8
        /// positions in 6 draft steps, with 2 x 2 x 4 - 2 x 3 = 10 window
        /// generations.
        @Test func aPlanWiderThanTheBatchDraftsAtItsOwnDepth() {
            let config = Self.config(speculative: 2, depth: 3, table: [1, 1, 3])
            let bounds = CBv2MTPAllocationLimits.roundBounds(config)
            #expect(bounds.positions == 8)
            #expect(bounds.steps == 6)
            #expect(CBv2MTPAllocationLimits.capturedWindowGenerations(config, perRequest: 3) == 10)
            let batchOnly = Self.batchOnly(config, perRequest: 3)
            #expect(batchOnly.positions == 4 && batchOnly.steps == 2 && batchOnly.generations == 2)
        }

        /// Where no wider plan drafts deeper (one depth, a non-increasing
        /// table, a table no longer than the batch, the adaptive policy that
        /// ignores tables) the bounds are the batch's own, unchanged.
        @Test(arguments: [
            config(speculative: 1, depth: 3, table: nil),
            config(speculative: 8, depth: 3, table: nil),
            config(speculative: 4, depth: 3, table: [3, 2, 2, 2, 1]),
            config(speculative: 2, depth: 3, table: [3, 1, 1, 1]),
            config(speculative: 8, depth: 3, table: [3, 2, 0]),
            config(speculative: 8, depth: 2, table: [1, 2]),
            CBv2MTPConfig(
                enabled: true, maxDraftTokens: 3, maxSpeculativeBatch: 2,
                fixedDraftTokensByDecodeRows: [1, 1, 3], verificationMode: .rectangularExact),
        ])
        func boundsWithoutADeeperWiderPlanAreUnchanged(_ config: CBv2MTPConfig) {
            let bounds = CBv2MTPAllocationLimits.roundBounds(config)
            let batchOnly = Self.batchOnly(config, perRequest: 3)
            #expect(bounds.positions == batchOnly.positions)
            #expect(bounds.steps == batchOnly.steps)
            #expect(
                CBv2MTPAllocationLimits.capturedWindowGenerations(config, perRequest: 3)
                    == batchOnly.generations)
        }
    }
}
