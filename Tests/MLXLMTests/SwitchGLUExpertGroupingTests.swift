import Foundation
import MLX
import MLXLMCommon
import Testing

/// The expert-grouping decision, tested as what it is: arithmetic on two
/// counts. Nothing here builds a module, launches a kernel, or evaluates an
/// array, so the whole suite runs while a GPU window is busy.
///
/// The exactness question this change raises — grouped and per-row gathers
/// producing the same tokens — is not testable here and must not be faked
/// here. It is a measured gate on the arm (see the report), because it is a
/// question about two MLX kernels, not about this predicate.
@Suite
struct SwitchGLUExpertGroupingTests {
    private func shouldGroup(rows: Int, slots: Int, union: Bool = true) -> Bool {
        SwitchGLUExpertGrouping.shouldGroup(
            assignments: rows * slots, rows: rows, unionAcrossRows: union)
    }

    @Test("one row is never grouped, so ordinary decode is untouched")
    func singleRowIsNeverGrouped() {
        // A single row has no collisions to exploit: every expert it selects
        // is read once either way. Grouping it would be pure permutation
        // overhead, and it is the shape production decode runs at.
        for slots in [1, 2, 4, 8, 16, 32, 63] {
            #expect(!shouldGroup(rows: 1, slots: slots))
            #expect(!shouldGroup(rows: 1, slots: slots, union: false))
        }
        // Except where the historic rule already grouped it, which is
        // preserved bit for bit.
        #expect(shouldGroup(rows: 1, slots: 64))
        #expect(shouldGroup(rows: 1, slots: 64, union: false))
    }

    @Test("an MTP verify groups from width 2 up, at every width")
    func rectangularVerifyGroupsAtEveryWidth() {
        // A [1, 1+k] verify at batch 1 with top-8 routing produces
        // (1+k) * 8 assignments. The historic rule put widths 1...7 below its
        // threshold and width 8 exactly on it, so the verify changed
        // algorithm at one width — a discontinuity keyed on a constant that
        // means "this is a prompt pass".
        for width in 1...8 {
            let grouped = shouldGroup(rows: width, slots: 8)
            #expect(grouped == (width >= 2))
            let historic = shouldGroup(rows: width, slots: 8, union: false)
            #expect(historic == (width >= 8))
        }
        // With the switch on there is no width at which the algorithm
        // changes above 1; with it off the old cliff at 8 is back exactly.
        let engaged = (1...8).map { shouldGroup(rows: $0, slots: 8) }
        #expect(engaged == [false, true, true, true, true, true, true, true])
    }

    @Test("the decision is a function of counts only — no length may enter it")
    func decisionDependsOnCountsAlone() {
        // The signature is the guard: there is nowhere to put a prompt
        // length, a context size, a KV window or a chunk size. A future
        // length-keyed branch would have to change it to compile.
        for rows in [1, 2, 5, 8, 64, 4096] {
            for slots in [1, 8, 128] {
                let first = shouldGroup(rows: rows, slots: slots)
                for _ in 0..<3 {
                    #expect(shouldGroup(rows: rows, slots: slots) == first)
                }
            }
        }
        // Grouping is monotone in rows at a fixed width: a bigger pass never
        // loses the grouping a smaller one had. Degradation across shapes is
        // one-directional, not a band.
        for slots in [1, 2, 8] {
            var seenGrouped = false
            for rows in 1...80 {
                let grouped = shouldGroup(rows: rows, slots: slots)
                if seenGrouped { #expect(grouped) }
                seenGrouped = seenGrouped || grouped
            }
        }
    }

    @Test("switching the union rule off restores the historic rule exactly")
    func switchOffIsTheHistoricRule() {
        for rows in [1, 2, 3, 5, 7, 8, 9, 33, 512] {
            for slots in [1, 2, 8, 64] {
                #expect(
                    shouldGroup(rows: rows, slots: slots, union: false)
                        == (rows * slots >= SwitchGLUExpertGrouping.historicGroupingThreshold))
            }
        }
    }

    @Test("row counting is rank-agnostic")
    func rowCountIsRankAgnostic() {
        // Shape metadata only: these arrays are never evaluated.
        #expect(SwitchGLUExpertGrouping.rowCount(MLXArray.zeros([5, 8], dtype: .int32)) == 5)
        #expect(SwitchGLUExpertGrouping.rowCount(MLXArray.zeros([2, 3, 8], dtype: .int32)) == 6)
        #expect(SwitchGLUExpertGrouping.rowCount(MLXArray.zeros([8], dtype: .int32)) == 1)
        #expect(SwitchGLUExpertGrouping.rowCount(MLXArray.zeros([1, 1, 1, 4], dtype: .int32)) == 1)
        // A packed [B, S, K] prompt pass and the flat [B*S, K] the MoE block
        // actually hands down must agree — the caller's layout is not allowed
        // to change the decision.
        #expect(
            SwitchGLUExpertGrouping.shouldGroup(MLXArray.zeros([2, 3, 8], dtype: .int32))
                == SwitchGLUExpertGrouping.shouldGroup(
                    MLXArray.zeros([6, 8], dtype: .int32)))
    }
}
