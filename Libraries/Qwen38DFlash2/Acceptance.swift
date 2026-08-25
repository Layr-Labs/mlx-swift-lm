// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx acceptance semantics at the revision in NOTICE.

public enum DFlash2Acceptance {
    public static func longestMatchingPrefix(
        draft: [Int],
        target: [Int]
    ) -> Int {
        let count = min(draft.count, target.count)
        for index in 0 ..< count where draft[index] != target[index] {
            return index
        }
        return count
    }
}

public struct DFlash2CommitPlan: Equatable, Sendable {
    public let commitRows: Int
    public let rejectedRows: Int
    public let fullyAccepted: Bool

    public init(verifyRows: Int, acceptedDraftTokens: Int) {
        commitRows = 1 + acceptedDraftTokens
        rejectedRows = verifyRows - commitRows
        fullyAccepted = commitRows == verifyRows
    }

    public init(commitRows: Int, rejectedRows: Int, fullyAccepted: Bool) {
        self.commitRows = commitRows
        self.rejectedRows = rejectedRows
        self.fullyAccepted = fullyAccepted
    }

    public func capped(to outputBudget: Int) -> DFlash2CommitPlan {
        let rows = min(commitRows, outputBudget)
        return DFlash2CommitPlan(
            commitRows: rows,
            rejectedRows: commitRows + rejectedRows - rows,
            fullyAccepted: rows == commitRows + rejectedRows)
    }
}
