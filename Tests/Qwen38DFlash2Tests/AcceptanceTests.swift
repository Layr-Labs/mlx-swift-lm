import Testing

@testable import Qwen38DFlash2

@Suite("DFlash2 target-authoritative acceptance")
struct DFlash2AcceptanceTests {
    @Test("returns the longest equal draft prefix")
    func longestPrefix() {
        #expect(DFlash2Acceptance.longestMatchingPrefix(draft: [1, 2, 3], target: [9, 2, 3]) == 0)
        #expect(DFlash2Acceptance.longestMatchingPrefix(draft: [1, 2, 3], target: [1, 9, 3]) == 1)
        #expect(DFlash2Acceptance.longestMatchingPrefix(draft: [1, 2, 3], target: [1, 2, 9]) == 2)
        #expect(DFlash2Acceptance.longestMatchingPrefix(draft: [1, 2, 3], target: [1, 2, 3]) == 3)
    }

    @Test("never reads beyond the shorter posterior")
    func shorterPosterior() {
        #expect(DFlash2Acceptance.longestMatchingPrefix(draft: [4, 5, 6], target: [4, 5]) == 2)
        #expect(DFlash2Acceptance.longestMatchingPrefix(draft: [], target: [4]) == 0)
    }

    @Test("commit plan keeps the authoritative row and trims only rejected verify rows")
    func commitPlan() {
        #expect(
            DFlash2CommitPlan(verifyRows: 1, acceptedDraftTokens: 0)
                == DFlash2CommitPlan(commitRows: 1, rejectedRows: 0, fullyAccepted: true))
        #expect(
            DFlash2CommitPlan(verifyRows: 6, acceptedDraftTokens: 0)
                == DFlash2CommitPlan(commitRows: 1, rejectedRows: 5, fullyAccepted: false))
        #expect(
            DFlash2CommitPlan(verifyRows: 6, acceptedDraftTokens: 3)
                == DFlash2CommitPlan(commitRows: 4, rejectedRows: 2, fullyAccepted: false))
        #expect(
            DFlash2CommitPlan(verifyRows: 6, acceptedDraftTokens: 5)
                == DFlash2CommitPlan(commitRows: 6, rejectedRows: 0, fullyAccepted: true))
    }

    @Test("final cycle commits no rows beyond the remaining output budget")
    func outputBudgetCapsCommit() {
        #expect(
            DFlash2CommitPlan(verifyRows: 8, acceptedDraftTokens: 7)
                .capped(to: 1)
                == DFlash2CommitPlan(commitRows: 1, rejectedRows: 7, fullyAccepted: false))
        #expect(
            DFlash2CommitPlan(verifyRows: 8, acceptedDraftTokens: 3)
                .capped(to: 3)
                == DFlash2CommitPlan(commitRows: 3, rejectedRows: 5, fullyAccepted: false))
    }
}
