// Copyright © 2026 Apple Inc.

import Foundation
import MLXSpeculative
import Testing

@Suite("SpeculativeWalk.single")
struct SpeculativeWalkSingleTests {

    @Test func emptyDraftReturnsJustMain() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [], main: [7]
        )
        #expect(accepted == 0)
        #expect(emitted == [7])
    }

    @Test func allDraftsMatch() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [1, 2, 3], main: [1, 2, 3, 4]
        )
        #expect(accepted == 3)
        #expect(emitted == [1, 2, 3, 4])
    }

    @Test func firstMismatchAtZero() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [9, 2, 3], main: [1, 2, 3, 4]
        )
        #expect(accepted == 0)
        #expect(emitted == [1])
    }

    @Test func firstMismatchInMiddle() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [1, 2, 99], main: [1, 2, 3, 4]
        )
        #expect(accepted == 2)
        #expect(emitted == [1, 2, 3])
    }

    @Test func emittedCountEqualsAcceptedPlusOne() {
        // Invariant: emitted.count == accepted + 1 for any non-trivial input.
        let draft = [1, 2, 3, 4, 5]
        let main = [1, 2, 99, 4, 5, 6]
        let (accepted, emitted) = SpeculativeWalk.single(draft: draft, main: main)
        #expect(emitted.count == accepted + 1)
    }
}
