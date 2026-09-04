import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Host-side rules of tree (multi-candidate) MTP drafting: column layout,
/// the verify rectangle's attention adjacency, the accept walk, and the KV
/// keep/rollback plan. No MLX graph, no engine — every rule is decided here.
@Suite("CBv2MTPTreeShape")
struct CBv2MTPTreeShapeTests {

    // MARK: - Column layout

    @Test func chainLaysOutOneColumnPerDraftPlusTheSeed() {
        let shape = CBv2MTPTreeShape.chain(4)
        #expect(shape.isChain)
        #expect(shape.columnCount == 5)
        #expect(shape.draftForwardCount == 4)
        #expect(shape.maximumCandidateRank == 0)
        #expect(!shape.mayRequireColumnMoves)
        let columns = shape.columns
        #expect(columns[0].parent == nil)
        #expect(columns[0].draftStep == 0)
        for depth in 1 ... 4 {
            #expect(columns[depth].parent == depth - 1)
            #expect(columns[depth].draftStep == depth)
            #expect(columns[depth].candidateRank == 0)
        }
    }

    @Test func depthZeroIsTheSeedColumnAlone() {
        let shape = CBv2MTPTreeShape.chain(0)
        #expect(shape.columnCount == 1)
        #expect(shape.columns.count == 1)
        #expect(shape.draftForwardCount == 0)
        #expect(shape.children() == [[]])
    }

    @Test func alternatesComeAfterTheWholeSpineAndReuseItsForward() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 4, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        #expect(shape.columnCount == 6)
        // The alternate does NOT add a drafter forward: it is rank 1 of the
        // forward that already produced the spine column at its depth.
        #expect(shape.draftForwardCount == 4)
        #expect(shape.maximumCandidateRank == 1)
        #expect(shape.mayRequireColumnMoves)
        let columns = shape.columns
        #expect(columns[5].parent == 0)
        #expect(columns[5].draftStep == 1)
        #expect(columns[5].candidateRank == 1)
        // Spine first, so a spine-only acceptance stays a plain prefix.
        #expect(columns[1 ... 4].allSatisfy { $0.candidateRank == 0 })
    }

    @Test func childrenListTheSpineBeforeAnyAlternate() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 3,
            alternates: [
                CBv2MTPTreeAlternate(depth: 1, rank: 1),
                CBv2MTPTreeAlternate(depth: 1, rank: 2),
            ])
        // Column 0's children: the spine's depth-1 column, then both
        // alternates, in that order.
        #expect(shape.children()[0] == [1, 4, 5])
        #expect(shape.children()[1] == [2])
        #expect(shape.children()[3] == [])
    }

    // MARK: - Width budget

    @Test func budgetIsAWidthBudgetOverColumnsNotOverDepth() {
        let chain = CBv2MTPTreeShape.chain(7)
        #expect(chain.fits(decodeRows: 1, budget: 8))
        #expect(!chain.fits(decodeRows: 2, budget: 8))
        let tree = CBv2MTPTreeShape(
            spineDepth: 6, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        #expect(tree.columnCount == 8)
        #expect(tree.fits(decodeRows: 1, budget: 8))
        #expect(!CBv2MTPTreeShape.chain(7).fits(decodeRows: 1, budget: 7))
    }

    @Test func clampShortensTheSpineAndNeverWidens() throws {
        let tree = CBv2MTPTreeShape(
            spineDepth: 6, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        let clamped = try #require(tree.clamped(decodeRows: 1, budget: 5))
        #expect(clamped.spineDepth == 3)
        #expect(clamped.alternates == tree.alternates)
        #expect(clamped.columnCount == 5)
        // Already inside the budget: unchanged, never grown to fill it.
        #expect(tree.clamped(decodeRows: 1, budget: 64) == tree)
    }

    @Test func clampRefusesWhenAnAlternateWouldLoseItsParent() {
        let tree = CBv2MTPTreeShape(
            spineDepth: 4, alternates: [CBv2MTPTreeAlternate(depth: 3, rank: 1)])
        // 1 seed + 3 spine + 1 alternate = 5 columns is the floor.
        #expect(tree.clamped(decodeRows: 1, budget: 5)?.spineDepth == 3)
        #expect(tree.clamped(decodeRows: 1, budget: 4) == nil)
        #expect(tree.clamped(decodeRows: 2, budget: 8) == nil)
        #expect(tree.clamped(decodeRows: 0, budget: 8) == nil)
    }

    @Test func shapeCarriesNoLengthOrPrefillCoupling() {
        // The same shape at any decode-row count depends only on the width
        // budget, never on how much context the row is carrying.
        let tree = CBv2MTPTreeShape(
            spineDepth: 4, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        #expect(tree.clamped(decodeRows: 1, budget: 8) == tree)
        #expect(tree.clamped(decodeRows: 1, budget: 8) == tree.clamped(decodeRows: 1, budget: 8))
    }

    // MARK: - Verify mask

    @Test func chainMaskIsExactlyTheCausalLowerTriangle() {
        let mask = CBv2MTPTreeShape.chain(3).ancestorMask()
        #expect(mask.count == 4)
        for query in 0 ..< 4 {
            for key in 0 ..< 4 {
                #expect(mask[query][key] == (key <= query))
            }
        }
    }

    @Test func alternateSeesItsParentChainAndItselfOnly() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 3, alternates: [CBv2MTPTreeAlternate(depth: 2, rank: 1)])
        // columns: 0 seed, 1..3 spine, 4 = alternate at depth 2 (parent 1)
        let mask = shape.ancestorMask()
        #expect(mask[4] == [true, true, false, false, true])
        // The spine is untouched by the alternate's presence: it must not
        // see a sibling that sits later in the rectangle.
        #expect(mask[3] == [true, true, true, true, false])
        #expect(mask[0] == [true, false, false, false, false])
    }

    @Test func maskIsReflexiveAndTransitivelyClosed() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 4,
            alternates: [
                CBv2MTPTreeAlternate(depth: 1, rank: 1),
                CBv2MTPTreeAlternate(depth: 3, rank: 1),
            ])
        let mask = shape.ancestorMask()
        let columns = shape.columns
        for query in 0 ..< shape.columnCount {
            #expect(mask[query][query], "every column attends itself")
            if let parent = columns[query].parent {
                for key in 0 ..< shape.columnCount where mask[parent][key] {
                    #expect(mask[query][key], "ancestors are inherited")
                }
            }
            // A column never sees a column that comes after it: the verify
            // rectangle is still written in one causal pass.
            for key in (query + 1) ..< shape.columnCount {
                #expect(!mask[query][key])
            }
        }
    }

    @Test func flattenedMaskIsRowMajor() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 1, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        #expect(
            shape.flattenedAncestorMask() == [
                true, false, false,
                true, true, false,
                true, false, true,
            ])
    }

    // MARK: - Accept walk

    /// The chain rule the engine ships (`targets[a] == drafts[a]`) restated
    /// for a tree must give the identical answer on a chain shape.
    @Test func acceptWalkOnAChainMatchesThePrefixRule() {
        let shape = CBv2MTPTreeShape.chain(4)
        // drafts d1..d4 = 11,12,13,14; targets t0..t4.
        let candidates = [0, 11, 12, 13, 14]
        for accepted in 0 ... 4 {
            var targets = [Int](repeating: 0, count: 5)
            for index in 0 ..< accepted { targets[index] = candidates[index + 1] }
            for index in accepted ..< 5 { targets[index] = 900 + index }
            let walk = shape.acceptWalk(candidates: candidates, targets: targets)
            #expect(walk.acceptedDraftCount == accepted)
            #expect(walk.path == Array(0 ... accepted))
            #expect(walk.emitted.count == accepted + 1)
            #expect(walk.emitted == (0 ... accepted).map { targets[$0] })
        }
    }

    @Test func alternateIsTakenOnlyWhenTheSpineColumnIsWrong() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 3, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        // columns: 0 seed, 1/2/3 spine, 4 alternate at depth 1.
        let candidates = [0, 11, 12, 13, 21]
        // Spine right at depth 1 -> the alternate can never be reached.
        var targets = [11, 12, 13, 77, 88]
        var walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0, 1, 2, 3])
        #expect(walk.emitted == [11, 12, 13, 77])
        // Spine wrong at depth 1, alternate right: the round commits two
        // tokens where the chain would have committed one.
        targets = [21, 55, 66, 77, 88]
        walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0, 4])
        #expect(walk.emitted == [21, 88])
        #expect(walk.acceptedDraftCount == 1)
        // Both wrong: one token, exactly the chain's outcome.
        targets = [99, 55, 66, 77, 88]
        walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0])
        #expect(walk.emitted == [99])
    }

    @Test func duplicateAlternateIsInertBecauseTheSpineIsTriedFirst() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 2, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        // A drafter whose runner-up equals its argmax (a degenerate head, or
        // a shortlist collapse) must not change the accepted path.
        let candidates = [0, 11, 12, 11]
        let targets = [11, 12, 33, 44]
        let walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0, 1, 2])
        #expect(walk.emitted == [11, 12, 33])
    }

    @Test func walkFollowsAnAlternateOnlyDownItsOwnAncestry() {
        // Two alternates at different depths: reaching the deeper one
        // requires the spine to have carried the round that far.
        let shape = CBv2MTPTreeShape(
            spineDepth: 3,
            alternates: [
                CBv2MTPTreeAlternate(depth: 1, rank: 1),
                CBv2MTPTreeAlternate(depth: 3, rank: 1),
            ])
        // columns: 0 seed, 1/2/3 spine, 4 alt@1 (parent 0), 5 alt@3 (parent 2)
        let candidates = [0, 11, 12, 13, 21, 23]
        let targets = [11, 12, 23, 44, 55, 66]
        let walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0, 1, 2, 5])
        #expect(walk.emitted == [11, 12, 23, 66])
    }

    // MARK: - KV keep plan

    @Test func chainKeepPlanIsASuffixRollbackWithNoMoves() {
        let shape = CBv2MTPTreeShape.chain(4)
        let candidates = [0, 11, 12, 13, 14]
        let targets = [11, 12, 99, 0, 0]
        let walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.acceptedDraftCount == 2)
        let plan = shape.keepPlan(for: walk, truncatedTo: walk.emitted.count)
        #expect(plan.moves.isEmpty)
        // seed + 2 accepted drafts keep their KV; the round's third emitted
        // token is the bonus read out of the third surviving column.
        #expect(plan.committedColumnCount == 3)
        #expect(plan.rollback == 5 - 3)
    }

    /// The whole reason a tree cannot reuse the shipped rollback unchanged:
    /// an accepted alternate leaves a HOLE in the column order, and
    /// `CBv2SequenceKV.rollback(_:)` can only drop a suffix.
    @Test func alternateHitLeavesAHoleThatMustBeClosedBeforeRollback() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 3, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)])
        let candidates = [0, 11, 12, 13, 21]
        let targets = [21, 55, 66, 77, 88]
        let walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0, 4])
        let plan = shape.keepPlan(for: walk, truncatedTo: walk.emitted.count)
        // Surviving columns are 0 and 4; column 4 has to become column 1
        // before the tail is dropped.
        #expect(plan.moves == [CBv2MTPTreeShape.KeepPlan.Move(from: 4, to: 1)])
        #expect(plan.committedColumnCount == 2)
        #expect(plan.rollback == 3)
    }

    @Test func alternateOnADeeperPathMovesItsColumnIntoPlace() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 3, alternates: [CBv2MTPTreeAlternate(depth: 3, rank: 1)])
        // columns: 0 seed, 1/2/3 spine, 4 = alt at depth 3 (parent 2)
        let candidates = [0, 11, 12, 13, 23]
        let targets = [11, 12, 23, 44, 55]
        let walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0, 1, 2, 4])
        #expect(walk.emitted == [11, 12, 23, 55])
        let plan = shape.keepPlan(for: walk, truncatedTo: 4)
        // Keep columns 0,1,2 in place and move column 4 down into slot 3, so
        // the surviving KV is contiguous and `rollback` stays a suffix drop.
        #expect(plan.moves == [CBv2MTPTreeShape.KeepPlan.Move(from: 4, to: 3)])
        #expect(plan.committedColumnCount == 4)
        #expect(plan.rollback == 1)
    }

    @Test func movesNeverClobberASourceThatHasNotBeenReadYet() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 4,
            alternates: [
                CBv2MTPTreeAlternate(depth: 1, rank: 1),
                CBv2MTPTreeAlternate(depth: 2, rank: 1),
            ])
        // Force a path that uses the depth-2 alternate: 0 -> 1 -> 6.
        let candidates = [0, 11, 12, 13, 14, 21, 22]
        let targets = [11, 22, 33, 44, 55, 66, 77]
        let walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.path == [0, 1, 6])
        let plan = shape.keepPlan(for: walk, truncatedTo: 3)
        #expect(plan.moves == [CBv2MTPTreeShape.KeepPlan.Move(from: 6, to: 2)])
        #expect(plan.committedColumnCount == 3)
        #expect(plan.rollback == 4)
        for move in plan.moves { #expect(move.to < move.from) }
    }

    @Test func truncationByStopOrLengthShortensTheKeptPrefix() {
        let shape = CBv2MTPTreeShape.chain(4)
        let candidates = [0, 11, 12, 13, 14]
        let targets = [11, 12, 13, 14, 99]
        let walk = shape.acceptWalk(candidates: candidates, targets: targets)
        #expect(walk.emitted.count == 5)
        // A stop token at the second emitted position: two columns of KV
        // survive and everything else rolls back.
        let plan = shape.keepPlan(for: walk, truncatedTo: 2)
        #expect(plan.moves.isEmpty)
        #expect(plan.committedColumnCount == 2)
        #expect(plan.rollback == 3)
        // One emitted token: only the seed column's KV stays, which is what
        // the shipped chain does when its first draft is rejected.
        let one = shape.keepPlan(for: walk, truncatedTo: 1)
        #expect(one.committedColumnCount == 1)
        #expect(one.rollback == 4)
        // No output budget left: the round leaves the row exactly where it
        // started, seed column included.
        let none = shape.keepPlan(for: walk, truncatedTo: 0)
        #expect(none.committedColumnCount == 0)
        #expect(none.rollback == 5)
        #expect(none.moves.isEmpty)
    }

    @Test func keepPlanTotalsAlwaysAccountForEveryColumn() {
        let shape = CBv2MTPTreeShape(
            spineDepth: 3,
            alternates: [
                CBv2MTPTreeAlternate(depth: 1, rank: 1),
                CBv2MTPTreeAlternate(depth: 2, rank: 1),
            ])
        let candidates = [0, 11, 12, 13, 21, 22]
        for target0 in [11, 21, 99] {
            for target1 in [12, 22, 98] {
                let targets = [target0, target1, 31, 41, 51, 61]
                let walk = shape.acceptWalk(candidates: candidates, targets: targets)
                for emitted in 0 ... walk.emitted.count {
                    let plan = shape.keepPlan(for: walk, truncatedTo: emitted)
                    #expect(plan.committedColumnCount + plan.rollback == shape.columnCount)
                    #expect(plan.committedColumnCount == emitted)
                    // Every move closes a hole: destinations are dense from 0
                    // and never above their source.
                    for move in plan.moves { #expect(move.to < move.from) }
                }
            }
        }
    }

    // MARK: - Shape grammar

    @Test func grammarParsesShapesAndRefusesAnythingElse() {
        #expect(CBv2MTPTreeShape.parse("off") == nil)
        #expect(CBv2MTPTreeShape.parse("") == nil)
        #expect(CBv2MTPTreeShape.parse("spine=5") == CBv2MTPTreeShape.chain(5))
        #expect(
            CBv2MTPTreeShape.parse("spine=4,alt=1")
                == CBv2MTPTreeShape(
                    spineDepth: 4, alternates: [CBv2MTPTreeAlternate(depth: 1, rank: 1)]))
        #expect(
            CBv2MTPTreeShape.parse("spine=4,alt=2:2")
                == CBv2MTPTreeShape(
                    spineDepth: 4, alternates: [CBv2MTPTreeAlternate(depth: 2, rank: 2)]))
        #expect(CBv2MTPTreeShape.parse("SPINE=3") == CBv2MTPTreeShape.chain(3))
        // An alternate whose parent is past the end of the spine is a typo,
        // not a shape to be silently repaired.
        #expect(CBv2MTPTreeShape.parse("spine=2,alt=5") == nil)
        #expect(CBv2MTPTreeShape.parse("alt=1") == nil)
        #expect(CBv2MTPTreeShape.parse("spine=-1") == nil)
        #expect(CBv2MTPTreeShape.parse("alt=0") == nil)
        #expect(CBv2MTPTreeShape.parse("spine=4,prefill=17408") == nil)
        #expect(CBv2MTPTreeShape.parse("spine") == nil)
    }

    @Test func treeDraftIsOffUnlessTheEnvironmentNamesAShape() {
        // Nothing in the process environment for a test run, so the shipped
        // chain is what every round gets.
        #expect(CBv2MTPRoundSwitches.treeDraft == nil)
    }

    // MARK: - Equivalence with the shipped verify

    /// A chain shape's adjacency must be the SAME visibility the target
    /// already applies to a `[1, 1+k]` rectangle. If this ever diverges,
    /// "tree" has stopped generalizing the shipped verify and has become a
    /// second, unverified attention rule.
    @Test func chainAdjacencyMatchesTheTargetsOwnCausalMask() throws {
        for k in 1 ... 7 {
            let shape = CBv2MTPTreeShape.chain(k)
            let columns = shape.columnCount
            let history = 37
            let mask = try #require(
                CBv2AttentionV1.boolMask(
                    L: columns, kL: history + columns, window: nil))
            let adjacency = shape.ancestorMask()
            let flat = mask.asArray(Bool.self)
            for query in 0 ..< columns {
                for key in 0 ..< (history + columns) {
                    let shipped = flat[query * (history + columns) + key]
                    if key < history {
                        #expect(shipped, "retained history is always visible")
                    } else {
                        #expect(shipped == adjacency[query][key - history])
                    }
                }
            }
        }
    }
}
