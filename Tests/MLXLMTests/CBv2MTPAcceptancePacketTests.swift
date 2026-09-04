import Foundation
import Testing

@testable import MLXLMCommon

/// The acceptance packet is the round's single host-sync boundary and the
/// offsets into it used to be open-coded on both sides. These pin the
/// arithmetic, including that adding a segment cannot move an older one.
@Suite("CBv2MTPAcceptancePacketLayout")
struct CBv2MTPAcceptancePacketTests {

    @Test func chainOffsetsMatchTheOpenCodedArithmeticTheyReplaced() {
        let rows = 3
        let k = 4
        let layout = CBv2MTPAcceptancePacketLayout(
            rows: rows, draftDepth: k, hasShortlistMass: true, hasRunnerUps: false)
        let draftCount = rows * k
        let targetWidth = 1 + k
        for row in 0 ..< rows {
            for position in 0 ..< k {
                #expect(layout.draftIndex(row: row, position: position) == row * k + position)
            }
            for column in 0 ..< targetWidth {
                #expect(
                    layout.targetIndex(row: row, column: column)
                        == draftCount + row * targetWidth + column)
                #expect(
                    layout.shortlistMassIndex(row: row, column: column)
                        == draftCount + rows * targetWidth + row * targetWidth + column)
            }
        }
    }

    /// Append-only: a reader that knows nothing about runner-ups still finds
    /// every earlier segment exactly where it was.
    @Test func addingRunnerUpsMovesNoExistingOffset() {
        for hasMass in [false, true] {
            let without = CBv2MTPAcceptancePacketLayout(
                rows: 2, draftDepth: 3, hasShortlistMass: hasMass, hasRunnerUps: false)
            let with = CBv2MTPAcceptancePacketLayout(
                rows: 2, draftDepth: 3, hasShortlistMass: hasMass, hasRunnerUps: true)
            #expect(without.draftsBase == with.draftsBase)
            #expect(without.targetsBase == with.targetsBase)
            #expect(without.shortlistMassBase == with.shortlistMassBase)
            #expect(without.runnerUpsBase == nil)
            #expect(with.runnerUpsBase == without.count)
            #expect(with.count == without.count + with.draftCount)
        }
    }

    @Test func segmentsAreDisjointAndCoverThePacketExactly() {
        for hasMass in [false, true] {
            for hasRunnerUps in [false, true] {
                let layout = CBv2MTPAcceptancePacketLayout(
                    rows: 3, draftDepth: 4,
                    hasShortlistMass: hasMass, hasRunnerUps: hasRunnerUps)
                var seen = Set<Int>()
                for row in 0 ..< layout.rows {
                    for position in 0 ..< layout.draftDepth {
                        #expect(seen.insert(layout.draftIndex(row: row, position: position)).inserted)
                        if layout.runnerUpsBase != nil {
                            #expect(
                                seen.insert(
                                    layout.runnerUpIndex(row: row, position: position)
                                ).inserted)
                        }
                    }
                    for column in 0 ..< layout.targetWidth {
                        #expect(seen.insert(layout.targetIndex(row: row, column: column)).inserted)
                        if layout.shortlistMassBase != nil {
                            #expect(
                                seen.insert(
                                    layout.shortlistMassIndex(row: row, column: column)
                                ).inserted)
                        }
                    }
                }
                #expect(seen.count == layout.count)
                #expect(seen.min() == 0)
                #expect(seen.max() == layout.count - 1)
            }
        }
    }

    /// Build a packet the way `mtpBuildVerifyGraph` does — row-major segments
    /// concatenated in order — and read every cell back through the layout.
    @Test func roundTripsEveryCellThroughAFlattenedPacket() {
        let rows = 2
        let k = 3
        let layout = CBv2MTPAcceptancePacketLayout(
            rows: rows, draftDepth: k, hasShortlistMass: true, hasRunnerUps: true)
        func draft(_ row: Int, _ p: Int) -> Int32 { Int32(1_000 + row * 10 + p) }
        func target(_ row: Int, _ c: Int) -> Int32 { Int32(2_000 + row * 10 + c) }
        func mass(_ row: Int, _ c: Int) -> Int32 { Int32(3_000 + row * 10 + c) }
        func runnerUp(_ row: Int, _ p: Int) -> Int32 { Int32(4_000 + row * 10 + p) }

        var packet: [Int32] = []
        for row in 0 ..< rows { for p in 0 ..< k { packet.append(draft(row, p)) } }
        for row in 0 ..< rows {
            for c in 0 ..< layout.targetWidth { packet.append(target(row, c)) }
        }
        for row in 0 ..< rows {
            for c in 0 ..< layout.targetWidth { packet.append(mass(row, c)) }
        }
        for row in 0 ..< rows { for p in 0 ..< k { packet.append(runnerUp(row, p)) } }

        #expect(packet.count == layout.count)
        for row in 0 ..< rows {
            for p in 0 ..< k {
                #expect(packet[layout.draftIndex(row: row, position: p)] == draft(row, p))
                #expect(packet[layout.runnerUpIndex(row: row, position: p)] == runnerUp(row, p))
            }
            for c in 0 ..< layout.targetWidth {
                #expect(packet[layout.targetIndex(row: row, column: c)] == target(row, c))
                #expect(packet[layout.shortlistMassIndex(row: row, column: c)] == mass(row, c))
            }
        }
    }

    /// The runner-up at position `p` must come from the same drafter forward
    /// as the draft at position `p` — same row, same offset within its
    /// segment. If those ever disagree, `r` is measured against the wrong
    /// column and the number is silently meaningless.
    @Test func runnerUpSharesItsDraftsRowAndPosition() throws {
        let layout = CBv2MTPAcceptancePacketLayout(
            rows: 4, draftDepth: 5, hasShortlistMass: false, hasRunnerUps: true)
        let base = try #require(layout.runnerUpsBase)
        for row in 0 ..< layout.rows {
            for position in 0 ..< layout.draftDepth {
                #expect(
                    layout.runnerUpIndex(row: row, position: position) - base
                        == layout.draftIndex(row: row, position: position)
                            - layout.draftsBase)
            }
        }
    }

    @Test func depthZeroHasNoDraftOrRunnerUpSegment() {
        let layout = CBv2MTPAcceptancePacketLayout(
            rows: 2, draftDepth: 0, hasShortlistMass: false, hasRunnerUps: true)
        #expect(layout.draftCount == 0)
        #expect(layout.runnerUpsBase == nil, "there is no forward to have a runner-up from")
        #expect(layout.count == layout.targetCount)
    }

    @Test func packetSizeGrowsOnlyWithRowsDepthAndSegments() {
        // Shape-generic: nothing about the packet is keyed on how much
        // context the rows carry.
        let narrow = CBv2MTPAcceptancePacketLayout(
            rows: 1, draftDepth: 4, hasShortlistMass: false, hasRunnerUps: true)
        #expect(narrow.count == 4 + 5 + 4)
        let wide = CBv2MTPAcceptancePacketLayout(
            rows: 8, draftDepth: 4, hasShortlistMass: false, hasRunnerUps: true)
        #expect(wide.count == 8 * narrow.count)
    }
}

/// `r` and `q` as the report reads them.
@Suite("CBv2MTPMetrics runner-up rate")
struct CBv2MTPRunnerUpMetricsTests {

    @Test func hitRateIsNilUntilARoundIsRejectedAtARealDivergence() {
        var metrics = CBv2MTPMetrics()
        #expect(metrics.runnerUpHitRate == nil)
        metrics.runnerUpObservations = 4
        metrics.runnerUpHits = 1
        #expect(metrics.runnerUpHitRate == 0.25)
    }

    @Test func perPositionRateIsNilWhereNothingWasObserved() {
        var metrics = CBv2MTPMetrics()
        metrics.perPositionRunnerUpObservations = [10, 0, 4]
        metrics.perPositionRunnerUpHits = [5, 0, 3]
        let rates = metrics.runnerUpHitRateByPosition
        #expect(rates.count == 3)
        #expect(rates[0] == 0.5)
        #expect(rates[1] == nil)
        #expect(rates[2] == 0.75)
    }

    /// The two numbers a single arm has to produce for D4: `q` per position
    /// (already recorded) and `r` per position (this change).
    @Test func qAndRAreBothReadableFromOneSnapshot() throws {
        var metrics = CBv2MTPMetrics()
        metrics.conditionalAcceptance = [0.85, 0.80, 0.78, 0.76]
        metrics.perPositionRunnerUpObservations = [30, 12, 5, 2]
        metrics.perPositionRunnerUpHits = [12, 5, 2, 1]
        metrics.runnerUpObservations = 49
        metrics.runnerUpHits = 20
        #expect(metrics.conditionalAcceptance.count == 4)
        #expect(metrics.runnerUpHitRate != nil)
        #expect(metrics.runnerUpHitRateByPosition.count == 4)
        // Sanity on the shape of the decision: at q = 0.85 an alternate needs
        // r above 1, so this snapshot falsifies tree drafting on its own.
        let q = metrics.conditionalAcceptance[0]
        let r = try #require(metrics.runnerUpHitRate)
        #expect((1 - q) * r < 0.18)
    }
}
