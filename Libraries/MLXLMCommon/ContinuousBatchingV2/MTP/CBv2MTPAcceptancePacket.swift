// CBv2MTPAcceptancePacket.swift
//
// The layout of the ONE lazy int32 array a verify round reads back at its
// single host-sync boundary (`CBv2MTPRoundInFlight.Verify.acceptancePacket`).
//
// The offsets used to be open-coded in two files: the build side concatenated
// segments in `mtpBuildVerifyGraph` and the read side recomputed
// `draftCount + batchIndex * targetWidth + column` and
// `massBase + batchIndex * targetWidth + hiddenColumn` in `finalizeMTPRound`.
// Two conditional segments were already enough for those to be a place where
// an off-by-one is invisible until a token stream is wrong; a third would be
// worse. This type owns the arithmetic, and it is the one part of the round
// that a CPU test can decide completely.
//
// Segment order is APPEND-ONLY on purpose: drafts, targets, then the optional
// segments, so adding one cannot move an offset an older reader computed.

import Foundation

/// Segment offsets for one round's acceptance packet.
///
///   drafts        [rows, k]      the chain's proposed token per draft position
///   targets       [rows, 1 + k]  the target's own token per verify column
///   shortlistMass [rows, 1 + k]  optional, parts-per-million (stateful drafters)
///   runnerUps     [rows, k]      optional, the drafter's RANK-2 token per
///                                draft position — instrumentation, see below
///
/// All segments are row-major, flattened, and concatenated in that order.
struct CBv2MTPAcceptancePacketLayout: Equatable {
    let rows: Int
    /// Draft depth `k`. The verify rectangle is `1 + k` columns wide.
    let draftDepth: Int
    let hasShortlistMass: Bool
    /// True when the drafter could report its runner-up alongside its argmax
    /// (`CBv2MTPDrafter.maximumDraftCandidates >= 2`). Nothing in the round
    /// BRANCHES on the runner-up — it is read at finalize and counted, so a
    /// drafter that cannot supply one only loses the measurement.
    let hasRunnerUps: Bool

    init(rows: Int, draftDepth: Int, hasShortlistMass: Bool, hasRunnerUps: Bool) {
        precondition(rows > 0, "CBv2MTPAcceptancePacketLayout: rows must be > 0")
        precondition(draftDepth >= 0, "CBv2MTPAcceptancePacketLayout: depth must be >= 0")
        self.rows = rows
        self.draftDepth = draftDepth
        self.hasShortlistMass = hasShortlistMass
        self.hasRunnerUps = hasRunnerUps && draftDepth > 0
    }

    var targetWidth: Int { 1 + draftDepth }
    var draftCount: Int { rows * draftDepth }
    var targetCount: Int { rows * targetWidth }

    var draftsBase: Int { 0 }
    var targetsBase: Int { draftCount }
    var shortlistMassBase: Int? { hasShortlistMass ? draftCount + targetCount : nil }
    var runnerUpsBase: Int? {
        guard hasRunnerUps else { return nil }
        return draftCount + targetCount + (hasShortlistMass ? targetCount : 0)
    }

    /// Total int32 elements. The one number the packet's `asArray` must return.
    var count: Int {
        draftCount + targetCount + (hasShortlistMass ? targetCount : 0)
            + (hasRunnerUps ? draftCount : 0)
    }

    func draftIndex(row: Int, position: Int) -> Int {
        precondition(row >= 0 && row < rows && position >= 0 && position < draftDepth)
        return draftsBase + row * draftDepth + position
    }

    func targetIndex(row: Int, column: Int) -> Int {
        precondition(row >= 0 && row < rows && column >= 0 && column < targetWidth)
        return targetsBase + row * targetWidth + column
    }

    func shortlistMassIndex(row: Int, column: Int) -> Int {
        guard let base = shortlistMassBase else {
            preconditionFailure("CBv2MTPAcceptancePacketLayout: no shortlist segment")
        }
        precondition(row >= 0 && row < rows && column >= 0 && column < targetWidth)
        return base + row * targetWidth + column
    }

    /// The drafter's runner-up at draft `position` — the rank-2 candidate of
    /// the SAME forward that produced `draftIndex(row:position:)`.
    func runnerUpIndex(row: Int, position: Int) -> Int {
        guard let base = runnerUpsBase else {
            preconditionFailure("CBv2MTPAcceptancePacketLayout: no runner-up segment")
        }
        precondition(row >= 0 && row < rows && position >= 0 && position < draftDepth)
        return base + row * draftDepth + position
    }
}
