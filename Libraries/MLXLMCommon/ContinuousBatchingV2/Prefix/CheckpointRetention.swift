/// Which boundaries of one complete-checkpoint donor are worth keeping.
///
/// A checkpoint earns its copy and its file only where a later request will
/// share exactly that prefix. There are two such places: the end of the
/// prompt (the next turn of a growing conversation restores the deepest
/// boundary) and a fork point, where other prompts share a shorter prefix
/// (a shared system prompt and tools, agent sub-tasks, regenerations). The
/// first boundary covers the shortest shared prefix; the coordinator's
/// observed repeated-prefix length names any other fork. Interior boundaries
/// elsewhere serve nobody.
///
/// One policy for both capture geometries, two ways of naming the target.
/// A historical (attention-only) donor's boundaries are every multiple of
/// the 1,024-token stride, so its fork target is known ahead of capture
/// (`plannedTarget`, the stride multiple at or below the hint) and the
/// historical path copies only that interior boundary. A recurrent (Qwen,
/// Nemotron, Bonsai) donor's boundaries are whatever aligned range ends
/// land, so it has no planned target: the target role goes to the deepest
/// committed boundary at or below the hint, and a deeper one supersedes it.
///
/// Positions only: the capture owns the staged copies and applies the
/// verdicts. Boundaries arrive in ascending order, as prefill computes them.
struct CBv2CheckpointRetention: Equatable, Sendable {
    /// First, fork target and rolling latest.
    static let maximumRetained = 3

    /// A fork target this close below the final deepest boundary is dropped
    /// at publication, whatever the layout's boundary spacing: the deepest
    /// already serves a prefix that long. One 1,024-token stride for a
    /// historical donor; for a recurrent donor a 2,048-chunk gap is kept.
    static let defaultTargetAdjacencyTokens = CBv2RecurrentCheckpointGeometry.historicalCheckpointStrideTokens

    /// What a retained boundary serves, in the order a donor gives them up:
    /// the first (a guess at a shared preamble) before the fork target
    /// (observed demand) before the rolling latest (the next turn).
    enum Role: Equatable, Sendable { case first, target, latest }

    /// Historical boundary spacing; nil for a recurrent donor.
    let stride: Int?
    /// `defaultTargetAdjacencyTokens` in production; test fixtures with
    /// tiny chunks scale it with their stride.
    let targetAdjacencyTokens: Int
    /// The coordinator's repeated-prefix length when it names a fork above
    /// the restore point; nil without a usable hint.
    let hintTokens: Int?
    /// Historical only: the stride-aligned boundary to copy for the target
    /// role. nil for a recurrent donor or without a usable hint.
    let plannedTarget: Int?
    /// False for an adopter. Nothing at or below its restored boundary is
    /// recomputed, so its first boundary is whatever the donor already made
    /// durable; the first boundary ABOVE the restore point is only interior.
    let keepsFirst: Bool
    private(set) var first: Int?
    /// The retained boundary holding the fork-target role, once one landed.
    private(set) var target: Int?
    /// Ascending retained positions, never more than `maximumRetained`.
    private(set) var retained: [Int] = []

    /// `hintTokens` is the coordinator's repeated-prefix length: nil without a
    /// hint, 0 for a fleet-novel prompt. `resumedAt` is the restored boundary
    /// of an adopter, 0 for a cold prefill.
    init(stride: Int?, hintTokens: Int?, resumedAt: Int = 0,
         targetAdjacencyTokens: Int = CBv2CheckpointRetention.defaultTargetAdjacencyTokens) {
        self.stride = stride
        self.targetAdjacencyTokens = targetAdjacencyTokens
        keepsFirst = resumedAt <= 0
        let usable = (hintTokens ?? 0) > max(0, resumedAt) ? hintTokens : nil
        self.hintTokens = usable
        if let stride, stride > 1, let usable {
            let aligned = usable / stride * stride
            plannedTarget = aligned > max(0, resumedAt) ? aligned : nil
        } else {
            plannedTarget = nil
        }
    }

    var firstIsOpen: Bool { keepsFirst && first == nil }

    /// Whether a boundary landing at `position` takes the target role: the
    /// planned boundary for a historical donor; for a recurrent donor any
    /// boundary at or below the hint, which then supersedes a shallower one.
    private func claimsTarget(_ position: Int) -> Bool {
        if let plannedTarget { return position == plannedTarget }
        guard let hintTokens else { return false }
        return position <= hintTokens && position > (target ?? 0)
    }

    /// Record one captured boundary and return the positions that lose their
    /// place: the previous latest unless it holds the first or target role,
    /// a target superseded by a deeper boundary at or below the hint, or the
    /// candidate itself when it does not advance the prompt.
    mutating func commit(_ position: Int) -> [Int] {
        guard position > (retained.last ?? 0) else { return [position] }
        var retired: [Int] = []
        if let previous = retained.last, previous != first, previous != target {
            retained.removeLast()
            retired.append(previous)
        }
        if firstIsOpen { first = position }
        if claimsTarget(position) {
            if let old = target, old != first, let index = retained.firstIndex(of: old) {
                retained.remove(at: index)
                retired.append(old)
            }
            target = position
        }
        retained.append(position)
        return retired
    }

    /// The rolling latest when it holds no other role: the next boundary to
    /// commit retires it, so a candidate replacing it adds nothing.
    var replaceableLatest: Int? {
        guard let latest = retained.last, latest != first, latest != target else { return nil }
        return latest
    }

    /// Retained boundaries this donor gives up, lowest priority first, to
    /// make room for a candidate of `role` under the slot-wide cap, or under
    /// its own byte budget once the candidate is the latest. The order is
    /// the value of what each serves: the rolling latest (the next turn of
    /// the same conversation) over the fork target (observed demand) over
    /// the first (a guess at a shared preamble).
    func sheddable(for role: Role) -> [Int] {
        var result: [Int] = []
        if role != .first, let first, first != target, retained.contains(first) { result.append(first) }
        if role == .latest, let target, retained.contains(target) { result.append(target) }
        return result
    }

    /// Give up one retained boundary under the slot-wide cap or the donor's
    /// byte budget. Its role stays taken: boundaries arrive ascending, so
    /// nothing later can fill it.
    mutating func shed(_ position: Int) {
        retained.removeAll { $0 == position }
    }

    /// Deepest first, then the fork target, then the first. A target within
    /// `targetAdjacencyTokens` of the final deepest boundary is dropped at
    /// publication: the deepest is only known at the end, and it already
    /// serves a prefix that long.
    var publication: (publish: [Int], drop: [Int]) {
        guard let deepest = retained.last else { return ([], []) }
        var drop: [Int] = []
        if let target, target != first, target != deepest, retained.contains(target),
           deepest - target <= targetAdjacencyTokens
        {
            drop = [target]
        }
        return (retained.reversed().filter { !drop.contains($0) }, drop)
    }
}

/// Slot-wide bound on staged historical windows.
///
/// Staged window copies and request chunk reservations draw on one admission
/// ledger. A retained checkpoint is an optimisation for a FUTURE request; a
/// chunk reservation is a request being served now, and when one fails the
/// scheduler preempts a running row. So the sum of staged windows across all
/// donors of a slot is capped, and a boundary that does not fit is simply
/// not captured.
enum CBv2HistoricalStagingCap {
    typealias Role = CBv2CheckpointRetention.Role

    /// Staged windows may hold at most this fraction of the slot's capacity.
    static let capacityDivisor = 8

    /// How many of `sheddable` (the donor's own staged bytes, lowest
    /// priority first) the candidate displaces to fit; nil refuses it.
    /// `slotBytes` is every donor's staged plus in-flight bytes;
    /// `replacingBytes` is the donor's role-less latest, which the
    /// candidate's commit retires anyway.
    static func displaced(
        candidateBytes: Int, slotBytes: Int, replacingBytes: Int, sheddable: [Int], cap: Int
    ) -> Int? {
        guard candidateBytes >= 0, slotBytes >= 0, replacingBytes >= 0, cap >= 0 else { return nil }
        let (grown, overflow) = slotBytes.addingReportingOverflow(candidateBytes)
        guard !overflow else { return nil }
        var total = grown - min(replacingBytes, slotBytes)
        var count = 0
        for bytes in sheddable where total > cap {
            total -= max(0, bytes)
            count += 1
        }
        return total <= cap ? count : nil
    }
}
