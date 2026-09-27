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
/// One policy for both capture geometries. A historical (attention-only)
/// donor's boundaries are every multiple of the 1,024-token stride; a
/// recurrent (Qwen, Nemotron, Bonsai) donor's boundaries are its uniform
/// chunk ends, so its `stride` is that chunk size and the fork target is the
/// deepest chunk end at or below the hint.
///
/// Positions only: the capture owns the staged copies and applies the
/// verdicts. Boundaries arrive in ascending order, as prefill computes them.
struct CBv2CheckpointRetention: Equatable, Sendable {
    /// First, fork target and rolling latest.
    static let maximumRetained = 3

    /// What a retained boundary serves, in the order a donor gives them up:
    /// the first (a guess at a shared preamble) before the fork target
    /// (observed demand) before the rolling latest (the next turn).
    enum Role: Equatable, Sendable { case first, target, latest }

    /// Boundary spacing: the historical stride, or the recurrent chunk size.
    let stride: Int
    /// Stride-aligned fork boundary; nil without a usable hint.
    let target: Int?
    /// False for an adopter. Nothing at or below its restored boundary is
    /// recomputed, so its first boundary is whatever the donor already made
    /// durable; the first boundary ABOVE the restore point is only interior.
    let keepsFirst: Bool
    private(set) var first: Int?
    /// Ascending retained positions, never more than `maximumRetained`.
    private(set) var retained: [Int] = []

    /// `hintTokens` is the coordinator's repeated-prefix length: nil without a
    /// hint, 0 for a fleet-novel prompt. `resumedAt` is the restored boundary
    /// of an adopter, 0 for a cold prefill.
    init(stride: Int, hintTokens: Int?, resumedAt: Int = 0) {
        self.stride = stride
        keepsFirst = resumedAt <= 0
        let aligned = stride > 1 ? max(0, hintTokens ?? 0) / stride * stride : 0
        target = aligned > max(0, resumedAt) ? aligned : nil
    }

    var firstIsOpen: Bool { keepsFirst && first == nil }

    /// Record one captured boundary and return the positions that lose their
    /// place: the previous latest unless it holds the first or target role,
    /// or the candidate itself when it does not advance the prompt.
    mutating func commit(_ position: Int) -> [Int] {
        guard position > (retained.last ?? 0) else { return [position] }
        var retired: [Int] = []
        if let previous = retained.last, previous != first, previous != target {
            retained.removeLast()
            retired.append(previous)
        }
        if firstIsOpen { first = position }
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
    /// one stride of the final deepest boundary is dropped at publication:
    /// the deepest is only known at the end, and it already serves a prefix
    /// that long.
    var publication: (publish: [Int], drop: [Int]) {
        guard let deepest = retained.last else { return ([], []) }
        var drop: [Int] = []
        if let target, target != first, target != deepest, retained.contains(target),
           deepest - target <= stride
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
