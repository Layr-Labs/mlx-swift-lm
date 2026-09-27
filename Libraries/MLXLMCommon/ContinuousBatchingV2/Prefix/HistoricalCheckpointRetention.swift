/// Which stride boundaries of one historical donor are worth keeping.
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
/// Positions only: the capture owns the staged copies and applies the
/// verdicts. Boundaries arrive in ascending order, as prefill computes them.
struct CBv2HistoricalCheckpointRetention: Equatable, Sendable {
    /// First, fork target and rolling latest.
    static let maximumRetained = 3

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

    /// Byte pressure gives up the fork target and keeps the first/latest
    /// pair. Returns the dropped position, nil when only that pair remains.
    mutating func dropInterior() -> Int? {
        guard retained.count > 2 else { return nil }
        return retained.remove(at: 1)
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
