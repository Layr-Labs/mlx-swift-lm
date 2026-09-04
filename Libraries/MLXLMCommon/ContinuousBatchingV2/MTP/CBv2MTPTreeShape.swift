// CBv2MTPTreeShape.swift
//
// The host-side core of TREE (multi-candidate) MTP drafting: the column
// layout, the verify rectangle's attention mask, the accept walk over a
// tree, and the KV keep/rollback plan. Pure value types — no MLX graph, no
// engine state — so every rule below is decided by a CPU test.
//
// WHY A TREE AT ALL. The rectangular budget is a WIDTH budget
// (`CBv2MTPRoundDriver.maximumAutomaticDepth`: `rows * (1 + k) <=
// maxAutomaticRectangularTokens`). A chain spends all of it on one line of
// descent, so a round commits `1 + sum_i P(prefix i accepted)` tokens. A
// tree spends some columns on a SIBLING of a chain node, which is on the
// accepted path exactly when the chain's node is wrong and the sibling is
// right — so it converts some 1-token rounds into 2-or-more-token rounds
// without deepening the chain.
//
// WHEN IT IS WORTH IT — read this before enabling the switch. Let `q` be
// the drafter's per-position acceptance and `r` the chance its rank-2
// candidate is the target's argmax GIVEN rank-1 is not. One alternate at
// depth d adds `q^(d-1) * (1 - q) * r` committed tokens. It needs no extra
// drafter forward (its token is rank 2 of a forward the spine already ran),
// so it costs `c_alt`: the verify-column share of the marginal round cost,
// not the whole of it. A column is worth adding only when it raises
// committed tokens by a larger FRACTION than it raises the round, so at the
// chain's own optimum depth k*
//
//      alternate wins  <=>  (1 - q) * r / committed(k*)  >  c_alt / round(k*)
//
// which is a threshold on `r` for each `q`. Measured at the production
// shape (17,408-token prompt, 1,024 output, B=1, 3,136 rounds over seven
// widths): `round_ms(k) = 13.38 + c*k`, `c = 4.70` today and ~2.60 with
// MTPLX_MTP_FUSED_VERIFY_ATTENTION on, `c_alt = c - 1.0`:
//
//      q         0.90  0.85  0.82  0.78  0.75  0.70  0.66  0.63  0.60  0.50
//   today  k*       4     3     3     2     2     2     2     1     1     1
//          r >   4.71  2.86  2.28  1.76  1.50  1.19  1.00  0.90  0.82  0.61
//   fused  k*       7     5     4     4     3     3     2     2     2     2
//          r >   2.89  1.68  1.31  0.99  0.83  0.64  0.53  0.47  0.42  0.30
//
// THE MEASURED `q` AT THAT SHAPE IS 0.63 (0.658 at position 1, ~0.62 flat
// after), not the 0.82 of the 64-token control. On the measured curve an
// alternate needs `r > 0.99` today and `r > 0.52` fused — so a tree is
// live, but ONLY downstream of the fused verify, and it is worth about
// +2.7 tok/s at r = 0.7 and +10 at a perfect r = 1.0. It is not a road to
// 200 at any `r`; `c` and `q` are. Best shapes at the measured curve:
// `spine=2,alt=1` at r >= 0.52 and `spine=2,alt=1,alt=2` at r near 1.
//
// The low `q` is itself under audit (the drafter's long-context capture and
// mask). If it is restored toward 0.9 this whole switch is dead again: the
// required `r` becomes 2.89 even fused, and the plain chain wins at every
// `r`. Do not enable this without reading `runnerUpHitRate` (measured by
// the instrument in `EngineLoopV2+MTPFinalize`) and the `q` of the same run.
//
// SHAPE-GENERIC BY CONSTRUCTION. Nothing here reads a prompt length, a
// prefill size, a ring size or a chunk size. A shape is (spine depth,
// alternates); the same shape is requested at 64 tokens and at the full
// context window, and the only thing that can shrink it is the width
// budget, which is keyed on the chip and the planned decode rows.

import Foundation

/// One column of the verify rectangle.
public struct CBv2MTPTreeColumn: Equatable, Sendable {
    /// Column index of this node's parent; nil only for the seed column.
    public let parent: Int?
    /// Drafter chain step that produced this column's token: 0 for the seed
    /// (it is the row's newest confirmed token, not a draft), `d >= 1` for a
    /// token proposed by the d-th drafter forward.
    public let draftStep: Int
    /// Which candidate of that forward this column carries. 0 is the greedy
    /// argmax (what the shipped chain drafts); 1 is the runner-up, and so on.
    public let candidateRank: Int

    public init(parent: Int?, draftStep: Int, candidateRank: Int) {
        self.parent = parent
        self.draftStep = draftStep
        self.candidateRank = candidateRank
    }
}

/// A sibling of the spine column at `depth`, carrying the drafter's
/// `rank`-th candidate from the SAME forward that produced that spine
/// column (rank 0 is the spine's own argmax, so an alternate's rank is >= 1).
public struct CBv2MTPTreeAlternate: Equatable, Sendable {
    public let depth: Int
    public let rank: Int
    public init(depth: Int, rank: Int) {
        self.depth = depth
        self.rank = rank
    }
}

/// A draft tree laid out as verify columns.
///
/// COLUMN ORDER IS LOAD-BEARING. The spine occupies columns `0 ... spine`
/// and every alternate comes after it, because the ACCEPTED path decides
/// which KV columns survive the round and `CBv2SequenceKV.rollback(_:)` can
/// only drop a SUFFIX. Putting the spine first makes the common outcome —
/// some prefix of the chain is accepted — a plain suffix rollback, exactly
/// as it is today; only the rare alternate hit needs column moves
/// (`KeepPlan.moves`), and those are the reason `supportsColumnMoves` is a
/// capability the caller must check before selecting a shape with
/// alternates.
public struct CBv2MTPTreeShape: Equatable, Sendable {
    /// Depth of the greedy chain: `spineDepth` drafted columns after the seed.
    public let spineDepth: Int
    /// Extra candidates, in column order after the spine.
    public let alternates: [CBv2MTPTreeAlternate]

    public init(spineDepth: Int, alternates: [CBv2MTPTreeAlternate] = []) {
        precondition(spineDepth >= 0, "CBv2MTPTreeShape: spine depth must be >= 0")
        for alternate in alternates {
            precondition(
                alternate.depth >= 1 && alternate.depth <= spineDepth,
                "CBv2MTPTreeShape: alternate depth \(alternate.depth) has no spine parent")
            precondition(
                alternate.rank >= 1,
                "CBv2MTPTreeShape: alternate rank must be >= 1 (rank 0 is the spine)")
        }
        self.spineDepth = spineDepth
        self.alternates = alternates
    }

    /// The shipped shape: one chain of `k` drafts, no alternates.
    public static func chain(_ k: Int) -> CBv2MTPTreeShape {
        CBv2MTPTreeShape(spineDepth: k)
    }

    public var isChain: Bool { alternates.isEmpty }

    /// Verify columns: seed + spine + alternates.
    public var columnCount: Int { 1 + spineDepth + alternates.count }

    /// Drafter forwards this shape costs. Alternates are RANK-k reads of a
    /// forward the spine already paid for, so they add none — that is the
    /// whole economic argument for a tree over a deeper chain.
    public var draftForwardCount: Int { spineDepth }

    /// Highest candidate rank any column needs from a drafter forward.
    /// A drafter that can only return its argmax supports 0.
    public var maximumCandidateRank: Int {
        alternates.map(\.rank).max() ?? 0
    }

    /// True when the accepted path can leave a gap in the column order, so
    /// finalize may have to move KV columns before the suffix rollback.
    public var mayRequireColumnMoves: Bool { !alternates.isEmpty }

    /// Columns in verify order.
    public var columns: [CBv2MTPTreeColumn] {
        var result: [CBv2MTPTreeColumn] = [
            CBv2MTPTreeColumn(parent: nil, draftStep: 0, candidateRank: 0)
        ]
        result.reserveCapacity(columnCount)
        var depth = 1
        while depth <= spineDepth {
            result.append(
                CBv2MTPTreeColumn(parent: depth - 1, draftStep: depth, candidateRank: 0))
            depth += 1
        }
        for alternate in alternates {
            result.append(
                CBv2MTPTreeColumn(
                    parent: alternate.depth - 1, draftStep: alternate.depth,
                    candidateRank: alternate.rank))
        }
        return result
    }

    /// Children of each column, in column order. Order matters: the accept
    /// walk takes the FIRST child whose token matches the target, so the
    /// spine is always preferred and a duplicate alternate (a drafter whose
    /// rank-2 equals its rank-1) is inert rather than ambiguous.
    public func children() -> [[Int]] {
        var result = [[Int]](repeating: [], count: columnCount)
        for (index, column) in columns.enumerated() {
            guard let parent = column.parent else { continue }
            result[parent].append(index)
        }
        return result
    }

    // MARK: - Width budget

    /// Does this shape fit the rectangular certification at `rows` decode
    /// rows? Same inequality the chain is clamped by
    /// (`CBv2MTPRoundDriver.maximumAutomaticDepth`), read with the tree's
    /// column count instead of `1 + k`.
    public func fits(decodeRows: Int, budget: Int) -> Bool {
        guard decodeRows > 0 else { return false }
        return decodeRows * columnCount <= budget
    }

    /// The widest shape of this FAMILY that fits: keep the alternates,
    /// shorten the spine. Returns nil when not even the alternates' own
    /// parents fit. Never widens — the budget is a correctness
    /// certification, and `CBv2MTPConfig.maxAutomaticRectangularTokens`
    /// tightens only.
    public func clamped(decodeRows: Int, budget: Int) -> CBv2MTPTreeShape? {
        guard decodeRows > 0, budget > 0 else { return nil }
        let columnsAvailable = budget / decodeRows
        guard columnsAvailable >= 1 else { return nil }
        if columnCount <= columnsAvailable { return self }
        let minimumSpine = alternates.map(\.depth).max() ?? 0
        // seed + spine + alternates <= columnsAvailable
        let spine = columnsAvailable - 1 - alternates.count
        guard spine >= minimumSpine, spine >= 0 else { return nil }
        return CBv2MTPTreeShape(spineDepth: spine, alternates: alternates)
    }

    // MARK: - Verify mask

    /// Attention adjacency over the verify rectangle's OWN columns:
    /// `mask[query][key] == true` iff `key` is `query` itself or one of its
    /// ancestors. Retained history is always fully visible (subject to the
    /// layer's window) and is not part of this matrix.
    ///
    /// For a chain this is exactly the lower triangle — i.e. the causal
    /// mask the target already applies to a `[1, 1+k]` rectangle — which is
    /// what makes "tree" a strict generalization of the shipped verify
    /// rather than a second code path.
    public func ancestorMask() -> [[Bool]] {
        let all = columns
        var mask = [[Bool]](
            repeating: [Bool](repeating: false, count: all.count), count: all.count)
        for query in 0 ..< all.count {
            var node: Int? = query
            while let current = node {
                mask[query][current] = true
                node = all[current].parent
            }
        }
        return mask
    }

    /// Row-major flattening of `ancestorMask()`, for handing the matrix to a
    /// device array without a second nested loop at the call site.
    public func flattenedAncestorMask() -> [Bool] {
        ancestorMask().flatMap { $0 }
    }

    // MARK: - Accept walk

    public struct Acceptance: Equatable, Sendable {
        /// Columns on the accepted root path, starting at the seed column 0.
        public let path: [Int]
        /// Tokens this round commits, in order. `emitted[i]` is the target's
        /// own token at `path[i]`, so the last one is the round's bonus and
        /// has no KV yet — exactly the chain's contract.
        public let emitted: [Int]
        /// Accepted DRAFT columns (the path without the seed). Equal to the
        /// chain's `accepted` counter when the shape is a chain.
        public var acceptedDraftCount: Int { path.count - 1 }
    }

    /// Target-authoritative accept walk. Exact and greedy: a candidate is
    /// accepted iff its token equals the target's own token at its PARENT
    /// column, which is precisely the chain rule
    /// (`targets[accepted] == drafts[accepted]`) restated for a tree.
    ///
    /// - candidates: token per column, `candidates[0]` unused (the seed is
    ///   already confirmed).
    /// - targets: the target's token at each column (its argmax, or its
    ///   pre-sampled token on the target-prefix path).
    public func acceptWalk(candidates: [Int], targets: [Int]) -> Acceptance {
        precondition(
            candidates.count == columnCount && targets.count == columnCount,
            "CBv2MTPTreeShape.acceptWalk: one candidate and one target per column")
        let childLists = children()
        var path = [0]
        var emitted = [targets[0]]
        var node = 0
        while let next = childLists[node].first(where: { candidates[$0] == targets[node] }) {
            path.append(next)
            emitted.append(targets[next])
            node = next
        }
        return Acceptance(path: path, emitted: emitted)
    }

    // MARK: - KV keep plan

    public struct KeepPlan: Equatable, Sendable {
        /// Column moves to apply to every storage-owning layer BEFORE the
        /// rollback, in this order: the KV written at `from` must end up at
        /// `to`. Empty whenever the accepted path is already a prefix, which
        /// is every outcome of a chain and the common outcome of a tree.
        public let moves: [Move]
        /// Columns to drop from the tail after the moves. This is the number
        /// `CBv2SequenceKV.rollback(_:)` takes today.
        public let rollback: Int
        /// Committed tokens whose KV now sits contiguously at the front.
        public let committedColumnCount: Int

        public struct Move: Equatable, Sendable {
            public let from: Int
            public let to: Int
            public init(from: Int, to: Int) {
                self.from = from
                self.to = to
            }
        }
    }

    /// What finalize must do to the row's KV so the surviving columns are the
    /// accepted path, contiguous, at the front.
    ///
    /// `truncatedTo` is the emitted count actually confirmed by the round's
    /// stop/length semantics, which can be shorter than the accepted path
    /// (`commonEmitted` in `finalizeMTPRound`) and can be zero when the row
    /// had no output budget left. One committed token == one surviving KV
    /// column, exactly as in the chain: the round's LAST emitted token is
    /// the target's bonus, and the column it was read from is the column
    /// holding the token before it — which is why `rejected = columns -
    /// confirmed` is the shipped arithmetic and stays true here.
    public func keepPlan(for acceptance: Acceptance, truncatedTo emitted: Int) -> KeepPlan {
        precondition(
            emitted >= 0 && emitted <= acceptance.emitted.count,
            "CBv2MTPTreeShape.keepPlan: emitted \(emitted) outside the accepted path")
        let kept = acceptance.path.prefix(emitted)
        var moves: [KeepPlan.Move] = []
        // Increasing order is safe: `path` is strictly increasing and
        // `path[i] >= i`, so a destination is always below every source that
        // has not been read yet.
        for (slot, column) in kept.enumerated() where column != slot {
            moves.append(KeepPlan.Move(from: column, to: slot))
        }
        return KeepPlan(
            moves: moves,
            rollback: columnCount - kept.count,
            committedColumnCount: kept.count)
    }

    // MARK: - Shape grammar

    /// `MTPLX_MTP_TREE_DRAFT` values.
    ///
    ///   off                      the shipped chain (this switch inert)
    ///   spine=5                  a chain of 5, expressed as a tree
    ///   spine=5,alt=1:1          + the drafter's runner-up at depth 1
    ///   spine=4,alt=1:1,alt=2:1  + one at depth 2 as well
    ///
    /// No length, prefill size or ring size is expressible, deliberately.
    public static func parse(_ raw: String) -> CBv2MTPTreeShape? {
        let text = raw.trimmingCharacters(in: .whitespaces).lowercased()
        guard !text.isEmpty, text != "off", text != "0", text != "false" else { return nil }
        var spine: Int?
        var alternates: [CBv2MTPTreeAlternate] = []
        for field in text.split(separator: ",") {
            let parts = field.split(separator: "=", maxSplits: 1)
            guard parts.count == 2 else { return nil }
            let value = String(parts[1])
            switch parts[0] {
            case "spine", "k":
                guard let depth = Int(value), depth >= 0 else { return nil }
                spine = depth
            case "alt":
                let pair = value.split(separator: ":", maxSplits: 1)
                guard let first = pair.first, let depth = Int(first), depth >= 1 else {
                    return nil
                }
                let rank = pair.count == 2 ? Int(pair[1]) : 1
                guard let rank, rank >= 1 else { return nil }
                alternates.append(CBv2MTPTreeAlternate(depth: depth, rank: rank))
            default:
                return nil
            }
        }
        guard let spine else { return nil }
        guard alternates.allSatisfy({ $0.depth <= spine }) else { return nil }
        return CBv2MTPTreeShape(spineDepth: spine, alternates: alternates)
    }
}
