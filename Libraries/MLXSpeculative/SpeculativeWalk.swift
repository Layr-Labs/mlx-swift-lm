// Copyright © 2026 Apple Inc.

import Foundation

/// Accept/reject walker for speculative decoding. Pure Swift — no MLX
/// dependency so it can be exhaustively unit-tested.
public enum SpeculativeWalk {

    /// Single-row greedy accept-prefix walker.
    ///
    /// The speculative-decoding contract: `main` contains `k + 1` tokens
    /// (one verify per draft plus one "bonus" token that the target sampled
    /// from the position after the last draft). `draft` contains the `k`
    /// tokens the drafter proposed. Walk left-to-right, accept while
    /// `draft[i] == main[i]`, and always emit one "correction" or "bonus"
    /// token from `main` past the accepted prefix.
    ///
    /// - Returns: `(acceptedCount, emittedTokens)` where
    ///   `emittedTokens.count == acceptedCount + 1`.
    public static func single(draft: [Int], main: [Int]) -> (Int, [Int]) {
        // If draft is empty, just emit main[0] (the bonus).
        guard !draft.isEmpty else {
            precondition(!main.isEmpty, "main must contain at least the bonus")
            return (0, [main[0]])
        }
        precondition(
            main.count >= draft.count + 1,
            "main must have at least draft.count + 1 tokens (drafts + bonus)"
        )
        var accepted = 0
        for i in 0 ..< draft.count {
            if main[i] != draft[i] { break }
            accepted += 1
        }
        // Emit main[0...accepted] inclusive — that's accepted + 1 tokens.
        return (accepted, Array(main[0 ... accepted]))
    }

    /// Multi-row greedy accept-prefix walker with per-row emit budgets.
    ///
    /// - Parameters:
    ///   - draft: per-row draft tokens, length `[B][k]`.
    ///   - main: per-row verify tokens, length `[B][k+1]`.
    ///   - budgets: per-row max emit count. A row emits at most
    ///     `budgets[i]` tokens; the accept count is capped accordingly so
    ///     the `emitted.count == accepted + 1` invariant holds per row.
    /// - Returns: `(acceptedPerRow, emittedPerRow)`.
    public static func batched(
        draft: [[Int]], main: [[Int]], budgets: [Int]
    ) -> ([Int], [[Int]]) {
        precondition(
            draft.count == main.count && draft.count == budgets.count,
            "batched: all inputs must have the same outer length B"
        )
        var acceptedOut: [Int] = []
        var emittedOut: [[Int]] = []
        acceptedOut.reserveCapacity(draft.count)
        emittedOut.reserveCapacity(draft.count)
        for i in 0 ..< draft.count {
            var (a, e) = single(draft: draft[i], main: main[i])
            let budget = Swift.max(0, budgets[i])
            if e.count > budget {
                e = Array(e.prefix(budget))
                a = Swift.max(0, e.count - 1)
            }
            acceptedOut.append(a)
            emittedOut.append(e)
        }
        return (acceptedOut, emittedOut)
    }
}
