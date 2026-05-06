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
}
