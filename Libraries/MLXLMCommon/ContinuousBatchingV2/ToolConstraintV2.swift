// Copyright © 2026 Eigen Labs.
//
// Per-request inference-time token constraints for ContinuousBatchingV2.
// The engine owns only this format-neutral contract. Model families compile
// their actual output grammar (Gemma tool envelopes, JSON schema, and so on)
// before submission and hand the immutable machine to the sampler.

import Foundation

/// Privacy-safe, low-cardinality tool-choice modes.
public enum CBv2TokenConstraintMode: String, Sendable {
    case none
    case auto
    case required
    case named
}

/// Immutable token-level automaton compiled before a request enters CBv2.
///
/// State is held per request by ``CBv2DefaultSampler``. Implementations must
/// be deterministic and row-local: the same `(state, remainingTokens)` must
/// always return the same token set regardless of batch composition.
public protocol CBv2TokenConstraint: AnyObject, Sendable {
    var mode: CBv2TokenConstraintMode { get }
    var initialState: Int { get }
    /// Exact request output budget used to reserve grammar-closing tokens.
    var maxTokens: Int { get }

    /// Token used only if the compiler/runtime reaches an impossible state.
    /// It must be a request stop token, so the engine terminates safely while
    /// surfacing the typed impossible-state failure instead of sampling an
    /// unconstrained token.
    var fallbackTokenID: Int { get }

    /// Sorted, unique allowed token ids for one state. `remainingTokens`
    /// includes the token selected by this call. Returning an empty set is an
    /// impossible state.
    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int]

    /// Advance after the engine materializes and confirms one sampled token.
    /// Returning nil means the sampled token was not a valid transition.
    func nextState(state: Int, tokenID: Int) -> Int?

    /// Optional device-mask cache key for `allowedTokenIDs(state:remainingTokens:)`.
    /// Two calls on the SAME constraint instance whose keys are equal and
    /// non-nil MUST return identical token sets; the sampler then reuses the
    /// uploaded mask and skips `allowedTokenIDs` entirely. nil (the default)
    /// disables caching for that call. A constraint whose allowed set ignores
    /// `remainingTokens` while it is positive can return `state` then.
    func maskCacheKey(state: Int, remainingTokens: Int) -> Int?

    /// Opt-in to chained (pipelined) decode. The engine then samples step
    /// N+1 before step N's token reaches the host, masking it with the
    /// constraint state confirmed through N-1 and the budget N+1 will have.
    /// When step N's confirmed token leaves that state, the engine redraws
    /// N+1 from the same logits with the correct mask, so the output is
    /// exactly the unchained stream. Requires only the protocol's existing
    /// determinism: `allowedTokenIDs` may be called for a state/budget pair
    /// whose sample is later discarded, and must have no observable effect
    /// beyond its return value. `nextState` is still called only for
    /// host-confirmed tokens, in order. Default false.
    var allowsChainedDecode: Bool { get }
}

extension CBv2TokenConstraint {
    public func maskCacheKey(state: Int, remainingTokens: Int) -> Int? { nil }
    public var allowsChainedDecode: Bool { false }
}

enum CBv2TokenConstraintFailure {
    static let impossibleState = "tool_constraint_impossible_state"
}
