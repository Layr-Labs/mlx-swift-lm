// Copyright © 2026 Eigen Labs.

/// One proposed long-prefill range split into its demanded target and its
/// original end. This is scheduler provenance, not a checkpoint or hit proof.
struct CBv2DemandedCheckpointContinuation: Sendable, Equatable {
    let start: Int
    let target: Int
    let originalEnd: Int
    let soloStripeTokens: Int
}

struct CBv2DemandedCheckpointRange: Sendable {
    let count: Int
    let continuation: CBv2DemandedCheckpointContinuation?
}

/// Optional state needs a wrapper because a dictionary assignment of nil
/// removes its entry. Every changed row can restore its exact prior state.
struct CBv2DemandedCheckpointContinuationUndo: Sendable {
    let previous: CBv2DemandedCheckpointContinuation?
}
