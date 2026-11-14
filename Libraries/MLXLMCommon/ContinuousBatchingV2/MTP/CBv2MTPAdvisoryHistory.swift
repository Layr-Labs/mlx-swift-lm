// Copyright © 2026 Eigen Labs.

import MLX

/// A request-stateful drafter whose request state is advisory.
///
/// Advisory state is rebuilt from the target rows the engine hands over and
/// is never needed for correctness: with exact acceptance a draft is kept
/// only when it equals the target's own token, so losing the state, or
/// starting it later, changes which tokens are proposed and never which are
/// emitted.
///
/// For such a drafter the engine keeps complete prefix checkpoints on the
/// generic paged path: they are captured and adopted with no assistant state,
/// exactly as for an engine without a drafter. A request that adopts a
/// checkpoint gets a new, empty request state, and its first observation
/// starts at the restored boundary, not at position zero.
///
/// Every other request-stateful drafter is unchanged: without this
/// conformance, or with `requestStateIsAdvisory == false`, its history must
/// travel with a checkpoint and the engine keeps refusing checkpoints that
/// cannot carry it. Typical acceptance is not exact, so it also keeps the
/// refusal.
public protocol CBv2MTPAdvisoryHistoryDrafter: CBv2MTPRequestStatefulDrafter {
    /// True when no emitted token can depend on this drafter's request state.
    var requestStateIsAdvisory: Bool { get }
    /// `observeCommittedTarget(_:requestState:)` with the absolute target
    /// position of the observation's first hidden row. The engine calls this
    /// form instead for an advisory drafter. Positions of one request only
    /// grow; after an adopted checkpoint the first one is the checkpoint's
    /// position.
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, startingAt position: Int,
        requestState: any CBv2MTPRequestState)
}
