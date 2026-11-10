import MLX

/// Immutable trusted assistant history at an exact target input boundary.
/// It never owns mutable draft KV or an unconfirmed speculative round.
public protocol CBv2MTPPrefixCheckpoint: AnyObject {
    var targetInputCount: Int { get }
    var materializedBytes: Int { get }
    var evaluationTargets: [MLXArray] { get }
}

/// Optional capability for assistants that can restore the same prompt history
/// and first-draft geometry as cold execution. Checkpoints stay model-owned.
public protocol CBv2MTPPrefixCheckpointDrafter: CBv2MTPRequestStatefulDrafter {
    /// True when a cached text request that cannot speculate (a stop string,
    /// logprobs, a logit bias, a penalty, a token constraint) still records
    /// this drafter's prompt history, so its prefix checkpoints carry the
    /// assistant state a speculating request restores. Its prompt then runs
    /// the hidden-returning prefill forward, whose logits must be the
    /// ordinary prefill forward's. Default false.
    var checkpointsNonSpeculatingRows: Bool { get }
    func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount: Int
    ) -> (any CBv2MTPPrefixCheckpoint)?
    func restorePrefixCheckpoint(
        _ checkpoint: any CBv2MTPPrefixCheckpoint
    ) -> (any CBv2MTPRequestState)?
}

extension CBv2MTPPrefixCheckpointDrafter {
    public var checkpointsNonSpeculatingRows: Bool { false }
}
