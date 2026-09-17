import Cmlx
import MLX
import MLXLMCommon

/// A prefill checkpoint preserves the exact pre-mixer HC residual history.
/// It is not Qwen3.5's normalized hidden state, and never contains speculative
/// head KV. Fresh mutable QSA head caches are created for each restore.
extension Qwen4ExpInlineMTPAssistant: CBv2MTPPrefixCheckpointCoding {
    private final class PrefixCheckpoint: CBv2MTPPrefixCheckpoint {
        let owner: ObjectIdentifier
        let targetInputCount: Int
        let hidden: MLXArray
        let tokens: MLXArray
        let frontier: MLXArray

        init(owner: ObjectIdentifier, count: Int, arrays: [MLXArray]) {
            self.owner = owner
            targetInputCount = count
            hidden = arrays[0]
            tokens = arrays[1]
            frontier = arrays[2]
        }

        var evaluationTargets: [MLXArray] { [hidden, tokens, frontier] }
        var materializedBytes: Int {
            evaluationTargets.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }
    }

    public var prefixCheckpointCodecID: String {
        "qwen4-hc-residual-history-v1:\(prefixCheckpointGeometry.verification)"
    }

    public func prefixCheckpointTensorDescriptors(targetInputCount count: Int)
        -> [CBv2CheckpointTensorDescriptor]?
    {
        let geometry = prefixCheckpointGeometry
        guard count > 1, count <= geometry.maximumLength, geometry.width > 0,
            let dtype = CBv2CheckpointDType(geometry.dtype),
            [.float16, .bfloat16, .float32].contains(dtype)
        else { return nil }
        return try? [
            .init(role: .assistantHidden, shape: [1, count - 1, geometry.width], dtype: dtype),
            .init(role: .assistantTokens, shape: [1, count - 1], dtype: .int32),
            .init(role: .assistantFrontier, shape: [1, 1, geometry.width], dtype: dtype),
        ]
    }

    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount count: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        guard let state = requestState as? RequestState,
            state.owner == ObjectIdentifier(self), !state.isReleased,
            !state.roundInFlight, state.cacheOffset == 0,
            count > 1, state.committedInputCount == count - 1,
            !state.backlogHidden.isEmpty,
            state.backlogHidden.count == state.backlogTokens.count,
            let frontier = state.targetHiddenFrontier,
            let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: count)
        else { return nil }
        let arrays = [concatenated(state.backlogHidden, axis: 1),
                      concatenated(state.backlogTokens, axis: 1), frontier]
        guard matches(arrays, descriptors: descriptors) else { return nil }
        // Byte-preserving compaction only, charged by complete-checkpoint
        // capture before this call. Never normalize or cast restored history.
        return PrefixCheckpoint(owner: ObjectIdentifier(self), count: count,
            arrays: arrays.map { MLX.where(MLXArray(true), $0, $0) })
    }

    public func restorePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint)
        -> (any CBv2MTPRequestState)?
    {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let state = makeRequestState() as? RequestState
        else { return nil }
        // This checkpoint has no assistant KV yet. Use the same initial-head
        // policy as an uncached prompt; retain the full trusted bytes for
        // discard/retry and codec validation. The old explicit cold-only
        // experiment still replays restored history.
        state.coldPromptReplayEligible = skipUnprimedRestoredReplay
        state.backlogHidden = [checkpoint.hidden]
        state.backlogTokens = [checkpoint.tokens]
        state.targetHiddenFrontier = checkpoint.frontier
        return state
    }

    public func encodePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> [MLXArray]? {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self)
        else { return nil }
        return checkpoint.evaluationTargets
    }

    public func decodePrefixCheckpoint(tensors: [MLXArray], prefixTokens: [Int])
        -> (any CBv2MTPPrefixCheckpoint)?
    {
        guard let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: prefixTokens.count),
            matches(tensors, descriptors: descriptors),
            prefixTokens.allSatisfy({ $0 >= 0 && $0 < prefixCheckpointGeometry.vocabulary }),
            tokensMatch(tensors[1], prefixTokens: prefixTokens)
        else { return nil }
        // Artifact, numerical profile, template and tenant identities are
        // checked by the enclosing authenticated complete-checkpoint import.
        return PrefixCheckpoint(owner: ObjectIdentifier(self), count: prefixTokens.count, arrays: tensors)
    }

    private func matches(_ arrays: [MLXArray], descriptors: [CBv2CheckpointTensorDescriptor]) -> Bool {
        arrays.count == descriptors.count && zip(arrays, descriptors).allSatisfy {
            $0.shape == $1.shape && $0.dtype == $1.dtype.mlxDType
        }
    }

    private func tokensMatch(_ tokens: MLXArray, prefixTokens: [Int]) -> Bool {
        guard let info = try? tokens.evaluatedBufferInfo(), info.isRowContiguous,
            info.dataElements == tokens.size, let pointer = mlx_array_data_int32(tokens.ctx)
        else { return false }
        // Compare imported evaluated storage directly. A lazy input is refused,
        // not evaluated here, so staging never creates an unreserved token copy.
        return withExtendedLifetime(tokens) {
            for (index, expected) in prefixTokens.dropFirst().enumerated() {
                if Int(pointer[index]) != expected { return false }
            }
            return true
        }
    }
}
