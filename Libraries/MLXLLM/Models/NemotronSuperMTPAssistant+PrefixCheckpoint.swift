import MLX
import MLXLMCommon

/// Prompt-history checkpoints of the Super assistant. While a request
/// prefills, its state is only the queued trusted history: the shifted
/// (next token, previous target hidden) pairs and the frontier hidden; the
/// FP8 draft history is built from them at the first draft step, in
/// canonical `historyChunkRows` chunks of the whole queue. A checkpoint is a
/// copy of that queue at an exact target boundary, so a restored request
/// builds the same draft history as a cold one whatever its chunks were. It
/// never holds draft KV or a speculative round; restore makes a fresh state.
extension NemotronSuperMTPAssistant: CBv2MTPPrefixCheckpointDrafter {
    final class PrefixCheckpoint: CBv2MTPPrefixCheckpoint {
        let owner: ObjectIdentifier
        let targetInputCount: Int
        /// `[1, count - 1, hidden]` target hidden paired with `tokens`.
        let hidden: MLXArray
        /// `[1, count - 1]` next input tokens.
        let tokens: MLXArray
        /// `[1, 1, hidden]` target hidden of the last input.
        let frontier: MLXArray

        init(
            owner: ObjectIdentifier, targetInputCount: Int, hidden: MLXArray, tokens: MLXArray,
            frontier: MLXArray
        ) {
            self.owner = owner
            self.targetInputCount = targetInputCount
            self.hidden = hidden
            self.tokens = tokens
            self.frontier = frontier
        }

        var evaluationTargets: [MLXArray] { [hidden, tokens, frontier] }
        var materializedBytes: Int {
            evaluationTargets.reduce(0) { total, array in
                let (sum, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : sum
            }
        }
    }

    /// Only a prompt state that has observed exactly `count` target inputs and
    /// has not drafted; nil otherwise, and the engine then captures nothing.
    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount count: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        let width = target.configuration.hiddenSize
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released,
            !s.started, s.stagedInputCount == 0, (s.history?.offset ?? 0) == 0,
            s.committedInputCount == count, count > 1,
            !s.pendingTokens.isEmpty, s.pendingTokens.count == s.pendingHidden.count,
            let frontier = s.frontier, frontier.shape == [1, 1, width]
        else { return nil }
        let tokens = concatenated(s.pendingTokens, axis: 1)
        let hidden = concatenated(s.pendingHidden, axis: 1)
        let rows = count - 1
        guard tokens.shape == [1, rows], hidden.shape == [1, rows, width] else { return nil }
        // Select copies the bytes without arithmetic and drops the views of
        // whole prefill chunks the queue entries keep alive.
        func copy(_ array: MLXArray) -> MLXArray { MLX.where(MLXArray(true), array, array) }
        return PrefixCheckpoint(
            owner: ObjectIdentifier(self), targetInputCount: count, hidden: copy(hidden),
            tokens: copy(tokens), frontier: copy(frontier))
    }

    /// A fresh request state holding the checkpoint's queue; the engine
    /// configures its history bound before the request continues.
    public func restorePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint)
        -> (any CBv2MTPRequestState)?
    {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let s = makeRequestState() as? State
        else { return nil }
        s.pendingTokens = [checkpoint.tokens]
        s.pendingHidden = [checkpoint.hidden]
        s.frontier = checkpoint.frontier
        s.committedInputCount = checkpoint.targetInputCount
        return s
    }
}
