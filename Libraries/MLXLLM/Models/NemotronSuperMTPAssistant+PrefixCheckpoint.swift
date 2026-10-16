import MLX
import MLXLMCommon

/// Prompt-history checkpoints of the Super assistant. While a request
/// prefills, its state is the FP8 draft history of every whole canonical
/// `historyChunkRows` block observed so far plus the queued partial block: the
/// shifted (next token, previous target hidden) pairs after the last block
/// and the frontier hidden. A checkpoint copies the history's FP8 codes and
/// the queue at an exact target boundary. Restore writes the same codes, and
/// the mirror decoded elementwise from them, into the request's own history,
/// and the first draft step writes the partial block in the same chunk as a
/// cold request, so a restored request builds the same draft history whatever
/// its scheduler chunks were. It never holds a speculative round; restore
/// makes a fresh state.
extension NemotronSuperMTPAssistant: CBv2MTPPrefixCheckpointDrafter {
    final class PrefixCheckpoint: CBv2MTPPrefixCheckpoint {
        let owner: ObjectIdentifier
        let targetInputCount: Int
        /// FP8 history codes `[K, V]`, each `[1, kvHeads, blocks x chunk, headDim]`;
        /// empty before the first whole block.
        let codes: [MLXArray]
        /// `[1, queued, hidden]` target hidden of the partial block, paired with
        /// `tokens`; nil when the pairs end on a block boundary.
        let hidden: MLXArray?
        /// `[1, queued]` next input tokens of the partial block.
        let tokens: MLXArray?
        /// `[1, 1, hidden]` target hidden of the last input.
        let frontier: MLXArray

        init(
            owner: ObjectIdentifier, targetInputCount: Int, codes: [MLXArray],
            hidden: MLXArray?, tokens: MLXArray?, frontier: MLXArray
        ) {
            self.owner = owner
            self.targetInputCount = targetInputCount
            self.codes = codes
            self.hidden = hidden
            self.tokens = tokens
            self.frontier = frontier
        }

        var historyRows: Int { codes.first?.dim(2) ?? 0 }
        var evaluationTargets: [MLXArray] { codes + [hidden, tokens, frontier].compactMap { $0 } }
        var materializedBytes: Int {
            evaluationTargets.reduce(0) { total, array in
                let (sum, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : sum
            }
        }
    }

    /// Only a prompt state that has observed exactly `count` target inputs and
    /// has not drafted, with whole blocks in its history and the rest queued;
    /// nil otherwise, and the engine then captures nothing.
    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount count: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        let width = target.configuration.hiddenSize
        let chunk = workspace.graph.historyChunkRows
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released,
            !s.started, s.stagedInputCount == 0, s.restoredCodes.isEmpty,
            s.committedInputCount == count, count > 1,
            s.pendingTokens.count == s.pendingHidden.count,
            let frontier = s.frontier, frontier.shape == [1, 1, width]
        else { return nil }
        let drained = s.history?.offset ?? 0
        let queued = s.pendingTokens.reduce(0) { $0 + $1.dim(1) }
        guard drained % chunk == 0, queued < chunk, drained + queued == count - 1 else {
            return nil
        }
        // Select copies the bytes without arithmetic and drops the views of
        // whole buffers and prefill chunks the state keeps alive.
        func copy(_ array: MLXArray) -> MLXArray { MLX.where(MLXArray(true), array, array) }
        var tokens: MLXArray?
        var hidden: MLXArray?
        if queued > 0 {
            let ids = concatenated(s.pendingTokens, axis: 1)
            let rows = concatenated(s.pendingHidden, axis: 1)
            guard ids.shape == [1, queued], rows.shape == [1, queued, width] else { return nil }
            tokens = copy(ids)
            hidden = copy(rows)
        }
        return PrefixCheckpoint(
            owner: ObjectIdentifier(self), targetInputCount: count,
            codes: (s.history?.committedCodes() ?? []).map(copy), hidden: hidden, tokens: tokens,
            frontier: copy(frontier))
    }

    /// A fresh request state holding the checkpoint's codes and queue; the
    /// engine configures its history bound, which writes the codes, before the
    /// request continues.
    public func restorePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint)
        -> (any CBv2MTPRequestState)?
    {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let s = makeRequestState() as? State
        else { return nil }
        s.restoredCodes = checkpoint.codes
        if let tokens = checkpoint.tokens, let hidden = checkpoint.hidden {
            s.pendingTokens = [tokens]
            s.pendingHidden = [hidden]
        }
        s.frontier = checkpoint.frontier
        s.committedInputCount = checkpoint.targetInputCount
        return s
    }
}
