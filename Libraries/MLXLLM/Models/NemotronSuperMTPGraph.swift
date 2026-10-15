import MLX
import MLXLMCommon
import MLXNN

/// Graph-only execution of the embedded Super prediction head. Same equations
/// as the eager `forward`; no routing readback, scalar `.item()` or tile
/// `eval`. Routed experts use `ModelOptNVFP4SwitchLinear.lazyDecode`, shared
/// experts `NemotronSuperMTPLazyNVFP4`, attention KV the request-owned
/// unit-scale `NemotronSuperFP8KVHistory`. The caller admits `GraphQuote`
/// before any call and checks the returned validity at its completion fence.
extension NemotronSuperMTPModule {
    /// Explicit array allowances, 32 bytes per intermediate element unless a
    /// component helper defines its own quote. Every term is live in one draft
    /// forward. Backend scratch is a separate caller-owned allowance.
    public struct GraphQuote: Sendable, Equatable {
        /// fc1 and fc2 selected-expert gathers and calibration intermediates.
        public let routedArrayBytes: Int
        public let routedFC1GraphUnits: Int
        public let routedFC2GraphUnits: Int
        public let sharedArrayBytes: Int
        /// Fusion, projections, FP8 KV append and attention scores over the
        /// complete admitted history.
        public let attentionArrayBytes: Int
        /// Router, latent projections, residual and norms.
        public let mixerArrayBytes: Int
        /// One trusted-history KV append of `historyChunkRows` rows.
        public let historyChunkArrayBytes: Int
        public let historyChunkRows: Int

        public var draftArrayBytes: Int {
            routedArrayBytes + sharedArrayBytes + attentionArrayBytes + mixerArrayBytes
        }
    }

    var attentionBlock: NemotronH35MTPAttentionBlock {
        layers[0] as! NemotronH35MTPAttentionBlock
    }
    var expertBlock: NemotronH35MTPExpertBlock { layers[1] as! NemotronH35MTPExpertBlock }

    private struct GraphProjections {
        let fc1: ModelOptNVFP4SwitchLinear
        let fc2: ModelOptNVFP4SwitchLinear
        let sharedUp: ModelOptCalibratedLinear
        let sharedDown: ModelOptCalibratedLinear
    }

    /// Only the exact ModelOpt NVFP4 layout produced by `load` is graph-capable.
    private func graphProjections() throws -> GraphProjections {
        let mixer = expertBlock.mixer
        guard let fc1 = mixer.switchMLP.fc1 as? ModelOptNVFP4SwitchLinear,
            let fc2 = mixer.switchMLP.fc2 as? ModelOptNVFP4SwitchLinear,
            let shared = mixer.sharedExperts,
            let up = shared.upProj as? ModelOptCalibratedLinear, case .nvfp4 = up.format,
            let down = shared.downProj as? ModelOptCalibratedLinear, case .nvfp4 = down.format,
            mixer.fc1LatentProj != nil, mixer.fc2LatentProj != nil
        else { throw Failure.inventory("graph-capable NVFP4 experts") }
        return GraphProjections(fc1: fc1, fc2: fc2, sharedUp: up, sharedDown: down)
    }

    public func graphQuote(maximumHistoryTokens: Int, historyChunkRows: Int) throws -> GraphQuote {
        let c = configuration
        guard maximumHistoryTokens > 0, historyChunkRows > 0, let latent = c.moeLatentSize else {
            throw Failure.budget
        }
        func bytes(_ terms: [(Int, Int)]) throws -> Int {
            var total = 1024
            for (a, b) in terms {
                let (elements, o1) = a.multipliedReportingOverflow(by: b)
                let (value, o2) = elements.multipliedReportingOverflow(by: 32)
                let (next, o3) = total.addingReportingOverflow(value)
                guard !o1, !o2, !o3 else { throw Failure.budget }
                total = next
            }
            return total
        }
        let projections = try graphProjections()
        let assignments = c.numExpertsPerTok
        let fc1 = try projections.fc1.lazyDecodeQuote(assignments: assignments)
        let fc2 = try projections.fc2.lazyDecodeQuote(assignments: assignments)
        let shared = try NemotronSuperMTPLazyNVFP4.arrayBytes(
            rows: 1, inputDimensions: c.hiddenSize,
            outputDimensions: c.moeSharedExpertIntermediateSize)
        let sharedDown = try NemotronSuperMTPLazyNVFP4.arrayBytes(
            rows: 1, inputDimensions: c.moeSharedExpertIntermediateSize,
            outputDimensions: c.hiddenSize)
        let h = c.hiddenSize
        let headDim = c.headDim ?? (h / c.numAttentionHeads)
        let queryWidth = c.numAttentionHeads * headDim
        let kvWidth = c.numKeyValueHeads * headDim
        // [2H] concatenation, fused/norm/residual rows, Q, new K/V, codec
        // temporaries and FP32 scores/probabilities over the whole history.
        let attention = try bytes([
            (1, 6 * h), (1, 2 * queryWidth), (4, kvWidth),
            (2 * c.numAttentionHeads, maximumHistoryTokens + 1),
        ])
        // Router scores/selection over every expert, routed weighted sum and
        // latent projections, shared-expert activation, residual and norms.
        let mixer = try bytes([
            (8, c.nRoutedExperts), (assignments, 2 * latent), (2, latent),
            (2, c.moeSharedExpertIntermediateSize), (6, h),
        ])
        let history = try bytes([(historyChunkRows, 8 * h + 4 * kvWidth)])
        func sum(_ values: Int...) throws -> Int {
            var total = 0
            for value in values {
                let (next, overflow) = total.addingReportingOverflow(value)
                guard !overflow else { throw Failure.budget }
                total = next
            }
            return total
        }
        return GraphQuote(
            routedArrayBytes: try sum(fc1.arrayBytes, fc2.arrayBytes),
            routedFC1GraphUnits: fc1.graphUnits, routedFC2GraphUnits: fc2.graphUnits,
            sharedArrayBytes: try sum(shared, sharedDown),
            attentionArrayBytes: attention, mixerArrayBytes: mixer,
            historyChunkArrayBytes: history, historyChunkRows: historyChunkRows)
    }

    /// Admission refusals precede the first KV append or graph operation.
    private func validate(quote: GraphQuote, projections: GraphProjections) throws {
        let c = configuration
        let fc1 = try projections.fc1.lazyDecodeQuote(assignments: c.numExpertsPerTok)
        let fc2 = try projections.fc2.lazyDecodeQuote(assignments: c.numExpertsPerTok)
        guard fc1.arrayBytes <= quote.routedArrayBytes, fc2.arrayBytes <= quote.routedArrayBytes,
            fc1.graphUnits <= quote.routedFC1GraphUnits,
            fc2.graphUnits <= quote.routedFC2GraphUnits
        else { throw Failure.budget }
        _ = try NemotronSuperMTPLazyNVFP4.validate(
            projections.sharedUp, inputShape: [1, 1, c.hiddenSize], inputDType: weightDType,
            admittedArrayBytes: quote.sharedArrayBytes)
        _ = try NemotronSuperMTPLazyNVFP4.validate(
            projections.sharedDown, inputShape: [1, 1, c.moeSharedExpertIntermediateSize],
            inputDType: weightDType, admittedArrayBytes: quote.sharedArrayBytes)
    }

    private func validateGraphInputs(_ hidden: MLXArray, _ embedding: MLXArray) throws {
        guard hidden.shape == [1, 1, configuration.hiddenSize],
            embedding.shape == hidden.shape, hidden.dtype == weightDType,
            embedding.dtype == weightDType
        else { throw Failure.shape("MTP graph input") }
    }

    private func validateHistory(_ history: NemotronSuperFP8KVHistory, rows: Int) throws {
        let headDim =
            configuration.headDim ?? (configuration.hiddenSize / configuration.numAttentionHeads)
        guard history.heads == configuration.numKeyValueHeads,
            history.headDimension == headDim, history.dtype == weightDType,
            rows <= history.capacityTokens - history.offset
        else { throw Failure.cachePolicy }
    }

    /// One draft step. Returns post-final-norm draft hidden `[1, 1, H]` and the
    /// device-side calibration validity of every runtime-scaled projection.
    public func graphForward(
        postNormHidden: MLXArray, nextEmbedding: MLXArray,
        history: NemotronSuperFP8KVHistory, quote: GraphQuote
    ) throws -> (hidden: MLXArray, valid: MLXArray) {
        try validateGraphInputs(postNormHidden, nextEmbedding)
        try validateHistory(history, rows: 1)
        let projections = try graphProjections()
        try validate(quote: quote, projections: projections)
        let attended = attentionBlock(
            hidden: postNormHidden, embedding: nextEmbedding, cache: history)
        let block = expertBlock
        let mixer = block.mixer
        let x = block.norm(attended)
        let (indices, scores) = mixer.gate(x)
        let selected = indices.reshaped([-1]).asType(.int32)
        let assignments = selected.dim(0)
        let routedInput = mixer.fc1LatentProj!(x)
        let up = try projections.fc1.lazyDecode(
            routedInput, selected, admittedArrayBytes: quote.routedArrayBytes,
            admittedGraphUnits: quote.routedFC1GraphUnits)
        let activated = maximum(up, MLXArray(0))
        let down = try projections.fc2.lazyDecode(
            activated * activated, selected, admittedArrayBytes: quote.routedArrayBytes,
            admittedGraphUnits: quote.routedFC2GraphUnits)
        // Same layout and reduction as NemotronHMoE: [1, 1, topK, latent].
        var routed = down.reshaped(1, 1, assignments, down.dim(-1))
        routed = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(routed.dtype)
        routed = mixer.fc2LatentProj!(routed)
        let sharedUp = try NemotronSuperMTPLazyNVFP4.forward(
            projections.sharedUp, x, admittedArrayBytes: quote.sharedArrayBytes)
        let hiddenShared = maximum(sharedUp.value, MLXArray(0))
        let sharedDown = try NemotronSuperMTPLazyNVFP4.forward(
            projections.sharedDown, hiddenShared * hiddenShared,
            admittedArrayBytes: quote.sharedArrayBytes)
        let output = block.finalNorm(attended + (routed + sharedDown.value))
        return (output, logicalAnd(sharedUp.valid, sharedDown.valid))
    }

    /// Trusted target rows only need the head's K/V projections; their query,
    /// attention, MoE and final norm cannot affect KV. Appends `[1, n]` rows.
    public func appendTrustedHistory(
        hidden: MLXArray, embedding: MLXArray, history: NemotronSuperFP8KVHistory,
        quote: GraphQuote
    ) throws {
        let rows = hidden.dim(1)
        guard hidden.ndim == 3, hidden.dim(0) == 1, rows > 0, rows <= quote.historyChunkRows,
            hidden.dim(2) == configuration.hiddenSize, embedding.shape == hidden.shape,
            hidden.dtype == weightDType, embedding.dtype == weightDType,
            rows <= history.maximumAppendRows
        else { throw Failure.shape("MTP history") }
        try validateHistory(history, rows: rows)
        _ = attentionBlock.appendTrustedKV(hidden: hidden, embedding: embedding, cache: history)
    }
}
