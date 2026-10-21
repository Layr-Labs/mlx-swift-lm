import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Graph-only execution of the embedded Super prediction head. Same equations
/// as the eager `forward`, on the fastpath-owned device-resident ModelOpt
/// wrappers the target uses: `ModelOptNVFP4SwitchLinear.graphForward` for
/// routed experts and `ModelOptCalibratedLinear.graphForward` for shared
/// experts. No routing readback, scalar `.item()` or tile `eval`. Attention KV
/// is the request-owned unit-scale `NemotronSuperFP8KVHistory`. The caller
/// admits `GraphQuote` before any call and checks the returned routing
/// validity at its completion fence.
extension NemotronSuperMTPModule {
    /// Explicit array allowances of one draft forward, all assumed live
    /// together. ModelOpt terms are the wrappers' own workspace quotes; other
    /// intermediates are charged at 32 bytes per element. Backend scratch is a
    /// separate caller-owned allowance.
    public struct GraphQuote: Sendable, Equatable {
        /// fc1 and fc2 routed activation, sums and outputs.
        public let routedArrayBytes: Int
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

    /// Selects how the head's ModelOpt modules run their ordinary forward.
    /// `graphForward` always uses the graph wrappers; this switches the eager
    /// `forward` between the same graph route and the eager reference.
    public func setModelOptExecution(_ execution: ModelOptExecution) {
        for (_, module) in leafModules().flattened() {
            (module as? ModelOptCalibratedLinear)?.execution = execution
            (module as? ModelOptNVFP4SwitchLinear)?.execution = execution
        }
    }

    /// FP32 partials of a draft step's one-query attention when MLX runs it
    /// fused (`fast::scaled_dot_product_attention` on the GPU: head dim 64, 96,
    /// 128 or 256, query rows times the GQA factor at most 32, no mask). Its
    /// vector kernels never materialize scores; the two-pass kernel keeps
    /// `queryHeads x blocks x (headDim + 2)` FP32 partials, at most 1,024
    /// blocks unless `MLX_SDPA_BLOCKS` sets more. Nil when the step would fall
    /// back to the unfused path, whose scores grow with the history.
    var fusedDraftAttentionBytes: Int? {
        let c = configuration
        let headDim = c.headDim ?? (c.hiddenSize / c.numAttentionHeads)
        guard Device.defaultDevice().deviceType == .gpu, [64, 96, 128, 256].contains(headDim),
            c.numKeyValueHeads > 0, c.numAttentionHeads % c.numKeyValueHeads == 0,
            c.numAttentionHeads / c.numKeyValueHeads <= 32
        else { return nil }
        var blocks = 1024
        if let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"], let value = Int(raw),
            value > 0
        {
            blocks = max(blocks, (value + 31) / 32 * 32)
        }
        let (bytes, overflow) = (c.numAttentionHeads * (headDim + 2) * 4)
            .multipliedReportingOverflow(by: blocks)
        return overflow ? nil : bytes
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
        func sum(_ values: Int...) throws -> Int {
            var total = 0
            for value in values {
                let (next, overflow) = total.addingReportingOverflow(value)
                guard !overflow else { throw Failure.budget }
                total = next
            }
            return total
        }
        let projections = try graphProjections()
        let assignments = c.numExpertsPerTok
        let elementBytes = weightDType.size
        let h = c.hiddenSize
        let headDim = c.headDim ?? (h / c.numAttentionHeads)
        let queryWidth = c.numAttentionHeads * headDim
        let kvWidth = c.numKeyValueHeads * headDim
        // [2H] concatenation, fused/norm/residual rows, Q, new K/V, codec
        // temporaries, and the fused attention's partials or, unfused, FP32
        // scores/probabilities over the whole history.
        let fused = fusedDraftAttentionBytes
        let rowTerms = [(1, 6 * h), (1, 2 * queryWidth), (4, kvWidth)]
        let attention =
            try fused.map { try sum(try bytes(rowTerms), $0) }
            ?? bytes(rowTerms + [(2 * c.numAttentionHeads, maximumHistoryTokens + 1)])
        // Router scores/selection over every expert, routed weighted sum and
        // latent projections, shared-expert activation, residual and norms.
        let mixer = try bytes([
            (8, c.nRoutedExperts), (assignments, 2 * latent), (2, latent),
            (2, c.moeSharedExpertIntermediateSize), (6, h),
        ])
        return GraphQuote(
            routedArrayBytes: try sum(
                try projections.fc1.graphWorkspaceBytes(
                    rows: assignments, outputBytesPerElement: elementBytes),
                try projections.fc2.graphWorkspaceBytes(
                    rows: assignments, outputBytesPerElement: elementBytes)),
            sharedArrayBytes: try sum(
                try projections.sharedUp.graphWorkspaceBytes(rows: 1),
                try projections.sharedDown.graphWorkspaceBytes(rows: 1)),
            attentionArrayBytes: attention, mixerArrayBytes: mixer,
            historyChunkArrayBytes: try bytes([(historyChunkRows, 8 * h + 4 * kvWidth)]),
            historyChunkRows: historyChunkRows)
    }

    /// Admission and calibration refusals precede the first KV append. The
    /// shared-expert projections report their host calibration validation, so
    /// no probe graph is built.
    private func validate(quote: GraphQuote, projections: GraphProjections) throws {
        let c = configuration
        let routed =
            try projections.fc1.graphWorkspaceBytes(
                rows: c.numExpertsPerTok, outputBytesPerElement: weightDType.size)
            + projections.fc2.graphWorkspaceBytes(
                rows: c.numExpertsPerTok, outputBytesPerElement: weightDType.size)
        let shared =
            try projections.sharedUp.graphWorkspaceBytes(rows: 1)
            + projections.sharedDown.graphWorkspaceBytes(rows: 1)
        guard routed <= quote.routedArrayBytes, shared <= quote.sharedArrayBytes else {
            throw Failure.budget
        }
        for projection in [projections.sharedUp, projections.sharedDown]
        where !projection.graphCalibrationValid {
            throw ModelOptCalibratedLinear.Failure.invalidMetadata
        }
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
    /// device-side routing validity of both expert projections.
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
        let up = try projections.fc1.graphForward(mixer.fc1LatentProj!(x), selected)
        let activated = maximum(up.value, MLXArray(0))
        let down = try projections.fc2.graphForward(activated * activated, selected)
        // Same layout and reduction as NemotronHMoE: [1, 1, topK, latent].
        var routed = down.value.reshaped(1, 1, assignments, down.value.dim(-1))
        routed = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(routed.dtype)
        routed = mixer.fc2LatentProj!(routed)
        let sharedUp = try projections.sharedUp.graphForward(x)
        let hiddenShared = maximum(sharedUp, MLXArray(0))
        let sharedDown = try projections.sharedDown.graphForward(hiddenShared * hiddenShared)
        let output = block.finalNorm(attended + (routed + sharedDown))
        return (output, logicalAnd(up.validIndices, down.validIndices))
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
