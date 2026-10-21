import MLX
import MLXLMCommon

extension NemotronSuperMTPAssistant {
    /// Every simultaneously live array allowance of one speculative round.
    /// Header/configuration arithmetic only; not a measured peak.
    public struct Workspace: Sendable, Equatable {
        public let graph: NemotronSuperMTPModule.GraphQuote
        /// Shared FP8 output head (`ModelOptCalibratedLinear.graphForward`), one row.
        public let outputHeadArrayBytes: Int
        public let draftAppendBytes: Int
        public let historyAppendBytes: Int
        public let backendScratchBytes: Int
        public let draftDepth: Int
        /// One draft forward, its output head and its one-row KV append.
        public let draftStepBytes: Int
        /// Every chained draft step plus one prompt-history chunk and scratch.
        public let totalBytes: Int
    }

    static func workspace(
        head: NemotronSuperMTPModule, outputHead: OutputHead, settings: Settings,
        history: NemotronSuperFP8KVHistory
    ) throws -> Workspace {
        let c = head.configuration
        let chunk = min(settings.historyChunkRows, history.maximumAppendRows)
        let graph = try head.graphQuote(
            maximumHistoryTokens: history.capacityTokens, historyChunkRows: chunk)
        let outputBytes: Int
        switch outputHead {
        case .fp8(let projection):
            // The shared unit-scale plane is resident with the target's head.
            outputBytes = try projection.graphWorkspaceBytes(rows: 1)
        case .dense:
            outputBytes = try sum([c.vocabSize * 32, 1024])
        }
        let draftAppend = try history.appendAllowanceBytes(rows: 1)
        let historyAppend = try history.appendAllowanceBytes(rows: chunk)
        let step = try sum([graph.draftArrayBytes, outputBytes, draftAppend])
        let (chain, overflow) = step.multipliedReportingOverflow(by: settings.maximumDraftTokens)
        guard !overflow else { throw Failure.workspace }
        return Workspace(
            graph: graph, outputHeadArrayBytes: outputBytes, draftAppendBytes: draftAppend,
            historyAppendBytes: historyAppend, backendScratchBytes: settings.backendScratchBytes,
            draftDepth: settings.maximumDraftTokens, draftStepBytes: step,
            totalBytes: try sum([
                chain, graph.historyChunkArrayBytes, historyAppend, settings.backendScratchBytes,
            ]))
    }

    private static func sum(_ values: [Int]) throws -> Int {
        try values.reduce(0) { total, value in
            let (next, overflow) = total.addingReportingOverflow(value)
            guard value >= 0, !overflow else { throw Failure.workspace }
            return next
        }
    }
}

extension NemotronSuperMTPAssistant: CBv2MTPSequenceScaledAllocationProviding {
    /// A request's charge that does not grow with its sequence is the frontier
    /// carries and the queued partial history block: fewer than
    /// `historyChunkRows` pairs in the head dtype with their tokens, once as
    /// queued and once as the copy an observation replaces it with (whole
    /// blocks go to the FP8 history as they are observed, and after the first
    /// draft only accepted rows queue). The round's work exists once per
    /// owner, whatever the number of requests (`ownerWorking`): every draft
    /// step the depth policy allows in one round (graph with the attention's
    /// partials or scores over its first history rows, FP8 output head,
    /// one-row codec), one prompt-history chunk's graph and codec (one
    /// request primes per round: one prompt chunk runs per step), the captured
    /// verify window at the round's largest position count, the backend
    /// scratch allowance, and the history copies MLX may hold at once over
    /// every request (`CBv2MTPAllocationLimits.historyCopies`), each the codes
    /// and mirror of the configured sequence bound: a history write copies its
    /// whole buffer whenever an earlier step's attention or the engine's
    /// retained roots still hold it. `boundedRequestTokenAllocation` charges
    /// the history itself per reserved token. Per request, the three cover
    /// `windowRequestAllocation` at the request's own bound.
    public func boundedRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        guard limits.maximumPrefillTokens > 0,
            (1 ... settings.maximumDraftTokens).contains(limits.maximumDraftTokens),
            workspace.draftStepBytes > 0,
            let graph = try? head.graphQuote(
                maximumHistoryTokens: 1, historyChunkRows: workspace.graph.historyChunkRows)
        else { return nil }
        let c = target.configuration
        let width = c.numKeyValueHeads * (c.headDim ?? (c.hiddenSize / c.numAttentionHeads))
        let (step, stepOverflow) = graph.draftArrayBytes.addingReportingOverflow(
            workspace.outputHeadArrayBytes + width * 2 * 32)
        let (history, historyOverflow) = graph.historyChunkArrayBytes.addingReportingOverflow(
            graph.historyChunkRows * width * 2 * 32)
        guard !stepOverflow, !historyOverflow else { return nil }
        // The owner bounds cover one request's deepest round. A depth table
        // can keep every reachable round below `maximumDraftTokens`.
        guard limits.maximumRoundDraftSteps >= limits.maximumRoundDepth,
            limits.maximumWindowPositions > limits.maximumRoundDepth,
            limits.maximumHistoryCopies > limits.maximumRoundDepth,
            let copy = historyCopyBytes
        else { return nil }
        var owner = [
            CBv2MTPFixedBufferSpec(
                logicalBytes: step, allocationCount: limits.maximumRoundDraftSteps),
            CBv2MTPFixedBufferSpec(logicalBytes: history),
            CBv2MTPFixedBufferSpec(
                logicalBytes: copy, allocationCount: limits.maximumHistoryCopies),
        ]
        if workspace.backendScratchBytes > 0 {
            owner.append(.init(logicalBytes: workspace.backendScratchBytes))
        }
        if settings.verification == .rectangularExact {
            guard let window = verifyWindowBytes(positions: limits.maximumWindowPositions) else {
                return nil
            }
            owner.append(.init(logicalBytes: window))
        }
        let (backlog, backlogOverflow) = graph.historyChunkRows.multipliedReportingOverflow(
            by: c.hiddenSize * head.weightDType.size + 4)
        guard !backlogOverflow else { return nil }
        return CBv2MTPBoundedAllocationSpec(
            resident: [
                .init(logicalBytes: 4 * c.hiddenSize * 4),
                .init(logicalBytes: backlog, allocationCount: 2),
            ], working: [], hostBytes: 4096, ownerWorking: owner)
    }

    /// One whole-history copy at the configured sequence bound: K and V codes
    /// and decoded mirror of `historyCapacity` rows.
    var historyCopyBytes: Int? {
        let c = target.configuration
        let width = c.numKeyValueHeads * (c.headDim ?? (c.hiddenSize / c.numAttentionHeads))
        let (bytes, overflow) = (2 * width * (1 + head.weightDType.size))
            .multipliedReportingOverflow(by: historyCapacity)
        return overflow ? nil : bytes
    }

    /// Request-owned state that grows with the request's sequence, per row of
    /// `reserved tokens + depth + 1` (the history capacity of a request bound,
    /// allocated once at the first write): FP8 codes and decoded mirror of K
    /// and V and, only when the draft attention would not run fused, each
    /// draft step's FP32 scores over the history (one row more). Copies of the
    /// history are charged once per owner (`boundedRequestAllocation`).
    public func boundedRequestTokenAllocation(limits: CBv2MTPAllocationLimits)
        -> [CBv2AuxiliaryAllocationSpec]?
    {
        guard limits.maximumPrefillTokens > 0,
            (1 ... settings.maximumDraftTokens).contains(limits.maximumDraftTokens)
        else { return nil }
        let c = target.configuration
        let width = c.numKeyValueHeads * (c.headDim ?? (c.hiddenSize / c.numAttentionHeads))
        let element = head.weightDType.size
        let padding = settings.maximumDraftTokens + 1
        let steps = limits.maximumDraftTokens
        var specs: [CBv2AuxiliaryAllocationSpec] = [
            .init(bytesPerToken: width, allocationCount: 2, tokenPadding: padding),
            .init(bytesPerToken: width * element, allocationCount: 2, tokenPadding: padding),
        ]
        if head.fusedDraftAttentionBytes == nil {
            specs.append(
                .init(
                    bytesPerToken: 2 * head.configuration.numAttentionHeads * 32,
                    allocationCount: steps, tokenPadding: padding + 1))
        }
        return specs
    }

    /// The verify window's target activations for `positions` positions:
    /// the heaviest layer's ModelOpt graph workspace for that chunk (the eager
    /// 64 MiB projection bound per row otherwise), the output head over every
    /// row, FP32 logits plus top-two partials, and hidden/residual rows.
    /// Captured conv/SSM stacks are charged by the engine as recurrent
    /// generations.
    func verifyWindowBytes(positions rows: Int) -> Int? {
        let c = target.configuration
        let head =
            (target.lmHead as? ModelOptCalibratedLinear)
            .flatMap { try? $0.graphWorkspaceBytes(rows: rows) } ?? 0
        let (layersPerRow, eagerOverflow) = rows.multipliedReportingOverflow(by: 64 << 20)
        let (outputs, outputOverflow) = rows.multipliedReportingOverflow(
            by: c.vocabSize * 8 + c.hiddenSize * 32)
        guard !eagerOverflow, !outputOverflow else { return nil }
        let layers = target.modelOptGraphPrefillWorkspaceBytes(tokens: rows) ?? layersPerRow
        let (partial, first) = layers.addingReportingOverflow(head)
        let (window, second) = partial.addingReportingOverflow(outputs)
        return first || second ? nil : window
    }

    /// The whole charge for a request of the configured maximum sequence, as
    /// one fixed declaration: resident FP8 codes and decoded mirror for the
    /// full sequence bound with its history copies (`historyCopies`), the queued
    /// partial history block twice and the frontier carries; working each
    /// chained draft step (graph, FP8 output head, KV append), one
    /// prompt-history chunk, and the explicit backend scratch allowance. The
    /// sequence-scaled split above is checked against it.
    func windowRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        guard limits.maximumPrefillTokens > 0,
            (1 ... settings.maximumDraftTokens).contains(limits.maximumDraftTokens)
        else { return nil }
        let c = target.configuration
        let rows = historyCapacity
        func product(_ values: Int...) -> Int? {
            var result = 1
            for value in values {
                let (next, overflow) = result.multipliedReportingOverflow(by: value)
                guard value > 0, !overflow else { return nil }
                result = next
            }
            return result
        }
        let headDim = c.headDim ?? (c.hiddenSize / c.numAttentionHeads)
        let element = head.weightDType.size
        guard
            let codes = product(rows, c.numKeyValueHeads, headDim),
            let decoded = product(codes, element),
            let copy = product(2, codes, 1 + element),
            let backlog = product(
                workspace.graph.historyChunkRows, c.hiddenSize * element + 4),
            workspace.draftStepBytes > 0
        else { return nil }
        let (historyWork, overflow) = workspace.graph.historyChunkArrayBytes
            .addingReportingOverflow(workspace.historyAppendBytes)
        guard !overflow else { return nil }
        var working = [
            CBv2MTPFixedBufferSpec(
                logicalBytes: workspace.draftStepBytes,
                allocationCount: limits.maximumDraftTokens),
            CBv2MTPFixedBufferSpec(logicalBytes: historyWork),
        ]
        if workspace.backendScratchBytes > 0 {
            working.append(.init(logicalBytes: workspace.backendScratchBytes))
        }
        if settings.verification == .rectangularExact {
            guard let window = verifyWindowBytes(positions: limits.maximumDraftTokens + 1) else {
                return nil
            }
            working.append(.init(logicalBytes: window))
        }
        return CBv2MTPBoundedAllocationSpec(
            resident: [
                .init(logicalBytes: codes, allocationCount: 2),
                .init(logicalBytes: decoded, allocationCount: 2),
                .init(logicalBytes: copy, allocationCount: limits.maximumHistoryCopies),
                .init(logicalBytes: backlog, allocationCount: 2),
                .init(logicalBytes: 4 * c.hiddenSize * 4),
            ],
            working: working, hostBytes: 4096)
    }
}
