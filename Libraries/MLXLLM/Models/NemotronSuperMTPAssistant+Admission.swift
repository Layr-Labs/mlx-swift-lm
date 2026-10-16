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
    /// The part of a request's charge that does not grow with its sequence,
    /// admitted before any request state exists and held until retirement:
    /// the frontier carries; each chained draft step's graph (scores over its
    /// first history rows), FP8 output head and one-row codec; one
    /// prompt-history chunk's graph and codec; the rectangular verify window;
    /// the backend scratch allowance. `boundedRequestTokenAllocation` charges
    /// the rest per reserved token, so the two together equal
    /// `windowRequestAllocation` evaluated at the request's own bound.
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
        var working = [
            CBv2MTPFixedBufferSpec(logicalBytes: step, allocationCount: limits.maximumDraftTokens),
            CBv2MTPFixedBufferSpec(logicalBytes: history),
        ]
        if workspace.backendScratchBytes > 0 {
            working.append(.init(logicalBytes: workspace.backendScratchBytes))
        }
        if settings.verification == .rectangularExact {
            guard let window = verifyWindowBytes(draftTokens: limits.maximumDraftTokens) else {
                return nil
            }
            working.append(.init(logicalBytes: window))
        }
        return CBv2MTPBoundedAllocationSpec(
            resident: [.init(logicalBytes: 4 * c.hiddenSize * 4)], working: working, hostBytes: 4096
        )
    }

    /// Request-owned state that grows with the request's sequence, per row of
    /// `reserved tokens + depth + 1` (the history capacity of a request bound):
    /// FP8 codes and decoded mirror of K and V in whole history blocks, the
    /// FP32 trusted backlog and two token carries; each draft step's FP32
    /// scores over the history (one row more) and its growth copy of the
    /// history; one prompt-history chunk's growth copy.
    public func boundedRequestTokenAllocation(limits: CBv2MTPAllocationLimits)
        -> [CBv2AuxiliaryAllocationSpec]?
    {
        guard limits.maximumPrefillTokens > 0,
            (1 ... settings.maximumDraftTokens).contains(limits.maximumDraftTokens)
        else { return nil }
        let c = target.configuration
        let width = c.numKeyValueHeads * (c.headDim ?? (c.hiddenSize / c.numAttentionHeads))
        let element = head.weightDType.size
        let block = NemotronSuperFP8KVHistory.blockTokens
        let padding = settings.maximumDraftTokens + 1
        let historyRow = 2 * width * (1 + element)
        let steps = limits.maximumDraftTokens
        return [
            .init(
                bytesPerToken: width, allocationCount: 2, tokenGranularity: block,
                tokenPadding: padding),
            .init(
                bytesPerToken: width * element, allocationCount: 2, tokenGranularity: block,
                tokenPadding: padding),
            .init(bytesPerToken: c.hiddenSize * 4, tokenPadding: padding),
            .init(bytesPerToken: 2 * 4, tokenPadding: padding),
            .init(
                bytesPerToken: 2 * head.configuration.numAttentionHeads * 32,
                allocationCount: steps, tokenPadding: padding + 1),
            .init(
                bytesPerToken: historyRow, allocationCount: steps, tokenGranularity: block,
                tokenPadding: padding),
            .init(bytesPerToken: historyRow, tokenGranularity: block, tokenPadding: padding),
        ]
    }

    /// The verify window's target activations for `draftTokens + 1` positions:
    /// the heaviest layer's ModelOpt graph workspace for that chunk (the eager
    /// 64 MiB projection bound per row otherwise), the output head over every
    /// row, FP32 logits plus top-two partials, and hidden/residual rows.
    /// Captured conv/SSM stacks are charged by the engine as recurrent
    /// generations.
    func verifyWindowBytes(draftTokens: Int) -> Int? {
        let c = target.configuration
        let rows = draftTokens + 1
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
    /// full sequence bound, the queued trusted backlog (target hidden slices
    /// can retain their parents, charged at FP32 for every row) and
    /// token/frontier carries; working each chained draft step (graph, FP8
    /// output head, KV append), one prompt-history chunk with its growth copy,
    /// and the explicit backend scratch allowance. The sequence-scaled split
    /// above is checked against it.
    func windowRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        guard limits.maximumPrefillTokens > 0,
            (1 ... settings.maximumDraftTokens).contains(limits.maximumDraftTokens)
        else { return nil }
        let c = target.configuration
        let rows = historyCapacity
        let blocks =
            (rows + NemotronSuperFP8KVHistory.blockTokens - 1)
            / NemotronSuperFP8KVHistory.blockTokens
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
        guard
            let codes = product(
                blocks, NemotronSuperFP8KVHistory.blockTokens, c.numKeyValueHeads, headDim),
            let decoded = product(codes, head.weightDType.size),
            let backlog = product(rows, c.hiddenSize, 4),
            let carries = product(2 * rows + 4 * c.hiddenSize, 4),
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
            guard let window = verifyWindowBytes(draftTokens: limits.maximumDraftTokens) else {
                return nil
            }
            working.append(.init(logicalBytes: window))
        }
        return CBv2MTPBoundedAllocationSpec(
            resident: [
                .init(logicalBytes: codes, allocationCount: 2),
                .init(logicalBytes: decoded, allocationCount: 2),
                .init(logicalBytes: backlog),
                .init(logicalBytes: carries),
            ],
            working: working, hostBytes: 4096)
    }
}
