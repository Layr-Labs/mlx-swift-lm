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

extension NemotronSuperMTPAssistant: CBv2MTPBoundedAllocationProviding {
    /// Fixed per-request charge the engine admits before any request state
    /// exists and holds until retirement. Resident: FP8 codes and decoded
    /// mirror for the full sequence bound, the queued trusted backlog (target
    /// hidden slices can retain their parents, charged at FP32 for every row)
    /// and token/frontier carries. Working: each chained draft step (graph,
    /// FP8 output head, KV append), one prompt-history chunk with its growth
    /// copy, and the explicit backend scratch allowance.
    public func boundedRequestAllocation(limits: CBv2MTPAllocationLimits)
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
