import MLX

extension CBv2InFlightStep {
    /// A successor can already be running when this step retires. Release
    /// only the workspaces captured by this exact step after its readbacks.
    func finishQuantizedScratchAfterSynchronization() {
        for lease in quantizedScratch { lease.finishAfterSynchronization() }
        quantizedScratch.removeAll()
    }
}

extension EngineLoopV2 {
    /// Provision the complete submitted step's arena shape before constructing
    /// any layer graph. Scheduler frontiers already include this assignment;
    /// lookahead also covers serial speculative verification columns.
    func beginQuantizedScratch(for plan: CBv2StepPlan) throws {
        guard let pool = (backend as? PagedKVBackend)?.pool,
              pool.config.quantization != nil else { return }
        var maximumFrontier = 1
        for assignment in plan.assignments {
            guard let record = scheduler.record(for: assignment.id) else { continue }
            let (frontier, overflow) = record.numComputedTokens.addingReportingOverflow(
                CBv2PagedSpeculation.maxSpeculativeSpan)
            guard !overflow, frontier > 0 else {
                throw CBv2KVError.backendIneligible(reason: "quantized step frontier overflow")
            }
            maximumFrontier = max(maximumFrontier, frontier)
        }
        try pool.beginQuantizedScratch(
            maximumQueries: max(8, scheduler.config.maxConcurrentRequests),
            maximumAttendLength: maximumFrontier)
    }

    /// Offline scoring does not construct a scheduler step. Each completed
    /// forward is its own bounded scope, with the same retirement rules.
    func beginQuantizedScoringScratch(state: [CBv2SequenceKV?], tokens: MLXArray) throws {
        guard let pool = (backend as? PagedKVBackend)?.pool,
              pool.config.quantization != nil else { return }
        let (end, endOverflow) = Self.positionOffset(state).addingReportingOverflow(tokens.dim(1))
        let (frontier, frontierOverflow) = end.addingReportingOverflow(CBv2PagedSpeculation.maxSpeculativeSpan)
        guard !endOverflow, !frontierOverflow, frontier > 0 else {
            throw CBv2KVError.backendIneligible(reason: "quantized scoring frontier overflow")
        }
        try pool.beginQuantizedScratch(maximumQueries: max(8, tokens.dim(0)),
                                       maximumAttendLength: frontier)
    }

    func attachQuantizedScratch(to step: CBv2InFlightStep?) {
        guard let pool = (backend as? PagedKVBackend)?.pool else { return }
        let leases = pool.takePendingQuantizedScratch()
        guard let step else {
            // An empty plan normally constructs no scratch. Do not return a
            // reservation while an exceptional submitted empty plan still runs.
            if !leases.isEmpty {
                Stream.gpu.synchronize()
                Stream.cpu.synchronize()
                for lease in leases { lease.finishAfterSynchronization() }
            }
            return
        }
        step.quantizedScratch.append(contentsOf: leases)
    }

    /// Failed graph builders have no in-flight step to own their leases.
    /// Call only after draining submitted work and discarding failed fences.
    func discardPendingQuantizedScratchAfterSynchronization() {
        guard let pool = (backend as? PagedKVBackend)?.pool else { return }
        for lease in pool.takePendingQuantizedScratch() {
            lease.finishAfterSynchronization()
        }
    }
}
