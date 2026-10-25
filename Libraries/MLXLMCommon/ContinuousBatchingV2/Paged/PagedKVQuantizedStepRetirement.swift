import MLX

extension PagedKVPool {
    /// Called only after the scheduler's confirmed step (including MTP
    /// rollback/commit). Quantized steps do not chain a successor. Keep every
    /// scratch holder and charge until its explicit output has completed.
    func finishQuantizedStorageStep() throws {
        let leases = pendingQuantizedScratch
        try withError { fault in
            let targets = leases.flatMap(\.evaluationTargets)
            if !targets.isEmpty { eval(targets) }
            try fault.check()
            StreamOrDevice.default.stream.synchronize()
            try fault.check()
        }
        completeQuantizedAttentionStep()
        try finishQuantizedRecentStep()
        // Neither an evaluation fault nor a refused compaction refunds a
        // generation that the device may still hold.
        _ = takePendingQuantizedScratch()
        for lease in leases { lease.finishAfterSynchronization() }
    }

    /// Failure-only cleanup AFTER the actual issuing streams have drained
    /// and affected rows/graphs have retired. Never evaluates a failed graph
    /// or tries to compact its native band while unwinding.
    package func discardQuantizedStorageStepAfterSynchronization() {
        completeQuantizedAttentionStep()
        for lease in takePendingQuantizedScratch() { lease.finishAfterSynchronization() }
    }
}

extension PagedKVBackend {
    /// For an owner that drives its own steps over these caches instead of
    /// `EngineV2` (a cluster stage host): after building a step's graph and
    /// before evaluating it, as the engine checks, throws a write the pool
    /// refused while building it (its attention returned a placeholder,
    /// never a result).
    public func checkOwnedStepWrites() throws {
        try pool.writeValidation.check()
    }

    /// For such an owner, once per confirmed step, after its outputs and every
    /// cache root (`innerState()`) were evaluated; an MTP window counts once,
    /// after its rollback and commit. Checks the write latch again, then
    /// completes packed storage's step as the engine does: scratch retired,
    /// borrowed views dropped and each row's native band compacted to its
    /// recent tokens.
    public func finishOwnedStep() throws {
        try checkOwnedStepWrites()
        guard usesQuantizedStorage else { return }
        try pool.finishQuantizedStorageStep()
    }

    /// Failure only, after the failed step's rows were released and its
    /// issuing streams drained: drops the step's packed scratch and clears the
    /// write latch, so the next request starts on a clean pool.
    public func discardFailedOwnedStepAfterSynchronization() {
        pool.discardQuantizedStorageStepAfterSynchronization()
        pool.writeValidation.clearAfterRetirement()
    }
}
