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
