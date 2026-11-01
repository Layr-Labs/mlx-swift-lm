import MLX

extension PagedKVBackend {
    /// Same failed-write boundary used by the AR scheduler, scoped to a native
    /// block step. Callers retire the failed request's rows before reusing them;
    /// resetting a fence is not a rollback of already-executed ring writes.
    package func performNativeBlockStep<Result>(_ body: () throws -> Result, onFailure: () -> Void)
        throws -> Result
    {
        let boundary = CBv2PagedWriteBoundary(pool: pool)
        do {
            return try MLX.withError { errors in
                try pool.writeValidation.check()
                let value = try body()
                try pool.writeValidation.check()
                try errors.check()
                if pool.config.quantization != nil {
                    StreamOrDevice.default.stream.synchronize()
                    try errors.check()
                    try pool.finishQuantizedStorageStep()
                }
                return value
            }
        } catch {
            StreamOrDevice.default.stream.synchronize()
            boundary.discardFailedGraphAfterSynchronization()
            onFailure()
            pool.discardQuantizedStorageStepAfterSynchronization()
            pool.writeValidation.clearAfterRetirement()
            throw error
        }
    }
}
