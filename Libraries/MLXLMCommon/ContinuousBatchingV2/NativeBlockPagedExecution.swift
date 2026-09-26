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
                return value
            }
        } catch {
            StreamOrDevice.default.stream.synchronize()
            boundary.discardFailedGraphAfterSynchronization()
            onFailure()
            pool.writeValidation.clearAfterRetirement()
            throw error
        }
    }
}
