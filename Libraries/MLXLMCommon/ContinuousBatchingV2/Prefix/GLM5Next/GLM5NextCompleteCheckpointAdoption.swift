import MLX

/// Implemented by the actual backend, never by a provider-supplied callback.
/// A refusal must not publish any part of the candidate row cohort.
protocol GLM5NextCompleteCheckpointRowRegistering: AnyObject {
    func registerGLM5NextCheckpointRows(_ state: [CBv2SequenceKV?]) throws
}

extension CBv2CompleteCheckpointCodec {
    /// Engine-queue component for an untracked GLM stage. The enclosing engine
    /// must exclude cancellation/publication while this move runs. Native
    /// profile stages use their counted preparation/commit path instead.
    ///
    /// Registration must validate the complete cohort before mutating its map;
    /// restoration must drop candidate recurrent/assistant aliases on failure.
    func adoptGLM5Next(
        _ staged: CBv2StagedCompleteCheckpoint, backend: PagedKVBackend,
        requestID: CBv2RequestID,
        restore: (CBv2RecurrentCheckpoint) throws -> Void
    ) throws -> [CBv2SequenceKV?] {
        guard let layout = glm5NextLayout, !staged.hasNativeTracking,
            staged.codec === self, staged.usesPagedBacking,
            backend.nativeModelBinding == nil, backend.pool.nativeModelBinding == nil,
            backend === layout.backend, backend.glm5NextModelBinding === layout.binding,
            backend.pool.memoryAdmission === admission,
            backend.pool.activeAttentionWork == nil
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        return try staged.consumePreparedState { prepared in
            guard let frame = prepared.pagedFrame else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            prepared.pagedFrame = nil
            let adopted = try backend.pool.importCheckpoint(
                frame, admission: admission, requestID: requestID, layerKinds: layerKinds,
                maximumTokens: staged.maximumSequenceLength)
            return try adopted.moveToActiveRequest { auxiliary in
                let checkpoint = try recurrentCheckpoint(
                    manifest: staged.manifest,
                    auxiliary: auxiliary)
                let rows: [CBv2SequenceKV?] = adopted.rows.map { $0 }
                do {
                    try backend.registerGLM5NextCheckpointRows(rows)
                    try restore(checkpoint)
                } catch {
                    // Ordinary backend retirement clears the same live row map
                    // that cold GLM requests use. The adoption owner then drops
                    // auxiliary aliases before returning its request charge.
                    backend.release(rows)
                    throw error
                }
            }
        }
    }
}
