import MLX

final class CBv2CompleteCheckpointRetiredState: @unchecked Sendable {
    private var state: [CBv2SequenceKV?]
    init(state: [CBv2SequenceKV?]) { self.state = state }
    func release(backend: CBv2KVBackend) {
        backend.release(state)
        state.removeAll()
    }
}

extension EngineLoopV2 {
    func adoptCompleteCheckpoint(_ staged: CBv2StagedCompleteCheckpoint, requestID: CBv2RequestID)
        throws -> [CBv2SequenceKV?]
    {
        if staged.hasNativeTracking {
            guard let capture = completeCheckpointCapture, let tracking = nativeShutdownState,
                  let request = scheduler.record(for: requestID)?.request else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            return try staged.consumeNativePreparedState(store: capture.store, request: request,
                engineID: tracking.engineID, expectedCodec: capture.codec) { prepared, codec, work in
                // Native graphs/fences were completed during staged lookup.
                // This helper performs the metadata-only atomic move
                // under the existing outer native commit.
                if codec.isNativePagedHistorical {
                    return try adoptPreparedNativePagedHistoricalState(prepared, codec: codec,
                        requestID: requestID, maximumSequenceLength: staged.maximumSequenceLength)
                }
                return try adoptPreparedContiguousHistoricalState(prepared, codec: codec,
                    requestID: requestID, maximumSequenceLength: staged.maximumSequenceLength)
            }
        }
        guard nativeShutdownState == nil else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        if staged.usesPagedBacking { return try adoptPagedCompleteCheckpoint(staged, requestID: requestID) }
        return try staged.consumePreparedState { prepared in
            guard let checkpoint = prepared.checkpoint else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            if staged.codec.contiguousLayout != nil {
                guard let contiguous = backend as? any CBv2ContiguousHistoricalBackend,
                      staged.codec === completeCheckpointCapture?.codec, checkpoint.layers.isEmpty,
                      checkpoint.assistant == nil, !(mtp?.tracksPersistentHistory ?? false),
                      recurrentStates[requestID] == nil else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                try contiguous.adoptContiguousHistoricalState(prepared.state, codec: staged.codec,
                    layerKinds: layerKinds, position: staged.manifest.position,
                    requestID: requestID, maximumSequenceLength: staged.maximumSequenceLength)
                recurrentCheckpointGeometry[requestID] = .init(position: checkpoint.position, chunkSize: checkpoint.chunkSize)
                return prepared.state
            }
            guard let contiguous = backend as? CBv2ContiguousKVBackend else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            try contiguous.adoptPreparedCheckpoint(prepared.state)
            do {
                try adoptRecurrentCheckpoint(checkpoint, requestID: requestID)
            } catch {
                contiguous.release(prepared.state)
                throw error
            }
            return prepared.state
        }
    }
}
