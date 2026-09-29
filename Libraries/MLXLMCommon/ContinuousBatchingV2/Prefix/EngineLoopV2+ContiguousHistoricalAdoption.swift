import MLX

extension EngineLoopV2 {
    /// Metadata-only transaction under the existing native enqueue/deadline
    /// commit. Mutable assistant restoration/evaluation MUST have completed in
    /// staged lookup; there is no model call, eval, or stream fence here.
    func adoptPreparedContiguousHistoricalState(
        _ prepared: CBv2PreparedCompleteCheckpoint,
        codec: CBv2CompleteCheckpointCodec, requestID: CBv2RequestID,
        maximumSequenceLength: Int
    ) throws -> [CBv2SequenceKV?] {
        guard let checkpoint = prepared.checkpoint,
            let contiguous = backend as? any CBv2ContiguousHistoricalBackend,
            codec === completeCheckpointCapture?.codec, codec.contiguousLayout != nil,
            checkpoint.layers.isEmpty, recurrentStates[requestID] == nil,
            kvStates[requestID] == nil,
            let request = scheduler.record(for: requestID)?.request,
            checkpoint.position < request.promptTokens.count
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let restored: (any CBv2MTPRequestState)?
        if checkpoint.assistant != nil {
            guard let mtp, mtp.canInstallHistoricalAssistant(for: requestID),
                let expected = codec.assistant,
                expected as AnyObject === mtp.drafter as AnyObject,
                let restoration = prepared.historicalAssistantRestoration,
                restoration.settled, let state = restoration.state
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            restored = state
        } else {
            guard codec.assistant == nil, !(mtp?.tracksPersistentHistory ?? false),
                prepared.historicalAssistantRestoration == nil
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            restored = nil
        }
        // Backend validates the fresh-row ledger and transfers its actual
        // Admission lease before publishing. It rolls back on registration
        // failure. All fallible assistant checks preceded this mutation.
        try contiguous.adoptContiguousHistoricalState(
            prepared.state, codec: codec,
            layerKinds: layerKinds, position: checkpoint.position,
            requestID: requestID, maximumSequenceLength: maximumSequenceLength)
        if let restored {
            _ = prepared.historicalAssistantRestoration?.takeSettledState()
            mtp!.restoreAssistantState(restored, for: requestID)
        }
        recurrentCheckpointGeometry[requestID] = .init(
            position: checkpoint.position, chunkSize: checkpoint.chunkSize)
        return prepared.state
    }
}
