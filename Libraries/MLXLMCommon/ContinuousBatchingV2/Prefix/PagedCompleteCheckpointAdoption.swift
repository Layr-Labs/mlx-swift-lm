import MLX

// Engine-queue only: stage transfer and restoration must finish before any
// request reservation can detach or any target/assistant forward can run.
extension EngineLoopV2 {
    /// Before beginCommit, in the same engine-queue turn as the eventual
    /// publication/refusal. No prepared live-pool map crosses an await.
    func prepareNativePagedCompletePrefix(
        _ adoption: CBv2PrefixAdoption?,
        request: CBv2Request
    ) throws {
        guard let staged = adoption?.completeCheckpoint, staged.hasNativeTracking,
            staged.usesPagedBacking
        else { return }
        guard let capture = completeCheckpointCapture, capture.codec.isNativePagedComplete,
            let backend = backend as? PagedKVBackend, let tracking = nativeShutdownState,
            let generation = registeredStreamGeneration(for: request.id)
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let prepared = try staged.prepareNativePagedTargets(
            store: capture.store, request: request,
            engineID: tracking.engineID, expectedCodec: capture.codec, backend: backend,
            streamGeneration: generation, mtpIssuer: mtp?.admittedMTPBindingIssuer)
        nativePagedPrefixPreparedForTesting?(request.id, prepared)
    }

    func adoptPreparedNativeGLM5NextState(
        _ prepared: CBv2PreparedCompleteCheckpoint, codec: CBv2CompleteCheckpointCodec,
        work: CBv2NativeCompletePrefixWork, requestID: CBv2RequestID,
        maximumSequenceLength: Int
    ) throws -> [CBv2SequenceKV?] {
        guard codec.isNativePagedGLM5Next, codec === completeCheckpointCapture?.codec,
            let checkpoint = prepared.checkpoint, !checkpoint.layers.isEmpty,
            let pages = prepared.nativePagedPreparation,
            let restoration = prepared.glm5NextAssistantPreparation, restoration.settled,
            restoration.recurrent != nil, restoration.state != nil,
            pages.requestID == requestID, pages.maximumTokens == maximumSequenceLength,
            kvStates[requestID] == nil, recurrentStates[requestID] == nil,
            let mtp, mtp.canInstallHistoricalAssistant(for: requestID),
            codec.assistant.map({ $0 as AnyObject }) === mtp.drafter as AnyObject,
            let generation = registeredStreamGeneration(for: requestID),
            generation == pages.streamGeneration,
            let request = scheduler.record(for: requestID)?.request,
            checkpoint.position < request.promptTokens.count,
            checkpoint.mediaIdentity == request.hybridPrefixIdentity
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        // No waits or graph construction under this outer healthy commit.
        // Assistant publication remains private until all target maps install.
        try restoration.commit(work: work)
        let state = try pages.publish(admission: codec.admission, streamGeneration: generation)
        guard let (recurrent, assistant) = restoration.take() else {
            preconditionFailure("settled GLM checkpoint lost its prepared state")
        }
        recurrentStates[requestID] = recurrent
        mtp.restoreAssistantState(assistant, for: requestID)
        recurrentCheckpointGeometry[requestID] = .init(
            position: checkpoint.position, chunkSize: checkpoint.chunkSize)
        nativePagedPrefixAdoptedForTesting?(requestID, state, assistant)
        return state
    }

    /// Atomic metadata-only target+assistant move. Every native root and the
    /// actual restored assistant were evaluated/fenced before this commit.
    func adoptPreparedNativePagedHistoricalState(
        _ prepared: CBv2PreparedCompleteCheckpoint,
        codec: CBv2CompleteCheckpointCodec, requestID: CBv2RequestID,
        maximumSequenceLength: Int
    ) throws -> [CBv2SequenceKV?] {
        guard codec.isNativePagedHistorical, codec === completeCheckpointCapture?.codec,
            let checkpoint = prepared.checkpoint, checkpoint.layers.isEmpty,
            let pages = prepared.nativePagedPreparation,
            pages.maximumTokens == maximumSequenceLength,
            pages.requestID == requestID, kvStates[requestID] == nil,
            recurrentStates[requestID] == nil,
            let request = scheduler.record(for: requestID)?.request,
            checkpoint.position < request.promptTokens.count,
            let generation = registeredStreamGeneration(for: requestID),
            generation == pages.streamGeneration
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let restored: (any CBv2MTPRequestState)?
        if checkpoint.assistant != nil {
            guard let mtp, mtp.canInstallHistoricalAssistant(for: requestID),
                let assistant = codec.assistant,
                assistant as AnyObject === mtp.drafter as AnyObject,
                let restoration = prepared.historicalAssistantRestoration,
                restoration.settled, let actual = restoration.state
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            restored = actual
        } else {
            guard codec.assistant == nil, !(mtp?.tracksPersistentHistory ?? false),
                prepared.historicalAssistantRestoration == nil
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            restored = nil
        }
        let state = try pages.publish(admission: codec.admission, streamGeneration: generation)
        // All fallible checks precede pool/row publication. No carry or missing
        // head state is synthesized; the first real suffix observation resumes.
        if let restored {
            _ = prepared.historicalAssistantRestoration?.takeSettledState()
            mtp!.restoreAssistantState(restored, for: requestID)
        }
        recurrentCheckpointGeometry[requestID] = .init(
            position: checkpoint.position,
            chunkSize: checkpoint.chunkSize)
        nativePagedPrefixAdoptedForTesting?(requestID, state, restored)
        return state
    }

    func adoptPagedCompleteCheckpoint(
        _ staged: CBv2StagedCompleteCheckpoint, requestID: CBv2RequestID
    ) throws -> [CBv2SequenceKV?] {
        guard let paged = backend as? PagedKVBackend,
            staged.codec === completeCheckpointCapture?.codec,
            recurrentStates[requestID] == nil
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        if staged.codec.glm5NextLayout != nil {
            return try staged.codec.adoptGLM5Next(staged, backend: paged, requestID: requestID) {
                checkpoint in
                do {
                    try adoptRecurrentCheckpoint(checkpoint, requestID: requestID)
                } catch {
                    mtp?.invalidateCarry(requestID)
                    if let recurrent = recurrentStates.removeValue(forKey: requestID) {
                        try recurrent.release()
                    }
                    recurrentCheckpointGeometry.removeValue(forKey: requestID)
                    throw error
                }
            }
        }
        return try staged.consumePreparedState { prepared in
            guard let frame = prepared.pagedFrame else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            prepared.pagedFrame = nil
            let adopted = try paged.pool.importCheckpoint(
                frame, admission: staged.codec.admission, requestID: requestID,
                layerKinds: layerKinds, maximumTokens: staged.maximumSequenceLength)
            // importCheckpoint has atomically replaced the stage destination
            // charge with physical backing + the full N request promise. No
            // generic capacity.reserve may precede or follow this handoff.
            return try adopted.moveToActiveRequest { auxiliary in
                do {
                    if staged.codec.historicalLayout != nil {
                        guard auxiliary.isEmpty, !(mtp?.tracksPersistentHistory ?? false) else {
                            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                        }
                        // Gemma's stateless drafter seeds from the next target
                        // round. No persistent assistant history is fabricated.
                        recurrentCheckpointGeometry[requestID] = .init(
                            position: staged.manifest.position, chunkSize: staged.manifest.chunkSize
                        )
                        return
                    }
                    let checkpoint = try staged.codec.recurrentCheckpoint(
                        manifest: staged.manifest, auxiliary: auxiliary)
                    try staged.codec.restoreQwen4(checkpoint, rows: adopted.rows)
                    try adoptRecurrentCheckpoint(checkpoint, requestID: requestID)
                } catch {
                    // Restoration builds only candidate state; none has run a
                    // model forward. Remove every candidate alias before the
                    // move owner releases pages and its generation-bound charge.
                    mtp?.invalidateCarry(requestID)
                    if let recurrent = recurrentStates.removeValue(forKey: requestID) {
                        try recurrent.release()
                    }
                    recurrentCheckpointGeometry.removeValue(forKey: requestID)
                    throw error
                }
            }
        }
    }
}
