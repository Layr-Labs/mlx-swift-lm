import Foundation

extension EngineV2 {
    /// Native historical capture still proves the donor's complete-chunk
    /// execution shape. Keep its scheduler bound until partition-independent
    /// target/head continuation is independently qualified.
    private func validateContiguousCheckpointChunk(
        _ manifest: CBv2CompleteCheckpointManifest, codec: CBv2CompleteCheckpointCodec
    ) throws {
        guard codec.contiguousLayout != nil || codec.isNativePagedHistorical else { return }
        guard manifest.chunkSize >= schedulerConfig.prefillChunkSize,
            manifest.chunkSize
                <= max(
                    schedulerConfig.prefillChunkSize,
                    schedulerConfig.soloPrefillStripeTokens ?? 0),
            CBv2AttentionV1.queryBlockSize <= 0
                || manifest.chunkSize % CBv2AttentionV1.queryBlockSize == 0
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
    }

    /// Charge the first manifest decrypt before the provider has a validated
    /// import plan. This shares the slot's admission ceiling, including when
    /// the caller has no provider-wide budget. No file or model work occurs.
    public func reserveCompleteCheckpointReadScratch() throws -> CBv2CompleteCheckpointIOLease {
        guard let completeCheckpointCodec,
            !completeCheckpointCodec.unsupportedAsymmetricGeometry
                || completeCheckpointCodec.contiguousLayout != nil
                || completeCheckpointCodec.isNativePagedHistorical
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        if completeCheckpointCodec.admission.hasProcessMemoryOwner {
            // The provider must charge manifest/decrypt buffers before reading.
            // Expose this binding so a missing host authority fails before IO.
            return CBv2CompleteCheckpointIOLease(
                reservation: .init(onRelease: {}), usesProcessMemoryOwner: true)
        }
        return CBv2CompleteCheckpointIOLease(
            reservation: try completeCheckpointCodec.admission.reserveTransient(
                bytes: CBv2CompleteCheckpointManifest.maximumProviderScratchBytes))
    }

    /// Called from the asynchronous provider stage path; no model execution,
    /// file access or tensor allocation occurs while planning.
    public func planCompleteCheckpointImport(
        manifest: CBv2CompleteCheckpointManifest, request: CBv2Request
    ) throws -> CBv2CompleteCheckpointImportPlan {
        guard let completeCheckpointCodec else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try validateContiguousCheckpointChunk(manifest, codec: completeCheckpointCodec)
        let work: CBv2NativeCompletePrefixWork?
        if let completePrefixCache {
            let factory = try nativeCompletePrefixWorkFactory(
                store: completePrefixCache,
                codec: completeCheckpointCodec, request: request)
            work = try factory?()  // counted BEFORE the plan retains codec/assistant
        } else if hasNativeCompletionTracking {
            throw CBv2NativeShutdownError.unsupportedConsumer
        } else {
            work = nil
        }
        do {
            let plan = try completeCheckpointCodec.plan(manifest: manifest, request: request)
            if let work, let completePrefixCache {
                try plan.bindNativeCompletePrefixWork(
                    work, store: completePrefixCache,
                    request: request, engineID: nativeShutdownEngineID)
            }
            return plan
        } catch {
            work?.finishAfterDroppingConsumers()
            throw error
        }
    }

    public func setCompletePrefixPublicationHandler(
        _ handler: (@Sendable (CBv2RequestID, [Int]) -> Void)?
    ) {
        completeCheckpointCapture?.setPublicationHandler(handler)
    }

    func completeCheckpointLookup(for request: CBv2Request) -> CBv2PrefixLookup {
        guard request.permitsHybridCheckpoint(layerKinds: layerKinds),
            let receiptID = request.prefixCacheReceiptID, let completePrefixCache
        else { return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0) }
        let (maximumLength, overflow) = request.promptTokens.count.addingReportingOverflow(
            max(1, request.maxTokens))
        guard !overflow else {
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        guard
            let staged = completePrefixCache.takeStaged(
                requestID: receiptID, tokens: request.promptTokens,
                cacheSalt: request.checkpointCacheSalt,
                maximumSequenceLength: maximumLength)
        else { return .init(adoption: nil, outcome: .miss, matchedTokens: 0) }
        do {
            if staged.hasNativeTracking {
                guard let completeCheckpointCodec, hasNativeCompletionTracking,
                    staged.maximumSequenceLength == maximumLength
                else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                try validateContiguousCheckpointChunk(
                    staged.manifest, codec: completeCheckpointCodec)
                try staged.prepareNativeHistoricalAssistant(
                    store: completePrefixCache,
                    request: request, engineID: nativeShutdownEngineID,
                    expectedCodec: completeCheckpointCodec)
                return try staged.withValidatedNativeCodec(
                    store: completePrefixCache,
                    request: request, engineID: nativeShutdownEngineID,
                    expectedCodec: completeCheckpointCodec
                ) { codec in
                    // Validation stays inside the stage's actual loan; do not
                    // create a second public deferred-plan loan just to check it.
                    _ = try codec.plan(manifest: staged.manifest, request: request)
                    let matched = staged.manifest.position
                    var plan =
                        codec.isNativePagedHistorical
                        ? try codec.historicalReusePlan(
                            position: matched, maximumSequenceLength: maximumLength)
                        : try codec.contiguousReusePlan(
                            position: matched, maximumSequenceLength: maximumLength)
                    plan.recurrentChunkSize = staged.manifest.chunkSize
                    plan.recurrentPromptLength = request.promptTokens.count
                    return .init(
                        adoption: .init(
                            requestID: receiptID, tokens: request.promptTokens,
                            matched: matched, plan: plan, prefix: [],
                            cacheSalt: request.checkpointCacheSalt,
                            completeCheckpoint: staged), outcome: .adoptionFailed,
                        matchedTokens: matched)
                }
            }
            // A tracked engine must never adopt an untracked/foreign stage.
            guard !hasNativeCompletionTracking else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            _ = try planCompleteCheckpointImport(manifest: staged.manifest, request: request)
            guard staged.maximumSequenceLength == maximumLength,
                staged.codec === completeCheckpointCodec
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let matched = staged.manifest.position
            var plan: CBv2PrefixReusePlan
            if staged.codec.contiguousLayout != nil {
                plan = try staged.codec.contiguousReusePlan(
                    position: matched, maximumSequenceLength: maximumLength)
            } else if staged.codec.historicalLayout != nil {
                plan = try staged.codec.historicalReusePlan(
                    position: matched, maximumSequenceLength: maximumLength)
            } else {
                let servingDescriptors =
                    try staged.codec.checkpointQuantization == nil
                    ? staged.manifest.tensors
                    : staged.codec.nativeTargetDescriptors(position: staged.manifest.position)
                let exactKVBytes = servingDescriptors.reduce(0) { total, tensor in
                    total
                        + ((tensor.role == .keys || tensor.role == .values) ? tensor.byteCount : 0)
                }
                let reuseBackend: CBv2PrefixReuseBackend =
                    staged.usesPagedBacking ? .pagedFP16 : .contiguousUnquantized
                let capability = CBv2PrefixReuseCapability.derive(
                    layerKinds: layerKinds, backend: reuseBackend)
                guard
                    let recurrentPlan = capability.plan(
                        matchedBoundary: matched, exactStagedFullKVBytes: exactKVBytes,
                        maximumSequenceLength: maximumLength,
                        nominalFullKVBytesPerToken: staged.codec.admission.fullKVBytesPerToken,
                        reserveFullSequenceTokens: true),
                    recurrentPlan.strategy == .direct, recurrentPlan.replayStart == matched
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                plan = recurrentPlan
            }
            if staged.codec.contiguousLayout != nil {
                plan.recurrentChunkSize = staged.manifest.chunkSize
                plan.recurrentPromptLength = request.promptTokens.count
            }
            // Chunk-agnostic complete-checkpoint adopters resume at `matched` under
            // ordinary scheduling: solo stripes, plain chunks and the
            // first-token projection all apply, and no bounded geometry wait
            // can cold-restart it. A recurrent adopter no longer continues
            // the donor's chunk geometry: on the dense Qwen target the state
            // and the continuation are partition-exact, and on the MoE a
            // cold run already depends on the partition, so forcing the
            // donor's stride restored nothing (see
            // `CBv2RecurrentCheckpointGeometry`). Chunk sizing is free;
            // packing is not: the adopter keeps the solo forward every
            // complete-checkpoint adopter has had, since a packed cohort was
            // never part of the parity evidence.
            plan.excludesPackedPrefill = true
            return .init(
                adoption: .init(
                    requestID: receiptID, tokens: request.promptTokens, matched: matched,
                    plan: plan, prefix: [], cacheSalt: request.checkpointCacheSalt,
                    completeCheckpoint: staged),
                outcome: .adoptionFailed, matchedTokens: matched)
        } catch {
            staged.close()
            return .init(
                adoption: nil, outcome: .adoptionFailed, matchedTokens: staged.manifest.position)
        }
    }
}
