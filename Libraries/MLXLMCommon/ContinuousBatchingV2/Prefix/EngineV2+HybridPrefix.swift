import Foundation

extension EngineV2 {
    public func setResidentPrefixPublicationHandler(
        _ handler: (@Sendable (CBv2RequestID, [Int]) -> Void)?
    ) {
        hybridPrefixCache?.setPublicationHandler(handler)
    }

    func hybridPrefixLookup(
        for request: CBv2Request, cache: CBv2HybridPrefixCache
    ) -> CBv2PrefixLookup {
        guard request.permitsHybridCheckpoint(layerKinds: layerKinds) else {
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        let (maximumSequenceLength, overflow) = request.promptTokens.count.addingReportingOverflow(
            max(1, request.maxTokens))
        guard !overflow else {
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        let maximumChunk = max(
            schedulerConfig.prefillChunkSize, schedulerConfig.soloPrefillStripeTokens ?? 0)
        guard let hit = cache.lookup(
            tokens: request.promptTokens, cacheSalt: request.checkpointCacheSalt,
            maximumChunkSize: maximumChunk)
        else { return .init(adoption: nil, outcome: .miss, matchedTokens: 0) }
        guard hit.checkpoint.mediaIdentity == request.hybridPrefixIdentity,
              hit.checkpoint.mediaTargetOnly == request.usesTargetOnlyMediaCheckpoint else {
            cache.endAdoption(pin: hit.pin)
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: layerKinds, backend: .contiguousUnquantized)
        guard var plan = capability.plan(
            matchedBoundary: hit.checkpoint.position,
            exactStagedFullKVBytes: hit.kvPrefix.reduce(0) { total, row in
                total + (row.map { $0.keys.nbytes + $0.values.nbytes } ?? 0)
            },
            maximumSequenceLength: maximumSequenceLength,
            // Native adoption builds a lazy copy into fresh request storage.
            // Keep its source backing covered even if the donor is evicted
            // before that graph finishes; do not infer dtype from slab slack.
            additionalStagedBackingBytes: hit.kvBackingBytes,
            reserveFullSequenceTokens: true),
            plan.strategy == .direct, plan.replayStart == hit.checkpoint.position
        else {
            cache.endAdoption(pin: hit.pin)
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: hit.checkpoint.position)
        }
        // Bank capture is chunk-agnostic (`isRecurrentBoundary`), so a bank
        // checkpoint may sit at any 256-token-aligned range end of a mixed
        // partition; forcing the checkpoint's last cap on the adopter would
        // neither reproduce that partition nor buy anything the durable path
        // does not have (see `CBv2RecurrentCheckpointGeometry`). The adopter
        // resumes at `matched` under ordinary chunking, out of packed
        // prefill, exactly as `completeCheckpointLookup` does.
        plan.excludesPackedPrefill = true
        return .init(
            adoption: .init(
                requestID: request.id, tokens: request.promptTokens,
                matched: hit.checkpoint.position, plan: plan, prefix: hit.kvPrefix,
                cacheSalt: request.checkpointCacheSalt,
                recurrentCheckpoint: hit.checkpoint, hybridPin: hit.pin),
            outcome: .adoptionFailed, matchedTokens: hit.checkpoint.position)
    }
}
