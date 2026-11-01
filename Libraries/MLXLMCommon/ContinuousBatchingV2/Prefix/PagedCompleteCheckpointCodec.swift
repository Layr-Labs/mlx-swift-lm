import MLX

extension CBv2CompleteCheckpointCodec {
    /// Page maps are captured on the engine queue while donor rows remain
    /// pinned. No whole-prefix snapshot/gather reaches the retirement queue.
    func exportPaged(
        checkpoint: CBv2RecurrentCheckpoint, state: [CBv2SequenceKV?],
        tokens: [Int], cacheSalt: String?
    ) throws -> CBv2CompleteCheckpointExport {
        guard !unsupportedAsymmetricGeometry || glm5NextLayout != nil else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        guard checkpoint.position < tokens.count, state.count == layerKinds.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        if glm5NextLayout != nil { try validateGLM5NextDonor(checkpoint: checkpoint, state: state) }
        let metadataPermit = try CBv2CheckpointManifestMemory.Permit(
            admission: admission, position: checkpoint.position)
        return try withExtendedLifetime(metadataPermit) {
            try makePagedExport(
                checkpoint: checkpoint, state: state, tokens: tokens,
                cacheSalt: cacheSalt, metadataPermit: metadataPermit)
        }
    }

    private func makePagedExport(
        checkpoint: CBv2RecurrentCheckpoint, state: [CBv2SequenceKV?], tokens: [Int],
        cacheSalt: String?,
        metadataPermit: CBv2CheckpointManifestMemory.Permit
    ) throws -> CBv2CompleteCheckpointExport {
        let descriptors = try tensorDescriptors(
            position: checkpoint.position, qwen4: qwen4Descriptors(checkpoint),
            mediaTargetOnly: checkpoint.mediaTargetOnly)
        // GLM payload roles may be written narrower than their FP32 pages
        // and state; every other layout writes its native dtype.
        let storage = glm5NextLayout?.storage
        var sources: [CBv2CompleteCheckpointTensorSource] = []
        for (index, entry) in state.enumerated() {
            guard let row = entry as? PagedSequenceKV,
                row.groupKey.dtype == kvDTypes[index], row.pool.layerKinds == layerKinds,
                row.groupKey == row.pool.groupKey(forLayer: index)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            let pageMap = try CBv2PagedCheckpointPageMap(
                row: row, position: checkpoint.position, admission: admission)
            let recent =
                try checkpoint.quantizedRecent[index]
                ?? ((row.groupKey.quantization?.recentTokenCount ?? 0) > 0
                    ? CBv2QuantizedCheckpointRecent(
                        row: row, position: checkpoint.position, admission: admission)
                    : nil)
            sources.append(
                try checkpointSource(
                    .paged(
                        try .init(
                            pageMap: pageMap, values: false,
                            recent: recent, wire: storage?.latent)), layer: index,
                    position: checkpoint.position, values: false))
            sources.append(
                try checkpointSource(
                    .paged(
                        try .init(
                            pageMap: pageMap, values: true,
                            recent: recent, wire: storage?.index)), layer: index,
                    position: checkpoint.position, values: true))
        }
        for spec in recurrentSpec?.layers ?? [] {
            guard let layer = checkpoint.layers[spec.modelLayerIndex], let conv = layer.conv,
                let ssm = layer.ssm
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            sources.append(
                storage.map { .exported(conv, wire: $0.convolution) } ?? .array(conv))
            sources.append(.array(ssm))
        }
        sources.append(contentsOf: try qwen4Arrays(checkpoint).map { .array($0) })
        if let assistant, !checkpoint.mediaTargetOnly {
            guard let state = checkpoint.assistant,
                let encoded = assistant.encodePrefixCheckpoint(state)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            sources.append(contentsOf: encoded.map { .array($0) })
        } else if checkpoint.assistant != nil {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        guard sources.count == descriptors.count,
            zip(sources, descriptors).allSatisfy({ $0.0.matches($0.1) })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let manifest = CBv2CompleteCheckpointManifest(
            schemaVersion: CBv2CompleteCheckpointManifest.currentSchemaVersion, identity: identity,
            backendLayout: backendLayout, position: checkpoint.position,
            chunkSize: checkpoint.chunkSize,
            cacheSalt: cacheSalt,
            assistantCodecID: checkpoint.mediaTargetOnly ? nil : assistant?.prefixCheckpointCodecID,
            mediaIdentity: checkpoint.mediaIdentity, mediaTargetOnly: checkpoint.mediaTargetOnly,
            glm5NextState: glm5NextLayout?.state(position: checkpoint.position),
            checkpointQuantization: checkpointQuantization,
            checkpointNativeDTypes: checkpointNativeDTypes,
            metadata: .init(
                tokens: Array(tokens.prefix(checkpoint.position)), tensors: descriptors,
                attentionLayers: glm5NextLayout?.attention.layers,
                permit: metadataPermit))
        _ = try manifest.validateStructure()
        return .init(
            manifest: manifest, sources: sources,
            usesProcessMemoryOwner: admission.hasProcessMemoryOwner)
    }
}
