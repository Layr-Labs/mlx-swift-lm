import Foundation
import MLX
import MLXLMCommon

/// Trusted engine-side context for an in-process committed prefix. These are
/// verified identities, not untrusted request metadata. A new weight/template,
/// tenant, media sequence, numerical contract or epoch must never reuse a hit.
public struct DiffusionGemmaPrefixIdentity: Sendable, Equatable {
    public let tenantScope: String
    public let artifact: String
    public let template: String
    public let media: String
    public let mediaIdentity: CBv2HybridPrefixIdentity?
    public let numericalProfile: String
    public let epoch: String

    public init(
        tenantScope: String, artifact: String, template: String, media: String,
        numericalProfile: String, epoch: String, mediaIdentity: CBv2HybridPrefixIdentity? = nil
    ) throws {
        guard
            [tenantScope, artifact, template, media, numericalProfile, epoch].allSatisfy({
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            })
        else { throw DiffusionGemmaModelError.invalidInput("incomplete prefix identity") }
        self.tenantScope = tenantScope
        self.artifact = artifact
        self.template = template
        self.media = media
        self.mediaIdentity = mediaIdentity
        self.numericalProfile = numericalProfile
        self.epoch = epoch
    }
}

/// Evaluated, immutable committed encoder state. It owns no model, denoising
/// canvas, RNG or self-conditioning logits. Only this SDK can construct it;
/// arrays are deliberately not exposed for mutation or unchecked disk restore.
/// This is an in-process reuse primitive, NOT an authenticated persistent format.
public final class DiffusionGemmaPrefixCheckpoint {
    public let tokenCount: Int
    /// Logical retained payload bytes. Shared allocation capacity can be larger;
    /// this must not be substituted for measured physical-memory admission.
    public let retainedBytes: Int
    public let physicalRetainedBytes: Int
    public let compact: Bool
    let configuration: DiffusionGemmaTextConfiguration
    let owner: UUID
    let identity: DiffusionGemmaPrefixIdentity
    let tokens: [Int32]
    let windowOrder: DiffusionGemmaWindowOrder
    private(set) var layers: [(keys: MLXArray, values: MLXArray, offset: Int)]
    private var importedStorage: CBv2NativeBlockCheckpointStorage?
    private var importedManifest: CBv2CompleteCheckpointManifest?
    var isImported: Bool { importedStorage != nil }

    fileprivate init(
        cache: DiffusionGemmaRequestCache, identity: DiffusionGemmaPrefixIdentity, compact: Bool
    )
        throws
    {
        guard let owner = cache.owner, cache.position > 0,
            cache.committedTokenCount == cache.position,
            cache.rows.allSatisfy({ $0.absoluteOffset == cache.position })
        else { throw DiffusionGemmaModelError.invalidInput("incomplete committed prefix") }
        let layers = cache.snapshots().map { layer in
            compact
                ? (
                    keys: MLX.where(MLXArray(true), layer.keys, layer.keys),
                    values: MLX.where(MLXArray(true), layer.values, layer.values),
                    offset: layer.offset
                ) : layer
        }
        guard layers.count == cache.configuration.layerCount else {
            throw DiffusionGemmaModelError.invalidInput("incomplete prefix layers")
        }
        // Freeze graph roots before a donor continues or is retired. Slice
        // views retain the old value, not mutable cache objects; later updates
        // cannot donate storage still referenced by this checkpoint.
        try MLX.withError { errors in
            eval(layers.flatMap { [$0.keys, $0.values] })
            try errors.check()
        }
        self.layers = layers
        self.compact = compact
        configuration = cache.configuration
        self.owner = owner
        self.identity = identity
        tokens = cache.committedTokenBatches.flatMap { $0.asArray(Int32.self) }
        windowOrder = cache.windowOrder
        tokenCount = cache.position
        retainedBytes = layers.reduce(0) { $0 + $1.keys.nbytes + $1.values.nbytes }
        physicalRetainedBytes = try layers.flatMap { [$0.keys, $0.values] }.reduce(0) {
            count, array in
            guard let info = try array.evaluatedBufferInfo(), info.allocatedBytes >= array.nbytes
            else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            let (total, overflow) = count.addingReportingOverflow(info.allocatedBytes)
            guard !overflow else { throw CBv2CompleteCheckpointError.allocationFailed }
            return total
        }
    }

    init(
        configuration: DiffusionGemmaTextConfiguration, owner: UUID,
        identity: DiffusionGemmaPrefixIdentity, manifest: CBv2CompleteCheckpointManifest,
        storage: CBv2NativeBlockCheckpointStorage
    ) {
        self.configuration = configuration
        self.owner = owner
        self.identity = identity
        tokens = manifest.prefixTokens.map(Int32.init)
        tokenCount = manifest.position
        let order = manifest.nativeBlockState!
        windowOrder = .init(physicalLength: order.windowPhysicalLength, cursor: order.windowCursor)
        layers = stride(from: 0, to: storage.arrays.count, by: 2).map {
            (keys: storage.arrays[$0], values: storage.arrays[$0 + 1], offset: manifest.position)
        }
        retainedBytes = storage.arrays.reduce(0) { $0 + $1.nbytes }
        physicalRetainedBytes = storage.nativeDestinationBytes
        compact = true
        importedStorage = storage
        importedManifest = manifest
    }

    deinit {
        // Drop every alias before the destination and metadata permits retire.
        layers.removeAll(keepingCapacity: false)
        importedStorage = nil
        importedManifest = nil
    }

    /// Token/identity lookup without exposing mutable tensor state.
    public func matches(tokens candidate: [Int32], identity: DiffusionGemmaPrefixIdentity) -> Bool {
        self.identity == identity && candidate.starts(with: tokens)
    }
    public func samePrefix(as other: DiffusionGemmaPrefixCheckpoint) -> Bool {
        owner == other.owner && identity == other.identity && tokens == other.tokens
    }
}

extension DiffusionGemmaTextDecoder {
    /// Capture only after successful prompt/finalized-block encoding. The
    /// caller owns identity verification and single-lane serialization.
    public func checkpoint(
        cache: DiffusionGemmaRequestCache, identity: DiffusionGemmaPrefixIdentity,
        compact: Bool = false
    ) throws -> DiffusionGemmaPrefixCheckpoint {
        try validateCache(cache)
        return try DiffusionGemmaPrefixCheckpoint(
            cache: cache, identity: identity, compact: compact)
    }

    /// Restore a complete committed prefix into a fresh independent request.
    /// No model forward is run and no trailing-window replay is required.
    /// The returned position is the amount to skip when encoding the suffix.
    public func restorePrefix(
        _ checkpoint: DiffusionGemmaPrefixCheckpoint,
        identity: DiffusionGemmaPrefixIdentity, promptTokenIds: MLXArray,
        maximumSequenceLength: Int? = nil, pagedBackend: PagedKVBackend? = nil
    ) throws -> DiffusionGemmaRequestCache {
        guard checkpoint.configuration == configuration,
            checkpoint.identity == identity,
            promptTokenIds.ndim == 2, promptTokenIds.dim(0) == 1,
            promptTokenIds.dim(1) >= checkpoint.tokenCount,
            promptTokenIds.dim(1) <= configuration.maxPositionEmbeddings,
            promptTokenIds.dtype == .int32 || promptTokenIds.dtype == .uint32
        else { throw DiffusionGemmaModelError.invalidInput("prefix identity/shape/capacity") }
        let tokens = promptTokenIds.asArray(Int32.self)
        guard tokens.allSatisfy({ $0 >= 0 && $0 < configuration.vocabularySize }),
            tokens.starts(with: checkpoint.tokens)
        else { throw DiffusionGemmaModelError.invalidInput("prefix token mismatch") }
        let cache = try DiffusionGemmaRequestCache(
            configuration: configuration, expectedPromptLength: tokens.count,
            maximumSequenceLength: maximumSequenceLength, pagedBackend: pagedBackend)
        // Bind before copying any arrays. Same-shape weights from another model
        // instance or a reload must not inherit a prior owner's checkpoint.
        try validateCache(cache)
        guard cache.owner == checkpoint.owner else {
            throw DiffusionGemmaModelError.invalidInput("prefix model owner")
        }
        cache.retainImportedPrefix(checkpoint)
        try cache.performStep {
            for (row, layer) in zip(cache.rows, checkpoint.layers) {
                if let paged = row as? PagedSequenceKV, let pagedBackend {
                    if paged.windowSize != nil {
                        paged.fastForward(to: layer.offset - layer.keys.dim(2))
                    }
                    // Restore through bounded native page writes, not one oversized
                    // ring update or a replay of model computation.
                    let length = layer.keys.dim(2)
                    for start in stride(
                        from: 0, to: length, by: pagedBackend.pool.config.maxPrefillChunk)
                    {
                        let end = min(length, start + pagedBackend.pool.config.maxPrefillChunk)
                        _ = paged.update(
                            keys: layer.keys[0, 0..., start ..< end, 0...],
                            values: layer.values[0, 0..., start ..< end, 0...])
                    }
                    continue
                }
                if let windowed = row as? CBv2WindowedSequenceKV {
                    windowed.fastForward(to: layer.offset - layer.keys.dim(2))
                }
                _ = row.update(keys: layer.keys, values: layer.values)
            }
            cache.recordCommittedTokens(
                MLXArray(checkpoint.tokens).reshaped(1, checkpoint.tokenCount),
                restoredWindowOrder: checkpoint.windowOrder)
            try validateCache(cache)
            try cache.finishPagedStep([])
        }
        return cache
    }
}
