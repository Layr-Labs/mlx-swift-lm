import Foundation
import MLX
import MLXLMCommon

/// Immutable validation for a particular loaded model and its verified disk
/// identity. Dtype is an explicit loaded-model contract, not accepted from the
/// incoming manifest. No weights or live model objects are retained here.
public final class DiffusionGemmaPersistentPrefixCodec: @unchecked Sendable {
    public let identity: CBv2CompleteCheckpointIdentity
    private let configuration: DiffusionGemmaTextConfiguration
    private let owner: UUID
    private let dtype: CBv2CheckpointDType
    private let codecIdentity = UUID()
    private let prefillChunkSize: Int?

    init(configuration: DiffusionGemmaTextConfiguration, owner: UUID,
         identity: CBv2CompleteCheckpointIdentity, dtype: DType, prefillChunkSize: Int? = nil) throws {
        guard let nativeType = CBv2CheckpointDType(dtype), nativeType.isFloatingPoint,
            [identity.modelAggregateHash, identity.promptContractID, identity.buildID, identity.numericsFingerprint]
                .allSatisfy({ $0.count == 64 && $0.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.configuration = configuration; self.owner = owner
        self.identity = identity; self.dtype = nativeType
        guard prefillChunkSize == nil || (prefillChunkSize! > 0 && prefillChunkSize! <= configuration.maxPositionEmbeddings)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.prefillChunkSize = prefillChunkSize
    }

    private func descriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        try configuration.layerTypes.enumerated().flatMap { index, kind in
            let sliding = kind == "sliding_attention"
            let heads = sliding ? configuration.keyValueHeads : (configuration.globalKeyValueHeads ?? configuration.keyValueHeads)
            let dimension = sliding ? configuration.headDimension : configuration.globalHeadDimension
            return try [CBv2CheckpointTensorRole.keys, .values].map {
                try CBv2CheckpointTensorDescriptor(role: $0, layer: index,
                          shape: [1, heads, sliding ? min(position, configuration.slidingWindow) : position, dimension],
                          dtype: dtype)
            }
        }
    }

    private func validate(_ manifest: CBv2CompleteCheckpointManifest,
                          prefixIdentity: DiffusionGemmaPrefixIdentity) throws {
        _ = try manifest.validateStructure()
        guard manifest.backendLayout == CBv2CompleteCheckpointManifest.diffusionBlockLayout,
            manifest.identity == identity,
            prefixIdentity.artifact == identity.modelAggregateHash,
            prefixIdentity.template == identity.promptContractID,
            prefixIdentity.numericalProfile == identity.numericsFingerprint + (prefillChunkSize.map { ":prefill-\($0)" } ?? ""),
            prefillChunkSize == nil || manifest.chunkSize == prefillChunkSize,
            manifest.mediaIdentity == prefixIdentity.mediaIdentity,
            prefixIdentity.media == (prefixIdentity.mediaIdentity.map {
                "media:" + $0.digest.map { String(format: "%02x", $0) }.joined()
            } ?? "text-only"),
            manifest.cacheSalt == prefixIdentity.tenantScope,
            manifest.position <= configuration.maxPositionEmbeddings,
            manifest.prefixTokens.allSatisfy({ $0 < configuration.vocabularySize }),
            manifest.tensors == (try descriptors(position: manifest.position)),
            let window = manifest.nativeBlockState,
            window.windowPhysicalLength <= max(manifest.position, configuration.slidingWindow)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        if manifest.position < configuration.slidingWindow {
            guard window.windowPhysicalLength >= manifest.position,
                window.windowCursor == manifest.position
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        } else {
            guard window.windowPhysicalLength >= configuration.slidingWindow,
                window.windowPhysicalLength == configuration.slidingWindow || window.windowCursor == window.windowPhysicalLength
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        }
    }

    /// Donor backing must remain charged by its capture owner through export
    /// completion. This adds a separate host-metadata reservation, not a second
    /// model or a dense copy of the prefix.
    public func export(_ checkpoint: DiffusionGemmaPrefixCheckpoint, chunkSize: Int,
                       engine: CBv2NativeBlockEngine) throws -> CBv2CompleteCheckpointExport {
        guard checkpoint.configuration == configuration, checkpoint.owner == owner, checkpoint.compact else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let manifest = CBv2CompleteCheckpointManifest(
            identity: identity, position: checkpoint.tokenCount, chunkSize: chunkSize,
            prefixTokens: checkpoint.tokens.map(Int.init), cacheSalt: checkpoint.identity.tenantScope,
            assistantCodecID: nil, tensors: try descriptors(position: checkpoint.tokenCount),
            backendLayout: CBv2CompleteCheckpointManifest.diffusionBlockLayout,
            mediaIdentity: checkpoint.identity.mediaIdentity,
            nativeBlockState: .init(windowPhysicalLength: checkpoint.windowOrder.physicalLength,
                                    windowCursor: checkpoint.windowOrder.cursor))
        try validate(manifest, prefixIdentity: checkpoint.identity)
        return try .init(nativeBlockManifest: manifest,
                         arrays: checkpoint.layers.flatMap { [$0.keys, $0.values] }, engine: engine)
    }

    /// No GPU allocation before exact artifact/build/numerics/scope/shape/token
    /// validation. Chunk size is the requesting engine's actual cold policy.
    public func importPlan(_ manifest: CBv2CompleteCheckpointManifest,
                           prefixIdentity: DiffusionGemmaPrefixIdentity,
                           promptTokens: [Int], chunkSize: Int, maximumNewTokens: Int = 1,
                           prefillGeometry: DiffusionGemmaPrefillGeometry? = nil,
                           engine: CBv2NativeBlockEngine) throws -> CBv2NativeBlockCheckpointImportPlan {
        try validate(manifest, prefixIdentity: prefixIdentity)
        if prefixIdentity.mediaIdentity != nil {
            guard let prefillGeometry, prefillGeometry.promptCount == promptTokens.count,
                prefillGeometry.chunkSize == chunkSize,
                prefillGeometry.permitsRestore(position: manifest.position)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        } else if prefillGeometry != nil {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let (maximumLength, overflow) = promptTokens.count.addingReportingOverflow(maximumNewTokens)
        guard manifest.chunkSize == chunkSize, promptTokens.count <= configuration.maxPositionEmbeddings,
            maximumNewTokens > 0, !overflow, maximumLength <= configuration.maxPositionEmbeddings,
            promptTokens.starts(with: manifest.prefixTokens),
            promptTokens.allSatisfy({ $0 >= 0 && $0 < configuration.vocabularySize })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        return try .init(manifest: manifest, engine: engine, codecIdentity: codecIdentity,
                         maximumSequenceLength: maximumLength)
    }

    public func adopt(_ staged: CBv2NativeBlockCheckpoint,
                      prefixIdentity: DiffusionGemmaPrefixIdentity) throws -> DiffusionGemmaPrefixCheckpoint {
        try validate(staged.manifest, prefixIdentity: prefixIdentity)
        guard staged.codecIdentity == codecIdentity else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let storage = try staged.consume()
        return .init(configuration: configuration, owner: owner, identity: prefixIdentity,
                     manifest: staged.manifest, storage: storage)
    }
}

extension DiffusionGemmaTextDecoder {
    public func makePersistentPrefixCodec(verifiedIdentity: CBv2CompleteCheckpointIdentity,
                                          kvDType: DType, prefillChunkSize: Int? = nil) throws -> DiffusionGemmaPersistentPrefixCodec {
        let empty = try DiffusionGemmaRequestCache(configuration: configuration, expectedPromptLength: 0)
        try validateCache(empty)
        return try .init(configuration: configuration, owner: empty.owner!, identity: verifiedIdentity,
                         dtype: kvDType, prefillChunkSize: prefillChunkSize)
    }
}
