// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Scalar two-scope metadata only. Session/generation never enter the durable
/// namespace; the host separately binds artifact/template/build/numerics.
/// No public initializer and no native handles cross the store-creation await.
public struct MiMoV26NativePagedPrefixMetadata: Sendable {
    public let prefix: MiMoV26NativeCompletePrefixMetadata
    public let pagedConfiguration: PagedKVPoolConfig
    fileprivate func matches(_ other: Self) -> Bool {
        let a = pagedConfiguration, b = other.pagedConfiguration
        return prefix == other.prefix && a.pageSize == b.pageSize && a.capacityBytes == b.capacityBytes
            && a.dtype == b.dtype && a.layerDTypes == b.layerDTypes
            && a.gatheredAttention == b.gatheredAttention && a.segmentSizeBytes == b.segmentSizeBytes
            && a.maxPrefillChunk == b.maxPrefillChunk && a.nominalMaxSequenceLength == b.nominalMaxSequenceLength
            && a.maxBufferLength == b.maxBufferLength && a.prefixSharingBlockSize == b.prefixSharingBlockSize
    }
}

/// Exact strict-loaded validator. Weak native references avoid a second weight
/// owner in an externally retained contract; real resources retain this object.
package final class MiMoV26NativePagedValidator: CBv2NativePagedModelValidating,
                                                CBv2NativeCompletePrefixBindingValidating {
    private weak var resources: MiMoV26LoadedResources?
    private weak var adapter: MiMoV26CBv2Adapter?
    private weak var assistant: MiMoV26MTPAssistant?
    private let hasAssistant: Bool
    private let sessionID, generation: UUID
    private let types: [DType]
    private let kinds: [CBv2LayerKind]
    private let prefixMetadata: MiMoV26NativePagedPrefixMetadata?
    init(resources: MiMoV26LoadedResources, adapter: MiMoV26CBv2Adapter, types: [DType],
         prefixMetadata: MiMoV26NativePagedPrefixMetadata? = nil) {
        self.resources = resources; self.adapter = adapter
        assistant = adapter.assistant; hasAssistant = adapter.assistant != nil
        sessionID = resources.loaded.receipt.sessionID
        generation = resources.nativePagedLifetime.generation
        self.types = types; kinds = adapter.layerKinds
        self.prefixMetadata = prefixMetadata
    }
    package func validateNativePagedModel() throws {
        guard let resources, let adapter,
              resources.loaded.receipt.sessionID == sessionID,
              resources.prepared.request.sessionID == sessionID,
              resources.nativePagedLifetime.generation == generation,
              resources.loaded.receipt.binding == resources.prepared.request.binding,
              adapter.target === resources.loaded.bundle.target,
              hasAssistant == (adapter.assistant != nil), adapter.assistant === assistant,
              !hasAssistant || (adapter.supportsRequestStatefulMTP
                  && adapter.assistant?.verificationMode == .serialTarget),
              adapter.layerKinds == kinds, adapter.cbv2CompleteCheckpointKVDTypes == types else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try resources.nativePagedLifetime.requireBinding()
        try adapter.validateNativePagedOwner(resources)
    }
    package func validateNativeCompletePrefixBinding() throws {
        guard let expected = prefixMetadata?.prefix, let adapter,
              expected.loadSessionID == sessionID, expected.loadedGeneration == generation,
              expected.layerKinds == kinds, expected.layerDTypes == types,
              expected.maximumContextTokens == adapter.target.configuration.maxPositionEmbeddings,
              expected.assistantCodecID == adapter.nativeCompletePrefixAssistantCodecID,
              expected.verificationMode == .serialTarget,
              expected.backendLayout == (hasAssistant
                ? CBv2CompleteCheckpointManifest.pagedAsymmetricMTPLayout
                : CBv2CompleteCheckpointManifest.pagedAsymmetricLayout) else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        // One actual page lifetime, including draining cleanup, owns BOTH
        // permissions. No second generation or second process owner is minted.
        try validateNativePagedModel()
    }
}

extension MiMoV26LoadedModel {
    public func nativePagedCompletePrefixMetadata(binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, maximumConcurrentRequests: Int, maximumQueryTokens: Int,
        maximumPrefillChunk: Int, retaining scope: NativeConstructionScope) throws
        -> MiMoV26NativePagedPrefixMetadata {
        let config = try nativePagedConfiguration(binding: binding, bytesCapacity: bytesCapacity,
            maximumConcurrentRequests: maximumConcurrentRequests, maximumQueryTokens: maximumQueryTokens,
            maximumPrefillChunk: maximumPrefillChunk, retaining: scope)
        try binding.adapter.validateNativeCompletePrefixOwner(resources)
        guard let types = binding.adapter.cbv2CompleteCheckpointKVDTypes else {
            throw MiMoV26CBv2Error.nativeProbeRequired
        }
        let codec = binding.adapter.nativeCompletePrefixAssistantCodecID
        guard (binding.assistant == nil) == (codec == nil) else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        let metadata = MiMoV26NativeCompletePrefixMetadata(modelType: "mimo_v2",
            loadSessionID: loadReceipt.sessionID, loadBindingFingerprint: try loadReceipt.binding.fingerprint(),
            loadedGeneration: resources.nativePagedLifetime.generation,
            layerKinds: binding.adapter.layerKinds, layerDTypes: types,
            maximumContextTokens: nativeConfiguration.maxPositionEmbeddings, assistantCodecID: codec,
            verificationMode: .serialTarget, backendLayout: codec == nil
                ? CBv2CompleteCheckpointManifest.pagedAsymmetricLayout
                : CBv2CompleteCheckpointManifest.pagedAsymmetricMTPLayout)
        try scope.retainOwner(binding.adapter)
        try scope.invalidateOnFailedCompletion(resources.nativePagedLifetime) {
            [weak lifetime = resources.nativePagedLifetime] in lifetime?.invalidate()
        }
        nativePagedPreparation = binding
        return .init(prefix: metadata, pagedConfiguration: config)
    }

    public func makeNativePagedCompletePrefixExecutionResources(binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, maximumConcurrentRequests: Int, maximumQueryTokens: Int,
        maximumPrefillChunk: Int, expectedMetadata: MiMoV26NativePagedPrefixMetadata,
        completePrefixCache: any CBv2NativeCompletePrefixCache,
        processMemoryOwner: any CBv2ProcessMemoryOwner,
        retaining scope: NativeConstructionScope) throws -> MiMoV26CBv2NativePagedExecutionResources {
        guard let previous = nativePagedPreparation,
              expectedMetadata.prefix.loadSessionID == loadReceipt.sessionID,
              expectedMetadata.prefix.loadedGeneration == resources.nativePagedLifetime.generation else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        let actual = try nativePagedCompletePrefixMetadata(binding: binding, bytesCapacity: bytesCapacity,
            maximumConcurrentRequests: maximumConcurrentRequests, maximumQueryTokens: maximumQueryTokens,
            maximumPrefillChunk: maximumPrefillChunk, retaining: scope)
        guard actual.matches(expectedMetadata) else {
            nativePagedPreparation = previous
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        return try issueNativePagedExecutionResources(binding: binding, expectedAssistant: binding.assistant,
            bytesCapacity: bytesCapacity, maximumConcurrentRequests: maximumConcurrentRequests,
            maximumQueryTokens: maximumQueryTokens, maximumPrefillChunk: maximumPrefillChunk,
            processMemoryOwner: processMemoryOwner, retaining: scope,
            completePrefixCache: completePrefixCache, prefixMetadata: actual)
    }

    /// Existing target-only issuer. A caller cannot upgrade its meaning by
    /// handing it an MTP binding; the explicit serial-MTP issuer is separate.
    /// All handles remain actor-local inside
    /// the already owned protected assembly operation.
    public func makeNativePagedExecutionResources(binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, maximumConcurrentRequests: Int, maximumQueryTokens: Int,
        maximumPrefillChunk: Int, processMemoryOwner: any CBv2ProcessMemoryOwner,
        retaining scope: NativeConstructionScope) throws -> MiMoV26CBv2NativePagedExecutionResources {
        guard binding.assistant == nil else { throw MiMoV26MultimodalError.incompatibleOwner }
        return try issueNativePagedExecutionResources(binding: binding, expectedAssistant: nil,
            bytesCapacity: bytesCapacity, maximumConcurrentRequests: maximumConcurrentRequests,
            maximumQueryTokens: maximumQueryTokens, maximumPrefillChunk: maximumPrefillChunk,
            processMemoryOwner: processMemoryOwner, retaining: scope)
    }

    /// Explicit genuine trained-head serialTarget composition. No Boolean or
    /// requested profile string substitutes for this actual bound assistant.
    public func makeNativePagedSerialMTPExecutionResources(binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, maximumConcurrentRequests: Int, maximumQueryTokens: Int,
        maximumPrefillChunk: Int, processMemoryOwner: any CBv2ProcessMemoryOwner,
        retaining scope: NativeConstructionScope) throws -> MiMoV26CBv2NativePagedExecutionResources {
        guard let assistant = binding.assistant, assistant.verificationMode == .serialTarget else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        return try issueNativePagedExecutionResources(binding: binding, expectedAssistant: assistant,
            bytesCapacity: bytesCapacity, maximumConcurrentRequests: maximumConcurrentRequests,
            maximumQueryTokens: maximumQueryTokens, maximumPrefillChunk: maximumPrefillChunk,
            processMemoryOwner: processMemoryOwner, retaining: scope)
    }

    private func issueNativePagedExecutionResources(binding: MiMoV26CBv2Binding,
        expectedAssistant: MiMoV26MTPAssistant?, bytesCapacity: Int,
        maximumConcurrentRequests: Int, maximumQueryTokens: Int, maximumPrefillChunk: Int,
        processMemoryOwner: any CBv2ProcessMemoryOwner,
        retaining scope: NativeConstructionScope,
        completePrefixCache: (any CBv2NativeCompletePrefixCache)? = nil,
        prefixMetadata: MiMoV26NativePagedPrefixMetadata? = nil) throws -> MiMoV26CBv2NativePagedExecutionResources {
        guard binding.assistant === expectedAssistant,
              (completePrefixCache != nil) == (prefixMetadata != nil) else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        let config = try nativePagedConfiguration(binding: binding, bytesCapacity: bytesCapacity,
            maximumConcurrentRequests: maximumConcurrentRequests, maximumQueryTokens: maximumQueryTokens,
            maximumPrefillChunk: maximumPrefillChunk, retaining: scope)
        try resources.nativePagedLifetime.beginIssuance()
        // The single paged lifetime owns the optional joint prefix. Contiguous
        // media/prefix alternatives cannot issue a competing tuple afterward.
        resources.nativePrefixLifetime.invalidate()
        let validator = MiMoV26NativePagedValidator(resources: resources,
            adapter: binding.adapter, types: config.layerDTypes!, prefixMetadata: prefixMetadata)
        resources.nativePagedValidator = validator
        nativePagedPreparation = binding
        do {
            try scope.retainOwner(validator); try scope.retainOwner(binding.adapter)
            try scope.retainOwner(processMemoryOwner)
            if let completePrefixCache { try scope.retainOwner(completePrefixCache) }
            try scope.invalidateOnFailedCompletion(resources.nativePagedLifetime) {
                [weak lifetime = resources.nativePagedLifetime] in lifetime?.invalidate()
            }
            let issued = try binding.adapter.makeNativePagedResources(config: config,
                processMemoryOwner: processMemoryOwner, loadedOwner: resources,
                validator: validator, retaining: scope,
                completePrefixCache: completePrefixCache,
                completePrefixValidator: prefixMetadata == nil ? nil : validator)
            try resources.nativePagedLifetime.recordContract(issued.contract.id)
            return issued
        } catch {
            resources.nativePagedLifetime.invalidate()
            throw error
        }
    }

    private func nativePagedConfiguration(binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, maximumConcurrentRequests: Int, maximumQueryTokens: Int,
        maximumPrefillChunk: Int, retaining scope: NativeConstructionScope) throws -> PagedKVPoolConfig {
        try scope.requireImmutableLoadedOwner(resources)
        try resources.nativePagedLifetime.requirePreparation()
        try resources.nativePrefixLifetime.requirePreparation()
        guard !hasIssuedManagedMediaProfile, nativeCompletePrefixPreparation == nil,
              resources.audioSidecar == nil,
              resources.loaded.receipt.sessionID == resources.prepared.request.sessionID,
              resources.loaded.receipt.binding == resources.prepared.request.binding,
              binding.adapter.target === resources.loaded.bundle.target,
              binding.assistant === binding.adapter.assistant,
              binding.assistant == nil || (binding.adapter.supportsRequestStatefulMTP
                  && binding.assistant?.verificationMode == .serialTarget),
              binding.mediaGeneration === resources.mediaGeneration,
              let types = binding.adapter.cbv2CompleteCheckpointKVDTypes,
              types.count == binding.adapter.layerKinds.count,
              types.allSatisfy({ $0 == .bfloat16 || $0 == .float16 }),
              bytesCapacity > 0, maximumConcurrentRequests > 0,
              maximumPrefillChunk > 0, maximumQueryTokens >= maximumPrefillChunk,
              maximumQueryTokens <= nativeConfiguration.maxPositionEmbeddings else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        if let previous = nativePagedPreparation {
            try previous.adapter.validateNativePagedOwner(resources)
        }
        let maximumBuffer: Int
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        guard Device.defaultDevice().deviceType == .gpu,
              StreamOrDevice.default.stream == MLX.Stream.gpu else {
            throw CBv2KVError.backendIneligible(reason: "native paged target requires actual default GPU stream")
        }
        maximumBuffer = GPU.deviceInfo().maxBufferSize
        guard maximumBuffer > 0 else { throw MiMoV26MultimodalError.incompatibleOwner }
        #else
        throw CBv2KVError.backendIneligible(reason: "native paged target requires Metal")
        #endif
        let limits = try CBv2PagedGatheredAttentionLimits(
            maximumBatchSize: maximumConcurrentRequests, maximumQueryTokens: maximumQueryTokens,
            maximumContextTokens: nativeConfiguration.maxPositionEmbeddings,
            maximumInFlightGraphs: 2, maximumScratchBytes: bytesCapacity,
            admissionMode: .stepOwned(.pinnedMetal))
        return PagedKVPoolConfig(capacityBytes: bytesCapacity, dtype: types[0],
            maxPrefillChunk: maximumPrefillChunk,
            nominalMaxSequenceLength: nativeConfiguration.maxPositionEmbeddings,
            maxBufferLength: maximumBuffer, prefixSharingBlockSize: nil,
            segmentSizeBytes: min(64 << 20, maximumBuffer), layerDTypes: types,
            gatheredAttention: limits)
    }

    public func beginNativePagedRetirement(executionContractID: UUID) throws {
        try resources.nativePagedLifetime.beginDrain(contract: executionContractID)
    }

    public func releaseNativePagedAfterNativeRetirement(_ receipt: CBv2NativeShutdownReceipt) throws {
        try resources.nativePagedLifetime.retire(receipt)
        nativePagedPreparation = nil
        resources.nativePagedValidator = nil
    }
}
