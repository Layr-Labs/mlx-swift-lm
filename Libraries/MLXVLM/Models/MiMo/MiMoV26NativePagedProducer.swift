// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Exact strict-loaded validator. Weak native references avoid a second weight
/// owner in an externally retained contract; real resources retain this object.
package final class MiMoV26NativePagedValidator: CBv2NativePagedModelValidating {
    private weak var resources: MiMoV26LoadedResources?
    private weak var adapter: MiMoV26CBv2Adapter?
    private let sessionID, generation: UUID
    private let types: [DType]
    private let kinds: [CBv2LayerKind]
    init(resources: MiMoV26LoadedResources, adapter: MiMoV26CBv2Adapter, types: [DType]) {
        self.resources = resources; self.adapter = adapter
        sessionID = resources.loaded.receipt.sessionID
        generation = resources.nativePagedLifetime.generation
        self.types = types; kinds = adapter.layerKinds
    }
    package func validateNativePagedModel() throws {
        guard let resources, let adapter,
              resources.loaded.receipt.sessionID == sessionID,
              resources.prepared.request.sessionID == sessionID,
              resources.nativePagedLifetime.generation == generation,
              resources.loaded.receipt.binding == resources.prepared.request.binding,
              adapter.target === resources.loaded.bundle.target, adapter.assistant == nil,
              adapter.layerKinds == kinds, adapter.cbv2CompleteCheckpointKVDTypes == types else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try resources.nativePagedLifetime.requireBinding()
        try adapter.validateNativePagedOwner(resources)
    }
}

extension MiMoV26LoadedModel {
    /// Target-only genuine segmented page-backed native attention. Not a
    /// persistent-prefix or MTP issuer. All handles remain actor-local inside
    /// the already owned protected assembly operation.
    public func makeNativePagedExecutionResources(binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, maximumConcurrentRequests: Int, maximumQueryTokens: Int,
        maximumPrefillChunk: Int, processMemoryOwner: any CBv2ProcessMemoryOwner,
        retaining scope: NativeConstructionScope) throws -> MiMoV26CBv2NativePagedExecutionResources {
        try scope.requireImmutableLoadedOwner(resources)
        try resources.nativePagedLifetime.requirePreparation()
        try resources.nativePrefixLifetime.requirePreparation()
        guard !hasIssuedManagedMediaProfile, nativeCompletePrefixPreparation == nil,
              resources.audioSidecar == nil, nativePagedPreparation == nil,
              binding.adapter.target === resources.loaded.bundle.target,
              binding.assistant == nil, binding.adapter.assistant == nil,
              binding.mediaGeneration === resources.mediaGeneration,
              let types = binding.adapter.cbv2CompleteCheckpointKVDTypes,
              types.count == binding.adapter.layerKinds.count,
              types.allSatisfy({ $0 == .bfloat16 || $0 == .float16 }),
              bytesCapacity > 0, maximumConcurrentRequests > 0,
              maximumPrefillChunk > 0, maximumQueryTokens >= maximumPrefillChunk,
              maximumQueryTokens <= nativeConfiguration.maxPositionEmbeddings else {
            throw MiMoV26MultimodalError.incompatibleOwner
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
        let config = PagedKVPoolConfig(capacityBytes: bytesCapacity, dtype: types[0],
            maxPrefillChunk: maximumPrefillChunk,
            nominalMaxSequenceLength: nativeConfiguration.maxPositionEmbeddings,
            maxBufferLength: maximumBuffer, prefixSharingBlockSize: nil,
            segmentSizeBytes: min(64 << 20, maximumBuffer), layerDTypes: types,
            gatheredAttention: limits)
        try resources.nativePagedLifetime.beginIssuance()
        // Profiles are one-shot alternatives. Existing combined/ordinary media
        // issuers already require the prefix lifetime's unissued state.
        resources.nativePrefixLifetime.invalidate()
        let validator = MiMoV26NativePagedValidator(resources: resources,
            adapter: binding.adapter, types: types)
        resources.nativePagedValidator = validator
        nativePagedPreparation = binding
        do {
            try scope.retainOwner(validator); try scope.retainOwner(binding.adapter)
            try scope.retainOwner(processMemoryOwner)
            try scope.invalidateOnFailedCompletion(resources.nativePagedLifetime) {
                [weak lifetime = resources.nativePagedLifetime] in lifetime?.invalidate()
            }
            let issued = try binding.adapter.makeNativePagedResources(config: config,
                processMemoryOwner: processMemoryOwner, loadedOwner: resources,
                validator: validator, retaining: scope)
            try resources.nativePagedLifetime.recordContract(issued.contract.id)
            return issued
        } catch {
            resources.nativePagedLifetime.invalidate()
            throw error // real preparation remains retained through actual failure retirement
        }
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
