// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Scalar, immutable two-scope revalidation only. Load-session/generation and
/// binding fingerprint MUST NOT become a durable prefix namespace. The provider
/// separately binds verified artifact/template/build/numerics/tenant identity.
public struct MiMoV26NativeCompletePrefixMetadata: Sendable, Equatable {
    public let modelType: String
    public let loadSessionID: UUID
    public let loadBindingFingerprint: String
    public let loadedGeneration: UUID
    public let layerKinds: [CBv2LayerKind]
    public let layerDTypes: [DType]
    public let maximumContextTokens: Int
    public let assistantCodecID: String?
    public let verificationMode: CBv2MTPVerificationMode
    public let backendLayout: String
    // No public initializer: metadata comes from the actual strict-loaded
    // owner and the adapter's completed native type probe, not caller claims.
}

/// Only lifecycle scalars are synchronized here. No model, array, provider
/// callback, reservation or evaluator is stored in this lock.
final class MiMoV26NativePrefixLifetime {
    let loadSessionID: UUID
    let generation = UUID()
    private enum Phase: Equatable { case active, draining, invalid, retired }
    private let lock = NSLock()
    private var phase = Phase.active
    private var issuanceStarted = false
    private var contractID: UUID?

    init(loadSessionID: UUID) { self.loadSessionID = loadSessionID }
    func requirePreparation() throws {
        try lock.withLock {
            guard phase == .active, !issuanceStarted else {
                throw MiMoV26MultimodalError.invalidatedOwner
            }
        }
    }
    func beginIssuance() throws {
        try lock.withLock {
            guard phase == .active, !issuanceStarted else {
                throw MiMoV26MultimodalError.invalidatedOwner
            }
            issuanceStarted = true
        }
    }
    func recordContract(_ id: UUID) throws {
        try lock.withLock {
            guard phase == .active, issuanceStarted, contractID == nil else {
                throw MiMoV26MultimodalError.invalidatedOwner
            }
            contractID = id
        }
    }
    func requireBinding() throws {
        try lock.withLock {
            // Common beginLoan(importing) refuses NEW imports once shutdown
            // is requested. Existing owned work still needs cleanup validity.
            guard issuanceStarted, phase == .active || phase == .draining else {
                throw MiMoV26MultimodalError.invalidatedOwner
            }
        }
    }
    func beginDrain(contract: UUID) throws {
        try lock.withLock {
            guard contractID == contract, phase == .active || phase == .draining else {
                throw MiMoV26MultimodalError.incompatibleOwner
            }
            phase = .draining
        }
    }
    func retire(_ receipt: CBv2NativeShutdownReceipt) throws {
        try lock.withLock {
            guard receipt.executionContractID == contractID, receipt.generation == 1,
                phase == .active || phase == .draining
            else {
                throw MiMoV26MultimodalError.incompatibleOwner
            }
            phase = .retired
        }
    }
    func invalidate() { lock.withLock { phase = .invalid } }
}

/// Kept by the real loaded resources, not by a public callback or a standalone
/// marker. Both references are weak: the adapter already owns those resources,
/// and a completed external contract must not create a second weight owner.
package final class MiMoV26NativeCompletePrefixValidator: CBv2NativeCompletePrefixBindingValidating
{
    private weak var resources: MiMoV26LoadedResources?
    private weak var adapter: MiMoV26CBv2Adapter?
    private let expected: MiMoV26NativeCompletePrefixMetadata

    init(
        resources: MiMoV26LoadedResources, adapter: MiMoV26CBv2Adapter,
        expected: MiMoV26NativeCompletePrefixMetadata
    ) {
        self.resources = resources
        self.adapter = adapter
        self.expected = expected
    }
    package func validateNativeCompletePrefixBinding() throws {
        guard let resources, let adapter,
            resources.loaded.receipt.sessionID == expected.loadSessionID,
            resources.prepared.request.sessionID == expected.loadSessionID,
            resources.nativePrefixLifetime.loadSessionID == expected.loadSessionID,
            resources.nativePrefixLifetime.generation == expected.loadedGeneration,
            adapter.target === resources.loaded.bundle.target,
            adapter.layerKinds == expected.layerKinds,
            adapter.cbv2CompleteCheckpointKVDTypes == expected.layerDTypes,
            adapter.nativeCompletePrefixAssistantCodecID == expected.assistantCodecID,
            (adapter.assistant?.verificationMode ?? .serialTarget) == expected.verificationMode
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try resources.nativePrefixLifetime.requireBinding()
        // Checks actual current assistant/weight generation and completed KV
        // probe metadata. No parameter traversal, array read or native call.
        try adapter.validateNativeCompletePrefixOwner(resources)
    }
    func invalidate() { resources?.nativePrefixLifetime.invalidate() }
}

extension MiMoV26LoadedModel {
    /// First protected scope. Only this DTO may cross the provider's store
    /// registration await. Native preparation remains with this loaded wrapper.
    public func nativeCompletePrefixMetadata(
        binding: MiMoV26CBv2Binding,
        retaining scope: NativeConstructionScope
    ) throws -> MiMoV26NativeCompletePrefixMetadata {
        try resources.nativePrefixLifetime.requirePreparation()
        try scope.requireImmutableLoadedOwner(resources)
        if let previous = nativeCompletePrefixPreparation {
            // A fresh adapter cannot conceal a stale trained-head generation
            // in the preparation that crossed the host's registration await.
            try previous.binding.adapter.validateNativeCompletePrefixOwner(resources)
        }
        guard !hasIssuedManagedMediaProfile,
            binding.adapter.target === resources.loaded.bundle.target,
            binding.assistant === binding.adapter.assistant,
            binding.mediaGeneration === resources.mediaGeneration,
            resources.loaded.receipt.sessionID == resources.prepared.request.sessionID,
            resources.loaded.receipt.binding == resources.prepared.request.binding
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try binding.adapter.validateNativeCompletePrefixOwner(resources)
        guard let types = binding.adapter.cbv2CompleteCheckpointKVDTypes else {
            throw MiMoV26CBv2Error.nativeProbeRequired
        }
        let codecID = binding.adapter.nativeCompletePrefixAssistantCodecID
        guard (binding.assistant == nil) == (codecID == nil) else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        let metadata = MiMoV26NativeCompletePrefixMetadata(
            modelType: "mimo_v2",
            loadSessionID: loadReceipt.sessionID,
            loadBindingFingerprint: try loadReceipt.binding.fingerprint(),
            loadedGeneration: resources.nativePrefixLifetime.generation,
            layerKinds: binding.adapter.layerKinds, layerDTypes: types,
            maximumContextTokens: nativeConfiguration.maxPositionEmbeddings,
            assistantCodecID: codecID,
            verificationMode: binding.assistant?.verificationMode ?? .serialTarget,
            backendLayout: codecID == nil
                ? CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout
                : CBv2CompleteCheckpointManifest.contiguousAsymmetricMTPLayout)
        try scope.retainOwner(binding.adapter)
        try scope.invalidateOnFailedCompletion(resources.nativePrefixLifetime) {
            [weak lifetime = resources.nativePrefixLifetime] in
            lifetime?.invalidate()
        }
        nativeCompletePrefixPreparation = (metadata, binding)
        return metadata
    }

    /// Second protected scope, after the host registers the ACTUAL store and
    /// process ledger owner. The caller assembles its engine locally; these
    /// native handles are not Sendable and must not cross the provider await.
    public func makeNativeCompletePrefixExecutionResources(
        binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, expectedMetadata: MiMoV26NativeCompletePrefixMetadata,
        completePrefixCache: any CBv2NativeCompletePrefixCache,
        processMemoryOwner: any CBv2ProcessMemoryOwner,
        retaining scope: NativeConstructionScope
    ) throws -> MiMoV26CBv2NativeExecutionResources {
        let validator = try beginNativeCompletePrefixIssuance(
            binding: binding,
            expectedMetadata: expectedMetadata, retaining: scope)
        do {
            let issued = try binding.adapter.makeNativeCompletePrefixResources(
                bytesCapacity: bytesCapacity,
                loadedOwner: resources, validator: validator,
                completePrefixCache: completePrefixCache,
                processMemoryOwner: processMemoryOwner, retaining: scope)
            try finishNativeCompletePrefixIssuance(issued.contract)
            return issued
        } catch {
            failNativeCompletePrefixIssuance()
            throw error
        }
    }

    /// Shared only by the protected text and joint-media issuers. Metadata is
    /// checked BEFORE the one-shot transition, so a foreign DTO cannot consume
    /// a valid preparation. No raw owner or validator is publicly returned.
    func beginNativeCompletePrefixIssuance(
        binding: MiMoV26CBv2Binding,
        expectedMetadata: MiMoV26NativeCompletePrefixMetadata,
        retaining scope: NativeConstructionScope
    ) throws -> MiMoV26NativeCompletePrefixValidator {
        guard let prepared = nativeCompletePrefixPreparation, prepared.metadata == expectedMetadata
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        let actual = try nativeCompletePrefixMetadata(binding: binding, retaining: scope)
        guard actual == expectedMetadata else {
            nativeCompletePrefixPreparation = prepared
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try resources.nativePrefixLifetime.beginIssuance()
        let validator = MiMoV26NativeCompletePrefixValidator(
            resources: resources,
            adapter: binding.adapter, expected: actual)
        resources.nativePrefixValidator = validator
        do {
            try scope.retainOwner(validator)
            try scope.invalidateOnFailedCompletion(validator) { [weak validator] in
                validator?.invalidate()
            }
            return validator
        } catch {
            resources.nativePrefixLifetime.invalidate()
            throw error
        }
    }

    func finishNativeCompletePrefixIssuance(_ contract: CBv2NativeExecutionContract) throws {
        try resources.nativePrefixLifetime.recordContract(contract.id)
    }
    func failNativeCompletePrefixIssuance() { resources.nativePrefixLifetime.invalidate() }

    /// Host teardown notification only. Does not fence, free, stop an engine or
    /// grant credit. The host must also start the real EngineV2 shutdown gate.
    public func beginNativeCompletePrefixRetirement(executionContractID: UUID) throws {
        try resources.nativePrefixLifetime.beginDrain(contract: executionContractID)
    }

    /// An authentic one-use SDK contract receipt, AFTER host consumer proofs.
    /// Dropping preparation/validator aliases is not physical memory credit.
    public func releaseNativeCompletePrefixAfterNativeRetirement(
        _ receipt: CBv2NativeShutdownReceipt
    ) throws {
        try resources.nativePrefixLifetime.retire(receipt)
        nativeCompletePrefixPreparation = nil
        resources.nativePrefixValidator = nil
    }
}
