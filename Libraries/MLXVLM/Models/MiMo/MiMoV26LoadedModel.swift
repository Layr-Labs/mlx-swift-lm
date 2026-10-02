// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Non-Module lifetime box. It retains the exact load receipt/reservation and
/// all root components without adding a second reflected parameter tree.
final class MiMoV26LoadedResources {
    let loaded: MiMoV26SerialLoadResult
    let prepared: MiMoV26ModelFactory.Prepared
    let mediaGeneration = MiMoV26MediaGeneration()
    let nativePrefixLifetime: MiMoV26NativePrefixLifetime
    var nativePrefixValidator: MiMoV26NativeCompletePrefixValidator?
    let nativePagedLifetime: MiMoV26NativePrefixLifetime
    var nativePagedValidator: MiMoV26NativePagedValidator?
    var audioSidecar: MiMoV26AudioSidecarLoaded?
    var audioInstallation: (ownerID: UUID, epoch: UInt64)?
    init(loaded: MiMoV26SerialLoadResult, prepared: MiMoV26ModelFactory.Prepared) {
        self.loaded = loaded
        self.prepared = prepared
        nativePrefixLifetime = .init(loadSessionID: loaded.receipt.sessionID)
        nativePagedLifetime = .init(loadSessionID: loaded.receipt.sessionID)
    }
}

/// Actor-local native serving handles. Both exported handles independently
/// retain the complete loaded owner; neither requires the wrapper to stay alive.
public struct MiMoV26CBv2Binding {
    public let adapter: MiMoV26CBv2Adapter
    public let assistant: MiMoV26MTPAssistant?
    public let stopTokenIDs: Set<Int>
    let mediaGeneration: MiMoV26MediaGeneration
}

/// Native-local construction result; no second target and no Sendable raw
/// component escape. Only the protected loaded wrapper can issue this profile.
public struct MiMoV26ManagedMediaExecutionResources {
    public let backend: MiMoV26CBv2Backend
    public let cacheProvider: any CBv2LayerCacheProvider
    public let contract: CBv2NativeExecutionContract
}

/// Immutable loaded wrapper for ordinary ModelContext/ModelContainer ownership.
/// It registers each existing component exactly once, never reloads/quantizes,
/// and retains the loaded media towers. Generic iteration is still refused;
/// managed decoded media is a separate explicitly issued native profile.
/// Inherited Module mutation is outside the verified immutable-load contract.
public final class MiMoV26LoadedModel: Module, LanguageModel, GenericGenerationValidating {
    let resources: MiMoV26LoadedResources
    private let target: MiMoV26TextModel
    private let vision: MiMoV26VisionTower
    private let audioPatch: MiMoV26AudioPatchEncoder
    private let mtp: MiMoV26MTP
    // Native-only preparation remains under the real loaded wrapper while the
    // provider registers its store using scalar metadata across an await.
    var nativePagedPreparation: MiMoV26CBv2Binding?
    var nativeCompletePrefixPreparation:
        (metadata: MiMoV26NativeCompletePrefixMetadata, binding: MiMoV26CBv2Binding)?
    var hasIssuedManagedMediaProfile: Bool { managedMedia != nil || managedAudio != nil }
    private var managedMedia: (processor: MiMoV26MultimodalProcessor, binding: MiMoV26CBv2Binding)?
    private var managedAudio:
        (
            processor: MiMoV26MultimodalProcessor,
            binding: MiMoV26CBv2Binding, contractID: UUID
        )?

    init(loaded: MiMoV26SerialLoadResult, prepared: MiMoV26ModelFactory.Prepared) {
        resources = .init(loaded: loaded, prepared: prepared)
        target = loaded.bundle.target
        vision = loaded.bundle.vision
        audioPatch = loaded.bundle.audioPatch
        mtp = loaded.bundle.mtp
        super.init()
    }

    public var nativeConfiguration: MiMoV26Configuration { target.configuration }
    public var loadReceipt: MiMoV26SerialLoadReceipt { resources.loaded.receipt }
    public var templateSHA256: String { resources.prepared.templateSHA256 }
    public var stopTokenIDs: Set<Int> { resources.prepared.stopTokenIDs }

    /// Provider must perform the adapter's explicit native KV type probe before
    /// creating a backend. MTP remains an explicit choice, disabled by default.
    /// adapter.target is an existing borrowed lower-layer escape hatch: extracting
    /// that raw target and dropping its adapter is not owner-guaranteed use.
    public func makeCBv2Binding(
        enableMTP: Bool = false,
        verificationMode: CBv2MTPVerificationMode = .serialTarget
    ) throws -> MiMoV26CBv2Binding {
        guard enableMTP || verificationMode == .serialTarget else {
            throw MiMoV26MTPError.unsupportedConfiguration(
                "Rectangular verification requires the real loaded assistant")
        }
        let assistant =
            enableMTP
            ? try MiMoV26MTPAssistant(
                target: target, predictor: mtp,
                retaining: resources, verificationMode: verificationMode) : nil
        let adapter = try MiMoV26CBv2Adapter(
            target: target, assistant: assistant, retaining: resources)
        return .init(
            adapter: adapter, assistant: assistant, stopTokenIDs: stopTokenIDs,
            mediaGeneration: resources.mediaGeneration)
    }

    /// Native decoded-input entrypoint. The same loaded binding must own
    /// the eventual EngineV2. Its real type probe still precedes backend/cache
    /// provider construction. Installed text MTP uses the real request-scoped
    /// target-only media fallback; this does not qualify media drafting.
    public func makeMultimodalProcessor(
        binding: MiMoV26CBv2Binding,
        limits: MiMoV26MultimodalLimits,
        audioCodec: MiMoV26OwnedAudioCodec? = nil
    ) throws -> MiMoV26MultimodalProcessor {
        guard binding.adapter.target === target,
            binding.assistant === binding.adapter.assistant,
            binding.adapter.supportsMultimodalPrefill(attention: .causal),
            binding.mediaGeneration === resources.mediaGeneration
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        return try .init(
            configuration: nativeConfiguration, tokenizer: resources.prepared.tokenizer,
            chatTemplate: resources.prepared.processor.chatTemplate, templateSHA256: templateSHA256,
            limits: limits, vision: vision, audioPatch: audioPatch, adapter: binding.adapter,
            stopTokens: stopTokenIDs, generation: resources.mediaGeneration, retaining: resources,
            audioCodec: audioCodec,
            audioSidecar: resources.audioSidecar?.codec === audioCodec
                ? resources.audioSidecar : nil)
    }

    /// Setup only, inside the existing factory construction scope, after the
    /// real KV probe. Warm requests never call this or reopen that scope.
    public func makeManagedMediaExecutionResources(
        binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, limits: MiMoV26MultimodalLimits,
        retaining scope: NativeConstructionScope
    ) throws -> MiMoV26ManagedMediaExecutionResources {
        guard managedMedia == nil, managedAudio == nil else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try resources.nativePrefixLifetime.requirePreparation()
        try scope.requireImmutableLoadedOwner(resources)
        let processor = try makeMultimodalProcessor(binding: binding, limits: limits)
        try scope.retainOwner(processor)
        let issued = try binding.adapter.makeNativeDecodedVisualResources(
            bytesCapacity: bytesCapacity, processor: processor, loadedOwner: resources,
            retaining: scope)
        managedMedia = (processor, binding)
        return .init(
            backend: issued.backend, cacheProvider: issued.cacheProvider, contract: issued.contract)
    }

    /// Load the separately admitted, authenticated input codec in the SAME
    /// protected construction scope. Store its actual owner before any late
    /// veto; never create a reflected Module or a second target/container.
    public func installAudioSidecar(
        session: MiMoV26AudioSidecarLoadSession,
        reservation: any MiMoV26AudioSidecarLoadReservation,
        retaining scope: NativeConstructionScope, isCancelled: () -> Bool
    ) throws
        -> MiMoV26AudioSidecarLoadReceipt
    {
        try scope.requireImmutableLoadedOwner(resources)
        try resources.nativePrefixLifetime.requirePreparation()
        guard resources.audioSidecar == nil, managedMedia == nil, managedAudio == nil,
            session.request.canonicalRoot == loadReceipt.binding.canonicalRoot,
            session.request.mainConfigurationSHA256 == loadReceipt.binding.configSHA256,
            session.mainConfiguration == nativeConfiguration
        else {
            throw MiMoV26AudioSidecarError.invalidBinding
        }
        let loaded = try session.load(
            reservation: reservation, retaining: scope, isCancelled: isCancelled)
        resources.audioSidecar = loaded
        resources.audioInstallation = (scope.snapshot.ownerID, scope.snapshot.epoch)
        try scope.retainOwner(loaded)
        try scope.authorizeImmutableLoadedOwner(loaded)
        try scope.invalidateOnFailedCompletion(loaded) { loaded.invalidate() }
        try loaded.validate()
        return loaded.receipt
    }

    /// Serialized installed-owner validation, with metadata-only return. No
    /// sidecar/session/codec reference escapes the strict loaded wrapper.
    public func validateInstalledAudioSidecar(expectedRequest: MiMoV26AudioSidecarLoadRequest)
        throws
        -> MiMoV26AudioSidecarLoadReceipt
    {
        guard let sidecar = resources.audioSidecar, sidecar.receipt.request == expectedRequest,
            expectedRequest.canonicalRoot == loadReceipt.binding.canonicalRoot,
            expectedRequest.mainConfigurationSHA256 == loadReceipt.binding.configSHA256
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try sidecar.validate()
        return sidecar.receipt
    }

    /// Cold setup only: a sidecar may have been installed before a later veto,
    /// and no managed profile was issued; host also requires its exact engine
    /// absent. The exact installation epoch must have
    /// genuinely completed. Failed construction produces no admissible receipt.
    /// The host still proves all enclosing Tasks/aliases and settles its own C.
    public func releaseInstalledAudioAfterConstructionCompletion(
        _ receipt: NativeConstructionReceipt
    ) throws {
        guard managedAudio == nil, managedMedia == nil else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        guard let sidecar = resources.audioSidecar else {
            guard resources.audioInstallation == nil else {
                throw MiMoV26MultimodalError.incompatibleOwner
            }
            return  // no installed codec, not a native completion/credit assertion
        }
        guard let origin = resources.audioInstallation, receipt.ownerID == origin.ownerID,
            receipt.epoch == origin.epoch, receipt.completion == .capturedStreamsCompleted
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        sidecar.invalidate()
        resources.audioSidecar = nil
        resources.audioInstallation = nil
    }

    /// Setup only. Bare codec conformance/digest metadata cannot issue this
    /// profile: the typed owner must come from installAudioSidecar above.
    public func makeManagedAudioExecutionResources(
        binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, limits: MiMoV26MultimodalLimits,
        retaining scope: NativeConstructionScope
    ) throws -> MiMoV26ManagedMediaExecutionResources {
        guard managedMedia == nil, managedAudio == nil, let sidecar = resources.audioSidecar else {
            throw MiMoV26MultimodalError.missingAudioCodec
        }
        try resources.nativePrefixLifetime.requirePreparation()
        try scope.requireImmutableLoadedOwner(resources)
        try sidecar.validate()
        guard sidecar.receipt.request.canonicalRoot == loadReceipt.binding.canonicalRoot,
            sidecar.receipt.request.mainConfigurationSHA256 == loadReceipt.binding.configSHA256,
            sidecar.codec.generation == sidecar.receipt.codecGeneration,
            sidecar.codec.sourceIdentity == sidecar.receipt.sourceIdentity
        else {
            throw MiMoV26AudioSidecarError.invalidBinding
        }
        // A loaded sidecar may survive an earlier successful setup epoch;
        // reauthorize exactly this still-valid owner in the current scope.
        try scope.retainOwner(sidecar)
        try scope.authorizeImmutableLoadedOwner(sidecar)
        try scope.invalidateOnFailedCompletion(sidecar) { sidecar.invalidate() }
        let processor = try makeMultimodalProcessor(
            binding: binding, limits: limits, audioCodec: sidecar.codec)
        try scope.retainOwner(processor)
        let issued = try binding.adapter.makeNativeDecodedAudioResources(
            bytesCapacity: bytesCapacity,
            processor: processor, loadedOwner: resources, audioOwner: sidecar,
            audioSessionID: sidecar.receipt.request.sessionID,
            audioSourceIdentity: sidecar.receipt.sourceIdentity,
            audioGeneration: sidecar.receipt.codecGeneration, retaining: scope)
        managedAudio = (processor, binding, issued.contract.id)
        return .init(
            backend: issued.backend, cacheProvider: issued.cacheProvider, contract: issued.contract)
    }

    /// One setup-only profile for text complete-prefix plus real decoded
    /// visual media. This never upgrades an already issued media/text-prefix
    /// profile, and media requests keep their existing noncacheable seals.
    public func makeManagedMediaCompletePrefixExecutionResources(
        binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, expectedMetadata: MiMoV26NativeCompletePrefixMetadata,
        completePrefixCache: any CBv2NativeCompletePrefixCache,
        processMemoryOwner: any CBv2ProcessMemoryOwner, limits: MiMoV26MultimodalLimits,
        retaining scope: NativeConstructionScope
    ) throws -> MiMoV26ManagedMediaExecutionResources {
        guard managedMedia == nil, managedAudio == nil else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try resources.nativePrefixLifetime.requirePreparation()
        try scope.requireImmutableLoadedOwner(resources)
        let processor = try makeMultimodalProcessor(binding: binding, limits: limits)
        try scope.retainOwner(processor)
        let validator = try beginNativeCompletePrefixIssuance(
            binding: binding,
            expectedMetadata: expectedMetadata, retaining: scope)
        do {
            let issued = try binding.adapter.makeNativeManagedCompletePrefixResources(
                bytesCapacity: bytesCapacity, processor: processor, loadedOwner: resources,
                validator: validator, completePrefixCache: completePrefixCache,
                processMemoryOwner: processMemoryOwner, retaining: scope)
            try finishNativeCompletePrefixIssuance(issued.contract)
            // All throwing validation precedes publishing the joint association.
            managedMedia = (processor, binding)
            return .init(
                backend: issued.backend, cacheProvider: issued.cacheProvider,
                contract: issued.contract)
        } catch {
            failNativeCompletePrefixIssuance()
            throw error
        }
    }

    /// Same joint ticket with the genuine separately loaded/admitted sidecar.
    /// No public codec, replacement target or second bank can be supplied.
    public func makeManagedAudioCompletePrefixExecutionResources(
        binding: MiMoV26CBv2Binding,
        bytesCapacity: Int, expectedMetadata: MiMoV26NativeCompletePrefixMetadata,
        completePrefixCache: any CBv2NativeCompletePrefixCache,
        processMemoryOwner: any CBv2ProcessMemoryOwner, limits: MiMoV26MultimodalLimits,
        retaining scope: NativeConstructionScope
    ) throws -> MiMoV26ManagedMediaExecutionResources {
        guard managedMedia == nil, managedAudio == nil, let sidecar = resources.audioSidecar else {
            throw MiMoV26MultimodalError.missingAudioCodec
        }
        try resources.nativePrefixLifetime.requirePreparation()
        try scope.requireImmutableLoadedOwner(resources)
        try sidecar.validate()
        guard sidecar.receipt.request.canonicalRoot == loadReceipt.binding.canonicalRoot,
            sidecar.receipt.request.mainConfigurationSHA256 == loadReceipt.binding.configSHA256,
            sidecar.codec.generation == sidecar.receipt.codecGeneration,
            sidecar.codec.sourceIdentity == sidecar.receipt.sourceIdentity
        else {
            throw MiMoV26AudioSidecarError.invalidBinding
        }
        try scope.retainOwner(sidecar)
        try scope.authorizeImmutableLoadedOwner(sidecar)
        try scope.invalidateOnFailedCompletion(sidecar) { sidecar.invalidate() }
        let processor = try makeMultimodalProcessor(
            binding: binding, limits: limits, audioCodec: sidecar.codec)
        try scope.retainOwner(processor)
        let validator = try beginNativeCompletePrefixIssuance(
            binding: binding,
            expectedMetadata: expectedMetadata, retaining: scope)
        do {
            let issued = try binding.adapter.makeNativeManagedCompletePrefixResources(
                bytesCapacity: bytesCapacity, processor: processor, loadedOwner: resources,
                validator: validator, completePrefixCache: completePrefixCache,
                processMemoryOwner: processMemoryOwner, audioOwner: sidecar,
                audioSessionID: sidecar.receipt.request.sessionID,
                audioSourceIdentity: sidecar.receipt.sourceIdentity,
                audioGeneration: sidecar.receipt.codecGeneration, retaining: scope)
            try sidecar.validate()
            try finishNativeCompletePrefixIssuance(issued.contract)
            managedAudio = (processor, binding, issued.contract.id)
            return .init(
                backend: issued.backend, cacheProvider: issued.cacheProvider,
                contract: issued.contract)
        } catch {
            failNativeCompletePrefixIssuance()
            throw error
        }
    }

    /// Decoded PCM (optionally mixed with ordered RGB/silent-video) uses the
    /// existing warm engine loan and one-shot seal, never construction epochs.
    public func prepareManagedDecodedAudioMedia(
        _ input: MiMoV26MultimodalInput, engine: EngineV2,
        authorize: (MiMoV26MultimodalPlan, Int) throws -> any MiMoV26MediaWorkReservation,
        retire: @escaping (any MiMoV26MediaWorkReservation) throws -> Void,
        isCancelled: () -> Bool
    ) throws -> CBv2Request {
        guard let audio = managedAudio, let sidecar = resources.audioSidecar,
            audio.processor.audioSidecar === sidecar,
            audio.processor.audioCodec === sidecar.codec,
            engine.nativeShutdownExecutionContractID == audio.contractID
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try sidecar.validate()
        return try engine.prepareNativeMedia(processor: audio.processor, loadedOwner: resources) {
            work in
            if isCancelled() { throw MiMoV26MultimodalError.cancelled }
            try sidecar.validate()
            work.retain(sidecar)
            let plan = try audio.processor.plan(input)
            guard !plan.spans.isEmpty else { throw MiMoV26MultimodalError.incompatiblePlan }
            try engine.requireNativeMediaCanEverFit(
                promptTokens: plan.promptTokens,
                maximumOutputTokens: plan.maximumOutputTokens)
            let bytes = try audio.processor.managedAudioCommitmentBytes(plan)
            let reservation = try authorize(plan, bytes)
            work.retain(reservation)
            work.afterRetirement { try retire(reservation) }
            let prepared = try audio.processor.prepareManagedAudio(
                plan, sidecar: sidecar,
                authorize: { actual in
                    try reservation.validate(plan: actual)
                    return reservation
                },
                isCancelled: isCancelled, work: work)
            work.retain(prepared)
            try work.validateAtBinding(prepared)
            return try prepared.makeRequest(binding: audio.binding, id: .init(0))
        }
    }

    /// Host calls only AFTER its real engine/bridge/consumer proof. The SDK
    /// receipt must belong to the single-use contract issued by this wrapper.
    /// Drops codec aliases only; separate sidecar ledger settlement stays host-
    /// owned and is NOT inferred from this method or a physical-free claim.
    public func releaseManagedAudioAfterNativeRetirement(_ receipt: CBv2NativeShutdownReceipt)
        throws
    {
        guard let audio = managedAudio, receipt.executionContractID == audio.contractID,
            receipt.generation == 1
        else { throw MiMoV26MultimodalError.incompatibleOwner }
        invalidateMultimodalPreparation()
        managedAudio = nil
        resources.audioSidecar = nil
        resources.audioInstallation = nil
    }

    /// Published-serving decoded RGB/silent-video path. Raw native arrays never
    /// cross the model/engine owner. The returned public closure refuses direct
    /// access; only the issuing engine can consume its one-shot sealed token.
    public func prepareManagedDecodedMedia(
        _ input: MiMoV26MultimodalInput, engine: EngineV2,
        authorize: (MiMoV26MultimodalPlan, Int) throws -> any MiMoV26MediaWorkReservation,
        retire: @escaping (any MiMoV26MediaWorkReservation) throws -> Void,
        isCancelled: () -> Bool
    ) throws -> CBv2Request {
        guard let media = managedMedia else { throw MiMoV26MultimodalError.incompatibleOwner }
        return try engine.prepareNativeMedia(processor: media.processor, loadedOwner: resources) {
            work in
            if isCancelled() { throw MiMoV26MultimodalError.cancelled }
            guard
                !input.messages.contains(where: { message in
                    message.content.contains {
                        switch $0 {
                        case .audio, .audiovisual: true
                        default: false
                        }
                    }
                })
            else { throw MiMoV26MultimodalError.missingAudioCodec }
            let plan = try media.processor.plan(input)
            guard !plan.spans.isEmpty else { throw MiMoV26MultimodalError.incompatiblePlan }
            try engine.requireNativeMediaCanEverFit(
                promptTokens: plan.promptTokens,
                maximumOutputTokens: plan.maximumOutputTokens)
            let bytes = try media.processor.managedVisualCommitmentBytes(plan)
            let reservation = try authorize(plan, bytes)
            // Installed before any pixel conversion/native work or later veto.
            work.retain(reservation)
            work.afterRetirement { try retire(reservation) }
            let prepared = try media.processor.prepareManaged(
                plan,
                authorize: { actual in
                    try reservation.validate(plan: actual)
                    return reservation
                }, isCancelled: isCancelled, work: work)
            work.retain(prepared)
            try work.validateAtBinding(prepared)
            return try prepared.makeRequest(binding: media.binding, id: .init(0))
        }
    }

    /// Call under the same serialized host owner before supported teardown or
    /// mutation. Already prepared/queued media refuses its next handoff. This
    /// does not replace draining an active engine before unloading weights.
    public func invalidateMultimodalPreparation() {
        resources.nativePagedLifetime.invalidate()
        resources.mediaGeneration.invalidate()
        resources.audioSidecar?.invalidate()
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        resources.nativePrefixLifetime.invalidate()
        invalidateMultimodalPreparation()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
    @discardableResult
    public override func update(
        modules: ModuleChildren, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        resources.nativePrefixLifetime.invalidate()
        invalidateMultimodalPreparation()
        return try super.update(
            modules: modules, verify: verify, path: path, modulePath: modulePath)
    }

    /// Generic iterator cache quantization, context limits and policy integration
    /// are not qualified by this factory slice. Reject through the standard
    /// recoverable boundary before the iterator attempts a model step.
    public func validateGenericGeneration() throws {
        throw MiMoV26FactoryError.nativeCBv2Required
    }
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard input.image == nil, input.video == nil else {
            throw MiMoV26FactoryError.unsupportedInput(
                "loaded text entrypoint rejects prepared media")
        }
        try validateGenericGeneration()
        throw MiMoV26FactoryError.nativeCBv2Required
    }
    public func newCache(parameters: GenerateParameters?) -> [KVCache] { target.newCache() }

    /// Direct native component diagnostic, with the target's real validation and
    /// arithmetic. Serving uses makeCBv2Binding, not a fabricated AR implementation.
    public func forwardText(inputIDs: MLXArray, cache: [KVCache]? = nil) throws -> MiMoV26TextOutput
    {
        try target.forward(inputIDs: inputIDs, cache: cache)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        do { return try forwardText(inputIDs: inputs, cache: cache).logits } catch {
            preconditionFailure(
                "MiMo native target contract bypassed: use throwing forwardText or native CBv2 (\(error))"
            )
        }
    }
}
