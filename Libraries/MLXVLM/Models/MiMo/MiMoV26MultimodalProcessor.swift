// Copyright © 2026 Eigen Labs. Native owned decoded-media composition only.
import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Caller-supplied, already materialized codec plus its REAL load owner. Exact
/// metadata/generation checks are not source-byte authentication or admission:
/// the provider's verified load session remains responsible for those gates.
public final class MiMoV26OwnedAudioCodec {
    let input: MiMoV26AudioInput
    private let lifetimeOwner: AnyObject
    public let sourceIdentity: String
    public let generation: UUID
    public init(
        input: MiMoV26AudioInput, retaining owner: AnyObject,
        expectedSourceIdentity: String, expectedGeneration: UUID,
        mainConfiguration: MiMoV26Configuration
    ) throws {
        guard expectedSourceIdentity == MiMoV26AudioTokenizerWeights.selectedPayloadSHA256,
            input.weights.sourceIdentity == expectedSourceIdentity,
            input.weights.generation == expectedGeneration
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        let checked = try MiMoV26AudioInputConfiguration(
            sidecarJSON: input.configuration.encodedSidecar(),
            mainConfiguration: mainConfiguration)
        guard checked == input.configuration else { throw MiMoV26MultimodalError.incompatibleOwner }
        self.input = input
        lifetimeOwner = owner
        sourceIdentity = expectedSourceIdentity
        generation = expectedGeneration
        try validate()
    }
    func validate() throws {
        guard input.weights.encoder.loadedGeneration == generation,
            input.weights.generation == generation, input.weights.sourceIdentity == sourceIdentity
        else {
            throw MiMoV26MultimodalError.invalidatedOwner
        }
    }
}

/// No Module inheritance: identity metadata never registers a target twice.
/// Access is serialized with the loaded owner, just like native module access.
final class MiMoV26MediaGeneration {
    let identity = UUID()
    private(set) var isValid = true
    func invalidate() { isValid = false }
    func validate() throws {
        guard isValid else { throw MiMoV26MultimodalError.invalidatedOwner }
    }
}

/// Opaque failed-work ownership, NEVER serialized. Host quarantine owns this
/// box after a failed drain. It retains original root handles (no tensor
/// copies), decoded input, loaded resources, codec and the actual reservation.
/// No global SDK quarantine and no inference that a numeric byte credit owns
/// in-flight allocations. Recovery must run on the host's granted native lane.
public final class MiMoV26FailedMediaWork {
    public let preparationSHA256: String
    public let loadedOwnerIdentity: UUID
    public let configurationSHA256, templateSHA256: String
    public private(set) var failedDrain = false
    public var retainedRootCount: Int { roots.count }
    private let stream: MLX.Stream
    private var roots: [MLXArray] = []
    private var plan: MiMoV26MultimodalPlan?
    private var loadedOwner: AnyObject?
    private var codec: MiMoV26OwnedAudioCodec?
    private var audioSidecar: MiMoV26AudioSidecarLoaded?
    private var reservation: (any MiMoV26MediaWorkReservation)?
    private var audioFailures: [MiMoV26FailedAudioWork] = []
    private let managedWork: CBv2NativeMediaPreparation?
    var managedAudioContext: CBv2NativeMediaPreparation? { managedWork }
    var retainedAudioFailureRootCountsForTesting: [Int] { audioFailures.map(\.retainedRootCount) }
    init(
        plan: MiMoV26MultimodalPlan, owner: AnyObject, codec: MiMoV26OwnedAudioCodec?,
        reservation: any MiMoV26MediaWorkReservation,
        managedWork: CBv2NativeMediaPreparation? = nil,
        audioSidecar: MiMoV26AudioSidecarLoaded? = nil
    ) {
        preparationSHA256 = plan.preparationSHA256
        loadedOwnerIdentity = plan.loadedOwnerIdentity
        configurationSHA256 = plan.configurationSHA256
        templateSHA256 = plan.templateSHA256
        self.plan = plan
        loadedOwner = owner
        self.codec = codec
        self.reservation = reservation
        self.managedWork = managedWork
        self.audioSidecar = audioSidecar
        stream = StreamOrDevice.default.stream
    }
    func track(_ values: [MLXArray]) { roots = values }
    func trackManaged(_ values: [MLXArray]) throws {
        roots = values
        try managedWork?.beforeNativeWork(values)
    }
    /// Preserve already-evaluated visual roots while arming the existing loan
    /// BEFORE the inner audio frontend/encoder/RVQ can submit native work.
    func beginManagedAudioPhase() throws {
        try managedWork?.beforeNativeWork(roots)
    }
    func requiredAudioCompletionFailed() { managedWork?.requiredCompletionFailed() }
    func requiredNativeCompletion<T>(_ body: () throws -> T) throws -> T {
        do { return try body() } catch {
            managedWork?.requiredCompletionFailed()
            throw error
        }
    }
    func retainAudioFailure(_ audio: MiMoV26FailedAudioWork, generation: MiMoV26MediaGeneration) {
        managedWork?.requiredCompletionFailed()
        audioFailures.append(audio)
        generation.invalidate()
        if !failedDrain {
            failedDrain = true
            reservation!.retainAfterFailedDrain(self)
        }
    }
    func drain(generation: MiMoV26MediaGeneration, synchronize: (() throws -> Void)? = nil) throws {
        do {
            if let synchronize {
                try synchronize()
            } else if let managedWork {
                try managedWork.fencePreparation(stream)
            } else {
                try withError { stream.synchronize() }
            }
        } catch {
            managedWork?.requiredCompletionFailed()
            failedDrain = true
            generation.invalidate()
            reservation!.retainAfterFailedDrain(self)
            throw MiMoV26MultimodalError.drainFailed
        }
        try managedWork?.completedPreparation()
    }
    func releaseAfterSuccessfulDrain() {
        for audio in audioFailures { audio.releaseAfterSuccessfulDrain() }
        audioFailures.removeAll()
        roots.removeAll()
        plan = nil
        loadedOwner = nil
        codec = nil
        audioSidecar = nil
        reservation = nil
    }
    /// Failed synchronization leaves ALL owners/roots in place. Only an actual
    /// successful synchronization on the captured stream allows release.
    func verifyRecoveryDrain(synchronize: (() throws -> Void)? = nil) throws {
        guard failedDrain else { throw MiMoV26MultimodalError.incompatiblePlan }
        if let synchronize { try synchronize() } else { try withError { stream.synchronize() } }
        for audio in audioFailures { try audio.verifyRecoveryDrain() }
    }
    public func recoverAndReleaseAfterDrain() throws {
        guard failedDrain, managedWork == nil else { throw MiMoV26MultimodalError.incompatiblePlan }
        do { try verifyRecoveryDrain() } catch { throw MiMoV26MultimodalError.drainFailed }
        releaseAfterSuccessfulDrain()
        failedDrain = false
    }
}

/// Delegates the audio phase to the SAME real request lease. It introduces no
/// new admission credit, fake authentication or duplicate numerical pipeline.
private final class MiMoV26MediaAudioReservation: MiMoV26AudioWorkReservation {
    let plan: MiMoV26MultimodalPlan
    let audioPlan: MiMoV26AudioInputPlan
    let codec: MiMoV26OwnedAudioCodec
    let audioSidecar: MiMoV26AudioSidecarLoaded?
    let reservation: any MiMoV26MediaWorkReservation
    let work: MiMoV26FailedMediaWork
    let generation: MiMoV26MediaGeneration
    init(
        plan: MiMoV26MultimodalPlan, audioPlan: MiMoV26AudioInputPlan,
        codec: MiMoV26OwnedAudioCodec,
        reservation: any MiMoV26MediaWorkReservation, work: MiMoV26FailedMediaWork,
        generation: MiMoV26MediaGeneration, audioSidecar: MiMoV26AudioSidecarLoaded? = nil
    ) {
        self.plan = plan
        self.audioPlan = audioPlan
        self.codec = codec
        self.reservation = reservation
        self.work = work
        self.generation = generation
        self.audioSidecar = audioSidecar
    }
    func validate(plan: MiMoV26AudioInputPlan, sourceIdentity: String, generation: UUID) throws {
        guard plan == audioPlan, sourceIdentity == codec.sourceIdentity,
            generation == codec.generation
        else {
            throw MiMoV26MultimodalError.incompatiblePlan
        }
        try self.generation.validate()
        try codec.validate()
        if let audioSidecar {
            guard audioSidecar.codec === codec else {
                throw MiMoV26MultimodalError.incompatibleOwner
            }
            try audioSidecar.validate()
        }
        try reservation.validate(plan: self.plan)
    }
    func retainAfterFailedDrain(_ failed: MiMoV26FailedAudioWork) {
        work.retainAudioFailure(failed, generation: generation)
    }
}

/// Created by the loaded wrapper, not a generic UserInputProcessor. Retains
/// exact private components/source owners without exposing tower aliases.
public final class MiMoV26MultimodalProcessor {
    let configuration: MiMoV26Configuration
    let tokenizer: any Tokenizer
    let chatTemplate, templateSHA256: String
    let limits: MiMoV26MultimodalLimits
    let profile: MiMoV26MultimodalProfile
    let identity = UUID()
    let audioCodec: MiMoV26OwnedAudioCodec?
    let audioSidecar: MiMoV26AudioSidecarLoaded?
    private let vision: MiMoV26VisionTower
    private let audioPatch: MiMoV26AudioPatchEncoder
    private let adapter: MiMoV26CBv2Adapter
    private let stopTokens: Set<Int>
    private let generation: MiMoV26MediaGeneration
    private let loadedOwner: AnyObject
    let configurationSHA256: String
    var loadedOwnerIdentity: UUID { generation.identity }

    init(
        configuration: MiMoV26Configuration, tokenizer: any Tokenizer, chatTemplate: String,
        templateSHA256: String, limits: MiMoV26MultimodalLimits,
        vision: MiMoV26VisionTower, audioPatch: MiMoV26AudioPatchEncoder,
        adapter: MiMoV26CBv2Adapter, stopTokens: Set<Int>, generation: MiMoV26MediaGeneration,
        retaining owner: AnyObject, audioCodec: MiMoV26OwnedAudioCodec?,
        audioSidecar: MiMoV26AudioSidecarLoaded? = nil
    ) throws {
        guard adapter.target.configuration.rawFields == configuration.rawFields,
            adapter.supportsMultimodalPrefill(attention: .causal),
            configuration.vision == vision.configuration,
            configuration.audio == audioPatch.configuration,
            Self.hash(Data(chatTemplate.utf8)) == templateSHA256
        else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.chatTemplate = chatTemplate
        self.templateSHA256 = templateSHA256
        self.limits = limits
        profile = try .init(configuration: configuration, tokenizer: tokenizer)
        self.vision = vision
        self.audioPatch = audioPatch
        self.adapter = adapter
        self.stopTokens = stopTokens
        self.generation = generation
        loadedOwner = owner
        self.audioCodec = audioCodec
        self.audioSidecar = audioSidecar
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        configurationSHA256 = Self.hash(try encoder.encode(configuration.rawFields))
        if let audioCodec {
            let checked = try MiMoV26AudioInputConfiguration(
                sidecarJSON: audioCodec.input.configuration.encodedSidecar(),
                mainConfiguration: configuration)
            guard checked == audioCodec.input.configuration else {
                throw MiMoV26MultimodalError.incompatibleOwner
            }
        }
        try checkOwner()
    }
    static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func profileIdentity() -> String {
        let codecIdentity =
            audioCodec.map { $0.sourceIdentity + ":" + $0.generation.uuidString } ?? "no-codec"
        return [
            MiMoV26MultimodalProfile.name, generation.identity.uuidString,
            configurationSHA256, templateSHA256, codecIdentity,
        ].joined(separator: ":")
    }
    func checkOwner() throws {
        try generation.validate()
        try audioCodec?.validate()
        if let audioSidecar {
            guard audioSidecar.codec === audioCodec else {
                throw MiMoV26MultimodalError.incompatibleOwner
            }
            try audioSidecar.validate()  // retained FD, separate load charge and invalidation flag
        }
        guard adapter.supportsMultimodalPrefill(attention: .causal) else {
            throw MiMoV26MultimodalError.invalidatedOwner
        }
    }

    /// Must run on the real host's serialized model lane. The mandatory
    /// callback obtains admission BEFORE pixel/native work. No default permit,
    /// network, source loading or regrouping is hidden inside this method.
    public func prepare(
        _ plan: MiMoV26MultimodalPlan,
        authorize: (MiMoV26MultimodalPlan) throws -> any MiMoV26MediaWorkReservation,
        isCancelled: () -> Bool = { false }
    ) throws -> MiMoV26PreparedMultimodal {
        try prepareOwned(plan, authorize: authorize, isCancelled: isCancelled, managedWork: nil)
    }
    package func prepareManaged(
        _ plan: MiMoV26MultimodalPlan,
        authorize: (MiMoV26MultimodalPlan) throws -> any MiMoV26MediaWorkReservation,
        isCancelled: () -> Bool, work: CBv2NativeMediaPreparation
    ) throws -> MiMoV26PreparedMultimodal {
        guard plan.audioPlan == nil else { throw MiMoV26MultimodalError.missingAudioCodec }
        return try prepareOwned(
            plan, authorize: authorize, isCancelled: isCancelled, managedWork: work)
    }
    package func prepareManagedAudio(
        _ plan: MiMoV26MultimodalPlan, sidecar: MiMoV26AudioSidecarLoaded,
        authorize: (MiMoV26MultimodalPlan) throws -> any MiMoV26MediaWorkReservation,
        isCancelled: () -> Bool, work: CBv2NativeMediaPreparation
    ) throws -> MiMoV26PreparedMultimodal {
        guard audioSidecar === sidecar, audioCodec === sidecar.codec else {
            throw MiMoV26MultimodalError.incompatibleOwner
        }
        try sidecar.validate()
        return try prepareOwned(
            plan, authorize: authorize, isCancelled: isCancelled, managedWork: work)
    }
    private func prepareOwned(
        _ plan: MiMoV26MultimodalPlan,
        authorize: (MiMoV26MultimodalPlan) throws -> any MiMoV26MediaWorkReservation,
        isCancelled: () -> Bool, managedWork: CBv2NativeMediaPreparation?
    ) throws -> MiMoV26PreparedMultimodal {
        try checkOwner()
        guard plan.processorIdentity == identity else {
            throw MiMoV26MultimodalError.incompatiblePlan
        }
        if isCancelled() { throw MiMoV26MultimodalError.cancelled }
        let reservation = try authorize(plan)
        try reservation.validate(plan: plan)
        let work = MiMoV26FailedMediaWork(
            plan: plan, owner: loadedOwner, codec: audioCodec,
            reservation: reservation, managedWork: managedWork, audioSidecar: audioSidecar)
        managedWork?.retain(work)
        do {
            let arrays = try withError { error in
                let result = try admitted(
                    plan, isCancelled: isCancelled, reservation: reservation, work: work,
                    errors: error)
                try error.check()
                return result
            }
            try work.drain(generation: generation)
            try checkOwner()
            let result = MiMoV26PreparedMultimodal(
                plan: plan, arrays: arrays, adapter: adapter, stopTokens: stopTokens,
                generation: generation, retaining: loadedOwner, codec: audioCodec,
                reservation: reservation,
                audioSidecar: audioSidecar)
            work.releaseAfterSuccessfulDrain()
            return result
        } catch {
            if managedWork != nil, !(error is MiMoV26MultimodalError), !(error is CancellationError)
            {
                managedWork?.requiredCompletionFailed()
            }
            if work.failedDrain || managedWork?.completionFailed == true {
                reservation.retainAfterFailedDrain(work)
                throw MiMoV26MultimodalError.drainFailed
            }
            // Roots were recorded BEFORE each explicit evaluation/read, so
            // admitted's Swift unwinding cannot drop failed issued work.
            try work.drain(generation: generation)
            work.releaseAfterSuccessfulDrain()
            throw error
        }
    }

    private func admitted(
        _ plan: MiMoV26MultimodalPlan, isCancelled: () -> Bool,
        reservation: any MiMoV26MediaWorkReservation,
        work: MiMoV26FailedMediaWork, errors: MLX.ErrorBox
    ) throws -> [MLXArray] {
        var features: [Int: MLXArray] = [:]
        var audioFeatures: [Int: MLXArray] = [:]
        let dtype: DType = configuration.dtype == "bfloat16" ? .bfloat16 : .float32
        func evaluate(_ array: MLXArray, count: Int) throws {
            try errors.check()
            guard array.shape == [count, configuration.hiddenSize], array.dtype == dtype else {
                throw MiMoV26MultimodalError.invalidFeatures
            }
            let finite = all(isFinite(array))
            try work.trackManaged(
                Array(features.values) + Array(audioFeatures.values) + [array, finite])
            try errors.check()
            try work.requiredNativeCompletion { try withError { eval(array, finite) } }
            guard try work.requiredNativeCompletion({ try withError { finite.item(Bool.self) } })
            else {
                throw MiMoV26MultimodalError.invalidFeatures
            }
            if isCancelled() { throw MiMoV26MultimodalError.cancelled }
        }
        for (index, item) in plan.parts.enumerated() {
            if isCancelled() { throw MiMoV26MultimodalError.cancelled }
            try checkOwner()
            try reservation.validate(plan: plan)
            let pixels: MiMoV26Pixels.Prepared
            switch item.content {
            case .image(let frame):
                pixels = try MiMoV26Pixels.image(
                    frame, settings: profile.settings, limits: limits.pixels)
            case .silentVideo(let video):
                pixels = try MiMoV26Pixels.video(
                    frames: video.frames,
                    sampledFrameCount: video.frames.count, settings: profile.settings,
                    limits: limits.pixels)
            case .audiovisual(let av):
                pixels = try MiMoV26Pixels.video(
                    frames: av.frames,
                    sampledFrameCount: av.frames.count, settings: profile.settings,
                    limits: limits.pixels)
            case .audio: continue
            case .text: throw MiMoV26MultimodalError.incompatiblePlan
            }
            guard let geometry = item.geometry, pixels.geometry == geometry else {
                throw MiMoV26MultimodalError.incompatiblePlan
            }
            let patchArray = MLXArray(
                pixels.patchValues, [geometry.patchCount, geometry.patchVectorSize])
            try work.trackManaged(Array(features.values) + [patchArray])
            try errors.check()
            let feature = try vision.forwardBounded(
                patches: patchArray,
                grids: [
                    .init(temporal: geometry.gridT, height: geometry.gridH, width: geometry.gridW)
                ], limits: limits.vision,
                checkpoint: { roots in
                    // Root ownership precedes eval and every possible throw.
                    // A failed native completion retains the existing loan;
                    // cancellation after a successful eval uses normal drain.
                    try work.trackManaged(
                        Array(features.values) + Array(audioFeatures.values) + roots)
                    try errors.check()
                    try work.requiredNativeCompletion { try withError { eval(roots) } }
                    if isCancelled() { throw MiMoV26MultimodalError.cancelled }
                })
            try evaluate(feature, count: geometry.mediaTokens)
            features[index] = feature
        }
        if let audioPlan = plan.audioPlan {
            guard let codec = audioCodec else { throw MiMoV26MultimodalError.missingAudioCodec }
            try checkOwner()
            try codec.validate()
            try reservation.validate(plan: plan)
            let clips: [MiMoV26DecodedPCM] = plan.parts.compactMap {
                switch $0.content {
                case .audio(let clip): return clip
                case .audiovisual(let av): return av.wholeAudio
                default: return nil
                }
            }
            try work.beginManagedAudioPhase()
            let codes: [MiMoV26AudioCodeClip]
            do {
                let audioAuthorize:
                    (MiMoV26AudioInputPlan) throws -> any MiMoV26AudioWorkReservation = { actual in
                        guard actual == audioPlan else {
                            throw MiMoV26MultimodalError.incompatiblePlan
                        }
                        return MiMoV26MediaAudioReservation(
                            plan: plan, audioPlan: audioPlan, codec: codec,
                            reservation: reservation, work: work, generation: self.generation,
                            audioSidecar: self.audioSidecar)
                    }
                if let native = work.managedAudioContext {
                    codes = try codec.input.encodeManaged(
                        clips: clips, limits: limits.audio, retaining: codec,
                        authorize: audioAuthorize, isCancelled: isCancelled, work: native)
                } else {
                    codes = try codec.input.encode(
                        clips: clips, limits: limits.audio, retaining: codec,
                        authorize: audioAuthorize, isCancelled: isCancelled)
                }
            } catch MiMoV26AudioInputError.cancelled {
                // encode returns this only before submission or after its
                // real required drain succeeded. Outer preparation still owes
                // its own captured-stream completion before retirement.
                throw MiMoV26MultimodalError.cancelled
            } catch MiMoV26AudioWorkFailure.drainFailed {
                work.requiredAudioCompletionFailed()
                throw MiMoV26AudioWorkFailure.drainFailed
            }
            guard codes.map(\.frameCount) == audioPlan.codeFrameCounts else {
                throw MiMoV26MultimodalError.invalidFeatures
            }
            let output = try audioPatch.forward(clips: codes, limits: limits.audioPatch)
            try evaluate(output.features, count: audioPlan.totalPatches)
            guard output.clipPatchRanges.map(\.count) == audioPlan.patchCounts else {
                throw MiMoV26MultimodalError.invalidFeatures
            }
            for (index, item) in plan.parts.enumerated() {
                if let audioIndex = item.audioIndex {
                    audioFeatures[index] = output.features[output.clipPatchRanges[audioIndex], 0...]
                }
            }
        }
        let arrays = try plan.spans.map { span -> MLXArray in
            // Linked AV has visual AND audio arrays at the same mediaIndex.
            // The whole audio is encoded once, then sliced by final patch rows.
            let selected =
                span.kind == .audio ? audioFeatures[span.mediaIndex] : features[span.mediaIndex]
            guard let feature = selected, span.featureOffset >= 0,
                span.length > 0, span.featureOffset <= feature.dim(0) - span.length
            else {
                throw MiMoV26MultimodalError.invalidFeatures
            }
            return feature[span.featureOffset ..< (span.featureOffset + span.length), 0...]
        }
        try work.trackManaged(arrays)
        try errors.check()
        try work.requiredNativeCompletion { try withError { eval(arrays) } }
        if isCancelled() { throw MiMoV26MultimodalError.cancelled }
        return arrays
    }
}

/// One admitted request, with immutable evaluated features and a source/work
/// owner. No public array getters. The returned CBv2 request retains this box
/// until engine retirement; host must retire/drain the engine before teardown.
public final class MiMoV26PreparedMultimodal: CBv2NativeMediaBindingValidating {
    public let plan: MiMoV26MultimodalPlan
    private let arrays: [MLXArray]
    private let adapter: MiMoV26CBv2Adapter
    private let stopTokens: Set<Int>
    private let generation: MiMoV26MediaGeneration
    private let loadedOwner: AnyObject
    private let codec: MiMoV26OwnedAudioCodec?
    private let audioSidecar: MiMoV26AudioSidecarLoaded?
    private let reservation: any MiMoV26MediaWorkReservation
    private let lock = NSLock()
    private var issued = false, consumed = false
    init(
        plan: MiMoV26MultimodalPlan, arrays: [MLXArray], adapter: MiMoV26CBv2Adapter,
        stopTokens: Set<Int>, generation: MiMoV26MediaGeneration, retaining owner: AnyObject,
        codec: MiMoV26OwnedAudioCodec?, reservation: any MiMoV26MediaWorkReservation,
        audioSidecar: MiMoV26AudioSidecarLoaded? = nil
    ) {
        self.plan = plan
        self.arrays = arrays
        self.adapter = adapter
        self.stopTokens = stopTokens
        self.generation = generation
        loadedOwner = owner
        self.codec = codec
        self.reservation = reservation
        self.audioSidecar = audioSidecar
    }
    package func validateNativeMediaBinding() throws { try validate() }
    private func validate() throws {
        try generation.validate()
        try codec?.validate()
        try reservation.validate(plan: plan)
        if let audioSidecar {
            guard audioSidecar.codec === codec else {
                throw MiMoV26MultimodalError.incompatibleOwner
            }
            try audioSidecar.validate()  // again at actual engine bind, not just prepare
        }
        guard adapter.supportsMultimodalPrefill(attention: .causal) else {
            throw MiMoV26MultimodalError.invalidatedOwner
        }
    }
    /// Use the same binding to construct the engine. Hand-crafted raw requests
    /// submitted to another engine are outside this owned producer contract.
    public func makeRequest(
        binding: MiMoV26CBv2Binding, id: CBv2RequestID,
        sampling: CBv2SamplingParams = .init(), stopStrings: [String] = []
    ) throws -> CBv2Request {
        try validate()
        guard binding.adapter === adapter, binding.mediaGeneration === generation,
            binding.assistant === adapter.assistant
        else { throw MiMoV26MultimodalError.incompatibleOwner }
        lock.lock()
        defer { lock.unlock() }
        guard !issued else { throw MiMoV26MultimodalError.incompatiblePlan }
        issued = true
        return CBv2Request(
            id: id, promptTokens: plan.promptTokens, sampling: sampling,
            maxTokens: plan.maximumOutputTokens, stopTokens: stopTokens, stopStrings: stopStrings,
            prefixCacheEnabled: false,
            multimodal: .init(spans: plan.spans.map(\.engineSpan), attention: .causal) { [self] in
                try validate()
                lock.lock()
                defer { lock.unlock() }
                guard !consumed else { throw MiMoV26MultimodalError.incompatiblePlan }
                consumed = true
                return arrays
            })
    }
}
