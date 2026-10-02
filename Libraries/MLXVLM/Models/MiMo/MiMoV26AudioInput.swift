// Copyright © 2026 Eigen Labs. One stateless request; no provider activation.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

public enum MiMoV26AudioWorkFailure: Error, Equatable, Sendable {
    case drainFailed, invalidatedOwner, noFailedWork
}

/// Host-owned physical reservation, distinct from numerical metadata. No
/// default permissive implementation. The source owner passed to encode must
/// retain the verified loaded codec and its actual load/source reservation.
public protocol MiMoV26AudioWorkReservation: AnyObject {
    func validate(plan: MiMoV26AudioInputPlan, sourceIdentity: String, generation: UUID) throws
    /// Transfer ALL failed work to host quarantine. Keep this bundle/lease
    /// until genuine recovery or process exit; never refund after failed drain.
    func retainAfterFailedDrain(_ work: MiMoV26FailedAudioWork)
}

/// Opaque, non-serialized failed work. Original array references, not copies or
/// nbytes stand-ins, retain the graph roots submitted at every explicit fence.
public final class MiMoV26FailedAudioWork {
    public let sourceIdentity: String
    public let generation: UUID
    public let inputPlanIdentity: Data
    public private(set) var failedDrain = false
    public var retainedRootCount: Int { roots.count }
    private let stream: MLX.Stream
    private var roots: [MLXArray] = []
    private var weights: MiMoV26AudioInputWeights?
    private var clips: [MiMoV26DecodedPCM]?
    private var sourceOwner: AnyObject?
    private var reservation: (any MiMoV26AudioWorkReservation)?
    private let managedWork: CBv2NativeMediaPreparation?
    private(set) var managedRequiredCompletionFailed = false
    var isManaged: Bool { managedWork != nil }
    init(
        weights: MiMoV26AudioInputWeights, clips: [MiMoV26DecodedPCM], plan: MiMoV26AudioInputPlan,
        sourceOwner: AnyObject, reservation: any MiMoV26AudioWorkReservation,
        managedWork: CBv2NativeMediaPreparation? = nil
    ) throws {
        self.weights = weights
        self.clips = clips
        self.sourceOwner = sourceOwner
        self.reservation = reservation
        self.managedWork = managedWork
        sourceIdentity = weights.sourceIdentity
        generation = weights.generation
        inputPlanIdentity = try plan.preparationIdentityData()
        stream = StreamOrDevice.default.stream
    }
    func track(_ values: [MLXArray]) { roots = values }
    func trackManaged(_ values: [MLXArray]) throws {
        roots = values
        try managedWork?.beforeNativeWork(values)  // actual inner roots before required ops
    }
    func markManagedRequiredFailure() {
        guard let managedWork else { return }
        managedWork.requiredCompletionFailed()  // first winner BEFORE cleanup/callback
        guard !managedRequiredCompletionFailed else { return }
        managedRequiredCompletionFailed = true
        reservation!.retainAfterFailedDrain(self)
    }
    func requiredCompletion<T>(_ body: () throws -> T) throws -> T {
        do { return try body() } catch {
            markManagedRequiredFailure()
            throw error
        }
    }
    func evaluateScratchCheckpoint(_ arrays: [MLXArray]) throws {
        if let managedWork {
            try managedWork.evaluateScratchCheckpoint(arrays)
        } else {
            try withError { eval(arrays) }
        }
    }
    func drain(invalidate: () -> Void, synchronize: (() throws -> Void)? = nil) throws {
        do {
            if let synchronize { try synchronize() } else { try withError { stream.synchronize() } }
        } catch {
            failedDrain = true
            invalidate()
            if managedWork != nil {
                markManagedRequiredFailure()
            } else {
                reservation!.retainAfterFailedDrain(self)
            }
            throw MiMoV26AudioWorkFailure.drainFailed
        }
    }
    func verifyRecoveryDrain(synchronize: (() throws -> Void)? = nil) throws {
        guard failedDrain else { throw MiMoV26AudioWorkFailure.noFailedWork }
        if let synchronize { try synchronize() } else { try withError { stream.synchronize() } }
    }
    func releaseAfterSuccessfulDrain() {
        guard !managedRequiredCompletionFailed, managedWork?.completionFailed != true else {
            return
        }
        roots.removeAll()
        clips = nil
        weights = nil
        sourceOwner = nil
        reservation = nil
        failedDrain = false
    }
    /// Caller keeps its native lane while recovering. A thrown synchronization
    /// preserves every owner. Success releases once; a repeated call refuses.
    public func recoverAndReleaseAfterDrain() throws {
        guard failedDrain, managedWork == nil else { throw MiMoV26AudioWorkFailure.noFailedWork }
        do { try verifyRecoveryDrain() } catch { throw MiMoV26AudioWorkFailure.drainFailed }
        releaseAfterSuccessfulDrain()
    }
}

/// PCM→codes composition. It never cross-batches calls or changes the original
/// plan to fit memory. The provider owns encoded decoding, real process permits, source
/// materialization/revalidation, the existing patch encoder and prompt assembly.
public final class MiMoV26AudioInput {
    public let weights: MiMoV26AudioInputWeights
    public var configuration: MiMoV26AudioInputConfiguration { weights.configuration }
    private var servingIsValid = true
    enum ManagedRequiredPhase: Equatable, Sendable {
        case frontendEval, frontendFinite, encoderEval, rvqEval, rvqFinite, codesReadback
    }
    // Per-instance refusal-only test hook; never substitutes evaluation/success.
    // Untracked encode never invokes it. Runs outside the outcome lock.
    #if DEBUG
        var beforeManagedRequiredCompletionForTesting: ((ManagedRequiredPhase) throws -> Void)?
    #endif
    public init(weights: MiMoV26AudioInputWeights) { self.weights = weights }

    /// Mandatory real work/source owners. Weights must already be materialized
    /// under the verified host load owner; this SDK does not authenticate it.
    /// A failed drain faults this input object and transfers roots/owners to
    /// host quarantine. Recovery does not silently reactivate this instance.
    public func encode(
        clips: [MiMoV26DecodedPCM], limits: MiMoV26AudioInputLimits,
        retaining sourceOwner: AnyObject,
        authorize: (MiMoV26AudioInputPlan) throws -> any MiMoV26AudioWorkReservation,
        isCancelled: () -> Bool = { false }
    ) throws -> [MiMoV26AudioCodeClip] {
        try encodeOwned(
            clips: clips, limits: limits, retaining: sourceOwner,
            authorize: authorize, isCancelled: isCancelled, managedWork: nil)
    }
    /// Trusted SDK producer uses its existing engine-issued loan; no public
    /// array callback, second loan/registry or successful-fence override.
    package func encodeManaged(
        clips: [MiMoV26DecodedPCM], limits: MiMoV26AudioInputLimits,
        retaining sourceOwner: AnyObject,
        authorize: (MiMoV26AudioInputPlan) throws -> any MiMoV26AudioWorkReservation,
        isCancelled: () -> Bool, work: CBv2NativeMediaPreparation
    ) throws -> [MiMoV26AudioCodeClip] {
        try encodeOwned(
            clips: clips, limits: limits, retaining: sourceOwner,
            authorize: authorize, isCancelled: isCancelled, managedWork: work)
    }
    private func encodeOwned(
        clips: [MiMoV26DecodedPCM], limits: MiMoV26AudioInputLimits,
        retaining sourceOwner: AnyObject,
        authorize: (MiMoV26AudioInputPlan) throws -> any MiMoV26AudioWorkReservation,
        isCancelled: () -> Bool, managedWork: CBv2NativeMediaPreparation?
    ) throws -> [MiMoV26AudioCodeClip] {
        guard servingIsValid else { throw MiMoV26AudioWorkFailure.invalidatedOwner }
        guard weights.encoder.loadedGeneration == weights.generation else {
            throw MiMoV26AudioInputError.weightsNotLoaded
        }
        let plan = try MiMoV26AudioInputPlan.make(
            clips: clips.map(\.descriptor), configuration: configuration, limits: limits)
        if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        guard !clips.isEmpty else { return [] }
        let owner = try authorize(plan)
        try owner.validate(
            plan: plan, sourceIdentity: weights.sourceIdentity, generation: weights.generation)
        let work = try MiMoV26FailedAudioWork(
            weights: weights, clips: clips, plan: plan, sourceOwner: sourceOwner,
            reservation: owner, managedWork: managedWork)
        managedWork?.retain(work)  // exact source/weights/PCM owner before graph creation
        do {
            let output = try withError { error in
                let result = try admitted(
                    clips: clips, plan: plan, isCancelled: isCancelled, owner: owner, work: work,
                    errors: error)
                try error.check()
                return result
            }
            try work.drain(invalidate: { servingIsValid = false })
            work.releaseAfterSuccessfulDrain()
            return output
        } catch {
            // Match outer managed classification BEFORE inner root cleanup.
            // Successfully drained typed cancellation/admission stay ordinary.
            if managedWork != nil, !(error is MiMoV26MultimodalError),
                !(error is CancellationError), error as? MiMoV26AudioInputError != .cancelled
            {
                work.markManagedRequiredFailure()
            }
            if work.managedRequiredCompletionFailed || managedWork?.completionFailed == true {
                servingIsValid = false
                work.markManagedRequiredFailure()
                throw MiMoV26AudioWorkFailure.drainFailed
            }
            if work.failedDrain { throw MiMoV26AudioWorkFailure.drainFailed }
            try work.drain(invalidate: { servingIsValid = false })
            work.releaseAfterSuccessfulDrain()
            throw error
        }
    }

    private func required<T>(
        _ phase: ManagedRequiredPhase, work: MiMoV26FailedAudioWork,
        _ body: () throws -> T
    ) throws -> T {
        try work.requiredCompletion {
            #if DEBUG
                if work.isManaged { try beforeManagedRequiredCompletionForTesting?(phase) }
            #endif
            return try body()
        }
    }

    private func admitted(
        clips: [MiMoV26DecodedPCM], plan: MiMoV26AudioInputPlan,
        isCancelled: () -> Bool, owner: any MiMoV26AudioWorkReservation,
        work: MiMoV26FailedAudioWork, errors: MLX.ErrorBox
    ) throws -> [MiMoV26AudioCodeClip] {
        let frontend = MiMoV26AudioFrontend(configuration: configuration)
        var mels: [MiMoV26PreparedMel] = []
        for (index, clip) in clips.enumerated() {
            mels.append(
                try frontend.prepare(
                    pcm: clip, plan: plan, clipIndex: index, isCancelled: isCancelled))
            try work.trackManaged(mels.map(\.values))
            try errors.check()
        }
        var melFinite = MLXArray(true)
        for mel in mels { melFinite = melFinite .&& all(isFinite(mel.values)) }
        try work.trackManaged(mels.map(\.values) + [melFinite])
        try errors.check()
        try owner.validate(
            plan: plan, sourceIdentity: weights.sourceIdentity, generation: weights.generation)
        // Explicit stage fence/readback, not a hidden property evaluation.
        try required(.frontendEval, work: work) {
            try withError { eval(mels.map(\.values) + [melFinite]) }
        }
        guard
            try required(
                .frontendFinite, work: work, { try withError { melFinite.item(Bool.self) } })
        else { throw MiMoV26AudioInputError.nonfiniteResult }
        if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        func checkpoint(_ roots: [MLXArray], phase: ManagedRequiredPhase) throws {
            try work.trackManaged(roots)
            try errors.check()
            try required(phase, work: work) { try work.evaluateScratchCheckpoint(roots) }
            if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        }
        let encoded = try weights.encoder.encodeFeaturesBounded(
            mels: mels, plan: plan, isCancelled: isCancelled,
            checkpoint: { try checkpoint($0, phase: .encoderEval) })
        try work.trackManaged([encoded.features])
        try errors.check()
        guard encoded.generation == weights.generation,
            encoded.inputPlanIdentity == (try plan.preparationIdentityData())
        else {
            throw MiMoV26AudioInputError.weightsNotLoaded
        }
        let quantized = try weights.quantizer.quantizeBounded(
            features: encoded.features, frameCounts: plan.codeFrameCounts,
            tileFrames: plan.limits.rvqTileFrames, isCancelled: isCancelled,
            checkpoint: { try checkpoint(mels.map(\.values) + $0, phase: .rvqEval) })
        try work.trackManaged([encoded.features, quantized.codes, quantized.allFinite])
        try errors.check()
        try owner.validate(
            plan: plan, sourceIdentity: weights.sourceIdentity, generation: weights.generation)
        try required(.rvqEval, work: work) {
            try withError { eval(quantized.codes, quantized.allFinite) }
        }
        guard
            try required(
                .rvqFinite, work: work, { try withError { quantized.allFinite.item(Bool.self) } })
        else { throw MiMoV26AudioInputError.nonfiniteResult }
        if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        let values = try required(.codesReadback, work: work) {
            try withError { quantized.codes.asArray(Int32.self) }
        }
        let expected = try MiMoV26AudioChecked.product(
            [plan.totalCodeFrames, configuration.quantizers], "host code count")
        guard values.count == expected else {
            throw MiMoV26AudioInputError.input("RVQ output accounting")
        }
        for (i, value) in values.enumerated() {
            guard value >= 0, Int(value) < configuration.codebookSizes[i % configuration.quantizers]
            else {
                throw MiMoV26AudioInputError.input("RVQ emitted an invalid channel code")
            }
        }
        var offset = 0
        return try plan.codeFrameCounts.map { frames in
            let count = try MiMoV26AudioChecked.product(
                [frames, configuration.quantizers], "clip code count")
            defer { offset += count }
            return MiMoV26AudioCodeClip(
                codes: Array(values[offset ..< (offset + count)]), frameCount: frames)
        }
    }
}
