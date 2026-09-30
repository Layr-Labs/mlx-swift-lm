import Foundation
import MLX

/// Additive: no requirement is added to generic CBv2Engine implementations.
public protocol CBv2NativeWorkShutdownReporting: Sendable {
    var nativeShutdownEngineID: UUID { get }
    var nativeShutdownExecutionContractID: UUID? { get }
    func shutdownReportingNativeCompletion() async -> CBv2NativeShutdownOutcome
}

public struct CBv2NativeShutdownReceipt: Sendable, Equatable {
    public let engineID: UUID
    public let generation: UInt64
    public let executionContractID: UUID
    public let capturedStreamCount: Int
    // No public initializer. This proves completion, NOT physical deallocation
    // or materialized-memory credit. Only the SDK barrier may issue it.
    init(engineID: UUID, generation: UInt64, executionContractID: UUID, capturedStreamCount: Int) {
        self.engineID = engineID
        self.generation = generation
        self.executionContractID = executionContractID
        self.capturedStreamCount = capturedStreamCount
    }
}

public struct CBv2NativeShutdownFault: Sendable, Equatable {
    public enum Reason: String, Sendable {
        case notTracked, unsupportedExecutionContract, shutdownTimedOut, stepWatchdog
        case capturedFenceFailed, nativeWorkFailed
    }
    public let engineID: UUID
    public let generation: UInt64
    public let reason: Reason
    init(engineID: UUID, generation: UInt64, reason: Reason) {
        self.engineID = engineID
        self.generation = generation
        self.reason = reason
    }
}

public enum CBv2NativeShutdownOutcome: Sendable, Equatable {
    case quiescent(CBv2NativeShutdownReceipt)
    case incomplete(CBv2NativeShutdownFault)
}

/// Minted only by the SDK's protected strict-loaded MiMo resource bundle. The
/// package producer must authorize the exact immutable, materialized loaded
/// owner and create the real backend/cache bank together. Not a generic marker
/// or a caller assertion about an arbitrary model/private stream.
public struct CBv2NativeExecutionContract: Sendable {
    public let id: UUID
    public let constructionOwnerID: UUID
    public let constructionEpoch: UInt64
    public let profile: String
    /// Bound to the same one-use model/cache/assistant ticket. This declares
    /// the permitted graph strategy, not a numerical certification.
    public let mtpVerificationMode: CBv2MTPVerificationMode
    public var supportsDecodedVisualMedia: Bool {
        profile == "mimo_decoded_visual_contiguous_default_and_cpu_v1"
            || profile == "mimo_complete_text_prefix_decoded_visual_contiguous_default_and_cpu_v1"
    }
    public var supportsDecodedAudioMedia: Bool {
        profile == "mimo_decoded_audio_contiguous_default_and_cpu_v1"
            || profile == "mimo_complete_text_prefix_decoded_audio_contiguous_default_and_cpu_v1"
    }
    public var supportsManagedDecodedMedia: Bool {
        supportsDecodedVisualMedia || supportsDecodedAudioMedia
    }
    public let audioSidecarSessionID: UUID?
    public let audioSidecarSourceIdentity: String?
    public let audioSidecarGeneration: UUID?
    public let supportsNativeCompletePrefix: Bool
    public let supportsNativePagedTarget: Bool
    /// Additional permission of the same exact page-bank ticket. It does not
    /// authorize rectangular verification or managed media. Joint prefix
    /// permission is separately validated against the same actual tuple.
    public var supportsNativePagedSerialMTP: Bool {
        supportsNativePagedTarget
            && (profile == "mimo_text_native_gathered_paged_serial_mtp_default_and_cpu_v1"
                || (supportsNativeCompletePrefix
                    && profile == "mimo_complete_text_prefix_native_gathered_paged_serial_mtp_v1"))
    }
    private let prefixIdentity: CBv2CompleteCheckpointIdentity?
    private let prefixAssistantCodecID: String?
    private let ticket: Ticket
    private final class Ticket: @unchecked Sendable {
        let lock = NSLock()
        weak var model: AnyObject?
        weak var backend: AnyObject?
        weak var cacheProvider: AnyObject?
        weak var assistant: AnyObject?
        weak var mediaProcessor: AnyObject?
        weak var loadedOwner: AnyObject?
        weak var audioOwner: AnyObject?
        weak var prefixStore: AnyObject?
        weak var prefixValidator: (any CBv2NativeCompletePrefixBindingValidating)?
        weak var prefixProcessOwner: AnyObject?
        let hasPrefixProcessOwner: Bool
        weak var pagedBinding: CBv2NativePagedModelBinding?
        weak var pagedProcessOwner: AnyObject?
        let hasAssistant: Bool
        var consumed = false
        init(
            model: AnyObject, backend: AnyObject, cacheProvider: AnyObject, assistant: AnyObject?,
            mediaProcessor: AnyObject?, loadedOwner: AnyObject?, audioOwner: AnyObject?,
            prefixStore: AnyObject?,
            prefixValidator: (any CBv2NativeCompletePrefixBindingValidating)?,
            prefixProcessOwner: AnyObject?, pagedBinding: CBv2NativePagedModelBinding?,
            pagedProcessOwner: AnyObject?
        ) {
            self.model = model
            self.backend = backend
            self.cacheProvider = cacheProvider
            self.assistant = assistant
            hasAssistant = assistant != nil
            self.mediaProcessor = mediaProcessor
            self.loadedOwner = loadedOwner
            self.audioOwner = audioOwner
            self.prefixStore = prefixStore
            self.prefixValidator = prefixValidator
            self.prefixProcessOwner = prefixProcessOwner
            hasPrefixProcessOwner = prefixProcessOwner != nil
            self.pagedBinding = pagedBinding
            self.pagedProcessOwner = pagedProcessOwner
        }
    }
    package init(
        model: AnyObject, backend: AnyObject, cacheProvider: AnyObject,
        assistant: AnyObject?, construction: NativeConstructionScope,
        mediaProcessor: AnyObject? = nil, loadedOwner: AnyObject? = nil,
        audioOwner: AnyObject? = nil, audioSessionID: UUID? = nil,
        audioSourceIdentity: String? = nil, audioGeneration: UUID? = nil,
        mtpVerificationMode: CBv2MTPVerificationMode = .serialTarget,
        completePrefixCache: (any CBv2NativeCompletePrefixCache)? = nil,
        completePrefixValidator: (any CBv2NativeCompletePrefixBindingValidating)? = nil,
        prefixProcessMemoryOwner: (any CBv2ProcessMemoryOwner)? = nil,
        nativePagedBinding: CBv2NativePagedModelBinding? = nil,
        nativePagedProcessMemoryOwner: (any CBv2ProcessMemoryOwner)? = nil
    ) throws {
        let origin = construction.snapshot
        guard case .active = origin.disposition, origin.epoch > 0 else {
            throw NativeConstructionError.inactiveScope
        }
        guard mtpVerificationMode == .serialTarget || mtpVerificationMode == .rectangular else {
            throw NativeConstructionError.inactiveScope
        }
        if mtpVerificationMode == .rectangular {
            // Only the protected package producer may issue this permission.
            // Match real loaded target identities and actual cache capability;
            // neither a requested enum nor a profile string is sufficient.
            guard let target = model as? any CBv2MTPSteppableModel,
                target.supportsRequestStatefulMTP,
                let targetID = target.mtpTargetIdentity,
                let drafter = assistant as? any CBv2MTPRequestStatefulDrafter,
                assistant is any CBv2NativeMTPCompletionSplitting,
                drafter.mtpTargetIdentity == targetID,
                drafter.requiredVerificationMode == .rectangular,
                drafter.maximumDraftTokens == 3, drafter.maximumSpeculativeBatch == 1,
                let bank = cacheProvider as? any CBv2LayerCacheProvider,
                bank.supportsMTPRectangularVerification
            else {
                throw NativeConstructionError.inactiveScope
            }
        }
        self.mtpVerificationMode = mtpVerificationMode
        constructionOwnerID = origin.ownerID
        constructionEpoch = origin.epoch
        id = UUID()
        let hasPrefix = completePrefixCache != nil
        let hasNativePaged = nativePagedBinding != nil
        guard hasNativePaged == (nativePagedProcessMemoryOwner != nil) else {
            throw NativeConstructionError.inactiveScope
        }
        if let nativePagedBinding {
            guard mediaProcessor == nil, audioOwner == nil,
                loadedOwner != nil, mtpVerificationMode == .serialTarget,
                let paged = backend as? PagedKVBackend,
                paged.nativeModelBinding === nativePagedBinding
            else {
                throw NativeConstructionError.inactiveScope
            }
            try nativePagedBinding.validate(
                model: model, backend: backend, bank: cacheProvider, assistant: assistant,
                processMemoryOwner: nativePagedProcessMemoryOwner,
                constructionOwnerID: origin.ownerID, constructionEpoch: origin.epoch)
            if let completePrefixCache {
                guard prefixProcessMemoryOwner === nativePagedProcessMemoryOwner else {
                    throw NativeConstructionError.inactiveScope
                }
                try nativePagedBinding.validateCompletePrefix(
                    store: completePrefixCache,
                    identity: completePrefixCache.identity, assistant: assistant,
                    processMemoryOwner: prefixProcessMemoryOwner)
            } else if nativePagedBinding.completePrefixIdentity != nil {
                throw NativeConstructionError.inactiveScope
            }
        }
        supportsNativePagedTarget = hasNativePaged
        guard hasPrefix == (completePrefixValidator != nil),
            hasPrefix || prefixProcessMemoryOwner == nil
        else {
            throw NativeConstructionError.inactiveScope
        }
        if let completePrefixCache, let completePrefixValidator {
            // Prefix authority is bound to the actual store/validator/owner.
            // Joint media authority is issued by the protected loaded producer,
            // never by upgrading a live ticket or copying a profile string.
            guard loadedOwner != nil,
                prefixProcessMemoryOwner != nil,
                completePrefixCache.identity.isValid,
                let target = model as? any CBv2HistoricalAttentionCheckpointProviding,
                target.cbv2SupportsHistoricalAttentionCheckpoint,
                let typed = model as? any CBv2CompleteCheckpointKVTypeProviding,
                let types = typed.cbv2CompleteCheckpointKVDTypes, !types.isEmpty,
                assistant == nil || assistant is any CBv2HistoricalMTPPrefixCheckpointCoding
            else {
                throw NativeConstructionError.inactiveScope
            }
            if mediaProcessor != nil {
                guard let mediaModel = model as? any CBv2MultimodalSteppableModel,
                    mediaModel.supportsMultimodalPrefill(attention: .causal),
                    let mediaBank = cacheProvider
                        as? any CBv2MultimodalAttentionCapabilityProviding,
                    mediaBank.supportsMultimodalPrefill(attention: .causal)
                else {
                    throw NativeConstructionError.inactiveScope
                }
            }
            try completePrefixValidator.validateNativeCompletePrefixBinding()
        }
        supportsNativeCompletePrefix = hasPrefix
        prefixIdentity = completePrefixCache?.identity
        prefixAssistantCodecID =
            (assistant as? any CBv2MTPPrefixCheckpointCoding)?.prefixCheckpointCodecID
        guard hasPrefix || hasNativePaged || (mediaProcessor == nil) == (loadedOwner == nil) else {
            throw NativeConstructionError.inactiveScope
        }
        if let loadedOwner { try construction.requireImmutableLoadedOwner(loadedOwner) }
        let hasAudio = audioOwner != nil
        guard hasAudio == (audioSessionID != nil), hasAudio == (audioSourceIdentity != nil),
            hasAudio == (audioGeneration != nil), !hasAudio || mediaProcessor != nil
        else {
            throw NativeConstructionError.inactiveScope
        }
        if let audioOwner, let source = audioSourceIdentity {
            try construction.requireImmutableLoadedOwner(audioOwner)
            guard source.utf8.count == 64,
                source.utf8.allSatisfy({ (48 ... 57).contains($0) || (97 ... 102).contains($0) })
            else {
                throw NativeConstructionError.inactiveScope
            }
        }
        audioSidecarSessionID = audioSessionID
        audioSidecarSourceIdentity = audioSourceIdentity
        audioSidecarGeneration = audioGeneration
        if hasNativePaged {
            if hasPrefix {
                profile =
                    assistant == nil
                    ? "mimo_complete_text_prefix_native_gathered_paged_v1"
                    : "mimo_complete_text_prefix_native_gathered_paged_serial_mtp_v1"
            } else {
                profile =
                    assistant == nil
                    ? "mimo_text_native_gathered_paged_default_and_cpu_v1"
                    : "mimo_text_native_gathered_paged_serial_mtp_default_and_cpu_v1"
            }
        } else if hasPrefix {
            profile =
                hasAudio
                ? "mimo_complete_text_prefix_decoded_audio_contiguous_default_and_cpu_v1"
                : mediaProcessor != nil
                    ? "mimo_complete_text_prefix_decoded_visual_contiguous_default_and_cpu_v1"
                    : "mimo_complete_text_prefix_contiguous_default_and_cpu_v1"
        } else {
            profile =
                hasAudio
                ? "mimo_decoded_audio_contiguous_default_and_cpu_v1"
                : (mediaProcessor == nil
                    ? "mimo_text_contiguous_default_and_cpu_v1"
                    : "mimo_decoded_visual_contiguous_default_and_cpu_v1")
        }
        ticket = Ticket(
            model: model, backend: backend, cacheProvider: cacheProvider, assistant: assistant,
            mediaProcessor: mediaProcessor, loadedOwner: loadedOwner, audioOwner: audioOwner,
            prefixStore: completePrefixCache, prefixValidator: completePrefixValidator,
            prefixProcessOwner: prefixProcessMemoryOwner, pagedBinding: nativePagedBinding,
            pagedProcessOwner: nativePagedProcessMemoryOwner)
    }
    func matchesMedia(processor: AnyObject, owner: AnyObject) -> Bool {
        ticket.lock.withLock {
            supportsManagedDecodedMedia && ticket.consumed && ticket.mediaProcessor === processor
                && ticket.loadedOwner === owner
                && (!supportsDecodedAudioMedia || ticket.audioOwner != nil)
        }
    }
    func consume(
        model: AnyObject, backend: AnyObject, cacheProvider: AnyObject, assistant: AnyObject?,
        mtpVerificationMode: CBv2MTPVerificationMode = .serialTarget,
        completePrefixCache: (any CBv2CompletePrefixCache)? = nil,
        processMemoryOwner: (any CBv2ProcessMemoryOwner)? = nil
    ) -> Bool {
        ticket.lock.withLock {
            // Weak liveness prevents a stale numeric ObjectIdentifier from
            // authorizing a different object after address reuse; no raw native
            // owner is transported or retained by this Sendable ticket.
            guard !ticket.consumed, self.mtpVerificationMode == mtpVerificationMode,
                ticket.model === model,
                ticket.backend === backend, ticket.cacheProvider === cacheProvider,
                ticket.hasAssistant == (assistant != nil), ticket.assistant === assistant,
                !supportsDecodedAudioMedia || ticket.audioOwner != nil
            else { return false }
            if mtpVerificationMode == .rectangular {
                guard let target = model as? any CBv2MTPSteppableModel,
                    target.supportsRequestStatefulMTP,
                    let targetID = target.mtpTargetIdentity,
                    let drafter = assistant as? any CBv2MTPRequestStatefulDrafter,
                    drafter.mtpTargetIdentity == targetID,
                    drafter.requiredVerificationMode == .rectangular,
                    (cacheProvider as? any CBv2LayerCacheProvider)?
                        .supportsMTPRectangularVerification == true
                else { return false }
            }
            if supportsNativeCompletePrefix {
                guard let completePrefixCache,
                    ticket.prefixStore === completePrefixCache,
                    prefixIdentity == completePrefixCache.identity,
                    ticket.loadedOwner != nil,
                    ticket.hasPrefixProcessOwner == (processMemoryOwner != nil),
                    ticket.prefixProcessOwner === processMemoryOwner,
                    let validator = ticket.prefixValidator
                else { return false }
                do { try validator.validateNativeCompletePrefixBinding() } catch { return false }
            }
            if supportsNativePagedTarget {
                guard completePrefixCache == nil || supportsNativeCompletePrefix,
                    ticket.loadedOwner != nil,
                    let binding = ticket.pagedBinding, ticket.pagedProcessOwner != nil,
                    ticket.pagedProcessOwner === processMemoryOwner
                else { return false }
                do {
                    try binding.validate(
                        model: model, backend: backend, bank: cacheProvider, assistant: assistant,
                        processMemoryOwner: processMemoryOwner,
                        constructionOwnerID: constructionOwnerID,
                        constructionEpoch: constructionEpoch)
                    if let completePrefixCache {
                        try binding.validateCompletePrefix(
                            store: completePrefixCache,
                            identity: completePrefixCache.identity, assistant: assistant,
                            processMemoryOwner: processMemoryOwner)
                    }
                } catch { return false }
            } else if !supportsNativeCompletePrefix
                && (completePrefixCache != nil || processMemoryOwner != nil)
            {
                return false
            }
            ticket.consumed = true
            return true
        }
    }

    /// Recheck immutable store/codec/loaded-generation binding when importing
    /// or publishing, not merely once at construction.
    func validateCompletePrefix(
        store: any CBv2CompletePrefixCache,
        codec: CBv2CompleteCheckpointCodec
    ) throws {
        try ticket.lock.withLock {
            guard supportsNativeCompletePrefix, ticket.consumed,
                ticket.prefixStore === store, prefixIdentity == store.identity,
                codec.identity == store.identity,
                codec.pagedConfig == nil
                    || (supportsNativePagedTarget && codec.isNativePagedHistorical),
                codec.recurrentSpec == nil, ticket.loadedOwner != nil,
                ticket.prefixProcessOwner != nil,
                let validator = ticket.prefixValidator,
                let typed = ticket.model as? any CBv2CompleteCheckpointKVTypeProviding,
                typed.cbv2CompleteCheckpointKVDTypes == codec.kvDTypes,
                ticket.assistant === codec.assistant.map({ $0 as AnyObject }),
                prefixAssistantCodecID == codec.assistant?.prefixCheckpointCodecID
            else {
                throw CBv2NativeShutdownError.unsupportedConsumer
            }
            try validator.validateNativeCompletePrefixBinding()
            if supportsNativePagedTarget {
                guard let binding = ticket.pagedBinding, codec.nativePagedBinding === binding,
                    ticket.prefixProcessOwner === ticket.pagedProcessOwner
                else {
                    throw CBv2NativeShutdownError.unsupportedConsumer
                }
                try binding.validateCompletePrefix(
                    store: store, identity: codec.identity,
                    assistant: codec.assistant.map { $0 as AnyObject },
                    processMemoryOwner: ticket.pagedProcessOwner as? any CBv2ProcessMemoryOwner)
            }
        }
    }
}

enum CBv2NativeShutdownError: Error { case operationClosed, unsupportedConsumer }

/// Only package-owned prepared media can supply this non-array handoff check.
/// It reuses the producer's generation/codec/reservation checks at actual bind.
package protocol CBv2NativeMediaBindingValidating: AnyObject {
    func validateNativeMediaBinding() throws
}

/// Opaque one-shot seal, not an admission assertion supplied by a caller.
/// Its native storage is accessed only on the owning engine queue.
public final class CBv2PreparedNativeMediaToken: @unchecked Sendable {
    public let id = UUID()
    let work: CBv2NativeMediaPreparation
    let prompt: [Int]
    let maximumOutputTokens: Int
    // Engine queue only. The actual stream object is the generation identity.
    var stream: CBv2OutputStream?
    var disposed = false
    init(work: CBv2NativeMediaPreparation, request: CBv2Request) {
        self.work = work
        prompt = request.promptTokens
        maximumOutputTokens = request.maxTokens
    }
    func matches(_ request: CBv2Request, engineID: UUID) -> Bool {
        !disposed && work.engineID == engineID && prompt == request.promptTokens
            && maximumOutputTokens == request.maxTokens && request.maxTokens > 0
            && request.multimodal?.nativeMediaToken === self
            && request.positionState == nil && request.multimodal?.positionState == nil
            && request.multimodal?.attention == .causal
    }
}

/// Package-local producer context, acquired before pixel/native work. This
/// reuses the engine's loan/root/first-winner state; it is not construction.
package final class CBv2NativeMediaPreparation: @unchecked Sendable {
    // Engine queue/loan guarantees liveness. A caller retaining a disposed
    // token must not retain this engine's unrelated roots or stream consumers.
    unowned let tracking: CBv2NativeShutdownState
    let engineID: UUID
    let generation: UInt64
    let executionContractID: UUID?
    let loan: UUID
    var rootIDs: [UInt64] = []
    private var scratchRootIDs: [UInt64] = []
    var owners: [AnyObject] = []
    var resolved: CBv2ResolvedMultimodal?
    var preparationCompleted = false
    var didStartNativeWork = false
    var onRequiredFailure: (() -> Void)?
    private var retirement: (() throws -> Void)?
    private var bindingValidator: (any CBv2NativeMediaBindingValidating)?
    init(tracking: CBv2NativeShutdownState, loan: UUID) {
        self.tracking = tracking
        engineID = tracking.engineID
        self.loan = loan
        generation = tracking.generation
        executionContractID = tracking.contractID
    }
    package func retain(_ owner: AnyObject) { owners.append(owner) }
    package func validateAtBinding(_ validator: any CBv2NativeMediaBindingValidating) throws {
        try tracking.requireWork()
        guard bindingValidator == nil else { throw CBv2NativeShutdownError.unsupportedConsumer }
        bindingValidator = validator
    }
    func validateBinding() throws {
        guard let bindingValidator else { throw CBv2NativeShutdownError.unsupportedConsumer }
        try bindingValidator.validateNativeMediaBinding()
    }
    package func beforeNativeWork(_ arrays: [MLXArray]) throws {
        try tracking.requireWork()
        tracking.captureStreams()
        let id = tracking.retain(arrays, owners: [])
        rootIDs.append(id)
        scratchRootIDs.append(id)
        didStartNativeWork = true
    }
    /// The current producer roots own every still-live input/output. Evaluate
    /// them synchronously before retiring older per-stage array registrations.
    /// The preparation owner, reservation and loan remain until final handoff
    /// retirement. Failed evaluation/first-winner state never releases roots.
    package func evaluateScratchCheckpoint(_ arrays: [MLXArray]) throws {
        try tracking.requireWork()
        try withError { eval(arrays) }
        try tracking.requireWork()
        tracking.retireCompleted(scratchRootIDs)
        try tracking.requireWork()
        let retired = Set(scratchRootIDs)
        rootIDs.removeAll { retired.contains($0) }
        scratchRootIDs.removeAll()
    }
    package func requiredCompletionFailed() {
        if let onRequiredFailure {
            onRequiredFailure()
        } else {
            _ = tracking.fail(.nativeWorkFailed)
        }
    }
    package func fencePreparation(_ stream: MLX.Stream) throws {
        do {
            try tracking.beforeFenceForTesting?(stream)
            try withError { error in
                stream.synchronize()
                try error.check()
            }
        } catch {
            requiredCompletionFailed()
            throw error
        }
    }
    package var completionFailed: Bool { tracking.isIncomplete }
    package func completedPreparation() throws {
        try tracking.requireWork()
        preparationCompleted = true
    }
    /// The reviewed host callback only retires its existing reservation; it
    /// must not perform native work or wait. Invoked outside outcome locks.
    package func afterRetirement(_ body: @escaping () throws -> Void) { retirement = body }
    // The short first-winner commit removes aliases, then a following queue
    // callback runs accounting outside that lock while the loan stays armed.
    func detachAfterCompletion() -> (() throws -> Void)? {
        guard tracking.mayExecute else { return nil }
        resolved = nil
        bindingValidator = nil
        owners.removeAll()
        tracking.retireCompleted(rootIDs)
        rootIDs.removeAll()
        scratchRootIDs.removeAll()
        let result = retirement
        retirement = nil
        return result
    }
}

/// Only the reviewed native assistant implements this package-only seam. Its
/// existing construction-stream fence is outside the outcome lock; replacement
/// of measured owners is a separate non-evaluating, non-waiting commit.
package protocol CBv2NativeMTPCompletionSplitting: AnyObject {
    func fenceRequestStateForNativeCompletion(_ state: any CBv2MTPRequestState) throws
    func commitRequestStateNativeCompletion(_ state: any CBv2MTPRequestState) throws
}

/// The status/first-winner mirror is locked; MLX roots and actual streams are
/// confined to the one engine queue. No native fence is performed under lock.
/// On incomplete, the strong engine reference intentionally becomes a permanent
/// restart-only retention cycle. There is no retry/reset/deinit-refund API.
final class CBv2NativeShutdownState: @unchecked Sendable {
    let engineID: UUID
    let generation: UInt64 = 1
    let contractID: UUID?
    let supported: Bool
    private let lock = NSRecursiveLock()
    private weak var engine: EngineV2?
    private var retainedFaultEngine: EngineV2?
    private var result: CBv2NativeShutdownOutcome?
    private var waiters: [CheckedContinuation<CBv2NativeShutdownOutcome, Never>] = []
    private var requested = false
    private var loans: [UUID: AnyObject?] = [:]
    private var capturedStreamCount = 0
    private var capturedRootCount = 0
    // Engine queue only. Array aliases never cross the status mirror.
    private var streams: [MLX.Stream] = []
    private var roots: [UInt64: ([MLXArray], [AnyObject])] = [:]
    private var nextRoot: UInt64 = 0
    // Refusal/hold hooks only, never a replacement for a successful fence.
    var beforeFenceForTesting: ((MLX.Stream) throws -> Void)?
    var afterSubmissionForTesting: (() -> Void)?
    var afterAssistantFenceForTesting: ((any CBv2MTPRequestState) -> Void)?

    init(engineID: UUID, contractID: UUID?, supported: Bool) {
        self.engineID = engineID
        self.contractID = contractID
        self.supported = supported
    }
    func installEngine(_ value: EngineV2) {
        lock.withLock { engine = value }
        if !supported { _ = fail(.unsupportedExecutionContract) }
    }
    var outcome: CBv2NativeShutdownOutcome? { lock.withLock { result } }
    var isIncomplete: Bool {
        lock.withLock {
            if case .incomplete? = result { return true }
            return false
        }
    }
    var mayExecute: Bool { lock.withLock { supported && result == nil } }
    var hasLoans: Bool { lock.withLock { !loans.isEmpty } }
    /// Private pool lifetime only. Missing/wrong UUID never authorizes teardown.
    func hasOnlyNativeLoan(_ id: UUID) -> Bool {
        lock.withLock { supported && result == nil && loans.count == 1 && loans[id] != nil }
    }
    var debugRetainedRootCount: Int { lock.withLock { capturedRootCount } }

    func beginLoan(owner: AnyObject? = nil, duringDrain: Bool = false) throws -> UUID {
        try lock.withLock {
            guard supported, result == nil, duringDrain || !requested else {
                throw CBv2NativeShutdownError.operationClosed
            }
            let id = UUID()
            loans[id] = .some(owner)
            return id
        }
    }
    func endLoan(_ id: UUID) {
        lock.withLock {
            // A late caller cannot release the only private diagnostic owner
            // after incomplete won. Its loan/owner remains part of the fault.
            guard result == nil else { return }
            loans.removeValue(forKey: id)
        }
    }

    /// Returns true for the one caller that must enqueue/schedule the drain.
    func register(_ waiter: CheckedContinuation<CBv2NativeShutdownOutcome, Never>) -> Bool {
        let answer: (CBv2NativeShutdownOutcome?, Bool) = lock.withLock {
            if let result { return (result, false) }
            waiters.append(waiter)
            let first = !requested
            requested = true
            return (nil, first)
        }
        if let result = answer.0 { waiter.resume(returning: result) }
        return answer.1
    }

    @discardableResult
    func fail(_ reason: CBv2NativeShutdownFault.Reason) -> Bool {
        let won:
            (CBv2NativeShutdownOutcome, [CheckedContinuation<CBv2NativeShutdownOutcome, Never>])? =
                lock.withLock {
                    guard result == nil else { return nil }
                    let value = CBv2NativeShutdownOutcome.incomplete(
                        .init(engineID: engineID, generation: generation, reason: reason))
                    retainedFaultEngine = engine
                    result = value
                    let parked = waiters
                    waiters = []
                    return (value, parked)
                }
        guard let won else { return false }
        for waiter in won.1 { waiter.resume(returning: won.0) }
        return true
    }

    /// Short metadata/ownership commit only. Never put eval/readback/fences or
    /// blocking hooks in this closure. Recursive solely for nested SDK cleanup.
    @discardableResult
    func commitIfHealthy(_ body: () -> Void) -> Bool {
        return lock.withLock {
            guard supported, result == nil else { return false }
            body()
            return true
        }
    }
    func beginCommit() -> Bool {
        lock.lock()
        guard supported, result == nil else {
            lock.unlock()
            return false
        }
        return true
    }
    func endCommit() { lock.unlock() }
    func commitIfHealthyThrowing(_ body: () throws -> Void) throws -> Bool {
        return try lock.withLock {
            guard supported, result == nil else { return false }
            try body()
            return true
        }
    }

    func requireWork() throws {
        guard mayExecute else { throw CBv2NativeShutdownError.operationClosed }
    }
    func captureStreams() {
        // Called on the engine queue BEFORE creating/submitting native work.
        for value in [StreamOrDevice.cpu.stream, StreamOrDevice.default.stream]
        where !streams.contains(value) {
            streams.append(value)
        }
        lock.withLock { capturedStreamCount = streams.count }
    }
    /// Engine queue only: an external prefix operation already fenced these
    /// actual streams before its completion callback reached this queue.
    func captureCompletedPrefixStreams(_ completed: [MLX.Stream]) {
        for value in completed where !streams.contains(value) { streams.append(value) }
        lock.withLock { capturedStreamCount = streams.count }
    }
    var streamsForPrefixWork: [MLX.Stream] { streams }
    var rootMark: UInt64 { nextRoot }
    @discardableResult
    func retain(_ arrays: [MLXArray] = [], owners: [AnyObject] = []) -> UInt64 {
        precondition(nextRoot < UInt64.max, "native root generation exhausted")
        nextRoot += 1
        roots[nextRoot] = (arrays, owners)
        lock.withLock { capturedRootCount = roots.count }
        return nextRoot
    }
    func rootIDs(since mark: UInt64) -> [UInt64] { roots.keys.filter { $0 > mark } }
    func retireCompleted(_ ids: [UInt64]) {
        // Completion authority is the existing successful step/future readback,
        // not a global memory delta or a new blanket per-step stream fence.
        _ = commitIfHealthy {
            for id in ids { roots.removeValue(forKey: id) }
            capturedRootCount = roots.count
        }
    }
    func didSubmit() throws {
        afterSubmissionForTesting?()
        try requireWork()
    }
    func fenceCapturedStreams() throws {
        // Engine queue only, outside the first-winner lock. Any failure retains
        // all roots. A watchdog can win while this actual fence is blocked.
        var firstFailure: Error?
        for stream in streams {
            do {
                try beforeFenceForTesting?(stream)
                try withError { error in
                    stream.synchronize()
                    try error.check()
                }
            } catch { if firstFailure == nil { firstFailure = error } }
        }
        if let firstFailure { throw firstFailure }
    }

    func completeQuiescent(_ cleanup: () -> Void) {
        let won:
            (CBv2NativeShutdownOutcome, [CheckedContinuation<CBv2NativeShutdownOutcome, Never>])? =
                lock.withLock {
                    guard supported, requested, result == nil, loans.isEmpty, let contractID else {
                        return nil
                    }
                    cleanup()  // only post-fence SDK state/ownership cleanup
                    roots.removeAll()
                    capturedRootCount = 0
                    let value = CBv2NativeShutdownOutcome.quiescent(
                        .init(
                            engineID: engineID, generation: generation,
                            executionContractID: contractID,
                            capturedStreamCount: capturedStreamCount))
                    result = value
                    let parked = waiters
                    waiters = []
                    return (value, parked)
                }
        if let won { for waiter in won.1 { waiter.resume(returning: won.0) } }
    }
}
