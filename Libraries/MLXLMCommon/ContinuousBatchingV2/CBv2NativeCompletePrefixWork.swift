// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// Package producer checks only sealed source/session/generation metadata.
/// Never a public caller assertion, array read, evaluator or user callback.
package protocol CBv2NativeCompletePrefixBindingValidating: AnyObject {
    func validateNativeCompletePrefixBinding() throws
}

enum CBv2NativeCompletePrefixPurpose: Equatable { case importing, publication, discard }

/// A real operation held by the engine's existing native loan. Roots may be
/// registered from the serial provider/capture queue; they never enter the
/// engine-queue-only native root dictionary. No deinit refund is permitted.
final class CBv2NativeCompletePrefixWork: @unchecked Sendable {
    let id = UUID()
    let engineID: UUID
    let executionContractID: UUID
    let codecIdentity: CBv2CompleteCheckpointIdentity
    let purpose: CBv2NativeCompletePrefixPurpose
    private weak var codecOwner: CBv2CompleteCheckpointCodec?
    private weak var storeOwner: (any CBv2NativeCompletePrefixCache)?
    private var request: CBv2Request?
    private let tracking: CBv2NativeShutdownState
    private let queue: DispatchQueue
    private let completionQueue: DispatchQueue
    private let wake: @Sendable () -> Void
    private let failure: @Sendable () -> Void
    private var validateBinding: (@Sendable () throws -> Void)?
    private let lock = NSLock()
    private var loan: UUID?
    private var streams: [MLX.Stream]
    private var arrays: [MLXArray] = []
    private var owners: [AnyObject] = []
    private var retirement: (() throws -> Void)?
    private var finishing = false
    private var finished = false
    private var failed = false
    private var promotionFenceCompleted = false
    private let beforeFence: ((MLX.Stream) throws -> Void)?

    init(
        tracking: CBv2NativeShutdownState, codec: CBv2CompleteCheckpointCodec,
        store: any CBv2NativeCompletePrefixCache, request: CBv2Request?,
        purpose: CBv2NativeCompletePrefixPurpose,
        streams: [MLX.Stream], duringDrain: Bool, queue: DispatchQueue,
        completionQueue: DispatchQueue, wake: @escaping @Sendable () -> Void,
        failure: @escaping @Sendable () -> Void,
        validateBinding: @escaping @Sendable () throws -> Void
    ) throws {
        guard let contractID = tracking.contractID else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        self.tracking = tracking
        self.codecOwner = codec
        self.storeOwner = store
        self.request = request
        codecIdentity = codec.identity
        self.purpose = purpose
        owners = [codec, store]
        engineID = tracking.engineID
        executionContractID = contractID
        self.streams = []
        for stream in streams where !self.streams.contains(stream) { self.streams.append(stream) }
        self.queue = queue
        self.completionQueue = completionQueue
        self.wake = wake
        self.failure = failure
        self.validateBinding = validateBinding
        beforeFence = tracking.beforeFenceForTesting
        // All stored properties exist before self is loan-owned.
        loan = try tracking.beginLoan(owner: self, duringDrain: duringDrain)
    }

    /// Root's import plan/stage must check this before touching/consuming its
    /// private native state; reused request IDs alone are not an authority.
    func validate(
        store: any CBv2CompletePrefixCache, codec: CBv2CompleteCheckpointCodec,
        request: CBv2Request, engineID: UUID
    ) throws {
        try tracking.requireWork()
        guard let validate = lock.withLock({ validateBinding }) else {
            throw CBv2NativeShutdownError.operationClosed
        }
        try validate()
        let state = lock.withLock { (self.request, finishing || finished || failed) }
        guard self.storeOwner === store, self.codecOwner === codec, self.engineID == engineID,
            self.executionContractID == tracking.contractID,
            let original = state.0, original.id == request.id,
            original.prefixCacheReceiptID == request.prefixCacheReceiptID,
            original.promptTokens == request.promptTokens,
            original.checkpointCacheSalt == request.checkpointCacheSalt,
            original.maxTokens == request.maxTokens,
            original.multimodal == nil, request.multimodal == nil,
            original.positionState == nil, request.positionState == nil,
            !state.1
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
    }

    func captureCurrentStreams() throws {
        try tracking.requireWork()
        do {
            guard let validate = lock.withLock({ validateBinding }) else {
                throw CBv2NativeShutdownError.operationClosed
            }
            try validate()
        } catch {
            requiredCompletionFailed()
            throw error
        }
        let actual = [StreamOrDevice.cpu.stream, StreamOrDevice.default.stream]
        let invalid = lock.withLock { () -> Bool in
            for stream in actual where !streams.contains(stream) { streams.append(stream) }
            promotionFenceCompleted = false
            return finishing || finished || failed
        }
        if invalid {
            requiredCompletionFailed()
            throw CBv2NativeShutdownError.operationClosed
        }
        try tracking.requireWork()
    }

    /// Keep late returned actual roots even if a concurrent watchdog won.
    /// The caller registers each fresh array/reservation BEFORE evaluation.
    func retain(arrays incoming: [MLXArray] = [], owners incomingOwners: [AnyObject] = []) throws {
        let invalid = lock.withLock { () -> Bool in
            for array in incoming where !arrays.contains(where: { $0 === array }) {
                arrays.append(array)
            }
            for owner in incomingOwners where !owners.contains(where: { $0 === owner }) {
                owners.append(owner)
            }
            promotionFenceCompleted = false
            return finishing || finished || failed
        }
        if invalid {
            requiredCompletionFailed()
            throw CBv2NativeShutdownError.operationClosed
        }
        try tracking.requireWork()
        do {
            guard let validate = lock.withLock({ validateBinding }) else {
                throw CBv2NativeShutdownError.operationClosed
            }
            try validate()
        } catch {
            requiredCompletionFailed()
            throw error
        }
    }

    /// Called only before the engine enters its metadata commit. Executes the
    /// real required stream fence without ending the operation or dropping C.
    func fenceForProtectedPromotion() throws {
        try tracking.requireWork()
        guard !lock.withLock({ finishing || finished || failed }) else {
            throw CBv2NativeShutdownError.operationClosed
        }
        let captured = lock.withLock { streams }
        var firstError: Error?
        for stream in captured {
            do {
                try beforeFence?(stream)
                try withError { errors in
                    stream.synchronize()
                    try errors.check()
                }
            } catch { if firstError == nil { firstError = error } }
        }
        if let firstError {
            requiredCompletionFailed()
            throw firstError
        }
        try tracking.requireWork()
        try lock.withLock {
            guard !failed, !finishing, !finished else {
                throw CBv2NativeShutdownError.operationClosed
            }
            promotionFenceCompleted = true
        }
    }

    var hasProtectedPromotionCompletion: Bool {
        lock.withLock { promotionFenceCompleted && !failed && !finishing && !finished }
    }

    /// Derived from the real sealed codec/binding, never a caller flag or
    /// manifest layout string. Contiguous/foreign work keeps refusing pages.
    func requireNativePagedSources() throws {
        try tracking.requireWork()
        guard purpose == .publication, let codec = codecOwner, codec.isNativePagedHistorical,
            let store = storeOwner, let binding = codec.nativePagedBinding
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        guard
            binding.validatesCompletePrefixCodec(
                identity: codec.identity,
                layerKinds: codec.layerKinds, layerDTypes: codec.kvDTypes,
                assistant: codec.assistant.map { $0 as AnyObject }),
            store.identity == codecIdentity
        else {
            requiredCompletionFailed()
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
    }

    /// The real imported assistant was evaluated/fenced off to the side.
    /// This performs only its non-waiting ownership measurement under the
    /// existing first-winner commit; it does not adopt a request or grant C.
    func commitPreparedAssistant(
        _ assistant: any CBv2NativeMTPCompletionSplitting,
        state: any CBv2MTPRequestState
    ) throws {
        guard purpose == .importing, hasProtectedPromotionCompletion,
            codecOwner?.assistant.map({ $0 as AnyObject }) === assistant as AnyObject
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        guard
            try tracking.commitIfHealthyThrowing({
                try assistant.commitRequestStateNativeCompletion(state)
            })
        else { throw CBv2NativeShutdownError.operationClosed }
    }

    /// Keep export scratch bounded to the live readback segment. Only the
    /// publication owner can use this, after copying bytes out of native
    /// storage. Executes a real fence; no caller-supplied completion boolean.
    /// Donor owners and the WHOLE-operation reservation/loan stay held.
    func retireReadbackTemporaries(_ temporary: [MLXArray]) throws {
        guard purpose == .publication else { throw CBv2NativeShutdownError.unsupportedConsumer }
        try fenceForProtectedPromotion()
        guard
            tracking.commitIfHealthy({
                lock.withLock {
                    arrays.removeAll { array in temporary.contains(where: { $0 === array }) }
                    promotionFenceCompleted = false
                }
            })
        else { throw CBv2NativeShutdownError.operationClosed }
    }

    func requiredCompletionFailed() {
        lock.withLock { failed = true }
        if tracking.fail(.nativeWorkFailed) { failure() }
    }

    /// Called after the producer has severed/handed off its consumer aliases.
    /// Returns after QUEUING, not after completion. Release the associated
    /// reservation only inside retirement, never immediately after this call.
    /// A first failure retains roots, loan and retirement closure permanently.
    @discardableResult
    func finishAfterDroppingConsumers(retirement: @escaping () throws -> Void = {}) -> Bool {
        let accepted = lock.withLock { () -> Bool in
            guard !finishing, !finished, !failed else { return false }
            finishing = true
            self.retirement = retirement
            return true
        }
        guard accepted else { return false }
        queue.async { [self] in
            guard tracking.mayExecute else { return }
            let captured = lock.withLock { streams }
            var firstError: Error?
            for stream in captured {
                do {
                    try beforeFence?(stream)
                    try withError { error in
                        stream.synchronize()
                        try error.check()
                    }
                } catch { if firstError == nil { firstError = error } }
            }
            if firstError != nil {
                requiredCompletionFailed()
                return
            }
            // The completion queue is the actual engine queue. No fence or
            // provider/accounting callback is performed under the commit lock.
            completionQueue.async { [self] in
                var detached:
                    ([MLXArray], [AnyObject], (() throws -> Void)?, (@Sendable () throws -> Void)?)?
                guard
                    tracking.commitIfHealthy({
                        lock.withLock {
                            guard !failed, !finished else { return }
                            detached = (arrays, owners, self.retirement, self.validateBinding)
                            arrays = []
                            owners = []
                            self.retirement = nil
                            self.validateBinding = nil
                            request = nil
                            finished = true
                        }
                        tracking.captureCompletedPrefixStreams(captured)
                    }), var released = detached
                else { return }
                // Move the tuple before clearing its arrays: a second COW
                // alias must not survive the retirement callback/endLoan/wake.
                detached = nil
                // Explicit order: native aliases, owner aliases, then credit.
                released.0.removeAll()
                released.1.removeAll()
                released.3 = nil
                do { try released.2?() } catch {
                    // A failed accounting/retirement callback retains its real
                    // captures too; no successful native receipt may follow.
                    lock.withLock {
                        self.retirement = released.2
                        failed = true
                    }
                    if tracking.fail(.nativeWorkFailed) { failure() }
                    return
                }
                released.2 = nil
                if let loan { tracking.endLoan(loan) }
                wake()
            }
        }
        return true
    }

    var debugRetainedArrayCount: Int { lock.withLock { arrays.count } }
    var debugIsFailed: Bool { lock.withLock { failed } }
}

/// One completed-group donor. Kept in the native engine until the actual
/// publication/drop consumer closes, not merely until a token terminal.
final class CBv2NativeCompletePrefixRetiredRows: @unchecked Sendable {
    let id = UUID()
    let requestID: CBv2RequestID
    let intent: CBv2DonationIntent?
    private(set) var state: [CBv2SequenceKV?]
    private var reservation: CBv2CheckpointReservation?
    private var completion: (() -> Void)?
    var started = false  // engine queue only
    init(
        requestID: CBv2RequestID, state: [CBv2SequenceKV?], intent: CBv2DonationIntent?,
        reservation: CBv2CheckpointReservation?, completion: @escaping () -> Void
    ) {
        self.requestID = requestID
        self.state = state
        self.intent = intent
        self.reservation = reservation
        self.completion = completion
    }
    func releaseRowsAndReservation(backend: CBv2KVBackend) {
        if !state.isEmpty { backend.release(state) }
        state.removeAll()
        reservation?.release()
        reservation = nil
        intent?.retiredReservation?.release()
    }
    func complete() {
        let callback = completion
        completion = nil
        callback?()
    }
}

/// An existing native loan owns both the exact store and its real async close.
/// The observer joins the close task's value before notifying the engine.
/// Neither task grants completion from a timeout or scalar counter.
final class CBv2NativeCompletePrefixStoreCloseJoin: @unchecked Sendable {
    private let store: any CBv2NativeCompletePrefixCache
    private var operation: Task<Void, Never>?
    private var observer: Task<Void, Never>?
    init(store: any CBv2NativeCompletePrefixCache) { self.store = store }
    func start(completion: @escaping @Sendable () -> Void) {
        precondition(operation == nil)
        let store = self.store
        let task = Task { await store.closeAndWait() }
        operation = task
        observer = Task {
            await task.value
            completion()
        }
    }
}
