// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

public enum NativeConstructionPhase: String, Sendable {
    case idle, filesystemLoad, serialLoad, modelFactory, nativeSetup, nativeKVProbe
}
public enum NativeConstructionCompletion: Sendable, Equatable {
    case noNativeSubmission, capturedStreamsCompleted
}
/// Completion of one construction epoch, not deallocation or memory credit.
public struct NativeConstructionReceipt: Sendable, Equatable {
    public let ownerID: UUID
    public let epoch: UInt64
    public let completion: NativeConstructionCompletion
    fileprivate init(ownerID: UUID, epoch: UInt64, completion: NativeConstructionCompletion) {
        self.ownerID = ownerID; self.epoch = epoch; self.completion = completion
    }
}
public struct NativeConstructionFault: Sendable, Equatable {
    public let phase: NativeConstructionPhase
    public let cause: String
    public let completionFailures: [String]
}
public enum NativeConstructionDisposition: Sendable, Equatable {
    case idle, active
    case completed(NativeConstructionReceipt)
    case retainedFault(NativeConstructionFault)
}
public struct NativeConstructionSnapshot: Sendable, Equatable {
    public let ownerID: UUID
    public let epoch: UInt64
    public let phase: NativeConstructionPhase
    public let disposition: NativeConstructionDisposition
    /// Counts of SDK references, never a physical allocation measurement.
    public let retainedArrayCount, retainedOwnerCount, capturedStreamCount: Int
    public var isRetainedFault: Bool {
        if case .retainedFault = disposition { return true }
        return false
    }
}
public enum NativeConstructionError: Error, Sendable, Equatable {
    case inactiveScope, alreadyConsumed, staleReceipt, invalidContainerAdoption
    case unqualifiedNativeOwner
    case retainedFault(NativeConstructionFault)
}

/// Only Sendable metadata crosses this lock. No native references are exposed.
private final class NativeConstructionStatus: @unchecked Sendable {
    private let lock = NSLock()
    private var value: NativeConstructionSnapshot
    init(_ value: NativeConstructionSnapshot) { self.value = value }
    func read() -> NativeConstructionSnapshot { lock.withLock { value } }
    func publish(_ value: NativeConstructionSnapshot) { lock.withLock { self.value = value } }
}
private final class NativeConstructionValue<Value> {
    let value: Value
    init(_ value: Value) { self.value = value }
}

/// Actor-local opaque lifetime. The caller MUST keep it alive after failed
/// completion, stop using the affected native model, and retain it until process
/// restart. There is intentionally no recovery, reset, or credit-refund method.
/// Do not share this scope across concurrent calls or isolation domains.
public final class NativeConstructionScope {
    private let ownerID = UUID()
    private var epoch: UInt64 = 0
    private var phase = NativeConstructionPhase.idle
    private var disposition = NativeConstructionDisposition.idle
    private var depth = 0
    private var mayHaveSubmitted = false
    private var arrays: [MLXArray] = []
    private var owners: [AnyObject] = []
    private var streams: [MLX.Stream] = []
    private var invalidations: [ObjectIdentifier: () -> Void] = [:]
    private var immutableLoadedOwners = Set<ObjectIdentifier>()
    private let publish: (@Sendable (NativeConstructionSnapshot) -> Void)?

    // Deterministic fault boundaries for package tests. These may only refuse
    // an operation/receipt, never replace native work with a successful result.
    package var testingBoundary: ((String, NativeConstructionScope) throws -> Void)?
    package var testingBeforeFence: ((MLX.Stream) throws -> Void)?
    package var retainedArraysForTesting: [MLXArray] { arrays }
    package var retainedOwnersForTesting: [AnyObject] { owners }
    package var capturedStreamsForTesting: [MLX.Stream] { streams }

    public init() { publish = nil }
    fileprivate init(publish: @escaping @Sendable (NativeConstructionSnapshot) -> Void) {
        self.publish = publish
        notify()
    }
    public var snapshot: NativeConstructionSnapshot {
        .init(ownerID: ownerID, epoch: epoch, phase: phase, disposition: disposition,
              retainedArrayCount: arrays.count, retainedOwnerCount: owners.count,
              capturedStreamCount: streams.count)
    }
    private func notify() { publish?(snapshot) }

    public func validate(_ receipt: NativeConstructionReceipt) throws {
        guard disposition == .completed(receipt), receipt.ownerID == ownerID,
              receipt.epoch == epoch else { throw NativeConstructionError.staleReceipt }
    }

    /// Nested SDK calls share their outer epoch. Only the outermost return may
    /// discard temporary owners. A probe explicitly fences before unbinding rows.
    package func withPhase<Value>(_ nextPhase: NativeConstructionPhase,
                                   _ body: () throws -> Value) throws -> Value {
        if case .retainedFault(let fault) = disposition {
            throw NativeConstructionError.retainedFault(fault)
        }
        let outermost = depth == 0
        let previousPhase = phase
        if outermost {
            let next = epoch.addingReportingOverflow(1)
            guard !next.overflow else { throw NativeConstructionError.alreadyConsumed }
            epoch = next.partialValue
            disposition = .active
            mayHaveSubmitted = false
        }
        phase = nextPhase; depth += 1; notify()
        defer {
            depth -= 1
            if !outermost { phase = previousPhase; notify() }
        }
        do {
            let value = try body()
            if outermost { try complete(cause: "normal return") }
            return value
        } catch {
            let original = error
            if outermost && !snapshot.isRetainedFault {
                try complete(cause: String(describing: original))
            }
            throw original
        }
    }

    private func requireActive() throws {
        guard depth > 0, disposition == .active else {
            if case .retainedFault(let fault) = disposition {
                throw NativeConstructionError.retainedFault(fault)
            }
            throw NativeConstructionError.inactiveScope
        }
    }
    package func retain(_ array: MLXArray) throws {
        try requireActive()
        // Preserve the actual wrapper identity. No copied/duplicated storage.
        if !arrays.contains(where: { $0 === array }) { arrays.append(array) }
        notify()
    }
    package func retain<S: Sequence>(arrays values: S) throws where S.Element == MLXArray {
        for value in values { try retain(value) }
    }
    package func retainOwner(_ owner: AnyObject) throws {
        try requireActive()
        if !owners.contains(where: { $0 === owner }) { owners.append(owner) }
        notify()
    }
    package func retainValue<Value>(_ value: Value) throws {
        try retainOwner(NativeConstructionValue(value))
    }
    /// Only the strict native factory authorizes the fully materialized loaded
    /// resources owner. An arbitrary active scope or successful target probe
    /// does not establish the provenance of untouched assistant parameters.
    package func authorizeImmutableLoadedOwner(_ owner: AnyObject) throws {
        try retainOwner(owner)
        immutableLoadedOwners.insert(ObjectIdentifier(owner))
    }
    package func requireImmutableLoadedOwner(_ owner: AnyObject?) throws {
        try requireActive()
        guard let owner, immutableLoadedOwners.contains(ObjectIdentifier(owner)) else {
            throw NativeConstructionError.unqualifiedNativeOwner
        }
    }
    /// Logical invalidation only, on this exclusive native construction segment.
    /// A later outer setup fence can fail after a nested probe already returned.
    package func invalidateOnFailedCompletion(_ owner: AnyObject, _ action: @escaping () -> Void) throws {
        try retainOwner(owner)
        invalidations[ObjectIdentifier(owner)] = action
    }
    package func capture(_ stream: MLX.Stream) throws {
        try requireActive()
        if !streams.contains(stream) { streams.append(stream) }
        notify()
    }
    package func willSubmit() throws {
        try requireActive()
        guard !streams.isEmpty else { throw NativeConstructionError.inactiveScope }
        mayHaveSubmitted = true
    }
    package func checkpoint(_ name: String) throws {
        try requireActive()
        try testingBoundary?(name, self)
    }

    /// Uses the captured streams, not whatever "default" means at unwind.
    /// All streams are attempted; any failure permanently retains every owner.
    package func fence(cause: String) throws {
        try requireActive()
        guard mayHaveSubmitted else { return }
        var failures: [String] = []
        for stream in streams {
            do {
                try testingBeforeFence?(stream)
                try withError { error in stream.synchronize(); try error.check() }
            } catch {
                failures.append("\(stream): \(error)")
            }
        }
        guard failures.isEmpty else {
            let fault = NativeConstructionFault(phase: phase, cause: cause,
                                                 completionFailures: failures)
            disposition = .retainedFault(fault)
            for invalidate in invalidations.values { invalidate() }
            invalidations.removeAll()
            immutableLoadedOwners.removeAll()
            testingBoundary = nil; testingBeforeFence = nil
            notify()
            throw NativeConstructionError.retainedFault(fault)
        }
    }

    private func complete(cause: String) throws {
        try fence(cause: cause)
        let completion: NativeConstructionCompletion =
            mayHaveSubmitted ? .capturedStreamsCompleted : .noNativeSubmission
        // The returned value or host container owns successful long-lived
        // objects. Clear only this epoch's extra roots after successful fences.
        arrays.removeAll(); owners.removeAll(); streams.removeAll()
        invalidations.removeAll()
        immutableLoadedOwners.removeAll()
        testingBoundary = nil; testingBeforeFence = nil
        disposition = .completed(.init(ownerID: ownerID, epoch: epoch, completion: completion))
        notify()
    }
}

/// Preinstall this opaque owner in a real host transaction BEFORE starting the
/// async factory. It stays reachable even if no model returns or a phase hangs.
/// Its mutex is held over the complete async operation, not actor-reentrant.
/// No member exposes an MLXArray, ModelContext, raw model, or mutable binding.
public final class NativeConstructionWork: Sendable {
    private final class State {
        let scope: NativeConstructionScope
        var loadStarted = false
        var pendingContainer: ModelContainer?
        // Do not retain the host's successfully adopted container, but do not
        // confuse a recyclable address with a still-live adopted identity.
        weak var adoptedContainer: ModelContainer?
        var sealed = false
        init(scope: NativeConstructionScope) { self.scope = scope }
    }
    private let status: NativeConstructionStatus
    private let state: SerialAccessContainer<State>

    public init() {
        let initial = NativeConstructionScope().snapshot
        let status = NativeConstructionStatus(initial)
        let scope = NativeConstructionScope(publish: { status.publish($0) })
        self.status = status
        self.state = .init(State(scope: scope))
    }
    public var snapshot: NativeConstructionSnapshot { status.read() }

    package func configureFailureForTesting(
        _ configure: @escaping @Sendable (NativeConstructionScope) -> Void
    ) async throws {
        try await state.read { state in
            guard !state.loadStarted, !state.sealed else { throw NativeConstructionError.alreadyConsumed }
            configure(state.scope)
        }
    }

    /// Atomically wins against any not-yet-started managed load. A metadata-only
    /// host failure can obtain a real no-submission receipt without assuming a
    /// cancelled task will never resume and try to begin construction.
    public func finishUnstartedConstruction() async throws -> NativeConstructionReceipt {
        try await state.read { state in
            guard !state.loadStarted, !state.sealed else { throw NativeConstructionError.alreadyConsumed }
            state.loadStarted = true; state.sealed = true
            try state.scope.withPhase(.modelFactory) {}
            guard case .completed(let receipt) = state.scope.snapshot.disposition else {
                throw NativeConstructionError.staleReceipt
            }
            return receipt
        }
    }

    public func validate(_ receipt: NativeConstructionReceipt) throws {
        let current = snapshot
        guard current.disposition == .completed(receipt),
              receipt.ownerID == current.ownerID, receipt.epoch == current.epoch else {
            throw NativeConstructionError.staleReceipt
        }
    }

    package func constructContainer(
        _ body: @escaping @Sendable (NativeConstructionScope) throws -> ModelContainer
    ) async throws -> ModelContainer {
        try await state.read { state in
            guard !state.loadStarted, !state.sealed else { throw NativeConstructionError.alreadyConsumed }
            state.loadStarted = true
            let container = try body(state.scope)
            // This write precedes the async return. If the caller is cancelled
            // before registering its newcomer, the preinstalled owner survives.
            state.pendingContainer = container
            return container
        }
    }

    /// Host MUST already strongly own the identical raw container. This only
    /// drops the pending-handoff alias; it does not complete future setup work.
    public func acknowledgeContainerAdoption(_ container: ModelContainer) async throws {
        try await state.read { state in
            guard !state.sealed, state.pendingContainer === container,
                  case .completed = state.scope.snapshot.disposition else {
                throw NativeConstructionError.invalidContainerAdoption
            }
            state.adoptedContainer = container
            state.pendingContainer = nil
        }
    }

    package func withContainer<Result: Sendable>(
        _ container: ModelContainer,
        _ body: @escaping @Sendable (ModelContext, NativeConstructionScope) throws -> Result
    ) async throws -> Result {
        try await state.read { state in
            guard !state.sealed, state.pendingContainer == nil,
                  state.adoptedContainer === container else {
                throw NativeConstructionError.invalidContainerAdoption
            }
            // Fixed lock order: work -> container. The scope is exclusively
            // borrowed while the work mutex stays held across this await.
            return try await container.perform(nonSendable: state.scope) { context, scope in
                try scope.withPhase(.nativeSetup) {
                    try scope.retainValue(context.model)
                    try scope.capture(StreamOrDevice.cpu.stream)
                    try scope.capture(StreamOrDevice.default.stream)
                    return try body(context, scope)
                }
            }
        }
    }

    /// Prevent any later construction epoch before the host's no-await final
    /// publication commit. This is not a host retirement/accounting receipt.
    public func sealForPublication(_ receipt: NativeConstructionReceipt) async throws {
        try await state.read { state in
            guard !state.sealed, state.adoptedContainer != nil,
                  state.pendingContainer == nil else { throw NativeConstructionError.invalidContainerAdoption }
            try state.scope.validate(receipt)
            state.sealed = true
        }
    }
}
