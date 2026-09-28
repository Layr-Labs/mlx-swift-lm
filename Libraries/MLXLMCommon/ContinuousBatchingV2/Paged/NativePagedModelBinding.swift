// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// SDK producer only; validates the genuine strict-loaded generation without
/// arrays, provider callbacks, model traversal or a new completion authority.
package protocol CBv2NativePagedModelValidating: AnyObject {
    func validateNativePagedModel() throws
}

/// Metadata returned only after exact pool/row/cohort validation. No snapshot,
/// gather, allocation or stream operation is needed to check a MiMo row.
package struct CBv2NativePagedRowMetadata {
    package let request: UUID
    package let offset, retainedCount, maximumLength, oldestValidPosition: Int
    package let keyWidth, valueWidth, kvHeads: Int
    package let window: Int?
    package let dtype: DType
}

/// One protected constructor's immutable association after seal(). Existing
/// PagedKVBackend constructors do not install one. Its callbacks are the
/// already-existing MiMo row ledger, not public registration or a second ledger.
package final class CBv2NativePagedModelBinding {
    package let ownerID = UUID()
    let constructionOwnerID: UUID
    let constructionEpoch: UInt64
    let layerKinds: [CBv2LayerKind]
    let layerDTypes: [DType]
    let maximumContextTokens: Int
    private weak var model: AnyObject?
    private weak var loadedOwner: AnyObject?
    private weak var actualBackend: PagedKVBackend?
    private weak var actualBank: CBv2LayerCacheBank?
    private weak var processOwner: (any CBv2ProcessMemoryOwner)?
    private let validator: any CBv2NativePagedModelValidating
    private let registerRows: ([CBv2SequenceKV?], UUID) throws -> Void
    private let rowRequest: (any CBv2SequenceKV, Int) throws -> UUID
    private let removeRows: ([CBv2SequenceKV?], UUID) throws -> Void
    private var sealed = false
    private var runtime: CBv2NativeShutdownState?
    private var lifetimeLoan: UUID?
    private var engineQueue: DispatchQueue?
    private let queueKey = DispatchSpecificKey<UInt8>()
    private var onFailure: (() -> Void)?
    private var onRetirement: (() -> Void)?
    private var onWorkCreated: ((Set<CBv2RequestID>) -> Void)?
    private var onWorkRetired: ((Set<CBv2RequestID>) -> Void)?
    private(set) var poolRetired = false

    package var hasSealedNativeResources: Bool {
        guard sealed, !poolRetired, actualBackend != nil, actualBank != nil else { return false }
        do { try validator.validateNativePagedModel(); return true }
        catch { return false }
    }

    package init(model: AnyObject, loadedOwner: AnyObject,
        layerKinds: [CBv2LayerKind], layerDTypes: [DType], maximumContextTokens: Int,
        processMemoryOwner: any CBv2ProcessMemoryOwner,
        validator: any CBv2NativePagedModelValidating, construction: NativeConstructionScope,
        registerRows: @escaping ([CBv2SequenceKV?], UUID) throws -> Void,
        rowRequest: @escaping (any CBv2SequenceKV, Int) throws -> UUID,
        removeRows: @escaping ([CBv2SequenceKV?], UUID) throws -> Void) throws {
        try construction.requireImmutableLoadedOwner(loadedOwner)
        try validator.validateNativePagedModel()
        let origin = construction.snapshot
        guard case .active = origin.disposition, origin.epoch > 0,
              !layerKinds.isEmpty, layerKinds.count <= 48, layerKinds.count == layerDTypes.count,
              maximumContextTokens > 0, maximumContextTokens <= Int(Int32.max),
              layerKinds.allSatisfy({ $0.sharesKVWithLayer == nil
                  && $0.headDim == 192 && $0.valueHeadDim == 128
                  && $0.queryHeads == 64 && [4, 8].contains($0.kvHeads)
                  && ($0.attention == .full || $0.attention == .slidingWindow(128)) }),
              layerDTypes.allSatisfy({ $0 == .bfloat16 || $0 == .float16 }) else {
            throw CBv2KVError.backendIneligible(reason: "unqualified native MiMo paged geometry")
        }
        self.model = model; self.loadedOwner = loadedOwner
        self.layerKinds = layerKinds; self.layerDTypes = layerDTypes
        self.maximumContextTokens = maximumContextTokens
        self.processOwner = processMemoryOwner; self.validator = validator
        constructionOwnerID = origin.ownerID; constructionEpoch = origin.epoch
        self.registerRows = registerRows; self.rowRequest = rowRequest; self.removeRows = removeRows
    }

    func attach(_ backend: PagedKVBackend) throws {
        guard actualBackend == nil, !sealed, backend.layerKinds == layerKinds,
              backend.pool.layerDTypes == layerDTypes,
              backend.pool.usesStepOwnedAttention, backend.pool.segmentGrant != nil,
              backend.pool.bytesReserved == 0, backend.pool.bytesMaterialized == 0,
              backend.residentPrefixIndex == nil,
              backend.pool.config.gatheredAttention?.maximumContextTokens == maximumContextTokens else {
            throw CBv2KVError.backendIneligible(reason: "native paged binding requires fresh exact segmented backing")
        }
        try validator.validateNativePagedModel()
        actualBackend = backend
    }

    package func seal(bank: CBv2LayerCacheBank, caches: [PagedLayerCache]) throws {
        guard !sealed, actualBank == nil, let backend = actualBackend,
              caches.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(reason: "native paged binding already sealed or incomplete")
        }
        for (index, cache) in caches.enumerated() {
            guard cache.pool === backend.pool, cache.layerIndex == index,
                  cache.kind == layerKinds[index],
                  backend.pool.attentionWorkCaches[index]?.value === cache else {
                throw CBv2KVError.backendIneligible(reason: "foreign native paged layer cache")
            }
        }
        try validator.validateNativePagedModel()
        actualBank = bank; sealed = true
    }

    func validate(model: AnyObject, backend: AnyObject, bank: AnyObject,
                  processMemoryOwner: (any CBv2ProcessMemoryOwner)?,
                  constructionOwnerID: UUID, constructionEpoch: UInt64) throws {
        guard sealed, !poolRetired, self.model === model, actualBackend === backend,
              actualBank === bank, loadedOwner != nil,
              processOwner != nil, processOwner === processMemoryOwner,
              self.constructionOwnerID == constructionOwnerID,
              self.constructionEpoch == constructionEpoch else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        try validator.validateNativePagedModel()
    }

    func installRuntime(_ tracking: CBv2NativeShutdownState, retaining engine: AnyObject,
                        queue: DispatchQueue,
                        onFailure: @escaping () -> Void, onRetirement: @escaping () -> Void,
                        onWorkCreated: @escaping (Set<CBv2RequestID>) -> Void,
                        onWorkRetired: @escaping (Set<CBv2RequestID>) -> Void) throws {
        guard runtime == nil, sealed, tracking.supported, tracking.mayExecute,
              actualBackend != nil, actualBank != nil else { throw CBv2NativeShutdownError.unsupportedConsumer }
        try validator.validateNativePagedModel()
        lifetimeLoan = try tracking.beginLoan(owner: engine)
        runtime = tracking; engineQueue = queue
        self.onFailure = onFailure; self.onRetirement = onRetirement
        self.onWorkCreated = onWorkCreated; self.onWorkRetired = onWorkRetired
        queue.setSpecific(key: queueKey, value: 1)
    }

    func requireEngineQueue() throws {
        guard DispatchQueue.getSpecific(key: queueKey) == 1, !poolRetired,
              let runtime else { throw CBv2NativeShutdownError.unsupportedConsumer }
        try runtime.requireWork()
        try validator.validateNativePagedModel()
    }

    func preflightCreation(backend: PagedKVBackend, kinds: [CBv2LayerKind], maximumLength: Int) throws {
        try requireEngineQueue()
        guard actualBackend === backend, kinds == layerKinds,
              CBv2NativePagedOperation.constructing?.tracking === runtime,
              maximumLength > 0, maximumLength <= maximumContextTokens else {
            throw CBv2KVError.backendIneligible(reason: "native paged target layout/context mismatch")
        }
    }

    /// Complete all checks before publishing ANY entry. Called only by the
    /// actual backend after its full cold row construction succeeded.
    func register(_ rows: [CBv2SequenceKV?], backend: PagedKVBackend) throws {
        try requireEngineQueue()
        try preflightRows(rows, backend: backend, requireCohort: false)
        try registerRows(rows, ownerID)
    }

    /// Existing target-only profile does not authorize import. The future
    /// complete-checkpoint transaction must call a separate typed adoption seam.
    func refuseImport() throws {
        throw CBv2KVError.backendIneligible(reason: "native paged COMPLETE prefix is not issued by target-only profile")
    }

    func remove(_ rows: [CBv2SequenceKV?], backend: PagedKVBackend) throws {
        try requireEngineQueue()
        try preflightRows(rows, backend: backend, requireCohort: true)
        try removeRows(rows, ownerID) // existing ledger itself validates all before removal
    }

    private func preflightRows(_ rows: [CBv2SequenceKV?], backend: PagedKVBackend,
                               requireCohort: Bool) throws {
        guard actualBackend === backend, rows.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(reason: "native paged release requires its whole actual request")
        }
        var seen = Set<UInt64>(), request: UUID?
        for (index, item) in rows.enumerated() {
            guard let row = item as? PagedSequenceKV, row.pool === backend.pool,
                  !row.isReleased, seen.insert(row.serial).inserted,
                  row.groupKey == backend.pool.groupKey(forLayer: index),
                  row.maxLength <= maximumContextTokens,
                  row.windowSize == expectedWindow(at: index) else {
                throw CBv2KVError.backendIneligible(reason: "foreign, aliased, released or wrong-layer native paged row")
            }
            if requireCohort {
                let actual = try rowRequest(row, index)
                guard request == nil || request == actual else {
                    throw CBv2KVError.backendIneligible(reason: "mixed native paged request cohorts")
                }
                request = actual
            }
        }
    }

    package func metadata(row: any CBv2SequenceKV, layer: Int) throws -> CBv2NativePagedRowMetadata {
        try requireEngineQueue()
        guard layerKinds.indices.contains(layer), let backend = actualBackend,
              let row = row as? PagedSequenceKV, row.pool === backend.pool, !row.isReleased,
              row.groupKey == backend.pool.groupKey(forLayer: layer),
              row.windowSize == expectedWindow(at: layer),
              row.groupKey.dtype == layerDTypes[layer] else {
            throw CBv2KVError.backendIneligible(reason: "foreign native paged metadata row")
        }
        let request = try rowRequest(row, layer)
        return .init(request: request, offset: row.absoluteOffset, retainedCount: row.retainedCount,
            maximumLength: row.maxLength, oldestValidPosition: row.oldestValidPosition,
            keyWidth: row.groupKey.headDim, valueWidth: row.groupKey.valueHeadDim,
            kvHeads: row.groupKey.kvHeads, window: row.windowSize, dtype: row.groupKey.dtype)
    }

    private func expectedWindow(at layer: Int) -> Int? {
        if case .slidingWindow(let window) = layerKinds[layer].attention { return window }
        return nil
    }

    func beginWork(requests: Set<CBv2RequestID> = [], nativeData: Bool = true) throws -> CBv2NativePagedOperation {
        try requireEngineQueue()
        if nativeData {
            guard MiMoV26NAXGatherQMM.gpuStream(.default),
                  StreamOrDevice.default.stream == MLX.Stream.gpu else {
                throw CBv2KVError.backendIneligible(reason: "native paged construction requires its actual GPU stream")
            }
        }
        guard let runtime, let queue = engineQueue else { throw CBv2NativeShutdownError.unsupportedConsumer }
        let operation = try CBv2NativePagedOperation(tracking: runtime, queue: queue,
            onFailure: { [weak self] in self?.onFailure?() },
            onRetirement: { [weak self] in
                self?.onWorkRetired?(requests); self?.onRetirement?()
            })
        onWorkCreated?(requests)
        return operation
    }

    func markPoolRetired(after operation: CBv2NativePagedOperation) throws {
        try requireEngineQueue()
        guard operation.completed, !operation.failed, operation.tracking === runtime else {
            throw CBv2NativeShutdownError.operationClosed
        }
        poolRetired = true
    }

    /// Excludes ONLY this exact issued lifetime loan. A media/import/work or
    /// foreign/unrelated loan can never be mistaken for idle pool ownership.
    var canStartPoolRetirement: Bool {
        guard let runtime, let lifetimeLoan, !poolRetired else { return false }
        return runtime.hasOnlyNativeLoan(lifetimeLoan)
    }

    func finishPoolLifetime() {
        precondition(poolRetired)
        if let runtime, let lifetimeLoan { runtime.endLoan(lifetimeLoan) }
        lifetimeLoan = nil
    }

}

/// Existing NativeState operation loan, not an accounting/receipt framework.
/// Used by conditional page growth and the existing paged attention owner.
final class CBv2NativePagedOperation: @unchecked Sendable {
    @TaskLocal static var constructing: CBv2NativePagedOperation?
    let tracking: CBv2NativeShutdownState
    let queue: DispatchQueue
    private let onFailure: () -> Void, onRetirement: () -> Void
    private let streams: [MLX.Stream]
    private var arrays: [MLXArray] = []
    private var owners: [AnyObject] = []
    private var loan: UUID?
    private(set) var failed = false
    private(set) var completed = false
    private var finished = false
    var hasArrays: Bool { !arrays.isEmpty }

    init(tracking: CBv2NativeShutdownState, queue: DispatchQueue,
         onFailure: @escaping () -> Void, onRetirement: @escaping () -> Void) throws {
        self.tracking = tracking; self.queue = queue
        self.onFailure = onFailure; self.onRetirement = onRetirement
        let actual = StreamOrDevice.default.stream
        streams = actual == MLX.Stream.cpu ? [actual] : [actual, MLX.Stream.cpu]
        loan = try tracking.beginLoan(owner: self, duringDrain: true)
        tracking.captureStreams()
    }
    func retain(_ array: MLXArray) { arrays.append(array) }
    func retain(owner: AnyObject) { owners.append(owner) }
    func withConstruction<Result>(_ body: () throws -> Result) rethrows -> Result {
        try Self.$constructing.withValue(self, operation: body)
    }
    func requireWork() throws { try tracking.requireWork() }
    private final class SameQueueAction: @unchecked Sendable {
        let body: () -> Void
        init(_ body: @escaping () -> Void) { self.body = body }
    }
    /// Transfer ONLY back to the captured engine queue; the actual native
    /// loan already retains its consumer. Not a detached task/completion proof.
    func enqueueRetirement(_ body: @escaping () -> Void) {
        let action = SameQueueAction(body)
        queue.async { action.body() }
    }
    func fail() {
        failed = true
        if tracking.fail(.nativeWorkFailed) { onFailure() }
    }
    func requiredDrain() throws {
        guard !failed else { throw CBv2NativeShutdownError.operationClosed }
        do {
            try tracking.requireWork()
            for stream in streams {
                try tracking.beforeFenceForTesting?(stream)
                try withError { errors in stream.synchronize(); try errors.check() }
            }
            try tracking.requireWork()
            completed = true
        } catch { fail(); throw error }
    }
    /// Only after actual successful drain, or a positively unstarted operation.
    /// Completion callbacks must be scalar/accounting only; no native waits.
    func finish(unstarted: Bool = false, _ release: () -> Void = {}) {
        guard !finished, !failed, completed || (unstarted && arrays.isEmpty && owners.isEmpty) else { return }
        var detached: ([MLXArray], [AnyObject])?
        guard tracking.commitIfHealthy({
            finished = true; detached = (arrays, owners); arrays = []; owners = []
        }), var held = detached else { return }
        detached = nil
        held.0.removeAll(); held.1.removeAll()
        release()
        if let loan { tracking.endLoan(loan) }
        onRetirement()
    }
}
