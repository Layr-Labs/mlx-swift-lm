// Copyright © 2026 Eigen Labs.
// Unregistered contiguous CBv2 adapter over the existing native MiMo target.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum MiMoV26CBv2Error: Error, Equatable, Sendable {
    case invalidInput(String)
    case invalidCache(layer: Int, reason: String)
    case contextExceeded
    case unloadedPrecision
    case nativeProbeRequired
}

/// Ownership metadata only. Entries never retain rows, arrays or request state.
final class MiMoV26CBv2RowLedger {
    final class Entry {
        weak var row: (any CBv2SequenceKV)?
        let backend: UUID
        let request: UUID
        let layer: Int
        init(row: any CBv2SequenceKV, backend: UUID, request: UUID, layer: Int) {
            self.row = row; self.backend = backend; self.request = request; self.layer = layer
        }
    }
    private let lock = NSLock()
    private var entries: [ObjectIdentifier: Entry] = [:]

    // Construction/retirement boundaries prune weak tombstones. Lookup stays
    // O(1) in the engine hot path and never retains a row merely for cleanup.
    private func pruneDeadRowsLocked() { entries = entries.filter { $0.value.row != nil } }
    var metadataEntryCount: Int {
        lock.lock(); defer { lock.unlock() }
        return entries.count
    }

    func register(_ rows: [CBv2SequenceKV?], backend: UUID) throws {
        lock.lock(); defer { lock.unlock() }
        pruneDeadRowsLocked()
        let request = UUID()
        var pending: [(ObjectIdentifier, Entry)] = []
        for (index, value) in rows.enumerated() {
            guard let row = value else { throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "backend omitted owning row") }
            let key = ObjectIdentifier(row)
            guard entries[key]?.row == nil, !pending.contains(where: { $0.0 == key }) else {
                throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "backend reused a live row")
            }
            pending.append((key, Entry(row: row, backend: backend, request: request, layer: index)))
        }
        for (key, entry) in pending { entries[key] = entry }
    }
    func identity(_ row: any CBv2SequenceKV, layer: Int) throws -> UUID {
        lock.lock(); defer { lock.unlock() }
        guard let entry = entries[ObjectIdentifier(row)], entry.row === row, entry.layer == layer else {
            throw MiMoV26CBv2Error.invalidCache(layer: layer, reason: "foreign, released or wrong-layer row")
        }
        return entry.request
    }
    func remove(_ rows: [CBv2SequenceKV?], backend: UUID) throws {
        lock.lock(); defer { lock.unlock() }
        pruneDeadRowsLocked()
        var keys = Set<ObjectIdentifier>()
        var request: UUID?
        for (index, value) in rows.enumerated() {
            guard let row = value, let entry = entries[ObjectIdentifier(row)], entry.row === row,
                  entry.backend == backend, entry.layer == index,
                  keys.insert(ObjectIdentifier(row)).inserted,
                  request == nil || request == entry.request else {
                throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "foreign or mixed request release")
            }
            request = entry.request
        }
        for key in keys { entries.removeValue(forKey: key) }
    }
    func invalidate(backend: UUID) {
        lock.lock(); defer { lock.unlock() }
        entries = entries.filter { $0.value.backend != backend && $0.value.row != nil }
    }
}

/// Thin owner facade over the existing contiguous allocator/accounting. It
/// adds row provenance only, with no second row implementation or tensor copy.
public final class MiMoV26CBv2Backend: CBv2KVBackend, CBv2ContiguousHistoricalBackend {
    private let backend: CBv2ContiguousKVBackend
    private let ledger: MiMoV26CBv2RowLedger
    private let owner = UUID()
    private let expectedKinds: [CBv2LayerKind]
    private let maximumContext: Int
    fileprivate init(kinds: [CBv2LayerKind], context: Int, dtype: DType, bytesCapacity: Int,
                     ledger: MiMoV26CBv2RowLedger) {
        expectedKinds = kinds; maximumContext = context; self.ledger = ledger
        backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: bytesCapacity, kvDType: dtype))
    }
    public var bytesCapacity: Int { backend.bytesCapacity }
    public var bytesInUse: Int { backend.bytesInUse }
    public var bytesReserved: Int { backend.bytesReserved }
    public var prefixReuseBackend: CBv2PrefixReuseBackend { .unknown }
    public func updateBytesCapacity(_ bytes: Int) { backend.updateBytesCapacity(bytes) }
    public func makeSequenceState(layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int) throws -> [CBv2SequenceKV?] {
        guard layerKinds == expectedKinds, maxLength <= maximumContext else {
            throw MiMoV26CBv2Error.invalidInput("backend target layout/context mismatch")
        }
        let rows = try backend.makeSequenceState(layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength)
        do { try ledger.register(rows, backend: owner) }
        catch { backend.release(rows); throw error }
        return rows
    }
    public func releaseValidated(_ state: [CBv2SequenceKV?]) throws {
        guard state.count == expectedKinds.count else { throw MiMoV26CBv2Error.invalidInput("release requires a complete request") }
        try ledger.remove(state, backend: owner)
        backend.release(state)
    }
    package func adoptContiguousHistoricalState(
        _ state: [CBv2SequenceKV?], codec: CBv2CompleteCheckpointCodec,
        layerKinds: [CBv2LayerKind], position: Int,
        requestID: CBv2RequestID, maximumSequenceLength: Int
    ) throws {
        guard layerKinds == expectedKinds, state.count == expectedKinds.count,
              maximumSequenceLength <= maximumContext, position > 1 else {
            throw MiMoV26CBv2Error.invalidInput("historical checkpoint target layout/context mismatch")
        }
        // Validate the real staged lease, native widths/dtypes and fresh request
        // transfer before publishing rows into this model's exact ownership map.
        try backend.adoptContiguousHistoricalState(state, codec: codec, layerKinds: layerKinds,
            position: position, requestID: requestID, maximumSequenceLength: maximumSequenceLength)
        do { try ledger.register(state, backend: owner) }
        catch { backend.release(state); throw error }
    }
    public func release(_ state: [CBv2SequenceKV?]) {
        do { try releaseValidated(state) }
        catch { preconditionFailure("MiMo CBv2 owned release invariant: \(error)") }
    }
    deinit { ledger.invalidate(backend: owner) }
}

/// Private model provenance facade; it is not a generic paging capability.
private protocol MiMoV26OwnedLayerCache: CBv2AttendingLayerCache, KVCache {
    var owner: UUID { get }
    var boundOffsets: [Int] { get }
    var mtpSerializesRectangularAttention: Bool { get set }
    var mtpBatchesRectangularAttention: Bool { get set }
}

/// Owns a causal-only common cache. No span-binding interface is exposed, so
/// external callers cannot install Gemma bidirectional overlays on MiMo.
private final class MiMoV26ContiguousLayerCache: MiMoV26OwnedLayerCache,
    CBv2MTPRectangularSerializing {
    let owner: UUID
    var blockBatchBudget: MiMoV26BlockBatchBudget? {
        get { base.mimoV26BlockBatchBudget }
        set { base.mimoV26BlockBatchBudget = newValue }
    }
    private let base: CBv2LayerCache
    private(set) var boundOffsets: [Int] = []
    var layerIndex: Int { base.layerIndex }
    var kind: CBv2LayerKind { base.kind }
    var rows: [CBv2SequenceKV] { base.rows }
    var positionOffsets: MLXArray { base.positionOffsets }
    // The exact owned base implements serial per-query causal/SWA attention.
    // No cache substitution, span capability, or attention kernel is added.
    var mtpSerializesRectangularAttention: Bool {
        get { base.mtpSerializesRectangularAttention }
        set { base.mtpSerializesRectangularAttention = newValue }
    }
    var mtpBatchesRectangularAttention: Bool {
        get { base.mtpBatchesRectangularAttention }
        set { base.mtpBatchesRectangularAttention = newValue }
    }

    init(owner: UUID, index: Int, kind: CBv2LayerKind) {
        self.owner = owner
        base = CBv2LayerCache(layerIndex: index, kind: kind,
                             mimoV26NAXAttention: true)
    }
    func setRows(_ rows: [CBv2SequenceKV]) {
        // Engine binding is nonthrowing; external callers use adapter.bindRows
        // for a recoverable all-layer preflight before any cache is rebound.
        precondition(rows.allSatisfy { $0.absoluteOffset >= 0 && $0.absoluteOffset <= Int(Int32.max) })
        boundOffsets = rows.map(\.absoluteOffset)
        base.setRows(rows)
    }
    func updateAndAttend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                         scale: Float, sinks: MLXArray?) -> MLXArray {
        let result = base.updateAndAttend(queries: queries, keys: keys, values: values,
                                          scale: scale, sinks: sinks)
        let length = queries.dim(2)
        boundOffsets = boundOffsets.map { $0 + length }
        return result
    }
    func attendBorrowing(source: CBv2AttendingLayerCache, queries: MLXArray,
                         scale: Float, sinks: MLXArray?) -> MLXArray {
        preconditionFailure("MiMo V2.6 owns every layer's K/V; no borrowed attention")
    }
    // EngineLoopV2 evaluates KVCache.innerState to collapse lazy position/KV
    // graphs. Preserve this bridge without permitting legacy update dispatch.
    func innerState() -> [MLXArray] { base.innerState() }
    var offset: Int { base.offset }
    var maxSize: Int? { base.maxSize }
    var state: [MLXArray] {
        get { [] }
        set { preconditionFailure("MiMo CBv2 cache has request-owned rows") }
    }
    var metaState: [String] {
        get { [] }
        set { preconditionFailure("MiMo CBv2 cache has no legacy metadata") }
    }
    var isTrimmable: Bool { false }
    func trim(_ n: Int) -> Int { 0 }
    func copy() -> any KVCache { preconditionFailure("Copy request rows through their owning backend") }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        preconditionFailure("MiMo CBv2 attention owns its row-local masks")
    }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("MiMo CBv2 must call updateAndAttend")
    }
}

/// Genuine page-backed attention facade over the SAME PagedLayerCache that
/// consumes pool-issued write/read tickets. No contiguous fallback or snapshot
/// gather is used to validate native row ownership.
private final class MiMoV26PagedLayerCache: MiMoV26OwnedLayerCache {
    let owner: UUID
    private let base: PagedLayerCache
    private(set) var boundOffsets: [Int] = []
    init(owner: UUID, base: PagedLayerCache) { self.owner = owner; self.base = base }
    var layerIndex: Int { base.layerIndex }
    var kind: CBv2LayerKind { base.kind }
    var rows: [CBv2SequenceKV] { base.rows }
    var positionOffsets: MLXArray { base.positionOffsets }
    // This facade intentionally does NOT conform to the public MTP capability.
    // Target-only issuance refuses a drafter before any engine is assembled.
    var mtpSerializesRectangularAttention: Bool {
        get { false }
        set { precondition(!newValue, "native paged MTP is not issued") }
    }
    var mtpBatchesRectangularAttention: Bool {
        get { false }
        set { precondition(!newValue, "native paged MTP is not issued") }
    }
    func setRows(_ rows: [CBv2SequenceKV]) {
        boundOffsets = rows.map(\.absoluteOffset)
        base.setRows(rows)
    }
    func updateAndAttend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                         scale: Float, sinks: MLXArray?) -> MLXArray {
        let result = base.updateAndAttend(queries: queries, keys: keys, values: values,
                                         scale: scale, sinks: sinks)
        boundOffsets = boundOffsets.map { $0 + queries.dim(2) }
        return result
    }
    func attendBorrowing(source: CBv2AttendingLayerCache, queries: MLXArray,
                         scale: Float, sinks: MLXArray?) -> MLXArray {
        preconditionFailure("MiMo owns every paged layer")
    }
    func innerState() -> [MLXArray] { base.innerState() }
    var offset: Int { base.offset }
    var maxSize: Int? { base.maxSize }
    var state: [MLXArray] {
        get { [] }
        set { preconditionFailure("native paged rows are request-owned") }
    }
    var metaState: [String] {
        get { [] }
        set { preconditionFailure("native paged rows have no legacy metadata") }
    }
    var isTrimmable: Bool { false }
    func trim(_ n: Int) -> Int { 0 }
    func copy() -> any KVCache { preconditionFailure("copy through the issued paged backend") }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        preconditionFailure("native paged attention owns masks")
    }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("native paged attention requires updateAndAttend")
    }
}

public struct MiMoV26CBv2NativePagedExecutionResources {
    public let backend: PagedKVBackend
    public let cacheProvider: CBv2LayerCacheBank
    public let contract: CBv2NativeExecutionContract
}

/// Native-local resources from one protected setup scope. The contract is issued
/// by the SDK for these exact identities, never asserted by provider metadata.
public struct MiMoV26CBv2NativeExecutionResources {
    public let backend: MiMoV26CBv2Backend
    public let cacheProvider: CBv2LayerCacheBank
    public let contract: CBv2NativeExecutionContract
}

/// One reference to one target; this is not another Module/parameter tree.
/// Batching here is an explicit component seam, not a packed-serving default.
public final class MiMoV26CBv2Adapter: CBv2SteppableModel, CBv2PrefillSteppableModel,
                                      CBv2ModelCapabilityProviding, CBv2MTPPrefillSteppableModel,
                                      CBv2MTPPolicyTopTwoProviding, CBv2MTPRequestScopedMediaFallback,
                                      CBv2HistoricalAttentionCheckpointProviding,
                                      CBv2CompleteCheckpointKVTypeProviding,
                                      MiMoV26BlockBatchAllocatingModel,
                                      MiMoV26RectangularDenseAllocatingModel {
    public let target: MiMoV26TextModel
    /// Retains the native factory's loaded components and host permit through
    /// this adapter's lifetime. Directly extracting target is a borrowed API;
    /// only the officially returned wrapper/binding carries this owner.
    private let loadLifetimeOwner: AnyObject?
    public let assistant: MiMoV26MTPAssistant?
    private let useRowLocalRectangularDense = MiMoV26RectangularDense.enabledByEnvironment
    package var cbv2MiMoRectangularDenseScratch: MiMoV26RectangularDenseScratchSpec? {
        guard useRowLocalRectangularDense, !target.model.useFusedDecodeNorms,
              assistant?.requiredVerificationMode == .rectangular,
              supportsRequestStatefulMTP else { return nil }
        return MiMoV26RectangularDense.scratchSpec(target)
    }
    public let layerKinds: [CBv2LayerKind]
    private let cacheOwner = UUID()
    private let rowLedger = MiMoV26CBv2RowLedger()
    private var nativePagedBinding: CBv2NativePagedModelBinding?
    private var nativeProbeScope = false
    private var nativeProbeFailed = false
    private var nativeProbeCaches = Set<ObjectIdentifier>()
    private var observedKVDTypes: [DType]?
    // Exact historical target KV is distinct from ordinary frozen replay.
    // Persistent assistants require their own complete-state codec, and the
    // protected provider profile must separately authorize store/I/O lifetime.
    public var cbv2SupportsHistoricalAttentionCheckpoint: Bool {
        guard !nativeProbeFailed, let types = observedKVDTypes,
              types.count == layerKinds.count,
              types.allSatisfy({ $0 == .bfloat16 || $0 == .float16 || $0 == .float32 }) else { return false }
        guard let assistant else { return true }
        // A real loaded native assistant must provide its complete three-head
        // codec. A declared MTP flag or target-only KV never enables this path.
        guard supportsRequestStatefulMTP,
              let codec = assistant as? any CBv2HistoricalMTPPrefixCheckpointCoding,
              let descriptors = codec.prefixCheckpointTensorDescriptors(targetInputCount: 4),
              !descriptors.isEmpty else { return false }
        return true
    }
    package var nativeCompletePrefixAssistantCodecID: String? {
        (assistant as? any CBv2HistoricalMTPPrefixCheckpointCoding)?.prefixCheckpointCodecID
    }
    package func validateNativeCompletePrefixOwner(_ owner: AnyObject) throws {
        guard loadLifetimeOwner === owner, cbv2SupportsHistoricalAttentionCheckpoint else {
            throw MiMoV26CBv2Error.invalidInput("invalid native historical-prefix owner/capability")
        }
    }
    public var cbv2CompleteCheckpointKVDTypes: [DType]? {
        nativeProbeFailed ? nil : observedKVDTypes
    }
    // ONE bounded weak association, not a provider registry. Never displaces
    // another live resource bundle; manually assembled banks stay unarmed.
    private final class WeakCache {
        weak var value: MiMoV26ContiguousLayerCache?
        init(_ value: MiMoV26ContiguousLayerCache) { self.value = value }
    }
    private let blockBatchLock = NSLock()
    private weak var blockBatchBackend: MiMoV26CBv2Backend?
    private weak var blockBatchProvider: AnyObject?
    private var blockBatchCaches: [WeakCache] = []
    private func associateBlockBatch(backend: MiMoV26CBv2Backend, provider: AnyObject,
                                     caches: [any CBv2AttendingLayerCache]) {
        guard caches.count == layerKinds.count, caches.count <= 48,
              caches.allSatisfy({ ($0 as? MiMoV26ContiguousLayerCache)?.owner == cacheOwner }) else { return }
        blockBatchLock.withLock {
            guard blockBatchBackend == nil && blockBatchProvider == nil else { return }
            blockBatchBackend = backend; blockBatchProvider = provider
            blockBatchCaches = caches.compactMap { ($0 as? MiMoV26ContiguousLayerCache).map(WeakCache.init) }
        }
    }
    package var cbv2MiMoBlockBatchLayerCount: Int? {
        guard !layerKinds.isEmpty, layerKinds.count <= 48,
              layerKinds.allSatisfy({ $0.headDim == 192 && $0.valueHeadDim == 128
                  && $0.queryHeads == 64 && [4,8].contains($0.kvHeads)
                  && ($0.attention == .full || $0.attention == .slidingWindow(128)) }) else { return nil }
        return layerKinds.count
    }
    package func cbv2TryInstallBlockBatchBudget(_ budget: MiMoV26BlockBatchBudget) -> Bool {
        blockBatchLock.withLock {
            guard budget.modelIdentity == ObjectIdentifier(self),
                  let backend = blockBatchBackend, let provider = blockBatchProvider,
                  budget.backendIdentity == ObjectIdentifier(backend),
                  budget.cacheProviderIdentity == ObjectIdentifier(provider),
                  blockBatchCaches.count == layerKinds.count else { return false }
            let caches = blockBatchCaches.compactMap(\.value)
            guard caches.count == layerKinds.count, caches.allSatisfy({ $0.blockBatchBudget == nil }) else { return false }
            for cache in caches { cache.blockBatchBudget = budget }
            return true
        }
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        .init(supportsPrefixReuse: false, supportsRecurrentCheckpointReuse: false,
              supportsPagedKV: nativePagedBinding?.hasSealedNativeResources == true, requiresNativePagedKV: false,
              supportsCompiledDecode: false, supportsPackedPrefill: false,
              supportsMTP: supportsRequestStatefulMTP, supportsCompactRecurrentMTPReplay: false)
    }

    public init(target: MiMoV26TextModel, assistant: MiMoV26MTPAssistant? = nil,
                retaining owner: AnyObject? = nil) throws {
        guard assistant == nil || assistant!.isCompatible(with: target) else {
            throw MiMoV26MTPError.incompatibleOwner
        }
        self.target = target
        loadLifetimeOwner = owner
        self.assistant = assistant
        layerKinds = try (0..<target.configuration.numHiddenLayers).map { index in
            let geometry = try target.configuration.attentionGeometry(at: index)
            return CBv2LayerKind(attention: geometry.slidingWindow.map { .slidingWindow($0) } ?? .full,
                hasSinks: geometry.hasSinks, headDim: geometry.headDim, valueHeadDim: geometry.valueHeadDim,
                kvHeads: geometry.keyValueHeads, queryHeads: geometry.queryHeads, modelLayerIndex: index)
        }
        guard target.hasLoadedEmbeddingPrecision, target.hasLoadedReadoutPrecision else {
            throw MiMoV26CBv2Error.unloadedPrecision
        }
    }

    public func makeCaches() -> [any CBv2AttendingLayerCache] {
        layerKinds.enumerated().map { MiMoV26ContiguousLayerCache(owner: cacheOwner, index: $0, kind: $1) }
    }

    /// Cold causal media stays on the ordinary target path. The explicit
    /// request-scoped engine opt-in never observes/drafts media with the heads;
    /// the same installed trained assistant remains available to text rows.
    public var supportsMultimodalPrefill: Bool { assistant == nil || supportsRequestStatefulMTP }
    public func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
        supportsMultimodalPrefill && attention == .causal
    }
    public func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        target.model.embedTokens(tokens)
    }
    public func forward(tokens: MLXArray, inputEmbeddings: MLXArray,
                        caches: [any CBv2AttendingLayerCache]) -> MLXArray {
        do {
            guard supportsMultimodalPrefill else {
                throw MiMoV26CBv2Error.invalidInput("native media owner is invalid")
            }
            return try forwardValidated(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
        } catch { preconditionFailure("MiMo causal media contract: \(error)") }
    }

    /// Uses the exact native rows/probe/ledger. No cache gains a span-mask
    /// setter, and ordinary arbitrary banks remain ineligible for media.
    public func makeMultimodalCacheProvider() throws -> any CBv2LayerCacheProvider {
        guard observedKVDTypes != nil else { throw MiMoV26CBv2Error.nativeProbeRequired }
        guard supportsMultimodalPrefill else {
            throw MiMoV26CBv2Error.invalidInput("native media owner is invalid")
        }
        return MiMoV26CausalMediaCacheProvider(adapter: self)
    }

    public func makeBackend(bytesCapacity: Int) throws -> MiMoV26CBv2Backend {
        guard nativePagedBinding == nil else { throw MiMoV26CBv2Error.invalidInput("paged adapter cannot issue a competing contiguous bank") }
        guard bytesCapacity >= 0 else { throw MiMoV26CBv2Error.invalidInput("negative backend capacity") }
        // Serving reservations must come from actual loaded projection/RoPE
        // storage, never an unproven activation-dtype assumption.
        guard let types = observedKVDTypes else { throw MiMoV26CBv2Error.nativeProbeRequired }
        // The common backend has one estimate dtype; actual rows retain their
        // native type. Mixed observed layers require its conservative FP32 bound.
        let dtype: DType = Set(types).count == 1 ? types[0] : .float32
        return MiMoV26CBv2Backend(kinds: layerKinds, context: target.configuration.maxPositionEmbeddings,
                                  dtype: dtype, bytesCapacity: bytesCapacity, ledger: rowLedger)
    }

    /// Native text/contiguous/default+CPU only. Call inside the factory's
    /// protected construction helper, before an active engine exists. Register
    /// the real owners before any subsequent veto or native engine construction.
    public func makeNativeExecutionResources(bytesCapacity: Int, retaining work: NativeConstructionScope)
        throws -> MiMoV26CBv2NativeExecutionResources {
        try work.requireImmutableLoadedOwner(loadLifetimeOwner)
        try work.retainOwner(self) // requires an already-active protected scope
        try work.invalidateOnFailedCompletion(self) { [weak self] in
            self?.nativeProbeFailed = true; self?.observedKVDTypes = nil
        }
        try work.capture(StreamOrDevice.cpu.stream)
        try work.capture(StreamOrDevice.default.stream)
        let backend = try makeBackend(bytesCapacity: bytesCapacity)
        try work.retainOwner(backend)
        let caches = makeCaches()
        try work.retainValue(caches)
        let bank = CBv2LayerCacheBank(caches: caches)
        try work.retainOwner(bank)
        associateBlockBatch(backend:backend,provider:bank,caches:caches)
        try work.checkpoint("adapter.nativeExecutionResources")
        let contract = try CBv2NativeExecutionContract(model: self, backend: backend,
            cacheProvider: bank, assistant: assistant, construction: work,
            mtpVerificationMode: assistant?.verificationMode ?? .serialTarget)
        return .init(backend: backend, cacheProvider: bank, contract: contract)
    }

    /// Package-only issuer: the strict loaded wrapper supplies its genuine
    /// session validator and exact registered store/process owner. No array
    /// capability is inferred from the store's conformance or metadata alone.
    package func makeNativeCompletePrefixResources(bytesCapacity: Int, loadedOwner: AnyObject,
        validator: any CBv2NativeCompletePrefixBindingValidating,
        completePrefixCache: any CBv2NativeCompletePrefixCache,
        processMemoryOwner: any CBv2ProcessMemoryOwner, retaining work: NativeConstructionScope) throws
        -> MiMoV26CBv2NativeExecutionResources {
        try validateNativeCompletePrefixOwner(loadedOwner)
        try validator.validateNativeCompletePrefixBinding()
        try work.requireImmutableLoadedOwner(loadedOwner)
        try work.retainOwner(self); try work.retainOwner(validator)
        try work.retainOwner(completePrefixCache); try work.retainOwner(processMemoryOwner)
        try work.invalidateOnFailedCompletion(self) { [weak self] in
            self?.nativeProbeFailed = true; self?.observedKVDTypes = nil
        }
        try work.capture(StreamOrDevice.cpu.stream)
        try work.capture(StreamOrDevice.default.stream)
        let backend = try makeBackend(bytesCapacity: bytesCapacity)
        try work.retainOwner(backend)
        let caches = makeCaches()
        try work.retainValue(caches)
        let bank = CBv2LayerCacheBank(caches: caches)
        try work.retainOwner(bank)
        associateBlockBatch(backend: backend, provider: bank, caches: caches)
        try work.checkpoint("adapter.nativeCompletePrefixResources")
        try validator.validateNativeCompletePrefixBinding()
        let contract = try CBv2NativeExecutionContract(model: self, backend: backend,
            cacheProvider: bank, assistant: assistant, construction: work, loadedOwner: loadedOwner,
            mtpVerificationMode: assistant?.verificationMode ?? .serialTarget,
            completePrefixCache: completePrefixCache, completePrefixValidator: validator,
            prefixProcessMemoryOwner: processMemoryOwner)
        return .init(backend: backend, cacheProvider: bank, contract: contract)
    }

    /// Protected joint issuer. One actual media-capable provider owns the
    /// same native caches used by text prefix/MTP and target-only media.
    /// The loaded wrapper creates the processor and validates any typed audio
    /// owner; Common then binds all identities in one consumed ticket.
    package func makeNativeManagedCompletePrefixResources(bytesCapacity: Int, processor: AnyObject,
        loadedOwner: AnyObject, validator: any CBv2NativeCompletePrefixBindingValidating,
        completePrefixCache: any CBv2NativeCompletePrefixCache,
        processMemoryOwner: any CBv2ProcessMemoryOwner,
        audioOwner: AnyObject? = nil, audioSessionID: UUID? = nil,
        audioSourceIdentity: String? = nil, audioGeneration: UUID? = nil,
        retaining work: NativeConstructionScope) throws
        -> (backend: MiMoV26CBv2Backend, cacheProvider: any CBv2LayerCacheProvider,
            contract: CBv2NativeExecutionContract) {
        try validateNativeCompletePrefixOwner(loadedOwner)
        try validator.validateNativeCompletePrefixBinding()
        try work.requireImmutableLoadedOwner(loadedOwner)
        try work.retainOwner(self); try work.retainOwner(processor)
        try work.retainOwner(validator); try work.retainOwner(completePrefixCache)
        try work.retainOwner(processMemoryOwner)
        if let audioOwner {
            try work.requireImmutableLoadedOwner(audioOwner)
            try work.retainOwner(audioOwner)
        }
        try work.invalidateOnFailedCompletion(self) { [weak self] in
            self?.nativeProbeFailed = true; self?.observedKVDTypes = nil
        }
        try work.capture(StreamOrDevice.cpu.stream)
        try work.capture(StreamOrDevice.default.stream)
        let backend = try makeBackend(bytesCapacity: bytesCapacity)
        try work.retainOwner(backend)
        let provider = try makeMultimodalCacheProvider()
        try work.retainOwner(provider)
        if let native = provider as? MiMoV26CausalMediaCacheProvider {
            associateBlockBatch(backend: backend, provider: native, caches: native.caches)
        }
        try work.checkpoint("adapter.nativeManagedCompletePrefixResources")
        try validator.validateNativeCompletePrefixBinding()
        let contract = try CBv2NativeExecutionContract(model: self, backend: backend,
            cacheProvider: provider, assistant: assistant, construction: work,
            mediaProcessor: processor, loadedOwner: loadedOwner, audioOwner: audioOwner,
            audioSessionID: audioSessionID, audioSourceIdentity: audioSourceIdentity,
            audioGeneration: audioGeneration,
            mtpVerificationMode: assistant?.verificationMode ?? .serialTarget,
            completePrefixCache: completePrefixCache, completePrefixValidator: validator,
            prefixProcessMemoryOwner: processMemoryOwner)
        return (backend, provider, contract)
    }

    /// Strict loaded producer only. All native handles stay inside its
    /// protected assembly scope; no raw model crosses an async boundary.
    package func validateNativePagedOwner(_ owner: AnyObject) throws {
        guard loadLifetimeOwner === owner, assistant == nil, !nativeProbeFailed,
              observedKVDTypes?.count == layerKinds.count else {
            throw MiMoV26CBv2Error.invalidInput("invalid native paged loaded owner")
        }
    }

    package func makeNativePagedResources(config: PagedKVPoolConfig,
        processMemoryOwner: any CBv2ProcessMemoryOwner, loadedOwner: AnyObject,
        validator: any CBv2NativePagedModelValidating,
        retaining work: NativeConstructionScope) throws -> MiMoV26CBv2NativePagedExecutionResources {
        guard nativePagedBinding == nil, assistant == nil, loadLifetimeOwner === loadedOwner,
              !nativeProbeFailed, let types = observedKVDTypes, config.layerDTypes == types,
              config.prefixSharingBlockSize == nil, config.segmentSizeBytes != nil,
              config.gatheredAttention?.admissionMode == .stepOwned(.pinnedMetal) else {
            throw MiMoV26CBv2Error.invalidInput("native paged target requires completed probe and exclusive target-only resources")
        }
        try work.requireImmutableLoadedOwner(loadedOwner)
        try validator.validateNativePagedModel()
        try work.retainOwner(self); try work.retainOwner(validator); try work.retainOwner(processMemoryOwner)
        try work.capture(StreamOrDevice.default.stream); try work.capture(StreamOrDevice.cpu.stream)
        let ledger = rowLedger
        let ownership = try CBv2NativePagedModelBinding(model: self, loadedOwner: loadedOwner,
            layerKinds: layerKinds, layerDTypes: types,
            maximumContextTokens: target.configuration.maxPositionEmbeddings,
            processMemoryOwner: processMemoryOwner, validator: validator, construction: work,
            registerRows: { try ledger.register($0, backend: $1) },
            rowRequest: { try ledger.identity($0, layer: $1) },
            removeRows: { try ledger.remove($0, backend: $1) })
        try work.retainOwner(ownership)
        // Capture the actual binding before a late constructor veto.
        nativePagedBinding = ownership
        try work.invalidateOnFailedCompletion(self) { [weak self] in
            self?.nativeProbeFailed = true; self?.observedKVDTypes = nil
        }
        let backend = try PagedKVBackend(layerKinds: layerKinds, config: config,
                                         nativeModelBinding: ownership)
        try work.retainOwner(backend)
        let raw = backend.makeLayerCaches()
        guard let bases = raw as? [PagedLayerCache], bases.count == layerKinds.count else {
            throw MiMoV26CBv2Error.invalidInput("native paged cache construction mismatch")
        }
        let caches = bases.map { MiMoV26PagedLayerCache(owner: cacheOwner, base: $0) }
        let bank = CBv2LayerCacheBank(caches: caches)
        try work.retainOwner(bank)
        try ownership.seal(bank: bank, caches: bases)
        try work.checkpoint("adapter.nativePagedResources")
        let contract = try CBv2NativeExecutionContract(model: self, backend: backend,
            cacheProvider: bank, assistant: nil, construction: work, loadedOwner: loadedOwner,
            nativePagedBinding: ownership, nativePagedProcessMemoryOwner: processMemoryOwner)
        return .init(backend: backend, cacheProvider: bank, contract: contract)
    }

    // Insert beside makeNativeExecutionResources in MiMoV26CBv2Adapter.
    // Package-only: the strict LoadedModel creates the exact processor, and
    // the already-active construction scope authenticates this loaded owner.
    package func makeNativeDecodedVisualResources(bytesCapacity: Int, processor: AnyObject,
        loadedOwner: AnyObject, retaining work: NativeConstructionScope) throws
        -> (backend: MiMoV26CBv2Backend, cacheProvider: any CBv2LayerCacheProvider,
            contract: CBv2NativeExecutionContract) {
        guard loadLifetimeOwner === loadedOwner else {
            throw MiMoV26CBv2Error.invalidInput("foreign decoded-media owner")
        }
        try work.requireImmutableLoadedOwner(loadLifetimeOwner)
        try work.retainOwner(self)
        try work.retainOwner(processor)
        try work.invalidateOnFailedCompletion(self) { [weak self] in
            self?.nativeProbeFailed = true; self?.observedKVDTypes = nil
        }
        try work.capture(StreamOrDevice.cpu.stream)
        try work.capture(StreamOrDevice.default.stream)
        let backend = try makeBackend(bytesCapacity: bytesCapacity)
        try work.retainOwner(backend)
        let provider = try makeMultimodalCacheProvider()
        try work.retainOwner(provider)
        if let native = provider as? MiMoV26CausalMediaCacheProvider {
            associateBlockBatch(backend:backend,provider:native,caches:native.caches)
        }
        try work.checkpoint("adapter.nativeDecodedVisualResources")
        let contract = try CBv2NativeExecutionContract(model: self, backend: backend,
            cacheProvider: provider, assistant: assistant, construction: work,
            mediaProcessor: processor, loadedOwner: loadedOwner,
            mtpVerificationMode: assistant?.verificationMode ?? .serialTarget)
        return (backend, provider, contract)
    }


    /// Package-only audio issuer. MLXVLM authenticates the concrete sidecar and
    /// authorizes its exact owner in this same protected construction scope.
    package func makeNativeDecodedAudioResources(bytesCapacity: Int, processor: AnyObject,
        loadedOwner: AnyObject, audioOwner: AnyObject, audioSessionID: UUID,
        audioSourceIdentity: String, audioGeneration: UUID, retaining work: NativeConstructionScope) throws
        -> (backend: MiMoV26CBv2Backend, cacheProvider: any CBv2LayerCacheProvider,
            contract: CBv2NativeExecutionContract) {
        guard loadLifetimeOwner === loadedOwner else {
            throw MiMoV26CBv2Error.invalidInput("foreign decoded-audio owner")
        }
        try work.requireImmutableLoadedOwner(loadLifetimeOwner)
        try work.requireImmutableLoadedOwner(audioOwner)
        try work.retainOwner(self); try work.retainOwner(processor); try work.retainOwner(audioOwner)
        try work.invalidateOnFailedCompletion(self) { [weak self] in
            self?.nativeProbeFailed = true; self?.observedKVDTypes = nil
        }
        try work.capture(StreamOrDevice.cpu.stream)
        try work.capture(StreamOrDevice.default.stream)
        let backend = try makeBackend(bytesCapacity: bytesCapacity)
        try work.retainOwner(backend)
        let provider = try makeMultimodalCacheProvider()
        try work.retainOwner(provider)
        if let native = provider as? MiMoV26CausalMediaCacheProvider {
            associateBlockBatch(backend:backend,provider:native,caches:native.caches)
        }
        try work.checkpoint("adapter.nativeDecodedAudioResources")
        let contract = try CBv2NativeExecutionContract(model: self, backend: backend,
            cacheProvider: provider, assistant: assistant, construction: work,
            mediaProcessor: processor, loadedOwner: loadedOwner, audioOwner: audioOwner,
            audioSessionID: audioSessionID, audioSourceIdentity: audioSourceIdentity,
            audioGeneration: audioGeneration,
            mtpVerificationMode: assistant?.verificationMode ?? .serialTarget)
        return (backend, provider, contract)
    }

    /// Serialized construction/warmup-only observation of this loaded target.
    /// The common probe installs its private RecordingRow, then unbinds it.
    /// No active engine may use this adapter concurrently with the probe.
    public func probeNativeKVTypes(retaining work: NativeConstructionScope) throws -> CBv2NativeKVTypeProbe.Result {
        guard nativePagedBinding == nil else { throw MiMoV26CBv2Error.invalidInput("cannot reprobe an issued paged adapter") }
        guard !nativeProbeFailed else {
            throw MiMoV26CBv2Error.invalidInput("native construction completion failed; process restart required")
        }
        guard !nativeProbeScope else { throw MiMoV26CBv2Error.invalidInput("nested native type probe") }
        do {
            return try work.withPhase(.nativeKVProbe) {
                try work.retainOwner(self)
                try work.invalidateOnFailedCompletion(self) { [weak self] in
                    self?.nativeProbeFailed = true; self?.observedKVDTypes = nil
                }
                try work.capture(StreamOrDevice.cpu.stream)
                try work.capture(StreamOrDevice.default.stream)
                observedKVDTypes = nil
                let caches = makeCaches()
                try work.retainValue(caches)
                nativeProbeCaches = Set(caches.map(ObjectIdentifier.init))
                nativeProbeScope = true
                defer {
                    if !work.snapshot.isRetainedFault {
                        nativeProbeScope = false; nativeProbeCaches.removeAll()
                    }
                }
                let result = try CBv2NativeKVTypeProbe.run(
                    model: self, layerKinds: layerKinds, caches: caches, retaining: work)
                observedKVDTypes = result.layerDTypes
                return result
            }
        } catch {
            if work.snapshot.isRetainedFault {
                nativeProbeFailed = true
                observedKVDTypes = nil
            }
            throw error
        }
    }

    /// Host-side external token validation. The engine forward intentionally
    /// trusts values from its validated prompt/sampler and never reads them back.
    public func validateTokenIDs(_ ids: [Int]) throws {
        guard !ids.isEmpty, ids.count <= target.configuration.maxPositionEmbeddings,
              ids.allSatisfy({ $0 >= 0 && $0 < target.configuration.vocabularySize }) else {
            throw MiMoV26CBv2Error.invalidInput("token IDs outside the native vocabulary/context")
        }
    }

    /// Recoverable external composition preflight. Checks all rows before the
    /// first setRows call; ordinary engine bank binding remains a trusted seam.
    public func bindRows(_ rowStates: [[CBv2SequenceKV?]], caches: [any CBv2AttendingLayerCache]) throws {
        guard observedKVDTypes != nil else { throw MiMoV26CBv2Error.nativeProbeRequired }
        let owned = try ownedCaches(caches)
        guard !rowStates.isEmpty, rowStates.allSatisfy({ $0.count == layerKinds.count }) else {
            throw MiMoV26CBv2Error.invalidInput("one state per layer and at least one row required")
        }
        var grouped: [[CBv2SequenceKV]] = []
        for index in layerKinds.indices {
            var rows: [CBv2SequenceKV] = []
            for state in rowStates {
                guard let row = state[index] else {
                    throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "missing owning row")
                }
                rows.append(row)
            }
            grouped.append(rows)
        }
        try validateRows(grouped, length: 0)
        for (cache, rows) in zip(owned, grouped) { cache.setRows(rows) }
    }

    /// Shape/owner/history-only preflight. Array token VALUES are trusted;
    /// callers accepting external IDs must run validateTokenIDs before upload.
    public func validate(tokens: MLXArray, inputEmbeddings: MLXArray? = nil,
                         caches: [any CBv2AttendingLayerCache]) throws {
        try validate(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches, allowMTPStaging: false)
    }

    private func validate(tokens: MLXArray, inputEmbeddings: MLXArray? = nil,
                          caches: [any CBv2AttendingLayerCache], allowMTPStaging: Bool) throws {
        guard !nativeProbeFailed else {
            throw MiMoV26CBv2Error.invalidInput("native construction completion failed; process restart required")
        }
        let probeRows = nativeProbeScope && Set(caches.map(ObjectIdentifier.init)) == nativeProbeCaches
        guard observedKVDTypes != nil || probeRows else { throw MiMoV26CBv2Error.nativeProbeRequired }
        guard tokens.ndim == 2, tokens.dtype == .int32 || tokens.dtype == .uint32,
              tokens.dim(0) > 0, tokens.dim(1) > 0 else {
            throw MiMoV26CBv2Error.invalidInput("tokens require nonempty rank-two int32/uint32")
        }
        if let inputEmbeddings {
            guard inputEmbeddings.shape == [tokens.dim(0), tokens.dim(1), target.configuration.hiddenSize],
                  inputEmbeddings.dtype == target.activationDType else {
                throw MiMoV26CBv2Error.invalidInput("native input embedding shape/dtype mismatch")
            }
        }
        guard target.hasLoadedEmbeddingPrecision, target.hasLoadedReadoutPrecision else {
            throw MiMoV26CBv2Error.unloadedPrecision
        }
        let owned = try ownedCaches(caches)
        for cache in owned {
            guard cache.rows.count == tokens.dim(0), cache.positionOffsets.shape == [tokens.dim(0)],
                  cache.positionOffsets.dtype == .int32,
                  cache.rows.map(\.absoluteOffset) == cache.boundOffsets else {
                throw MiMoV26CBv2Error.invalidCache(layer: cache.layerIndex,
                    reason: "batch or position frontier changed without cache rebinding")
            }
        }
        try validateRows(owned.map(\.rows), length: tokens.dim(1), allowProbeRows: probeRows,
                         allowMTPStaging: allowMTPStaging)
    }

    private func ownedCaches(_ caches: [any CBv2AttendingLayerCache]) throws -> [any MiMoV26OwnedLayerCache] {
        guard caches.count == layerKinds.count else { throw MiMoV26CBv2Error.invalidInput("one native cache per layer required") }
        var result: [any MiMoV26OwnedLayerCache] = [], identities = Set<ObjectIdentifier>()
        for (index, cache) in caches.enumerated() {
            guard let owned = cache as? any MiMoV26OwnedLayerCache,
                  (nativePagedBinding == nil ? owned is MiMoV26ContiguousLayerCache : owned is MiMoV26PagedLayerCache),
                  owned.owner == cacheOwner, owned.layerIndex == index, owned.kind == layerKinds[index],
                  identities.insert(ObjectIdentifier(owned)).inserted else {
                throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "foreign, reordered or aliased cache owner")
            }
            result.append(owned)
        }
        return result
    }

    private func validateRows(_ rowsByLayer: [[CBv2SequenceKV]], length: Int, allowProbeRows: Bool = false,
                              allowMTPStaging: Bool = false) throws {
        let maximum = target.configuration.maxPositionEmbeddings
        guard length >= 0, length <= maximum else { throw MiMoV26CBv2Error.contextExceeded }
        let batch = rowsByLayer.first?.count ?? 0
        let offsets = rowsByLayer.first?.map(\.absoluteOffset) ?? []
        var identities = Set<ObjectIdentifier>()
        var requestIdentities = Array<UUID?>(repeating: nil, count: batch)
        for (index, rows) in rowsByLayer.enumerated() {
            let kind = layerKinds[index]
            guard rows.count == batch else { throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "row count differs across layers") }
            for (rowIndex, row) in rows.enumerated() {
                let pagedMetadata: CBv2NativePagedRowMetadata?
                if row is PagedSequenceKV {
                    guard !allowProbeRows, let nativePagedBinding else {
                        throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "unissued paged row")
                    }
                    pagedMetadata = try nativePagedBinding.metadata(row: row, layer: index)
                } else { pagedMetadata = nil }
                guard pagedMetadata != nil || row is CBv2FullSequenceKV || row is CBv2WindowedSequenceKV
                        || (allowProbeRows && !(row is PagedSequenceKV) && !(row is CBv2FrozenReplayFullSequenceKV)) else {
                    throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "unqualified row backend")
                }
                if !allowProbeRows {
                    let request = try pagedMetadata?.request ?? rowLedger.identity(row, layer: index)
                    guard requestIdentities[rowIndex] == nil || requestIdentities[rowIndex] == request else {
                        throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "different requests spliced across layers")
                    }
                    requestIdentities[rowIndex] = request
                }
                guard identities.insert(ObjectIdentifier(row)).inserted else {
                    throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "row aliases another layer/request slot")
                }
                let offset = row.absoluteOffset
                guard offset >= 0, offset <= maximum - length else { throw MiMoV26CBv2Error.contextExceeded }
                guard offset == offsets[rowIndex] else {
                    throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "request history differs across layers")
                }
                if let full = row as? CBv2FullSequenceKV {
                    guard case .full = kind.attention, full.maxLength >= offset + length else {
                        throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "full row policy/capacity mismatch")
                    }
                }
                if let window = row as? CBv2WindowedSequenceKV {
                    guard kind.attention == .slidingWindow(window.window),
                          window.retainedCount >= min(offset, window.window),
                          window.retainedCount <= min(offset, window.window + (allowMTPStaging ? 3 : 0)) else {
                        throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "sliding row policy mismatch")
                    }
                }
                if let pagedMetadata {
                    guard pagedMetadata.offset == offset,
                          pagedMetadata.retainedCount == row.retainedCount,
                          pagedMetadata.maximumLength >= offset + length,
                          pagedMetadata.keyWidth == kind.headDim,
                          pagedMetadata.valueWidth == kind.valueHeadDim,
                          pagedMetadata.kvHeads == kind.kvHeads,
                          pagedMetadata.dtype == observedKVDTypes?[index],
                          pagedMetadata.oldestValidPosition <= offset - row.retainedCount else {
                        throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "native paged scalar geometry/history mismatch")
                    }
                    continue // NEVER gather/snapshot merely to validate ownership.
                }
                // Public snapshots expose only metadata here: no eval/item or
                // data copy. Do not retain them or derive positions from data.
                // The real native type probe uses its private RecordingRow,
                // whose empty placeholders are FP16 before its first write.
                let snapshot = row.snapshot()
                guard snapshot.offset == offset, snapshot.keys.ndim == 4, snapshot.values.ndim == 4,
                      snapshot.keys.shape == [1, kind.kvHeads, row.retainedCount, kind.headDim],
                      snapshot.values.shape == [1, kind.kvHeads, row.retainedCount, kind.valueHeadDim] else {
                    throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "native K/V snapshot geometry mismatch")
                }
                if row.retainedCount > 0 {
                    let expected = observedKVDTypes?[index] ?? target.activationDType
                    let validDType = allowProbeRows
                        ? snapshot.keys.dtype == snapshot.values.dtype
                            && [.float16, .bfloat16, .float32].contains(snapshot.keys.dtype)
                        : snapshot.keys.dtype == expected && snapshot.values.dtype == expected
                    guard validDType else {
                        throw MiMoV26CBv2Error.invalidCache(layer: index, reason: "native K/V storage dtype mismatch")
                    }
                }
            }
        }
    }

    public func forwardValidated(tokens: MLXArray, inputEmbeddings: MLXArray? = nil,
                                 caches: [any CBv2AttendingLayerCache]) throws -> MLXArray {
        try validate(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
        let hidden = trunk(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
        return target.lmHead.map { $0(hidden) } ?? target.model.embedTokens.asLinear(hidden)
    }

    public func prefillValidated(tokens: MLXArray, inputEmbeddings: MLXArray? = nil,
                                 caches: [any CBv2AttendingLayerCache],
                                 requirement: CBv2PrefillRequirement) throws -> MLXArray {
        try validate(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
        let hidden = trunk(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
        switch requirement {
        case .evaluationOnly:
            // Full-trunk reduction proves all rows participate; no vocabulary
            // projection is constructed for an intermediate prompt chunk.
            return depends(input: hidden.sum(axes: [1, 2]).reshaped(-1, 1),
                           dependencies: cacheRoots(caches))
        case .lastPositionLogits:
            let last = hidden[0..., -1, 0...]
            let logits = target.lmHead.map { $0(last) } ?? target.model.embedTokens.asLinear(last)
            return depends(input: logits, dependencies: cacheRoots(caches))
        }
    }

    public func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache]) -> MLXArray {
        do { return try forwardValidated(tokens: tokens, caches: caches) }
        catch { preconditionFailure("MiMo CBv2 validated engine state invariant: \(error)") }
    }
    public func prefill(tokens: MLXArray, inputEmbeddings: MLXArray?,
                        caches: [any CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement) -> MLXArray {
        do { return try prefillValidated(tokens: tokens, inputEmbeddings: inputEmbeddings,
                                         caches: caches, requirement: requirement) }
        catch { preconditionFailure("MiMo CBv2 validated prefill state invariant: \(error)") }
    }

    public var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
    public var supportsRequestStatefulMTP: Bool { assistant?.isCompatible(with: target) == true }
    public var mtpTargetIdentity: ObjectIdentifier? {
        supportsRequestStatefulMTP ? ObjectIdentifier(target) : nil
    }
    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        cbv2TopTwoRows(logits)
    }

    /// MiMo's native heads consume the target's POST-final-norm features.
    /// Logits and target KV use the identical ordinary trunk and readout.
    public func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray) {
        do {
            guard supportsRequestStatefulMTP else { throw MiMoV26MTPError.weightsNotLoaded }
            try validate(tokens: tokens, caches: caches, allowMTPStaging: true)
            let rowLocalDense = MiMoV26RectangularDense.eligible(tokens: tokens, caches: caches,
                requested: useRowLocalRectangularDense
                    && MiMoV26RectangularDenseAdmission.isActive(for: self),
                fusedNorms: target.model.useFusedDecodeNorms)
            if rowLocalDense { MiMoV26RectangularDenseAdmission.recordSubmission(for: self) }
            let hidden = trunk(tokens: tokens, inputEmbeddings: nil, caches: caches,
                               rowLocalDense: rowLocalDense)
            return (MiMoV26RectangularDense.readout(target, hidden, enabled: rowLocalDense), hidden)
        } catch { preconditionFailure("MiMo CBv2 hidden contract: \(error)") }
    }

    public func forwardWithHiddenForPrefill(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
                                            requirement: CBv2PrefillRequirement)
        -> (logits: MLXArray, lastHidden: MLXArray) {
        do {
            guard supportsRequestStatefulMTP else { throw MiMoV26MTPError.weightsNotLoaded }
            try validate(tokens: tokens, caches: caches)
            let hidden = trunk(tokens: tokens, inputEmbeddings: nil, caches: caches)
            switch requirement {
            case .evaluationOnly:
                return (depends(input: hidden.sum(axes: [1, 2]).reshaped(-1, 1),
                                dependencies: cacheRoots(caches)), hidden)
            case .lastPositionLogits:
                let last = hidden[0..., -1, 0...]
                let logits = target.lmHead.map { $0(last) } ?? target.model.embedTokens.asLinear(last)
                return (depends(input: logits, dependencies: cacheRoots(caches)), hidden)
            }
        } catch { preconditionFailure("MiMo CBv2 hidden prefill contract: \(error)") }
    }

    private func cacheRoots(_ caches: [any CBv2AttendingLayerCache]) -> [MLXArray] {
        caches.flatMap { ($0 as? KVCache)?.innerState() ?? [] }
    }
    private func trunk(tokens: MLXArray, inputEmbeddings: MLXArray?,
                       caches: [any CBv2AttendingLayerCache], rowLocalDense: Bool = false) -> MLXArray {
        var hidden = inputEmbeddings ?? target.model.embedTokens(tokens)
        var nextInput: MLXArray?
        for (index, layer) in target.model.layers.enumerated() {
            let attention = layer.selfAttention
            let normalized = nextInput ?? layer.inputNorm(hidden)
            let (batch, length) = (hidden.dim(0), hidden.dim(1))
            let geometry = attention.geometry
            var q = MiMoV26RectangularDense.projection(attention.qProj, normalized, enabled: rowLocalDense)
                .reshaped(batch, length, geometry.queryHeads, geometry.headDim).transposed(0, 2, 1, 3)
            var k = MiMoV26RectangularDense.projection(attention.kProj, normalized, enabled: rowLocalDense)
                .reshaped(batch, length, geometry.keyValueHeads, geometry.headDim).transposed(0, 2, 1, 3)
            let v = MiMoV26RectangularDense.projection(attention.vProj, normalized, enabled: rowLocalDense)
                .reshaped(batch, length, geometry.keyValueHeads, geometry.valueHeadDim)
                .transposed(0, 2, 1, 3) * attention.valueScale
            let offsets = caches[index].positionOffsets + 0
            q = attention.rope(q, offset: offsets)
            k = attention.rope(k, offset: offsets)
            let output = caches[index].updateAndAttend(queries: q, keys: k, values: v,
                scale: attention.scale, sinks: attention.attentionSinkBias)
            let projected = MiMoV26RectangularDense.projection(attention.oProj,
                output.transposed(0, 2, 1, 3).reshaped(batch, length, -1), enabled: rowLocalDense)
            let nextNorm = index + 1 < target.model.layers.count
                ? target.model.layers[index + 1].inputNorm : target.model.norm
            if let fused = MiMoV26DecodeKernels.finishLayer(
                hidden, attentionOutput: projected, layer: layer, nextNorm: nextNorm,
                enabled: target.model.useFusedDecodeNorms) {
                hidden = fused.residual
                nextInput = fused.normalized
            } else {
                let residual = hidden + projected
                hidden = residual + MiMoV26RectangularDense.mlp(
                    layer.mlp, layer.postAttentionNorm(residual), enabled: rowLocalDense)
                nextInput = nil
            }
        }
        // This remains target POST-final-norm for every MTP consumer.
        return nextInput ?? target.model.norm(hidden)
    }
}

private final class MiMoV26CausalMediaCacheProvider: CBv2MultimodalAttentionCapabilityProviding,
                                                  CBv2CompositionInvalidating {
    private let adapter: MiMoV26CBv2Adapter
    private let bank: CBv2LayerCacheBank
    fileprivate let caches: [any CBv2AttendingLayerCache]
    init(adapter: MiMoV26CBv2Adapter) {
        self.adapter = adapter
        caches = adapter.makeCaches()
        bank = CBv2LayerCacheBank(caches: caches)
    }
    func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
        adapter.supportsMultimodalPrefill(attention: attention)
    }
    // This admits only the existing eligible text rounds. Media request
    // exclusion, all-row scheduling costs and native head B=1 stay unchanged.
    var supportsMTPRectangularVerification: Bool {
        bank.supportsMTPRectangularVerification
    }
    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [any CBv2AttendingLayerCache] {
        do { try adapter.bindRows(rowStates, caches: caches) }
        catch { preconditionFailure("MiMo media provider row ownership: \(error)") }
        return bank.layerCaches(rowStates: rowStates)
    }
    func invalidateBoundComposition() { bank.invalidateBoundComposition() }
    func releaseBoundRows() { bank.releaseBoundRows() }
}
