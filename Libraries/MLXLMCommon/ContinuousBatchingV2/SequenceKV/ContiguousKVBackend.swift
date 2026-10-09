// ContiguousKVBackend.swift
//
// The v1 `CBv2KVBackend`: per-sequence contiguous MLX buffers
// (`CBv2FullSequenceKV` / `CBv2WindowedSequenceKV`).
// The paged backend (workstream C) implements the same protocol behind a
// Metal kernel; the scheduler and models never see the difference.

import Foundation
import MLX

/// Configuration for `CBv2ContiguousKVBackend`.
public struct CBv2ContiguousBackendConfig: Sendable {
    /// Byte budget for all live sequence KV (admission ceiling). This is
    /// the INITIAL budget; the backend's live ceiling can be re-sliced at
    /// runtime via `CBv2ContiguousKVBackend.updateBytesCapacity(_:)`.
    public var bytesCapacity: Int
    /// dtype assumed for admission estimates (actual allocation adopts the
    /// dtype of the first appended K/V).
    public var kvDType: DType
    /// Grow fresh sliding-window buffers with retained tokens. Complete
    /// checkpoint restoration retains its existing full-ring allocation.
    public var elasticWindowStorage: Bool

    public init(
        bytesCapacity: Int,
        kvDType: DType = .float16,
        elasticWindowStorage: Bool = false
    ) {
        self.bytesCapacity = bytesCapacity
        self.kvDType = kvDType
        self.elasticWindowStorage = elasticWindowStorage
    }
}

/// Factory + accounting for per-sequence contiguous KV state.
///
/// Thread-safe: the live-row registry is lock-protected (`makeSequenceState`
/// runs on the admission path while `release` runs on the engine loop).
/// `bytesInUse` sums live logical array extents. `bytesReserved` additionally
/// retains v2 imported allocator bounds; logical nbytes are not a physical
/// allocator receipt or materialization credit.
///
/// Admission RESERVES: rows allocate lazily (`byteCount == 0` until their
/// first update), so judging capacity against `bytesInUse` alone would let
/// several same-step admissions collectively exceed `bytesCapacity`. Each
/// admitted row therefore holds a reservation equal to its estimated
/// initial bytes until its actual allocation exceeds it
/// (`max(byteCount, reservation)` per row — see `bytesReserved`), and the
/// capacity check + registration are a single atomic section.
public final class CBv2ContiguousKVBackend: CBv2KVBackend {

    public let config: CBv2ContiguousBackendConfig
    public var prefixReuseBackend: CBv2PrefixReuseBackend { .contiguousUnquantized }

    private let lock = NSLock()
    private var live: [ObjectIdentifier: CBv2SequenceKV] = [:]
    /// Admission reservation per live row (estimated initial bytes),
    /// released with the row. NOTE: estimates assume `config.kvDType`; a
    /// model that caches wider elements (e.g. fp32) under-reserves until
    /// the first update trues the row up to its actual `byteCount` —
    /// `AdmissionV2` (which can carry per-layer element sizes) remains the
    /// primary admission gate.
    private var reservations: [ObjectIdentifier: Int] = [:]
    /// Live byte budget, seeded from `config.bytesCapacity` and resizable
    /// at runtime (`updateBytesCapacity`). Lock-protected: the atomic
    /// admit-and-register check reads it inside its critical section.
    private var liveBytesCapacity: Int
    // Deterministic failure seam after ledger transfer, before row publication.
    var checkpointBeforeRegistration: (() throws -> Void)?

    public init(config: CBv2ContiguousBackendConfig) {
        self.config = config
        self.liveBytesCapacity = config.bytesCapacity
    }

    public var bytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return liveBytesCapacity
    }

    /// Runtime capacity update (multi-model co-residency re-slicing).
    /// Shrink never evicts live rows: registrations above a new lower
    /// ceiling stay resident and new admissions fail until usage drains
    /// below the new ceiling; grow admits immediately
    /// (`CBv2KVBackend.updateBytesCapacity`).
    public func updateBytesCapacity(_ bytes: Int) {
        lock.lock()
        liveBytesCapacity = max(0, bytes)
        lock.unlock()
    }

    public var bytesInUse: Int {
        lock.lock()
        defer { lock.unlock() }
        return live.values.reduce(0) { CBv2KVGeometry.add($0, $1.byteCount) ?? Int.max }
    }

    /// Actual bytes plus outstanding admission reservations — what the
    /// capacity check judges against (`CBv2KVBackend.bytesReserved`).
    public var bytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return accountedBytesLocked()
    }

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard promptLength >= 0, maxLength > 0, maxLength <= Int(Int32.max),
            promptLength <= maxLength
        else {
            throw CBv2KVError.backendIneligible(
                reason: "promptLength \(promptLength) exceeds maxLength \(maxLength)")
        }
        let estimates = try rowEstimates(
            layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength)
        let state = layerKinds.map { kind -> CBv2SequenceKV? in
            makeRow(kind: kind, promptLength: promptLength, maxLength: maxLength)
        }
        try registerReserving(
            state,
            estimates: estimates)
        return state
    }

    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard maxLength > 0, maxLength <= Int(Int32.max) else {
            throw CBv2KVError.backendIneligible(reason: "invalid prefix maximum length")
        }
        guard prefix.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix count \(prefix.count) != layer count \(layerKinds.count)")
        }

        guard plan.backend == prefixReuseBackend else {
            throw CBv2KVError.backendIneligible(
                reason:
                    "prefix plan backend \(plan.backend.rawValue) != \(prefixReuseBackend.rawValue)"
            )
        }
        guard plan.matchedBoundary <= maxLength,
            plan.replayStart >= 0,
            plan.replayStart <= plan.matchedBoundary,
            plan.replayTokens == plan.matchedBoundary - plan.replayStart
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid prefix replay plan")
        }

        // Full snapshots use either C (ordinary safe-layout replay) or M
        // (frozen-full replay). Every owning full row must agree.
        let expectedSnapshotOffset = plan.restoredFullTokens
        guard expectedSnapshotOffset >= 0, expectedSnapshotOffset <= maxLength,
            plan.strategy != .frozenFullReplay || plan.replayStart < expectedSnapshotOffset
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid frozen/full snapshot boundary")
        }
        var sawOwningFull = false
        for (index, entry) in prefix.enumerated() {
            let kind = layerKinds[index]
            if kind.sharesKVWithLayer != nil {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) is KV-shared but received a prefix snapshot")
                }
                continue
            }
            if case .slidingWindow = kind.attention {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index) is windowed but received a prefix snapshot")
                }
                continue
            }
            sawOwningFull = true
            guard let entry else {
                throw CBv2KVError.backendIneligible(
                    reason: "owning full layer \(index) is missing its prefix snapshot")
            }
            guard kind.sharesKVWithLayer == nil else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index) is KV-shared but received a prefix snapshot")
            }
            guard case .full = kind.attention else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "layer \(index) is windowed but received a prefix snapshot (windowed layers are recomputed)"
                )
            }
            guard entry.offset == expectedSnapshotOffset else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "prefix offset \(entry.offset) != planned \(expectedSnapshotOffset) at layer \(index)"
                )
            }
            guard entry.keys.shape == [1, kind.kvHeads, entry.offset, kind.headDim],
                entry.values.shape == [1, kind.kvHeads, entry.offset, kind.valueHeadDim],
                entry.keys.dtype == entry.values.dtype,
                [.float16, .bfloat16, .float32].contains(entry.keys.dtype)
            else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "full prefix snapshot at layer \(index) has incompatible shape, dtype or offset"
                )
            }
        }
        guard sawOwningFull else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix replay requires at least one storage-owning full layer")
        }

        let estimates = try adoptionRowEstimates(
            prefix: prefix, layerKinds: layerKinds, maxLength: maxLength)
        let state = layerKinds.enumerated().map { index, kind -> CBv2SequenceKV? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                // Sliding rows always start empty at C and rebuild through R.
                return CBv2WindowedSequenceKV(
                    window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                    valueHeadDim: kind.valueHeadDim,
                    initialOffset: plan.replayStart,
                    elasticStorage: config.elasticWindowStorage, maximumSequenceLength: maxLength)
            case .full:
                let entry = prefix[index]!
                if plan.strategy == .frozenFullReplay {
                    return CBv2FrozenReplayFullSequenceKV(
                        snapshot: entry,
                        replayStart: plan.replayStart,
                        maxLength: maxLength,
                        kvHeads: kind.kvHeads,
                        headDim: kind.headDim, valueHeadDim: kind.valueHeadDim)
                }
                let row = makeRow(
                    kind: kind,
                    promptLength: expectedSnapshotOffset,
                    maxLength: maxLength)!
                _ = row.update(keys: entry.keys, values: entry.values)
                return row
            }
        }
        try registerReserving(
            state,
            estimates: estimates)
        return state
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        lock.lock()
        defer { lock.unlock() }
        for row in state {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live.removeValue(forKey: key)
            reservations.removeValue(forKey: key)
        }
    }

    /// Prepared rows were filled under an external stage reservation. Register
    /// their exact final allocation atomically before that reservation ends.
    func adoptPreparedCheckpoint(_ state: [CBv2SequenceKV?]) throws {
        guard !state.isEmpty, state.allSatisfy({ $0 is CBv2FullSequenceKV }) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try registerReserving(state, estimates: state.map { $0?.byteCount })
    }

    /// New layout only. Legacy adoption remains unchanged and full-row-only.
    func adoptPreparedCheckpoint(
        _ state: [CBv2SequenceKV?], codec: CBv2CompleteCheckpointCodec, position: Int,
        requestID: CBv2RequestID, maximumSequenceLength: Int
    ) throws {
        try codec.validateContiguousRows(state, position: position, exactWindow: true)
        func backing(_ row: CBv2SequenceKV) -> CBv2ContiguousCheckpointBacking? {
            (row as? CBv2FullSequenceKV)?.checkpointBacking
                ?? (row as? CBv2WindowedSequenceKV)?.checkpointBacking
        }
        guard let owner = state.compactMap({ $0 }).first.flatMap(backing), owner.requestID == nil,
            owner.lease.admission === codec.admission,
            owner.allocationBoundsByLayer.count == state.count,
            state.compactMap({ $0 }).allSatisfy({
                $0.absoluteOffset == position && backing($0) === owner
            })
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        lock.lock()
        defer { lock.unlock() }
        var total = 0
        for (i, row) in state.enumerated() {
            guard let row else {
                guard owner.allocationBoundsByLayer[i] == nil else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                continue
            }
            guard live[ObjectIdentifier(row)] == nil, let bound = owner.allocationBoundsByLayer[i],
                bound >= row.byteCount,
                let sum = CBv2KVGeometry.add(total, bound)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            if let full = row as? CBv2FullSequenceKV {
                guard full.maxLength == maximumSequenceLength else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
            }
            total = sum
        }
        let available = max(0, liveBytesCapacity - accountedBytesLocked())
        guard total <= available else {
            throw CBv2KVError.capacityExhausted(needed: total, available: available)
        }
        let ticket = try codec.admission.transferContiguousCheckpointStage(
            owner.lease,
            requestID: requestID, maximumTokens: maximumSequenceLength)
        owner.arm(requestID: requestID)
        do {
            try checkpointBeforeRegistration?()
            for (i, row) in state.enumerated() {
                guard let row else { continue }
                let key = ObjectIdentifier(row)
                live[key] = row
                reservations[key] = owner.allocationBoundsByLayer[i]!
            }
            ticket.commit()
            owner.finishAdoption()
        } catch {
            // No row was published. The owner retains C through any remaining
            // prepare/diagnostic aliases, then resolves rollback exactly once.
            owner.abandonAdoption(ticket)
            throw error
        }
    }

    // MARK: - Private

    /// Bytes currently charged against capacity: each live row counts for
    /// the LARGER of its actual allocation and its outstanding admission
    /// reservation, so a not-yet-allocated row still occupies its estimate
    /// and a grown row is charged its true size (`bytesInUse`-truthful).
    /// Caller holds `lock`.
    private func accountedBytesLocked() -> Int {
        live.reduce(0) { total, entry in
            CBv2KVGeometry.add(total, max(entry.value.byteCount, reservations[entry.key] ?? 0))
                ?? Int.max
        }
    }

    /// Atomically admit + register: the capacity check and the reservation
    /// write share one critical section so N same-step admissions cannot
    /// collectively overshoot `bytesCapacity` (the pre-fix bug — rows report
    /// `byteCount == 0` until first update).
    private func registerReserving(
        _ state: [CBv2SequenceKV?], estimates: [Int?], requireFreshOwnership: Bool = false
    ) throws {
        precondition(state.count == estimates.count, "estimate/state count mismatch")
        lock.lock()
        defer { lock.unlock() }
        if requireFreshOwnership {
            let identifiers = state.compactMap { $0 }.map(ObjectIdentifier.init)
            guard Set(identifiers).count == identifiers.count,
                identifiers.allSatisfy({ live[$0] == nil })
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
        var needed = 0
        for (row, estimate) in zip(state, estimates) {
            guard let row else { continue }
            guard let sum = CBv2KVGeometry.add(needed, max(row.byteCount, estimate ?? 0)) else {
                throw CBv2KVError.backendIneligible(reason: "contiguous reservation overflow")
            }
            needed = sum
        }
        let available = liveBytesCapacity - accountedBytesLocked()
        guard needed <= available else {
            throw CBv2KVError.capacityExhausted(needed: needed, available: max(0, available))
        }
        for (row, estimate) in zip(state, estimates) {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live[key] = row
            reservations[key] = estimate ?? 0
        }
    }

    private func makeRow(kind: CBv2LayerKind, promptLength: Int, maxLength: Int)
        -> CBv2SequenceKV?
    {
        guard kind.sharesKVWithLayer == nil else { return nil }
        switch kind.attention {
        case .slidingWindow(let window):
            return CBv2WindowedSequenceKV(
                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                valueHeadDim: kind.valueHeadDim, elasticStorage: config.elasticWindowStorage,
                maximumSequenceLength: maxLength)
        case .full:
            return CBv2FullSequenceKV(
                promptLength: promptLength, maxLength: maxLength,
                kvHeads: kind.kvHeads, headDim: kind.headDim, valueHeadDim: kind.valueHeadDim)
        }
    }

    private func validate(layerKinds: [CBv2LayerKind]) throws {
        guard config.bytesCapacity >= 0, [.float16, .bfloat16, .float32].contains(config.kvDType)
        else {
            throw CBv2KVError.backendIneligible(
                reason: "invalid contiguous native dtype or capacity")
        }
        for (index, kind) in layerKinds.enumerated() {
            guard kind.kvGeometry != nil, kind.queryHeads > 0, kind.queryHeads <= Int(Int32.max),
                kind.queryHeads.isMultiple(of: kind.kvHeads)
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): invalid native K/V geometry")
            }
            if case .slidingWindow(let window) = kind.attention,
                window <= 0 || window > Int(Int32.max)
            {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): non-positive window \(window)")
            }
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count, source != index else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): invalid KV-share source \(source)")
                }
                guard layerKinds[source].sharesKVWithLayer == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index): KV-share source \(source) is itself a shared layer")
                }
                let owner = layerKinds[source]
                guard owner.kvHeads == kind.kvHeads, owner.headDim == kind.headDim,
                    owner.valueHeadDim == kind.valueHeadDim, owner.attention == kind.attention
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): KV borrower geometry differs from owner")
                }
            }
            // The v1 contiguous backend attends through MLXFast SDPA, which
            // supports attention sinks natively — sink models are eligible.
        }
    }

    /// Per-layer estimated initial allocation bytes, aligned to `layerKinds`
    /// (nil for KV-shared layers, which own no storage). Full layers allocate
    /// `promptLength + 256` slots capped at maxLength; windowed layers
    /// allocate their full ring up front. The sum is the reservation charged
    /// against capacity at admission.
    private func rowEstimates(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [Int?] {
        let itemSize = config.kvDType.size
        return try layerKinds.map { kind -> Int? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            let slots: Int
            let maximumStoredTokens: Int
            switch kind.attention {
            case .slidingWindow(let window):
                slots = window
                maximumStoredTokens = window
            case .full:
                slots = min(maxLength, max(1, promptLength + CBv2FullSequenceKV.initialSlack))
                maximumStoredTokens = maxLength
            }
            // Prove both initial and eventual row allocations representable.
            guard let geometry = kind.kvGeometry,
                geometry.storageBytes(tokens: maximumStoredTokens, elementBytes: itemSize) != nil,
                let bytes = geometry.storageBytes(tokens: slots, elementBytes: itemSize)
            else {
                throw CBv2KVError.backendIneligible(reason: "contiguous row byte geometry overflow")
            }
            return bytes
        }
    }

    /// Adoption transfers native-dtype full rows from staging. Reserve their
    /// full request span before publication so later capacity growth cannot
    /// outrun the backend hard ceiling. Sliding rows retain their fixed ring
    /// estimate.
    private func adoptionRowEstimates(
        prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind],
        maxLength: Int
    ) throws -> [Int?] {
        try layerKinds.enumerated().map { index, kind in
            guard kind.sharesKVWithLayer == nil else { return nil }
            let tokens: Int
            let elementBytes: Int
            switch kind.attention {
            case .slidingWindow(let window):
                tokens = window
                elementBytes = config.kvDType.size
            case .full:
                guard let entry = prefix[index] else { return 0 }
                tokens = maxLength
                elementBytes = entry.keys.dtype.size
            }
            guard
                let bytes = kind.kvGeometry?.storageBytes(
                    tokens: tokens, elementBytes: elementBytes)
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "contiguous adopted row byte geometry overflow")
            }
            return bytes
        }
    }
}
