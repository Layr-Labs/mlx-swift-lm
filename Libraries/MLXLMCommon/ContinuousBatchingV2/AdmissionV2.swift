// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: truthful admission with soft reservations.
//
// Submit-time check: could the request EVER fit (worst case promptLen +
// maxTokens, window-capped per layer) in total KV capacity? Step-time check:
// vLLM-style optimism — admit while `bytesReserved + nextStepNeed <
// capacity - watermark`; preemption is the backstop when optimism loses.

import Foundation
import MLX

// MARK: - Capacity oracle (scheduler ↔ admission)

/// Soft KV-capacity oracle consulted by `SchedulerV2.plan()` for every
/// assignment. Implemented by `AdmissionV2`; tests use scripted fakes.
/// (WS-B-internal — not part of the frozen contract; see
/// docs/engine-v2/CONTRACT-ISSUES-B-scheduler.md §6.)
public protocol CBv2StepCapacity: AnyObject {
    /// Reserve headroom for `additionalTokens` more KV entries of request
    /// `id`. Throws `CBv2KVError.capacityExhausted` when the soft ledger
    /// would cross the watermark — the scheduler preempts in response.
    func reserve(id: CBv2RequestID, additionalTokens: Int) throws
    /// Reserve token-derived bytes plus an exact native-dtype adjustment in
    /// one admission operation.
    func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws
    /// Partially undo a reservation (optimistic-advance rollback).
    func unreserve(id: CBv2RequestID, tokens: Int)
    func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int)
    /// Drop every reservation held by `id` (finish/cancel/preempt).
    func releaseAll(id: CBv2RequestID)
    /// Non-throwing conservative probe used by the chained-decode fast path.
    func hasHeadroom(additionalTokens: Int) -> Bool
    /// The ledger's current byte ceiling (runtime-resizable). Feeds the
    /// engine's capacity snapshot so re-slices read back consistently.
    /// Defaulted to 0 (= unknown) so simple test capacities need not
    /// implement it.
    var bytesCapacity: Int { get }
}

extension CBv2StepCapacity {
    public var bytesCapacity: Int { 0 }
    public func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws {
        guard additionalBytes == 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "capacity oracle cannot reserve exact native KV bytes")
        }
        try reserve(id: id, additionalTokens: additionalTokens)
    }
    public func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int) {
        precondition(bytes == 0, "capacity oracle cannot unreserve exact native KV bytes")
        unreserve(id: id, tokens: tokens)
    }
}

// MARK: - AdmissionV2

/// Byte-ledger admission controller. Thread-safe: `canEverFit` runs on the
/// submitter's thread while reserve/release run on the engine thread.
///
/// Bytes are estimated from `CBv2LayerKind`: layers that share KV storage
/// (`sharesKVWithLayer != nil`) own no bytes; sliding-window layers plateau
/// at `window` tokens — so a long-context request on Gemma-style hybrids is
/// charged truthfully, not as if every layer were full-attention.
public final class AdmissionV2: CBv2StepCapacity, @unchecked Sendable {
    public struct Config: Sendable {
        /// Fraction of capacity kept free as the optimism watermark.
        public var watermarkFraction: Double
        /// Default bytes per KV element (2 = fp16/bf16) — the fallback for
        /// layers not listed in `layerElementBytes`.
        public var elementBytes: Int
        /// OPTIONAL per-layer bytes-per-element (aligned to `layerKinds`),
        /// for models whose layers cache K/V at different precisions —
        /// e.g. GPT-OSS full-attention layers cache fp32 K/V (YarnRoPE
        /// computes in fp32) while sliding layers cache bf16. Assuming a
        /// flat 2 bytes/element there under-charges the fp32 rows ~2x and
        /// over-admits. Derive it from the model's probed cache dtypes
        /// (the compiled decode path probes per-layer dtypes at warmup —
        /// see `CBv2CompiledDecode`) via `Config.elementBytes(forDTypes:)`.
        /// nil ⇒ uniform `elementBytes`. Entries for KV-shared layers are
        /// ignored (those layers own no storage).
        public var layerElementBytes: [Int]?
        public init(
            watermarkFraction: Double = 0.05, elementBytes: Int = 2,
            layerElementBytes: [Int]? = nil
        ) {
            self.watermarkFraction = watermarkFraction
            self.elementBytes = elementBytes
            self.layerElementBytes = layerElementBytes
        }

        /// Build a per-layer element-bytes table from probed cache dtypes
        /// (nil entries — KV-shared layers, unprobed — fall back to
        /// `defaultElementBytes`). Conservative: a wider dtype is charged
        /// its full element size.
        public static func elementBytes(
            forDTypes dtypes: [DType?], defaultElementBytes: Int = 2
        ) -> [Int] {
            dtypes.map { $0.map(\.size) ?? defaultElementBytes }
        }
    }

    private let lock = NSLock()
    private let layerKinds: [CBv2LayerKind]
    /// Resolved bytes-per-element per layer (aligned to `layerKinds`).
    private let perLayerElementBytes: [Int]
    /// Watermark fraction retained so `updateBytesCapacity` can recompute
    /// `watermark` against the new capacity.
    private let watermarkFraction: Double
    /// Lock-protected: recomputed whenever the capacity changes.
    private var watermark: Int
    /// Per-token bytes if every storage-owning layer retained the token
    /// (upper bound; used for the conservative headroom probe).
    private let maxPerTokenBytes: Int

    /// Total KV byte budget. Runtime-resizable via `updateBytesCapacity`
    /// (multi-model co-residency re-slicing); reads take the ledger lock.
    public var bytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return _bytesCapacity
    }
    private var _bytesCapacity: Int

    /// Bytes carved out of the budget for an EXTERNAL worst-case obligation
    /// the ledger cannot see per-request — today the compiled decode path's
    /// padding reserve. REFUNDABLE: if the obligation disappears (compiled
    /// decode disables itself at warmup after a trace failure), the engine
    /// calls `refundExternalReserve()` so admission is not permanently
    /// tighter than the hardware truth (PR#62 review). Lock-protected.
    private var externalReserveBytes: Int

    /// Cumulative reserved tokens per request (window capping is applied when
    /// converting to bytes, so decode reservations past a layer's window add
    /// zero bytes for that layer).
    private var reservedTokens: [CBv2RequestID: Int] = [:]
    private var reservedExactBytes: [CBv2RequestID: Int] = [:]
    private var ledgerBytes = 0

    public init(
        layerKinds: [CBv2LayerKind], bytesCapacity: Int, config: Config = .init(),
        externalReserveBytes: Int = 0
    ) {
        self.layerKinds = layerKinds
        self._bytesCapacity = bytesCapacity
        self.externalReserveBytes = max(0, externalReserveBytes)
        if let table = config.layerElementBytes {
            precondition(
                table.count == layerKinds.count,
                "AdmissionV2: layerElementBytes count \(table.count) != layer count \(layerKinds.count)"
            )
            self.perLayerElementBytes = table
        } else {
            self.perLayerElementBytes = Array(
                repeating: config.elementBytes, count: layerKinds.count)
        }
        self.watermarkFraction = config.watermarkFraction
        self.watermark = Int(Double(bytesCapacity) * config.watermarkFraction)
        var perToken = 0
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            perToken += 2 * kind.kvHeads * kind.headDim * self.perLayerElementBytes[index]
        }
        self.maxPerTokenBytes = perToken
    }

    // MARK: Estimation

    /// KV bytes retained after processing `tokens` tokens of one sequence.
    public func estimatedBytes(forTokens tokens: Int) -> Int {
        guard tokens > 0 else { return 0 }
        var total = 0
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let retained: Int
            switch kind.attention {
            case .full: retained = tokens
            case .slidingWindow(let window): retained = min(tokens, window)
            }
            total += retained * 2 * kind.kvHeads * kind.headDim * perLayerElementBytes[index]
        }
        return total
    }

    /// Bytes a single request may ever reserve: `reserve` enforces
    /// `capacity - externalReserve - watermark`, so feasibility must be
    /// judged against the same ceiling. (Judging against full capacity
    /// admitted requests in `(ceiling, capacity]` that could NEVER reserve
    /// their last tokens — they hit the wall, self-preempted, restarted,
    /// and livelocked until their deadline.)
    public var admissibleBytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return _bytesCapacity - externalReserveBytes - watermark
    }

    /// The ceiling every reservation is checked against. Callers hold `lock`.
    private var reserveCeiling: Int { _bytesCapacity - externalReserveBytes - watermark }

    /// Runtime capacity update (multi-model co-residency re-slicing: the
    /// provider shrinks resident engines to fair shares before granting a
    /// newcomer, and grows survivors back on unload). The watermark is
    /// recomputed from the configured fraction against the new capacity.
    ///
    /// Shrink semantics are safe by construction: reservations already in
    /// the ledger are untouched (nothing is evicted here — preemption
    /// remains the scheduler's job), while NEW `reserve` calls fail with
    /// `capacityExhausted` until the pool drains below the new ceiling.
    /// Grow takes effect immediately.
    public func updateBytesCapacity(_ bytes: Int) {
        lock.lock()
        _bytesCapacity = max(0, bytes)
        watermark = Int(Double(_bytesCapacity) * watermarkFraction)
        lock.unlock()
    }

    /// Release the external carve-out (idempotent). Called by the engine
    /// when the obligation it covered no longer exists — compiled decode
    /// disabled itself at warmup, so its padding can never materialize and
    /// the bytes belong to regular admission again (PR#62 review).
    /// Live external (compiled padding) reserve — the construction value
    /// until `refundExternalReserve()`, then 0. Read by the engine loop's
    /// gauge publish so the snapshot's `kvBytesReserved` carries the FULL
    /// not-available-for-new-admissions figure (backend promises + this
    /// carve), keeping "capacity − reserved" truthful for planners.
    public var bytesExternallyReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return externalReserveBytes
    }

    public func refundExternalReserve() {
        lock.lock()
        externalReserveBytes = 0
        lock.unlock()
    }

    /// Truthful submit-time check: worst case (promptLen + maxTokens) vs the
    /// watermark-adjusted capacity (`admissibleBytesCapacity` — the most
    /// `reserve` will ever grant). Requests that could never fit are
    /// rejected up front; requests that fit only sometimes are admitted
    /// optimistically and preempted if optimism loses.
    public func canEverFit(promptTokens: Int, maxTokens: Int) -> Bool {
        estimatedBytes(forTokens: promptTokens + max(maxTokens, 0)) <= admissibleBytesCapacity
    }

    // MARK: CBv2StepCapacity

    public func reserve(id: CBv2RequestID, additionalTokens: Int) throws {
        try reserve(id: id, additionalTokens: additionalTokens, additionalBytes: 0)
    }

    public func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws {
        guard additionalTokens >= 0, additionalBytes >= 0 else {
            throw CBv2KVError.backendIneligible(reason: "negative KV reservation")
        }
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let new = old + additionalTokens
        let oldExact = reservedExactBytes[id] ?? 0
        let (newExact, exactOverflow) = oldExact.addingReportingOverflow(additionalBytes)
        guard !exactOverflow else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        let (tokenDelta, tokenOverflow) = estimatedBytes(forTokens: new)
            .subtractingReportingOverflow(estimatedBytes(forTokens: old))
        let (delta, deltaOverflow) = tokenDelta.addingReportingOverflow(additionalBytes)
        guard !tokenOverflow, !deltaOverflow else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        let (after, afterOverflow) = ledgerBytes.addingReportingOverflow(delta)
        guard !afterOverflow else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        guard after <= reserveCeiling else {
            throw CBv2KVError.capacityExhausted(
                needed: delta,
                available: max(0, reserveCeiling - ledgerBytes))
        }
        reservedTokens[id] = new
        reservedExactBytes[id] = newExact
        ledgerBytes = after
    }

    public func unreserve(id: CBv2RequestID, tokens: Int) {
        unreserve(id: id, tokens: tokens, bytes: 0)
    }

    public func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int) {
        precondition(tokens >= 0 && bytes >= 0)
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let new = max(0, old - tokens)
        let oldExact = reservedExactBytes[id] ?? 0
        let newExact = max(0, oldExact - bytes)
        ledgerBytes += estimatedBytes(forTokens: new) - estimatedBytes(forTokens: old)
            + newExact - oldExact
        if new == 0 {
            reservedTokens.removeValue(forKey: id)
        } else {
            reservedTokens[id] = new
        }
        if newExact == 0 {
            reservedExactBytes.removeValue(forKey: id)
        } else {
            reservedExactBytes[id] = newExact
        }
    }

    public func releaseAll(id: CBv2RequestID) {
        lock.lock()
        defer { lock.unlock() }
        guard let old = reservedTokens.removeValue(forKey: id) else { return }
        let exact = reservedExactBytes.removeValue(forKey: id) ?? 0
        ledgerBytes -= estimatedBytes(forTokens: old) + exact
    }

    /// Conservative (window caps ignored): may under-report headroom, which
    /// only breaks a decode chain — never over-admits.
    public func hasHeadroom(additionalTokens: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ledgerBytes + additionalTokens * maxPerTokenBytes <= reserveCeiling
    }

    // MARK: Telemetry

    /// Ledger bytes currently reserved (soft truth; the backend's
    /// `bytesInUse` is the hard truth once arrays materialize).
    public var bytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return ledgerBytes
    }

    public func snapshot(
        activeRequests: Int, waitingRequests: Int, activeTokens: Int, backendBytesInUse: Int? = nil
    ) -> CBv2CapacitySnapshot {
        lock.lock()
        let ledger = ledgerBytes
        // Honor the field's contract (see `CBv2CapacitySnapshot
        // .kvBytesReserved`): reserved carries the live external (compiled
        // padding) carve too, so "capacity − reserved" matches what
        // `canEverFit`/`reserve` will actually admit — the same figure the
        // engine loop's gauge publish reports. The carve is NOT storage,
        // so the in-use fallback stays ledger-only.
        let reserved = ledger + externalReserveBytes
        let capacity = _bytesCapacity
        lock.unlock()
        return CBv2CapacitySnapshot(
            activeRequests: activeRequests,
            waitingRequests: waitingRequests,
            kvBytesInUse: backendBytesInUse ?? ledger,
            kvBytesCapacity: capacity,
            kvBytesReserved: reserved,
            activeTokens: activeTokens)
    }
}
