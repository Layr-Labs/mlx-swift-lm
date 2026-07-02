// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: truthful admission with soft reservations.
//
// Submit-time check: could the request EVER fit (worst case promptLen +
// maxTokens, window-capped per layer) in total KV capacity? Step-time check:
// vLLM-style optimism — admit while `bytesReserved + nextStepNeed <
// capacity - watermark`; preemption is the backstop when optimism loses.

import Foundation

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
    /// Partially undo a reservation (optimistic-advance rollback).
    func unreserve(id: CBv2RequestID, tokens: Int)
    /// Drop every reservation held by `id` (finish/cancel/preempt).
    func releaseAll(id: CBv2RequestID)
    /// Non-throwing conservative probe used by the chained-decode fast path.
    func hasHeadroom(additionalTokens: Int) -> Bool
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
        /// Bytes per KV element (2 = fp16/bf16).
        public var elementBytes: Int
        public init(watermarkFraction: Double = 0.05, elementBytes: Int = 2) {
            self.watermarkFraction = watermarkFraction
            self.elementBytes = elementBytes
        }
    }

    private let lock = NSLock()
    private let layerKinds: [CBv2LayerKind]
    private let elementBytes: Int
    private let watermark: Int
    /// Per-token bytes if every storage-owning layer retained the token
    /// (upper bound; used for the conservative headroom probe).
    private let maxPerTokenBytes: Int

    public let bytesCapacity: Int

    /// Cumulative reserved tokens per request (window capping is applied when
    /// converting to bytes, so decode reservations past a layer's window add
    /// zero bytes for that layer).
    private var reservedTokens: [CBv2RequestID: Int] = [:]
    private var ledgerBytes = 0

    public init(layerKinds: [CBv2LayerKind], bytesCapacity: Int, config: Config = .init()) {
        self.layerKinds = layerKinds
        self.bytesCapacity = bytesCapacity
        self.elementBytes = config.elementBytes
        self.watermark = Int(Double(bytesCapacity) * config.watermarkFraction)
        var perToken = 0
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            perToken += 2 * kind.kvHeads * kind.headDim * config.elementBytes
        }
        self.maxPerTokenBytes = perToken
    }

    // MARK: Estimation

    /// KV bytes retained after processing `tokens` tokens of one sequence.
    public func estimatedBytes(forTokens tokens: Int) -> Int {
        guard tokens > 0 else { return 0 }
        var total = 0
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            let retained: Int
            switch kind.attention {
            case .full: retained = tokens
            case .slidingWindow(let window): retained = min(tokens, window)
            }
            total += retained * 2 * kind.kvHeads * kind.headDim * elementBytes
        }
        return total
    }

    /// Bytes a single request may ever reserve: `reserve` enforces
    /// `capacity - watermark`, so feasibility must be judged against the
    /// same ceiling. (Judging against full capacity admitted requests in
    /// `(capacity - watermark, capacity]` that could NEVER reserve their
    /// last tokens — they hit the wall, self-preempted, restarted, and
    /// livelocked until their deadline.)
    public var admissibleBytesCapacity: Int { bytesCapacity - watermark }

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
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let new = old + additionalTokens
        let delta = estimatedBytes(forTokens: new) - estimatedBytes(forTokens: old)
        let after = ledgerBytes + delta
        guard after <= bytesCapacity - watermark else {
            throw CBv2KVError.capacityExhausted(
                needed: delta,
                available: max(0, bytesCapacity - watermark - ledgerBytes))
        }
        reservedTokens[id] = new
        ledgerBytes = after
    }

    public func unreserve(id: CBv2RequestID, tokens: Int) {
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let new = max(0, old - tokens)
        ledgerBytes += estimatedBytes(forTokens: new) - estimatedBytes(forTokens: old)
        if new == 0 {
            reservedTokens.removeValue(forKey: id)
        } else {
            reservedTokens[id] = new
        }
    }

    public func releaseAll(id: CBv2RequestID) {
        lock.lock()
        defer { lock.unlock() }
        guard let old = reservedTokens.removeValue(forKey: id) else { return }
        ledgerBytes -= estimatedBytes(forTokens: old)
    }

    /// Conservative (window caps ignored): may under-report headroom, which
    /// only breaks a decode chain — never over-admits.
    public func hasHeadroom(additionalTokens: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ledgerBytes + additionalTokens * maxPerTokenBytes <= bytesCapacity - watermark
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
        CBv2CapacitySnapshot(
            activeRequests: activeRequests,
            waitingRequests: waitingRequests,
            kvBytesInUse: backendBytesInUse ?? bytesReserved,
            kvBytesCapacity: bytesCapacity,
            activeTokens: activeTokens)
    }
}
