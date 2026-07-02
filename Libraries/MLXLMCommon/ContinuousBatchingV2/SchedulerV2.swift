// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: unified token-budget scheduler (vLLM-V1 style).
//
// There is no "prefill phase" and no "decode phase". Each request carries a
// token list (prompt + confirmed generated tokens) and `numComputedTokens`
// (tokens fed through the model). At every step `plan()` assigns each request
// `min(remaining, budget)` new tokens so `numComputedTokens` catches up to
// the number of known tokens; decode rows naturally request exactly 1.
// (Reference: vLLM v1 scheduler.py:395-404 — see research report 08 §1.)
//
// This file is PURE, SYNCHRONOUS bookkeeping: no MLX imports, no arrays, no
// I/O. It is fully unit-testable without model weights. All methods must be
// called from a single thread (the engine thread); the engine serializes.

import Foundation

// MARK: - Scheduler errors

/// Submit-side scheduler contract violations (distinct from
/// `CBv2KVError.capacityExhausted`, which is transient back-off).
public enum CBv2SchedulerError: Error, Equatable {
    /// A request with this id is already live (waiting or running). Request
    /// ids are engine-scoped: reusing one is only legal after the previous
    /// request carrying it has fully finished. Without this rejection a
    /// duplicate would silently clobber the live record in `byID` while the
    /// stale record kept its queue slot — orphaning the first request's
    /// bookkeeping. The provider already guards against duplicate ids; the
    /// engine now enforces it (PR#62 review).
    case duplicateRequestID(CBv2RequestID)
}

// MARK: - Per-request scheduling record

/// Book-keeping for one request inside the v2 scheduler.
///
/// Token accounting (the vLLM optimistic-advance model):
/// - `tokens` — prompt + CONFIRMED generated token values (host-visible).
/// - `pendingSamples` — samples launched but not yet confirmed (deferred stop
///   detection inspects tokens one step late; chained decode keeps them lazy).
/// - `numComputedTokens` — tokens fed through the model, advanced
///   OPTIMISTICALLY at plan time and rolled back on failure/rejection.
public final class CBv2ScheduledRequest {
    public let request: CBv2Request
    /// Monotonic admission sequence — FCFS tie-break within a priority class.
    public let arrivalSeq: UInt64
    public let submittedAt: Date
    /// Absolute wall-clock deadline; the engine error-finishes past this.
    public let deadline: Date?

    /// Prompt + confirmed generated tokens.
    public internal(set) var tokens: [Int]
    /// Tokens fed through the model (optimistically advanced at plan time).
    public internal(set) var numComputedTokens: Int = 0
    /// Samples launched but not yet confirmed on the host.
    public internal(set) var pendingSamples: Int = 0
    public internal(set) var status: CBv2RequestStatus = .waiting
    /// Backpressure: slot retained, scheduling skipped until the consumer
    /// drains its event stream.
    public internal(set) var isPaused: Bool = false
    /// Set by the engine's cancel path; the row is dropped at the next step
    /// boundary. `plan()` never assigns work to a cancel-pending row.
    public internal(set) var cancelRequested: Bool = false
    /// Times this request was preempted (telemetry).
    public internal(set) var preemptionCount: Int = 0

    public var id: CBv2RequestID { request.id }
    public var numTokens: Int { tokens.count }
    /// Known + in-flight tokens: what `numComputedTokens` catches up to.
    public var effectiveTokenCount: Int { tokens.count + pendingSamples }
    public var generatedTokenCount: Int { tokens.count - request.promptTokens.count }
    /// A decode row: exactly one un-computed token remains (its next input).
    public var isDecodeReady: Bool { effectiveTokenCount - numComputedTokens == 1 }
    var remainingTokens: Int { effectiveTokenCount - numComputedTokens }

    init(request: CBv2Request, arrivalSeq: UInt64, submittedAt: Date, deadline: Date?) {
        self.request = request
        self.arrivalSeq = arrivalSeq
        self.submittedAt = submittedAt
        self.deadline = deadline
        self.tokens = request.promptTokens
    }
}

// MARK: - SchedulerV2

/// vLLM-V1-style scheduler: single token budget per step, RUNNING first in
/// order, WAITING admitted while budget and slots allow (chunked prefill =
/// partial token counts), optimistic advance with rollback, and preemption
/// (lowest priority / youngest victim, requeued front) as the capacity
/// backstop.
public final class SchedulerV2 {
    public let config: CBv2SchedulerConfig
    /// Soft KV capacity oracle (AdmissionV2). Optional so pure simulations
    /// can run without any capacity model.
    let capacity: CBv2StepCapacity?

    /// RUNNING requests, in admission order (plan preserves this order).
    public private(set) var running: [CBv2ScheduledRequest] = []
    /// WAITING requests, sorted by (priority desc, arrivalSeq asc); preempted
    /// requests are re-inserted at the FRONT of their priority class.
    public private(set) var waiting: [CBv2ScheduledRequest] = []

    private var byID: [CBv2RequestID: CBv2ScheduledRequest] = [:]
    private var nextArrivalSeq: UInt64 = 0

    public init(config: CBv2SchedulerConfig, capacity: CBv2StepCapacity? = nil) {
        self.config = config
        self.capacity = capacity
    }

    // MARK: Queries

    public var waitingCount: Int { waiting.count }
    public var runningCount: Int { running.count }
    public var hasWork: Bool { !running.isEmpty || !waiting.isEmpty }
    /// Tokens known + in flight across running requests (capacity snapshot).
    public var activeTokens: Int { running.reduce(0) { $0 + $1.effectiveTokenCount } }

    public func record(for id: CBv2RequestID) -> CBv2ScheduledRequest? { byID[id] }

    // MARK: Submission

    /// Enqueue a new request. Throws `CBv2SchedulerError.duplicateRequestID`
    /// when a request with the same id is still live (waiting or running),
    /// and `capacityExhausted` when the waiting queue is full (`maxWaiting`).
    @discardableResult
    public func enqueue(
        _ request: CBv2Request, now: Date = Date(), deadline: Date? = nil
    ) throws -> CBv2ScheduledRequest {
        guard byID[request.id] == nil else {
            throw CBv2SchedulerError.duplicateRequestID(request.id)
        }
        guard waiting.count < config.maxWaiting else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }
        let record = CBv2ScheduledRequest(
            request: request, arrivalSeq: nextArrivalSeq, submittedAt: now, deadline: deadline)
        nextArrivalSeq += 1
        byID[request.id] = record
        insertWaiting(record, preemptedRequeue: false)
        return record
    }

    // MARK: Plan (the vLLM-V1 core)

    /// Produce one step's work assignment under a single token budget.
    ///
    /// - RUNNING first, in order: each gets `min(remaining, chunk, budget)`;
    ///   decode rows request exactly 1.
    /// - On `capacityExhausted` from the capacity oracle, preempt the lowest
    ///   priority / youngest running request (free KV via the returned
    ///   `preemptions`, keep generated tokens, requeue front,
    ///   `numComputedTokens = 0`). If the victim is the requester itself,
    ///   scheduling stops (vLLM scheduler.py:573-575).
    /// - WAITING admitted only if nothing was preempted this step
    ///   (vLLM scheduler.py:634), while budget and `maxConcurrentRequests`
    ///   allow; chunked prefill admits with a partial token count.
    /// - `numComputedTokens` advances optimistically at plan time; use
    ///   `rollback(_:)` if the planned step is never executed.
    public func plan() -> CBv2StepPlan {
        var budget = config.maxBatchedTokensPerStep
        var assignments: [(id: CBv2RequestID, numTokens: Int)] = []
        var assignmentIndex: [CBv2RequestID: Int] = [:]
        var preemptions: [CBv2RequestID] = []
        var stopScheduling = false

        // 1. RUNNING first, in order.
        var idx = 0
        while idx < running.count, budget > 0, !stopScheduling {
            let rec = running[idx]
            if rec.isPaused || rec.cancelRequested || rec.remainingTokens <= 0 {
                idx += 1
                continue
            }
            var n = rec.remainingTokens
            if n > 1 { n = min(n, config.prefillChunkSize) }  // chunk prefill only
            n = min(n, budget)

            // Reserve KV headroom; preemption is the backstop.
            var reserved = capacity == nil
            while !reserved {
                do {
                    try capacity?.reserve(id: rec.id, additionalTokens: n)
                    reserved = true
                } catch {
                    guard let victim = preemptionVictim() else {
                        stopScheduling = true
                        break
                    }
                    if victim === rec {
                        preempt(
                            victim, assignments: &assignments,
                            assignmentIndex: &assignmentIndex, budget: &budget)
                        preemptions.append(victim.id)
                        stopScheduling = true  // victim == requester ⇒ stop
                        break
                    }
                    if let vIdx = running.firstIndex(where: { $0 === victim }), vIdx < idx {
                        idx -= 1
                    }
                    preempt(
                        victim, assignments: &assignments,
                        assignmentIndex: &assignmentIndex, budget: &budget)
                    preemptions.append(victim.id)
                }
            }
            if !reserved { break }

            rec.numComputedTokens += n  // optimistic advance
            budget -= n
            assignmentIndex[rec.id] = assignments.count
            assignments.append((id: rec.id, numTokens: n))
            idx += 1
        }

        // 2. WAITING admission — skipped entirely if anything was preempted.
        if preemptions.isEmpty, !stopScheduling {
            while budget > 0, running.count < config.maxConcurrentRequests,
                let rec = waiting.first
            {
                if rec.cancelRequested { break }  // engine cleans at the boundary
                // A preempted request whose in-flight sample is unconfirmed
                // cannot re-prefill yet (its token values are not host-visible).
                guard rec.pendingSamples == 0 else { break }
                let chunk = min(rec.remainingTokens, config.prefillChunkSize, budget)
                guard chunk > 0 else { break }
                if let capacity {
                    do { try capacity.reserve(id: rec.id, additionalTokens: chunk) } catch {
                        break  // no preemption on behalf of WAITING requests
                    }
                }
                waiting.removeFirst()
                rec.status = .running
                rec.numComputedTokens += chunk
                budget -= chunk
                assignmentIndex[rec.id] = assignments.count
                assignments.append((id: rec.id, numTokens: chunk))
                running.append(rec)
            }
        }

        return CBv2StepPlan(
            assignments: assignments.filter { $0.numTokens > 0 },
            preemptions: preemptions)
    }

    /// Undo the optimistic advance of an UNEXECUTED plan (failure/rejection
    /// path). Requests admitted from waiting by this plan stay in `running`
    /// with zero progress — the next `plan()` reassigns them (vLLM never
    /// un-admits except via preemption). Preemptions are NOT undone.
    public func rollback(_ plan: CBv2StepPlan) {
        for (id, n) in plan.assignments {
            guard let rec = byID[id] else { continue }
            rec.numComputedTokens = max(0, rec.numComputedTokens - n)
            capacity?.unreserve(id: id, tokens: n)
        }
    }

    // MARK: Post-execution accounting (engine → scheduler)

    /// The engine launched a step that will sample one token for each of
    /// `ids` (deferred confirmation — the values are still lazy).
    public func markPendingSamples(ids: [CBv2RequestID]) {
        for id in ids { byID[id]?.pendingSamples += 1 }
    }

    /// Confirm one sampled token (called at step finalization, one step
    /// late). Valid for running AND preempted records — a preempted request
    /// keeps its generated tokens.
    public func recordSampled(id: CBv2RequestID, token: Int) {
        guard let rec = byID[id] else { return }
        rec.tokens.append(token)
        rec.pendingSamples = max(0, rec.pendingSamples - 1)
    }

    /// Remove a request in any state. Returns the record for usage reporting.
    @discardableResult
    public func finish(id: CBv2RequestID, reason: CBv2FinishReason) -> CBv2ScheduledRequest? {
        guard let rec = byID.removeValue(forKey: id) else { return nil }
        running.removeAll { $0 === rec }
        waiting.removeAll { $0 === rec }
        rec.status = .finished(reason)
        return rec
    }

    // MARK: Cancellation & backpressure

    /// Mark a request for cancellation. The engine drops the row at the next
    /// step boundary (O(1)); `plan()` never assigns to a marked row.
    public func requestCancel(_ id: CBv2RequestID) {
        byID[id]?.cancelRequested = true
    }

    /// Backpressure: retain the slot, skip scheduling until resumed.
    public func pause(_ id: CBv2RequestID) { byID[id]?.isPaused = true }
    public func resume(_ id: CBv2RequestID) { byID[id]?.isPaused = false }

    // MARK: Chained-decode eligibility

    /// The exact row set the next `plan()` would schedule IF it is a pure
    /// rectangular-decode step with unchanged membership — or nil when the
    /// next step cannot chain (mid-prefill rows, joins possible, cancels
    /// pending, everything paused, or budget too small).
    ///
    /// The engine compares this against the in-flight step's sampled rows to
    /// decide whether to build step N+1 on top of step N's lazy tokens
    /// (SGLang-MLX chained overlap; chain breaks on ANY membership change).
    public func chainCandidateIDs() -> [CBv2RequestID]? {
        guard !running.isEmpty else { return nil }
        var ids: [CBv2RequestID] = []
        ids.reserveCapacity(running.count)
        for rec in running {
            if rec.cancelRequested { return nil }
            if rec.isPaused { continue }
            guard rec.isDecodeReady else { return nil }
            ids.append(rec.id)
        }
        guard !ids.isEmpty, ids.count <= config.maxBatchedTokensPerStep else { return nil }
        // A join would change membership: if any waiting request could be
        // admitted, the chain must break so the mixed step can run.
        if !waiting.isEmpty, running.count < config.maxConcurrentRequests { return nil }
        return ids
    }

    // MARK: Preemption internals

    /// Victim = LOWEST priority; tie → YOUNGEST (largest arrivalSeq).
    private func preemptionVictim() -> CBv2ScheduledRequest? {
        running.min { a, b in
            if a.request.priority != b.request.priority {
                return a.request.priority < b.request.priority
            }
            return a.arrivalSeq > b.arrivalSeq
        }
    }

    private func preempt(
        _ victim: CBv2ScheduledRequest,
        assignments: inout [(id: CBv2RequestID, numTokens: Int)],
        assignmentIndex: inout [CBv2RequestID: Int],
        budget: inout Int
    ) {
        // Refund an assignment the victim received earlier in this same plan
        // (vLLM scheduler.py:550-567).
        if let aIdx = assignmentIndex.removeValue(forKey: victim.id) {
            budget += assignments[aIdx].numTokens
            assignments[aIdx].numTokens = 0  // filtered out on return
        }
        capacity?.releaseAll(id: victim.id)
        running.removeAll { $0 === victim }
        // Full restart: keep generated tokens, recompute everything (the
        // prefix cache makes re-prefill cheap once WS-D lands).
        victim.numComputedTokens = 0
        victim.status = .preempted
        victim.preemptionCount += 1
        insertWaiting(victim, preemptedRequeue: true)
    }

    /// New arrivals: before the first STRICTLY lower priority (FCFS within a
    /// class). Preempted requeues: before the first SAME-or-lower priority
    /// (front of their class — vLLM prepends preempted requests).
    private func insertWaiting(_ rec: CBv2ScheduledRequest, preemptedRequeue: Bool) {
        let priority = rec.request.priority
        let idx: Int
        if preemptedRequeue {
            idx = waiting.firstIndex { $0.request.priority <= priority } ?? waiting.count
        } else {
            idx = waiting.firstIndex { $0.request.priority < priority } ?? waiting.count
        }
        waiting.insert(rec, at: idx)
    }
}
