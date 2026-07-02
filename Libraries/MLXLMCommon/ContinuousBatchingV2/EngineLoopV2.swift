// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: the execution loop.
//
// Single engine thread (serial GCD queue, the EngineCore idiom): the loop
// does graph-build + `asyncEval` ONLY. Tokenization/detokenization state
// machines are pluggable (WS-E), prefix-cache donation and SSD I/O live
// elsewhere. Decode is rectangular [B, 1]; prefill runs per-request
// [1, chunk] under the shared token budget. There is no left padding, no
// shared frontier, and no batch-wide trim anywhere in this file.
//
// Chained async decode (SGLang-MLX pattern, report 09 §7): step N+1's [B, 1]
// forward is built ON TOP of step N's still-lazy sampled-token array and
// `asyncEval`ed BEFORE the loop blocks on step N's tokens for stop detection.
// Tokens are therefore inspected one step late; a finished request wastes at
// most one slot-step, and its extra token + KV tail are rolled back
// (`CBv2SequenceKV.rollback(1)`). The chain breaks on ANY membership change
// (prefill completion into a different set, finish, cancel, join, pause).

import Foundation
import MLX

// MARK: - Model interface (WS-F adapters / WS-G fixtures conform)

/// Minimal steppable-model surface the loop drives. `tokens` is [B, L] int32
/// ([B, 1] decode, [1, chunk] prefill); `caches` has one entry per model
/// layer with rows matching batch rows. Returns logits [B, L, vocab].
public protocol CBv2SteppableModel: AnyObject {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray
}

/// Builds per-layer batch-facing cache views for a set of rows
/// (`rowStates[b][layer]`, row order == batch row order). WS-A's
/// `LayerCacheV2` conforms; see CONTRACT-ISSUES-B-scheduler.md §1.
public protocol CBv2LayerCacheProvider: AnyObject {
    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache]
}

// MARK: - Sampler interface (WS-E's CBv2DefaultSampler is the production impl)

/// Samples next tokens from last-position logits [B, vocab] → lazy token
/// array [B] (int32). MUST NOT host-sync; see
/// CONTRACT-ISSUES-B-scheduler.md §2 and CONTRACT-DECISIONS.md.
///
/// Stateful samplers (penalties, keyed RNG) reconfigure on membership
/// change using `rowContext` and track per-request progress:
///  - `params`/`requestIDs`: per-row, row order == logits row order.
///  - `stepIndex`: global engine step (telemetry only — per-request RNG must
///    key on per-request progress, never on this).
///  - `pendingSampledTokens`: lazy [B] tokens sampled for exactly these rows
///    by the still-in-flight previous step (chained decode), row-aligned;
///    nil when every row's history is fully confirmed. A sampler that
///    reconfigures from `rowContext` (confirmed history only) must fold
///    these in on-device to stay exact.
///  - `rowContext`: materializes full per-row context (id, params, prompt,
///    CONFIRMED output tokens). Copies token arrays — only call it on
///    membership change, never on the chained fast path.
public protocol CBv2StepSampler: AnyObject {
    func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray
}

/// Greedy stub — vectorized argmax, batch-composition invariant by
/// construction (per-row reduction, no cross-row ops). Kept as the
/// deterministic fallback for scheduler/loop tests; production uses
/// `CBv2DefaultSampler`.
public final class CBv2GreedySampler: CBv2StepSampler {
    public init() {}
    public func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray {
        argMax(logits, axis: -1).asType(.int32)
    }
}

// MARK: - Detokenizer interface (WS-E conforms; null stub until then)

/// Incremental per-request detokenizer with UTF-8 + stop-string holdback.
/// See CONTRACT-ISSUES-B-scheduler.md §3.
public protocol CBv2IncrementalDetokenizer: AnyObject {
    /// Append confirmed tokens; returns text now safe to emit.
    func push(_ tokens: [Int]) -> String
    /// True once a stop string has matched (engine finishes with `.stop`).
    var matchedStopString: Bool { get }
    /// Held-back text still emittable at finish (excludes matched stop text).
    func flush() -> String
}

public protocol CBv2DetokenizerFactory: AnyObject {
    func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer
}

/// Default until WS-E lands: deltas carry token ids with empty text, stop
/// strings never match (stop tokens / maxTokens / deadlines still work).
public final class CBv2NullDetokenizerFactory: CBv2DetokenizerFactory {
    final class NullDetokenizer: CBv2IncrementalDetokenizer {
        let matchedStopString = false
        func push(_ tokens: [Int]) -> String { "" }
        func flush() -> String { "" }
    }
    public init() {}
    public func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer {
        NullDetokenizer()
    }
}

// MARK: - Loop configuration

public struct CBv2EngineLoopConfig: Sendable {
    /// Per-request wall-clock deadline; overdue requests are error-finished
    /// at the next step boundary.
    public var requestTimeout: TimeInterval
    /// Single-step watchdog: a step (graph build + blocking eval) exceeding
    /// this marks the engine unhealthy, error-finishes all live streams, and
    /// fires `onStepWedge`.
    public var stepTimeout: TimeInterval
    /// Watchdog polling interval.
    public var watchdogInterval: TimeInterval
    /// Idle re-check interval when there is no work.
    public var idleRecheckInterval: TimeInterval
    /// Per-request event buffer before backpressure pauses scheduling.
    public var eventBufferCapacity: Int

    public init(
        requestTimeout: TimeInterval = 120, stepTimeout: TimeInterval = 30,
        watchdogInterval: TimeInterval = 0.25, idleRecheckInterval: TimeInterval = 0.001,
        eventBufferCapacity: Int = 256
    ) {
        self.requestTimeout = requestTimeout
        self.stepTimeout = stepTimeout
        self.watchdogInterval = watchdogInterval
        self.idleRecheckInterval = idleRecheckInterval
        self.eventBufferCapacity = eventBufferCapacity
    }
}

// MARK: - In-flight step

/// One launched-but-not-finalized step. Its sampled tokens are still lazy;
/// finalization materializes them (the ONE host sync per step, overlapped
/// with the next step's GPU work when chained).
final class CBv2InFlightStep {
    /// Every request that computed anything this step (KV release for any of
    /// these must be deferred until finalization — see CONTRACT-ISSUES §4).
    let participants: Set<CBv2RequestID>
    /// Rows that sampled a token, in plan order (== row order of
    /// `sampledTokens`).
    let sampledRows: [CBv2RequestID]
    /// Lazy [K] int32, or nil when no row sampled (all mid-prefill chunks).
    let sampledTokens: MLXArray?
    /// Cheap handles that force evaluation of non-sampling prefill chunks.
    let evalTargets: [MLXArray]
    /// Rows finished/cancelled AFTER launch: their sampled token is
    /// discarded at finalization (the ≤1 wasted slot-step).
    var discard: Set<CBv2RequestID> = []
    /// KV states whose release is fenced behind this step's completion.
    /// `rollbackOne` scrubs the wasted-token KV tail before release;
    /// `donateTokens` (non-nil for natural finishes with prefix caching on)
    /// routes the retired state through the donation queue.
    var deferredReleases:
        [(
            id: CBv2RequestID, state: [CBv2SequenceKV?], rollbackOne: Bool,
            donateTokens: [Int]?
        )] = []

    init(
        participants: Set<CBv2RequestID>, sampledRows: [CBv2RequestID],
        sampledTokens: MLXArray?, evalTargets: [MLXArray]
    ) {
        self.participants = participants
        self.sampledRows = sampledRows
        self.sampledTokens = sampledTokens
        self.evalTargets = evalTargets
    }
}

// MARK: - Prefix adoption handoff

/// A prefix-cache hit prepared on the submit thread (lookup + graph-only
/// slicing) and applied on the engine thread at enqueue. `prefix` is already
/// sliced to `effective = matched - cbv2RequiredRecompute(...)`, the uniform
/// offset every storage-owning row adopts; the engine replays
/// `[effective, prompt)` through all layers. The lookup's in-use pin is
/// balanced by exactly one `endAdoption(tokens:matched:)` when the adoption
/// is consumed or abandoned.
struct CBv2PrefixAdoption {
    let tokens: [Int]
    let matched: Int
    let effective: Int
    let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?]
}

// MARK: - EngineLoopV2

/// The engine thread. All scheduler and MLX mutations happen on
/// `engineQueue` (serial, self-rescheduling — the EngineCore idiom that
/// keeps the GPU pipeline full with only GCD dispatch overhead between
/// steps). Cross-thread surface: `requestCancel`, `setPaused`, stream
/// registration, and the watchdog — all lock-protected.
public final class EngineLoopV2: @unchecked Sendable {
    let scheduler: SchedulerV2
    let capacity: CBv2StepCapacity?
    let backend: CBv2KVBackend
    let cacheProvider: CBv2LayerCacheProvider
    let model: CBv2SteppableModel
    let sampler: CBv2StepSampler
    let detokenizerFactory: CBv2DetokenizerFactory
    let layerKinds: [CBv2LayerKind]
    /// Non-nil only when prefix caching is active (instance supplied AND
    /// `CBv2SchedulerConfig.enablePrefixCache`).
    let prefixCache: CBv2PrefixCache?
    let config: CBv2EngineLoopConfig
    let gauges: CBv2EngineGauges

    private let engineQueue = DispatchQueue(
        label: "com.eigen.cbv2.engine", qos: .userInitiated)
    private let watchdogQueue = DispatchQueue(
        label: "com.eigen.cbv2.watchdog", qos: .utility)
    /// Prefix-cache donation runs here (hashing + indexing + optional device
    /// materialization) — never on the engine step thread (invariant 6).
    private let donationQueue = DispatchQueue(
        label: "com.eigen.cbv2.donation", qos: .utility)

    // Cross-thread state (stateLock).
    private let stateLock = NSLock()
    private var streams: [CBv2RequestID: CBv2OutputStream] = [:]
    private var pendingCancels: Set<CBv2RequestID> = []
    private var stepStartedNanos: UInt64 = 0
    private var wedgeReported = false
    private var _healthy = true

    // Engine-thread-confined state.
    private var detokenizers: [CBv2RequestID: CBv2IncrementalDetokenizer] = [:]
    private var kvStates: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
    /// Tokens skipped via prefix-cache adoption, reported in usage.
    private var prefixHitTokens: [CBv2RequestID: Int] = [:]
    private var inFlight: CBv2InFlightStep?
    private var running = false
    private var draining = false
    private var drainContinuations: [CheckedContinuation<Void, Never>] = []

    /// Telemetry / test hooks.
    public private(set) var stepCount = 0
    public private(set) var chainedStepCount = 0
    public private(set) var preemptionCount = 0
    /// Fired (from the watchdog thread) when a step exceeds `stepTimeout`.
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)?

    public var isHealthy: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _healthy
    }

    init(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        backend: CBv2KVBackend,
        cacheProvider: CBv2LayerCacheProvider,
        sampler: CBv2StepSampler,
        detokenizerFactory: CBv2DetokenizerFactory,
        scheduler: SchedulerV2,
        capacity: CBv2StepCapacity?,
        prefixCache: CBv2PrefixCache? = nil,
        config: CBv2EngineLoopConfig,
        gauges: CBv2EngineGauges
    ) {
        self.model = model
        self.layerKinds = layerKinds
        self.backend = backend
        self.cacheProvider = cacheProvider
        self.sampler = sampler
        self.detokenizerFactory = detokenizerFactory
        self.scheduler = scheduler
        self.capacity = capacity
        self.prefixCache = prefixCache
        self.config = config
        self.gauges = gauges
    }

    // MARK: Lifecycle

    func start() {
        engineQueue.async { [self] in
            guard !running else { return }
            running = true
            startWatchdog()
            engineQueue.async { [weak self] in self?.engineStep() }
        }
    }

    /// Graceful drain: waiting requests are cancelled, running requests
    /// finish naturally, then the loop stops. Idempotent.
    func drain() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            engineQueue.async { [self] in
                guard running else {
                    c.resume()
                    return
                }
                draining = true
                for rec in scheduler.waiting {
                    finishRequest(rec.id, reason: .cancelled)
                }
                publishGauges()
                if !scheduler.hasWork, inFlight == nil {
                    completeStop()
                    c.resume()
                } else {
                    drainContinuations.append(c)
                }
            }
        }
    }

    private func completeStop() {
        running = false
        draining = false
        stopWatchdog()
    }

    // MARK: Submission (from EngineV2)

    func register(stream: CBv2OutputStream) {
        stateLock.lock()
        streams[stream.id] = stream
        stateLock.unlock()
    }

    /// Runs on the engine queue. The stream must already be registered.
    func enqueue(_ request: CBv2Request, adoption: CBv2PrefixAdoption? = nil) {
        engineQueue.async { [self] in
            defer {
                gauges.endSubmit()
                publishGauges()
            }
            guard running, !draining else {
                releaseAbandonedAdoption(adoption)
                takeStream(request.id)?.finish(
                    reason: .error("engine is shutting down"),
                    usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
                return
            }
            do {
                try scheduler.enqueue(
                    request,
                    deadline: Date().addingTimeInterval(config.requestTimeout))
                detokenizers[request.id] =
                    detokenizerFactory.makeDetokenizer(stopStrings: request.stopStrings)
                if let adoption {
                    applyAdoption(adoption, requestID: request.id)
                }
            } catch {
                releaseAbandonedAdoption(adoption)
                // Precise maxWaiting enforcement (the submit-side gauge check
                // is the fast path; this is the authoritative one).
                takeStream(request.id)?.finish(
                    reason: .error("capacity: waiting queue is full"),
                    usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
            }
        }
    }

    // MARK: Prefix-cache adoption (engine thread)

    /// Adopt a prepared prefix hit into fresh KV state and fast-forward the
    /// scheduler record. Best-effort: any failure (capacity, backend) falls
    /// back to a full prefill — the request itself never fails here. Always
    /// balances the lookup pin.
    private func applyAdoption(_ adoption: CBv2PrefixAdoption, requestID: CBv2RequestID) {
        defer {
            prefixCache?.endAdoption(tokens: adoption.tokens, matched: adoption.matched)
        }
        guard let rec = scheduler.record(for: requestID), kvStates[requestID] == nil else {
            return
        }
        if let capacity {
            do {
                try capacity.reserve(id: requestID, additionalTokens: adoption.effective)
            } catch {
                return  // capacity tight — full prefill with the usual backstops
            }
        }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state = try backend.makeSequenceState(
                adopting: adoption.prefix, layerKinds: layerKinds, maxLength: maxLength)
            kvStates[requestID] = state
            rec.numComputedTokens = adoption.effective
            prefixHitTokens[requestID] = adoption.effective
        } catch {
            capacity?.unreserve(id: requestID, tokens: adoption.effective)
        }
    }

    /// Balance a lookup pin for an adoption that never reached
    /// `applyAdoption` (shutdown, queue-full rejection).
    private func releaseAbandonedAdoption(_ adoption: CBv2PrefixAdoption?) {
        guard let adoption else { return }
        prefixCache?.endAdoption(tokens: adoption.tokens, matched: adoption.matched)
    }

    // MARK: Cancellation & backpressure (any thread)

    /// O(1): marks the request; the row is dropped at the next step boundary.
    /// Lock-based (not a queue hop) so cancellation works even while the
    /// engine thread is blocked inside a long step.
    func requestCancel(_ id: CBv2RequestID) {
        stateLock.lock()
        pendingCancels.insert(id)
        stateLock.unlock()
    }

    func setPaused(_ id: CBv2RequestID, _ paused: Bool) {
        engineQueue.async { [self] in
            if paused {
                scheduler.pause(id)
            } else {
                scheduler.resume(id)
            }
        }
    }

    // MARK: The step loop

    private func engineStep() {
        guard running else { return }
        markStepStarted()
        defer { markStepEnded() }

        processCancellations()
        processDeadlines()

        // Chained decode fast path: build step N+1 on step N's lazy tokens.
        if let previous = inFlight,
            previous.sampledTokens != nil,
            let ids = scheduler.chainCandidateIDs(),
            ids == previous.sampledRows,
            ids.allSatisfy({ kvStates[$0] != nil }),
            capacity?.hasHeadroom(additionalTokens: ids.count) ?? true
        {
            let plan = scheduler.plan()
            if isPureDecodePlan(plan, matching: ids) {
                let next = launchChainedDecode(plan, feeding: previous.sampledTokens!)
                inFlight = next
                chainedStepCount += 1
                stepCount += 1
                finalize(previous)
                publishGauges()
                scheduleNextStep()
                return
            }
            // Defensive: chainCandidateIDs and plan() disagree (only
            // possible when the capacity oracle's `hasHeadroom` was
            // optimistic and `reserve` preempted mid-plan). Roll the
            // optimistic advance back and fall through to the general path.
            // Preemptions are NOT rolled back (the scheduler already
            // requeued the victims), so their KV must be released here —
            // fenced behind the still-in-flight step that references it.
            // The victims are NOT added to `discard`: their in-flight
            // sample must be recorded at finalization (preemption keeps
            // generated tokens, and an unconfirmed `pendingSamples` would
            // block their re-admission forever).
            scheduler.rollback(plan)
            for id in plan.preemptions {
                preemptionCount += 1
                guard let state = kvStates.removeValue(forKey: id) else { continue }
                if previous.participants.contains(id) {
                    previous.deferredReleases.append(
                        (id: id, state: state, rollbackOne: false, donateTokens: nil))
                } else {
                    backend.release(state)
                }
            }
        }

        // Chain broken (or nothing chained): finalize before re-planning so
        // the plan sees confirmed tokens and post-stop membership.
        if let previous = inFlight {
            inFlight = nil
            finalize(previous)
        }

        guard scheduler.hasWork else {
            publishGauges()
            if draining {
                completeStop()
                let continuations = drainContinuations
                drainContinuations = []
                for c in continuations { c.resume() }
                return
            }
            scheduleIdleRecheck()
            return
        }

        let plan = scheduler.plan()
        handlePreemptions(plan.preemptions)
        guard !plan.assignments.isEmpty else {
            publishGauges()
            scheduleIdleRecheck()
            return
        }
        inFlight = executeMixed(plan)
        stepCount += 1
        publishGauges()
        scheduleNextStep()
    }

    private func scheduleNextStep() {
        engineQueue.async { [weak self] in self?.engineStep() }
    }

    private func scheduleIdleRecheck() {
        engineQueue.asyncAfter(deadline: .now() + config.idleRecheckInterval) { [weak self] in
            self?.engineStep()
        }
    }

    // MARK: Step execution

    private func isPureDecodePlan(_ plan: CBv2StepPlan, matching ids: [CBv2RequestID]) -> Bool {
        guard plan.preemptions.isEmpty, plan.assignments.count == ids.count else { return false }
        for (i, assignment) in plan.assignments.enumerated() {
            guard assignment.numTokens == 1, assignment.id == ids[i] else { return false }
        }
        return true
    }

    /// Pure-decode step fed by the previous step's still-lazy tokens.
    private func launchChainedDecode(
        _ plan: CBv2StepPlan, feeding lazyTokens: MLXArray
    ) -> CBv2InFlightStep {
        let ids = plan.assignments.map(\.id)
        let rowStates = ids.map { kvStates[$0]! }  // presence pre-checked
        var params: [CBv2SamplingParams] = []
        params.reserveCapacity(ids.count)
        for id in ids { params.append(scheduler.record(for: id)!.request.sampling) }

        let inputs = lazyTokens.reshaped([ids.count, 1])
        let caches = cacheProvider.layerCaches(rowStates: rowStates)
        let logits = model.forward(tokens: inputs, caches: caches)
        let last = logits[0..., -1, 0...]  // [B, vocab]
        // `pendingSampledTokens` = the fed lazy tokens: each row has exactly
        // one launched-but-unconfirmed sample here (the chain invariant).
        let sampled = sampler.sample(
            logits: last, params: params, requestIDs: ids, stepIndex: stepCount,
            pendingSampledTokens: lazyTokens,
            rowContext: { [scheduler] in
                ids.map { Self.samplerRow(scheduler.record(for: $0)!) }
            })
        scheduler.markPendingSamples(ids: ids)
        asyncEval([sampled])
        return CBv2InFlightStep(
            participants: Set(ids), sampledRows: ids, sampledTokens: sampled, evalTargets: [])
    }

    /// General step: decode batch [B, 1] + per-request [1, chunk] prefills,
    /// interleaved per the plan, ONE `asyncEval` for the whole step.
    private func executeMixed(_ plan: CBv2StepPlan) -> CBv2InFlightStep? {
        struct RowWork {
            let rec: CBv2ScheduledRequest
            let start: Int
            let count: Int
            let samples: Bool
            let isDecode: Bool
        }

        var work: [RowWork] = []
        work.reserveCapacity(plan.assignments.count)
        for (id, n) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            guard ensureKVState(rec) != nil else { continue }  // error-finished
            let start = rec.numComputedTokens - n  // pre-optimistic-advance position
            // The step samples iff it computes through the last known token.
            // (`pendingSamples == 0` here: finalize always precedes
            // executeMixed, so every planned token value is host-visible.)
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            let isDecode = n == 1 && samples && start == rec.tokens.count - 1
            work.append(
                RowWork(rec: rec, start: start, count: n, samples: samples, isDecode: isDecode))
        }
        guard !work.isEmpty else { return nil }

        // Rectangular decode batch, in plan order.
        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        if !decodeRows.isEmpty {
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let caches = cacheProvider.layerCaches(
                rowStates: decodeRows.map { kvStates[$0.rec.id]! })
            let logits = model.forward(tokens: inputs, caches: caches)
            decodeSampled = sampler.sample(
                logits: logits[0..., -1, 0...],
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,  // finalize preceded: all confirmed
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
        }

        // Per-request prefill chunks [1, chunk].
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var evalTargets: [MLXArray] = []
        for row in work where !row.isDecode {
            let rec = row.rec
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let caches = cacheProvider.layerCaches(rowStates: [kvStates[rec.id]!])
            let logits = model.forward(tokens: inputs, caches: caches)
            if row.samples {
                prefillSampled[rec.id] = sampler.sample(
                    logits: logits[0..., -1, 0...],
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
            } else {
                // Cheap handle that forces this chunk's graph (incl. KV
                // writes) without materializing full logits on the host.
                evalTargets.append(logits[0, row.count - 1, 0 ..< 1])
            }
        }

        // Assemble sampled tokens in plan order so a following pure-decode
        // plan (same membership, same order) can chain off this array.
        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIdx = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIdx ..< decodeIdx + 1])
                decodeIdx += 1
                sampledRows.append(row.rec.id)
            } else if let s = prefillSampled[row.rec.id] {
                pieces.append(s)
                sampledRows.append(row.rec.id)
            }
        }
        let sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))

        scheduler.markPendingSamples(ids: sampledRows)
        var toEval = evalTargets
        if let sampledTokens { toEval.append(sampledTokens) }
        asyncEval(toEval)

        return CBv2InFlightStep(
            participants: Set(work.map(\.rec.id)),
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            evalTargets: evalTargets)
    }

    // MARK: Finalization (deferred stop detection)

    private func finalize(_ step: CBv2InFlightStep) {
        // THE host sync — overlapped with the successor step's GPU work when
        // chained. All-prefill steps block on their eval targets instead so
        // graph pipelining stays bounded at two steps.
        var host: [Int32] = []
        if let tokens = step.sampledTokens {
            host = tokens.asArray(Int32.self)
        } else if !step.evalTargets.isEmpty {
            eval(step.evalTargets)
        }

        for (i, id) in step.sampledRows.enumerated() {
            if step.discard.contains(id) { continue }
            guard let rec = scheduler.record(for: id) else { continue }
            let token = Int(host[i])
            scheduler.recordSampled(id: id, token: token)

            let detokenizer = detokenizers[id]
            let text = detokenizer?.push([token]) ?? ""
            stream(for: id)?.emit(.delta(text: text, tokens: [token], logprobs: nil))

            // Stop detection — one step late by construction.
            if rec.request.stopTokens.contains(token) {
                finishRequest(id, reason: .stop)
            } else if let detokenizer, detokenizer.matchedStopString {
                finishRequest(id, reason: .stop)
            } else if rec.generatedTokenCount >= rec.request.maxTokens {
                finishRequest(id, reason: .length)
            } else if let deadline = rec.deadline, Date() > deadline {
                finishRequest(
                    id, reason: .error("request exceeded \(Int(config.requestTimeout))s deadline"))
            }
        }

        // Fenced frees: rows finished/cancelled while this step was in
        // flight. Scrub the wasted-token KV tail, then retire (donate to
        // the prefix cache when eligible, else release).
        for (_, state, rollbackOne, donateTokens) in step.deferredReleases {
            if rollbackOne {
                for sequence in state { sequence?.rollback(1) }
            }
            retire(state: state, donatingTokens: donateTokens)
        }
    }

    // MARK: Request completion

    private func finishRequest(_ id: CBv2RequestID, reason: CBv2FinishReason) {
        guard let rec = scheduler.finish(id: id, reason: reason) else {
            // Unknown to the scheduler (already finished) — make sure no
            // stream leaks regardless.
            prefixHitTokens.removeValue(forKey: id)
            takeStream(id)?.finish(
                reason: reason, usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
            return
        }
        capacity?.releaseAll(id: id)

        if let state = kvStates.removeValue(forKey: id) {
            let donateTokens = donationTokens(for: rec, reason: reason)
            if let inFlight, inFlight.participants.contains(id) {
                // The in-flight step still references this state — fence the
                // free behind its completion; roll back the wasted token iff
                // that step sampled for this row.
                inFlight.discard.insert(id)
                inFlight.deferredReleases.append(
                    (
                        id: id, state: state,
                        rollbackOne: inFlight.sampledRows.contains(id),
                        donateTokens: donateTokens
                    ))
            } else {
                retire(state: state, donatingTokens: donateTokens)
            }
        }

        var trailing = ""
        if let detokenizer = detokenizers.removeValue(forKey: id) {
            trailing = detokenizer.flush()
        }
        let stream = takeStream(id)
        if !trailing.isEmpty {
            stream?.emit(.delta(text: trailing, tokens: [], logprobs: nil))
        }
        stream?.finish(
            reason: reason,
            usage: CBv2Usage(
                promptTokens: rec.request.promptTokens.count,
                completionTokens: rec.generatedTokenCount,
                prefixCacheHitTokens: prefixHitTokens.removeValue(forKey: id) ?? 0))
    }

    // MARK: Prefix-cache donation (engine thread → donation queue)

    /// Tokens to donate for a finished request, or nil when donation does
    /// not apply. Only NATURAL completions donate: their KV is a complete,
    /// confirmed prefix. The last confirmed token was sampled but never fed
    /// through the model, so it is dropped at donation time.
    private func donationTokens(for rec: CBv2ScheduledRequest, reason: CBv2FinishReason)
        -> [Int]?
    {
        guard prefixCache != nil else { return nil }
        switch reason {
        case .stop, .length: break
        case .cancelled, .error: return nil
        }
        // Donation requires the full prompt to have been processed (at
        // least one sampled token) — mid-prefill finishes carry partial KV.
        guard rec.generatedTokenCount >= 1, rec.tokens.count > 1 else { return nil }
        return rec.tokens
    }

    /// Retire a finished request's KV state: donate to the prefix cache
    /// (snapshots graph-built HERE on the engine thread, so views over
    /// shared paged slabs are consistent with in-flight writes; hashing +
    /// indexing + optional materialization run on the donation queue), then
    /// release the storage back on the engine queue (the paged pool is
    /// engine-thread-affine).
    private func retire(state: [CBv2SequenceKV?], donatingTokens: [Int]?) {
        guard let prefixCache, let tokens = donatingTokens else {
            backend.release(state)
            return
        }
        let donated = Array(tokens.dropLast())
        var snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        snapshots.reserveCapacity(layerKinds.count)
        for (i, kind) in layerKinds.enumerated() {
            var cacheable = kind.sharesKVWithLayer == nil
            if case .slidingWindow = kind.attention { cacheable = false }
            guard cacheable, let seq = state[i] else {
                snapshots.append(nil)
                continue
            }
            snapshots.append(seq.snapshot())
        }
        let layerKinds = self.layerKinds
        // Strong self on purpose: the deferred release is a pending
        // obligation of this loop — it must survive until the donation
        // lands, or the retired state leaks its pages (the paged pool is
        // engine-thread-affine, so the free hops back to the engine queue).
        // No cycle: the block releases its captures once it runs.
        donationQueue.async {
            prefixCache.donate(tokens: donated, snapshots: snapshots, layerKinds: layerKinds)
            self.releaseOnEngineQueue(state)
        }
    }

    private func releaseOnEngineQueue(_ state: [CBv2SequenceKV?]) {
        engineQueue.async { [backend] in backend.release(state) }
    }

    // MARK: Sampler row context

    /// Full per-row sampler context (confirmed history only; in-flight
    /// chained samples travel separately as `pendingSampledTokens`).
    private static func samplerRow(_ rec: CBv2ScheduledRequest) -> CBv2SamplerRow {
        CBv2SamplerRow(
            id: rec.id,
            params: rec.request.sampling,
            promptTokens: rec.request.promptTokens,
            outputTokens: Array(rec.tokens.dropFirst(rec.request.promptTokens.count)))
    }

    // MARK: Boundary housekeeping

    private func processCancellations() {
        stateLock.lock()
        let cancels = pendingCancels
        pendingCancels.removeAll()
        stateLock.unlock()
        for id in cancels {
            guard scheduler.record(for: id) != nil else { continue }
            finishRequest(id, reason: .cancelled)
        }
    }

    private func processDeadlines() {
        let now = Date()
        var overdue: [CBv2RequestID] = []
        for rec in scheduler.running where rec.deadline.map({ now > $0 }) == true {
            overdue.append(rec.id)
        }
        for rec in scheduler.waiting where rec.deadline.map({ now > $0 }) == true {
            overdue.append(rec.id)
        }
        for id in overdue {
            finishRequest(
                id, reason: .error("request exceeded \(Int(config.requestTimeout))s deadline"))
        }
    }

    private func handlePreemptions(_ ids: [CBv2RequestID]) {
        // Preemption only happens on the non-chained path, where the
        // previous step was finalized first — no in-flight step can
        // reference these states, so the release is immediate. The
        // scheduler already released the capacity reservations and requeued
        // the victims (generated tokens kept).
        assert(inFlight == nil, "preemption with a step in flight")
        for id in ids {
            preemptionCount += 1
            if let state = kvStates.removeValue(forKey: id) {
                backend.release(state)
            }
        }
    }

    /// Create per-layer KV state on first execution. On allocation failure
    /// (admission + preemption should prevent this; backstop only) the
    /// request is error-finished.
    private func ensureKVState(_ rec: CBv2ScheduledRequest) -> [CBv2SequenceKV?]? {
        if let state = kvStates[rec.id] { return state }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state = try backend.makeSequenceState(
                layerKinds: layerKinds, promptLength: rec.tokens.count, maxLength: maxLength)
            kvStates[rec.id] = state
            return state
        } catch {
            finishRequest(rec.id, reason: .error("KV allocation failed: \(error)"))
            return nil
        }
    }

    // MARK: Streams

    private func stream(for id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams[id]
    }

    private func takeStream(_ id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams.removeValue(forKey: id)
    }

    // MARK: Test/telemetry introspection

    /// Engine-queue-synchronized snapshot of paused (backpressured) rows.
    /// Blocks until the current step boundary; test/telemetry use only.
    func pausedIDsSnapshot() -> Set<CBv2RequestID> {
        engineQueue.sync { Set(scheduler.running.filter(\.isPaused).map(\.id)) }
    }

    // MARK: Gauges

    private func publishGauges() {
        gauges.update(
            CBv2CapacitySnapshot(
                activeRequests: scheduler.runningCount,
                waitingRequests: scheduler.waitingCount,
                kvBytesInUse: backend.bytesInUse,
                kvBytesCapacity: backend.bytesCapacity,
                activeTokens: scheduler.activeTokens,
                stepsExecuted: stepCount))
    }

    // MARK: Watchdog (engine health signal)

    private var watchdogTimer: DispatchSourceTimer?

    private func markStepStarted() {
        stateLock.lock()
        stepStartedNanos = DispatchTime.now().uptimeNanoseconds
        stateLock.unlock()
    }

    private func markStepEnded() {
        stateLock.lock()
        stepStartedNanos = 0
        wedgeReported = false
        _healthy = true
        stateLock.unlock()
    }

    deinit {
        watchdogTimer?.cancel()
    }

    private func startWatchdog() {
        let timer = DispatchSource.makeTimerSource(queue: watchdogQueue)
        timer.schedule(
            deadline: .now() + config.watchdogInterval, repeating: config.watchdogInterval)
        timer.setEventHandler { [weak self] in self?.watchdogTick() }
        timer.resume()
        watchdogTimer = timer
    }

    private func stopWatchdog() {
        watchdogTimer?.cancel()
        watchdogTimer = nil
    }

    /// Runs on the watchdog queue. The engine thread may be blocked inside a
    /// wedged eval, so this touches ONLY lock-protected state and the
    /// (thread-safe) streams: consumers get a timely error, every live
    /// request is marked for cancellation (cleaned up if the loop ever
    /// resumes), and the health signal flips for provider heartbeats.
    private func watchdogTick() {
        stateLock.lock()
        let started = stepStartedNanos
        guard started != 0 else {
            stateLock.unlock()
            return
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds &- started) / 1_000_000_000
        guard elapsed > config.stepTimeout, !wedgeReported else {
            stateLock.unlock()
            return
        }
        wedgeReported = true
        _healthy = false
        let liveStreams = streams
        pendingCancels.formUnion(liveStreams.keys)
        stateLock.unlock()

        // Signal BEFORE erroring the streams: a consumer woken by the
        // terminal event must be able to observe the wedge side effects
        // (health metric, telemetry) immediately.
        onStepWedge?(elapsed)
        for (_, stream) in liveStreams {
            stream.finish(
                reason: .error(
                    "engine step exceeded \(Int(config.stepTimeout))s watchdog"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
    }
}
