// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: the public engine (`CBv2Engine`).
//
// Thin wiring layer: scheduler + admission + loop + sampler + model adapter.
// Tokenization happens on the caller's task (requests already carry token
// ids per the contract); the engine thread only ever does graph-build +
// asyncEval. Cancellation drops the row at the next step boundary, O(1).

import Foundation
import MLX

// MARK: - Shared gauges (submit-side admission ⇄ engine-thread truth)

/// Lock-protected counters bridging the caller-thread `submit`/`capacity`
/// surface and the engine thread. The loop publishes a full snapshot after
/// every step; `pendingSubmits` covers requests accepted on the caller
/// thread but not yet picked up by the engine queue (so the `maxWaiting`
/// bound holds even while a step blocks the queue).
final class CBv2EngineGauges: @unchecked Sendable {
    private let lock = NSLock()
    private var snapshot: CBv2CapacitySnapshot
    private var pendingSubmits = 0

    init(kvBytesCapacity: Int) {
        self.snapshot = CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0,
            kvBytesCapacity: kvBytesCapacity, activeTokens: 0)
    }

    func update(_ newValue: CBv2CapacitySnapshot) {
        lock.lock()
        snapshot = newValue
        lock.unlock()
    }

    func read() -> CBv2CapacitySnapshot {
        lock.lock()
        defer { lock.unlock() }
        var value = snapshot
        value.waitingRequests += pendingSubmits
        return value
    }

    /// Fast-path `maxWaiting` bound; the scheduler's own check (on the
    /// engine thread) is authoritative.
    func beginSubmit(maxWaiting: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard snapshot.waitingRequests + pendingSubmits < maxWaiting else { return false }
        pendingSubmits += 1
        return true
    }

    func endSubmit() {
        lock.lock()
        pendingSubmits = max(0, pendingSubmits - 1)
        lock.unlock()
    }
}

// MARK: - EngineV2

/// Production continuous-batching engine v2 (WS-B).
///
/// ```swift
/// let engine = EngineV2(
///     model: adapter, layerKinds: kinds, backend: kvBackend,
///     cacheProvider: layerCacheFactory)
/// let events = try engine.submit(request)
/// for await event in events { ... }
/// ```
/// `@unchecked Sendable` justification (contract `CBv2Engine: Sendable`):
/// all mutable state is either lock-protected (`stateLock`,
/// `CBv2EngineGauges`) or confined to the engine's serial dispatch queue
/// inside `EngineLoopV2`; the only cross-thread surfaces are `submit`
/// (lock + queue hop), `cancel` (lock), `capacity()` (lock), and
/// `shutdown()` (queue-synchronized drain).
public final class EngineV2: CBv2Engine, @unchecked Sendable {
    private let loop: EngineLoopV2
    private let admission: AdmissionV2
    private let schedulerConfig: CBv2SchedulerConfig
    private let loopConfig: CBv2EngineLoopConfig
    private let gauges: CBv2EngineGauges

    private let stateLock = NSLock()
    private var rejectingSubmissions = false

    /// Engine health (step watchdog): false while a step exceeds the
    /// configured step timeout. Providers surface this in heartbeats.
    public var isHealthy: Bool { loop.isHealthy }
    /// Fired (once per wedge, from the watchdog thread) when a step exceeds
    /// the step timeout.
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)? {
        get { loop.onStepWedge }
        set { loop.onStepWedge = newValue }
    }
    /// Telemetry/test hooks.
    public var stepCount: Int { loop.stepCount }
    public var chainedStepCount: Int { loop.chainedStepCount }
    public var preemptionCount: Int { loop.preemptionCount }
    /// Internal test hook (engine-queue synchronized).
    var loopForTesting: EngineLoopV2 { loop }

    public init(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        backend: CBv2KVBackend,
        cacheProvider: CBv2LayerCacheProvider,
        sampler: CBv2StepSampler = CBv2GreedySampler(),
        detokenizerFactory: CBv2DetokenizerFactory = CBv2NullDetokenizerFactory(),
        schedulerConfig: CBv2SchedulerConfig = CBv2SchedulerConfig(),
        loopConfig: CBv2EngineLoopConfig = CBv2EngineLoopConfig(),
        admissionConfig: AdmissionV2.Config = AdmissionV2.Config()
    ) {
        self.schedulerConfig = schedulerConfig
        self.loopConfig = loopConfig
        let admission = AdmissionV2(
            layerKinds: layerKinds, bytesCapacity: backend.bytesCapacity, config: admissionConfig)
        self.admission = admission
        let gauges = CBv2EngineGauges(kvBytesCapacity: backend.bytesCapacity)
        self.gauges = gauges
        self.loop = EngineLoopV2(
            model: model,
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: cacheProvider,
            sampler: sampler,
            detokenizerFactory: detokenizerFactory,
            scheduler: SchedulerV2(config: schedulerConfig, capacity: admission),
            capacity: admission,
            config: loopConfig,
            gauges: gauges)
        loop.start()
    }

    // MARK: CBv2Engine

    /// Submit a request; events stream until `.finished`. Throws
    /// `CBv2KVError.capacityExhausted` when truthful admission fails (worst
    /// case could never fit), the waiting queue is full, or the engine is
    /// shutting down — the provider maps this to 429/503 exactly as today.
    public func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        stateLock.lock()
        let rejecting = rejectingSubmissions
        stateLock.unlock()
        guard !rejecting else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }

        // Degenerate requests: uniform event surface, no engine round-trip.
        if request.maxTokens <= 0 {
            return Self.immediateStream(
                reason: .length,
                usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
        }
        if request.promptTokens.isEmpty {
            return Self.immediateStream(
                reason: .error("empty prompt"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }

        // Truthful admission: reject what could NEVER fit; everything else
        // is admitted optimistically (preemption is the backstop).
        guard
            admission.canEverFit(
                promptTokens: request.promptTokens.count, maxTokens: request.maxTokens)
        else {
            throw CBv2KVError.capacityExhausted(
                needed: admission.estimatedBytes(
                    forTokens: request.promptTokens.count + request.maxTokens),
                available: admission.bytesCapacity)
        }
        guard gauges.beginSubmit(maxWaiting: schedulerConfig.maxWaiting) else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }

        let loop = self.loop
        let stream = CBv2OutputStream(
            id: request.id,
            capacity: loopConfig.eventBufferCapacity,
            onBackpressure: { id, paused in loop.setPaused(id, paused) },
            onAbandoned: { id in loop.requestCancel(id) })
        loop.register(stream: stream)
        loop.enqueue(request)
        return stream.makeStream()
    }

    /// Cancel promptly: the in-flight step completes, the row is dropped
    /// O(1) at the next step boundary.
    public func cancel(_ id: CBv2RequestID) {
        loop.requestCancel(id)
    }

    /// Truthful capacity snapshot (actual bytes/tokens, not worst case),
    /// published by the engine thread after every step.
    public func capacity() -> CBv2CapacitySnapshot {
        gauges.read()
    }

    /// Graceful drain: new submissions are rejected, waiting requests are
    /// cancelled, running requests finish naturally, then the loop stops.
    public func shutdown() async {
        beginRejectingSubmissions()
        await loop.drain()
    }

    /// Synchronous helper: `NSLock` is not async-safe to hold across
    /// suspension points, so the flag flip lives outside the async context.
    private func beginRejectingSubmissions() {
        stateLock.lock()
        rejectingSubmissions = true
        stateLock.unlock()
    }

    // MARK: Helpers

    private static func immediateStream(
        reason: CBv2FinishReason, usage: CBv2Usage
    ) -> AsyncStream<CBv2Event> {
        AsyncStream { continuation in
            continuation.yield(.finished(reason: reason, usage: usage))
            continuation.finish()
        }
    }
}
