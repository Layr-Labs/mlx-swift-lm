// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: per-request event stream with bounded
// buffering and scheduling backpressure.
//
// The engine thread NEVER blocks on a slow consumer: when a request's buffer
// reaches capacity the stream fires the backpressure callback, the engine
// pauses that request's scheduling (slot retained, decode skipped), and the
// buffer stops growing except for the ≤2 steps already in flight. Draining
// below the low watermark resumes scheduling. Events are never dropped.

import Foundation

/// Per-request producer/consumer channel behind `AsyncStream<CBv2Event>`.
///
/// Thread model: `emit`/`finish` are called from the engine thread (and, for
/// error finishes, the watchdog thread); the consumer pulls from an arbitrary
/// task. All state is lock-protected; single consumer.
public final class CBv2OutputStream: @unchecked Sendable {
    public let id: CBv2RequestID

    private let lock = NSLock()
    private var buffer: [CBv2Event] = []
    private var waiter: CheckedContinuation<CBv2Event?, Never>?
    /// Terminal event has been enqueued (or delivered); further emits no-op.
    private var finished = false
    /// Terminal event has been handed to the consumer; `next()` returns nil.
    private var terminalDelivered = false
    /// Consumer task was cancelled. Set under `lock` in the cancellation
    /// handler so a cancellation racing the park cannot strand a waiter.
    private var consumerCancelled = false
    private var pausedForBackpressure = false

    private let capacity: Int
    private var lowWatermark: Int { max(0, capacity / 2) }
    /// `(id, paused)` — fired OUTSIDE the lock, at most once per transition.
    private let onBackpressure: (@Sendable (CBv2RequestID, Bool) -> Void)?
    /// Fired when the consumer abandons the stream before the terminal event
    /// (client disconnect) — the engine maps this to `cancel(id)`.
    private let onAbandoned: (@Sendable (CBv2RequestID) -> Void)?

    public init(
        id: CBv2RequestID,
        capacity: Int = 256,
        onBackpressure: (@Sendable (CBv2RequestID, Bool) -> Void)? = nil,
        onAbandoned: (@Sendable (CBv2RequestID) -> Void)? = nil
    ) {
        precondition(capacity > 0, "CBv2OutputStream capacity must be positive")
        self.id = id
        self.capacity = capacity
        self.onBackpressure = onBackpressure
        self.onAbandoned = onAbandoned
    }

    // MARK: Producer (engine thread / watchdog)

    /// Non-blocking append. Buffer may exceed `capacity` only by the steps
    /// already in flight when the pause fired (bounded, never unbounded).
    public func emit(_ event: CBv2Event) {
        var firePause = false
        lock.lock()
        if finished {
            lock.unlock()
            return
        }
        if case .finished = event { finished = true }
        if let waiter {
            self.waiter = nil
            if case .finished = event { terminalDelivered = true }
            lock.unlock()
            waiter.resume(returning: event)
            return
        }
        buffer.append(event)
        if !pausedForBackpressure, !finished, buffer.count >= capacity {
            pausedForBackpressure = true
            firePause = true
        }
        lock.unlock()
        if firePause { onBackpressure?(id, true) }
    }

    /// Terminal emit; idempotent (safe from both loop and watchdog).
    public func finish(reason: CBv2FinishReason, usage: CBv2Usage) {
        emit(.finished(reason: reason, usage: usage))
    }

    /// True once a terminal event has been enqueued.
    public var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return finished
    }

    // MARK: Consumer

    /// Single-consumer stream of events, ending after `.finished`.
    ///
    /// PULL-based (`AsyncStream(unfolding:)`) on purpose: a push design
    /// (inner task + `continuation.yield`) would eagerly drain this bounded
    /// buffer into AsyncStream's own UNBOUNDED buffer, defeating
    /// backpressure entirely. With unfolding, `next()` runs only when the
    /// real consumer asks, so a slow consumer really does fill this buffer
    /// and trigger the scheduling pause.
    ///
    /// Cancelling the consuming task fires `onAbandoned` (client
    /// disconnect ⇒ engine cancels the request).
    ///
    /// The closures capture `self` STRONGLY on purpose: the engine drops its
    /// reference the moment it enqueues the terminal event (`takeStream`), so
    /// the returned sequence is what keeps this buffer alive until the
    /// consumer has drained it. A weak capture here would race consumer pulls
    /// against deallocation and silently drop buffered events. No cycle:
    /// `self` never references the returned AsyncStream.
    public func makeStream() -> AsyncStream<CBv2Event> {
        AsyncStream(
            unfolding: { await self.next() },
            onCancel: {
                guard !self.isFinished else { return }
                self.onAbandoned?(self.id)
            })
    }

    /// Pop the next event, parking until one arrives. Returns nil after the
    /// terminal event has been delivered or on task cancellation.
    func next() async -> CBv2Event? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<CBv2Event?, Never>) in
                var fireResume = false
                var result: CBv2Event??  // .some(event), .some(nil)=done, nil=parked
                lock.lock()
                if terminalDelivered || consumerCancelled {
                    result = .some(nil)
                } else if !buffer.isEmpty {
                    let event = buffer.removeFirst()
                    if case .finished = event { terminalDelivered = true }
                    if pausedForBackpressure, buffer.count <= lowWatermark {
                        pausedForBackpressure = false
                        fireResume = true
                    }
                    result = .some(event)
                } else {
                    waiter = c
                }
                lock.unlock()
                if fireResume { onBackpressure?(id, false) }
                if let result { c.resume(returning: result) }
            }
        } onCancel: {
            lock.lock()
            consumerCancelled = true
            let parked = waiter
            waiter = nil
            lock.unlock()
            parked?.resume(returning: nil)
        }
    }

    /// Buffered event count (tests/telemetry).
    public var bufferedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return buffer.count
    }
}
