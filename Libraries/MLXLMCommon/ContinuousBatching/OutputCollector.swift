// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/output_collector.py — vLLM-style streaming output collector.
// https://github.com/jundot/omlx/blob/main/omlx/output_collector.py

import Foundation

/// Per-request output collector with smart buffering.
///
/// Implements the vLLM pattern for efficient streaming:
/// - Non-blocking `getNowait()` to avoid unnecessary task switches
/// - Output aggregation when producer is faster than consumer
/// - Event-based signaling for efficient waiting
///
/// ### Usage
/// ```swift
/// let collector = RequestOutputCollector()
///
/// // Producer side (engine loop):
/// collector.put(output)
///
/// // Consumer side (streaming generator):
/// if let output = collector.getNowait() {
///     // use it
/// } else {
///     // await collector.get()
/// }
/// ```
public final class RequestOutputCollector: @unchecked Sendable {
    // A small FIFO of pending outputs. Adjacent COALESCEABLE outputs are merged
    // (or overwritten when !aggregate) into the tail, so an all-streaming run
    // still holds at most one element — byte-for-byte the prior single-slot
    // behavior. A NON-coalesceable output (the prefill-start admission marker)
    // is appended as its own discrete entry and never merged with a neighbor, so
    // the consumer observes admission and the first token as separate outputs
    // even when it drains slowly (the case that previously collapsed the
    // admitted→first-token window).
    private var _buffer: [RequestOutput] = []
    private let _lock = NSLock()
    private var _waitingContinuations: [CheckedContinuation<RequestOutput, Never>] = []
    private let aggregate: Bool

    public init(aggregate: Bool = true) {
        self.aggregate = aggregate
    }

    /// Put an output into the collector (non-blocking).
    /// When aggregation is enabled the new output is merged with the buffered
    /// tail — UNLESS either the tail or the incoming output is non-coalesceable,
    /// in which case it is appended as a discrete entry.
    public func put(_ output: RequestOutput) {
        _lock.lock()

        // Fast path: deliver directly to a waiting consumer without buffering.
        // Mirrors omlx's asyncio.Event pattern. A waiter is only ever registered
        // while the buffer is empty (see get()), and put/get are serialized by
        // `_lock`, so the buffer is guaranteed empty here — direct delivery
        // therefore preserves FIFO order.
        if !_waitingContinuations.isEmpty {
            let continuations = _waitingContinuations
            _waitingContinuations.removeAll()
            _lock.unlock()
            for cont in continuations {
                cont.resume(returning: output)
            }
            return
        }

        // Merge only a run of adjacent coalesceable outputs into the tail; a
        // non-coalesceable marker (or a marker already at the tail) forces a new
        // discrete entry so admission and the first token are never collapsed.
        if let last = _buffer.last, last.coalesceable, output.coalesceable {
            if aggregate {
                _buffer[_buffer.count - 1] = mergeOutputs(last, output)
            } else {
                _buffer[_buffer.count - 1] = output
            }
        } else {
            _buffer.append(output)
        }
        _lock.unlock()
    }

    /// Get output without blocking. Returns the oldest buffered output (FIFO), or
    /// nil if none available.
    public func getNowait() -> RequestOutput? {
        _lock.lock()
        defer { _lock.unlock() }
        return _buffer.isEmpty ? nil : _buffer.removeFirst()
    }

    /// Get output, blocking only if none available.
    /// For low-latency streaming, prefer:
    /// ```swift
    /// let output = collector.getNowait() ?? await collector.get()
    /// ```
    public func get() async -> RequestOutput {
        await withCheckedContinuation { continuation in
            _lock.lock()
            if !_buffer.isEmpty {
                let output = _buffer.removeFirst()
                _lock.unlock()
                continuation.resume(returning: output)
            } else {
                _waitingContinuations.append(continuation)
                _lock.unlock()
            }
        }
    }

    /// Clear any pending output.
    public func clear() {
        _lock.lock()
        _buffer.removeAll()
        _waitingContinuations.removeAll()
        _lock.unlock()
    }

    // MARK: - Merge

    private func mergeOutputs(_ existing: RequestOutput, _ new: RequestOutput) -> RequestOutput {
        RequestOutput(
            requestId: new.requestId,
            newTokenIds: existing.newTokenIds + new.newTokenIds,
            newText: existing.newText + new.newText,
            outputTokenIds: new.outputTokenIds,
            outputText: new.outputText,
            finished: new.finished,
            finishReason: new.finishReason,
            promptTokens: new.promptTokens,
            completionTokens: new.completionTokens,
            cachedTokens: new.cachedTokens,
            error: new.error ?? existing.error
        )
    }
}

/// Tracks streaming state for a request.
/// Used to implement stream_interval batching,
/// allowing tokens to be accumulated before sending.
public struct RequestStreamState: Sendable {
    public let streamInterval: Int
    public private(set) var sentTokens: Int

    public init(streamInterval: Int = 1) {
        self.streamInterval = streamInterval
        self.sentTokens = 0
    }

    /// Determine if output should be sent based on streamInterval.
    /// Always sends on finish and first token (for low TTFT).
    public func shouldSend(totalTokens: Int, finished: Bool) -> Bool {
        if finished { return true }
        if sentTokens == 0 { return true }
        return (totalTokens - sentTokens) >= streamInterval
    }

    public mutating func markSent(totalTokens: Int) {
        sentTokens = totalTokens
    }
}
