// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/engine_core.py — Async engine loop for continuous batching.
// https://github.com/jundot/omlx/blob/main/omlx/engine_core.py
//
// The EngineCore runs the scheduler step loop on a background dispatch queue,
// while the high-level interface (BatchedEngine) provides the async API.
//
// Thread-safety model (mirrors omlx's single-executor design):
//   omlx routes ALL MLX + scheduler work through a single-worker ThreadPoolExecutor,
//   making dict access effectively single-threaded.  Here we replicate that by
//   running all dict mutations (add, cleanup) on `engineQueue` and protecting
//   reads from external callers with `_lock`.  The engine loop output distribution
//   also runs inside the engineQueue block so it is serialised with addRequest
//   and cleanupRequest.

import Foundation
import MLX

// MARK: - EngineConfig

/// Configuration for the continuous-batching engine.
public struct ContinuousBatchingConfig: Sendable {
    public var schedulerConfig: SchedulerConfig
    /// Interval (in seconds) between engine loop iterations when idle.
    public var stepInterval: TimeInterval
    /// How often (in steps) to yield to the event loop.
    public var yieldInterval: Int
    /// Optional in-memory prefix cache configuration.
    /// Set to non-nil to enable block-level KV reuse across requests.
    public var prefixCacheConfig: PrefixCacheConfig?
    public init(
        schedulerConfig: SchedulerConfig = SchedulerConfig(),
        stepInterval: TimeInterval = 0.001,
        yieldInterval: Int = 5,
        prefixCacheConfig: PrefixCacheConfig? = nil
    ) {
        self.schedulerConfig = schedulerConfig
        self.stepInterval = stepInterval
        self.yieldInterval = yieldInterval
        self.prefixCacheConfig = prefixCacheConfig
    }
}

// MARK: - EngineCore

/// Core engine for continuous batching.
///
/// This engine runs the generation loop and manages request lifecycle.
/// It provides both sync and async interfaces for request handling.
///
/// Design mirrors omlx's EngineCore:
/// - Scheduler manages request lifecycle and BatchGenerator interaction
/// - EngineCore owns the async loop, output collectors, and streaming
/// - BatchedEngine provides the high-level generate/stream API
public final class EngineCore: @unchecked Sendable {
    public let scheduler: Scheduler
    public let config: ContinuousBatchingConfig

    private let engineQueue = DispatchQueue(
        label: "com.eigen.engine",
        qos: .userInitiated
    )

    // Output collectors for streaming (vLLM pattern).
    // Mutations happen ONLY on engineQueue; reads from external contexts
    // must hold _lock.  The engine queue operations also hold _lock so that
    // external reads never race with a concurrent engine-queue write.
    private var outputCollectors: [String: RequestOutputCollector] = [:]
    private var streamStates: [String: RequestStreamState] = [:]
    private let _lock = NSLock()

    // Engine state
    private var _running = false
    private var _task: Task<Void, Never>?
    private var _startTime: Date?
    public private(set) var stepsExecuted: Int = 0

    // Steps spent idle after the last active request completed.
    // After `deferredClearDelay` idle steps we flush the Metal buffer cache
    // to reclaim GPU memory. The delay prevents races with IOKit's async
    // completeMemory() callbacks that cause kernel panics on M4 hardware.
    private var _idleSteps = 0
    private static let deferredClearDelay = 8

    public init(
        scheduler: Scheduler,
        config: ContinuousBatchingConfig = ContinuousBatchingConfig()
    ) {
        self.scheduler = scheduler
        self.config = config
    }

    // MARK: - Lifecycle

    /// Start the engine loop.
    public func start() {
        guard !_running else { return }
        _running = true
        _startTime = Date()
        _task = Task { [weak self] in await self?.engineLoop() }
    }

    /// Stop the engine loop.
    public func stop() {
        _running = false
        _task?.cancel()
        _task = nil
    }

    public var isRunning: Bool { _running }

    // MARK: - Request Management

    /// Add a request for processing.
    ///
    /// Dict setup and scheduler submission both happen on `engineQueue` so that
    /// state mutations are serialised with the step loop — mirroring omlx's
    /// `loop.run_in_executor(self._mlx_executor, self.scheduler.add_request, request)`.
    @discardableResult
    public func addRequest(
        _ request: Request
    ) async -> String {
        let rid = request.requestId
        await withCheckedContinuation { continuation in
            engineQueue.async { [weak self] in
                guard let self else { continuation.resume(); return }
                _lock.lock()
                outputCollectors[rid] = RequestOutputCollector(aggregate: true)
                streamStates[rid] = RequestStreamState(
                    streamInterval: config.schedulerConfig.streamInterval
                )
                _lock.unlock()
                scheduler.addRequest(request)
                continuation.resume()
            }
        }
        return rid
    }

    /// Abort a request.
    ///
    /// Signals the consumer immediately (the collector is internally locked),
    /// then defers the scheduler abort to `engineQueue` to avoid racing
    /// scheduler state — matching omlx's deferred-abort comment.
    public func abortRequest(_ requestId: String) -> Bool {
        _lock.lock()
        let collector = outputCollectors[requestId]
        _lock.unlock()
        guard let collector else { return false }

        collector.put(RequestOutput(
            requestId: requestId,
            finished: true,
            finishReason: "abort",
            error: "Request aborted"
        ))

        engineQueue.async { [weak self] in
            _ = self?.scheduler.abortRequest(requestId)
        }
        return true
    }

    /// Abort all active requests.
    @discardableResult
    public func abortAllRequests() -> Int {
        _lock.lock()
        let rids = Array(outputCollectors.keys)
        _lock.unlock()

        for rid in rids {
            engineQueue.async { [weak self] in
                _ = self?.scheduler.abortRequest(rid)
            }
            _lock.lock()
            let collector = outputCollectors[rid]
            _lock.unlock()
            collector?.put(RequestOutput(
                requestId: rid,
                finished: true,
                finishReason: "error",
                error: "All requests aborted"
            ))
        }
        return rids.count
    }

    // MARK: - Output Streaming

    /// Stream outputs for a request using non-blocking pattern.
    public func streamOutputs(
        requestId: String,
        timeout: TimeInterval? = nil
    ) -> AsyncStream<RequestOutput> {
        _lock.lock()
        let collector = outputCollectors[requestId]
        _lock.unlock()

        return AsyncStream { continuation in
            Task {
                guard let collector else {
                    continuation.finish()
                    return
                }

                while true {
                    if let output = collector.getNowait() {
                        continuation.yield(output)
                        if output.finished || output.error != nil {
                            break
                        }
                    } else {
                        let output = await collector.get()
                        continuation.yield(output)
                        if output.finished || output.error != nil {
                            break
                        }
                    }
                }

                cleanupRequest(requestId)
                continuation.finish()
            }
        }
    }

    /// Generate a complete response (non-streaming).
    ///
    /// Reuses `streamOutputs` rather than a blocking semaphore wait on
    /// `engineQueue`, eliminating the thread-stall / potential deadlock from
    /// the original `DispatchSemaphore.wait()` design.
    public func generate(
        prompt: String,
        samplingParams: SamplingParams
    ) async throws -> RequestOutput {
        let request = Request(
            requestId: UUID().uuidString,
            prompt: prompt,
            samplingParams: samplingParams
        )

        let rid = await addRequest(request)

        return try await withTaskCancellationHandler {
            for await output in streamOutputs(requestId: rid) {
                if output.finished || output.error != nil {
                    if let error = output.error {
                        throw EngineError.generationFailed(error)
                    }
                    return output
                }
            }
            throw EngineError.missingOutput
        } onCancel: { [weak self] in
            _ = self?.abortRequest(rid)
        }
    }

    // MARK: - Engine Loop

    /// Main engine loop — runs scheduler steps on the engine queue.
    ///
    /// Output distribution runs INSIDE the `engineQueue.async` block so that
    /// all `outputCollectors` access is serialised with `addRequest` and
    /// `cleanupRequest`, matching omlx's single-executor model.
    private func engineLoop() async {
        while _running {
            if scheduler.hasRequests() {
                _idleSteps = 0
                await withCheckedContinuation { continuation in
                    engineQueue.async { [weak self] in
                        guard let self else {
                            continuation.resume()
                            return
                        }
                        let output = scheduler.step()
                        stepsExecuted += 1

                        // Distribute outputs to collectors inside the queue block.
                        if !output.outputs.isEmpty {
                            for reqOutput in output.outputs {
                                _lock.lock()
                                let collector = outputCollectors[reqOutput.requestId]
                                _lock.unlock()
                                collector?.put(reqOutput)
                            }
                        }

                        continuation.resume()
                    }
                }
            } else {
                _idleSteps += 1
                // Flush the Metal buffer cache after a brief idle window to
                // reclaim GPU memory. The delay avoids races with IOKit's
                // async completeMemory() callbacks (M4 kernel-panic fix).
                if _idleSteps == Self.deferredClearDelay {
                    Stream().synchronize()
                    Memory.clearCache()
                }
                try? await Task.sleep(nanoseconds: UInt64(config.stepInterval * 1_000_000_000))
            }

            if Task.isCancelled { break }
        }
    }

    // MARK: - Cleanup

    /// Dispatch cleanup to `engineQueue` so dict mutations are serialised
    /// with the step loop — the caller (streamOutputs Task) may run on any thread.
    private func cleanupRequest(_ requestId: String) {
        engineQueue.async { [weak self] in
            guard let self else { return }
            scheduler.removeFinishedRequest(requestId)
            _lock.lock()
            outputCollectors.removeValue(forKey: requestId)
            streamStates.removeValue(forKey: requestId)
            _lock.unlock()
        }
    }

    // MARK: - Stats

    public func getStats() -> [String: Any] {
        _lock.lock()
        let activeRequests = outputCollectors.count
        _lock.unlock()
        var stats: [String: Any] = [
            "running": _running,
            "steps_executed": stepsExecuted,
            "active_requests": activeRequests,
        ]
        stats.merge(scheduler.getStats()) { $1 }
        return stats
    }
}

// MARK: - Errors

public enum EngineError: Error {
    case missingOutput
    case generationFailed(String)
}
