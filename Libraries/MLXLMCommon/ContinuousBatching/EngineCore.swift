// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/engine_core.py — Async engine loop for continuous batching.
// https://github.com/jundot/omlx/blob/main/omlx/engine_core.py
//
// The EngineCore runs the scheduler step loop on a background dispatch queue,
// while the high-level interface (BatchedEngine) provides the async API.

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
    /// Optional SSD cache configuration.
    /// Requires `prefixCacheConfig` to also be set.
    public var ssdCacheConfig: SSDCacheConfig?

    public init(
        schedulerConfig: SchedulerConfig = SchedulerConfig(),
        stepInterval: TimeInterval = 0.001,
        yieldInterval: Int = 5,
        prefixCacheConfig: PrefixCacheConfig? = nil,
        ssdCacheConfig: SSDCacheConfig? = nil
    ) {
        self.schedulerConfig = schedulerConfig
        self.stepInterval = stepInterval
        self.yieldInterval = yieldInterval
        self.prefixCacheConfig = prefixCacheConfig
        self.ssdCacheConfig = ssdCacheConfig
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

    // Output collectors for streaming (vLLM pattern)
    private var outputCollectors: [String: RequestOutputCollector] = [:]
    private var streamStates: [String: RequestStreamState] = [:]
    private var finishedEvents: [String: DispatchSemaphore] = [:]

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
    @discardableResult
    public func addRequest(
        _ request: Request
    ) async -> String {
        let rid = request.requestId
        outputCollectors[rid] = RequestOutputCollector(aggregate: true)
        streamStates[rid] = RequestStreamState(
            streamInterval: config.schedulerConfig.streamInterval
        )
        finishedEvents[rid] = DispatchSemaphore(value: 0)

        // Submit to scheduler on the engine queue
        await withCheckedContinuation { continuation in
            engineQueue.async { [weak self] in
                self?.scheduler.addRequest(request)
                continuation.resume()
            }
        }

        return rid
    }

    /// Abort a request.
    public func abortRequest(_ requestId: String) -> Bool {
        let result = scheduler.abortRequest(requestId)

        // Signal the consumer
        if let collector = outputCollectors[requestId] {
            collector.put(RequestOutput(
                requestId: requestId,
                finished: true,
                finishReason: "abort",
                error: "Request aborted"
            ))
        }
        finishedEvents[requestId]?.signal()

        return result
    }

    /// Abort all active requests.
    @discardableResult
    public func abortAllRequests() -> Int {
        let rids = Array(outputCollectors.keys)
        for rid in rids {
            scheduler.abortRequest(rid)
            outputCollectors[rid]?.put(RequestOutput(
                requestId: rid,
                finished: true,
                finishReason: "error",
                error: "All requests aborted"
            ))
            finishedEvents[rid]?.signal()
        }
        return rids.count
    }

    // MARK: - Output Streaming

    /// Stream outputs for a request using non-blocking pattern.
    public func streamOutputs(
        requestId: String,
        timeout: TimeInterval? = nil
    ) -> AsyncStream<RequestOutput> {
        AsyncStream { continuation in
            let collector = outputCollectors[requestId]

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
                        // Wait for the next output
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
    public func generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> RequestOutput {
        let request = Request(
            requestId: UUID().uuidString,
            prompt: prompt,
            samplingParams: SamplingParams(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP
            )
        )

        let rid = await addRequest(request)

        // Wait for completion
        guard let event = finishedEvents[rid] else {
            throw EngineError.missingEvent
        }

        _ = await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            engineQueue.async {
                event.wait()
                cont.resume()
            }
        }

        // Drain the collector for the final output
        guard let collector = outputCollectors[rid] else {
            throw EngineError.missingOutput
        }

        var finalOutput: RequestOutput?
        while let output = collector.getNowait() {
            finalOutput = output
        }

        cleanupRequest(rid)

        guard let output = finalOutput else {
            throw EngineError.missingOutput
        }

        if let error = output.error {
            throw EngineError.generationFailed(error)
        }

        return output
    }

    // MARK: - Engine Loop

    /// Main engine loop — runs scheduler steps on the engine queue.
    private func engineLoop() async {
        while _running {
            if scheduler.hasRequests() {
                _idleSteps = 0
                let output = await withCheckedContinuation { continuation in
                    engineQueue.async { [weak self] in
                        guard let self else {
                            continuation.resume(returning: SchedulerOutput())
                            return
                        }
                        let result = self.scheduler.step()
                        stepsExecuted += 1
                        continuation.resume(returning: result)
                    }
                }

                // Distribute outputs to collectors
                let outputs = output.outputs
                if !outputs.isEmpty {
                    for reqOutput in outputs {
                        if let collector = outputCollectors[reqOutput.requestId] {
                            collector.put(reqOutput)
                        }

                        if reqOutput.finished {
                            finishedEvents[reqOutput.requestId]?.signal()
                        }
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

    private func cleanupRequest(_ requestId: String) {
        scheduler.removeFinishedRequest(requestId)
        outputCollectors.removeValue(forKey: requestId)
        streamStates.removeValue(forKey: requestId)
        finishedEvents.removeValue(forKey: requestId)
    }

    // MARK: - Stats

    public func getStats() -> [String: Any] {
        var stats: [String: Any] = [
            "running": _running,
            "steps_executed": stepsExecuted,
            "active_requests": outputCollectors.count,
        ]
        stats.merge(scheduler.getStats()) { $1 }
        return stats
    }
}

// MARK: - Errors

public enum EngineError: Error {
    case missingEvent
    case missingOutput
    case generationFailed(String)
}
