// Cross-token, cross-layer speculative expert prefetch for MoE SSD
// streaming.
//
// THE IDLE-SSD PROBLEM: while the GPU computes layer L for token t, the SSD
// sits idle -- `StreamingQuantizedSwitchGLU` only issues a fetch for a
// layer's experts once that layer's router has run, and layer L+1's router
// hasn't run yet (it needs layer L's output). So each layer's fetch is
// entirely on the critical path: compute layer L -> block on I/O for layer
// L+1's experts -> compute layer L+1 -> ...
//
// THE FIX (temporal locality across tokens, not within one): MoE routing is
// noisy per-token but the routing WEIGHTS are fixed, so which experts a
// layer picks is strongly correlated across consecutive tokens for the same
// decode stream -- token t's layer-L+1 experts heavily overlap token t-1's.
// So instead of trying to predict token t's future layers from token t's
// OWN (not-yet-available) routing, this coordinator uses token t-1's
// ALREADY-KNOWN routing for those layers as a speculative guess, and kicks
// background fetches for it while token t's forward pass is still busy
// computing earlier layers. By the time the forward pass actually reaches
// that layer, some or all of its experts are already warm in `ExpertCache`.
//
// This file is split into:
//  - pure bookkeeping (`recordSelection`, `plannedPrefetchTargets`) that
//    tracks "what did each layer select last time" and computes prefetch
//    targets with NO I/O -- independently unit-testable.
//  - I/O-side-effecting scheduling (`prefetchAhead`) that fans the planned
//    targets out to a bounded-concurrency background queue.

import Foundation
import MLX

/// Speculatively warms `ExpertCache` with the experts a downstream layer is
/// LIKELY to need, based on what that layer selected for the previous
/// token. Never blocks the forward pass: `prefetchAhead` only enqueues
/// background work and returns immediately.
///
/// IN-FLIGHT DEDUP CHOICE (v1): a simple "in-flight keys" `Set` guards
/// against scheduling the SAME (layer, expert) prefetch twice while one is
/// already running, but does NOT prevent the race where the foreground
/// (synchronous, on-the-critical-path) fetch in `StreamingQuantizedSwitchGLU`
/// asks for the same expert while a background prefetch for it is still in
/// flight. In that case both issue a real `pread` and the loser's result is
/// simply discarded (`ExpertCache.insert` is a no-op if the key already
/// exists). This is the "let the double-read happen" option from the
/// design brief: it costs one redundant disk read in the rare case the
/// forward pass catches up to an in-flight prefetch, which is strictly
/// cheaper than adding a condition-variable wait that could stall the
/// foreground path on a background fetch's completion -- and blocking the
/// forward pass is the one thing this feature must never do.
public final class PrefetchCoordinator: @unchecked Sendable {
    private let lock = NSLock()
    /// Most recently observed expert selection per layer, i.e. "what did
    /// this layer pick last time it ran" -- for the layer the forward pass
    /// is CURRENTLY at, that's this token's pick; for layers further ahead
    /// (not yet reached this token), it's still the previous token's pick,
    /// which is exactly the speculative signal this coordinator exploits.
    private var lastSelectedExperts: [Int: [Int]] = [:]
    private var inFlightKeys: Set<ExpertCache.Key> = []
    private var scheduledCountLocked: Int = 0
    private var completedCountLocked: Int = 0
    private var alreadyWarmCountLocked: Int = 0

    public let enabled: Bool
    public let totalLayers: Int
    public let lookaheadLayers: Int
    public let maxInFlight: Int

    private let cache: ExpertCache
    private let store: ExpertShardStore
    private let queue: OperationQueue

    public init(
        cache: ExpertCache,
        store: ExpertShardStore,
        totalLayers: Int,
        lookaheadLayers: Int = 2,
        maxInFlight: Int = 3,
        enabled: Bool = true
    ) {
        self.cache = cache
        self.store = store
        self.totalLayers = totalLayers
        self.lookaheadLayers = max(0, lookaheadLayers)
        self.maxInFlight = max(1, maxInFlight)
        self.enabled = enabled

        let queue = OperationQueue()
        queue.name = "com.eigenlabs.dsv4.expert-prefetch"
        // .utility, not .userInitiated: prefetch is a speculative bonus,
        // not real work. It must not compete for I/O/CPU priority against
        // the foreground fetch that the forward pass is actually blocked
        // on (ExpertShardStore.fetch's own DispatchQueue.concurrentPerform).
        queue.qualityOfService = .utility
        queue.maxConcurrentOperationCount = self.maxInFlight
        self.queue = queue
    }

    /// Reads `DSV4_STREAM_PREFETCH` (default DISABLED -- set to `"1"` to
    /// enable).
    ///
    /// Opt-in, not opt-out, per the measured result that motivated it
    /// (dd3f27d + review): at production cache sizes (70 GiB) the previous-
    /// token signal schedules ZERO real fetches -- everything it would warm
    /// is already resident from that token's own foreground fetch -- and at
    /// eviction-heavy cache sizes where it DOES fire it measured neutral to
    /// slightly NEGATIVE (I/O contention with the foreground fetch on an
    /// already I/O-bound path). The machinery is kept for a future smarter
    /// signal (frequency-based prediction, deeper lookahead); flip the
    /// default back only with a measurement that justifies it.
    public static func enabledFromEnv() -> Bool {
        ProcessInfo.processInfo.environment["DSV4_STREAM_PREFETCH"] == "1"
    }

    /// Reads `DSV4_STREAM_PREFETCH_LOOKAHEAD` (int, default 2 -- matches
    /// the L+1..L+2 lookahead from the design brief).
    public static func lookaheadFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_PREFETCH_LOOKAHEAD"],
            let n = Int(raw), n >= 0
        {
            return n
        }
        return 2
    }

    /// Reads `DSV4_STREAM_PREFETCH_MAX_INFLIGHT` (int, default 3).
    public static func maxInFlightFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_PREFETCH_MAX_INFLIGHT"],
            let n = Int(raw), n > 0
        {
            return n
        }
        return 3
    }

    // MARK: - Bookkeeping (pure, no I/O, unit-testable without a real store)

    /// Record the expert ids a layer just selected for the token currently
    /// being processed. Overwrites whatever was recorded for that layer
    /// (from the previous token) -- by design, once the forward pass
    /// reaches a layer, its OWN routing for this token becomes the best
    /// available signal for the NEXT token's speculative prefetch.
    public func recordSelection(layer: Int, experts: [Int]) {
        lock.lock()
        defer { lock.unlock() }
        lastSelectedExperts[layer] = experts
    }

    /// Test/inspection hook: what's currently recorded for `layer` (nil if
    /// that layer hasn't run yet in this process).
    func recordedSelection(layer: Int) -> [Int]? {
        lock.lock()
        defer { lock.unlock() }
        return lastSelectedExperts[layer]
    }

    /// Pure computation of prefetch targets: having just finished
    /// `fromLayer`'s router pick for the CURRENT token, which (layer,
    /// expert) pairs should we speculatively warm? Looks at layers
    /// `fromLayer+1 ... fromLayer+lookaheadLayers`, each still holding the
    /// PREVIOUS token's recorded pick (this token hasn't reached them yet),
    /// clipped to `totalLayers`. No I/O, no cache/store access -- safe to
    /// call from a unit test with a fake coordinator state.
    func plannedPrefetchTargets(fromLayer: Int) -> [(layer: Int, expert: Int)] {
        lock.lock()
        defer { lock.unlock() }
        guard lookaheadLayers > 0 else { return [] }
        var targets: [(layer: Int, expert: Int)] = []
        for lookahead in 1 ... lookaheadLayers {
            let targetLayer = fromLayer + lookahead
            guard targetLayer < totalLayers, let experts = lastSelectedExperts[targetLayer] else {
                continue
            }
            for e in experts { targets.append((targetLayer, e)) }
        }
        return targets
    }

    // MARK: - Scheduling (I/O side-effecting)

    /// Kick background, non-blocking prefetch for the layers ahead of
    /// `fromLayer`, using each target layer's PREVIOUS-token selection.
    /// Always returns immediately -- all actual disk I/O happens on the
    /// bounded-concurrency background queue.
    public func prefetchAhead(fromLayer: Int) {
        guard enabled else { return }
        for target in plannedPrefetchTargets(fromLayer: fromLayer) {
            schedule(layer: target.layer, expert: target.expert)
        }
    }

    private func schedule(layer: Int, expert: Int) {
        // Already resident -- nothing to do. Checked via `contains` (not
        // `get`) so this speculative peek never pollutes the cache's
        // reported hit/miss stats or recency order.
        guard !cache.contains(layer: layer, expert: expert) else {
            lock.lock()
            alreadyWarmCountLocked += 1
            lock.unlock()
            return
        }

        let key = ExpertCache.Key(layer: layer, expert: expert)
        lock.lock()
        guard !inFlightKeys.contains(key) else {
            lock.unlock()
            return
        }
        inFlightKeys.insert(key)
        scheduledCountLocked += 1
        lock.unlock()

        queue.addOperation { [weak self] in
            guard let self else { return }
            defer {
                self.lock.lock()
                self.inFlightKeys.remove(key)
                self.completedCountLocked += 1
                self.lock.unlock()
            }
            // Re-check: the foreground path (or another prefetch queued
            // just ahead of this one) may have already resolved this
            // expert while this operation waited for a queue slot.
            guard !self.cache.contains(layer: layer, expert: expert) else { return }
            do {
                let fetched = try self.store.fetch(layerIndex: layer, experts: [expert])
                if let w = fetched[expert] {
                    self.cache.insert(layer: layer, expert: expert, weights: w)
                }
            } catch {
                // Best-effort: a real checkpoint/layout mismatch will also
                // fail the foreground fetch loudly (`fatalError` in
                // `StreamingQuantizedSwitchGLU.fetchChunk`); a background
                // prefetch failure must not crash the process on its own.
            }
        }
    }

    /// Test hook: number of prefetches currently in flight (running or
    /// queued behind `maxInFlight` running operations).
    var inFlightCountForTesting: Int {
        lock.lock()
        defer { lock.unlock() }
        return inFlightKeys.count
    }

    /// Blocks until all currently scheduled prefetch operations finish.
    /// Test-only -- production code must never wait on the prefetch queue.
    func waitUntilIdleForTesting() {
        queue.waitUntilAllOperationsAreFinished()
    }

    public struct Stats: Sendable {
        /// Background fetches actually dispatched to the queue (target was
        /// cold and not already in flight).
        public let scheduled: Int
        /// Background fetches that finished (success or failure).
        public let completed: Int
        /// Prefetch targets that were ALREADY resident by the time
        /// `schedule` looked -- i.e. either a previous prefetch or the
        /// foreground path already warmed them. High relative to
        /// `scheduled` means the natural LRU cache is already doing most of
        /// the work and prefetch has little left to contribute.
        public let alreadySkippedWarm: Int
    }

    public var stats: Stats {
        lock.lock()
        defer { lock.unlock() }
        return Stats(
            scheduled: scheduledCountLocked, completed: completedCountLocked,
            alreadySkippedWarm: alreadyWarmCountLocked)
    }
}
