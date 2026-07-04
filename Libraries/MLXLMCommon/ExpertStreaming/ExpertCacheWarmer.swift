// Background cache warming for MoE SSD streaming, driven by a persisted
// `ExpertUsageProfile` (see ExpertUsageProfile.swift).
//
// THE FIX THIS ADDRESSES: `ExpertCache` starts a fresh process at 0% hit
// rate and only converges to steady state after enough real forward passes
// have organically populated it -- costing the first request(s) a much
// higher TTFT and a slower early decode. If a usage profile from a PAST
// session against the same checkpoint already exists, this type kicks a
// background task at model-load time that fetches the historically hottest
// experts (per the profile) into the cache BEFORE the first request needs
// them, so as much of that convergence as possible happens off the
// request's critical path.
//
// SAFETY CONTRACT (all enforced below):
//   - Never blocks the caller: `start()` returns immediately; all I/O runs
//     on a background queue.
//   - Bounded concurrency, background QoS: mirrors `PrefetchCoordinator`'s
//     `.utility` + `maxConcurrentOperationCount` pattern so warming never
//     competes for I/O/CPU priority against a real request.
//   - Yields to foreground fetches: warming issues ONE expert fetch at a
//     time per queue slot (not the batched multi-expert fetch the
//     foreground path uses) and re-checks cancellation between every
//     single fetch, so it releases queue slots quickly rather than holding
//     them through a long batch while a real request is waiting on the
//     same SSD queue depth.
//   - Stops once real traffic has enough of its own signal: see
//     `shouldStopForForegroundTraffic`.
//   - Disable-able via `DSV4_STREAM_WARM=0`.
//   - Dedupes against in-flight foreground fetches the same way
//     `PrefetchCoordinator` already does (see the note on `warmOneKey`
//     below) -- a `cache.contains` check immediately before the read, and
//     accepting that a genuine race against a concurrent foreground fetch
//     costs one redundant `pread` rather than adding lock/wait machinery
///    on the hot path.
//   - Respects the byte budget by construction: every insert goes through
//     `ExpertCache.insertAtColdEnd`, which evicts under the SAME budget
//     `evictLocked()` already enforces for the resident cache.

import Foundation

public final class ExpertCacheWarmer: @unchecked Sendable {

    public struct Stats: Sendable {
        /// Experts actually fetched from disk and inserted (cold-end) by
        /// this warmer.
        public let warmed: Int
        /// Total bytes warmed (sum of `warmed` experts' `byteCount`).
        public let warmedBytes: Int
        /// Warm candidates skipped because they were already resident by
        /// the time this warmer got to them (organic traffic, or a prior
        /// PrefetchCoordinator run, beat it there).
        public let skippedAlreadyResident: Int
        /// True once the task has stopped (completed, exhausted the
        /// profile, cancelled, or disabled) -- false while still running.
        public let finished: Bool
        /// Why the task stopped, for diagnostics (empty while running).
        public let stopReason: String
        /// Wall-clock milliseconds from `start()` to completion (nil while
        /// still running).
        public let elapsedMs: Double?
        public let profileEntryCount: Int
    }

    private let lock = NSLock()
    private var startedLocked = false
    private var cancelledLocked = false
    private var stoppedForForegroundLocked = false
    private var warmedCountLocked = 0
    private var warmedBytesLocked = 0
    private var skippedLocked = 0
    private var finishedLocked = false
    private var stopReasonLocked = ""
    private var startTime: CFAbsoluteTime = 0
    private var elapsedMsLocked: Double?

    private let cache: ExpertCache
    private let store: ExpertShardStore
    private let profile: ExpertUsageProfile
    private let targetFraction: Double
    private let maxInFlight: Int
    /// Forward calls (see `ExpertUsageProfile.totalForwardCalls`) after
    /// which the warmer voluntarily stops even if it hasn't exhausted its
    /// candidate list or hit the byte target -- see
    /// `shouldStopForForegroundTraffic`.
    private let stopAfterForegroundCalls: Int
    private let queue: OperationQueue

    public let enabled: Bool

    public init(
        cache: ExpertCache,
        store: ExpertShardStore,
        profile: ExpertUsageProfile,
        targetFraction: Double = ExpertCacheWarmer.targetFractionFromEnv(),
        maxInFlight: Int = ExpertCacheWarmer.maxInFlightFromEnv(),
        stopAfterForegroundCalls: Int? = nil,
        enabled: Bool = ExpertCacheWarmer.enabledFromEnv()
    ) {
        self.cache = cache
        self.store = store
        self.profile = profile
        self.targetFraction = min(max(targetFraction, 0), 1)
        self.maxInFlight = max(1, maxInFlight)
        // Default: four full decode steps' worth of streamed-layer forward
        // calls (`4 * totalLayers`). By then the organic LRU already holds
        // this SPECIFIC request's real routing for every streamed layer at
        // least four times over, which is a stronger, request-specific
        // signal than the historical global profile the warmer is
        // guessing from -- and a live request's own synchronous fetches
        // are now competing with the warmer for the same SSD queue depth,
        // so continuing to warm past this point trades a request's own
        // latency for decreasingly-useful speculation.
        self.stopAfterForegroundCalls = stopAfterForegroundCalls ?? max(1, profile.totalLayers * 4)
        self.enabled = enabled

        let queue = OperationQueue()
        queue.name = "com.eigenlabs.dsv4.expert-cache-warmer"
        queue.qualityOfService = .utility
        queue.maxConcurrentOperationCount = self.maxInFlight
        self.queue = queue
    }

    /// Reads `DSV4_STREAM_WARM` (default ENABLED -- set to `"0"` to
    /// disable background warm-on-load entirely).
    public static func enabledFromEnv() -> Bool {
        ProcessInfo.processInfo.environment["DSV4_STREAM_WARM"] != "0"
    }

    /// Reads `DSV4_STREAM_WARM_TARGET_FRACTION` (double in (0, 1], default
    /// 0.8). Warming stops at 80% of budget by default (rather than 100%)
    /// so the LAST slice of cache headroom is always immediately available
    /// to the first real request's own fetches without having to evict a
    /// warm guess first -- a small amount of always-free headroom is
    /// cheaper than the eviction-then-fetch latency it would otherwise add
    /// to that very first foreground miss.
    public static func targetFractionFromEnv() -> Double {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_WARM_TARGET_FRACTION"],
            let f = Double(raw), f > 0, f <= 1
        {
            return f
        }
        return 0.8
    }

    /// Reads `DSV4_STREAM_WARM_MAX_INFLIGHT` (int, default 4).
    public static func maxInFlightFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_WARM_MAX_INFLIGHT"],
            let n = Int(raw), n > 0
        {
            return n
        }
        return 4
    }

    /// Whether the background task has been kicked (idempotent: a second
    /// `start()` call is a no-op so callers building this per-layer, same
    /// pattern as `DeepseekV4ExpertStreaming.store`/`prefetchCoordinator`,
    /// never accidentally launch it twice).
    public func start() {
        guard enabled else {
            lock.lock()
            finishedLocked = true
            stopReasonLocked = "disabled"
            lock.unlock()
            return
        }
        lock.lock()
        guard !startedLocked else {
            lock.unlock()
            return
        }
        startedLocked = true
        startTime = CFAbsoluteTimeGetCurrent()
        lock.unlock()

        // Candidate computation can do a real (single, tiny) probe read
        // (see `estimatedBytesPerExpert`) when the cache is empty, so it
        // runs on the SAME background queue as the actual warm fetches --
        // `start()` itself must return immediately no matter what state
        // the cache/profile are in, since it is called from the model-load
        // path (`DeepseekV4MoE.init`), and a hosting hooking model load
        // lazily off the first request's own execution must never see
        // this block for even one disk read.
        queue.addOperation { [weak self] in
            self?.buildAndScheduleWarmCandidates()
        }
    }

    private func buildAndScheduleWarmCandidates() {
        lock.lock()
        let cancelledBeforeStart = cancelledLocked
        lock.unlock()
        guard !cancelledBeforeStart else {
            finish(reason: "cancelled")
            return
        }

        let candidates = warmCandidates()
        print(
            "[expert-warm] starting background warm: \(candidates.count) candidates, "
                + "target=\(Int(targetFraction * 100))% of budget, maxInFlight=\(maxInFlight), "
                + "profileEntries=\(profile.snapshotCounts().count)")

        guard !candidates.isEmpty else {
            finish(reason: "empty profile")
            return
        }

        // All candidates are enqueued up front; `OperationQueue`'s
        // `maxConcurrentOperationCount` (== maxInFlight) is what actually
        // bounds concurrency -- NOT a hand-rolled "one in flight, schedule
        // the next on completion" chain, which would serialize warming to
        // one fetch at a time regardless of `maxInFlight` and defeat the
        // whole point of bounding (as opposed to just limiting)
        // concurrency. Each operation independently re-checks
        // cancellation/foreground-traffic before doing any I/O, so a
        // `cancel()` (or foreground traffic crossing the stop threshold)
        // mid-run makes every not-yet-started operation a fast no-op
        // rather than actually being pulled out of the queue.
        let group = DispatchGroup()
        for key in candidates {
            group.enter()
            queue.addOperation { [weak self] in
                defer { group.leave() }
                self?.runIfStillWarranted(key)
            }
        }
        group.notify(queue: .global(qos: .utility)) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let reason: String
            if self.stoppedForForegroundLocked {
                reason = "foreground traffic active"
            } else if self.cancelledLocked {
                reason = "cancelled"
            } else {
                reason = "completed"
            }
            self.lock.unlock()
            self.finish(reason: reason)
        }
    }

    /// Runs inside one queue operation: bails out fast (without touching
    /// disk) if warming should no longer proceed, otherwise performs
    /// exactly one expert fetch+insert.
    private func runIfStillWarranted(_ key: ExpertCache.Key) {
        lock.lock()
        let alreadyStopped = cancelledLocked
        lock.unlock()
        guard !alreadyStopped else { return }

        guard !shouldStopForForegroundTraffic() else {
            lock.lock()
            cancelledLocked = true
            stoppedForForegroundLocked = true
            lock.unlock()
            return
        }

        warmOneKey(key)
    }

    /// Cancel any remaining warm work -- called on model unload/reload
    /// (`DeepseekV4ExpertStreaming.purgeCache()`) so a stale warm task
    /// never keeps issuing reads for a checkpoint the process has already
    /// moved on from. Sets the same flag every queued operation already
    /// checks first thing (`runIfStillWarranted`) rather than relying on
    /// `OperationQueue`'s own cancellation (which only marks `Operation`
    /// objects cancelled -- plain closures added via `addOperation(_:)`
    /// don't consult that automatically), so already-queued-but-not-yet-run
    /// operations become near-instant no-ops without needing a handle to
    /// each individual `Operation`.
    public func cancel() {
        lock.lock()
        cancelledLocked = true
        lock.unlock()
    }

    public var stats: Stats {
        lock.lock()
        defer { lock.unlock() }
        return Stats(
            warmed: warmedCountLocked, warmedBytes: warmedBytesLocked,
            skippedAlreadyResident: skippedLocked, finished: finishedLocked,
            stopReason: stopReasonLocked, elapsedMs: elapsedMsLocked,
            profileEntryCount: profile.snapshotCounts().count)
    }

    // MARK: - Candidate computation

    /// Byte-per-expert estimate used to size the candidate list against
    /// `targetFraction * budget`. There is no cheap way to know a
    /// particular expert's exact byte count without reading it (that's the
    /// whole point of streaming), so this prefers the cache's CURRENT
    /// average resident bytes-per-entry when anything is already resident
    /// (a live, checkpoint-correct estimate needing no extra I/O), and
    /// otherwise does a single real probe read of the single
    /// highest-ranked candidate (`topCandidate`, already computed by the
    /// caller from the SAME ranking `warmOrder` will use) -- a fresh
    /// process's cache is empty by definition the first time this runs, so
    /// there is no free estimate available yet, and every real DeepSeek-V4
    /// routed expert in a given checkpoint has the same per-expert byte
    /// count (same shapes/quantization across the `switch_mlp` stack), so
    /// one probe accurately sizes every other candidate too. Falls back to
    /// a conservative flat 16 MiB guess only if the probe itself fails
    /// (e.g. a checkpoint/layout mismatch that will fail loudly elsewhere
    /// anyway) -- overestimating there is intentionally safe: it just
    /// means the candidate list undershoots the true target a bit rather
    /// than risking scheduling more than fits.
    private func estimatedBytesPerExpert(topCandidate: ExpertCache.Key?) -> Int {
        let stats = cache.stats
        if stats.residentCount > 0 {
            return max(1, stats.residentBytes / stats.residentCount)
        }
        if let topCandidate,
            let probe = try? store.fetch(layerIndex: topCandidate.layer, experts: [topCandidate.expert]),
            let weights = probe[topCandidate.expert]
        {
            return max(1, weights.byteCount)
        }
        return 16 * 1024 * 1024
    }

    private func warmCandidates() -> [ExpertCache.Key] {
        let budget = cache.currentByteBudget
        let targetBytes = Int(Double(budget) * targetFraction)
        let counts = profile.snapshotCounts()
        guard targetBytes > 0, !counts.isEmpty else { return [] }

        // Full ranking (no byte limit yet) so the probe above and the
        // final budget-limited list are guaranteed to agree on which
        // candidate is "the top one" -- avoids duplicating warmOrder's
        // tie-break comparator here.
        let fullRanking = ExpertUsageProfile.warmOrder(
            counts: counts, byteBudget: Int.max, bytesPerExpert: 1)
        let bytesPerExpert = estimatedBytesPerExpert(topCandidate: fullRanking.first)
        let order = ExpertUsageProfile.warmOrder(
            counts: counts, byteBudget: targetBytes, bytesPerExpert: bytesPerExpert)

        // Skip anything already resident up front -- cheap, avoids
        // scheduling (and occupying a bounded queue slot for) operations
        // that would immediately no-op in `warmOneKey`. Still counted into
        // `skippedAlreadyResident` here (not just inside `warmOneKey`) so
        // `stats` reports the SAME total regardless of whether a candidate
        // was recognized as resident before or after being scheduled --
        // callers/tests must be able to rely on
        // `warmed + skippedAlreadyResident == candidates considered`.
        var kept: [ExpertCache.Key] = []
        kept.reserveCapacity(order.count)
        var preSkipped = 0
        for key in order {
            if cache.contains(layer: key.layer, expert: key.expert) {
                preSkipped += 1
            } else {
                kept.append(key)
            }
        }
        if preSkipped > 0 {
            lock.lock()
            skippedLocked += preSkipped
            lock.unlock()
        }
        return kept
    }

    // MARK: - Scheduling

    /// Whether real traffic has produced enough of its own signal that
    /// further speculative warming should stop (see the constructor
    /// comment on `stopAfterForegroundCalls` for the threshold rationale).
    private func shouldStopForForegroundTraffic() -> Bool {
        profile.totalForwardCalls >= stopAfterForegroundCalls
    }

    /// Fetch and insert exactly one candidate expert, if it's still worth
    /// fetching by the time this queue slot runs.
    ///
    /// DEDUPE VS. FOREGROUND, v1 (same trade-off `PrefetchCoordinator`
    /// already makes -- see its file header): re-check `cache.contains`
    /// immediately before the `pread`, so the common case (foreground
    /// already resolved this expert while this operation waited for a
    /// queue slot) is skipped entirely. A genuine race where the
    /// foreground path starts fetching the SAME expert at the SAME instant
    /// this warm operation does is still possible and not specially
    /// guarded against -- both issue a real disk read, and
    /// `ExpertCache.insertAtColdEnd`/`insert` silently no-ops whichever one
    /// loses, so correctness is unaffected either way. Adding a
    /// cross-path in-flight registry to close that narrow window would add
    /// a lock a real request's foreground fetch has to check on every
    /// miss, which is not worth it to avoid an occasional redundant read
    /// on an already I/O-bound path -- consistent with the existing
    /// prefetch coordinator's documented reasoning.
    private func warmOneKey(_ key: ExpertCache.Key) {
        guard !cache.contains(layer: key.layer, expert: key.expert) else {
            lock.lock()
            skippedLocked += 1
            lock.unlock()
            return
        }
        do {
            let fetched = try store.fetch(layerIndex: key.layer, experts: [key.expert])
            guard let weights = fetched[key.expert] else { return }
            // Re-check once more right before inserting: the fetch itself
            // took real wall-clock time, during which the foreground path
            // may have resolved (and touched) this same expert.
            guard !cache.contains(layer: key.layer, expert: key.expert) else {
                lock.lock()
                skippedLocked += 1
                lock.unlock()
                return
            }
            cache.insertAtColdEnd(layer: key.layer, expert: key.expert, weights: weights)
            lock.lock()
            warmedCountLocked += 1
            warmedBytesLocked += weights.byteCount
            lock.unlock()
        } catch {
            // Best-effort, same rationale as PrefetchCoordinator: a real
            // checkpoint/layout mismatch also fails the foreground path
            // loudly; a background warm failure must never crash the
            // process on its own.
        }
    }

    private func finish(reason: String) {
        lock.lock()
        guard !finishedLocked else {
            lock.unlock()
            return
        }
        finishedLocked = true
        stopReasonLocked = reason
        elapsedMsLocked = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        let warmed = warmedCountLocked
        let warmedBytes = warmedBytesLocked
        let skipped = skippedLocked
        let elapsed = elapsedMsLocked ?? 0
        lock.unlock()
        print(
            String(
                format:
                    "[expert-warm] finished reason=%@ warmed=%d warmedBytes=%.2fGiB skipped=%d elapsedMs=%.0f",
                reason, warmed, Double(warmedBytes) / 1_073_741_824, skipped, elapsed))
    }

    /// Test-only: block until the warm task (if started) has finished.
    func waitUntilFinishedForTesting(timeout: TimeInterval = 30) {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            lock.lock()
            let done = finishedLocked
            lock.unlock()
            if done { return }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }
}
