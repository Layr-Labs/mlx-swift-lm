// Process-wide, byte-budgeted LRU cache for streamed MoE expert weights.
//
// Keyed by (layerIndex, expertIndex) rather than per-layer instances because
// a single process only ever has one model loaded (CLI / provider), and a
// shared budget lets "hot" experts from any layer compete for cache space
// rather than partitioning the budget upfront per-layer (some layers'
// routing may be far more concentrated than others).

import Foundation
import MLX

/// LRU eviction is approximated with a plain recency list rather than a
/// doubly-linked list: cache occupancy is bounded by `byteBudget / ~12 MB`
/// (a few thousand entries even at 70 GiB), so an O(n) `touch`/eviction scan
/// costs low-single-digit microseconds — not worth the extra code to make
/// O(1) unless profiling says otherwise.
public final class ExpertCache: @unchecked Sendable {
    public struct Key: Hashable, Sendable {
        public let layer: Int
        public let expert: Int
        public init(layer: Int, expert: Int) {
            self.layer = layer
            self.expert = expert
        }
    }

    private let lock = NSLock()
    private var storage: [Key: ExpertWeights] = [:]
    /// Recency order, least-recently-used first.
    private var order: [Key] = []
    private var currentBytes: Int = 0
    /// Locked (not `let`) so `setByteBudget(_:)` can resize a live cache —
    /// e.g. a provider reloading DeepSeek-V4 with a different
    /// `expert_cache_gb` no longer has to live with whatever budget the
    /// FIRST load picked for the lifetime of the process.
    private var byteBudget: Int

    private var hitCountLocked: Int = 0
    private var missCountLocked: Int = 0

    public init(byteBudget: Int) {
        self.byteBudget = max(0, byteBudget)
    }

    /// Reads `DSV4_EXPERT_CACHE_GB` (float, default 8.0).
    public static func budgetFromEnv() -> Int {
        let raw = ProcessInfo.processInfo.environment["DSV4_EXPERT_CACHE_GB"]
        let gb = raw.flatMap(Double.init) ?? 8.0
        return max(0, Int(gb * 1024 * 1024 * 1024))
    }

    public func get(layer: Int, expert: Int) -> ExpertWeights? {
        let key = Key(layer: layer, expert: expert)
        lock.lock()
        defer { lock.unlock() }
        guard let value = storage[key] else {
            missCountLocked += 1
            return nil
        }
        hitCountLocked += 1
        touchLocked(key)
        return value
    }

    /// Membership check that does NOT touch hit/miss counters or recency
    /// order. Used by the prefetch coordinator to decide whether an expert
    /// is worth fetching without polluting the reported cache-hit-rate
    /// stats with speculative lookups (a prefetch "miss" isn't a real
    /// decode-path miss) or promoting recency for an expert nothing has
    /// actually used yet.
    public func contains(layer: Int, expert: Int) -> Bool {
        let key = Key(layer: layer, expert: expert)
        lock.lock()
        defer { lock.unlock() }
        return storage[key] != nil
    }

    /// Insert a freshly-fetched expert. A no-op if another concurrent fetch
    /// (e.g. from a sibling `DispatchQueue.concurrentPerform` iteration in
    /// `ExpertShardStore.fetch`) already inserted the same key — the
    /// redundant read is wasted disk I/O but never corrupts cache state.
    public func insert(layer: Int, expert: Int, weights: ExpertWeights) {
        let key = Key(layer: layer, expert: expert)
        lock.lock()
        defer { lock.unlock() }
        guard storage[key] == nil else { return }
        storage[key] = weights
        order.append(key)
        currentBytes += weights.byteCount
        evictLocked()
    }

    /// Insert a speculatively-warmed expert (see `ExpertCacheWarmer`) at
    /// the COLD end of the LRU order instead of the normal MRU end.
    ///
    /// WHY cold-end, not the normal `insert`'s MRU-end: a warm entry is a
    /// PREDICTION (this checkpoint's historical usage profile says this
    /// expert is likely to matter), not yet confirmed by an actual
    /// request. Foreground traffic must never lose real, already-useful
    /// cache residents to make room for a guess. Inserting at the cold end
    /// means every subsequent `evictLocked()` call evicts pending warm
    /// entries BEFORE it ever touches an organically-inserted (real)
    /// entry, no matter how large the warm batch is relative to headroom
    /// -- so warming can only ever compete with OTHER warm entries for
    /// space, never with real traffic. The moment a warmed expert is
    /// actually used by a real request, `get()`'s `touchLocked` promotes
    /// it to the MRU end exactly like any organic hit, and from then on
    /// it is indistinguishable from (and exactly as protected as) an
    /// expert the foreground path fetched itself.
    ///
    /// Repeatedly warming (multiple calls, each prepending) naturally
    /// preserves relative priority among the warmed set too: since
    /// `ExpertUsageProfile.warmOrder` yields highest-frequency-first, and
    /// each new prepend pushes earlier entries one step further from the
    /// eviction end, the FIRST (highest-priority) warmed expert ends up
    /// LAST to be evicted among the warm cohort -- lowest-priority warm
    /// guesses are sacrificed first if the cache is under budget pressure
    /// before real traffic confirms any of them.
    ///
    /// A no-op if the key is already resident (organically or from an
    /// earlier warm call) -- same idempotence contract as `insert`.
    public func insertAtColdEnd(layer: Int, expert: Int, weights: ExpertWeights) {
        let key = Key(layer: layer, expert: expert)
        lock.lock()
        defer { lock.unlock() }
        guard storage[key] == nil else { return }
        storage[key] = weights
        order.insert(key, at: 0)
        currentBytes += weights.byteCount
        evictLocked()
    }

    /// Resolve a batch of experts for one layer: cache hits return
    /// immediately, misses are fetched from `store` in parallel and
    /// inserted before returning. This is the single entry point
    /// `StreamingQuantizedSwitchGLU` uses per chunk.
    public func fetch(layer: Int, experts: [Int], from store: ExpertShardStore) throws -> [Int:
        ExpertWeights]
    {
        var resolved: [Int: ExpertWeights] = [:]
        var misses: [Int] = []
        for e in experts {
            if let w = get(layer: layer, expert: e) {
                resolved[e] = w
            } else {
                misses.append(e)
            }
        }
        guard !misses.isEmpty else { return resolved }

        let fetched = try store.fetch(layerIndex: layer, experts: misses)
        for (e, w) in fetched {
            insert(layer: layer, expert: e, weights: w)
            resolved[e] = w
        }
        return resolved
    }

    /// Drop every cached entry and reset occupancy to zero — used when a
    /// model unloads so its (potentially tens-of-GB) resident expert
    /// weights are actually released rather than lingering until the next
    /// model's cache pressure evicts them one LRU entry at a time.
    ///
    /// Hit/miss counters are reset to zero as well: they are reported as
    /// per-model-lifetime cache statistics (e.g. provider diagnostics), and
    /// carrying counts across an unrelated model's load would make the
    /// reported hit rate meaningless. A caller that wants cumulative
    /// process-lifetime stats should snapshot `stats` before purging.
    ///
    /// Freeing the underlying `MLXArray`s is a side effect of dropping the
    /// last Swift reference here, not something this method does directly
    /// — callers that want the freed bytes to actually return to the OS in
    /// the same sweep should call `MLX.Memory.clearCache()` (or equivalent)
    /// shortly after this returns.
    public func purgeAll() {
        lock.lock()
        defer { lock.unlock() }
        storage.removeAll()
        order.removeAll()
        currentBytes = 0
        hitCountLocked = 0
        missCountLocked = 0
    }

    /// Resize the byte budget of a live cache, evicting LRU entries
    /// immediately if the new budget is smaller than current occupancy.
    /// Growing the budget never evicts (it just raises the ceiling future
    /// inserts are checked against).
    public func setByteBudget(_ newBudget: Int) {
        lock.lock()
        defer { lock.unlock() }
        byteBudget = max(0, newBudget)
        evictLocked()
    }

    /// Current byte budget (primarily for tests/diagnostics).
    public var currentByteBudget: Int {
        lock.lock()
        defer { lock.unlock() }
        return byteBudget
    }

    private func touchLocked(_ key: Key) {
        guard let idx = order.firstIndex(of: key) else { return }
        order.remove(at: idx)
        order.append(key)
    }

    private func evictLocked() {
        while currentBytes > byteBudget, !order.isEmpty {
            let oldest = order.removeFirst()
            if let removed = storage.removeValue(forKey: oldest) {
                currentBytes -= removed.byteCount
            }
        }
    }

    public struct Stats: Sendable {
        public let hits: Int
        public let misses: Int
        public let residentBytes: Int
        public let residentCount: Int
    }

    public var stats: Stats {
        lock.lock()
        defer { lock.unlock() }
        return Stats(
            hits: hitCountLocked, misses: missCountLocked,
            residentBytes: currentBytes, residentCount: storage.count)
    }
}
