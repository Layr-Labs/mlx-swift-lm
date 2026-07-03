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
    private let byteBudget: Int

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
