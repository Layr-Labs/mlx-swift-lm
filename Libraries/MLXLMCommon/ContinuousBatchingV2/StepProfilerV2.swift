// StepProfilerV2.swift
//
// Opt-in phase timers and independently armed event counters for the decode
// hot loops (v2 EngineLoopV2 and the legacy GenerationBatch/Scheduler path).
// BenchCBv2's profile mode uses phase timing to decompose per-step wall time;
// performance cells arm only graph-submission events for provenance.
//
// DISABLED by default: each instrumentation point reads one plain static Bool,
// so the production step path pays one predictable branch and nothing else.
// Timing can be enabled programmatically or via CBV2_STEP_PROFILE=1. Event
// counters are armed explicitly at an idle benchmark boundary.

import Foundation

public enum CBv2StepProfiler {

    /// Master switch. Read on hot paths; set it BEFORE the run under
    /// measurement and do not toggle mid-run (plain non-atomic Bool).
    nonisolated(unsafe) public static var enabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_STEP_PROFILE"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

    /// Independent event-counter switch. Benchmark boundaries guarantee that
    /// no engine work is in flight while this plain Bool is changed.
    nonisolated(unsafe) public private(set) static var eventsEnabled = false

    private static let lock = NSLock()
    nonisolated(unsafe) private static var samples: [String: [Double]] = [:]
    nonisolated(unsafe) private static var eventCounts: [String: Int] = [:]

    /// Record one duration (seconds) for a phase. No-op when disabled.
    @inline(__always)
    public static func record(_ phase: StaticString, seconds: Double) {
        guard enabled else { return }
        let key = "\(phase)"
        lock.lock()
        samples[key, default: []].append(seconds)
        lock.unlock()
    }

    /// Time a closure and record it under `phase`. No-op overhead when
    /// disabled beyond the closure call itself.
    @inline(__always)
    public static func time<T>(_ phase: StaticString, _ body: () -> T) -> T {
        guard enabled else { return body() }
        let start = CFAbsoluteTimeGetCurrent()
        let result = body()
        record(phase, seconds: CFAbsoluteTimeGetCurrent() - start)
        return result
    }
    /// Count an observable graph-submission event without enabling phase
    /// clocks. The disabled path is one predictable plain-Bool branch and
    /// performs no string conversion, locking, allocation, or atomic access.
    @inline(__always)
    public static func recordEvent(_ event: StaticString) {
        guard eventsEnabled else { return }
        let key = "\(event)"
        lock.lock()
        defer { lock.unlock() }
        // Arm/disarm happens only while the engine is idle. This under-lock
        // recheck additionally rejects a recorder that passed the fast branch
        // immediately before a boundary acquired the lock.
        guard eventsEnabled else { return }
        eventCounts[key, default: 0] += 1
    }

    /// Clear and arm only event counting at an idle measurement boundary.
    public static func armEvents() {
        lock.lock()
        eventCounts.removeAll(keepingCapacity: true)
        eventsEnabled = true
        lock.unlock()
    }

    /// Disarm and snapshot one measured event scope.
    public static func snapshotAndDisarmEvents() -> [String: Int] {
        lock.lock()
        eventsEnabled = false
        defer { lock.unlock() }
        return eventCounts
    }

    /// Clear all profiler state. Call only at an idle boundary.
    public static func reset() {
        lock.lock()
        eventsEnabled = false
        samples.removeAll()
        eventCounts.removeAll()
        lock.unlock()
    }

    /// Snapshot of all recorded phases (name → sorted durations, seconds).
    public static func snapshot() -> [String: [Double]] {
        lock.lock()
        defer { lock.unlock() }
        return samples.mapValues { $0.sorted() }
    }
    /// Snapshot of explicitly counted events (name → cumulative count).
    public static func eventCountsSnapshot() -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        return eventCounts
    }

    /// Snapshot of all recorded phases in INSERTION order (name → durations,
    /// seconds), i.e. not sorted. `dropFirst` in `summaryTable` peels warm-up
    /// steps off the front, so it needs the samples in the order recorded.
    private static func rawSnapshot() -> [String: [Double]] {
        lock.lock()
        defer { lock.unlock() }
        return samples
    }

    /// Aggregated statistics for one phase's durations (milliseconds).
    public struct PhaseStat: Sendable, Equatable {
        public let count: Int
        public let totalMS: Double
        public let meanMS: Double
        public let p50MS: Double
        public let p95MS: Double
        public let maxMS: Double
    }

    /// Linear-interpolated percentile of a sorted sample (`q` in 0...1).
    /// Rank = q·(n−1), interpolated between the bracketing order statistics.
    static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let rank = q * Double(sorted.count - 1)
        let lo = Int(rank.rounded(.down)), hi = Int(rank.rounded(.up))
        if lo == hi { return sorted[lo] }
        let w = rank - Double(lo)
        return sorted[lo] * (1 - w) + sorted[hi] * w
    }

    /// Pure aggregation of one phase's durations (seconds in, milliseconds
    /// out): count, total, mean, p50, p95, max. Sorts a local copy for the
    /// order statistics; the input order is untouched. Empty input yields an
    /// all-zero, count-0 stat. This is the unit the aggregation tests cover.
    public static func aggregate(_ samplesSeconds: [Double]) -> PhaseStat {
        let sorted = samplesSeconds.sorted()
        let n = sorted.count
        let total = sorted.reduce(0, +)
        let mean = n > 0 ? total / Double(n) : 0
        return PhaseStat(
            count: n,
            totalMS: total * 1e3,
            meanMS: mean * 1e3,
            p50MS: percentile(sorted, 0.5) * 1e3,
            p95MS: percentile(sorted, 0.95) * 1e3,
            maxMS: (sorted.last ?? 0) * 1e3)
    }

    /// Markdown decomposition table: per phase count, total, mean, p50, p95,
    /// max (milliseconds), sorted by total descending. `dropFirst` peels that
    /// many leading (warm-up) samples off each phase before aggregating — pass
    /// the warm-up step count to report steady-state decode only. A phase with
    /// no samples left after the drop is omitted.
    public static func summaryTable(dropFirst n: Int = 0) -> String {
        let raw = rawSnapshot()
        let drop = max(0, n)
        let rows = raw.map {
            (name: $0.key, stat: aggregate(Array($0.value.dropFirst(drop))))
        }
        .filter { $0.stat.count > 0 }
        .sorted { $0.stat.totalMS > $1.stat.totalMS }
        var out = "| phase | n | total ms | mean ms | p50 ms | p95 ms | max ms |\n"
        out += "|---|---|---|---|---|---|---|\n"
        for row in rows {
            let s = row.stat
            out += String(
                format: "| %@ | %d | %.1f | %.3f | %.3f | %.3f | %.3f |\n",
                row.name, s.count, s.totalMS, s.meanMS, s.p50MS, s.p95MS, s.maxMS)
        }
        return out
    }
}
