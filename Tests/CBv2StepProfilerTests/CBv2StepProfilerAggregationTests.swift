// CBv2StepProfilerAggregationTests.swift
//
// Pure-CPU unit tests for CBv2StepProfiler's aggregation math: the
// count/total/mean/p50/p95/max reduction used by the decode-step profile
// table, its order-independence, its empty/degenerate handling, and the
// `dropFirst` warm-up peel that summaryTable applies. No model weights, no
// Metal — this whole suite runs on any host.
//
// Serialized: a few cases exercise the process-wide CBv2StepProfiler sample
// buffer (record → summaryTable), so they must not run concurrently with each
// other or with the pure-aggregate cases that share the type.

import Foundation
import MLXLMCommon
import Testing

@Suite(.serialized)
struct CBv2StepProfilerAggregationTests {

    /// Milliseconds are compared with a tolerance: the fixtures are whole
    /// milliseconds expressed as seconds, so the ×1e3 round-trip is only
    /// float-exact to within a few ULP. Returns Bool so `#expect` reports the
    /// failing call site.
    private func close(_ a: Double, _ b: Double, _ tol: Double = 1e-6) -> Bool {
        abs(a - b) <= tol
    }

    // 1 ms, 2 ms, ... 10 ms expressed in seconds.
    private let oneToTenMS: [Double] = (1...10).map { Double($0) / 1000.0 }

    @Test func aggregatesMeanMedianAndTailOfAKnownSample() {
        let stat = CBv2StepProfiler.aggregate(oneToTenMS)
        #expect(stat.count == 10)
        #expect(close(stat.totalMS, 55.0))
        #expect(close(stat.meanMS, 5.5))
        // p50: rank 0.5·9 = 4.5 → halfway between the 5th (5 ms) and 6th (6 ms).
        #expect(close(stat.p50MS, 5.5))
        // p95: rank 0.95·9 = 8.55 → 9 ms + 0.55·(10−9) ms.
        #expect(close(stat.p95MS, 9.55))
        #expect(close(stat.maxMS, 10.0))
    }

    @Test func aggregationIsOrderIndependent() {
        let shuffled = CBv2StepProfiler.aggregate([0.003, 0.001, 0.002])
        #expect(shuffled.count == 3)
        #expect(close(shuffled.totalMS, 6.0))
        #expect(close(shuffled.meanMS, 2.0))
        // sorted [1,2,3] ms: p50 rank 1.0 → 2 ms exactly.
        #expect(close(shuffled.p50MS, 2.0))
        // p95 rank 0.95·2 = 1.9 → 2 ms + 0.9·(3−2) ms.
        #expect(close(shuffled.p95MS, 2.9))
        #expect(close(shuffled.maxMS, 3.0))
    }

    @Test func emptySampleIsAllZeroCountZero() {
        let stat = CBv2StepProfiler.aggregate([])
        #expect(stat.count == 0)
        #expect(close(stat.totalMS, 0.0))
        #expect(close(stat.meanMS, 0.0))
        #expect(close(stat.p50MS, 0.0))
        #expect(close(stat.p95MS, 0.0))
        #expect(close(stat.maxMS, 0.0))
    }

    @Test func singleSampleCollapsesEveryQuantile() {
        let stat = CBv2StepProfiler.aggregate([0.007])
        #expect(stat.count == 1)
        #expect(close(stat.totalMS, 7.0))
        #expect(close(stat.meanMS, 7.0))
        #expect(close(stat.p50MS, 7.0))
        #expect(close(stat.p95MS, 7.0))
        #expect(close(stat.maxMS, 7.0))
    }

    @Test func dropFirstPeelsLeadingWarmupSamples() {
        // The warm-up peel is `Array(samples.dropFirst(n))` before aggregate,
        // so aggregating the peeled slice must give the steady-state stats.
        let steady = CBv2StepProfiler.aggregate(Array(oneToTenMS.dropFirst(8)))
        #expect(steady.count == 2)  // [9 ms, 10 ms]
        #expect(close(steady.totalMS, 19.0))
        #expect(close(steady.meanMS, 9.5))
        #expect(close(steady.p50MS, 9.5))
        // p95 rank 0.95·1 = 0.95 → 9 ms + 0.95·(10−9) ms.
        #expect(close(steady.p95MS, 9.95))
        #expect(close(steady.maxMS, 10.0))
    }

    @Test func summaryTableSortsByTotalDescendingAndCountsSamples() {
        CBv2StepProfiler.reset()
        CBv2StepProfiler.enabled = true
        defer {
            CBv2StepProfiler.enabled = false
            CBv2StepProfiler.reset()
        }

        // "slow" totals 10 ms in one sample; "fast" totals 6 ms across three.
        CBv2StepProfiler.record("fast", seconds: 0.001)
        CBv2StepProfiler.record("fast", seconds: 0.002)
        CBv2StepProfiler.record("fast", seconds: 0.003)
        CBv2StepProfiler.record("slow", seconds: 0.010)

        let table = CBv2StepProfiler.summaryTable()
        let lines = table.split(separator: "\n").map(String.init)
        // Header (2 rows) then one row per phase, slow before fast (10 > 6 ms).
        let dataRows = lines.filter { $0.hasPrefix("| ") && !$0.contains("---") && !$0.contains("phase ") }
        #expect(dataRows.count == 2)
        #expect(dataRows[0].hasPrefix("| slow |"))
        #expect(dataRows[1].hasPrefix("| fast |"))
        #expect(dataRows[1].contains("| fast | 3 |"))
        #expect(dataRows[0].contains("| slow | 1 |"))
    }

    @Test func summaryTableDropFirstOmitsExhaustedPhases() {
        CBv2StepProfiler.reset()
        CBv2StepProfiler.enabled = true
        defer {
            CBv2StepProfiler.enabled = false
            CBv2StepProfiler.reset()
        }

        CBv2StepProfiler.record("keep", seconds: 0.001)
        CBv2StepProfiler.record("keep", seconds: 0.002)
        CBv2StepProfiler.record("keep", seconds: 0.003)
        CBv2StepProfiler.record("gone", seconds: 0.010)  // single warm-up sample

        // Dropping the first 2 samples leaves "keep" with one sample and
        // exhausts "gone" (0 left → omitted, never a divide-by-zero row).
        let table = CBv2StepProfiler.summaryTable(dropFirst: 2)
        let dataRows = table.split(separator: "\n").map(String.init)
            .filter { $0.hasPrefix("| ") && !$0.contains("---") && !$0.contains("phase ") }
        #expect(dataRows.count == 1)
        #expect(dataRows[0].contains("| keep | 1 |"))
        #expect(!table.contains("| gone |"))

        // Dropping past every sample yields just the header (no data rows).
        let empty = CBv2StepProfiler.summaryTable(dropFirst: 99)
        let emptyRows = empty.split(separator: "\n").map(String.init)
            .filter { $0.hasPrefix("| ") && !$0.contains("---") && !$0.contains("phase ") }
        #expect(emptyRows.isEmpty)
    }
}
