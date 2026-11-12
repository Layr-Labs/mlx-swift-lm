import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// Replays a busy provider's arrivals and retirements through the paged
    /// page allocator (`PagedKVFreePages`) and counts the segmented decode's
    /// part dispatches per step with the real planner
    /// (`PagedSegmentDispatchPlan`: one dispatch per bucket per layer).
    ///
    /// The trace keeps the pool rules that decide where pages land: a row
    /// reserves `prompt + max_tokens` at admission and growth commits
    /// size-class segments (`PagedKVSegmentLayout.adding`); every layer of a
    /// row has its own table in the one shared group; prefill allocates in
    /// 512-token chunks layer by layer; a retiring row releases its tables
    /// tail-first; each release retires whole free segments, highest first,
    /// while the rest still covers the reservations
    /// (`PagedKVGroup.trimSegments`). Prefix-cache pages are not modelled.
    @Suite struct PagedKVAllocationTraceTests {

        /// Qwen 3.6 under GPQA at c = 4: 10 full-attention layers, 32 KiB
        /// pages (1,024-page segments), short prompts, long reasoning
        /// outputs, `max_tokens` 131,072.
        @Test func reasoningRowsKeepTheFreshDispatchCount() throws {
            let summary = try Trace(
                layers: 10, pageBytes: 32 << 10, concurrency: 4, rows: 16, sampleEvery: 1_000,
                prompt: { $0.integer(200 ... 600) },
                output: { $0.logNormal(median: 10_000, sigma: 0.6, in: 2_000 ... 30_000) },
                maxTokens: { _ in 131_072 }
            ).run()
            summary.expectBounded(layers: 10)
        }

        /// The 9B under production-like load at c = 3: 8 full-attention
        /// layers, 64 KiB pages (512-page segments), ~4K-token prompts,
        /// ~350-token outputs, mixed `max_tokens`.
        @Test func productionRowsKeepTheFreshDispatchCount() throws {
            let summary = try Trace(
                layers: 8, pageBytes: 64 << 10, concurrency: 3, rows: 400, sampleEvery: 500,
                prompt: { $0.logNormal(median: 4_000, sigma: 0.8, in: 200 ... 30_000) },
                output: { $0.logNormal(median: 350, sigma: 0.9, in: 64 ... 4_000) },
                maxTokens: {
                    [1_000, 1_000, 1_000, 1_000, 1_000, 4_096, 4_096, 16_384, 32_768, 64_000][
                        $0.integer(0 ... 9)]
                }
            ).run()
            summary.expectBounded(layers: 8)
        }
    }
}

/// Part dispatches per decode step over one trace.
private struct DispatchSummary {
    let name: String
    /// Largest count before the first row retired.
    let fresh: Int
    /// Counts sampled after the first row retired.
    let afterRetirement: [Int]

    var mean: Double { Double(afterRetirement.reduce(0, +)) / Double(afterRetirement.count) }
    var maximum: Int { afterRetirement.max() ?? 0 }

    /// One bucket per layer is the fresh pool's count. A busy run may cost
    /// a few more buckets, never one per partition.
    func expectBounded(layers: Int) {
        let report =
            "\(name): fresh \(fresh), after the first retirement mean "
            + String(format: "%.1f", mean) + ", max \(maximum) part dispatches per step"
        print(report)
        #expect(fresh == layers, "\(report)")
        #expect(!afterRetirement.isEmpty, "\(report)")
        #expect(mean <= 1.5 * Double(layers), "\(report)")
        #expect(maximum <= 3 * layers, "\(report)")
    }
}

private struct Trace {
    let layers: Int
    let pageBytes: Int
    let concurrency: Int
    let rows: Int
    let sampleEvery: Int
    let prompt: (inout SplitMix64) -> Int
    let output: (inout SplitMix64) -> Int
    let maxTokens: (inout SplitMix64) -> Int

    static let pageSize = 16
    static let prefillChunk = 512

    private struct Row {
        let prompt: Int
        let output: Int
        let reservedPages: Int
        var length = 0
        var tables: [[Int32]]
    }

    func run(seed: UInt64 = 1) throws -> DispatchSummary {
        var random = SplitMix64(seed: seed)
        var pool = try TracePool(pageBytes: pageBytes)
        var live: [Row] = []
        var started = 0
        var retired = 0
        var fresh = 0
        var afterRetirement: [Int] = []

        func pages(_ tokens: Int) -> Int { (tokens + Self.pageSize - 1) / Self.pageSize }

        func admit() throws {
            let maxTokens = maxTokens(&random)
            let prompt = prompt(&random)
            var row = Row(
                prompt: prompt, output: min(output(&random), maxTokens),
                reservedPages: pages(prompt + maxTokens),
                tables: Array(repeating: [], count: layers))
            started += 1
            try pool.reserve(row.reservedPages * layers)
            var done = 0
            while done < row.prompt {
                let chunk = min(Self.prefillChunk, row.prompt - done)
                for layer in 0 ..< layers {
                    while row.tables[layer].count < pages(done + chunk) {
                        row.tables[layer].append(pool.allocate(after: row.tables[layer].last))
                    }
                }
                done += chunk
            }
            row.length = row.prompt
            live.append(row)
        }

        for _ in 0 ..< concurrency { try admit() }
        var step = 0
        while !live.isEmpty {
            step += 1
            var index = 0
            while index < live.count {
                if live[index].length % Self.pageSize == 0 {
                    for layer in 0 ..< layers {
                        live[index].tables[layer].append(
                            pool.allocate(after: live[index].tables[layer].last))
                    }
                }
                live[index].length += 1
                if live[index].length >= live[index].prompt + live[index].output {
                    let row = live.remove(at: index)
                    for table in row.tables { pool.release(table.reversed()) }
                    pool.unreserve(row.reservedPages * layers)
                    retired += 1
                    if started < rows { try admit() }
                } else {
                    index += 1
                }
            }
            if step == 1 || step % sampleEvery == 0, !live.isEmpty {
                let count = dispatches(live, layout: pool.layout)
                if retired == 0 { fresh = max(fresh, count) } else { afterRetirement.append(count) }
            }
        }
        return DispatchSummary(
            name:
                "\(layers) layers, \(pageBytes >> 10) KiB pages, c = \(concurrency), \(rows) rows",
            fresh: fresh, afterRetirement: afterRetirement)
    }

    private func dispatches(_ rows: [Row], layout: PagedKVSegmentLayout) -> Int {
        (0 ..< layers).reduce(0) { total, layer in
            let planned = rows.map { row in
                let table = row.tables[layer]
                let last = row.length - 1
                return PagedSegmentDispatchPlan.Row(
                    pages: table,
                    info: PagedAttentionKernel.SeqInfoRow(
                        attendStart: 0, attendLength: row.length, tableLength: table.count,
                        writePage: table[last / Self.pageSize], writeSlot: last % Self.pageSize))
            }
            return total
                + PagedSegmentDispatchPlan(
                    rows: planned, layout: layout, pageSize: Self.pageSize,
                    partitionTokens: PagedAttentionKernel.partitionTokens, hasWrite: true
                ).buckets.count
        }
    }
}

/// The page bookkeeping of one segmented `PagedKVGroup`, without storage.
private struct TracePool {
    private(set) var layout: PagedKVSegmentLayout
    private var free: PagedKVFreePages
    private var committed: [Int: Range<Int>] = [:]
    private var heldBySegment: [Int: Int] = [:]
    private var reserved = 0

    init(pageBytes: Int) throws {
        layout = try PagedKVSegmentLayout(
            pageBytes: pageBytes, targetBytes: PagedKVSegmentLayout.defaultTargetBytes,
            maximumBufferBytes: 1 << 40)
        free = PagedKVFreePages(layout: layout)
    }

    private var committedUsable: Int { committed.values.reduce(0) { $0 + $1.count - 1 } }

    mutating func reserve(_ pages: Int) throws {
        reserved += pages
        let missing = reserved - committedUsable
        guard missing > 0 else { return }
        let grown = try layout.adding(usablePages: missing, excluding: Set(committed.keys))
        free.extend(to: grown.layout)
        layout = grown.layout
        for index in grown.segmentIDs {
            committed[index] = layout.range(index)
            free.addFresh(layout.range(index).dropFirst().lazy.map { Int32($0) })
        }
    }

    mutating func unreserve(_ pages: Int) {
        reserved -= pages
        var remaining = committedUsable
        for index in committed.keys.sorted(by: >) {
            let range = committed[index]!
            guard remaining - (range.count - 1) >= reserved, heldBySegment[index, default: 0] == 0
            else { continue }
            for page in range.dropFirst() { free.remove(Int32(page)) }
            committed.removeValue(forKey: index)
            remaining -= range.count - 1
        }
    }

    mutating func allocate(after hint: Int32?) -> Int32 {
        let page = free.next(after: hint)!
        free.remove(page)
        heldBySegment[layout.segmentIndex(page: page), default: 0] += 1
        return page
    }

    mutating func release(_ pages: some Sequence<Int32>) {
        let pages = Array(pages)
        for page in pages { heldBySegment[layout.segmentIndex(page: page)]! -= 1 }
        free.addReleased(uncached: pages, cached: [])
    }
}

/// Seeded generator, so the trace is the same on every run.
private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    mutating func unit() -> Double { (Double(next() >> 11) + 0.5) / Double(1 << 53) }

    mutating func integer(_ range: ClosedRange<Int>) -> Int {
        range.lowerBound + Int(next() % UInt64(range.count))
    }

    mutating func logNormal(median: Double, sigma: Double, in range: ClosedRange<Int>) -> Int {
        let normal = (-2 * log(unit())).squareRoot() * cos(2 * Double.pi * unit())
        let value = Int(median * exp(sigma * normal))
        return min(range.upperBound, max(range.lowerBound, value))
    }
}
