import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Random operations on a real `PagedKVGroup` with the invariants checked
    /// after each one. Operations: hinted and unhinted allocation, prefix-cache
    /// bundles that dissolve on reuse and retire (`reclassifyAsUncached`),
    /// tail-first release into both tiers, prefix hits that resurrect or share
    /// pages (`retain`), deferred speculative frees, size-class growth through
    /// `PreparedGrowth` (installed or dropped) and whole-segment trim.
    /// Invariants: refcounts equal table references, `pagesInUse`, the free list
    /// is exactly the zero-ref usable pages of committed segments, unaliased free
    /// pages come before aliased ones, every reuse bumps the generation, and
    /// stale handles are invalid.
    ///
    /// Fixed-slab arrays stay lazy and segments wrap zero-filled host pages, so
    /// no kernel runs; a growth that would evaluate on the device throws.
    @Suite(.serialized)
    struct PagedKVGroupPropertyTests {
        struct Rng {
            var state: UInt64

            mutating func next() -> UInt64 {
                state &+= 0x9E37_79B9_7F4A_7C15
                var z = state
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                return z ^ (z >> 31)
            }

            mutating func int(_ n: Int) -> Int { n <= 0 ? 0 : Int(next() % UInt64(n)) }
            mutating func chance(_ p: Double) -> Bool { Double(next() >> 11) / Double(1 << 53) < p }
        }

        enum DeviceEvaluation: Error { case refused }

        final class Driver {
            let group: PagedKVGroup
            let segmented: Bool
            var rng: Rng
            var tables: [[Int32]] = []
            var deferred: [Int32] = []
            /// Prefix-index stand-in: a page is cached while a live bundle names it.
            var bundles: [Int: [PagedKVPageHandle]] = [:]
            var bundleOfHandle: [PagedKVPageHandle: Int] = [:]
            var nextBundle = 0
            var lastGeneration: [Int32: UInt64] = [:]
            var staleHandles: [PagedKVPageHandle] = []
            var stats: [String: Int] = [:]
            var issues = 0

            init(group: PagedKVGroup, segmented: Bool, seed: UInt64) {
                self.group = group
                self.segmented = segmented
                rng = Rng(state: seed)
            }

            func expect(_ ok: Bool, _ message: @autoclosure () -> String) {
                guard !ok else { return }
                issues += 1
                if issues <= 10 { Issue.record(Comment(rawValue: message())) }
            }

            func isCached(_ handle: PagedKVPageHandle) -> Bool { bundleOfHandle[handle] != nil }

            /// Like the index dropping an entry: reclassify its orphaned pages.
            func dissolve(_ id: Int) {
                guard let handles = bundles.removeValue(forKey: id) else { return }
                var orphans: [PagedKVPageHandle] = []
                for handle in handles where bundleOfHandle[handle] == id {
                    bundleOfHandle.removeValue(forKey: handle)
                    orphans.append(handle)
                }
                for handle in orphans { group.reclassifyAsUncached(handle) }
            }

            func willReuse(_ handle: PagedKVPageHandle) {
                guard let id = bundleOfHandle[handle] else { return }
                stats["dissolve.onReuse", default: 0] += 1
                dissolve(id)
            }

            var usablePages: [Int32] {
                if let layout = group.segmentLayout {
                    return group.segments.keys.sorted().flatMap {
                        layout.range($0).dropFirst().map(Int32.init)
                    }
                }
                return (1 ..< group.pageCount).map(Int32.init)
            }

            func allocate(into t: Int?) {
                guard !group.freeList.isEmpty else {
                    stats["alloc.full", default: 0] += 1
                    return
                }
                let hint = t.flatMap { rng.chance(0.9) ? tables[$0].last : nil }
                let freeBefore = Set(group.freeList)
                let cachedBefore = freeBefore.filter { isCached(group.currentHandle($0)) }
                let page = group.allocatePage(after: hint) { self.willReuse($0) }
                expect(freeBefore.contains(page), "allocated \(page) was not free")
                if cachedBefore.contains(page) {
                    stats["alloc.cachedTier", default: 0] += 1
                    expect(
                        cachedBefore.count == freeBefore.count,
                        "took cached \(page) while an uncached page was free")
                }
                expect(group.refCounts[Int(page)] == 1, "allocated \(page) has refcount != 1")
                let generation = group.generations[Int(page)]
                if let last = lastGeneration[page] {
                    expect(generation > last, "page \(page) reused without a generation bump")
                    staleHandles.append(
                        PagedKVPageHandle(group: group.key, page: page, generation: last))
                }
                lastGeneration[page] = generation
                stats["alloc", default: 0] += 1
                if let t { tables[t].append(page) } else { tables.append([page]) }
            }

            func release(_ t: Int, keep: Int) {
                let doomed = Array(tables[t][keep...].reversed())
                group.release(doomed) { self.isCached($0) }
                stats["release.pages", default: 0] += doomed.count
                if keep == 0 { tables.remove(at: t) } else { tables[t].removeSubrange(keep...) }
            }

            /// One bundle per block naming the pages of two tables, like two layers
            /// of a published prompt.
            func publish() {
                guard tables.count >= 2 else { return }
                let a = rng.int(tables.count)
                let b = rng.int(tables.count)
                guard a != b else { return }
                for i in 0 ..< min(tables[a].count, tables[b].count) {
                    let handles = [
                        group.currentHandle(tables[a][i]), group.currentHandle(tables[b][i]),
                    ]
                    guard handles.allSatisfy({ bundleOfHandle[$0] == nil }) else { continue }
                    nextBundle += 1
                    bundles[nextBundle] = handles
                    for handle in handles { bundleOfHandle[handle] = nextBundle }
                    stats["publish", default: 0] += 1
                }
            }

            /// A prefix hit: retain every page of a valid bundle into new tables.
            func hit() {
                let ids = bundles.keys.sorted()
                guard !ids.isEmpty else { return }
                let id = ids[rng.int(ids.count)]
                let handles = bundles[id]!
                guard handles.allSatisfy({ group.isValid($0) }) else {
                    dissolve(id)
                    return
                }
                for handle in handles {
                    let wasFree = group.refCounts[Int(handle.page)] == 0
                    group.retain(handle)
                    stats[wasFree ? "hit.resurrect" : "hit.share", default: 0] += 1
                    tables.append([handle.page])
                }
            }

            func speculativeRollback() {
                guard !tables.isEmpty else { return }
                let t = rng.int(tables.count)
                guard tables[t].count > 1 else { return }
                let page = tables[t].removeLast()
                group.deferFree(page)
                deferred.append(page)
                stats["deferred", default: 0] += 1
            }

            func drain() {
                group.drainDeferredFrees { self.isCached($0) }
                deferred.removeAll()
            }

            func grow() throws {
                guard segmented, group.freeList.count < 4, rng.chance(0.5) else { return }
                let plan = try group.planGrowth(
                    usablePages: group.committedUsablePages + 1 + rng.int(160))
                guard !plan.segmentIDs.isEmpty else { return }
                let before = group.freeList
                let prepared = try group.prepareGrowth(
                    plan, evaluate: { _ in throw DeviceEvaluation.refused }, hostZeroed: true)
                expect(group.freeList == before, "prepareGrowth changed the live free list")
                if rng.chance(0.15) {
                    stats["grow.dropped", default: 0] += 1
                    return
                }
                group.installGrowth(prepared)
                stats["grow.segments", default: 0] += plan.segmentIDs.count
            }

            func trim() {
                guard segmented else { return }
                let held = tables.reduce(0) { $0 + $1.count } + deferred.count
                group.pagesReserved = held + rng.int(64)
                let before = group.segments.count
                group.trimSegments { handle in
                    if let id = self.bundleOfHandle[handle] { self.dissolve(id) }
                }
                stats["trim.segments", default: 0] += before - group.segments.count
            }

            func verify(_ context: String) {
                var references: [Int32: Int] = [:]
                for page in tables.joined() { references[page, default: 0] += 1 }
                for page in deferred { references[page, default: 0] += 1 }
                let usable = usablePages
                let usableSet = Set(usable)
                let freeList = group.freeList
                let freeSet = Set(freeList)
                expect(freeSet.count == freeList.count, "\(context): duplicate in the free list")
                expect(
                    freeSet.isSubset(of: usableSet),
                    "\(context): the free list names a poison, retired or unknown page")
                var inUse = 0
                for page in usable {
                    let count = group.refCounts[Int(page)]
                    let expected = references[page, default: 0]
                    expect(
                        count == expected,
                        "\(context): page \(page) refcount \(count) != \(expected)")
                    expect(
                        (count == 0) == freeSet.contains(page),
                        "\(context): page \(page) refcount \(count), free \(freeSet.contains(page))"
                    )
                    if count > 0 { inUse += 1 }
                }
                for page in references.keys {
                    expect(usableSet.contains(page), "\(context): held page \(page) is not usable")
                }
                expect(
                    group.pagesInUse == inUse,
                    "\(context): pagesInUse \(group.pagesInUse) != \(inUse)")
                if let layout = group.segmentLayout {
                    for index in group.segments.keys {
                        expect(
                            group.refCounts[layout.range(index).lowerBound] == 1,
                            "\(context): poison refcount")
                    }
                } else {
                    expect(group.refCounts[0] == 1, "\(context): poison refcount")
                }
                let aliased = freeList.filter { isCached(group.currentHandle($0)) }
                let unaliased = freeList.filter { !isCached(group.currentHandle($0)) }
                expect(
                    freeList == unaliased.sorted() + aliased, "\(context): tier order \(freeList)")
                for handle in staleHandles.suffix(64) {
                    expect(!group.isValid(handle), "\(context): stale handle \(handle) is valid")
                }
            }

            func step() throws {
                switch rng.int(100) {
                case 0 ..< 8: try grow()
                case 8 ..< 50:
                    if tables.isEmpty || rng.chance(0.15) {
                        allocate(into: nil)
                    } else {
                        allocate(into: rng.int(tables.count))
                    }
                case 50 ..< 62:
                    if !tables.isEmpty { release(rng.int(tables.count), keep: 0) }
                case 62 ..< 66:
                    guard !tables.isEmpty else { return }
                    let t = rng.int(tables.count)
                    if tables[t].count > 1 { release(t, keep: 1 + rng.int(tables[t].count - 1)) }
                case 66 ..< 74: publish()
                case 74 ..< 80: hit()
                case 80 ..< 84:
                    if let id = bundles.keys.min() { dissolve(id) }
                case 84 ..< 90: speculativeRollback()
                case 90 ..< 94: drain()
                default: trim()
                }
            }
        }

        static let key = PagedKVGroupKey(kvHeads: 1, headDim: 256, dtype: .bfloat16)

        @Test func segmentedGroupKeepsEveryInvariant() throws {
            var total: [String: Int] = [:]
            var issues = 0
            for seed in UInt64(1) ... 12 {
                for maxUsable in [16, 64] {
                    let pageBytes = 16 << 10
                    let layout = try PagedKVSegmentLayout(
                        pageBytes: pageBytes, targetBytes: pageBytes * (maxUsable + 1),
                        maximumBufferBytes: 1 << 30)
                    let group = PagedKVGroup(
                        key: Self.key, pageCount: 0, pageSize: 16, dtype: .bfloat16,
                        segmentLayout: layout)
                    let driver = Driver(
                        group: group, segmented: true, seed: seed * 31 + UInt64(maxUsable))
                    for op in 0 ..< 700 {
                        try driver.step()
                        driver.verify("seed \(seed) max \(maxUsable) op \(op)")
                        if driver.issues > 0 { break }
                    }
                    driver.drain()
                    while !driver.tables.isEmpty { driver.release(0, keep: 0) }
                    driver.verify("seed \(seed) max \(maxUsable) end")
                    issues += driver.issues
                    for (name, count) in driver.stats { total[name, default: 0] += count }
                }
            }
            #expect(issues == 0)
            #expect(total["grow.segments", default: 0] > 100, "\(total)")
            #expect(total["trim.segments", default: 0] > 100, "\(total)")
            #expect(total["hit.resurrect", default: 0] > 20, "\(total)")
            #expect(total["alloc.cachedTier", default: 0] > 5, "\(total)")
        }

        @Test func fixedSlabGroupKeepsEveryInvariant() throws {
            var total: [String: Int] = [:]
            var issues = 0
            for seed in UInt64(1) ... 12 {
                for pages in [2, 18, 70] {
                    let group = PagedKVGroup(
                        key: Self.key, pageCount: pages, pageSize: 16, dtype: .bfloat16)
                    let driver = Driver(
                        group: group, segmented: false, seed: seed * 17 + UInt64(pages))
                    for op in 0 ..< 500 {
                        try driver.step()
                        driver.verify("seed \(seed) fixed \(pages) op \(op)")
                        if driver.issues > 0 { break }
                    }
                    issues += driver.issues
                    for (name, count) in driver.stats { total[name, default: 0] += count }
                }
            }
            #expect(issues == 0)
            #expect(total["hit.resurrect", default: 0] > 20, "\(total)")
            #expect(total["alloc.cachedTier", default: 0] > 5, "\(total)")
        }
    }
}
