import Foundation

/// The free pages of one paged-KV group and the choice of the next page.
///
/// Free pages come in two tiers. A zero-ref page that a prefix-cache alias
/// still names is "cached": it waits in eviction order (`PagedBlockFreeQueue`)
/// and is taken only when no uncached page is free. Every other free page is
/// "uncached" and is placed by address.
///
/// Placement keeps each row layer's pages in few segments, because the
/// segmented decode binds at most `PagedSegmentDispatchPlan.maximumBindings`
/// segments per dispatch: once a layer's partitions touch more segments than
/// one dispatch binds, the decode issues one dispatch per bucket instead of
/// one per layer. Each segment's usable pages split into aligned runs of
/// `runPages`, one decode partition at the usual page size. A row layer
/// - continues in the run of the page it took last;
/// - then takes a fully free run in that page's segment;
/// - then a fully free run in the segment with the most pages in fully free
///   runs (lowest address on a tie), so concurrent row layers spread over
///   segments instead of interleaving in the same ones;
/// - and only when no run is fully free, the lowest free page.
///
/// `PagedKVGroup` owns refcounts and generations; this type only orders the
/// free pages. Page choice never changes what a page holds or who may read
/// it, only where a row's next tokens are stored.
struct PagedKVFreePages {
    static let runPages = PagedSegmentDispatchPlan.maximumBindings - 1

    /// Address ranges in address order; the first page of each is poison.
    private var segments: [Range<Int>] = []
    /// Index of each segment's first run in `runFree`.
    private var firstRun: [Int] = [0]
    /// Uncached free pages in each run.
    private var runFree: [Int] = []
    /// Pages in each segment's fully free runs.
    private var fullRunPages: [Int] = []
    /// One bit per page id: an uncached free page.
    private var uncached: [UInt64] = []
    /// One bit per word of `uncached`: the word is not zero.
    private var occupiedWords: [UInt64] = []
    private var uncachedCount = 0
    private var cached: PagedBlockFreeQueue

    /// Fixed slab: one range over every page; all but the poison page start free.
    init(fixedPageCount: Int) {
        cached = PagedBlockFreeQueue(pageCount: fixedPageCount)
        addRanges([0 ..< fixedPageCount])
        for page in 1 ..< fixedPageCount { insertUncached(Int32(page)) }
    }

    /// Segmented pool: pages become free when their segment is committed.
    init(layout: PagedKVSegmentLayout) {
        cached = PagedBlockFreeQueue(pageCount: layout.pageCount)
        addRanges(layout.ranges)
    }

    var count: Int { uncachedCount + cached.count }

    /// Snapshot: uncached pages by address, then cached pages in eviction
    /// order. Tests and diagnostics only.
    var elements: [Int32] {
        var result: [Int32] = []
        result.reserveCapacity(count)
        for (word, bits) in uncached.enumerated() {
            var remaining = bits
            while remaining != 0 {
                result.append(Int32(word * 64 + remaining.trailingZeroBitCount))
                remaining &= remaining - 1
            }
        }
        return result + cached.elements
    }

    func contains(_ page: Int32) -> Bool { isUncached(page) || cached.contains(page) }

    /// Adopt the address ranges `layout` added. Their pages stay absent until
    /// `addFresh` installs a committed segment.
    mutating func extend(to layout: PagedKVSegmentLayout) {
        precondition(Array(layout.ranges.prefix(segments.count)) == segments)
        cached.extend(to: layout.pageCount)
        addRanges(layout.ranges.dropFirst(segments.count))
    }

    /// Pages of a newly committed segment.
    mutating func addFresh<S: Sequence>(_ pages: S) where S.Element == Int32 {
        for page in pages { insertUncached(page) }
    }

    /// Pages whose last owner released them. Cached pages enter eviction
    /// order in the order given (a row releases tail-first, so a divergent
    /// suffix is evicted before its ancestors).
    mutating func addReleased(uncached pages: [Int32], cached aliased: [Int32]) {
        for page in pages { insertUncached(page) }
        cached.append(contentsOf: aliased)
    }

    /// A queued page lost its last cache alias: it is now uncached.
    mutating func markUncached(_ page: Int32) {
        guard cached.contains(page) else { return }
        cached.remove(page)
        insertUncached(page)
    }

    mutating func remove(_ page: Int32) {
        if isUncached(page) {
            removeUncached(page)
        } else {
            cached.remove(page)
        }
    }

    /// The page the next allocation takes, without removing it. `hint` is
    /// the page the same row layer took last, if any.
    func next(after hint: Int32?) -> Int32? {
        guard uncachedCount > 0 else { return cached.count > 0 ? cached.first : nil }
        if let hint, let segment = segmentIndex(of: hint) {
            let run = runIndex(of: Int(hint), in: segment)
            if let page = firstUncached(
                in: (Int(hint) + 1) ..< runBounds(run, in: segment).upperBound)
            {
                return page
            }
            if fullRunPages[segment] > 0 { return firstFullRun(in: segment) }
        }
        var roomiest: Int?
        for segment in segments.indices where fullRunPages[segment] > 0 {
            if roomiest.map({ fullRunPages[segment] > fullRunPages[$0] }) ?? true {
                roomiest = segment
            }
        }
        if let roomiest { return firstFullRun(in: roomiest) }
        return lowestUncached()
    }

    // MARK: - Geometry

    private mutating func addRanges<C: Collection>(_ ranges: C) where C.Element == Range<Int> {
        for range in ranges {
            precondition(range.lowerBound == (segments.last?.upperBound ?? 0) && range.count >= 2)
            segments.append(range)
            let runs = (range.count - 1 + Self.runPages - 1) / Self.runPages
            runFree.append(contentsOf: repeatElement(0, count: runs))
            firstRun.append(firstRun[firstRun.count - 1] + runs)
            fullRunPages.append(0)
        }
        let words = ((segments.last?.upperBound ?? 0) + 63) / 64
        if words > uncached.count {
            uncached.append(contentsOf: repeatElement(0, count: words - uncached.count))
            let summary = (words + 63) / 64
            occupiedWords.append(
                contentsOf: repeatElement(0, count: summary - occupiedWords.count))
        }
    }

    /// The segment of a usable page; nil for poison and unknown ids.
    private func segmentIndex(of page: Int32) -> Int? {
        let page = Int(page)
        var low = 0
        var high = segments.count
        while low < high {
            let middle = (low + high) / 2
            if segments[middle].upperBound <= page { low = middle + 1 } else { high = middle }
        }
        guard low < segments.count, segments[low].lowerBound < page else { return nil }
        return low
    }

    private func runIndex(of page: Int, in segment: Int) -> Int {
        firstRun[segment] + (page - segments[segment].lowerBound - 1) / Self.runPages
    }

    private func runBounds(_ run: Int, in segment: Int) -> Range<Int> {
        let start = segments[segment].lowerBound + 1 + (run - firstRun[segment]) * Self.runPages
        return start ..< min(start + Self.runPages, segments[segment].upperBound)
    }

    // MARK: - Uncached pages

    private func isUncached(_ page: Int32) -> Bool {
        let page = Int(page)
        guard page >= 0, page / 64 < uncached.count else { return false }
        return uncached[page / 64] & (1 << UInt64(page % 64)) != 0
    }

    private mutating func insertUncached(_ page: Int32) {
        guard let segment = segmentIndex(of: page), !contains(page) else {
            preconditionFailure("free page \(page) is poison, unknown or already free")
        }
        let index = Int(page)
        uncached[index / 64] |= 1 << UInt64(index % 64)
        occupiedWords[index / 4096] |= 1 << UInt64((index / 64) % 64)
        uncachedCount += 1
        let run = runIndex(of: index, in: segment)
        runFree[run] += 1
        let length = runBounds(run, in: segment).count
        if runFree[run] == length { fullRunPages[segment] += length }
    }

    private mutating func removeUncached(_ page: Int32) {
        let index = Int(page)
        let segment = segmentIndex(of: page)!
        let run = runIndex(of: index, in: segment)
        let length = runBounds(run, in: segment).count
        if runFree[run] == length { fullRunPages[segment] -= length }
        runFree[run] -= 1
        uncached[index / 64] &= ~(1 << UInt64(index % 64))
        if uncached[index / 64] == 0 {
            occupiedWords[index / 4096] &= ~(1 << UInt64((index / 64) % 64))
        }
        uncachedCount -= 1
    }

    private func firstUncached(in range: Range<Int>) -> Int32? {
        for page in range where uncached[page / 64] & (1 << UInt64(page % 64)) != 0 {
            return Int32(page)
        }
        return nil
    }

    private func firstFullRun(in segment: Int) -> Int32 {
        for run in firstRun[segment] ..< firstRun[segment + 1]
        where runFree[run] == runBounds(run, in: segment).count {
            return Int32(runBounds(run, in: segment).lowerBound)
        }
        preconditionFailure("segment \(segment) counts a fully free run it does not have")
    }

    private func lowestUncached() -> Int32? {
        for (summary, words) in occupiedWords.enumerated() where words != 0 {
            let word = summary * 64 + words.trailingZeroBitCount
            return Int32(word * 64 + uncached[word].trailingZeroBitCount)
        }
        return nil
    }
}
