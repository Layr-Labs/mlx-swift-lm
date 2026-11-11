import Foundation

/// The free pages of one paged-KV group and the choice of the next page.
///
/// Zero-ref pages that a prefix-cache alias still names are "cached"; the
/// rest are "uncached". `PagedKVGroup` owns refcounts and generations; this
/// type only orders the free pages.
struct PagedKVFreePages {
    private var queue: PagedBlockFreeQueue

    /// Fixed slab: every page but the poison page starts free.
    init(fixedPageCount: Int) {
        queue = PagedBlockFreeQueue(pageCount: fixedPageCount, excluding: PagedKVGroup.poisonPage)
    }

    /// Segmented pool: pages become free when their segment is committed.
    init(layout: PagedKVSegmentLayout) {
        queue = PagedBlockFreeQueue(
            pageCount: layout.pageCount, excluding: PagedKVGroup.poisonPage,
            initiallyEmpty: true)
    }

    var count: Int { queue.count }

    /// Snapshot in allocation order. Tests and diagnostics only.
    var elements: [Int32] { queue.elements }

    func contains(_ page: Int32) -> Bool { queue.contains(page) }

    /// Adopt the address ranges `layout` added. Their pages stay absent until
    /// `addFresh` installs a committed segment.
    mutating func extend(to layout: PagedKVSegmentLayout) {
        queue.extend(to: layout.pageCount)
    }

    /// Pages of a newly committed segment.
    mutating func addFresh<S: Sequence>(_ pages: S) where S.Element == Int32 {
        queue.append(contentsOf: pages)
    }

    /// Pages whose last owner released them, in the caller's eviction order.
    /// Uncached pages are reused first; cached pages queue behind them.
    mutating func addReleased(uncached: [Int32], cached: [Int32]) {
        queue.prepend(contentsOf: uncached)
        queue.append(contentsOf: cached)
    }

    /// A queued page lost its last cache alias: reuse it before cached pages.
    mutating func markUncached(_ page: Int32) {
        queue.moveToFront(page)
    }

    mutating func remove(_ page: Int32) {
        queue.remove(page)
    }

    /// The page the next allocation takes. `hint` is the page the same row
    /// layer took last, if any. Does not remove the page.
    func next(after hint: Int32?) -> Int32? {
        queue.count > 0 ? queue.first : nil
    }
}
