// PagedBlockFreeQueue.swift
//
// Eviction queue of the paged KV allocator's cached tier: zero-ref pages that
// a prefix-cache alias still names. Prefix-cache hits must be able to
// resurrect a zero-ref page from the middle in O(1), which a stack or
// Array.remove(at:) cannot provide.

import Foundation

/// Queue membership is stored in arrays indexed by physical page id. The
/// front is evicted first.
struct PagedBlockFreeQueue {
    private static let none: Int32 = -1

    private var previous: [Int32]
    private var next: [Int32]
    private var present: [Bool]
    private(set) var first: Int32 = none
    private(set) var last: Int32 = none
    private(set) var count = 0

    /// An empty queue over page ids `0 ..< pageCount`.
    init(pageCount: Int) {
        precondition(pageCount >= 0)
        previous = [Int32](repeating: Self.none, count: pageCount)
        next = [Int32](repeating: Self.none, count: pageCount)
        present = [Bool](repeating: false, count: pageCount)
    }

    /// Extend an uncommitted replacement queue; new addresses are absent
    /// until their native segment is installed in the same transaction.
    mutating func extend(to pageCount: Int) {
        precondition(pageCount >= present.count)
        let extra = pageCount - present.count
        previous.append(contentsOf: repeatElement(Self.none, count: extra))
        next.append(contentsOf: repeatElement(Self.none, count: extra))
        present.append(contentsOf: repeatElement(false, count: extra))
    }

    func contains(_ page: Int32) -> Bool {
        guard page >= 0, Int(page) < present.count else { return false }
        return present[Int(page)]
    }

    /// Snapshot in eviction order. Test and diagnostics only.
    var elements: [Int32] {
        var result: [Int32] = []
        result.reserveCapacity(count)
        var cursor = first
        while cursor != Self.none {
            result.append(cursor)
            cursor = next[Int(cursor)]
        }
        return result
    }

    mutating func remove(_ page: Int32) {
        let index = Int(page)
        precondition(index >= 0 && index < present.count && present[index])
        let before = previous[index]
        let after = next[index]
        if before == Self.none {
            first = after
        } else {
            next[Int(before)] = after
        }
        if after == Self.none {
            last = before
        } else {
            previous[Int(after)] = before
        }
        previous[index] = Self.none
        next[index] = Self.none
        present[index] = false
        count -= 1
    }

    mutating func append(_ page: Int32) {
        let index = Int(page)
        precondition(index >= 0 && index < present.count && !present[index])
        let oldLast = last
        previous[index] = oldLast
        next[index] = Self.none
        present[index] = true
        if oldLast == Self.none {
            first = page
        } else {
            next[Int(oldLast)] = page
        }
        last = page
        count += 1
    }

    mutating func append<S: Sequence>(contentsOf pages: S) where S.Element == Int32 {
        for page in pages { append(page) }
    }
}
