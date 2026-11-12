import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// The paged allocator's free pages: the cached and uncached tiers, and
    /// where an uncached page is placed.
    @Suite struct PagedKVFreePagesTests {

        /// Two committed segments of 64 usable pages: pages 1...64 and 66...129.
        private func twoSegments() throws -> PagedKVFreePages {
            let empty = try PagedKVSegmentLayout(
                pageBytes: 1, targetBytes: 65, maximumBufferBytes: 1 << 20)
            let layout = try empty.adding(usablePages: 128, excluding: []).layout
            #expect(layout.ranges == [0 ..< 65, 65 ..< 130])
            var free = PagedKVFreePages(layout: empty)
            free.extend(to: layout)
            for range in layout.ranges { free.addFresh(range.dropFirst().map { Int32($0) }) }
            return free
        }

        private func take(_ free: inout PagedKVFreePages, after hint: Int32?) -> Int32 {
            let page = free.next(after: hint)!
            free.remove(page)
            return page
        }

        @Test func uncachedPagesGoFirstAndCachedPagesKeepTheirEvictionOrder() {
            var free = PagedKVFreePages(fixedPageCount: 12)
            #expect(free.count == 11 && !free.contains(0))
            var held: [Int32] = []
            while free.count > 0 { held.append(take(&free, after: held.last)) }
            #expect(held == Array(1 ... 11))

            free.addReleased(uncached: [], cached: [5, 3, 9])
            free.addReleased(uncached: [7], cached: [])
            #expect(free.elements == [7, 5, 3, 9])
            #expect(take(&free, after: nil) == 7)
            #expect(take(&free, after: nil) == 5)

            // A cached page whose last alias went away is reused first.
            free.markUncached(9)
            #expect(free.elements == [9, 3])
            #expect(take(&free, after: nil) == 9)

            // A prefix-cache hit resurrects a page from the middle of either tier.
            free.addReleased(uncached: [2, 4], cached: [6, 8])
            free.remove(4)
            free.remove(6)
            #expect(free.elements == [2, 3, 8])
            #expect(free.contains(3) && free.contains(2) && !free.contains(6))
        }

        @Test func aRowLayerFillsItsRunThenAnotherRunInItsSegment() throws {
            var free = try twoSegments()
            var table: [Int32] = []
            for _ in 0 ..< 20 { table.append(take(&free, after: table.last)) }
            #expect(table == Array(1 ... 20))
        }

        @Test func rowLayersStartingTogetherTakeSeparateSegments() throws {
            var free = try twoSegments()
            var first: [Int32] = []
            var second: [Int32] = []
            for _ in 0 ..< 40 {
                first.append(take(&free, after: first.last))
                second.append(take(&free, after: second.last))
            }
            #expect(first == Array(1 ... 40))
            #expect(second == Array(66 ... 105))
        }

        @Test func aRowLayerLeavesAFullSegmentForTheRoomiestOne() throws {
            var free = try twoSegments()
            // Segment 0 keeps one fully free run (49...64); segment 1 keeps two.
            for page in Int32(1) ... 48 { free.remove(page) }
            for page in Int32(66) ... 97 { free.remove(page) }
            var table: [Int32] = [48]
            for _ in 0 ..< 16 { table.append(take(&free, after: table.last)) }
            #expect(table.dropFirst() == ArraySlice(Int32(49) ... 64))
            // Its segment is full now: the next run is in the other segment.
            #expect(take(&free, after: table.last) == 98)
            // With no hint, the roomiest segment's lowest fully free run.
            #expect(take(&free, after: nil) == 114)
        }

        @Test func withNoFullyFreeRunTheLowestFreePageIsTaken() throws {
            var free = try twoSegments()
            for page in Int32(1) ... 129 where page != 65 && page % 4 != 0 { free.remove(page) }
            #expect(take(&free, after: nil) == 4)
            // The rest of the hint's run still comes first.
            #expect(take(&free, after: 9) == 12)
            #expect(take(&free, after: 100) == 104)
            #expect(take(&free, after: 128) == 8)
        }

        /// With no fully free run, the fallback scans the summary words. Emptying
        /// a word must clear its summary bit, or the scan returns a held page.
        @Test func theFallbackSkipsAnEmptiedWord() throws {
            let empty = try PagedKVSegmentLayout(
                pageBytes: 1, targetBytes: 129, maximumBufferBytes: 1 << 20)
            let layout = try empty.adding(usablePages: 256, excluding: []).layout
            #expect(layout.ranges == [0 ..< 129, 129 ..< 258])
            var free = PagedKVFreePages(layout: empty)
            free.extend(to: layout)
            for range in layout.ranges { free.addFresh(range.dropFirst().map { Int32($0) }) }
            // Hold the first page of every run, so no run is fully free.
            for range in layout.ranges {
                for start in stride(from: range.lowerBound + 1, to: range.upperBound, by: 16) {
                    free.remove(Int32(start))
                }
            }
            // Empty word 0 (pages 1...63) and hold page 64, the first page of word 1.
            for page in Int32(1) ..< 64 where free.contains(page) { free.remove(page) }
            free.remove(64)
            #expect(free.next(after: nil) == 66)
        }

        @Test func aRetiredSegmentIsNeverChosen() throws {
            var free = try twoSegments()
            for page in Int32(66) ... 129 { free.remove(page) }
            free.remove(1)
            #expect(take(&free, after: nil) == 17)
            #expect(free.count == 62)
        }
    }
}
