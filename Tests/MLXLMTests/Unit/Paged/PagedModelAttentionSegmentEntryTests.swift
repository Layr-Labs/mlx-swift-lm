import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// The entries of the in-place model-attention segment tables,
    /// `binding << 24 | local page`, on synthetic layouts.
    @Suite
    struct PagedModelAttentionSegmentEntryTests {

        @Test func entriesEncodeBindingAndLocalPage() throws {
            let last = PagedLayerCache.modelAttentionMaximumLocalPages - 1
            #expect(last == 0xFF_FFFF)
            let entries = PagedLayerCache.modelAttentionSegmentEntries(
                tables: [[3, 7], [9]], width: 3, binding: { $0 == 9 ? 15 : 2 },
                localPage: { $0 == 9 ? last : Int($0) })
            let flat = try #require(entries)
            let expected: [Int32] = [0x0200_0003, 0x0200_0007, 0, 0x0FFF_FFFF, 0, 0]
            #expect(flat == expected)
        }

        /// A segment of more than 2^24 pages: its local page would spill into
        /// the binding bits, so the rows are not bound in place.
        @Test func localPagesPastTwentyFourBitsAreRefused() {
            let limit = PagedLayerCache.modelAttentionMaximumLocalPages
            #expect(limit == 1 << 24)
            let pastLimit = PagedLayerCache.modelAttentionSegmentEntries(
                tables: [[0, 1, 2]], width: 8, binding: { _ in 0 },
                localPage: { $0 == 2 ? limit : Int($0) })
            #expect(pastLimit == nil)
            let negative = PagedLayerCache.modelAttentionSegmentEntries(
                tables: [[0]], width: 8, binding: { _ in 1 }, localPage: { _ in -1 })
            #expect(negative == nil)
        }

        @Test func bindingsPastOneDispatchAreRefused() {
            let bindings = PagedLayerCache.modelAttentionMaximumBindings
            let entries = PagedLayerCache.modelAttentionSegmentEntries(
                tables: [[4]], width: 8, binding: { _ in bindings }, localPage: { _ in 0 })
            #expect(entries == nil)
        }
    }
}
