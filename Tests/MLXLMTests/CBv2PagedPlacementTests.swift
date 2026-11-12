import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Where a segmented pool places the pages that rows write through
/// `PagedSequenceKV`: each row layer in whole 16-page runs, row layers that
/// start together in different segments, and a retired row's holes not
/// scattered into the next row.
@Suite("CBv2 paged page placement", .serialized)
struct CBv2PagedPlacementTests {
    private static let kind = CBv2LayerKind(
        attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)

    /// Two layers in one group; segments of 64 usable pages (4 runs).
    private func backend() throws -> PagedKVBackend {
        let pageBytes = 2 * 16 * 64 * DType.bfloat16.size
        return try PagedKVBackend(
            layerKinds: [Self.kind, Self.kind],
            config: PagedKVPoolConfig(
                capacityBytes: 4 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 2048, maxBufferLength: 16 << 20,
                segmentSizeBytes: pageBytes * 65))
    }

    private func rows(_ state: [CBv2SequenceKV?]) throws -> [PagedSequenceKV] {
        try state.map { try #require($0 as? PagedSequenceKV) }
    }

    /// Decode `tokens` positions on every row layer, interleaved as a batch does.
    private func decode(_ tokens: Int, _ rows: [[PagedSequenceKV]]) {
        for _ in 0 ..< tokens {
            for layers in rows {
                for row in layers { _ = row.prepareDecodeWrite() }
            }
        }
    }

    private func segments(_ table: [Int32], _ group: PagedKVGroup) -> Set<Int> {
        Set(table.map { group.segmentLayout!.segmentIndex(page: $0) })
    }

    private func isOneRun(_ pages: ArraySlice<Int32>) -> Bool {
        zip(pages, pages.dropFirst()).allSatisfy { $0 + 1 == $1 }
    }

    /// Every 256-token partition's 16 pages are consecutive.
    private func partitionsAreRuns(_ table: [Int32]) -> Bool {
        stride(from: 0, to: table.count, by: 16).allSatisfy {
            isOneRun(table[$0 ..< min($0 + 16, table.count)])
        }
    }

    @Test func interleavedRowLayersEachFillWholeRuns() throws {
        let backend = try backend()
        let first = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: 0, maxLength: 512)
        defer { backend.release(first) }
        let second = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: 0, maxLength: 512)
        defer { backend.release(second) }
        let tables = [try rows(first), try rows(second)]
        let group = backend.pool.group(tables[0][0].groupKey)
        #expect(group.segments.count == 2)

        decode(256, tables)
        for row in tables.joined() {
            #expect(row.table.count == 16)
            #expect(partitionsAreRuns(row.table), "\(row.table)")
        }
        // The two layers of a row start together and take different segments.
        for layers in tables {
            #expect(segments(layers[0].table, group) != segments(layers[1].table, group))
        }
    }

    @Test func aRetiredRowsHolesDoNotScatterTheNextRow() throws {
        let backend = try backend()
        let longLived = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: 0, maxLength: 512)
        defer { backend.release(longLived) }
        let retiring = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: 0, maxLength: 512)
        let live = try rows(longLived)
        decode(160, [live, try rows(retiring)])
        backend.release(retiring)

        let next = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: 0, maxLength: 512)
        defer { backend.release(next) }
        let fresh = try rows(next)
        decode(352, [live, fresh])
        let group = backend.pool.group(live[0].groupKey)
        #expect(live.map(\.table.count) == [32, 32] && fresh.map(\.table.count) == [22, 22])
        for row in live + fresh {
            #expect(segments(row.table, group).count == 1, "\(row.table)")
            #expect(partitionsAreRuns(row.table), "\(row.table)")
        }
    }
}
