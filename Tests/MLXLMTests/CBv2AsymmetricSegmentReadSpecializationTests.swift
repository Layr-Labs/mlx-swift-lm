import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// MLX compiles one Metal library per distinct custom-kernel template tuple,
/// so asymmetric segment reads take their token extent from the output shape.
final class CBv2AsymmetricSegmentReadSpecializationTests: XCTestCase {
    /// The former read, specialized on the transfer length `N`.
    private static let lengthSpecializedRead = MLXFast.metalKernel(
        name: "cbv2_asymmetric_segment_read_length_specialized",
        inputNames: ["storage", "records", "keys", "values", "previous"],
        outputNames: ["fence"],
        source: """
            const int d = int(thread_position_in_grid.x);
            const int h = int(thread_position_in_grid.y);
            const int r = int(thread_position_in_grid.z);
            const int token = records[r * 3], page = records[r * 3 + 1], slot = records[r * 3 + 2];
            const size_t row = ((size_t)page * H + h) * S + slot;
            const size_t target = (size_t)h * N + token;
            if (d < DK) { device T* dst = keys; dst[target * DK + d] = storage[row * DK + d]; }
            if (d < DV) { device T* dst = values; dst[target * DV + d] = storage[VBASE + row * DV + d]; }
            if (d == 0 && h == 0 && r == 0) fence[0] = previous[0] + 1;
            """, ensureRowContiguous: true, mutableInputs: ["keys", "values"])

    func testSegmentTemplateCarriesNoTransferLength() throws {
        let kind = asymPagedKind(192, 128, heads: 2, queries: 4)
        let pageBytes = 2 * 16 * (192 + 128) * 4
        let pool = try PagedKVPool(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 4 << 20, dtype: .float32, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 128, maxBufferLength: 4 << 20,
                segmentSizeBytes: pageBytes * 3))
        let key = pool.groupKey(forLayer: 0)
        try pool.reserve([key: 5])
        try pool.materializeReservedSegments()
        defer { pool.unreserve([key: 5]) }
        let group = pool.group(key)
        for segment in group.segments.values {
            let names = PagedAsymmetricTransfers.segmentTemplate(group, segment).map(\.0)
            XCTAssertEqual(names, ["T", "H", "DK", "DV", "S", "VBASE"])
        }
    }

    /// Bitwise equality with the length-specialized kernel across counts,
    /// first slots, both width orders and dtypes.
    func testSegmentReadMatchesLengthSpecializedKernelBitwise() throws {
        for (dk, dv) in [(192, 128), (128, 192)] {
            for dtype: DType in [.float32, .bfloat16] {
                let kind = asymPagedKind(dk, dv, heads: 2, queries: 4)
                let pageBytes = 2 * 16 * (dk + dv) * dtype.size
                let pool = try PagedKVPool(
                    layerKinds: [kind],
                    config: .init(
                        capacityBytes: 4 << 20, dtype: dtype, maxPrefillChunk: 32,
                        nominalMaxSequenceLength: 128, maxBufferLength: 4 << 20,
                        segmentSizeBytes: pageBytes * 3))
                let key = pool.groupKey(forLayer: 0)
                let group = pool.group(key)
                try pool.reserve([key: 5])
                try pool.materializeReservedSegments()
                let allocated = (0 ..< 5).map { _ in pool.allocatePage(group: key) }
                let pages = Array(allocated.reversed())
                defer {
                    StreamOrDevice.default.stream.synchronize()
                    pool.freePages(group: key, pages: allocated)
                    pool.unreserve([key: 5])
                }
                let total = 5 * 16
                let slots = (0 ..< total).map { pages[$0 / 16] * 16 + Int32($0 % 16) }
                pool.writeTokens(
                    group: key, slots: slots,
                    keys: asymPagedTensor(heads: 2, start: 0, count: total, width: dk, dtype: dtype)
                        .squeezed(axis: 0),
                    values: asymPagedTensor(
                        heads: 2, start: 0, count: total, width: dv, dtype: dtype, bias: -4
                    ).squeezed(axis: 0))
                for (first, count) in [(0, 1), (0, 7), (3, 13), (0, 16), (5, 17), (1, 50), (0, 80)]
                {
                    guard first + count <= total else { continue }
                    let current = PagedAsymmetricTransfers.gatherSegmented(
                        group: group, pages: pages, firstSlot: first, count: count,
                        publishReadFence: false)
                    let former = formerGather(
                        group: group, pages: pages, firstSlot: first, count: count)
                    eval(current.keys, current.values, former.keys, former.values)
                    XCTAssertEqual(
                        current.keys.asData(access: .copy).data,
                        former.keys.asData(access: .copy).data, "\(dk)/\(dv) \(dtype) \(count)")
                    XCTAssertEqual(
                        current.values.asData(access: .copy).data,
                        former.values.asData(access: .copy).data, "\(dk)/\(dv) \(dtype) \(count)")
                }
            }
        }
    }

    private func formerGather(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int)
        -> (keys: MLXArray, values: MLXArray)
    {
        let h = group.key.kvHeads
        let dk = group.key.headDim
        let dv = group.key.valueHeadDim
        let keys = MLXArray.zeros([1, h, count, dk], dtype: group.dtype)
        let values = MLXArray.zeros([1, h, count, dv], dtype: group.dtype)
        let slots = (0 ..< count).map { token -> Int32 in
            let offset = firstSlot + token
            return pages[offset / group.pageSize] * Int32(group.pageSize)
                + Int32(offset % group.pageSize)
        }
        let width = max(dk, dv)
        var fence = group.writeFence
        for (segment, triples) in PagedSegmentTransfers.buckets(group: group, slots: slots) {
            fence =
                Self.lengthSpecializedRead(
                    [segment.storage, PagedSegmentTransfers.records(triples), keys, values, fence],
                    template: [
                        ("T", group.dtype), ("H", h), ("DK", dk), ("DV", dv),
                        ("S", group.pageSize), ("VBASE", segment.valueOffset), ("N", count),
                    ],
                    grid: (width, h, triples.count / 3), threadGroup: (min(256, width), 1, 1),
                    outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
        return (
            depends(input: keys, dependencies: [fence]),
            depends(input: values, dependencies: [fence])
        )
    }
}
