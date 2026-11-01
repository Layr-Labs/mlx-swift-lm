import Foundation
import MLX
import MLXFast

extension PagedQuantizedTransfers {
    private static let snapshotLock = NSLock()
    nonisolated(unsafe) private static var snapshotKernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let snapshotCompletion = MLXFast.metalKernel(
        name: "cbv2_packed_snapshot_visible", inputNames: ["previous"], outputNames: ["fence"],
        source: "fence[0] = previous[0] + 1;", ensureRowContiguous: false)

    /// Copy the exact packed codes and FP32 affine metadata. No quantization,
    /// dequantization or basis transformation occurs in this checkpoint read.
    /// A false publication flag requires the caller's ordinary step barrier to
    /// hold successor writes until this private copy completed or drained.
    static func gatherPacked(
        group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int,
        publishReadFence: Bool = true, stream: StreamOrDevice = .default
    ) -> (keys: MLXArray, values: MLXArray) {
        precondition(group.key.quantization != nil && group.segmentLayout != nil)
        let keyLayout = try! group.key.quantization!.rowLayout(headDim: group.key.headDim)
        let valueLayout = try! group.key.quantization!.rowLayout(headDim: group.key.valueHeadDim)
        let heads = group.key.kvHeads
        let keyBytes = keyLayout.keyRowBytes
        let valueBytes = valueLayout.valueRowBytes
        precondition(count >= 0 && firstSlot >= 0 && firstSlot < group.pageSize)
        let keys = MLXArray.zeros([1, heads, count, keyBytes], dtype: .uint8, stream: stream)
        let values = MLXArray.zeros([1, heads, count, valueBytes], dtype: .uint8, stream: stream)
        guard count > 0 else { return (keys, values) }
        precondition(pages.count * group.pageSize >= firstSlot + count)
        let slots = (0 ..< count).map { token -> Int32 in
            let offset = firstSlot + token
            return pages[offset / group.pageSize] * Int32(group.pageSize)
                + Int32(offset % group.pageSize)
        }
        let name = "cbv2_packed_snapshot_k\(keyBytes)v\(valueBytes)"
        let kernel = snapshotLock.withLock {
            if let existing = snapshotKernels[name] { return existing }
            let made = MLXFast.metalKernel(
                name: name,
                inputNames: [
                    "storage", "records", "key_output", "value_output", "previous", "value_base",
                ],
                outputNames: ["fence"],
                source: """
                    const int column = int(thread_position_in_grid.x);
                    const int head = int(thread_position_in_grid.y);
                    const int record = int(thread_position_in_grid.z);
                    const int token = records[record * 3];
                    const int page = records[record * 3 + 1];
                    const int slot = records[record * 3 + 2];
                    const size_t row = ((size_t)page * H + head) * S + slot;
                    if (column < KROW) key_output[((size_t)head * key_output_shape[2] + token) * KROW + column]
                        = storage[row * KROW + column];
                    if (column < VROW) value_output[((size_t)head * value_output_shape[2] + token) * VROW + column]
                        = storage[size_t(value_base[0]) + row * VROW + column];
                    if (column == 0 && head == 0 && record == 0) fence[0] = previous[0] + 1;
                    """, ensureRowContiguous: false, mutableInputs: ["key_output", "value_output"])
            snapshotKernels[name] = made
            return made
        }
        var fence = group.writeFence
        for (segment, records) in PagedSegmentTransfers.buckets(group: group, slots: slots) {
            fence =
                kernel(
                    [
                        segment.storage, PagedSegmentTransfers.records(records), keys, values,
                        fence,
                        PagedSegmentTransfers.quantizedValueBase(segment),
                    ],
                    template: [
                        ("H", heads), ("S", group.pageSize), ("KROW", keyBytes),
                        ("VROW", valueBytes),
                    ],
                    grid: (max(keyBytes, valueBytes), heads, records.count / 3),
                    threadGroup: (min(256, max(keyBytes, valueBytes)), 1, 1),
                    outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        }
        fence =
            snapshotCompletion(
                [fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        if publishReadFence { group.writeFence = fence }
        return (
            depends(input: keys, dependencies: [fence]),
            depends(input: values, dependencies: [fence])
        )
    }
}
