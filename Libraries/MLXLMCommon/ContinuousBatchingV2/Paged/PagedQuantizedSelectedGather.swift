import Foundation
import MLX
import MLXFast

/// Qwen4's compact QSA reader reconstructs selected rows only. Packed keys
/// return to the native RoPE basis, while the confirmed recent band and all
/// pending tokens are copied directly from their original-precision owner.
enum PagedQuantizedSelectedGather {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let completion = MLXFast.metalKernel(
        name: "cbv2_quantized_selected_gather_complete", inputNames: ["previous"],
        outputNames: ["fence"], source: "fence[0] = previous[0] + 1;",
        ensureRowContiguous: false)

    private static func kernel(bindings: Int) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let existing = kernels[bindings] { return existing }
            let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
            let source = """
                const uint d = thread_position_in_threadgroup.x;
                const int h = int(threadgroup_position_in_grid.y);
                const int selected = int(threadgroup_position_in_grid.z);
                const int token = indices[int64_t(selected) * indices_strides[0]];
                const int count = indices_shape[0];
                const bool valid = token >= 0 && token < parameters[0];
                const bool recent = valid && token >= parameters[1]
                    && token - parameters[1] < parameters[2];
                const size_t target = ((size_t)h * count + selected) * D + d;
                threadgroup float keys[D];
                device T* output = destination;
                if (recent) {
                    const int64_t k = int64_t(h) * native_keys_strides[0]
                        + int64_t(token - parameters[1]) * native_keys_strides[1]
                        + int64_t(d) * native_keys_strides[2];
                    const int64_t v = int64_t(h) * native_values_strides[0]
                        + int64_t(token - parameters[1]) * native_values_strides[1]
                        + int64_t(d) * native_values_strides[2];
                    output[target] = native_keys[k];
                    output[(size_t)H * count * D + target] = native_values[v];
                } else if (valid) {
                    const device uchar* buffers[\(bindings)] = {\(pointers)};
                    const int page = table[token / S];
                    int binding = -1;
                    for (int i = 0; i < N; i++) {
                        if (page > bounds[2 * i] && page < bounds[2 * i + 1]) {
                            binding = i;
                            break;
                        }
                    }
                    if (binding >= 0) {
                        constexpr int KROW = D * KB / 8 + 8 * (D / G);
                        constexpr int VROW = D * VB / 8 + 8 * (D / G);
                        const int local = page - bounds[2 * binding];
                        const size_t row = ((size_t)local * H + h) * S + token % S;
                        keys[d] = cbv2::quant_load<KB, D, G>(buffers[binding] + row * KROW, d);
                        const float value = cbv2::quant_load<VB, D, G>(
                            buffers[binding] + size_t(value_offsets[binding]) + row * VROW, d);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint stride = 1; stride < ROTATION; stride <<= 1) {
                            const float a = keys[d];
                            const float b = keys[d ^ stride];
                            const float next = (d & stride) ? b - a : a + b;
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            keys[d] = next;
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }
                        const float key = ROTATION == 0 ? keys[d]
                            : keys[d] * rsqrt(float(ROTATION)) * cbv2::quant_sign(d % max(ROTATION, 1));
                        output[target] = T(key);
                        output[(size_t)H * count * D + target] = T(value);
                    }
                }
                if (h == 0 && d == 0 && selected == 0) fence[0] = previous[0] + 1;
                """
            let made = MLXFast.metalKernel(
                name: "cbv2_quantized_selected_gather_bound\(bindings)",
                inputNames: (0 ..< bindings).map { "segment\($0)" }
                    + [
                        "table", "indices", "parameters", "native_keys", "native_values",
                        "bounds", "value_offsets", "destination", "previous",
                    ],
                outputNames: ["fence"], source: source, header: PagedQuantizedMetal.header,
                ensureRowContiguous: false, mutableInputs: ["destination"])
            kernels[bindings] = made
            return made
        }
    }

    static func gather(row: PagedSequenceKV, indices: MLXArray) throws
        -> (keys: MLXArray, values: MLXArray)
    {
        let pool = row.pool
        let group = pool.group(row.groupKey)
        guard row.supportsQwen4SelectedGather, let format = group.key.quantization,
            group.segmentLayout != nil, group.key.headDim == group.key.valueHeadDim,
            indices.ndim == 1, indices.dtype == .int32, indices.size > 0,
            indices.size <= Int(Int32.max), row.absoluteOffset > 0,
            row.absoluteOffset <= Int(Int32.max), !row.table.isEmpty,
            row.table.allSatisfy(group.isAllocatable)
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid packed selected KV reader")
        }
        let h = group.key.kvHeads
        let d = group.key.headDim
        let count = indices.size
        let segmentIDs = Array(
            Set(
                row.table.map {
                    group.segmentLayout!.segmentIndex(page: $0)
                })
        ).sorted()
        guard !segmentIDs.isEmpty, segmentIDs.allSatisfy({ group.segments[$0] != nil }) else {
            throw CBv2KVError.backendIneligible(reason: "packed selected KV names missing backing")
        }
        let bound = PagedSelectedGather.boundEnabled()
        let batches =
            bound
            ? PagedSelectedGather.bindingBatches(segmentIDs: segmentIDs) : segmentIDs.map { [$0] }
        let bindingCounts = batches.map { batch in
            PagedSelectedGather.bindingClasses.first { $0 >= batch.count }!
        }
        let recentKeys = row.nativeRecentKeys
        let recentValues = row.nativeRecentValues
        guard (recentKeys == nil) == (recentValues == nil),
            format.recentTokenCount == 0 || recentKeys != nil
        else {
            throw CBv2KVError.backendIneligible(
                reason: "packed selected KV has no native recent owner")
        }
        let reserved = try reservationBytes(
            count: count, heads: h, dimension: d, elementBytes: group.dtype.size,
            tableCount: row.table.count, bindingCounts: bindingCounts,
            needsNativePlaceholder: recentKeys == nil,
            maximumBufferBytes: pool.config.maxBufferLength)
        let lease = PagedQuantizedScratchLease(
            reservation: try pool.memoryAdmission?.reserveTransient(bytes: reserved),
            bytes: reserved)
        pool.pendingQuantizedScratch.append(lease)
        if let owner = row.nativeRecentStorageOwner { lease.retainOwners([owner]) }
        let nativeKeys = recentKeys ?? MLXArray.zeros([1, 8, d], dtype: group.dtype)
        let nativeValues = recentValues ?? MLXArray.zeros([1, 8, d], dtype: group.dtype)
        let parameters = MLXArray([
            Int32(row.absoluteOffset), Int32(row.nativeRecentStart),
            Int32(recentKeys?.dim(1) ?? 0), 0, 0, 0, 0, 0,
        ])
        let table = MLXArray(row.table)
        let destination = MLXArray.zeros([2, 1, h, count, d], dtype: group.dtype)
        lease.retainAdditional([indices, table, parameters, nativeKeys, nativeValues, destination])
        if bound {
            PagedSelectedGatherInvocation.record(
                pages: table.size, segments: segmentIDs.count,
                passes: batches.count, selectedRows: count)
        }
        var fence = group.writeFence
        for (batch, bindingCount) in zip(batches, bindingCounts) {
            let segments = batch.map { group.segments[$0]! }
            var storage = segments.map(\.storage)
            storage.append(
                contentsOf: repeatElement(storage[0], count: bindingCount - storage.count))
            var bounds = segments.flatMap {
                [Int32($0.pages.lowerBound), Int32($0.pages.upperBound)]
            }
            bounds.append(
                contentsOf: repeatElement(Int32(0), count: 2 * (bindingCount - segments.count)))
            var offsets = segments.map { Int64($0.valueOffset) }
            offsets.append(
                contentsOf: repeatElement(Int64(0), count: bindingCount - segments.count))
            let boundsArray = MLXArray(bounds)
            let offsetsArray = MLXArray(offsets)
            lease.retainAdditional(storage + [boundsArray, offsetsArray])
            fence =
                kernel(bindings: bindingCount)(
                    storage + [
                        table, indices, parameters, nativeKeys, nativeValues,
                        boundsArray, offsetsArray, destination, fence,
                    ],
                    template: [
                        ("T", group.dtype), ("H", h), ("D", d), ("S", group.pageSize),
                        ("N", bindingCount), ("G", format.groupSize), ("KB", format.keyBits),
                        ("VB", format.valueBits),
                        ("ROTATION", format.resolvedRotationBlockSize(headDim: d)),
                    ],
                    grid: (d, h, count), threadGroup: (d, 1, 1),
                    outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
        fence =
            completion(
                [fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
        group.writeFence = fence
        lease.retainCompletion(fence)
        let output = depends(input: destination, dependencies: [fence])
        return (output[0], output[1])
    }

    private static func reservationBytes(
        count: Int, heads: Int, dimension: Int, elementBytes: Int, tableCount: Int,
        bindingCounts: [Int], needsNativePlaceholder: Bool, maximumBufferBytes: Int
    ) throws -> Int {
        // Host page-to-segment plans can grow with a long lifetime even though
        // the selected destination is bounded. Cover their vectors, hash-set
        // capacity and control wrappers independently of native buffer bounds.
        var total = try CBv2CheckpointAllocationFootprint.add(
            64 << 10, PagedKVQuantizationConfig.multiply(tableCount, 128))
        func buffer(_ factors: [Int], copies: Int = 1) throws {
            let bytes = try factors.reduce(1, PagedKVQuantizationConfig.multiply)
            guard bytes > 0, bytes <= maximumBufferBytes else {
                throw CBv2KVError.backendIneligible(
                    reason: "packed selected KV exceeds buffer limit")
            }
            total = try CBv2CheckpointAllocationFootprint.add(
                total,
                PagedKVQuantizationConfig.multiply(
                    copies, CBv2CheckpointAllocationFootprint.bound(bytes)))
        }
        try buffer([2, heads, count, dimension, elementBytes])
        try buffer([tableCount, 4])
        try buffer([8, 4])
        try buffer([elementBytes], copies: needsNativePlaceholder ? 3 : 1)
        try buffer([4], copies: bindingCounts.count + 1)
        for count in bindingCounts {
            try buffer([2, count, 4])
            try buffer([count, 8])
        }
        if needsNativePlaceholder { try buffer([8, dimension, elementBytes], copies: 2) }
        return total
    }
}
