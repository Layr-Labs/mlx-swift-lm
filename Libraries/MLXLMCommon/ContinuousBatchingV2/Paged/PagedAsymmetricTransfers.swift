import MLX
import MLXFast

/// Role-addressed copies only. No attention math and no padding/truncation.
/// Equal-width callers continue through their original kernels.
enum PagedAsymmetricTransfers {
    private static let fixedWrite = MLXFast.metalKernel(
        name: "cbv2_asymmetric_fixed_write", inputNames: ["keys", "values", "kslab", "vslab", "slots", "previous"],
        outputNames: ["fence"], source: """
        const int d = int(thread_position_in_grid.x);
        const int h = int(thread_position_in_grid.y);
        const int r = int(thread_position_in_grid.z);
        const int n = keys_shape[1];
        const int page = slots[r] / S, slot = slots[r] % S;
        if (page > 0) {
            const size_t row = ((size_t)page * H + h) * S + slot;
            if (d < DK) { device T* dst = kslab; dst[row * DK + d] = keys[((size_t)h * n + r) * DK + d]; }
            if (d < DV) { device T* dst = vslab; dst[row * DV + d] = values[((size_t)h * n + r) * DV + d]; }
        }
        if (d == 0 && h == 0 && r == 0) fence[0] = previous[0] + 1;
        """, ensureRowContiguous: true, mutableInputs: ["kslab", "vslab"])

    private static let segmentWrite = MLXFast.metalKernel(
        name: "cbv2_asymmetric_segment_write", inputNames: ["keys", "values", "storage", "records", "previous"],
        outputNames: ["fence"], source: """
        const int d = int(thread_position_in_grid.x);
        const int h = int(thread_position_in_grid.y);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3], page = records[r * 3 + 1], slot = records[r * 3 + 2];
        const int n = keys_shape[1];
        if (page > 0) {
            const size_t row = ((size_t)page * H + h) * S + slot;
            device T* dst = storage;
            if (d < DK) dst[row * DK + d] = keys[((size_t)h * n + token) * DK + d];
            if (d < DV) dst[VBASE + row * DV + d] = values[((size_t)h * n + token) * DV + d];
        }
        if (d == 0 && h == 0 && r == 0) fence[0] = previous[0] + 1;
        """, ensureRowContiguous: true, mutableInputs: ["storage"])

    private static let segmentRead = MLXFast.metalKernel(
        name: "cbv2_asymmetric_segment_read", inputNames: ["storage", "records", "keys", "values", "previous"],
        outputNames: ["fence"], source: """
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

    private static let complete = MLXFast.metalKernel(
        name: "cbv2_asymmetric_transfer_complete", inputNames: ["previous"], outputNames: ["witness"],
        source: "witness[0] = previous[0] + 1;", ensureRowContiguous: true)

    private static func arguments(_ group: PagedKVGroup) -> [(String, any KernelTemplateArg)] {
        [("T",group.dtype),("H",group.key.kvHeads),("DK",group.key.headDim),
         ("DV",group.key.valueHeadDim),("S",group.pageSize)]
    }

    private static func validate(group: PagedKVGroup, slots: [Int32], keys: MLXArray, values: MLXArray) -> Bool {
        guard group.writeValidation.validate(keys: keys, values: values, expected: group.dtype),
              group.writeValidation.validateShape(keys: keys, values: values, group: group.key,
                rank: 3, batch: nil, tokens: slots.count) else { return false }
        guard Set(slots).count == slots.count, slots.allSatisfy({ slot in
            slot >= 0 && group.isAllocatable(slot / Int32(group.pageSize))
                && group.refCounts[Int(slot / Int32(group.pageSize))] > 0
        }) else { return group.writeValidation.refuse("invalid or duplicate asymmetric write destination", expected: group.dtype) }
        return true
    }

    static func writeFixed(group: PagedKVGroup, slots: [Int32], keys: MLXArray, values: MLXArray) {
        guard !slots.isEmpty, validate(group: group, slots: slots, keys: keys, values: values) else { return }
        let width = max(group.key.headDim, group.key.valueHeadDim)
        let padded = slots + Array(repeating: Int32(0), count: max(0, 8 - slots.count))
        group.writeFence = fixedWrite([keys,values,group.kSlab,group.vSlab,MLXArray(padded),group.writeFence],
            template: arguments(group), grid: (width,group.key.kvHeads,slots.count),
            threadGroup: (min(256,width),1,1), outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    static func writeSegmented(group: PagedKVGroup, slots: [Int32], keys: MLXArray, values: MLXArray,
                               work: CBv2PagedAttentionStepOwner? = nil) {
        guard !slots.isEmpty, validate(group: group, slots: slots, keys: keys, values: values) else { return }
        let width = max(group.key.headDim, group.key.valueHeadDim)
        for (segment, triples) in PagedSegmentTransfers.buckets(group: group, slots: slots) {
            let records = PagedSegmentTransfers.records(triples)
            group.writeFence = segmentWrite([keys,values,segment.storage,records,group.writeFence],
                template: arguments(group) + [("VBASE",segment.valueOffset)],
                grid: (width,group.key.kvHeads,triples.count / 3), threadGroup: (min(256,width),1,1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
            work?.retainRoots([segment.storage, keys, values, records, group.writeFence])
        }
    }

    static func gatherSegmented(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int,
                                work: CBv2PagedAttentionStepOwner? = nil,
                                publishReadFence: Bool = true, stream: StreamOrDevice = .default)
        -> (keys: MLXArray, values: MLXArray) {
        let h = group.key.kvHeads, dk = group.key.headDim, dv = group.key.valueHeadDim
        precondition(count >= 0)
        let keys = MLXArray.zeros([1,h,count,dk],dtype: group.dtype,stream: stream)
        let values = MLXArray.zeros([1,h,count,dv],dtype: group.dtype,stream: stream)
        guard count > 0 else { return (keys,values) }
        precondition(firstSlot >= 0 && firstSlot < group.pageSize && pages.count * group.pageSize >= firstSlot + count)
        let slots = (0..<count).map { token -> Int32 in
            let offset = firstSlot + token
            return pages[offset / group.pageSize] * Int32(group.pageSize) + Int32(offset % group.pageSize)
        }
        let width = max(dk,dv)
        var fence = group.writeFence
        for (segment,triples) in PagedSegmentTransfers.buckets(group: group, slots: slots) {
            let records = PagedSegmentTransfers.records(triples)
            fence = segmentRead([segment.storage,records,keys,values,fence],
                template: arguments(group) + [("VBASE",segment.valueOffset),("N",count)],
                grid: (width,h,triples.count / 3),threadGroup: (min(256,width),1,1),
                outputShapes: [[1]],outputDTypes: [.int32],stream: stream)[0]
            work?.retainRoots([segment.storage, records, fence])
        }
        // A real final consumer supplies the read barrier before any later
        // in-place overwrite. Depends alone is only an alias dependency.
        fence = complete([fence],grid: (1,1,1),threadGroup: (1,1,1),outputShapes: [[1]],outputDTypes: [.int32],stream: stream)[0]
        // Historical copies are owned/fenced by their step before successors;
        // a failed private copy must not replace the live serving write fence.
        if publishReadFence { group.writeFence = fence }
        work?.retainRoots([keys, values, fence])
        return (depends(input: keys,dependencies: [fence]),depends(input: values,dependencies: [fence]))
    }
}
