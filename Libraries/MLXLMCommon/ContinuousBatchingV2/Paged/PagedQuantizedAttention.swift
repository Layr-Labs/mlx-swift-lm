import Foundation
import MLX
import MLXFast

/// Original-precision tokens that a packed attention call must prefer. The
/// optional owner carries the admission charge through every aliased reader.
struct PagedQuantizedNativeView {
    let start: Int
    let keys: MLXArray
    let values: MLXArray
    let owner: AnyObject?
}

/// Register-only affine reads and one bounded query-block online reduction.
/// Native recent/pending tokens use original Q; old packed K uses rotated Q.
enum PagedQuantizedAttention {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let completionKernel = MLXFast.metalKernel(
        name: "cbv2_packed_attention_visible", inputNames: ["previous"],
        outputNames: ["fence"], source: "fence[0] = previous[0] + 1;",
        ensureRowContiguous: false)

    static func partThreadgroupBytes(headDim: Int, gqa: Int, simdgroups: Int) -> Int {
        PagedAttentionKernel.partThreadgroupBytes(
            headDim: headDim, gqa: gqa, simdgroups: simdgroups)
            + PagedAttentionKernel.headsPerThreadgroup(headDim: headDim, gqa: gqa) * headDim * 4
    }

    static func simdgroups(headDim: Int, gqa: Int) -> Int? {
        PagedAttentionKernel.simdgroupCandidates.first {
            partThreadgroupBytes(headDim: headDim, gqa: gqa, simdgroups: $0)
                <= PagedAttentionKernel.threadgroupMemoryLimit
        }
    }

    static func partBody(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const int query = int(threadgroup_position_in_grid.z);
            const cbv2::PagedMixedQuantizedSegmentAccessor<NT, SEGMENTS, D, S, G, KB, VB> cache{
                {\(pointers)}, value_offsets, record, KVH,
                native_keys, native_values, int(native_info[0]), int(native_info[1]),
                native_info[2], native_info[3], native_info[4]};
            const uint3 position(threadgroup_position_in_grid.x, query, record[1]);
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0 && thread_position_in_grid.z == 0) {
                fence[0] = previous[0] + 1;
            }
            threadgroup float q_smem[HPT * D];
            threadgroup float native_q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_cached_impl<T, D, S, GQA, HPT, NSG, PTOK, false, HAS_SOFTCAP>(
                q, q, q, cache, seqinfo, seqinfo, params, KVH, 0, int(native_info[5]),
                q_smem, red_smem, partials, meta, position,
                thread_position_in_threadgroup, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup, q_strides[2], q_strides[1], seqinfo[query * 8 + 5],
                R, q_strides[3], true, native_q_smem, true,
                HAS_MASK ? bool_mask : nullptr, native_info[6], native_info[7]);
            """
    }

    /// MLX appends one Metal buffer for each referenced shape/stride vector.
    /// Capture the evaluated layouts in one eight-word input instead: the
    /// 17-segment part then uses 31 bindings, including its completion output.
    /// A GPU metadata pass also avoids observing a lazy array's early strides.
    private static let nativeLayoutKernel = MLXFast.metalKernel(
        name: "cbv2_quantized_native_layout", inputNames: ["native_keys", "bool_mask", "bounds"],
        outputNames: ["layout"],
        source: """
            const int i = int(thread_position_in_grid.x);
            const int64_t fields[8] = {
                bounds[0], int64_t(native_keys_shape[1]),
                native_keys_strides[0], native_keys_strides[1], native_keys_strides[2],
                bounds[1], bool_mask_strides[2], bool_mask_strides[3]};
            layout[i] = fields[i];
            """, ensureRowContiguous: false)

    private static func partKernel(
        key: PagedAttentionKernelKey, group: PagedKVGroup,
        bindings: Int, source: String, hasMask: Bool
    ) -> MLXFast.MLXFastKernel {
        let quant = group.key.quantization!
        let name =
            key.kernelName + "_mixed_k\(quant.keyBits)v\(quant.valueBits)g\(quant.groupSize)"
            + "_r\(quant.resolvedRotationBlockSize(headDim: key.headDim))_native\(group.dtype)_h\(group.key.kvHeads)_n\(bindings)_mask\(hasMask ? 1 : 0)"
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            let kernel = MLXFast.metalKernel(
                name: name,
                inputNames: ["q"] + (0 ..< bindings).map { "segment\($0)" }
                    + [
                        "native_keys", "native_values", "native_info", "seqinfo", "params",
                        "previous", "value_offsets", "records", "partials", "meta", "bool_mask",
                    ],
                outputNames: ["fence"], source: partBody(bindings: bindings),
                header: source + "\n" + PagedQuantizedMetal.header,
                ensureRowContiguous: false, mutableInputs: ["partials", "meta"])
            kernels[name] = kernel
            return kernel
        }
    }

    private static func mergeKernel(key: PagedAttentionKernelKey, source: String)
        -> MLXFast.MLXFastKernel
    {
        let name = key.kernelName + "_mixed_scatter"
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            let kernel = MLXFast.metalKernel(
                name: name,
                inputNames: ["partials", "meta", "seqinfo", "sinks", "previous", "destination"],
                outputNames: ["fence"],
                source: """
                    const int heads = partials_shape[1];
                    const int maxpart = partials_shape[2];
                    cbv2::paged_attention_merge_impl<T, D, PTOK, HAS_SINKS>(
                        partials, meta, seqinfo, sinks, heads, maxpart, destination,
                        threadgroup_position_in_grid, thread_index_in_simdgroup, destination_shape[2]);
                    if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) fence[0] = previous[0] + 1;
                    """, header: source, ensureRowContiguous: false, mutableInputs: ["destination"])
            kernels[name] = kernel
            return kernel
        }
    }

    static func metadataRecordCounts(
        row: PagedSequenceKV, visibleStart: Int, visibleEnd: Int,
        virtualPages: [Int32]? = nil
    ) -> [Int] {
        let group = row.pool.group(row.groupKey)
        let partition = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: group.pageSize)
        let pages = virtualPages ?? row.table
        if pages.isEmpty {
            return [
                ((visibleEnd - visibleStart - 1) / partition + 1)
                    * PagedSegmentDispatchPlan.recordStride
            ]
        }
        let info = PagedAttentionKernel.SeqInfoRow(
            attendStart: visibleStart,
            attendLength: visibleEnd - visibleStart,
            tableLength: virtualPages == nil ? row.decodeTableLength : pages.count)
        return PagedSegmentDispatchPlan(
            rows: [.init(pages: pages, info: info)],
            layout: group.segmentLayout!, pageSize: group.pageSize,
            partitionTokens: partition, hasWrite: false
        ).buckets.map { $0.records.count }
    }

    private struct Prepared {
        let buckets: [PagedSegmentDispatchPlan.Bucket]
        let metadata: [PagedSegmentPreparedDispatch.Metadata]
        init(_ value: PagedSegmentPreparedDispatch) {
            buckets = value.plan.buckets
            metadata = value.metadata
        }
        init(nativeOnlyLength: Int, partitionTokens: Int) {
            let count = (nativeOnlyLength - 1) / partitionTokens + 1
            var records: [Int32] = []
            for partition in 0 ..< count {
                records.append(contentsOf: [0, Int32(partition), 0, 0, 0, 0, Int32(count), 0])
                records.append(
                    contentsOf: repeatElement(0, count: PagedSegmentDispatchPlan.recordStride - 8))
            }
            // No physical page is addressed. The accessor's native-band branch
            // covers every token; an eight-byte inert input only gives Metal a
            // stable binding when the committed prefix is empty.
            buckets = [.init(segmentIDs: [], records: records)]
            metadata = [
                .init(
                    records: MLXArray(records),
                    valueOffsets: MLXArray(Array(repeating: Int64(0), count: 8)))
            ]
        }
    }

    /// One physical row per call; packed batches use this same operation per
    /// row. No current-workspace array scales as queryCount * historyLength.
    static func attend(
        queries: MLXArray, row: PagedSequenceKV, native: PagedQuantizedNativeView,
        queryStart: Int, visibleStart: Int, visibleEnd: Int,
        queryBounds: [Range<Int>], sinks: MLXArray?, params: MLXArray,
        softcap: Bool, workspace: PagedQuantizedAttentionWorkspace, source: String,
        booleanMask: MLXArray? = nil, virtualPages: [Int32]? = nil
    ) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        let quant = group.key.quantization!
        let heads = queries.dim(1)
        let width = queries.dim(3)
        let kvh = group.key.kvHeads
        let count = queries.dim(2)
        let gqa = heads / kvh
        let nsg = simdgroups(headDim: width, gqa: gqa)!
        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: width, gqa: gqa)
        let pages = virtualPages ?? row.table
        let tableLength = virtualPages == nil ? row.decodeTableLength : max(1, pages.count)
        let prepared: Prepared
        if pages.isEmpty {
            prepared = Prepared(
                nativeOnlyLength: visibleEnd - visibleStart,
                partitionTokens: workspace.partitionTokens)
        } else {
            let info = PagedAttentionKernel.SeqInfoRow(
                attendStart: visibleStart,
                attendLength: visibleEnd - visibleStart, tableLength: tableLength)
            let value = PagedSegmentPreparedDispatch(
                rows: [.init(pages: pages, info: info)], group: group,
                partitionTokens: workspace.partitionTokens, hasWrite: false)
            prepared = Prepared(value)
        }
        workspace.lease.retainAdditional(
            prepared.metadata.flatMap { [$0.records, $0.valueOffsets] }
                + [queries, native.keys, native.values, params])
        if let owner = native.owner { workspace.lease.retainOwners([owner]) }
        let noSinks = sinks ?? MLXArray.zeros([max(8, heads)], dtype: .float32)
        let mask = booleanMask ?? MLXArray.ones([1, 1, 1, 8], dtype: .bool)
        let nativeBounds = MLXArray([Int64(native.start), Int64(workspace.maximumPartitions)])
        let nativeInfo = nativeLayoutKernel(
            [native.keys, mask, nativeBounds],
            grid: (8, 1, 1), threadGroup: (8, 1, 1), outputShapes: [[8]], outputDTypes: [.int64])[0]
        let dummyPacked = pages.isEmpty ? MLXArray.zeros([8], dtype: .uint8) : nil
        workspace.lease.retainAdditional(
            [nativeInfo, nativeBounds, noSinks, mask] + (dummyPacked.map { [$0] } ?? []))
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: queries.dtype, headDim: width,
            pageSize: group.pageSize, gqa: gqa, simdgroups: nsg, hasSinks: false,
            hasSoftcap: softcap, partitionTokens: workspace.partitionTokens)
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: queries.dtype, headDim: width,
            pageSize: group.pageSize, gqa: gqa, simdgroups: 1, hasSinks: sinks != nil,
            hasSoftcap: false, partitionTokens: workspace.partitionTokens)
        for offset in stride(
            from: 0, to: count, by: PagedQuantizedAttentionWorkspace.queryBlockSize)
        {
            let block = min(PagedQuantizedAttentionWorkspace.queryBlockSize, count - offset)
            let flat = (offset ..< offset + block).flatMap { index -> [Int32] in
                let bounds = queryBounds[index]
                return [
                    Int32(visibleStart), Int32(visibleEnd - visibleStart), Int32(tableLength),
                    Int32(bounds.lowerBound), Int32(bounds.upperBound), Int32(index), 0, 0,
                ]
            }
            let seqinfo = MLXArray(flat, [block, 8])
            workspace.lease.retainAdditional([seqinfo])
            for (bucket, metadata) in zip(prepared.buckets, prepared.metadata) {
                var backing = bucket.segmentIDs.map { group.segments[$0]!.storage }
                if backing.isEmpty { backing = [dummyPacked!] }
                backing.append(
                    contentsOf: repeatElement(
                        backing[0], count: bucket.bindingClass - backing.count))
                group.writeFence =
                    partKernel(
                        key: partKey, group: group, bindings: bucket.bindingClass, source: source,
                        hasMask: booleanMask != nil)(
                        [queries] + backing + [
                            native.keys, native.values, nativeInfo, seqinfo, params,
                            group.writeFence, metadata.valueOffsets, metadata.records,
                            workspace.partials, workspace.meta, mask,
                        ],
                        template: [
                            ("T", queries.dtype), ("NT", group.dtype), ("D", width),
                            ("S", group.pageSize),
                            ("KVH", kvh), ("GQA", gqa), ("HPT", hpt), ("NSG", nsg),
                            ("PTOK", workspace.partitionTokens), ("HAS_SOFTCAP", softcap),
                            ("SEGMENTS", bucket.bindingClass),
                            ("STRIDE", PagedSegmentDispatchPlan.recordStride),
                            ("G", quant.groupSize), ("KB", quant.keyBits), ("VB", quant.valueBits),
                            ("R", quant.resolvedRotationBlockSize(headDim: width)),
                            ("HAS_MASK", booleanMask != nil),
                        ],
                        grid: (kvh * (gqa / hpt) * 32 * nsg, bucket.workCount, block),
                        threadGroup: (32 * nsg, 1, 1), outputShapes: [[1]], outputDTypes: [.int32])[
                        0]
            }
            group.writeFence =
                mergeKernel(key: mergeKey, source: source)(
                    [
                        workspace.partials, workspace.meta, seqinfo, noSinks, group.writeFence,
                        workspace.output,
                    ],
                    template: [
                        ("T", queries.dtype), ("D", width), ("PTOK", workspace.partitionTokens),
                        ("HAS_SINKS", sinks != nil),
                    ],
                    grid: (heads * 32, block, 1), threadGroup: (32, 1, 1), outputShapes: [[1]],
                    outputDTypes: [.int32])[0]
            // The next query block consumes this actual encoded merge fence
            // before overwriting the exact same partial/meta allocation.
        }
        group.writeFence =
            completionKernel(
                [group.writeFence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
        workspace.lease.retainCompletion(group.writeFence)
        return depends(input: workspace.output, dependencies: [group.writeFence])
    }
}
