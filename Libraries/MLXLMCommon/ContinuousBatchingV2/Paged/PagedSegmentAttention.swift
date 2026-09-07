import Foundation
import MLX
import MLXFast

/// Decode over any number of native segments using bounded binding buckets.
/// Only address resolution differs from the direct-slab numerical reference.
enum PagedSegmentAttention {
    private static let completionKernel = MLXFast.metalKernel(
        name: "cbv2_quantized_attention_consumed", inputNames: ["output", "previous"],
        outputNames: ["fence"], source: "fence[0] = previous[0] + 1;", ensureRowContiguous: true)
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    static func inputNames(bindings: Int) -> [String] {
        ["q", "knew", "vnew"] + (0 ..< bindings).map { "segment\($0)" }
            + ["seqinfo", "params", "previous", "value_offsets", "records", "partials", "meta"]
    }

    static func body(bindings: Int, quantized: Bool = false, broadcast: Bool = false) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        let accessor = quantized
            ? "cbv2::PagedQuantizedSegmentAccessor<T, SEGMENTS, D, S, G, KB, VB>"
            : "cbv2::PagedSegmentAccessor<T, SEGMENTS>"
        let geometry = quantized ? "KVH" : "(size_t)KVH * S * D"
        // Reused query-block workspaces have a larger physical partition
        // stride than early causal rows. Both passes must index that stride.
        let partitionStride = quantized ? "partials_shape[2]" : "record[6]"
        let queryRow = broadcast ? "threadgroup_position_in_grid.z" : "record[0]"
        let queryArguments = quantized
            ? (broadcast ? ", q_strides[2], q_strides[1], seqinfo[position.y * 8 + 5], QR, q_strides[3], true"
                         : ", q_strides[0], q_strides[1], -1, QR, q_strides[2], true") : ""
        let writeGuard = quantized ? "static_assert(!HAS_WRITE, \"packed writes require the separate fenced encoder\");" : ""
        return """
            \(writeGuard)
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const \(accessor) cache{
                {\(pointers)}, value_offsets, record, \(geometry)};
            const uint3 position(threadgroup_position_in_grid.x, \(queryRow), record[1]);
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0 && threadgroup_position_in_grid.z == 0) {
                fence[0] = previous[0] + 1;
            }
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_cached_impl<T, D, S, GQA, HPT, NSG, PTOK, HAS_WRITE, HAS_SOFTCAP>(
                q, knew, vnew, cache, seqinfo, seqinfo, params, KVH, 0, \(partitionStride),
                q_smem, red_smem, const_cast<device float*>(partials), const_cast<device float*>(meta),
                position, thread_position_in_threadgroup, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup\(queryArguments));
            """
    }

    private static func kernel(key: PagedAttentionKernelKey, bindings: Int, kvHeads: Int,
                               source: String, quantization: PagedKVQuantizationConfig? = nil,
                               broadcast: Bool = false) -> MLXFast.MLXFastKernel {
        let name = key.kernelName + "_segments\(bindings)_kvh\(kvHeads)"
            + (quantization.map { "_k\($0.keyBits)v\($0.valueBits)g\($0.groupSize)" } ?? "")
            + (broadcast ? "_broadcast" : "")
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            let made = MLXFast.metalKernel(
                name: name, inputNames: inputNames(bindings: bindings), outputNames: ["fence"],
                source: body(bindings: bindings, quantized: quantization != nil, broadcast: broadcast),
                header: source + (quantization == nil ? "" : "\n" + PagedQuantizedMetal.header),
                ensureRowContiguous: quantization == nil)
            kernels[name] = made
            return made
        }
    }

    private static func mergeKernel(key: PagedAttentionKernelKey, source: String)
        -> MLXFast.MLXFastKernel
    {
        let name = key.kernelName + "_segments_ordered"
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            // previous is a binding-only dependency. CustomKernel registers
            // every input with the Metal encoder, which inserts a full buffer
            // barrier when this prior kernel output is consumed. An alias-only
            // Depends node cannot provide that barrier for hidden writes.
            let made = MLXFast.metalKernel(
                name: name, inputNames: ["partials", "meta", "seqinfo", "sinks", "previous"],
                outputNames: ["out"], source: PagedAttentionMSL.mergeBody,
                header: source, ensureRowContiguous: true)
            kernels[name] = made
            return made
        }
    }

    static func decode(
        queries: MLXArray, newKeys: MLXArray?, newValues: MLXArray?,
        group: PagedKVGroup, rows: [PagedSegmentDispatchPlan.Row],
        sinks: MLXArray?, params: MLXArray, softcap: Bool, source: String,
        dispatchCache: PagedSegmentDispatchCache? = nil,
        workspace suppliedWorkspace: PagedQuantizedAttentionWorkspace? = nil,
        prefill: PagedQuantizedPrefillDispatch? = nil
    ) -> MLXArray {
        var q = prefill != nil ? queries : (queries.ndim == 4 ? queries.squeezed(axis: 2) : queries)
        guard !group.writeValidation.isFaulted else { return q }
        if let newKeys, let newValues {
            guard group.writeValidation.validate(keys: newKeys, values: newValues, expected: group.dtype)
            else { return q }
        }
        precondition(prefill != nil ? (q.ndim == 4 && q.dim(0) == 1)
            : (q.ndim == 3 && q.dim(0) == rows.count))
        precondition((newKeys == nil) == (newValues == nil))
        let quantization = group.key.quantization
        let packed = quantization != nil
        let workspace: PagedQuantizedAttentionWorkspace?
        if packed {
            do {
                workspace = try suppliedWorkspace ?? PagedQuantizedAttentionWorkspace(
                    pool: group.pool!, group: group, queryCount: q.dim(0), blockSize: q.dim(0),
                    queryHeads: q.dim(1), maxAttendLength: rows.map { $0.info.attendLength }.max()!,
                    nativeOutputBytes: q.size * q.dtype.size)
            } catch {
                group.writeValidation.record(error)
                return q
            }
        } else { workspace = nil }
        if packed {
            if let newKeys, let newValues {
                for (index, row) in rows.enumerated() {
                    let slot = row.info.writePage * Int32(group.pageSize) + Int32(row.info.writeSlot)
                    PagedSegmentTransfers.write(
                        group: group, slots: [slot],
                        keys: newKeys[index].expandedDimensions(axis: 1),
                        values: newValues[index].expandedDimensions(axis: 1))
                }
            }
        }
        let hasWrite = newKeys != nil && !packed
        let dtype = packed ? q.dtype : group.dtype
        let b = rows.count, qh = q.dim(1), d = q.dim(-1), kvh = group.key.kvHeads
        precondition(d == group.key.headDim && qh % kvh == 0)
        let gqa = qh / kvh
        let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa)!
        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: gqa)
        let splits = gqa / hpt
        let (seqinfo, maxLength) = PagedAttentionKernel.seqinfo(rows.map(\.info))
        let ptok = workspace?.partitionTokens ?? PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: maxLength, batch: b, kvHeads: kvh,
                headSplits: splits, pageSize: group.pageSize), pageSize: group.pageSize)
        // Quantized metadata is owned by the current workspace lease. Keeping
        // a prepared cache after that lease retires needs a separate memory
        // owner, so the experimental path builds metadata per dispatch.
        let activeDispatchCache = packed ? nil : dispatchCache
        let prepared = prefill?.prepared ?? activeDispatchCache?.prepare(
            rows: rows, group: group, partitionTokens: ptok, hasWrite: hasWrite)
            ?? PagedSegmentPreparedDispatch(
                rows: rows, group: group, partitionTokens: ptok, hasWrite: hasWrite)
        let plan = prepared.plan
        if q.dtype != dtype { q = q.asType(dtype) }
        let k = packed ? q : (newKeys ?? q)
        let v = packed ? q : (newValues ?? q)
        if hasWrite {
            precondition(k.shape == [b, kvh, d] && v.shape == k.shape)
        }
        precondition(workspace == nil || (b <= workspace!.maximumQueries && plan.maxPartitions <= workspace!.maximumPartitions))
        let partials = workspace?.partials ?? MLXArray.zeros([b, qh, plan.maxPartitions, d], dtype: .float32)
        // Padding preserves a device pointer even for a one-head/one-partition probe.
        let meta = workspace?.meta ?? MLXArray.zeros([max(8, b * qh * plan.maxPartitions * 2)], dtype: .float32)
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap,
            partitionTokens: ptok, hasWrite: hasWrite)
        if let workspace {
            group.writeFence = workspace.acquire(after: group.writeFence)
        }
        for (bucket, metadata) in zip(plan.buckets, prepared.metadata) {
            let segments = bucket.segmentIDs.map { group.segments[$0]! }
            let bindingCount = bucket.bindingClass
            var backing = segments.map(\.storage)
            // Empty records never select padded bindings. They alias an input,
            // never a separately declared writable output.
            backing.append(contentsOf: repeatElement(backing[0], count: bindingCount - backing.count))
            let inputs = [q, k, v] + backing + [
                seqinfo, params, group.writeFence, metadata.valueOffsets,
                metadata.records, partials, meta]
            group.writeFence = kernel(
                key: partKey, bindings: bindingCount, kvHeads: kvh, source: source,
                quantization: quantization, broadcast: prefill != nil)(
                inputs,
                template: [("T", dtype), ("D", d), ("S", group.pageSize),
                           ("KVH", kvh), ("GQA", gqa), ("HPT", hpt), ("NSG", nsg),
                           ("PTOK", ptok), ("HAS_WRITE", hasWrite),
                           ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindingCount),
                           ("STRIDE", PagedSegmentDispatchPlan.recordStride)]
                    + (quantization.map { [("KB", $0.keyBits), ("VB", $0.valueBits), ("G", $0.groupSize),
                                            ("QR", $0.resolvedRotationBlockSize(headDim: d))] } ?? []),
                grid: (kvh * splits * 32 * nsg, bucket.workCount, prefill != nil ? b : 1),
                threadGroup: (32 * nsg, 1, 1), outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false, partitionTokens: ptok)
        let zeroSinks = MLXArray.zeros([max(8, qh)], dtype: .float32)
        if let prefill {
            group.writeFence = PagedQuantizedPrefillMerge.dispatch(
                partials: partials, meta: meta, seqinfo: seqinfo, sinks: sinks ?? zeroSinks,
                previous: group.writeFence, output: prefill.output, queryCount: b,
                headDim: d, queryHeads: qh, partitionTokens: ptok, hasSinks: sinks != nil,
                source: source)
            workspace?.recordCompletion(group.writeFence)
            return prefill.output
        }
        let output = mergeKernel(key: mergeKey, source: source)(
            [partials, meta, seqinfo, sinks ?? zeroSinks, group.writeFence],
            template: [("T", dtype), ("D", d), ("PTOK", ptok), ("HAS_SINKS", sinks != nil)],
            grid: (qh * 32, b, 1), threadGroup: (32, 1, 1),
            outputShapes: [[b, qh, d]], outputDTypes: [dtype])[0]
        if let workspace {
            // Every next partial overwrite waits for this merge to consume the
            // shared buffers. Binding output establishes a real Metal read barrier.
            group.writeFence = completionKernel(
                [output, group.writeFence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
            workspace.recordCompletion(group.writeFence)
        }
        return output
    }
}
