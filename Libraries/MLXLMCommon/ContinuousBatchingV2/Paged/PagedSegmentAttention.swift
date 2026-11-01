import Foundation
import MLX
import MLXFast

/// Decode over any number of native segments using bounded binding buckets.
/// Only address resolution differs from the direct-slab numerical reference.
enum PagedSegmentAttention {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    /// `allocatesScratch` is the first bucket's variant: `partials` and `meta`
    /// are its uninitialized outputs instead of mutable inputs. The body and
    /// arithmetic are identical; only the buffers' origin differs.
    static func inputNames(bindings: Int, allocatesScratch: Bool = false) -> [String] {
        ["q", "knew", "vnew"] + (0 ..< bindings).map { "segment\($0)" }
            + ["seqinfo", "params", "previous", "value_offsets", "records"]
            + (allocatesScratch ? [] : ["partials", "meta"])
    }

    static func body(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            const uint3 position(threadgroup_position_in_grid.x, record[0], record[1]);
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_cached_impl<T, D, S, GQA, HPT, NSG, PTOK, HAS_WRITE, HAS_SOFTCAP>(
                q, knew, vnew, cache, seqinfo, seqinfo, params, KVH, 0, record[6],
                q_smem, red_smem, partials, meta,
                position, thread_position_in_threadgroup, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """
    }

    private static func kernel(
        key: PagedAttentionKernelKey, bindings: Int, kvHeads: Int,
        source: String, allocatesScratch: Bool = false
    ) -> MLXFast.MLXFastKernel {
        let name =
            key.kernelName + "_segments\(bindings)_kvh\(kvHeads)"
            + (allocatesScratch ? "_scratch" : "")
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            let made = MLXFast.metalKernel(
                name: name,
                inputNames: inputNames(bindings: bindings, allocatesScratch: allocatesScratch),
                outputNames: ["fence"] + (allocatesScratch ? ["partials", "meta"] : []),
                source: body(bindings: bindings), header: source, ensureRowContiguous: true,
                mutableInputs: (allocatesScratch ? [] : ["partials", "meta"])
                    + (key.hasWrite ? (0 ..< bindings).map { "segment\($0)" } : []))
            kernels[name] = made
            return made
        }
    }

    static func mergeKernel(key: PagedAttentionKernelKey, source: String)
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
        zeroFilledScratch: Bool = false,
        seqinfoCache: PagedSegmentSeqinfoCache? = nil
    ) -> MLXArray {
        var q = queries.ndim == 4 ? queries.squeezed(axis: 2) : queries
        guard !group.writeValidation.isFaulted else { return q }
        if let newKeys, let newValues {
            guard
                group.writeValidation.validate(
                    keys: newKeys, values: newValues, expected: group.dtype)
            else { return q }
        }
        precondition(q.ndim == 3 && q.dim(0) == rows.count)
        precondition((newKeys == nil) == (newValues == nil))
        let dtype = group.dtype
        let b = q.dim(0)
        let qh = q.dim(1)
        let d = q.dim(2)
        let kvh = group.key.kvHeads
        precondition(d == group.key.headDim && qh % kvh == 0)
        let gqa = qh / kvh
        let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa)!
        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: gqa)
        let splits = gqa / hpt
        let (seqinfo, maxLength) =
            seqinfoCache?.seqinfo(rows.map(\.info))
            ?? PagedAttentionKernel.seqinfo(rows.map(\.info))
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: maxLength, batch: b, kvHeads: kvh,
                headSplits: splits, pageSize: group.pageSize), pageSize: group.pageSize)
        let prepared =
            dispatchCache?.prepare(
                rows: rows, group: group, partitionTokens: ptok, hasWrite: newKeys != nil)
            ?? PagedSegmentPreparedDispatch(
                rows: rows, group: group, partitionTokens: ptok, hasWrite: newKeys != nil)
        let plan = prepared.plan
        if q.dtype != dtype { q = q.asType(dtype) }
        let k = newKeys ?? q
        let v = newValues ?? q
        if newKeys != nil {
            precondition(k.shape == [b, kvh, d] && v.shape == k.shape)
        }
        let partialsShape = [b, qh, plan.maxPartitions, d]
        // Padding preserves a device pointer even for a one-head/one-partition probe.
        let metaCount = max(8, b * qh * plan.maxPartitions * 2)
        // No fill dispatch: the first bucket allocates the merge scratch as
        // uninitialized kernel outputs and later buckets write their disjoint
        // slots in place. The merge reads only partitions [0, ceil(len / PTOK))
        // of each row, the plan issues exactly one work record per such
        // (row, partition), and each record's threadgroups write every query
        // head's full partial and (m, l) — so no unwritten slot is ever read
        // and the output bytes cannot change. `zeroFilledScratch` keeps the
        // original zero fills as the exactness reference for tests.
        var partials: MLXArray? =
            zeroFilledScratch ? MLXArray.zeros(partialsShape, dtype: .float32) : nil
        var meta: MLXArray? = zeroFilledScratch ? MLXArray.zeros([metaCount], dtype: .float32) : nil
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap,
            partitionTokens: ptok, hasWrite: newKeys != nil)
        for (bucket, metadata) in zip(plan.buckets, prepared.metadata) {
            let segments = bucket.segmentIDs.map { group.segments[$0]! }
            let bindingCount = bucket.bindingClass
            var backing = segments.map(\.storage)
            // Empty records never select padded bindings. They alias an input,
            // never a separately declared writable output.
            backing.append(
                contentsOf: repeatElement(backing[0], count: bindingCount - backing.count))
            let inputs =
                [q, k, v] + backing + [
                    seqinfo, params, group.writeFence, metadata.valueOffsets, metadata.records,
                ]
            let template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("D", d), ("S", group.pageSize),
                ("KVH", kvh), ("GQA", gqa), ("HPT", hpt), ("NSG", nsg),
                ("PTOK", ptok), ("HAS_WRITE", newKeys != nil),
                ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindingCount),
                ("STRIDE", PagedSegmentDispatchPlan.recordStride),
            ]
            let grid = (kvh * splits * 32 * nsg, bucket.workCount, 1)
            if let partials, let meta {
                group.writeFence =
                    kernel(
                        key: partKey, bindings: bindingCount, kvHeads: kvh, source: source)(
                        inputs + [partials, meta], template: template, grid: grid,
                        threadGroup: (32 * nsg, 1, 1), outputShapes: [[1]], outputDTypes: [.int32])[
                        0]
            } else {
                let outputs = kernel(
                    key: partKey, bindings: bindingCount, kvHeads: kvh, source: source,
                    allocatesScratch: true)(
                        inputs, template: template, grid: grid, threadGroup: (32 * nsg, 1, 1),
                        outputShapes: [[1], partialsShape, [metaCount]],
                        outputDTypes: [.int32, .float32, .float32])
                group.writeFence = outputs[0]
                partials = outputs[1]
                meta = outputs[2]
            }
        }
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false, partitionTokens: ptok)
        let mergeSinks = sinks ?? MLXArray.zeros([max(8, qh)], dtype: .float32)
        return mergeKernel(key: mergeKey, source: source)(
            [partials!, meta!, seqinfo, mergeSinks, group.writeFence],
            template: [("T", dtype), ("D", d), ("PTOK", ptok), ("HAS_SINKS", sinks != nil)],
            grid: (qh * 32, b, 1), threadGroup: (32, 1, 1),
            outputShapes: [[b, qh, d]], outputDTypes: [dtype])[0]
    }
}

/// One decode seqinfo upload per distinct value instead of one per layer.
///
/// The segmented kernels read a row's attend start, length, table length and
/// write slot from seqinfo, but never its write page: the part kernel takes
/// the fused write's destination from its work record (`write_page` ignores
/// its argument) and nothing else reads that word. With the page zeroed, the
/// layers of one group present identical seqinfo in a decode step (same
/// rows' positions, ring length and slot), so the pool shares one immutable
/// array per value. Every value the kernels read is unchanged.
final class PagedSegmentSeqinfoCache {
    private var entries: [(rows: [PagedAttentionKernel.SeqInfoRow], array: MLXArray)] = []
    /// Arrays built from host values (test hook).
    private(set) var uploads = 0
    static let capacity = 4

    func clear() { entries.removeAll() }

    func seqinfo(_ rows: [PagedAttentionKernel.SeqInfoRow]) -> (
        array: MLXArray, maxAttendLength: Int
    ) {
        var visible = rows
        for index in visible.indices { visible[index].writePage = 0 }
        let maxAttendLength = max(1, visible.map(\.attendLength).max() ?? 1)
        if let index = entries.firstIndex(where: { $0.rows == visible }) {
            let hit = entries.remove(at: index)
            entries.insert(hit, at: 0)
            return (hit.array, maxAttendLength)
        }
        let made = PagedAttentionKernel.seqinfo(visible)
        precondition(made.maxAttendLength == maxAttendLength)
        uploads += 1
        entries.insert((visible, made.array), at: 0)
        if entries.count > Self.capacity { entries.removeLast() }
        return made
    }
}

/// Experimental dispatch batching, not a new attention reduction. A single
/// global-cache sequence becomes one independent native decode row per query.
/// Fixed PTOK is required: adaptive batch sizing would change arithmetic.
enum PagedMTPBatchedColumns {
    static let enabled =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_NEMOTRON35_MTP_BATCHED_ATTENTION"] != "0"

    static func eligible(queries: MLXArray, row: PagedSequenceKV, group: PagedKVGroup) -> Bool {
        queries.dim(0) == 1 && (2 ... 8).contains(queries.dim(2))
            && row.windowSize == nil && group.segmentLayout != nil
            && row.absoluteOffset >= row.frozenHighWater
            && PagedAttentionKernel.partitionTargetThreadgroups == 0
    }

    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        row: PagedSequenceKV, group: PagedKVGroup,
        sinks: MLXArray?, params: MLXArray, softcap: Bool,
        source: String, dispatchCache: PagedSegmentDispatchCache
    ) -> MLXArray {
        precondition(eligible(queries: queries, row: row, group: group))
        let count = queries.dim(2)
        let heads = queries.dim(1)
        let width = queries.dim(3)
        // One ordinary chunk write, followed by independent causal query rows.
        row.write(keys: keys.squeezed(axis: 0), values: values.squeezed(axis: 0))
        if PagedMultiQueryAttention.enabled,
            PagedMultiQueryAttention.eligible(
                columns: count, queryHeads: heads, headDim: width, group: group)
        {
            // Item 24: the window's columns share ONE pass over the context.
            let batch = queries.transposed(0, 2, 1, 3).reshaped(count, heads, width)
            let result = PagedMultiQueryAttention.attend(
                queries: batch, row: row, group: group, sinks: sinks, params: params,
                softcap: softcap, source: source, dispatchCache: dispatchCache)
            let output = result.reshaped(1, count, heads, width).transposed(0, 2, 1, 3)
            return output.dtype == queries.dtype ? output : output.asType(queries.dtype)
        }
        let descriptors = (0 ..< count).map { position in
            let end = row.absoluteOffset - count + position + 1
            return PagedSegmentDispatchPlan.Row(
                pages: row.table,
                info: row.seqInfoRow(
                    attending: (
                        start: row.baseOffset,
                        length: end - row.baseOffset
                    )),
                identity: .init(serial: row.serial, tableVersion: row.tableVersion))
        }
        let batch = queries.transposed(0, 2, 1, 3).reshaped(count, heads, width)
        let result = PagedSegmentAttention.decode(
            queries: batch, newKeys: nil, newValues: nil,
            group: group, rows: descriptors, sinks: sinks, params: params, softcap: softcap,
            source: source, dispatchCache: dispatchCache)
        let output = result.reshaped(1, count, heads, width).transposed(0, 2, 1, 3)
        return output.dtype == queries.dtype ? output : output.asType(queries.dtype)
    }
}

/// Item 24: multi-query paged verification. A speculative window's `1 + k`
/// columns of ONE row attend the same pages, column `c` over the first
/// `length - (count - 1 - c)` positions. `PagedMTPBatchedColumns` used to run
/// them as `count` independent decode rows, so every column re-read the whole
/// context (5x the KV bytes per layer per round at depth 4). Here one
/// threadgroup owns one (kv head, partition) and runs one SIMDGROUP per
/// (column, query head) pair, as MLX's `sdpa_vector_2pass` does for `qL > 1`:
/// every simdgroup streams the same K/V rows, so each byte comes from memory
/// once per window. Each simdgroup walks its partition's tokens in order with
/// the same online softmax as pass A, writes the same unnormalised partial and
/// `(m, l)` meta, and the ordinary merge pass reduces them per column. Each
/// simdgroup keeps the per-column kernel's NSG strided online-softmax states
/// (token `i` to state `(i - lo) % NSG`) and merges them in the same order,
/// so a column's partial replays the one-token decode kernel's arithmetic for
/// that column: the same operations in the same order (the hardware runs and
/// the near-tie rule are the proof, not this comment). Kill switch:
/// `DARKBLOOM_PAGED_MTP_MULTI_QUERY=0` (falls back to the per-column rows).
enum PagedMultiQueryAttention {
    static let enabled =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_PAGED_MTP_MULTI_QUERY"] != "0"
    /// Dispatches issued (test/telemetry hook; process-wide).
    nonisolated(unsafe) private(set) static var dispatches = 0
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    static func eligible(columns: Int, queryHeads: Int, headDim: Int, group: PagedKVGroup) -> Bool {
        let kvh = group.key.kvHeads
        guard group.segmentLayout != nil, !group.key.isAsymmetric, kvh > 0,
            queryHeads % kvh == 0, headDim == group.key.headDim, headDim % 32 == 0,
            PagedAttentionKernel.supportedHeadDims.contains(headDim)
        else { return false }
        // One simdgroup per (column, query head of the kv head), at least one
        // column per threadgroup.
        return (2 ... 8).contains(columns)
            && queryHeads / kvh
                <= maximumPairs(
                    headDim: headDim,
                    gqa: queryHeads / kvh)
    }

    /// (column, head) simdgroups one threadgroup may hold. Each simdgroup
    /// keeps NSG online-softmax states of `headDim / 32` floats; register
    /// pressure lowers a pipeline's thread limit below 1,024, and a dispatch
    /// over it fails, so heavy states get 512 threads. Override:
    /// `DARKBLOOM_PAGED_MTP_MQ_THREADS` (64...1024).
    static func maximumPairs(headDim: Int, gqa: Int) -> Int {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_PAGED_MTP_MQ_THREADS"],
            let threads = Int(raw), (64 ... 1024).contains(threads)
        {
            return threads / 32
        }
        let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) ?? 8
        return nsg * headDim / 32 <= 32 ? 32 : 16
    }

    static func body(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            constexpr int EPT = D / 32;
            const int kv_head = threadgroup_position_in_grid.x / GROUPS;
            const int part = record[1];
            const int maxpart = record[6];
            const int pair = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int col = (threadgroup_position_in_grid.x % GROUPS) * CPT + pair / GQA;
            if (col >= COLS) {
                return;
            }
            const int head = kv_head * GQA + pair % GQA;
            const int qh = KVH * GQA;
            const device int32_t* info = seqinfo + col * 8;
            const int attend_start = info[0];
            const int attend_len = info[1];
            const int lo = part * PTOK;
            if (lo >= attend_len) {
                return;
            }
            const int hi = min(attend_len, lo + PTOK);
            const float softcap = params[0];
            const float scale = params[1];
            float qv[EPT];
            const device T* qrow = q + ((size_t)col * (size_t)qh + (size_t)head) * D + lane * EPT;
            for (int e = 0; e < EPT; e++) {
                qv[e] = float(qrow[e]) * scale;
            }
            // NSG online-softmax states, token i to state (i - lo) % NSG, merged
            // in state order: the per-column kernel's simdgroup split and
            // merge, replayed by one simdgroup, so every column's partial is
            // the one-token decode kernel's for that column.
            float m[NSG];
            float l[NSG];
            float acc[NSG][EPT];
            for (int s = 0; s < NSG; s++) {
                m[s] = -FLT_MAX;
                l[s] = 0.0f;
                for (int e = 0; e < EPT; e++) {
                    acc[s][e] = 0.0f;
                }
            }
            float kf[EPT];
            float vf[EPT];
            // Unrolled by NSG so each state index is a compile-time constant
            // (no dynamically indexed register arrays).
            for (int first = lo; first < hi; first += NSG) {
                #pragma unroll
                for (int s = 0; s < NSG; s++) {
                    const int i = first + s;
                    if (i < hi) {
                        const int pos = attend_start + i;
                        const auto page = cache.read_page(seqinfo, pos / S, 0);
                        const size_t base = ((size_t)kv_head * S + (size_t)(pos % S)) * D;
                        cbv2::load_row<T, EPT>(page.key + base, lane, kf);
                        cbv2::load_row<T, EPT>(page.value + base, lane, vf);
                        float partial = 0.0f;
                        for (int e = 0; e < EPT; e++) {
                            partial += qv[e] * kf[e];
                        }
                        float qk = simd_sum(partial);
                        if (HAS_SOFTCAP) {
                            qk = softcap * precise::tanh(qk / softcap);
                        }
                        qk *= M_LOG2E_F;
                        const float m_new = max(m[s], qk);
                        const float corr = exp2(m[s] - m_new);
                        const float p = exp2(qk - m_new);
                        l[s] = l[s] * corr + p;
                        for (int e = 0; e < EPT; e++) {
                            acc[s][e] = acc[s][e] * corr + p * vf[e];
                        }
                        m[s] = m_new;
                    }
                }
            }
            float pm = -FLT_MAX;
            for (int t = 0; t < NSG; t++) {
                pm = max(pm, m[t]);
            }
            float pl = 0.0f;
            for (int t = 0; t < NSG; t++) {
                pl += l[t] * exp2(m[t] - pm);
            }
            const size_t slot = ((size_t)col * (size_t)qh + (size_t)head) * (size_t)maxpart + (size_t)part;
            device float* prow = partials + slot * D + lane * EPT;
            for (int e = 0; e < EPT; e++) {
                float v = 0.0f;
                for (int t = 0; t < NSG; t++) {
                    v += acc[t][e] * exp2(m[t] - pm);
                }
                prow[e] = v;
            }
            if (lane == 0) {
                meta[slot * 2] = pm;
                meta[slot * 2 + 1] = pl;
            }
            """
    }

    private static func kernel(name: String, bindings: Int, source: String) -> MLXFast.MLXFastKernel
    {
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            let made = MLXFast.metalKernel(
                name: name, inputNames: PagedSegmentAttention.inputNames(bindings: bindings),
                outputNames: ["fence"], source: body(bindings: bindings), header: source,
                ensureRowContiguous: true, mutableInputs: ["partials", "meta"])
            kernels[name] = made
            return made
        }
    }

    /// `queries` is `[count, queryHeads, D]` (column-major batch, as the
    /// per-column path lays it out); `row` already holds the window's
    /// columns (written by the caller), so column `c` attends
    /// `[base, absoluteOffset - count + c + 1)`.
    static func attend(
        queries: MLXArray, row: PagedSequenceKV, group: PagedKVGroup,
        sinks: MLXArray?, params: MLXArray, softcap: Bool, source: String,
        dispatchCache: PagedSegmentDispatchCache
    ) -> MLXArray {
        var q = queries
        let count = q.dim(0)
        let qh = q.dim(1)
        let d = q.dim(2)
        guard !group.writeValidation.isFaulted else { return q }
        let kvh = group.key.kvHeads
        let gqa = qh / kvh
        guard let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa) else {
            return q
        }
        // Columns per threadgroup, balanced over the fewest groups: each group
        // reads the context once, so the window reads it `groups` times.
        let groups =
            (count + max(1, maximumPairs(headDim: d, gqa: gqa) / gqa) - 1)
            / max(1, maximumPairs(headDim: d, gqa: gqa) / gqa)
        let cpt = (count + groups - 1) / groups
        let dtype = group.dtype
        let last = (start: row.baseOffset, length: row.absoluteOffset - row.baseOffset)
        // The partition grid is the LONGEST column's; shorter columns skip
        // the tokens past their own end (at most `count - 1` of the last
        // partition) and the merge counts each column's own partitions.
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: group.pageSize)
        let descriptor = PagedSegmentDispatchPlan.Row(
            pages: row.table, info: row.seqInfoRow(attending: last),
            identity: .init(serial: row.serial, tableVersion: row.tableVersion))
        let prepared = dispatchCache.prepare(
            rows: [descriptor], group: group, partitionTokens: ptok, hasWrite: false)
        let plan = prepared.plan
        let (seqinfo, _) = PagedAttentionKernel.seqinfo(
            (0 ..< count).map { c in
                row.seqInfoRow(
                    attending: (start: last.start, length: last.length - (count - 1 - c)))
            })
        if q.dtype != dtype { q = q.asType(dtype) }
        let partials = MLXArray.zeros([count, qh, plan.maxPartitions, d], dtype: .float32)
        let meta = MLXArray.zeros([max(8, count * qh * plan.maxPartitions * 2)], dtype: .float32)
        let dname: String
        switch dtype {
        case .float16: dname = "f16"
        case .float32: dname = "f32"
        case .bfloat16: dname = "bf16"
        default: dname = "dt\(dtype)"
        }
        for (bucket, metadata) in zip(plan.buckets, prepared.metadata) {
            let segments = bucket.segmentIDs.map { group.segments[$0]! }
            let bindingCount = bucket.bindingClass
            var backing = segments.map(\.storage)
            backing.append(
                contentsOf: repeatElement(backing[0], count: bindingCount - backing.count))
            let inputs =
                [q, q, q] + backing + [
                    seqinfo, params, group.writeFence, metadata.valueOffsets,
                    metadata.records, partials, meta,
                ]
            let name =
                "cbv2_paged_mq_\(dname)_d\(d)_s\(group.pageSize)_g\(gqa)_n\(nsg)_c\(count)_x\(cpt)"
                + "_t\(ptok)_kvh\(kvh)_cap\(softcap ? 1 : 0)_segments\(bindingCount)"
            group.writeFence =
                kernel(name: name, bindings: bindingCount, source: source)(
                    inputs,
                    template: [
                        ("T", dtype), ("D", d), ("S", group.pageSize), ("KVH", kvh),
                        ("GQA", gqa), ("NSG", nsg), ("PTOK", ptok), ("HAS_SOFTCAP", softcap),
                        ("GROUPS", groups), ("CPT", cpt), ("COLS", count),
                        ("SEGMENTS", bindingCount),
                        ("STRIDE", PagedSegmentDispatchPlan.recordStride),
                    ],
                    grid: (kvh * groups * 32 * gqa * cpt, bucket.workCount, 1),
                    threadGroup: (32 * gqa * cpt, 1, 1), outputShapes: [[1]],
                    outputDTypes: [.int32])[0]
        }
        lock.withLock { dispatches += 1 }
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false, partitionTokens: ptok)
        let zeroSinks = MLXArray.zeros([max(8, qh)], dtype: .float32)
        return PagedSegmentAttention.mergeKernel(key: mergeKey, source: source)(
            [partials, meta, seqinfo, sinks ?? zeroSinks, group.writeFence],
            template: [("T", dtype), ("D", d), ("PTOK", ptok), ("HAS_SINKS", sinks != nil)],
            grid: (qh * 32, count, 1), threadGroup: (32, 1, 1),
            outputShapes: [[count, qh, d]], outputDTypes: [dtype])[0]
    }
}
