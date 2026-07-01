// PagedLayerCache.swift
//
// Batch-facing per-layer cache for the paged backend (WS-C): the
// `CBv2AttendingLayerCache` that owns BOTH the KV update and the attention
// dispatch, so models and the scheduler never see the storage layout.
//
// Attention paths (numerically pinned — one path per phase, never switching
// mask representation across steps):
//   - decode (`L == 1`, any B, including B == 1): the paged Metal kernel.
//     Per-row window clamping is start-offset arithmetic on absolute
//     positions computed host-side from Swift Ints; masks do not exist.
//   - prefill chunk (`B == 1`, `L > 1`): per-request SDPA over gathered
//     pages with an explicit BOOL mask built from absolute positions
//     (always `.array`, never `.none`/`.causal`, so the path cannot drift
//     — see MLX #3384). Models with an attention-logit softcap use the
//     composed reference path instead (SDPA cannot express softcap).
//
// Rows with different retained lengths are handled internally: the caller
// never pads and never builds masks.

import Foundation
import MLX
import MLXFast

public final class PagedLayerCache: CBv2AttendingLayerCache {
    public let layerIndex: Int
    public let kind: CBv2LayerKind
    let pool: PagedKVPool
    /// Optional attention-logit soft cap. Not part of the contract's
    /// `updateAndAttend` signature, so it is layer-cache configuration
    /// (from model config) instead.
    public let attentionSoftcap: Float?

    private var pagedRows: [PagedSequenceKV] = []

    // Device block-table cache: rebuilt only when a row's page table
    // changes (page allocated / rollback / composition change), not on
    // every step.
    private var cachedTables: MLXArray?
    private var cachedTablesFingerprint: [(ObjectIdentifier, Int)] = []

    public init(
        layerIndex: Int, kind: CBv2LayerKind, pool: PagedKVPool,
        attentionSoftcap: Float? = nil
    ) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.pool = pool
        self.attentionSoftcap = attentionSoftcap
    }

    // MARK: - Rows

    public var rows: [CBv2SequenceKV] { pagedRows }

    /// Set the current batch rows (row order == batch row order). O(B);
    /// join = append a row object, leave = drop it — no storage moves.
    public func setRows(_ rows: [CBv2SequenceKV]) {
        precondition(
            kind.sharesKVWithLayer == nil || rows.isEmpty,
            "KV-shared layers own no rows; attention borrows via attendBorrowing")
        pagedRows = rows.map { row in
            guard let paged = row as? PagedSequenceKV else {
                fatalError("[PagedLayerCache] rows must be PagedSequenceKV from the same backend")
            }
            precondition(
                paged.groupKey == PagedKVGroupKey(kind),
                "row group \(paged.groupKey) does not match layer kind")
            return paged
        }
    }

    public var positionOffsets: MLXArray {
        MLXArray(pagedRows.map { Int32($0.absoluteOffset) })
    }

    // MARK: - Attention

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer == nil, "shared layers must call attendBorrowing")
        let b = queries.dim(0)
        let l = queries.dim(2)
        precondition(b == pagedRows.count, "queries batch \(b) != rows \(pagedRows.count)")

        if l == 1 {
            // Decode: write each row's K/V, then one kernel dispatch.
            for (i, row) in pagedRows.enumerated() {
                row.write(keys: keys[i], values: values[i])
            }
            return dispatchDecode(
                queries: queries, rows: pagedRows, scale: scale, sinks: sinks)
        } else {
            precondition(b == 1, "prefill chunks are per-request [1, chunk]")
            let row = pagedRows[0]
            row.write(keys: keys.squeezed(axis: 0), values: values.squeezed(axis: 0))
            return prefillAttend(
                queries: queries, row: row, scale: scale, sinks: sinks)
        }
    }

    public func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer != nil, "attendBorrowing requires a KV-shared layer")
        guard let src = source as? PagedLayerCache else {
            fatalError("[PagedLayerCache] can only borrow from another PagedLayerCache")
        }
        precondition(
            kind.attention == src.kind.attention,
            "KV-shared layer must share the source layer's attention type")
        let l = queries.dim(2)
        if l == 1 {
            return dispatchDecode(
                queries: queries, rows: src.pagedRows, scale: scale, sinks: sinks,
                tableProvider: src)
        } else {
            precondition(queries.dim(0) == 1 && src.pagedRows.count == 1)
            return prefillAttend(
                queries: queries, row: src.pagedRows[0], scale: scale, sinks: sinks)
        }
    }

    // MARK: - Decode path (paged kernel)

    private func dispatchDecode(
        queries: MLXArray, rows: [PagedSequenceKV], scale: Float, sinks: MLXArray?,
        tableProvider: PagedLayerCache? = nil
    ) -> MLXArray {
        let provider = tableProvider ?? self
        let group = pool.group(rows[0].groupKey)
        let tables = provider.deviceTables(rows: rows)

        var info = [Int32]()
        info.reserveCapacity(rows.count * 8)
        for row in rows {
            let (start, length) = row.decodeAttendRange
            info.append(Int32(start))
            info.append(Int32(length))
            info.append(Int32(row.table.count))
            info.append(contentsOf: [0, 0, 0, 0, 0])
        }
        let seqinfo = MLXArray(info, [rows.count, 8])

        let out = PagedAttentionKernel.decode(
            queries: queries,
            kSlab: group.kSlab,
            vSlab: group.vSlab,
            tables: tables,
            seqinfo: seqinfo,
            sinks: sinks,
            scale: scale,
            softcap: attentionSoftcap,
            pageSize: pool.config.pageSize
        )
        // [B, QH, D] -> [B, QH, 1, D], back in the model dtype.
        var result = out.expandedDimensions(axis: 2)
        if result.dtype != queries.dtype {
            result = result.asType(queries.dtype)
        }
        return result
    }

    /// Device `[B, maxPages]` int32 block tables, rebuilt only when some
    /// row's page table changed since the last dispatch.
    private func deviceTables(rows: [PagedSequenceKV]) -> MLXArray {
        let fingerprint = rows.map { (ObjectIdentifier($0), $0.tableVersion) }
        if let cached = cachedTables,
            fingerprint.count == cachedTablesFingerprint.count,
            zip(fingerprint, cachedTablesFingerprint).allSatisfy({ $0 == $1 })
        {
            return cached
        }
        // Pad to >= 8 columns so the kernel signature's address space is
        // stable across batch shapes (see PagedAttentionKernel).
        let maxPages = max(8, rows.map { $0.table.count }.max() ?? 0)
        var flat = [Int32](repeating: 0, count: rows.count * maxPages)
        for (i, row) in rows.enumerated() {
            flat.replaceSubrange(
                (i * maxPages) ..< (i * maxPages + row.table.count), with: row.table)
        }
        let tables = MLXArray(flat, [rows.count, maxPages])
        cachedTables = tables
        cachedTablesFingerprint = fingerprint
        return tables
    }

    // MARK: - Prefill path (per-request SDPA over gathered pages)

    private func prefillAttend(
        queries: MLXArray, row: PagedSequenceKV, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        let l = queries.dim(2)
        let end = row.absoluteOffset
        let retained = row.retainedCount
        let (rawK, rawV) = row.attendableViews()
        let k = rawK.dtype == queries.dtype ? rawK : rawK.asType(queries.dtype)
        let v = rawV.dtype == queries.dtype ? rawV : rawV.asType(queries.dtype)

        // Absolute positions: queries are the LAST l tokens written; keys
        // are the retained tail. Bool mask = causal AND in-window.
        let qStart = end - l
        let kStart = end - retained
        let qpos = MLXArray(Int32(qStart) ..< Int32(end)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(kStart) ..< Int32(end)).expandedDimensions(axis: 0)
        var mask = kpos .<= qpos
        if case .slidingWindow(let window) = kind.attention {
            mask = mask & (kpos .> (qpos - Int32(window)))
        }

        if let softcap = attentionSoftcap {
            // SDPA cannot express logit softcapping — composed path, pinned
            // for softcap configs.
            return PagedAttentionReference.composedAttention(
                queries: queries, keys: k, values: v, scale: scale,
                boolMask: mask, sinks: sinks, softcap: softcap)
        }

        // MLX SDPA requires the sink dtype to promote to the output dtype
        // (fp16 queries + fp32 sinks trap), so match the query dtype here.
        // The decode kernel is unaffected: it consumes sinks in fp32.
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: k, values: v, scale: scale,
            mask: .array(mask), sinks: sinks?.asType(queries.dtype))
    }
}
