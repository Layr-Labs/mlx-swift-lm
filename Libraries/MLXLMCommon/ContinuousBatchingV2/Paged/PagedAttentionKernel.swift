// PagedAttentionKernel.swift
//
// Swift dispatch wrapper for the CBv2 paged-attention decode kernels
// (`pagedattention.metal`, shipped as a bundle resource). Decode is a
// two-pass flash-decoding dispatch: pass A computes unnormalized partials
// over PTOK-token partitions (one threadgroup per (sequence, kv head,
// partition)); pass B merges the partials per (sequence, query head) with
// sinks folded into the denominator. See the .metal header for the full
// contract and attribution.
//
// Kernel instances are JIT-compiled by MLX and cached both by MLX (per
// kernel name) and here (per configuration). Every distinct configuration
// gets a distinct kernel NAME: MLX's custom-kernel library cache is keyed
// by name and re-JITs whenever the generated source for a name changes, so
// sharing one name across dtypes/head-dims would thrash the pipeline cache.

import Foundation
import MLX
import MLXFast

/// Configuration key for one compiled paged-attention kernel variant.
struct PagedAttentionKernelKey: Hashable {
    enum Pass: Hashable {
        case part
        case merge
    }
    var pass: Pass
    var dtype: DType
    var headDim: Int
    var pageSize: Int
    var gqa: Int
    var simdgroups: Int
    var hasSinks: Bool
    var hasSoftcap: Bool

    var kernelName: String {
        let d: String
        switch dtype {
        case .float16: d = "f16"
        case .float32: d = "f32"
        case .bfloat16: d = "bf16"
        default: d = "dt\(dtype)"
        }
        let p = pass == .part ? "part" : "merge"
        return "cbv2_paged_\(p)_\(d)_d\(headDim)_s\(pageSize)_g\(gqa)"
            + "_n\(simdgroups)_sink\(hasSinks ? 1 : 0)_cap\(hasSoftcap ? 1 : 0)"
    }
}

/// Loads the MSL source for the paged-attention kernels from the package
/// bundle exactly once.
enum PagedAttentionMSL {
    static let header: String = {
        guard
            let url = Bundle.module.url(
                forResource: "pagedattention", withExtension: "metal")
        else {
            fatalError(
                "[PagedAttentionKernel] missing bundle resource pagedattention.metal "
                    + "— check Package.swift resources for MLXLMCommon")
        }
        guard let source = try? String(contentsOf: url, encoding: .utf8) else {
            fatalError("[PagedAttentionKernel] unable to read pagedattention.metal")
        }
        return source
    }()

    /// Bodies of the auto-generated kernel functions. They reference the
    /// thread attributes and `_shape` helpers by name so MLX includes them
    /// in the generated signature (MLX scans the body source for tokens).
    static let partBody: String = """
            const int kvh = kcache_shape[1];
            const int maxp = tables_shape[1];
            // grid z extent == the partial buffers' partition capacity
            // (output `_shape` params are not injected by MLX, only inputs').
            const int maxpart = threadgroups_per_grid.z;
            threadgroup float q_smem[GQA * D];
            threadgroup float red_smem[NSG * GQA * (D + 2)];
            cbv2::paged_attention_part_impl<T, D, S, GQA, NSG, PTOK, HAS_SOFTCAP>(
                q, kcache, vcache, tables, seqinfo, params,
                kvh, maxp, maxpart, q_smem, red_smem, partials, meta,
                threadgroup_position_in_grid,
                thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
        """

    static let mergeBody: String = """
            const int heads = partials_shape[1];
            const int maxpart = partials_shape[2];
            cbv2::paged_attention_merge_impl<T, D, PTOK, HAS_SINKS>(
                partials, meta, seqinfo, sinks, heads, maxpart, out,
                threadgroup_position_in_grid,
                thread_index_in_simdgroup);
        """
}

/// Batched single-token (decode) paged attention.
public enum PagedAttentionKernel {

    /// Head dims with a validated shared-memory/register budget. Gemma-4
    /// global layers are 512; everything else in the product fleet is
    /// 64/128/256.
    public static let supportedHeadDims: Set<Int> = [64, 128, 256, 512]

    /// Tokens per flash-decoding partition. Must be a multiple of the page
    /// size. 256 gives ceil(len/256) partitions per (row, kv head) —
    /// enough threadgroups to saturate the GPU at B=1 (the per-token online
    /// softmax is ALU-serialized within a simdgroup, so parallelism comes
    /// from partition count) without bloating the partial buffers.
    public static let partitionTokens = 256

    /// Threadgroup shared memory is capped at 32 KB by the Metal API; pick
    /// the largest simdgroup count whose staging + merge buffers fit.
    static func simdgroupsPerThreadgroup(headDim: Int, gqa: Int) -> Int {
        let limit = 32 * 1024
        for nsg in [8, 4, 2, 1] {
            let bytes = (gqa * headDim + nsg * gqa * (headDim + 2)) * 4
            if bytes <= limit { return nsg }
        }
        return 1
    }

    private final class KernelCache: @unchecked Sendable {
        private var kernels: [PagedAttentionKernelKey: MLXFast.MLXFastKernel] = [:]
        private let lock = NSLock()

        func kernel(for key: PagedAttentionKernelKey) -> MLXFast.MLXFastKernel {
            lock.lock()
            defer { lock.unlock() }
            if let k = kernels[key] { return k }
            let k: MLXFast.MLXFastKernel
            switch key.pass {
            case .part:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: ["q", "kcache", "vcache", "tables", "seqinfo", "params"],
                    outputNames: ["partials", "meta"],
                    source: PagedAttentionMSL.partBody,
                    header: PagedAttentionMSL.header,
                    ensureRowContiguous: true
                )
            case .merge:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: ["partials", "meta", "seqinfo", "sinks"],
                    outputNames: ["out"],
                    source: PagedAttentionMSL.mergeBody,
                    header: PagedAttentionMSL.header,
                    ensureRowContiguous: true
                )
            }
            kernels[key] = k
            return k
        }
    }

    private static let cache = KernelCache()

    static func kernel(for key: PagedAttentionKernelKey) -> MLXFast.MLXFastKernel {
        cache.kernel(for: key)
    }

    /// Shared dummy sinks (the generated signature keeps a `device` input
    /// even when HAS_SINKS is false). Read-only after creation; MLXArray is
    /// not Sendable but this one is never mutated (engine-thread discipline).
    nonisolated(unsafe) private static let zeroSinks: MLXArray = {
        let z = MLXArray.zeros([8], dtype: .float32)
        eval(z)
        return z
    }()

    /// Dispatch decode attention for `B` rows.
    ///
    /// - Parameters:
    ///   - queries: `[B, queryHeads, 1, headDim]` or `[B, queryHeads, headDim]`,
    ///     any float dtype (converted to the slab dtype if needed).
    ///   - kSlab/vSlab: pool slabs `[P, kvHeads, pageSize, headDim]`.
    ///   - tables: `[B, maxPages]` int32, `maxPages >= 8`.
    ///   - seqinfo: `[B, 8]` int32 rows `{attendStart, attendLen, tableLen, 0…}`.
    ///   - maxAttendLength: max over rows of the attended length (host-side
    ///     Swift Int — sizes the partial buffers, never a device sync).
    ///   - sinks: optional per-query-head sink logits `[queryHeads]`.
    ///   - params: `[8]` float32 `{softcap, scale, 0…}` (cache it per layer —
    ///     it is constant across steps).
    ///   - softcap: whether params[0] is an active softcap.
    /// - Returns: `[B, queryHeads, headDim]` in the slab dtype.
    public static func decode(
        queries: MLXArray,
        kSlab: MLXArray,
        vSlab: MLXArray,
        tables: MLXArray,
        seqinfo: MLXArray,
        maxAttendLength: Int,
        sinks: MLXArray?,
        params: MLXArray,
        softcap: Bool,
        pageSize: Int,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var q = queries
        if q.ndim == 4 {
            precondition(q.dim(2) == 1, "decode kernel requires L == 1")
            q = q.squeezed(axis: 2)
        }
        precondition(q.ndim == 3, "queries must be [B, QH, D]")

        let dtype = kSlab.dtype
        let b = q.dim(0)
        let queryHeads = q.dim(1)
        let headDim = q.dim(2)
        let kvHeads = kSlab.dim(1)
        precondition(kSlab.dim(3) == headDim && vSlab.dim(3) == headDim)
        precondition(kSlab.dim(2) == pageSize)
        precondition(queryHeads % kvHeads == 0, "GQA requires QH % KVH == 0")
        precondition(supportedHeadDims.contains(headDim), "unsupported head dim \(headDim)")
        precondition(tables.dim(0) == b && seqinfo.dim(0) == b)
        precondition(tables.dim(1) >= 8, "pad tables to >= 8 columns for a stable signature")
        precondition(partitionTokens % pageSize == 0, "PTOK must be a page multiple")
        precondition(maxAttendLength >= 1)

        let gqa = queryHeads / kvHeads
        let nsg = simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa)
        let maxParts = (maxAttendLength + partitionTokens - 1) / partitionTokens

        if q.dtype != dtype { q = q.asType(dtype) }

        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap)
        let tg = 32 * nsg
        let partOut = kernel(for: partKey)(
            [q, kSlab, vSlab, tables, seqinfo, params],
            template: [
                ("T", dtype),
                ("D", headDim),
                ("S", pageSize),
                ("GQA", gqa),
                ("NSG", nsg),
                ("PTOK", partitionTokens),
                ("HAS_SOFTCAP", softcap),
            ],
            grid: (kvHeads * tg, b, maxParts),
            threadGroup: (tg, 1, 1),
            outputShapes: [[b, queryHeads, maxParts, headDim], [b, queryHeads, maxParts, 2]],
            outputDTypes: [.float32, .float32],
            stream: stream
        )

        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false)
        let outputs = kernel(for: mergeKey)(
            [partOut[0], partOut[1], seqinfo, sinks ?? zeroSinks],
            template: [
                ("T", dtype),
                ("D", headDim),
                ("PTOK", partitionTokens),
                ("HAS_SINKS", sinks != nil),
            ],
            grid: (queryHeads * 32, b, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[b, queryHeads, headDim]],
            outputDTypes: [dtype],
            stream: stream
        )
        return outputs[0]
    }
}
