// PagedAttentionKernel.swift
//
// Swift dispatch wrapper for the CBv2 paged-attention decode kernel
// (`pagedattention.metal`, shipped as a bundle resource). The kernel runs
// one threadgroup per (sequence, kv head) pair; see the .metal header for
// the full contract and attribution.
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
        return "cbv2_paged_attn_\(d)_d\(headDim)_s\(pageSize)_g\(gqa)"
            + "_n\(simdgroups)_sink\(hasSinks ? 1 : 0)_cap\(hasSoftcap ? 1 : 0)"
    }
}

/// Loads the MSL source for the paged-attention kernel from the package
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

    /// Body of the auto-generated kernel function. References the thread
    /// attributes and `_shape` helpers by name so MLX includes them in the
    /// generated signature (MLX scans the body source for these tokens).
    static let body: String = """
            const int kvh = kcache_shape[1];
            const int maxp = tables_shape[1];
            threadgroup float q_smem[GQA * D];
            threadgroup float red_smem[NSG * GQA * (D + 2)];
            cbv2::paged_attention_impl<T, D, S, GQA, NSG, HAS_SINKS, HAS_SOFTCAP>(
                q, kcache, vcache, tables, seqinfo, sinks, params,
                kvh, maxp, q_smem, red_smem, out,
                threadgroup_position_in_grid,
                thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
        """
}

/// Batched single-token (decode) paged attention.
public enum PagedAttentionKernel {

    /// Head dims with a validated shared-memory/register budget. Gemma-4
    /// global layers are 512; everything else in the product fleet is
    /// 64/128/256.
    public static let supportedHeadDims: Set<Int> = [64, 128, 256, 512]

    /// Threadgroup shared memory is capped at 32 KB by the Metal API; pick
    /// the largest simdgroup count whose staging + merge buffers fit.
    static func simdgroupsPerThreadgroup(headDim: Int, gqa: Int) -> Int {
        let limit = 32 * 1024
        for nsg in [4, 2, 1] {
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
            let k = MLXFast.metalKernel(
                name: key.kernelName,
                inputNames: ["q", "kcache", "vcache", "tables", "seqinfo", "sinks", "params"],
                outputNames: ["out"],
                source: PagedAttentionMSL.body,
                header: PagedAttentionMSL.header,
                ensureRowContiguous: true
            )
            kernels[key] = k
            return k
        }
    }

    private static let cache = KernelCache()

    static func kernel(for key: PagedAttentionKernelKey) -> MLXFast.MLXFastKernel {
        cache.kernel(for: key)
    }

    /// Dispatch decode attention for `B` rows.
    ///
    /// - Parameters:
    ///   - queries: `[B, queryHeads, 1, headDim]` or `[B, queryHeads, headDim]`,
    ///     any float dtype (converted to the slab dtype if needed).
    ///   - kSlab/vSlab: pool slabs `[P, kvHeads, pageSize, headDim]`.
    ///   - tables: `[B, maxPages]` int32, `maxPages >= 8`.
    ///   - seqinfo: `[B, 8]` int32 rows `{attendStart, attendLen, tableLen, 0…}`.
    ///   - sinks: optional per-query-head sink logits `[queryHeads]`.
    ///   - scale: query scale (applied inside the kernel in fp32).
    ///   - softcap: optional attention-logit soft cap.
    /// - Returns: `[B, queryHeads, headDim]` in the slab dtype.
    public static func decode(
        queries: MLXArray,
        kSlab: MLXArray,
        vSlab: MLXArray,
        tables: MLXArray,
        seqinfo: MLXArray,
        sinks: MLXArray?,
        scale: Float,
        softcap: Float? = nil,
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

        let gqa = queryHeads / kvHeads
        let nsg = simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa)

        if q.dtype != dtype { q = q.asType(dtype) }

        // Pad sinks to >= 8 elements so the generated signature keeps the
        // `device` address space regardless of head count.
        let sinksIn: MLXArray
        if let sinks {
            var s = sinks.asType(.float32).reshaped([-1])
            if s.dim(0) < 8 {
                s = concatenated([s, MLXArray.zeros([8 - s.dim(0)], dtype: .float32)])
            }
            sinksIn = s
        } else {
            sinksIn = MLXArray.zeros([8], dtype: .float32)
        }

        var paramValues: [Float] = [softcap ?? 1.0, scale]
        paramValues.append(contentsOf: [Float](repeating: 0, count: 8 - paramValues.count))
        let params = MLXArray(paramValues)

        let key = PagedAttentionKernelKey(
            dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: sinks != nil, hasSoftcap: softcap != nil)

        let tg = 32 * nsg
        let outputs = kernel(for: key)(
            [q, kSlab, vSlab, tables, seqinfo, sinksIn, params],
            template: [
                ("T", dtype),
                ("D", headDim),
                ("S", pageSize),
                ("GQA", gqa),
                ("NSG", nsg),
                ("HAS_SINKS", sinks != nil),
                ("HAS_SOFTCAP", softcap != nil),
            ],
            grid: (kvHeads * tg, b, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[b, queryHeads, headDim]],
            outputDTypes: [dtype],
            stream: stream
        )
        return outputs[0]
    }
}
