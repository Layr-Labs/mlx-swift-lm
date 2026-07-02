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
//
// PERFORMANCE NOTE (2026-07, real-weights re-baseline): the kernel is
// numerically CORRECT but UNOPTIMIZED on real-model shapes — it was
// validated on synthetic parity only. On GPT-OSS-20B real weights the
// paged decode path runs ~100x slower than the contiguous backend
// (~2.0 s/token, all GPU-side); requests correctly error via the engine's
// 30 s step watchdog (see benchmarks/reports/gptoss-20b-mxfp4q8-main.md).
// The paged backend remains NON-DEFAULT and gated behind explicit
// construction; kernel optimization is tracked as follow-up work. Known
// issue to investigate alongside the perf work: a teardown SIGSEGV was
// observed after watchdog-errored runs (stack captured during the
// benchmarks/reports GPT-OSS runs).

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

    /// Head dims with a validated shared-memory/register budget. Necessary
    /// but NOT sufficient for eligibility: the threadgroup-memory budget
    /// also depends on the GQA factor (see `ineligibilityReason`) — e.g.
    /// Gemma-4 global layers (headDim 512, GQA 8) are over budget even at
    /// one simdgroup. Everything else in the product fleet is 64/128/256.
    public static let supportedHeadDims: Set<Int> = [64, 128, 256, 512]

    /// Tokens per flash-decoding partition. Must be a multiple of the page
    /// size. 256 gives ceil(len/256) partitions per (row, kv head) —
    /// enough threadgroups to saturate the GPU at B=1 (the per-token online
    /// softmax is ALU-serialized within a simdgroup, so parallelism comes
    /// from partition count) without bloating the partial buffers.
    public static let partitionTokens = 256

    // MARK: - Threadgroup-memory budget (single source of truth)
    //
    // These constants mirror the part kernel's threadgroup allocations —
    // `PagedAttentionMSL.partBody` above plus the RSTRIDE layout in
    // pagedattention.metal (which carries a matching keep-in-sync comment):
    //
    //   threadgroup float q_smem[GQA * D];               // staged queries
    //   threadgroup float red_smem[NSG * GQA * (D + 2)]; // merge records:
    //                                                    // acc[D], m, l
    //
    // Both buffers are float32 regardless of the slab dtype T (K/V rows are
    // converted to float on load), the merge pass allocates NO threadgroup
    // memory, and neither the HAS_SOFTCAP nor the HAS_SINKS variant adds
    // any — so the budget is a function of (headDim, gqa, simdgroups)
    // alone. `CBv2PagedEligibilityTests` asserts the generated bodies still
    // match this model.

    /// Metal's per-threadgroup memory cap (`setThreadgroupMemoryLength`
    /// limit on Apple GPUs). A dispatch over this limit is an UNCATCHABLE
    /// process fatal ("Threadgroup memory size (...) exceeds the maximum
    /// (32768)"), not a thrown error — eligibility must be refused
    /// statically, before any dispatch.
    public static let threadgroupMemoryLimit = 32 * 1024

    /// Per-(simdgroup, query head) merge record trailer: the shader's
    /// RSTRIDE is `D + 2` floats — acc[D] plus (m, l).
    public static let mergeRecordMetaFloats = 2

    /// Candidate simdgroup counts for the part kernel, largest first.
    static let simdgroupCandidates = [8, 4, 2, 1]

    /// Exact threadgroup bytes the part kernel allocates for one
    /// threadgroup: (q_smem + red_smem) float32 elements.
    public static func partThreadgroupBytes(
        headDim: Int, gqa: Int, simdgroups: Int
    ) -> Int {
        let qSmem = gqa * headDim
        let redSmem = simdgroups * gqa * (headDim + mergeRecordMetaFloats)
        return (qSmem + redSmem) * MemoryLayout<Float32>.size
    }

    /// Largest simdgroup count whose staging + merge buffers fit the Metal
    /// threadgroup-memory cap, or nil when even NSG=1 exceeds it — the
    /// configuration is statically ineligible for the paged kernels.
    static func simdgroupsPerThreadgroup(headDim: Int, gqa: Int) -> Int? {
        simdgroupCandidates.first {
            partThreadgroupBytes(headDim: headDim, gqa: gqa, simdgroups: $0)
                <= threadgroupMemoryLimit
        }
    }

    /// Static eligibility of one attention shape for the paged decode
    /// kernels: nil when eligible, else a human-readable reason. Callers
    /// building paged state (`PagedKVBackend.init`) MUST refuse ineligible
    /// shapes before any dispatch (see `threadgroupMemoryLimit`).
    public static func ineligibilityReason(headDim: Int, gqa: Int) -> String? {
        guard supportedHeadDims.contains(headDim) else {
            return "paged kernel does not support headDim \(headDim); "
                + "supported: \(supportedHeadDims.sorted())"
        }
        guard gqa >= 1 else {
            return "invalid GQA factor \(gqa)"
        }
        guard simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) != nil else {
            let bytes = partThreadgroupBytes(headDim: headDim, gqa: gqa, simdgroups: 1)
            return "headDim \(headDim) with GQA \(gqa) needs \(bytes) B of "
                + "threadgroup memory even at 1 simdgroup, over the "
                + "\(threadgroupMemoryLimit) B Metal limit — the paged part "
                + "kernel would trap at first dispatch"
        }
        return nil
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
        guard let nsg = simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) else {
            preconditionFailure(
                "paged decode dispatched for a statically ineligible shape: "
                    + (ineligibilityReason(headDim: headDim, gqa: gqa)
                        ?? "headDim \(headDim), GQA \(gqa)"))
        }
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
