// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Tensor-unit sparse MLA prefill attention adapted from oMLX (Apache-2.0,
// © oMLX contributors), tag 33e0b0c:
//   omlx/patches/glm_moe_dsa/sparse_mla_nax.py (#3986, #3996).
// Rewritten for batched `[B,H,T,C]` absorbed queries, `[B,N,C]` latent
// histories and the -1-padded token indices of `GLM5NextSparseAttention`.

import Foundation
import MLX

/// Absorbed attention of `GLM5NextExecutionPolicy.SparsePrefill.nativeKernel`
/// for query blocks of any length over a contiguous latent history: per query
/// and head, `p = softmax(scale · qlat·c_j)` over the valid selected rows `j`
/// and `out = Σ p_j c_j`. Selected rows are read in-kernel through the index
/// list; no gathered latent copy or per-query partial buffer exists, so the
/// only transient is the `[B,H,T,C]` output. Default GPU stream only.
///
/// With M5 tensor units (NAX) and the GLM-5.3 geometry (C 512, heads a
/// multiple of 32, 16-bit queries and latents) one threadgroup per (query,
/// 32 heads) walks 128-slot tiles: 16-bit QK products accumulate in FP32,
/// an online exp2 softmax keeps FP32 maxima and sums, and each unnormalized
/// probability enters PV as an FP16 hi + lo pair (within 2^-24 of its FP32
/// value) against the 16-bit latents with FP32 accumulation; tiles without a
/// valid slot are skipped. 16-bit products are exact, so only the
/// accumulation order and the probability rounding differ from the gathered
/// reference: that reference rounds each NORMALIZED probability to the
/// activation dtype before PV, an online softmax normalizes once after PV and
/// so keeps FP32-accurate probabilities (not bitwise; the gap is the
/// reference's own probability rounding). Other devices and geometries run
/// `GLM5NextSparseDecodeKernel.attend` over query blocks whose FP32 split
/// partials stay within `portablePartialBytes`.
enum GLM5NextSparsePrefillKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry
        case unsupportedDeviceOrStream
    }

    static let latentDimension = 512
    static let headGroup = 32
    static let threads = 256
    /// Bound on one portable launch's FP32 split partials.
    static let portablePartialBytes = 64 << 20

    static var tensorUnitsAvailable: Bool { glm5NextNAXAvailable }

    /// Whether `attend` runs the tensor-unit kernel for these operands.
    static func usesTensorUnits(queryLatents: MLXArray, latents: MLXArray) -> Bool {
        tensorUnitsAvailable && queryLatents.dim(3) == latentDimension
            && queryLatents.dim(1) % headGroup == 0
            && [.bfloat16, .float16].contains(queryLatents.dtype)
            && [.bfloat16, .float16].contains(latents.dtype)
    }

    /// Absorbed queries `[B,H,T,C]`, latent history `[B,N,C]` and `[B,T,W]`
    /// int32 token indices (-1, or any index outside `0..<N`, is skipped).
    /// Returns `[B,H,T,C]` attended latents in `outputType` (zero for a query
    /// with no valid index); `scale` multiplies each raw score. The
    /// tensor-unit path needs 16-bit queries and latents; callers narrow a
    /// losslessly widened FP32 history first.
    static func attend(
        queryLatents: MLXArray, latents: MLXArray, indices: MLXArray, scale: Float,
        outputType: DType
    ) throws -> MLXArray {
        try attend(
            queryLatents: queryLatents, latents: latents, indices: indices, scale: scale,
            outputType: outputType, tensorUnits: true)
    }

    /// `tensorUnits: false` forces the portable split-K path.
    static func attend(
        queryLatents: MLXArray, latents: MLXArray, indices: MLXArray, scale: Float,
        outputType: DType, tensorUnits: Bool
    ) throws -> MLXArray {
        let (batch, heads, count, dimension) = (
            queryLatents.dim(0), queryLatents.dim(1), queryLatents.dim(2), queryLatents.dim(3)
        )
        guard queryLatents.ndim == 4, latents.ndim == 3, indices.ndim == 3,
            latents.dim(0) == batch, latents.dim(1) > 0, latents.dim(2) == dimension,
            indices.dim(0) == batch, indices.dim(1) == count, indices.dim(2) > 0,
            indices.dtype == .int32, batch > 0, heads > 0, count > 0, dimension > 0,
            [.float16, .bfloat16, .float32].contains(outputType),
            [.float16, .bfloat16, .float32].contains(latents.dtype),
            batch * heads * count * dimension <= Int(Int32.max),
            batch * latents.dim(1) * dimension <= Int(Int32.max)
        else { throw Failure.invalidGeometry }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        if tensorUnits, usesTensorUnits(queryLatents: queryLatents, latents: latents) {
            #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
                return tensorKernel(
                    [queryLatents, latents, indices, MLXArray(scale)],
                    template: [
                        ("QT", queryLatents.dtype), ("KT", latents.dtype), ("OT", outputType),
                    ],
                    grid: (threads * count, heads / headGroup, batch),
                    threadGroup: (threads, 1, 1),
                    outputShapes: [[batch, heads, count, dimension]],
                    outputDTypes: [outputType])[0]
            #else
                throw Failure.unsupportedDeviceOrStream
            #endif
        }
        let perQuery = GLM5NextSparseDecodeKernel.attentionPartialBytes(
            rows: batch, width: indices.dim(2), heads: heads, latentDimension: dimension)
        let block = max(1, portablePartialBytes / perQuery)
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: count, by: block) {
            let queries = start ..< min(start + block, count)
            parts.append(
                try GLM5NextSparseDecodeKernel.attend(
                    queryLatents: queryLatents[0..., 0..., queries], latents: latents,
                    indices: indices[0..., queries], scale: scale, outputType: outputType))
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// One threadgroup per (query, 32-head group, batch row), 8
        /// SIMD-groups. Per 128-slot tile, SIMD-group (hg, j4) scores heads
        /// hg*16..+16 against key quarter j4 over the whole latent, exchanges
        /// the four partial row maxima once, writes exp2 and the FP16 hi/lo
        /// split of its 16 heads x 32 keys in the PV operand fragment layout,
        /// then multiplies the tile's probabilities with latent columns
        /// j4*128..+128. Fragment layout (MLX BaseNAXFrag, 16x16): lane ->
        /// rows fm, fm + 8 and columns fn .. fn + 3. Every load's result is
        /// used: in-flight loads into dead registers made a software-pipelined
        /// variant nondeterministic on M5 (oMLX).
        private static let tensorKernel = MLXFast.metalKernel(
            name: "glm5_next_sparse_prefill_attention",
            inputNames: ["queries", "latents", "ids", "scale"],
            outputNames: ["attended"],
            source: """
                constexpr int D = 512;
                constexpr int BK = 128;
                const int H = queries_shape[1];
                const int L = queries_shape[2];
                const int N = latents_shape[1];
                const int W = ids_shape[2];
                const float scale_log2 = scale * 1.44269504088896341f;
                const int qi = int(threadgroup_position_in_grid.x);
                const int hh = int(threadgroup_position_in_grid.y);
                const int b = int(threadgroup_position_in_grid.z);
                const uint sg = simdgroup_index_in_threadgroup;
                const uint lane = thread_index_in_simdgroup;
                const uint tid = sg * 32 + lane;
                const int hg = int(sg) / 4;
                const int j4 = int(sg) % 4;   // QK + softmax: key quarter; PV: latent quarter
                const int head0 = hh * 32 + hg * 16;

                const short qid = short(lane >> 2);
                const short fm = short((qid & 4) | ((lane >> 1) & 3));
                const short fn = short(((qid & 2) | (lane & 1)) * 4);
                // One lane per fragment row (fm) after the xor-1 / xor-8 reductions.
                const bool row_writer = (lane & 9u) == 0u;

                threadgroup int sel[2][BK];
                threadgroup int live[2][BK / 32];
                threadgroup float red_max[2][4][16];
                threadgroup float red_sum[2][4][16];
                // Per (head group, 16-key step, lane): the lane's 8 fragment
                // values (rows fm, fm + 8 x keys fn .. fn + 3).
                threadgroup half p_hi[2][BK / 16][32 * 8];
                threadgroup half p_lo[2][BK / 16][32 * 8];

                constexpr auto qk_desc = matmul2d_descriptor(
                    16, 32, 16, false, true, true,
                    matmul2d_descriptor::mode::multiply_accumulate);
                matmul2d<qk_desc, execution_simdgroup> qk_op;
                constexpr auto pv_desc = matmul2d_descriptor(
                    16, 32, 16, false, false, true,
                    matmul2d_descriptor::mode::multiply_accumulate);
                matmul2d<pv_desc, execution_simdgroup> pv_op;

                auto pa = pv_op.template get_left_input_cooperative_tensor<half, KT, float>();
                auto pm = pv_op.template get_left_input_cooperative_tensor<half, KT, float>();
                auto pb = pv_op.template get_right_input_cooperative_tensor<half, KT, float>();
                auto o0 = pv_op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
                auto o1 = pv_op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
                auto o2 = pv_op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
                auto o3 = pv_op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
                for (short e = 0; e < 16; ++e) {
                    o0[e] = 0.0f;
                    o1[e] = 0.0f;
                    o2[e] = 0.0f;
                    o3[e] = 0.0f;
                }
                float m_run[2] = {-FLT_MAX, -FLT_MAX};
                float l_run[2] = {0.0f, 0.0f};

                const device QT* qr0 = queries + ((ulong(b) * H + head0 + fm) * L + qi) * D + fn;
                const device QT* qr1 = queries + ((ulong(b) * H + head0 + fm + 8) * L + qi) * D + fn;
                const device int32_t* idx_row = ids + (ulong(b) * L + qi) * W;
                const device KT* kv = latents + ulong(b) * N * D;

                auto stage = [&](int t, int buffer) {
                    if (tid < uint(BK)) {
                        const int slot = t * BK + int(tid);
                        int kp = slot < W ? int(idx_row[slot]) : -1;
                        if (kp < 0 || kp >= N) {
                            kp = -1;
                        }
                        sel[buffer][tid] = kp;
                        const bool any_live = simd_any(kp >= 0);
                        if (lane == 0) {
                            live[buffer][sg] = any_live ? 1 : 0;
                        }
                    }
                };

                const int n_tiles = (W + BK - 1) / BK;
                stage(0, 0);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int t = 0; t < n_tiles; ++t) {
                    const int buf = t & 1;
                    const int tile_keys = min(BK, W - t * BK);
                    // Uniform across the threadgroup: skip tiles with no valid slot.
                    if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (t + 1 < n_tiles) {
                            stage(t + 1, buf ^ 1);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        continue;
                    }

                    // ---- S = Q K^T for key quarter j4 (16 heads x 32 keys).
                    auto qa = qk_op.template get_left_input_cooperative_tensor<QT, KT, float>();
                    auto kb = qk_op.template get_right_input_cooperative_tensor<QT, KT, float>();
                    auto sc = qk_op.template get_destination_cooperative_tensor<
                        metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
                    for (short e = 0; e < 16; ++e) {
                        sc[e] = 0.0f;
                    }
                    if (j4 * 32 < tile_keys) {
                        const device KT* kr[2][2];
                        for (short tn = 0; tn < 2; ++tn) {
                            for (short i = 0; i < 2; ++i) {
                                const int kp = sel[buf][j4 * 32 + tn * 16 + fm + i * 8];
                                kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;
                            }
                        }
                        _Pragma("clang loop unroll_count(4)")
                        for (short kk = 0; kk < D; kk += 16) {
                            for (short j = 0; j < 4; ++j) {
                                qa[j] = qr0[kk + j];
                                qa[4 + j] = qr1[kk + j];
                            }
                            for (short tn = 0; tn < 2; ++tn) {
                                for (short i = 0; i < 2; ++i) {
                                    for (short j = 0; j < 4; ++j) {
                                        kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                                    }
                                }
                            }
                            qk_op.run(qa, kb, sc);
                        }
                    }
                    bool ok[2][4];
                    for (short tn = 0; tn < 2; ++tn) {
                        for (short j = 0; j < 4; ++j) {
                            ok[tn][j] = sel[buf][j4 * 32 + tn * 16 + fn + j] >= 0;
                        }
                    }
                    // Partial row max of the raw scores (scale > 0 and rounding
                    // is monotonic: max(s) * c == max(s * c) bitwise).
                    float pmx[2];
                    for (short i = 0; i < 2; ++i) {
                        float m = -FLT_MAX;
                        for (short tn = 0; tn < 2; ++tn) {
                            for (short j = 0; j < 4; ++j) {
                                m = ok[tn][j] ? max(m, float(sc[tn * 8 + i * 4 + j])) : m;
                            }
                        }
                        m = max(m, simd_shuffle_xor(m, ushort(1)));
                        m = max(m, simd_shuffle_xor(m, ushort(8)));
                        pmx[i] = m;
                    }
                    if (row_writer) {
                        red_max[hg][j4][fm] = pmx[0];
                        red_max[hg][j4][fm + 8] = pmx[1];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (t + 1 < n_tiles) {
                        stage(t + 1, buf ^ 1);
                    }

                    // ---- Tile row max (the same value in the group's 4 SIMD-groups).
                    float factor[2];
                    float mnew[2];
                    for (short i = 0; i < 2; ++i) {
                        const int r = fm + i * 8;
                        const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]),
                                            max(red_max[hg][2][r], red_max[hg][3][r]));
                        const float cand = m == -FLT_MAX ? -FLT_MAX : m * scale_log2;
                        mnew[i] = max(m_run[i], cand);
                        factor[i] = fast::exp2(m_run[i] - mnew[i]);
                        m_run[i] = mnew[i];
                    }
                    // ---- P of this quarter: exp2, FP16 hi + lo pieces, row sums.
                    float rs[2] = {0.0f, 0.0f};
                    for (short tn = 0; tn < 2; ++tn) {
                        vec<half, 8> h, lo;
                        for (short i = 0; i < 2; ++i) {
                            for (short j = 0; j < 4; ++j) {
                                const float e = ok[tn][j]
                                    ? fast::exp2(float(sc[tn * 8 + i * 4 + j]) * scale_log2 - mnew[i])
                                    : 0.0f;
                                const half hi = half(e);
                                h[i * 4 + j] = hi;
                                lo[i * 4 + j] = half(e - float(hi));
                                rs[i] += e;
                            }
                        }
                        const int ks = j4 * 2 + tn;
                        *(threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]) = h;
                        *(threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]) = lo;
                    }
                    for (short i = 0; i < 2; ++i) {
                        rs[i] += simd_shuffle_xor(rs[i], ushort(1));
                        rs[i] += simd_shuffle_xor(rs[i], ushort(8));
                    }
                    if (row_writer) {
                        red_sum[hg][j4][fm] = rs[0];
                        red_sum[hg][j4][fm + 8] = rs[1];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // ---- O = O * factor + P V for latent quarter j4 over the tile.
                    if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
                        for (short e = 0; e < 16; ++e) {
                            const float f = factor[(e >> 2) & 1];
                            o0[e] *= f;
                            o1[e] *= f;
                            o2[e] *= f;
                            o3[e] *= f;
                        }
                    }
                    for (short i = 0; i < 2; ++i) {
                        const int r = fm + i * 8;
                        const float tsum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r])
                            + red_sum[hg][3][r];
                        l_run[i] = l_run[i] * factor[i] + tsum;
                    }
                    const int n_ks = (tile_keys + 15) / 16;
                    for (short ks = 0; ks < n_ks; ++ks) {
                        const vec<half, 8> h = *(const threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]);
                        const vec<half, 8> lo = *(const threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]);
                        for (short e = 0; e < 8; ++e) {
                            pa[e] = h[e];
                            pm[e] = lo[e];
                        }
                        const int kp0 = sel[buf][ks * 16 + fm];
                        const int kp1 = sel[buf][ks * 16 + fm + 8];
                        const device KT* v0 = kv + ulong(max(kp0, 0)) * D + j4 * 128 + fn;
                        const device KT* v1 = kv + ulong(max(kp1, 0)) * D + j4 * 128 + fn;
                        for (short np = 0; np < 4; ++np) {
                            for (short tn = 0; tn < 2; ++tn) {
                                for (short j = 0; j < 4; ++j) {
                                    pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                                    pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                                }
                            }
                            if (np == 0) {
                                pv_op.run(pa, pb, o0);
                                pv_op.run(pm, pb, o0);
                            } else if (np == 1) {
                                pv_op.run(pa, pb, o1);
                                pv_op.run(pm, pb, o1);
                            } else if (np == 2) {
                                pv_op.run(pa, pb, o2);
                                pv_op.run(pm, pb, o2);
                            } else {
                                pv_op.run(pa, pb, o3);
                                pv_op.run(pm, pb, o3);
                            }
                        }
                    }
                }

                for (short i = 0; i < 2; ++i) {
                    device OT* orow = attended
                        + ((ulong(b) * H + head0 + fm + i * 8) * L + qi) * D + j4 * 128 + fn;
                    const float denom = l_run[i] > 0.0f ? l_run[i] : 1.0f;
                    for (short tn = 0; tn < 2; ++tn) {
                        for (short j = 0; j < 4; ++j) {
                            const short e = tn * 8 + i * 4 + j;
                            orow[tn * 16 + j] = static_cast<OT>(o0[e] / denom);
                            orow[32 + tn * 16 + j] = static_cast<OT>(o1[e] / denom);
                            orow[64 + tn * 16 + j] = static_cast<OT>(o2[e] / denom);
                            orow[96 + tn * 16 + j] = static_cast<OT>(o3[e] / denom);
                        }
                    }
                }
                """,
            header: """
                #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
                using namespace mpp::tensor_ops;
                """,
            ensureRowContiguous: true)
    #endif
}
