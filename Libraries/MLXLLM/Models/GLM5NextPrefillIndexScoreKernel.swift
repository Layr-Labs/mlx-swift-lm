// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Tensor-unit DSA indexer scores adapted from oMLX (Apache-2.0, © oMLX
// contributors), tag 33e0b0c: omlx/patches/glm_moe_dsa/indexer_nax.py.
// Rewritten for batched `[B,Q,H,D]` index queries, unscaled head weights,
// FP32 output and the per-row pool anchors of `GLM5NextSparseAttention`.

import Foundation
import MLX

/// Masked pool scores of `GLM5NextSparseAttention`'s blocked selection under
/// `GLM5NextExecutionPolicy.SparsePrefill.nativeKernel`:
///
///     score[b,q,p] = Σ_h relu((q[b,q,h]·k[b,p]) · indexScale) · (w[b,q,h] · headScale)
///
/// for eligible pools (complete and ending at or before the query's raw
/// position), the finite FP32 minimum elsewhere. One launch writes the
/// `[B,Q,P]` FP32 scores directly; no `[B,Q,H,P]` per-head tensor exists.
/// Each SIMD-group owns a 32 x 32 (query x pool) block and runs every head's
/// 16-bit product on the tensor units with FP32 accumulation (exact
/// products, so only summation order differs from the FP32 GEMM), applies
/// the reference's FP32 scale, ReLU and weight products, and sums heads in
/// order. Blocks whose first pool ends after the block's last query are
/// written masked without any product. The head sum order differs from the
/// reference reduction, so a near-tie pool can change, as with the other
/// blocked score shapes. Runs only where `supports` holds (default GPU
/// stream, M5 tensor units, 16-bit queries and keys, 128-wide index heads);
/// callers keep the score graph elsewhere.
enum GLM5NextPrefillIndexScoreKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry
        case unsupportedDeviceOrStream
    }

    static let headDimension = 128
    /// Query x pool blocks of a threadgroup, in 32 x 32 SIMD-group blocks.
    static let blockRows = 2
    static let blockColumns = 2

    static func supports(indexQueries: MLXArray, pooledKeys: MLXArray) -> Bool {
        glm5NextOnDefaultGPUStream() && GLM5NextSparsePrefillKernel.tensorUnitsAvailable
            && indexQueries.ndim == 4 && indexQueries.dim(3) == headDimension
            && [.bfloat16, .float16].contains(indexQueries.dtype)
            && pooledKeys.dtype == indexQueries.dtype
    }

    /// Queries `[B,Q,H,D]`, unscaled head weights `[B,Q,H]`, pooled keys
    /// `[B,P,D]`, complete-pool mask `[B,P]` and each pool's last raw key
    /// position `[B,P]` (int32). Query i sits at raw position `queryStart + i`.
    static func scores(
        indexQueries: MLXArray, headWeights: MLXArray, pooledKeys: MLXArray,
        completePools: MLXArray, poolEnds: MLXArray, queryStart: Int
    ) throws -> MLXArray {
        let (batch, queries, heads, dimension) = (
            indexQueries.dim(0), indexQueries.dim(1), indexQueries.dim(2), indexQueries.dim(3)
        )
        let pools = pooledKeys.dim(1)
        guard indexQueries.ndim == 4, pooledKeys.shape == [batch, pools, dimension],
            headWeights.shape == [batch, queries, heads], completePools.shape == [batch, pools],
            completePools.dtype == .bool, poolEnds.shape == [batch, pools],
            batch > 0, queries > 0, heads > 0, pools > 0, queryStart >= 0,
            batch * queries * pools <= Int(Int32.max),
            [.bfloat16, .float16, .float32].contains(headWeights.dtype)
        else { throw Failure.invalidGeometry }
        guard supports(indexQueries: indexQueries, pooledKeys: pooledKeys) else {
            throw Failure.unsupportedDeviceOrStream
        }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let columns = (pools + 32 * blockColumns - 1) / (32 * blockColumns)
            let rows = (queries + 32 * blockRows - 1) / (32 * blockRows)
            return kernel(
                [
                    indexQueries, pooledKeys, headWeights, completePools,
                    poolEnds.asType(.int32), MLXArray(Int32(queryStart)),
                    MLXArray([1 / sqrt(Float(dimension)), 1 / sqrt(Float(heads))]),
                ],
                template: [
                    ("T", indexQueries.dtype), ("WM", blockRows), ("WN", blockColumns),
                ],
                grid: (columns * blockRows * blockColumns * 32, rows, batch),
                threadGroup: (blockRows * blockColumns * 32, 1, 1),
                outputShapes: [[batch, queries, pools]], outputDTypes: [.float32])[0]
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// Fragment layout (MLX BaseNAXFrag, 16x16): lane -> rows fm, fm + 8
        /// and columns fn .. fn + 3 of each fragment.
        private static let kernel = MLXFast.metalKernel(
            name: "glm5_next_prefill_index_scores",
            inputNames: ["q", "k", "w", "complete", "ends", "query_start", "scales"],
            outputNames: ["out"],
            source: """
                constexpr int D = 128;
                const int S = q_shape[1];
                const int H = q_shape[2];
                const int P = k_shape[1];
                const int b = int(threadgroup_position_in_grid.z);
                const uint sg = simdgroup_index_in_threadgroup;
                const uint lane = thread_index_in_simdgroup;
                const int sm = int(threadgroup_position_in_grid.y) * (32 * WM) + int(sg / WN) * 32;
                const int sn = int(threadgroup_position_in_grid.x) * (32 * WN) + int(sg % WN) * 32;
                const float index_scale = scales[0];
                const float head_scale = scales[1];
                const device T* qb = q + ulong(b) * S * H * D;
                const device T* kbatch = k + ulong(b) * P * D;
                const device bool* cb = complete + ulong(b) * P;
                const device int* eb = ends + ulong(b) * P;

                const short qid = short(lane >> 2);
                const short fm = short((qid & 4) | ((lane >> 1) & 3));
                const short fn = short(((qid & 2) | (lane & 1)) * 4);

                // Pool ends ascend with the pool index: the whole block is
                // masked when its first pool ends after the block's last query.
                const int s_last = min(sm + 31, S - 1);
                const bool live = sm < S && sn < P && eb[sn] <= query_start + s_last;

                float acc[2][16];
                for (short a = 0; a < 2; ++a) {
                    for (short e = 0; e < 16; ++e) {
                        acc[a][e] = 0.0f;
                    }
                }
                int qrow[2][2];
                for (short tm = 0; tm < 2; ++tm) {
                    for (short i = 0; i < 2; ++i) {
                        qrow[tm][i] = min(sm + tm * 16 + fm + i * 8, S - 1);
                    }
                }

                if (live) {
                    constexpr auto desc = matmul2d_descriptor(
                        16, 32, 16, false, true, true,
                        matmul2d_descriptor::mode::multiply_accumulate);
                    matmul2d<desc, execution_simdgroup> op;
                    const device T* kbase[2][2];
                    for (short tn = 0; tn < 2; ++tn) {
                        for (short i = 0; i < 2; ++i) {
                            kbase[tn][i] = kbatch + ulong(min(sn + tn * 16 + fm + i * 8, P - 1)) * D + fn;
                        }
                    }
                    for (int h = 0; h < H; ++h) {
                        auto ct_a0 = op.template get_left_input_cooperative_tensor<T, T, float>();
                        auto ct_a1 = op.template get_left_input_cooperative_tensor<T, T, float>();
                        auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
                        auto c0 = op.template get_destination_cooperative_tensor<
                            metal::remove_addrspace_t<decltype(ct_a0)>,
                            metal::remove_addrspace_t<decltype(ct_b)>, float>();
                        auto c1 = op.template get_destination_cooperative_tensor<
                            metal::remove_addrspace_t<decltype(ct_a0)>,
                            metal::remove_addrspace_t<decltype(ct_b)>, float>();
                        for (short e = 0; e < 16; ++e) {
                            c0[e] = 0.0f;
                            c1[e] = 0.0f;
                        }
                        const device T* a00 = qb + (ulong(qrow[0][0]) * H + h) * D + fn;
                        const device T* a01 = qb + (ulong(qrow[0][1]) * H + h) * D + fn;
                        const device T* a10 = qb + (ulong(qrow[1][0]) * H + h) * D + fn;
                        const device T* a11 = qb + (ulong(qrow[1][1]) * H + h) * D + fn;
                        for (short kk = 0; kk < D; kk += 16) {
                            for (short j = 0; j < 4; ++j) {
                                ct_a0[j] = a00[kk + j];
                                ct_a0[4 + j] = a01[kk + j];
                                ct_a1[j] = a10[kk + j];
                                ct_a1[4 + j] = a11[kk + j];
                            }
                            for (short tn = 0; tn < 2; ++tn) {
                                for (short i = 0; i < 2; ++i) {
                                    for (short j = 0; j < 4; ++j) {
                                        ct_b[tn * 8 + i * 4 + j] = kbase[tn][i][kk + j];
                                    }
                                }
                            }
                            op.run(ct_a0, ct_b, c0);
                            op.run(ct_a1, ct_b, c1);
                        }
                        // The reference's FP32 roundings: scaled score, ReLU,
                        // scaled weight and their product; heads summed in order.
                        for (short i = 0; i < 2; ++i) {
                            const float w0 = float(w[(ulong(b) * S + qrow[0][i]) * H + h]) * head_scale;
                            const float w1 = float(w[(ulong(b) * S + qrow[1][i]) * H + h]) * head_scale;
                            for (short tn = 0; tn < 2; ++tn) {
                                for (short j = 0; j < 4; ++j) {
                                    const short e = tn * 8 + i * 4 + j;
                                    acc[0][e] += max(float(c0[e]) * index_scale, 0.0f) * w0;
                                    acc[1][e] += max(float(c1[e]) * index_scale, 0.0f) * w1;
                                }
                            }
                        }
                    }
                }

                const float masked = -3.402823466e+38f;
                for (short tm = 0; tm < 2; ++tm) {
                    for (short i = 0; i < 2; ++i) {
                        const int s = sm + tm * 16 + fm + i * 8;
                        if (s >= S) {
                            continue;
                        }
                        device float* orow = out + (ulong(b) * S + s) * P;
                        for (short tn = 0; tn < 2; ++tn) {
                            for (short j = 0; j < 4; ++j) {
                                const int p = sn + tn * 16 + fn + j;
                                if (p >= P) {
                                    continue;
                                }
                                const bool valid = live && cb[p] && eb[p] <= query_start + s;
                                orow[p] = valid ? acc[tm][tn * 8 + i * 4 + j] : masked;
                            }
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
