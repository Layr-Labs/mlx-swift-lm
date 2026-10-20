// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Small-query pool scores and split-K selected-latent attention adapted from
// oMLX (Apache-2.0, © oMLX contributors), tag 33e0b0c:
//   omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/
//   small_indexer.py (small_scores) and sparse_mla_decode.py (split-K + merge).
// Rewritten for the pinned GLM5Next pooling/selection semantics of
// `GLM5NextSparseAttention.selectTokens` and portable simdgroup matrices.

import Foundation
import MLX

/// Native kernels of `GLM5NextExecutionPolicy.SparseDecode.nativeKernels` for
/// steps of at most `maximumQueries` sparse queries. Default GPU stream only.
///
/// - `poolScores`: one launch scoring every pool for every query directly from
///   raw index keys, pool gates and key validity. Pooling keeps the
///   reference's roundings (MLX softmax over the four gate logits, weights
///   rounded to the pooling dtype, rounded products, sequential rounded sum);
///   the FP32 head dot products and head sum use this kernel's order.
/// - `expand`: one launch turning selector output into `selectTokens`' token
///   indices (ascending pools, eligibility, -1 padding, deduplicated tail,
///   query validity). Bit-identical for the same chosen pools.
/// - `attend`: absorbed attention over the selected latents read in-kernel
///   through the index list (no gathered copy), split over 64-key blocks with
///   FP32 partial max/sum/PV and a log-sum-exp merge. The reference rounds
///   each normalized probability to the activation dtype before PV; a split
///   merge normalizes after PV, so probabilities stay FP32 (not bitwise).
enum GLM5NextSparseDecodeKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry
        case unsupportedDeviceOrStream
    }

    static let maximumQueries = 8
    static let maximumIndexDimension = 128
    static let poolTile = 32
    static let scoreThreads = 256
    /// One thread per selected pool for the in-threadgroup sort.
    static let expandThreads = GLM5NextSparseSelection.maximumSelectedPools
    static let keyBlock = 64
    static let headGroup = 32
    static let attentionThreads = 512
    static let mergeThreads = 256

    /// FP32 bytes of `attend`'s split partials (max, sum, PV), the only
    /// transient beyond its `[B,H,T,C]` output.
    static func attentionPartialBytes(rows: Int, width: Int, heads: Int, latentDimension: Int)
        -> Int
    {
        let blocks = (width + keyBlock - 1) / keyBlock
        return rows * blocks * heads * (latentDimension + 2) * 4
    }

    /// Queries `[B,Q,H,D]`, head weights `[B,Q,H]`, raw keys/gates `[B,N,D]`,
    /// position bias `[4,D]` and boolean key validity `[B,N]` (views allowed).
    /// Returns FP32 `[B,Q,P]` scores (ineligible pools at the finite minimum)
    /// and the `[B,P]` complete-pool mask.
    static func poolScores(
        indexQueries: MLXArray, headWeights: MLXArray, indexKeys: MLXArray, poolGates: MLXArray,
        poolPositionBias: MLXArray, keyValidity: MLXArray, queryStart: Int, poolingType: DType
    ) throws -> (scores: MLXArray, complete: MLXArray) {
        let (batch, queries, heads, dimension) = (
            indexQueries.dim(0), indexQueries.dim(1), indexQueries.dim(2), indexQueries.dim(3)
        )
        let history = indexKeys.dim(1)
        guard (1 ... maximumQueries).contains(queries), dimension <= maximumIndexDimension,
            indexKeys.shape == [batch, history, dimension], poolGates.shape == indexKeys.shape,
            headWeights.shape == [batch, queries, heads], poolPositionBias.shape == [4, dimension],
            keyValidity.shape == [batch, history], keyValidity.dtype == .bool,
            [.float16, .bfloat16, .float32].contains(poolingType)
        else { throw Failure.invalidGeometry }
        try requireGPU()
        let pools = (history + 3) / 4
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let outputs = scoreKernel(
                [
                    indexQueries, headWeights, indexKeys, poolGates, poolPositionBias,
                    keyValidity, GLM5NextKernelConstants.shared.int32(Int32(queryStart)),
                    GLM5NextKernelConstants.shared.floats([
                        1 / sqrt(Float(dimension)), 1 / sqrt(Float(heads)),
                    ]),
                ],
                template: [
                    ("R", poolingType), ("PT", poolTile), ("MaxD", maximumIndexDimension),
                    ("Threads", scoreThreads),
                ],
                grid: (scoreThreads * ((pools + poolTile - 1) / poolTile), batch, 1),
                threadGroup: (scoreThreads, 1, 1),
                outputShapes: [[batch, queries, pools], [batch, pools]],
                outputDTypes: [.float32, .bool])
            return (outputs[0], outputs[1])
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    /// `chosen` is the selector's `[B*Q,K]` pool IDs (any order, padded with
    /// P). Returns int32 `[B,Q,tokenBudget+3]` token indices.
    static func expand(
        chosen: MLXArray, keyValidity: MLXArray, queryValidity: MLXArray, queryStart: Int,
        tokenBudget: Int
    ) throws -> MLXArray {
        let (batch, queries) = (queryValidity.dim(0), queryValidity.dim(1))
        let pools = (keyValidity.dim(1) + 3) / 4
        guard chosen.ndim == 2, chosen.dtype == .int32, chosen.dim(0) == batch * queries,
            chosen.dim(1) == min(tokenBudget / 4, pools), chosen.dim(1) <= expandThreads,
            keyValidity.ndim == 2, keyValidity.dim(0) == batch, keyValidity.dtype == .bool,
            queryValidity.dtype == .bool
        else { throw Failure.invalidGeometry }
        try requireGPU()
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            return expandKernel(
                [
                    chosen, keyValidity, queryValidity,
                    GLM5NextKernelConstants.shared.int32s([
                        Int32(queryStart), Int32(tokenBudget),
                    ]),
                ],
                template: [("Threads", expandThreads)],
                grid: (expandThreads, batch * queries, 1), threadGroup: (expandThreads, 1, 1),
                outputShapes: [[batch, queries, tokenBudget + 3]], outputDTypes: [.int32])[0]
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    /// Absorbed queries `[B,H,T,C]`, latent history `[B,N,C]` (views allowed)
    /// and `[B,T,W]` token indices (-1 skipped). Returns `[B,H,T,C]` attended
    /// latents in `outputType`; `scale` multiplies each raw score.
    static func attend(
        queryLatents: MLXArray, latents: MLXArray, indices: MLXArray, scale: Float,
        outputType: DType
    ) throws -> MLXArray {
        let (batch, heads, count, dimension) = (
            queryLatents.dim(0), queryLatents.dim(1), queryLatents.dim(2), queryLatents.dim(3)
        )
        guard queryLatents.ndim == 4, latents.ndim == 3, indices.ndim == 3,
            latents.dim(0) == batch, latents.dim(1) > 0, latents.dim(2) == dimension,
            indices.dim(0) == batch, indices.dim(1) == count, indices.dim(2) > 0,
            indices.dtype == .int32, batch > 0, heads > 0, count > 0, dimension > 0,
            [.float16, .bfloat16, .float32].contains(outputType)
        else { throw Failure.invalidGeometry }
        try requireGPU()
        let rows = batch * count
        let blocks = (indices.dim(2) + keyBlock - 1) / keyBlock
        let (elements, overflow) = (rows * blocks * heads).multipliedReportingOverflow(
            by: dimension)
        guard !overflow, elements <= Int(Int32.max) else { throw Failure.invalidGeometry }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let partials = blockKernel(
                [queryLatents, latents, indices, GLM5NextKernelConstants.shared.float(scale)],
                template: [
                    ("H", heads), ("C", dimension), ("BK", keyBlock), ("HG", headGroup),
                    ("Threads", attentionThreads),
                ],
                grid: (attentionThreads * blocks, (heads + headGroup - 1) / headGroup, rows),
                threadGroup: (attentionThreads, 1, 1),
                outputShapes: [
                    [rows, blocks, heads], [rows, blocks, heads], [rows, blocks, heads, dimension],
                ],
                outputDTypes: [.float32, .float32, .float32])
            let outputs = rows * heads * dimension
            return mergeKernel(
                partials + [GLM5NextKernelConstants.shared.int32(Int32(count))],
                template: [("H", heads), ("C", dimension), ("OT", outputType)],
                grid: ((outputs + mergeThreads - 1) / mergeThreads * mergeThreads, 1, 1),
                threadGroup: (mergeThreads, 1, 1),
                outputShapes: [[batch, heads, count, dimension]], outputDTypes: [outputType])[0]
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    private static func requireGPU() throws {
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// First valid key of row `b` (N when none), reduced over the threadgroup.
        private static let firstKeySource = """
            int first = N;
            for (int n = int(tid); n < N; n += Threads) {
                if (validity[vrow + n * vstep]) { first = n; break; }
            }
            first = simd_min(first);
            if (lane == 0) { shared_int[group] = first; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            first = N;
            for (int i = 0; i < Groups; ++i) { first = metal::min(first, shared_int[i]); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """

        /// One threadgroup per (32-pool tile, batch row). Pools of the tile are
        /// built once into threadgroup memory, then one SIMD-group per
        /// (query, pool) item strides the index heads.
        private static let scoreKernel = MLXFast.metalKernel(
            name: "glm5_next_sparse_decode_pool_scores",
            inputNames: [
                "queries", "weights", "keys", "gates", "bias", "validity", "query_start", "scales",
            ],
            outputNames: ["scores", "complete_pools"],
            source: """
                constexpr int Groups = Threads / 32;
                threadgroup float pooled[PT * MaxD];
                threadgroup int shared_int[Groups];
                threadgroup bool tile_complete[PT];
                const int Q = queries_shape[1];
                const int H = queries_shape[2];
                const int D = queries_shape[3];
                const int N = keys_shape[1];
                const int P = (N + 3) / 4;
                const int b = int(threadgroup_position_in_grid.y);
                const int tile = int(threadgroup_position_in_grid.x);
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                const int64_t vrow = int64_t(b) * validity_strides[0];
                const int64_t vstep = validity_strides[1];
                \(firstKeySource)
                if (int(tid) < PT) {
                    const int p = tile * PT + int(tid);
                    bool complete = p < P;
                    for (int j = 0; j < 4 && complete; ++j) {
                        const int n = first + 4 * p + j;
                        complete = n < N && validity[vrow + n * vstep];
                    }
                    tile_complete[tid] = complete;
                    if (p < P) { complete_pools[size_t(b) * P + p] = complete; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // The reference pooling: MLX's FP32 softmax of gate + bias over
                // the pool's four keys, weights rounded to R, rounded products,
                // and MLX's sequential R-precision sum of the four products.
                const int64_t krow = int64_t(b) * keys_strides[0];
                const int64_t grow = int64_t(b) * gates_strides[0];
                for (int e = int(tid); e < PT * D; e += Threads) {
                    const int pl = e / D;
                    const int d = e - pl * D;
                    float value = 0.0f;
                    if (tile_complete[pl]) {
                        const int n0 = first + 4 * (tile * PT + pl);
                        float logits[4];
                        float maximum = -3.402823466e+38f;
                        for (int j = 0; j < 4; ++j) {
                            logits[j] =
                                float(gates[grow + (n0 + j) * gates_strides[1] + d * gates_strides[2]])
                                + float(bias[j * bias_strides[0] + d * bias_strides[1]]);
                            maximum = (maximum < logits[j]) ? logits[j] : maximum;
                        }
                        float exps[4];
                        float normalizer = 0.0f;
                        for (int j = 0; j < 4; ++j) {
                            exps[j] = metal::fast::exp(logits[j] - maximum);
                            normalizer += exps[j];
                        }
                        normalizer = 1 / normalizer;
                        R total = R(0);
                        for (int j = 0; j < 4; ++j) {
                            const R w = static_cast<R>(exps[j] * normalizer);
                            const R k = static_cast<R>(
                                keys[krow + (n0 + j) * keys_strides[1] + d * keys_strides[2]]);
                            const R product = static_cast<R>(float(w) * float(k));
                            total = j == 0 ? product : static_cast<R>(float(total) + float(product));
                        }
                        value = float(total);
                    }
                    pooled[pl * MaxD + d] = value;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const float index_scale = scales[0];
                const float head_scale = scales[1];
                for (int item = int(group); item < Q * PT; item += Groups) {
                    const int qi = item / PT;
                    const int pl = item - qi * PT;
                    const int p = tile * PT + pl;
                    if (p >= P) { continue; }
                    const bool eligible =
                        tile_complete[pl] && first + 4 * p + 3 <= query_start + qi;
                    float score = -3.402823466e+38f;
                    if (eligible) {
                        const int64_t qrow =
                            int64_t(b) * queries_strides[0] + int64_t(qi) * queries_strides[1];
                        const int64_t wrow =
                            int64_t(b) * weights_strides[0] + int64_t(qi) * weights_strides[1];
                        const threadgroup float* key = pooled + pl * MaxD;
                        float v = 0.0f;
                        for (int h = int(lane); h < H; h += 32) {
                            const int64_t qh = qrow + int64_t(h) * queries_strides[2];
                            float acc = 0.0f;
                            for (int d = 0; d < D; ++d) {
                                acc += float(queries[qh + d * queries_strides[3]]) * key[d];
                            }
                            const float s = metal::max(acc * index_scale, 0.0f);
                            v += s * (float(weights[wrow + h * weights_strides[2]]) * head_scale);
                        }
                        score = simd_sum(v);
                    }
                    if (lane == 0) { scores[(size_t(b) * Q + qi) * P + p] = score; }
                }
                """,
            ensureRowContiguous: false)

        /// One threadgroup per query row: bitonic sort of the chosen pools,
        /// then the members, padding and tail exactly as `selectTokens`.
        private static let expandKernel = MLXFast.metalKernel(
            name: "glm5_next_sparse_decode_expand",
            inputNames: ["chosen", "validity", "query_validity", "params"],
            outputNames: ["tokens"],
            source: """
                constexpr int Groups = Threads / 32;
                threadgroup int order[Threads];
                threadgroup int shared_int[Groups];
                threadgroup int overlap;
                const int Q = query_validity_shape[1];
                const int N = validity_shape[1];
                const int P = (N + 3) / 4;
                const int count = chosen_shape[1];
                const int r = int(threadgroup_position_in_grid.y);
                const int b = r / Q;
                const int qi = r - b * Q;
                const int pos = params[0] + qi;
                const int budget = params[1];
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                const int64_t vrow = int64_t(b) * validity_strides[0];
                const int64_t vstep = validity_strides[1];
                \(firstKeySource)
                int visible = 0;
                for (int n = int(tid); n <= pos && n < N; n += Threads) {
                    visible += validity[vrow + n * vstep] ? 1 : 0;
                }
                visible = simd_sum(visible);
                if (lane == 0) { shared_int[group] = visible; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                visible = 0;
                for (int i = 0; i < Groups; ++i) { visible += shared_int[i]; }

                order[tid] = int(tid) < count
                    ? chosen[int64_t(r) * chosen_strides[0] + int64_t(tid) * chosen_strides[1]]
                    : 0x7FFFFFFF;
                if (tid == 0) { overlap = 0; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 2; k <= uint(Threads); k <<= 1) {
                    for (uint j = k >> 1; j > 0; j >>= 1) {
                        const uint other = tid ^ j;
                        if (other > tid) {
                            const int a = order[tid];
                            const int c = order[other];
                            if ((a > c) == ((tid & k) == 0)) {
                                order[tid] = c;
                                order[other] = a;
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }

                const int tail_count = visible % 4;
                const int tail_pool = (visible - tail_count) / 4;
                const bool query_valid =
                    query_validity[int64_t(b) * query_validity_strides[0]
                        + int64_t(qi) * query_validity_strides[1]];
                device int* row = tokens + size_t(r) * (budget + 3);
                if (int(tid) < count) {
                    const int p = order[tid];
                    bool eligible = p >= 0 && p < P && first + 4 * p + 3 <= pos;
                    for (int j = 0; j < 4 && eligible; ++j) {
                        const int n = first + 4 * p + j;
                        eligible = n < N && validity[vrow + n * vstep];
                    }
                    // The tail lies in pool tail_pool; a selected copy of it
                    // already holds every tail token.
                    if (eligible && p == tail_pool) { overlap = 1; }
                    for (int j = 0; j < 4; ++j) {
                        row[4 * int(tid) + j] = eligible && query_valid ? first + 4 * p + j : -1;
                    }
                }
                for (int i = 4 * count + int(tid); i < budget; i += Threads) { row[i] = -1; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < 3) {
                    const int t = first + visible - tail_count + int(tid);
                    const int clipped = metal::clamp(t, 0, N - 1);
                    const bool valid = int(tid) < tail_count && t < N
                        && validity[vrow + clipped * vstep] && clipped <= pos && overlap == 0;
                    row[budget + int(tid)] = valid && query_valid ? t : -1;
                }
                """,
            ensureRowContiguous: false)

        /// One threadgroup per (64-key block, 32-head group, query row): 16
        /// SIMD-groups of 8 heads x one quarter of the block (scores) or of the
        /// latent width (PV), on 8x8 FP32 simdgroup matrices. Partial PV is
        /// relative to the block's FP32 maximum in log2 units.
        private static let blockKernel = MLXFast.metalKernel(
            name: "glm5_next_sparse_decode_attention_blocks",
            inputNames: ["queries", "latents", "ids", "scale"],
            outputNames: ["block_max", "block_sum", "block_values"],
            source: """
                constexpr int WM = HG / 8;
                constexpr int WN = Threads / 32 / WM;
                constexpr int KQ = BK / WN;
                constexpr int KF = KQ / 8;
                constexpr int CQ = ((C + 8 * WN - 1) / (8 * WN)) * 8;
                constexpr int CF = CQ / 8;
                constexpr int LPH = Threads / HG;
                constexpr int KPL = BK / LPH;
                constexpr float NEG = -3.402823466e+38f;
                static_assert(KQ % 8 == 0 && LPH <= 32 && BK % LPH == 0, "block layout");
                threadgroup int sel[BK];
                threadgroup float probs[HG * BK];

                const uint tid = thread_position_in_threadgroup.x;
                const ushort lane = thread_index_in_simdgroup;
                const ushort sg = simdgroup_index_in_threadgroup;
                const int kb = int(threadgroup_position_in_grid.x);
                const int hg = int(threadgroup_position_in_grid.y);
                const int r = int(threadgroup_position_in_grid.z);
                const int T = queries_shape[2];
                const int b = r / T;
                const int t = r - b * T;
                const int N = latents_shape[1];
                const int W = ids_shape[2];
                const int NB = (W + BK - 1) / BK;
                const float scale2 = scale * 1.44269504089f;

                for (int i = int(tid); i < BK; i += Threads) {
                    const int slot = kb * BK + i;
                    const int id = slot < W
                        ? ids[int64_t(b) * ids_strides[0] + int64_t(t) * ids_strides[1]
                            + int64_t(slot) * ids_strides[2]]
                        : -1;
                    sel[i] = id >= 0 && id < N ? id : -1;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const int wm = int(sg) / WN;
                const int wn = int(sg) % WN;
                const int hl0 = wm * 8;
                const int h0 = hg * HG + hl0;
                const short qid = short(lane / 4);
                const short fm = (qid & 4) + short((lane / 2) % 4);
                const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
                const int64_t qbase = int64_t(b) * queries_strides[0] + int64_t(t) * queries_strides[2];
                const int64_t qs1 = queries_strides[1];
                const int64_t qs3 = queries_strides[3];
                const int64_t lbase = int64_t(b) * latents_strides[0];
                const int64_t ls1 = latents_strides[1];
                const int64_t ls2 = latents_strides[2];

                // Scores, transposed: S^T[key][head] = sum_c K[key][c] Q[head][c].
                simdgroup_matrix<float, 8, 8> score[KF];
                for (int kf = 0; kf < KF; ++kf) { score[kf] = simdgroup_matrix<float, 8, 8>(0.0f); }
                if (h0 < H) {
                    int key_id[KF];
                    for (int kf = 0; kf < KF; ++kf) { key_id[kf] = sel[wn * KQ + kf * 8 + fm]; }
                    const int ha = h0 + fn;
                    for (int c0 = 0; c0 < C; c0 += 8) {
                        const int cq = c0 + fm;
                        float2 qv = float2(0.0f);
                        if (cq < C) {
                            if (ha < H) { qv.x = float(queries[qbase + ha * qs1 + cq * qs3]); }
                            if (ha + 1 < H) { qv.y = float(queries[qbase + (ha + 1) * qs1 + cq * qs3]); }
                        }
                        simdgroup_matrix<float, 8, 8> qm;
                        reinterpret_cast<thread float2&>(qm.thread_elements()) = qv;
                        const int ca = c0 + fn;
                        for (int kf = 0; kf < KF; ++kf) {
                            const int id = key_id[kf];
                            float2 kv = float2(0.0f);
                            if (id >= 0) {
                                const int64_t row = lbase + id * ls1;
                                if (ca < C) { kv.x = float(latents[row + ca * ls2]); }
                                if (ca + 1 < C) { kv.y = float(latents[row + (ca + 1) * ls2]); }
                            }
                            simdgroup_matrix<float, 8, 8> km;
                            reinterpret_cast<thread float2&>(km.thread_elements()) = kv;
                            simdgroup_multiply_accumulate(score[kf], km, qm, score[kf]);
                        }
                    }
                }
                for (int kf = 0; kf < KF; ++kf) {
                    const float2 e = reinterpret_cast<thread float2&>(score[kf].thread_elements());
                    const int key = wn * KQ + kf * 8 + fm;
                    probs[(hl0 + fn) * BK + key] = e.x;
                    probs[(hl0 + fn + 1) * BK + key] = e.y;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                {
                    const int hl = int(tid) / LPH;
                    const int k0 = (int(tid) % LPH) * KPL;
                    const int h = hg * HG + hl;
                    float s[KPL];
                    float m = NEG;
                    for (int i = 0; i < KPL; ++i) {
                        const int key = k0 + i;
                        s[i] = sel[key] >= 0 ? probs[hl * BK + key] * scale2 : NEG;
                        m = metal::max(m, s[i]);
                    }
                    for (ushort o = 1; o < LPH; o <<= 1) { m = metal::max(m, simd_shuffle_xor(m, o)); }
                    float l = 0.0f;
                    for (int i = 0; i < KPL; ++i) {
                        const int key = k0 + i;
                        const float p = sel[key] >= 0 ? metal::fast::exp2(s[i] - m) : 0.0f;
                        probs[hl * BK + key] = p;
                        l += p;
                    }
                    for (ushort o = 1; o < LPH; o <<= 1) { l += simd_shuffle_xor(l, o); }
                    if (int(tid) % LPH == 0 && h < H) {
                        const size_t index = (size_t(r) * NB + kb) * H + h;
                        block_max[index] = m;
                        block_sum[index] = l;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (h0 < H) {
                    simdgroup_matrix<float, 8, 8> out[CF];
                    for (int cf = 0; cf < CF; ++cf) { out[cf] = simdgroup_matrix<float, 8, 8>(0.0f); }
                    for (int ks = 0; ks < BK / 8; ++ks) {
                        const int pk = (hl0 + fm) * BK + ks * 8 + fn;
                        simdgroup_matrix<float, 8, 8> pmat;
                        reinterpret_cast<thread float2&>(pmat.thread_elements()) =
                            float2(probs[pk], probs[pk + 1]);
                        const int id = sel[ks * 8 + fm];
                        const int64_t row = lbase + int64_t(metal::max(id, 0)) * ls1;
                        for (int cf = 0; cf < CF; ++cf) {
                            const int c = wn * CQ + cf * 8 + fn;
                            float2 kv = float2(0.0f);
                            if (id >= 0) {
                                if (c < C) { kv.x = float(latents[row + c * ls2]); }
                                if (c + 1 < C) { kv.y = float(latents[row + (c + 1) * ls2]); }
                            }
                            simdgroup_matrix<float, 8, 8> vm;
                            reinterpret_cast<thread float2&>(vm.thread_elements()) = kv;
                            simdgroup_multiply_accumulate(out[cf], pmat, vm, out[cf]);
                        }
                    }
                    const int h = h0 + fm;
                    if (h < H) {
                        device float* orow = block_values + ((size_t(r) * NB + kb) * H + h) * C;
                        for (int cf = 0; cf < CF; ++cf) {
                            const float2 e = reinterpret_cast<thread float2&>(out[cf].thread_elements());
                            const int c = wn * CQ + cf * 8 + fn;
                            if (c < C) { orow[c] = e.x; }
                            if (c + 1 < C) { orow[c + 1] = e.y; }
                        }
                    }
                }
                """,
            header: "#include <metal_simdgroup_matrix>\n",
            ensureRowContiguous: false)

        /// One thread per output element: FP32 log-sum-exp merge of the block
        /// partials, one rounding to `OT`.
        private static let mergeKernel = MLXFast.metalKernel(
            name: "glm5_next_sparse_decode_attention_merge",
            inputNames: ["block_max", "block_sum", "block_values", "queries_per_row"],
            outputNames: ["latent"],
            source: """
                constexpr float NEG = -3.402823466e+38f;
                const int R = block_max_shape[0];
                const int NB = block_max_shape[1];
                const int T = queries_per_row;
                const uint gid = thread_position_in_grid.x;
                const int c = int(gid % uint(C));
                const int h = int((gid / uint(C)) % uint(H));
                const int r = int(gid / uint(C * H));
                if (r >= R) { return; }
                const size_t base = size_t(r) * NB * H + h;
                float M = NEG;
                for (int k = 0; k < NB; ++k) { M = metal::max(M, block_max[base + size_t(k) * H]); }
                float L = 0.0f;
                float acc = 0.0f;
                for (int k = 0; k < NB; ++k) {
                    const size_t index = base + size_t(k) * H;
                    const float m = block_max[index];
                    if (m == NEG) { continue; }
                    const float f = metal::fast::exp2(m - M);
                    L += block_sum[index] * f;
                    acc += block_values[index * C + c] * f;
                }
                const int b = r / T;
                const int t = r - b * T;
                latent[((size_t(b) * H + h) * T + t) * C + c] =
                    static_cast<OT>(L > 0.0f ? acc / L : 0.0f);
                """)
    #endif
}
