// Copyright © 2026 Eigen Labs / Darkbloom.
//
// GPT-OSS's residual steps as one kernel each on decode-shaped calls:
//
// - `attentionTail`: the output projection's bias, the residual add and the
//   post-attention norm (`h = residual + (y + bias)`, `rmsNorm(h, w)`): three
//   launches become one.
// - `expertTail`: each expert row's bias, its routing weight, the experts'
//   sum (as MLX's small column reduction adds them), the residual add and the
//   next layer's input norm (or the final norm): five or six launches become
//   one.
//
// Each value is the expression MLX's kernels write (`T + T`, `T * T`, and
// `rms_single_row` with the mean form `RMSNormMeanForm.measured` finds), one
// thread group per row, so the outputs are the unfused ops' bits whatever the
// row count.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

enum GPTOSSFusedNorms {
    /// Diagnostic only: `DARKBLOOM_GPTOSS_FUSED_NORMS=0` keeps the unfused ops.
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_FUSED_NORMS"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// Whether the kernels cover rows of `width` in `dtype` on this device
    /// (one thread group per row: MLX's looped norm, above 4096, is not copied).
    static func supports(width: Int, dtype: DType) -> Bool {
        isEnabled && Device.defaultDevice().deviceType == .gpu
            && [.bfloat16, .float16].contains(dtype) && (1 ... 4096).contains(width)
            && RMSNormMeanForm.measured(width: width, dtype: dtype) != nil
    }

    /// `h = residual + (projection + bias)` and `rmsNorm(h, weight)` for rows
    /// `[..., D]` (`projection` is the output projection before its bias).
    static func attentionTail(
        _ projection: MLXArray, bias: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float
    ) -> (sum: MLXArray, normed: MLXArray) {
        let width = projection.dim(-1)
        let rows = projection.size / width
        let flat = [rows, width]
        let outputs = attentionTailKernel(
            [
                projection.reshaped(flat), bias, residual.reshaped(flat), weight,
                ResidualRMSNorm.Constants.shared.epsilon(eps),
            ],
            template: [
                ("T", projection.dtype), ("D", width),
                ("MEAN_FORM", Gemma4FusedNorms.meanForm(width, projection.dtype)),
            ],
            grid: (rows * Gemma4FusedNorms.threads(width), 1, 1),
            threadGroup: (Gemma4FusedNorms.threads(width), 1, 1),
            outputShapes: [flat, flat], outputDTypes: [projection.dtype, projection.dtype])
        let shape = projection.shape
        return (outputs[0].reshaped(shape), outputs[1].reshaped(shape))
    }

    /// `out = residual + sum_k((rows_k + biases[indices_k]) * weights_k)` and
    /// `rmsNorm(out, next)`, for expert rows `[N, K, D]` before their bias,
    /// `indices` and `weights` `[N, K]`, the bias table `[E, D]` and
    /// `residual` `[N, D]`. The sum is MLX's `sum(axis: -2)` of the weighted
    /// rows: each row plus zero, then added to the running sum in expert
    /// order, in the activation dtype (`col_reduce_small`).
    static func expertTail(
        rows: MLXArray, indices: MLXArray, biases: MLXArray, weights: MLXArray,
        residual: MLXArray, next: MLXArray, eps: Float
    ) -> (out: MLXArray, normed: MLXArray) {
        let width = rows.dim(-1)
        let experts = indices.dim(-1)
        let count = indices.size / experts
        let flat = [count, width]
        let outputs = expertTailKernel(
            [
                rows.reshaped(count, experts, width),
                indices.reshaped(count, experts).asType(.uint32), biases,
                weights.reshaped(count, experts), residual.reshaped(flat), next,
                ResidualRMSNorm.Constants.shared.epsilon(eps),
            ],
            template: [
                ("T", rows.dtype), ("D", width), ("KE", experts),
                ("MEAN_FORM", Gemma4FusedNorms.meanForm(width, rows.dtype)),
            ],
            grid: (count * Gemma4FusedNorms.threads(width), 1, 1),
            threadGroup: (Gemma4FusedNorms.threads(width), 1, 1),
            outputShapes: [flat, flat], outputDTypes: [rows.dtype, rows.dtype])
        let shape = residual.shape
        return (outputs[0].reshaped(shape), outputs[1].reshaped(shape))
    }

    /// Diagnostic only: `DARKBLOOM_GPTOSS_FUSED_EXPERT_BIAS=0` keeps the
    /// gate/up projection's bias add before the compiled SwiGLU.
    static let expertBiasEnabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_FUSED_EXPERT_BIAS"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var expertBiasOverride: Bool?
    static var isExpertBiasEnabled: Bool {
        isEnabled && (expertBiasOverride ?? expertBiasEnabledByEnvironment)
    }

    // MARK: - Attention prologue

    /// Diagnostic only: `DARKBLOOM_GPTOSS_FUSED_PROLOGUE=0` keeps the
    /// projections' bias adds, YaRN's scale and `MLXFast.RoPE` separate.
    static let prologueEnabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_FUSED_PROLOGUE"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var prologueOverride: Bool?
    static var isPrologueEnabled: Bool {
        isEnabled && (prologueOverride ?? prologueEnabledByEnvironment)
    }

    /// GPT-OSS attention's prologue in one kernel. For each of the `B * L`
    /// rows (row `b * L + t` sits at position `offsets[b] + t`) and head:
    /// `rope((q + qBias) * mscale)`, `rope((k + kBias) * mscale)` and
    /// `v + vBias`, each op in the activation dtype, the rotation as
    /// `rotation.form` measures MLX's `rope` (non-traditional, scale 1, from
    /// `frequencies`). Inputs are the quantized products `[B, L, heads * D]`
    /// before their bias; outputs are `[B, heads, L, D]`, the values of the
    /// bias adds, transpose, YaRN's multiply and `MLXFast.RoPE`.
    static func attentionPrologue(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        queryBias: MLXArray, keyBias: MLXArray, valueBias: MLXArray,
        queryHeads: Int, keyValueHeads: Int, headDim: Int, mscale: Float,
        offsets: MLXArray, rotation: Gemma4FusedNorms.Rotation
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (batch, tokens) = (queries.dim(0), queries.dim(1))
        let dtype = queries.dtype
        let half = headDim / 2
        let threads = 32 * ((half + 31) / 32)
        let outputs = attentionPrologueKernel(
            [
                queries, keys, values, queryBias, keyBias, valueBias, MLXArray([mscale]),
                offsets.asType(.int32), rotation.baseArray, rotation.frequencyArray,
            ],
            template: [
                ("T", dtype), ("D", headDim), ("HALF", half), ("TPG", threads),
                ("NQ", queryHeads), ("NK", keyValueHeads), ("TOKENS", tokens),
                ("SCALED", mscale != 1 ? 1 : 0),
                ("ROPE_FREQS", rotation.usesFrequencies ? 1 : 0),
            ] + rotation.form.templates,
            grid: ((queryHeads + 2 * keyValueHeads) * threads, batch * tokens, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [
                [batch, queryHeads, tokens, headDim], [batch, keyValueHeads, tokens, headDim],
                [batch, keyValueHeads, tokens, headDim],
            ],
            outputDTypes: [dtype, dtype, dtype])
        return (outputs[0], outputs[1], outputs[2])
    }

    private static let attentionPrologueKernel = MLXFast.metalKernel(
        name: "gptoss_attention_prologue",
        inputNames: [
            "q", "k", "v", "bq", "bk", "bv", "mscale", "offsets", "rope_base", "rope_freqs",
        ],
        outputNames: ["qo", "ko", "vo"],
        source: """
            const uint slot = threadgroup_position_in_grid.x;
            const uint r = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint b = r / TOKENS;
            const uint t = r % TOKENS;
            const bool is_q = slot < NQ;
            const bool is_k = !is_q && slot < NQ + NK;
            const uint head = is_q ? slot : (is_k ? slot - NQ : slot - NQ - NK);
            const uint heads = is_q ? NQ : NK;
            const device T* src = is_q ? q : (is_k ? k : v);
            const device T* bias = is_q ? bq : (is_k ? bk : bv);
            const size_t row = (size_t(r) * heads + head) * D;
            const size_t out_row = ((size_t(b) * heads + head) * TOKENS + t) * D;
            const size_t bias_row = size_t(head) * D;
            if (is_q || is_k) {
              device T* dst = is_q ? qo : ko;
              const float position = static_cast<float>(t + uint(offsets[b]));
              const T scale = static_cast<T>(mscale[0]);
              for (uint i = lid; i < HALF; i += TPG) {
                T a = src[row + i] + bias[bias_row + i];
                T c = src[row + i + HALF] + bias[bias_row + i + HALF];
                if (SCALED) {
                  a = a * scale;
                  c = c * scale;
                }
                const float x1 = static_cast<float>(a);
                const float x2 = static_cast<float>(c);
                \(RoPEForm.rotationSource)
                dst[out_row + i] = static_cast<T>(rx1);
                dst[out_row + i + HALF] = static_cast<T>(rx2);
              }
            } else {
              for (uint i = lid; i < D; i += TPG) {
                vo[out_row + i] = src[row + i] + bias[bias_row + i];
              }
            }
            """)

    // MARK: - Routing

    /// Diagnostic only: `DARKBLOOM_GPTOSS_FUSED_ROUTING=0` keeps the unfused
    /// routing ops (with the fused residual steps).
    static let routingEnabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_FUSED_ROUTING"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var routingOverride: Bool?

    /// The measured softmax form for `topK` values of `dtype`, where the
    /// routing kernel covers `experts` (nil: keep the unfused ops).
    static func routingForm(experts: Int, topK: Int, dtype: DType) -> SoftmaxForm? {
        guard isEnabled, routingOverride ?? routingEnabledByEnvironment,
            Device.defaultDevice().deviceType == .gpu, [.bfloat16, .float16].contains(dtype),
            (1 ... 1024).contains(experts), (1 ... min(128, experts)).contains(topK)
        else { return nil }
        return SoftmaxForm.measured(axis: topK, dtype: dtype)
    }

    /// The router after its quantized product, in one kernel per row:
    /// `logits + bias` in the activation dtype, the `topK` highest in the
    /// order `mlxTopK` leaves them (`argPartition`'s GPU path is a stable
    /// merge sort, so the last `topK` of a stable ascending order, NaN last),
    /// and `softmax(precise: true)` of those values. The same indices and bits
    /// as the bias add, `argPartition`, `takeAlong` and `softmax`.
    static func route(logits: MLXArray, bias: MLXArray, topK: Int, form: SoftmaxForm) -> (
        indices: MLXArray, weights: MLXArray
    ) {
        let experts = logits.dim(-1)
        let rows = logits.size / experts
        let threads = 32 * ((experts + 31) / 32)
        let shape = Array(logits.shape.dropLast()) + [topK]
        let outputs = routeKernel(
            [logits.reshaped(rows, experts), bias],
            template: [
                ("T", logits.dtype), ("E", experts), ("AXIS", topK), ("TPG", threads),
                ("SOFTMAX_FORM", form.rawValue),
            ],
            grid: (rows * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, topK], [rows, topK]], outputDTypes: [.uint32, logits.dtype])
        return (outputs[0].reshaped(shape), outputs[1].reshaped(shape))
    }

    // `Gemma4FusedNorms.route`'s selection on the biased logits, without a scale.
    private static let routeKernel = MLXFast.metalKernel(
        name: "gptoss_route",
        inputNames: ["scores", "bias"],
        outputNames: ["indices", "weights"],
        source: """
            const uint r = threadgroup_position_in_grid.x;
            const uint j = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float vals[E];
            threadgroup uint top_index[AXIS];
            threadgroup float top_value[AXIS];
            threadgroup float local_max[32];
            threadgroup float local_normalizer[32];
            const size_t row = size_t(r) * E;
            for (uint e = j; e < E; e += TPG) {
              const T biased = scores[row + e] + bias[e];
              vals[e] = static_cast<float>(biased);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // A stable ascending sort's position: the values before this one
            // (MLX's LessThan: NaN last), then equal values with lower indices.
            for (uint e = j; e < E; e += TPG) {
              const float v = vals[e];
              const bool vn = metal::isnan(v);
              uint rank = 0;
              for (uint i = 0; i < E; i++) {
                const float u = vals[i];
                const bool un = metal::isnan(u);
                const bool before = (un | vn) ? (!un & vn) : (u < v);
                const bool after = (un | vn) ? (!vn & un) : (v < u);
                rank += (before || (!after && i < e)) ? 1u : 0u;
              }
              if (rank >= E - AXIS) {
                top_index[rank - (E - AXIS)] = e;
                top_value[rank - (E - AXIS)] = v;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #define softmax_load(p) top_value[(p)]
            #define softmax_store(p, value) weights[size_t(r) * AXIS + (p)] = static_cast<T>(value)
            \(SoftmaxForm.rowSource)
            #undef softmax_load
            #undef softmax_store
            for (uint p = j; p < AXIS; p += TPG) { indices[size_t(r) * AXIS + p] = top_index[p]; }
            """)

    // MARK: - Kernels

    private static let attentionTailSource =
        Gemma4FusedNorms.prologue + """
              float hv[N_READS];
              for (int i = 0; i < N_READS; i++) {
                if (full || base + i < axis_size) {
                  const T projected = y[row + base + i] + b[base + i];
                  const T sum = res[row + base + i] + projected;
                  h[row + base + i] = sum;
                  hv[i] = sum;
                } else {
                  hv[i] = 0;
                }
              }

            """ + Gemma4FusedNorms.squares("hv") + Gemma4FusedNorms.reduce("inv_h") + """
              for (int i = 0; i < N_READS; i++) {
                if (full || base + i < axis_size) {
                  n[row + base + i] = w[base + i] * static_cast<T>(hv[i] * inv_h);
                }
              }

            """

    private static let expertTailSource =
        Gemma4FusedNorms.prologue + """
              float ov[N_READS];
              for (int i = 0; i < N_READS; i++) {
                if (full || base + i < axis_size) {
                  T acc = T(0);
                  for (int j = 0; j < KE; j++) {
                    const size_t slot = size_t(gid) * KE + j;
                    const T biased = rows[slot * axis_size + base + i]
                        + biases[size_t(idx[slot]) * axis_size + base + i];
                    const T product = biased * rw[slot];
                    const T lane = product + T(0);
                    acc = j == 0 ? lane : static_cast<T>(lane + acc);
                  }
                  const T sum = res[row + base + i] + acc;
                  out[row + base + i] = sum;
                  ov[i] = sum;
                } else {
                  ov[i] = 0;
                }
              }

            """ + Gemma4FusedNorms.squares("ov") + Gemma4FusedNorms.reduce("inv_o") + """
              for (int i = 0; i < N_READS; i++) {
                if (full || base + i < axis_size) {
                  nn[row + base + i] = wn[base + i] * static_cast<T>(ov[i] * inv_o);
                }
              }

            """

    private static let attentionTailKernel = MLXFast.metalKernel(
        name: "gptoss_attention_tail", inputNames: ["y", "b", "res", "w", "eps"],
        outputNames: ["h", "n"], source: attentionTailSource)

    private static let expertTailKernel = MLXFast.metalKernel(
        name: "gptoss_expert_tail",
        inputNames: ["rows", "idx", "biases", "rw", "res", "wn", "eps"],
        outputNames: ["out", "nn"], source: expertTailSource)
}
