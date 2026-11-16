// Copyright © 2026 Eigen Labs / Darkbloom.
//
// Gemma 4's norm chains around the feed-forward as two kernels per layer:
// `attentionTail` (post-attention norm, residual add, the sum's norms) and
// `feedforwardTail` (branch post-norms, their sum, post-norm, residual add,
// layer scalar, and the next layer's input norm). Each value is the expression
// MLX's kernels write (`rms_single_row`, N_READS = 4, with the mean form of
// `RMSNormMeanForm.measured`; `T + T`, `T * T`), one thread group per row, so
// the outputs are the unfused ops' bits whatever the row count.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

enum Gemma4FusedNorms {
    /// Diagnostic only: `DARKBLOOM_GEMMA4_FUSED_NORMS=0` keeps the unfused ops.
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_FUSED_NORMS"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// Whether the kernels cover rows of `width` in `dtype` (MLX's looped norm,
    /// above 4096, is not copied).
    static func supports(width: Int, dtype: DType) -> Bool {
        isEnabled && (1 ... 4096).contains(width)
            && RMSNormMeanForm.measured(width: width, dtype: dtype) != nil
    }

    private static func meanForm(_ width: Int, _ dtype: DType) -> Int {
        guard let form = RMSNormMeanForm.measured(width: width, dtype: dtype) else {
            preconditionFailure("Gemma4FusedNorms: called where `supports` is false")
        }
        return form.rawValue
    }

    private static func threads(_ width: Int) -> Int { 32 * (((width + 3) / 4 + 31) / 32) }

    /// `sum = residual + rmsNorm(attention, postAttention)` and
    /// `rmsNorm(sum, w)` for each `w` in `sumNorms` (one, or three for a
    /// mixture-of-experts layer: dense pre-norm, experts pre-norm, router).
    static func attentionTail(
        _ attention: MLXArray, residual: MLXArray, postAttention: MLXArray,
        sumNorms: [MLXArray], eps: Float
    ) -> (sum: MLXArray, norms: [MLXArray]) {
        precondition(sumNorms.count == 1 || sumNorms.count == 3)
        let width = attention.dim(-1)
        let rows = attention.size / width
        let kernel = sumNorms.count == 3 ? attentionTailThree : attentionTailOne
        let flat = [rows, width]
        let outputs = kernel(
            [
                attention.reshaped(flat), residual.reshaped(flat), postAttention,
            ] + sumNorms + [ResidualRMSNorm.Constants.shared.epsilon(eps)],
            template: [
                ("T", attention.dtype), ("D", width),
                ("MEAN_FORM", meanForm(width, attention.dtype)),
            ],
            grid: (rows * threads(width), 1, 1), threadGroup: (threads(width), 1, 1),
            outputShapes: Array(repeating: flat, count: 1 + sumNorms.count),
            outputDTypes: Array(repeating: attention.dtype, count: 1 + sumNorms.count))
        let shape = attention.shape
        return (outputs[0].reshaped(shape), outputs.dropFirst().map { $0.reshaped(shape) })
    }

    /// The feed-forward's end. With `experts`:
    /// `s = rmsNorm(dense, postDense) + rmsNorm(experts, postExperts)`;
    /// without: `s = dense` (the dense layer's MLP output). Then
    /// `out = (residual + rmsNorm(s, post)) * layerScalar`, and, with `next`,
    /// `rmsNorm(out, next)` (the next layer's input norm or the final norm).
    static func feedforwardTail(
        dense: MLXArray, experts: (output: MLXArray, postDense: MLXArray, postExperts: MLXArray)?,
        residual: MLXArray, post: MLXArray, layerScalar: MLXArray, next: MLXArray?, eps: Float
    ) -> (out: MLXArray, next: MLXArray?) {
        let width = dense.dim(-1)
        let rows = dense.size / width
        let flat = [rows, width]
        var inputs = [dense.reshaped(flat)]
        if let experts {
            inputs += [experts.output.reshaped(flat), experts.postDense, experts.postExperts]
        }
        inputs += [residual.reshaped(flat), post, layerScalar.reshaped([1])]
        if let next { inputs.append(next) }
        inputs.append(ResidualRMSNorm.Constants.shared.epsilon(eps))
        let kernel: MLXFast.MLXFastKernel
        switch (experts != nil, next != nil) {
        case (true, true): kernel = feedforwardTailExpertsNext
        case (true, false): kernel = feedforwardTailExperts
        case (false, true): kernel = feedforwardTailDenseNext
        case (false, false): kernel = feedforwardTailDense
        }
        let count = next == nil ? 1 : 2
        let outputs = kernel(
            inputs,
            template: [
                ("T", dense.dtype), ("D", width), ("MEAN_FORM", meanForm(width, dense.dtype)),
            ],
            grid: (rows * threads(width), 1, 1), threadGroup: (threads(width), 1, 1),
            outputShapes: Array(repeating: flat, count: count),
            outputDTypes: Array(repeating: dense.dtype, count: count))
        let shape = dense.shape
        return (outputs[0].reshaped(shape), next == nil ? nil : outputs[1].reshaped(shape))
    }

    // MARK: - Attention prologue

    /// Diagnostic only: `DARKBLOOM_GEMMA4_FUSED_ATTENTION_PROLOGUE=0` keeps the
    /// unfused q/k/v norms and RoPE.
    static let prologueEnabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_FUSED_ATTENTION_PROLOGUE"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var prologueOverride: Bool?
    static var isPrologueEnabled: Bool {
        isEnabled && (prologueOverride ?? prologueEnabledByEnvironment)
    }

    /// A layer's rotation: MLX `rope` over the whole head, non-traditional,
    /// scale 1, from `base` or from `frequencies`, with its measured form.
    struct Rotation {
        let form: RoPEForm
        let usesFrequencies: Bool
        /// `[log2(base)]`, or a placeholder.
        let baseArray: MLXArray
        /// The frequencies, or a placeholder.
        let frequencyArray: MLXArray
    }

    /// A layer's measured rotation per activation dtype.
    final class RotationCache: @unchecked Sendable {
        private let lock = NSLock()
        private var rotations: [DType: Rotation?] = [:]

        /// The rotation for heads of `headDim` in `dtype` where the norm's
        /// mean form and the RoPE form are both measured; nil otherwise.
        /// Callers check `isPrologueEnabled` first.
        func rotation(headDim: Int, base: Float?, frequencies: MLXArray?, dtype: DType)
            -> Rotation?
        {
            if let known = lock.withLock({ rotations[dtype] }) { return known }
            var made: Rotation?
            if (1 ... 4096).contains(headDim),
                RMSNormMeanForm.measured(width: headDim, dtype: dtype) != nil,
                let form = RoPEForm.measured(
                    dimensions: headDim, base: base, frequencies: frequencies, dtype: dtype)
            {
                let baseArray = MLXArray([base.map { log2($0) } ?? 0])
                let frequencyArray = frequencies ?? MLXArray([Float(1)])
                eval(baseArray, frequencyArray)
                made = Rotation(
                    form: form, usesFrequencies: frequencies != nil, baseArray: baseArray,
                    frequencyArray: frequencyArray)
            }
            lock.withLock { rotations[dtype] = made }
            return made
        }
    }

    /// Gemma 4 attention's prologue in one kernel. For each of the `B * L`
    /// rows (row `b * L + t` sits at position `offsets[b] + t`) and head:
    /// `rope(rmsNorm(q, queryWeight))`, `rope(rmsNorm(k, keyWeight))` and
    /// `rmsNorm(v)` (no weight). Inputs are the projections, `[B, L, heads * D]`;
    /// outputs are `[B, heads, L, D]`, the layout and bits of today's
    /// `RMSNorm`, transpose and `MLXFast.RoPE` (one thread group per head
    /// row, as MLX's norm, so a row's bits do not depend on the batch).
    static func attentionPrologue(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        queryHeads: Int, keyValueHeads: Int, headDim: Int,
        queryWeight: MLXArray, keyWeight: MLXArray, offsets: MLXArray,
        rotation: Rotation, eps: Float
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (batch, tokens) = (queries.dim(0), queries.dim(1))
        let dtype = queries.dtype
        let threads = threads(headDim)
        let outputs = attentionPrologueKernel(
            [
                queries, keys, values, queryWeight, keyWeight, offsets.asType(.int32),
                rotation.baseArray, rotation.frequencyArray, ResidualRMSNorm.Constants.shared.epsilon(eps),
            ],
            template: [
                ("T", dtype), ("D", headDim), ("HALF", headDim / 2), ("TPG", threads),
                ("NQ", queryHeads), ("NK", keyValueHeads), ("TOKENS", tokens),
                ("MEAN_FORM", meanForm(headDim, dtype)),
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
        name: "gemma4_attention_prologue",
        inputNames: [
            "q", "k", "v", "wq", "wk", "offsets", "rope_base", "rope_freqs", "eps",
        ],
        outputNames: ["qo", "ko", "vo"],
        source: """
            constexpr int N_READS = 4;
            constexpr int SIMD_SIZE = 32;
            const uint slot = threadgroup_position_in_grid.x;
            const uint r = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint axis_size = D;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            threadgroup T normed[D];
            const uint b = r / TOKENS;
            const uint t = r % TOKENS;
            const bool is_q = slot < NQ;
            const bool is_k = !is_q && slot < NQ + NK;
            const uint head = is_q ? slot : (is_k ? slot - NQ : slot - NQ - NK);
            const uint heads = is_q ? NQ : NK;
            const device T* src = is_q ? q : (is_k ? k : v);
            const size_t row = (size_t(r) * heads + head) * axis_size;
            const size_t out_row = ((size_t(b) * heads + head) * TOKENS + t) * axis_size;
            const uint base = lid * N_READS;
            const bool full = base + N_READS <= axis_size;
            float acc = 0;
            float xv[N_READS];
            for (int i = 0; i < N_READS; i++) {
              xv[i] = (full || base + i < axis_size) ? static_cast<float>(src[row + base + i]) : 0;
            }

            """ + squares("xv") + reduce("inv") + """
              if (is_q || is_k) {
                const device T* w = is_q ? wq : wk;
                for (int i = 0; i < N_READS; i++) {
                  if (full || base + i < axis_size) {
                    normed[base + i] = w[base + i] * static_cast<T>(xv[i] * inv);
                  }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                device T* dst = is_q ? qo : ko;
                const float position = static_cast<float>(t + uint(offsets[b]));
                for (uint i = lid; i < HALF; i += TPG) {
                  const float x1 = static_cast<float>(normed[i]);
                  const float x2 = static_cast<float>(normed[i + HALF]);
                  \(RoPEForm.rotationSource)
                  dst[out_row + i] = static_cast<T>(rx1);
                  dst[out_row + i + HALF] = static_cast<T>(rx2);
                }
              } else {
                for (int i = 0; i < N_READS; i++) {
                  if (full || base + i < axis_size) {
                    vo[out_row + base + i] = static_cast<T>(xv[i] * inv);
                  }
                }
              }
            """)

    // MARK: - Kernels

    // rms_single_row's prologue.
    private static let prologue = """
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        const uint gid = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint simd_lane_id = thread_index_in_simdgroup;
        const uint simd_group_id = simdgroup_index_in_threadgroup;
        const uint axis_size = D;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];
        const size_t row = gid * size_t(axis_size);
        const uint base = lid * N_READS;
        const bool full = base + N_READS <= axis_size;
        float acc = 0;

        """

    private static func reduce(_ into: String) -> String {
        """
          acc = simd_sum(acc);
          if (simd_group_id == 0) { local_sums[simd_lane_id] = 0; }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_lane_id == 0) { local_sums[simd_group_id] = acc; }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group_id == 0) {
            acc = simd_sum(local_sums[simd_lane_id]);
            if (simd_lane_id == 0) {
              local_inv_mean[0] = \(RMSNormMeanForm.inverseMeanExpression);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          const float \(into) = local_inv_mean[0];
          // Every thread has read the mean before the next reduction reuses it.
          threadgroup_barrier(mem_flags::mem_threadgroup);

        """
    }

    // Sum of squares of `values` (zero past the row's end).
    private static func squares(_ values: String) -> String {
        """
          acc = 0;
          for (int i = 0; i < N_READS; i++) { acc += \(values)[i] * \(values)[i]; }

        """
    }

    // `name`'s row as floats, zero past its end.
    private static func load(_ name: String, into: String) -> String {
        """
          float \(into)[N_READS];
          for (int i = 0; i < N_READS; i++) {
            \(into)[i] = (full || base + i < axis_size) ? static_cast<float>(\(name)[row + base + i]) : 0;
          }

        """
    }

    private static func attentionTailSource(norms: Int) -> String {
        let names = norms == 3 ? ["n1", "n2", "n3"] : ["n1"]
        let weights = norms == 3 ? ["w1", "w2", "w3"] : ["w1"]
        let writes = zip(names, weights).map { n, w in
            "      \(n)[row + base + i] = \(w)[base + i] * static_cast<T>(hv[i] * inv_h);"
        }.joined(separator: "\n")
        return prologue + load("attn", into: "av") + squares("av") + reduce("inv_a") + """
              float hv[N_READS];
              for (int i = 0; i < N_READS; i++) {
                if (full || base + i < axis_size) {
                  const T post = wpa[base + i] * static_cast<T>(av[i] * inv_a);
                  const T sum = res[row + base + i] + post;
                  h[row + base + i] = sum;
                  hv[i] = sum;
                } else {
                  hv[i] = 0;
                }
              }

            """ + squares("hv") + reduce("inv_h") + """
                  for (int i = 0; i < N_READS; i++) {
                    if (full || base + i < axis_size) {
                \(writes)
                    }
                  }

                """
    }

    private static func feedforwardTailSource(experts: Bool, next: Bool) -> String {
        var s = prologue + load("dense", into: "dv")
        if experts {
            s +=
                squares("dv") + reduce("inv_d") + """
                      T dn[N_READS];
                      for (int i = 0; i < N_READS; i++) {
                        dn[i] = (full || base + i < axis_size) ? wd[base + i] * static_cast<T>(dv[i] * inv_d) : T(0);
                      }

                    """ + load("experts", into: "ev") + squares("ev") + reduce("inv_e") + """
                      float sv[N_READS];
                      for (int i = 0; i < N_READS; i++) {
                        if (full || base + i < axis_size) {
                          const T en = we[base + i] * static_cast<T>(ev[i] * inv_e);
                          const T sum = dn[i] + en;
                          sv[i] = sum;
                        } else {
                          sv[i] = 0;
                        }
                      }

                    """
        } else {
            s += """
                  float sv[N_READS];
                  for (int i = 0; i < N_READS; i++) { sv[i] = dv[i]; }

                """
        }
        s +=
            squares("sv") + reduce("inv_s") + """
                  float ov[N_READS];
                  for (int i = 0; i < N_READS; i++) {
                    if (full || base + i < axis_size) {
                      const T post = wp[base + i] * static_cast<T>(sv[i] * inv_s);
                      const T sum = res[row + base + i] + post;
                      const T scaled = sum * ls[0];
                      out[row + base + i] = scaled;
                      ov[i] = scaled;
                    } else {
                      ov[i] = 0;
                    }
                  }

                """
        if next {
            s +=
                squares("ov") + reduce("inv_o") + """
                      for (int i = 0; i < N_READS; i++) {
                        if (full || base + i < axis_size) {
                          nn[row + base + i] = wn[base + i] * static_cast<T>(ov[i] * inv_o);
                        }
                      }

                    """
        }
        return s
    }

    private static let attentionTailThree = MLXFast.metalKernel(
        name: "gemma4_attention_tail_moe",
        inputNames: ["attn", "res", "wpa", "w1", "w2", "w3", "eps"],
        outputNames: ["h", "n1", "n2", "n3"], source: attentionTailSource(norms: 3))
    private static let attentionTailOne = MLXFast.metalKernel(
        name: "gemma4_attention_tail_dense",
        inputNames: ["attn", "res", "wpa", "w1", "eps"],
        outputNames: ["h", "n1"], source: attentionTailSource(norms: 1))

    private static func feedforwardTailKernel(experts: Bool, next: Bool) -> MLXFast.MLXFastKernel {
        var inputs = ["dense"]
        if experts { inputs += ["experts", "wd", "we"] }
        inputs += ["res", "wp", "ls"]
        if next { inputs.append("wn") }
        inputs.append("eps")
        return MLXFast.metalKernel(
            name: "gemma4_feedforward_tail_\(experts ? "moe" : "dense")\(next ? "_next" : "")",
            inputNames: inputs, outputNames: next ? ["out", "nn"] : ["out"],
            source: feedforwardTailSource(experts: experts, next: next))
    }
    private static let feedforwardTailExpertsNext = feedforwardTailKernel(experts: true, next: true)
    private static let feedforwardTailExperts = feedforwardTailKernel(experts: true, next: false)
    private static let feedforwardTailDenseNext = feedforwardTailKernel(experts: false, next: true)
    private static let feedforwardTailDense = feedforwardTailKernel(experts: false, next: false)
}
