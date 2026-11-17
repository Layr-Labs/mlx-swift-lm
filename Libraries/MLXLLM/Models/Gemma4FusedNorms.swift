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
import os

enum Gemma4FusedNorms {
    /// `DARKBLOOM_GEMMA4_FUSED_NORMS=0` keeps the unfused ops (the rollback).
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_FUSED_NORMS"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// GPU families where the kernels were checked against MLX's norm on
    /// real weights (M5). Others keep the unfused ops.
    static let verifiedFamilies: Set<Int> = [17]

    /// Tests only: the mean form the kernels use (nil: the measured one).
    @TaskLocal static var meanFormOverride: RMSNormMeanForm?

    private static let logger = Logger(subsystem: "darkbloom", category: "Gemma4FusedNorms")

    /// Whether the kernels cover rows of `width` in `dtype` (MLX's looped norm,
    /// above 4096, is not copied): a verified family, a measured mean form,
    /// and the kernels' own check against the unfused ops (`Readiness`).
    static func supports(width: Int, dtype: DType) -> Bool {
        guard isEnabled, (1 ... 4096).contains(width),
            Device.defaultDevice().deviceType == .gpu,
            verifiedFamilies.contains(AffineQuantizedVectors.family()),
            RMSNormMeanForm.measured(width: width, dtype: dtype) != nil
        else { return false }
        return Readiness.shared.ready(width: width, dtype: dtype)
    }

    private static func meanForm(_ width: Int, _ dtype: DType) -> Int {
        if let meanFormOverride { return meanFormOverride.rawValue }
        guard let form = RMSNormMeanForm.measured(width: width, dtype: dtype) else {
            preconditionFailure("Gemma4FusedNorms: called where `supports` is false")
        }
        return form.rawValue
    }

    /// Each (width, dtype)'s kernels run once on the mean-form calibration rows
    /// and must write the unfused ops' bytes. A Metal error or any difference
    /// keeps the unfused ops for that key, logged once.
    final class Readiness: @unchecked Sendable {
        static let shared = Readiness()

        private struct Key: Hashable {
            let width: Int
            let dtype: DType
        }

        private let lock = NSLock()
        private var results: [Key: Bool] = [:]

        func ready(width: Int, dtype: DType) -> Bool {
            lock.withLock {
                let key = Key(width: width, dtype: dtype)
                if let known = results[key] { return known }
                let failure = Self.check(width: width, dtype: dtype)
                if let failure {
                    Gemma4FusedNorms.logger.error(
                        "gemma4_fused_norms_unavailable width=\(width) dtype=\(String(describing: dtype), privacy: .public) fallback=unfused reason=\(failure, privacy: .public)"
                    )
                }
                results[key] = failure == nil
                return failure == nil
            }
        }

        /// Nil when all six kernels write the unfused ops' bytes on the rows
        /// that separate the two mean forms (and on plain rows), else why not.
        static func check(width: Int, dtype: DType) -> String? {
            do {
                return try withError { () -> String? in
                    let (separating, plain) = RMSNormMeanForm.Calibration.rows(
                        width: width, dtype: dtype, separate: width & (width - 1) != 0)
                    let rows = separating + plain
                    let x = MLXArray(rows.flatMap { $0 }, [rows.count, 1, width]).asType(dtype)
                    let y = MLXArray(rows.reversed().flatMap { $0 }, [rows.count, 1, width])
                        .asType(dtype)
                    return mismatch(x, y, width: width, dtype: dtype)
                }
            } catch {
                return "\(error)"
            }
        }

        /// The first kernel output that differs from the unfused ops, with
        /// unit weights so each norm's rounding of the mean shows.
        static func mismatch(_ x: MLXArray, _ y: MLXArray, width: Int, dtype: DType) -> String? {
            let eps = RMSNormMeanForm.Calibration.epsilon
            let one = MLXArray.ones([width], dtype: dtype)
            let zero = MLXArray.zeros(x.shape, dtype: dtype)
            let scalar = MLXArray([Float(1)]).asType(dtype)
            func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
                eval(a, b)
                return a.dtype == b.dtype && a.shape == b.shape
                    && a.asData(access: .copy).data == b.asData(access: .copy).data
            }
            for count in [1, 3] {
                let tail = attentionTail(
                    x, residual: zero, postAttention: one,
                    sumNorms: Array(repeating: one, count: count), eps: eps)
                let sum = zero + MLXFast.rmsNorm(x, weight: one, eps: eps)
                guard same(tail.sum, sum) else { return "attentionTail(\(count)) sum" }
                for norm in tail.norms
                where !same(norm, MLXFast.rmsNorm(sum, weight: one, eps: eps)) {
                    return "attentionTail(\(count)) norm"
                }
            }
            for experts in [true, false] {
                let s =
                    experts
                    ? MLXFast.rmsNorm(x, weight: one, eps: eps)
                        + MLXFast.rmsNorm(y, weight: one, eps: eps)
                    : x
                let out = (zero + MLXFast.rmsNorm(s, weight: one, eps: eps)) * scalar
                for next in [true, false] {
                    let tail = feedforwardTail(
                        dense: x, experts: experts ? (y, one, one) : nil, residual: zero,
                        post: one, layerScalar: scalar, next: next ? one : nil, eps: eps)
                    guard same(tail.out, out) else { return "feedforwardTail out" }
                    if next, let n = tail.next,
                        !same(n, MLXFast.rmsNorm(out, weight: one, eps: eps))
                    {
                        return "feedforwardTail next"
                    }
                }
            }
            return nil
        }
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
