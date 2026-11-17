// Copyright © 2026 Eigen Labs / Darkbloom.
//
// The Gated DeltaNet's causal convolution step as one kernel: the retained
// convolution state and the new rows concatenated (`convInput`, which the
// recurrent state and the verify replay tape keep), the depthwise
// convolution over it and its SiLU, written in one launch instead of three
// (concatenate, `depthwise_conv_1d`, the compiled silu). Every value is the
// expression MLX's kernels write: the concatenation copies, the convolution
// accumulates `float(in) * w` over the kernel taps in order and rounds once
// (`depthwise_conv_1d` in mlx/backend/metal/kernels/conv.metal), and the
// SiLU is the compiled `x * sigmoid(x)` with MLX's `Sigmoid` functor and the
// activation dtype at each op. The inputs are read through their strides,
// so the state and the projection's column slice are not copied first.
// Byte parity against the unfused MLX ops is checked per GPU family
// (`ResidualRMSNorm.verifiedFamilies` lists the families measured).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum GatedDeltaConvolution {
    /// Diagnostic only: `DARKBLOOM_GDN_FUSED_CONV=0` keeps the unfused ops.
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GDN_FUSED_CONV"] != "0"
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// `(convInput, silu(conv1d(convInput)))` for `convState [B, K-1, C]`,
    /// `inputs [B, S, C]` and the depthwise weight `[C, K, 1]`; nil where not
    /// covered (the unfused ops are the caller's).
    static func apply(state: MLXArray, inputs: MLXArray, weight: MLXArray) -> (
        convInput: MLXArray, output: MLXArray
    )? {
        guard isEnabled, Device.defaultDevice().deviceType == .gpu,
            ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family()),
            state.ndim == 3, inputs.ndim == 3, weight.ndim == 3, weight.dim(2) == 1,
            state.dim(0) == inputs.dim(0), state.dim(2) == inputs.dim(2),
            weight.dim(0) == inputs.dim(2), weight.dim(1) == state.dim(1) + 1,
            state.dtype == inputs.dtype, weight.dtype == inputs.dtype,
            [.bfloat16, .float16, .float32].contains(inputs.dtype),
            inputs.dim(1) >= 1, inputs.dim(1) <= 8
        else { return nil }
        let batch = inputs.dim(0)
        let steps = inputs.dim(1)
        let channels = inputs.dim(2)
        let taps = weight.dim(1)
        let kept = taps - 1
        let outputs = kernel(
            [state, inputs, weight],
            template: [("T", inputs.dtype), ("KS", taps), ("C", channels), ("S", steps)],
            grid: (channels, kept + steps, batch), threadGroup: (min(256, channels), 1, 1),
            outputShapes: [[batch, kept + steps, channels], [batch, steps, channels]],
            outputDTypes: [inputs.dtype, inputs.dtype])
        return (outputs[0], outputs[1])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "gated_delta_conv_silu", inputNames: ["st", "xin", "w"],
        outputNames: ["conv_input", "conv_out"], source: source, header: header,
        ensureRowContiguous: false)

    // MLX's `Sigmoid` functor (mlx/backend/metal/kernels/unary_ops.h), and row
    // `r` of the concatenation [state; inputs] read through their strides.
    // Shared with `GatedDeltaConvolutionQKNorm` and `GatedDeltaOutputNorm`.
    static let header = """
        struct GdnSigmoid {
          template <typename U>
          U operator()(U x) thread {
            auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
          }
        };

        template <typename T>
        inline T gdn_conv_row(
            const device T* st, const constant int64_t* sts,
            const device T* xin, const constant int64_t* xs,
            int kept, int r, uint b, uint c) {
          return r < kept
              ? st[b * sts[0] + r * sts[1] + c * sts[2]]
              : xin[b * xs[0] + (r - kept) * xs[1] + c * xs[2]];
        }
        """

    private static let source = """
        constexpr int KEPT = KS - 1;
        const uint c = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        const uint b = thread_position_in_grid.z;
        if (c >= uint(C)) { return; }
        conv_input[(size_t(b) * (KEPT + S) + row) * C + c] =
            gdn_conv_row<T>(st, st_strides, xin, xin_strides, KEPT, int(row), b, c);
        if (row >= uint(KEPT)) {
          const int t = int(row) - KEPT;
          float acc = 0.0;
          for (int i = 0; i < KS; ++i) {
            acc += static_cast<float>(gdn_conv_row<T>(st, st_strides, xin, xin_strides, KEPT, t + i, b, c))
                * w[c * w_strides[0] + i * w_strides[1]];
          }
          const T conv = static_cast<T>(acc);
          const T sig = GdnSigmoid()(conv);
          const T activated = conv * sig;
          conv_out[(size_t(b) * S + t) * C + c] = activated;
        }
        """
}

/// The Qwen 3.5 Gated DeltaNet's query and key normalization as one kernel:
/// for every head of q and k (the first `2 * keys` channels of the
/// convolution output, read in place), `scale * rmsNorm(head)` with no norm
/// weight, as `MLXFast.rmsNorm` (MLX's `rms_single_row`: four values a lane,
/// one SIMD sum, `precise::rsqrt(mean + eps)`, the unit weight) and the
/// activation-dtype multiply compute it: four launches become one, the same
/// values. Heads of 32...128 values (four a lane, at most one SIMD group).
enum GatedDeltaQKNorm {
    /// Diagnostic only: `DARKBLOOM_GDN_FUSED_QK_NORM=0` keeps the unfused ops.
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GDN_FUSED_QK_NORM"] != "0"
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// `(q, k)` `[B, S, heads, dim]` from `convOut [B, S, C]` whose channels
    /// `[0, heads*dim)` are q and `[heads*dim, 2*heads*dim)` are k; nil where
    /// not covered.
    static func apply(
        convOut: MLXArray, heads: Int, dim: Int, qScale: MLXArray, kScale: MLXArray, eps: Float
    ) -> (q: MLXArray, k: MLXArray)? {
        guard isEnabled, Device.defaultDevice().deviceType == .gpu,
            ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family()),
            convOut.ndim == 3, dim % 4 == 0, dim >= 4, dim <= 128,
            convOut.dim(2) >= 2 * heads * dim,
            [.bfloat16, .float16, .float32].contains(convOut.dtype),
            qScale.dtype == convOut.dtype, kScale.dtype == convOut.dtype,
            qScale.size == 1, kScale.size == 1
        else { return nil }
        let batch = convOut.dim(0)
        let steps = convOut.dim(1)
        let outputs = kernel(
            [
                convOut, qScale.reshaped([1]), kScale.reshaped([1]),
                ResidualRMSNorm.Constants.shared.epsilon(eps),
            ],
            template: [("T", convOut.dtype), ("D", dim), ("H", heads)],
            grid: (32, heads, batch * steps * 2), threadGroup: (32, 1, 1),
            outputShapes: [[batch, steps, heads, dim], [batch, steps, heads, dim]],
            outputDTypes: [convOut.dtype, convOut.dtype])
        return (outputs[0], outputs[1])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "gated_delta_qk_norm", inputNames: ["conv", "qs", "ks", "eps"],
        outputNames: ["qn", "kn"], source: source, ensureRowContiguous: false)

    // `rms_single_row` (axis D, N_READS = 4, one SIMD group) with the unit
    // weight, then the scale multiply.
    private static let source = """
        constexpr int N_READS = 4;
        const uint lane = thread_index_in_simdgroup;
        const uint head = threadgroup_position_in_grid.y;
        const uint z = threadgroup_position_in_grid.z;
        const uint which = z % 2;            // 0: q, 1: k
        const uint bs = z / 2;               // b * S + s
        const int64_t row_stride = conv_strides[1];
        const int64_t batch_stride = conv_strides[0];
        const uint S_ = conv_shape[1];
        const uint b = bs / S_;
        const uint s = bs % S_;
        auto src = conv + b * batch_stride + s * row_stride
            + (which * H + head) * D * conv_strides[2];
        device T* dst = (which == 0 ? qn : kn) + (size_t(bs) * H + head) * D;
        const T unit = static_cast<T>(1);
        const T scale = which == 0 ? qs[0] : ks[0];
        float thread_x[N_READS];
        float acc = 0;
        const uint base = lane * N_READS;
        for (int i = 0; i < N_READS; i++) {
          thread_x[i] = (base + i < uint(D)) ? static_cast<float>(src[(base + i) * conv_strides[2]]) : 0;
          acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        // `rms_single_row`'s threadgroup step over one SIMD group adds zeros.
        const float inv_mean = metal::precise::rsqrt(acc / D + eps[0]);
        if (base < uint(D)) {
          for (int i = 0; i < N_READS; i++) {
            const T normed = unit * static_cast<T>(thread_x[i] * inv_mean);
            dst[base + i] = scale * normed;
          }
        }
        """
}

/// `GatedDeltaConvolution` and `GatedDeltaQKNorm` as one kernel for decode
/// steps: one thread group per `dim` channels of the convolution writes those
/// channels of `[state; inputs]`, their convolution and SiLU (the same
/// expressions as `GatedDeltaConvolution`), and then either the scaled RMS
/// norm of that q or k head (the same expressions as `GatedDeltaQKNorm`) or
/// the value channels as they are. The convolution output's q and k channels
/// are no longer written and read back, and the two launches on every Gated
/// DeltaNet layer's critical path become one. The same values.
enum GatedDeltaConvolutionQKNorm {
    /// Diagnostic only: `DARKBLOOM_GDN_FUSED_CONV_QK_NORM=0` keeps the two kernels.
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GDN_FUSED_CONV_QK_NORM"] != "0"
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// `(convInput [B, K-1+S, C], q [B, S, heads, dim], k [B, S, heads, dim],
    /// v [B, S, C - 2*heads*dim])` for `state [B, K-1, C]`, `inputs [B, S, C]`
    /// and the depthwise weight `[C, K, 1]`, where the convolution output's
    /// channels `[0, heads*dim)` are q and the next `heads*dim` are k; nil
    /// where not covered (the caller keeps the two kernels or the unfused ops).
    static func apply(
        state: MLXArray, inputs: MLXArray, weight: MLXArray, heads: Int, dim: Int,
        qScale: MLXArray, kScale: MLXArray, eps: Float
    ) -> (convInput: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray)? {
        guard isEnabled, GatedDeltaConvolution.isEnabled, GatedDeltaQKNorm.isEnabled,
            Device.defaultDevice().deviceType == .gpu,
            ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family()),
            state.ndim == 3, inputs.ndim == 3, weight.ndim == 3, weight.dim(2) == 1,
            state.dim(0) == inputs.dim(0), state.dim(2) == inputs.dim(2),
            weight.dim(0) == inputs.dim(2), weight.dim(1) == state.dim(1) + 1,
            state.dtype == inputs.dtype, weight.dtype == inputs.dtype,
            [.bfloat16, .float16, .float32].contains(inputs.dtype),
            inputs.dim(1) >= 1, inputs.dim(1) <= 8,
            dim % 4 == 0, dim >= 4, dim <= 128, heads >= 1,
            inputs.dim(2) > 2 * heads * dim, inputs.dim(2) % dim == 0,
            qScale.dtype == inputs.dtype, kScale.dtype == inputs.dtype,
            qScale.size == 1, kScale.size == 1
        else { return nil }
        let batch = inputs.dim(0)
        let steps = inputs.dim(1)
        let channels = inputs.dim(2)
        let taps = weight.dim(1)
        let kept = taps - 1
        let outputs = kernel(
            [
                state, inputs, weight, qScale.reshaped([1]), kScale.reshaped([1]),
                ResidualRMSNorm.Constants.shared.epsilon(eps),
            ],
            template: [
                ("T", inputs.dtype), ("KS", taps), ("C", channels), ("S", steps), ("D", dim),
                ("H", heads),
            ],
            grid: (32 * (channels / dim), batch, 1), threadGroup: (32, 1, 1),
            outputShapes: [
                [batch, kept + steps, channels], [batch, steps, heads, dim],
                [batch, steps, heads, dim], [batch, steps, channels - 2 * heads * dim],
            ],
            outputDTypes: [inputs.dtype, inputs.dtype, inputs.dtype, inputs.dtype])
        return (outputs[0], outputs[1], outputs[2], outputs[3])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "gated_delta_conv_qk_norm", inputNames: ["st", "xin", "w", "qs", "ks", "eps"],
        outputNames: ["conv_input", "qn", "kn", "vout"], source: source,
        header: GatedDeltaConvolution.header, ensureRowContiguous: false)

    // Lane `l` of thread group `block` owns channels `block * D + 4l ...`.
    // The convolution is `GatedDeltaConvolution`'s expression per channel;
    // the norm is `GatedDeltaQKNorm`'s (`rms_single_row` over one SIMD group,
    // the unit weight, then the scale).
    private static let source = """
        constexpr int KEPT = KS - 1;
        constexpr int N_READS = 4;
        constexpr int VC = C - 2 * H * D;
        const uint lane = thread_index_in_simdgroup;
        const uint block = threadgroup_position_in_grid.x;
        const uint b = threadgroup_position_in_grid.y;
        const uint base = lane * N_READS;
        const bool active = base < uint(D);
        const uint c0 = block * D + base;
        if (active) {
          for (int r = 0; r < KEPT + S; ++r) {
            for (int i = 0; i < N_READS; i++) {
              conv_input[(size_t(b) * (KEPT + S) + r) * C + c0 + i] =
                  gdn_conv_row<T>(st, st_strides, xin, xin_strides, KEPT, r, b, c0 + i);
            }
          }
        }
        const uint which = block < uint(H) ? 0 : (block < uint(2 * H) ? 1 : 2);
        const uint head = which == 0 ? block : block - H;
        const T unit = static_cast<T>(1);
        const T scale = which == 0 ? qs[0] : ks[0];
        for (int t = 0; t < S; ++t) {
          T activated[N_READS];
          float thread_x[N_READS];
          for (int i = 0; i < N_READS; i++) {
            if (active) {
              const uint c = c0 + i;
              float acc = 0.0;
              for (int j = 0; j < KS; ++j) {
                acc += static_cast<float>(gdn_conv_row<T>(st, st_strides, xin, xin_strides, KEPT, t + j, b, c))
                    * w[c * w_strides[0] + j * w_strides[1]];
              }
              const T conv = static_cast<T>(acc);
              const T sig = GdnSigmoid()(conv);
              activated[i] = conv * sig;
            } else {
              activated[i] = static_cast<T>(0);
            }
          }
          if (which == 2) {
            if (active) {
              for (int i = 0; i < N_READS; i++) {
                vout[(size_t(b) * S + t) * VC + (c0 - 2 * H * D) + i] = activated[i];
              }
            }
            continue;
          }
          float acc = 0;
          for (int i = 0; i < N_READS; i++) {
            thread_x[i] = active ? static_cast<float>(activated[i]) : 0;
            acc += thread_x[i] * thread_x[i];
          }
          acc = simd_sum(acc);
          const float inv_mean = metal::precise::rsqrt(acc / D + eps[0]);
          if (active) {
            device T* dst = (which == 0 ? qn : kn) + ((size_t(b) * S + t) * H + head) * D;
            for (int i = 0; i < N_READS; i++) {
              const T normed = unit * static_cast<T>(thread_x[i] * inv_mean);
              dst[base + i] = scale * normed;
            }
          }
        }
        """
}

/// The Qwen 3.5 Gated DeltaNet's gated output norm (`Qwen3NextRMSNormGated`
/// with the SiLU gate) as one kernel for decode steps: per head row,
/// `MLXFast.rmsNorm` (MLX's `rms_single_row` over one SIMD group, as
/// `GatedDeltaQKNorm`) and then the compiled tail `(silu(float(gate)) *
/// float(normed))` rounded to the activation dtype, with MLX's `Sigmoid`
/// functor. Two launches on every Gated DeltaNet layer's critical path
/// become one; the same values. Heads of 4...128 values.
enum GatedDeltaOutputNorm {
    /// Diagnostic only: `DARKBLOOM_GDN_FUSED_OUTPUT_NORM=0` keeps the two kernels.
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GDN_FUSED_OUTPUT_NORM"] != "0"
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// `gatedSiluNormTail(rmsNorm(out, weight), gate)` for `out` and `gate`
    /// `[..., dim]` of one dtype and `weight [dim]` of that dtype; nil where
    /// not covered.
    static func apply(_ out: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float) -> MLXArray? {
        guard isEnabled, Device.defaultDevice().deviceType == .gpu,
            ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family()),
            out.ndim >= 2, out.shape == gate.shape, weight.ndim == 1,
            weight.dim(0) == out.dim(-1), out.dtype == gate.dtype, weight.dtype == out.dtype,
            [.bfloat16, .float16, .float32].contains(out.dtype),
            out.dim(-1) % 4 == 0, out.dim(-1) >= 4, out.dim(-1) <= 128,
            out.size / out.dim(-1) <= 4096
        else { return nil }
        let dim = out.dim(-1)
        let rows = out.size / dim
        return kernel(
            [
                out.reshaped([rows, dim]), gate.reshaped([rows, dim]), weight,
                ResidualRMSNorm.Constants.shared.epsilon(eps),
            ],
            template: [("T", out.dtype), ("D", dim)],
            grid: (32, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [out.shape], outputDTypes: [out.dtype])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "gated_delta_output_norm", inputNames: ["x", "gate", "w", "eps"],
        outputNames: ["y"], source: source, header: GatedDeltaConvolution.header,
        ensureRowContiguous: false)

    private static let source = """
        constexpr int N_READS = 4;
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.y;
        const uint base = lane * N_READS;
        const bool active = base < uint(D);
        auto xr = x + row * x_strides[0];
        auto gr = gate + row * gate_strides[0];
        float thread_x[N_READS];
        float acc = 0;
        for (int i = 0; i < N_READS; i++) {
          thread_x[i] = active ? static_cast<float>(xr[(base + i) * x_strides[1]]) : 0;
          acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        const float inv_mean = metal::precise::rsqrt(acc / D + eps[0]);
        if (active) {
          for (int i = 0; i < N_READS; i++) {
            const T normed = w[(base + i) * w_strides[0]] * static_cast<T>(thread_x[i] * inv_mean);
            const float g = static_cast<float>(gr[(base + i) * gate_strides[1]]);
            const float s = GdnSigmoid()(g);
            const float activated = g * s;
            const float product = activated * static_cast<float>(normed);
            y[size_t(row) * D + base + i] = static_cast<T>(product);
          }
        }
        """
}
