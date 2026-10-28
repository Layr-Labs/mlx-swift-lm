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
    private static let header = """
        struct GdnSigmoid {
          template <typename U>
          U operator()(U x) thread {
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
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
