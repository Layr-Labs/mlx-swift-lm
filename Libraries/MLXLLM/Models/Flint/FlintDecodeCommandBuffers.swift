// Flint decode command-buffer limits. See ThirdPartyNotices/Flint.md for the
// model's sources (Prince Canuma/mlx-vlm, MIT; Recursive rec, Apache-2.0).
import MLX
import MLXLMCommon

/// Command-buffer limits for Flint's plain decode steps (every row one token;
/// see ``CBv2DecodeCommandBufferLimitsProviding``). MLX's per-device defaults
/// commit every 20 (phone), 40 (base/Pro) or 50 (Max/Ultra) operations, or
/// once a buffer's distinct inputs and outputs exceed 40-50 "MB" — counted in
/// ELEMENTS, so a single routed-expert bank or `lm_head` read crosses it and
/// decode committed after nearly every MoE layer. The CPU then cannot encode
/// ahead of the GPU.
///
/// The decode scope commits by operation count only. Weights add no memory,
/// and a decode step's temporaries are small (the largest is the FP32 logits
/// row), so the input bound is disabled on every GPU family. The operation
/// bound is the value Nemotron Super measured across M-series machines
/// (100, with 200/300/1000 no faster); it is a constant, independent of the
/// device architecture suffix, so M1–M4 and M5 run the same partitioning.
/// Prefill chunks keep MLX's defaults. Results never depend on the
/// partitioning: every kernel and its arithmetic are unchanged.
public enum FlintDecodeCommandBuffers {
    public static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)
}

extension FlintModel: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        FlintDecodeCommandBuffers.limits
    }
}
