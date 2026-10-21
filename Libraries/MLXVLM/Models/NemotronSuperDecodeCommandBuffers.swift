import MLX
import MLXLMCommon

/// Command-buffer limits for Super's plain decode steps (every row one token;
/// see ``CBv2DecodeCommandBufferLimitsProviding``). MLX's defaults for this GPU
/// commit every 50 operations or 50 MB of distinct inputs, and the inputs
/// include the expert banks, so decode committed after almost every MoE
/// operation and the CPU could not encode ahead of the GPU. The decode scope
/// commits by operation count only: weights do not grow memory, and a decode
/// step's temporaries are small. Prefill chunks and speculative rounds keep the
/// defaults. Results do not depend on the partitioning.
///
/// Full checkpoint (M3 Ultra, EngineV2, greedy, 27-token prompt, 64 tokens):
/// plain decode 43.0 against 34.2 tok/s at one row and 91.4 against 81.9 at
/// eight rows with 100 operations per buffer; 200, 300 and 1000 gave the same
/// rates. MLX's peak during decode, the 8K prefill peak (81.1 GB) and every
/// token (60-token golden run, MTP and batched rows) were unchanged.
public enum NemotronSuperDecodeCommandBuffers {
    /// 100 operations per command buffer; the input bound (1 TiB) never
    /// triggers. A constant: every plain decode step reads it.
    public static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)
}

extension NemotronSuperRecurrentAdapter: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        NemotronSuperDecodeCommandBuffers.limits
    }
}

extension NemotronSuperMTPTextTarget: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        NemotronSuperDecodeCommandBuffers.limits
    }
}
