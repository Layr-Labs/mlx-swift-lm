// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Command-buffer limits for GPT-OSS's plain decode steps (every row one token;
/// see `MLXLMCommon.CBv2DecodeCommandBufferLimitsProviding`). MLX's defaults
/// commit every 40-50 operations or 40-50 M distinct input elements. A step's
/// inputs include the expert banks and each 64 MiB KV segment the history reads,
/// so decode committed after almost every MoE layer, reached MLX's limit of ten
/// command buffers in flight, and waited inside evaluation instead of encoding
/// ahead of the GPU. These steps commit by operation count only: weights and
/// segments add no memory. Prefill chunks keep the defaults. Results do not
/// depend on the partitioning.
enum GPTOSSDecodeCommandBuffers {
    /// Nemotron Super's measured value: 100 operations per command buffer; the
    /// input bound (1 TiB) never triggers.
    static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)
}

extension GPTOSSModel: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        GPTOSSDecodeCommandBuffers.limits
    }
}
