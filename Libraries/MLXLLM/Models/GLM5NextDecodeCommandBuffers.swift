// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Command-buffer limits for GLM's plain decode steps (every row one token;
/// see ``CBv2DecodeCommandBufferLimitsProviding``). MLX's defaults commit
/// every 50 operations or 50 M distinct input elements. A step's inputs
/// include the expert banks and, for paged rows, each 64 MiB KV segment the
/// history reads, so decode committed after almost every MoE layer and every
/// few segment reads. The decode scope commits by operation count only:
/// weights and segments add no memory, and a decode step's temporaries are
/// priced by the step-scratch envelope. Prefill chunks and speculative rounds
/// keep the defaults. Results do not depend on the partitioning.
public enum GLM5NextDecodeCommandBuffers {
    /// Nemotron Super's measured value: 100 operations per command buffer;
    /// the input bound (1 TiB) never triggers.
    public static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)
}
