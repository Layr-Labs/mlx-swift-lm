// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Command-buffer limits for GLM's plain decode steps and MTP rounds of
/// decode and verification rows (see ``CBv2DecodeCommandBufferLimitsProviding``).
/// MLX's defaults commit every 50 operations or 50 M distinct input elements.
/// A step's inputs include the expert banks and, for paged rows, each 64 MiB
/// KV segment the history reads, so decode committed after almost every MoE
/// layer and every few segment reads. These steps commit by operation count
/// only: weights and segments add no memory, and their temporaries are priced
/// by the step-scratch envelope. Prefill chunks, and MTP rounds that carry
/// one, keep the defaults. Results do not depend on the partitioning.
public enum GLM5NextDecodeCommandBuffers {
    /// Nemotron Super's measured value: 100 operations per command buffer;
    /// the input bound (1 TiB) never triggers.
    public static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)
}
