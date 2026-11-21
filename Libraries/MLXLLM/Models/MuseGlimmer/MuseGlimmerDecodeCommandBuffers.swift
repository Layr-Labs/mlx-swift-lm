// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Command-buffer limits for plain decode steps (every row one token). MLX's
/// per-device defaults commit after 40 to 50 operations or once a buffer's
/// inputs pass a size bound that one projection's weight read already crosses,
/// so the CPU cannot encode ahead of the GPU. A decode step's temporaries are
/// small, so the scope commits by operation count only: the value Nemotron
/// Super and Flint use. Prefill chunks keep MLX's defaults. The partitioning
/// never changes a kernel or its arithmetic.
public enum MuseGlimmerDecodeCommandBuffers {
    public static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)
}

extension MuseGlimmerTextModel: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        MuseGlimmerDecodeCommandBuffers.limits
    }
}
