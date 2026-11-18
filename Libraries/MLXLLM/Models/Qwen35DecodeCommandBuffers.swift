// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon

/// Command-buffer limits for the Qwen 3.5 family's plain decode steps (every row
/// one token; see `MLXLMCommon.CBv2DecodeCommandBufferLimitsProviding`): Qwen 3.5
/// 9B and 35B-A3B, Qwen 3.6 35B-A3B and Qwen 3.8 27B. MLX's defaults commit every
/// 40-50 operations or 40-50 MB of distinct inputs. A step's inputs include the
/// routed expert banks (about 140 MB per 4-bit projection at the 35B-A3B
/// geometry) and each 64 MiB KV segment the history reads, so decode committed
/// after almost every MoE operation, reached MLX's limit of ten command buffers
/// in flight, and waited inside evaluation instead of encoding ahead of the GPU.
/// These steps commit by operation count only: weights and segments add no
/// memory. Prefill chunks and MTP rounds keep the defaults. Results do not
/// depend on the partitioning.
enum Qwen35DecodeCommandBuffers {
    /// Nemotron Super's measured value: 100 operations per command buffer; the
    /// input bound (1 TiB) never triggers.
    static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)

    static let environmentKey = "DARKBLOOM_QWEN35_DECODE_COMMAND_BUFFERS"

    /// Plain decode steps take `limits` unless `DARKBLOOM_QWEN35_DECODE_COMMAND_BUFFERS`
    /// is `0`, `off`, `false` or `no` (trimmed, case-insensitive). Read once.
    static let enabled = enabled(ProcessInfo.processInfo.environment[environmentKey])

    static func enabled(_ raw: String?) -> Bool {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
            !value.isEmpty
        else { return true }
        return !["0", "off", "false", "no"].contains(value)
    }

    /// Tests only: replaces the environment's choice (nil: the environment's).
    /// The engine reads it on its own queue at each step; a test sets it only
    /// while no engine runs.
    nonisolated(unsafe) static var overrideForTesting: Bool?

    static var current: GPU.CommandBufferLimits? {
        (overrideForTesting ?? enabled) ? limits : nil
    }
}
