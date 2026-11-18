// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon

/// Command-buffer limits for the Qwen 3.5 family's plain (non-speculative) decode
/// steps (every row one token; see `MLXLMCommon.CBv2DecodeCommandBufferLimitsProviding`)
/// on the single-Mac CBv2 engine: every model it serves as `Qwen35Model` or its MoE
/// subclass (Qwen 3.5 9B and 35B-A3B, Qwen 3.6 35B-A3B, Qwen 3.8 27B), except the
/// Prism Hadamard text pack, which keeps MLX's defaults. MLX's defaults commit after
/// 40-50 kernel dispatches or 40-50 MB of distinct inputs. A step's inputs include
/// the routed expert banks (about 140 MB per 4-bit projection at the 35B-A3B
/// geometry) and each 64 MiB KV segment the history reads, so decode committed after
/// almost every MoE operation, reached MLX's limit of ten command buffers in flight,
/// and waited inside evaluation instead of encoding ahead of the GPU. These steps
/// commit by dispatch count only: weights and segments add no memory. Prefill chunks
/// and MTP rounds keep the defaults. Results do not depend on the partitioning.
/// Cluster Qwen sessions set their own (identical) limits; this switch doesn't reach
/// them.
enum Qwen35DecodeCommandBuffers {
    /// Nemotron Super's measured value: about 100 kernel dispatches per command
    /// buffer; the input bound (1 TiB) never triggers.
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

    /// Tests only: replaces the environment's choice (nil: the environment's). Set
    /// it only from a `.serialized` suite while no engine runs; the engine reads it
    /// on its own queue at each step.
    nonisolated(unsafe) static var overrideForTesting: Bool?

    static var current: GPU.CommandBufferLimits? {
        (overrideForTesting ?? enabled) ? limits : nil
    }
}

extension Qwen35Model: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        // The Prism Hadamard pack (Bonsai) wasn't measured with these limits.
        self is PrismHadamardQwen35TextModel ? nil : Qwen35DecodeCommandBuffers.current
    }
}
