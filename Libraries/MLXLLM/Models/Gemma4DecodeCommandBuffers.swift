// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon

/// Command-buffer limits for Gemma 4's plain decode steps (every row one token)
/// and its MTP rounds whose rows each carry at most 8 tokens (decode and
/// verification rows, or a short prompt chunk; see
/// `MLXLMCommon.CBv2DecodeCommandBufferLimitsProviding`). MLX's defaults commit
/// every 40-50 operations or 40-50 M distinct input elements. A step's inputs
/// include the weights and each 64 MiB KV segment the history reads, so decode
/// committed many times per step, reached MLX's limit of ten command buffers in
/// flight, and waited inside evaluation instead of encoding ahead of the GPU.
/// These steps commit by operation count only: weights and segments add no
/// memory. Longer prefill chunks, and MTP rounds with a longer row, keep the
/// defaults. Results do not depend on the partitioning.
enum Gemma4DecodeCommandBuffers {
    /// Nemotron Super's measured value: 100 operations per command buffer; the
    /// input bound (1 TiB) never triggers.
    static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)

    /// MTP rounds take `limits` unless `DARKBLOOM_GEMMA4_MTP_COMMAND_BUFFERS`
    /// is set to something other than 1, true, yes or on.
    static let speculativeEnabled = speculativeEnabled(
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_MTP_COMMAND_BUFFERS"])

    static func speculativeEnabled(_ raw: String?) -> Bool {
        guard let raw else { return true }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }

    /// Tests only: replaces the environment's choice for MTP rounds (nil: the
    /// environment's). The engine reads it on its own queue at each round; a
    /// test sets it only while no engine runs, and test runs that read the
    /// environment's value must not run in parallel with it (CI runs shards
    /// with `--no-parallel`).
    nonisolated(unsafe) static var speculativeOverrideForTesting: Bool?
}

extension Gemma4TextModel: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        Gemma4DecodeCommandBuffers.limits
    }

    public var speculativeCommandBufferLimits: GPU.CommandBufferLimits? {
        let enabled =
            Gemma4DecodeCommandBuffers.speculativeOverrideForTesting
            ?? Gemma4DecodeCommandBuffers.speculativeEnabled
        return enabled ? Gemma4DecodeCommandBuffers.limits : nil
    }
}
