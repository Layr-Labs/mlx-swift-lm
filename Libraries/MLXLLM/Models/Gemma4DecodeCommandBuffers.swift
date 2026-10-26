// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Bounded-width CBv2 steps can encode further ahead without changing their
/// operations. The engine retains its existing row-width and memory checks.
enum Gemma4DecodeCommandBuffers {
    // The second field is MLX's existing input-size threshold: despite its
    // API name, the current backend counts M distinct elements, not MiB.
    // Both thresholds are checked after a primitive, so these are commit
    // heuristics rather than strict upper bounds on operations or memory.
    static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 512)

    static func resolve(
        _ config: Gemma4TextConfiguration,
        device: DeviceType = Device.defaultDevice().deviceType
    ) -> GPU.CommandBufferLimits? {
        #if canImport(Metal)
        // Keep this a scalar configuration check on the serving path.
        // The VLM loader merges root quantization into the text config.
        guard device == .gpu,
            gemma4SupportsProductionExpertTopology(config),
            config.quantizationBits == 4, config.quantizationGroupSize == 64,
            config.quantizationMode == .affine
        else { return nil }
        return limits
        #else
        return nil
        #endif
    }
}

extension Gemma4TextModel: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        Gemma4DecodeCommandBuffers.resolve(configuration)
    }

    public var speculativeCommandBufferLimits: GPU.CommandBufferLimits? {
        Gemma4DecodeCommandBuffers.resolve(configuration)
    }
}
