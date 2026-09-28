// Copyright © 2026 Eigen Labs. Decoded-PCM work commitment, not measured residency.

import Foundation
import MLXLLM

/// Additional audio work only. The issued audio profile must add the existing
/// decoded/visual/features commitment, preserve the separate codec load charge,
/// and leave target KV and existing OS/activation reserves unchanged.
public enum MiMoV26ManagedAudioCommitment {
    public static func additionalBytes(input: MiMoV26AudioInputPlan,
                                       patchConfiguration c: MiMoV26AudioConfiguration,
                                       limits: MiMoV26AudioPatchLimits) throws -> Int {
        guard let clips = input.pcmDescriptors, !clips.isEmpty,
              input.codeFrameCounts.count == clips.count,
              c.channels == input.configuration.quantizers, c.groupSize == 4,
              c.hiddenSize > 0, c.hiddenSize <= 1024, c.layers > 0, c.layers <= 6,
              c.queryHeads > 0, c.queryHeads <= 16, c.outputHiddenSize > 0,
              c.outputHiddenSize <= 4096, clips.count <= limits.maximumClips,
              input.limits.rvqTileFrames > 0 else { throw MiMoV26AudioSidecarError.invalidBinding }
        func add(_ a: Int, _ b: Int) throws -> Int {
            try MiMoV26AudioChecked.add(a, b, "managed audio commitment")
        }
        func mul(_ values: Int...) throws -> Int {
            try MiMoV26AudioChecked.product(values, "managed audio commitment")
        }
        guard case .number(let rawIntermediate) = c.rawFields["input_local_intermediate_size"] else {
            throw MiMoV26AudioSidecarError.invalidConfiguration
        }
        let value = NSDecimalNumber(decimal: rawIntermediate).int64Value
        guard rawIntermediate == Decimal(value), let intermediate = Int(exactly: value),
              intermediate > 0, intermediate <= 4096 else { throw MiMoV26AudioSidecarError.invalidConfiguration }
        let frames = try input.codeFrameCounts.reduce(0, add)
        let patches = try input.codeFrameCounts.reduce(0) {
            try add($0, MiMoV26AudioChecked.ceilDivide($1, c.groupSize))
        }
        guard input.codeFrameCounts.allSatisfy({ $0 > 0 }), frames <= limits.maximumFrames,
              patches == input.totalPatches, patches <= limits.maximumPatches else {
            throw MiMoV26AudioSidecarError.invalidBinding
        }
        // Same shape equations as AudioPatchPlan, without inventing code IDs.
        let paddedFrames = try mul(patches, c.groupSize)
        let codes = try mul(paddedFrames, c.channels)
        let layer = try add(mul(5, c.hiddenSize), mul(3, intermediate))
        let hidden = try mul(paddedFrames, add(mul(2, c.hiddenSize), mul(c.layers, layer)))
        let scores = try mul(patches, c.queryHeads, c.groupSize, c.groupSize, c.layers)
        let projectionHidden = try mul(c.groupSize, c.hiddenSize, 4)
        let projection = try mul(patches, add(projectionHidden, c.outputHiddenSize))
        let patchWork = try add(add(codes, hidden), add(scores, projection))
        guard patchWork <= limits.maximumWorkingElements else { throw MiMoV26AudioSidecarError.insufficientReservation }

        // Price all possible RVQ tiles, including retained lazy graph nodes;
        // do not reinterpret the component's per-tile bound as a physical peak.
        let tiles = try input.codeFrameCounts.reduce(0) {
            try add($0, MiMoV26AudioChecked.ceilDivide($1, input.limits.rvqTileFrames))
        }
        let tableRows = try input.configuration.codebookSizes.reduce(0, add)
        let tableElements = try mul(tableRows, input.configuration.hiddenSize)
        var bytes = try mul(input.workingElementUpperBound, max(1, tiles), 16)
        bytes = try add(bytes, mul(tableElements, max(1, tiles), 16))
        bytes = try add(bytes, mul(patchWork, 16))
        bytes = try add(bytes, mul(try add(mul(frames, c.channels), codes), 8))
        // Conservative node/page slack for frontend, per-segment transformer,
        // per-tile RVQ and vectorized patch graph. Qualification must measure it.
        var nodes = try mul(clips.count, 1024)
        nodes = try add(nodes, mul(input.segments.count, input.configuration.layers, 256))
        nodes = try add(nodes, mul(tiles, input.configuration.quantizers, 128))
        nodes = try add(nodes, add(mul(c.layers, 256), 512))
        return try add(bytes, mul(nodes, 16_384))
    }
}
