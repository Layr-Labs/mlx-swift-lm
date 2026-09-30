import Foundation
import MLXLLM

extension MiMoV26MultimodalProcessor {
    /// Conservative source-derived commitment, not measured residency: decoded
    /// buffers, pixel working bound, retained patches/features, plus the entire
    /// FP32 vision graph (all blocks, full-frame score upper bound even for
    /// tiled local attention). Existing global/OS reserves remain additional.
    /// Never represents target KV, which the bridge already charges.
    func managedVisualCommitmentBytes(_ plan: MiMoV26MultimodalPlan) throws -> Int {
        guard plan.audioPlan == nil else {
            throw MiMoV26MultimodalError.incompatiblePlan
        }
        return try managedBaseCommitmentBytes(plan)
    }
    func managedAudioCommitmentBytes(_ plan: MiMoV26MultimodalPlan) throws -> Int {
        try checkOwner()
        guard let sidecar = audioSidecar, audioCodec === sidecar.codec else {
            throw MiMoV26MultimodalError.missingAudioCodec
        }
        let base = try managedBaseCommitmentBytes(plan)
        guard let input = plan.audioPlan else { return base }
        guard let patch = configuration.audio else { throw MiMoV26MultimodalError.incompatiblePlan }
        let audio = try MiMoV26ManagedAudioCommitment.additionalBytes(
            input: input,
            patchConfiguration: patch, limits: limits.audioPatch)
        return try MiMoV26AudioChecked.add(base, audio, "managed decoded audio commitment")
    }
    private func managedBaseCommitmentBytes(_ plan: MiMoV26MultimodalPlan) throws -> Int {
        guard let c = configuration.vision else { throw MiMoV26MultimodalError.incompatiblePlan }
        func mul(_ a: Int, _ b: Int) throws -> Int {
            try MiMoV26AudioChecked.product([a, b], "managed visual commitment")
        }
        func add(_ a: Int, _ b: Int) throws -> Int {
            try MiMoV26AudioChecked.add(a, b, "managed visual commitment")
        }
        var bytes = try add(Self.managedPixelWorkingBytes(plan), mul(plan.decodedElements, 4))
        bytes = try add(bytes, mul(plan.patchElements, 8))
        bytes = try add(bytes, mul(plan.featureElements, 16))
        for geometry in plan.visionGeometryByMediaIndex.values {
            let n = geometry.patchCount
            // Covers projection/rotary/normalization/gated-MLP/merger
            // temporaries and layout indices with FP32 widths and slack.
            let widths = try add(try mul(c.hiddenSize, 64), try mul(c.intermediateSize, 16))
            let activations = try mul(try mul(n, widths), 4)
            // Attention is independent for each temporal grid, never across
            // the concatenated video. Keep every frame's graph charged.
            let framePatches = try mul(geometry.gridH, geometry.gridW)
            let scores = try Self.managedVisionScoreBytes(geometry, queryHeads: c.queryHeads)
            bytes = try add(bytes, try mul(try add(activations, scores), try add(c.depth, 2)))
            // 16-KiB native-load profile: reserve per-node rounding/slack in
            // addition to logical tensor bytes. The reviewed vision expression
            // graph has fewer than 256 non-tile nodes/block and 96 nodes/tile;
            // full-frame attention uses fewer tiles than this local upper bound.
            let tiles = try mul(geometry.gridT, try add(framePatches, 127) / 128)
            let nodes = try add(64, try mul(c.depth, try add(256, try mul(96, tiles))))
            bytes = try add(bytes, try mul(nodes, 16384))
        }
        for part in plan.parts {
            if case .audiovisual = part.content, let geometry = part.geometry {
                // Two bounded view/node metadata allowances per AV unit;
                // whole audio feature/data backing is separately kept above
                // and in ManagedAudioCommitment, never priced as slice-only.
                bytes = try add(bytes, try mul(geometry.timestampCount, 32768))
            }
        }
        return bytes
    }

    static func managedPixelWorkingBytes(_ plan: MiMoV26MultimodalPlan) throws -> Int {
        var peak = 0
        for part in plan.parts {
            guard let geometry = part.geometry else { continue }
            let frames: [MiMoV26Pixels.DecodedRGB]
            switch part.content {
            case .image(let frame): frames = [frame]
            case .silentVideo(let video): frames = video.frames
            case .audiovisual(let video): frames = video.frames
            case .audio, .text: continue
            }
            let elements = try frames.reduce(0) {
                try MiMoV26AudioChecked.add($0, $1.planarRGB.count, "managed pixel elements")
            }
            peak = max(peak, try MiMoV26Pixels.workingByteCount(
                inputElements: elements, frameCount: frames.count, plan: geometry))
        }
        return peak
    }

    static func managedVisionScoreBytes(_ geometry: MiMoV26MediaGeometry.Plan,
                                        queryHeads: Int) throws -> Int {
        try MiMoV26AudioChecked.product(
            [geometry.gridT, geometry.gridH, geometry.gridW,
             geometry.gridH, geometry.gridW, queryHeads, 16], "managed vision scores")
    }

}
