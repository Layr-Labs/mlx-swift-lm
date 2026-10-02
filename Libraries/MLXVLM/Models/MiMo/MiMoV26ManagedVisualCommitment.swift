import Foundation
import MLXLLM

extension MiMoV26MultimodalProcessor {
    /// Conservative source-derived commitment, not measured residency: decoded
    /// buffers, pixel working bound, retained patches/features, plus one FP32
    /// frame/block graph. Managed serving synchronously completes each block
    /// and frame through forwardBounded before constructing its successor.
    /// Existing global/OS reserves remain additional.
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
        var visionPeak = 0
        for geometry in plan.visionGeometryByMediaIndex.values {
            visionPeak = max(
                visionPeak, try MiMoV26VisionWorkingSet.frameBytes(geometry, configuration: c))
        }
        bytes = try add(bytes, visionPeak)
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
            peak = max(
                peak,
                try MiMoV26Pixels.workingByteCount(
                    inputElements: elements, frameCount: frames.count, plan: geometry))
        }
        return peak
    }

}
