// Copyright © 2026 Eigen Labs.
import MLX
import MLXNN
import Testing

@testable import MLXVLM

/// MLX peak growth while one image evaluates at the largest admitted grid
/// (104x128 = 13312 patches), compared with the derived activation bound.
/// Run filtered: other suites in the same process share MLX's peak counter.
@Suite(.tags(.integration), .serialized)
struct NemotronSuperImageActivationTests {
    private let rows = 104
    private let columns = 128

    private func composition(
        width: Int, heads: Int, mlpWidth: Int, prefix: Int, positionSide: Int,
        hidden: Int, output: Int, dtype: DType
    ) -> NemotronSuperImageComposition {
        let dimensions = NemotronRadioDimensions(
            width: width, layers: 1, heads: heads, channels: 3, patch: 16,
            positionSide: positionSide, prefix: prefix, mlpWidth: mlpWidth)
        let component = NemotronSuperImageComposition(
            tower: NemotronRadio(dimensions),
            projector: NemotronSuperVisionProjector(
                visionWidth: width, hiddenWidth: hidden, outputWidth: output),
            width: output, imageTokenID: 18, visionDType: dtype)
        component.update(
            parameters: ModuleParameters.unflattened(
                component.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
        eval(component)
        return component
    }

    /// Peak growth above the evaluated, retained normalized input: one image, or
    /// one tubelet when `frames` is 2.
    private func measuredPeak(_ component: NemotronSuperImageComposition, frames: Int = 1) throws
        -> Int
    {
        let pixels = MLXRandom.uniform(-2 ..< 2, [frames, 3, rows * 16, columns * 16])
        eval(pixels)
        Memory.clearCache()
        Memory.peakMemory = 0
        let base = Memory.activeMemory
        let feature =
            try frames == 1
            ? component.imageFeatures(normalizedPixels: pixels)
            : component.videoFeatures(normalizedFrames: pixels)
        eval(feature)
        return Memory.peakMemory - base
    }

    @Test func unfusedAttentionPeakStaysWithinBound() throws {
        // The provider fixture tower: head dimension 2 has no fused kernel.
        let component = composition(
            width: 2, heads: 1, mlpWidth: 3, prefix: 1, positionSide: 32,
            hidden: 3, output: 16, dtype: .float32)
        let bound = component.activationBytes(patchRows: rows, patchColumns: columns)
        let peak = try measuredPeak(component)
        let tokens = rows * columns + 1
        #expect(peak >= tokens * tokens * 4, "score matrix not materialized: \(peak)")
        #expect(peak <= bound, "peak \(peak) bound \(bound)")
    }

    @Test func fusedAttentionPeakStaysWithinBound() throws {
        // Head dimension 80, as in the pinned tower, takes the fused kernel.
        let component = composition(
            width: 160, heads: 2, mlpWidth: 640, prefix: 10, positionSide: 128,
            hidden: 2560, output: 512, dtype: .bfloat16)
        let bound = component.activationBytes(patchRows: rows, patchColumns: columns)
        let peak = try measuredPeak(component)
        let tokens = rows * columns + 10
        #expect(peak < 2 * tokens * tokens * 2, "scores materialized: \(peak)")
        #expect(peak <= bound, "peak \(peak) bound \(bound)")
    }

    @Test func fusedAttentionTubeletPeakStaysWithinBound() throws {
        let component = composition(
            width: 160, heads: 2, mlpWidth: 640, prefix: 10, positionSide: 128,
            hidden: 2560, output: 512, dtype: .bfloat16)
        let bound = component.tubeletActivationBytes(patchRows: rows, patchColumns: columns)
        let peak = try measuredPeak(component, frames: 2)
        #expect(peak <= bound, "peak \(peak) bound \(bound)")
    }

    @Test func pinnedTubeletBoundAddsOnlyTheSecondFramesPixels() {
        // A 672 x 384 frame grid: 1008 patches of 3 x 16 x 16 BF16 samples, cast
        // and rearranged once more for the tubelet's second frame.
        #expect(
            NemotronSuperImageComposition.pinnedTubeletActivationBytes(
                patchRows: 24, patchColumns: 42)
                - NemotronSuperImageComposition.pinnedActivationBytes(
                    patchRows: 24, patchColumns: 42) == 2 * 1008 * 768 * 2)
        #expect(
            NemotronSuperImageComposition.pinnedTubeletActivationBytes(
                patchRows: 105, patchColumns: 128) == .max)
    }

    @Test func pinnedBoundAtTheLargestGrid() {
        #expect(NemotronSuperImageComposition.fusedAttention(headDimension: 80, queryLength: 13322))
        #expect(!NemotronSuperImageComposition.fusedAttention(headDimension: 2, queryLength: 13313))
        #expect(
            !NemotronSuperImageComposition.fusedAttention(headDimension: 256, queryLength: 13322))
        // Embeddings 431,565,824 + block 886,712,320 + projector 776,994,816 bytes.
        #expect(
            NemotronSuperImageComposition.pinnedActivationBytes(
                patchRows: rows, patchColumns: columns)
                == 2_095_272_960)
        #expect(
            NemotronSuperImageComposition.pinnedActivationBytes(patchRows: 105, patchColumns: 128)
                == .max)
    }
}
