// Copyright © 2026 Apple Inc.

import MLX
@testable import MLXLMCommon
import MLXRandom
import Testing

@Suite("Gemma4 expert kernels")
struct Gemma4ExpertKernelTests {
    @Test func fusedDownKernelMatchesDequantizedWeightedGatherMM() throws {
        MLXRandom.seed(2026)

        let rows = 7
        let topK = 8
        let numExperts = 11
        let hiddenDims = 128
        let outputDims = 256
        let groupSize = 64
        let bits = 4

        let hidden = MLXRandom.normal([rows, topK, hiddenDims]).asType(.bfloat16)
        let weights = MLX.softmax(
            MLXRandom.normal([rows, topK]).asType(.bfloat16),
            axis: -1,
            precise: true
        ).asType(.bfloat16)
        let routePattern = (0 ..< rows * topK).map { Int32(($0 * 7 + 3) % numExperts) }
        let indices = MLXArray(routePattern, [rows, topK])

        let down = MLXRandom.normal([numExperts, outputDims, hiddenDims]).asType(.bfloat16)
        let (downWeight, downScales, downBiases) = MLX.quantized(
            down,
            groupSize: groupSize,
            bits: bits,
            mode: .affine
        )
        let biases = try #require(downBiases)

        let hiddenFlat = hidden.reshaped(rows * topK, 1, hiddenDims).contiguous()
        let indicesFlat = indices.flattened()
        let expectedRoutes = MLX.gatherQuantizedMM(
            hiddenFlat,
            downWeight,
            scales: downScales,
            biases: biases,
            rhsIndices: indicesFlat,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: false
        )
        let expected = (
            MLX.squeezed(expectedRoutes, axis: -2)
                .reshaped(rows, topK, outputDims)
                * MLX.expandedDimensions(weights, axis: -1)
        ).sum(axis: -2)
        let dequantizedDown = MLX.dequantized(
            downWeight,
            scales: downScales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: .float32
        )
        let dequantizedRoutes = MLX.gatherMM(
            hiddenFlat.asType(.float32),
            dequantizedDown.swappedAxes(-1, -2),
            rhsIndices: indicesFlat,
            sortedIndices: false
        )
        let dequantizedExpected = (
            MLX.squeezed(dequantizedRoutes, axis: -2)
                .reshaped(rows, topK, outputDims)
                * MLX.expandedDimensions(weights.asType(.float32), axis: -1)
        ).sum(axis: -2).asType(.bfloat16)

        let actual = SwitchGLU.gemma4FusedDownKernelForTesting(
            hidden: hidden.contiguous(),
            indices: indices.contiguous(),
            weights: weights.contiguous(),
            downWeight: downWeight.contiguous(),
            downScales: downScales.contiguous(),
            downBiases: biases.contiguous(),
            rows: rows,
            topK: topK,
            hiddenDims: hiddenDims,
            outputDims: outputDims,
            dtype: hidden.dtype
        )

        eval(expected, dequantizedExpected, actual)

        #expect(allClose(actual, dequantizedExpected, rtol: 5e-2, atol: 5e-2).item(Bool.self))
    }
}
