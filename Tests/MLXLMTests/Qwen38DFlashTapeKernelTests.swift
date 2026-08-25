// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0

import MLX
import XCTest

@testable import MLXLLM

final class Qwen38DFlashTapeKernelTests: XCTestCase {
    func testInnovationTapeUsesSourceFinalThreadgroupGeometry() {
        XCTAssertEqual(qwen38InnovationTapeThreadgroupY, 8)
    }

    private let batch = 1
    private let keyHeads = 16
    private let valueHeads = 48
    private let keyDimension = 128
    private let valueDimension = 128
    private var projectedKeyDimension: Int { keyHeads * keyDimension }
    private var projectedValueDimension: Int { valueHeads * valueDimension }

    private func inputs(steps: Int, seed: UInt64) -> (
        convOutput: MLXArray,
        g: MLXArray, beta: MLXArray, state: MLXArray
    ) {
        MLXRandom.seed(seed)
        let convOutput = MLXRandom.normal([
            batch, steps,
            2 * projectedKeyDimension + projectedValueDimension,
        ]).asType(.bfloat16)
        let g = exp(
            -abs(MLXRandom.normal([batch, steps, valueHeads]).asType(.float32)))
        let beta = sigmoid(
            MLXRandom.normal([batch, steps, valueHeads]).asType(.bfloat16)
        ).asType(.float32)
        let state = MLXRandom.normal(
            [batch, valueHeads, valueDimension, keyDimension]).asType(.float32)
        return (convOutput, g, beta, state)
    }

    private func normalizedQKV(_ convOutput: MLXArray) -> (
        q: MLXArray, k: MLXArray, v: MLXArray
    ) {
        let batch = convOutput.dim(0)
        let steps = convOutput.dim(1)
        let split = MLX.split(
            convOutput,
            indices: [projectedKeyDimension, 2 * projectedKeyDimension],
            axis: -1)
        let q = split[0].reshaped(batch, steps, keyHeads, keyDimension)
        let k = split[1].reshaped(batch, steps, keyHeads, keyDimension)
        let v = split[2].reshaped(batch, steps, valueHeads, valueDimension)
        let invScale = pow(Float(keyDimension), -0.5)
        let dtype = convOutput.dtype
        return (
            MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6),
            MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6),
            v
        )
    }

    func testTapeRecurrenceMatchesStockQwen38Recurrence() throws {
        let input = inputs(steps: 8, seed: 0xDFA5_2001)
        let qkv = normalizedQKV(input.convOutput)
        let expected = gatedDeltaKernel(
            q: qkv.q, k: qkv.k, v: qkv.v,
            g: input.g, beta: input.beta, state: input.state)
        let actual = qwen38GatedDeltaFromConvWithInnovationTape(
            convOutput: input.convOutput,
            g: input.g, beta: input.beta, state: input.state)
        eval(expected.0, expected.1, actual.output, actual.state, actual.tape)

        XCTAssertEqual(actual.tape.dtype, .float32)
        XCTAssertEqual(actual.tape.shape, [batch, 8, valueHeads, valueDimension])
        XCTAssertTrue(
            allClose(actual.output, expected.0, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertTrue(
            allClose(actual.state, expected.1, rtol: 0, atol: 0).item(Bool.self))
    }

    func testInnovationReplayMatchesAcceptedPrefixState() throws {
        let input = inputs(steps: 8, seed: 0xDFA5_2002)
        let qkv = normalizedQKV(input.convOutput)
        let recurrence = qwen38GatedDeltaFromConvWithInnovationTape(
            convOutput: input.convOutput,
            g: input.g, beta: input.beta, state: input.state)

        for acceptedRows in 1 ..< 8 {
            let rows = 0 ..< acceptedRows
            let expected = gatedDeltaKernel(
                q: qkv.q[0..., rows, 0...],
                k: qkv.k[0..., rows, 0...],
                v: qkv.v[0..., rows, 0...],
                g: input.g[0..., rows, 0...],
                beta: input.beta[0..., rows, 0...],
                state: input.state
            ).1
            let actual = qwen38ReplayInnovationTape(
                tape: recurrence.tape,
                convOutput: input.convOutput,
                g: input.g,
                state: input.state,
                steps: acceptedRows)
            eval(expected, actual)
            XCTAssertTrue(
                allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self),
                "acceptedRows=\(acceptedRows)")
        }
    }
}
