// Qwen4ExpHCFusionTests.swift
//
// The compiled hyper-connection chains compute what the unfused chains
// computed. Float32 inputs, so both paths carry float32 between the ops and
// the comparison is tight; bf16 activations agree to activation precision.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpHCFusionTests: XCTestCase {

    private func mixer() throws -> (Qwen4ExpGatedResidual, Qwen4ExpTextConfiguration) {
        let config = try Qwen4ExpFixture.configuration()
        MLXRandom.seed(3)
        let mixer = Qwen4ExpGatedResidual(config, useInject: true)
        let replaced = mixer.parameters().flattened().map { key, value -> (String, MLXArray) in
            (key, MLXRandom.normal(value.shape) * 0.3)
        }
        mixer.update(parameters: ModuleParameters.unflattened(replaced))
        eval(mixer)
        return (mixer, config)
    }

    /// The unfused chain, written out as it was before the compiled functions.
    private func reference(
        _ mixer: Qwen4ExpGatedResidual, _ hyper: MLXArray, hc: Int, dims: Int
    ) -> (input: MLXArray, inject: MLXArray) {
        let normed = mixer.hcNorm(hyper)
        var w = silu(mixer.mixDown(normed) / Float(hc))
        w = sigmoid(mixer.mixUp(w))
        let lead = w.shape.dropLast()
        let input = (w.reshaped(lead + [hc, dims]) * normed.reshaped(lead + [hc, dims])).mean(
            axis: -2)
        let inject = 2 * sigmoid(mixer.blockInject!(normed) / Float(hc))
        return (input, inject)
    }

    func testMixWithInjectMatchesTheUnfusedChain() throws {
        let (mixer, config) = try mixer()
        for S in [1, 7] {
            let hyper = MLXRandom.normal([1, S, config.hcCount * config.hiddenSize])
            let (input, residual, inject) = mixer.mixWithInject(hyper)
            let expected = reference(mixer, hyper, hc: config.hcCount, dims: config.hiddenSize)
            eval(input, inject, expected.input, expected.inject)
            XCTAssertTrue(
                allClose(input, expected.input, rtol: 1e-5, atol: 1e-6).item(Bool.self),
                "S=\(S) input")
            XCTAssertTrue(
                allClose(inject, expected.inject, rtol: 1e-5, atol: 1e-6).item(Bool.self),
                "S=\(S) inject")
            XCTAssertTrue(allClose(residual, hyper).item(Bool.self))

            let output = MLXRandom.normal([1, S, config.hiddenSize])
            let injected = qwen4ExpInject(residual: residual, output: output, inject: inject)
            let spread = output[.ellipsis, .newAxis, 0...] * expected.inject[.ellipsis, .newAxis]
            let expectedInjected = hyper + spread.reshaped([1, S, -1])
            eval(injected, expectedInjected)
            XCTAssertEqual(injected.shape, hyper.shape)
            XCTAssertTrue(
                allClose(injected, expectedInjected, rtol: 1e-5, atol: 1e-6).item(Bool.self),
                "S=\(S) inject-add")
        }
    }
}
