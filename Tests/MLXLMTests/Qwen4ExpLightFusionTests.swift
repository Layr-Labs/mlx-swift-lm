// Qwen4ExpLightFusionTests.swift
//
// The compiled light chains (MoE tail, attention gate, partial rope) compute
// what the unfused chains computed. Float32 inputs, tight tolerance.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpLightFusionTests: XCTestCase {
    private func assertClose(_ a: MLXArray, _ b: MLXArray, _ what: String) {
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape, what)
        XCTAssertTrue(allClose(a, b, rtol: 1e-5, atol: 1e-6).item(Bool.self), what)
    }

    func testMoETail() {
        MLXRandom.seed(1)
        for S in [1, 4] {
            let experts = MLXRandom.normal([1, S, 10, 64])
            let weights = MLXRandom.uniform(0 ..< 1, [1, S, 10])[.ellipsis, .newAxis]
            let gate = MLXRandom.normal([1, S, 1])
            let shared = MLXRandom.normal([1, S, 64])
            let expected = (experts * weights).sum(axis: -2) + sigmoid(gate) * shared
            assertClose(
                qwen4ExpMoETail(experts, weights, gate, shared), expected, "moe tail S=\(S)")
        }
    }

    func testAttentionGate() {
        MLXRandom.seed(2)
        let out = MLXRandom.normal([1, 3, 96])
        let gate = MLXRandom.normal([1, 3, 96])
        assertClose(qwen4ExpAttentionGate(out, gate), out * sigmoid(gate), "attention gate")
    }

    func testRopeRotate() {
        MLXRandom.seed(3)
        for S in [1, 5] {
            let d = 16
            let x = MLXRandom.normal([1, 2, S, 24])
            let c = MLXRandom.normal([1, 1, S, d])
            let s = MLXRandom.normal([1, 1, S, d])
            let rotated = x[.ellipsis, ..<d]
            let x1 = rotated[.ellipsis, ..<(d / 2)]
            let x2 = rotated[.ellipsis, (d / 2)...]
            let swapped = concatenated([-x2, x1], axis: -1)
            let expected = rotated * c + swapped * s
            let halves = qwen4ExpRopeRotate([
                x1, x2, c[.ellipsis, ..<(d / 2)], c[.ellipsis, (d / 2)...],
                s[.ellipsis, ..<(d / 2)], s[.ellipsis, (d / 2)...],
            ])
            assertClose(concatenated(halves, axis: -1), expected, "rope S=\(S)")
            assertClose(
                qwen4ExpRopePartial(x, cos: c, sin: s),
                concatenated([expected, x[.ellipsis, d...]], axis: -1), "rope partial S=\(S)")
        }
    }
}
