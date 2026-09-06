// Qwen4ExpGDNFusionTests.swift
//
// The compiled GDN gate chain computes what the unfused chain computed.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpGDNFusionTests: XCTestCase {
    func testGatesMatchTheChain() throws {
        MLXRandom.seed(8)
        let a = MLXRandom.normal([1, 3, 4]).asType(.bfloat16)
        let b = MLXRandom.normal([1, 3, 4]).asType(.bfloat16)
        let aLog = MLXRandom.normal([4])
        let dtBias = MLXRandom.normal([4])
        let out = Qwen4ExpGDNKernels.gates([a, b, aLog, dtBias])
        let decay = computeGatedDeltaG(aLog, a, dtBias)
        let beta = sigmoid(b).asType(.float32)
        eval(out, decay, beta)
        XCTAssertTrue(allClose(out[0], decay, rtol: 1e-5, atol: 1e-6).item(Bool.self))
        XCTAssertTrue(allClose(out[1], beta, rtol: 1e-5, atol: 1e-6).item(Bool.self))
    }
}
