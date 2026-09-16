// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpOutputGateTests: XCTestCase {
    private func configuration(gate: String) -> Qwen4ExpTextConfiguration {
        var args = Qwen4ExpTextConfiguration()
        args.hiddenSize = 16
        args.hiddenLayers = 1
        args.layerTypes = ["linear_attention"]
        args.linearNumValueHeads = 2
        args.linearNumKeyHeads = 1
        args.linearKeyHeadDim = 8
        args.linearValueHeadDim = 8
        args.hcCount = 2
        args.hcLowrank = 4
        args.pleLayerIds = []
        args.numExperts = 2
        args.numExpertsPerTok = 1
        args.sharedExpertIntermediateSize = 16
        args.moeIntermediateSize = 16
        args.outputGateType = gate
        return args
    }

    func testSupportedNondefaultGateSurvivesDecodeAndRoundTrip() throws {
        for gate in ["sigmoid", "silu"] {
            let original = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
                from: Data("{\"output_gate_type\":\"\(gate)\"}".utf8))
            let restored = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
                from: JSONEncoder().encode(original))
            XCTAssertEqual(restored.outputGateType, gate)
            _ = try restored.recurrentOutputGate()
        }
    }

    func testUnknownGateIsRejectedAtDecodeIncludingNestedConfig() {
        for gate in ["identity", "relu", "SiLU", ""] {
            let text = "{\"output_gate_type\":\"\(gate)\"}"
            XCTAssertThrowsError(try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
                from: Data(text.utf8))) { error in
                    guard case DecodingError.dataCorrupted(let context) = error else {
                        return XCTFail("Expected typed configuration error, got \(error)")
                    }
                    XCTAssertEqual(context.codingPath.map(\.stringValue), ["output_gate_type"])
                }
            XCTAssertThrowsError(try JSONDecoder().decode(Qwen4ExpConfiguration.self,
                from: Data("{\"text_config\":\(text)}".utf8))) { error in
                    guard case DecodingError.dataCorrupted(let context) = error else {
                        return XCTFail("Expected typed configuration error, got \(error)")
                    }
                    XCTAssertEqual(context.codingPath.map(\.stringValue), ["text_config", "output_gate_type"])
                }
        }
    }

    func testProgrammaticUnknownGateIsRejectedBeforeRecurrentConstruction() {
        XCTAssertThrowsError(try Qwen4ExpDecoderLayer(configuration(gate: "identity"),
            layerIdx: 0, mmapPLE: false))
    }

    func testConfiguredGateExecutesCorrectMathIncludingFusedPosture() throws {
        // Exercise the actual tail called by decode, prefill and MTP GDN paths.
        // SiLU must bypass the sigmoid-only fusion; sigmoid keeps that fusion.
        for gateType in ["sigmoid", "silu"] {
            let layer = try Qwen4ExpDecoderLayer(configuration(gate: gateType),
                layerIdx: 0, mmapPLE: false)
            let gdn = try XCTUnwrap(layer.linearAttn)
            for dtype: DType in [.float32, .bfloat16, .float16] {
                for width in [1, 6, 64] {
                    let count = width * 2 * 8
                    let shape = [1, width, 2, 8]
                    let input = MLXArray((0 ..< count).map { Float($0 % 23 - 11) / 7 })
                        .reshaped(shape).asType(dtype)
                    let gate = MLXArray((0 ..< count).map { Float($0 % 17 - 8) / 3 })
                        .reshaped(shape).asType(dtype)
                    let normalized = MLXFast.rmsNorm(input, weight: gdn.norm.weight, eps: gdn.norm.eps)
                    let rawGate = gate.asType(.float32)
                    let activation = gateType == "sigmoid" ? sigmoid(rawGate) : silu(rawGate)
                    let expected = (activation * normalized.asType(.float32)).asType(dtype)
                    let actual = gdn.gatedOutputNorm(input, gate: gate)
                    eval(actual, expected)
                    XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                        expected.asType(.float32).asArray(Float.self),
                        "gate=\(gateType), dtype=\(dtype), width=\(width)")
                }
            }
        }
    }
}
