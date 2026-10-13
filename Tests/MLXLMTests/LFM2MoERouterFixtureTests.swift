// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

/// Deterministic, offline router fixtures for LFM2-MoE (mlx-lm 0.32.0
/// `lfm2_moe.py` semantics): sigmoid gate, float32 expert bias added for
/// SELECTION only, unbiased gathered weights, renormalized with +1e-6, then
/// routed_scaling_factor. Each fixture row is chosen so that a specific wrong
/// implementation produces a measurably different selection or weight; the
/// suite proves that discrimination with in-test mutants and then requires the
/// production `route()` to match the reference.
final class LFM2MoERouterFixtureTests: XCTestCase {
    private struct Route: Equatable {
        let experts: [Int]  // sorted
        let weights: [Int: Double]
    }

    private enum Variant: CaseIterable {
        case reference, softmaxGate, biasInWeights, noRenormalization, epsilon1e20,
            bfloat16Bias, legacyMLXLM0313
    }

    private let experts = 4
    private let topK = 2

    private func block(bias: [Float], scaling: Float = 1) throws -> Lfm2MoeSparseMoeBlock {
        let json = """
            {"model_type":"lfm2_moe","vocab_size":32,"hidden_size":4,"intermediate_size":8,
             "moe_intermediate_size":8,"num_hidden_layers":1,"num_experts":\(experts),
             "num_experts_per_tok":\(topK),"norm_topk_prob":true,"num_attention_heads":2,
             "num_key_value_heads":1,"max_position_embeddings":128,"use_expert_bias":true,
             "num_dense_layers":0,"norm_eps":1e-6,"conv_bias":false,"conv_L_cache":3,
             "routed_scaling_factor":\(scaling)}
            """
        let config = try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(json.utf8))
        let block = Lfm2MoeSparseMoeBlock(config)
        var eye = [Float](repeating: 0, count: experts * experts)
        for i in 0 ..< experts { eye[i * experts + i] = 1 }
        block.update(
            parameters: ModuleParameters.unflattened([
                "gate.weight": MLXArray(eye, [experts, experts]),
                "expert_bias": MLXArray(bias),
            ]))
        return block
    }

    private func sigmoid(_ v: Double) -> Double { 1 / (1 + exp(-v)) }

    private func bf16(_ v: Float) -> Double {
        // Round-to-nearest-even to bfloat16, as MLX's astype does.
        let bits = v.bitPattern
        let rounded = (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000
        return Double(Float(bitPattern: rounded))
    }

    private func route(_ logits: [Float], bias: [Float], scaling: Double, _ variant: Variant) -> Route {
        let x = logits.map(Double.init)
        let gate: [Double]
        switch variant {
        case .softmaxGate, .legacyMLXLM0313:
            let m = x.max()!
            let e = x.map { exp($0 - m) }
            let s = e.reduce(0, +)
            gate = e.map { $0 / s }
        default:
            gate = x.map(sigmoid)
        }
        let b = bias.map { variant == .bfloat16Bias ? bf16($0) : Double($0) }
        let selection = zip(gate, b).map { $0 + $1 }
        let chosen = selection.enumerated().sorted { $0.element > $1.element }.prefix(topK).map(\.offset)
        var raw: [Int: Double] = [:]
        for e in chosen {
            raw[e] = (variant == .biasInWeights || variant == .legacyMLXLM0313) ? selection[e] : gate[e]
        }
        let eps = (variant == .epsilon1e20 || variant == .legacyMLXLM0313) ? 1e-20 : 1e-6
        let denominator = raw.values.reduce(0, +) + eps
        var weights: [Int: Double] = [:]
        for (e, w) in raw {
            let normalized = variant == .noRenormalization ? w : w / denominator
            weights[e] = normalized * (variant == .legacyMLXLM0313 ? 1 : scaling)
        }
        return Route(experts: chosen.sorted(), weights: weights)
    }

    private func production(_ block: Lfm2MoeSparseMoeBlock, _ logits: [Float]) -> Route {
        let (indices, weights) = block.route(MLXArray(logits, [1, experts]))
        eval(indices, weights)
        XCTAssertEqual(weights.dtype, .float32, "routing weights are computed in float32")
        let idx = indices.asArray(Int32.self).map(Int.init)
        let w = weights.asArray(Float.self).map(Double.init)
        return Route(experts: idx.sorted(), weights: Dictionary(uniqueKeysWithValues: zip(idx, w)))
    }

    private func distance(_ a: Route, _ b: Route) -> Double {
        guard a.experts == b.experts else { return .infinity }
        return a.experts.map { abs(a.weights[$0]! - b.weights[$0]!) }.max() ?? 0
    }

    /// Rows: (logits, bias, scaling, variants this row must expose).
    private var fixtures: [([Float], [Float], Double, [Variant])] {
        [
            // Bias lifts low-logit expert 2 into the top-2 (selection-only).
            ([2.0, 1.0, 0.5, 0.0], [0, 0, 1, 0], 1, [.softmaxGate, .biasInWeights, .noRenormalization, .legacyMLXLM0313]),
            // Unequal bias: folding it into the weights changes their ratio.
            ([0.3, -0.2, 0.1, 0.0], [0.4, 0.05, 0, 0], 1, [.biasInWeights, .softmaxGate]),
            // Tiny sigmoid mass: the +1e-6 epsilon is a 40% effect here.
            ([-14.0, -14.5, -20.0, -20.0], [0, 0, 0, 0], 1, [.epsilon1e20, .softmaxGate]),
            // Near tie broken only by float32 bias precision: 1.003 rounds to
            // 1.0 in bfloat16, flipping expert 0 out of the selection.
            ([-0.008, 0.0, 3.0, -5.0], [1.003, 1.0, 1.0, 0], 1, [.bfloat16Bias]),
            // routed_scaling_factor scales the normalized weights.
            ([1.0, 0.5, -1.0, -2.0], [0, 0.1, 0, 0], 2.5, [.legacyMLXLM0313, .noRenormalization]),
        ]
    }

    func testFixtureRowsDiscriminateEveryKnownWrongRouter() {
        var exposed = Set<String>()
        for (logits, bias, scaling, required) in fixtures {
            let reference = route(logits, bias: bias, scaling: scaling, .reference)
            for variant in required {
                let d = distance(route(logits, bias: bias, scaling: scaling, variant), reference)
                XCTAssertGreaterThan(d, 1e-3, "fixture \(logits) does not expose \(variant)")
                if d > 1e-3 { exposed.insert("\(variant)") }
            }
        }
        XCTAssertEqual(
            exposed,
            Set(Variant.allCases.filter { $0 != .reference }.map { "\($0)" }),
            "every wrong router must be exposed by at least one fixture row")
    }

    func testProductionRouteMatchesReferenceOnEveryFixtureRow() throws {
        for (logits, bias, scaling, _) in fixtures {
            let block = try block(bias: bias, scaling: Float(scaling))
            let actual = production(block, logits)
            let reference = route(logits, bias: bias, scaling: scaling, .reference)
            XCTAssertEqual(actual.experts, reference.experts, "selection for \(logits)")
            XCTAssertLessThan(distance(actual, reference), 1e-5, "weights for \(logits)")
            for variant in Variant.allCases where variant != .reference {
                let wrong = route(logits, bias: bias, scaling: scaling, variant)
                if distance(wrong, reference) > 1e-3 {
                    XCTAssertGreaterThan(
                        distance(actual, wrong), 1e-4, "production route behaves like \(variant)")
                }
            }
        }
    }

    /// The mixture applies the unbiased weights, cast to the activation
    /// dtype only after normalization and scaling.
    func testMixtureUsesUnbiasedWeightsInActivationDType() throws {
        let block = try block(bias: [0, 0, 1, 0])
        // Production precision: experts in bf16, router gate and bias float32.
        block.update(
            parameters: ModuleParameters.unflattened(
                block.parameters().flattened().map { key, value in
                    (key, key.hasPrefix("switch_mlp") ? value.asType(.bfloat16) : value)
                }))
        XCTAssertEqual(block.gate.weight.dtype, .float32)
        let x = MLXArray([Float(2.0), 1.0, 0.5, 0.0], [1, 4]).asType(.bfloat16)
        let (indices, weights) = block.route(x)
        XCTAssertEqual(weights.dtype, .float32)
        let expected = (block.switchMLP(x, indices)
            * weights.asType(.bfloat16).expandedDimensions(axis: -1)).sum(axis: -2)
        let actual = block(x)
        eval(expected, actual)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertLessThanOrEqual(
            abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self), 1e-6)
    }

    /// A bf16 gate weight or expert bias would change routing on the real
    /// artifact; the module keeps whatever float32 tensors it is given.
    func testFloat32RouterParametersAreNotNarrowedByTheModule() throws {
        let block = try block(bias: [1.003, 1.0, 1.0, 0])
        XCTAssertEqual(block.gate.weight.dtype, .float32)
        XCTAssertEqual(block.expertBias?.dtype, .float32)
        XCTAssertEqual(block.expertBias?.asArray(Float.self)[0], 1.003)
    }
}
