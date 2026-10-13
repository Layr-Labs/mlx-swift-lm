// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Short-convolution semantics (mlx-lm 0.32.0 `ShortConv`): BCx = in_proj(x),
/// Bx = B * x, history = last L-1 Bx rows (zeros initially), depthwise causal
/// conv over [history, Bx], y = C * conv, out_proj. A float64 host oracle
/// checks the serial module and the CBv2 request-owned path; in-test mutants
/// show each fixture would catch a wrong gate, order or kernel orientation.
final class LFM2MoEShortConvTests: XCTestCase {
    private let hidden = 4

    private func makeConv(lCache: Int) throws -> LFM2MoEShortConv {
        let json = """
            {"model_type":"lfm2_moe","vocab_size":32,"hidden_size":\(hidden),"intermediate_size":8,
             "moe_intermediate_size":8,"num_hidden_layers":1,"num_experts":4,
             "num_experts_per_tok":2,"norm_topk_prob":true,"num_attention_heads":1,
             "num_key_value_heads":1,"max_position_embeddings":128,"use_expert_bias":true,
             "num_dense_layers":1,"norm_eps":1e-6,"conv_bias":false,"conv_L_cache":\(lCache)}
            """
        let config = try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(json.utf8))
        let conv = LFM2MoEShortConv(config, layerIdx: 0)
        MLXRandom.seed(1_337 + UInt64(lCache))
        var identity = [Float](repeating: 0, count: hidden * hidden)
        for i in 0 ..< hidden { identity[i * hidden + i] = 1 }
        conv.update(
            parameters: ModuleParameters.unflattened([
                "in_proj.weight": MLXRandom.uniform(low: -1, high: 1, [3 * hidden, hidden]),
                // Identity output projection exposes y = C * conv directly.
                "out_proj.weight": MLXArray(identity, [hidden, hidden]),
                // MLX depthwise layout [out, kernel, in/groups].
                "conv.weight": MLXRandom.uniform(low: -1, high: 1, [hidden, lCache, 1]),
            ]))
        eval(conv)
        return conv
    }

    private enum Mutant { case none, ungatedInput, ungatedOutput, flippedKernel, historyFirstRows }

    /// Float64 oracle. Returns outputs [T][H] and the final history rows.
    private func oracle(
        _ conv: LFM2MoEShortConv, input: [[Double]], lCache: Int, mutant: Mutant = .none
    ) -> (out: [[Double]], history: [[Double]]) {
        let w = conv.inProj.weight.asArray(Float.self).map(Double.init)  // [3H, H]
        let k = conv.conv.weight.asArray(Float.self).map(Double.init)  // [H, L, 1]
        var history = Array(repeating: Array(repeating: 0.0, count: hidden), count: lCache - 1)
        var outputs: [[Double]] = []
        var allBx: [[Double]] = history
        for x in input {
            let bcx = (0 ..< 3 * hidden).map { r in (0 ..< hidden).reduce(0) { $0 + w[r * hidden + $1] * x[$1] } }
            let b = Array(bcx[0 ..< hidden]), c = Array(bcx[hidden ..< 2 * hidden])
            let xc = Array(bcx[2 * hidden ..< 3 * hidden])
            let bx = mutant == .ungatedInput ? xc : zip(b, xc).map(*)
            let window = history + [bx]  // oldest -> newest, L rows
            var y = [Double](repeating: 0, count: hidden)
            for ch in 0 ..< hidden {
                var acc = 0.0
                for j in 0 ..< lCache {
                    let tap = mutant == .flippedKernel ? lCache - 1 - j : j
                    acc += k[ch * lCache + tap] * window[j][ch]
                }
                y[ch] = mutant == .ungatedOutput ? acc : c[ch] * acc
            }
            outputs.append(y)
            allBx.append(bx)
            history = mutant == .historyFirstRows
                ? Array(allBx.prefix(lCache - 1)) : Array(window.suffix(lCache - 1))
        }
        return (outputs, history)
    }

    private func input(_ t: Int, seed: Int) -> [[Double]] {
        (0 ..< t).map { i in (0 ..< hidden).map { h in sin(Double(i * 7 + h * 3 + seed)) } }
    }

    private func mlx(_ rows: [[Double]]) -> MLXArray {
        MLXArray(rows.flatMap { $0.map(Float.init) }, [1, rows.count, hidden])
    }

    private func maxDiff(_ a: MLXArray, _ b: [[Double]]) -> Double {
        let flat = a.asType(.float32).asArray(Float.self).map(Double.init)
        return zip(flat, b.flatMap { $0 }).map { abs($0 - $1) }.max() ?? 0
    }

    /// Serial module: zero init, B*x and C* gating, kernel orientation,
    /// one-shot == chunked (including chunks shorter than L-1), and the cache
    /// holds exactly the last L-1 gated inputs.
    func testSerialShortConvMatchesFloat64OracleOneShotAndChunked() throws {
        for lCache in [3, 4] {
            let conv = try makeConv(lCache: lCache)
            let rows = input(11, seed: lCache)
            let expected = oracle(conv, input: rows, lCache: lCache)
            for mutant in [Mutant.ungatedInput, .ungatedOutput, .flippedKernel, .historyFirstRows] {
                let wrong = oracle(conv, input: rows, lCache: lCache, mutant: mutant).out
                let gap = zip(wrong.flatMap { $0 }, expected.out.flatMap { $0 }).map { abs($0 - $1) }.max()!
                XCTAssertGreaterThan(gap, 1e-2, "fixture does not expose \(mutant) (L=\(lCache))")
            }

            let oneShot = conv(mlx(rows), mask: nil, cache: nil)
            XCTAssertLessThan(maxDiff(oneShot, expected.out), 1e-5, "uncached (zero history) L=\(lCache)")

            for split in [[11], [1, 1, 1, 8], [2, 9], [5, 1, 5]] {
                let cache = MambaCache()
                var outs: [MLXArray] = []
                var start = 0
                for n in split {
                    outs.append(conv(mlx(Array(rows[start ..< start + n])), mask: nil, cache: cache))
                    start += n
                }
                XCTAssertLessThan(
                    maxDiff(concatenated(outs, axis: 1), expected.out), 1e-5,
                    "chunked \(split) L=\(lCache)")
                let state = try XCTUnwrap(cache[0])
                XCTAssertEqual(state.shape, [1, lCache - 1, hidden])
                XCTAssertLessThan(maxDiff(state, expected.history), 1e-5, "history rows \(split)")
            }
        }
    }

    /// CBv2 request-owned path: transactional staging, decode after prefill,
    /// rows with different histories in one rectangle, rollback.
    func testCBv2ShortConvStagesExactHistoryPerRow() throws {
        let lCache = 3
        let conv = try makeConv(lCache: lCache)
        let spec = CBv2RecurrentStateSpec(layers: [
            .convolutionOnly(modelLayerIndex: 7, convShape: [1, lCache - 1, hidden], convDType: .float32)
        ])
        let rowsA = input(9, seed: 1), rowsB = input(4, seed: 2)
        let expectedA = oracle(conv, input: rowsA, lCache: lCache)
        let expectedB = oracle(conv, input: rowsB, lCache: lCache)
        let a = try CBv2RecurrentRequestState(spec: spec)
        let b = try CBv2RecurrentRequestState(spec: spec)

        func step(_ states: [CBv2RecurrentRequestState], _ x: MLXArray, commit: Bool = true) throws
            -> (MLXArray, [CBv2RecurrentStateEvaluation])
        {
            let bindings = try states.map { try $0.bind() }
            let out = conv.cbv2Forward(x, modelLayerIndex: 7, recurrentState: bindings)
            eval([out] + (try bindings.flatMap { try $0.evaluate() }))
            if commit { for binding in bindings { try binding.commit() } }
            return (out, bindings)
        }

        // Prefill A in chunks of 2 + 1 + 3 (chunk < L-1 included), B one-shot.
        var outA: [MLXArray] = []
        var start = 0
        for n in [2, 1, 3] {
            outA.append(try step([a], mlx(Array(rowsA[start ..< start + n]))).0)
            start += n
        }
        let outB = try step([b], mlx(Array(rowsB.prefix(3)))).0
        // Rectangular decode: A then B in one [2, 1] step, then reversed.
        let decode1 = try step(
            [a, b], concatenated([mlx([rowsA[6]]), mlx([rowsB[3]])], axis: 0)).0
        let pendingB = try step([b, a], concatenated([mlx([rowsB[3]]), mlx([rowsA[7]])], axis: 0), commit: false)
        // Roll the speculative step back: both rows return to their histories.
        for binding in pendingB.1 { try binding.rollback() }
        let decode2 = try step([a], mlx([rowsA[7]])).0
        let decode3 = try step([a], mlx([rowsA[8]])).0

        let prefixA = oracle(conv, input: Array(rowsA.prefix(6)), lCache: lCache)
        XCTAssertLessThan(maxDiff(concatenated(outA, axis: 1), prefixA.out), 1e-5)
        XCTAssertLessThan(maxDiff(outB, Array(expectedB.out.prefix(3))), 1e-5)
        XCTAssertLessThan(maxDiff(decode1[0 ..< 1], [expectedA.out[6]]), 1e-5)
        XCTAssertLessThan(maxDiff(decode1[1 ..< 2], [expectedB.out[3]]), 1e-5)
        XCTAssertLessThan(maxDiff(decode2, [expectedA.out[7]]), 1e-5)
        XCTAssertLessThan(maxDiff(decode3, [expectedA.out[8]]), 1e-5)

        let historyA = try XCTUnwrap(a.confirmedStateSnapshot()?[7])
        XCTAssertNil(historyA.ssm, "conv-only layers stage no SSM state")
        XCTAssertLessThan(maxDiff(try XCTUnwrap(historyA.conv), expectedA.history), 1e-6)
        let historyB = try XCTUnwrap(b.confirmedStateSnapshot()?[7]?.conv)
        XCTAssertLessThan(maxDiff(historyB, expectedB.history), 1e-6)
        try a.release()
        try b.release()
    }

    /// A Hugging Face depthwise weight [H, 1, L] is transposed to MLX
    /// [H, L, 1] with the newest tap last; MLX-layout weights pass through.
    func testSanitizeOrientsDepthwiseKernel() throws {
        let model = LFM2MoEModel(try LFM2MoEFixture.configuration())
        let hf = MLXArray((0 ..< 256 * 3).map(Float.init), [256, 1, 3])
        let sanitized = model.sanitize(weights: ["model.layers.0.conv.conv.weight": hf])
        let w = try XCTUnwrap(sanitized["model.layers.0.conv.conv.weight"])
        XCTAssertEqual(w.shape, [256, 3, 1])
        XCTAssertEqual(w[5, 2, 0].item(Float.self), hf[5, 0, 2].item(Float.self))
        let mlxLayout = MLXArray((0 ..< 256 * 3).map(Float.init), [256, 3, 1])
        let kept = model.sanitize(weights: ["model.layers.0.conv.conv.weight": mlxLayout])
        XCTAssertEqual(
            kept["model.layers.0.conv.conv.weight"]?.asArray(Float.self), mlxLayout.asArray(Float.self))
    }

    /// Spec/stage contract for convolution-only layers in the shared infra.
    func testConvolutionOnlySpecStagesAndPricesNoSSMTensor() throws {
        let spec = CBv2RecurrentStateSpec(layers: [
            .convolutionOnly(modelLayerIndex: 0, convShape: [1, 2, 8], convDType: .bfloat16),
            .init(modelLayerIndex: 1, convShape: [1, 3, 8], convDType: .bfloat16, ssmShape: [1, 2, 2, 2]),
        ])
        XCTAssertFalse(spec.layers[0].hasSSMState)
        XCTAssertTrue(spec.layers[1].hasSSMState)
        XCTAssertEqual(try spec.fixedBytesPerRequest(), 2 * 8 * 2 + 3 * 8 * 2 + 8 * 4)
        if let policy = Memory.allocationFootprintPolicy() {
            XCTAssertGreaterThan(try spec.allocationBytesPerGeneration(policy: policy), 0)
            let convOnly = CBv2RecurrentStateSpec(layers: [spec.layers[0]])
            XCTAssertNoThrow(
                try convOnly.allocationBytesPerGeneration(policy: policy),
                "an absent SSM tensor must not make admission pricing fail")
        }
        let state = try CBv2RecurrentRequestState(spec: spec)
        let tx = try state.bind()
        XCTAssertThrowsError(
            try tx.stage(modelLayerIndex: 0, conv: MLXArray.zeros([1, 2, 8]), ssm: MLXArray.zeros([1, 1])),
            "SSM on a convolution-only layer")
        XCTAssertThrowsError(
            try tx.stage(modelLayerIndex: 1, conv: MLXArray.zeros([1, 3, 8]), ssm: nil),
            "missing SSM on a layer that declares one")
        try tx.stage(modelLayerIndex: 0, conv: MLXArray.zeros([1, 2, 8], dtype: .bfloat16), ssm: nil)
        try tx.stage(
            modelLayerIndex: 1, conv: MLXArray.zeros([1, 3, 8], dtype: .bfloat16),
            ssm: MLXArray.zeros([1, 2, 2, 2]))
        let roots = try tx.evaluate()
        XCTAssertEqual(roots.count, 3, "two conv tensors and one SSM tensor")
        try tx.commit()
        let snapshot = try XCTUnwrap(state.confirmedStateSnapshot())
        XCTAssertNil(snapshot[0]?.ssm)
        XCTAssertNoThrow(try CBv2RecurrentRequestState(spec: spec, adoptedCommitted: snapshot))
        var missing = snapshot
        missing[1] = .init(conv: snapshot[1]!.conv, ssm: nil)
        XCTAssertThrowsError(try CBv2RecurrentRequestState(spec: spec, adoptedCommitted: missing))
        try state.release()
    }
}
