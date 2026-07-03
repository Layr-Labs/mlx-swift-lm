// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM

/// Tests for the DeepSeek-V4 port (reference: ml-explore/mlx-lm#1192).
///
/// Uses a tiny random-weight config so every attention variant (local,
/// compressed, sparse-compressed) and the MoE/hyper-connection stack run in
/// milliseconds. These pin exactly the behaviours the port got wrong before:
/// shared-expert clamping, sink handling, masked-score sentinels, pooling
/// cache accounting across chunked prefill, and quantized wo_a loading.
final class DeepseekV4Tests: XCTestCase {

    // MARK: - Fixtures

    /// Tiny config covering all three attention kinds: ratios [0, 4, 128, 0].
    /// Layer 0 is a hash-routing MoE layer (num_hash_layers = 1).
    private func makeConfig(
        compressRatios: [Int] = [0, 4, 128, 0],
        slidingWindow: Int = 8
    ) throws -> DeepseekV4Configuration {
        let json = """
            {
                "vocab_size": 64,
                "hidden_size": 32,
                "moe_intermediate_size": 16,
                "num_hidden_layers": \(compressRatios.count),
                "num_attention_heads": 4,
                "head_dim": 16,
                "q_lora_rank": 16,
                "qk_rope_head_dim": 8,
                "rms_norm_eps": 1e-6,
                "o_groups": 2,
                "o_lora_rank": 8,
                "sliding_window": \(slidingWindow),
                "compress_ratios": \(compressRatios),
                "compress_rope_theta": 160000.0,
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "scoring_func": "sqrtsoftplus",
                "routed_scaling_factor": 1.5,
                "swiglu_limit": 10.0,
                "num_hash_layers": 1,
                "num_nextn_predict_layers": 0,
                "norm_topk_prob": true,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 1e-6,
                "rope_theta": 10000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 512,
                    "beta_fast": 32,
                    "beta_slow": 1
                },
                "max_position_embeddings": 8192,
                "index_n_heads": 4,
                "index_head_dim": 8,
                "index_topk": 4
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(DeepseekV4Configuration.self, from: json)
    }

    /// Model with small random weights (deterministic). Integer tables
    /// (tid2eid) keep their zero init, which is a valid expert index.
    private func makeModel(_ config: DeepseekV4Configuration, seed: UInt64 = 7)
        -> DeepseekV4Model
    {
        MLXRandom.seed(seed)
        let model = DeepseekV4Model(config)
        let randomized = model.parameters().flattened().map {
            (key, value) -> (String, MLXArray) in
            guard value.dtype == .float32 else { return (key, value) }
            return (key, MLXRandom.normal(value.shape) * 0.05)
        }
        try! model.update(
            parameters: ModuleParameters.unflattened(randomized), verify: .none)
        eval(model)
        return model
    }

    private func assertFinite(_ x: MLXArray, _ message: String) {
        let f = x.asType(.float32)
        let finite = MLX.logicalAnd(f .== f, abs(f) .< Float.infinity)
        XCTAssertTrue(finite.all().item(Bool.self), message)
    }

    // MARK: - Forward pass

    func testPrefillAndDecodeProduceFiniteLogits() throws {
        let config = try makeConfig()
        let model = makeModel(config)
        let cache = model.makeCache(parameters: GenerateParameters())

        XCTAssertEqual(cache.count, config.numHiddenLayers)

        let prompt = MLXArray([Int32(1), 5, 9, 13, 2], [1, 5])
        let logits = model(prompt, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, config.vocabSize])
        assertFinite(logits, "prefill logits must be finite")

        // Decode across a ratio-4 window boundary so pooling kicks in.
        var last = MLXArray([Int32(3)], [1, 1])
        for step in 0 ..< 8 {
            let out = model(last, cache: cache)
            eval(out)
            XCTAssertEqual(out.shape, [1, 1, config.vocabSize])
            assertFinite(out, "decode step \(step) logits must be finite")
            last = out.argMax(axis: -1).asType(.int32)
        }
    }

    /// Long prompt (> sliding window, > several ratio-4 windows) exercising the
    /// pooled and masked paths that previously produced -inf/NaN.
    func testLongPromptPrefillIsFinite() throws {
        let config = try makeConfig()
        let model = makeModel(config)
        let cache = model.makeCache(parameters: GenerateParameters())

        let tokens = (0 ..< 40).map { Int32($0 % 64) }
        let logits = model(MLXArray(tokens, [1, 40]), cache: cache)
        eval(logits)
        assertFinite(logits, "long prefill must not produce NaN")
    }

    /// PoolingCache remainder accounting: chunked prefill must match one-shot.
    func testChunkedPrefillMatchesOneShot() throws {
        let config = try makeConfig(compressRatios: [0, 4, 4, 0])
        let model = makeModel(config)

        let tokens = (0 ..< 12).map { Int32(($0 * 5) % 64) }

        let oneShotCache = model.makeCache(parameters: GenerateParameters())
        let oneShot = model(MLXArray(tokens, [1, 12]), cache: oneShotCache)
        let oneShotLast = oneShot[0..., -1, 0...]
        eval(oneShotLast)

        // Chunks deliberately misaligned with the ratio-4 window (5 + 4 + 3).
        let chunkedCache = model.makeCache(parameters: GenerateParameters())
        var chunkedLast: MLXArray = MLXArray([Float(0)])
        for chunk in [Array(tokens[0 ..< 5]), Array(tokens[5 ..< 9]), Array(tokens[9 ..< 12])] {
            let out = model(MLXArray(chunk, [1, chunk.count]), cache: chunkedCache)
            chunkedLast = out[0..., -1, 0...]
        }
        eval(chunkedLast)

        let maxDiff = abs(oneShotLast - chunkedLast).max().item(Float.self)
        XCTAssertLessThan(
            maxDiff, 1e-3,
            "chunked prefill must reproduce one-shot logits (pooling remainder accounting)")
    }

    // MARK: - HyperConnection

    /// The fused Metal kernel and the pure-ops fallback must agree.
    func testHcKernelMatchesOpsFallback() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Metal kernel requires GPU")
        }

        MLXRandom.seed(11)
        let (B, S, hc, D) = (2, 3, 4, 32)
        let x = MLXRandom.normal([B, S, hc, D]).asType(.bfloat16)
        let fn = MLXRandom.normal([(2 + hc) * hc, hc * D]) * 0.1
        let scale = MLXArray([Float(1.0), 1.2, 0.8])
        let base = MLXRandom.normal([(2 + hc) * hc]) * 0.1
        let eps: Float = 1e-6

        let xFlat = x.reshaped([B, S, hc * D]).asType(.float32)
        let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
        let mixes = matmul(xFlat, fn.T) * normScale

        let (kCollapsed, kPost, kComb) = hcKernel(
            x: x, mixes: mixes.reshaped([B * S, (2 + hc) * hc]),
            scale: scale, base: base, hcMult: hc, sinkhornIters: 20, eps: eps)

        let (pre, oPost, oComb) = hcSplitSinkhorn(
            mixes, hcScale: scale, hcBase: base, hcMult: hc, sinkhornIters: 20, eps: eps)
        let oCollapsed = (pre.expandedDimensions(axis: -1) * x.asType(.float32))
            .sum(axis: -2).asType(x.dtype)

        eval(kCollapsed, kPost, kComb, oCollapsed, oPost, oComb)

        XCTAssertLessThan(
            abs(kPost - oPost).max().item(Float.self), 1e-3, "post gates must match")
        XCTAssertLessThan(
            abs(kComb - oComb).max().item(Float.self), 1e-3, "sinkhorn comb must match")
        XCTAssertLessThan(
            abs(kCollapsed.asType(.float32) - oCollapsed.asType(.float32)).max()
                .item(Float.self),
            2e-2, "collapsed hidden must match (bf16 tolerance)")

        // Doubly-stochastic invariant: rows and columns both sum to ~1.
        let rowSums = kComb.sum(axis: -1)
        let colSums = kComb.sum(axis: -2)
        XCTAssertLessThan(abs(rowSums - 1).max().item(Float.self), 1e-2)
        XCTAssertLessThan(abs(colSums - 1).max().item(Float.self), 1e-2)
    }

    // MARK: - MoE gate

    func testGateBiasSteersSelectionButNotWeights() throws {
        let config = try makeConfig()
        // layerIdx 3 >= numHashLayers(1) → bias routing
        let gate = DeepseekV4Gate(config: config, layerIdx: 3)

        // Identity-ish gate: logits = first 8 dims of x.
        var w = [Float](repeating: 0, count: 8 * 32)
        for e in 0 ..< 8 { w[e * 32 + e] = 1 }
        // Bias lifts expert 5 (logit 0) over expert 1 (logit 1.0) for selection.
        var bias = [Float](repeating: 0, count: 8)
        bias[5] = 10
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXArray(w, [8, 32]),
                "e_score_correction_bias": MLXArray(bias),
            ]), verify: .none)

        var xData = [Float](repeating: 0, count: 32)
        xData[0] = 2.0  // expert 0 logit
        xData[1] = 1.0  // expert 1 logit
        let x = MLXArray(xData, [1, 1, 32])

        let (inds, weights) = gate(x, inputIds: nil)
        eval(inds, weights)

        let selected = Set(inds.asArray(Int32.self).map { Int($0) })
        XCTAssertEqual(
            selected, Set([0, 5]),
            "e_score_correction_bias must steer selection (5 in, 1 out)")

        // Weights come from UNbiased sqrtsoftplus scores, normalized, scaled.
        func sqrtSoftplus(_ v: Double) -> Double { (log(1 + exp(v))).squareRoot() }
        let s0 = sqrtSoftplus(2.0)
        let s5 = sqrtSoftplus(0.0)
        let denom = s0 + s5
        let expected: [Int: Double] = [
            0: s0 / denom * 1.5,
            5: s5 / denom * 1.5,
        ]
        let idx = inds.asArray(Int32.self).map { Int($0) }
        let wOut = weights.asArray(Float.self)
        for (j, expert) in idx.enumerated() {
            let want = try XCTUnwrap(expected[expert])
            XCTAssertEqual(Double(wOut[j]), want, accuracy: 1e-3)
        }
    }

    func testGateHashRoutingUsesTokenTable() throws {
        let config = try makeConfig()
        // layerIdx 0 < numHashLayers(1) → hash routing
        let gate = DeepseekV4Gate(config: config, layerIdx: 0)

        var table = [Int32](repeating: 0, count: 64 * 2)
        table[7 * 2] = 3  // token 7 → experts (3, 6)
        table[7 * 2 + 1] = 6
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([8, 32]) * 0.1,
                "tid2eid": MLXArray(table, [64, 2]),
            ]), verify: .none)

        let x = MLXRandom.normal([1, 1, 32])
        let ids = MLXArray([Int32(7)], [1, 1])
        let (inds, weights) = gate(x, inputIds: ids)
        eval(inds, weights)

        XCTAssertEqual(
            inds.asArray(Int32.self), [3, 6],
            "hash layer routing must come from tid2eid, not logits")
        assertFinite(weights, "hash routing weights must be finite")
    }

    // MARK: - Shared experts

    /// Shared experts must be UNclamped (reference applies swiglu_limit to
    /// routed experts only). With weights scaled so gate outputs exceed the
    /// limit, a clamped MLP would visibly diverge from the exact SwiGLU.
    func testSharedExpertsAreNotClamped() throws {
        let config = try makeConfig()
        let moe = DeepseekV4MoE(config: config, layerIdx: 3)

        let shared = moe.sharedExperts
        MLXRandom.seed(3)
        let big = MLXRandom.normal([16, 32]) * 4.0  // drives |gate| >> swiglu_limit
        try shared.update(
            parameters: ModuleParameters.unflattened([
                "gate_proj.weight": big,
                "up_proj.weight": MLXRandom.normal([16, 32]),
                "down_proj.weight": MLXRandom.normal([32, 16]) * 0.1,
            ]), verify: .none)

        let x = MLXRandom.normal([1, 1, 32]) * 3.0
        let got = shared(x)

        let g = matmul(x, big.T)
        let u = matmul(x, shared.up_proj.weight.T)
        let want = matmul(silu(g) * u, shared.down_proj.weight.T)
        eval(got, want)

        XCTAssertLessThan(
            abs(got - want).max().item(Float.self), 1e-4,
            "shared experts must compute exact (unclamped) SwiGLU")
    }

    // MARK: - Cache semantics

    func testMakeCacheLayoutFollowsCompressRatios() throws {
        let config = try makeConfig(compressRatios: [0, 4, 128, 0])
        let model = makeModel(config)
        let cache = model.makeCache(parameters: GenerateParameters())

        XCTAssertTrue(cache[0] is RotatingKVCache)
        let sparse = try XCTUnwrap(cache[1] as? DeepseekV4LayerCache)
        XCTAssertEqual(sparse.pooling.count, 2, "ratio 4 → compressor + indexer pools")
        let compressed = try XCTUnwrap(cache[2] as? DeepseekV4LayerCache)
        XCTAssertEqual(compressed.pooling.count, 1, "ratio 128 → compressor pool only")
        XCTAssertTrue(cache[3] is RotatingKVCache)
    }

    func testLayerCacheIsNotTrimmableAndCopyIsIndependent() throws {
        let layerCache = DeepseekV4LayerCache(
            rotating: RotatingKVCache(maxSize: 8),
            pooling: [PoolingCache(ratio: 4)])

        XCTAssertFalse(
            layerCache.isTrimmable,
            "pooled state cannot be rolled back; must not advertise trimmability")
        XCTAssertEqual(layerCache.trim(4), 0)

        // copy() must not share PoolingCache instances.
        let kv = MLXRandom.normal([1, 4, 8])
        let gate = MLXRandom.normal([1, 4, 8])
        let copy = layerCache.copy() as! DeepseekV4LayerCache
        _ = layerCache.pooling[0].accumulateWindows(kv: kv, gate: gate, offset: 4)
        _ = layerCache.pooling[0].updateAndFetch(MLXRandom.normal([1, 1, 8]))
        XCTAssertEqual(layerCache.pooling[0].pooledCount, 1)
        XCTAssertEqual(
            copy.pooling[0].pooledCount, 0,
            "mutating the original must not affect the copy")
    }

    // MARK: - Quantization

    func testQuantizedMultiLinearMatchesFloat() throws {
        MLXRandom.seed(5)
        let ml = DeepseekV4MultiLinear(inFeatures: 64, outFeatures: 8, groups: 2)
        try ml.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([2, 8, 64])
            ]), verify: .none)

        let x = MLXRandom.normal([1, 2, 3, 64])
        let want = ml(x)

        let quantized = ml.toQuantized(groupSize: 64, bits: 8, mode: .affine)
        let q = try XCTUnwrap(quantized as? QuantizedDeepseekV4MultiLinear)
        let got = q(x)
        eval(want, got)

        XCTAssertEqual(got.shape, want.shape)
        let denom = abs(want).max().item(Float.self)
        let relErr = abs(got - want).max().item(Float.self) / max(denom, 1e-6)
        XCTAssertLessThan(relErr, 0.05, "8-bit quantized wo_a must track the float output")
    }

    /// End-to-end: a 4-bit-style quantize pass over the whole model (the load
    /// path for community quantized checkpoints) must leave the model runnable —
    /// in particular wo_a must convert instead of breaking the forward pass.
    func testModelSurvivesQuantizePass() throws {
        let config = try makeConfig()
        let model = makeModel(config)

        quantize(model: model) { path, module in
            // Same predicate shape as Load.swift: quantize everything quantizable
            // whose innermost dim fits the minimum supported group size (32).
            if let ml = module as? DeepseekV4MultiLinear, ml.weight.dim(-1) % 32 == 0 {
                return (32, 4, .affine)
            }
            if let linear = module as? Linear, linear.weight.dim(-1) % 32 == 0 {
                return (32, 4, .affine)
            }
            return nil
        }

        var sawQuantizedWoA = false
        model.visit { path, module in
            if module is QuantizedDeepseekV4MultiLinear { sawQuantizedWoA = true }
        }
        XCTAssertTrue(sawQuantizedWoA, "wo_a must be quantizable via the standard pass")

        let cache = model.makeCache(parameters: GenerateParameters())
        let logits = model(MLXArray([Int32(1), 2, 3], [1, 3]), cache: cache)
        eval(logits)
        assertFinite(logits, "quantized model forward must be finite")
    }

    // MARK: - Sanitize

    func testSanitizeRemapsCheckpointKeys() throws {
        let config = try makeConfig(compressRatios: [0, 4])
        let model = DeepseekV4Model(config)

        var weights: [String: MLXArray] = [
            "embed.weight": zeros([64, 32]),
            "norm.weight": zeros([32]),
            "head.weight": zeros([64, 32]),
            "hc_head_fn": zeros([4, 4 * 32]),
            "hc_head_base": zeros([4]),
            "hc_head_scale": ones([1]),
            "layers.0.hc_attn_fn": zeros([24, 128]),
            "layers.0.ffn.gate.bias": zeros([8]),
            "layers.0.ffn.gate.tid2eid": zeros([64, 2], dtype: .float32),
            "layers.1.attn.wo_a.weight": zeros([2 * 8, 64]),
        ]
        for e in 0 ..< 8 {
            weights["layers.0.ffn.experts.\(e).w1.weight"] = zeros([16, 32])
        }

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.norm.weight"])
        XCTAssertNotNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["model.hc_head.fn"])
        XCTAssertNotNil(sanitized["model.layers.0.attn_hc.fn"])
        XCTAssertNotNil(sanitized["model.layers.0.ffn.gate.e_score_correction_bias"])

        let tid2eid = try XCTUnwrap(sanitized["model.layers.0.ffn.gate.tid2eid"])
        XCTAssertEqual(tid2eid.dtype, .int32, "hash tables must be cast to int32")

        let stacked = try XCTUnwrap(sanitized["model.layers.0.ffn.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(stacked.shape, [8, 16, 32], "per-expert weights must stack")
        XCTAssertNil(sanitized["model.layers.0.ffn.experts.0.w1.weight"])

        let woA = try XCTUnwrap(sanitized["model.layers.1.attn.wo_a.weight"])
        XCTAssertEqual(woA.shape, [2, 8, 64], "wo_a must reshape to [groups, rank, -1]")
    }
}
