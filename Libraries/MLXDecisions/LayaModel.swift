// Copyright © 2026 Eigen Labs Inc.
// Native Swift port of mizorewww/laya-mlx (Apache-2.0). See NOTICE.
import Foundation
import MLX
import MLXNN

/// Bidirectional ModernBERT and Laya's decision/action heads. No KV cache or decoding.
final class LayaModel {
    let encoder: LayaEncoderConfiguration
    let agent: LayaAgentConfiguration
    private let weights: [String: MLXArray]

    init(directory: URL, dtype: DType = .float16) throws {
        let decoder = JSONDecoder()
        encoder = try decoder.decode(
            LayaEncoderConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "encoder/config.json")))
        agent = try decoder.decode(
            LayaAgentConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "rl_agent_config.json")))
        try encoder.validate()
        try agent.validate(encoder: encoder)
        let source = try MLX.loadArrays(url: directory.appending(path: "model.safetensors"))
        var mapped: [String: MLXArray] = [:]
        for (original, tensor) in source {
            var name = original.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
                .replacingOccurrences(of: ".in_proj_bias", with: ".in_proj.bias")
            for prefix in ["scorer", "act_head"]
            where name.hasPrefix(prefix + ".")
                && !name.hasPrefix(prefix + ".layers.")
            {
                name = prefix + ".layers." + name.dropFirst(prefix.count + 1)
            }
            guard mapped[name] == nil else {
                throw LayaError.invalidCheckpoint("Duplicate weight key")
            }
            mapped[name] = tensor.asType(dtype)
        }
        weights = mapped
        try validateWeights()
        eval(Array(weights.values))
    }

    private func validateWeights() throws {
        var expected: [String: [Int]] = [:]
        func linear(_ key: String, _ input: Int, _ output: Int, _ bias: Bool = true) {
            expected[key + ".weight"] = [output, input]
            if bias { expected[key + ".bias"] = [output] }
        }
        func norm(_ key: String, _ dimension: Int, _ bias: Bool = true) {
            expected[key + ".weight"] = [dimension]
            if bias { expected[key + ".bias"] = [dimension] }
        }
        let d = encoder.hidden_size
        expected["encoder.embeddings.tok_embeddings.weight"] = [encoder.vocab_size, d]
        norm("encoder.embeddings.norm", d, encoder.norm_bias)
        norm("encoder.final_norm", d, encoder.norm_bias)
        for i in 0 ..< encoder.num_hidden_layers {
            let p = "encoder.layers.\(i)"
            if i > 0 { norm(p + ".attn_norm", d, encoder.norm_bias) }
            norm(p + ".mlp_norm", d, encoder.norm_bias)
            linear(p + ".attn.Wqkv", d, 3 * d, encoder.attention_bias)
            linear(p + ".attn.Wo", d, d, encoder.attention_bias)
            linear(p + ".mlp.Wi", d, 2 * encoder.intermediate_size, encoder.mlp_bias)
            linear(p + ".mlp.Wo", encoder.intermediate_size, d, encoder.mlp_bias)
        }
        for i in 0 ..< agent.head_layers {
            let p = "head.layers.\(i)"
            norm(p + ".norm1", d)
            norm(p + ".norm2", d)
            linear(p + ".self_attn.in_proj", d, 3 * d)
            linear(p + ".self_attn.out_proj", d, d)
            linear(p + ".linear1", d, 4 * d)
            linear(p + ".linear2", 4 * d, d)
        }
        expected["type_emb.weight"] = [3, d]
        expected["temperature"] = [3]
        norm("scorer.layers.0", d)
        linear("scorer.layers.1", d, d)
        linear("scorer.layers.3", d, 1)
        linear("act_head.layers.0", d + 4, 256)
        linear("act_head.layers.2", 256, agent.act_costs.count + 1)
        guard Set(weights.keys) == Set(expected.keys) else {
            throw LayaError.invalidCheckpoint("Checkpoint parameter names do not match Laya")
        }
        for (name, shape) in expected where weights[name]!.shape != shape {
            throw LayaError.invalidCheckpoint("Invalid checkpoint shape for \(name)")
        }
    }

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        let weight = weights[key + ".weight"]!.T
        // Match MLX nn.Linear: fusing bias avoids rounding the FP16 matmul
        // before addition, which materially changes the large action logits.
        if let bias = weights[key + ".bias"] { return addMM(bias, x, weight) }
        return matmul(x, weight)
    }
    private func norm(_ x: MLXArray, _ key: String, eps: Float = 1e-5) -> MLXArray {
        MLXFast.layerNorm(
            x, weight: weights[key + ".weight"], bias: weights[key + ".bias"], eps: eps)
    }
    private func attention(
        _ x: MLXArray, mask: MLXArray, key: String,
        heads: Int, rope: Float? = nil, encoderLayer: Bool
    ) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let h = x.dim(2) / heads
        let qkv = linear(x, key + (encoderLayer ? ".Wqkv" : ".in_proj"))
            .reshaped(b, n, 3, heads, h)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        if let rope {
            q = MLXFast.RoPE(q, dimensions: h, traditional: false, base: rope, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: h, traditional: false, base: rope, scale: 1, offset: 0)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(h), -0.5), mask: .array(mask)
        )
        .transposed(0, 2, 1, 3).reshaped(b, n, -1)
        return linear(out, key + (encoderLayer ? ".Wo" : ".out_proj"))
    }

    func callAsFunction(
        ids: MLXArray, valid: MLXArray, markers: MLXArray,
        markerMask: MLXArray, types: MLXArray
    ) -> (MLXArray, MLXArray) {
        let b = ids.dim(0)
        let n = ids.dim(1)
        let full = valid.reshaped(b, 1, 1, n)
        let positions = MLXArray(0 ..< n)
        let distance = abs(positions[.newAxis, 0...] - positions[0..., .newAxis])
        let window = distance .<= (encoder.local_attention / 2)
        let local = logicalAnd(
            logicalOr(
                window.reshaped(1, 1, n, n),
                logicalNot(valid.reshaped(b, 1, n, 1))), full)
        var h = norm(
            weights["encoder.embeddings.tok_embeddings.weight"]![ids],
            "encoder.embeddings.norm", eps: encoder.norm_eps)
        for i in 0 ..< encoder.num_hidden_layers {
            let p = "encoder.layers.\(i)"
            let x = i == 0 ? h : norm(h, p + ".attn_norm", eps: encoder.norm_eps)
            h =
                h
                + attention(
                    x, mask: encoder.isGlobal(i) ? full : local,
                    key: p + ".attn", heads: encoder.num_attention_heads,
                    rope: encoder.ropeBase(i), encoderLayer: true)
            let parts = split(
                linear(
                    norm(h, p + ".mlp_norm", eps: encoder.norm_eps),
                    p + ".mlp.Wi"), parts: 2, axis: -1)
            h = h + linear(gelu(parts[0]) * parts[1], p + ".mlp.Wo")
        }
        h = norm(h, "encoder.final_norm", eps: encoder.norm_eps)
        h = h + weights["type_emb.weight"]![types][0..., .newAxis, 0...]
        for i in 0 ..< agent.head_layers {
            let p = "head.layers.\(i)"
            h =
                h
                + attention(
                    norm(h, p + ".norm1"), mask: full,
                    key: p + ".self_attn", heads: max(1, encoder.hidden_size / 64),
                    encoderLayer: false)
            h = h + linear(relu(linear(norm(h, p + ".norm2"), p + ".linear1")), p + ".linear2")
        }
        let selected = h[MLXArray(0 ..< b)[0..., .newAxis], markers]
        let logits = linear(
            gelu(
                linear(
                    norm(selected, "scorer.layers.0"),
                    "scorer.layers.1")), "scorer.layers.3"
        ).squeezed(axis: -1).asType(.float32)
        let masked = MLX.where(markerMask, logits, MLXArray(Float(-10000)))
        let p = softmax(masked, axis: -1)
        let k = maximum(sum(markerMask.asType(.int32), axis: -1), MLXArray(2)).asType(.float32)
        let entropy = -sum(p * log(maximum(p, MLXArray(Float(1e-9)))), axis: -1) / log(k)
        let sorted = MLX.sorted(p, axis: -1)
        let first = sorted[0..., -1]
        let second = sorted[0..., -2]
        let features = stacked([first, first - second, entropy, k / 255], axis: -1)
        let pooled = concatenated([h[0..., 0, 0...].asType(.float32), features], axis: -1)
            .asType(weights["act_head.layers.0.weight"]!.dtype)
        let action = linear(gelu(linear(pooled, "act_head.layers.0")), "act_head.layers.2")
        return (masked, action.asType(.float32))
    }
}
