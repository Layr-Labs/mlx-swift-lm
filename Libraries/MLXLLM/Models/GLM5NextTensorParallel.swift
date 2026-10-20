// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// One rank of a tensor-parallel GLM text model: every layer on every rank,
/// each holding `1/size` of each head- or channel-parallel weight.
///
/// Split (rank-local contiguous half):
/// - KDA: per-head q/k/v projections and short convolutions, f/g second
///   projections, `b_proj`, `A_log`, `dt_bias` (axis 0); `o_proj` input columns.
/// - DSA: `q_b_proj` and `kv_b_proj` head rows (axis 0); `o_proj` input columns.
///   The latent projections, their norms and the whole indexer are replicated,
///   so every rank selects the same tokens.
/// - Dense MLP, shared experts and every routed expert: gate/up output rows,
///   down input columns. The router is replicated, so ranks route identically.
/// Everything else (embedding, hyper-connections, norms, router, head) is
/// replicated. Each layer then needs two sums across ranks: the attention
/// output and the feed-forward output (`GLM5NextTextLayer.reduce`).
public struct GLM5NextTensorShard: Sendable, Equatable {
    public let rank: Int
    public let size: Int

    public init(rank: Int, size: Int) throws {
        guard size == 2, (0 ..< size).contains(rank) else {
            throw GLM5NextLoadError.invalid("GLM tensor parallel supports two ranks")
        }
        self.rank = rank
        self.size = size
    }

    /// Which axis of a stored tensor this rank keeps a part of; nil replicates it.
    public static func splitAxis(_ name: String) -> Int? {
        let layer = "model.language_model.layers."
        guard name.hasPrefix(layer) else { return nil }
        // weight / scales / biases of packed linears split together; the NVFP4
        // second-level scale is one scalar per matrix and stays replicated.
        guard !name.hasSuffix(".weight_scale_2") else { return nil }
        let rows = [
            ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.",
            ".self_attn.q_conv1d.", ".self_attn.k_conv1d.", ".self_attn.v_conv1d.",
            ".self_attn.f_b_proj.", ".self_attn.g_b_proj.", ".self_attn.b_proj.",
            ".self_attn.q_b_proj.", ".self_attn.kv_b_proj.",
            ".gate_proj.", ".up_proj.",
        ]
        if rows.contains(where: { name.contains($0) }) { return 0 }
        if name.hasSuffix(".self_attn.A_log") || name.hasSuffix(".self_attn.dt_bias") { return 0 }
        if name.contains(".self_attn.o_proj.") || name.contains(".down_proj.") { return 1 }
        return nil
    }

    /// This rank's part of a full stored tensor, as an array owning its own
    /// storage (a slice alone would keep the whole source buffer alive).
    public func slice(_ name: String, _ full: MLXArray) throws -> MLXArray {
        guard let axis = Self.splitAxis(name) else { return full }
        guard full.ndim > axis, full.dim(axis) % size == 0 else {
            throw GLM5NextLoadError.invalid("tensor-parallel split of \(name) \(full.shape)")
        }
        let width = full.dim(axis) / size
        let range = (rank * width) ..< ((rank + 1) * width)
        let part = axis == 0 ? full[range] : full[0..., range]
        return stacked([part]).squeezed(axis: 0)
    }

    /// The configuration every module of this rank is built from: the same
    /// model with half the attention heads and feed-forward channels.
    public func localConfiguration(_ c: GLM5NextTextConfiguration) throws -> GLM5NextTextConfiguration {
        guard c.attentionHeads % size == 0, c.linearAttention.numHeads % size == 0,
            c.intermediateSize % size == 0, c.moeIntermediateSize % size == 0
        else { throw GLM5NextLoadError.invalid("GLM geometry does not divide across ranks") }
        var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(c)) as? [String: Any] ?? [:]
        func halve(_ key: String, in dictionary: inout [String: Any]) throws {
            guard let value = dictionary[key] as? Int else {
                throw GLM5NextLoadError.invalid("tensor-parallel configuration lacks \(key)")
            }
            dictionary[key] = value / size
        }
        try halve("num_attention_heads", in: &object)
        // MLA: one latent key/value head per query head.
        try halve("num_key_value_heads", in: &object)
        try halve("intermediate_size", in: &object)
        try halve("moe_intermediate_size", in: &object)
        guard var linear = object["linear_attn_config"] as? [String: Any] else {
            throw GLM5NextLoadError.invalid("tensor-parallel configuration lacks linear_attn_config")
        }
        try halve("num_heads", in: &linear)
        object["linear_attn_config"] = linear
        let local = try JSONDecoder().decode(
            GLM5NextTextConfiguration.self, from: JSONSerialization.data(withJSONObject: object))
        guard local.attentionHeads * size == c.attentionHeads,
            local.linearAttention.numHeads * size == c.linearAttention.numHeads,
            local.hiddenSize == c.hiddenSize, local.hiddenLayers == c.hiddenLayers,
            local.layerTypes == c.layerTypes, local.hcMult == c.hcMult
        else { throw GLM5NextLoadError.invalid("tensor-parallel local configuration differs") }
        return local
    }
}
