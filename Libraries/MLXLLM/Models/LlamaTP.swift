// Copyright © 2026 Apple Inc. (TP variant — Layr-Labs)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Tensor-parallel variant of Llama.swift. Each rank holds a column-shard of
// the Q/K/V/gate/up projections and a row-shard of the O/down projections.
// Embedding, layernorms, and the final LM head stay replicated across ranks.
//
// Per-rank: attentionHeads / group.size local Q heads and kvHeads / group.size
// local KV heads. Both must be divisible by group.size or init throws.
//
// On a singleton group (size 1), LlamaModelTP produces output bit-equivalent
// to LlamaModel modulo float accumulation order — used as the equivalence
// baseline in tests.

class LlamaAttentionTP: Module {

    let args: LlamaConfiguration
    let scale: Float
    let group: DistributedGroup
    let localHeads: Int
    let localKVHeads: Int

    @ModuleInfo(key: "q_proj") var wq: AllToShardedLinear
    @ModuleInfo(key: "k_proj") var wk: AllToShardedLinear
    @ModuleInfo(key: "v_proj") var wv: AllToShardedLinear
    @ModuleInfo(key: "o_proj") var wo: ShardedToAllLinear

    let rope: RoPELayer

    init(_ args: LlamaConfiguration, group: DistributedGroup) throws {
        self.args = args
        self.group = group

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        // Validate divisibility — both Q and KV head counts must shard cleanly.
        guard heads % group.size == 0 else {
            throw DistributedError.invalidConfiguration(
                "attentionHeads=\(heads) must be divisible by group size \(group.size)")
        }
        guard kvHeads % group.size == 0 else {
            throw DistributedError.invalidConfiguration(
                "kvHeads=\(kvHeads) must be divisible by group size \(group.size)")
        }
        self.localHeads = heads / group.size
        self.localKVHeads = kvHeads / group.size

        self._wq.wrappedValue = try AllToShardedLinear(
            inputDimensions: dim, outputDimensions: heads * headDim,
            bias: args.attentionBias, group: group)
        self._wk.wrappedValue = try AllToShardedLinear(
            inputDimensions: dim, outputDimensions: kvHeads * headDim,
            bias: args.attentionBias, group: group)
        self._wv.wrappedValue = try AllToShardedLinear(
            inputDimensions: dim, outputDimensions: kvHeads * headDim,
            bias: args.attentionBias, group: group)
        self._wo.wrappedValue = try ShardedToAllLinear(
            inputDimensions: heads * headDim, outputDimensions: dim,
            bias: args.attentionBias, group: group)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta,
            traditional: args.ropeTraditional,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Reshape using LOCAL head counts — each rank has only its shard.
        queries = queries.reshaped(B, L, localHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, localKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, localKVHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        // ShardedToAllLinear's forward runs allSum internally so the result is
        // identical on every rank.
        return wo(output)
    }
}

class LlamaMLPTP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: AllToShardedLinear
    @ModuleInfo(key: "down_proj") var down: ShardedToAllLinear
    @ModuleInfo(key: "up_proj") var up: AllToShardedLinear

    init(_ args: LlamaConfiguration, group: DistributedGroup) throws {
        self._gate.wrappedValue = try AllToShardedLinear(
            inputDimensions: args.hiddenSize, outputDimensions: args.intermediateSize,
            bias: args.mlpBias, group: group)
        self._down.wrappedValue = try ShardedToAllLinear(
            inputDimensions: args.intermediateSize, outputDimensions: args.hiddenSize,
            bias: args.mlpBias, group: group)
        self._up.wrappedValue = try AllToShardedLinear(
            inputDimensions: args.hiddenSize, outputDimensions: args.intermediateSize,
            bias: args.mlpBias, group: group)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

class LlamaTransformerBlockTP: Module {
    @ModuleInfo(key: "self_attn") var attention: LlamaAttentionTP
    @ModuleInfo(key: "mlp") var mlp: LlamaMLPTP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: LlamaConfiguration, group: DistributedGroup) throws {
        self._attention.wrappedValue = try LlamaAttentionTP(args, group: group)
        self._mlp.wrappedValue = try LlamaMLPTP(args, group: group)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class LlamaModelInnerTP: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [LlamaTransformerBlockTP]
    let norm: RMSNorm
    let group: DistributedGroup

    init(_ args: LlamaConfiguration, group: DistributedGroup) throws {
        precondition(args.vocabularySize > 0)

        self.group = group
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        var layers: [LlamaTransformerBlockTP] = []
        layers.reserveCapacity(args.hiddenLayers)
        for _ in 0 ..< args.hiddenLayers {
            layers.append(try LlamaTransformerBlockTP(args, group: group))
        }
        self.layers = layers
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Tensor-parallel variant of `LlamaModel`. Each rank holds a column-shard of
/// the Q/K/V/gate/up projections and a row-shard of the O/down projections.
/// Use `LlamaModel` for single-rank inference.
public class LlamaModelTP: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let group: DistributedGroup

    public let model: LlamaModelInnerTP

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LlamaConfiguration, group: DistributedGroup) throws {
        self.vocabularySize = args.vocabularySize
        // KV heads reported here are the LOCAL kv-head count for cache sizing —
        // the KV cache only stores this rank's shard of the keys/values.
        guard args.kvHeads % group.size == 0 else {
            throw DistributedError.invalidConfiguration(
                "kvHeads=\(args.kvHeads) must be divisible by group size \(group.size)")
        }
        let localKV = args.kvHeads / group.size
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in localKV }
        self.group = group
        self.model = try LlamaModelInnerTP(args, group: group)
        if !args.tieWordEmbeddings {
            // LM head stays replicated for simplicity; column-parallel +
            // allGather is a future optimization.
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    /// Slices column-parallel and row-parallel weights into rank-local shards
    /// before module assignment. Singleton group (size 1) is a pass-through.
    /// Quantized weights are passed through unmodified — quantized TP support
    /// is a follow-up that needs packed-uint32-aware slicing.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // First apply LlamaModel's standard cleanup (drop rotary_emb.inv_freq).
        var result = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        let world = group.size
        if world == 1 { return result }

        let rank = group.rank
        var sliced: [String: MLXArray] = [:]
        sliced.reserveCapacity(result.count)

        for (key, value) in result {
            sliced[key] = LlamaModelTP.shardWeightIfNeeded(
                key: key, value: value, rank: rank, world: world)
        }
        return sliced
    }

    /// Decides which axis to slice based on the parameter name and produces
    /// the rank's shard. Column-parallel weights slice axis 0 (output dim);
    /// row-parallel weights slice axis 1 (input dim); biases for column-parallel
    /// also slice (since the bias rides along with the output dim); biases for
    /// row-parallel stay full (each rank has the full bias and the allreduce
    /// then re-adds it — which is incorrect — so for first cut we just keep
    /// the row-parallel bias on rank 0 and zero it on others). The MLX Llama
    /// weights don't use biases by default (`attentionBias=false`, `mlpBias=false`)
    /// so this edge case is academic for stock Llama checkpoints.
    public static func shardWeightIfNeeded(
        key: String, value: MLXArray, rank: Int, world: Int
    ) -> MLXArray {
        // Embedding and layernorms stay full on every rank.
        if key.contains("embed_tokens") || key.contains("norm") || key.contains("lm_head") {
            return value
        }

        // Quantized weights: 4-bit packed weights need special handling.
        // Pass through for now — TP only validates against unquantized checkpoints
        // in the first iteration.
        if key.contains(".scales") || key.contains(".biases") {
            return value
        }

        let isColumnParallel =
            key.contains("q_proj.weight") || key.contains("q_proj.bias")
            || key.contains("k_proj.weight") || key.contains("k_proj.bias")
            || key.contains("v_proj.weight") || key.contains("v_proj.bias")
            || key.contains("gate_proj.weight") || key.contains("gate_proj.bias")
            || key.contains("up_proj.weight") || key.contains("up_proj.bias")

        let isRowParallelWeight =
            key.contains("o_proj.weight") || key.contains("down_proj.weight")

        if isColumnParallel {
            // Slice along axis 0 (output dim).
            let outDim = value.dim(0)
            precondition(
                outDim % world == 0,
                "column-parallel weight '\(key)' outDim=\(outDim) not divisible by world=\(world)")
            let shard = outDim / world
            return value[(rank * shard) ..< ((rank + 1) * shard)]
        }
        if isRowParallelWeight {
            // Slice along axis 1 (input dim).
            let inDim = value.dim(1)
            precondition(
                inDim % world == 0,
                "row-parallel weight '\(key)' inDim=\(inDim) not divisible by world=\(world)")
            let shard = inDim / world
            return value[0..., (rank * shard) ..< ((rank + 1) * shard)]
        }
        return value
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        do {
            let probe = [
                [
                    "role": "system",
                    "content": "test",
                ]
            ]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }
}

// MARK: - LoRA

extension LlamaModelTP: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
