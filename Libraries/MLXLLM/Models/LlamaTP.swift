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
        // Embedding, layernorms, and the replicated LM head stay full on
        // every rank. Use path-segment-precise checks (not a substring
        // "norm" match) so future TP variants with per-head q_norm/k_norm
        // don't accidentally skip sharding for those.
        if key.contains("embed_tokens") || isLayerNormKey(key) || key.contains("lm_head") {
            return value
        }

        // This fp16 sanitize doesn't handle quantized weights; LlamaModelTPQ
        // owns that path. If a caller loads a quantized checkpoint into
        // LlamaModelTP, the .scales / .biases passes through unsliced and
        // the .weight (uint32 packed) gets sliced along the wrong axis size,
        // producing a shape mismatch at module assignment. Fail loudly here
        // instead of silently producing garbage downstream.
        if key.contains(".scales") || key.contains(".biases") {
            preconditionFailure(
                "LlamaModelTP cannot load quantized weights (got '\(key)'). Use LlamaModelTPQ for 4-bit / 8-bit MLX checkpoints — see makeLlamaTP(args:quantization:group:) factory."
            )
        }

        let isColumnParallel =
            isProjectionKey(key, name: "q_proj")
            || isProjectionKey(key, name: "k_proj")
            || isProjectionKey(key, name: "v_proj")
            || isProjectionKey(key, name: "gate_proj")
            || isProjectionKey(key, name: "up_proj")

        let isRowParallel =
            isProjectionKey(key, name: "o_proj")
            || isProjectionKey(key, name: "down_proj")

        if isColumnParallel {
            // Slice along axis 0 (output dim).
            let outDim = value.dim(0)
            precondition(
                outDim % world == 0,
                "column-parallel weight '\(key)' outDim=\(outDim) not divisible by world=\(world)")
            let shard = outDim / world
            return value[(rank * shard) ..< ((rank + 1) * shard)]
        }
        if isRowParallel {
            // Bias for row-parallel stays full on every rank — addition
            // after the implicit allSum applies once on the summed result.
            if key.hasSuffix(".bias") {
                return value
            }
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

// MARK: - Quantized TP variant
//
// Stock MLX Llama checkpoints are 4-bit quantized. Because mlx-swift's
// `AllToShardedLinear` / `ShardedToAllLinear` and their quantized
// counterparts (`QuantizedAllToShardedLinear` / `QuantizedShardedToAllLinear`)
// are sibling Module classes — NOT in an inheritance hierarchy — we can't
// swap one for the other at runtime the way MLXNN's `quantize()` swaps
// `Linear` for `QuantizedLinear`. So we ship a parallel class hierarchy
// here. Use `makeLlamaTP(args:quantization:group:)` to pick the right
// variant based on the model's quantization config.

class LlamaAttentionTPQ: Module {

    let args: LlamaConfiguration
    let scale: Float
    let group: DistributedGroup
    let localHeads: Int
    let localKVHeads: Int

    @ModuleInfo(key: "q_proj") var wq: QuantizedAllToShardedLinear
    @ModuleInfo(key: "k_proj") var wk: QuantizedAllToShardedLinear
    @ModuleInfo(key: "v_proj") var wv: QuantizedAllToShardedLinear
    @ModuleInfo(key: "o_proj") var wo: QuantizedShardedToAllLinear

    let rope: RoPELayer

    init(
        _ args: LlamaConfiguration, group: DistributedGroup,
        groupSize: Int, bits: Int
    ) throws {
        self.args = args
        self.group = group

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

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

        self._wq.wrappedValue = try QuantizedAllToShardedLinear(
            inputDimensions: dim, outputDimensions: heads * headDim,
            bias: args.attentionBias, groupSize: groupSize, bits: bits, group: group)
        self._wk.wrappedValue = try QuantizedAllToShardedLinear(
            inputDimensions: dim, outputDimensions: kvHeads * headDim,
            bias: args.attentionBias, groupSize: groupSize, bits: bits, group: group)
        self._wv.wrappedValue = try QuantizedAllToShardedLinear(
            inputDimensions: dim, outputDimensions: kvHeads * headDim,
            bias: args.attentionBias, groupSize: groupSize, bits: bits, group: group)
        self._wo.wrappedValue = try QuantizedShardedToAllLinear(
            inputDimensions: heads * headDim, outputDimensions: dim,
            bias: args.attentionBias, groupSize: groupSize, bits: bits, group: group)

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

        queries = queries.reshaped(B, L, localHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, localKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, localKVHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class LlamaMLPTPQ: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: QuantizedAllToShardedLinear
    @ModuleInfo(key: "down_proj") var down: QuantizedShardedToAllLinear
    @ModuleInfo(key: "up_proj") var up: QuantizedAllToShardedLinear

    init(
        _ args: LlamaConfiguration, group: DistributedGroup,
        groupSize: Int, bits: Int
    ) throws {
        self._gate.wrappedValue = try QuantizedAllToShardedLinear(
            inputDimensions: args.hiddenSize, outputDimensions: args.intermediateSize,
            bias: args.mlpBias, groupSize: groupSize, bits: bits, group: group)
        self._down.wrappedValue = try QuantizedShardedToAllLinear(
            inputDimensions: args.intermediateSize, outputDimensions: args.hiddenSize,
            bias: args.mlpBias, groupSize: groupSize, bits: bits, group: group)
        self._up.wrappedValue = try QuantizedAllToShardedLinear(
            inputDimensions: args.hiddenSize, outputDimensions: args.intermediateSize,
            bias: args.mlpBias, groupSize: groupSize, bits: bits, group: group)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

class LlamaTransformerBlockTPQ: Module {
    @ModuleInfo(key: "self_attn") var attention: LlamaAttentionTPQ
    @ModuleInfo(key: "mlp") var mlp: LlamaMLPTPQ

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(
        _ args: LlamaConfiguration, group: DistributedGroup,
        groupSize: Int, bits: Int
    ) throws {
        self._attention.wrappedValue = try LlamaAttentionTPQ(
            args, group: group, groupSize: groupSize, bits: bits)
        self._mlp.wrappedValue = try LlamaMLPTPQ(
            args, group: group, groupSize: groupSize, bits: bits)
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

public class LlamaModelInnerTPQ: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [LlamaTransformerBlockTPQ]
    let norm: RMSNorm
    let group: DistributedGroup

    init(
        _ args: LlamaConfiguration, group: DistributedGroup,
        groupSize: Int, bits: Int
    ) throws {
        precondition(args.vocabularySize > 0)

        self.group = group
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        var layers: [LlamaTransformerBlockTPQ] = []
        layers.reserveCapacity(args.hiddenLayers)
        for _ in 0 ..< args.hiddenLayers {
            layers.append(
                try LlamaTransformerBlockTPQ(
                    args, group: group, groupSize: groupSize, bits: bits))
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

/// Quantized tensor-parallel variant of `LlamaModel`. Same structure as
/// `LlamaModelTP` but each linear layer is `QuantizedAllToShardedLinear`
/// or `QuantizedShardedToAllLinear`. Use this for 4-bit MLX checkpoints
/// (the common production case). Embedding stays unquantized + replicated
/// to match the upstream `LlamaModel` behavior.
public class LlamaModelTPQ: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let group: DistributedGroup
    public let groupSize: Int
    public let bits: Int

    public let model: LlamaModelInnerTPQ

    @ModuleInfo(key: "lm_head") var lmHead: QuantizedLinear?

    public init(
        _ args: LlamaConfiguration, group: DistributedGroup,
        groupSize: Int = 64, bits: Int = 4
    ) throws {
        self.vocabularySize = args.vocabularySize
        guard args.kvHeads % group.size == 0 else {
            throw DistributedError.invalidConfiguration(
                "kvHeads=\(args.kvHeads) must be divisible by group size \(group.size)")
        }
        let localKV = args.kvHeads / group.size
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in localKV }
        self.group = group
        self.groupSize = groupSize
        self.bits = bits
        self.model = try LlamaModelInnerTPQ(
            args, group: group, groupSize: groupSize, bits: bits)
        if !args.tieWordEmbeddings {
            // LM head stays replicated + quantized (no row/column split).
            // Each rank holds a full QuantizedLinear; cheap to broadcast and
            // avoids the allreduce / allgather complexity for the last step.
            self._lmHead.wrappedValue = QuantizedLinear(
                args.hiddenSize, args.vocabularySize, bias: false,
                groupSize: groupSize, bits: bits)
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

    /// Slices a 4-bit packed quantized weight tree into per-rank shards.
    ///
    /// Packed-uint32 weight layout (MLX 4-bit): `weight` has shape
    /// `[outDim, inDim/8]` (8 weights per uint32). `scales`/`biases` have
    /// shape `[outDim, inDim/groupSize]` (one fp16 per quantization group).
    ///
    /// Column-parallel (Q/K/V/gate/up): slice all three along axis 0 (outDim).
    /// Row-parallel (O/down): slice along axis 1. For weight, axis-1 length
    /// is `inDim/8` so `inDim` must be divisible by `8 × world_size = 16`
    /// for `world=2`. For scales/biases, axis-1 length is `inDim/groupSize`
    /// so `inDim` must be divisible by `groupSize × world_size = 128` for
    /// default groupSize=64, world=2. Llama hidden dims (4096, 8192, ...)
    /// satisfy both constraints.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
        let world = group.size
        if world == 1 { return result }
        let rank = group.rank

        var sliced: [String: MLXArray] = [:]
        sliced.reserveCapacity(result.count)
        for (key, value) in result {
            sliced[key] = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
                key: key, value: value, rank: rank, world: world)
        }
        return sliced
    }

    /// Slices a single quantized parameter tensor along the right axis
    /// based on its key. Returns the value unchanged if the key doesn't
    /// belong to a Q/K/V/O/gate/up/down projection.
    ///
    /// Embedding, layernorms, and the (replicated) LM head are always
    /// returned unchanged.
    public static func shardQuantizedWeightIfNeeded(
        key: String, value: MLXArray, rank: Int, world: Int
    ) -> MLXArray {
        // Don't shard embedding, layernorms, or the replicated LM head.
        if key.contains("embed_tokens") || isLayerNormKey(key) || key.contains("lm_head") {
            return value
        }

        let isColumnParallel =
            isProjectionKey(key, name: "q_proj")
            || isProjectionKey(key, name: "k_proj")
            || isProjectionKey(key, name: "v_proj")
            || isProjectionKey(key, name: "gate_proj")
            || isProjectionKey(key, name: "up_proj")

        let isRowParallel =
            isProjectionKey(key, name: "o_proj")
            || isProjectionKey(key, name: "down_proj")

        if isColumnParallel {
            // Slice along axis 0 (outDim) for weight, scales, biases, bias.
            let outDim = value.dim(0)
            precondition(
                outDim % world == 0,
                "quantized column-parallel '\(key)' outDim=\(outDim) not divisible by world=\(world)")
            let shard = outDim / world
            return value[(rank * shard) ..< ((rank + 1) * shard)]
        }
        if isRowParallel {
            // Bias for row-parallel stays full on every rank — addition after
            // allSum applies once on the summed result. Match LlamaModelTP's
            // convention: row-parallel biases pass through unsliced.
            if key.hasSuffix(".bias") {
                return value
            }
            // Slice along axis 1 (inDim-derived).
            let secondDim = value.dim(1)
            precondition(
                secondDim % world == 0,
                "quantized row-parallel '\(key)' axis-1=\(secondDim) not divisible by world=\(world)")
            let shard = secondDim / world
            return value[0..., (rank * shard) ..< ((rank + 1) * shard)]
        }
        return value
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        do {
            let probe = [
                ["role": "system", "content": "test"]
            ]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }
}

extension LlamaModelTPQ: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

// MARK: - Shared helpers + factory

/// Path-segment-precise check for `*_layernorm` and the final `model.norm`
/// (so future TP variants that ALSO have `q_norm`/`k_norm` per-head norms
/// don't get caught by a naive substring match on "norm").
private func isLayerNormKey(_ key: String) -> Bool {
    key.contains("input_layernorm")
        || key.contains("post_attention_layernorm")
        || key.hasSuffix("model.norm.weight")
        || key.hasSuffix("model.norm.bias")
}

/// Returns true iff `key` names a leaf parameter (weight, bias, scales,
/// biases) of the linear layer named `name`. Uses path-segment boundaries
/// to avoid false matches (e.g. "q_proj_extra").
private func isProjectionKey(_ key: String, name: String) -> Bool {
    key.contains(".\(name).weight")
        || key.contains(".\(name).bias")
        || key.contains(".\(name).scales")
        || key.contains(".\(name).biases")
}

/// Build the right TP variant for the model based on quantization config.
///
/// - Parameters:
///   - args: Llama configuration parsed from `config.json`.
///   - quantization: Quantization config from the checkpoint (`nil` for
///     fp16/bf16 checkpoints; non-nil for MLX 4-bit / 8-bit quantized).
///   - group: The distributed group across which to shard.
///
/// - Returns: An opaque `LLMModel` that's either `LlamaModelTP` (fp16) or
///   `LlamaModelTPQuantized` based on `quantization`. Caller doesn't have to
///   downcast — the LLMModel protocol covers everything needed for inference.
public func makeLlamaTP(
    args: LlamaConfiguration,
    quantization: BaseConfiguration.Quantization?,
    group: DistributedGroup
) throws -> any LLMModel {
    if let q = quantization {
        return try LlamaModelTPQ(args, group: group, groupSize: q.groupSize, bits: q.bits)
    }
    return try LlamaModelTP(args, group: group)
}
