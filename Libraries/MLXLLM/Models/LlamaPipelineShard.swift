// Copyright © 2026 Eigen Labs.
//
// LlamaPipelineShard -- a per-rank SLICE of a Llama model for pipeline-parallel
// inference across a cluster (Darkbloom). Each rank instantiates only the
// transformer blocks in its owned half-open layer interval [start, end), so the
// aggregate cluster RAM -- not any single machine's -- bounds the model size.
//
//   - rank 0 (head): owns embed_tokens + layers[start..<end]
//   - last rank (tail): owns layers[start..<end] + final norm + lm_head
//   - middle ranks: only layers[start..<end]
//
// It reuses the EXISTING `LlamaTransformerBlock` (so attention/MLP/RoPE behavior
// is identical to the monolithic model), and a custom weight loader that reads
// only this rank's safetensors keys, remapping global layer index i ->
// local index (i - start) so MLXNN's `update(parameters:)` matches.
//
// This file lives inside MLXLLM because LlamaTransformerBlock is internal to the
// module. It is additive -- the single-node LlamaModel path is untouched.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The half-open layer interval `[start, end)` a rank owns, plus the model's
/// total layer count (so head/tail roles can be derived).
public struct LlamaShardRange: Sendable, Equatable {
    public let start: Int
    public let end: Int
    public let totalLayers: Int
    public init(start: Int, end: Int, totalLayers: Int) {
        self.start = start
        self.end = end
        self.totalLayers = totalLayers
    }
    public var isHead: Bool { start == 0 }
    public var isTail: Bool { end == totalLayers }
    public var count: Int { end - start }
}

/// One rank's slice of a Llama model. Owns only its layers (+ embed on head,
/// norm+lm_head on tail). Not an `LLMModel` -- it exposes the partial-forward
/// primitives the cluster pipeline drives (`embed` / `runOwnedLayers` /
/// `projectToLogits`) rather than a full `callAsFunction`.
public class LlamaPipelineShard: Module {

    public let range: LlamaShardRange
    private let args: LlamaConfiguration

    // Owned submodules. Optionals are nil unless this rank owns that role,
    // so they neither allocate nor expect weights.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    let layers: [LlamaTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LlamaConfiguration, range: LlamaShardRange) {
        precondition(args.vocabularySize > 0)
        precondition(range.start >= 0 && range.end <= args.hiddenLayers && range.start < range.end)
        self.args = args
        self.range = range

        if range.isHead {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        }
        self.layers = (range.start ..< range.end).map { _ in LlamaTransformerBlock(args) }
        if range.isTail {
            self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            if !args.tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }
        }
    }

    // MARK: - Partial forward primitives

    /// HEAD only: embed token ids into the initial hidden state.
    public func embed(_ tokens: MLXArray) -> MLXArray {
        guard let embedTokens else {
            fatalError("embed() called on non-head shard (rank does not own embed_tokens)")
        }
        return embedTokens(tokens)
    }

    /// Run this rank's owned transformer blocks on a hidden state.
    /// `cache` (if provided) must be sized to this rank's layer count.
    public func runOwnedLayers(_ hidden: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = hidden
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return h
    }

    /// TAIL only: final norm + lm_head -> vocabulary logits.
    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        guard let norm else {
            fatalError("projectToLogits() called on non-tail shard (rank does not own norm/lm_head)")
        }
        let normed = norm(hidden)
        if let lmHead {
            return lmHead(normed)
        }
        // Tied embeddings: project through the (head's) embedding matrix. In a
        // pipeline split the tail does NOT own embed_tokens, so tied-embedding
        // models must be configured with an explicit lm_head for the tail, or
        // the embedding weight replicated to the tail. Llama-3.3-70B ships an
        // untied lm_head, so this path is not exercised for the target model.
        fatalError("tied-embedding tail projection requires embed_tokens on the tail; "
            + "use a model with an explicit lm_head (Llama-3.3-70B has one)")
    }

    // MARK: - KV cache sizing

    /// Number of KV caches this rank needs (one per owned layer).
    public var ownedLayerCount: Int { layers.count }
}
