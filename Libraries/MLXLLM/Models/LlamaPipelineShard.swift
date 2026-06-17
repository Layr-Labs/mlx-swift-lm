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
    /// Embedding matrix. Present on the head (to embed tokens) AND on a tied-
    /// embedding tail (to project logits via `asLinear`, mirroring the
    /// monolithic LlamaModel). The loader replicates `embed_tokens` weights to
    /// the tail when the model is tied.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    let layers: [LlamaTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// Whether this model ties input/output embeddings (no separate lm_head).
    public let tieWordEmbeddings: Bool

    public init(_ args: LlamaConfiguration, range: LlamaShardRange) {
        precondition(args.vocabularySize > 0)
        precondition(range.start >= 0 && range.end <= args.hiddenLayers && range.start < range.end)
        self.args = args
        self.range = range
        self.tieWordEmbeddings = args.tieWordEmbeddings

        // embed_tokens lives on the head (always) and on a tied tail (for the
        // output projection). A single-node head==tail keeps just one copy.
        let needsEmbed = range.isHead || (range.isTail && args.tieWordEmbeddings)
        if needsEmbed {
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
    ///
    /// `evalEvery` forces an `eval()` every N layers so a single Metal command
    /// buffer never spans all owned layers — without it, ~40 layers of a 70B
    /// run as one buffer and trip macOS's ~5s GPU watchdog
    /// (kIOGPUCommandBufferCallbackErrorTimeout). 0 disables the periodic eval.
    public func runOwnedLayers(_ hidden: MLXArray, cache: [KVCache]? = nil, evalEvery: Int = 8) -> MLXArray {
        var h = hidden
        // Derive the mask from the cache via makeMask so a batched cache
        // (BatchKVCache) supplies its per-row left-padding-aware mask for ragged
        // batches. For KVCacheSimple this returns .causal/.none — identical to
        // the previous plain-causal behavior, so the B=1 path is unchanged.
        let mask = makeAttentionMask(n: h.dim(1), cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if evalEvery > 0 && (i + 1) % evalEvery == 0 {
                eval(h)
            }
        }
        return h
    }

    /// TAIL only: final norm + lm_head -> vocabulary logits.
    ///
    /// `lastPositionOnly` projects just the final sequence position (all that's
    /// needed to sample the next token), avoiding an [seq × vocab] matmul over
    /// every position — a large, watchdog-tripping op for long prompts.
    public func projectToLogits(_ hidden: MLXArray, lastPositionOnly: Bool = true) -> MLXArray {
        guard let norm else {
            fatalError("projectToLogits() called on non-tail shard (rank does not own norm/lm_head)")
        }
        // hidden is [1, seq, dim]; slice to the last position before the big
        // vocabulary projection when only the next token is needed.
        let h = lastPositionOnly && hidden.ndim == 3
            ? hidden[0..., (hidden.dim(1) - 1)..., 0...]
            : hidden
        let normed = norm(h)
        if let lmHead {
            return lmHead(normed)
        }
        // Tied embeddings: project through the embedding matrix (replicated to
        // the tail by the loader), exactly as the monolithic LlamaModel does.
        if let embedTokens {
            return embedTokens.asLinear(normed)
        }
        fatalError("tied-embedding tail is missing its embed_tokens copy (loader bug)")
    }

    // MARK: - KV cache sizing

    /// Number of KV caches this rank needs (one per owned layer).
    public var ownedLayerCount: Int { layers.count }

    /// Batched KV caches for this rank's owned layers (continuous-batching path).
    /// Llama is full-attention throughout ⇒ one `BatchKVCache` per owned layer.
    /// `leftPadding` has one entry per batch row.
    public func makeBatchedCaches(leftPadding: [Int]) -> [any KVCache] {
        (0 ..< layers.count).map { _ in BatchKVCache(leftPadding: leftPadding) }
    }
}
