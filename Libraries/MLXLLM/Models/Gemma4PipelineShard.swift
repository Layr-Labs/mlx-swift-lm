// Copyright © 2026 Eigen Labs.
//
// Gemma4PipelineShard -- a per-rank SLICE of a Gemma 4 text model for pipeline-
// parallel inference across a cluster (Darkbloom), mirroring GPTOSSPipelineShard.
//
// Like GPT-OSS, Gemma 4's layer list slices cleanly into WHOLE blocks: each
// `Gemma4DecoderLayer` carries its own attention + MLP + (on 26B-A4B) the
// parallel dense+sparse MoE branch, so the per-block MoE rides along with its
// layer automatically — no special MoE handling in the shard. The Gemma-4
// specifics the shard must honor are:
//   * per-layer attention type (sliding vs full) → matching mask + the layer's
//     own RoPE/head-dim (the block already builds the right RoPE from its
//     layerIdx, so the shard only supplies the correct per-type mask),
//   * tied embeddings → the tail projects logits through embed_tokens.asLinear,
//   * final logit softcap (tanh(x/cap)*cap) → applied in projectToLogits.
//
// IMPORTANT — this shard targets the SHIPPED build
// (mlx-community/gemma-4-26B-A4B-it-qat-4bit and its fp8 sibling), whose config
// has `num_kv_shared_layers = 0` and `hidden_size_per_layer_input = 0`. Those
// two switch OFF cross-layer KV sharing and Per-Layer-Input embeddings — the
// only two features that would force tensors other than the single hidden state
// across the pipeline cut. With both off, a whole-block slice is self-contained
// and the existing one-hop ring loop suffices (same as Llama / GPT-OSS). A build
// with either feature enabled is rejected by the loader.
//
// Lives inside MLXLLM because Gemma4DecoderLayer is internal to the module.
// Additive: the single-node Gemma4TextModel path is untouched.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One rank's slice of a Gemma 4 text model. Owns only its layers (+ embed on
/// the head; norm + tied-embedding logit projection on the tail). Exposes the
/// partial-forward primitives the cluster pipeline drives, not a full
/// callAsFunction.
public class Gemma4PipelineShard: Module {

    public let range: LlamaShardRange   // reuse the shared range type
    private let config: Gemma4TextConfiguration
    private let embedScale: Float
    private let logitSoftcap: Float

    // embed_tokens lives on the head (to embed) AND on a tied tail (to project
    // logits via asLinear), exactly like the Llama shard. Gemma 4 is always tied.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    fileprivate let layers: [Gemma4DecoderLayer]
    @ModuleInfo public var norm: RMSNorm?

    // Per-owned-layer attention type ("full_attention" / "sliding_attention").
    private let ownedLayerTypes: [String]
    private let windowSize: Int

    public init(_ config: Gemma4TextConfiguration, range: LlamaShardRange) {
        precondition(config.vocabSize > 0)
        precondition(range.start >= 0 && range.end <= config.numHiddenLayers && range.start < range.end)
        // Pipeline sharding ships ONLY the hidden state across the cut, so the
        // two cross-layer features that would need extra tensors must be off.
        precondition(config.numKvSharedLayers == 0,
            "Gemma4PipelineShard requires num_kv_shared_layers == 0 (cross-layer KV would cross the pipeline cut)")
        precondition(config.hiddenSizePerLayerInput == 0,
            "Gemma4PipelineShard requires hidden_size_per_layer_input == 0 (per-layer-input embeddings would cross the pipeline cut)")
        self.config = config
        self.range = range
        self.windowSize = config.slidingWindow
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self.logitSoftcap = config.finalLogitSoftcapping

        // Global layer-type pattern, sliced to this rank's interval. Each owned
        // Gemma4DecoderLayer is built with its GLOBAL layer index so it picks the
        // correct attention type, head-dim, and RoPE internally.
        self.ownedLayerTypes = Array(config.layerTypes[range.start ..< range.end])

        // embed_tokens on head, and on a tied tail for the output projection.
        let needsEmbed = range.isHead || (range.isTail && config.tieWordEmbeddings)
        if needsEmbed {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        }
        self.layers = (range.start ..< range.end).map {
            Gemma4DecoderLayer(config, layerIdx: $0)
        }
        if range.isTail {
            self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            // Gemma 4 ties word embeddings (no separate lm_head); the tail
            // projects through embed_tokens.asLinear. If a future untied build
            // appears, this is where an explicit lm_head would be allocated.
        }
    }

    // MARK: - Partial forward primitives

    /// HEAD only: embed token ids into the initial hidden state, applying
    /// Gemma's √hidden_size embedding scale (matches Gemma4TextModelInner).
    public func embed(_ tokens: MLXArray) -> MLXArray {
        guard let embedTokens else {
            fatalError("embed() called on non-head Gemma4 shard")
        }
        return embedTokens(tokens) * embedScale
    }

    /// Run this rank's owned blocks, applying the correct per-layer mask
    /// (full vs sliding window). `cache` is sized to this rank's layer count.
    ///
    /// Gemma 4 has `num_kv_shared_layers == 0` here, so every layer computes its
    /// own K/V (sharedKV: nil) and PLE is off (perLayerInput: nil) — the block's
    /// returned KV pair / position offset are not needed by the pipeline.
    public func runOwnedLayers(_ hidden: MLXArray, cache: [KVCache]? = nil, evalEvery: Int = 4) -> MLXArray {
        var x = hidden
        let caches: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)
        let seqLen = x.dim(1)

        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?
        // First owned index of each type, for the cache used to size the mask.
        let firstFull = ownedLayerTypes.firstIndex(of: "full_attention") ?? 0
        let firstSliding = ownedLayerTypes.firstIndex(of: "sliding_attention") ?? 0

        for (i, layer) in layers.enumerated() {
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if ownedLayerTypes[i] == "full_attention" {
                if fullMask == nil {
                    fullMask = makeAttentionMask(n: seqLen, cache: caches[firstFull], windowSize: nil)
                }
                maskMode = fullMask!
            } else {
                if slidingMask == nil {
                    slidingMask = makeAttentionMask(n: seqLen, cache: caches[firstSliding], windowSize: windowSize)
                }
                maskMode = slidingMask!
            }
            let (out, _, _) = layer(x, mask: maskMode, cache: caches[i])
            x = out
            if evalEvery > 0 && (i + 1) % evalEvery == 0 { eval(x) }
        }
        return x
    }

    /// TAIL only: final norm + tied-embedding logit projection + final-logit
    /// softcap. Mirrors `Gemma4TextModel.applyLMHead`.
    public func projectToLogits(_ hidden: MLXArray, lastPositionOnly: Bool = true) -> MLXArray {
        guard let norm, let embedTokens else {
            fatalError("projectToLogits() called on non-tail Gemma4 shard (missing norm/embed copy)")
        }
        let h = lastPositionOnly && hidden.ndim == 3
            ? hidden[0..., (hidden.dim(1) - 1)..., 0...]
            : hidden
        let logits = embedTokens.asLinear(norm(h))
        // Final-logit softcap: tanh(logits / cap) * cap. cap == 0 disables it.
        guard logitSoftcap > 0 else { return logits }
        return tanh(logits / logitSoftcap) * logitSoftcap
    }

    public var ownedLayerCount: Int { layers.count }

    /// Batched KV caches for this rank's owned layers (continuous-batching path).
    /// Gemma 4 interleaves full- and sliding-attention layers, so each owned
    /// layer gets the matching batched cache type (sliding ⇒ BatchRotatingKVCache
    /// capped at the window). `leftPadding` has one entry per batch row.
    public func makeBatchedCaches(leftPadding: [Int]) -> [any KVCache] {
        ownedLayerTypes.map { type in
            type == "sliding_attention"
                ? BatchRotatingKVCache(maxSize: windowSize, leftPadding: leftPadding)
                : BatchKVCache(leftPadding: leftPadding)
        }
    }
}
