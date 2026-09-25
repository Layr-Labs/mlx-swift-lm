// Copyright © 2026 Eigen Labs.
//
// GPTOSSPipelineShard -- a per-rank SLICE of a GPT-OSS model for pipeline-
// parallel inference across a cluster (Darkbloom), a per-rank shard (mirrors the GPT-OSS shard structure).
//
// GPT-OSS is MoE with sliding-window attention, but its layer-list shape matches
// Llama's: embed_tokens + layers[GPTOSSTransformerBlock] + norm + a separate
// lm_head. Pipeline-sharding slices WHOLE blocks, so the per-block MoE
// (experts + router) rides along with its layer automatically — no special MoE
// handling in the shard. The only GPT-OSS-specific piece is the per-layer
// sliding/full attention mask.
//
// IMPORTANT — quantization variant: this shard targets the PRE-QUANTIZED Q8
// build (mlx-community/gpt-oss-20b-MXFP4-Q8), whose weights stay quantized and
// fit a small cluster. The plain MXFP4 build dequantizes its MoE experts to
// bf16 at load (~40 GB in RAM) and is NOT a fit for pipeline-on-2-Macs.
//
// Lives inside MLXLLM because GPTOSSTransformerBlock is internal to the module.
// Additive: the single-node GPTOSSModel path is untouched.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One rank's slice of a GPT-OSS model. Owns only its layers (+ embed on head,
/// norm + lm_head on tail). Exposes the partial-forward primitives the cluster
/// pipeline drives, not a full callAsFunction.
public class GPTOSSPipelineShard: Module {

    public let range: PipelineShardRange   // reuse the shared range type
    private let config: GPTOSSConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    fileprivate let layers: [GPTOSSTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // Per-owned-layer attention type ("full_attention" / "sliding_attention").
    private let ownedLayerTypes: [String]
    private let windowSize: Int

    public init(_ config: GPTOSSConfiguration, range: PipelineShardRange) {
        precondition(config.vocabularySize > 0)
        precondition(range.start >= 0 && range.end <= config.hiddenLayers && range.start < range.end)
        self.config = config
        self.range = range
        self.windowSize = config.slidingWindow

        // Global layer-type pattern (same default as GPTOSSModelInner), sliced
        // to this rank's interval.
        let allTypes = config.layerTypes
            ?? Array(repeating: ["sliding_attention", "full_attention"],
                     count: config.hiddenLayers / 2).flatMap { $0 }
        self.ownedLayerTypes = Array(allTypes[range.start ..< range.end])

        if range.isHead {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        }
        self.layers = (range.start ..< range.end).map { _ in GPTOSSTransformerBlock(config) }
        if range.isTail {
            self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            // GPT-OSS always has a separate lm_head (not tied).
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    // MARK: - Partial forward primitives

    /// HEAD only: embed token ids into the initial hidden state.
    public func embed(_ tokens: MLXArray) -> MLXArray {
        guard let embedTokens else {
            fatalError("embed() called on non-head GPT-OSS shard")
        }
        return embedTokens(tokens)
    }

    /// Run this rank's owned blocks, applying the correct per-layer mask
    /// (full vs sliding window). `cache` is sized to this rank's layer count.
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
            x = layer(x, mask: maskMode, cache: caches[i])
            if evalEvery > 0 && (i + 1) % evalEvery == 0 { eval(x) }
        }
        return x
    }

    /// TAIL only: final norm + lm_head -> vocabulary logits.
    public func projectToLogits(_ hidden: MLXArray, lastPositionOnly: Bool = true) -> MLXArray {
        guard let norm, let lmHead else {
            fatalError("projectToLogits() called on non-tail GPT-OSS shard")
        }
        let h = lastPositionOnly && hidden.ndim == 3
            ? hidden[0..., (hidden.dim(1) - 1)..., 0...]
            : hidden
        return lmHead(norm(h))
    }

    public var ownedLayerCount: Int { layers.count }

    /// Batched KV caches for this rank's owned layers (continuous-batching path).
    /// GPT-OSS interleaves full- and sliding-attention layers, so each owned
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
