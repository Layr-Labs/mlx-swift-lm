// Copyright © 2026 Eigen Labs / Darkbloom.
//
// One proposal step for several requests' rounds at once, every row with the
// arithmetic of its own one-row step.
//
// A caller drafting a cohort (the cluster's tensor-parallel batch owner) runs
// each round's first step per request, as `draftHidden` does, then may run the
// round's later steps for all of its requests through `draftHiddens`: one
// embedding, one pre-FC norm pair, one FC, one decoder layer and one final
// norm for the rows together. The caller supplies `project`, the product of one
// projection for all rows, and it must give every row the bits that row gets
// alone (a batch-invariant kernel such as `AffineQuantizedVectors`, or one call
// per row). Attention runs per row against that row's own history (its cache,
// rotary offset and mask), and every norm and element-wise step treats each
// row independently, so the step is the rows' own steps side by side: a row's
// result does not depend on how many rows share the call. Nothing here is
// reached through `draftStep`, `draftHidden` or the engine's own drafter, so
// the single-Mac drafting path keeps its arithmetic.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension Qwen35Attention {
    /// `callAsFunction` for one column per row, each row against its own cache
    /// and mask, with every projection through `project`.
    func draftStepRows(
        _ x: MLXArray, caches: [any KVCache],
        masks: [MLXFast.ScaledDotProductAttentionMaskMode],
        project: (Linear, MLXArray) -> MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        precondition(
            L == 1 && caches.count == B && masks.count == B,
            "inline Qwen MTP draft rows take one column per row")

        let qProjOutput = project(qProj, x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        let gate = qSplit[1].reshaped(B, L, -1)
        let queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let keys = kNorm(project(kProj, x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = project(vProj, x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for row in 0 ..< B {
            let range = row ..< (row + 1)
            let cache = caches[row]
            let rowQueries = applyRotaryPosition(rope, to: queries[range], cache: cache)
            let rowKeys = applyRotaryPosition(rope, to: keys[range], cache: cache)
            outputs.append(
                attentionWithCacheUpdate(
                    queries: rowQueries, keys: rowKeys, values: values[range],
                    cache: cache, scale: scale, mask: masks[row]))
        }
        let output = concatenated(outputs, axis: 0)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
        return project(oProj, sigmoidMultiply(output, gate))
    }
}

extension Qwen35MTPDecoderLayer {
    /// Whether `draftStepRows` covers this layer: a dense MLP (a routed-expert
    /// head keeps its own per-row forward).
    var draftsRowsTogether: Bool { mlp is Qwen3NextMLP }

    /// `callAsFunction` for one column per row (`Qwen35Attention.draftStepRows`),
    /// the dense MLP's projections through `project`.
    func draftStepRows(
        _ x: MLXArray, caches: [any KVCache],
        masks: [MLXFast.ScaledDotProductAttentionMaskMode],
        project: (Linear, MLXArray) -> MLXArray
    ) -> MLXArray {
        guard let dense = mlp as? Qwen3NextMLP else {
            preconditionFailure("inline Qwen MTP draft rows require a dense MLP head")
        }
        let r = selfAttn.draftStepRows(
            inputLayerNorm(x), caches: caches, masks: masks, project: project)
        let h = x + r
        let normed = postAttentionLayerNorm(h)
        return h
            + project(
                dense.downProj,
                silu(project(dense.gateProj, normed)) * project(dense.upProj, normed))
    }
}

extension Qwen35MTPModule {
    /// Whether `draftStepRows` covers this head: one decoder layer with a dense MLP.
    var draftsRowsTogether: Bool { layers.count == 1 && layers[0].draftsRowsTogether }

    /// `callAsFunction` for one column per row (`hidden [B, 1, H]`,
    /// `nextTokenIds [B, 1]`), row `b` against `caches[b]`.
    func draftStepRows(
        hidden: MLXArray, nextTokenIds: MLXArray, embedTokens: Embedding,
        caches: [any KVCache], project: (Linear, MLXArray) -> MLXArray
    ) -> MLXArray {
        let rows = hidden.dim(0)
        precondition(
            draftsRowsTogether && hidden.ndim == 3 && hidden.dim(1) == 1
                && nextTokenIds.shape == [rows, 1] && caches.count == rows,
            "inline Qwen MTP draft rows differ from one column per row")
        let e = preFcNormEmbedding(embedTokens(nextTokenIds))
        let h = preFcNormHidden(hidden)
        let fused = project(fc, concatenated([e, h], axis: -1))
        let masks = (0 ..< rows).map { row -> MLXFast.ScaledDotProductAttentionMaskMode in
            let rowCache: (any KVCache)? = caches[row]
            return createAttentionMask(h: fused[row ..< (row + 1)], cache: rowCache)
        }
        return norm(layers[0].draftStepRows(fused, caches: caches, masks: masks, project: project))
    }
}
