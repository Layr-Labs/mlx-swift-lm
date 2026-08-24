// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx DFlash2 at the revision recorded in NOTICE.

import MLX
import MLXNN

public struct DFlash2DraftProposal {
    public let tokenIDs: MLXArray
    public let candidateIDs: MLXArray?
    public let candidateProbabilities: MLXArray?

    public init(
        tokenIDs: MLXArray,
        candidateIDs: MLXArray? = nil,
        candidateProbabilities: MLXArray? = nil
    ) {
        self.tokenIDs = tokenIDs
        self.candidateIDs = candidateIDs
        self.candidateProbabilities = candidateProbabilities
    }
}

final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(configuration: DFlash2Configuration) {
        topK = configuration.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.selectorRank,
            bias: false)
        super.init()
    }

    func select(
        hidden: MLXArray,
        logits: MLXArray,
        anchorIDs: MLXArray,
        temperature: Float,
        captureQ: Bool
    ) -> DFlash2DraftProposal {
        let candidates = argPartition(logits, kth: -topK, axis: -1)[
            0..., 0..., (-topK)...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projectedHidden = hiddenProjection(hidden)
        let successors = successorCodebook(candidates)
        var predecessor = anchorIDs.reshaped([-1])
        var selectedPath = [MLXArray]()
        var probabilityRows = [MLXArray]()
        selectedPath.reserveCapacity(hidden.dim(1))
        probabilityRows.reserveCapacity(hidden.dim(1))

        for position in 0 ..< hidden.dim(1) {
            let predecessorVector = predecessorCodebook(predecessor)
            let edges =
                (predecessorVector[0..., .newAxis, 0...]
                * projectedHidden[0..., position, .newAxis, 0...]
                * successors[0..., position, 0..., 0...]).sum(axis: -1)
            let scores = unary[0..., position, 0...] + edges
            if temperature > 0 || captureQ {
                let divisor = temperature > 0 ? temperature : 1
                probabilityRows.append(
                    softmax(scores.asType(.float32) / divisor, axis: -1))
            }
            let selected =
                temperature > 0
                ? categorical(scores.asType(.float32) / temperature)
                : argMax(scores, axis: -1)
            predecessor = takeAlong(
                candidates[0..., position, 0...],
                selected.expandedDimensions(axis: -1),
                axis: -1
            ).squeezed(axis: -1)
            selectedPath.append(predecessor)
        }

        return DFlash2DraftProposal(
            tokenIDs: stacked(selectedPath, axis: 1),
            candidateIDs: captureQ ? candidates : nil,
            candidateProbabilities: captureQ ? stacked(probabilityRows, axis: 1) : nil)
    }
}
