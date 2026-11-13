// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

// The target side of block drafting. A drafter conditions on the residual
// stream after `mtpHiddenLayerIDs`, embeds tokens without `embed_norm` and
// reads its proposals through the target's own head.

extension MuseGlimmerTextModel {
    /// Width of the hidden a drafter receives: one residual per captured layer.
    public var mtpHiddenWidth: Int { mtpHiddenLayerIDs.count * configuration.hiddenSize }

    /// Final-normed hidden `[B, L, hidden]` and the residuals after
    /// `mtpHiddenLayerIDs`, concatenated on the last axis in that order.
    func hiddenWithMTPCapture(_ embeddings: MLXArray, cache: [KVCache]?)
        -> (normed: MLXArray, captured: MLXArray)
    {
        precondition(!mtpHiddenLayerIDs.isEmpty, "no drafter hidden layers configured")
        let (hidden, captured) = model.residual(
            embeddings, cache: cache, capture: mtpHiddenLayerIDs)
        return (
            model.norm(hidden), captured.count == 1 ? captured[0] : concatenated(captured, axis: -1)
        )
    }

    /// The plain forward's logits `[B, L, vocab]` with the drafter hidden
    /// `[B, L, mtpHiddenWidth]`. Works on eager and on CBv2 caches.
    public func forwardWithMTPHidden(_ tokens: MLXArray, cache: [KVCache]?)
        -> (logits: MLXArray, hidden: MLXArray)
    {
        let (normed, captured) = hiddenWithMTPCapture(model.inputEmbeddings(tokens), cache: cache)
        return (outputLogits(normed), captured)
    }

    /// A prompt chunk that keeps the drafter hidden of every row and projects
    /// vocabulary logits for the last row only: `[B, 1]` for an intermediate
    /// chunk, `[B, vocab]` at the prompt frontier.
    public func prefillWithMTPHidden(
        _ tokens: MLXArray, cache: [KVCache]?, requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let (normed, captured) = hiddenWithMTPCapture(model.inputEmbeddings(tokens), cache: cache)
        switch requirement {
        case .evaluationOnly: return (normed[0..., -1, 0 ..< 1], captured)
        case .lastPositionLogits: return (outputLogits(normed[0..., -1, 0...]), captured)
        }
    }
}

/// Lets the generic adapter drive stateless drafters and offline harnesses.
/// The capture layers name the last full and the last sliding layer.
extension MuseGlimmerTextModel: CBv2MTPForwardable {
    public var cbv2MTPCaptureLayers: CBv2MTPCaptureLayers? {
        guard
            let full = configuration.layerTypes.lastIndex(
                of: MuseGlimmerTextModelConfiguration.fullAttention),
            let sliding = configuration.layerTypes.lastIndex(
                of: MuseGlimmerTextModelConfiguration.slidingAttention)
        else { return nil }
        return CBv2MTPCaptureLayers(full: full, sliding: sliding)
    }

    public func cbv2ForwardWithHidden(_ tokens: MLXArray, caches: [KVCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
    {
        let result = forwardWithMTPHidden(tokens, cache: caches)
        return (result.logits, result.hidden)
    }
}
