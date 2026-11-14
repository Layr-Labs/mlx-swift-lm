// Copyright © 2026 Eigen Labs.

import MLX

/// The native text tower as the DFlash drafter's target: raw embeddings, the
/// bare head and the tapped residual stream it hands to the MTP seam.
extension MuseGlimmerTextModel: MuseGlimmerDFlashTarget {
    public var dflashTargetIdentity: ObjectIdentifier { ObjectIdentifier(self) }
    public var dflashHiddenSize: Int { configuration.hiddenSize }
    public var dflashVocabularySize: Int { configuration.vocabularySize }
    public var dflashContextWidth: Int { mtpHiddenWidth }
    public var dflashTappedLayerIDs: [Int] { mtpHiddenLayerIDs }
    public func dflashTokenEmbeddings(_ tokens: MLXArray) -> MLXArray {
        rawTokenEmbeddings(tokens)
    }
    public func dflashHeadProduct(_ hidden: MLXArray) -> MLXArray { headProduct(hidden) }
    /// `output_multiplier`, then the tanh softcap: the two steps
    /// `outputLogits` applies after the head product.
    public func dflashReadout(_ headProduct: MLXArray) -> MLXArray {
        var logits = headProduct * configuration.outputMultiplier
        let cap = configuration.finalLogitSoftcapping
        if cap > 0 { logits = tanh(logits / cap) * cap }
        return logits
    }
    public var dflashVerifiesWindowsExactly: Bool { verifiesWindowsExactly }
}
