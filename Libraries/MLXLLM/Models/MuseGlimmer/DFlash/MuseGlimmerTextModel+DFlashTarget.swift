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
    public var dflashVerifiesWindowsExactly: Bool { verifiesWindowsExactly }
}
