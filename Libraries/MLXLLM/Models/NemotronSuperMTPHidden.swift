import MLX
import MLXLMCommon

extension NemotronHModel {
    /// Explicit Super onboarding seam, not a type-wide MTP capability. The
    /// caller owns the exact loaded model and request transactions. Hidden is
    /// post-norm_f, matching the pinned NVIDIA MTP conditioning boundary.
    public func superMTPTextForward(_ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) throws -> (logits: MLXArray, lastHidden: MLXArray) {
        try validateSuperMTPTextLayout(tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState)
        return (logits(hidden), hidden)
    }

    public enum SuperMTPTextError: Error { case target, layout, positions, media }

    private func validateSuperMTPTextLayout(_ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?) throws {
        guard type(of: self) == NemotronHModel.self else { throw SuperMTPTextError.target }
        guard positionIds == nil else { throw SuperMTPTextError.positions }
        guard tokens.ndim == 2, tokens.dim(0) == 1, tokens.dim(1) > 0,
              [.int32, .uint32].contains(tokens.dtype), recurrentState.count == 1,
              caches.count == cbv2LayerKinds.count,
              caches.allSatisfy({ $0 is any CBv2AttendingLayerCache }) else { throw SuperMTPTextError.layout }
    }

    /// Preserve every trusted text hidden row but only score the final position
    /// during prefill. No image-embedding argument is accepted by this seam.
    public func superMTPTextPrefill(_ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) throws -> (logits: MLXArray, lastHidden: MLXArray) {
        try validateSuperMTPTextLayout(tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState)
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        switch requirement {
        case .evaluationOnly: return (last[0..., 0..., 0..<1], hidden)
        case .lastPositionLogits: return (logits(last), hidden)
        }
    }
}
