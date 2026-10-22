import MLX
import MLXLMCommon

/// Reuses the native CBv2 prompt-output contract also used by GPTOSS and Gemma.
/// Every token traverses the full trunk and writes KV; only unused vocabulary
/// projections are omitted. The ordinary forward remains the full-logits path.
extension FlintModel: CBv2LanguageModelPrefillForwardable {
    public func cbv2Prefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        precondition(inputEmbedding == nil, "Flint supports text input only")
        let hidden = model(inputs, cache: cache)
        switch requirement {
        case .evaluationOnly:
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            let last = hidden[0..., -1, 0...].asType(.float32)
            return lmHead?(last) ?? model.embedding.asLinear(last)
        }
    }
    // Packed prefill retains the protocol's fail-closed default.
}
