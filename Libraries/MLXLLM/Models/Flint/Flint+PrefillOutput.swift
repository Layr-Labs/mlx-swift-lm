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
        switch requirement {
        case .evaluationOnly:
            // Nothing reads an intermediate chunk's output, so on a qualifying
            // paged cache the final layer only commits its K/V. Every layer's
            // K/V bytes are unchanged; the handle's values are unspecified.
            let hidden =
                commitsFinalPromptKVOnly
                ? model.committingPromptKV(inputs, cache: cache) : model(inputs, cache: cache)
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            let hidden = model(inputs, cache: cache)
            let last = hidden[0..., -1, 0...].asType(.float32)
            return lmHead?(last) ?? model.embedding.asLinear(last)
        }
    }
    // Packed prefill retains the protocol's fail-closed default.
}
