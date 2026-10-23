import MLX
import MLXLMCommon

/// GPT-OSS's model-local entry for a batch whose requests own independent
/// attention histories. It preserves the caller's attention implementation;
/// it does not convert storage, pad histories or choose an engine backend.
/// Ordinary KVCache and CBv2 caches retain their existing forward paths.
public protocol GPTOSSPerRowAttentionCache: KVCache {
    /// Absolute positions of the next tokens, int32 [batch]. The model reads
    /// these before updateAndAttend advances any request. Rows follow input
    /// order, and joining or leaving a row must not change another's position.
    var positionOffsets: MLXArray { get }

    /// The same bounded positions as host integers, for the singleton RoPE
    /// graph. Native stage caches override this to avoid device readback.
    var scalarPositionOffsets: [Int] { get }

    /// Updates each request's KV and attends its own history, including the
    /// supplied learned sinks. All arrays use [batch, heads, tokens, width].
    /// The implementation owns causal/window masking and must keep its native
    /// dtype and arithmetic policy across membership changes. No caller may
    /// release request storage until the submitted forward has completed.
    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray
}

extension GPTOSSPerRowAttentionCache {
    public var scalarPositionOffsets: [Int] { positionOffsets.asArray(Int32.self).map(Int.init) }
}
