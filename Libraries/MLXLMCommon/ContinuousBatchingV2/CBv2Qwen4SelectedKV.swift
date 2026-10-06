import MLX

/// Optional native-paging seam. Indices are absolute token positions, packed
/// in the caller's exact attention traversal order; -1 requests a zero row.
/// The returned independent typed buffers live only through this engine step.
/// Implementations must order the read after writes AND before later writes.
public protocol CBv2Qwen4SelectedKVCache: CBv2Qwen4GatheredCache {
    func qwen4CanGatherSelectedKV(keys: MLXArray, values: MLXArray) -> Bool
    func qwen4UpdateAndGatherSelectedKV(
        keys: MLXArray, values: MLXArray, tokenIndices: MLXArray
    ) -> (keys: MLXArray, values: MLXArray)
}

extension PagedLayerCache: CBv2Qwen4SelectedKVCache {}
