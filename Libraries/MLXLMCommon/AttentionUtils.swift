import Foundation
import MLX

/// Cache-native attention that can reuse an owner's KV without appending it again.
package protocol SharedAttentionKVCache: KVCache {
    func updateForAttention(keys: MLXArray, values: MLXArray)
    func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}

/// An owner's attention presentation, valid until its next cache update.
/// Keep this within one forward pass; serialization state is not an attention view.
package enum AttentionKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int, bits: Int, mode: QuantizationMode
    )
    case native(cache: any SharedAttentionKVCache, sequenceLength: Int)

    package var sequenceLength: Int {
        switch self {
        case .regular(let keys, _): keys.dim(2)
        case .quantized(let keys, _, _, _, _): keys.0.dim(2)
        case .native(_, let sequenceLength): sequenceLength
        }
    }

    package static func update(keys: MLXArray, values: MLXArray, cache: KVCache?) -> Self {
        if let cache = cache as? SharedAttentionKVCache {
            cache.updateForAttention(keys: keys, values: values)
            return .native(cache: cache, sequenceLength: cache.offset)
        } else if let cache = cache as? QuantizedKVCacheProtocol {
            let (keys, values) = cache.updateQuantized(keys: keys, values: values)
            return .quantized(
                keys: keys, values: values,
                groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)
        } else if let cache {
            let (keys, values) = cache.update(keys: keys, values: values)
            return .regular(keys: keys, values: values)
        } else {
            return .regular(keys: keys, values: values)
        }
    }

    package func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        switch self {
        case .regular(let keys, let values):
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        case .quantized(let keys, let values, let groupSize, let bits, let mode):
            return quantizedScaledDotProductAttention(
                queries: queries, quantizedKeys: keys, quantizedValues: values,
                scale: scale, mask: mask, groupSize: groupSize, bits: bits, mode: mode)
        case .native(let cache, let sequenceLength):
            // Native storage must still describe the owner's presentation.
            precondition(cache.offset == sequenceLength)
            return cache.attend(queries: queries, scale: scale, mask: mask)
        }
    }
}

/// Whether attention can be split around a plain `KVCache.update` call.
///
/// Quantized and TurboQuant caches own their complete attention operation, so
/// compiled model segments must leave those cache routes on the general path.
package func usesPlainAttentionCacheRoute(_ cache: KVCache) -> Bool {
    !(cache is QuantizedKVCacheProtocol) && !(cache is TurboQuantKVCache)
}

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Detects cache-native attention implementations
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
///
/// LIMITATION (ContinuousBatchingV2 caches): when `cache` is a
/// `CBv2AttendingLayerCache`, the layer cache owns BOTH the KV update and
/// the attention computation, INCLUDING masking — the `mask` parameter is
/// DISCARDED on that path (v2 derives causal/window masks from per-row
/// absolute positions), and `sinks` are passed as nil. A non-adapted model
/// driven with CBv2 caches therefore silently loses any CUSTOM array mask
/// (e.g. bidirectional/prefix-LM or padding masks) and any attention
/// sinks; sinks-bearing or custom-mask models must call `updateAndAttend`
/// directly instead. Array masks fail HARD here — in release builds too
/// (`preconditionFailure`, not a debug-only assertion): silently swapping
/// the model's required mask for v2's causal/window mask would corrupt
/// output in exactly the builds users run (PR#62 review).
///
/// MULTI-ROW LIMITATION: this compatibility path is B == 1 ONLY. Legacy
/// (non-v2-adapted) models apply scalar RoPE via `KVCache.offset` BEFORE
/// calling this helper; a CBv2 cache's legacy `offset` is the MAX row
/// offset, so at B > 1 every shorter row would be silently mis-rotated.
/// B == 1 stays allowed (the scalar offset is exact for a single row).
/// Generic models must be v2-adapted — capture `positionOffsets` before
/// dispatch and call `updateAndAttend` directly — before they can serve
/// multi-row CBv2 batches. This fails loudly rather than mis-rotating.
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    // ContinuousBatchingV2 hook — see the LIMITATION notes above.
    if let v2 = cache as? CBv2AttendingLayerCache {
        if let violation = cbv2CustomMaskViolation(mask: mask, layerIndex: v2.layerIndex) {
            preconditionFailure(violation)
        }
        if let violation = cbv2LegacyAttentionBatchViolation(
            batch: queries.dim(0), layerIndex: v2.layerIndex)
        {
            preconditionFailure(violation)
        }
        return v2.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
    }
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let turboCache = cache as? TurboQuantKVCache {
        let L = queries.dim(2)
        if L > 1 && !turboCache.isCompressed {
            // Prefill (L>1) on a raw cache: plain update + standard SDPA, // zero overhead; compression is deferred to the first decode step.
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: mask
            )
        }
        // Decode (L=1) or any call once the cache is compressed (speculative
        // verify chunks, multi-turn re-prefill): the compressed path. The raw
        // update() path is invalid after compression, its raw buffers are
        // gone. First decode call triggers compressRawCache().
        return turboCache.compressedAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
    } else if let attentionCache = cache as? KVCacheAttentionProtocol {
        return attentionCache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        // A cache that attends a step itself (paged storage read in place,
        // the same bits as the gathered history's attention), where it can.
        if let attending = cache as? KVCacheStepAttending,
            let output = attending.updateAndAttendStep(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        {
            return output
        }
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

/// A `KVCache` that can append a step's keys and values and attend it in one
/// go, computing exactly what `scaledDotProductAttention` computes over the
/// keys and values `update` would return (a paged cache reading its pages
/// in place instead of gathering them). Returns nil, before changing
/// anything, when it does not take this step: the caller then runs `update`
/// and `scaledDotProductAttention` as for any cache.
public protocol KVCacheStepAttending: KVCache {
    func updateAndAttendStep(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray?
}

/// Custom-mask guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the LIMITATION doc there). `.none`/`.causal` are subsumed by v2's own
/// position-derived masks; any other mode (custom `.array`/`.arrays`) would
/// be silently DISCARDED by `updateAndAttend`, so it must fail in ALL build
/// configurations — a debug-only `assertionFailure` compiles out of release
/// builds and lets the wrong mask ship (PR#62 review). Returns the failure
/// description for an illegal call, or nil when the call is allowed.
/// Internal (not private) so tests can pin the exact condition without
/// tripping the precondition.
func cbv2CustomMaskViolation(
    mask: MLXFast.ScaledDotProductAttentionMaskMode, layerIndex: Int
) -> String? {
    switch mask {
    case .none, .causal:
        return nil
    default:
        return """
            attentionWithCacheUpdate: a custom array mask was passed with a \
            CBv2 layer cache (layer \(layerIndex)). CBv2 caches own their \
            masks and DISCARD this parameter — the model must be v2-adapted \
            (call updateAndAttend with its own semantics) instead.
            """
    }
}

/// Multi-row guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the MULTI-ROW LIMITATION doc there). Returns the failure description for
/// an illegal call, or nil when the call is allowed. Internal (not private)
/// so tests can pin the exact condition without tripping the precondition.
func cbv2LegacyAttentionBatchViolation(batch: Int, layerIndex: Int) -> String? {
    guard batch > 1 else { return nil }
    return """
        attentionWithCacheUpdate: a multi-row batch (B=\(batch)) reached the \
        legacy CBv2 compatibility path (layer \(layerIndex)). Legacy models \
        apply scalar RoPE via `KVCache.offset` — the MAX row offset — so \
        shorter rows would be silently mis-rotated at B > 1. This model must \
        be v2-adapted (read `positionOffsets` before dispatch and call \
        `updateAndAttend` directly) before multi-row CBv2 serving; B == 1 \
        remains supported.
        """
}
