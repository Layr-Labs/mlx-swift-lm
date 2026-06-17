// SharedKVCache.swift — K==V cache deduplication for MLA-style attention.
//
// DeepSeek-V4's MLA attention passes the SAME tensor as keys and values
// (k = v = kv_norm(wkv(x)) with rope on the tail dims). A standard KV cache
// stores two identical copies — doubling KV memory for no information.
//
// `SharedKVCache` wraps any inner `KVCache` and feeds it a ZERO-WIDTH values
// tensor (shape [..., 0]). All cache bookkeeping (concat growth, rotation,
// trimming, masks) operates on shapes/offsets and treats the degenerate
// values buffer identically, so behavior is preserved while values storage
// costs ~nothing. `update` returns the cached keys for BOTH outputs.
//
// On a 4K context DSV4 run this saves ~180 MB; at 32K it saves ~1.4 GB —
// headroom that matters on 24–48 GB machines.
//
// Escape hatch: MLX_SHARED_KV=0 disables (models then use their default caches).

import Foundation
import MLX
import MLXNN

/// Whether K==V cache sharing is enabled (default on; MLX_SHARED_KV=0 disables).
public let sharedKVEnabled: Bool = {
    let v = ProcessInfo.processInfo.environment["MLX_SHARED_KV"] ?? ""
    return !(v == "0" || v.lowercased() == "false")
}()

/// KV cache for attention layers where keys == values by construction.
/// Stores one copy; returns it for both. Wraps any inner cache implementation
/// (KVCacheSimple, RotatingKVCache, …) so windowing semantics are inherited.
public final class SharedKVCache: KVCache, Evaluatable {
    // var (not let): KVCache is not class-constrained, so mutating members of
    // the existential require settability even though all impls are classes.
    public private(set) var inner: KVCache

    public init(wrapping inner: KVCache) {
        self.inner = inner
    }

    public var offset: Int { inner.offset }
    public var maxSize: Int? { inner.maxSize }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        // values is identical to keys in K==V attention — store a zero-width
        // slice instead so the inner cache's values buffer stays empty.
        let zeroWidthValues = values[.ellipsis, 0 ..< 0]
        let (cachedKeys, _) = inner.update(keys: keys, values: zeroWidthValues)
        return (cachedKeys, cachedKeys)
    }

    public func innerState() -> [MLXArray] { inner.innerState() }

    public var state: [MLXArray] {
        get { inner.state }
        set { inner.state = newValue }
    }

    public var metaState: [String] {
        get { inner.metaState }
        set { inner.metaState = newValue }
    }

    public var isTrimmable: Bool { inner.isTrimmable }

    @discardableResult
    public func trim(_ n: Int) -> Int { inner.trim(n) }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        inner.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        SharedKVCache(wrapping: inner.copy())
    }
}
