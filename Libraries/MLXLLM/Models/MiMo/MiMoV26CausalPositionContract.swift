// Copyright © 2026 Eigen Labs.
import MLXLMCommon

extension MiMoV26CBv2Adapter {
    /// Native MiMo media occupies ordinary causal token positions. The real
    /// trunk applies 1D RoPE at each row's cache offset, including chunked
    /// prefill and sliding-window wrap. No Qwen-style position tensor is
    /// synthesized or silently discarded. Supplied positions still pass the
    /// engine's normal shape/positioned-forwarding checks and are rejected by
    /// this non-positioned adapter. Tracked native media remains unqualified.
    public var causalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
}
