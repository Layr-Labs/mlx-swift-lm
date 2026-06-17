// DSAKVCache.swift — sliding-window + compressed-slot KV cache for DSA-style
// sparse attention (DeepSeek-V4). The attention layer consumes this cache type
// directly: raw KV window (K==V stored once) plus a growing buffer of learned
// compressed slots, with the compressor's pending-block state carried here.
//
// Storage follows the KVCacheSimple pattern: preallocated buffers mutated with
// in-place index assignment (MLX donates the underlying buffer, so each token
// is a true in-place write) instead of per-token concat+slice chains — the
// latter cost ~6 fresh allocations per layer per token and made long decodes
// drift from ~7 to ~3 tok/s as the buffers grew.
//
// The raw window is a RING: after warmup, token at absolute position p lives at
// row p % windowSize. SDPA over a set of keys is permutation-invariant (rope is
// baked into the keys), so decode uses the ring directly with NO mask. Prefill
// chunks build an order-agnostic mask from the per-row absolute positions.

import Foundation
import MLX
import MLXNN

/// Whether the DSA window+compressor attention structure is enabled
/// (default on; MLX_DSA=0 falls back to dense full attention — known to
/// degenerate past ~300-500 tokens because the model never trained on it).
public let dsaEnabled: Bool = {
    let v = ProcessInfo.processInfo.environment["MLX_DSA"] ?? ""
    return !(v == "0" || v.lowercased() == "false")
}()

/// Per-layer DSA cache. Attention handles this cache type directly (no
/// generic update()).
public final class DSAKVCache: BaseKVCache {
    public let windowSize: Int
    public let ratio: Int  // 0 = pure sliding window (layers 0, 1)

    /// Slot-buffer growth quantum (slots), KVCacheSimple-style.
    static let slotStep = 256

    // ── Raw window ring: [B, 1, windowSize, headDim], roped keys.
    // Valid rows: min(offset, windowSize). Row of absolute position p is
    //   p              while offset ≤ windowSize (no wrap yet)
    //   p % windowSize afterwards.
    public var windowBuf: MLXArray?
    // Absolute position stored in each ring row (host-side, tiny), for
    // order-agnostic prefill masks.
    public var windowPos: [Int] = []

    // ── Compressed slots: [B, 1, capacity, headDim]; valid prefix slotCount.
    public var slotsBuf: MLXArray?
    public var slotCount: Int = 0

    // ── Compressor pending state (fp32), preallocated [B, ratio, coff*d].
    // Valid prefix pendLen. Scores stored RAW (ape added at consume time).
    public var pendKV: MLXArray?
    public var pendScore: MLXArray?
    public var pendLen: Int = 0
    // Overlap variant: previous complete block's first-half channels
    // (kv raw, score WITH ape), each [B, 1, ratio, d].
    public var prevBlockKV: MLXArray?
    public var prevBlockScore: MLXArray?
    // Absolute position of the first pending token (always ≡ 0 mod ratio).
    public var pendStart: Int = 0

    public init(windowSize: Int, ratio: Int) {
        self.windowSize = windowSize
        self.ratio = ratio
        super.init()
    }

    // ── Window ring operations ──────────────────────────────────────────────

    /// Append a chunk of roped kv ([B, 1, L, d]) to the ring and return the
    /// valid window content to attend over (a view, not a copy, once warm).
    /// Caller must update `offset` afterwards.
    public func appendWindow(_ kv: MLXArray) -> MLXArray {
        let L = kv.dim(2)
        if windowBuf == nil {
            let B = kv.dim(0), d = kv.dim(3)
            windowBuf = MLXArray.zeros([B, 1, windowSize, d], dtype: kv.dtype)
            windowPos = Array(repeating: -1, count: windowSize)
        }
        // Invariant: absolute position p is stored at ring row p % windowSize.
        // For chunks larger than the window only the last windowSize tokens
        // matter; either way every written row keeps the invariant, so the
        // write splits into ≤ 2 contiguous in-place assignments.
        let skip = max(0, L - windowSize)           // tokens that fall straight off
        var written = skip
        while written < L {
            let pos = offset + written
            let row = pos % windowSize
            let span = min(L - written, windowSize - row)
            windowBuf?[0..., 0..., row ..< (row + span), 0...] =
                kv[0..., 0..., written ..< (written + span), 0...]
            for j in 0 ..< span { windowPos[row + j] = pos + j }
            written += span
        }
        let valid = min(offset + L, windowSize)
        return valid < windowSize
            ? windowBuf![0..., 0..., ..<valid, 0...]
            : windowBuf!
    }

    /// Absolute positions of the rows returned by the last `appendWindow`
    /// (first `count` rows of the ring / warmup prefix).
    public func windowPositions(count: Int) -> [Int] {
        Array(windowPos.prefix(count))
    }

    // ── Compressed-slot operations ─────────────────────────────────────────

    /// Append `n` slots ([B, 1, n, d]) in-place, growing capacity in
    /// `slotStep` quanta. Returns nothing; read via `slotsView`.
    public func appendSlots(_ slots: MLXArray) {
        let n = slots.dim(2)
        let B = slots.dim(0), d = slots.dim(3)
        let needed = slotCount + n
        let capacity = slotsBuf?.dim(2) ?? 0
        if needed > capacity {
            let newCap = ((needed + Self.slotStep - 1) / Self.slotStep) * Self.slotStep
            let grown = MLXArray.zeros([B, 1, newCap, d], dtype: slots.dtype)
            if let old = slotsBuf, slotCount > 0 {
                grown[0..., 0..., ..<slotCount, 0...] = old[0..., 0..., ..<slotCount, 0...]
            }
            slotsBuf = grown
        }
        slotsBuf?[0..., 0..., slotCount ..< needed, 0...] = slots
        slotCount = needed
    }

    /// View of the valid slots ([B, 1, slotCount, d]) or nil when empty.
    public var slotsView: MLXArray? {
        guard slotCount > 0, let buf = slotsBuf else { return nil }
        return buf[0..., 0..., ..<slotCount, 0...]
    }

    public override func innerState() -> [MLXArray] {
        [windowBuf, slotsBuf, pendKV, pendScore, prevBlockKV, prevBlockScore].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("DSAKVCache is consumed directly by DeepseekV4Attention, not via update()")
    }
}
