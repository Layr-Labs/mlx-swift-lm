//
//  Qwen4ExpCaches.swift
//  mlx-swift-lm
//
//  Caches for the `qwen4_exp` / `qwen4_exp_text` model family
//  (Qwen 3.8 Flash-Next). Derived from the cache classes of the MIT-licensed
//  mlx-lm reference port (ml-explore/mlx-lm PR #1788, `mlx_lm/models/qwen4_exp.py`,
//  head c961f839): `_AttnCache` / `_IndexerCache` and `_LayerCache`.
//
//  These live in MLXLMCommon rather than next to the model because
//  `KVCacheSimple` and `ArraysCache` are `public` (not `open`) and can only be
//  subclassed inside their own module.
//

import Foundation
import MLX
import MLXFast

// MARK: - QSA indexer tape, selection and gathered attention

/// One row's indexer tape: the raw indexer keys, one per processed token,
/// and the pooled blocks derived from them.
///
/// Both live in capacity-doubling buffers, so an append is a slice write and
/// not a copy of the history, and both follow the row's rollback: `truncate`
/// drops the tail, and a pooled block that overlapped the dropped tail is
/// recomputed by the next `blocks` call. Everything else pooled stays.
public final class Qwen4ExpIndexerTape {
    public static let slack = 256

    public init() {}

    public private(set) var length = 0
    private var raw: MLXArray?
    private var pooled: MLXArray?
    private var pooledCount = 0
    /// Shortest length the tape was truncated to since the pooled blocks
    /// were last brought up to date; `Int.max` when they are current.
    private var lowWater = Int.max

    /// Storage the engine loop evaluates each step (graph hygiene).
    public var arrays: [MLXArray] { [raw, pooled].compactMap { $0 } }

    /// The tape `[1, length, headDim]`, or nil before the first append.
    public var view: MLXArray? {
        guard let raw, length > 0 else { return nil }
        return raw[0..., ..<length, 0...]
    }

    /// Replace the tape with `keys` `[1, n, headDim]` (state round-trips);
    /// nothing pooled survives.
    public func load(_ keys: MLXArray?) {
        raw = keys
        length = keys?.dim(1) ?? 0
        pooled = nil
        pooledCount = 0
        lowWater = Int.max
    }

    public func truncate(to count: Int) {
        guard count < length else { return }
        length = count
        lowWater = Swift.min(lowWater, count)
    }

    /// Append `[1, n, headDim]` and return the tape `[1, length, headDim]`.
    public func append(_ keys: MLXArray) -> MLXArray {
        let n = keys.dim(1)
        raw = Self.reserve(raw, rows: length + n, like: keys)
        raw![0..., length ..< (length + n), 0...] = keys
        length += n
        return raw![0..., ..<length, 0...]
    }

    /// Pooled blocks `[1, length / compressRatio, headDim]`. Only the blocks
    /// completed since the last call are pooled: `pool` receives their raw
    /// keys `[1, n, compressRatio, headDim]` and the index of the first one.
    public func blocks(compressRatio: Int, pool: (MLXArray, Int) -> MLXArray) -> MLXArray {
        if lowWater < Int.max {
            pooledCount = Swift.min(pooledCount, lowWater / compressRatio)
            lowWater = Int.max
        }
        let total = length / compressRatio
        if total > pooledCount {
            let start = pooledCount * compressRatio
            let fresh = pool(
                raw![0..., start ..< (total * compressRatio), 0...]
                    .reshaped(1, total - pooledCount, compressRatio, -1),
                pooledCount)
            pooled = Self.reserve(pooled, rows: total, like: fresh)
            pooled![0..., pooledCount ..< total, 0...] = fresh
            pooledCount = total
        }
        guard let pooled, pooledCount > 0 else {
            return MLXArray.zeros([1, 0, raw?.dim(-1) ?? 0], dtype: .float32)
        }
        return pooled[0..., ..<pooledCount, 0...]
    }

    /// `buffer` with room for `rows` entries on axis 1, grown by doubling.
    private static func reserve(_ buffer: MLXArray?, rows: Int, like template: MLXArray)
        -> MLXArray
    {
        guard let buffer else {
            return MLXArray.zeros([1, rows + slack, template.dim(-1)], dtype: template.dtype)
        }
        let capacity = buffer.dim(1)
        guard rows > capacity else { return buffer }
        let grown = Swift.max(capacity * 2, rows + slack)
        return concatenated(
            [buffer, MLXArray.zeros([1, grown - capacity, buffer.dim(-1)], dtype: buffer.dtype)],
            axis: 1)
    }
}

/// What the QSA indexer chose for one forward: the keys attention may read.
public enum Qwen4ExpQSASelection {
    /// The context fits the budget: attend everything.
    case all
    /// Dense keep mask `[1, 1, S, kvLength]`, true == attend, for wide windows.
    case keepMask(MLXArray)
    /// Per-query gathered keys: `indices [S, n]` int32 tape columns and
    /// `valid [S, n]` bool (false == a padding column to ignore).
    case gather(indices: MLXArray, valid: MLXArray)
}


/// Attention of `queries` `[1, heads, S, D]` over, per query, the cached
/// columns a `Qwen4ExpQSASelection.gather` names: `indices` / `valid` are
/// `[S, n]`. Each query reads `n` keys (budget + compressRatio), whatever the
/// context length; a padding column is masked off.
public func qwen4ExpGatherAttention(
    queries: MLXArray, cachedKeys: MLXArray, cachedValues: MLXArray,
    scale: Float, indices: MLXArray, valid: MLXArray
) -> MLXArray {
    let S = queries.dim(2)
    precondition(
        indices.dim(0) == S && valid.dim(0) == S,
        "Qwen4Exp QSA gather \(indices.shape) does not match \(S) queries")
    let dtype = queries.dtype
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(S)
    for s in 0 ..< S {
        let columns = indices[s]
        let k = take(cachedKeys, columns, axis: 2)
        let v = take(cachedValues, columns, axis: 2)
        outputs.append(
            MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., s ..< (s + 1), 0...],
                keys: k.dtype == dtype ? k : k.asType(dtype),
                values: v.dtype == dtype ? v : v.asType(dtype),
                scale: scale, mask: .array(valid[s].reshaped(1, 1, 1, -1)),
                sinks: nil))
    }
    return S == 1 ? outputs[0] : concatenated(outputs, axis: 2)
}

/// KV cache for one Qwen4-Exp full-attention layer.
///
/// A full-attention layer runs a QSA indexer beside the ordinary attention. The
/// indexer keeps its own key tape (`Qwen4ExpIndexerTape`): one raw key vector
/// per token, plus the blocks pooled from them. The tape has to live with the
/// KV tape so that a trim, a copy or a state round-trip keeps the two in step:
/// `trim` moves the KV offset and truncates the tape to it, and every
/// `updateIndexer` re-synchronizes the tape to `offset` before appending.
public final class Qwen4ExpAttentionCache: KVCacheSimple {

    private let tape = Qwen4ExpIndexerTape()

    /// Raw indexer keys, shape `[B, offset, indexerHeadDim]`, or `nil` before
    /// the first update.
    public var indexerKeys: MLXArray? { tape.view }

    public override init() {
        super.init()
    }

    /// Append `keys` to the indexer tape and return the whole tape. Call
    /// BEFORE the key-value update of the same step: the tape is truncated to
    /// the pre-update `offset` first.
    public func updateIndexer(keys: MLXArray) -> MLXArray {
        tape.truncate(to: offset)
        return tape.append(keys)
    }

    /// Pooled indexer blocks `[1, offset / compressRatio, headDim]`, pooling
    /// only the blocks completed since the last call. Call AFTER
    /// `updateIndexer` in the same step.
    public func indexerBlocks(
        compressRatio: Int, pool: (_ raw: MLXArray, _ firstBlock: Int) -> MLXArray
    ) -> MLXArray {
        tape.blocks(compressRatio: compressRatio, pool: pool)
    }

    /// Append this step's K/V and attend over the indexer's selection.
    /// `mask` is the causal/window mask the caller would pass to plain
    /// `attentionWithCacheUpdate`; it applies to `.all` and `.keepMask`.
    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, selection: Qwen4ExpQSASelection
    ) -> MLXArray {
        let (cachedKeys, cachedValues) = update(keys: keys, values: values)
        switch selection {
        case .all:
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                mask: mask)
        case .keepMask(let keep):
            let S = queries.dim(2)
            let kvLength = cachedKeys.dim(2)
            let composed: MLXArray
            switch mask {
            case .none:
                composed = keep
            case .causal:
                let rinds = MLXArray(Int32(0) ..< Int32(kvLength))
                let linds = MLXArray(Int32(kvLength - S) ..< Int32(kvLength))[0..., .newAxis]
                composed = (linds .>= rinds) & keep
            case .array(let m) where m.dtype == .bool:
                composed = m & keep
            default:
                preconditionFailure("Qwen4ExpAttentionCache: cannot combine the keep mask with \(mask)")
            }
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                mask: .array(composed))
        case .gather(let indices, let valid):
            return qwen4ExpGatherAttention(
                queries: queries, cachedKeys: cachedKeys, cachedValues: cachedValues,
                scale: scale, indices: indices, valid: valid)
        }
    }

    public override func innerState() -> [MLXArray] {
        super.innerState() + tape.arrays
    }

    /// The indexer tape is serialized as the last entry. An empty array stands
    /// for "no tape yet", because `nil` is not serializable.
    public override var state: [MLXArray] {
        get {
            let kv = super.state
            let tape = indexerKeys ?? MLXArray([Float]())
            return kv.isEmpty ? [tape] : kv + [tape]
        }
        set {
            guard let keys = newValue.last else {
                fatalError("Qwen4ExpAttentionCache state must carry the indexer tape")
            }
            tape.load(keys.size > 0 ? keys : nil)
            let kv = Array(newValue.dropLast())
            if !kv.isEmpty {
                super.state = kv
            }
        }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = super.trim(n)
        tape.truncate(to: offset)
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = Qwen4ExpAttentionCache()
        new.step = self.step
        let s = self.state
        if s.count > 1 {
            new.state = s.map { $0[.ellipsis] }
        } else if let indexerKeys {
            new.tape.load(indexerKeys[.ellipsis])
        }
        return new
    }
}

/// Cache for one Qwen4-Exp linear-attention (gated deltanet) layer.
///
/// Four slots, because a linear layer can also carry the model's single PLE
/// layer:
///
/// - 0: gated deltanet short-convolution state
/// - 1: gated deltanet recurrent (SSM) state
/// - 2: PLE short-convolution state
/// - 3: the last `ngram_size - 1` token ids, the n-gram hash history
///
/// Slots 2 and 3 stay `nil` on every layer except the PLE layer.
public final class Qwen4ExpLayerCache: ArraysCache {

    public static let deltaConvSlot = 0
    public static let deltaStateSlot = 1
    public static let pleConvSlot = 2
    public static let ngramHistorySlot = 3

    public init(leftPadding: [Int]? = nil) {
        super.init(size: 4, leftPadding: leftPadding)
    }

    public override func copy() -> any KVCache {
        let new = Qwen4ExpLayerCache()
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        return new
    }

    public override func extract(_ idx: Int) -> ArraysCache {
        let extracted = Qwen4ExpLayerCache()
        extracted.state = state.map { $0[idx ..< (idx + 1)] }
        return extracted
    }
}

// MARK: - Speculative rollback

/// One layer's cache state at a draft boundary.
///
/// WHY A SNAPSHOT AND NOT A TRIM. A speculative round feeds the target tokens
/// it may have to take back. On the 12 full-attention layers that is a trim:
/// the key-value tape is append-only and `offset` says how much of it is real.
/// On the 36 gated-deltanet layers it is NOT. Their state is a RECURRENCE --
/// each forward overwrites the previous state -- so there is no row to drop and
/// no offset to move back. The state after the accepted prefix simply is not in
/// the cache any more once the rejected tokens have been folded in.
///
/// So the stack is snapshotted BEFORE the verify forward, and a rollback
/// restores that snapshot and replays the accepted tokens. The replay is one
/// forward of the accepted width, which is at most the draft depth, and it is
/// the only way to reach the recurrent state of a prefix that was overwritten.
public enum Qwen4ExpCacheSnapshot {
    /// A full-attention layer: how much of the tape was real.
    case attention(offset: Int)
    /// A linear-attention layer: every slot, plus the position count.
    case layer(slots: [MLXArray?], offset: Int)
}

extension Qwen4ExpAttentionCache {
    public func snapshot() -> Qwen4ExpCacheSnapshot { .attention(offset: offset) }

    /// Restore to `offset`. The key-value buffers keep their bytes past the
    /// offset and the next update overwrites them in place, so only the offset
    /// and the EXACT indexer tape have to move.
    public func restore(toOffset target: Int) {
        precondition(
            target >= 0 && target <= offset,
            "Qwen4ExpAttentionCache: cannot restore forward, from \(offset) to \(target)")
        offset = target
        tape.truncate(to: target)
    }
}

extension Qwen4ExpLayerCache {
    public func snapshot() -> Qwen4ExpCacheSnapshot {
        // The slot arrays are REPLACED, never written in place, by every writer
        // in this model (the deltanet's conv and state slots, the per-layer
        // embedding's conv slot, and the n-gram history), so holding the
        // references is a snapshot.
        .layer(slots: (0 ..< 4).map { self[$0] }, offset: offset)
    }

    public func restore(slots: [MLXArray?], offset target: Int) {
        precondition(slots.count == 4, "Qwen4ExpLayerCache has four slots")
        for index in 0 ..< 4 { self[index] = slots[index] }
        offset = target
    }
}

/// Snapshot a whole hybrid stack at a draft boundary.
public func qwen4ExpCheckpointCaches(_ caches: [KVCache]) -> [Qwen4ExpCacheSnapshot] {
    caches.map { cache in
        if let attention = cache as? Qwen4ExpAttentionCache {
            return attention.snapshot()
        }
        if let layer = cache as? Qwen4ExpLayerCache {
            return layer.snapshot()
        }
        preconditionFailure(
            "Qwen4Exp speculative rollback needs a Qwen4Exp cache stack, got "
                + "\(type(of: cache))")
    }
}

/// Restore a whole hybrid stack to a snapshot.
///
/// This returns the caches to the state they were in when the snapshot was
/// taken. It does NOT by itself put them at the accepted prefix: the caller
/// replays the accepted tokens after restoring. Splitting it that way keeps
/// the one thing that cannot be derived -- the pre-forward state -- separate
/// from the thing that can.
public func qwen4ExpRestoreCaches(
    _ caches: [KVCache], to snapshots: [Qwen4ExpCacheSnapshot]
) {
    precondition(
        caches.count == snapshots.count,
        "Qwen4Exp rollback: \(caches.count) caches against \(snapshots.count) snapshots")
    for (cache, snapshot) in zip(caches, snapshots) {
        switch snapshot {
        case .attention(let offset):
            guard let attention = cache as? Qwen4ExpAttentionCache else {
                preconditionFailure(
                    "Qwen4Exp rollback: an attention snapshot met \(type(of: cache))")
            }
            attention.restore(toOffset: offset)
        case .layer(let slots, let offset):
            guard let layer = cache as? Qwen4ExpLayerCache else {
                preconditionFailure(
                    "Qwen4Exp rollback: a layer snapshot met \(type(of: cache))")
            }
            layer.restore(slots: slots, offset: offset)
        }
    }
}
