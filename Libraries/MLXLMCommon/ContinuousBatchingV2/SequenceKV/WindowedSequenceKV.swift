// WindowedSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for SLIDING-WINDOW attention.
//
// The window is enforced by STORAGE EVICTION keyed to absolute positions —
// never by masks over shared buffers (report 10 §4 invariant 6). Storage is
// a ring of exactly `window` slots; the token at absolute position `p` lives
// in physical slot `p % window`, so eviction is simply "newer tokens
// overwrite slots window positions behind them". The RECENT end is always
// kept. `absoluteOffset` keeps counting past the window.

import Foundation
import MLX

/// `CBv2SequenceKV` for sliding-window attention.
///
/// ## Temporal-order guarantee
/// Every array this class returns (from `update` and `snapshot`) is in
/// temporal order (oldest → newest). Unwinding the ring costs at most one
/// concat of two slices; in the common pre-wrap decode case it is a single
/// zero-copy slice.
///
/// ## Multi-token updates (prefill chunks)
/// For `n > 1` the returned views are `retainedHistory (≤ window-1 entries)
/// ++ the n new tokens` — i.e. up to `window - 1 + n` entries, which can
/// exceed `retainedCount`. This is required for correctness: the FIRST token
/// of the chunk must attend to the `window - 1` tokens before it, which the
/// ring evicts as the chunk is written. The attention layer applies a
/// causal∧window mask whenever the returned length exceeds `window`
/// (a pure function of lengths — see `CBv2AttentionV1.maskMode`).
///
/// ## Rollback and un-wrapping
/// `rollback(n)` moves `absoluteOffset` back by `n`. Before the ring wraps
/// this recovers the previous state exactly. After wrapping, the speculative
/// tokens' writes have already destroyed the `n` OLDEST in-window entries
/// (their slots alias positions exactly `window` behind), so the retained
/// count shrinks to `window - n` until fresh tokens refill the window. This
/// is tracked by `oldestValidPosition`, which is monotonically non-decreasing
/// — destroyed history can never be re-exposed, and the un-confirmed tail is
/// structurally unreachable (all views are keyed to
/// `[oldestValidPosition, absoluteOffset)`).
///
/// ## Staged speculative writes (MTP verify rounds)
/// The post-wrap window shrink above is CORRECT but numerically divergent
/// from a non-speculative run, which breaks the MTP acceptance gate (MTP-on
/// must be token-exact vs MTP-off for greedy). `beginSpeculativeWrite()`
/// therefore arms a STAGED mode for the next `update()`: the update returns
/// exactly the views a plain update would return and advances
/// `absoluteOffset`, but the destructive ring writes (and the
/// `oldestValidPosition` advance) are deferred to
/// `commitSpeculativeWrite()`. A `rollback(m)` in between is a pure counter
/// move — nothing was destroyed — so after commit the state is value-exactly
/// what a plain `update()` of only the confirmed tokens would have produced.
public final class CBv2WindowedSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {

    /// Window size in tokens == number of physical ring slots.
    public let window: Int

    /// Absolute position of the next token to be written.
    public private(set) var absoluteOffset: Int

    /// Absolute position of the oldest entry that is still physically valid.
    /// Monotonically non-decreasing (see rollback discussion above).
    private var oldestValidPosition: Int

    public var retainedCount: Int { absoluteOffset - oldestValidPosition }

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?

    /// Step-scoped PRE-EVICTION views captured by the most recent MULTI-token
    /// `update()` (`retainedHistory ++ chunk`, up to `window - 1 + n` entries).
    /// KV-borrowing layers (Gemma-4 cross-layer sharing) attend these instead
    /// of the post-eviction ring, so a chunk's earliest queries still see
    /// their full window — the ring writes have already destroyed those
    /// entries (slot aliasing at distance `window`). nil after a decode
    /// update, rollback, or before any update. Retaining these views keeps
    /// the pre-write buffer alive only until the next `update()` replaces
    /// them (bounded: one extra window-sized buffer between a chunk update
    /// and the following update).
    private var borrowableChunkViews: (keys: MLXArray, values: MLXArray)?

    /// Armed by `beginSpeculativeWrite()`; consumed (disarmed) by the next
    /// `update()`, which stages instead of writing the ring.
    private var speculativeWriteArmed = false

    /// The staged speculative update: the K/V tensors the armed `update()`
    /// received, plus the absolute position it started at. The ring writes
    /// for the still-confirmed range `[basePosition, absoluteOffset)` happen
    /// at `commitSpeculativeWrite()`. At most one staged update per row.
    private var staged: (keys: MLXArray, values: MLXArray, basePosition: Int)?

    /// - Parameters:
    ///   - window: sliding window in tokens (> 0).
    ///   - initialOffset: absolute position this sequence starts at. Non-zero
    ///     when a prefix-cache hit adopts full-attention KV at offset `m` and
    ///     the scheduler recomputes the trailing `window` tokens for windowed
    ///     layers: those rows start at `max(0, m - window)` so that absolute
    ///     RoPE positions line up across layers.
    public init(window: Int, kvHeads: Int, headDim: Int, initialOffset: Int = 0) {
        precondition(window > 0, "CBv2WindowedSequenceKV: window must be > 0")
        precondition(initialOffset >= 0, "CBv2WindowedSequenceKV: negative initialOffset")
        self.window = window
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.absoluteOffset = initialOffset
        self.oldestValidPosition = initialOffset
    }

    public var byteCount: Int {
        // Staged tensors are physically held until commit (bounded: one
        // 1+k-token chunk per in-flight MTP round).
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
            + (staged.map { $0.keys.nbytes + $0.values.nbytes } ?? 0)
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2WindowedSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2WindowedSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2WindowedSequenceKV: keys/values token count mismatch")
        precondition(n > 0, "CBv2WindowedSequenceKV: empty update")

        if speculativeWriteArmed {
            return stageSpeculativeUpdate(newKeys: newKeys, newValues: newValues, count: n)
        }
        precondition(
            staged == nil,
            "CBv2WindowedSequenceKV: plain update with a staged speculative write pending — commit first"
        )

        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)

        if n == 1 {
            // Decode fast path: one modular slot write, then the retained
            // window in temporal order (1 slice pre-wrap, 2-slice concat after).
            borrowableChunkViews = nil  // ring == window: snapshot is exact
            writeRing(keys!, tokens: newKeys, firstPosition: absoluteOffset)
            writeRing(values!, tokens: newValues, firstPosition: absoluteOffset)
            absoluteOffset += 1
            oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
            return (
                temporalOrder(keys!, from: oldestValidPosition, to: absoluteOffset),
                temporalOrder(values!, from: oldestValidPosition, to: absoluteOffset)
            )
        }

        // Prefill chunk: capture the retained history VIEWS before the ring
        // writes evict it (the views reference the pre-write buffer contents;
        // slice-update produces a new buffer, so they stay valid).
        let historyCount = min(retainedCount, window - 1)
        let historyFrom = absoluteOffset - historyCount
        var kParts = ringSlices(keys!, from: historyFrom, to: absoluteOffset)
        var vParts = ringSlices(values!, from: historyFrom, to: absoluteOffset)
        kParts.append(newKeys)
        vParts.append(newValues)
        let returnedKeys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let returnedValues = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)

        // Write the new tokens into the ring. Only the last `window` matter
        // when the chunk itself exceeds the window.
        let writeCount = min(n, window)
        let firstWritten = absoluteOffset + n - writeCount
        let kTail = writeCount == n ? newKeys : newKeys[.ellipsis, (n - writeCount)..., 0...]
        let vTail = writeCount == n ? newValues : newValues[.ellipsis, (n - writeCount)..., 0...]
        writeRing(keys!, tokens: kTail, firstPosition: firstWritten)
        writeRing(values!, tokens: vTail, firstPosition: firstWritten)

        absoluteOffset += n
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)

        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    // MARK: - Speculative (MTP) staging

    /// Staging always supported: the ring defers its destructive writes to
    /// `commitSpeculativeWrite()`, so speculative rollback is value-exact.
    public var supportsSpeculativeWrites: Bool { true }

    public func beginSpeculativeWrite() {
        precondition(
            !speculativeWriteArmed,
            "CBv2WindowedSequenceKV: beginSpeculativeWrite while already armed")
        precondition(
            staged == nil,
            "CBv2WindowedSequenceKV: beginSpeculativeWrite with a staged update pending — commit first"
        )
        speculativeWriteArmed = true
    }

    /// The armed `update()`: return EXACTLY the views a plain update would
    /// return and advance `absoluteOffset`, but touch NEITHER the ring
    /// buffers NOR `oldestValidPosition` — the destructive writes are
    /// deferred to `commitSpeculativeWrite()` so a `rollback` in between is
    /// a pure counter move.
    ///
    /// n == 1 equivalence with the plain decode return: plain writes the
    /// token then returns the ring `[max(oldest, offset+1-window),
    /// offset+1)`. `history ++ token` with `historyCount = min(retained,
    /// window - 1)` yields the same entries — a below-full ring keeps all
    /// history, and a full ring drops exactly the one entry (position
    /// `offset - window`) the plain write would have destroyed. Pinned by
    /// `CBv2MTPKVStagingTests`.
    private func stageSpeculativeUpdate(
        newKeys: MLXArray, newValues: MLXArray, count n: Int
    ) -> (MLXArray, MLXArray) {
        speculativeWriteArmed = false
        let historyCount = min(retainedCount, window - 1)
        let historyFrom = absoluteOffset - historyCount
        var kParts: [MLXArray] = []
        var vParts: [MLXArray] = []
        if historyCount > 0 {
            kParts = ringSlices(keys!, from: historyFrom, to: absoluteOffset)
            vParts = ringSlices(values!, from: historyFrom, to: absoluteOffset)
        }
        kParts.append(newKeys)
        vParts.append(newValues)
        let returnedKeys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let returnedValues = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)

        staged = (newKeys, newValues, absoluteOffset)
        absoluteOffset += n
        // KV-borrowing layers attend the SAME views this step (the ring does
        // not hold the staged tokens yet) — set for n == 1 too, unlike the
        // plain decode path where the post-write ring is already exact.
        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    public func commitSpeculativeWrite() {
        speculativeWriteArmed = false
        guard let staged else { return }
        self.staged = nil
        // Confirmed range after finalize-time rollback: [basePosition,
        // absoluteOffset). Write it into the ring exactly as the plain
        // multi-token path would (only the last `window` matter when the
        // confirmed span exceeds the window); a fully rolled-back
        // (cancelled) row writes nothing.
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            allocateIfNeeded(keyTemplate: staged.keys, valueTemplate: staged.values)
            let writeCount = min(confirmed, window)
            let skip = confirmed - writeCount
            writeRing(
                keys!, tokens: staged.keys[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount)
            writeRing(
                values!, tokens: staged.values[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount)
        }
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
        // Step-scoped views die at finalize (the plain paths replace them on
        // the next update; nothing borrows between commit and that update).
        borrowableChunkViews = nil
    }

    /// Non-nil when compiled decode would observe staged-but-uncommitted
    /// speculative state — compiled decode never runs during an MTP round.
    /// Testable twin of the preconditions in `compiledStorage` /
    /// `noteCompiledAdvance`.
    func cbv2SpeculativePendingViolation() -> String? {
        guard speculativeWriteArmed || staged != nil else { return nil }
        return "CBv2WindowedSequenceKV: compiled decode with a speculative write pending "
            + "(armed=\(speculativeWriteArmed), staged=\(staged != nil)) — "
            + "MTP rounds and compiled decode are mutually exclusive"
    }

    /// Views a KV-BORROWING layer must attend against in the CURRENT step:
    /// exactly what the most recent `update()` returned. After a multi-token
    /// (prefill-chunk) update this is the PRE-eviction history + chunk — the
    /// borrowing layer's chunk queries need the same `window - 1 + n` entries
    /// the source layer attended, and the ring has already evicted the old
    /// ones. After a decode update (or a rollback) it is the retained ring,
    /// identical to `snapshot()`. Step-scoped: valid between the source
    /// layer's `update()` and the next mutation; never retain across steps.
    public func borrowableViews() -> (keys: MLXArray, values: MLXArray) {
        if let views = borrowableChunkViews { return views }
        let snap = snapshot()
        return (snap.keys, snap.values)
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        if let staged {
            return stagedSnapshot(staged)
        }
        guard let keys, let values, retainedCount > 0 else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            temporalOrder(keys, from: oldestValidPosition, to: absoluteOffset),
            temporalOrder(values, from: oldestValidPosition, to: absoluteOffset),
            absoluteOffset
        )
    }

    /// Value-exact snapshot while a speculative update is staged: the ring
    /// holds `[oldestValidPosition, basePosition)` and the staged tensors
    /// hold the confirmed tail `[basePosition, absoluteOffset)` (rollback
    /// may have shrunk it). Transiently up to `window + n` entries — same
    /// exposure as a plain chunk update's pre-eviction views. Never on an
    /// engine path (windowed rows are not donated), but keeps
    /// `borrowableViews()`'s fallback correct after a mid-staged rollback.
    private func stagedSnapshot(
        _ staged: (keys: MLXArray, values: MLXArray, basePosition: Int)
    ) -> (keys: MLXArray, values: MLXArray, offset: Int) {
        var kParts: [MLXArray] = []
        var vParts: [MLXArray] = []
        if let keys, let values, staged.basePosition > oldestValidPosition {
            kParts = ringSlices(keys, from: oldestValidPosition, to: staged.basePosition)
            vParts = ringSlices(values, from: oldestValidPosition, to: staged.basePosition)
        }
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            kParts.append(staged.keys[.ellipsis, ..<confirmed, 0...])
            vParts.append(staged.values[.ellipsis, ..<confirmed, 0...])
        }
        guard !kParts.isEmpty else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2),
            vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2),
            absoluteOffset
        )
    }

    /// Advance the position counter WITHOUT writing storage (contract
    /// `CBv2SequenceKV.fastForward(to:)`). Only valid on a FRESH state, used
    /// during prefix-cache adoption so the engine's trailing-window replay
    /// lands at true absolute positions.
    public func fastForward(to offset: Int) {
        precondition(
            keys == nil && absoluteOffset == oldestValidPosition,
            "CBv2WindowedSequenceKV.fastForward requires a fresh state")
        // A fully-rolled-back staged row can look "fresh" (offset back at
        // oldestValidPosition, ring never allocated) while a commit is
        // still owed — exclude it explicitly.
        precondition(
            !speculativeWriteArmed && staged == nil,
            "CBv2WindowedSequenceKV.fastForward with a speculative write pending")
        precondition(offset >= absoluteOffset, "fastForward cannot move backwards")
        absoluteOffset = offset
        oldestValidPosition = offset
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2WindowedSequenceKV.rollback: negative n")
        if let staged {
            // Pure counter move: the staged tokens were never written to
            // the ring, so nothing was destroyed and the retained window
            // does not shrink. The engine only ever rolls back tokens from
            // the staged round (a cancelled row may roll back ALL of them).
            precondition(
                n <= absoluteOffset - staged.basePosition,
                "CBv2WindowedSequenceKV.rollback(\(n)) exceeds staged range "
                    + "\(absoluteOffset - staged.basePosition)")
            absoluteOffset -= n
            borrowableChunkViews = nil
            return
        }
        precondition(
            n <= retainedCount,
            "CBv2WindowedSequenceKV.rollback(\(n)) exceeds retained \(retainedCount)")
        // oldestValidPosition deliberately does NOT decrease: if the ring had
        // wrapped, the rolled-back tokens' writes destroyed the oldest `n`
        // in-window entries (slot aliasing at distance `window`), so the
        // window shrinks until refilled. Pre-wrap, oldestValidPosition is
        // still the initial offset and the rollback is a full recovery.
        absoluteOffset -= n
        // Any captured pre-eviction chunk views now cover rolled-back
        // positions — invalidate so borrowing falls back to the ring.
        borrowableChunkViews = nil
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    // MARK: - Compiled-decode bridge (CBv2CompiledDecode)

    /// The ring buffers for the compiled [B, 1] decode step. The ring is
    /// already fixed-shape (`[1, kvHeads, window, headDim]`, slot = absolute
    /// position mod window), so the compiled path shares this storage
    /// directly — the SAME `MLXArray` objects, mutated via `_updateInternal`
    /// by the compiled graph and by slice-assignment on the eager path.
    /// Allocates zeros when the row has never been written (single-token
    /// prompt joins decode straight away). K and V dtypes are checked
    /// INDEPENDENTLY against their traced dtypes (PR#62 review).
    ///
    /// Bind-time only (membership changes) — never on the per-step path.
    func compiledStorage(
        keysDType: DType, valuesDType: DType
    ) -> (keys: MLXArray, values: MLXArray)? {
        if let violation = cbv2SpeculativePendingViolation() {
            preconditionFailure(violation)
        }
        if keys == nil {
            keys = MLXArray.zeros([1, kvHeads, window, headDim], dtype: keysDType)
            values = MLXArray.zeros([1, kvHeads, window, headDim], dtype: valuesDType)
        }
        guard let keys, let values, keys.dtype == keysDType, values.dtype == valuesDType else {
            return nil
        }
        return (keys, values)
    }

    /// Oldest absolute position still physically valid — the compiled mask's
    /// static per-session lower bound (`validFrom = max(entry, offset+1-w)`).
    var compiledOldestValidPosition: Int { oldestValidPosition }

    /// Host-side bookkeeping for one compiled decode step: the compiled
    /// graph wrote one token at slot `absoluteOffset % window` and advanced
    /// the device-side offset state. Mirrors the eager decode update's
    /// counter math exactly (append then evict to `window`, RECENT end
    /// kept). Any captured pre-eviction chunk views are step-scoped and are
    /// invalidated, same as an eager decode write.
    func noteCompiledAdvance() {
        if let violation = cbv2SpeculativePendingViolation() {
            preconditionFailure(violation)
        }
        borrowableChunkViews = nil
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    // MARK: - Ring geometry

    /// Views covering absolute positions `[from, to)` in temporal order:
    /// one slice when the modular range does not cross the wrap point,
    /// two slices when it does. `to - from` must be ≤ `window`.
    private func ringSlices(_ array: MLXArray, from: Int, to: Int) -> [MLXArray] {
        guard to > from else { return [] }
        let count = to - from
        precondition(count <= window, "ring range exceeds window")
        let start = from % window
        if start + count <= window {
            return [array[.ellipsis, start ..< (start + count), 0...]]
        }
        return [
            array[.ellipsis, start ..< window, 0...],
            array[.ellipsis, 0 ..< (start + count - window), 0...],
        ]
    }

    private func temporalOrder(_ array: MLXArray, from: Int, to: Int) -> MLXArray {
        let slices = ringSlices(array, from: from, to: to)
        switch slices.count {
        case 0:
            return array[.ellipsis, 0 ..< 0, 0...]
        case 1:
            return slices[0]
        default:
            return concatenated(slices, axis: 2)
        }
    }

    /// Write `tokens` (≤ window of them) into their modular slots, splitting
    /// at the wrap point when needed (at most two slice assignments).
    private func writeRing(_ buffer: MLXArray, tokens: MLXArray, firstPosition: Int) {
        let n = tokens.dim(2)
        precondition(n <= window, "writeRing: more tokens than slots")
        let start = firstPosition % window
        if start + n <= window {
            buffer[.ellipsis, start ..< (start + n), 0...] = tokens
        } else {
            let first = window - start
            buffer[.ellipsis, start ..< window, 0...] = tokens[.ellipsis, ..<first, 0...]
            buffer[.ellipsis, 0 ..< (n - first), 0...] = tokens[.ellipsis, first..., 0...]
        }
    }

    private func allocateIfNeeded(keyTemplate: MLXArray, valueTemplate: MLXArray) {
        guard keys == nil else { return }
        keys = MLXArray.zeros(
            [1, kvHeads, window, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
        values = MLXArray.zeros(
            [1, kvHeads, window, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
    }
}
