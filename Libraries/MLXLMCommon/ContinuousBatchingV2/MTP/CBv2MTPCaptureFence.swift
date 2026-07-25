// CBv2MTPCaptureFence.swift
//
// Ordering an MTP round's pre-write KV captures ahead of the in-place writes
// the very same graph is about to perform.

import MLX

/// Publishes a graph edge from a round's KV captures back into the storage
/// they were gathered from, so the round's own writes cannot overtake them.
///
/// Kept as a named seam rather than inlined in `mtpBuildVerifyGraph` for two
/// reasons: it is the one place in the MTP loop that knows a concrete storage
/// backend, and the edge it publishes is not observable from the round's
/// outputs, so it needs to be assertable on its own.
enum CBv2MTPCaptureFence {

    /// **Why this is not automatic.** On the contiguous backend `snapshot()`
    /// hands back the live `MLXArray`s and MLX's own versioning does the
    /// work: the round's `concatenated` update produces a *different* array,
    /// so the capture keeps referring to the pre-round bytes. On the PAGED
    /// backend it does not. `PagedSequenceKV.snapshot()` is a LAZY gather
    /// over slabs that the write kernels mutate IN PLACE, and in-place writes
    /// are invisible to MLX's hazard tracking. Both the capture and the
    /// round's writes merely *consume* the group's current `writeFence`
    /// (`PagedKVPool.gather`, `PagedKVPool.writeTokens`), so they are
    /// siblings in one unevaluated graph — first forced together at
    /// `executeMTPRound`'s `asyncEval` — and which one the scheduler runs
    /// first is an MLX detail, not a guarantee. If a write wins, the
    /// drafter's "frozen" pre-round KV already contains the round's own
    /// speculative tokens: drafts diverge from the target, acceptance
    /// collapses, and greedy token-exactness can break SILENTLY, with no
    /// crash to point at.
    ///
    /// It does not corrupt today only because a windowed ring is sized
    /// `window + maxPrefillChunk`, and the resulting ~528-token alias margin
    /// puts the round's writes outside the gathered range. That margin is an
    /// accident, and it is exactly what WS-1.2/3.1's ring shrink deletes —
    /// which is why the shrink is held out of this wave and this is not.
    ///
    /// **The fix: a fence BACK-edge.** `PagedKVPool.gather` already folds the
    /// group's write fence into its page index (`MLXArray(pages) +
    /// g.writeFence * 0`) so a read orders AFTER every prior write. That edge
    /// is one-directional. Here we run it the other way and fold the capture
    /// back INTO the fence, so every LATER write is forced to order after the
    /// gather: both write paths — `PagedKVPool.writeTokens`' bulk write and
    /// `PagedLayerCache`'s fused decode write — consume `group.writeFence`
    /// and re-publish its successor, so nothing reaches the slabs without
    /// first taking a dependency on the captures. `* 0` in `int32` is exactly
    /// zero for every input, including whatever an out-of-range float→int
    /// conversion produces, so the fence keeps its VALUE and gains only the
    /// graph edge. No host sync, so the round keeps its pipelining; the cost
    /// is one reduction pass over tensors the drafter forces anyway.
    ///
    /// A `.copy()` or a `stopGradient` would NOT do. The hazard is invisible
    /// to MLX, so the remedy has to be a real graph edge or a real
    /// evaluation — anything lazy just adds another sibling node.
    ///
    /// - Parameter captured: each capture paired with the row it was
    ///   gathered from.
    /// - Returns: the capture arrays that could NOT be fenced, because their
    ///   row exposes no write fence to hook. The caller must `eval` them:
    ///   that is the blunt equivalent, one host sync instead of a graph edge.
    ///   Empty for both backends that exist today — a future recyclable
    ///   backend must not silently inherit no protection at all.
    @discardableResult
    static func publish(
        _ captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)]
    ) -> [MLXArray] {
        var unfenceable: [MLXArray] = []
        for capture in captured {
            guard let paged = capture.row as? PagedSequenceKV else {
                unfenceable.append(capture.keys)
                unfenceable.append(capture.values)
                continue
            }
            let group = paged.pool.group(paged.groupKey)
            let edge =
                (capture.keys.sum() + capture.values.sum())
                .asType(group.writeFence.dtype) * 0
            group.writeFence = group.writeFence + edge
        }
        return unfenceable
    }
}
