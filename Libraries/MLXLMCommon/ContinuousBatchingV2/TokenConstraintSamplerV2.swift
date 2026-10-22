// Copyright © 2026 Eigen Labs.
//
// Row-local host state and dense logit masking for CBv2 token constraints.
// Ordinary batches never enter this file's mask construction path.
//
// Device masks: each row keeps a small LRU of uploaded `[1, vocab]` bool
// masks keyed by the constraint's opt-in `maskCacheKey`. A hit performs no
// O(vocab) host work and no upload; the hard mask is one `which` over the
// batch. Masked values are bitwise those of the historical
// takeAlong/putAlong construction (allowed logits unchanged, every other
// entry -infinity).
//
// Chained decode (opt-in, `CBv2TokenConstraint.allowsChainedDecode`): a
// speculative mask is built for step N+1 while step N's token is still on
// the GPU, from the state confirmed through N-1 and the budget N+1 will
// have. It is a pure function of that pair and latches nothing. Confirming
// step N's token either validates it (the confirmed state and budget equal
// the ones it used, so the unchained path would have built the same mask)
// or reports the row through `takeInvalidated()` for an exact redraw.

import MLX

final class CBv2TokenConstraintSampler {
    /// The mask an in-flight chained successor was sampled under.
    private struct Speculation {
        let state: Int
        /// Confirmed tokens when the mask was built; the successor's sample is
        /// output index `consumed + 1`.
        let consumed: Int
        /// The impossible-state failure that mask produced, if any.
        let failure: String?
    }

    /// Bounded per-row LRU of uploaded masks.
    private struct MaskCache {
        static let capacity = 8
        private(set) var entries: [(key: Int, mask: MLXArray)] = []

        mutating func lookup(_ key: Int, vocab: Int) -> MLXArray? {
            guard let index = entries.firstIndex(where: { $0.key == key }) else { return nil }
            let entry = entries.remove(at: index)
            guard entry.mask.dim(-1) == vocab else { return nil }
            entries.append(entry)
            return entry.mask
        }

        mutating func insert(_ key: Int, _ mask: MLXArray) {
            entries.removeAll { $0.key == key }
            if entries.count == Self.capacity { entries.removeFirst() }
            entries.append((key, mask))
        }
    }

    private struct RowState {
        let constraint: any CBv2TokenConstraint
        var state: Int
        var consumed: Int
        var failure: String?
        var speculation: Speculation?
        /// A validated speculative mask failed. The failure belongs to the
        /// successor's sample, so it is latched when that sample is
        /// confirmed, exactly where the unchained path would observe it.
        var successorFailure: String?
        var masks = MaskCache()
    }

    private var rows: [CBv2RequestID: RowState] = [:]
    private var invalidated: [CBv2RequestID] = []
    private var openRow: MLXArray?

    /// Test hooks: mask builds that ran `allowedTokenIDs` + upload, and
    /// cached-mask reuses.
    private(set) var maskBuildCount = 0
    private(set) var maskCacheHitCount = 0

    var hasRows: Bool { !rows.isEmpty }

    func hasRows(in ids: [CBv2RequestID]) -> Bool {
        ids.contains { rows[$0] != nil }
    }

    func configure(_ descriptions: [CBv2SamplerRow]) {
        for row in descriptions {
            guard let constraint = row.tokenConstraint else {
                rows.removeValue(forKey: row.id)
                continue
            }
            let existing = rows[row.id]
            let sameConstraint =
                existing.map {
                    ObjectIdentifier($0.constraint) == ObjectIdentifier(constraint)
                } ?? false
            if let existing, sameConstraint, existing.consumed == row.outputTokens.count {
                continue
            }

            var rebuilt = RowState(
                constraint: constraint,
                state: constraint.initialState,
                consumed: 0,
                failure: nil)
            // Masks are a function of (constraint instance, key) only.
            if let existing, sameConstraint { rebuilt.masks = existing.masks }
            for token in row.outputTokens {
                guard
                    let next = constraint.nextState(
                        state: rebuilt.state, tokenID: token)
                else {
                    rebuilt.failure = CBv2TokenConstraintFailure.impossibleState
                    break
                }
                rebuilt.state = next
                rebuilt.consumed += 1
            }
            rows[row.id] = rebuilt
        }
    }

    /// A speculative sample starts a new chain epoch: invalidations reported
    /// for any earlier successor are stale.
    func beginSpeculation() {
        invalidated.removeAll(keepingCapacity: true)
    }

    /// Apply a per-row boolean mask to `[B, vocab]` logits. The common path
    /// returns the original array without allocating a mask.
    ///
    /// `speculative`: every row has exactly one launched-but-unconfirmed
    /// sample (chained decode). Masks use the confirmed state with the
    /// successor's budget and latch nothing; see the file header.
    func mask(
        _ logits: MLXArray, requestIDs: [CBv2RequestID], speculative: Bool = false
    ) -> MLXArray {
        guard hasRows(in: requestIDs) else { return logits }

        let vocab = logits.dim(-1)
        var rowMasks: [MLXArray] = []
        rowMasks.reserveCapacity(requestIDs.count)
        for id in requestIDs {
            guard var row = rows[id] else {
                rowMasks.append(open(vocab: vocab))
                continue
            }
            let consumed = row.consumed + (speculative ? 1 : 0)
            let remaining = max(0, row.constraint.maxTokens - consumed)
            let key =
                row.failure == nil
                ? row.constraint.maskCacheKey(state: row.state, remainingTokens: remaining)
                : nil
            var failure: String?
            if let key, let cached = row.masks.lookup(key, vocab: vocab) {
                maskCacheHitCount += 1
                rowMasks.append(cached)
            } else {
                var allowed =
                    row.failure == nil
                    ? row.constraint.allowedTokenIDs(
                        state: row.state, remainingTokens: remaining)
                    : []
                if allowed.isEmpty || allowed.contains(where: { $0 < 0 || $0 >= vocab }) {
                    failure = CBv2TokenConstraintFailure.impossibleState
                    allowed = [row.constraint.fallbackTokenID]
                }
                let mask = Self.deviceMask(allowed: allowed, vocab: vocab)
                maskBuildCount += 1
                if let key, failure == nil { row.masks.insert(key, mask) }
                rowMasks.append(mask)
            }
            if speculative {
                row.speculation = Speculation(
                    state: row.state, consumed: row.consumed, failure: failure)
            } else {
                row.speculation = nil
                row.successorFailure = nil
                if let failure { row.failure = failure }
            }
            rows[id] = row
        }
        let mask = rowMasks.count == 1 ? rowMasks[0] : concatenated(rowMasks, axis: 0)
        return which(mask, logits, MLXArray(-Float.infinity).asType(logits.dtype))
    }

    /// `[1, vocab]` bool, true exactly at `allowed`. An out-of-range id
    /// (only possible for a malformed fallback) yields an all-false row,
    /// i.e. every logit -infinity, as before.
    private static func deviceMask(allowed: [Int], vocab: Int) -> MLXArray {
        var flags = [Bool](repeating: false, count: vocab)
        if allowed.allSatisfy({ $0 >= 0 && $0 < vocab }) {
            for token in allowed { flags[token] = true }
        }
        return MLXArray(flags).reshaped([1, vocab])
    }

    private func open(vocab: Int) -> MLXArray {
        if let openRow, openRow.dim(-1) == vocab { return openRow }
        let row = MLXArray.ones([1, vocab], type: Bool.self)
        openRow = row
        return row
    }

    func confirm(tokens: [Int], requestIDs: [CBv2RequestID]) {
        precondition(tokens.count == requestIDs.count)
        for (token, id) in zip(tokens, requestIDs) {
            guard var row = rows[id] else { continue }
            let speculation = row.speculation
            row.speculation = nil
            if let pending = row.successorFailure {
                row.failure = pending
                row.successorFailure = nil
            }
            if row.failure == nil {
                if let next = row.constraint.nextState(state: row.state, tokenID: token) {
                    row.state = next
                    row.consumed += 1
                } else {
                    row.failure = CBv2TokenConstraintFailure.impossibleState
                }
            }
            if let speculation {
                // The successor's mask is exact iff it was built from this
                // confirmed state and budget. A failed row finishes at this
                // boundary (its successor is discarded); report it anyway so
                // the engine, not this class, decides.
                if row.failure == nil, speculation.state == row.state,
                    speculation.consumed + 1 == row.consumed
                {
                    row.successorFailure = speculation.failure
                } else {
                    invalidated.append(id)
                }
            }
            rows[id] = row
        }
    }

    /// Rows whose chained successor was sampled under a mask the confirmed
    /// token invalidated (take semantics).
    func takeInvalidated() -> [CBv2RequestID] {
        defer { invalidated.removeAll(keepingCapacity: true) }
        return invalidated
    }

    func failure(for id: CBv2RequestID) -> String? {
        rows[id]?.failure
    }

    func requestDidFinish(_ id: CBv2RequestID) {
        rows.removeValue(forKey: id)
    }
}
