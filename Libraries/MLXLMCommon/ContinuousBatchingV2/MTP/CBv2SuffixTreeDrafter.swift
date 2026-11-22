// CBv2SuffixTreeDrafter.swift
//
// ContinuousBatchingV2 — Zero-draft-cost adaptive multi-branch suffix-automaton
// drafter (`CBv2TokenOnlyMTPDrafter`).
//
// Maintains one incremental O(N) online suffix automaton (SAM) per active
// request over its confirmed `Int32` token history. At each decode step,
// suffix matching and branch scoring run purely on CPU (~0.02 ms) with zero
// extra GPU memory, zero KV snapshots, zero hidden-state carries, and zero
// seed steps.

import Foundation
import MLX

// MARK: - Online Suffix Automaton

/// Incremental O(N) online suffix automaton over `Int32` token ids.
///
/// Tracks per-state context length (`len`), suffix link (`link`), most recent
/// ending position (`endPos`), occurrence frequency (`freq`), and outgoing
/// transitions (`next`). Extending by `delta` newly confirmed tokens takes
/// amortized O(delta) time and memory with zero re-indexing of earlier tokens.
public struct OnlineSuffixAutomaton: Sendable {
    public struct State: Sendable {
        public var len: Int32
        public var link: Int32
        public var endPos: Int32
        public var freq: Int32
        public var next: [Int32: Int32]

        public init(
            len: Int32, link: Int32, endPos: Int32, freq: Int32, next: [Int32: Int32] = [:]
        ) {
            self.len = len
            self.link = link
            self.endPos = endPos
            self.freq = freq
            self.next = next
        }
    }

    public static let defaultMatchWindow = 64
    /// Cap cold-prompt initial indexing to the trailing window so 8K-128K prompts
    /// never stall `engineQueue` with tens of milliseconds of CPU dictionary allocation.
    public static let defaultMaxInitialIndexTokens = 4_096
    private static let ancestorUpdateHops = 4

    public private(set) var states: [State]
    public private(set) var lastState: Int32 = 0
    public private(set) var consumedCount: Int = 0

    public init() {
        self.states = [State(len: 0, link: -1, endPos: -1, freq: 0, next: [:])]
    }

    /// Drop all indexed states and reset back to the empty root state.
    public mutating func reset() {
        states.removeAll(keepingCapacity: true)
        states.append(State(len: 0, link: -1, endPos: -1, freq: 0, next: [:]))
        lastState = 0
        consumedCount = 0
    }

    /// Incrementally index only `tokens[consumedCount...]`. If `tokens` shrank
    /// (for example after preemption or request reuse), resets and re-indexes.
    /// On a cold prompt exceeding `maxInitialTokens`, indexes only the trailing
    /// `maxInitialTokens` suffix so initial construction stays sub-millisecond.
    public mutating func append(
        _ tokens: [Int],
        maxInitialTokens: Int = OnlineSuffixAutomaton.defaultMaxInitialIndexTokens
    ) {
        if tokens.count < consumedCount {
            reset()
        }
        guard tokens.count > consumedCount else { return }
        if consumedCount == 0 && maxInitialTokens > 0 && tokens.count > maxInitialTokens {
            consumedCount = tokens.count - maxInitialTokens
        }
        let delta = tokens.count - consumedCount
        let desiredCapacity = states.count + delta * 2
        if states.capacity < desiredCapacity {
            states.reserveCapacity(desiredCapacity)
        }
        for index in consumedCount ..< tokens.count {
            extend(Int32(truncatingIfNeeded: tokens[index]))
        }
    }

    private mutating func extend(_ token: Int32) {
        let pos = Int32(consumedCount)
        consumedCount += 1

        let cur = Int32(states.count)
        let curLen = states[Int(lastState)].len + 1
        states.append(State(len: curLen, link: 0, endPos: pos, freq: 1, next: [:]))

        var p = lastState
        while p != -1 && states[Int(p)].next[token] == nil {
            states[Int(p)].next[token] = cur
            p = states[Int(p)].link
        }

        if p == -1 {
            states[Int(cur)].link = 0
        } else {
            let q = states[Int(p)].next[token]!
            if states[Int(p)].len + 1 == states[Int(q)].len {
                states[Int(cur)].link = q
            } else {
                let clone = Int32(states.count)
                let qState = states[Int(q)]
                states.append(
                    State(
                        len: states[Int(p)].len + 1,
                        link: qState.link,
                        endPos: qState.endPos,
                        freq: qState.freq,
                        next: qState.next))
                while p != -1 && states[Int(p)].next[token] == q {
                    states[Int(p)].next[token] = clone
                    p = states[Int(p)].link
                }
                states[Int(q)].link = clone
                states[Int(cur)].link = clone
            }
        }

        // Bounded constant-hop frequency/recency propagation preserves strict
        // O(1) amortized extend while keeping branch statistics accurate.
        var anc = states[Int(cur)].link
        var hops = 0
        while anc > 0 && hops < Self.ancestorUpdateHops {
            states[Int(anc)].freq &+= 1
            states[Int(anc)].endPos = pos
            anc = states[Int(anc)].link
            hops += 1
        }

        lastState = cur
    }

    /// Match the longest suffix of `tokens` (up to `matchWindow` tokens) that
    /// occurred earlier in the automaton and greedily roll out the highest-
    /// scoring continuation up to an adaptive draft horizon:
    ///   `k = min(maxDraft, matchLen >= 6 ? maxDraft : (matchLen >= 4 ? 5 : 3))`.
    ///
    /// Outgoing branches at `state` are scored by:
    ///   `2.0 * Float(state.len) + log1p(Float(next.freq))`
    ///   `+ 1.5 * (Float(next.endPos) / Float(max(1, consumedCount)))`
    /// with deterministic tie-breaking on `(endPos desc, token asc)`.
    public func proposeAdaptive(
        tokens: [Int],
        maxDraft: Int,
        minMatchLen: Int = 2,
        matchWindow: Int = Self.defaultMatchWindow
    ) -> (matchLen: Int, draft: [Int32]) {
        guard maxDraft > 0, minMatchLen > 0, !tokens.isEmpty, consumedCount > 0 else {
            return (0, [])
        }

        let windowStart = max(0, tokens.count - max(1, matchWindow))
        var state: Int32 = 0
        var matched = 0
        for index in windowStart ..< tokens.count {
            let tok = Int32(truncatingIfNeeded: tokens[index])
            while state != 0 && states[Int(state)].next[tok] == nil {
                state = states[Int(state)].link
                matched = min(matched, Int(states[Int(state)].len))
            }
            if let nextState = states[Int(state)].next[tok] {
                state = nextState
                matched += 1
            } else {
                state = 0
                matched = 0
            }
        }

        // If the matched state occurs only at the current stream frontier
        // (empty `next`), walk suffix links to the longest suffix with a
        // previously observed continuation.
        while state > 0 && states[Int(state)].next.isEmpty {
            state = states[Int(state)].link
            matched = state > 0 ? min(matched, Int(states[Int(state)].len)) : 0
        }

        guard state > 0, matched >= minMatchLen else {
            return (0, [])
        }

        let matchLen = matched
        let adaptiveLimit = matchLen >= 6 ? maxDraft : (matchLen >= 4 ? 5 : 3)
        let targetK = min(maxDraft, adaptiveLimit)
        var draft: [Int32] = []
        draft.reserveCapacity(targetK)

        let invConsumed = 1.0 / Float(max(1, consumedCount))
        var cur = state
        for _ in 0 ..< targetK {
            while cur > 0 && states[Int(cur)].next.isEmpty {
                cur = states[Int(cur)].link
            }
            guard cur > 0, Int(states[Int(cur)].len) >= minMatchLen else { break }
            let curState = states[Int(cur)]
            guard !curState.next.isEmpty else { break }

            let contextTerm = 2.0 * Float(curState.len)
            var bestToken: Int32 = 0
            var bestNext: Int32 = -1
            var bestScore: Float = -.infinity
            var bestEndPos: Int32 = -1

            for (tok, nextID) in curState.next {
                let nextState = states[Int(nextID)]
                let score =
                    contextTerm
                    + Foundation.log1p(Float(nextState.freq))
                    + 1.5 * (Float(nextState.endPos) * invConsumed)
                if score > bestScore
                    || (score == bestScore
                        && (nextState.endPos > bestEndPos
                            || (nextState.endPos == bestEndPos && tok < bestToken)))
                {
                    bestScore = score
                    bestToken = tok
                    bestNext = nextID
                    bestEndPos = nextState.endPos
                }
            }
            guard bestNext >= 0 else { break }
            draft.append(bestToken)
            cur = bestNext
        }

        guard !draft.isEmpty else { return (0, []) }
        return (matchLen, draft)
    }
}

// MARK: - Suffix-Automaton Token-Only Drafter

/// Universal zero-draft-cost CBv2 drafter powered by per-request online suffix
/// automata. Works with Gemma 4 and any attention-only `CBv2SteppableModel`
/// (including compacted Block-SDCM rows) with zero model weights or KV capture.
public final class CBv2SuffixTreeDrafter: CBv2TokenOnlyMTPDrafter {
    /// `nil` means universal (compatible with any non-recurrent target model).
    public var mtpTargetIdentity: ObjectIdentifier?
    public var supportsTargetPrefixAcceptance: Bool { true }
    public var commitsAcceptedPrefixPerRow: Bool { true }
    public var prefersBatchedRectangularAttention: Bool { true }
    public var maximumSpeculativeBatch: Int? { CBv2MTPConfig.maxTokenOnlySpeculativeBatch }

    public let minMatchLen: Int
    public let matchWindow: Int

    private var automata: [CBv2RequestID: OnlineSuffixAutomaton] = [:]

    public init(
        targetIdentity: ObjectIdentifier? = nil,
        minMatchLen: Int = 3,
        matchWindow: Int = OnlineSuffixAutomaton.defaultMatchWindow
    ) {
        self.mtpTargetIdentity = targetIdentity
        self.minMatchLen = max(1, minMatchLen)
        self.matchWindow = max(1, matchWindow)
    }

    public func hasCandidateMatch(
        for rows: [(id: CBv2RequestID, tokens: [Int])],
        maxDraftTokens: Int
    ) -> Bool {
        guard !rows.isEmpty, maxDraftTokens > 0 else { return false }
        let proposals = collectProposals(for: rows, maxDraftTokens: maxDraftTokens)
        return isProfitable(
            proposals: proposals, batchCount: rows.count, maxDraftTokens: maxDraftTokens)
    }

    public func proposeDrafts(
        for rows: [(id: CBv2RequestID, tokens: [Int])],
        maxDraftTokens: Int
    ) -> (depth: Int, draftsByRequest: [CBv2RequestID: [Int32]]) {
        guard !rows.isEmpty, maxDraftTokens > 0 else { return (0, [:]) }
        let proposals = collectProposals(for: rows, maxDraftTokens: maxDraftTokens)
        guard
            isProfitable(
                proposals: proposals, batchCount: rows.count, maxDraftTokens: maxDraftTokens)
        else {
            return (0, [:])
        }

        var maxProposed = 0
        for item in proposals where item.draft.count > maxProposed {
            maxProposed = item.draft.count
        }
        let depth = min(
            Self.batchDraftCap(batchCount: rows.count, maxDraftTokens: maxDraftTokens), maxProposed)
        guard depth > 0 else { return (0, [:]) }

        var draftsByRequest: [CBv2RequestID: [Int32]] = [:]
        draftsByRequest.reserveCapacity(rows.count)
        for item in proposals {
            if item.draft.count == depth {
                draftsByRequest[item.id] = item.draft
            } else if item.draft.count > depth {
                draftsByRequest[item.id] = Array(item.draft.prefix(depth))
            } else {
                // Non-matching or shorter-draft rows receive zero-padded dummy
                // drafts to keep the target verify forward rectangular. With
                // `commitsAcceptedPrefixPerRow == true`, they commit their own
                // exact target prefix and roll back rejected slots cleanly.
                var padded = item.draft
                padded.reserveCapacity(depth)
                padded.append(
                    contentsOf: repeatElement(Int32(0), count: depth - item.draft.count))
                draftsByRequest[item.id] = padded
            }
        }
        return (depth, draftsByRequest)
    }

    public func requestDidFinish(_ id: CBv2RequestID) {
        automata.removeValue(forKey: id)
    }

    public func removeAllRequestState() {
        automata.removeAll(keepingCapacity: false)
    }

    /// Number of active per-request automata currently retained (for unit tests).
    var trackedRequestCountForTesting: Int { automata.count }

    // MARK: - CBv2MTPDrafter neural stubs (unreachable on token-only path)

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        preconditionFailure(
            "CBv2SuffixTreeDrafter is token-only; prepare(rows:) must not be called")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure(
            "CBv2SuffixTreeDrafter is token-only; draftStep(tokens:hidden:prepared:) must not be called"
        )
    }

    // MARK: - Private helpers

    private struct RowProposal {
        let id: CBv2RequestID
        let matchLen: Int
        let draft: [Int32]
    }

    private func collectProposals(
        for rows: [(id: CBv2RequestID, tokens: [Int])],
        maxDraftTokens: Int
    ) -> [RowProposal] {
        var proposals: [RowProposal] = []
        proposals.reserveCapacity(rows.count)
        for row in rows {
            // Mutate the automaton in-place via `Dictionary.subscript(_:default:)`
            // so `states` stays uniquely referenced (zero CoW copies).
            let result = withAutomaton(for: row.id) { sam -> (matchLen: Int, draft: [Int32]) in
                sam.append(row.tokens)
                return sam.proposeAdaptive(
                    tokens: row.tokens,
                    maxDraft: maxDraftTokens,
                    minMatchLen: minMatchLen,
                    matchWindow: matchWindow)
            }
            proposals.append(
                RowProposal(id: row.id, matchLen: result.matchLen, draft: result.draft))
        }
        return proposals
    }

    @inline(__always)
    private func withAutomaton<R>(
        for id: CBv2RequestID,
        _ body: (inout OnlineSuffixAutomaton) -> R
    ) -> R {
        body(&automata[id, default: OnlineSuffixAutomaton()])
    }

    /// Compiled-trace-safe draft depth cap per batch size:
    /// - `B == 1`: full `maxDraftTokens` (memory-bandwidth bound).
    /// - `B == 2`: `<= 2` (`width = 3`, `2 * 3 * 8 = 48 < 64` stays on Gemma 4's compiled width-3 trace).
    /// - `B >= 3`: `<= 1` (`width = 2`, stays on Gemma 4's compiled sorted width-2 trace up to `B = maxRows / 2`).
    private static func batchDraftCap(batchCount: Int, maxDraftTokens: Int) -> Int {
        switch batchCount {
        case ...1:
            return maxDraftTokens
        case 2:
            return min(maxDraftTokens, 2)
        default:
            return min(maxDraftTokens, 1)
        }
    }

    /// Batch profitability gate evaluated on post-cap effective draft lengths
    /// so admission stays monotonic across `maxDraftTokens`:
    /// - `B == 1`: require `matchLen >= minMatchLen` (default 3) and at least
    ///   `min(2, maxDraftTokens)` drafted tokens so spurious short n-grams never
    ///   break GPU decode chaining.
    /// - `B > 1`: require strong matches (`matchLen >= max(minMatchLen, 4)`) on at least
    ///   half the active batch (`max(2, (B + 1) / 2)`) and post-cap mean drafted tokens
    ///   `>= 0.5 * Double(batchCap)`.
    private func isProfitable(
        proposals: [RowProposal], batchCount: Int, maxDraftTokens: Int
    ) -> Bool {
        guard batchCount > 0, maxDraftTokens > 0 else { return false }
        if batchCount == 1 {
            guard let first = proposals.first else { return false }
            return first.matchLen >= minMatchLen && first.draft.count >= min(2, maxDraftTokens)
        }
        let cap = Self.batchDraftCap(batchCount: batchCount, maxDraftTokens: maxDraftTokens)
        let batchMinMatch = max(minMatchLen, 4)
        var strongMatches = 0
        var totalEffectiveProposed = 0
        for item in proposals {
            let effectiveDraftCount = min(item.draft.count, cap)
            if item.matchLen >= batchMinMatch && effectiveDraftCount > 0 {
                strongMatches += 1
                totalEffectiveProposed += effectiveDraftCount
            }
        }
        let minMatchingRows = max(2, (batchCount + 1) / 2)
        let meanEffective = Double(totalEffectiveProposed) / Double(batchCount)
        return strongMatches >= minMatchingRows && meanEffective >= 0.5 * Double(cap)
    }
}
