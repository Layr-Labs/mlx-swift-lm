// PrefixReusePlan.swift
//
// Typed model/backend capability and per-match M/C/R contract for exact
// prefix reuse. Unknown layouts and backends fail cold.

import Foundation

/// KV backend identity used by the prefix-reuse planner. Prefix reuse is an
/// explicit backend contract: unknown implementations never inherit support.
public enum CBv2PrefixReuseBackend: String, Sendable, Equatable {
    /// Contiguous, unquantized native floating-point rows. Current Gemma QAT
    /// caches fp16; GPT-OSS full-attention rows are fp32. Dynamic plans bind
    /// exact staged `nbytes`, so neither is estimated as the other.
    case contiguousUnquantized = "contiguous_unquantized"
    case pagedFP16 = "paged_fp16"
    case unknown
}

/// Exact reuse mechanism selected for one model layout.
public enum CBv2PrefixReuseStrategy: String, Sendable, Equatable {
    case direct
    case tailReplay = "tail_replay"
    case frozenFullReplay = "frozen_full_replay"
}

/// Stable, low-cardinality construction refusal reasons.
public enum CBv2PrefixReuseUnsupportedReason: String, Sendable, Equatable {
    case emptyLayout = "empty_layout"
    case invalidLayout = "invalid_layout"
    case pagedHybridRequiresDualCursor = "paged_hybrid_requires_dual_cursor"
    case unknownBackend = "unknown_backend"
    case accountingOverflow = "accounting_overflow"
}

/// Model/backend capability established before a prefix cache is constructed.
/// A dynamic `CBv2PrefixReusePlan` is then derived for each matched boundary.
public struct CBv2PrefixReuseCapability: Sendable, Equatable {
    public let backend: CBv2PrefixReuseBackend
    public let strategy: CBv2PrefixReuseStrategy?
    public let conservativeReplayBoundTokens: Int
    public let fullKVBytesPerToken: Int
    public let unsupportedReason: CBv2PrefixReuseUnsupportedReason?

    public var isSupported: Bool {
        strategy != nil && unsupportedReason == nil
    }

    private init(
        backend: CBv2PrefixReuseBackend,
        strategy: CBv2PrefixReuseStrategy?,
        conservativeReplayBoundTokens: Int,
        fullKVBytesPerToken: Int,
        unsupportedReason: CBv2PrefixReuseUnsupportedReason?
    ) {
        self.backend = backend
        self.strategy = strategy
        self.conservativeReplayBoundTokens = conservativeReplayBoundTokens
        self.fullKVBytesPerToken = fullKVBytesPerToken
        self.unsupportedReason = unsupportedReason
    }

    /// Derive support from the exact layer layout and backend. The
    /// conservative dependency span is windowed-layer count × largest window.
    public static func derive(
        layerKinds: [CBv2LayerKind],
        backend: CBv2PrefixReuseBackend
    ) -> Self {
        guard !layerKinds.isEmpty else {
            return unsupported(backend: backend, reason: .emptyLayout)
        }

        var maxWindow = 0
        var windowCount = 0
        var sawWindowed = false
        var hasOwningFullAfterWindow = false
        var fullKVBytesPerToken = 0

        for (index, kind) in layerKinds.enumerated() {
            guard kind.headDim > 0, kind.kvHeads > 0, kind.queryHeads > 0 else {
                return unsupported(backend: backend, reason: .invalidLayout)
            }
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < index,
                    layerKinds[source].sharesKVWithLayer == nil,
                    layerKinds[source].attention == kind.attention,
                    layerKinds[source].kvHeads == kind.kvHeads,
                    layerKinds[source].headDim == kind.headDim
                else {
                    return unsupported(backend: backend, reason: .invalidLayout)
                }
            }
            switch kind.attention {
            case .slidingWindow(let window):
                guard window > 0 else {
                    return unsupported(backend: backend, reason: .invalidLayout)
                }
                maxWindow = max(maxWindow, window)
                windowCount += 1
                sawWindowed = true
            case .full:
                guard kind.sharesKVWithLayer == nil else { continue }
                if sawWindowed { hasOwningFullAfterWindow = true }
                let (elements, elementsOverflow) = kind.kvHeads.multipliedReportingOverflow(
                    by: kind.headDim)
                let (kvElements, kvOverflow) = elements.multipliedReportingOverflow(by: 2)
                let (bytes, bytesOverflow) = kvElements.multipliedReportingOverflow(by: 2)
                let (sum, sumOverflow) = fullKVBytesPerToken.addingReportingOverflow(bytes)
                guard !elementsOverflow, !kvOverflow, !bytesOverflow, !sumOverflow else {
                    return unsupported(backend: backend, reason: .accountingOverflow)
                }
                fullKVBytesPerToken = sum
            }
        }

        let (product, overflow) = windowCount.multipliedReportingOverflow(by: maxWindow)
        let replayBound = overflow ? Int.max : product

        if hasOwningFullAfterWindow {
            switch backend {
            case .contiguousUnquantized:
                return Self(
                    backend: backend,
                    strategy: .frozenFullReplay,
                    conservativeReplayBoundTokens: replayBound,
                    fullKVBytesPerToken: fullKVBytesPerToken,
                    unsupportedReason: nil)
            case .pagedFP16:
                // WS-4.1. Paged pays ONE EXTRA WINDOW of replay over
                // contiguous, and the reason is a difference between the two
                // prefill paths, not a safety margin.
                //
                // `CBv2FrozenReplayFullSequenceKV.update` discards the
                // replayed projections and returns the CACHED keys for the
                // whole chunk, diagonal included, so a contiguous frozen
                // replay is exact from the first position whose sliding cone
                // fits inside [C, M) — the `windowCount * maxWindow` bound.
                // `PagedLayerCache.prefillKV` cannot: it assembles
                // `gather([baseOffset, queryStart)) ++ chunk`, and the chunk
                // half is the freshly projected K/V the layer was called
                // with. A frozen paged row therefore contributes exact keys
                // for everything BEFORE the current chunk and poisoned ones
                // inside it, which pushes the first exact position back by at
                // most one `maxPrefillChunk`.
                //
                // The capability cannot see `maxPrefillChunk` (it is pool
                // config, and `derive` is model shape plus backend identity),
                // so it pays `maxWindow` — one more windowed layer's worth —
                // and `PagedKVBackend.makeSequenceState(adopting:)` re-checks
                // the plan against the pool's actual chunk before adopting,
                // refusing to a cold prefill if a hostile config ever set
                // `maxPrefillChunk > maxWindow`. Buying the slack here and
                // verifying it there keeps the unsafe combination
                // unreachable without plumbing pool config into the planner.
                let (padded, padOverflow) = replayBound.addingReportingOverflow(maxWindow)
                guard !padOverflow else {
                    return unsupported(
                        backend: backend,
                        reason: .accountingOverflow,
                        replayBound: replayBound,
                        fullKVBytesPerToken: fullKVBytesPerToken)
                }
                return Self(
                    backend: backend,
                    strategy: .frozenFullReplay,
                    conservativeReplayBoundTokens: padded,
                    fullKVBytesPerToken: fullKVBytesPerToken,
                    unsupportedReason: nil)
            case .unknown:
                return unsupported(
                    backend: backend,
                    reason: .unknownBackend,
                    replayBound: replayBound,
                    fullKVBytesPerToken: fullKVBytesPerToken)
            }
        }

        guard backend != .unknown else {
            return unsupported(
                backend: backend,
                reason: .unknownBackend,
                replayBound: replayBound,
                fullKVBytesPerToken: fullKVBytesPerToken)
        }
        return Self(
            backend: backend,
            strategy: replayBound == 0 ? .direct : .tailReplay,
            conservativeReplayBoundTokens: replayBound,
            fullKVBytesPerToken: fullKVBytesPerToken,
            unsupportedReason: nil)
    }

    /// Produce one immutable match-specific contract. Exact staged bytes come
    /// from the hit arrays, so native fp16/fp32 layouts are accounted as held.
    ///
    /// `restoringWindowsAtBoundary` is the adopter's promise that it holds an
    /// EXACT sliding window ending at `matchedBoundary` for every windowed
    /// layer — the paged seam's `CBv2PagedWindowSnapshot`, sourced from the
    /// provider's per-block window sidecars. It collapses R to zero and puts
    /// both cursors at M, which is the form
    /// `PagedSeamContract` calls "no second cursor". The promise is CHECKED,
    /// not trusted: `PagedKVBackend.makeSequenceState(adopting:)` refuses a
    /// zero-replay plan whose prefix does not carry an admissible window for
    /// every owning windowed layer, and the engine then cold-prefills.
    public func plan(
        matchedBoundary: Int,
        exactStagedFullKVBytes: Int? = nil,
        maximumSequenceLength: Int? = nil,
        nominalFullKVBytesPerToken: Int? = nil,
        fixedWindowCapacityBytes: Int = 0,
        restoringWindowsAtBoundary: Bool = false
    ) -> CBv2PrefixReusePlan? {
        guard let strategy, unsupportedReason == nil, matchedBoundary > 0 else {
            return nil
        }
        let nominalBytesPerToken =
            nominalFullKVBytesPerToken ?? fullKVBytesPerToken
        guard nominalBytesPerToken >= 0, fixedWindowCapacityBytes >= 0 else {
            return nil
        }
        // A restored window makes the replay unnecessary, but only for the
        // strategy that keeps the full rows at M. Tail replay restores its
        // full rows to C, so its windowed rows have to march from C too.
        let restoresWindows = restoringWindowsAtBoundary && strategy == .frozenFullReplay
        let replayTokens =
            restoresWindows ? 0 : min(matchedBoundary, conservativeReplayBoundTokens)
        let replayStart = matchedBoundary - replayTokens
        guard replayStart > 0 || replayTokens == 0 else { return nil }
        let restoredFullTokens =
            strategy == .frozenFullReplay ? matchedBoundary : replayStart
        let capacityReservationTokens =
            strategy == .frozenFullReplay ? matchedBoundary : replayStart

        let exactBytesPerToken: Int
        let stagedBytes: Int
        if let exactStagedFullKVBytes {
            guard exactStagedFullKVBytes >= 0,
                exactStagedFullKVBytes % matchedBoundary == 0
            else { return nil }
            exactBytesPerToken = exactStagedFullKVBytes / matchedBoundary
            stagedBytes = exactStagedFullKVBytes
        } else {
            exactBytesPerToken = fullKVBytesPerToken
            guard let computed = Self.multiply(matchedBoundary, exactBytesPerToken) else {
                return nil
            }
            stagedBytes = computed
        }
        guard let residentBytes = Self.multiply(restoredFullTokens, exactBytesPerToken) else {
            return nil
        }
        let additionalFullBytesPerToken = max(
            0, exactBytesPerToken - nominalBytesPerToken)
        let fullCapacityTokens = max(
            restoredFullTokens,
            maximumSequenceLength ?? restoredFullTokens)
        guard
            let exactFullCapacityBytes = Self.multiply(
                fullCapacityTokens, exactBytesPerToken),
            let nominalInitiallyReserved = Self.multiply(
                capacityReservationTokens, nominalBytesPerToken)
        else { return nil }
        let fullCapacityAdjustment = max(
            0, exactFullCapacityBytes - nominalInitiallyReserved)
        let (initialAdditionalCapacityBytes, capacityOverflow) =
            fullCapacityAdjustment.addingReportingOverflow(fixedWindowCapacityBytes)
        guard !capacityOverflow else { return nil }
        return CBv2PrefixReusePlan(
            backend: backend,
            strategy: strategy,
            matchedBoundary: matchedBoundary,
            replayStart: replayStart,
            replayTokens: replayTokens,
            prefillTokensSaved: replayStart,
            restoredFullTokens: restoredFullTokens,
            capacityReservationTokens: capacityReservationTokens,
            nominalFullKVBytesPerToken: nominalBytesPerToken,
            fullKVBytesPerToken: exactBytesPerToken,
            additionalFullKVBytesPerToken: additionalFullBytesPerToken,
            initialAdditionalCapacityBytes: initialAdditionalCapacityBytes,
            fullCapacityTokensReserved: fullCapacityTokens,
            stagedFullKVBytes: stagedBytes,
            residentFullKVBytes: residentBytes)
    }

    /// Preserve the historical backend-only contract: callers already hand
    /// this overload snapshots sliced to the adopted offset C and perform any
    /// required replay themselves. Interleaved hybrids require the explicit
    /// M/C dual-cursor plan and therefore fail cold here.
    public func compatibilityPlan(
        adoptedOffset: Int,
        exactStagedFullKVBytes: Int,
        maximumSequenceLength: Int
    ) -> CBv2PrefixReusePlan? {
        guard let strategy,
            strategy != .frozenFullReplay,
            unsupportedReason == nil,
            adoptedOffset > 0,
            maximumSequenceLength >= adoptedOffset,
            exactStagedFullKVBytes >= 0,
            exactStagedFullKVBytes % adoptedOffset == 0
        else { return nil }
        let exactBytesPerToken = exactStagedFullKVBytes / adoptedOffset
        guard
            let exactFullCapacityBytes = Self.multiply(
                maximumSequenceLength, exactBytesPerToken),
            let nominalInitiallyReserved = Self.multiply(
                adoptedOffset, fullKVBytesPerToken)
        else { return nil }
        return CBv2PrefixReusePlan(
            backend: backend,
            strategy: strategy,
            matchedBoundary: adoptedOffset,
            replayStart: adoptedOffset,
            replayTokens: 0,
            prefillTokensSaved: adoptedOffset,
            restoredFullTokens: adoptedOffset,
            capacityReservationTokens: adoptedOffset,
            nominalFullKVBytesPerToken: fullKVBytesPerToken,
            fullKVBytesPerToken: exactBytesPerToken,
            additionalFullKVBytesPerToken: max(
                0, exactBytesPerToken - fullKVBytesPerToken),
            initialAdditionalCapacityBytes: max(
                0, exactFullCapacityBytes - nominalInitiallyReserved),
            fullCapacityTokensReserved: maximumSequenceLength,
            stagedFullKVBytes: exactStagedFullKVBytes,
            residentFullKVBytes: exactStagedFullKVBytes)
    }

    private static func unsupported(
        backend: CBv2PrefixReuseBackend,
        reason: CBv2PrefixReuseUnsupportedReason,
        replayBound: Int = 0,
        fullKVBytesPerToken: Int = 0
    ) -> Self {
        Self(
            backend: backend,
            strategy: nil,
            conservativeReplayBoundTokens: replayBound,
            fullKVBytesPerToken: fullKVBytesPerToken,
            unsupportedReason: reason)
    }

    private static func multiply(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }
}

/// Match-specific exact prefix-reuse contract.
///
/// M = `matchedBoundary`, C = `replayStart`, R = `replayTokens`.
public struct CBv2PrefixReusePlan: Sendable, Equatable {
    public let backend: CBv2PrefixReuseBackend
    public let strategy: CBv2PrefixReuseStrategy
    public let matchedBoundary: Int
    public let replayStart: Int
    public let replayTokens: Int
    public let prefillTokensSaved: Int
    public let restoredFullTokens: Int
    public let capacityReservationTokens: Int
    public let nominalFullKVBytesPerToken: Int
    public let fullKVBytesPerToken: Int
    public let additionalFullKVBytesPerToken: Int
    public let initialAdditionalCapacityBytes: Int
    public let fullCapacityTokensReserved: Int
    public let stagedFullKVBytes: Int
    public let residentFullKVBytes: Int

    /// A frozen-full plan with nothing to replay: the adopter promised an
    /// EXACT sliding window ending at M for every windowed layer, so both
    /// cursors sit at M and the request resumes with an ordinary prefill of
    /// `[M, promptLength)`.
    ///
    /// The two frozen-full forms are mutually exclusive and a backend must
    /// branch on this rather than on `replayTokens > 0` ad hoc: the restore
    /// form REQUIRES a window payload for every owning windowed layer and
    /// must refuse without one, while the replay form REQUIRES the opposite
    /// (windowed layers are recomputed, so a payload for one means the
    /// donor and the plan disagree about what is being adopted).
    public var requiresExactWindowRestore: Bool {
        strategy == .frozenFullReplay && replayTokens == 0
    }

    /// Clamp a proposed prefill chunk so it cannot cross C or M.
    public func clampedChunk(start: Int, proposed: Int) -> Int {
        guard proposed > 0 else { return proposed }
        var result = proposed
        for boundary in [replayStart, matchedBoundary]
        where start < boundary && result > boundary - start {
            result = boundary - start
        }
        return result
    }

    /// Frozen replay reserves through M atomically; work below M must not
    /// reserve those positions a second time.
    public func capacityTokensForChunk(start: Int, count: Int) -> Int {
        guard count > 0 else { return 0 }
        // Owning full rows are prepaid through maxLength, and every sliding
        // row's fixed ring was reserved at state construction. No later chunk
        // can increase physical KV for this adopted request.
        if fullCapacityTokensReserved > restoredFullTokens { return 0 }
        guard strategy == .frozenFullReplay else { return count }
        guard start < matchedBoundary else { return count }
        return max(0, count - (matchedBoundary - start))
    }

    public func capacityBytesForChunk(start: Int, count: Int) -> Int {
        if fullCapacityTokensReserved > restoredFullTokens { return 0 }
        let tokens = capacityTokensForChunk(start: start, count: count)
        let (bytes, overflow) = tokens.multipliedReportingOverflow(
            by: additionalFullKVBytesPerToken)
        return overflow ? Int.max : bytes
    }
}
