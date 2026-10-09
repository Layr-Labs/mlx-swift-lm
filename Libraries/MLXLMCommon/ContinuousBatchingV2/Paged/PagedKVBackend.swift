// PagedKVBackend.swift
//
// `CBv2KVBackend` factory over a `PagedKVPool` (WS-C). Swappable with the
// WS-A contiguous backend behind the same contract: the scheduler and
// models never see the difference.
//
// Eligibility is validated at construction (engine build time), per the
// contract: unsupported head dims, shapes over the paged kernel's
// threadgroup-memory budget (`PagedAttentionKernel.ineligibilityReason` —
// dispatching one is an uncatchable Metal fatal; the kernel's head split
// keeps every supported head dim within budget, incl. Gemma-4 global
// layers at headDim 512 / GQA 8), quant schemes, or malformed KV-sharing
// throw `CBv2KVError.backendIneligible` before any request is admitted.
// Attention sinks ARE supported (they are a kernel parameter here).
//
// Admission model: the worst-case page count for a request's `maxLength`
// is reserved UP FRONT — at admission via `reserve(layerKinds:maxLength:)`,
// or (when no admission-time reservation was taken) lazily by
// `makeSequenceState`, which reconciles against any prior reservation so the
// pages are charged exactly once. `reserve` throws `capacityExhausted` when
// the pool cannot honor the demand, so an admission controller can reject or
// queue a request that would otherwise fail at materialization (Codex P2:
// several same-step admissions must not over-commit the pool). Physical
// pages materialize lazily as tokens are written (`bytesInUse` stays
// truthful); `CBv2SequenceKV.update` therefore never fails mid-decode. See
// PagedKVPool.swift for the rationale.
//
// Prefix adoption (WS-4.1): paged serves `.direct`, `.tailReplay` and BOTH
// forms of `.frozenFullReplay` — the dual-cursor replay, and the zero-replay
// restore that arrives once a window payload exists. See
// `makeSequenceState(adopting:plan:layerKinds:maxLength:)` for the three
// shapes. Two properties of that path are load-bearing and easy to lose:
//
//   * a refusal is a THROW, never a trap. The engine's only recovery is a
//     cold prefill, and it can only take it if the adoption reports failure
//     instead of aborting the daemon;
//   * validation of EVERY layer completes before the first page is reserved.
//     A frozen-full hit whose sliding side cannot be completed must leave the
//     pool exactly as it found it, because a half-restored hybrid does not
//     fail loudly — the sliding rows simply attend fewer keys than they
//     should, and nothing downstream can see it.

import Foundation
import MLX

private final class PagedGatheredRequestOwner {
    weak var row: PagedSequenceKV?
    init(_ row: PagedSequenceKV) { self.row = row }
}

public final class PagedKVBackend: CBv2KVBackend {
    public var prefixReuseBackend: CBv2PrefixReuseBackend { .pagedFP16 }
    public let pool: PagedKVPool
    /// The model's per-layer structure this backend was built for.
    public let layerKinds: [CBv2LayerKind]
    package private(set) var nativeModelBinding: CBv2NativePagedModelBinding?
    /// Non-owning physical-page prefix index. nil preserves the historical
    /// paged backend byte-for-byte; the engine discovers the optional native
    /// capability without changing the snapshot-cache contract.
    let residentPrefixIndex: PagedPrefixBlockIndex?
    /// WHEN this backend's slabs become MLX-resident. See
    /// PagedKVSlabCommitment.swift — the default defers the commitment past
    /// engine construction so an idle pool does not pre-empt a co-resident
    /// model's post-load headroom measurement (D1).
    public let slabCommitment: PagedKVSlabCommitment
    /// Fixed-reference commitment flag. Segmented pools instead track each
    /// live buffer and may retire it after the final request releases it.
    private(set) var slabsAreWired = false
    private var gatheredRequestOwners: [PagedGatheredRequestOwner] = []
    private var gatheredPendingReservations = 0
    private var attentionRowLayers: [UInt64: Int] = [:]

    private func preflightGatheredRequest(reserved: Bool) throws {
        guard pool.hasAsymmetricLayers, let limits = pool.config.gatheredAttention else { return }
        gatheredRequestOwners.removeAll { $0.row == nil || $0.row!.isReleased }
        guard !reserved || gatheredPendingReservations > 0,
            gatheredRequestOwners.count + (reserved ? 0 : gatheredPendingReservations)
                < limits.maximumBatchSize
        else {
            throw CBv2KVError.backendIneligible(
                reason:
                    "explicit asymmetric gathered-attention maximum live request/batch bound exceeded"
            )
        }
    }
    /// The only writer of `slabsAreWired`; `private(set)` keeps the flag out
    /// of reach of everything except `commitSlabs()`, which lives in the
    /// other file.
    func markSlabsWired() { slabsAreWired = true }

    public init(
        layerKinds: [CBv2LayerKind],
        config: PagedKVPoolConfig,
        slabCommitment: PagedKVSlabCommitment = .atFirstAdmission,
        residentPrefixCache: CBv2PagedPrefixCacheConfig? = nil
    ) throws {
        for (index, kind) in layerKinds.enumerated() {
            guard kind.kvGeometry != nil else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): invalid native K/V geometry")
            }
            if kind.valueHeadDim != kind.headDim {
                guard config.segmentSizeBytes != nil, config.gatheredAttention != nil,
                    kind.headDim <= 512, kind.valueHeadDim <= 512,
                    kind.qwen4IndexerCompressRatio == nil, !kind.isBidirectional,
                    residentPrefixCache == nil, slabCommitment != .atConstruction
                else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index): asymmetric serving requires explicit segmented native-gathered policy; fixed/fused/indexed/bidirectional/resident-prefix/eager paths are not qualified"
                    )
                }
            }
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count,
                    layerKinds[source].sharesKVWithLayer == nil
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) shares KV with invalid layer \(source)")
                }
                let src = layerKinds[source]
                guard src.kvHeads == kind.kvHeads, src.headDim == kind.headDim,
                    src.valueHeadDim == kind.valueHeadDim,
                    src.attention == kind.attention
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) KV-shares with structurally different layer "
                            + "\(source)")
                }
            }
            guard kind.kvHeads > 0, kind.queryHeads > 0, kind.queryHeads % kind.kvHeads == 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): queryHeads \(kind.queryHeads) not a multiple "
                        + "of kvHeads \(kind.kvHeads)")
            }
            // Kernel-level static eligibility (head dim support + the part
            // kernel's threadgroup-memory budget). Checked for EVERY layer
            // that will dispatch paged attention — including KV-shared
            // layers, which borrow storage but launch with their own GQA.
            // One over-budget layer makes the whole model ineligible.
            if kind.valueHeadDim == kind.headDim,
                let reason = PagedAttentionKernel.ineligibilityReason(
                    headDim: kind.headDim, gqa: kind.queryHeads / kind.kvHeads)
            {
                throw CBv2KVError.backendIneligible(reason: "layer \(index): \(reason)")
            }
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): invalid sliding window \(window)")
            }
        }
        self.layerKinds = layerKinds
        self.slabCommitment = slabCommitment
        var effectiveConfig = config
        if let residentPrefixCache {
            if let declared = effectiveConfig.prefixSharingBlockSize,
                declared != residentPrefixCache.blockSize
            {
                throw CBv2KVError.backendIneligible(
                    reason: "pool prefixSharingBlockSize \(declared) does not match resident "
                        + "cache block size \(residentPrefixCache.blockSize)")
            }
            effectiveConfig.prefixSharingBlockSize = residentPrefixCache.blockSize
        }
        self.pool = try PagedKVPool(layerKinds: layerKinds, config: effectiveConfig)
        if let residentPrefixCache {
            self.residentPrefixIndex = try PagedPrefixBlockIndex(
                pool: pool, layerKinds: layerKinds, config: residentPrefixCache)
        } else {
            self.residentPrefixIndex = nil
        }
        if slabCommitment == .atConstruction {
            // An eager commit that cannot fit fails the BUILD (throwing
            // `capacityExhausted`), which is the honest posture for the
            // profiler/single-slot deployments that opt into it.
            if config.segmentSizeBytes != nil {
                try pool.materializeSlabs()
            } else {
                try commitSlabs()
            }
        }
    }

    /// Protected MiMo constructor only. Ordinary public construction remains
    /// byte-for-byte nil-binding policy; no caller can replace the association.
    package convenience init(
        layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig,
        nativeModelBinding: CBv2NativePagedModelBinding
    ) throws {
        try self.init(
            layerKinds: layerKinds, config: config,
            slabCommitment: .atFirstAdmission, residentPrefixCache: nil)
        try nativeModelBinding.attach(self)
        self.nativeModelBinding = nativeModelBinding
        pool.nativeModelBinding = nativeModelBinding
    }

    // MARK: - Admission-time reservation (Codex P2)

    /// Reserve the worst-case page demand for a request of `maxLength` tokens
    /// BEFORE it is admitted, so several same-step admissions cannot
    /// over-commit the pool. Charges the pool up front (reflected in
    /// `bytesReserved`) and throws `capacityExhausted` when it cannot fit —
    /// the admission controller then rejects or queues the request instead of
    /// accepting one that would only fail at `makeSequenceState`.
    ///
    /// A subsequent `makeSequenceState` for the SAME request must be told the
    /// pages are already held (`reserved: true`) so it does not double-charge;
    /// `release`/`makeSequenceState(adopting:)`/finish balance the hold via
    /// the per-row `reservedPages` bookkeeping exactly once. Balance an
    /// admission that never materializes with `unreserve(layerKinds:maxLength:)`.
    public func reserve(layerKinds: [CBv2LayerKind], maxLength: Int) throws {
        try nativeModelBinding?.preflightCreation(
            backend: self, kinds: layerKinds, maximumLength: maxLength)
        precondition(maxLength > 0)
        guard layerKinds == self.layerKinds else {
            throw CBv2KVError.backendIneligible(
                reason: "paged reservation layout differs from its owner")
        }
        try preflightGatheredRequest(reserved: false)
        try pool.prepareGatheredAttention(maximumSequenceLength: maxLength)
        let needs = pageNeeds(layerKinds: layerKinds, maxLength: maxLength)
        try pool.reserve(needs)
        // The charge succeeded, so this pool is no longer idle. Wire the
        // slabs NOW — before any row can reach `ensurePage` — so a deferred
        // commitment never turns an accepted admission into an unbacked
        // page. Deferring the ALLOCATION is the D1 fix; deferring the
        // GUARANTEE would be a daemon abort under load.
        do {
            try commitSlabs()
            if pool.hasAsymmetricLayers { gatheredPendingReservations += 1 }
        } catch {
            // A refused commit must leave the pool exactly as it found it:
            // unwind the page charge so the rejected admission leaves no
            // residue and the retry re-charges from a clean ledger.
            if nativeModelBinding == nil
                || CBv2NativePagedOperation.constructing?.tracking.mayExecute == true
            {
                pool.unreserve(needs)
            }
            throw error
        }
    }

    /// Release an admission-time `reserve` that never reached
    /// `makeSequenceState` (rejected, superseded, or shut down).
    public func unreserve(layerKinds: [CBv2LayerKind], maxLength: Int) {
        pool.unreserve(pageNeeds(layerKinds: layerKinds, maxLength: maxLength))
        if pool.hasAsymmetricLayers {
            gatheredPendingReservations = max(0, gatheredPendingReservations - 1)
        }
    }

    // MARK: - CBv2KVBackend

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try makeSequenceState(
            layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength,
            reserved: false)
    }

    /// `reserved: true` skips the pool reservation because `reserve(...)`
    /// already charged the pages at admission — the per-row `reservedPages`
    /// still governs allocation and `release` still unreserves them, so the
    /// hold is charged and released exactly once.
    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int, reserved: Bool
    ) throws -> [CBv2SequenceKV?] {
        try nativeModelBinding?.preflightCreation(
            backend: self, kinds: layerKinds, maximumLength: maxLength)
        precondition(maxLength >= promptLength && maxLength > 0)
        guard layerKinds == self.layerKinds else {
            throw CBv2KVError.backendIneligible(reason: "paged row layout differs from its owner")
        }
        try preflightGatheredRequest(reserved: reserved)
        try pool.prepareGatheredAttention(maximumSequenceLength: maxLength)
        let needs = pageNeeds(layerKinds: layerKinds, maxLength: maxLength)
        if !reserved {
            try pool.reserve(needs)
        }
        // Same guarantee as `reserve`: every page this row may touch is
        // backed before the row exists. Idempotent and free after the first
        // admission. On a refused commit, unwind exactly the charge THIS
        // call took — a `reserved: true` caller still owns its own hold and
        // balances it with `unreserve` per the admission contract above.
        do {
            try commitSlabs()
        } catch {
            if !reserved,
                nativeModelBinding == nil
                    || CBv2NativePagedOperation.constructing?.tracking.mayExecute == true
            {
                pool.unreserve(needs)
            }
            throw error
        }
        var states: [CBv2SequenceKV?] = []
        states.reserveCapacity(layerKinds.count)
        for (index, kind) in layerKinds.enumerated() {
            if kind.sharesKVWithLayer != nil {
                states.append(nil)
            } else {
                let reserved = PagedKVPool.pageDemand(
                    kind: kind, maxLength: maxLength, config: pool.config)
                states.append(
                    PagedSequenceKV(
                        pool: pool, kind: kind, groupKey: pool.groupKey(forLayer: index),
                        maxLength: maxLength, reservedPages: reserved))
            }
        }
        if let binding = nativeModelBinding {
            // Retain every actual partial row before the throwing ledger edge.
            // Failed native publication must not refund another live promise.
            for row in states.compactMap({ $0 }) {
                CBv2NativePagedOperation.constructing?.retain(owner: row)
            }
            do { try binding.register(states, backend: self) } catch {
                CBv2NativePagedOperation.constructing?.fail()
                throw error
            }
        }
        if pool.hasAsymmetricLayers {
            if reserved { gatheredPendingReservations -= 1 }
            if let row = states.compactMap({ $0 as? PagedSequenceKV }).first {
                gatheredRequestOwners.append(.init(row))
            }
        }
        if pool.usesStepOwnedAttention {
            for (index, state) in states.enumerated() {
                if let row = state as? PagedSequenceKV { attentionRowLayers[row.serial] = index }
            }
        }
        return states
    }

    /// Adopt a donated prefix. Snapshots are written into fresh pages via the
    /// in-place bulk-write kernel.
    ///
    /// Three shapes, one entry point:
    ///
    ///  * `.direct` / `.tailReplay` — owning full rows restored to C, windowed
    ///    rows fast-forwarded to C and left empty, engine replays `[C, M)`.
    ///    Both cursors are C, which is why paged has always served these.
    ///  * `.frozenFullReplay` with R == 0 (`requiresExactWindowRestore`) —
    ///    owning full rows restored to M and EVERY owning windowed row
    ///    restored to M from an admissible `CBv2PagedWindowSnapshot`. Both
    ///    cursors are M and there is nothing to replay, which is the form
    ///    that makes the paged-hybrid dual-cursor problem evaporate rather
    ///    than solving it. (The capability refusal that named that problem
    ///    is gone: `derive` no longer refuses paged hybrids.)
    ///  * `.frozenFullReplay` with R > 0 — owning full rows adopted FROZEN
    ///    through M via `PagedSequenceKV.adoptFrozen`, so their storage is
    ///    exact and immutable while the logical cursor reports C; windowed
    ///    rows fast-forwarded to C and left empty; engine replays `[C, M)`.
    ///    This is the genuine dual cursor, and it is the form the engine
    ///    actually produces today — `PrefixCacheV2` nils every windowed layer,
    ///    so no window payload reaches the restore form yet.
    ///
    /// EVERY refusal is a thrown `backendIneligible`, never a trap and never a
    /// partial install. `EngineLoopV2.applyAdoption` catches it, unreserves the
    /// admission charge and the request cold-prefills — which is the only safe
    /// answer, because a frozen-full hit that restores the full layers but
    /// cannot complete the sliding side leaves windows that are SHORT rather
    /// than absent: attention silently ignores the missing oldest entries and
    /// no later replay can recover them. Validation therefore completes for
    /// every layer BEFORE a single page is reserved.
    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        guard layerKinds == self.layerKinds else {
            throw CBv2KVError.backendIneligible(
                reason: "paged restore layout differs from its owner")
        }
        guard !pool.groupKeys.contains(where: { $0.quantization != nil }) else {
            throw CBv2KVError.backendIneligible(
                reason: "packed KV prefix reuse requires an authenticated complete checkpoint frame"
            )
        }
        try nativeModelBinding?.refuseImport()
        guard !pool.usesStepOwnedAttention else {
            throw CBv2KVError.backendIneligible(
                reason: "step-owned paging prefix adoption requires separate transfer qualification"
            )
        }
        guard plan.backend == .pagedFP16 else {
            throw CBv2KVError.backendIneligible(
                reason: "paged adoption received \(plan.backend.rawValue) prefix plan")
        }
        guard prefix.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix count \(prefix.count) != layer count \(layerKinds.count)")
        }
        guard layerKinds.count == pool.layerDTypes.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix layer count does not match paged pool")
        }
        // Snapshot import is a throwing boundary, unlike model forward.
        // Reject all source dtypes before reserving/mutating any destination;
        // a bad cached frame must not poison another live request's latch.
        try pool.writeValidation.check()
        for (index, entry) in prefix.enumerated() {
            guard let entry else { continue }
            let kind = layerKinds[index]
            guard kind.sharesKVWithLayer == nil, entry.offset > 0,
                entry.keys.ndim == 3 || entry.keys.ndim == 4,
                entry.values.ndim == entry.keys.ndim,
                entry.keys.ndim != 4 || (entry.keys.dim(0) == 1 && entry.values.dim(0) == 1)
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "invalid paged source snapshot owner/rank")
            }
            let k = Self.rowShaped(entry.keys)
            let v = Self.rowShaped(entry.values)
            guard k.dim(0) == kind.kvHeads, v.dim(0) == kind.kvHeads,
                k.dim(2) == kind.headDim, v.dim(2) == kind.valueHeadDim,
                k.dim(1) == v.dim(1), k.dim(1) > 0
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "paged source snapshot role geometry mismatch")
            }
            if case .full = kind.attention, k.dim(1) != entry.offset {
                throw CBv2KVError.backendIneligible(
                    reason: "paged source snapshot does not cover its full boundary")
            }
            let expected = pool.layerDTypes[index]
            guard entry.keys.dtype == expected, entry.values.dtype == expected else {
                throw CBv2PagedKVWriteError(
                    layerIndex: index, expected: expected,
                    keys: entry.keys.dtype, values: entry.values.dtype)
            }
        }
        guard plan.matchedBoundary > 0, plan.matchedBoundary <= maxLength,
            plan.replayStart >= 0, plan.replayStart <= plan.matchedBoundary,
            plan.replayTokens == plan.matchedBoundary - plan.replayStart
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid prefix replay plan")
        }
        if plan.strategy == .frozenFullReplay {
            return try makeFrozenFullState(
                adopting: prefix, plan: plan, layerKinds: layerKinds, maxLength: maxLength)
        }
        var matched = 0
        for (index, entry) in prefix.enumerated() {
            guard let entry else { continue }
            guard layerKinds[index].sharesKVWithLayer == nil,
                case .full = layerKinds[index].attention,
                matched == 0 || matched == entry.offset
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "ordinary paged restore has invalid owner/window/nonuniform boundary")
            }
            matched = entry.offset
        }
        guard matched == plan.restoredFullTokens, matched == plan.replayStart else {
            throw CBv2KVError.backendIneligible(
                reason: "paged prefix offset does not match ordinary replay plan")
        }
        let states = try makeSequenceState(
            layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
        for (index, state) in states.enumerated() {
            guard let state = state as? PagedSequenceKV else { continue }
            if let snapshot = prefix[index] {
                precondition(
                    state.windowSize == nil,
                    "prefix donated to a windowed layer \(index)")
                var keys = snapshot.keys
                var values = snapshot.values
                if keys.ndim == 4 {
                    keys = keys.squeezed(axis: 0)
                    values = values.squeezed(axis: 0)
                }
                precondition(
                    keys.dim(1) == snapshot.offset,
                    "full-attention prefix snapshot must cover [0, offset)")
                state.write(keys: keys, values: values)
            } else if state.windowSize != nil, matched > 0 {
                state.fastForward(to: matched)
            }
        }
        return states
    }

    // MARK: - Frozen-full hybrid adoption (WS-4.1)

    private func makeFrozenFullState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        let matched = plan.matchedBoundary
        guard plan.restoredFullTokens == matched else {
            throw CBv2KVError.backendIneligible(
                reason: "frozen-full plan restores \(plan.restoredFullTokens) tokens but "
                    + "matched \(matched) — full rows must be exact through M")
        }
        let restoringWindows = plan.requiresExactWindowRestore
        if !restoringWindows {
            // The dual-cursor form. Neither condition is visible to the
            // planner, and both are cheap, so they are checked here.
            guard plan.replayStart > 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "frozen replay with C == 0 saves nothing")
            }
            // The SAME bound contiguous uses, from the SAME function — paged
            // briefly kept its own edition (`requiredFrozenReplayTokens`,
            // `cbv2RequiredRecompute` plus one prefill chunk) because a
            // frozen paged row attended its chunk's freshly projected keys.
            // `PagedLayerCache.prefillKVWritingChunk` reads the cached
            // diagonal out of the frozen pages now, so the extra term is gone
            // and two copies of one bound would only drift. Still CHECKED
            // rather than assumed: a plan reaches this function from outside
            // the type. If a frozen paged row is ever made to attend fresh
            // projections again, this bound is short by one chunk and it will
            // accept a replay that leaves the sliding rows silently inexact.
            let required = cbv2RequiredRecompute(
                layerKinds: layerKinds, matched: plan.matchedBoundary)
            guard plan.replayTokens >= required else {
                throw CBv2KVError.backendIneligible(
                    reason: "frozen replay of \(plan.replayTokens) tokens is shorter than the "
                        + "\(required) this layout needs — the sliding rows would come back "
                        + "inexact")
            }
        }

        // --- Validate every layer BEFORE reserving a single page. A frozen
        //     full restore that cannot complete its sliding side must leave
        //     the pool untouched so the cold prefill can have the capacity.
        var fullKV: [Int: (keys: MLXArray, values: MLXArray)] = [:]
        var windows: [Int: CBv2PagedWindowSnapshot] = [:]
        var sawOwningFull = false
        for (index, kind) in layerKinds.enumerated() {
            let entry = prefix[index]
            guard kind.sharesKVWithLayer == nil else {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) is KV-shared but received a prefix snapshot")
                }
                continue
            }
            // An owning full layer always needs its snapshot. A windowed one
            // needs a window under the restore form and must NOT have one
            // under the replay form, which the switch below decides.
            let needsEntry: Bool
            if case .slidingWindow = kind.attention {
                needsEntry = restoringWindows
            } else {
                needsEntry = true
            }
            guard let entry else {
                guard !needsEntry else {
                    throw CBv2KVError.backendIneligible(
                        reason: "frozen-full adoption at boundary \(matched): owning layer "
                            + "\(index) has no donated KV")
                }
                continue
            }
            switch kind.attention {
            case .full:
                guard entry.offset == matched else {
                    throw CBv2KVError.backendIneligible(
                        reason: "full prefix offset \(entry.offset) != matched \(matched) at "
                            + "layer \(index)")
                }
                let keys = Self.rowShaped(entry.keys)
                let values = Self.rowShaped(entry.values)
                guard keys.ndim == 3, values.ndim == 3,
                    keys.dim(0) == kind.kvHeads, values.dim(0) == kind.kvHeads,
                    keys.dim(2) == kind.headDim, values.dim(2) == kind.valueHeadDim,
                    keys.dim(1) == matched, values.dim(1) == matched
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "full prefix snapshot at layer \(index) does not exactly cover "
                            + "[0, \(matched)) at this layer's geometry")
                }
                fullKV[index] = (keys, values)
                sawOwningFull = true
            case .slidingWindow(let window):
                guard restoringWindows else {
                    // The replay form recomputes windowed layers from C. A
                    // payload here means the donor and the plan disagree
                    // about what is being adopted; installing it would put
                    // the row at M while every other row sits at C.
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) is windowed but received a prefix snapshot "
                            + "under a replay plan (windowed layers are recomputed)")
                }
                guard entry.keys.ndim == 4,
                    let snapshot = CBv2PagedWindowSnapshot(
                        keys: entry.keys, values: entry.values,
                        base: entry.offset - entry.keys.dim(2), valueHeadDim: kind.valueHeadDim)
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "windowed prefix at layer \(index) is not a well-formed "
                            + "window snapshot")
                }
                // The seam's rule, not a local reimplementation of it: the
                // payload may be installed at exactly one absolute boundary,
                // and a short window is refused rather than partially placed.
                do {
                    try snapshot.requireAdmissible(at: matched, window: window)
                } catch let refusal as CBv2PagedWindowRestoreRefusal {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): \(refusal.description)")
                }
                guard snapshot.keys.dim(1) == kind.kvHeads,
                    snapshot.keys.dim(3) == kind.headDim,
                    snapshot.values.dim(1) == kind.kvHeads,
                    snapshot.values.dim(3) == kind.valueHeadDim
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "windowed prefix at layer \(index) has the wrong KV geometry")
                }
                windows[index] = snapshot
            }
        }
        guard sawOwningFull else {
            throw CBv2KVError.backendIneligible(
                reason: "frozen-full adoption requires at least one storage-owning full layer")
        }

        // --- Install. Everything above validated, so nothing here throws and
        //     no half-built state can escape.
        //
        //     Both forms leave EVERY row on the same logical cursor, which is
        //     what keeps RoPE offsets, masks and `retainedCount` uniform
        //     across layers: M for the restore form, C for the replay form.
        let states = try makeSequenceState(
            layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
        for (index, state) in states.enumerated() {
            guard let row = state as? PagedSequenceKV else { continue }
            if let full = fullKV[index] {
                if restoringWindows {
                    row.write(keys: full.keys, values: full.values)
                } else {
                    // Storage [0, M) is exact and immutable; the cursor
                    // reports C so the replay of [C, M) advances it without
                    // overwriting a byte.
                    row.adoptFrozen(
                        keys: full.keys, values: full.values, replayStart: plan.replayStart)
                }
            } else if let window = windows[index] {
                installWindow(window, into: row, at: matched)
            } else if plan.replayStart > 0 {
                // Windowed row on the replay form: empty, at C, and its
                // `baseOffset` set so no query reaches behind the replay.
                row.fastForward(to: plan.replayStart)
            }
        }
        return states
    }

    /// Place a validated window at its one admissible base. A BYTE WRITE:
    /// the payload is copied into the row's own pages. Nothing is shared and
    /// no page refcount is adopted.
    ///
    /// `fastForward` is what makes the absolute positions right: it sets both
    /// `absoluteOffset` and `baseOffset` to `snapshot.base`, so the tokens
    /// land in the ring slots `(p / pageSize) % ringPages` a cold row would
    /// have used for the same absolute positions, and every later gather and
    /// decode window resolves against the true positions rather than against
    /// a payload that merely starts somewhere.
    ///
    /// The write is SPLIT: `PagedSequenceKV.write` refuses a windowed run
    /// longer than `maxPrefillChunk` because the ring cannot hold one, and a
    /// full window is longer than a chunk by construction (gemma-4: 1,024
    /// against 512).
    ///
    /// Admissibility is re-asserted here rather than trusted from the
    /// caller. `CBv2PagedWindowSnapshot.requireAdmissible` is what stops a
    /// donation taken at position 4,096 being written into an adopter's
    /// `[0, 1024)` — silent wrong answers, no trap, no telemetry — and the
    /// validating call is in `makeFrozenFullState`, twenty lines away and
    /// separated by an install loop. A `precondition` rather than a throw:
    /// the install phase is deliberately non-throwing so no half-built state
    /// can escape, and by this point the refusal path has already run, so
    /// this can only fire on a programming error.
    private func installWindow(
        _ snapshot: CBv2PagedWindowSnapshot, into row: PagedSequenceKV,
        at matchedBoundary: Int
    ) {
        do {
            try snapshot.requireAdmissible(at: matchedBoundary, window: row.windowSize)
        } catch {
            preconditionFailure(
                "[PagedKVBackend] installWindow reached with an inadmissible snapshot "
                    + "at boundary \(matchedBoundary): \(error) — validation must run in "
                    + "makeFrozenFullState before the install phase")
        }
        let keys = snapshot.keys.squeezed(axis: 0)
        let values = snapshot.values.squeezed(axis: 0)
        row.fastForward(to: snapshot.base)
        var written = 0
        while written < snapshot.tokens {
            let count = min(pool.config.maxPrefillChunk, snapshot.tokens - written)
            let range = written ..< (written + count)
            row.write(
                keys: keys[.ellipsis, range, 0...],
                values: values[.ellipsis, range, 0...])
            written += count
        }
    }

    /// `[1, kvHeads, tokens, headDim]` (a donated snapshot) or
    /// `[kvHeads, tokens, headDim]` (already row-shaped) -> row shape.
    private static func rowShaped(_ array: MLXArray) -> MLXArray {
        array.ndim == 4 && array.dim(0) == 1 ? array.squeezed(axis: 0) : array
    }

    /// A recoverable all-row ownership check for package-native callers/tests.
    /// A refusal has not removed any ledger entry or freed any pool page.
    package func releaseNativeValidated(_ state: [CBv2SequenceKV?]) throws {
        guard let nativeModelBinding else {
            throw CBv2KVError.backendIneligible(reason: "no issued native paged binding")
        }
        try nativeModelBinding.remove(state, backend: self)
        releaseRowsAfterValidation(state)
    }

    /// Fresh page-prefix rows are registered through the SAME backend and
    /// MiMo cohort ledger as cold rows, before atomic pool publication.
    /// The private preparation proves its actual required completion.
    func registerPreparedNativeCheckpoint(
        _ state: [CBv2SequenceKV?],
        preparation: CBv2PreparedNativePagedCheckpoint
    ) throws {
        guard let nativeModelBinding, preparation.authorizesRegistration(state, backend: self),
            state.count == layerKinds.count,
            state.enumerated().allSatisfy({ index, item in
                guard let row = item as? PagedSequenceKV else { return false }
                return row.pool === pool && row.table.isEmpty && row.absoluteOffset == 0
                    && !row.isReleased && attentionRowLayers[row.serial] == nil
                    && row.groupKey == pool.groupKey(forLayer: index)
            })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        try preflightGatheredRequest(reserved: false)
        try nativeModelBinding.register(state, backend: self)
        if let first = state.first.flatMap({ $0 as? PagedSequenceKV }) {
            gatheredRequestOwners.append(.init(first))
        }
        for (index, item) in state.enumerated() {
            attentionRowLayers[(item as! PagedSequenceKV).serial] = index
        }
    }

    /// Refusal before publication owns no pool pages. Remove only that exact
    /// fresh cohort; do not call releaseStorage/unreserve for uninstalled rows.
    func rollbackPreparedNativeCheckpointRegistration(
        _ state: [CBv2SequenceKV?],
        preparation: CBv2PreparedNativePagedCheckpoint
    ) throws {
        guard let nativeModelBinding, preparation.authorizesRegistration(state, backend: self),
            state.allSatisfy({ ($0 as? PagedSequenceKV)?.table.isEmpty == true })
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try nativeModelBinding.remove(state, backend: self)
        let serials = Set(state.compactMap { ($0 as? PagedSequenceKV)?.serial })
        for serial in serials { attentionRowLayers.removeValue(forKey: serial) }
        gatheredRequestOwners.removeAll { $0.row.map { serials.contains($0.serial) } ?? true }
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        if nativeModelBinding != nil {
            do { try releaseNativeValidated(state) } catch {
                preconditionFailure("native MiMo paged owned release invariant: \(error)")
            }
            return
        }
        releaseRowsAfterValidation(state)
    }

    private func releaseRowsAfterValidation(_ state: [CBv2SequenceKV?]) {
        for entry in state {
            guard let entry else { continue }
            guard let paged = entry as? PagedSequenceKV else {
                fatalError("[PagedKVBackend] release of a foreign sequence state")
            }
            paged.releaseStorage()
            attentionRowLayers.removeValue(forKey: paged.serial)
        }
    }

    /// Only the grant changes off queue. Live page/range ownership survives
    /// a shrink and new growth must revalidate the current grant epoch.
    public func updateBytesCapacity(_ bytes: Int) { pool.segmentGrant?.update(bytes: bytes) }

    public var bytesInUse: Int { pool.bytesInUse }
    public var bytesCapacity: Int { pool.bytesCapacity }
    /// Admission-relevant bytes (worst-case reservations of live requests).
    public var bytesReserved: Int { pool.bytesReserved }
    /// Current physical grant. A shrink preserves live segmented owners;
    /// bytesWired may temporarily exceed this ceiling until they release.
    public var bytesPhysical: Int { pool.bytesPhysical }
    /// Fixed-reference snapshots can alias recyclable slab storage. Require
    /// materialized donations conservatively for both layouts; segmented
    /// gathers already return a separate exact native destination.
    public var requiresMaterializedSnapshots: Bool { true }

    // MARK: - Helpers

    func pageNeeds(layerKinds: [CBv2LayerKind], maxLength: Int) -> [PagedKVGroupKey: Int] {
        var needs: [PagedKVGroupKey: Int] = [:]
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let pages = PagedKVPool.pageDemand(
                kind: kind, maxLength: maxLength, config: pool.config)
            needs[pool.groupKey(forLayer: index), default: 0] += pages
        }
        return needs
    }

    /// One layer cache per model layer (KV-shared layers get a borrowing
    /// cache with no rows). `attentionSoftcap` comes from model config —
    /// it is not part of the contract's per-call surface.
    public func makeLayerCaches(attentionSoftcap: Float? = nil) -> [PagedLayerCache] {
        layerKinds.enumerated().map { index, kind in
            PagedLayerCache(
                layerIndex: index, kind: kind, pool: pool,
                attentionSoftcap: attentionSoftcap)
        }
    }

    /// Scalar planning and real Admission acquisition precede every forward.
    /// Row serials are pool-issued routing identities, never allocation IDs.
    func prepareAttentionWork(
        assignments: [(id: CBv2RequestID, range: Range<Int>)],
        states: [CBv2RequestID: [CBv2SequenceKV?]]
    ) throws -> CBv2PagedAttentionStepOwner? {
        guard pool.usesStepOwnedAttention else { return nil }
        guard pool.attentionWorkEnginePrepared, pool.attentionWorkEngineRefusal == nil,
            pool.activeAttentionWork == nil, !assignments.isEmpty,
            let limits = pool.config.gatheredAttention,
            let admission = pool.memoryAdmission, admission.hasProcessMemoryOwner,
            let policy = Memory.allocationFootprintPolicy(),
            StreamOrDevice.default.stream == Stream.gpu,
            assignments.count <= limits.maximumBatchSize,
            pool.attentionWorkOwners.values.filter({ !$0.completed }).count
                < limits.maximumInFlightGraphs,
            Set(assignments.map(\.id)).count == assignments.count
        else {
            throw CBv2KVError.backendIneligible(
                reason: pool.attentionWorkEngineRefusal
                    ?? "step-owned paging has no valid engine/process/stream preparation")
        }
        let environment = CBv2PagedAttentionWorkEnvironment.capture()
        try environment.requireSupported()
        var descriptors: [CBv2PagedAttentionTicketKey: CBv2PagedAttentionRowDescriptor] = [:]
        var allocations: [CBv2PagedAttentionAllocation] = []
        var rowRequests: [UInt64: CBv2RequestID] = [:]
        var possibleBatchBuffers: [String: Int] = [:]
        for assignment in assignments {
            guard let rows = states[assignment.id], rows.count == layerKinds.count,
                !assignment.range.isEmpty, assignment.range.lowerBound >= 0,
                assignment.range.upperBound <= limits.maximumContextTokens,
                assignment.range.count <= limits.maximumQueryTokens
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "invalid actual paged attention work range")
            }
            for (index, kind) in layerKinds.enumerated() {
                let owner = kind.sharesKVWithLayer ?? index
                guard kind.headDim != kind.valueHeadDim,
                    let row = rows[owner] as? PagedSequenceKV,
                    row.pool === pool, !row.isReleased,
                    row.speculativeBase == nil
                        || CBv2NativePagedMTPWork.current?
                            .permitsPlannedColumn(row, range: assignment.range) == true,
                    attentionRowLayers[row.serial] == owner,
                    row.absoluteOffset == assignment.range.lowerBound,
                    row.maxLength >= assignment.range.upperBound,
                    row.windowSize == nil || assignment.range.count <= pool.config.maxPrefillChunk,
                    row.groupKey == pool.groupKey(forLayer: owner),
                    let cache = pool.attentionWorkCaches[index]?.value,
                    cache.pool === pool, cache.kind == kind,
                    row.frozenHighWater <= assignment.range.lowerBound
                        || assignment.range.upperBound <= row.frozenHighWater
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "paged work row/cache identity, generation or frozen range mismatch"
                    )
                }
                if let previous = rowRequests[row.serial], previous != assignment.id {
                    throw CBv2KVError.backendIneligible(
                        reason: "paged work aliases one row across requests")
                }
                rowRequests[row.serial] = assignment.id
                let q = assignment.range.count
                let start = assignment.range.lowerBound
                let frozen = row.frozenHighWater > start
                let keyStart: Int
                if q == 1 {
                    keyStart =
                        row.windowSize.map { max(row.baseOffset, assignment.range.upperBound - $0) }
                        ?? row.baseOffset
                } else {
                    keyStart =
                        row.windowSize.map { max(row.baseOffset, start - $0 + 1) }
                        ?? row.baseOffset
                }
                let keyCount = assignment.range.upperBound - keyStart
                let gather: Range<Int>?
                if q == 1 || (owner == index && frozen) {
                    gather = keyStart ..< assignment.range.upperBound
                } else if owner == index && keyStart < start {
                    gather = keyStart ..< start
                } else {
                    gather = nil
                }
                var blocks: [CBv2PagedAttentionBlock] = []
                if q > 1 && environment.queryBlockSize > 0 && q > environment.queryBlockSize {
                    var offset = 0
                    while offset < q {
                        let count = min(environment.queryBlockSize, q - offset)
                        let bounds = CBv2AttentionV1.queryBlockBounds(
                            historyCount: start - keyStart, offset: offset, count: count,
                            window: row.windowSize)
                        blocks.append(
                            .init(
                                queryOffset: offset, queryCount: count,
                                visibleStart: bounds.visibleStart, visibleEnd: bounds.visibleEnd))
                        offset += count
                    }
                } else {
                    blocks = [
                        .init(queryOffset: 0, queryCount: q, visibleStart: 0, visibleEnd: keyCount)
                    ]
                }
                let descriptor = CBv2PagedAttentionRowDescriptor(
                    requestID: assignment.id, row: .init(row), rowSerial: row.serial,
                    layerIndex: index, ownerLayerIndex: owner, range: assignment.range,
                    baseOffset: row.baseOffset, frozenHighWater: row.frozenHighWater,
                    gatheredRange: gather, keyStart: keyStart, keyCount: keyCount,
                    blocks: blocks, softcap: cache.attentionSoftcap != nil)
                descriptors[.init(layer: index, serial: row.serial)] = descriptor
                let rowAllocations = try CBv2PagedAttentionWorkProjection.allocations(
                    kind: kind, dtype: pool.layerDTypes[index], descriptor: descriptor,
                    pageSize: pool.config.pageSize, policy: policy)
                for allocation in rowAllocations {
                    guard allocation.maximumLogicalBufferBytes <= pool.config.maxBufferLength else {
                        throw CBv2KVError.backendIneligible(
                            reason:
                                "actual attention workspace exceeds native maximum buffer length")
                    }
                    if allocation.role == "output.batchConcat"
                        || allocation.role.hasPrefix("position.offset")
                    {
                        let key = "\(index):\(allocation.role)"
                        possibleBatchBuffers[key] = try CBv2PagedWorkMath.add(
                            possibleBatchBuffers[key] ?? 0, allocation.logicalBytes)
                        guard possibleBatchBuffers[key]! <= pool.config.maxBufferLength else {
                            throw CBv2KVError.backendIneligible(
                                reason:
                                    "actual packed workspace exceeds native maximum buffer length")
                        }
                    }
                }
                allocations += rowAllocations
            }
        }
        let bytes = try allocations.reduce(64 << 10) {
            try CBv2PagedWorkMath.add($0, $1.upperBoundBytes)
        }
        let coexistence = try CBv2PagedWorkMath.add(pool.attentionWorkBytesReserved, bytes)
        guard coexistence <= limits.maximumScratchBytes else {
            throw CBv2KVError.capacityExhausted(
                needed: bytes,
                available: max(0, limits.maximumScratchBytes - pool.attentionWorkBytesReserved))
        }
        let (generation, overflow) = pool.attentionWorkGeneration.addingReportingOverflow(1)
        guard !overflow else {
            throw CBv2KVError.backendIneligible(reason: "paged work generation exhausted")
        }
        let nativeOperation = try nativeModelBinding?.beginWork(
            requests: Set(assignments.map(\.id)))
        do {
            let reservation = try admission.reserveTransient(bytes: bytes)
            nativeOperation?.retain(owner: reservation)  // before any late veto
            let work = CBv2PagedAttentionStepOwner(
                generation: generation, pool: pool,
                environment: environment, descriptors: descriptors, allocations: allocations,
                bytes: bytes, reservation: reservation, allocationPolicy: policy,
                nativeOperation: nativeOperation)
            nativeOperation?.retain(owner: work)
            try nativeOperation?.requireWork()
            pool.attentionWorkGeneration = generation
            pool.attentionWorkOwners[generation] = work
            pool.publishAttentionWorkCharge()
            pool.activeAttentionWork = work
            return work
        } catch {
            if let nativeOperation {
                if nativeOperation.tracking.mayExecute {
                    // Only a reserveTransient refusal can reach this healthy
                    // catch before a work/reservation owner was installed.
                    nativeOperation.finish(unstarted: true)
                } else {
                    nativeOperation.fail()
                }
            }
            throw error
        }
    }
}
