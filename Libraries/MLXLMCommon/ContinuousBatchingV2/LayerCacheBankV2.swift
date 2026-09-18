// LayerCacheBankV2.swift
//
// The engine's `CBv2LayerCacheProvider` over PERSISTENT per-layer
// attending caches (WS-A `CBv2LayerCache` or WS-C `PagedLayerCache`).
//
// The loop asks for layer caches every step (including every chained
// decode step). Rebinding rows costs a host-integer positionOffsets
// rebuild on the contiguous backend, so the bank fingerprints the batch
// composition by row-object identity and calls the contract's canonical
// `setRows(_:)` ONLY when composition actually changed — the chained
// pure-decode hot path performs zero host rebuilds (the caches advance
// their offsets on-device), preserving WS-A's membership-change-only
// discipline (`CBv2CoreInstrumentation.positionOffsetsHostRebuilds`).
//
// Identity safety: the previous batch's row objects stay retained by the
// caches' `rows` arrays until the next `setRows`, so an ObjectIdentifier
// can never alias a deallocated row.
//
// Engine-thread-confined (no locking) — the loop is the only caller.

import Foundation

/// A layer-cache provider whose composition fingerprint can be forced
/// stale. The engine loop invalidates when rows advanced OUTSIDE the
/// provider's caches, so the next eager bind rebuilds `positionOffsets`
/// from host truth instead of trusting its own stale on-device advance
/// chain.
///
/// The invalidating caller used to be compiled decode. That path is gone;
/// the sole writer of `EngineLoopV2.eagerCompositionStale` is now the MTP
/// round finalizer (`MTP/EngineLoopV2+MTPFinalize.swift`), which rolls the
/// on-device offsets back after a rejecting round — the same hazard, a
/// different producer.
public protocol CBv2CompositionInvalidating: AnyObject {
    func invalidateBoundComposition()

    /// Drop the provider's strong row bindings entirely (each cache's
    /// `rows` array). The loop calls this when the eager caches will not be
    /// rebound for an unbounded time — the engine went idle, or compiled
    /// decode took over the step stream — so retired rows' KV buffers are
    /// not pinned by a stale binding until some future eager rebind
    /// (PR#62 review). Implies `invalidateBoundComposition()`; must be a
    /// cheap no-op when nothing is bound.
    func releaseBoundRows()
}

/// Affirmative capability: this cache keeps every bound row's attention
/// independent under a rectangular `[B > 1, L > 1]` prompt pass, so a packed
/// row is bit-identical to that row run alone.
///
/// This is a CLAIM a cache makes, not a type the bank recognises. The gate
/// used to be `cache is CBv2LayerCache`, which made every backend that is
/// not the contiguous cache second-class BY CONSTRUCTION: a new cache could
/// satisfy the requirement perfectly and still be refused, and the only
/// remedy was to edit the bank. Fail-safe is unchanged — a cache that does
/// not conform makes no claim, and one silent cache closes the gate for the
/// whole bank.
public protocol CBv2PackedPrefillCapableCache: AnyObject {
    var keepsRowsIndependentWhenPacked: Bool { get }
}

/// Affirmative capability: this cache's attention actually APPLIES the span
/// context bound through `CBv2SpanMaskBinding`.
///
/// It REFINES the binding protocol because a cache cannot honour a context
/// it has no way to receive — and conformance to `CBv2SpanMaskBinding` alone
/// was too weak a gate: it proves only that the setter exists, never that
/// the bound spans reach the mask. A cache with a no-op `bindSpanContext`
/// passed that check and would have served vision chunks under plain causal
/// masks.
public protocol CBv2MultimodalSpanCapableCache: CBv2SpanMaskBinding {
    var honorsSpanMaskContexts: Bool { get }
}

extension CBv2LayerCache: CBv2PackedPrefillCapableCache, CBv2MultimodalSpanCapableCache {
    /// `updateAndAttend` dispatches per row against that row's own KV, so
    /// batchmates cannot influence each other's attention.
    public var keepsRowsIndependentWhenPacked: Bool { true }

    /// The bound context feeds this cache's chunk mask builder directly
    /// (`CBv2LayerCache.boundSpanContext`).
    public var honorsSpanMaskContexts: Bool { true }
}

/// A KV-owning cache that must retain per-chunk attention state for the
/// KV-shared siblings that borrow from it.
///
/// `CBv2LayerKind.sharesKVWithLayer` points from the BORROWER to its source,
/// so a source cannot see its own consumers; only something holding the
/// whole bank can. The paged cache defaults to retaining (correct standalone)
/// and the bank switches it OFF for every layer nothing borrows — which is
/// every layer of both supported models, so the retention costs nothing
/// unless a model actually shares KV.
public protocol CBv2KVSourceChunkRetaining: AnyObject {
    func setRetainsChunkForBorrowers(_ retains: Bool)
}

protocol CBv2PositionBindingCoordinator: AnyObject {
    var isActive: Bool { get }
    func accepts(_ caches: [any CBv2AttendingLayerCache]) -> Bool
    func finishBinding()
}

protocol CBv2CoordinatedPositionBinding: AnyObject {
    var positionBindingCoordinator: (any CBv2PositionBindingCoordinator)? { get }
    func setRowsForPositionBinding(_ rows: [CBv2SequenceKV])
}

public final class CBv2LayerCacheBank: CBv2LayerCacheProvider, CBv2CompositionInvalidating {

    private let caches: [any CBv2AttendingLayerCache]
    private var boundRowIdentity: [ObjectIdentifier] = []
    private var hasBound = false
    private let positionCoordinator: (any CBv2PositionBindingCoordinator)?

    /// Wrap pre-built caches — e.g. `model.newCacheV2 { ... }` output (the
    /// GPT-OSS path, which also primes sink activation at build time) or
    /// `PagedKVBackend.makeLayerCaches(attentionSoftcap:)`.
    public init(caches: [any CBv2AttendingLayerCache]) {
        self.caches = caches
        let proposed = caches.first.flatMap { ($0 as? any CBv2CoordinatedPositionBinding)?.positionBindingCoordinator }
        positionCoordinator = proposed?.accepts(caches) == true ? proposed : nil
        var borrowedSources = Set<Int>()
        for cache in caches {
            guard let source = cache.kind.sharesKVWithLayer else { continue }
            borrowedSources.insert(source)
        }
        for cache in caches {
            guard let retaining = cache as? CBv2KVSourceChunkRetaining else { continue }
            retaining.setRetainsChunkForBorrowers(borrowedSources.contains(cache.layerIndex))
        }
    }

    /// Contiguous-backend convenience: one `CBv2LayerCache` per layer kind
    /// (KV-shared layers get a rowless borrowing cache), with the
    /// construction-time attention softcap threaded identically to the
    /// paged backend.
    public convenience init(layerKinds: [CBv2LayerKind], attentionSoftcap: Float? = nil) {
        self.init(
            caches: layerKinds.enumerated().map { index, kind in
                CBv2LayerCache(
                    layerIndex: index, kind: kind, attentionSoftcap: attentionSoftcap)
            })
    }

    /// Force the next `layerCaches` call to rebind rows even when the
    /// composition is identity-identical — required after compiled decode
    /// steps advanced the rows outside these caches (their cached
    /// `positionOffsets` no longer reflect the rows' true positions).
    public func invalidateBoundComposition() {
        hasBound = false
        boundRowIdentity = []
    }

    /// Unbind every storage-owning cache (`setRows([])`) and reset the
    /// fingerprint. Without this, the previous eager batch's row objects —
    /// including RETIRED rows whose backend accounting was already released
    /// — stay strongly retained by the caches' `rows` arrays for as long as
    /// the engine is idle or serving compiled-only steps (PR#62 review).
    /// No-op when nothing is bound, so idle-loop callers pay nothing.
    public func releaseBoundRows() {
        guard hasBound else { return }
        for cache in caches where cache.kind.sharesKVWithLayer == nil {
            bind(cache, rows: [])
        }
        positionCoordinator?.finishBinding()
        hasBound = false
        boundRowIdentity = []
    }

    /// Vision prefill eligibility: every cache must AFFIRM that it applies a
    /// bound span context (`CBv2MultimodalSpanCapableCache`), or a vision
    /// chunk's bidirectional-span mask could silently not apply on some
    /// layer. All-or-nothing, checked at submit.
    public var supportsMultimodalSpans: Bool {
        caches.allSatisfy {
            ($0 as? CBv2MultimodalSpanCapableCache)?.honorsSpanMaskContexts ?? false
        }
    }

    /// Rectangular packed prefill eligibility: every cache must AFFIRM that
    /// its rows stay independent under a `[B > 1, L > 1]` pass
    /// (`CBv2PackedPrefillCapableCache`), so a packed row is bit-identical
    /// to running alone. A cache that makes no claim keeps prompt chunks
    /// per-request.
    public var supportsPackedPrefill: Bool {
        caches.allSatisfy {
            ($0 as? CBv2PackedPrefillCapableCache)?.keepsRowsIndependentWhenPacked ?? false
        }
    }

    /// Packed vision rows additionally require one independently bound
    /// optional span context per row on every owning and borrowing layer.
    public var supportsPackedMultimodalSpans: Bool {
        caches.allSatisfy { $0 is CBv2PackedSpanMaskBinding }
    }

    public var supportsMTPRectangularVerification: Bool {
        caches.allSatisfy { $0 is any CBv2MTPRectangularSerializing }
    }

    /// Preserve the existing first-storage-owner identity contract without
    /// allocating every non-nil entry of the row.
    private static func rowIdentity(_ row: [CBv2SequenceKV?]) -> ObjectIdentifier {
        for case let anchor? in row { return ObjectIdentifier(anchor) }
        preconditionFailure("CBv2LayerCacheBank: row owns no storage at any layer")
    }

    public func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        // Ported from the final Gemma MLXFast challenge: unchanged membership
        // needs no new fingerprint array. Invalidation and release still force
        // the original binding path, for both contiguous and paged caches.
        if hasBound && rowStates.count == boundRowIdentity.count {
            var unchanged = true
            for row in rowStates.indices {
                if Self.rowIdentity(rowStates[row]) != boundRowIdentity[row] {
                    unchanged = false
                    break
                }
            }
            if unchanged { return caches }
        }
        let identity = rowStates.map(Self.rowIdentity)
        if !hasBound || identity != boundRowIdentity {
            for (layer, cache) in caches.enumerated() {
                guard cache.kind.sharesKVWithLayer == nil else { continue }
                bind(cache, rows: rowStates.map { states in
                        guard let state = states[layer] else {
                            preconditionFailure(
                                "CBv2LayerCacheBank: missing sequence state for layer \(layer)")
                        }
                        return state
                    })
            }
            positionCoordinator?.finishBinding()
            boundRowIdentity = identity
            hasBound = true
        }
        return caches
    }

    private func bind(_ cache: any CBv2AttendingLayerCache, rows: [CBv2SequenceKV]) {
        if positionCoordinator?.isActive == true,
            let coordinated = cache as? any CBv2CoordinatedPositionBinding {
            coordinated.setRowsForPositionBinding(rows)
        } else {
            cache.setRows(rows)
        }
    }
}
