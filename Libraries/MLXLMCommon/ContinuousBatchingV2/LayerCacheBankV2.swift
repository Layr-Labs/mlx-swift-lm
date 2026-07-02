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
/// stale. The engine loop invalidates after compiled decode steps advanced
/// rows OUTSIDE the provider's caches, so the next eager bind rebuilds
/// `positionOffsets` from host truth instead of trusting its own stale
/// on-device advance chain.
public protocol CBv2CompositionInvalidating: AnyObject {
    func invalidateBoundComposition()
}

public final class CBv2LayerCacheBank: CBv2LayerCacheProvider, CBv2CompositionInvalidating {

    private let caches: [any CBv2AttendingLayerCache]
    private var boundRowIdentity: [ObjectIdentifier] = []
    private var hasBound = false

    /// Wrap pre-built caches — e.g. `model.newCacheV2 { ... }` output (the
    /// GPT-OSS path, which also primes sink activation at build time) or
    /// `PagedKVBackend.makeLayerCaches(attentionSoftcap:)`.
    public init(caches: [any CBv2AttendingLayerCache]) {
        self.caches = caches
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

    public func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        let identity = rowStates.map { row -> ObjectIdentifier in
            guard let anchor = row.compactMap({ $0 }).first else {
                preconditionFailure("CBv2LayerCacheBank: row owns no storage at any layer")
            }
            return ObjectIdentifier(anchor)
        }
        if !hasBound || identity != boundRowIdentity {
            for (layer, cache) in caches.enumerated() {
                guard cache.kind.sharesKVWithLayer == nil else { continue }
                cache.setRows(
                    rowStates.map { states in
                        guard let state = states[layer] else {
                            preconditionFailure(
                                "CBv2LayerCacheBank: missing sequence state for layer \(layer)")
                        }
                        return state
                    })
            }
            boundRowIdentity = identity
            hasBound = true
        }
        return caches
    }
}
