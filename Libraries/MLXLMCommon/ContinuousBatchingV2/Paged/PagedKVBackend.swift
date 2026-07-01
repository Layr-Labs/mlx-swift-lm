// PagedKVBackend.swift
//
// `CBv2KVBackend` factory over a `PagedKVPool` (WS-C). Swappable with the
// WS-A contiguous backend behind the same contract: the scheduler and
// models never see the difference.
//
// Eligibility is validated at construction (engine build time), per the
// contract: unsupported head dims, quant schemes, or malformed KV-sharing
// throw `CBv2KVError.backendIneligible` before any request is admitted.
// Attention sinks ARE supported (they are a kernel parameter here).
//
// Admission model: `makeSequenceState` reserves the worst-case page count
// for the request's `maxLength` and throws `capacityExhausted` when the
// pool cannot honor it. Physical pages materialize lazily as tokens are
// written (`bytesInUse` stays truthful); `CBv2SequenceKV.update` therefore
// never fails mid-decode. See PagedKVPool.swift for the rationale.

import Foundation
import MLX

public final class PagedKVBackend: CBv2KVBackend {
    public let pool: PagedKVPool
    /// The model's per-layer structure this backend was built for.
    public let layerKinds: [CBv2LayerKind]

    public init(layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig) throws {
        for (index, kind) in layerKinds.enumerated() {
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count,
                    layerKinds[source].sharesKVWithLayer == nil
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) shares KV with invalid layer \(source)")
                }
                let src = layerKinds[source]
                guard src.kvHeads == kind.kvHeads, src.headDim == kind.headDim,
                    src.attention == kind.attention
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) KV-shares with structurally different layer "
                            + "\(source)")
                }
                continue
            }
            guard PagedAttentionKernel.supportedHeadDims.contains(kind.headDim) else {
                throw CBv2KVError.backendIneligible(
                    reason: "paged kernel does not support headDim \(kind.headDim) "
                        + "(layer \(index)); supported: "
                        + "\(PagedAttentionKernel.supportedHeadDims.sorted())")
            }
            guard kind.kvHeads > 0, kind.queryHeads % kind.kvHeads == 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): queryHeads \(kind.queryHeads) not a multiple "
                        + "of kvHeads \(kind.kvHeads)")
            }
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): invalid sliding window \(window)")
            }
        }
        self.layerKinds = layerKinds
        self.pool = try PagedKVPool(layerKinds: layerKinds, config: config)
    }

    // MARK: - CBv2KVBackend

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        precondition(maxLength >= promptLength && maxLength > 0)
        let needs = pageNeeds(layerKinds: layerKinds, maxLength: maxLength)
        try pool.reserve(needs)
        var states: [CBv2SequenceKV?] = []
        states.reserveCapacity(layerKinds.count)
        for kind in layerKinds {
            if kind.sharesKVWithLayer != nil {
                states.append(nil)
            } else {
                let reserved = PagedKVPool.pageDemand(
                    kind: kind, maxLength: maxLength, config: pool.config)
                states.append(
                    PagedSequenceKV(
                        pool: pool, kind: kind, maxLength: maxLength, reservedPages: reserved))
            }
        }
        return states
    }

    /// Adopt a donated prefix. Snapshots are written into fresh pages
    /// (bulk page-run slice updates — off the hot decode path); windowed
    /// layers receive fresh EMPTY states, per the contract ("windowed
    /// layers are recomputed by the scheduler"). The scheduler can use
    /// `PagedSequenceKV.fastForward(to:)` to place the recompute at true
    /// absolute positions once the replay policy is fixed at integration
    /// (see docs/engine-v2/CONTRACT-ISSUES-C-paged-backend.md).
    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        precondition(prefix.count == layerKinds.count, "prefix/layer count mismatch")
        let states = try makeSequenceState(
            layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
        for (index, snapshot) in prefix.enumerated() {
            guard let snapshot else { continue }
            guard let state = states[index] as? PagedSequenceKV else {
                preconditionFailure("prefix donated to a KV-shared layer \(index)")
            }
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
        }
        return states
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        for entry in state {
            guard let entry else { continue }
            guard let paged = entry as? PagedSequenceKV else {
                fatalError("[PagedKVBackend] release of a foreign sequence state")
            }
            paged.releaseStorage()
        }
    }

    public var bytesInUse: Int { pool.bytesInUse }
    public var bytesCapacity: Int { pool.bytesCapacity }
    /// Admission-relevant bytes (worst-case reservations of live requests).
    public var bytesReserved: Int { pool.bytesReserved }

    // MARK: - Helpers

    func pageNeeds(layerKinds: [CBv2LayerKind], maxLength: Int) -> [PagedKVGroupKey: Int] {
        var needs: [PagedKVGroupKey: Int] = [:]
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            let pages = PagedKVPool.pageDemand(
                kind: kind, maxLength: maxLength, config: pool.config)
            needs[PagedKVGroupKey(kind), default: 0] += pages
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
}
