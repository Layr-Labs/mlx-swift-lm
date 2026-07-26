// PagedKVSlabCommitment.swift
//
// WHEN a paged pool's slabs become MLX-resident (D1).
//
// The problem
// -----------
// `PagedKVGroup` builds its two slabs with `MLXArray.zeros(...)`, which is a
// LAZY `Full` primitive: no Metal buffer exists until something evaluates it.
// The only thing that forced them resident at engine-construction time was an
// explicit `PagedKVPool.materializeSlabs()` call in the provider's production
// factory. That single call is what broke two-model co-residency on a 36 GiB
// box: the FIRST model's pool — idle, holding no KV for anyone — landed in
// `MLX.GPU.activeMemory` before the SECOND model's post-load headroom guard
// re-measured, so the second model measured 0.15 GiB against a 1 GiB
// serveable-KV minimum and was unloaded with a 503. An all-contiguous pair
// measured 2.40 GiB and served, because a contiguous grant is an admission
// CEILING that allocates as it goes, never a preallocation.
//
// So paged was being charged its worst case at construction while contiguous
// was charged as it went. This file removes that asymmetry.
//
// What is deferred, and what is NOT
// ---------------------------------
// Deferred: the ALLOCATION. `commitSlabs()` is the one place that evaluates
// them, and it runs at the pool's first admission rather than at construction.
//
// NOT deferred: the GUARANTEE. Every page a row can reach must be backed
// before the row exists, or `PagedSequenceKV.ensurePage` reaches a page whose
// buffer was never allocated. `PagedKVBackend.reserve` and
// `makeSequenceState` therefore commit the WHOLE pool — every group, both
// slabs — immediately after the admission charge succeeds and before any
// `PagedSequenceKV` is minted. An admission that throws `capacityExhausted`
// commits nothing.
//
// Nothing else changes: `pageCount` is still fixed at `PagedKVPool.init`, the
// slabs are still `let`, and there is still no resize primitive. Lazy FIRST
// commitment is not a resize — it is the same buffer, allocated later.
//
// Why this is correctness-neutral
// -------------------------------
// The slabs are passed to the write/decode Metal kernels as INPUTS, never as
// declared outputs. MLX's `eval_impl` walks the tape leaf-first, so a slab's
// `Full` primitive runs (allocating and zero-filling the buffer) BEFORE the
// custom kernel that const-casts and writes into it, with a real
// `memoryBarrier(BarrierScopeBuffers)` between them because the encoder sees
// the slab as a prior output and then as an input. `eval_impl` then detaches
// the primitive, so the buffer is never recomputed or re-zeroed and its
// identity is stable for the pool's lifetime — which is the only property the
// in-place write design actually requires (see pagedattention.metal,
// "In-place slab writes"). The one hazard, a `Full` surviving inside a
// `compile` tracer and re-zeroing on a later eval, cannot arise here:
// `producesCompiledDecodeEligibleRows` is false for paged, so no slab is ever
// traced.
//
// Byte accounting
// ---------------
// Lazy commitment does NOT make any existing figure time-varying.
// `bytesInUse`, `bytesReserved`, `bytesCapacity` and `bytesPhysical` are all
// host-side arithmetic over page bookkeeping fixed at `PagedKVPool.init`;
// none of them observes evaluation state. In particular `bytesPhysical`
// remains the allocation CEILING (`pageCount * pageBytes`, poison pages
// included) and stays the right input for sizing and wired-limit consumers.
// `PagedKVBackend.bytesWired` is the new — and only — time-varying figure,
// and it is deliberately diagnostic: nothing admits or refuses on it.

import Foundation

/// When a paged pool's slabs are evaluated into real Metal residency.
public enum PagedKVSlabCommitment: String, Sendable, Equatable, CaseIterable {
    /// Wire the slabs during `PagedKVBackend.init`. First-token latency never
    /// pays the allocation, at the cost of an idle pool occupying unified
    /// memory that a co-resident model's headroom measurement will see. Use
    /// for microbenchmarks and profilers that want allocation out of the
    /// timed region, and for single-slot deployments that will never share a
    /// box.
    case atConstruction

    /// Wire the slabs at the pool's first admission — the moment it stops
    /// being idle. The production default: an unused pool contributes zero
    /// bytes to a co-resident model's post-load headroom measurement, exactly
    /// as an unused contiguous grant does.
    case atFirstAdmission
}

extension PagedKVBackend {
    /// Bytes the slabs have ACTUALLY committed to MLX right now: zero until
    /// the pool's first admission under `.atFirstAdmission`, `bytesPhysical`
    /// thereafter.
    ///
    /// TIME-VARYING BY CONSTRUCTION. Diagnostics and telemetry only — never
    /// an admission input, never a sizing input, never a wired-limit input.
    /// Anything that must not move under this backend's feet wants
    /// `bytesCapacity` (the budgeted, admission-relevant figure) or
    /// `bytesPhysical` (the allocation ceiling), both of which are fixed at
    /// pool construction.
    public var bytesWired: Int { slabsAreWired ? pool.bytesPhysical : 0 }

    /// Evaluate every group's slabs, making the pool's pages physically
    /// resident. Idempotent: after the first call this is a bool test, so the
    /// admission path can call it unconditionally.
    ///
    /// Thread-affinity is the pool's: the engine loop thread, no locking.
    public func commitSlabs() {
        guard !slabsAreWired else { return }
        pool.materializeSlabs()
        markSlabsWired()
    }
}
