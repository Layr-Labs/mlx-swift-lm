// PagedKVSlabCommitment.swift
//
// WHEN a paged pool's slabs become MLX-resident (D1) — and what happens
// when the commitment no longer fits the box.
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
// commits nothing, and an admission whose COMMIT throws is unwound — the
// page charge is released, the pool is left idle and unwired, and the next
// admission retries the commit from a clean ledger.
//
// Nothing else changes: `pageCount` is still fixed at `PagedKVPool.init`, the
// slabs are still `let`, and there is still no resize primitive. Lazy FIRST
// commitment is not a resize — it is the same buffer, allocated later.
//
// Why the ORDERING is correctness-neutral
// ---------------------------------------
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
// `compile` tracer and re-zeroing on a later eval, cannot arise here: there
// is no tracing path at all. The compiled [B, 1] decode graph was deleted
// with the v0.8.0 migration, so nothing in the engine puts a slab — or any
// other array — inside a `compile` tracer.
//
// Ordering, however, is only HALF of what the old eager commit provided.
// The other half was TIMING: an eager commit ran while the just-measured
// load headroom was still true. A deferred commit runs at first admission —
// possibly minutes later, after a co-resident model has consumed the very
// headroom this pool was deferred to yield (the provider's physical-capacity
// policy deliberately reports an uncommitted pool as ZERO bytes, so nothing
// holds the pool's bytes in escrow between load and first admission). The
// deferred eval can therefore genuinely fail, and MLX's answer to a Metal
// allocation failure is the installed error handler — which is `fatalError`
// when nobody bound one: a daemon abort, on a machine whose watchdog
// restarts it into the same state.
//
// Why a failed commitment is a REFUSAL, not an abort
// --------------------------------------------------
// `commitSlabs()` protects the eval in two layers:
//
//  1. PROACTIVE. Before the eval, the pool's full physical byte demand
//     (`bytesPhysical`) is re-checked against MLX's OWN memory accounting —
//     active bytes vs the allocator's configured limit (`commitShortfall`).
//     A pool that no longer fits throws `CBv2KVError.capacityExhausted`
//     naming needed and available bytes, without touching MLX state. This
//     closes the co-residency window: a neighbor that ate the headroom
//     since construction turns the admission into a retryable rejection.
//  2. DEFENSIVE. The eval itself runs under MLX's SCOPED error handler
//     (`withError`, task-local — never a process-global handler swap): an
//     allocation failure inside the C++ layer is caught at the mlx-c
//     boundary after a clean C++ unwind and surfaces as a thrown Swift
//     error. Nothing throws across C++ frames, and the failed slab keeps
//     its `Full` primitive (MLX only marks arrays evaluated AFTER their
//     primitive ran), so a retry re-evaluates exactly the missing slabs.
//     See `PagedKVPool.materializeSlabs`.
//
// Either refusal surfaces as `capacityExhausted` — the engine's RETRYABLE
// capacity class: `EngineLoopV2.ensureKVState` requeues the request while
// the pool waits for room, then finish-errors with
// `capacityExhaustedFinishPrefix`, which bridges map to a retryable
// capacity rejection (429-class), never a server error and never an abort.
// The pool itself stays UNWIRED and IDLE (`slabsAreWired` remains false),
// the admission charge is unwound by the caller, and the next admission
// retries the whole commit — `guard !slabsAreWired` is the idempotence
// that makes the call free once it finally succeeds.
//
// A PARTIAL commit (some slabs evaluated, a later one failed) retries
// exactly the REMAINDER. `materializeSlabs` evaluates slab-by-slab and
// records each slab's residency the moment its blocking eval returns
// (`PagedKVGroup.kSlabMaterialized`/`vSlabMaterialized`), so the retry's
// headroom re-check demands `bytesUnmaterialized` — never the full
// `bytesPhysical`. Demanding the full pool would count the resident slabs
// TWICE: once in the demand and once inside `activeMemory` (their buffers
// are owned by the pool's live MLXArrays), permanently refusing an
// exactly-fitting retry — the wedge this file exists to prevent.
//
// Byte accounting
// ---------------
// Lazy commitment does NOT make any existing figure time-varying.
// `bytesInUse`, `bytesReserved`, `bytesCapacity` and `bytesPhysical` are all
// host-side arithmetic over page bookkeeping fixed at `PagedKVPool.init`;
// none of them observes evaluation state. In particular `bytesPhysical`
// remains the allocation CEILING (`pageCount * pageBytes`, poison pages
// included) and stays the right input for sizing and wired-limit consumers.
// The time-varying figures are `PagedKVPool.bytesUnmaterialized` (what the
// commit re-check demands — the only admission-relevant one, and it only
// ever shrinks) and its complements `bytesMaterialized` /
// `PagedKVBackend.bytesWired`, which are deliberately diagnostic: nothing
// admits or refuses on them.

import Foundation
import MLX

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

/// Memory accounting consulted by `commitSlabs()` before the slabs are
/// wired. Production reads MLX's own counters (`live`); tests inject
/// deterministic values to drive the refusal and retry paths without
/// mutating process-global MLX limits.
struct PagedKVCommitMemoryProbe {
    /// Bytes held by live MLXArrays right now (`Memory.activeMemory`).
    var activeBytes: () -> Int
    /// The allocator's configured ceiling (`Memory.memoryLimit` — MLX's
    /// `block_limit_`, min(1.5 × recommended working set, 0.95 × RAM)
    /// unless the embedding process lowered it).
    var limitBytes: () -> Int

    /// MLX's real accounting.
    static var live: PagedKVCommitMemoryProbe {
        PagedKVCommitMemoryProbe(
            activeBytes: { Memory.activeMemory },
            limitBytes: { Memory.memoryLimit })
    }
}

extension PagedKVBackend {
    /// Bytes the slabs have ACTUALLY committed to MLX right now: zero until
    /// the pool's first admission under `.atFirstAdmission`, `bytesPhysical`
    /// once wired, and the honest resident amount in the (transient)
    /// partially-committed state between a failed commit and its retry.
    ///
    /// TIME-VARYING BY CONSTRUCTION. Diagnostics and telemetry only — never
    /// an admission input, never a sizing input, never a wired-limit input.
    /// Anything that must not move under this backend's feet wants
    /// `bytesCapacity` (the budgeted, admission-relevant figure) or
    /// `bytesPhysical` (the allocation ceiling), both of which are fixed at
    /// pool construction.
    public var bytesWired: Int { pool.bytesMaterialized }

    /// Bytes of headroom MISSING for a slab commitment of `required` bytes,
    /// or nil when the commitment fits.
    ///
    /// MLX's own accounting, not a restatement of provider policy: headroom
    /// is `limitBytes - activeBytes`. Cache memory is deliberately NOT
    /// subtracted — the Metal allocator reclaims cached buffers before it
    /// fails an allocation, so cache is available to the slabs by
    /// construction. No slab byte can hide in that cache and be counted as
    /// available twice: buffers enter the cache only when FREED
    /// (`MetalAllocator::free` recycles), the resident slabs' buffers are
    /// owned by the pool's live MLXArrays (never freed while the backend
    /// exists, hence in `activeBytes`), and a slab whose allocation FAILED
    /// never received a buffer at all (the malloc threw before any
    /// assignment). An over-committed box (`activeBytes > limitBytes`)
    /// reports the full deficit.
    static func commitShortfall(required: Int, activeBytes: Int, limitBytes: Int) -> Int? {
        let available = limitBytes - activeBytes
        return required <= available ? nil : required - available
    }

    /// Evaluate every group's slabs, making the pool's pages physically
    /// resident. Idempotent after success: once wired this is a bool test,
    /// so the admission path can call it unconditionally.
    ///
    /// REFUSES rather than traps when the box can no longer take the pool:
    /// throws `CBv2KVError.capacityExhausted` — the engine's retryable
    /// capacity class — and leaves the pool unwired so a later admission
    /// retries the commit. See the file header for the two layers.
    ///
    /// Thread-affinity is the pool's: the engine loop thread, no locking.
    public func commitSlabs() throws {
        guard !slabsAreWired else { return }
        // Only the bytes still needing ALLOCATION. After a partial commit
        // the resident slabs already sit inside `activeBytes`; demanding
        // the full `bytesPhysical` would count them on both sides of the
        // shortfall inequality and permanently refuse an exactly-fitting
        // retry (PR #100 review). First commit: equals `bytesPhysical`.
        let required = pool.bytesUnmaterialized

        // Layer 1 — proactive: the headroom measured at model load is stale
        // by first admission; re-check against what MLX says is left NOW.
        let active = commitMemoryProbe.activeBytes()
        let limit = commitMemoryProbe.limitBytes()
        if Self.commitShortfall(required: required, activeBytes: active, limitBytes: limit)
            != nil
        {
            throw CBv2KVError.capacityExhausted(
                needed: required, available: max(0, limit - active))
        }

        // Layer 2 — defensive: the eval can still lose a race with a
        // co-resident allocator (or hit the Metal resource-count limit);
        // a failure here is a thrown error, never the process-fatal
        // default handler. `needed` reports what is STILL missing after
        // the partial progress this attempt made.
        do {
            try pool.materializeSlabs()
        } catch {
            throw CBv2KVError.capacityExhausted(
                needed: pool.bytesUnmaterialized,
                available: max(0, commitMemoryProbe.limitBytes() - commitMemoryProbe.activeBytes()))
        }
        markSlabsWired()
    }
}
