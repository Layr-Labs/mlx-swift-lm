// CBv2PagedSlabCommitmentTests.swift
//
// D1: WHEN a paged pool's slabs become MLX-resident.
//
// The provider used to force them resident at engine construction, which put
// an idle pool — one holding no KV for anyone — into `MLX.GPU.activeMemory`
// before the next model's post-load headroom guard measured. This file pins
// the engine half of the fix: the pool stays unwired until its first
// admission, the byte accessors do NOT become time-varying, and the
// guarantee that every admitted page is backed before any row exists is
// preserved rather than merely deferred alongside the allocation.
//
// The provider-side arithmetic (36 GiB two-model co-residency, the 48 GiB
// third-slot refusal, the coordinator's free_for_load_gb) lives in
// PagedKVPhysicalCapacityPolicyTests over there; this file is only about the
// backend's own contract.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 paged slab commitment")
struct CBv2PagedSlabCommitmentTests {

    private func fullKind(
        headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func config(capacityBytes: Int = 8 << 20) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: capacityBytes, maxPrefillChunk: 64,
            nominalMaxSequenceLength: 1024)
    }

    /// The D1 property. A pool nobody has admitted to contributes nothing to
    /// MLX residency, while still reporting its full budgeted and physical
    /// ceilings — those are what a co-resident model's admission arithmetic
    /// and the post-build serveability guard read.
    @Test("an unadmitted pool wires nothing but still reports its full ceiling")
    func idlePoolIsUnwiredYetFullySized() throws {
        let backend = try PagedKVBackend(layerKinds: [fullKind()], config: config())

        #expect(backend.slabCommitment == .atFirstAdmission, "the production default")
        #expect(!backend.slabsAreWired)
        #expect(backend.bytesWired == 0, "an idle pool pre-empts nobody")

        // Everything an admission or a capacity report reads is already final.
        #expect(backend.bytesCapacity > 0)
        #expect(backend.bytesPhysical > backend.bytesCapacity, "poison page per group")
        #expect(backend.bytesPhysical <= config().capacityBytes)
        #expect(backend.bytesInUse == 0)
        #expect(backend.bytesReserved == 0)
    }

    /// `.atConstruction` is the pre-D1 posture and must stay reachable — the
    /// decode profiler wants its allocation outside the timed region, and a
    /// single-slot box has nothing to yield headroom to.
    @Test("the eager posture still wires at construction")
    func eagerPostureWiresImmediately() throws {
        let backend = try PagedKVBackend(
            layerKinds: [fullKind()], config: config(), slabCommitment: .atConstruction)
        #expect(backend.slabsAreWired)
        #expect(backend.bytesWired == backend.bytesPhysical)
    }

    /// First admission is the moment the pool stops being idle, and it is
    /// where the commitment lands — via either entry point.
    @Test("the first admission wires the whole pool")
    func firstAdmissionWiresThePool() throws {
        let viaReserve = try PagedKVBackend(layerKinds: [fullKind()], config: config())
        #expect(viaReserve.bytesWired == 0)
        try viaReserve.reserve(layerKinds: [fullKind()], maxLength: 128)
        #expect(viaReserve.slabsAreWired)
        #expect(viaReserve.bytesWired == viaReserve.bytesPhysical)
        viaReserve.unreserve(layerKinds: [fullKind()], maxLength: 128)

        let viaState = try PagedKVBackend(layerKinds: [fullKind()], config: config())
        #expect(viaState.bytesWired == 0)
        let states = try viaState.makeSequenceState(
            layerKinds: [fullKind()], promptLength: 0, maxLength: 128)
        #expect(viaState.slabsAreWired)
        #expect(viaState.bytesWired == viaState.bytesPhysical)
        viaState.release(states)
    }

    /// R1's hazard, pinned: deferring the ALLOCATION is the fix, deferring
    /// the GUARANTEE would be a daemon abort. Whatever admission charges must
    /// be backed before a row can reach it, so the commitment has to be
    /// complete by the time `makeSequenceState` hands back a row — not lazily
    /// discovered on the first write.
    @Test("every admitted row finds its pages already backed")
    func admittedRowsNeverSeeAnUnwiredPool() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())

        // The reserve-then-materialize path the scheduler actually uses.
        try backend.reserve(layerKinds: [kind], maxLength: 128)
        #expect(backend.bytesWired == backend.bytesPhysical, "wired by the charge")
        let states = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 128, reserved: true)
        let row = try #require(states[0] as? PagedSequenceKV)

        // A real write through the in-place kernel against a pool that was
        // never eagerly materialized: this is the path that would trap if
        // laziness had left a reserved page unbacked.
        let keys = MLXArray.zeros([kind.kvHeads, 32, kind.headDim], dtype: .float16)
        let values = MLXArray.zeros([kind.kvHeads, 32, kind.headDim], dtype: .float16)
        row.write(keys: keys, values: values)
        eval(backend.pool.group(PagedKVGroupKey(kind)).writeFence)
        #expect(backend.bytesInUse > 0)

        backend.release(states)
    }

    /// A charge that FAILS must not wire anything: a pool that refused every
    /// admission it was ever offered is still idle, and taking residency for
    /// a request that was rejected is exactly the pre-emption D1 removes.
    @Test("a refused admission leaves the pool unwired")
    func refusedAdmissionDoesNotWire() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        let usable = backend.pool.usablePageCount(group: PagedKVGroupKey(kind))
        let overCommit = (usable + 1) * backend.pool.config.pageSize

        #expect(throws: CBv2KVError.self) {
            try backend.reserve(layerKinds: [kind], maxLength: overCommit)
        }
        #expect(!backend.slabsAreWired)
        #expect(backend.bytesWired == 0)
        #expect(backend.bytesReserved == 0, "the failed charge left no residue either")
    }

    /// `bytesWired` is the ONLY figure lazy commitment makes time-varying.
    /// The three the engine and provider actually admit and size against are
    /// page arithmetic fixed at `PagedKVPool.init`, so they must be identical
    /// either side of the commitment — otherwise an admission ledger or a
    /// capacity report would move under a co-resident model's feet.
    @Test("commitment moves bytesWired and nothing else")
    func commitmentDoesNotDisturbTheSizingFigures() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        let before = (
            capacity: backend.bytesCapacity,
            physical: backend.bytesPhysical,
            inUse: backend.bytesInUse,
            reserved: backend.bytesReserved)

        try backend.commitSlabs()

        #expect(backend.bytesCapacity == before.capacity)
        #expect(backend.bytesPhysical == before.physical)
        #expect(backend.bytesInUse == before.inUse)
        #expect(backend.bytesReserved == before.reserved)
        #expect(backend.bytesWired == before.physical, "the one figure that moved")

        // Idempotent: the admission path calls this on every single admission.
        try backend.commitSlabs()
        #expect(backend.bytesWired == before.physical)
    }

    /// An eager and a deferred backend of the same shape are indistinguishable
    /// to every consumer except `bytesWired`, once both have served. Lazy
    /// commitment is a change of TIMING, never of size.
    @Test("eager and deferred pools converge exactly once both have served")
    func posturesConvergeAfterFirstAdmission() throws {
        let kind = fullKind()
        let eager = try PagedKVBackend(
            layerKinds: [kind], config: config(), slabCommitment: .atConstruction)
        let deferred = try PagedKVBackend(layerKinds: [kind], config: config())

        #expect(eager.bytesWired != deferred.bytesWired, "they differ while idle")

        try deferred.reserve(layerKinds: [kind], maxLength: 128)
        try eager.reserve(layerKinds: [kind], maxLength: 128)

        #expect(deferred.bytesWired == eager.bytesWired)
        #expect(deferred.bytesCapacity == eager.bytesCapacity)
        #expect(deferred.bytesPhysical == eager.bytesPhysical)
        #expect(deferred.bytesReserved == eager.bytesReserved)
    }

    // MARK: - Recoverable commitment (the v0.8.0 co-residency window)

    /// The daemon-killer, pinned as a refusal. A pool that fit at
    /// construction can stop fitting by first admission (a co-resident
    /// model consumed the headroom — nothing holds an uncommitted pool's
    /// bytes in escrow). The commit must THROW the engine's retryable
    /// capacity error, never trap, and the failed admission must leave no
    /// residue: pool unwired, page charge unwound.
    @Test("a commit that no longer fits refuses instead of aborting")
    func commitRefusalThrowsInsteadOfAborting() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        // A neighbor ate the box since construction: MLX reports zero
        // bytes of headroom left under its configured limit.
        let limit = 1 << 30
        backend.commitMemoryProbe = PagedKVCommitMemoryProbe(
            activeBytes: { limit }, limitBytes: { limit })

        do {
            try backend.reserve(layerKinds: [kind], maxLength: 128)
            Issue.record("reserve must refuse when the commit cannot fit")
        } catch let error as CBv2KVError {
            guard case .capacityExhausted(let needed, let available) = error else {
                Issue.record("expected capacityExhausted, got \(error)")
                return
            }
            #expect(needed == backend.bytesPhysical, "names the pool's full byte demand")
            #expect(available == 0, "names what the box actually has left")
        }
        #expect(!backend.slabsAreWired)
        #expect(backend.bytesWired == 0)
        #expect(backend.bytesReserved == 0, "the refused admission left no page charge behind")
    }

    /// The production entry point (`makeSequenceState`, what
    /// `EngineLoopV2.ensureKVState` calls) refuses the same way and unwinds
    /// the charge it took — this is the shape the engine's capacity-requeue
    /// path consumes.
    @Test("makeSequenceState unwinds its charge on a refused commit")
    func makeSequenceStateUnwindsOnRefusedCommit() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        backend.commitMemoryProbe = PagedKVCommitMemoryProbe(
            activeBytes: { 1 << 30 }, limitBytes: { 1 << 30 })

        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 128)
        }
        #expect(!backend.slabsAreWired)
        #expect(backend.bytesReserved == 0, "no row was minted, so no hold may survive")
    }

    /// The other half of the unwind contract: a `reserved: true` caller
    /// already owns its page hold (charged by `reserve` at admission), and
    /// a refused commit inside `makeSequenceState` must NOT unwind it —
    /// the caller balances its own hold exactly once via `unreserve`. In
    /// production `reserve` wires the slabs itself, so this branch is
    /// defensive; the ledger contract still deserves pinning.
    @Test("a reserved-true refusal preserves the caller's hold")
    func reservedTrueRefusalPreservesCallersHold() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        // Charge the pages the way `reserve` does, but leave the slabs
        // unwired so the refusal fires inside makeSequenceState itself.
        try backend.pool.reserve(backend.pageNeeds(layerKinds: [kind], maxLength: 128))
        let held = backend.bytesReserved
        #expect(held > 0)
        backend.commitMemoryProbe = PagedKVCommitMemoryProbe(
            activeBytes: { 1 << 30 }, limitBytes: { 1 << 30 })

        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 0, maxLength: 128, reserved: true)
        }
        #expect(!backend.slabsAreWired)
        #expect(
            backend.bytesReserved == held,
            "the hold belongs to the caller — makeSequenceState must not unwind it")
        // The caller balances its own hold, exactly once.
        backend.unreserve(layerKinds: [kind], maxLength: 128)
        #expect(backend.bytesReserved == 0)
    }

    /// A refused commit is a DELAY, not a verdict: when the pressure
    /// clears, the next admission retries the commit, wires the slabs
    /// exactly once, and the pool serves real writes. Once wired, the
    /// probe is never consulted again (the `!slabsAreWired` guard is the
    /// idempotence that makes the admission-path call free).
    @Test("a refused commit leaves the slot retryable and wires exactly once")
    func refusedCommitIsRetryable() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        let required = backend.bytesPhysical
        var neighborResident = true
        var probeConsults = 0
        backend.commitMemoryProbe = PagedKVCommitMemoryProbe(
            activeBytes: {
                probeConsults += 1
                return neighborResident ? required : 0
            },
            // Exactly the pool's demand: the cleared retry is also an
            // exact-fit admission, the tightest headroom that must pass.
            limitBytes: { required })

        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 128)
        }
        #expect(!backend.slabsAreWired)
        #expect(backend.bytesReserved == 0)

        // The neighbor unloaded; the SAME entry point retries the commit.
        neighborResident = false
        let states = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 128)
        #expect(backend.slabsAreWired)
        #expect(backend.bytesWired == backend.bytesPhysical)

        // Wired is terminal: later commits are a bool test, no re-probe.
        let consultsAtWire = probeConsults
        try backend.commitSlabs()
        #expect(probeConsults == consultsAtWire, "the probe is dead once the pool is wired")

        // And the retried pool genuinely serves: a real kernel write lands.
        let row = try #require(states[0] as? PagedSequenceKV)
        let keys = MLXArray.zeros([kind.kvHeads, 32, kind.headDim], dtype: .float16)
        let values = MLXArray.zeros([kind.kvHeads, 32, kind.headDim], dtype: .float16)
        row.write(keys: keys, values: values)
        eval(backend.pool.group(PagedKVGroupKey(kind)).writeFence)
        #expect(backend.bytesInUse > 0)
        backend.release(states)
    }

    /// The headroom re-check arithmetic, pinned at the boundary. "Fits
    /// exactly" must admit — the re-check exists to stop genuine
    /// overshoots, not to shave usable headroom — and one byte short must
    /// refuse with the exact deficit.
    @Test("the headroom re-check is exact at the boundary")
    func headroomArithmeticIsExactAtTheBoundary() {
        // Fits exactly: zero slack is still a fit.
        #expect(
            PagedKVBackend.commitShortfall(required: 1024, activeBytes: 0, limitBytes: 1024)
                == nil)
        #expect(
            PagedKVBackend.commitShortfall(required: 1024, activeBytes: 512, limitBytes: 1536)
                == nil)
        // Short by one byte: refused, and the shortfall says one byte.
        #expect(
            PagedKVBackend.commitShortfall(required: 1024, activeBytes: 1, limitBytes: 1024)
                == 1)
        // Nothing required always fits, even on a saturated box.
        #expect(
            PagedKVBackend.commitShortfall(required: 0, activeBytes: 4096, limitBytes: 4096)
                == nil)
        // Over-committed box (active beyond the limit): the deficit compounds.
        #expect(
            PagedKVBackend.commitShortfall(required: 8, activeBytes: 4100, limitBytes: 4096)
                == 12)
    }

    /// An exact fit admits IN VIVO too, through the real commit path.
    @Test("a commit that fits exactly is admitted")
    func exactFitCommits() throws {
        let backend = try PagedKVBackend(layerKinds: [fullKind()], config: config())
        let required = backend.bytesPhysical
        backend.commitMemoryProbe = PagedKVCommitMemoryProbe(
            activeBytes: { 0 }, limitBytes: { required })
        try backend.commitSlabs()
        #expect(backend.slabsAreWired)
        #expect(backend.bytesWired == required)
    }

    /// Layer 2's seam contract, pinned against the vendored MLX: an error
    /// raised inside MLX's C++ layer during a `withError` scope surfaces as
    /// a thrown Swift error — the process-fatal default handler is NOT
    /// engaged. `PagedKVPool.materializeSlabs` relies on exactly this to
    /// turn a Metal allocation failure at slab-commit time into a throw. If
    /// the vendored mlx-swift ever loses the scoped-handler semantics, this
    /// fails loudly instead of the daemon dying in production.
    @Test("MLX's scoped error handler converts C++ errors into Swift throws")
    func mlxScopedHandlerThrows() {
        #expect(throws: (any Error).self) {
            try withError {
                let a = MLXArray(0 ..< 10, [2, 5])
                let b = MLXArray(0 ..< 15, [3, 5])
                // Broadcast error, raised inside the C++ layer and routed
                // through the same handler chain an allocator failure uses.
                _ = a + b
            }
        }
    }
}
