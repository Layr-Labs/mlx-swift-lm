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

        backend.commitSlabs()

        #expect(backend.bytesCapacity == before.capacity)
        #expect(backend.bytesPhysical == before.physical)
        #expect(backend.bytesInUse == before.inUse)
        #expect(backend.bytesReserved == before.reserved)
        #expect(backend.bytesWired == before.physical, "the one figure that moved")

        // Idempotent: the admission path calls this on every single admission.
        backend.commitSlabs()
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
}
