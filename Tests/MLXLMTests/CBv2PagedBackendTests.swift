// CBv2PagedBackendTests.swift
//
// WS-C unit tests: paged pool bookkeeping, per-sequence page tables,
// windowed rings, reservation-based admission, and backend lifecycle.
// No model weights required.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedBackend")
struct CBv2PagedBackendTests {

    // MARK: - Helpers

    private func fullKind(
        headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func windowedKind(
        _ window: Int, headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
            queryHeads: queryHeads)
    }

    private func config(
        capacityBytes: Int = 8 << 20, maxPrefillChunk: Int = 64,
        nominalMaxLen: Int = 1024
    ) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: capacityBytes, maxPrefillChunk: maxPrefillChunk,
            nominalMaxSequenceLength: nominalMaxLen)
    }

    private func randomKV(heads: Int, tokens: Int, dim: Int) -> MLXArray {
        MLXRandom.normal([heads, tokens, dim], dtype: .float16)
    }

    private func assertEqualArrays(
        _ a: MLXArray, _ b: MLXArray,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(a.shape == b.shape, sourceLocation: sourceLocation)
        #expect(
            arrayEqual(a, b).item(Bool.self),
            "arrays differ",
            sourceLocation: sourceLocation)
    }

    // MARK: - Pool

    @Test func poolGroupsAndAccounting() throws {
        let kinds = [fullKind(), fullKind(), windowedKind(32)]
        let pool = try PagedKVPool(layerKinds: kinds, config: config())
        // One group: all layers share (kvHeads=2, headDim=64).
        #expect(pool.groupKeys == [PagedKVGroupKey(kvHeads: 2, headDim: 64)])
        #expect(pool.bytesInUse == 0)
        #expect(pool.bytesReserved == 0)
        #expect(pool.bytesCapacity > 0)
        #expect(pool.bytesCapacity <= 8 << 20)
    }

    @Test func poolRejectsQuantSchemes() {
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVPool(
                layerKinds: [fullKind()],
                config: PagedKVPoolConfig(
                    capacityBytes: 1 << 20,
                    quantScheme: .affine(groupSize: 64, bits: 4)))
        }
    }

    @Test func reservationExhaustionThrowsAtomically() throws {
        let kinds = [fullKind()]
        // Tiny pool: room for a handful of pages only.
        let pageBytes = 2 * 2 * 16 * 64 * 2
        let pool = try PagedKVPool(
            layerKinds: kinds,
            config: config(capacityBytes: pageBytes * 8, nominalMaxLen: 128))
        let key = PagedKVGroupKey(kvHeads: 2, headDim: 64)
        try pool.reserve([key: 6])
        #expect(throws: CBv2KVError.self) {
            try pool.reserve([key: 3])
        }
        // Failed reserve must leave no partial state.
        try pool.reserve([key: 2])
        pool.unreserve([key: 8])
        #expect(pool.bytesReserved == 0)
    }

    // MARK: - Sequence state: full attention

    @Test func fullSequenceRoundtrip() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        let row = try #require(state[0] as? PagedSequenceKV)

        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        for n in [5, 16, 12, 1] {
            let k = randomKV(heads: 2, tokens: n, dim: 64)
            let v = randomKV(heads: 2, tokens: n, dim: 64)
            mirrorK.append(k)
            mirrorV.append(v)
            let (viewK, viewV) = row.update(
                keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0))
            #expect(viewK.shape == [1, 2, row.retainedCount, 64])
            #expect(viewV.shape == viewK.shape)
        }
        #expect(row.absoluteOffset == 34)
        #expect(row.retainedCount == 34)
        // 34 tokens => 3 pages of 16.
        #expect(row.table.count == 3)
        #expect(backend.bytesInUse == 3 * backend.pool.group(row.groupKey).pageBytes)

        let expectedK = concatenated(mirrorK, axis: 1).expandedDimensions(axis: 0)
        let expectedV = concatenated(mirrorV, axis: 1).expandedDimensions(axis: 0)
        let snap = row.snapshot()
        #expect(snap.offset == 34)
        assertEqualArrays(snap.keys, expectedK)
        assertEqualArrays(snap.values, expectedV)

        backend.release(state)
        #expect(backend.bytesInUse == 0)
        #expect(backend.pool.bytesReserved == 0)
    }

    @Test func rollbackFreesTailPagesAndRewrites() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128)
        let row = try #require(state[0] as? PagedSequenceKV)

        let k0 = randomKV(heads: 2, tokens: 20, dim: 64)
        let v0 = randomKV(heads: 2, tokens: 20, dim: 64)
        row.write(keys: k0, values: v0)
        #expect(row.table.count == 2)

        row.rollback(6)
        #expect(row.absoluteOffset == 14)
        #expect(row.table.count == 1)

        // Rewrite different tail tokens; gather must reflect the new tail.
        let k1 = randomKV(heads: 2, tokens: 4, dim: 64)
        let v1 = randomKV(heads: 2, tokens: 4, dim: 64)
        row.write(keys: k1, values: v1)
        #expect(row.absoluteOffset == 18)

        let expectedK = concatenated(
            [k0[0..., 0 ..< 14, 0...], k1], axis: 1
        ).expandedDimensions(axis: 0)
        let (gotK, _) = row.attendableViews()
        assertEqualArrays(gotK, expectedK)
        backend.release(state)
    }

    // MARK: - Sequence state: sliding window ring

    @Test func windowedRingWrapKeepsRecentEnd() throws {
        let window = 32
        let kinds = [windowedKind(window)]
        let cfg = config(maxPrefillChunk: 16)
        let backend = try PagedKVBackend(layerKinds: kinds, config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)
        let ring = PagedKVPool.ringPageCount(window: window, config: cfg)
        // ceil((32 + 16) / 16) + 1 = 4
        #expect(ring == 4)

        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        var written = 0
        for n in [16, 7, 16, 16, 3, 16, 16, 10] {
            let k = randomKV(heads: 2, tokens: n, dim: 64)
            let v = randomKV(heads: 2, tokens: n, dim: 64)
            mirrorK.append(k)
            mirrorV.append(v)
            row.write(keys: k, values: v)
            written += n
            #expect(row.absoluteOffset == written)
            #expect(row.table.count <= ring)

            let retained = row.retainedCount
            #expect(retained == min(written, window - 1 + n))
            let allK = concatenated(mirrorK, axis: 1)
            let allV = concatenated(mirrorV, axis: 1)
            let (gotK, gotV) = row.attendableViews()
            assertEqualArrays(
                gotK, allK[0..., (written - retained) ..< written, 0...]
                    .expandedDimensions(axis: 0))
            assertEqualArrays(
                gotV, allV[0..., (written - retained) ..< written, 0...]
                    .expandedDimensions(axis: 0))
        }
        // Ring must have wrapped (100 tokens through a 4-page/64-token ring).
        #expect(row.table.count == ring)
        backend.release(state)
        #expect(backend.bytesInUse == 0)
    }

    @Test func decodeAttendRangeClampsToWindow() throws {
        let kinds = [windowedKind(32)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 16))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)

        row.write(keys: randomKV(heads: 2, tokens: 10, dim: 64),
                  values: randomKV(heads: 2, tokens: 10, dim: 64))
        var range = row.decodeAttendRange
        #expect(range.start == 0 && range.length == 10)

        for _ in 0 ..< 6 {
            row.write(keys: randomKV(heads: 2, tokens: 16, dim: 64),
                      values: randomKV(heads: 2, tokens: 16, dim: 64))
        }
        // 106 tokens written; query position 105; window 32 => start 74.
        range = row.decodeAttendRange
        #expect(range.start == 74 && range.length == 32)
        backend.release(state)
    }

    // MARK: - Backend lifecycle

    @Test func sharedLayersGetNilStateAndValidation() throws {
        var shared = fullKind()
        shared.sharesKVWithLayer = 0
        let kinds = [fullKind(), shared]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 64)
        #expect(state.count == 2)
        #expect(state[0] != nil)
        #expect(state[1] == nil)
        backend.release(state)

        // Invalid head dim -> ineligible at build.
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(headDim: 96)], config: config())
        }
        // Bad GQA ratio -> ineligible at build.
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(kvHeads: 3, queryHeads: 4)], config: config())
        }
        // Shared layer pointing at another shared layer -> ineligible.
        var badShared = fullKind()
        badShared.sharesKVWithLayer = 1
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(), shared, badShared].map { $0 },
                config: config())
        }
    }

    @Test func admissionThrowsCapacityExhaustedAndRecovers() throws {
        let kinds = [fullKind()]
        let pageBytes = 2 * 2 * 16 * 64 * 2
        // Room for ~8 pages == 128 tokens.
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: config(capacityBytes: pageBytes * 8, nominalMaxLen: 128))

        let a = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 96)  // 6 pages
        #expect(throws: CBv2KVError.self) {
            _ = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 0, maxLength: 96)
        }
        backend.release(a)
        let b = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 96)
        backend.release(b)
    }

    @Test func adoptPrefixRoundtrip() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())

        // Donor sequence.
        let donor = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128)
        let donorRow = try #require(donor[0] as? PagedSequenceKV)
        let k = randomKV(heads: 2, tokens: 40, dim: 64)
        let v = randomKV(heads: 2, tokens: 40, dim: 64)
        donorRow.write(keys: k, values: v)
        let snap = donorRow.snapshot()
        // Materialize the snapshot before the donor pages are recycled —
        // gathered views reference the live slabs.
        eval(snap.keys, snap.values)
        backend.release(donor)

        let adopted = try backend.makeSequenceState(
            adopting: [(keys: snap.keys, values: snap.values, offset: snap.offset)],
            layerKinds: kinds, maxLength: 128)
        let row = try #require(adopted[0] as? PagedSequenceKV)
        #expect(row.absoluteOffset == 40)
        let (gotK, gotV) = row.attendableViews()
        assertEqualArrays(gotK, k.expandedDimensions(axis: 0))
        assertEqualArrays(gotV, v.expandedDimensions(axis: 0))
        backend.release(adopted)
    }

    @Test func fastForwardPlacesWindowedRecompute() throws {
        let kinds = [windowedKind(32)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 16))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)

        row.fastForward(to: 100)
        #expect(row.absoluteOffset == 100)
        #expect(row.retainedCount == 0)

        row.write(keys: randomKV(heads: 2, tokens: 16, dim: 64),
                  values: randomKV(heads: 2, tokens: 16, dim: 64))
        #expect(row.absoluteOffset == 116)
        // Only replayed tokens are attendable.
        #expect(row.retainedCount == 16)
        let range = row.decodeAttendRange
        #expect(range.start == 100 && range.length == 16)
        backend.release(state)
    }

    @Test func updateHonorsMaxLengthReservation() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 32)
        let row = try #require(state[0] as? PagedSequenceKV)
        row.write(keys: randomKV(heads: 2, tokens: 32, dim: 64),
                  values: randomKV(heads: 2, tokens: 32, dim: 64))
        #expect(row.table.count == 2)
        backend.release(state)
    }
}
