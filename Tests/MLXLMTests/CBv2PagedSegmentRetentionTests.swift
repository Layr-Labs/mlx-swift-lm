import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2 paged segment retention", .serialized)
struct CBv2PagedSegmentRetentionTests {
    private func kind(_ dim: Int = 64, kvHeads: Int = 1) -> CBv2LayerKind {
        CBv2LayerKind(attention: .full, headDim: dim, kvHeads: kvHeads, queryHeads: 2 * kvHeads)
    }

    private func backend(
        retain: Int, hostZeroed: Bool = false, kinds: [CBv2LayerKind]? = nil,
        segmentPages: Int = 9, pageBytes: Int = 2 * 16 * 64 * 2
    ) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: kinds ?? [kind()],
            config: PagedKVPoolConfig(
                capacityBytes: 8 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 2048, maxBufferLength: 16 << 20,
                segmentSizeBytes: pageBytes * segmentPages,
                retainedFreeSegmentBytes: retain, hostZeroedSegments: hostZeroed))
    }

    private func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

    private func storages(_ backend: PagedKVBackend) -> Set<ObjectIdentifier> {
        let group = backend.pool.group(backend.pool.groupKey(forLayer: 0))
        return Set(group.segments.values.map { ObjectIdentifier($0.storage) })
    }

    @Test func releasedSegmentsAreReinstalledWithoutAllocatingOrStayingCharged() throws {
        let backend = try backend(retain: 4 << 20)
        let first = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 64)
        let original = storages(backend)
        let committed = backend.bytesWired
        #expect(committed > 0)
        backend.release(first)
        // Parked backing is uncommitted: no grant, floor or reservation.
        #expect(backend.bytesWired == 0 && backend.bytesReserved == 0)
        #expect(backend.pool.retainedSegmentBytes == committed)
        #expect(backend.pool.segmentStorageSnapshot?.committedBytes == 0)

        var allocations = 0
        backend.pool.slabEval = {
            allocations += 1
            eval($0)
        }
        let second = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 64)
        #expect(allocations == 0)
        #expect(storages(backend) == original)
        #expect(backend.bytesWired == committed && backend.pool.retainedSegmentBytes == 0)
        let row = try #require(second[0] as? PagedSequenceKV)
        let k = MLXArray.ones([1, 40, 64], dtype: .bfloat16)
        row.write(keys: k, values: k)
        #expect(bytes(row.snapshot().keys) == bytes(k.expandedDimensions(axis: 0)))
        backend.release(second)
    }

    @Test func retentionBudgetBoundsParkedBackingAndZeroKeepsReference() throws {
        // 0 is the reference behavior: every free segment is dropped.
        let none = try backend(retain: 0)
        let a = try none.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 64)
        let oneBacking = none.bytesWired
        none.release(a)
        #expect(none.bytesWired == 0 && none.pool.retainedSegmentBytes == 0)

        // A budget smaller than one backing drops it too.
        let tiny = try backend(retain: oneBacking - 1)
        let b = try tiny.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 64)
        tiny.release(b)
        #expect(tiny.bytesWired == 0 && tiny.pool.retainedSegmentBytes == 0)

        // Two concurrent rows own two backings; a one-backing budget parks one.
        let limited = try backend(retain: oneBacking)
        let c = try limited.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 64)
        let d = try limited.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 64)
        #expect(limited.bytesWired == 2 * oneBacking)
        limited.release(c)
        limited.release(d)
        #expect(limited.bytesWired == 0)
        #expect(limited.pool.retainedSegmentBytes == oneBacking)
    }

    @Test func idleEngineReleasesRetainedBacking() throws {
        let backend = try backend(retain: 4 << 20)
        let first = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 64)
        backend.release(first)
        let parked = backend.pool.retainedSegmentBytes
        #expect(parked > 0)
        let touched = backend.pool.retainedSegmentsTouchedNanos
        backend.pool.releaseRetainedSegmentsIfIdle(nowNanos: touched + 1_000_000_000)
        #expect(backend.pool.retainedSegmentBytes == parked)
        backend.pool.releaseRetainedSegmentsIfIdle(nowNanos: touched + 31_000_000_000)
        #expect(backend.pool.retainedSegmentBytes == 0)
    }

    @Test func processReleaseRequestDropsRetainedBackingOnce() throws {
        let backend = try backend(retain: 4 << 20)
        let first = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 64)
        backend.release(first)
        let parked = backend.pool.retainedSegmentBytes
        #expect(parked > 0)
        // No request since the pool was built: nothing to honor.
        backend.pool.releaseRetainedSegmentsIfRequested()
        #expect(backend.pool.retainedSegmentBytes == parked)
        CBv2RetainedKVSegments.requestRelease()
        backend.pool.releaseRetainedSegmentsIfRequested()
        #expect(backend.pool.retainedSegmentBytes == 0)
        #expect(backend.bytesWired == 0 && backend.bytesReserved == 0)
        // Honored once: later retention survives until the next request.
        let second = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 64)
        backend.release(second)
        #expect(backend.pool.retainedSegmentBytes == parked)
        backend.pool.releaseRetainedSegmentsIfRequested()
        #expect(backend.pool.retainedSegmentBytes == parked)
    }

    @Test func processGaugeTracksParkReinstallReleaseAndTeardown() throws {
        let base = CBv2RetainedKVSegments.retainedBytes
        do {
            let backend = try backend(retain: 4 << 20)
            let first = try backend.makeSequenceState(
                layerKinds: [kind()], promptLength: 0, maxLength: 64)
            backend.release(first)
            let parked = backend.pool.retainedSegmentBytes
            #expect(parked > 0)
            #expect(CBv2RetainedKVSegments.retainedBytes == base + parked)
            let second = try backend.makeSequenceState(
                layerKinds: [kind()], promptLength: 0, maxLength: 64)
            #expect(CBv2RetainedKVSegments.retainedBytes == base)
            backend.release(second)
            #expect(CBv2RetainedKVSegments.retainedBytes == base + parked)
            backend.pool.releaseRetainedSegments()
            #expect(CBv2RetainedKVSegments.retainedBytes == base)
            let third = try backend.makeSequenceState(
                layerKinds: [kind()], promptLength: 0, maxLength: 64)
            backend.release(third)
            #expect(CBv2RetainedKVSegments.retainedBytes == base + parked)
        }
        // A pool torn down with parked backing takes it out of the gauge.
        #expect(CBv2RetainedKVSegments.retainedBytes == base)
    }

    @Test func refusedGrowthDropsRetainedBacking() throws {
        let backend = try backend(retain: 4 << 20)
        let first = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 64)
        backend.release(first)
        #expect(backend.pool.retainedSegmentBytes > 0)
        // 128 tokens need an 8-page class the parked 4-page backing cannot serve.
        backend.pool.slabEval = { _ in
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 128)
        }
        #expect(backend.pool.retainedSegmentBytes == 0)
        #expect(backend.bytesWired == 0 && backend.bytesReserved == 0)
        backend.pool.slabEval = { eval($0) }
        let retry = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 128)
        backend.release(retry)
    }

    @Test func hostZeroedSegmentsAreAvailableZeroedAndExactlySized() throws {
        // One page of this geometry is 16 KiB, so every segment is page-granular.
        let layer = kind(128, kvHeads: 2)
        let pageBytes = 2 * 16 * 2 * 128 * DType.bfloat16.size
        let backend = try backend(
            retain: 0, hostZeroed: true, kinds: [layer], pageBytes: pageBytes)
        var allocations = 0
        backend.pool.slabEval = {
            allocations += 1
            eval($0)
        }
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: 0, maxLength: 64)
        #expect(allocations == 0)
        let group = backend.pool.group(backend.pool.groupKey(forLayer: 0))
        #expect(!group.segments.isEmpty)
        for segment in group.segments.values {
            let info = try #require(try segment.storage.evaluatedBufferInfo())
            #expect(info.isUnique && info.allocatedBytes == segment.storage.nbytes)
            #expect(abs(segment.storage.asType(.float32)).max().item(Float.self) == 0)
        }
        let row = try #require(state[0] as? PagedSequenceKV)
        let k = MLXRandom.normal([2, 40, 128], key: MLXRandom.key(5)).asType(.bfloat16)
        let v = MLXRandom.normal([2, 40, 128], key: MLXRandom.key(6)).asType(.bfloat16)
        row.write(keys: k, values: v)
        let snapshot = row.snapshot()
        #expect(bytes(snapshot.keys) == bytes(k.expandedDimensions(axis: 0)))
        #expect(bytes(snapshot.values) == bytes(v.expandedDimensions(axis: 0)))
        backend.release(state)
        #expect(backend.bytesWired == 0)
    }

    /// Reinstalled backing still holds the previous row's KV. Decode must read
    /// only the new row's written tokens, bit-identical to a direct slab.
    @Test func reinstalledStaleSegmentsDecodeLikeFreshStorage() throws {
        let dtype = DType.bfloat16
        let backend = try backend(retain: 4 << 20, segmentPages: 2)
        let stale = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 512)
        let staleRow = try #require(stale[0] as? PagedSequenceKV)
        staleRow.write(
            keys: MLXRandom.normal([1, 512, 64], key: MLXRandom.key(81)).asType(dtype),
            values: MLXRandom.normal([1, 512, 64], key: MLXRandom.key(82)).asType(dtype))
        eval(staleRow.snapshot().keys)
        let original = storages(backend)
        backend.release(stale)
        #expect(backend.pool.retainedSegmentBytes > 0)

        let state = try backend.makeSequenceState(
            layerKinds: [kind()], promptLength: 0, maxLength: 512)
        defer { backend.release(state) }
        #expect(storages(backend) == original)
        let row = try #require(state[0] as? PagedSequenceKV)
        let count = 301
        row.write(
            keys: MLXRandom.normal([1, count, 64], key: MLXRandom.key(91)).asType(dtype),
            values: MLXRandom.normal([1, count, 64], key: MLXRandom.key(92)).asType(dtype))
        let descriptor = PagedSegmentDispatchPlan.Row(
            pages: row.table, info: row.seqInfoRow(attending: row.decodeAttendRange))
        let snap = row.snapshot()
        let group = backend.pool.group(backend.pool.groupKey(forLayer: 0))
        let q = MLXRandom.normal([1, 2, 64], key: MLXRandom.key(93)).asType(dtype)
        let params = MLXArray([Float(0), 0.125, 0, 0, 0, 0, 0, 0])
        let pages = descriptor.pages.count
        let padding = pages * 16 - descriptor.info.attendLength
        func pageShape(_ a: MLXArray) -> MLXArray {
            let padded =
                padding == 0
                ? a
                : concatenated([a, MLXArray.zeros([1, 1, padding, 64], dtype: dtype)], axis: 2)
            return padded.reshaped([pages, 1, 16, 64])
        }
        let (info, maxLength) = PagedAttentionKernel.seqinfo([descriptor.info])
        let direct = try PagedAttentionKernel.decode(
            queries: q, kSlab: pageShape(snap.keys), vSlab: pageShape(snap.values),
            tables: MLXArray((0 ..< pages).map { Int32($0) }, [1, pages]), seqinfo: info,
            maxAttendLength: maxLength, sinks: nil, params: params, softcap: false,
            pageSize: 16, writeFence: group.writeFence, kernelSource: backend.pool.kernelSource
        ).out
        let segmented = PagedSegmentAttention.decode(
            queries: q, newKeys: nil, newValues: nil, group: group, rows: [descriptor],
            sinks: nil, params: params, softcap: false, source: backend.pool.kernelSource)
        eval(direct, segmented)
        #expect(bytes(direct) == bytes(segmented))
    }
}
