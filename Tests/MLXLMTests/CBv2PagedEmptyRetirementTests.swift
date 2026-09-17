import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Paged empty-pool native retirement ordering", .serialized)
struct CBv2PagedEmptyRetirementTests {
    private let kind = CBv2LayerKind(
        attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)

    private func fixture() throws -> (PagedKVBackend, AdmissionV2) {
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 1 << 20, dtype: .bfloat16, maxPrefillChunk: 64,
            nominalMaxSequenceLength: 2048, maxBufferLength: 1 << 20,
            segmentSizeBytes: 9 * 4096))
        let admission = AdmissionV2(
            layerKinds: [kind], bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2),
            residency: backend.kvResidency)
        backend.pool.bindAdmission(admission)
        return (backend, admission)
    }

    private final class Probe: @unchecked Sendable {
        weak var storage: MLXArray?
        var events: [String] = []
        var drains = 0
        var drainCompleted = false
    }

    @Test("last backing dies before the owning stream drains and its floor refunds",
          arguments: [false, true])
    func emptyRetirementOrdersOwnersDrainAndRefund(scopedStream: Bool) throws {
        let outer = StreamOrDevice.default.stream
        if scopedStream {
            try Stream.withNewDefaultStream(device: .gpu) {
                #expect(StreamOrDevice.default.stream != outer)
                try exerciseEmptyRetirement()
            }
        } else {
            try exerciseEmptyRetirement()
        }
    }

    private func exerciseEmptyRetirement() throws {
        let stream = StreamOrDevice.default.stream
        let (backend, admission) = try fixture()
        let pool = backend.pool
        defer { pool.physicalLease?.close() }
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let rows = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 16, maxLength: 64)
        let row = try #require(rows[0] as? PagedSequenceKV)
        let keys = MLXArray.ones([1, 16, 64], dtype: .bfloat16)
        row.write(keys: keys, values: keys)
        eval(pool.group(row.groupKey).writeFence)

        let probe = Probe()
        probe.storage = pool.group(row.groupKey).segments.values.first?.storage
        #expect(probe.storage != nil)
        let floor = backend.bytesWired
        #expect(floor > 0)
        let original = try #require(pool.physicalLease)
        pool.physicalLease = CBv2BackendPhysicalLease(bytes: original.bytes, resize: { bytes in
            if bytes == 0 {
                #expect(probe.storage == nil)
                #expect(probe.drainCompleted)
                #expect(probe.events == ["drain-enter", "drain-complete"])
                #expect(admission.bytesReserved == floor)
                probe.events.append("refund")
            }
            try original.resize(to: bytes)
        }, onClose: { original.close() })
        pool.synchronizeEmptyRetirement = { [weak pool] actual in
            #expect(actual == stream)
            #expect(pool?.bytesMaterialized == 0)
            #expect(probe.storage == nil)
            #expect(admission.bytesReserved == floor)
            #expect(probe.events.isEmpty)
            probe.events.append("drain-enter")
            // Keep the production barrier in this ordering test. The separate
            // exclusive allocator gate verifies actual native bytes disappear.
            actual.synchronize()
            probe.drainCompleted = true
            probe.events.append("drain-complete")
        }
        admission.releaseAll(id: .init(1))
        backend.release(rows)
        #expect(probe.events == ["drain-enter", "drain-complete", "refund"])
        #expect(backend.bytesWired == 0 && admission.bytesReserved == 0)
    }

    @Test("nonempty and unchanged pools do not drain the decode stream")
    func nonemptyAndUnchangedTrimsDoNotDrain() throws {
        let (backend, admission) = try fixture()
        let pool = backend.pool
        defer { pool.physicalLease?.close() }
        let probe = Probe()
        pool.synchronizeEmptyRetirement = { stream in
            probe.drains += 1
            stream.synchronize()
        }
        pool.trimFreeSegments()
        #expect(probe.drains == 0)
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let first = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 16, maxLength: 64)
        try admission.reserve(id: .init(2), additionalTokens: 64)
        let second = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 16, maxLength: 64)
        pool.trimFreeSegments()
        #expect(probe.drains == 0)
        admission.releaseAll(id: .init(1))
        backend.release(first)
        #expect(backend.bytesWired > 0 && probe.drains == 0)
        pool.trimFreeSegments()
        #expect(probe.drains == 0)
        admission.releaseAll(id: .init(2))
        backend.release(second)
        #expect(backend.bytesWired == 0 && admission.bytesReserved == 0)
        #expect(probe.drains == 1)
        pool.trimFreeSegments()
        #expect(probe.drains == 1)
    }
}
