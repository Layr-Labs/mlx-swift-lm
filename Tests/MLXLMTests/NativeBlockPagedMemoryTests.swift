import Foundation
import MLX
import Testing

@testable import MLXLMCommon

final class NativePageTestProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
    private let lock = NSLock()
    private var charge: UInt64 = 0
    private var materialized: UInt64 = 0
    private var retired = false
    let maximum: UInt64
    init(maximum: UInt64) { self.maximum = maximum }
    var bytes: UInt64 { lock.withLock { charge } }
    var coverage: UInt64 { lock.withLock { materialized } }
    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard !retired, bytes <= maximum else {
                throw CBv2NativeBlockError.invalidConfiguration
            }
            charge = bytes
        }
    }
    func recordMaterialization(_ bytes: UInt64) throws {
        try lock.withLock {
            guard !retired, bytes <= charge else { throw CBv2NativeBlockError.invalidConfiguration }
            materialized = bytes
        }
    }
    func withdrawCoverage(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= materialized else { throw CBv2NativeBlockError.invalidConfiguration }
            materialized -= bytes
        }
    }
    func retire() {
        lock.withLock {
            charge = 0
            materialized = 0
            retired = true
        }
    }
}

@Suite("Native paged process admission", .serialized)
struct NativeBlockPagedMemoryTests {
    @Test func nativeImportOwnsCoverageOnceAndRetainsItsPermitPastShutdown() async throws {
        let owner = NativePageTestProcessOwner(maximum: 64 << 20)
        let memory = try memory(owner: owner)
        let engine = try CBv2NativeBlockEngine(
            tokenizer: TestTokenizer(vocabularySize: 128), kvBytesCapacity: 64 << 20,
            pagedMemory: memory, reservationForRequest: { _ in 1 << 20 },
            makeSession: { _, _ in throw CBv2NativeBlockError.unsupportedRequest("unused") })
        #expect(engine.usesProcessMemoryOwner && engine.usesPagedStorage)
        let hostLease = try engine.reserveNativeCheckpointReadScratch()
        #expect(hostLease.usesProcessMemoryOwner && owner.bytes == 0)
        hostLease.close()
        await importThenDrain(engine: engine, owner: owner)
        #expect(owner.bytes == 0 && owner.coverage == 0)
        #expect(engine.capacity().kvBytesReserved == 0)
    }

    private func importThenDrain(engine: CBv2NativeBlockEngine, owner: NativePageTestProcessOwner)
        async
    {
        do {
            let manifest = CBv2CompleteCheckpointManifest(
                identity: .init(
                    modelAggregateHash: "m", promptContractID: "t", buildID: "b",
                    numericsFingerprint: "n"),
                position: 16, chunkSize: 16, prefixTokens: Array(repeating: 2, count: 16),
                cacheSalt: "scope", assistantCodecID: nil,
                tensors: try [CBv2CheckpointTensorRole.keys, .values].map {
                    try .init(role: $0, layer: 0, shape: [1, 1, 16, 64], dtype: .float32)
                },
                backendLayout: CBv2CompleteCheckpointManifest.diffusionBlockLayout,
                nativeBlockState: .init(windowPhysicalLength: 16, windowCursor: 16))
            let plan = try CBv2NativeBlockCheckpointImportPlan(
                manifest: manifest, engine: engine, codecIdentity: UUID(), maximumSequenceLength: 32
            )
            #expect(
                plan.usesProcessMemoryOwner
                    && plan.scratchBytes
                        < CBv2CompleteCheckpointManifest.maximumProviderScratchBytes
            )
            let importer = try plan.allocate()
            #expect(owner.coverage > 0 && owner.coverage <= owner.bytes)
            for (index, descriptor) in manifest.tensors.enumerated() {
                try importer.appendSegment(
                    tensorIndex: index, byteOffset: 0,
                    data: Data(repeating: 0, count: descriptor.byteCount))
            }
            let staged = try importer.finish()
            importer.close()
            let before = owner.bytes
            await engine.shutdown()
            #expect(
                owner.bytes == before && owner.coverage > 0,
                "Engine shutdown cannot refund a ticket still owned by its consumer")
            staged.close()
            #expect(owner.coverage == 0)
            #expect(
                owner.bytes > 0, "The retained manifest keeps only its separate metadata permit")
        } catch { Issue.record(error) }
    }
    private func memory(owner: NativePageTestProcessOwner) throws -> CBv2NativeBlockPagedMemory {
        _ = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .slidingWindow(32), headDim: 64, kvHeads: 1, queryHeads: 2),
        ]
        return try .init(
            layerKinds: kinds,
            configuration: .init(
                capacityBytes: 64 << 20, dtype: .float32, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 128, segmentSizeBytes: 64 << 10,
                layerDTypes: [.float32, .float32]),
            processMemoryOwner: owner)
    }

    @Test func pagesOverlapPhysicalBackingWhileCanvasAndSnapshotsRemainAdditive() throws {
        let owner = NativePageTestProcessOwner(maximum: 64 << 20)
        let memory = try memory(owner: owner)
        let nominal = memory.nominalBytes(tokens: 64)
        let auxiliary = 1 << 20
        let snapshots = 2 << 20
        try memory.setSharedReservation(bytes: snapshots)
        try memory.reserve(id: .init(1), tokens: 64, totalBytes: nominal + auxiliary)
        #expect(owner.bytes == UInt64(nominal + auxiliary + snapshots))
        let backend = memory.backend
        let rows = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: 32, maxLength: 64)
        #expect(backend.bytesWired >= nominal)
        #expect(
            owner.bytes == UInt64(backend.bytesWired + auxiliary + snapshots),
            "Nominal pages must not be added a second time to actual backing")
        #expect(owner.coverage == UInt64(backend.bytesWired))
        try MLX.withError { errors in
            for row in rows.compactMap({ $0 }) {
                let values = MLXArray.ones([1, 1, 32, 64])
                let output = row.update(keys: values, values: values)
                eval(output.0, output.1)
                try errors.check()
            }
            StreamOrDevice.default.stream.synchronize()
        }
        backend.release(rows)
        memory.release(id: .init(1))
        #expect(owner.bytes == UInt64(snapshots) && owner.coverage == 0)
        try memory.setSharedReservation(bytes: 0)
        #expect(owner.bytes == 0 && memory.reservedBytes == 0)
    }

    @Test func refusalAndShrinkDoNotRefundLiveRequestsOrExternalTransfers() throws {
        let owner = NativePageTestProcessOwner(maximum: 4 << 20)
        let memory = try memory(owner: owner)
        try memory.setSharedReservation(bytes: 2 << 20)
        let transient = try memory.reserveTransient(bytes: 1 << 20)
        let before = owner.bytes
        #expect(throws: (any Error).self) {
            try memory.reserve(id: .init(2), tokens: 64, totalBytes: 2 << 20)
        }
        #expect(owner.bytes == before)
        memory.updateCapacity(1 << 20)
        #expect(owner.bytes == before, "Lowering a grant cannot erase live backing promises")
        try memory.setSharedReservation(bytes: 0)
        #expect(owner.bytes == 1 << 20)
        #expect(throws: (any Error).self) { try memory.reserveTransient(bytes: 1) }
        transient.release()
        transient.release()
        #expect(owner.bytes == 0 && memory.reservedBytes == 0)
        memory.updateCapacity(8 << 20)
        try memory.reserve(id: .init(3), tokens: 32, totalBytes: 1 << 20)
        #expect(owner.bytes == 1 << 20)
        memory.release(id: .init(3))
        #expect(owner.bytes == 0)
    }
}
