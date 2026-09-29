import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// Prepared only. Run serially under the exclusive synthetic native lane.
// These tests use real PagedKVBackend/Admission/Metal work, not model payloads.
// The synchronized process delegate records native contract calls; provider
// ProcessMemoryLedger integration requires a separate provider-level test.
private final class PagedWorkProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
    private let lock = NSLock()
    private var c: UInt64 = 0, m: UInt64 = 0
    private var cap: UInt64 = 2 << 30
    var snapshot: (charge: UInt64, materialized: UInt64) { lock.withLock { (c, m) } }
    func limit(_ value: UInt64) { lock.withLock { cap = value } }
    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= cap, bytes >= m else {
                throw MLXError.caught("test process capacity refusal")
            }
            c = bytes
        }
    }
    func recordMaterialization(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes >= m, bytes <= c else {
                throw MLXError.caught("test invalid materialization")
            }
            m = bytes
        }
    }
    func withdrawCoverage(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= m else { throw MLXError.caught("test duplicate coverage withdrawal") }
            m -= bytes
        }
    }
    func retire() {}  // no artificial refund; real leases must remove their C/M
}

private func workKind(
    _ dk: Int = 192, _ dv: Int = 128, window: Int? = nil,
    heads: Int = 1, gqa: Int = 2, shares: Int? = nil, sinks: Bool = false
) -> CBv2LayerKind {
    .init(
        attention: window.map { .slidingWindow($0) } ?? .full,
        sharesKVWithLayer: shares, hasSinks: sinks, headDim: dk, valueHeadDim: dv,
        kvHeads: heads, queryHeads: heads * gqa)
}

private final class PagedWorkFixture {
    let kinds: [CBv2LayerKind]
    let backend: PagedKVBackend
    let caches: [PagedLayerCache]
    let bank: CBv2LayerCacheBank
    let process: PagedWorkProcessOwner
    let admission: AdmissionV2
    var states: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
    private var boundaries: [UInt64: CBv2PagedWriteBoundary] = [:]
    init(
        _ kinds: [CBv2LayerKind], dtype: DType = .float32, batch: Int = 2,
        softcap: Float? = nil, bindAdmission: Bool = true
    ) throws {
        let process = PagedWorkProcessOwner()
        let limits = try CBv2PagedGatheredAttentionLimits(
            maximumBatchSize: batch, maximumQueryTokens: 129, maximumContextTokens: 1_048_576,
            maximumInFlightGraphs: 2, maximumScratchBytes: 1 << 30,
            admissionMode: .stepOwned(.pinnedMetal))
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 1 << 30, dtype: dtype, maxPrefillChunk: 129,
                nominalMaxSequenceLength: 1_048_576, maxBufferLength: 8 << 20,
                segmentSizeBytes: 128 << 10,
                layerDTypes: Array(repeating: dtype, count: kinds.count),
                gatheredAttention: limits))
        let caches = backend.makeLayerCaches(attentionSoftcap: softcap)
        let bank = CBv2LayerCacheBank(caches: caches)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 30,
            config: .init(
                watermarkFraction: 0, elementBytes: dtype.size,
                layerElementBytes: Array(repeating: dtype.size, count: kinds.count)),
            residency: backend.kvResidency, processMemoryOwner: process)
        self.kinds = kinds
        self.backend = backend
        self.caches = caches
        self.bank = bank
        self.admission = admission
        self.process = process
        if bindAdmission { backend.pool.bindAdmission(admission) }
        // Component tests prepare the same mandatory owner; actual EngineV2
        // tests below exercise the production envelope binding instead.
        backend.pool.attentionWorkEnginePrepared = bindAdmission
    }
    func add(_ id: CBv2RequestID = .init(1), maximum: Int = 512) throws {
        states[id] = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: maximum)
    }
    func prepare(_ ids: [CBv2RequestID] = [.init(1)], count: Int) throws
        -> CBv2PagedAttentionStepOwner
    {
        let ranges = ids.map { id -> (id: CBv2RequestID, range: Range<Int>) in
            let start = states[id]!.compactMap { $0 }.first!.absoluteOffset
            return (id, start ..< (start + count))
        }
        let boundary = CBv2PagedWriteBoundary(pool: backend.pool)
        let owner = try XCTUnwrap(backend.prepareAttentionWork(assignments: ranges, states: states))
        boundaries[owner.generation] = boundary
        return owner
    }
    func forward(_ ids: [CBv2RequestID] = [.init(1)], count: Int) throws -> [MLXArray] {
        var outputs: [MLXArray] = []
        for (index, cache) in caches.enumerated() {
            let kind = kinds[index]
            let dtype = backend.pool.layerDTypes[index]
            let query = MLXArray.zeros(
                [ids.count, kind.queryHeads, count, kind.headDim], dtype: dtype)
            if let source = kind.sharesKVWithLayer {
                outputs.append(
                    cache.attendBorrowing(
                        source: caches[source], queries: query, scale: 0.125, sinks: nil))
            } else {
                cache.setRows(ids.map { states[$0]![index]! })
                let keys = MLXArray.zeros(
                    [ids.count, kind.kvHeads, count, kind.headDim], dtype: dtype)
                let values =
                    MLXArray.ones(
                        [ids.count, kind.kvHeads, count, kind.valueHeadDim], dtype: dtype)
                    * Float(index + 3)
                outputs.append(
                    cache.updateAndAttend(
                        queries: query, keys: keys, values: values, scale: 0.125, sinks: nil))
            }
        }
        try backend.pool.writeValidation.check()
        return outputs
    }
    func publish(_ owner: CBv2PagedAttentionStepOwner, outputs: [MLXArray]) throws {
        try owner.seal()
        try withError { fault in
            asyncEval(outputs + owner.evaluationTargets)
            try fault.check()
        }
        owner.publish()
        boundaries.removeValue(forKey: owner.generation)
        backend.pool.endAttentionConstruction(owner)
    }
    func retire() {
        StreamOrDevice.default.stream.synchronize()
        for owner in backend.pool.attentionWorkOwners.values where !owner.published {
            boundaries[owner.generation]?.discardFailedGraphAfterSynchronization()
        }
        boundaries.removeAll()
        caches.forEach { $0.setRows([]) }
        backend.pool.activeAttentionLayer = nil
        backend.pool.activeAttentionWork = nil
        backend.pool.discardUnpublishedAttentionWorkAfterDrain()
        for owner in Array(backend.pool.attentionWorkOwners.values) where owner.completed {
            owner.closeGraph()
        }
        for rows in states.values { backend.release(rows) }
        states.removeAll()
    }
    deinit { retire() }
}

private final class PagedWorkEngineModel: CBv2SteppableModel {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let b = tokens.dim(0)
        let n = tokens.dim(1)
        var hidden = tokens.asType(.float32).reshaped([b, 1, n, 1]) / Float(32)
        for cache in caches {
            let q = MLXArray.zeros([b, cache.kind.queryHeads, n, cache.kind.headDim])
            let output: MLXArray
            if let owner = cache.kind.sharesKVWithLayer {
                output = cache.attendBorrowing(
                    source: caches[owner], queries: q, scale: 0.125, sinks: nil)
            } else {
                output = cache.updateAndAttend(
                    queries: q,
                    keys: broadcast(hidden, to: [b, cache.kind.kvHeads, n, cache.kind.headDim]),
                    values: broadcast(
                        hidden, to: [b, cache.kind.kvHeads, n, cache.kind.valueHeadDim]),
                    scale: 0.125, sinks: nil)
            }
            hidden = mean(output, axes: [1, 3], keepDims: true)
        }
        let target = MLX.round(hidden.reshaped([b, n, 1]) * Float(128)).asType(.int32) % 31
        return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
    }
}

final class CBv2PagedAttentionWorkReservationTests: XCTestCase {
    func testLegacyPolicyDefaultAndUniformBackendStayUnchanged() throws {
        let limits = try CBv2PagedGatheredAttentionLimits(
            maximumBatchSize: 1, maximumQueryTokens: 16,
            maximumContextTokens: 128, maximumInFlightGraphs: 2, maximumScratchBytes: 512 << 20)
        XCTAssertEqual(limits.admissionMode, .poolLifetime)
        XCTAssertGreaterThan(
            try limits.scratchUpperBound(layerKinds: [workKind()], pageSize: 16), 0)
        let backend = try PagedKVBackend(
            layerKinds: [workKind(64, 64)],
            config: .init(capacityBytes: 4 << 20, maxPrefillChunk: 16, segmentSizeBytes: 32 << 10))
        XCTAssertFalse(backend.pool.usesStepOwnedAttention)
        XCTAssertEqual(backend.pool.gatheredAttentionScratchBound, 0)
        XCTAssertNil(try backend.prepareAttentionWork(assignments: [], states: [:]))
    }

    func testShortActualRangeKeepsNativeMillionTokenMetadata() throws {
        let f = try PagedWorkFixture([workKind(window: 128), workKind()])
        try f.add()
        let baseline = f.admission.bytesReserved
        let m = f.process.snapshot.materialized
        let work = try f.prepare(count: 16)
        XCTAssertEqual(f.backend.pool.config.nominalMaxSequenceLength, 1_048_576)
        XCTAssertEqual(f.backend.pool.config.gatheredAttention?.maximumContextTokens, 1_048_576)
        XCTAssertEqual(f.backend.pool.gatheredAttentionScratchBound, 0)
        XCTAssertTrue(
            work.descriptors.values.allSatisfy { $0.range == 0 ..< 16 && $0.keyCount == 16 })
        XCTAssertLessThan(work.reservedBytes, 32 << 20)
        XCTAssertEqual(f.admission.bytesReserved - baseline, work.reservedBytes)
        XCTAssertEqual(f.process.snapshot.charge, UInt64(f.admission.bytesReserved))
        XCTAssertEqual(f.process.snapshot.materialized, m, "future work grants no M")
        f.backend.pool.endAttentionConstruction(work)
        try work.discardAfterDrain()
        XCTAssertEqual(f.admission.bytesReserved, baseline)
    }

    func testProjectionSeparatesViewsCopiesSoftcapAndIndependentPadding() throws {
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        for (dk, dv) in [(192, 128), (128, 192)] {
            for gqa in [1, 8, 16] {
                for count in [1, 2, 4, 8, 9, 128, 129] {
                    for blockSize in [0, 128] {
                        for softcap in [false, true] {
                            let kind = workKind(dk, dv, heads: 1, gqa: gqa, sinks: true)
                            let history = 33
                            let total = history + count
                            var blocks: [CBv2PagedAttentionBlock] = []
                            var offset = 0
                            while offset < count {
                                let n = min(blockSize == 0 ? count : blockSize, count - offset)
                                let range = CBv2AttentionV1.queryBlockBounds(
                                    historyCount: history, offset: offset, count: n, window: nil)
                                blocks.append(
                                    .init(
                                        queryOffset: offset, queryCount: n,
                                        visibleStart: range.visibleStart,
                                        visibleEnd: range.visibleEnd))
                                offset += n
                            }
                            let d = CBv2PagedAttentionRowDescriptor(
                                requestID: .init(1), row: .init(nil), rowSerial: 1,
                                layerIndex: 0, ownerLayerIndex: 0,
                                range: history ..< (history + count),
                                baseOffset: 0, frozenHighWater: 0, gatheredRange: 0 ..< history,
                                keyStart: 0, keyCount: total, blocks: blocks, softcap: softcap)
                            for dtype: DType in [.float16, .bfloat16, .float32] {
                                let allocations = try CBv2PagedAttentionWorkProjection.allocations(
                                    kind: kind, dtype: dtype, descriptor: d, pageSize: 16,
                                    policy: policy)
                                XCTAssertTrue(
                                    allocations.allSatisfy {
                                        $0.upperBoundBytes >= $0.logicalBytes
                                            && $0.allocationCount > 0
                                    })
                                XCTAssertEqual(
                                    allocations.contains { $0.role.contains("repeatKeys") },
                                    softcap && gqa > 1)
                                XCTAssertTrue(allocations.contains { $0.role.contains("qk.copyK") })
                                XCTAssertEqual(
                                    allocations.first { $0.role == "gather.keys" }?.logicalBytes,
                                    history * dk * dtype.size)
                                XCTAssertTrue(allocations.contains { $0.role == "gather.records" })
                                XCTAssertTrue(
                                    allocations.contains { $0.role == "output.blockConcat" }
                                        == (blocks.count > 1))
                            }
                        }
                    }
                }
            }
        }
    }

    func testCheckedOverflowAndUnsupportedWorkspaceProfileRefuse() throws {
        XCTAssertThrowsError(try CBv2PagedWorkMath.add(Int.max, 1))
        XCTAssertThrowsError(try CBv2PagedWorkMath.product([Int.max, 2]))
        XCTAssertThrowsError(
            try CBv2PagedAttentionWorkEnvironment(queryBlockSize: 128, sdpaBlocks: "4097")
                .requireSupported())
        XCTAssertThrowsError(
            try CBv2PagedAttentionWorkEnvironment(queryBlockSize: 128, sdpaBlocks: "0")
                .requireSupported())
        try CBv2PagedAttentionWorkEnvironment(queryBlockSize: 0, sdpaBlocks: nil).requireSupported()
    }

    func testLocalAndProcessTightCapsRefuseBeforeAnyCursorPageOrFenceMutation() throws {
        for processCap in [false, true] {
            let f = try PagedWorkFixture([workKind(window: 17)])
            try f.add()
            let work = try f.prepare(count: 16)
            let needed = work.reservedBytes
            f.backend.pool.endAttentionConstruction(work)
            try work.discardAfterDrain()
            let row = try XCTUnwrap(f.states[.init(1)]![0] as? PagedSequenceKV)
            let before = f.admission.bytesReserved
            let charge = f.process.snapshot.charge
            let group = f.backend.pool.group(row.groupKey)
            let fence = group.writeFence
            let table = row.table
            let version = row.tableVersion
            if processCap {
                f.process.limit(charge + UInt64(needed) - 1)
            } else {
                f.admission.updateBytesCapacity(before + needed - 1)
            }
            XCTAssertThrowsError(try f.prepare(count: 16))
            XCTAssertEqual(row.absoluteOffset, 0)
            XCTAssertEqual(row.table, table)
            XCTAssertEqual(row.tableVersion, version)
            XCTAssertTrue(group.writeFence === fence)
            XCTAssertEqual(f.admission.bytesReserved, before)
            XCTAssertEqual(f.process.snapshot.charge, charge)
            if processCap {
                f.process.limit(charge + UInt64(needed))
            } else {
                f.admission.updateBytesCapacity(before + needed)
            }
            let accepted = try f.prepare(count: 16)
            XCTAssertEqual(accepted.reservedBytes, needed)
            f.backend.pool.endAttentionConstruction(accepted)
            try accepted.discardAfterDrain()
        }
    }

    func testMissingAndRepeatedCacheTicketsRefuseBeforeWrites() throws {
        for repeated in [false, true] {
            let f = try PagedWorkFixture([workKind()])
            try f.add()
            let row = try XCTUnwrap(f.states[.init(1)]![0] as? PagedSequenceKV)
            f.caches[0].setRows([row])
            if repeated {
                _ = try f.prepare(count: 16)
                _ = try f.forward(count: 16)
            }
            let offset = row.absoluteOffset
            let table = row.table
            let version = row.tableVersion
            let group = f.backend.pool.group(row.groupKey)
            let fence = group.writeFence
            let q = MLXArray.zeros([1, 2, 16, 192])
            let k = MLXArray.zeros([1, 1, 16, 192])
            let v = MLXArray.zeros([1, 1, 16, 128])
            _ = f.caches[0].updateAndAttend(
                queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
            XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
            XCTAssertEqual(row.absoluteOffset, offset)
            XCTAssertEqual(row.table, table)
            XCTAssertEqual(row.tableVersion, version)
            XCTAssertTrue(group.writeFence === fence)
            // Never evaluate the refused output.
        }
    }

    func testDirectRowMutationCannotAdvanceCursorWithoutTicket() throws {
        let f = try PagedWorkFixture([workKind()])
        try f.add()
        let row = try XCTUnwrap(f.states[.init(1)]![0] as? PagedSequenceKV)
        let k = MLXArray.zeros([1, 1, 1, 192])
        let v = MLXArray.zeros([1, 1, 1, 128])
        _ = row.update(keys: k, values: v)
        XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
        XCTAssertEqual(row.absoluteOffset, 0)
        XCTAssertTrue(row.table.isEmpty)
        f.backend.pool.writeValidation.clearAfterRetirement()
        row.write(keys: k.squeezed(axis: 0), values: v.squeezed(axis: 0))
        XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
        XCTAssertEqual(row.absoluteOffset, 0)
        XCTAssertTrue(row.table.isEmpty)
    }

    func testCoexistingGenerationsHaveIndependentChargesUntilBothComplete() throws {
        let f = try PagedWorkFixture([workKind()])
        try f.add()
        let baseline = f.admission.bytesReserved
        let first = try f.prepare(count: 16)
        let out1 = try f.forward(count: 16)
        try f.publish(first, outputs: out1)
        let second = try f.prepare(count: 1)
        XCTAssertNotEqual(first.generation, second.generation)
        XCTAssertEqual(
            f.admission.bytesReserved, baseline + first.reservedBytes + second.reservedBytes)
        let out2 = try f.forward(count: 1)
        try f.publish(second, outputs: out2)
        XCTAssertThrowsError(
            try f.prepare(count: 1),
            "a third unfinished native generation exceeds the sealed envelope")
        XCTAssertEqual(
            f.admission.bytesReserved, baseline + first.reservedBytes + second.reservedBytes)
        try first.finishEvaluation()
        first.closeGraph()
        XCTAssertEqual(f.admission.bytesReserved, baseline + second.reservedBytes)
        try second.finishEvaluation()
        second.closeGraph()
        XCTAssertEqual(f.admission.bytesReserved, baseline)
        first.closeGraph()
        second.closeGraph()
        XCTAssertEqual(f.admission.bytesReserved, baseline, "idempotent retirement")
    }

    func testBorrowedPrefillLoanSurvivesCompletionUntilCacheUnbind() throws {
        let f = try PagedWorkFixture([workKind(window: 17), workKind(window: 17, shares: 0)])
        try f.add()
        let baseline = f.admission.bytesReserved
        let m = f.process.snapshot.materialized
        let work = try f.prepare(count: 16)
        let outputs = try f.forward(count: 16)
        try f.publish(work, outputs: outputs)
        try work.finishEvaluation()
        work.closeGraph()
        XCTAssertTrue(work.completed)
        XCTAssertFalse(work.released)
        XCTAssertEqual(outputs[0].asArray(Float.self), outputs[1].asArray(Float.self))
        XCTAssertEqual(f.admission.bytesReserved, baseline + work.reservedBytes)
        XCTAssertEqual(f.process.snapshot.materialized, m)
        f.caches[0].setRows([])
        XCTAssertTrue(work.released)
        XCTAssertEqual(f.admission.bytesReserved, baseline)
    }

    func testRequestIDReuseCannotSpendThePriorRowsTicket() throws {
        let f = try PagedWorkFixture([workKind()])
        try f.add()
        let oldRows = f.states[.init(1)]!
        let old = try XCTUnwrap(oldRows[0] as? PagedSequenceKV)
        let work = try f.prepare(count: 16)
        f.backend.release(oldRows)
        try f.add()
        let replacement = try XCTUnwrap(f.states[.init(1)]![0] as? PagedSequenceKV)
        XCTAssertNotEqual(old.serial, replacement.serial)
        f.caches[0].setRows([replacement])
        _ = f.caches[0].updateAndAttend(
            queries: MLXArray.zeros([1, 2, 16, 192]),
            keys: MLXArray.zeros([1, 1, 16, 192]), values: MLXArray.zeros([1, 1, 16, 128]),
            scale: 0.125, sinks: nil)
        XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
        XCTAssertEqual(replacement.absoluteOffset, 0)
        XCTAssertTrue(replacement.table.isEmpty)
        f.backend.pool.endAttentionConstruction(work)
    }

    func testSameGeometryWrongBorrowerOwnerRefusesBeforeReadingOrChangingFence() throws {
        for count in [1, 16] {
            let f = try PagedWorkFixture([workKind(), workKind(), workKind(shares: 0)])
            try f.add()
            let original = f.states[.init(1)]!
            let charge = f.process.snapshot.charge
            f.states[.init(1)] = [original[1], original[0], nil]
            XCTAssertThrowsError(try f.prepare(count: count)) { error in
                guard let failure = error as? CBv2KVError,
                    case .backendIneligible(let reason) = failure
                else {
                    return XCTFail("expected sealed row-owner refusal, got \(error)")
                }
                XCTAssertTrue(reason.contains("row/cache identity"))
            }
            XCTAssertEqual(f.process.snapshot.charge, charge)
            XCTAssertTrue(original.compactMap { $0 }.allSatisfy { $0.absoluteOffset == 0 })
            f.states[.init(1)] = original
            let work = try f.prepare(count: count)
            let q = MLXArray.zeros([1, 2, count, 192])
            let k = MLXArray.zeros([1, 1, count, 192])
            for index in 0 ..< 2 {
                f.caches[index].setRows([f.states[.init(1)]![index]!])
                _ = f.caches[index].updateAndAttend(
                    queries: q, keys: k,
                    values: MLXArray.ones([1, 1, count, 128]) * Float(index == 0 ? 3 : 9),
                    scale: 0.125, sinks: nil)
            }
            let row = try XCTUnwrap(f.states[.init(1)]![0] as? PagedSequenceKV)
            let group = f.backend.pool.group(row.groupKey)
            let fence = group.writeFence
            let cursor = row.absoluteOffset
            let table = row.table
            _ = f.caches[2].attendBorrowing(
                source: f.caches[1], queries: q, scale: 0.125, sinks: nil)
            XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
            XCTAssertEqual(row.absoluteOffset, cursor)
            XCTAssertEqual(row.table, table)
            XCTAssertTrue(group.writeFence === fence)
            XCTAssertThrowsError(try work.seal())
            f.backend.pool.endAttentionConstruction(work)
        }
    }

    func testPackedRowsWithDifferentActualRangesStayIndependent() throws {
        let f = try PagedWorkFixture([workKind(window: 17)])
        try f.add()
        let seed = try f.prepare(count: 16)
        let seeded = try f.forward(count: 16)
        try f.publish(seed, outputs: seeded)
        try seed.finishEvaluation()
        seed.closeGraph()
        try f.add(.init(2))
        let work = try f.prepare([.init(1), .init(2)], count: 16)
        XCTAssertEqual(Set(work.descriptors.values.map(\.range)), Set([16 ..< 32, 0 ..< 16]))
        f.caches[0].setRows([f.states[.init(1)]![0]!, f.states[.init(2)]![0]!])
        let output = f.caches[0].updateAndAttend(
            queries: MLXArray.zeros([2, 2, 16, 192]),
            keys: MLXArray.zeros([2, 1, 16, 192]),
            values: broadcast(
                MLXArray([Float(3), Float(9)]).reshaped([2, 1, 1, 1]), to: [2, 1, 16, 128]),
            scale: 0.125, sinks: nil)
        try f.backend.pool.writeValidation.check()
        try f.publish(work, outputs: [output])
        try work.finishEvaluation()
        for value in output[0].asArray(Float.self) { XCTAssertEqual(value, 3, accuracy: 0.0001) }
        for value in output[1].asArray(Float.self) { XCTAssertEqual(value, 9, accuracy: 0.0001) }
        XCTAssertEqual(f.states[.init(1)]![0]!.absoluteOffset, 32)
        XCTAssertEqual(f.states[.init(2)]![0]!.absoluteOffset, 16)
        work.closeGraph()
    }

    func testDisablingCacheLoanDoesNotRetireUncompletedBorrowerGraph() throws {
        let f = try PagedWorkFixture([workKind(window: 17), workKind(window: 17, shares: 0)])
        try f.add()
        let baseline = f.admission.bytesReserved
        let work = try f.prepare(count: 16)
        let outputs = try f.forward(count: 16)
        try f.publish(work, outputs: outputs)
        f.caches[0].setRetainsChunkForBorrowers(false)
        work.closeGraph()
        XCTAssertFalse(work.released)
        XCTAssertEqual(f.admission.bytesReserved, baseline + work.reservedBytes)
        try work.finishEvaluation()
        XCTAssertEqual(outputs[0].asArray(Float.self), outputs[1].asArray(Float.self))
        XCTAssertTrue(work.released)
        XCTAssertEqual(f.admission.bytesReserved, baseline)
    }

    func testFailedCompletionKeepsRootsChargeAndNoMaterializationCredit() throws {
        let f = try PagedWorkFixture([workKind()])
        try f.add()
        let baseline = f.admission.bytesReserved
        let m = f.process.snapshot.materialized
        let work = try f.prepare(count: 16)
        let outputs = try f.forward(count: 16)
        try f.publish(work, outputs: outputs)
        work.beforeCompletionDrainForTesting = {
            throw MLXError.caught("injected completion refusal")
        }
        XCTAssertThrowsError(try work.finishEvaluation())
        work.closeGraph()
        XCTAssertFalse(work.completed)
        XCTAssertFalse(work.released)
        XCTAssertFalse(work.evaluationTargets.isEmpty)
        XCTAssertEqual(f.admission.bytesReserved, baseline + work.reservedBytes)
        XCTAssertEqual(f.process.snapshot.materialized, m)
        work.beforeCompletionDrainForTesting = nil
        // A later successful stream wait cannot rehabilitate the failed
        // required completion. Root ownership/credit remain quarantined.
        XCTAssertThrowsError(try work.finishEvaluation())
        XCTAssertThrowsError(try work.discardAfterDrain())
        XCTAssertTrue(work.completionFailed)
        XCTAssertFalse(work.released)
        XCTAssertFalse(work.evaluationTargets.isEmpty)
        XCTAssertEqual(f.admission.bytesReserved, baseline + work.reservedBytes)
    }

    func testNativeBothWidthOrdersAndBF16FP32KeepScalarAttentionOracle() throws {
        for (dk, dv) in [(192, 128), (128, 192)] {
            for dtype: DType in [.bfloat16, .float32] {
                for softcap: Float? in [nil, 20] {
                    let f = try PagedWorkFixture(
                        [workKind(dk, dv, window: 17, sinks: softcap != nil)], dtype: dtype,
                        softcap: softcap)
                    try f.add()
                    for count in [16, 1, 16, 1] {
                        let start = f.states[.init(1)]![0]!.absoluteOffset
                        let work = try f.prepare(count: count)
                        let outputs: [MLXArray]
                        if softcap != nil {
                            f.caches[0].setRows([f.states[.init(1)]![0]!])
                            outputs = [
                                f.caches[0].updateAndAttend(
                                    queries: MLXArray.zeros([1, 2, count, dk], dtype: dtype),
                                    keys: MLXArray.zeros([1, 1, count, dk], dtype: dtype),
                                    values: MLXArray.ones([1, 1, count, dv], dtype: dtype)
                                        * Float(3),
                                    scale: 0.125, sinks: MLXArray.zeros([2], dtype: dtype))
                            ]
                            try f.backend.pool.writeValidation.check()
                        } else {
                            outputs = try f.forward(count: count)
                        }
                        try f.publish(work, outputs: outputs)
                        try work.finishEvaluation()
                        XCTAssertEqual(outputs[0].shape, [1, 2, count, dv])
                        for (index, value) in outputs[0].asType(.float32).asArray(Float.self)
                            .enumerated()
                        {
                            let visible = Float(min(17, start + (index / dv) % count + 1))
                            let expected: Float = softcap == nil ? 3 : 3 * visible / (visible + 1)
                            XCTAssertEqual(value, expected, accuracy: 0.04)
                        }
                        work.closeGraph()
                    }
                    XCTAssertEqual(f.backend.pool.attentionWorkBytesReserved, 0)
                }
            }
        }
    }

    private func actualEngine(_ f: PagedWorkFixture) -> EngineV2 {
        .init(
            model: PagedWorkEngineModel(), layerKinds: f.kinds, backend: f.backend,
            cacheProvider: f.bank,
            schedulerConfig: .init(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 16, prefillChunkSize: 16, maxWaiting: 4,
                enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0, elementBytes: 4),
            processMemoryOwner: f.process)
    }

    func testActualEngineTokensMatchContiguousAndRetireFutureWork() async throws {
        let kinds = [workKind(window: 17), workKind()]
        let f = try PagedWorkFixture(kinds, bindAdmission: false)
        let paged = actualEngine(f)
        XCTAssertNil(paged.pagedAttentionWorkInactiveReason)
        let coldBackend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 30, kvDType: .float32))
        let cold = EngineV2(
            model: PagedWorkEngineModel(), layerKinds: kinds, backend: coldBackend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 16,
                prefillChunkSize: 16, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0, elementBytes: 4))
        let request = CBv2Request(
            id: .init(80), promptTokens: (0 ..< 41).map { ($0 * 7) % 29 },
            sampling: .init(temperature: 0, seed: 0), maxTokens: 5)
        let expected = await cbv2SchedCollect(try cold.submit(request))
        let result = await cbv2SchedCollect(try paged.submit(request))
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens, expected.tokens)
        await cold.shutdown()
        await paged.shutdown()
        XCTAssertEqual(paged.pagedAttentionWorkBytesReserved, 0)
        XCTAssertEqual(paged.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(f.process.snapshot.charge, 0)
        XCTAssertEqual(f.process.snapshot.materialized, 0)
    }

    func testActualEngineCancellationDrainsSubmittedOwner() async throws {
        let f = try PagedWorkFixture([workKind(window: 17)], bindAdmission: false)
        let engine = actualEngine(f)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = 1
        }
        let id = CBv2RequestID(81)
        let stream = try engine.submit(
            .init(id: id, promptTokens: Array(repeating: 1, count: 64), maxTokens: 8))
        let launched = await cbv2SchedWait { engine.stepCount == 1 }
        XCTAssertTrue(launched)
        XCTAssertGreaterThan(engine.pagedAttentionWorkBytesReserved, 0)
        engine.cancel(id)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        let result = await cbv2SchedCollect(stream)
        XCTAssertEqual(result.finishReason, .cancelled)
        await engine.shutdown()
        XCTAssertEqual(engine.pagedAttentionWorkBytesReserved, 0)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(f.process.snapshot.charge, 0)
        XCTAssertEqual(f.process.snapshot.materialized, 0)
    }

    func testActualEngineMissingOwnerOrLargerSchedulerEnvelopeRefusesPublication() throws {
        for scenario in ["owner", "batchGrowth", "stripeGrowth"] {
            let f = try PagedWorkFixture([workKind()], bindAdmission: false)
            let engine = EngineV2(
                model: PagedWorkEngineModel(), layerKinds: f.kinds, backend: f.backend,
                cacheProvider: f.bank,
                schedulerConfig: .init(
                    maxConcurrentRequests: 2,
                    maxBatchedTokensPerStep: scenario == "batchGrowth" ? 130 : 16,
                    prefillChunkSize: 16,
                    soloPrefillStripeTokens: scenario == "stripeGrowth" ? 130 : nil,
                    enablePrefixCache: false),
                processMemoryOwner: scenario == "owner" ? nil : f.process)
            XCTAssertNotNil(engine.pagedAttentionWorkInactiveReason)
            XCTAssertThrowsError(
                try engine.submit(.init(id: .init(82), promptTokens: [1, 2], maxTokens: 1)))
            XCTAssertEqual(engine.stepCount, 0)
            XCTAssertEqual(f.backend.bytesReserved, 0)
            XCTAssertEqual(engine.pagedAttentionWorkBytesReserved, 0)
            XCTAssertEqual(f.backend.pool.config.gatheredAttention?.maximumContextTokens, 1_048_576)
        }
    }
}
