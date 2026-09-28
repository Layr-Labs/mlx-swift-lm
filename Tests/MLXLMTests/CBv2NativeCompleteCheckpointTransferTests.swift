import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

/// Real Common native graphs/rows/issuer/async store jobs, not a MiMo model or
/// artifact qualification. Strict-loaded MiMo factory and MTP codec tests are
/// separate integration gates. Every retained-fault selector needs a fresh process.
final class CBv2NativeCompleteCheckpointTransferTests: XCTestCase {
    private enum Failure: Error { case invalidated, capacity, fence, noReceipt, adoption }

    private final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt64 = 0
        private var closed = false
        private var materialized: UInt64 = 0
        var bytes: UInt64 { lock.withLock { value } }
        var retired: Bool { lock.withLock { closed } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= 256 << 20 else { throw Failure.capacity }
                value = bytes
            }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock {
                guard bytes >= materialized, bytes <= value else { throw Failure.capacity }
                materialized = bytes
            }
        }
        func withdrawCoverage(_ bytes: UInt64) throws {
            try lock.withLock {
                guard bytes <= materialized else { throw Failure.capacity }
                materialized -= bytes
            }
        }
        func retire() { lock.withLock { XCTAssertEqual(value, 0); closed = true } }
    }

    /// Same real asymmetric attention pattern used by the existing historical
    /// checkpoint fixture, with immutable materialized native input and a
    /// package-only generation validator. This is not an NSObject stand-in.
    private final class Model: CBv2SteppableModel, CBv2HistoricalAttentionCheckpointProviding,
        CBv2CompleteCheckpointKVTypeProviding, CBv2NativeCompletePrefixBindingValidating {
        let loadedScale = MLXArray(Float(1.0 / 32.0))
        private let lock = NSLock()
        private var valid = true
        var cbv2SupportsHistoricalAttentionCheckpoint: Bool { true }
        var cbv2CompleteCheckpointKVDTypes: [DType]? { [.float32, .float32] }
        func invalidate() { lock.withLock { valid = false } }
        func validateNativeCompletePrefixBinding() throws {
            if !lock.withLock({ valid }) { throw Failure.invalidated }
        }
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            let b = tokens.dim(0), n = tokens.dim(1)
            var hidden = tokens.asType(.float32).reshaped([b, 1, n, 1]) * loadedScale
            for cache in caches {
                let q = MLXArray.zeros([b, 2, n, 192], dtype: .float32)
                let k = broadcast(hidden, to: [b, 1, n, 192])
                let v = broadcast(hidden, to: [b, 1, n, 128])
                hidden = mean(cache.updateAndAttend(queries: q, keys: k, values: v,
                    scale: 0.125, sinks: nil), axes: [1, 3], keepDims: true)
            }
            let target = MLX.round(hidden.reshaped([b, n, 1]) * Float(128)).asType(.int32) % 31
            return MLX.where(MLXArray(Int32(0)..<Int32(32)) .== target, Float(10), Float(-10))
        }
    }

    private final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var flag = false
        var value: Bool { lock.withLock { flag } }
        func set() { lock.withLock { flag = true } }
    }

    private final class Gate: @unchecked Sendable {
        let entered: XCTestExpectation
        private let lock = NSLock()
        private var first = true
        private let resume = DispatchSemaphore(value: 0)
        init(_ entered: XCTestExpectation) { self.entered = entered }
        func hold() {
            guard lock.withLock({ let old = first; first = false; return old }) else { return }
            entered.fulfill()
            _ = resume.wait(timeout: .now() + 8)
        }
        func release() { resume.signal() }
    }

    private final class Store: CBv2NativeCompletePrefixCache, @unchecked Sendable {
        let base: CompleteCheckpointFixtureStore
        private let jobs = DispatchGroup()
        private let lock = NSLock()
        private var supplied: [CBv2RequestID: CBv2StagedCompleteCheckpoint] = [:]
        var closeGate: Gate?
        let closeJoined = Flag()
        init(writeGate: CheckpointPublicationGate? = nil,
             archives: [CompleteCheckpointFixtureStore.Archive] = []) {
            base = .init(archives: archives, gate: writeGate, segmentBytes: 1 << 20)
        }
        var identity: CBv2CompleteCheckpointIdentity { base.identity }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
            base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
        }
        func takeStaged(requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
                        maximumSequenceLength: Int) -> CBv2StagedCompleteCheckpoint? {
            if let actual = lock.withLock({ supplied.removeValue(forKey: requestID) }) { return actual }
            return base.takeStaged(requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                            maximumSequenceLength: maximumSequenceLength)
        }
        func supplyActualForeignStage(_ stage: CBv2StagedCompleteCheckpoint, receipt: CBv2RequestID) {
            lock.withLock { precondition(supplied[receipt] == nil); supplied[receipt] = stage }
        }
        func donate(_ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
                    tokens: [Int], cacheSalt: String?,
                    completion: @escaping @Sendable ([Int]) -> Void) {
            jobs.enter()
            base.donate(source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt) { [self] positions in
                completion(positions)
                jobs.leave() // real underlying source read/close callback returned.
            }
        }
        func close() {
            let stages = lock.withLock { let old = Array(supplied.values); supplied = [:]; return old }
            stages.forEach { $0.close() }
            base.close()
        }
        func closeAndWait() async {
            closeGate?.hold()
            close()
            await withCheckedContinuation { continuation in
                jobs.notify(queue: .global(qos: .utility)) { continuation.resume() }
            }
            closeJoined.set()
        }
    }

    private struct Fixture {
        let engine: EngineV2
        let backend: CBv2ContiguousKVBackend
        let model: Model
        let process: ProcessOwner
        let store: Store
        let scope: NativeConstructionScope
    }
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }

    private func lane(fault: String? = nil) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["DARKBLOOM_TEST_NATIVE_COMPLETE_PREFIX"] == "1" else {
            throw XCTSkip("Requires an exclusively owned real native process")
        }
        guard env["DARKBLOOM_TEST_NATIVE_COMPLETE_PREFIX_FAULT"] == fault else {
            throw XCTSkip("Retained fault cells run alone in fresh processes")
        }
    }

    private func fixture(store: Store = Store(), shutdownTimeout: TimeInterval = 10,
                         beforeConsume: ((CBv2NativeExecutionContract, Model, CBv2ContiguousKVBackend,
                                          CBv2LayerCacheBank, ProcessOwner) throws -> Void)? = nil) throws -> Fixture {
        let scope = NativeConstructionScope(), model = Model(), process = ProcessOwner()
        let kinds: [CBv2LayerKind] = [
            .init(attention: .slidingWindow(17), headDim: 192, valueHeadDim: 128, kvHeads: 1, queryHeads: 2),
            .init(attention: .full, headDim: 192, valueHeadDim: 128, kvHeads: 1, queryHeads: 2),
        ]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 128 << 20, kvDType: .float32))
        let bank = CBv2LayerCacheBank(layerKinds: kinds)
        let engine = try scope.withPhase(.nativeSetup) {
            try scope.capture(StreamOrDevice.cpu.stream); try scope.capture(StreamOrDevice.default.stream)
            try scope.retain(model.loadedScale); try scope.willSubmit()
            try withError { errors in eval(model.loadedScale); try errors.check() }
            try scope.authorizeImmutableLoadedOwner(model)
            try scope.retainOwner(backend); try scope.retainOwner(bank); try scope.retainOwner(store)
            let contract = try CBv2NativeExecutionContract(model: model, backend: backend,
                cacheProvider: bank, assistant: nil, construction: scope, loadedOwner: model,
                completePrefixCache: store, completePrefixValidator: model,
                prefixProcessMemoryOwner: process)
            try beforeConsume?(contract, model, backend, bank, process)
            let engine = EngineV2(model: model, layerKinds: kinds, backend: backend, cacheProvider: bank,
                schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: chunk,
                    prefillChunkSize: chunk, enablePrefixCache: true),
                loopConfig: .init(stepTimeout: 60, watchdogInterval: 0.01, shutdownTimeout: shutdownTimeout),
                admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
                processMemoryOwner: process, nativeCompletionTracking: true, nativeExecutionContract: contract)
            try scope.retainOwner(engine)
            return engine
        }
        XCTAssertNil(engine.nativeCompletionFault)
        return .init(engine: engine, backend: backend, model: model, process: process, store: store, scope: scope)
    }

    private func request(_ id: UInt64) -> CBv2Request {
        .init(id: .init(id), promptTokens: (0..<(3 * chunk + 1)).map { ($0 * 7) % 29 },
              sampling: .init(temperature: 0), maxTokens: 3, cacheSalt: "isolated-tenant",
              prefixCacheReceiptID: .init(id + 1000))
    }

    private func shutDown(_ f: Fixture) async throws {
        guard case .quiescent = await f.engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(f.engine)
            throw Failure.noReceipt
        }
        XCTAssertTrue(f.store.closeJoined.value)
        XCTAssertEqual(f.backend.bytesReserved, 0)
        XCTAssertEqual(f.process.bytes, 0)
        XCTAssertTrue(f.process.retired)
    }


    private final class WeakArray: @unchecked Sendable {
        weak var value: MLXArray?
        init(_ value: MLXArray) { self.value = value }
    }
    private final class ArrayWitness: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [WeakArray] = []
        func record(_ arrays: [MLXArray]) { lock.withLock { values = arrays.map(WeakArray.init) } }
        var liveCount: Int { lock.withLock { values.filter { $0.value != nil }.count } }
    }
    private func tracking(_ f: Fixture) throws -> CBv2NativeShutdownState {
        try XCTUnwrap(f.engine.loopForTesting.nativeShutdownState)
    }
    private func capturedArchive() async throws -> CompleteCheckpointFixtureStore.Archive {
        let f = try fixture()
        let submitted = try f.engine.submitWithNativeRetirement(request(800))
        let result = await cbv2SchedCollect(submitted.events)
        guard result.finishReason == .length else {
            _ = await f.engine.shutdownReportingNativeCompletion()
            throw Failure.noReceipt
        }
        await submitted.retirement.wait()
        let archive = try XCTUnwrap(f.store.base.saved.first { $0.manifest.position == chunk })
        try await shutDown(f)
        return archive // encoded Data + wire-roundtripped manifest; no donor native aliases
    }
    private func importPlan(_ f: Fixture, _ archive: CompleteCheckpointFixtureStore.Archive,
                            _ input: CBv2Request, retireGate: Gate? = nil, armed: Flag? = nil)
        throws -> CBv2CompleteCheckpointImportPlan {
        let state = try tracking(f)
        f.engine.loopForTesting.onEngineQueueSync {
            state.beforeFenceForTesting = { _ in if armed?.value == true { retireGate?.hold() } }
        }
        defer { f.engine.loopForTesting.onEngineQueueSync { state.beforeFenceForTesting = nil } }
        return try f.engine.planCompleteCheckpointImport(manifest: archive.manifest, request: input)
    }
    private func fill(_ sink: CBv2CompleteCheckpointImport, archive: CompleteCheckpointFixtureStore.Archive) throws {
        for piece in archive.chunks {
            try sink.appendSegment(tensorIndex: piece.tensor, byteOffset: piece.offset, data: piece.bytes)
        }
    }

    func testDeferredPlanOwnsRealLoanAndExplicitCloseLeavesRetainedAliasMetadataOnly() async throws {
        try lane()
        let archive = try await capturedArchive(), f = try fixture()
        let entered = expectation(description: "deferred plan real retirement fence")
        let gate = Gate(entered), armed = Flag()
        var plan: CBv2CompleteCheckpointImportPlan? = try importPlan(f, archive, request(801),
            retireGate: gate, armed: armed)
        XCTAssertTrue(try tracking(f).hasLoans)
        armed.set()
        plan!.close()
        await fulfillment(of: [entered], timeout: 3)
        XCTAssertTrue(try tracking(f).hasLoans, "queued completion is not retirement")
        XCTAssertThrowsError(try plan!.allocate(onRelease: {}))
        gate.release()
        let nativeTracking = try tracking(f)
        let loanEnded = await cbv2SchedWait { !nativeTracking.hasLoans }
        XCTAssertTrue(loanEnded)
        XCTAssertNotNil(plan, "public metadata alias remains, without a native codec loan")
        plan = nil // metadata permit has its OWN lifetime; final-zero waits for it
        try await shutDown(f)
    }

    func testMoveStageCloseKeepsActualArraysAndChargeUntilQueuedFenceCompletes() async throws {
        try lane()
        let archive = try await capturedArchive(), f = try fixture()
        let entered = expectation(description: "import close real fence")
        let gate = Gate(entered), armed = Flag(), released = Flag(), witness = ArrayWitness()
        var plan: CBv2CompleteCheckpointImportPlan? = try importPlan(f, archive, request(802),
            retireGate: gate, armed: armed)
        plan!.evaluateDestinations = { arrays in
            witness.record(arrays)
            try withError { eval(arrays) }
        }
        var sink: CBv2CompleteCheckpointImport? = try plan!.allocate { released.set() }
        try fill(sink!, archive: archive)
        var stage: CBv2StagedCompleteCheckpoint? = try sink!.finish()
        sink!.close() // source wrapper no longer owns the moved stage
        XCTAssertThrowsError(try plan!.allocate(onRelease: {}), "native plan is single use")
        XCTAssertGreaterThan(witness.liveCount, 0)
        let charge = f.process.bytes
        armed.set(); stage!.close()
        await fulfillment(of: [entered], timeout: 3)
        XCTAssertFalse(released.value)
        XCTAssertGreaterThan(witness.liveCount, 0)
        XCTAssertEqual(f.process.bytes, charge)
        XCTAssertTrue(try tracking(f).hasLoans)
        gate.release()
        let fullyRetired = await cbv2SchedWait { released.value && witness.liveCount == 0 }
        XCTAssertTrue(fullyRetired)
        XCTAssertFalse(try tracking(f).hasLoans)
        XCTAssertNotNil(plan); XCTAssertNotNil(sink); XCTAssertNotNil(stage)
        // Retained closed handles keep manifest/plan metadata, not native arrays.
        stage = nil; sink = nil; plan = nil
        try await shutDown(f)
    }

    func testForeignNativeConsumptionDoesNotMoveAndActualBackendRefusalRetiresAfterFence() async throws {
        try lane()
        let archive = try await capturedArchive(), f = try fixture()
        let input = request(803), codec = try XCTUnwrap(f.engine.completeCheckpointCodec)
        let entered = expectation(description: "failed adoption retires through real import fence")
        let gate = Gate(entered), armed = Flag(), released = Flag()
        var plan: CBv2CompleteCheckpointImportPlan? = try importPlan(f, archive, input, retireGate: gate, armed: armed)
        var sink: CBv2CompleteCheckpointImport? = try plan!.allocate { released.set() }
        try fill(sink!, archive: archive)
        var stage: CBv2StagedCompleteCheckpoint? = try sink!.finish()
        XCTAssertThrowsError(try stage!.consumeNativePreparedState(store: f.store,
            request: input, engineID: UUID(), expectedCodec: codec) { _, _, _ in XCTFail("foreign consumed"); return () })
        XCTAssertNoThrow(try stage!.withValidatedNativeCodec(store: f.store, request: input,
            engineID: f.engine.nativeShutdownEngineID, expectedCodec: codec) { _ in () })
        f.backend.checkpointBeforeRegistration = { throw Failure.adoption }
        f.store.supplyActualForeignStage(stage!, receipt: try XCTUnwrap(input.prefixCacheReceiptID))
        let engine = f.engine
        let nativeTracking = try tracking(f)
        defer { gate.release() }
        armed.set()
        // Submission/collection can depend on this real retirement fence.
        // Do not await them before observing and releasing the held gate.
        let collector = Task { await cbv2SchedCollect(try engine.submit(input)) }
        await fulfillment(of: [entered], timeout: 3)
        XCTAssertFalse(released.value)
        XCTAssertTrue(nativeTracking.hasLoans)
        XCTAssertGreaterThan(f.process.bytes, 0)
        XCTAssertNil(f.engine.nativeCompletionFault)
        gate.release()
        let result = try await collector.value
        XCTAssertEqual(result.finishReason, .length, "ordinary metadata veto falls back, not a fabricated native fault")
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved, 0)
        let callbackReturned = await cbv2SchedWait { released.value }
        XCTAssertTrue(callbackReturned)
        stage = nil; sink = nil; plan = nil
        try await shutDown(f)
    }

    func testRequiredImportEvaluationFailureCannotBeRehabilitatedByLaterSuccessfulFence() async throws {
        try lane(fault: "import-evaluation")
        let archive = try await capturedArchive(), f = try fixture(), released = Flag(), witness = ArrayWitness()
        var plan: CBv2CompleteCheckpointImportPlan? = try f.engine.planCompleteCheckpointImport(
            manifest: archive.manifest, request: request(804))
        plan!.evaluateDestinations = { arrays in
            witness.record(arrays)
            try withError { eval(arrays) }
            throw Failure.fence // injected required-completion failure AFTER real native evaluation
        }
        XCTAssertThrowsError(try plan!.allocate { released.set() })
        let retainedCharge = f.process.bytes
        XCTAssertGreaterThan(retainedCharge, 0); XCTAssertGreaterThan(witness.liveCount, 0)
        XCTAssertFalse(released.value)
        XCTAssertNotNil(f.engine.nativeCompletionFault)
        try withError { StreamOrDevice.default.stream.synchronize(); StreamOrDevice.cpu.stream.synchronize() }
        plan!.close(); plan = nil
        XCTAssertFalse(released.value)
        XCTAssertGreaterThanOrEqual(f.process.bytes, retainedCharge)
        XCTAssertGreaterThan(witness.liveCount, 0)
        guard case .incomplete = await f.engine.shutdownReportingNativeCompletion() else {
            return XCTFail("original required failure became quiescent")
        }
        _ = Unmanaged.passRetained(f.engine)
        // Fresh-process retained-fault test, NOT a physical GPU-failure claim.
    }

    func testActualStridedExportSegmentsKeepOnlyBoundedSourceRoots() async throws {
        try lane()
        let archive = try await capturedArchive(), f = try fixture()
        let codec = try XCTUnwrap(f.engine.completeCheckpointCodec)
        let work = try f.engine.loopForTesting.onEngineQueueSync {
            try f.engine.loopForTesting.makeNativeCompletePrefixWork(purpose: .publication, requestID: .init(805))
        }
        let permit = try codec.admission.reserveTransient(bytes: 8 << 20)
        try work.retain(owners: [permit])
        try work.captureCurrentStreams()
        var arrays: [MLXArray] = []
        var expected: [Data] = []
        for descriptor in archive.manifest.tensors {
            let p = descriptor.shape[2], d = descriptor.shape[3]
            let words = (0..<(p*d)).map { UInt32($0 % 1024) | 0x3f000000 }
            let backing = MLXArray(words, [1,1,d,p])
            let array = backing.view(dtype: .float32).transposed(0,1,3,2)
            try work.retain(arrays: [backing, array])
            arrays.append(array)
            var bytes = Data()
            for time in 0..<p {
                for feature in 0..<d {
                    var word = words[feature*p+time].littleEndian
                    withUnsafeBytes(of: &word) { bytes.append(contentsOf: $0) }
                }
            }
            expected.append(bytes)
        }
        try withError { eval(arrays) }
        let source = CBv2CompleteCheckpointExport(manifest: archive.manifest, arrays: arrays,
            usesProcessMemoryOwner: true, retainedOwners: [permit])
        try work.retain(owners: [source])
        try source.bindNativeCompletePrefixWork(work)
        let roots = work.debugRetainedArrayCount, charge = f.process.bytes
        for (index, descriptor) in archive.manifest.tensors.enumerated() {
            var data = Data(), offset = 0
            while offset < descriptor.byteCount {
                let part = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: 260)
                data.append(part); offset += part.count
                XCTAssertEqual(work.debugRetainedArrayCount, roots, "no per-segment native packing allocation")
                XCTAssertEqual(f.process.bytes, charge, "no temporary-readback refund of whole work")
            }
            XCTAssertEqual(data, expected[index])
        }
        source.close(); arrays.removeAll()
        let retired = expectation(description: "publication operation and scratch retired")
        XCTAssertTrue(work.finishAfterDroppingConsumers { permit.release(); retired.fulfill() })
        await fulfillment(of: [retired], timeout: 3)
        try await shutDown(f)
    }

    func testRealAdmissionTransferKeepsFullTargetExcessAuxiliaryAndHostThroughDetach() throws {
        try lane()
        // Actual Admission + native evaluated buffers; this is accounting,
        // not a trained-head or model numerical equivalence test.
        for releaseLeaseFirst in [true, false] {
            let process = ProcessOwner(), position = 128, maximum = 256
            let kinds: [CBv2LayerKind] = [
                .init(attention: .slidingWindow(17), headDim: 192, valueHeadDim: 128, kvHeads: 1, queryHeads: 2),
                .init(attention: .full, headDim: 192, valueHeadDim: 128, kvHeads: 1, queryHeads: 2)]
            let fixed = 2 << 20
            let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 32 << 20,
                config: .init(watermarkFraction: 0, elementBytes: 2, fixedBytesPerRequest: fixed),
                processMemoryOwner: process)
            let targetShapes = [[1,1,17,192],[1,1,17,128],[1,1,maximum,192],[1,1,maximum,128]]
            let targetSpecs: [([Int], DType)] = targetShapes.map { ($0, .bfloat16) }
            let auxSpecs: [([Int], DType)] = Array(repeating: ([1,1,17,64], .bfloat16), count: 6)
                + [([1,3,64], .bfloat16), ([1,position], .int32), ([3,7], .int64)]
            XCTAssertEqual(auxSpecs.count, 9)
            func bound(_ specs: [([Int], DType)]) throws -> Int {
                try specs.reduce(0) { total, spec in
                    try CBv2CheckpointAllocationFootprint.add(total,
                        CBv2CheckpointAllocationFootprint.bound(spec.0.reduce(1, *) * spec.1.size))
                }
            }
            let target = try bound(targetSpecs), nativeAux = try bound(auxSpecs)
            let host = 8 * position + (64 << 10), auxiliary = nativeAux + host, scratch = 1 << 20
            let nominalTarget = (17 + maximum) * (192 + 128) * 2
            let ordinary = nominalTarget + fixed
            XCTAssertEqual(admission.allocatedBytes(forTokens: maximum), ordinary)
            let excess = target - nominalTarget
            XCTAssertGreaterThan(excess, 0, "Metal allocator-bound target padding must be discriminating")
            XCTAssertLessThan(auxiliary, fixed, "discounting aux against fixed state would erase this witness")
            let stage = try admission.reserveCheckpointStage(targetBytes: target, auxiliaryBytes: auxiliary, scratchBytes: scratch)
            var arrays: [MLXArray]? = (targetSpecs + auxSpecs).map { MLXArray.zeros($0.0, dtype: $0.1) }
            try withError { eval(arrays!); StreamOrDevice.default.stream.synchronize() }
            _ = try CBv2CheckpointAllocationFootprint.freshBytes(arrays!)
            try stage.settleDestinationAfterEvaluation(targetBytes: target, auxiliaryBytes: auxiliary)
            let ticket = try admission.transferContiguousCheckpointStage(stage, requestID: .init(806), maximumTokens: maximum)
            ticket.commit()
            XCTAssertEqual(admission.bytesReserved, ordinary + excess + auxiliary + scratch)
            XCTAssertEqual(admission.nonBackendBytesReserved, fixed + auxiliary + scratch)
            XCTAssertEqual(admission.targetBytesReserved(partitionedBy: [.init(806)]).materialized, nominalTarget + excess)
            stage.closeAfterDroppingOwners() // destination moved; scratch only
            XCTAssertEqual(admission.bytesReserved, ordinary + excess + auxiliary)
            let detached = admission.detachReservation(id: .init(806))
            XCTAssertEqual(admission.nonBackendBytesReserved, fixed + auxiliary)
            XCTAssertEqual(admission.bytesReserved, ordinary + excess + auxiliary)
            if releaseLeaseFirst {
                detached.release()
                XCTAssertEqual(admission.bytesReserved, ordinary + excess + auxiliary)
                arrays = nil
                admission.retireContiguousCheckpointRows(id: .init(806), owner: stage.identity)
            } else {
                arrays = nil
                admission.retireContiguousCheckpointRows(id: .init(806), owner: stage.identity)
                XCTAssertEqual(admission.bytesReserved, ordinary + excess + auxiliary)
                detached.release()
            }
            XCTAssertEqual(admission.bytesReserved, 0)
            XCTAssertEqual(admission.nonBackendBytesReserved, 0)
            XCTAssertEqual(process.bytes, 0) // fixture ledger zero, not physical-free proof
        }
    }
}
