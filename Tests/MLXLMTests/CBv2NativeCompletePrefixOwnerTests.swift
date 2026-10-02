import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Real Common native graphs/rows/issuer/async store jobs, not a MiMo model or
/// artifact qualification. Strict-loaded MiMo factory and MTP codec tests are
/// separate integration gates. Every retained-fault selector needs a fresh process.
final class CBv2NativeCompletePrefixOwnerTests: XCTestCase {
    private enum Failure: Error { case invalidated, capacity, fence, noReceipt }

    private final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt64 = 0
        private var closed = false
        private var materialized: UInt64 = 0
        var bytes: UInt64 { lock.withLock { value } }
        var retired: Bool { lock.withLock { closed } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= 256 << 20 else {
                    throw Failure.capacity
                }
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
        func retire() {
            lock.withLock {
                XCTAssertEqual(value, 0)
                closed = true
            }
        }
    }

    /// Same real asymmetric attention pattern used by the existing historical
    /// checkpoint fixture, with immutable materialized native input and a
    /// package-only generation validator. This is not an NSObject stand-in.
    private final class Model: CBv2SteppableModel, CBv2HistoricalAttentionCheckpointProviding,
        CBv2CompleteCheckpointKVTypeProviding, CBv2NativeCompletePrefixBindingValidating
    {
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
            let b = tokens.dim(0)
            let n = tokens.dim(1)
            var hidden = tokens.asType(.float32).reshaped([b, 1, n, 1]) * loadedScale
            for cache in caches {
                let q = MLXArray.zeros([b, 2, n, 192], dtype: .float32)
                let k = broadcast(hidden, to: [b, 1, n, 192])
                let v = broadcast(hidden, to: [b, 1, n, 128])
                hidden = mean(
                    cache.updateAndAttend(
                        queries: q, keys: k, values: v,
                        scale: 0.125, sinks: nil), axes: [1, 3], keepDims: true)
            }
            let target = MLX.round(hidden.reshaped([b, n, 1]) * Float(128)).asType(.int32) % 31
            return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
        }
    }

    /// Observation only: the actual owners below are a native MLXArray and a
    /// native KV row. This box owns neither, and is not a completion receipt.
    private final class RetirementRootWitness {
        weak var array: MLXArray?
        weak var row: CBv2WindowedSequenceKV?
        init(array: MLXArray, row: CBv2WindowedSequenceKV) {
            self.array = array
            self.row = row
        }
    }

    /// A real function return ends every setup-local strong alias before the
    /// queued finish begins. Do not return an array/row/tuple of strong roots.
    @inline(never)
    private func retainEvaluatedRetirementRoots(_ work: CBv2NativeCompletePrefixWork) throws
        -> RetirementRootWitness
    {
        try work.captureCurrentStreams()
        let array = (MLXArray(Int32(0) ..< Int32(1024)).asType(.float32) + Float(1))
        let row = CBv2WindowedSequenceKV(window: 17, kvHeads: 1, headDim: 192, valueHeadDim: 128)
        // The real counted work owns both graph and row before evaluation.
        try work.retain(arrays: [array], owners: [row])
        let pair = row.update(
            keys: MLXArray.ones([1, 1, 3, 192], dtype: .float32),
            values: MLXArray.ones([1, 1, 3, 128], dtype: .float32))
        try withError { errors in
            eval(array, pair.0, pair.1)
            try errors.check()
        }
        return .init(array: array, row: row)
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
            guard
                lock.withLock({
                    let old = first
                    first = false
                    return old
                })
            else { return }
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
        init(
            writeGate: CheckpointPublicationGate? = nil,
            archives: [CompleteCheckpointFixtureStore.Archive] = []
        ) {
            base = .init(archives: archives, gate: writeGate, segmentBytes: 1 << 20)
        }
        var identity: CBv2CompleteCheckpointIdentity { base.identity }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
            base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
        }
        func takeStaged(
            requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
            maximumSequenceLength: Int
        ) -> CBv2StagedCompleteCheckpoint? {
            if let actual = lock.withLock({ supplied.removeValue(forKey: requestID) }) {
                return actual
            }
            return base.takeStaged(
                requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                maximumSequenceLength: maximumSequenceLength)
        }
        func supplyActualForeignStage(_ stage: CBv2StagedCompleteCheckpoint, receipt: CBv2RequestID)
        {
            lock.withLock {
                precondition(supplied[receipt] == nil)
                supplied[receipt] = stage
            }
        }
        func donate(
            _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
            tokens: [Int], cacheSalt: String?,
            completion: @escaping @Sendable ([Int]) -> Void
        ) {
            jobs.enter()
            base.donate(source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt) {
                [self] positions in
                completion(positions)
                jobs.leave()  // real underlying source read/close callback returned.
            }
        }
        func close() {
            let stages = lock.withLock {
                let old = Array(supplied.values)
                supplied = [:]
                return old
            }
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

    private func fixture(
        store: Store = Store(), shutdownTimeout: TimeInterval = 10,
        beforeConsume: (
            (
                CBv2NativeExecutionContract, Model, CBv2ContiguousKVBackend,
                CBv2LayerCacheBank, ProcessOwner
            ) throws -> Void
        )? = nil
    ) throws -> Fixture {
        let scope = NativeConstructionScope()
        let model = Model()
        let process = ProcessOwner()
        let kinds: [CBv2LayerKind] = [
            .init(
                attention: .slidingWindow(17), headDim: 192, valueHeadDim: 128, kvHeads: 1,
                queryHeads: 2),
            .init(attention: .full, headDim: 192, valueHeadDim: 128, kvHeads: 1, queryHeads: 2),
        ]
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 128 << 20, kvDType: .float32))
        let bank = CBv2LayerCacheBank(layerKinds: kinds)
        let engine = try scope.withPhase(.nativeSetup) {
            try scope.capture(StreamOrDevice.cpu.stream)
            try scope.capture(StreamOrDevice.default.stream)
            try scope.retain(model.loadedScale)
            try scope.willSubmit()
            try withError { errors in
                eval(model.loadedScale)
                try errors.check()
            }
            try scope.authorizeImmutableLoadedOwner(model)
            try scope.retainOwner(backend)
            try scope.retainOwner(bank)
            try scope.retainOwner(store)
            let contract = try CBv2NativeExecutionContract(
                model: model, backend: backend,
                cacheProvider: bank, assistant: nil, construction: scope, loadedOwner: model,
                completePrefixCache: store, completePrefixValidator: model,
                prefixProcessMemoryOwner: process)
            try beforeConsume?(contract, model, backend, bank, process)
            let engine = EngineV2(
                model: model, layerKinds: kinds, backend: backend, cacheProvider: bank,
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: chunk,
                    prefillChunkSize: chunk, enablePrefixCache: true),
                loopConfig: .init(
                    stepTimeout: 60, watchdogInterval: 0.01, shutdownTimeout: shutdownTimeout),
                admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
                processMemoryOwner: process, nativeCompletionTracking: true,
                nativeExecutionContract: contract)
            try scope.retainOwner(engine)
            return engine
        }
        XCTAssertNil(engine.nativeCompletionFault)
        return .init(
            engine: engine, backend: backend, model: model, process: process, store: store,
            scope: scope)
    }

    private func request(_ id: UInt64) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: (0 ..< (3 * chunk + 1)).map { ($0 * 7) % 29 },
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

    func testProtectedIssuedTicketRejectsForeignActualStoreAndProcessOwnerWithoutConsumption()
        async throws
    {
        try lane()
        let foreignStore = Store()
        let foreignProcess = ProcessOwner()
        let localStore = Store()
        let f = try fixture(store: localStore) { contract, model, backend, bank, process in
            XCTAssertFalse(
                contract.consume(
                    model: model, backend: backend, cacheProvider: bank,
                    assistant: nil, completePrefixCache: foreignStore, processMemoryOwner: process))
            XCTAssertFalse(
                contract.consume(
                    model: model, backend: backend, cacheProvider: bank,
                    assistant: nil, completePrefixCache: localStore,
                    processMemoryOwner: foreignProcess))
            XCTAssertFalse(
                contract.consume(
                    model: model, backend: backend, cacheProvider: bank,
                    assistant: nil, completePrefixCache: Store(), processMemoryOwner: process))
        }
        let result = await cbv2SchedCollect(try f.engine.submit(request(101)))
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, 3)
        XCTAssertFalse(f.store.base.saved.isEmpty)
        try await shutDown(f)
    }

    func testBlockedPublicationAndActualCloseJoinRetainRowsAndRequestRetirement() async throws {
        try lane()
        let writeGate = CheckpointPublicationGate()
        let store = Store(writeGate: writeGate)
        let closeEntered = expectation(description: "actual store close entered")
        let closeGate = Gate(closeEntered)
        store.closeGate = closeGate
        let f = try fixture(store: store)
        let submitted = try f.engine.submitWithNativeRetirement(request(102))
        let retired = Flag()
        let retirement = Task {
            await submitted.retirement.wait()
            retired.set()
        }
        let collector = Task { await cbv2SchedCollect(submitted.events) }
        XCTAssertTrue(writeGate.waitUntilEntered())
        XCTAssertGreaterThan(f.backend.bytesReserved, 0)
        XCTAssertGreaterThan(f.process.bytes, 0)
        XCTAssertFalse(retired.value)
        let shutdownEngine = f.engine
        let shutdown = Task { await shutdownEngine.shutdownReportingNativeCompletion() }
        await fulfillment(of: [closeEntered], timeout: 3)
        XCTAssertFalse(store.closeJoined.value)
        XCTAssertFalse(retired.value)
        XCTAssertGreaterThan(f.backend.bytesReserved, 0)
        closeGate.release()  // real close cancels/unblocks the pending store write.
        let result = await collector.value
        await retirement.value
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, 3)
        guard case .quiescent = await shutdown.value else { throw Failure.noReceipt }
        XCTAssertTrue(store.closeJoined.value)
        XCTAssertTrue(retired.value)
        XCTAssertEqual(f.backend.bytesReserved, 0)
        XCTAssertEqual(f.process.bytes, 0)
    }

    func testIdleEngineWaitsForActualStoreCloseJoinBeforeQuiescentOutcome() async throws {
        try lane()
        let store = Store()
        let entered = expectation(description: "idle store closeAndWait entered")
        let gate = Gate(entered)
        store.closeGate = gate
        defer { gate.release() }
        let f = try fixture(store: store)
        let engine = f.engine
        // Never submit a request: no publication/import/drop/terminal consumer
        // can independently mask a missing store-close join.
        try engine.loopForTesting.onEngineQueueSync {
            let tracking = try XCTUnwrap(engine.loopForTesting.nativeShutdownState)
            XCTAssertNil(tracking.outcome)
            XCTAssertFalse(tracking.hasLoans)
            XCTAssertEqual(tracking.debugRetainedRootCount, 0)
            XCTAssertEqual(engine.stepCount, 0)
            XCTAssertEqual(f.backend.bytesReserved, 0)
            XCTAssertEqual(f.process.bytes, 0)
            XCTAssertFalse(f.process.retired)
        }
        XCTAssertTrue(store.base.saved.isEmpty)
        let shutdown = Task { await engine.shutdownReportingNativeCompletion() }
        await fulfillment(of: [entered], timeout: 3)
        // The first barrier follows the initial drain frame; the next also
        // follows a completion callback queued by that frame. Inspect the
        // actual outcome, not whether the awaiting test Task has resumed.
        engine.loopForTesting.onEngineQueueSync {}
        let held = try engine.loopForTesting.onEngineQueueSync {
            let tracking = try XCTUnwrap(engine.loopForTesting.nativeShutdownState)
            return (
                outcome: tracking.outcome, loan: tracking.hasLoans,
                processRetired: f.process.retired
            )
        }
        XCTAssertNil(held.outcome, "closeAndWait is the only unfinished consumer")
        XCTAssertTrue(held.loan, "the actual store-close native loan must remain live")
        XCTAssertFalse(held.processRetired)
        XCTAssertFalse(store.closeJoined.value)
        XCTAssertEqual(f.backend.bytesReserved, 0)
        XCTAssertEqual(f.process.bytes, 0)
        gate.release()
        let outcome = await shutdown.value
        guard case .quiescent(let receipt) = outcome else {
            _ = Unmanaged.passRetained(engine)
            throw Failure.noReceipt
        }
        XCTAssertEqual(receipt.engineID, engine.nativeShutdownEngineID)
        XCTAssertEqual(
            receipt.executionContractID, try XCTUnwrap(engine.nativeShutdownExecutionContractID))
        XCTAssertTrue(store.closeJoined.value)
        try engine.loopForTesting.onEngineQueueSync {
            let tracking = try XCTUnwrap(engine.loopForTesting.nativeShutdownState)
            XCTAssertEqual(tracking.outcome, outcome)
            XCTAssertFalse(tracking.hasLoans)
            XCTAssertEqual(tracking.debugRetainedRootCount, 0)
        }
        XCTAssertTrue(f.process.retired)
        XCTAssertEqual(f.process.bytes, 0)
        XCTAssertEqual(f.backend.bytesReserved, 0)
    }

    func testCancellationAfterRealCaptureDrainReleasesOnlyAfterLoanCompletion() async throws {
        try lane()
        let f = try fixture()
        let entered = expectation(description: "required capture drain")
        let gate = Gate(entered)
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.completeCheckpointCapture?.makeContiguousCheckpoint = {
                codec, position, chunk, state in
                let value = try CBv2ContiguousHistoricalCheckpoint(
                    codec: codec, position: position,
                    chunkSize: chunk, state: state)
                value.beforeRequiredDrainForTesting = { gate.hold() }
                return value
            }
        }
        let submitted = try f.engine.submitWithNativeRetirement(request(103))
        let collector = Task { await cbv2SchedCollect(submitted.events) }
        let retired = Flag()
        let retirement = Task {
            await submitted.retirement.wait()
            retired.set()
        }
        await fulfillment(of: [entered], timeout: 3)
        f.engine.cancel(.init(103))
        XCTAssertFalse(retired.value)
        XCTAssertGreaterThan(f.process.bytes, 0)
        gate.release()
        let result = await collector.value
        await retirement.value
        XCTAssertEqual(result.finishReason, .cancelled)
        XCTAssertEqual(f.store.base.saved.count, 0)
        try await shutDown(f)
    }

    func testImportOwnerRejectsForeignCodecRequestStoreAndStaleGenerationBeforeNativeAdoption()
        async throws
    {
        try lane()
        let a = try fixture()
        let b = try fixture()
        let req = request(104)
        let codec = try XCTUnwrap(a.engine.completeCheckpointCodec)
        let factory = try XCTUnwrap(
            a.engine.nativeCompletePrefixWorkFactory(
                store: a.store, codec: codec, request: req))
        let work = try factory()
        try work.validate(
            store: a.store, codec: codec, request: req, engineID: a.engine.nativeShutdownEngineID)
        XCTAssertThrowsError(
            try work.validate(
                store: b.store, codec: codec,
                request: req, engineID: a.engine.nativeShutdownEngineID))
        XCTAssertThrowsError(
            try work.validate(
                store: a.store, codec: XCTUnwrap(b.engine.completeCheckpointCodec),
                request: req, engineID: a.engine.nativeShutdownEngineID))
        var changed = req
        changed.promptTokens[0] += 1
        XCTAssertThrowsError(
            try work.validate(
                store: a.store, codec: codec,
                request: changed, engineID: a.engine.nativeShutdownEngineID))
        changed = req
        changed.prefixCacheReceiptID = .init(99999)
        XCTAssertThrowsError(
            try work.validate(
                store: a.store, codec: codec,
                request: changed, engineID: a.engine.nativeShutdownEngineID))
        let released = expectation(description: "real native-work retirement callback")
        let reservation = try codec.admission.reserveTransient(bytes: 1 << 20)
        var values: MLXArray? = MLXArray.zeros([1024], dtype: .float32)
        try work.captureCurrentStreams()
        try work.retain(arrays: [try XCTUnwrap(values)], owners: [reservation])
        try withError { errors in
            eval(try XCTUnwrap(values))
            try errors.check()
        }
        a.model.invalidate()
        XCTAssertThrowsError(
            try work.validate(
                store: a.store, codec: codec,
                request: req, engineID: a.engine.nativeShutdownEngineID))
        values = nil
        XCTAssertTrue(
            work.finishAfterDroppingConsumers {
                reservation.release()
                released.fulfill()
            })
        await fulfillment(of: [released], timeout: 3)
        XCTAssertFalse(work.finishAfterDroppingConsumers())
        try await shutDown(a)
        try await shutDown(b)
    }

    func testActualForeignImportRefusesThenSameEngineArchiveRestoresExactContinuation() async throws
    {
        try lane()
        let seed = try fixture()
        let cold = await cbv2SchedCollect(try seed.engine.submit(request(200)))
        XCTAssertEqual(cold.finishReason, .length)
        let archives = seed.store.base.saved.filter { $0.manifest.position == chunk }
        XCTAssertEqual(archives.count, 1)
        try await shutDown(seed)
        let a = try fixture(store: Store(archives: archives))
        let b = try fixture(store: Store(archives: archives))
        let foreignRequest = request(201)
        XCTAssertTrue(try a.store.base.stage(engine: a.engine, request: foreignRequest))
        XCTAssertTrue(
            a.engine.loopForTesting.onEngineQueueSync {
                a.engine.loopForTesting.nativeShutdownState?.hasLoans == true
            }, "actual staged native import must hold a native operation loan")
        var foreignStage: CBv2StagedCompleteCheckpoint? = try XCTUnwrap(
            a.store.takeStaged(
                requestID: XCTUnwrap(foreignRequest.prefixCacheReceiptID),
                tokens: foreignRequest.promptTokens, cacheSalt: foreignRequest.checkpointCacheSalt,
                maximumSequenceLength: foreignRequest.promptTokens.count + foreignRequest.maxTokens)
        )
        XCTAssertTrue(
            try XCTUnwrap(foreignStage).withValidatedNativeCodec(
                store: a.store, request: foreignRequest, engineID: a.engine.nativeShutdownEngineID,
                expectedCodec: XCTUnwrap(a.engine.completeCheckpointCodec)
            ) { codec in
                codec === a.engine.completeCheckpointCodec
            })
        b.store.supplyActualForeignStage(
            try XCTUnwrap(foreignStage),
            receipt: try XCTUnwrap(foreignRequest.prefixCacheReceiptID))
        let rejected = await cbv2SchedCollect(try b.engine.submit(foreignRequest))
        XCTAssertEqual(rejected.finishReason, .length)
        XCTAssertEqual(rejected.tokens, cold.tokens)
        XCTAssertEqual(rejected.usage?.prefixCachePrefillTokensSaved, 0)
        let released = await cbv2SchedWait { a.store.base.releaseCount == 1 }
        XCTAssertTrue(released, "actual foreign native stage callback must retire once")
        let importLoanEnded = await cbv2SchedWait {
            a.engine.loopForTesting.nativeShutdownState?.hasLoans == false
        }
        XCTAssertTrue(importLoanEnded)
        // The native loan is retired, but this closed handle still owns its
        // host manifest permit. End that lifetime before final-zero shutdown.
        foreignStage = nil
        let validRequest = request(202)
        XCTAssertTrue(try a.store.base.stage(engine: a.engine, request: validRequest))
        XCTAssertTrue(a.engine.loopForTesting.nativeShutdownState?.hasLoans == true)
        let accepted = await cbv2SchedCollect(try a.engine.submit(validRequest))
        XCTAssertEqual(accepted.tokens, cold.tokens)
        XCTAssertEqual(accepted.usage?.prefixCachePrefillTokensSaved, chunk)
        XCTAssertEqual(accepted.usage?.prefixCacheReplayTokens, 0)
        try await shutDown(a)
        try await shutDown(b)
    }

    /// The provider bridge stages with a placeholder engine ID and mints the
    /// real one just before submit. The stage must still adopt for the
    /// submission that owns its receipt, with the exact cold continuation.
    func testStageUnderPlaceholderEngineIDAdoptsForTheSubmissionReceipt() async throws {
        try lane()
        let seed = try fixture()
        let cold = await cbv2SchedCollect(try seed.engine.submit(request(300)))
        XCTAssertEqual(cold.finishReason, .length)
        let archives = seed.store.base.saved.filter { $0.manifest.position == chunk }
        XCTAssertEqual(archives.count, 1)
        try await shutDown(seed)
        let a = try fixture(store: Store(archives: archives))
        let submitted = request(301)
        var placeholder = submitted
        placeholder.id = CBv2RequestID(0)
        XCTAssertTrue(try a.store.base.stage(engine: a.engine, request: placeholder))
        let accepted = await cbv2SchedCollect(try a.engine.submit(submitted))
        XCTAssertEqual(accepted.finishReason, .length)
        XCTAssertEqual(accepted.tokens, cold.tokens)
        XCTAssertEqual(accepted.usage?.prefixCacheOutcome, .hit)
        XCTAssertEqual(accepted.usage?.prefixCachePrefillTokensSaved, chunk)
        XCTAssertEqual(accepted.usage?.prefixCacheReplayTokens, 0)
        try await shutDown(a)
    }

    /// The receipt is the binding: the same engine ID under another
    /// submission's receipt is refused, a placeholder engine ID under the
    /// stage's own receipt validates, and the owner still adopts exactly.
    func testStageBindsToItsSubmissionReceiptRatherThanTheEngineID() async throws {
        try lane()
        let seed = try fixture()
        let cold = await cbv2SchedCollect(try seed.engine.submit(request(310)))
        let archives = seed.store.base.saved.filter { $0.manifest.position == chunk }
        XCTAssertEqual(archives.count, 1)
        try await shutDown(seed)
        let a = try fixture(store: Store(archives: archives))
        let staged = request(311)
        XCTAssertTrue(try a.store.base.stage(engine: a.engine, request: staged))
        let receipt = try XCTUnwrap(staged.prefixCacheReceiptID)
        var stage: CBv2StagedCompleteCheckpoint? = try XCTUnwrap(
            a.store.takeStaged(
                requestID: receipt, tokens: staged.promptTokens,
                cacheSalt: staged.checkpointCacheSalt,
                maximumSequenceLength: staged.promptTokens.count + staged.maxTokens))
        let codec = try XCTUnwrap(a.engine.completeCheckpointCodec)
        var otherSubmission = staged
        otherSubmission.prefixCacheReceiptID = CBv2RequestID(9_999)
        XCTAssertThrowsError(
            try XCTUnwrap(stage).withValidatedNativeCodec(
                store: a.store, request: otherSubmission,
                engineID: a.engine.nativeShutdownEngineID, expectedCodec: codec
            ) { _ in () })
        var placeholder = staged
        placeholder.id = CBv2RequestID(0)
        XCTAssertNoThrow(
            try XCTUnwrap(stage).withValidatedNativeCodec(
                store: a.store, request: placeholder,
                engineID: a.engine.nativeShutdownEngineID, expectedCodec: codec
            ) { _ in () })
        a.store.supplyActualForeignStage(try XCTUnwrap(stage), receipt: receipt)
        let accepted = await cbv2SchedCollect(try a.engine.submit(staged))
        XCTAssertEqual(accepted.tokens, cold.tokens)
        XCTAssertEqual(accepted.usage?.prefixCachePrefillTokensSaved, chunk)
        // The consumed handle still owns its host manifest permit. End that
        // lifetime before final-zero shutdown.
        stage = nil
        try await shutDown(a)
    }

    func testPublicationReadbackTemporariesStayBoundedUntilRealWholeWorkRetirement() async throws {
        try lane()
        let f = try fixture()
        let work = try f.engine.loopForTesting.onEngineQueueSync {
            try f.engine.loopForTesting.makeNativeCompletePrefixWork(
                purpose: .publication,
                requestID: .init(203))
        }
        let codec = try XCTUnwrap(f.engine.completeCheckpointCodec)
        let reservation = try codec.admission.reserveTransient(bytes: 1 << 20)
        try work.retain(owners: [reservation])
        let held = f.process.bytes
        for _ in 0 ..< 8 {
            var temporary: MLXArray? = MLXArray.zeros([1024], dtype: .float32)
            try work.captureCurrentStreams()
            try work.retain(arrays: [try XCTUnwrap(temporary)])
            try withError { errors in
                eval(try XCTUnwrap(temporary))
                _ = try XCTUnwrap(temporary).asData(access: .copy).data
                try errors.check()
            }
            XCTAssertEqual(work.debugRetainedArrayCount, 1)
            try work.retireReadbackTemporaries([try XCTUnwrap(temporary)])
            temporary = nil
            XCTAssertEqual(work.debugRetainedArrayCount, 0)
            XCTAssertEqual(f.process.bytes, held, "readback never refunds whole-work C")
        }
        let retired = expectation(description: "whole publication scratch retired")
        XCTAssertTrue(
            work.finishAfterDroppingConsumers {
                reservation.release()
                retired.fulfill()
            })
        await fulfillment(of: [retired], timeout: 3)
        try await shutDown(f)
    }

    func testTypedRequiredFenceFailureRetainsActualRootsAndCreditAfterLaterDrain() async throws {
        try lane(fault: "typed-fence")
        let f = try fixture()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.completeCheckpointCapture?.makeContiguousCheckpoint = {
                codec, position, chunk, state in
                let value = try CBv2ContiguousHistoricalCheckpoint(
                    codec: codec, position: position,
                    chunkSize: chunk, state: state)
                value.beforeRequiredDrainForTesting = { throw Failure.fence }
                return value
            }
        }
        let submitted = try f.engine.submitWithNativeRetirement(request(105))
        _ = await cbv2SchedCollect(submitted.events)
        guard case .incomplete(let first) = await f.engine.shutdownReportingNativeCompletion()
        else {
            throw Failure.noReceipt
        }
        XCTAssertEqual(first.reason, .nativeWorkFailed)
        XCTAssertGreaterThan(f.process.bytes, 0)
        XCTAssertGreaterThan(f.backend.bytesReserved, 0)
        try withError { errors in
            Stream.cpu.synchronize()
            Stream.gpu.synchronize()
            try errors.check()
        }
        guard case .incomplete(let later) = await f.engine.shutdownReportingNativeCompletion()
        else {
            throw Failure.noReceipt
        }
        XCTAssertEqual(first, later)
        XCTAssertGreaterThan(f.process.bytes, 0)
        // The real retirement token must NOT be awaited/claimed complete.
        _ = Unmanaged.passRetained(f.engine)
    }
    func testQueuedRetirementDropsActualArrayAndRowOwnerBeforeCreditCallback() async throws {
        try lane()
        let f = try fixture()
        let engine = f.engine
        let process = f.process
        let work = try engine.loopForTesting.onEngineQueueSync {
            try engine.loopForTesting.makeNativeCompletePrefixWork(
                purpose: .publication,
                requestID: .init(204))
        }
        let codec = try XCTUnwrap(engine.completeCheckpointCodec)
        let reservation = try codec.admission.reserveTransient(bytes: 1 << 20)
        try work.retain(owners: [reservation])
        let witness: RetirementRootWitness
        do { witness = try retainEvaluatedRetirementRoots(work) } catch {
            work.requiredCompletionFailed()
            _ = Unmanaged.passRetained(engine)
            _ = Unmanaged.passRetained(work)
            throw error
        }
        XCTAssertTrue(witness.array != nil)
        XCTAssertTrue(witness.row != nil)
        XCTAssertEqual(work.debugRetainedArrayCount, 1)
        let charged = process.bytes
        XCTAssertGreaterThan(charged, 0)
        let returned = expectation(description: "actual queued work retirement callback returned")
        let callbackReached = Flag()
        XCTAssertTrue(
            work.finishAfterDroppingConsumers {
                // This executes on the actual completion/engine queue after the
                // required stream fences, BEFORE this callback releases C and
                // BEFORE finishAfterDroppingConsumers ends the native loan/wakes.
                // No onEngineQueueSync here: it forbids self-queue dispatch.
                let arrayGone = witness.array == nil
                let rowGone = witness.row == nil
                XCTAssertTrue(
                    arrayGone, "a cleared arrays.count cannot hide a detached COW root alias")
                XCTAssertTrue(
                    rowGone, "the real native KV owner must be gone before retirement credit")
                XCTAssertEqual(work.debugRetainedArrayCount, 0)
                XCTAssertEqual(
                    process.bytes, charged, "credit is still owned at the weak-root observation")
                XCTAssertTrue(engine.loopForTesting.nativeShutdownState?.hasLoans == true)
                // An oracle failure must not release the genuine charge. The real
                // work's existing throwing-retirement path keeps its owner/fault.
                guard arrayGone, rowGone else { throw Failure.noReceipt }
                reservation.release()
                callbackReached.set()
                returned.fulfill()
            })
        await fulfillment(of: [returned], timeout: 3)
        guard callbackReached.value else {
            _ = Unmanaged.passRetained(engine)
            _ = Unmanaged.passRetained(work)
            throw Failure.noReceipt
        }
        XCTAssertTrue(witness.array == nil)
        XCTAssertTrue(witness.row == nil)
        XCTAssertFalse(
            work.finishAfterDroppingConsumers(), "one-shot work cannot repeat retirement")
        // Keep the actual work handle alive through the observation; success
        // must come from explicit root movement, not the work object's deinit.
        withExtendedLifetime(work) {}
        try await shutDown(f)
    }

}
