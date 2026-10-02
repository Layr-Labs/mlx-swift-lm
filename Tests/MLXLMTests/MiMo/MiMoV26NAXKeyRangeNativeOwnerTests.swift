// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Real Common native execution contracts, process-owner charging, cache rows,
/// grouped shader dispatch and completion. This deliberately small attention
/// model is NOT a strict-loaded MiMo/artifact or full-model qualification.
final class MiMoV26NAXKeyRangeNativeOwnerTests: XCTestCase {
    private enum Failure: Error { case capacity, fence, gateTimeout, missingCompletion }

    private final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt64 = 0, materialized: UInt64 = 0
        private var frozenCeiling: UInt64?
        private var closed = false
        private var refusals = 0
        var bytes: UInt64 { lock.withLock { value } }
        var rejected: Int { lock.withLock { refusals } }
        var retired: Bool { lock.withLock { closed } }
        func refuseNewGrowth() { lock.withLock { frozenCeiling = value } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= (1 << 30),
                    frozenCeiling.map({ bytes <= $0 }) ?? true
                else {
                    refusals += 1
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

    private final class Store: CBv2NativeCompletePrefixCache, @unchecked Sendable {
        let base = CompleteCheckpointFixtureStore(segmentBytes: 1 << 20)
        private let jobs = DispatchGroup()
        var identity: CBv2CompleteCheckpointIdentity { base.identity }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool { false }
        func takeStaged(
            requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
            maximumSequenceLength: Int
        ) -> CBv2StagedCompleteCheckpoint? { nil }
        func donate(
            _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
            tokens: [Int], cacheSalt: String?,
            completion: @escaping @Sendable ([Int]) -> Void
        ) {
            jobs.enter()
            base.donate(source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt) {
                [self] result in
                completion(result)
                jobs.leave()
            }
        }
        func close() { base.close() }
        func closeAndWait() async {
            close()
            await withCheckedContinuation { continuation in
                jobs.notify(queue: .global(qos: .utility)) { continuation.resume() }
            }
        }
    }

    private final class Model: CBv2SteppableModel, CBv2HistoricalAttentionCheckpointProviding,
        CBv2CompleteCheckpointKVTypeProviding, CBv2NativeCompletePrefixBindingValidating,
        MiMoV26BlockBatchAllocatingModel
    {
        let loadedScale = MLXArray(Float(1.0 / 97.0))
        weak var backend: CBv2ContiguousKVBackend?
        weak var bank: CBv2LayerCacheBank?
        let cache: CBv2LayerCache
        var verifyOutput: ((MLXArray, CBv2AttendingLayerCache, MLXArray) -> Void)?
        init(kind: CBv2LayerKind) {
            cache = .init(layerIndex: 0, kind: kind, mimoV26NAXAttention: true)
        }
        var cbv2SupportsHistoricalAttentionCheckpoint: Bool { true }
        var cbv2CompleteCheckpointKVDTypes: [DType]? { [.bfloat16] }
        func validateNativeCompletePrefixBinding() throws {}
        var cbv2MiMoBlockBatchLayerCount: Int? { 1 }
        func cbv2TryInstallBlockBatchBudget(_ budget: MiMoV26BlockBatchBudget) -> Bool {
            guard let backend, let bank, cache.mimoV26BlockBatchBudget == nil,
                budget.modelIdentity == ObjectIdentifier(self),
                budget.backendIdentity == ObjectIdentifier(backend),
                budget.cacheProviderIdentity == ObjectIdentifier(bank)
            else { return false }
            cache.mimoV26BlockBatchBudget = budget
            return true
        }
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            let b = tokens.dim(0)
            let n = tokens.dim(1)
            let hidden = (tokens.asType(.float32) * loadedScale).asType(.bfloat16)
                .reshaped([b, 1, n, 1])
            // Nontrivial, deterministic native BF16 scores and values. Views
            // intentionally carry zero inner strides into the real cache path.
            let q = broadcast(hidden + Float(0.25), to: [b, 64, n, 192])
            let k = broadcast(hidden - Float(0.125), to: [b, 4, n, 192])
            let v = broadcast(hidden, to: [b, 4, n, 128])
            let attended = caches[0].updateAndAttend(
                queries: q, keys: k, values: v,
                scale: Float(1.0 / sqrt(192.0)), sinks: nil)
            verifyOutput?(q, caches[0], attended)
            let summary = mean(attended.asType(.float32), axes: [1, 3])
            let target = MLX.round(summary * Float(64)).asType(.int32).reshaped([b, n, 1]) % 31
            return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
        }
    }

    private final class Witness: @unchecked Sendable {
        private let lock = NSLock()
        private var owners: [MiMoV26NAXKeyRangeWork] = []
        private var retirement = false
        private var fenceBytes = 0, fenceArrays = 0
        weak var firstArray: MLXArray?
        func add(_ work: MiMoV26NAXKeyRangeWork) { lock.withLock { owners.append(work) } }
        var work: [MiMoV26NAXKeyRangeWork] { lock.withLock { owners } }
        func retired() { lock.withLock { retirement = true } }
        var didRetire: Bool { lock.withLock { retirement } }
        func observedFence(_ work: MiMoV26NAXKeyRangeWork) {
            lock.withLock {
                fenceBytes = work.reservedBytes
                fenceArrays = work.retainedArrayCount
                firstArray = work.evaluationTargets.first {
                    $0.dtype == .float32 && $0.size == 4 * 64 * 2 * 64 * 130
                }
                XCTAssertNotNil(
                    firstArray, "witness must name an actual full FP32 range-state allocation")
            }
        }
        var held: (bytes: Int, arrays: Int) { lock.withLock { (fenceBytes, fenceArrays) } }
    }

    private final class Gate: @unchecked Sendable {
        let entered: XCTestExpectation
        private let semaphore = DispatchSemaphore(value: 0)
        init(_ entered: XCTestExpectation) { self.entered = entered }
        func hold() throws {
            entered.fulfill()
            guard semaphore.wait(timeout: .now() + 10) == .success else {
                throw Failure.gateTimeout
            }
        }
        func release() { semaphore.signal() }
    }

    private final class Proof: @unchecked Sendable {
        var equal: MLXArray?, ulp: MLXArray?
        private(set) var bitExact: Bool?, maximumULP: Int32?
        func readAndDropNativeAliases() throws {
            defer {
                equal = nil
                ulp = nil
            }
            try withError { errors in
                bitExact = equal?.item(Bool.self)
                maximumULP = ulp?.item(Int32.self)
                try errors.check()
            }
        }
    }

    private struct Fixture {
        let engine: EngineV2
        let backend: CBv2ContiguousKVBackend
        let process: ProcessOwner
        let store: Store
        let model: Model
        let scope: NativeConstructionScope
    }

    private func lane(fault: String? = nil) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["DARKBLOOM_TEST_NATIVE_KEY_RANGES"] == "1",
            env["DARKBLOOM_MIMO_V26_NAX_KEY_RANGES"] == "1",
            env["DARKBLOOM_MIMO_BLOCK_BATCH_PREFILL"] == "1",
            env["DARKBLOOM_MIMO_V26_NAX_ATTENTION"] == "1"
        else {
            throw XCTSkip("Explicit exclusive native key-range test process required")
        }
        guard env["DARKBLOOM_TEST_NATIVE_KEY_RANGES_FAULT"] == fault else {
            throw XCTSkip("Sticky fault selectors require their own fresh process")
        }
        guard MiMoV26NAXGatherQMM.gpuStream(.default), MiMoV26NAXGatherQMM.naxAvailable else {
            throw XCTSkip("M5 NAX required; M3 fallback is not a kernel pass")
        }
        XCTAssertEqual(CBv2AttentionV1.queryBlockSize, 128)
    }

    private func fixture(tracked: Bool = true) throws -> Fixture {
        let kind = CBv2LayerKind(
            attention: .full, headDim: 192, valueHeadDim: 128,
            kvHeads: 4, queryHeads: 64, modelLayerIndex: 0)
        let model = Model(kind: kind)
        let scope = NativeConstructionScope()
        let process = ProcessOwner()
        let store = Store()
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 768 << 20, kvDType: .bfloat16))
        let bank = CBv2LayerCacheBank(caches: [model.cache])
        model.backend = backend
        model.bank = bank
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
            let contract: CBv2NativeExecutionContract?
            if tracked {
                contract = try CBv2NativeExecutionContract(
                    model: model, backend: backend,
                    cacheProvider: bank, assistant: nil, construction: scope, loadedOwner: model,
                    completePrefixCache: store, completePrefixValidator: model,
                    prefixProcessMemoryOwner: process)
            } else {
                contract = nil
            }
            let engine = EngineV2(
                model: model, layerKinds: [kind], backend: backend, cacheProvider: bank,
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 512,
                    prefillChunkSize: 512, enablePrefixCache: tracked),
                loopConfig: .init(stepTimeout: 120, watchdogInterval: 0.01, shutdownTimeout: 20),
                admissionConfig: .init(watermarkFraction: 0),
                completePrefixCache: tracked ? store : nil,
                processMemoryOwner: tracked ? process : nil,
                nativeCompletionTracking: tracked, nativeExecutionContract: contract)
            try scope.retainOwner(engine)
            return engine
        }
        XCTAssertNil(engine.nativeCompletionFault)
        XCTAssertGreaterThan(engine.groupedPrefillScratchBytes, 0)
        return .init(
            engine: engine, backend: backend, process: process, store: store, model: model,
            scope: scope)
    }

    private func request(_ id: UInt64, maxTokens: Int = 1) -> CBv2Request {
        // With the unchanged round-to-even range policy, the last 512-query
        // group is the FIRST all-four-descriptor R=2 group (keys 12416...12800).
        .init(
            id: .init(id), promptTokens: (0 ..< 12800).map { ($0 * 7 + 3) % 29 },
            sampling: .init(temperature: 0), maxTokens: maxTokens, prefixCacheEnabled: false)
    }

    private func shutdown(_ f: Fixture) async throws {
        guard case .quiescent = await f.engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(f.engine)
            throw Failure.missingCompletion
        }
        XCTAssertEqual(f.process.bytes, 0)
        XCTAssertTrue(f.process.retired)
        XCTAssertEqual(f.backend.bytesReserved, 0)
        XCTAssertTrue(f.store.base.saved.isEmpty)
    }

    func testIssuedNativeDispatchReleasesActualScratchWhileWorkObjectIsStillRetained() async throws
    {
        try lane()
        let f = try fixture()
        let witness = Witness()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.mimoKeyRangeWorkCreatedForTesting = { work in
                witness.add(work)
                work.beforeRequiredDrainForTesting = { [weak work] in
                    if let work { witness.observedFence(work) }
                }
            }
        }
        let before = MiMoV26BlockBatchAttention.keyRangeEncodedDispatches()
        let submitted = try f.engine.submitWithNativeRetirement(request(7001))
        let result = await cbv2SchedCollect(submitted.events)
        await submitted.retirement.wait()
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, 1)
        XCTAssertEqual(MiMoV26BlockBatchAttention.keyRangeEncodedDispatches() - before, 6)
        XCTAssertGreaterThan(witness.held.bytes, 0)
        XCTAssertGreaterThan(witness.held.arrays, 0)
        try f.engine.loopForTesting.onEngineQueueSync {
            let work = try XCTUnwrap(witness.work.first)
            XCTAssertTrue(work.completed)
            XCTAssertTrue(work.released)
            XCTAssertEqual(work.retainedArrayCount, 0)
            XCTAssertEqual(work.reservedBytes, 0)
            XCTAssertFalse(work.completionFailed)
        }
        // Witness still strongly owns work: neither charge nor arrays need deinit.
        XCTAssertEqual(witness.work.count, 1)
        try await shutdown(f)
    }

    func testActualAdmittedGroupedRangesAreBitExactToUnchangedThreePassGroupedBody() async throws {
        try lane()
        let f = try fixture()
        let witness = Witness()
        let proof = Proof()
        let admission = f.engine.admissionForTesting
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        // Test-only reference + contiguous words + bool/ordered-positive ULP
        // reductions. Price each possible distinct allocation BEFORE building
        // this extra oracle graph. The production reservation is not reused.
        let n = 512 * 64 * 128
        let logical = [
            512 * 64 * 192 * 2, n * 2, n * 2, n * 2, n,
            n * 4, n * 4, n * 4, n * 4, 1, 4, 320, 128, 1, 4, 2,
        ]
        var bytes = 128 << 10  // conservative bounded host/control source allowance
        for count in logical {
            let bound = try XCTUnwrap(policy.upperBound(byteCount: count))
            let sum = bytes.addingReportingOverflow(bound)
            XCTAssertFalse(sum.overflow)
            bytes = sum.partialValue
        }
        let referenceBytes = bytes
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.mimoKeyRangeWorkCreatedForTesting = { work in
                witness.add(work)
                work.beforeRequiredDrainForTesting = { try proof.readAndDropNativeAliases() }
            }
            f.model.verifyOutput = { q, cache, actual in
                guard q.dim(2) == 512, cache.rows.first?.absoluteOffset == 12800 else { return }
                guard let work = witness.work.last else {
                    XCTFail("real native owner was not installed")
                    return
                }
                do {
                    let reservation = try admission.reserveTransient(bytes: referenceBytes)
                    work.retain(reservation: reservation, bytes: referenceBytes)
                    let original = cache.rows[0].snapshot()
                    let plan = try XCTUnwrap(
                        MiMoV26BlockBatchAttention.makePlan(
                            queries: q, keys: original.keys, values: original.values,
                            scale: Float(1.0 / sqrt(192.0)), sinks: nil, window: nil,
                            maximumQueries: 512, production: true))
                    // No new range context/helper in this independent reference.
                    let expected = MiMoV26BlockBatchAttention.launch(
                        queries: q,
                        keys: original.keys, values: original.values,
                        scale: Float(1.0 / sqrt(192.0)), sinks: nil, plan: plan)
                    let a = contiguous(actual).view(dtype: .uint16)
                    let b = contiguous(expected).view(dtype: .uint16)
                    let equal = all(a .== b)
                    // This fixture's V and outputs are nonnegative; uint16
                    // word distance is exact ULP distance (including zero).
                    let ulp = MLX.max(abs(a.asType(.int32) - b.asType(.int32)))
                    work.retain(arrays: [expected, a, b, equal, ulp])
                    proof.equal = equal
                    proof.ulp = ulp
                } catch {
                    XCTFail("oracle could not obtain/complete genuine extra admission: \(error)")
                    work.failCompletion()
                }
            }
        }
        let before = MiMoV26BlockBatchAttention.keyRangeEncodedDispatches()
        let submitted = try f.engine.submitWithNativeRetirement(request(7007))
        let result = await cbv2SchedCollect(submitted.events)
        await submitted.retirement.wait()
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, 1)
        XCTAssertEqual(MiMoV26BlockBatchAttention.keyRangeEncodedDispatches() - before, 6)
        XCTAssertEqual(proof.bitExact, true)
        XCTAssertEqual(proof.maximumULP, 0)
        XCTAssertNil(proof.equal)
        XCTAssertNil(proof.ulp)
        print(
            "NAX_KEY_RANGES grouped-native-byte-oracle maxULP=\(String(describing: proof.maximumULP))"
        )
        f.engine.loopForTesting.onEngineQueueSync { f.model.verifyOutput = nil }
        try await shutdown(f)
    }

    func testActualProcessOwnerRefusalFallsBackBeforeEncodingWithExactTargetTokens() async throws {
        try lane()
        let reference = try fixture()
        let baseline = await cbv2SchedCollect(try reference.engine.submit(request(7002)))
        try await shutdown(reference)
        let f = try fixture()
        let witness = Witness()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.mimoKeyRangeWorkCreatedForTesting = { work in
                witness.add(work)
                f.process.refuseNewGrowth()  // real shared owner's replacement refuses EXTRA C.
            }
        }
        let before = MiMoV26BlockBatchAttention.keyRangeEncodedDispatches()
        let submitted = try f.engine.submitWithNativeRetirement(request(7003))
        let result = await cbv2SchedCollect(submitted.events)
        await submitted.retirement.wait()
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens, baseline.tokens)
        XCTAssertEqual(MiMoV26BlockBatchAttention.keyRangeEncodedDispatches(), before)
        XCTAssertGreaterThan(f.process.rejected, 0)
        f.engine.loopForTesting.onEngineQueueSync {
            XCTAssertEqual(witness.work.count, 1)
            XCTAssertTrue(
                witness.work.allSatisfy { $0.released && $0.completed && $0.reservedBytes == 0 })
        }
        try await shutdown(f)
    }

    func testCancellationHeldAtRealRequiredDrainKeepsScratchAndRetirementPending() async throws {
        try lane()
        let f = try fixture()
        let witness = Witness()
        let entered = expectation(description: "actual key-range required drain")
        let gate = Gate(entered)
        defer { gate.release() }
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.mimoKeyRangeWorkCreatedForTesting = { work in
                witness.add(work)
                work.beforeRequiredDrainForTesting = { [weak work] in
                    if let work { witness.observedFence(work) }
                    try gate.hold()
                }
            }
        }
        let submitted = try f.engine.submitWithNativeRetirement(request(7004, maxTokens: 8))
        let collector = Task { await cbv2SchedCollect(submitted.events) }
        let retirement = Task {
            await submitted.retirement.wait()
            witness.retired()
        }
        await fulfillment(of: [entered], timeout: 90)
        f.engine.cancel(.init(7004))
        XCTAssertFalse(witness.didRetire)
        XCTAssertGreaterThan(witness.held.bytes, 0)
        XCTAssertGreaterThan(witness.held.arrays, 0)
        XCTAssertGreaterThan(f.process.bytes, 0)
        gate.release()
        let result = await collector.value
        await retirement.value
        XCTAssertEqual(result.finishReason, .cancelled)
        XCTAssertTrue(witness.didRetire)
        try await shutdown(f)
    }

    func testUntrackedSameGeometryKeepsExistingFallbackAndDoesNotCreateRangeOwner() async throws {
        try lane()
        let f = try fixture(tracked: false)
        let witness = Witness()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.mimoKeyRangeWorkCreatedForTesting = { witness.add($0) }
        }
        let before = MiMoV26BlockBatchAttention.keyRangeEncodedDispatches()
        let result = await cbv2SchedCollect(try f.engine.submit(request(7005)))
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, 1)
        XCTAssertEqual(MiMoV26BlockBatchAttention.keyRangeEncodedDispatches(), before)
        XCTAssertTrue(witness.work.isEmpty)
        XCTAssertEqual(f.process.bytes, 0)
        await f.engine.shutdown()
        // Deliberately no native completion/retirement assertion for untracked.
    }

    func testTypedRequiredFenceFailureRetainsRealArraysChargeAndFirstFaultAfterLaterDrain()
        async throws
    {
        try lane(fault: "required-fence")
        let f = try fixture()
        let witness = Witness()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.mimoKeyRangeWorkCreatedForTesting = { work in
                witness.add(work)
                work.beforeRequiredDrainForTesting = { [weak work] in
                    if let work { witness.observedFence(work) }
                    throw Failure.fence
                }
            }
        }
        let submitted = try f.engine.submitWithNativeRetirement(request(7006))
        _ = await cbv2SchedCollect(submitted.events)
        guard case .incomplete(let first) = await f.engine.shutdownReportingNativeCompletion()
        else {
            throw Failure.missingCompletion
        }
        XCTAssertEqual(first.reason, .nativeWorkFailed)
        XCTAssertGreaterThan(witness.held.arrays, 0)
        XCTAssertGreaterThan(f.process.bytes, 0)
        XCTAssertNotNil(witness.firstArray)
        XCTAssertTrue(
            witness.work.contains { $0.completionFailed && !$0.released && $0.reservedBytes > 0 })
        try withError { errors in
            Stream.cpu.synchronize()
            Stream.gpu.synchronize()
            try errors.check()
        }
        guard case .incomplete(let later) = await f.engine.shutdownReportingNativeCompletion()
        else {
            throw Failure.missingCompletion
        }
        XCTAssertEqual(first, later)
        XCTAssertNotNil(witness.firstArray)
        XCTAssertGreaterThan(f.process.bytes, 0)
        XCTAssertFalse(f.process.retired)
        // Never await or manufacture submitted.retirement on this fault.
        _ = Unmanaged.passRetained(f.engine)
    }
}
