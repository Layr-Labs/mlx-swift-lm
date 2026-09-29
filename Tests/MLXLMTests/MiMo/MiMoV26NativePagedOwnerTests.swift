import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Real Common native pages/loans + existing MiMo ledger. Not a strict-loaded
/// artifact/provider or speed/physical-peak qualification. Fault cells isolated.
final class MiMoV26NativePagedOwnerTests: XCTestCase {
    private enum Failure: Error { case capacity, invalidated, noReceipt, injected }
    private final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var c: UInt64 = 0, m: UInt64 = 0
        private var closed = false
        var bytes: UInt64 { lock.withLock { c } }
        var retired: Bool { lock.withLock { closed } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= m, bytes <= 768 << 20 else { throw Failure.capacity }
                c = bytes
            }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock {
                guard bytes >= m, bytes <= c else { throw Failure.capacity }
                m = bytes
            }
        }
        func withdrawCoverage(_ bytes: UInt64) throws {
            try lock.withLock {
                guard bytes <= m else { throw Failure.capacity }
                m -= bytes
            }
        }
        func retire() {
            lock.withLock {
                XCTAssertEqual(c, 0)
                XCTAssertEqual(m, 0)
                closed = true
            }
        }
    }
    private final class Model: CBv2SteppableModel, CBv2NativePagedModelValidating {
        let scalar = MLXArray(Float(1.0 / 97.0)).asType(.bfloat16)
        var valid = true
        var duringForward: (() -> Void)?
        func validateNativePagedModel() throws { if !valid { throw Failure.invalidated } }
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            duringForward?()
            let b = tokens.dim(0)
            let n = tokens.dim(1)
            var hidden = (tokens.asType(.bfloat16) * scalar).reshaped([b, 1, n, 1])
            for cache in caches {
                let q = broadcast(hidden, to: [b, 64, n, 192])
                let k = broadcast(hidden, to: [b, cache.kind.kvHeads, n, 192])
                let v = broadcast(hidden, to: [b, cache.kind.kvHeads, n, 128])
                hidden = mean(
                    cache.updateAndAttend(
                        queries: q, keys: k, values: v,
                        scale: Float(1.0 / sqrt(192.0)), sinks: nil
                    ).asType(.float32),
                    axes: [1, 3], keepDims: true
                ).asType(.bfloat16)
            }
            let target =
                MLX.round(hidden.asType(.float32).reshaped([b, n, 1]) * Float(64)).asType(.int32)
                % 31
            return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
        }
    }
    private final class WeakBuffer: @unchecked Sendable { weak var value: MLXArray? }
    private final class WeakEngine: @unchecked Sendable { weak var value: EngineV2? }
    private struct Fixture {
        let engine: EngineV2, backend: PagedKVBackend, model: Model
        let binding: CBv2NativePagedModelBinding?
        let owner: ProcessOwner
        let scope: NativeConstructionScope
    }
    private func lane(fault: String? = nil) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_NATIVE_PAGED_TESTS"] == "1" else {
            throw XCTSkip("Requires exclusively owned M3/M5 native lane")
        }
        guard env["MIMO_V26_NATIVE_PAGED_FAULT"] == fault else {
            throw XCTSkip("Fault selectors run alone in fresh processes")
        }
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("Metal GPU required") }
    }
    private func fixture(tracked: Bool = true) throws -> Fixture {
        let kinds: [CBv2LayerKind] = [
            .init(
                attention: .full, headDim: 192, valueHeadDim: 128, kvHeads: 4, queryHeads: 64,
                modelLayerIndex: 0),
            .init(
                attention: .slidingWindow(128), headDim: 192, valueHeadDim: 128, kvHeads: 8,
                queryHeads: 64, modelLayerIndex: 1),
        ]
        let model = Model()
        let owner = ProcessOwner()
        let scope = NativeConstructionScope()
        let ledger = MiMoV26CBv2RowLedger()
        let limits = try CBv2PagedGatheredAttentionLimits(
            maximumBatchSize: 2, maximumQueryTokens: 32,
            maximumContextTokens: 512, maximumInFlightGraphs: 2, maximumScratchBytes: 512 << 20,
            admissionMode: .stepOwned(.pinnedMetal))
        let config = PagedKVPoolConfig(
            capacityBytes: 512 << 20, dtype: .bfloat16,
            maxPrefillChunk: 16, nominalMaxSequenceLength: 512, segmentSizeBytes: 1 << 20,
            layerDTypes: [.bfloat16, .bfloat16], gatheredAttention: limits)
        return try scope.withPhase(.nativeSetup) {
            try scope.capture(StreamOrDevice.cpu.stream)
            try scope.capture(StreamOrDevice.default.stream)
            try scope.retain(model.scalar)
            try scope.willSubmit()
            try withError { errors in
                eval(model.scalar)
                try errors.check()
            }
            try scope.authorizeImmutableLoadedOwner(model)
            let binding: CBv2NativePagedModelBinding?
            let backend: PagedKVBackend
            if tracked {
                let actual = try CBv2NativePagedModelBinding(
                    model: model, loadedOwner: model,
                    layerKinds: kinds, layerDTypes: [.bfloat16, .bfloat16],
                    maximumContextTokens: 512,
                    processMemoryOwner: owner, validator: model, construction: scope,
                    registerRows: { try ledger.register($0, backend: $1) },
                    rowRequest: { try ledger.identity($0, layer: $1) },
                    removeRows: { try ledger.remove($0, backend: $1) })
                backend = try PagedKVBackend(
                    layerKinds: kinds, config: config, nativeModelBinding: actual)
                binding = actual
            } else {
                backend = try PagedKVBackend(layerKinds: kinds, config: config)
                binding = nil
            }
            let caches = backend.makeLayerCaches()
            let bank = CBv2LayerCacheBank(caches: caches)
            try scope.retainOwner(backend)
            try scope.retainOwner(bank)
            try binding?.seal(bank: bank, caches: try XCTUnwrap(caches as? [PagedLayerCache]))
            let contract: CBv2NativeExecutionContract?
            if let binding {
                contract = try CBv2NativeExecutionContract(
                    model: model, backend: backend,
                    cacheProvider: bank, assistant: nil, construction: scope, loadedOwner: model,
                    nativePagedBinding: binding, nativePagedProcessMemoryOwner: owner)
            } else {
                contract = nil
            }
            let engine = EngineV2(
                model: model, layerKinds: kinds, backend: backend, cacheProvider: bank,
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 16, enablePrefixCache: false),
                loopConfig: .init(stepTimeout: 30, watchdogInterval: 0.01, shutdownTimeout: 10),
                admissionConfig: .init(watermarkFraction: 0), processMemoryOwner: owner,
                nativeCompletionTracking: tracked, nativeExecutionContract: contract)
            XCTAssertNil(engine.pagedAttentionWorkInactiveReason)
            XCTAssertNil(engine.nativeCompletionFault)
            return .init(
                engine: engine, backend: backend, model: model, binding: binding, owner: owner,
                scope: scope)
        }
    }
    private func request(_ id: UInt64) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: (0 ..< 145).map { ($0 * 7 + 3) % 29 },
            sampling: .init(temperature: 0), maxTokens: 4, prefixCacheEnabled: false)
    }
    private func shutdown(_ f: Fixture) async throws -> CBv2NativeShutdownOutcome {
        let outcome = await f.engine.shutdownReportingNativeCompletion()
        guard case .quiescent = outcome else {
            _ = Unmanaged.passRetained(f.engine)
            throw Failure.noReceipt
        }
        XCTAssertTrue(f.owner.retired)
        XCTAssertEqual(f.owner.bytes, 0)
        XCTAssertEqual(f.backend.pool.bytesMaterialized, 0)
        return outcome
    }

    func testActualPageGrowthReuseAndIdleFloorMatchUntrackedReference() async throws {
        try lane()
        let reference = try fixture(tracked: false)
        let expected = await cbv2SchedCollect(try reference.engine.submit(request(8101)))
        XCTAssertEqual(expected.finishReason, .length)
        XCTAssertEqual(expected.tokens.count, 4)
        await reference.engine.shutdown()
        XCTAssertEqual(
            reference.backend.pool.bytesMaterialized, 0, "old nil-binding trim is unchanged")

        let f = try fixture()
        for id in [UInt64(8102), 8103] {
            let submitted = try f.engine.submitWithNativeRetirement(request(id))
            let result = await cbv2SchedCollect(submitted.events)
            await submitted.retirement.wait()
            XCTAssertEqual(result.finishReason, .length)
            XCTAssertEqual(result.tokens, expected.tokens)
            f.engine.loopForTesting.onEngineQueueSync {
                XCTAssertGreaterThan(
                    f.backend.pool.bytesMaterialized, 0, "idle pool backing is still real")
                XCTAssertGreaterThan(f.owner.bytes, 0, "no per-request physical refund")
                XCTAssertEqual(f.backend.bytesReserved, 0)
                XCTAssertEqual(f.engine.pagedAttentionWorkBytesReserved, 0)
            }
        }
        let first = try await shutdown(f)
        let second = await f.engine.shutdownReportingNativeCompletion()
        XCTAssertEqual(second, first)
    }

    func testActualForeignLayerAndMixedCohortReleaseRefuseAtomically() async throws {
        try lane()
        let f = try fixture()
        let foreign = try fixture()
        let foreignRows = try foreign.engine.loopForTesting.onEngineQueueSync {
            let operation = try XCTUnwrap(foreign.binding).beginWork()
            let rows = try operation.withConstruction {
                try foreign.backend.makeSequenceState(
                    layerKinds: foreign.backend.layerKinds,
                    promptLength: 16, maxLength: 32)
            }
            try operation.requiredDrain()
            operation.finish()
            return rows
        }
        try f.engine.loopForTesting.onEngineQueueSync {
            let operation = try XCTUnwrap(f.binding).beginWork()
            let pair = try operation.withConstruction {
                (
                    try f.backend.makeSequenceState(
                        layerKinds: f.backend.layerKinds, promptLength: 16, maxLength: 32),
                    try f.backend.makeSequenceState(
                        layerKinds: f.backend.layerKinds, promptLength: 16, maxLength: 32)
                )
            }
            try operation.requiredDrain()
            let before = f.backend.bytesReserved
            let foreignBefore = foreign.backend.bytesReserved
            XCTAssertThrowsError(try f.backend.releaseNativeValidated(foreignRows))
            XCTAssertEqual(foreign.backend.bytesReserved, foreignBefore)
            let mixed = [pair.0[0], pair.1[1]]
            XCTAssertThrowsError(try f.backend.releaseNativeValidated(mixed))
            XCTAssertThrowsError(try f.backend.releaseNativeValidated(Array(pair.0.reversed())))
            XCTAssertThrowsError(try f.backend.releaseNativeValidated([pair.0[0], pair.0[0]]))
            XCTAssertEqual(f.backend.bytesReserved, before)
            XCTAssertNoThrow(try XCTUnwrap(f.binding).metadata(row: XCTUnwrap(pair.0[0]), layer: 0))
            XCTAssertThrowsError(
                try XCTUnwrap(f.binding).metadata(row: XCTUnwrap(pair.0[0]), layer: 1))
            try f.backend.releaseNativeValidated(pair.0)
            try f.backend.releaseNativeValidated(pair.1)
            operation.finish()
        }
        try foreign.engine.loopForTesting.onEngineQueueSync {
            try foreign.backend.releaseNativeValidated(foreignRows)
        }
        _ = try await shutdown(foreign)
        _ = try await shutdown(f)
    }

    func testAnUnrelatedRealLoanBlocksPoolRetirementAndReceipt() async throws {
        try lane()
        let f = try fixture()
        let engine = f.engine
        let submitted = try engine.submitWithNativeRetirement(request(8104))
        _ = await cbv2SchedCollect(submitted.events)
        await submitted.retirement.wait()
        let extra = try engine.loopForTesting.onEngineQueueSync {
            try XCTUnwrap(f.binding).beginWork()
        }
        let task = Task { await engine.shutdownReportingNativeCompletion() }
        let entered = await cbv2SchedWait {
            engine.loopForTesting.onEngineQueueSync { engine.loopForTesting.isDrainingForTesting }
        }
        XCTAssertTrue(entered, "the real drain frame must reach the other-loan barrier")
        engine.loopForTesting.onEngineQueueSync {}
        engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(engine.loopForTesting.nativeShutdownState?.outcome)
            XCTAssertFalse(f.owner.retired)
            XCTAssertGreaterThan(f.owner.bytes, 0)
            XCTAssertFalse(f.binding?.canStartPoolRetirement ?? true)
            extra.finish(unstarted: true)  // this actual independent scope never allocated/submitted
        }
        guard case .quiescent = await task.value else { throw Failure.noReceipt }
        XCTAssertTrue(f.owner.retired)
        XCTAssertEqual(f.owner.bytes, 0)
    }

    private func dropFixtureWithoutShutdown() async throws -> (WeakEngine, ProcessOwner) {
        var f: Fixture? = try fixture()
        let owner = try XCTUnwrap(f).owner
        let actual = WeakEngine()
        actual.value = f?.engine
        var stream: AsyncStream<CBv2Event>? = try XCTUnwrap(f).engine.submit(request(8105))
        _ = await cbv2SchedCollect(try XCTUnwrap(stream))
        stream = nil
        f = nil  // also drops the original construction scope/model/bank references
        return (actual, owner)  // no original fixture/stream survives this async frame
    }

    func testDroppedUnshutdownEngineStaysOwnedUntilRealPoolRetirement() async throws {
        try lane()
        let (actual, owner) = try await dropFixtureWithoutShutdown()
        XCTAssertNotNil(actual.value, "the actual native lifetime loan, not ARC, owns shutdown")
        XCTAssertGreaterThan(owner.bytes, 0)
        XCTAssertFalse(owner.retired)
        let held = try XCTUnwrap(actual.value)
        guard case .quiescent = await held.shutdownReportingNativeCompletion() else {
            throw Failure.noReceipt
        }
        XCTAssertTrue(owner.retired)
        XCTAssertEqual(owner.bytes, 0)
        withExtendedLifetime(held) {}
    }

    func testOrdinaryCapacityRefusalDoesNotBecomeARequiredNativeFault() async throws {
        try lane()
        let f = try fixture()
        f.engine.updateKVBytesCapacity(1)
        XCTAssertThrowsError(try f.engine.submit(request(8106)))
        XCTAssertNil(f.engine.nativeCompletionFault)
        XCTAssertEqual(f.backend.pool.bytesMaterialized, 0)
        _ = try await shutdown(f)
    }

    func testPartialEvaluatedSlabFailureRetainsActualBufferPeakAndFirstFault() async throws {
        try lane(fault: "slab")
        let f = try fixture()
        let witness = WeakBuffer()
        f.engine.loopForTesting.onEngineQueueSync {
            f.backend.pool.slabEval = { array in
                witness.value = array
                try withError { errors in
                    eval(array)
                    try errors.check()
                }
                throw Failure.injected  // after real evaluation; never ordinary capacity
            }
        }
        _ = await cbv2SchedCollect(try f.engine.submit(request(8107)))
        guard case .incomplete(let first) = await f.engine.shutdownReportingNativeCompletion()
        else {
            throw Failure.noReceipt
        }
        XCTAssertEqual(first.reason, .nativeWorkFailed)
        XCTAssertNotNil(witness.value)
        XCTAssertGreaterThan(f.owner.bytes, 0)
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
        XCTAssertNotNil(witness.value)
        XCTAssertGreaterThan(f.owner.bytes, 0)
        XCTAssertFalse(f.owner.retired)
        _ = Unmanaged.passRetained(f.engine)
    }
}
