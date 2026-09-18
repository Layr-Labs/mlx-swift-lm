import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class PrismPrefillCarryPolicyTests: XCTestCase {
    func testDefaultAndExplicitPolicy() {
        XCTAssertTrue(PrismHadamardPrefillCarry.isEnabled(environmentValue: nil))
        XCTAssertTrue(PrismHadamardPrefillCarry.isEnabled(environmentValue: "1"))
        for value in ["0", "", "false", "true", "off", "on", " 1", "invalid"] {
            XCTAssertFalse(PrismHadamardPrefillCarry.isEnabled(environmentValue: value))
        }
    }

    func testProcessOverrideMatchesLatchedPolicy() {
        let override = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_PREFILL_CARRY_ASYNC"]
        XCTAssertEqual(PrismHadamardPrefillCarry.enabled, override == nil || override == "1")
        XCTAssertNil(PrismHadamardPrefillCarry.context)
    }
}

/// Scheduling/retirement tests, not model-quality or throughput evidence.
/// Run in a dedicated process with the exclusive GPU and diagnostic witness enabled.
final class PrismPrefillCarrySubmissionTests: XCTestCase {
    override func setUpWithError() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST"] == "1"
            && PrismHadamardPrefillCarry.enabled
            && env["DARKBLOOM_BONSAI_PREFILL_CARRY_DIAGNOSTICS"] == "1")
    }

    private func backend() throws -> PagedKVBackend {
        let kinds = (0..<2).map { _ in CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2) }
        return try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 8 << 20, maxPrefillChunk: 128,
            nominalMaxSequenceLength: 512, segmentSizeBytes: 32768,
            layerDTypes: [.bfloat16, .bfloat16]))
    }

    private func bound(_ backend: PagedKVBackend) throws -> ([PagedLayerCache], [CBv2SequenceKV?]) {
        let rows = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: 0, maxLength: 512)
        let caches = backend.makeLayerCaches()
        for (cache, row) in zip(caches, rows) { cache.setRows([try XCTUnwrap(row)]) }
        return (caches, rows)
    }

    private func scoped<T>(_ caches: [any CBv2AttendingLayerCache],
        body: () -> T) -> T {
        PrismHadamardPrefillCarry.withScope(isPacked: true, batch: 1, width: 128,
            hasEmbeddings: false, hasPositions: false, capturesState: false,
            caches: caches, body: body)
    }

    func testEligibilityAndNestedFallbackDoNotLeakScope() throws {
        let backend = try backend()
        let (caches, rows) = try bound(backend)
        defer {
            Stream.gpu.synchronize()
            for cache in caches { cache.setRows([]) }
            backend.release(rows)
        }
        let carry = MLXArray.ones([1, 3, 4])
        XCTAssertFalse(PrismHadamardPrefillCarry.submit(carry))
        let before = PrismHadamardPrefillCarry.submittedCount
        for (packed, batch, width, embeds, positions, captures) in [
            (false, 1, 128, false, false, false),
            (true, 0, 128, false, false, false),
            (true, 1, 1, false, false, false),
            (true, 1, 127, false, false, false),
            (true, 1, 128, true, false, false),
            (true, 1, 128, false, true, false),
            (true, 1, 128, false, false, true),
        ] {
            PrismHadamardPrefillCarry.withScope(isPacked: packed, batch: batch,
                width: width, hasEmbeddings: embeds, hasPositions: positions,
                capturesState: captures, caches: caches) {
                XCTAssertNil(PrismHadamardPrefillCarry.context)
                XCTAssertFalse(PrismHadamardPrefillCarry.submit(carry))
            }
        }
        scoped([]) { XCTAssertFalse(PrismHadamardPrefillCarry.submit(carry)) }
        scoped(caches) {
            XCTAssertTrue(PrismHadamardPrefillCarry.submit(carry))
            XCTAssertFalse(PrismHadamardPrefillCarry.submit(MLXArray.ones([2, 3, 4])))
            XCTAssertFalse(PrismHadamardPrefillCarry.submit(MLXArray.ones([1, 4])))
            PrismHadamardPrefillCarry.withScope(isPacked: false, batch: 1,
                width: 128, hasEmbeddings: false, hasPositions: false,
                capturesState: false, caches: caches) {
                XCTAssertNil(PrismHadamardPrefillCarry.context)
                XCTAssertFalse(PrismHadamardPrefillCarry.submit(carry))
            }
            XCTAssertNotNil(PrismHadamardPrefillCarry.context)
            caches[0].setRows([])
            XCTAssertFalse(PrismHadamardPrefillCarry.submit(carry))
        }
        XCTAssertNil(PrismHadamardPrefillCarry.context)
        XCTAssertEqual(PrismHadamardPrefillCarry.submittedCount - before, 1)
    }

    func testDeferredInputsAreFilledAndRegistrationRemainsOpen() throws {
        let backend = try backend()
        let (caches, rows) = try bound(backend)
        defer {
            Stream.gpu.synchronize()
            for cache in caches { cache.setRows([]) }
            backend.release(rows)
        }
        let slot = MLXArray.zeros([1, 3, 4], dtype: .float32)
        eval(slot)
        let view = slot.asData(access: .noCopy)
        let hostFill = CBv2DeferredHostFill.open()
        defer { hostFill.close(); hostFill.run() }
        var fills = 0
        hostFill.register {
            XCTAssertNil(CBv2DeferredHostFill.current)
            view.data.withUnsafeBytes { raw in
                let pointer = UnsafeMutableRawPointer(mutating: raw.baseAddress!)
                    .assumingMemoryBound(to: Float.self)
                for index in 0..<12 { pointer[index] = 7 }
            }
            fills += 1
        }
        let before = PrismHadamardPrefillCarry.submittedCount
        scoped(caches) {
            let output = slot + Float(2)
            XCTAssertTrue(PrismHadamardPrefillCarry.submit(output))
            XCTAssertEqual(fills, 1)
            XCTAssertTrue(CBv2DeferredHostFill.current === hostFill)
            eval(output)
            XCTAssertEqual(output.asArray(Float.self), Array(repeating: 9, count: 12))
            hostFill.register { fills += 1 }
            XCTAssertTrue(PrismHadamardPrefillCarry.submit(output + Float(1)))
            XCTAssertEqual(fills, 2)
        }
        hostFill.close(); hostFill.run()
        XCTAssertEqual(fills, 2)
        XCTAssertEqual(PrismHadamardPrefillCarry.submittedCount - before, 2)
    }

    func testExistingAndDeferredWriteFaultsPreventSubmission() throws {
        let backend = try backend()
        let (caches, rows) = try bound(backend)
        defer {
            Stream.gpu.synchronize(); Stream.cpu.synchronize()
            for cache in caches { cache.setRows([]) }
            backend.release(rows)
            backend.pool.writeValidation.clearAfterRetirement()
        }
        let carry = MLXArray.ones([1, 3, 4])
        let native = MLXArray.ones([1, 1, 1, 64], dtype: .bfloat16)
        let wrong = native.asType(.float32)
        let hostFill = CBv2DeferredHostFill.open()
        defer { hostFill.close(); hostFill.run() }
        hostFill.register {
            XCTAssertFalse(backend.pool.writeValidation.validate(
                keys: native, values: wrong, expected: .bfloat16, layerIndex: 1))
        }
        let before = PrismHadamardPrefillCarry.submittedCount
        scoped(caches) {
            XCTAssertFalse(PrismHadamardPrefillCarry.submit(carry))
            XCTAssertFalse(PrismHadamardPrefillCarry.submit(carry))
        }
        XCTAssertTrue(CBv2DeferredHostFill.current === hostFill)
        XCTAssertEqual(PrismHadamardPrefillCarry.submittedCount, before)
    }

    func testPartialSubmissionFaultRetiresAndSameIDRecovers() async throws {
        for failOnCall in [1, 3] {
            let backend = try backend()
            let model = PrismCarryFaultModel(failOnCall: failOnCall)
            let engine = EngineV2(model: model, layerKinds: backend.layerKinds,
                backend: backend, cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: CBv2GreedySampler(), schedulerConfig: .init(maxConcurrentRequests: 1,
                    maxBatchedTokensPerStep: 128, prefillChunkSize: 128),
                admissionConfig: .init(watermarkFraction: 0))
            let before = PrismHadamardPrefillCarry.submittedCount
            var finish: CBv2FinishReason?
            for await event in try engine.submit(.init(id: .init(77),
                promptTokens: Array(repeating: 1, count: 384),
                sampling: .init(temperature: 0), maxTokens: 2)) {
                if case .finished(let reason, _) = event { finish = reason }
            }
            guard case .some(.error(let message)) = finish else {
                await engine.shutdown()
                XCTFail("Fault must be returned recoverably as a request error")
                return
            }
            XCTAssertTrue(message.contains("paged KV dtype mismatch"))
            XCTAssertGreaterThan(PrismHadamardPrefillCarry.submittedCount, before)
            engine.loopForTesting.onEngineQueueSync {
                XCTAssertEqual(backend.bytesReserved, 0)
                XCTAssertEqual(backend.bytesWired, 0)
                XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
                XCTAssertFalse(backend.pool.writeValidation.isFaulted)
            }
            model.allowValidCalls()
            var outputCount = 0
            for await event in try engine.submit(.init(id: .init(77),
                promptTokens: Array(repeating: 2, count: 128),
                sampling: .init(temperature: 0), maxTokens: 2)) {
                switch event {
                case .delta(_, let tokens, _): outputCount += tokens.count
                case .finished(let reason, _): finish = reason
                }
            }
            XCTAssertEqual(finish, .length)
            XCTAssertEqual(outputCount, 2)
            await engine.shutdown()
            XCTAssertEqual(backend.bytesReserved, 0)
            XCTAssertEqual(backend.bytesWired, 0)
        }
    }
}

private final class PrismCarryFaultModel: CBv2SteppableModel, @unchecked Sendable {
    private let lock = NSLock()
    private var failOnCall: Int?
    private var calls = 0
    init(failOnCall: Int) { self.failOnCall = failOnCall }
    func allowValidCalls() { lock.withLock { failOnCall = nil } }
    func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache]) -> MLXArray {
        let fail = lock.withLock { calls += 1; return calls == failOnCall }
        let length = tokens.dim(1)
        return PrismHadamardPrefillCarry.withScope(isPacked: true, batch: 1,
            width: length, hasEmbeddings: false, hasPositions: false,
            capturesState: false, caches: caches) {
            let queries = MLXArray.ones([1, 2, length, 64], dtype: .float32)
            let keys = MLXArray.ones([1, 1, length, 64], dtype: .bfloat16)
            var output = queries
            for (index, cache) in caches.enumerated() {
                output = cache.updateAndAttend(queries: queries, keys: keys,
                    values: fail && index == 1 ? keys.asType(.float32) : keys,
                    scale: 0.125, sinks: nil)
                PrismHadamardPrefillCarry.submit(
                    output.transposed(0, 2, 1, 3).reshaped(1, length, 128))
            }
            return broadcast(MLXArray([Float(0), Float(1)]).reshaped(1, 1, 2), to: [1, length, 2])
                + sum(output) * Float(0)
        }
    }
}
