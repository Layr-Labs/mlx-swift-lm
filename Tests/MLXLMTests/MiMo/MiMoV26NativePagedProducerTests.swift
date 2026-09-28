import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import Tokenizers
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Strict-loaded small fixture, real native producer and serving engine.
/// Fixture creation/full-model qualification are separate coordinator gates.
final class MiMoV26NativePagedProducerTests: XCTestCase {
    private enum Failure: Error { case fixtureRequired, noReceipt, capacity, injectedSlabCompletion }
    private final class WeakSlab: @unchecked Sendable { weak var value: MLXArray? }
    private final class Permit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        var reservedLoadBytes: UInt64 { request.requiredLoadBytes }
        init(_ request: MiMoV26SerialLoadRequest) { self.request = request }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let raw = try await AutoTokenizer.from(modelFolder: directory)
            return #adaptHuggingFaceTokenizer(raw)
        }
    }
    private final class Owner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var c: UInt64 = 0, m: UInt64 = 0
        private var retirementCalls = 0
        var bytes: UInt64 { lock.withLock { c } }
        var observedRetirementCalls: Int { lock.withLock { retirementCalls } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock { guard bytes >= m, bytes <= 1 << 30 else { throw Failure.capacity }; c = bytes }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock { guard bytes >= m, bytes <= c else { throw Failure.capacity }; m = bytes }
        }
        func withdrawCoverage(_ bytes: UInt64) throws {
            try lock.withLock { guard bytes <= m else { throw Failure.capacity }; m -= bytes }
        }
        func retire() { lock.withLock { XCTAssertEqual(c, 0); XCTAssertEqual(m, 0); retirementCalls += 1 } }
    }
    private struct Fixture: Sendable {
        let container: ModelContainer
        let construction: NativeConstructionWork
    }
    private struct Built: Sendable { let engine: EngineV2; let contract: CBv2NativeExecutionContract }
    private func fixture() async throws -> Fixture {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_NATIVE_PAGED_TESTS"] == "1" else { throw XCTSkip("Owned native lane required") }
        guard let path = env["MIMO_V26_NATIVE_PAGED_FIXTURE_ROOT"] else { throw Failure.fixtureRequired }
        let root = URL(fileURLWithPath: path)
        let handle = try FileHandle(forReadingFrom: root.appendingPathComponent("provenance.json"))
        defer { try? handle.close() }
        let data = try XCTUnwrap(handle.read(upToCount: 65537))
        guard data.count <= 65536 else { throw Failure.fixtureRequired }
        let p = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]), sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 64 << 20, maximumTotalFileBytes: 128 << 20))
        let config = plan.bundlePlan.configuration
        guard config.hiddenSize <= 64, config.numHiddenLayers <= 4,
              config.maxPositionEmbeddings >= 256, config.vocabularySize >= 32 else { throw Failure.fixtureRequired }
        for index in 0..<config.numHiddenLayers {
            let g = try config.attentionGeometry(at: index)
            guard g.headDim == 192, g.valueHeadDim == 128, g.queryHeads == 64,
                  [4, 8].contains(g.keyValueHeads) else { throw Failure.fixtureRequired }
        }
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let construction = NativeConstructionWork()
        let container = try await MiMoV26ModelFactory.loadContainer(session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        return .init(container: container, construction: construction)
    }
    private func build(_ f: Fixture, paged: Bool, owner: Owner) async throws -> Built {
        try await MiMoV26ModelFactory.withNativeConstruction(container: f.container,
            retaining: f.construction) { model, scope in
            let binding = try model.makeCBv2Binding(enableMTP: false)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let backend: any CBv2KVBackend
            let bank: CBv2LayerCacheBank
            let contract: CBv2NativeExecutionContract
            if paged {
                let actual = try model.makeNativePagedExecutionResources(binding: binding,
                    bytesCapacity: 768 << 20, maximumConcurrentRequests: 2, maximumQueryTokens: 32,
                    maximumPrefillChunk: 16, processMemoryOwner: owner, retaining: scope)
                backend = actual.backend; bank = actual.cacheProvider; contract = actual.contract
                XCTAssertTrue(binding.adapter.cbv2Capabilities.supportsPagedKV)
                XCTAssertTrue(contract.supportsNativePagedTarget)
                XCTAssertFalse(contract.supportsNativeCompletePrefix)
                XCTAssertFalse(contract.supportsManagedDecodedMedia)
            } else {
                let actual = try binding.adapter.makeNativeExecutionResources(bytesCapacity: 768 << 20, retaining: scope)
                backend = actual.backend; bank = actual.cacheProvider; contract = actual.contract
            }
            let engine = EngineV2(model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: backend, cacheProvider: bank,
                schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 16, enablePrefixCache: false),
                loopConfig: .init(stepTimeout: 30, watchdogInterval: 0.01, shutdownTimeout: 10),
                processMemoryOwner: paged ? owner : nil,
                nativeCompletionTracking: true, nativeExecutionContract: contract)
            try scope.retainOwner(engine)
            XCTAssertNil(engine.nativeCompletionFault)
            if paged { XCTAssertNil(engine.pagedAttentionWorkInactiveReason) }
            return .init(engine: engine, contract: contract)
        }
    }
    private func stop(_ built: Built, fixture: Fixture, paged: Bool) async throws {
        guard case .quiescent(let receipt) = await built.engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(built.engine); throw Failure.noReceipt
        }
        if paged {
            try await fixture.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                try model.releaseNativePagedAfterNativeRetirement(receipt)
                XCTAssertNil(model.nativePagedPreparation)
                XCTAssertNil(model.resources.nativePagedValidator)
                XCTAssertThrowsError(try model.releaseNativePagedAfterNativeRetirement(receipt))
            }
        }
    }
    func testStrictLoadedPagedFacadeMatchesColdContiguousTokensAcrossPageAndWindowBoundaries() async throws {
        let a = try await fixture(), b = try await fixture(), owner = Owner()
        let baseline = try await build(a, paged: false, owner: Owner())
        let candidate = try await build(b, paged: true, owner: owner)
        for (index, count) in [15, 16, 17, 127, 128, 129, 145].enumerated() {
            let request = CBv2Request(id: .init(UInt64(8300 + index)),
                promptTokens: (0..<count).map { 1 + ($0 * 7) % 29 },
                sampling: .init(temperature: 0), maxTokens: 4, prefixCacheEnabled: false)
            let x = try baseline.engine.submitWithNativeRetirement(request)
            let y = try candidate.engine.submitWithNativeRetirement(request)
            let expected = await cbv2SchedCollect(x.events)
            let actual = await cbv2SchedCollect(y.events)
            await x.retirement.wait(); await y.retirement.wait()
            XCTAssertEqual(expected.finishReason, .length); XCTAssertEqual(actual.finishReason, .length)
            XCTAssertEqual(actual.tokens, expected.tokens); XCTAssertEqual(actual.tokens.count, 4)
        }
        try await stop(baseline, fixture: a, paged: false)
        try await stop(candidate, fixture: b, paged: true)
        XCTAssertEqual(owner.bytes, 0)
    }

    func testStrictLoadedProducerRefusesMTPAndInvalidatedGenerationBeforeIssuance() async throws {
        let f = try await fixture()
        try await MiMoV26ModelFactory.withNativeConstruction(container: f.container,
            retaining: f.construction) { model, scope in
            let mtp = try model.makeCBv2Binding(enableMTP: true)
            _ = try mtp.adapter.probeNativeKVTypes(retaining: scope)
            XCTAssertThrowsError(try model.makeNativePagedExecutionResources(binding: mtp,
                bytesCapacity: 128 << 20, maximumConcurrentRequests: 1, maximumQueryTokens: 16,
                maximumPrefillChunk: 16, processMemoryOwner: Owner(), retaining: scope))
            XCTAssertNil(model.nativePagedPreparation)
            let text = try model.makeCBv2Binding()
            _ = try text.adapter.probeNativeKVTypes(retaining: scope)
            model.invalidateMultimodalPreparation()
            XCTAssertThrowsError(try model.makeNativePagedExecutionResources(binding: text,
                bytesCapacity: 128 << 20, maximumConcurrentRequests: 1, maximumQueryTokens: 16,
                maximumPrefillChunk: 16, processMemoryOwner: Owner(), retaining: scope))
            XCTAssertNil(model.nativePagedPreparation)
        }
    }

    /// Strict loaded native owner + real partial slab evaluation. This is the
    /// existing bounded SDK test owner, not provider GlobalKVCacheBudget proof.
    /// Run this selector alone; retained faults intentionally survive to exit.
    func testStrictLoadedPartialSlabCompletionFailureRetainsIssuedOwnerAndCharge() async throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_NATIVE_PAGED_FAULT"] == "strict-slab" else {
            throw XCTSkip("Isolated strict-loaded slab-fault process required")
        }
        let f = try await fixture(), owner = Owner(), weakSlab = WeakSlab()
        var engineToRetain: EngineV2?
        defer {
            // The actual failed epoch cannot be freed by ARC or an invented
            // retirement receipt, even if an assertion or later call throws.
            _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
            if let engineToRetain { _ = Unmanaged.passRetained(engineToRetain) }
        }
        let built = try await build(f, paged: true, owner: owner)
        engineToRetain = built.engine
        let setup = f.construction.snapshot
        guard case .completed(let completedSetup) = setup.disposition else { throw Failure.noReceipt }
        try f.construction.validate(completedSetup)
        XCTAssertEqual(setup.retainedArrayCount, 0)
        XCTAssertEqual(setup.retainedOwnerCount, 0)
        XCTAssertEqual(setup.capturedStreamCount, 0)
        // Real protected construction borrow has returned before serving work.
        // This receipt is setup completion, NEVER runtime retirement or credit.
        try await f.construction.sealForPublication(completedSetup)
        let backend = try XCTUnwrap(built.engine.loopForTesting.backend as? PagedKVBackend)
        let tracking = try XCTUnwrap(built.engine.loopForTesting.nativeShutdownState)
        XCTAssertTrue(built.contract.supportsNativePagedTarget)
        XCTAssertEqual(built.engine.nativeShutdownExecutionContractID, built.contract.id)
        built.engine.loopForTesting.onEngineQueueSync {
            XCTAssertEqual(backend.pool.bytesMaterialized, 0)
            backend.pool.slabEval = { actual in
                weakSlab.value = actual
                try withError { errors in eval(actual); try errors.check() }
                // Real buffer evaluated first; then the required completion
                // boundary refuses. Never a replacement successful result.
                throw Failure.injectedSlabCompletion
            }
        }
        let request = CBv2Request(id: .init(8401), promptTokens: (0..<17).map { 1 + $0 % 29 },
            sampling: .init(temperature: 0), maxTokens: 4, prefixCacheEnabled: false)
        _ = await cbv2SchedCollect(try built.engine.submit(request))
        XCTAssertNotNil(weakSlab.value)
        XCTAssertGreaterThan(owner.bytes, 0)
        XCTAssertEqual(owner.observedRetirementCalls, 0)
        XCTAssertTrue(tracking.hasLoans)
        // Only metadata crosses the container actor. No model/array escapes.
        let aliasesBeforeShutdown = try await f.container.perform { context -> Bool in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            try model.beginNativePagedRetirement(executionContractID: built.contract.id)
            return model.nativePagedPreparation != nil && model.resources.nativePagedValidator != nil
        }
        XCTAssertTrue(aliasesBeforeShutdown)
        guard case .incomplete(let first) = await built.engine.shutdownReportingNativeCompletion() else {
            throw Failure.noReceipt
        }
        XCTAssertEqual(first.reason, .nativeWorkFailed)
        XCTAssertEqual(f.construction.snapshot.disposition, .completed(completedSetup))
        try withError { errors in Stream.cpu.synchronize(); Stream.gpu.synchronize(); try errors.check() }
        guard case .incomplete(let later) = await built.engine.shutdownReportingNativeCompletion() else {
            throw Failure.noReceipt
        }
        XCTAssertEqual(first, later)
        XCTAssertNotNil(weakSlab.value); XCTAssertGreaterThan(owner.bytes, 0)
        XCTAssertEqual(owner.observedRetirementCalls, 0)
        let stillRetained = try await f.container.perform { context -> Bool in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            return model.nativePagedPreparation != nil && model.resources.nativePagedValidator != nil
        }
        XCTAssertTrue(stillRetained)
        // No fabricated CBv2NativeShutdownReceipt and no success-only alias
        // release call. The separate fresh process owns this restart-only fault.
    }
}
