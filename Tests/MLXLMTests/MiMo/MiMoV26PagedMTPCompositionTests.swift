import Foundation
import MLX
import MLXHuggingFace
import Tokenizers
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Real strict-loaded tiny target + genuine trained heads/page bank. Source
/// prepared only. No prefix/media/rectangular or full-model qualification.
final class MiMoV26PagedMTPCompositionTests: XCTestCase {
    private enum Failure: Error { case fixture, capacity, completion, injected }
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
        private var charge: UInt64 = 0, materialized: UInt64 = 0
        private var closed = false
        var retired: Bool { lock.withLock { closed } }
        var bytes: UInt64 { lock.withLock { charge } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= 1 << 30 else { throw Failure.capacity }
                charge = bytes
            }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock { guard bytes >= materialized, bytes <= charge else { throw Failure.capacity }; materialized = bytes }
        }
        func withdrawCoverage(_ bytes: UInt64) throws {
            try lock.withLock { guard bytes <= materialized else { throw Failure.capacity }; materialized -= bytes }
        }
        func retire() { lock.withLock { XCTAssertEqual(charge, 0); XCTAssertEqual(materialized, 0); closed = true } }
    }
    private struct Fixture: Sendable {
        let container: ModelContainer
        let construction: NativeConstructionWork
        let engine: EngineV2
        let contract: CBv2NativeExecutionContract
        let owner: Owner
        let paged: Bool
    }
    private struct Built: Sendable { let engine: EngineV2; let contract: CBv2NativeExecutionContract }
    private func lane(_ fault: String? = nil) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_PAGED_MTP_TESTS"] == "1",
              env["MIMO_V26_NATIVE_PAGED_TESTS"] == "1",
              Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Requires the root's exclusive bounded Metal lane and strict tiny fixture")
        }
        guard env["MIMO_V26_PAGED_MTP_FAULT"] == fault else {
            throw XCTSkip("Fault cases require their own fresh bounded process")
        }
    }
    private func fixture(paged: Bool = true, depth: Int = 3) async throws -> Fixture {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_NATIVE_PAGED_FIXTURE_ROOT"] else { throw Failure.fixture }
        let root = URL(fileURLWithPath: path)
        let file = try FileHandle(forReadingFrom: root.appendingPathComponent("provenance.json"))
        defer { try? file.close() }
        let data = try XCTUnwrap(file.read(upToCount: 65537))
        guard data.count <= 65536 else { throw Failure.fixture }
        let p = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]), sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 64 << 20, maximumTotalFileBytes: 128 << 20))
        let config = plan.bundlePlan.configuration
        guard config.hiddenSize <= 64, config.numHiddenLayers <= 4,
              config.maxPositionEmbeddings >= 256, config.vocabularySize >= 32,
              config.numNextnPredictLayers == 3 else { throw Failure.fixture }
        for index in 0..<config.numHiddenLayers {
            let g = try config.attentionGeometry(at: index)
            guard g.headDim == 192, g.valueHeadDim == 128, g.queryHeads == 64,
                  [4, 8].contains(g.keyValueHeads) else { throw Failure.fixture }
        }
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let construction = NativeConstructionWork(), owner = Owner()
        let container = try await MiMoV26ModelFactory.loadContainer(session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        let built = try await MiMoV26ModelFactory.withNativeConstruction(container: container,
            retaining: construction) { model, scope -> Built in
            if paged {
                let refused = try model.makeCBv2Binding(enableMTP: true, verificationMode: .rectangular)
                _ = try refused.adapter.probeNativeKVTypes(retaining: scope)
                XCTAssertThrowsError(try model.makeNativePagedSerialMTPExecutionResources(binding: refused,
                    bytesCapacity: 768 << 20, maximumConcurrentRequests: 2, maximumQueryTokens: 32,
                    maximumPrefillChunk: 16, processMemoryOwner: owner, retaining: scope))
                XCTAssertNil(model.nativePagedPreparation, "negative mode must not consume issuance")
            }
            let binding = try model.makeCBv2Binding(enableMTP: true, verificationMode: .serialTarget)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let backend: any CBv2KVBackend
            let bank: CBv2LayerCacheBank
            let contract: CBv2NativeExecutionContract
            if paged {
                XCTAssertThrowsError(try model.makeNativePagedExecutionResources(binding: binding,
                    bytesCapacity: 768 << 20, maximumConcurrentRequests: 2, maximumQueryTokens: 32,
                    maximumPrefillChunk: 16, processMemoryOwner: owner, retaining: scope),
                    "the existing target-only issuer must keep its old refusal")
                let resources = try model.makeNativePagedSerialMTPExecutionResources(binding: binding,
                    bytesCapacity: 768 << 20, maximumConcurrentRequests: 2, maximumQueryTokens: 32,
                    maximumPrefillChunk: 16, processMemoryOwner: owner, retaining: scope)
                backend = resources.backend; bank = resources.cacheProvider; contract = resources.contract
                XCTAssertTrue(contract.supportsNativePagedSerialMTP)
                XCTAssertFalse(contract.supportsNativeCompletePrefix)
                XCTAssertFalse(contract.supportsManagedDecodedMedia)
                let foreign = try model.makeCBv2Binding(enableMTP: true, verificationMode: .serialTarget)
                XCTAssertFalse(contract.consume(model: binding.adapter, backend: backend,
                    cacheProvider: bank, assistant: foreign.assistant, processMemoryOwner: owner))
                XCTAssertFalse(contract.consume(model: binding.adapter, backend: backend,
                    cacheProvider: bank, assistant: binding.assistant, processMemoryOwner: Owner()))
                XCTAssertFalse(contract.consume(model: binding.adapter, backend: backend,
                    cacheProvider: bank, assistant: binding.assistant, mtpVerificationMode: .rectangular,
                    processMemoryOwner: owner))
            } else {
                let resources = try binding.adapter.makeNativeExecutionResources(bytesCapacity: 768 << 20, retaining: scope)
                backend = resources.backend; bank = resources.cacheProvider; contract = resources.contract
            }
            let engine = EngineV2(model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: backend, cacheProvider: bank,
                schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 16, enablePrefixCache: false),
                loopConfig: .init(stepTimeout: 30, watchdogInterval: 0.01, shutdownTimeout: 10),
                mtpDrafter: binding.assistant,
                mtpConfig: .init(enabled: true, maxDraftTokens: depth, maxSpeculativeBatch: 1,
                    fixedDraftTokens: depth, verificationMode: .serialTarget),
                processMemoryOwner: paged ? owner : nil, nativeCompletionTracking: true,
                nativeExecutionContract: contract)
            try scope.retainOwner(engine)
            XCTAssertNil(engine.nativeCompletionFault); XCTAssertNil(engine.pagedAttentionWorkInactiveReason)
            XCTAssertNil(engine.mtpInactiveReason)
            guard case .bounded? = engine.resolvedMTPAdmission else { throw Failure.completion }
            XCTAssertFalse(contract.consume(model: binding.adapter, backend: backend,
                cacheProvider: bank, assistant: binding.assistant, processMemoryOwner: paged ? owner : nil),
                "actual engine must consume the genuine ticket exactly once")
            return .init(engine: engine, contract: contract)
        }
        guard case .completed(let receipt) = construction.snapshot.disposition else { throw Failure.completion }
        try await construction.sealForPublication(receipt)
        return .init(container: container, construction: construction, engine: built.engine,
            contract: built.contract, owner: owner, paged: paged)
    }
    private func request(_ id: UInt64, count: Int, output: Int = 24) -> CBv2Request {
        .init(id: .init(id), promptTokens: (0..<count).map { 1 + ($0 * 7) % 29 },
              sampling: .init(temperature: 0), maxTokens: output, prefixCacheEnabled: false)
    }
    private func stop(_ f: Fixture) async throws {
        if f.paged {
            try await f.container.perform { context in
                try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                    .beginNativePagedRetirement(executionContractID: f.contract.id)
            }
        }
        guard case .quiescent(let receipt) = await f.engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction); throw Failure.completion
        }
        if f.paged {
            try await f.container.perform { context in
                try XCTUnwrap(context.model as? MiMoV26LoadedModel).releaseNativePagedAfterNativeRetirement(receipt)
            }
            XCTAssertTrue(f.owner.retired); XCTAssertEqual(f.owner.bytes, 0)
        }
        XCTAssertEqual(f.engine.admissionForTesting.bytesReserved, 0)
    }
    private func collect(_ f: Fixture, _ request: CBv2Request) async throws -> [Int] {
        let before = try XCTUnwrap(f.engine.mtpMetricsSnapshot())
        var completed = 0
        if f.paged {
            f.engine.loopForTesting.onEngineQueueSync {
                f.engine.loopForTesting.nativeRetirementBoundaryForTesting = { phase, step in
                    guard phase == "beforeMTPFinalization", let verify = step?.mtpRound?.verify else { return }
                    let work = step?.nativePagedMTPWork
                    XCTAssertNotNil(work)
                    XCTAssertEqual(work?.completedColumns, verify.k + 1)
                    XCTAssertGreaterThan(f.owner.bytes, 0)
                    XCTAssertGreaterThan(f.engine.pagedAttentionWorkBytesReserved, 0)
                    XCTAssertEqual(verify.rows.count, 1)
                    completed += 1
                }
            }
        }
        let submission = try f.engine.submitWithNativeRetirement(request)
        let result = await cbv2SchedCollect(submission.events)
        guard f.engine.nativeCompletionFault == nil else { throw Failure.completion }
        await submission.retirement.wait()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
            XCTAssertNil(f.engine.loopForTesting.mtp?.assistantStateCountsForTesting(request.id))
            if f.paged {
                XCTAssertEqual(f.engine.pagedAttentionWorkBytesReserved, 0)
                XCTAssertTrue((f.engine.loopForTesting.backend as? PagedKVBackend)?
                    .nativeModelBinding?.canStartPoolRetirement == true)
            }
        }
        XCTAssertEqual(result.finishReason, .length); XCTAssertEqual(result.tokens.count, request.maxTokens)
        let after = try XCTUnwrap(f.engine.mtpMetricsSnapshot())
        XCTAssertGreaterThan(after.draftedTokens - before.draftedTokens, 0)
        XCTAssertGreaterThan(after.serialVerificationRounds - before.serialVerificationRounds, 0)
        XCTAssertEqual(after.rectangularVerificationRounds, 0)
        if f.paged { XCTAssertGreaterThan(completed, 0) }
        return result.tokens
    }

    func testIssuedSerialMTPPagesMatchContiguousAcrossPageAndWindowBoundaries() async throws {
        try lane()
        for depth in 1...3 {
            let reference = try await fixture(paged: false, depth: depth)
            let candidate = try await fixture(depth: depth)
            for (index, count) in [15, 16, 17, 127, 128, 129, 145].enumerated() {
                let value = request(UInt64(9400 + index), count: count)
                let expected = try await collect(reference, value)
                let actual = try await collect(candidate, value)
                XCTAssertEqual(actual, expected, "real page-backed target + real three-head proposals")
            }
            try await stop(candidate); try await stop(reference)
        }
    }

    func testOrdinaryNeighborKeepsItsOwnTicketsWhileOneTextRowSpeculates() async throws {
        try lane()
        var outputs: [[[Int]]] = []
        for paged in [false, true] {
            let f = try await fixture(paged: paged, depth: 2)
            var mixed = 0
            f.engine.loopForTesting.onEngineQueueSync { f.engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0 }
            let text = request(9501, count: 17)
            var ordinary = request(9502, count: 49, output: 12)
            ordinary.sampling = .init(temperature: 0, repetitionPenalty: 1.1)
            f.engine.loopForTesting.onEngineQueueSync {
                f.engine.loopForTesting.nativeRetirementBoundaryForTesting = { phase, step in
                    guard phase == "beforeMTPFinalization", let step,
                          let verify = step.mtpRound?.verify,
                          verify.rows.map(\.id) == [text.id], step.participants.contains(ordinary.id) else { return }
                    if paged { XCTAssertEqual(step.nativePagedMTPWork?.completedColumns, verify.k + 1) }
                    mixed += 1
                }
            }
            let a = try f.engine.submitWithNativeRetirement(text)
            let b = try f.engine.submitWithNativeRetirement(ordinary)
            f.engine.loopForTesting.onEngineQueueSync { f.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil }
            async let x = cbv2SchedCollect(a.events)
            async let y = cbv2SchedCollect(b.events)
            let pair = await (x, y)
            guard f.engine.nativeCompletionFault == nil else { throw Failure.completion }
            await a.retirement.wait(); await b.retirement.wait()
            f.engine.loopForTesting.onEngineQueueSync { f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil }
            XCTAssertEqual(pair.0.finishReason, .length); XCTAssertEqual(pair.1.finishReason, .length)
            XCTAssertEqual(pair.0.tokens.count, 24); XCTAssertEqual(pair.1.tokens.count, 12)
            XCTAssertGreaterThan(try XCTUnwrap(f.engine.mtpMetricsSnapshot()).serialVerificationRounds, 0)
            XCTAssertGreaterThan(mixed, 0, "must actually overlap an ordinary neighbor with trained verification")
            outputs.append([pair.0.tokens, pair.1.tokens])
            try await stop(f)
        }
        XCTAssertEqual(outputs[1], outputs[0])
    }

    func testSinglePrefillTerminalDoesNotStrandAnUnusedBorrowerLoan() async throws {
        try lane()
        let f = try await fixture()
        let submission = try f.engine.submitWithNativeRetirement(request(9551, count: 16, output: 1))
        let result = await cbv2SchedCollect(submission.events)
        XCTAssertEqual(result.finishReason, .length); XCTAssertEqual(result.tokens.count, 1)
        guard f.engine.nativeCompletionFault == nil else { throw Failure.completion }
        await submission.retirement.wait()
        XCTAssertEqual(f.engine.mtpMetricsSnapshot()?.draftedTokens, 0)
        XCTAssertEqual(f.engine.pagedAttentionWorkBytesReserved, 0)
        try await stop(f)
    }

    func testCancelAfterCompletedColumnsKeepsChargesUntilRetirementThenReusesIDCold() async throws {
        try lane()
        let f = try await fixture(), reference = try await fixture(paged: false)
        let entered = expectation(description: "actual paged verification completed before cancellation")
        let gate = DispatchSemaphore(value: 0)
        defer { gate.signal() }
        var first = true
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRetirementBoundaryForTesting = { phase, step in
                guard first, phase == "beforeMTPFinalization", let verify = step?.mtpRound?.verify else { return }
                first = false
                XCTAssertEqual(step?.nativePagedMTPWork?.completedColumns, verify.k + 1)
                entered.fulfill()
                _ = gate.wait(timeout: .now() + 15)
            }
        }
        let submitted = try f.engine.submitWithNativeRetirement(request(9552, count: 145))
        await fulfillment(of: [entered], timeout: 10)
        let charged = f.owner.bytes
        XCTAssertGreaterThan(charged, 0)
        f.engine.cancel(.init(9552))
        XCTAssertEqual(f.owner.bytes, charged, "cancel intent cannot return live target/head/work promises")
        gate.signal()
        let result = await cbv2SchedCollect(submitted.events)
        guard f.engine.nativeCompletionFault == nil else { throw Failure.completion }
        await submitted.retirement.wait()
        XCTAssertEqual(result.finishReason, .cancelled)
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
            XCTAssertNil(f.engine.loopForTesting.mtp?.assistantStateCountsForTesting(.init(9552)))
            XCTAssertEqual(f.engine.pagedAttentionWorkBytesReserved, 0)
        }
        let replacement = request(9552, count: 17)
        let expected = try await collect(reference, replacement)
        let actual = try await collect(f, replacement)
        XCTAssertEqual(actual, expected)
        try await stop(f); try await stop(reference)
    }

    func testPartialRealColumnFailureRequiresGenuineDrainBeforeRefund() async throws {
        try lane("partial-column")
        let f = try await fixture()
        var injected = false
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeSubmittedWorkFailureForTesting = {
                guard let work = CBv2NativePagedMTPWork.current, work.completedColumns == 1 else { return }
                injected = true
                throw Failure.injected // second actual column evaluated; no counterfeit output
            }
        }
        let submission = try f.engine.submitWithNativeRetirement(request(9601, count: 17))
        let result = await cbv2SchedCollect(submission.events)
        XCTAssertTrue(injected)
        guard case .error? = result.finishReason else { return XCTFail("partial-column failure was hidden") }
        guard f.engine.nativeCompletionFault == nil else { throw Failure.completion }
        await submission.retirement.wait()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeSubmittedWorkFailureForTesting = nil
            XCTAssertEqual(f.engine.pagedAttentionWorkBytesReserved, 0)
        }
        try await stop(f)
    }

    func testRequiredSecondColumnFenceFailureStaysOwnedAfterLaterSuccessfulFence() async throws {
        try lane("column-fence")
        let f = try await fixture()
        defer {
            _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
        }
        let tracking = try XCTUnwrap(f.engine.loopForTesting.nativeShutdownState)
        var injected = false
        f.engine.loopForTesting.onEngineQueueSync {
            tracking.beforeFenceForTesting = { _ in
                guard let work = CBv2NativePagedMTPWork.current, work.completedColumns == 1 else { return }
                injected = true
                throw Failure.injected
            }
        }
        _ = await cbv2SchedCollect(try f.engine.submit(request(9602, count: 17)))
        XCTAssertTrue(injected)
        let first = await f.engine.shutdownReportingNativeCompletion()
        guard case .incomplete = first else { return XCTFail("failed required fence produced quiescence") }
        XCTAssertGreaterThan(f.owner.bytes, 0); XCTAssertFalse(f.owner.retired)
        XCTAssertTrue(tracking.hasLoans)
        f.engine.loopForTesting.onEngineQueueSync { tracking.beforeFenceForTesting = nil }
        try withError { errors in Stream.gpu.synchronize(); Stream.cpu.synchronize(); try errors.check() }
        let later = await f.engine.shutdownReportingNativeCompletion()
        XCTAssertEqual(later, first)
        XCTAssertGreaterThan(f.owner.bytes, 0); XCTAssertFalse(f.owner.retired)
    }
}
