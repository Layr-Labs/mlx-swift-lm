import Cmlx
import Foundation
import MLX
import MLXHuggingFace
import Tokenizers
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Source-prepared, UNRUN. Actual strict-loaded MiMo192/128/W128/trained
/// three-head profile. Fixture transport is bounded encoded memory, NOT the
/// provider encrypted-store/global-budget proof or selected full-model proof.
final class MiMoV26NativePagedPrefixCompositionTests: XCTestCase {
    private enum Failure: Error { case fixture, capacity, completion, injected }

    func testSameStageUpwardExtensionRefusesOverbudgetAndSettlesOnlyOncePerPhase() throws {
        let admission = AdmissionV2(layerKinds: [], bytesCapacity: 256,
                                     config: .init(watermarkFraction: 0))
        let stage = try admission.reserveCheckpointStage(targetBytes: 96, auxiliaryBytes: 16, scratchBytes: 16)
        XCTAssertThrowsError(try stage.extendForNativePagedPreparation(targetBytes: 1, auxiliaryBytes: 0))
        try stage.settleDestinationAfterEvaluation(targetBytes: 96, auxiliaryBytes: 16)
        let neighbor = try admission.reserveTransient(bytes: 128)
        XCTAssertEqual(admission.bytesReserved, 256)
        XCTAssertThrowsError(try stage.extendForNativePagedPreparation(targetBytes: 16, auxiliaryBytes: 8))
        XCTAssertEqual(stage.targetBytes, 96); XCTAssertEqual(stage.auxiliaryBytes, 16)
        XCTAssertEqual(admission.bytesReserved, 256)
        neighbor.release() // actual ordinary host reservation, no native owner exists
        try stage.extendForNativePagedPreparation(targetBytes: 16, auxiliaryBytes: 8)
        XCTAssertEqual(admission.bytesReserved, 152)
        XCTAssertThrowsError(try stage.extendForNativePagedPreparation(targetBytes: 1, auxiliaryBytes: 0))
        try stage.settleDestinationAfterEvaluation(targetBytes: 104, auxiliaryBytes: 20)
        XCTAssertEqual(admission.bytesReserved, 140)
        XCTAssertThrowsError(try stage.settleDestinationAfterEvaluation(targetBytes: 100, auxiliaryBytes: 16))
        XCTAssertThrowsError(try admission.extendCheckpointStage(identity: stage.identity,
            expectedBytes: 140, additionalBytes: 1), "ledger must reject a replay too")
        stage.closeAfterDroppingOwners()
        XCTAssertEqual(admission.bytesReserved, 0)
        XCTAssertThrowsError(try admission.extendCheckpointStage(identity: stage.identity,
            expectedBytes: 140, additionalBytes: 1))
        XCTAssertThrowsError(try admission.extendCheckpointStage(identity: UUID(),
            expectedBytes: 0, additionalBytes: Int.max))
        XCTAssertEqual(admission.bytesReserved, 0)
    }

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
    /// This records calls from REAL Admission/backing owners. It does not
    /// substitute a page budget, completion receipt, or provider ledger.
    private final class Owner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var c: UInt64 = 0, m: UInt64 = 0
        private var closed = false
        var charge: UInt64 { lock.withLock { c } }
        var coverage: UInt64 { lock.withLock { m } }
        var retired: Bool { lock.withLock { closed } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock { guard !closed, bytes >= m, bytes <= 1 << 30 else { throw Failure.capacity }; c = bytes }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock { guard !closed, bytes >= m, bytes <= c else { throw Failure.capacity }; m = bytes }
        }
        func withdrawCoverage(_ bytes: UInt64) throws {
            try lock.withLock { guard bytes <= m else { throw Failure.capacity }; m -= bytes }
        }
        func retire() { lock.withLock { XCTAssertEqual(c, 0); XCTAssertEqual(m, 0); closed = true } }
    }
    private final class Store: CBv2NativeCompletePrefixCache, @unchecked Sendable {
        let base: CompleteCheckpointFixtureStore
        let identity: CBv2CompleteCheckpointIdentity
        private let jobs = DispatchGroup()
        private let lock = NSLock()
        private var supplied: [CBv2RequestID: CBv2StagedCompleteCheckpoint] = [:]
        private var joined = false
        var didJoin: Bool { lock.withLock { joined } }
        init(identity: CBv2CompleteCheckpointIdentity,
             archives: [CompleteCheckpointFixtureStore.Archive] = []) {
            self.identity = identity
            base = .init(archives: archives, maximumPosition: 256, segmentBytes: 1 << 20)
        }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
            packedBytes <= 32 << 20 && base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
        }
        func takeStaged(requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
                        maximumSequenceLength: Int) -> CBv2StagedCompleteCheckpoint? {
            if let stage = lock.withLock({ supplied.removeValue(forKey: requestID) }) { return stage }
            return base.takeStaged(requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                                  maximumSequenceLength: maximumSequenceLength)
        }
        func supply(_ stage: CBv2StagedCompleteCheckpoint, receipt: CBv2RequestID) {
            lock.withLock { precondition(supplied[receipt] == nil); supplied[receipt] = stage }
        }
        func donate(_ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
                    tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void) {
            jobs.enter()
            base.donate(source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt) { [self] positions in
                completion(positions); jobs.leave()
            }
        }
        func close() {
            let pending = lock.withLock { let old = Array(supplied.values); supplied = [:]; return old }
            pending.forEach { $0.close() }; base.close()
        }
        func closeAndWait() async {
            close()
            await withCheckedContinuation { continuation in
                jobs.notify(queue: .global()) { continuation.resume() }
            }
            lock.withLock { joined = true }
        }
    }
    private struct Fixture: Sendable {
        let container: ModelContainer
        let construction: NativeConstructionWork
        let engine: EngineV2
        let contract: CBv2NativeExecutionContract
        let owner: Owner
        let store: Store
    }
    private struct Built: Sendable { let engine: EngineV2; let contract: CBv2NativeExecutionContract }

    private func lane(_ fault: String? = nil) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_NATIVE_PAGED_TESTS"] == "1", env["MIMO_V26_PAGED_PREFIX_TESTS"] == "1" else {
            throw XCTSkip("Requires the root's exclusive bounded native lane and genuine strict fixture")
        }
        guard env["MIMO_V26_PAGED_PREFIX_FAULT"] == fault else {
            throw XCTSkip("Retained-fault selectors require their own fresh process")
        }
    }

    private func fixture(mtp: Bool = false, depth: Int = 3,
                         archives: [CompleteCheckpointFixtureStore.Archive] = []) async throws -> Fixture {
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
        guard config.hiddenSize <= 64, config.numHiddenLayers <= 4, config.maxPositionEmbeddings >= 256,
              config.vocabularySize >= 32, config.numNextnPredictLayers == 3,
              config.slidingWindow == 128 else { throw Failure.fixture }
        for index in 0..<config.numHiddenLayers {
            let g = try config.attentionGeometry(at: index)
            guard g.headDim == 192, g.valueHeadDim == 128, g.queryHeads == 64,
                  [4, 8].contains(g.keyValueHeads) else { throw Failure.fixture }
        }
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let construction = NativeConstructionWork(), owner = Owner()
        // Test namespace only; real fixture provenance, never another family's
        // identity. Provider tests bind the actual aggregate/binary/encryption.
        let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: try XCTUnwrap(p["conversionManifestSHA256"]),
            promptContractID: "strict-mimo-paged-fixture-tokens-v1", buildID: "source-test",
            numericsFingerprint: "native-paged-serial-mtp-\(mtp)")
        let store = Store(identity: identity, archives: archives)
        let container = try await MiMoV26ModelFactory.loadContainer(session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        let built = try await MiMoV26ModelFactory.withNativeConstruction(container: container,
            retaining: construction) { model, scope -> Built in
            let binding = try model.makeCBv2Binding(enableMTP: mtp, verificationMode: .serialTarget)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let metadata = try model.nativePagedCompletePrefixMetadata(binding: binding,
                bytesCapacity: 768 << 20, maximumConcurrentRequests: 2, maximumQueryTokens: 32,
                maximumPrefillChunk: 16, retaining: scope)
            let resources = try model.makeNativePagedCompletePrefixExecutionResources(binding: binding,
                bytesCapacity: 768 << 20, maximumConcurrentRequests: 2, maximumQueryTokens: 32,
                maximumPrefillChunk: 16, expectedMetadata: metadata, completePrefixCache: store,
                processMemoryOwner: owner, retaining: scope)
            XCTAssertTrue(resources.contract.supportsNativePagedTarget)
            XCTAssertTrue(resources.contract.supportsNativeCompletePrefix)
            XCTAssertEqual(resources.contract.supportsNativePagedSerialMTP, mtp)
            XCTAssertFalse(resources.contract.supportsManagedDecodedMedia)
            XCTAssertFalse(resources.contract.consume(model: binding.adapter, backend: resources.backend,
                cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                completePrefixCache: Store(identity: identity), processMemoryOwner: owner))
            XCTAssertFalse(resources.contract.consume(model: binding.adapter, backend: resources.backend,
                cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                completePrefixCache: store, processMemoryOwner: Owner()))
            XCTAssertFalse(resources.contract.consume(model: binding.adapter, backend: resources.backend,
                cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                mtpVerificationMode: .rectangular, completePrefixCache: store, processMemoryOwner: owner))
            let engine = EngineV2(model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 16, enablePrefixCache: true),
                loopConfig: .init(stepTimeout: 30, watchdogInterval: 0.01, shutdownTimeout: 10),
                completePrefixCache: store, mtpDrafter: binding.assistant,
                mtpConfig: .init(enabled: mtp, maxDraftTokens: depth, maxSpeculativeBatch: 1,
                    fixedDraftTokens: depth, verificationMode: .serialTarget),
                processMemoryOwner: owner, nativeCompletionTracking: true,
                nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            XCTAssertNil(engine.nativeCompletionFault); XCTAssertNil(engine.pagedAttentionWorkInactiveReason)
            XCTAssertTrue(engine.completeCheckpointCodec?.isNativePagedHistorical == true)
            if mtp {
                XCTAssertNil(engine.mtpInactiveReason)
                guard case .bounded? = engine.resolvedMTPAdmission else { throw Failure.completion }
            }
            XCTAssertFalse(resources.contract.consume(model: binding.adapter, backend: resources.backend,
                cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                completePrefixCache: store, processMemoryOwner: owner), "actual engine consumed it once")
            return .init(engine: engine, contract: resources.contract)
        }
        guard case .completed(let receipt) = construction.snapshot.disposition else { throw Failure.completion }
        try await construction.sealForPublication(receipt)
        return .init(container: container, construction: construction, engine: built.engine,
            contract: built.contract, owner: owner, store: store)
    }

    private func request(_ id: UInt64, count: Int = 145, output: Int = 24, prefix: Bool = true) -> CBv2Request {
        .init(id: .init(id), promptTokens: (0..<count).map { 1 + ($0 * 7) % 29 },
            sampling: .init(temperature: 0), maxTokens: output, cacheSalt: "isolated-paged-tenant",
            prefixCacheEnabled: prefix, prefixCacheReceiptID: .init(id + 10_000))
    }
    private func stop(_ f: Fixture) async throws {
        try await f.container.perform { context in
            try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                .beginNativePagedRetirement(executionContractID: f.contract.id)
        }
        guard case .quiescent(let receipt) = await f.engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction); throw Failure.completion
        }
        XCTAssertTrue(f.store.didJoin)
        try await f.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            try model.releaseNativePagedAfterNativeRetirement(receipt)
            XCTAssertNil(model.nativePagedPreparation); XCTAssertNil(model.resources.nativePagedValidator)
            XCTAssertThrowsError(try model.releaseNativePagedAfterNativeRetirement(receipt))
        }
        XCTAssertTrue(f.owner.retired); XCTAssertEqual(f.owner.charge, 0); XCTAssertEqual(f.owner.coverage, 0)
        XCTAssertEqual(f.engine.admissionForTesting.bytesReserved, 0)
    }
    private func collect(_ f: Fixture, _ request: CBv2Request) async throws -> CBv2SchedCollected {
        let submission = try f.engine.submitWithNativeRetirement(request)
        let result = await cbv2SchedCollect(submission.events)
        guard result.finishReason == .length, result.tokens.count == request.maxTokens else {
            _ = await f.engine.shutdownReportingNativeCompletion()
            throw Failure.completion
        }
        await submission.retirement.wait()
        return result
    }

    private func archiveBytes(_ archive: CompleteCheckpointFixtureStore.Archive) throws -> [Data] {
        try archive.manifest.tensors.indices.map { index in
            var result = Data()
            for piece in archive.chunks.filter({ $0.tensor == index }).sorted(by: { $0.offset < $1.offset }) {
                guard piece.offset == result.count else { throw Failure.completion }
                result.append(piece.bytes)
            }
            guard result.count == archive.manifest.tensors[index].byteCount else { throw Failure.completion }
            return result
        }
    }

    private func witnessed(_ f: Fixture, _ request: CBv2Request) async throws
        -> (CBv2SchedCollected, Witness, CBv2CheckpointReservation) {
        let witness = Witness()
        let permit = try f.engine.admissionForTesting.reserveTransient(bytes: witness.bound)
        f.engine.loopForTesting.onEngineQueueSync { [weak engine = f.engine] in
            guard let loop = engine?.loopForTesting else { return }
            loop.nativePagedPrefixAdoptedForTesting = { id, rows, state in
                guard id == request.id else { return }
                witness.adoption(rows: rows, state: state,
                    assistant: loop.mtp?.drafter as? MiMoV26MTPAssistant)
                XCTAssertNil(loop.mtp?.pendingHistoryCarry(for: id), "no carry is synthesized at adoption")
            }
            loop.nativeRetirementBoundaryForTesting = { [weak engine] phase, step in
                guard phase == "beforeMTPFinalization", let step,
                      step.mtpRound?.verify?.rows.first?.id == request.id,
                      let assistant = engine?.loopForTesting.mtp?.drafter as? MiMoV26MTPAssistant else { return }
                witness.observe(step, assistant: assistant)
            }
        }
        do {
            let result = try await collect(f, request)
            f.engine.loopForTesting.onEngineQueueSync {
                f.engine.loopForTesting.nativePagedPrefixAdoptedForTesting = nil
                f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
            }
            guard witness.failure == nil else { throw witness.failure! }
            return (result, witness, permit)
        } catch {
            f.engine.loopForTesting.onEngineQueueSync {
                f.engine.loopForTesting.nativePagedPrefixAdoptedForTesting = nil
                f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
            }
            witness.discard(); permit.release()
            throw error
        }
    }

    func testIssuedPagedPrefixOffAndSerialOnRestoreActualBytesAndFirstSuffixContinuation() async throws {
        try lane()
        // Partial and wrapped native W128 histories. P+1 enforces ONE actual
        // suffix token before the first trained-head round, not an overwrite.
        for depth in [0, 1, 2, 3] {
          var observedRejectionWithContinuation = false
          for count in [33, 145] {
            var fixtures: [Fixture] = []
            do {
                let cold = try await fixture(mtp: depth > 0, depth: max(1, depth))
                fixtures.append(cold)
                let hit = try await fixture(mtp: depth > 0, depth: max(1, depth))
                fixtures.append(hit)
                let donor = request(9200, count: count)
                _ = try await collect(hit, donor)
                let archive = try XCTUnwrap(hit.store.base.saved.max { $0.manifest.position < $1.manifest.position })
                XCTAssertEqual(archive.manifest.position, count - 1)
                XCTAssertEqual(archive.manifest.assistantCodecID != nil, depth > 0)
                let expectedBytes = try archiveBytes(archive)
                var cached = request(9201, count: count)
                XCTAssertTrue(try hit.store.base.stage(engine: hit.engine, request: cached))
                let before = hit.engine.mtpMetricsSnapshot()
                let actual = try await witnessed(hit, cached)
                defer { actual.1.discard(); actual.2.release() }
                cached.prefixCacheEnabled = false
                let reference = try await witnessed(cold, cached)
                defer { reference.1.discard(); reference.2.release() }
                XCTAssertEqual(actual.0.tokens, reference.0.tokens)
                XCTAssertEqual(actual.0.usage?.prefixCachePrefillTokensSaved, count - 1)
                XCTAssertEqual(reference.0.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
                let adoption = try XCTUnwrap(actual.1.adopted)
                XCTAssertEqual(adoption.tensors, expectedBytes,
                    "all target roles and all nine assistant tensors/metadata BEFORE suffix")
                for (index, layer) in try XCTUnwrap(archive.manifest.attentionLayers).enumerated() {
                    XCTAssertEqual(adoption.metadata[index][0], count - 1)
                    XCTAssertEqual(adoption.metadata[index][1], min(count - 1, layer.window ?? count - 1))
                    XCTAssertEqual(adoption.metadata[index][2], layer.window ?? 0)
                }
                XCTAssertNil(reference.1.adopted)
                if depth > 0 {
                    let after = try XCTUnwrap(hit.engine.mtpMetricsSnapshot())
                    XCTAssertGreaterThan(after.draftedTokens - (before?.draftedTokens ?? 0), 0)
                    XCTAssertEqual(actual.1.rounds.count, 2, "requires real continuation after a verify")
                    XCTAssertEqual(actual.1.rounds, reference.1.rounds,
                        "complete meaningful target/head/draft/feature state, exact native bytes")
                    XCTAssertEqual(actual.1.roundBases, reference.1.roundBases)
                    XCTAssertEqual(actual.1.accepted, reference.1.accepted)
                    XCTAssertEqual(actual.1.roundBases[1],
                        actual.1.roundBases[0] + actual.1.accepted[0] + 1,
                        "next real round must observe the actual accepted/rollback frontier")
                    observedRejectionWithContinuation = observedRejectionWithContinuation || actual.1.accepted[0] < depth
                }
                // Retained host copies drop BEFORE their actual Admission C.
                actual.1.discard(); actual.2.release()
                reference.1.discard(); reference.2.release()
                try await stop(hit); try await stop(cold)
            } catch {
                for value in fixtures { try? await stop(value) }
                throw error
            }
          }
          XCTAssertTrue(depth == 0 || observedRejectionWithContinuation,
              "INCOMPLETE natural rollback coverage; never script proposals or relabel it a pass")
        }
    }

    func testRepeatedImportsSoftGrantGrowthShrinkRefusalAndActualMapRetirement() async throws {
        try lane()
        let f = try await fixture()
        do {
            let input = request(9300, output: 1)
            let authority = try await collect(f, input)
            let backend = try XCTUnwrap(f.engine.loopForTesting.backend as? PagedKVBackend)
            let binding = try XCTUnwrap(backend.nativeModelBinding)
            // This exceeds the ORIGINAL soft capacity. No new "reassembly"
            // ceiling may be smuggled in to simplify host accounting.
            f.engine.updateKVBytesCapacity(896 << 20)
            XCTAssertGreaterThan(try XCTUnwrap(backend.pool.segmentGrant).snapshot().bytes,
                                 backend.pool.config.capacityBytes)
            var lastPhysical = backend.pool.bytesMaterialized
            var lastHost = f.engine.loopForTesting.onEngineQueueSync { binding.hostMetadataBytesForTesting }
            XCTAssertGreaterThan(lastHost, 0)
            for index in 0..<3 {
                let next = request(UInt64(9301 + index), output: 1)
                XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: next))
                let result = try await collect(f, next)
                XCTAssertEqual(result.tokens, authority.tokens)
                XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved, 144)
                let physical = backend.pool.bytesMaterialized
                let host = f.engine.loopForTesting.onEngineQueueSync { binding.hostMetadataBytesForTesting }
                XCTAssertGreaterThan(physical, lastPhysical, "actual zero-copy imports keep their native pages")
                XCTAssertGreaterThanOrEqual(host, lastHost)
                XCTAssertGreaterThanOrEqual(f.owner.charge, UInt64(physical + host))
                lastPhysical = physical; lastHost = host
            }
            let charge = f.owner.charge
            f.engine.updateKVBytesCapacity(1024)
            XCTAssertEqual(backend.pool.bytesMaterialized, lastPhysical)
            XCTAssertEqual(f.owner.charge, charge, "grant shrink cannot refund retained maps/backing")
            XCTAssertThrowsError(try f.store.base.stage(engine: f.engine, request: request(9310, output: 1)))
            let closed = await cbv2SchedWait {
                f.engine.loopForTesting.onEngineQueueSync { binding.canStartPoolRetirement }
            }
            XCTAssertTrue(closed, "the refused actual import loan must finish, leaving only the real pool lifetime")
            XCTAssertEqual(backend.pool.bytesMaterialized, lastPhysical)
            XCTAssertEqual(f.engine.loopForTesting.onEngineQueueSync { binding.hostMetadataBytesForTesting }, lastHost)
            XCTAssertEqual(f.owner.charge, charge)
            f.engine.updateKVBytesCapacity(896 << 20)
            let retry = request(9311, output: 1)
            XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: retry))
            let result = try await collect(f, retry)
            XCTAssertEqual(result.tokens, authority.tokens)
            XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved, 144)
            try await stop(f)
            XCTAssertTrue(backend.pool.groups.isEmpty, "actual maps must drop before their final allowance")
            XCTAssertEqual(binding.hostMetadataBytesForTesting, 0)
        } catch { try? await stop(f); throw error }
    }

    func testPrefixAdoptionCoexistsWithActualOrdinaryNeighborAndNoDuplicateChargeOwner() async throws {
        try lane()
        let f = try await fixture(mtp: true), cold = try await fixture(mtp: true)
        let suspended = expectation(description: "actual ordinary neighbor submitted")
        let adopted = expectation(description: "actual prefix row adopted while neighbor owns pages")
        var neighbor = request(9401, count: 209, output: 32, prefix: false)
        neighbor.sampling.repetitionPenalty = 1.05 // existing ordinary-row fallback
        let cached = request(9402, output: 64)
        var didSuspend = false
        do {
            _ = try await collect(f, request(9400, output: 1))
            XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: cached))
            let loop = f.engine.loopForTesting
            loop.onEngineQueueSync {
                loop.nativeShutdownState?.afterSubmissionForTesting = {
                    guard !didSuspend, loop.kvStates[neighbor.id] != nil else { return }
                    didSuspend = true
                    loop.suspendStepExecutionAtCountForTesting = loop.stepCount
                    suspended.fulfill()
                }
                loop.nativePagedPrefixAdoptedForTesting = { id, rows, state in
                    guard id == cached.id else { return }
                    XCTAssertNotNil(loop.kvStates[neighbor.id])
                    XCTAssertEqual(rows.compactMap { $0 }.count, loop.layerKinds.count)
                    XCTAssertNotNil(state)
                    adopted.fulfill()
                }
            }
            let ordinary = try f.engine.submitWithNativeRetirement(neighbor)
            await fulfillment(of: [suspended], timeout: 8)
            guard loop.onEngineQueueSync({ didSuspend }) else { throw Failure.completion }
            let prefix = try f.engine.submitWithNativeRetirement(cached)
            await fulfillment(of: [adopted], timeout: 8)
            loop.onEngineQueueSync {
                loop.nativeShutdownState?.afterSubmissionForTesting = nil
                loop.nativePagedPrefixAdoptedForTesting = nil
                loop.suspendStepExecutionAtCountForTesting = nil
            }
            let actualNeighbor = await cbv2SchedCollect(ordinary.events)
            let actualPrefix = await cbv2SchedCollect(prefix.events)
            await ordinary.retirement.wait(); await prefix.retirement.wait()
            XCTAssertEqual(actualNeighbor.finishReason, .length); XCTAssertEqual(actualPrefix.finishReason, .length)
            XCTAssertEqual(actualPrefix.usage?.prefixCachePrefillTokensSaved, 144)
            let referenceNeighbor = try await collect(cold, neighbor)
            var plain = cached; plain.prefixCacheEnabled = false
            let referencePrefix = try await collect(cold, plain)
            XCTAssertEqual(actualNeighbor.tokens, referenceNeighbor.tokens)
            XCTAssertEqual(actualPrefix.tokens, referencePrefix.tokens)
            XCTAssertNil(f.engine.nativeCompletionFault)
            try await stop(f); try await stop(cold)
        } catch {
            f.engine.loopForTesting.onEngineQueueSync {
                f.engine.loopForTesting.nativeShutdownState?.afterSubmissionForTesting = nil
                f.engine.loopForTesting.nativePagedPrefixAdoptedForTesting = nil
                f.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            try? await stop(f); try? await stop(cold)
            throw error
        }
    }

    func testPreparedCancellationAndStaleGrantLeaveNoAdoptedStateAndPermitSameIDRetry() async throws {
        try lane()
        for cancel in [true, false] {
            let f = try await fixture(mtp: true), cold = try await fixture(mtp: true)
            do {
                _ = try await collect(f, request(9450, output: 1))
                let backend = try XCTUnwrap(f.engine.loopForTesting.backend as? PagedKVBackend)
                let binding = try XCTUnwrap(backend.nativeModelBinding)
                let loop = f.engine.loopForTesting
                var input = request(9451)
                let baselineCharge = f.owner.charge, baselineCoverage = f.owner.coverage
                let baselinePhysical = backend.pool.bytesMaterialized
                let baselineHost = loop.onEngineQueueSync { binding.hostMetadataBytesForTesting }
                XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: input))
                var preparedCount = 0, adoptedCount = 0
                var oldGeneration: UInt64?
                loop.onEngineQueueSync {
                    loop.nativePagedPrefixAdoptedForTesting = { id, _, _ in
                        if id == input.id { adoptedCount += 1 }
                    }
                    loop.nativePagedPrefixPreparedForTesting = { id, pages in
                        guard id == input.id else { return }
                        preparedCount += 1
                        XCTAssertTrue(pages.scalarWitness.ready, "genuine native preparation/fences completed")
                        XCTAssertGreaterThan(pages.scalarWitness.rows, 0)
                        XCTAssertGreaterThan(pages.scalarWitness.hostBound, 0)
                        oldGeneration = pages.streamGeneration
                        XCTAssertNil(loop.kvStates[id]); XCTAssertNil(loop.mtp?.assistantStateCountsForTesting(id))
                        if cancel {
                            f.engine.cancel(id) // real pending-admission cancellation, not a fake outcome
                        } else {
                            // Legal growth invalidates this captured epoch while
                            // keeping ample capacity. No pool mutation or refund.
                            f.engine.updateKVBytesCapacity((768 << 20) + 4096)
                        }
                    }
                }
                let policy = CBv2FirstTokenDeadlineAdmission(
                    deadline: .now.advanced(by: .seconds(60)),
                    conservativePrefillTokensPerSecond: 1000,
                    conservativeDecodeTokensPerSecond: 1000)
                do {
                    _ = try await f.engine.submit(input, firstTokenDeadline: policy)
                    XCTFail("prepared refusal must not become admitted or silently cold-prefill")
                } catch {
                    if cancel { XCTAssertTrue(error is CancellationError) }
                    else {
                        guard case .capacityExhausted? = error as? CBv2KVError else { throw error }
                    }
                }
                loop.onEngineQueueSync {
                    loop.nativePagedPrefixPreparedForTesting = nil
                    loop.nativePagedPrefixAdoptedForTesting = nil
                }
                XCTAssertEqual(preparedCount, 1); XCTAssertEqual(adoptedCount, 0)
                let retired = await cbv2SchedWait {
                    loop.onEngineQueueSync {
                        binding.canStartPoolRetirement && loop.kvStates[input.id] == nil
                            && loop.mtp?.assistantStateCountsForTesting(input.id) == nil
                            && loop.registeredStreamGeneration(for: input.id) == nil
                    } && f.owner.charge == baselineCharge && f.owner.coverage == baselineCoverage
                }
                XCTAssertTrue(retired, "actual stage/suffix/map loans must retire before ID reuse")
                XCTAssertEqual(backend.pool.bytesMaterialized, baselinePhysical)
                XCTAssertEqual(loop.onEngineQueueSync { binding.hostMetadataBytesForTesting }, baselineHost)
                XCTAssertNil(f.engine.nativeCompletionFault)
                input.prefixCacheReceiptID = .init(29_451)
                XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: input))
                var retryGeneration: UInt64?
                loop.onEngineQueueSync {
                    loop.nativePagedPrefixPreparedForTesting = { id, pages in
                        if id == input.id { retryGeneration = pages.streamGeneration }
                    }
                }
                let actual = try await collect(f, input)
                loop.onEngineQueueSync { loop.nativePagedPrefixPreparedForTesting = nil }
                XCTAssertNotNil(oldGeneration); XCTAssertNotNil(retryGeneration)
                XCTAssertNotEqual(oldGeneration, retryGeneration)
                XCTAssertEqual(actual.usage?.prefixCachePrefillTokensSaved, 144)
                input.prefixCacheEnabled = false
                let reference = try await collect(cold, input)
                XCTAssertEqual(actual.tokens, reference.tokens)
                try await stop(f); try await stop(cold)
            } catch {
                f.engine.loopForTesting.onEngineQueueSync {
                    f.engine.loopForTesting.nativePagedPrefixPreparedForTesting = nil
                    f.engine.loopForTesting.nativePagedPrefixAdoptedForTesting = nil
                }
                try? await stop(f); try? await stop(cold)
                throw error
            }
        }
    }

    func testPartialSuffixAllocationFailureRetainsActualBufferAndOrdinaryNeighbor() async throws {
        try lane("suffix-allocation")
        let f = try await fixture(mtp: true)
        defer {
            _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
        }
        _ = try await collect(f, request(9470, output: 1))
        let cached = request(9472, output: 96) // actual 241-token N promise
        XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: cached))
        var neighbor = request(9471, count: 209, output: 32, prefix: false)
        neighbor.sampling.repetitionPenalty = 1.05
        let loop = f.engine.loopForTesting
        let backend = try XCTUnwrap(loop.backend as? PagedKVBackend)
        let tracking = try XCTUnwrap(loop.nativeShutdownState)
        let suspended = expectation(description: "real ordinary row holds its N page promise")
        var didSuspend = false, allocations = 0, evaluatedBytes = 0
        loop.onEngineQueueSync {
            tracking.afterSubmissionForTesting = {
                guard !didSuspend, loop.kvStates[neighbor.id] != nil else { return }
                didSuspend = true
                loop.suspendStepExecutionAtCountForTesting = loop.stepCount
                suspended.fulfill()
            }
        }
        let ordinary = try f.engine.submitWithNativeRetirement(neighbor)
        await fulfillment(of: [suspended], timeout: 8)
        guard loop.onEngineQueueSync({ didSuspend }) else { throw Failure.completion }
        let oldCharge = f.owner.charge, oldPhysical = backend.pool.bytesMaterialized
        loop.onEngineQueueSync {
            let original = backend.pool.slabEval
            backend.pool.slabEval = { array in
                try original(array) // genuine newly allocated/evaluated native suffix buffer
                guard let info = try array.evaluatedBufferInfo(), info.allocatedBytes > 0 else {
                    throw Failure.completion
                }
                allocations += 1; evaluatedBytes += info.allocatedBytes
                throw Failure.injected
            }
            tracking.afterSubmissionForTesting = nil
        }
        let failed = await cbv2SchedCollect(try f.engine.submit(cached))
        XCTAssertTrue(failed.tokens.isEmpty)
        XCTAssertGreaterThan(allocations, 0, "INCOMPLETE unless real missing-suffix allocation occurred")
        XCTAssertGreaterThan(evaluatedBytes, 0)
        XCTAssertEqual(backend.pool.bytesMaterialized, oldPhysical, "private partial growth was never published")
        guard case .incomplete(let first) = await f.engine.shutdownReportingNativeCompletion() else {
            throw Failure.completion
        }
        XCTAssertEqual(first.reason, .nativeWorkFailed)
        XCTAssertGreaterThanOrEqual(f.owner.charge, oldCharge)
        XCTAssertGreaterThan(f.owner.coverage, 0); XCTAssertTrue(tracking.hasLoans)
        XCTAssertFalse(f.owner.retired)
        // No retirement.wait or fake successful completion on an intentionally
        // sticky native failure. The actual request/stream remains owned.
        withExtendedLifetime(ordinary) {}
    }

    func testFailedRequiredPagedImportFenceKeepsRealStageMapsRootsAndFirstOutcome() async throws {
        try lane("required-fence")
        let f = try await fixture(mtp: true)
        defer {
            // A required native failure is restart-only. Preserve actual
            // objects/charges; no successful receipt or manual reset is made.
            _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
        }
        _ = try await collect(f, request(9500, output: 1))
        let cached = request(9501)
        XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: cached))
        let tracking = try XCTUnwrap(f.engine.loopForTesting.nativeShutdownState)
        let before = f.owner.charge
        var fences = 0
        f.engine.loopForTesting.onEngineQueueSync {
            tracking.beforeFenceForTesting = { _ in fences += 1; throw Failure.injected }
        }
        let result = await cbv2SchedCollect(try f.engine.submit(cached))
        XCTAssertTrue(result.tokens.isEmpty)
        XCTAssertGreaterThan(fences, 0)
        guard case .incomplete(let first) = await f.engine.shutdownReportingNativeCompletion() else {
            throw Failure.completion
        }
        XCTAssertEqual(first.reason, .nativeWorkFailed)
        XCTAssertGreaterThanOrEqual(f.owner.charge, before)
        XCTAssertGreaterThan(f.owner.coverage, 0); XCTAssertTrue(tracking.hasLoans)
        let held = f.owner.charge
        f.engine.loopForTesting.onEngineQueueSync { tracking.beforeFenceForTesting = nil }
        try withError { fault in Stream.gpu.synchronize(); Stream.cpu.synchronize(); try fault.check() }
        guard case .incomplete(let later) = await f.engine.shutdownReportingNativeCompletion() else {
            throw Failure.completion
        }
        XCTAssertEqual(first, later); XCTAssertEqual(f.owner.charge, held)
        XCTAssertTrue(tracking.hasLoans); XCTAssertFalse(f.owner.retired)
    }

    private struct Snapshot: Equatable {
        let tensors: [Data]
        let metadata: [[Int]]
    }
    /// Borrow-only, post-fence reader. Never eval/asData/slice/reshape/astype;
    /// the actual Admission host allowance bounds every retained copy.
    private final class Witness {
        private(set) var adopted: Snapshot?
        private(set) var rounds: [Snapshot] = []
        private(set) var roundBases: [Int] = []
        private(set) var accepted: [Int] = []
        private(set) var failure: Error?
        private var used = 0
        let bound = 64 << 20
        func discard() { adopted = nil; rounds.removeAll(); roundBases.removeAll(); accepted.removeAll() }
        private func charge(_ bytes: Int) throws {
            let next = used.addingReportingOverflow(bytes)
            guard bytes >= 0, !next.overflow, next.partialValue <= bound else { throw Failure.capacity }
            used = next.partialValue
        }
        private func read(_ array: MLXArray) throws -> Data {
            guard let info = try array.evaluatedBufferInfo(),
                  let rawStrides = mlx_array_strides(array.ctx), array.ndim <= 4,
                  array.shape.allSatisfy({ $0 > 0 }) else { throw Failure.completion }
            let shape = array.shape, item = array.dtype.size
            let strides = shape.indices.map { Int(rawStrides[$0]) }
            guard strides.allSatisfy({ $0 >= 0 }) else { throw Failure.completion }
            var last = 0
            for (count, stride) in zip(shape, strides) {
                let product = (count - 1).multipliedReportingOverflow(by: stride)
                let next = last.addingReportingOverflow(product.partialValue)
                guard !product.overflow, !next.overflow else { throw Failure.capacity }
                last = next.partialValue
            }
            let nextElement = last.addingReportingOverflow(1)
            guard !nextElement.overflow else { throw Failure.capacity }
            let extent = nextElement.partialValue.multipliedReportingOverflow(by: item)
            let end = info.dataOffset.addingReportingOverflow(extent.partialValue)
            guard !extent.overflow, !end.overflow, info.dataOffset >= 0,
                  end.partialValue <= info.allocatedBytes,
                  let pointer = mlx_array_data_uint8(array.ctx) else { throw Failure.completion }
            try charge(array.nbytes)
            return withExtendedLifetime(array) {
                if info.isRowContiguous { return Data(bytes: pointer, count: array.nbytes) }
                var result = Data(count: array.nbytes)
                result.withUnsafeMutableBytes { destination in
                    for i in 0..<array.size {
                        var rest = i, offset = 0
                        for axis in shape.indices.reversed() {
                            offset += (rest % shape[axis]) * strides[axis]; rest /= shape[axis]
                        }
                        destination.baseAddress!.advanced(by: i * item).copyMemory(
                            from: pointer.advanced(by: offset * item), byteCount: item)
                    }
                }
                return result
            }
        }
        private func target(_ row: PagedSequenceKV, through end: Int) throws -> [Data] {
            let start = row.windowSize.map { max(row.baseOffset, end - $0) } ?? 0
            guard start >= row.oldestValidPosition, end <= row.absoluteOffset, end > start else {
                throw Failure.completion
            }
            let key = row.groupKey, pool = row.pool, group = pool.group(key), s = pool.config.pageSize
            return try [false, true].map { values in
                let width = values ? key.valueHeadDim : key.headDim
                let rowBytes = width * key.dtype.size
                let total = key.kvHeads * (end - start) * rowBytes
                try charge(total)
                var result = Data(count: total)
                try result.withUnsafeMutableBytes { destination in
                    for head in 0..<key.kvHeads { for token in start..<end {
                        let logical = token / s, index = row.ringPages.map { logical % $0 } ?? logical
                        guard row.table.indices.contains(index) else { throw Failure.completion }
                        let page = row.table[index], segment = group.segment(for: page)
                        let array = segment.storage
                        guard let info = try array.evaluatedBufferInfo(), info.isRowContiguous,
                              info.dataOffset == 0, info.dataElements == array.size,
                              let pointer = mlx_array_data_uint8(array.ctx) else { throw Failure.completion }
                        let local = Int(page) - segment.pages.lowerBound
                        let element = (((local * key.kvHeads + head) * s + token % s) * width)
                            + (values ? segment.valueOffset : 0)
                        guard element >= 0, element * key.dtype.size + rowBytes <= array.nbytes else {
                            throw Failure.completion
                        }
                        let output = (head * (end - start) + token - start) * rowBytes
                        destination.baseAddress!.advanced(by: output).copyMemory(
                            from: pointer.advanced(by: element * key.dtype.size), byteCount: rowBytes)
                    } }
                }
                return result
            }
        }
        private func heads(_ cache: MiMoV26MTPRequestCache) throws -> ([Data], [[Int]]) {
            let arrays = cache.innerState(), rows = cache.retainedHistoryMetadataForTesting
            guard arrays.count == 6, rows.count == 3 else { throw Failure.completion }
            let numbers = try rows.enumerated().map { depth, strings -> [Int] in
                let value = strings.compactMap(Int.init)
                guard value.count == 6, value[0] == depth + 1, value[1] == 0, value[2] > 0,
                      value[3] == 256, value[4] == cache.consumedTokenCounts[depth],
                      value[0] + value[4] == cache.nextTokenPositions[depth] else { throw Failure.completion }
                return value + [min(value[4], value[2])]
            }
            for depth in 0..<3 {
                guard arrays[depth * 2].dtype == .bfloat16, arrays[depth * 2 + 1].dtype == .bfloat16,
                      arrays[depth * 2].ndim == 4, arrays[depth * 2 + 1].ndim == 4,
                      arrays[depth * 2].dim(2) == numbers[depth][6],
                      arrays[depth * 2 + 1].dim(2) == numbers[depth][6] else { throw Failure.completion }
            }
            return (try arrays.map(read), numbers)
        }
        func adoption(rows: [CBv2SequenceKV?], state: (any CBv2MTPRequestState)?,
                      assistant: MiMoV26MTPAssistant?) {
            do {
                guard adopted == nil else { throw Failure.completion }
                try charge(64 << 10)
                var data: [Data] = [], metadata: [[Int]] = []
                for row in rows {
                    let value = try XCTUnwrap(row as? PagedSequenceKV)
                    data += try target(value, through: value.absoluteOffset)
                    metadata.append([value.absoluteOffset, value.retainedCount, value.windowSize ?? 0])
                }
                if let state {
                    let value = try XCTUnwrap(state as? MiMoV26MTPState), cache = try XCTUnwrap(value.cache)
                    guard value.owner === assistant, value.generation == assistant?.generation,
                          value.prefixRestoredBoundary == value.observedCount,
                          !value.hasRequiredRestoredSuffix, value.round == nil,
                          value.pendingTokens == nil, value.pendingHidden == nil, value.pendingLastToken == nil,
                          !value.isReleased, !value.hasUnmeasuredResidency else { throw Failure.completion }
                    let actual = try heads(cache)
                    data += actual.0
                    data.append(try read(XCTUnwrap(value.tail)))
                    data.append(try read(XCTUnwrap(value.prefixObservedTokens)))
                    let words = actual.1.flatMap { $0.map { Int64($0).littleEndian } }
                    try charge(words.count * 8)
                    data.append(words.withUnsafeBufferPointer {
                        Data(bytes: $0.baseAddress!, count: $0.count * MemoryLayout<Int64>.size)
                    })
                    metadata += actual.1
                } else { guard assistant == nil else { throw Failure.completion } }
                adopted = .init(tensors: data, metadata: metadata)
            } catch { failure = error }
        }
        func observe(_ step: CBv2InFlightStep, assistant: MiMoV26MTPAssistant) {
            guard rounds.count < 2 else { return }
            do {
                guard let verify = step.mtpRound?.verify, verify.rows.count == 1,
                      let state = verify.rows[0].assistantState as? MiMoV26MTPState,
                      let committed = state.cache, let draft = state.round,
                      let range = step.computedRanges[verify.rows[0].id],
                      range.count == verify.k + 1,
                      state.owner === assistant, state.generation == assistant.generation,
                      !state.isReleased, !state.hasUnmeasuredResidency,
                      state.hasRequiredRestoredSuffix, state.observedCount == range.lowerBound,
                      state.pendingTokens == nil, state.pendingHidden == nil, state.pendingLastToken == nil,
                      state.stagedInputCount == verify.k else { throw Failure.completion }
                try charge(64 << 10)
                var data: [Data] = [], metadata: [[Int]] = []
                for row in verify.rows[0].storageRows {
                    let value = try XCTUnwrap(row as? PagedSequenceKV)
                    guard value.absoluteOffset == range.upperBound else { throw Failure.completion }
                    data += try target(value, through: range.lowerBound)
                    data += try target(value, through: range.upperBound)
                    // Effective live frontier, not irrelevant pre-import
                    // history below the last W tokens or allocator padding.
                    metadata.append([value.absoluteOffset, value.retainedCount, value.windowSize ?? 0,
                                     max(value.baseOffset, range.lowerBound - (value.windowSize ?? range.lowerBound))])
                }
                for cache in [committed, draft.cache] {
                    let actual = try heads(cache); data += actual.0; metadata += actual.1
                }
                for array in [state.tail, Optional(verify.lastHidden)] + draft.inputs.map(Optional.some) {
                    data.append(try read(XCTUnwrap(array)))
                }
                data.append(try read(XCTUnwrap(state.prefixObservedTokens)))
                let packet = try read(verify.acceptancePacket)
                guard verify.acceptancePacket.dtype == .int32, packet.count % 4 == 0 else {
                    throw Failure.completion
                }
                let ids = packet.withUnsafeBytes { raw in
                    (0..<(raw.count / 4)).map { Int(raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
                }
                guard ids.count == 2 * verify.k + 1 else { throw Failure.completion }
                var n = 0
                while n < verify.k && ids[n] == ids[verify.k + n] { n += 1 }
                metadata += [[verify.k, state.observedCount, state.committedInputCount, state.stagedInputCount,
                              state.retainedFeatureRows, state.maximumSequenceLength ?? -1],
                             state.headInputCounts, committed.nextTokenPositions, state.headProposalCounts, ids]
                roundBases.append(range.lowerBound); accepted.append(n)
                rounds.append(.init(tensors: data, metadata: metadata))
            } catch { failure = error }
        }
    }
}
