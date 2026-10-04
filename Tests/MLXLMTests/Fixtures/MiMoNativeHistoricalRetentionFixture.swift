import Foundation
import MLXHuggingFace
import Tokenizers
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Strict small native-contiguous target with model-issued execution resources.
/// Encoded fixture transport does not qualify provider encryption or full weights.
struct MiMoNativeHistoricalRetentionFixture: Sendable {
    enum Failure: Error { case fixture, capacity, completion }
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
    final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var charge: UInt64 = 0, materialized: UInt64 = 0
        private var closed = false
        var bytes: UInt64 { lock.withLock { charge } }
        var retired: Bool { lock.withLock { closed } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= 256 << 20 else {
                    throw Failure.capacity
                }
                charge = bytes
            }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= charge else {
                    throw Failure.capacity
                }
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
                XCTAssertEqual(charge, 0)
                XCTAssertEqual(materialized, 0)
                closed = true
            }
        }
    }
    final class Store: CBv2NativeCompletePrefixCache, @unchecked Sendable {
        let identity: CBv2CompleteCheckpointIdentity
        let base: CompleteCheckpointFixtureStore
        private let jobs = DispatchGroup()
        init(
            identity: CBv2CompleteCheckpointIdentity,
            archives: [CompleteCheckpointFixtureStore.Archive]
        ) {
            self.identity = identity
            base = .init(archives: archives, maximumPosition: 1_024, segmentBytes: 1 << 20)
        }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
            packedBytes <= 32 << 20
                && base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
        }
        func takeStaged(
            requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
            maximumSequenceLength: Int
        ) -> CBv2StagedCompleteCheckpoint? {
            base.takeStaged(
                requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                maximumSequenceLength: maximumSequenceLength)
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
                jobs.leave()
            }
        }
        func close() { base.close() }
        func closeAndWait() async {
            close()
            await withCheckedContinuation { continuation in
                jobs.notify(queue: .global()) { continuation.resume() }
            }
        }
    }
    private struct Built: Sendable {
        let engine: EngineV2
        let contract: CBv2NativeExecutionContract
    }
    let container: ModelContainer
    let construction: NativeConstructionWork
    let engine: EngineV2
    let contract: CBv2NativeExecutionContract
    let store: Store
    let process: ProcessOwner

    static func load(
        mtp: Bool,
        archives: [CompleteCheckpointFixtureStore.Archive] = []
    ) async throws -> Self {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1",
            let path = environment["MIMO_V26_SERIAL_LOAD_FIXTURES"]
        else {
            throw XCTSkip("Requires existing strict native-contiguous fixture and exclusive lane")
        }
        let fixtures = URL(fileURLWithPath: path)
        let root = fixtures.appendingPathComponent("tiny-bf16")
        let provenanceObject = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with:
                    Data(contentsOf: fixtures.appendingPathComponent("provenance.json")))
                as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(
            artifactID: XCTUnwrap(provenanceObject["artifactID"]),
            sourceRepository: XCTUnwrap(provenanceObject["sourceRepository"]),
            sourceRevision: XCTUnwrap(provenanceObject["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(provenanceObject["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 4 << 20, maximumTotalFileBytes: 16 << 20))
        let config = plan.bundlePlan.configuration
        guard config.hiddenSize <= 64, config.numHiddenLayers <= 4,
            config.maxPositionEmbeddings >= 1_024, config.vocabularySize >= 32,
            config.numNextnPredictLayers == 3, CBv2AttentionV1.queryBlockSize == 128
        else { throw Failure.fixture }
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(
            request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let construction = NativeConstructionWork()
        defer {
            if construction.snapshot.isRetainedFault { _ = Unmanaged.passRetained(construction) }
        }
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        let process = ProcessOwner()
        let identity = CBv2CompleteCheckpointIdentity(
            modelAggregateHash: try XCTUnwrap(provenanceObject["conversionManifestSHA256"]),
            promptContractID: "strict-mimo-contiguous-fixture-tokens-v1", buildID: "source-test",
            numericsFingerprint: "native-contiguous-serial-mtp-\(mtp)")
        let store = Store(identity: identity, archives: archives)
        let built = try await MiMoV26ModelFactory.withNativeConstruction(
            container: container,
            retaining: construction
        ) { model, scope -> Built in
            let binding = try model.makeCBv2Binding(enableMTP: mtp)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let metadata = try model.nativeCompletePrefixMetadata(
                binding: binding, retaining: scope)
            let resources = try model.makeNativeCompletePrefixExecutionResources(
                binding: binding,
                bytesCapacity: 256 << 20, expectedMetadata: metadata, completePrefixCache: store,
                processMemoryOwner: process, retaining: scope)
            XCTAssertTrue(resources.contract.supportsNativeCompletePrefix)
            XCTAssertFalse(resources.contract.supportsNativePagedTarget)
            let engine = EngineV2(
                model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 128,
                    prefillChunkSize: 128, enablePrefixCache: true),
                loopConfig: .init(stepTimeout: 30, watchdogInterval: 0.01, shutdownTimeout: 10),
                completePrefixCache: store, mtpDrafter: binding.assistant,
                mtpConfig: .init(
                    enabled: mtp, maxDraftTokens: 3, maxSpeculativeBatch: 1,
                    fixedDraftTokens: 3, verificationMode: .serialTarget),
                processMemoryOwner: process, nativeCompletionTracking: true,
                nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            XCTAssertNil(engine.nativeCompletionFault)
            XCTAssertTrue(engine.completeCheckpointCodec?.contiguousLayout != nil)
            XCTAssertEqual(engine.mtpMetricsSnapshot()?.active == true, mtp)
            if mtp {
                XCTAssertNil(engine.mtpInactiveReason)
                guard case .bounded? = engine.resolvedMTPAdmission else { throw Failure.completion }
            }
            return .init(engine: engine, contract: resources.contract)
        }
        guard case .completed(let receipt) = construction.snapshot.disposition else {
            throw Failure.completion
        }
        try await construction.sealForPublication(receipt)
        return .init(
            container: container, construction: construction, engine: built.engine,
            contract: built.contract, store: store, process: process)
    }

    func stop() async throws {
        guard !process.retired else { return }
        try await container.perform { context in
            try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                .beginNativeCompletePrefixRetirement(executionContractID: contract.id)
        }
        guard case .quiescent(let receipt) = await engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(engine)
            _ = Unmanaged.passRetained(container)
            _ = Unmanaged.passRetained(construction)
            throw Failure.completion
        }
        try await container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            try model.releaseNativeCompletePrefixAfterNativeRetirement(receipt)
            XCTAssertNil(model.nativeCompletePrefixPreparation)
            XCTAssertNil(model.resources.nativePrefixValidator)
        }
        XCTAssertTrue(process.retired)
        XCTAssertEqual(process.bytes, 0)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
    }

    func collect(_ request: CBv2Request) async throws -> CBv2SchedCollected {
        let submission = try engine.submitWithNativeRetirement(request)
        let result = await cbv2SchedCollect(submission.events)
        guard result.finishReason == .length, result.tokens.count == request.maxTokens else {
            throw Failure.completion
        }
        await submission.retirement.wait()
        store.base.finishPublicationCallbacks(engine: engine)
        return result
    }
}
