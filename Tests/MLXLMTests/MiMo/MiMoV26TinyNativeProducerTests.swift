import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Native complete-prefix and paged producers of the strict-loaded tiny
/// synthetic checkpoint. The tiny model has unequal K/V head widths (32/16),
/// which the contiguous asymmetric prefix layout requires. The test store and
/// process owner are ownership witnesses only; they do no I/O.
final class MiMoV26TinyNativeProducerTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint
    private enum Failure: Error { case noReceipt, capacity }
    private final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var charge: UInt64 = 0, materialized: UInt64 = 0
        private var closed = false
        var bytes: UInt64 { lock.withLock { charge } }
        var isRetired: Bool { lock.withLock { closed } }
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
        func retire() { lock.withLock { closed = true } }
    }
    private final class Store: CBv2NativeCompletePrefixCache, @unchecked Sendable {
        let identity = CBv2CompleteCheckpointIdentity(
            modelAggregateHash: "tiny-synthetic-only", promptContractID: "mock-tokenizer",
            buildID: "source-test", numericsFingerprint: "native")
        private let lock = NSLock()
        private var didClose = false
        var isClosed: Bool { lock.withLock { didClose } }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool { false }
        func takeStaged(
            requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
            maximumSequenceLength: Int
        ) -> CBv2StagedCompleteCheckpoint? { nil }
        func donate(
            _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
            tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
        ) {
            source.close()
            completion([])
        }
        func close() { lock.withLock { didClose = true } }
        func closeAndWait() async { close() }  // This empty store starts no I/O job.
    }

    private func loaded() async throws -> MiMoV26LoadedModel {
        let value = try await Fixture.loaded()
        addTeardownBlock { try? FileManager.default.removeItem(at: value.root) }
        return value.model
    }
    private func scope<T>(
        _ model: MiMoV26LoadedModel, _ body: (NativeConstructionScope) throws -> T
    ) throws -> T {
        try Fixture.withScope { work in
            try work.withPhase(.nativeSetup) {
                try work.authorizeImmutableLoadedOwner(model.resources)
                return try body(work)
            }
        }
    }
    private func metadata(_ model: MiMoV26LoadedModel, _ binding: MiMoV26CBv2Binding) throws
        -> MiMoV26NativeCompletePrefixMetadata
    {
        try scope(model) { work in
            _ = try binding.adapter.probeNativeKVTypes(retaining: work)
            return try model.nativeCompletePrefixMetadata(binding: binding, retaining: work)
        }
    }

    // MARK: - Complete prefix

    func testPrefixMetadataRequiresTheProbeAndTheActualLoadedOwner() async throws {
        let model = try await loaded()
        let foreign = try await loaded()
        let binding = try model.makeCBv2Binding()
        XCTAssertFalse(binding.adapter.cbv2SupportsHistoricalAttentionCheckpoint)
        XCTAssertThrowsError(
            try scope(model) {
                try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
            })
        let value = try metadata(model, binding)
        XCTAssertEqual(value.modelType, "mimo_v2")
        XCTAssertEqual(value.loadSessionID, model.loadReceipt.sessionID)
        XCTAssertEqual(value.loadBindingFingerprint, try model.loadReceipt.binding.fingerprint())
        XCTAssertEqual(value.layerKinds, binding.adapter.layerKinds)
        XCTAssertEqual(value.layerDTypes, binding.adapter.cbv2CompleteCheckpointKVDTypes)
        XCTAssertEqual(value.layerDTypes.count, 2)
        XCTAssertEqual(value.maximumContextTokens, 256)
        XCTAssertEqual(
            value.backendLayout, CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout)
        XCTAssertNil(value.assistantCodecID)
        XCTAssertEqual(value.verificationMode, .serialTarget)
        XCTAssertNotNil(model.nativeCompletePrefixPreparation)
        let otherBinding = try foreign.makeCBv2Binding()
        _ = try metadata(foreign, otherBinding)
        XCTAssertThrowsError(
            try scope(model) {
                try model.nativeCompletePrefixMetadata(binding: otherBinding, retaining: $0)
            })
        XCTAssertNotEqual(value.loadedGeneration, foreign.resources.nativePrefixLifetime.generation)
    }

    func testMTPPrefixMetadataUsesTheThreeHeadCodec() async throws {
        let model = try await loaded()
        let binding = try model.makeCBv2Binding(enableMTP: true)
        let value = try metadata(model, binding)
        let assistant = try XCTUnwrap(binding.assistant)
        XCTAssertEqual(value.assistantCodecID, assistant.prefixCheckpointCodecID)
        XCTAssertNotNil(value.assistantCodecID)
        XCTAssertEqual(
            value.backendLayout, CBv2CompleteCheckpointManifest.contiguousAsymmetricMTPLayout)
        XCTAssertTrue(binding.adapter.cbv2SupportsHistoricalAttentionCheckpoint)
        XCTAssertTrue(binding.adapter.supportsMultimodalPrefill(attention: .causal))
        // A predictor reload advances its loaded generation. The old binding
        // and a fresh one cannot hide that stale preparation.
        let predictor = model.resources.loaded.bundle.mtp
        let parameters = Dictionary(
            uniqueKeysWithValues: predictor.parameters().flattened().map { ("mtp." + $0.0, $0.1) })
        try predictor.loadConvertedWeights(parameters)
        XCTAssertFalse(binding.adapter.cbv2SupportsHistoricalAttentionCheckpoint)
        XCTAssertThrowsError(
            try scope(model) {
                try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
            })
        let replacement = try model.makeCBv2Binding(enableMTP: true)
        XCTAssertThrowsError(try metadata(model, replacement))
    }

    func testPrefixIssuanceBindsOneStoreAndProcessOwnerOnce() async throws {
        let model = try await loaded()
        let other = try await loaded()
        let binding = try model.makeCBv2Binding()
        let value = try metadata(model, binding)
        let foreignMetadata = try metadata(other, other.makeCBv2Binding())
        let store = Store()
        let process = ProcessOwner()
        XCTAssertThrowsError(
            try scope(model) {
                try model.makeNativeCompletePrefixExecutionResources(
                    binding: binding, bytesCapacity: 32 << 20,
                    expectedMetadata: foreignMetadata, completePrefixCache: store,
                    processMemoryOwner: process, retaining: $0)
            })
        let issued = try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(
                binding: binding, bytesCapacity: 32 << 20, expectedMetadata: value,
                completePrefixCache: store, processMemoryOwner: process, retaining: $0)
        }
        XCTAssertTrue(issued.contract.supportsNativeCompletePrefix)
        XCTAssertFalse(issued.contract.supportsManagedDecodedMedia)
        XCTAssertFalse(
            issued.contract.consume(
                model: binding.adapter, backend: issued.backend,
                cacheProvider: issued.cacheProvider, assistant: nil,
                completePrefixCache: Store(), processMemoryOwner: process))
        XCTAssertFalse(
            issued.contract.consume(
                model: binding.adapter, backend: issued.backend,
                cacheProvider: issued.cacheProvider, assistant: nil,
                completePrefixCache: store, processMemoryOwner: ProcessOwner()))
        XCTAssertTrue(
            issued.contract.consume(
                model: binding.adapter, backend: issued.backend,
                cacheProvider: issued.cacheProvider, assistant: nil,
                completePrefixCache: store, processMemoryOwner: process))
        XCTAssertFalse(
            issued.contract.consume(
                model: binding.adapter, backend: issued.backend,
                cacheProvider: issued.cacheProvider, assistant: nil,
                completePrefixCache: store, processMemoryOwner: process))
        XCTAssertThrowsError(
            try scope(model) {
                try model.makeNativeCompletePrefixExecutionResources(
                    binding: binding, bytesCapacity: 32 << 20, expectedMetadata: value,
                    completePrefixCache: store, processMemoryOwner: process, retaining: $0)
            })
    }

    func testDrainingKeepsTheValidatorUntilTheEngineReceipt() async throws {
        let model = try await loaded()
        XCTAssertEqual(model.nativeConfiguration.fullAttention.headDim, 32)
        XCTAssertEqual(model.nativeConfiguration.fullAttention.valueHeadDim, 16)
        let binding = try model.makeCBv2Binding()
        let value = try metadata(model, binding)
        let store = Store()
        let process = ProcessOwner()
        let issued = try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(
                binding: binding, bytesCapacity: 32 << 20, expectedMetadata: value,
                completePrefixCache: store, processMemoryOwner: process, retaining: $0)
        }
        let validator = try XCTUnwrap(model.resources.nativePrefixValidator)
        let engine = EngineV2(
            model: binding.adapter, layerKinds: binding.adapter.layerKinds,
            backend: issued.backend, cacheProvider: issued.cacheProvider,
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4, enablePrefixCache: true),
            completePrefixCache: store, processMemoryOwner: process,
            nativeCompletionTracking: true, nativeExecutionContract: issued.contract)
        XCTAssertNil(engine.nativeCompletionFault)
        try model.beginNativeCompletePrefixRetirement(executionContractID: issued.contract.id)
        XCTAssertNoThrow(try validator.validateNativeCompletePrefixBinding())
        XCTAssertThrowsError(
            try scope(model) {
                try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
            })
        XCTAssertThrowsError(
            try model.beginNativeCompletePrefixRetirement(executionContractID: UUID()))
        guard case .quiescent(let receipt) = await engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(engine)
            throw Failure.noReceipt
        }
        XCTAssertTrue(store.isClosed)
        XCTAssertTrue(process.isRetired)
        XCTAssertEqual(process.bytes, 0)
        XCTAssertNoThrow(try model.releaseNativeCompletePrefixAfterNativeRetirement(receipt))
        XCTAssertNil(model.nativeCompletePrefixPreparation)
        XCTAssertNil(model.resources.nativePrefixValidator)
        XCTAssertThrowsError(try validator.validateNativeCompletePrefixBinding())
        XCTAssertThrowsError(try model.releaseNativeCompletePrefixAfterNativeRetirement(receipt))
    }

    func testMutationAndLateRefusalInvalidateThePreparedGeneration() async throws {
        let model = try await loaded()
        let binding = try model.makeCBv2Binding()
        let value = try metadata(model, binding)
        try model.update(parameters: .unflattened([]), verify: .noUnusedKeys)
        XCTAssertThrowsError(
            try scope(model) {
                try model.makeNativeCompletePrefixExecutionResources(
                    binding: binding, bytesCapacity: 32 << 20, expectedMetadata: value,
                    completePrefixCache: Store(), processMemoryOwner: ProcessOwner(),
                    retaining: $0)
            })
        XCTAssertThrowsError(
            try scope(model) {
                try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
            })
        let late = try await loaded()
        var lateBinding: MiMoV26CBv2Binding? = try late.makeCBv2Binding()
        weak var adapter = lateBinding?.adapter
        let lateValue = try metadata(late, XCTUnwrap(lateBinding))
        XCTAssertThrowsError(
            try scope(late) {
                try late.makeNativeCompletePrefixExecutionResources(
                    binding: XCTUnwrap(lateBinding), bytesCapacity: -1,
                    expectedMetadata: lateValue, completePrefixCache: Store(),
                    processMemoryOwner: ProcessOwner(), retaining: $0)
            })
        lateBinding = nil
        XCTAssertNotNil(adapter, "the loaded wrapper keeps the failed preparation")
        XCTAssertNotNil(late.nativeCompletePrefixPreparation)
        let validator = try XCTUnwrap(late.resources.nativePrefixValidator)
        XCTAssertThrowsError(try validator.validateNativeCompletePrefixBinding())
        XCTAssertThrowsError(try metadata(late, late.makeCBv2Binding()))
    }

    // MARK: - Paged

    private struct Container: Sendable {
        let container: ModelContainer
        let construction: NativeConstructionWork
    }
    private func container() async throws -> Container {
        let root = try Fixture.writeNativeBundle(to: Fixture.temporaryRoot("paged"))
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let session = try MiMoV26SerialLoadSession(plan: Fixture.preflight(root))
        let prepared = try await MiMoV26ModelFactory.prepare(
            request: session.request, configuration: .init(directory: root),
            tokenizerLoader: Fixture.Loader(tokenizer: .init()))
        let construction = NativeConstructionWork()
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: session, reservation: Fixture.Permit(session.request), prepared: prepared,
            retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        return .init(container: container, construction: construction)
    }

    func testPagedProducerRefusesMTPInvalidatedGenerationAndBadBounds() async throws {
        let f = try await container()
        try await MiMoV26ModelFactory.withNativeConstruction(
            container: f.container, retaining: f.construction
        ) { model, scope in
            let mtp = try model.makeCBv2Binding(enableMTP: true)
            _ = try mtp.adapter.probeNativeKVTypes(retaining: scope)
            XCTAssertThrowsError(
                try model.makeNativePagedExecutionResources(
                    binding: mtp, bytesCapacity: 64 << 20, maximumConcurrentRequests: 1,
                    maximumQueryTokens: 16, maximumPrefillChunk: 16,
                    processMemoryOwner: ProcessOwner(), retaining: scope)
            ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner) }
            XCTAssertNil(model.nativePagedPreparation)
            let text = try model.makeCBv2Binding()
            XCTAssertThrowsError(
                try model.makeNativePagedSerialMTPExecutionResources(
                    binding: text, bytesCapacity: 64 << 20, maximumConcurrentRequests: 1,
                    maximumQueryTokens: 16, maximumPrefillChunk: 16,
                    processMemoryOwner: ProcessOwner(), retaining: scope)
            ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner) }
            // The KV type probe is required before any paged configuration.
            XCTAssertThrowsError(
                try model.nativePagedCompletePrefixMetadata(
                    binding: text, bytesCapacity: 64 << 20, maximumConcurrentRequests: 1,
                    maximumQueryTokens: 16, maximumPrefillChunk: 16, retaining: scope))
            _ = try text.adapter.probeNativeKVTypes(retaining: scope)
            for (capacity, requests, query, chunk) in [
                (0, 1, 16, 16), (64 << 20, 0, 16, 16), (64 << 20, 1, 8, 16),
                (64 << 20, 1, 512, 16), (64 << 20, 1, 16, 0),
            ] {
                XCTAssertThrowsError(
                    try model.nativePagedCompletePrefixMetadata(
                        binding: text, bytesCapacity: capacity,
                        maximumConcurrentRequests: requests, maximumQueryTokens: query,
                        maximumPrefillChunk: chunk, retaining: scope)
                ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner) }
            }
            XCTAssertNil(model.nativePagedPreparation)
            model.invalidateMultimodalPreparation()
            XCTAssertThrowsError(
                try model.makeNativePagedExecutionResources(
                    binding: text, bytesCapacity: 64 << 20, maximumConcurrentRequests: 1,
                    maximumQueryTokens: 16, maximumPrefillChunk: 16,
                    processMemoryOwner: ProcessOwner(), retaining: scope))
            XCTAssertNil(model.nativePagedPreparation)
        }
    }

    func testPagedPrefixMetadataDescribesTheProbedLoadedOwner() async throws {
        let f = try await container()
        try await MiMoV26ModelFactory.withNativeConstruction(
            container: f.container, retaining: f.construction
        ) { model, scope in
            let binding = try model.makeCBv2Binding()
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let value = try model.nativePagedCompletePrefixMetadata(
                binding: binding, bytesCapacity: 64 << 20, maximumConcurrentRequests: 2,
                maximumQueryTokens: 32, maximumPrefillChunk: 16, retaining: scope)
            XCTAssertEqual(value.prefix.modelType, "mimo_v2")
            XCTAssertEqual(value.prefix.loadSessionID, model.loadReceipt.sessionID)
            XCTAssertEqual(
                value.prefix.loadedGeneration, model.resources.nativePagedLifetime.generation)
            XCTAssertEqual(
                value.prefix.backendLayout, CBv2CompleteCheckpointManifest.pagedAsymmetricLayout)
            XCTAssertNil(value.prefix.assistantCodecID)
            XCTAssertEqual(value.prefix.maximumContextTokens, 256)
            XCTAssertEqual(value.pagedConfiguration.capacityBytes, 64 << 20)
            XCTAssertEqual(value.pagedConfiguration.maxPrefillChunk, 16)
            XCTAssertEqual(value.pagedConfiguration.nominalMaxSequenceLength, 256)
            XCTAssertEqual(value.pagedConfiguration.layerDTypes, value.prefix.layerDTypes)
            XCTAssertNil(value.pagedConfiguration.prefixSharingBlockSize)
            XCTAssertNotNil(model.nativePagedPreparation)
            // An issued complete-prefix preparation excludes the paged producer.
            XCTAssertNoThrow(
                try model.nativeCompletePrefixMetadata(binding: binding, retaining: scope))
            XCTAssertThrowsError(
                try model.nativePagedCompletePrefixMetadata(
                    binding: binding, bytesCapacity: 64 << 20, maximumConcurrentRequests: 2,
                    maximumQueryTokens: 32, maximumPrefillChunk: 16, retaining: scope))
        }
    }
}
