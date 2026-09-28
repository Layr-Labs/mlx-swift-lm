// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Prepared / UNRUN. Reuses the existing strict tiny-bf16 serial-load fixture.
/// Mock tokenizer routing and test ledger/store ownership are not provider
/// encryption, full-artifact, memory-peak or whole-engine prefix-hit evidence.
final class MiMoV26NativePrefixProducerTests: XCTestCase {
    private enum Failure: Error { case noReceipt, retired, capacity }
    private final class Permit: MiMoV26SerialLoadReservation {
        let request: MiMoV26SerialLoadRequest
        var reservedLoadBytes: UInt64 { request.requiredLoadBytes }
        init(_ request: MiMoV26SerialLoadRequest) { self.request = request }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private final class TokenizerFixture: Tokenizer, @unchecked Sendable {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [6] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map(String.init).joined(separator: " ") }
        func convertTokenToId(_ token: String) -> Int? {
            ["<|im_start|>": 1, "<|im_end|>": 3, "<think>": 4, "</think>": 5][token]
        }
        func convertIdToToken(_ id: Int) -> String? { id == 3 ? "<|im_end|>" : nil }
        var bosToken: String? { nil }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(messages: [Message], tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, tools: nil, additionalContext: nil)
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String, tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] {
            [1, 2] + ((additionalContext?["enable_thinking"] as? Bool) == false ? [4, 5] : []) + [6]
        }
    }
    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { TokenizerFixture() }
    }
    private final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var charge: UInt64 = 0, materialized: UInt64 = 0
        private var closed = false
        var bytes: UInt64 { lock.withLock { charge } }
        var isRetired: Bool { lock.withLock { closed } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= 256 << 20 else { throw Failure.capacity }
                charge = bytes
            }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= materialized, bytes <= charge else { throw Failure.capacity }
                materialized = bytes
            }
        }
        func withdrawCoverage(_ bytes: UInt64) throws {
            try lock.withLock {
                guard bytes <= materialized else { throw Failure.capacity }
                materialized -= bytes
            }
        }
        func retire() { lock.withLock { XCTAssertEqual(charge, 0); closed = true } }
    }
    private final class Store: CBv2NativeCompletePrefixCache, @unchecked Sendable {
        let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: "strict-tiny-fixture-only",
            promptContractID: "mock-tokenizer", buildID: "source-test", numericsFingerprint: "native")
        private let lock = NSLock()
        private var didClose = false
        var isClosed: Bool { lock.withLock { didClose } }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool { false }
        func takeStaged(requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
                        maximumSequenceLength: Int) -> CBv2StagedCompleteCheckpoint? { nil }
        func donate(_ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
                    tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void) {
            source.close(); completion([])
        }
        func close() { lock.withLock { didClose = true } }
        func closeAndWait() async { close() } // This empty fixture starts no I/O job.
    }

    private func scope<T>(_ model: MiMoV26LoadedModel,
                          _ body: (NativeConstructionScope) throws -> T) throws -> T {
        let work = NativeConstructionScope()
        defer { if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) } }
        return try work.withPhase(.nativeSetup) {
            try work.authorizeImmutableLoadedOwner(model.resources)
            return try body(work)
        }
    }
    private func loaded(asymmetric: Bool = false) async throws -> MiMoV26LoadedModel {
        guard ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1",
              let path = ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_LOAD_FIXTURES"] else {
            throw XCTSkip("Requires existing strict tiny-bf16 fixture and coordinator-owned native lane")
        }
        let fixtures = URL(fileURLWithPath: path)
        let root = fixtures.appendingPathComponent("native-prefix-producer-" + UUID().uuidString)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent(
            asymmetric ? "tiny-asymmetric-bf16" : "tiny-bf16"), to: root)
        // Same mock tokenizer sidecars as MiMoV26FactoryTests; native payload
        // bytes and their converted provenance are copied without modification.
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["chat_template": "embedded alternate", "eos_token": "<|im_end|>"])
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data("{{ messages }}{% if enable_thinking is false %}<think></think>{% endif %}\n".utf8)
            .write(to: root.appendingPathComponent("chat_template.jinja"))
        try JSONSerialization.data(withJSONObject: ["eos_token_id": [4, 5]])
            .write(to: root.appendingPathComponent("generation_config.json"))
        let p = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: fixtures.appendingPathComponent(
                asymmetric ? "provenance-asymmetric.json" : "provenance.json"))) as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]), sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304))
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let work = NativeConstructionScope()
        defer { if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) } }
        let context = try MiMoV26ModelFactory.load(session: session, reservation: Permit(session.request),
            prepared: prepared, retaining: work)
        return try XCTUnwrap(context.model as? MiMoV26LoadedModel)
    }
    private func metadata(_ model: MiMoV26LoadedModel, _ binding: MiMoV26CBv2Binding) throws
        -> MiMoV26NativeCompletePrefixMetadata {
        try scope(model) { work in
            _ = try binding.adapter.probeNativeKVTypes(retaining: work)
            return try model.nativeCompletePrefixMetadata(binding: binding, retaining: work)
        }
    }

    func testMetadataRequiresObservedProbeAndActualStrictLoadedOwner() async throws {
        let model = try await loaded(), foreign = try await loaded()
        let binding = try model.makeCBv2Binding()
        XCTAssertFalse(binding.adapter.cbv2SupportsHistoricalAttentionCheckpoint)
        XCTAssertThrowsError(try scope(model) {
            try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
        })
        let value = try metadata(model, binding)
        XCTAssertEqual(value.modelType, "mimo_v2")
        XCTAssertEqual(value.loadSessionID, model.loadReceipt.sessionID)
        XCTAssertEqual(value.loadBindingFingerprint, try model.loadReceipt.binding.fingerprint())
        XCTAssertEqual(value.layerKinds, binding.adapter.layerKinds)
        XCTAssertEqual(value.layerDTypes, binding.adapter.cbv2CompleteCheckpointKVDTypes)
        XCTAssertFalse(value.layerDTypes.isEmpty)
        XCTAssertEqual(value.maximumContextTokens, model.nativeConfiguration.maxPositionEmbeddings)
        XCTAssertEqual(value.backendLayout, CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout)
        XCTAssertNil(value.assistantCodecID)
        XCTAssertFalse(binding.adapter.cbv2Capabilities.supportsPagedKV)
        let otherBinding = try foreign.makeCBv2Binding()
        _ = try metadata(foreign, otherBinding)
        XCTAssertThrowsError(try scope(model) {
            try model.nativeCompletePrefixMetadata(binding: otherBinding, retaining: $0)
        })
        XCTAssertNotEqual(value.loadedGeneration, foreign.resources.nativePrefixLifetime.generation)
    }

    func testMTPMetadataUsesGenuineThreeHeadCodecAndStaleHeadGenerationRefuses() async throws {
        let model = try await loaded()
        let binding = try model.makeCBv2Binding(enableMTP: true)
        let value = try metadata(model, binding)
        let assistant = try XCTUnwrap(binding.assistant)
        XCTAssertEqual(assistant.maximumDraftTokens, 3)
        XCTAssertEqual(value.assistantCodecID, assistant.prefixCheckpointCodecID)
        XCTAssertNotNil(value.assistantCodecID)
        XCTAssertEqual(value.backendLayout, CBv2CompleteCheckpointManifest.contiguousAsymmetricMTPLayout)
        XCTAssertEqual(value.verificationMode, .serialTarget)
        XCTAssertTrue(binding.adapter.cbv2SupportsHistoricalAttentionCheckpoint)
        XCTAssertTrue(binding.adapter.supportsMultimodalPrefill(attention: .causal))
        // Actual quiescent predictor reload advances its real loaded generation;
        // a fresh enum or metadata record cannot rehabilitate this old binding.
        let predictor = model.resources.loaded.bundle.mtp
        let parameters = Dictionary(uniqueKeysWithValues: predictor.parameters().flattened().map {
            ("mtp." + $0.0, $0.1)
        })
        try predictor.loadConvertedWeights(parameters)
        XCTAssertFalse(binding.adapter.cbv2SupportsHistoricalAttentionCheckpoint)
        XCTAssertThrowsError(try scope(model) {
            try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
        })
        let replacement = try model.makeCBv2Binding(enableMTP: true)
        XCTAssertThrowsError(try metadata(model, replacement),
            "a fresh assistant must not hide the stale preparation's real head generation")
    }

    func testExactStoreAndProcessTupleIsOneUseAndMetadataMismatchDoesNotIssue() async throws {
        let model = try await loaded(), other = try await loaded()
        let binding = try model.makeCBv2Binding(), otherBinding = try other.makeCBv2Binding()
        let value = try metadata(model, binding), foreignMetadata = try metadata(other, otherBinding)
        let store = Store(), process = ProcessOwner(), foreignStore = Store(), foreignProcess = ProcessOwner()
        XCTAssertThrowsError(try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(binding: binding, bytesCapacity: 32 << 20,
                expectedMetadata: foreignMetadata, completePrefixCache: store, processMemoryOwner: process, retaining: $0)
        })
        let issued = try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(binding: binding, bytesCapacity: 32 << 20,
                expectedMetadata: value, completePrefixCache: store, processMemoryOwner: process, retaining: $0)
        }
        XCTAssertTrue(issued.contract.supportsNativeCompletePrefix)
        XCTAssertFalse(issued.contract.supportsManagedDecodedMedia)
        XCTAssertFalse(issued.contract.consume(model: binding.adapter, backend: issued.backend,
            cacheProvider: issued.cacheProvider, assistant: nil,
            completePrefixCache: foreignStore, processMemoryOwner: process))
        XCTAssertFalse(issued.contract.consume(model: binding.adapter, backend: issued.backend,
            cacheProvider: issued.cacheProvider, assistant: nil,
            completePrefixCache: store, processMemoryOwner: foreignProcess))
        XCTAssertTrue(issued.contract.consume(model: binding.adapter, backend: issued.backend,
            cacheProvider: issued.cacheProvider, assistant: nil,
            completePrefixCache: store, processMemoryOwner: process))
        XCTAssertFalse(issued.contract.consume(model: binding.adapter, backend: issued.backend,
            cacheProvider: issued.cacheProvider, assistant: nil,
            completePrefixCache: store, processMemoryOwner: process))
        XCTAssertThrowsError(try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(binding: binding, bytesCapacity: 32 << 20,
                expectedMetadata: value, completePrefixCache: store, processMemoryOwner: process, retaining: $0)
        })
        // Component ticket consumption only: this is NOT an engine completion
        // receipt, and the test does not call a success-shaped retirement API.
    }

    func testDrainingKeepsActualValidatorUntilAuthenticEngineReceipt() async throws {
        // Keep the original equal-width refusal as a real codec-geometry
        // control; never issue a success-shaped retirement for that layout.
        let equal = try await loaded(), equalBinding = try equal.makeCBv2Binding()
        let equalMetadata = try metadata(equal, equalBinding)
        XCTAssertTrue(equalMetadata.layerKinds.allSatisfy { $0.headDim == $0.valueHeadDim })
        XCTAssertThrowsError(try CBv2CheckpointAttentionLayer.resolveContiguousAsymmetric(
            layerKinds: equalMetadata.layerKinds, dtypes: equalMetadata.layerDTypes))
        let model = try await loaded(asymmetric: true)
        XCTAssertEqual(model.nativeConfiguration.fullAttention.headDim, 64)
        XCTAssertEqual(model.nativeConfiguration.fullAttention.valueHeadDim, 32)
        let binding = try model.makeCBv2Binding(), value = try metadata(model, binding)
        let store = Store(), process = ProcessOwner()
        let issued = try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(binding: binding, bytesCapacity: 32 << 20,
                expectedMetadata: value, completePrefixCache: store, processMemoryOwner: process, retaining: $0)
        }
        let validator = try XCTUnwrap(model.resources.nativePrefixValidator)
        let engine = EngineV2(model: binding.adapter, layerKinds: binding.adapter.layerKinds,
            backend: issued.backend, cacheProvider: issued.cacheProvider,
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 4,
                                   prefillChunkSize: 4, enablePrefixCache: true),
            completePrefixCache: store, processMemoryOwner: process,
            nativeCompletionTracking: true, nativeExecutionContract: issued.contract)
        XCTAssertNil(engine.nativeCompletionFault)
        try model.beginNativeCompletePrefixRetirement(executionContractID: issued.contract.id)
        XCTAssertNoThrow(try validator.validateNativeCompletePrefixBinding())
        XCTAssertThrowsError(try scope(model) {
            try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
        })
        guard case .quiescent(let receipt) = await engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(engine)
            throw Failure.noReceipt
        }
        XCTAssertTrue(store.isClosed); XCTAssertTrue(process.isRetired)
        XCTAssertEqual(process.bytes, 0)
        XCTAssertNoThrow(try model.releaseNativeCompletePrefixAfterNativeRetirement(receipt))
        XCTAssertNil(model.nativeCompletePrefixPreparation)
        XCTAssertNil(model.resources.nativePrefixValidator)
        XCTAssertThrowsError(try validator.validateNativeCompletePrefixBinding())
        XCTAssertThrowsError(try model.releaseNativeCompletePrefixAfterNativeRetirement(receipt))
    }

    func testSupportedWrapperMutationInvalidatesPreparedLoadedGeneration() async throws {
        let model = try await loaded()
        let binding = try model.makeCBv2Binding(), value = try metadata(model, binding)
        try model.update(parameters: .unflattened([]), verify: .noUnusedKeys)
        XCTAssertThrowsError(try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(binding: binding, bytesCapacity: 32 << 20,
                expectedMetadata: value, completePrefixCache: Store(), processMemoryOwner: ProcessOwner(), retaining: $0)
        })
        XCTAssertThrowsError(try scope(model) {
            try model.nativeCompletePrefixMetadata(binding: binding, retaining: $0)
        })
    }

    func testLateSetupRefusalRetainsPreparationAndCannotBeReissued() async throws {
        let model = try await loaded()
        var binding: MiMoV26CBv2Binding? = try model.makeCBv2Binding()
        weak var adapter = binding?.adapter
        let value = try metadata(model, XCTUnwrap(binding))
        XCTAssertThrowsError(try scope(model) {
            try model.makeNativeCompletePrefixExecutionResources(binding: XCTUnwrap(binding), bytesCapacity: -1,
                expectedMetadata: value, completePrefixCache: Store(), processMemoryOwner: ProcessOwner(), retaining: $0)
        })
        binding = nil
        XCTAssertNotNil(adapter, "real loaded wrapper retains the failed/late preparation")
        XCTAssertNotNil(model.nativeCompletePrefixPreparation)
        let validator = try XCTUnwrap(model.resources.nativePrefixValidator)
        XCTAssertThrowsError(try validator.validateNativeCompletePrefixBinding())
        let fresh = try model.makeCBv2Binding()
        XCTAssertThrowsError(try metadata(model, fresh))
        // No fabricated completion or cleanup call. The enclosing strict-loaded
        // owner and actual construction scope retain their ordinary obligations.
    }
}
