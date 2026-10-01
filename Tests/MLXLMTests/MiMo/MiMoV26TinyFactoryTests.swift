import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Two-stage factory and loaded wrapper over the tiny synthetic checkpoint.
/// The mock tokenizer routes tokens only; it is not a tokenizer or Jinja oracle.
final class MiMoV26TinyFactoryTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint
    private final class ShortPermit: MiMoV26SerialLoadReservation {
        let request: MiMoV26SerialLoadRequest
        var reservedLoadBytes: UInt64 { request.requiredLoadBytes - 1 }
        init(_ request: MiMoV26SerialLoadRequest) { self.request = request }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }

    private func bundle() throws -> URL {
        let root = try Fixture.writeNativeBundle(to: Fixture.temporaryRoot("factory"))
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func session(_ root: URL) throws -> MiMoV26SerialLoadSession {
        try MiMoV26SerialLoadSession(plan: Fixture.preflight(root))
    }
    private func prepare(
        _ root: URL, _ session: MiMoV26SerialLoadSession,
        tokenizer: Fixture.Tokenizer = .init(),
        configure: (inout ResolvedModelConfiguration) -> Void = { _ in }
    ) async throws -> MiMoV26ModelFactory.Prepared {
        var configuration = ResolvedModelConfiguration(directory: root)
        configure(&configuration)
        return try await MiMoV26ModelFactory.prepare(
            request: session.request, configuration: configuration,
            tokenizerLoader: Fixture.Loader(tokenizer: tokenizer))
    }
    /// Returns the factory error of one preparation, or nil when it succeeds.
    private func prepareError(
        _ root: URL, tokenizer: Fixture.Tokenizer = .init(),
        configure: (inout ResolvedModelConfiguration) -> Void = { _ in }
    ) async throws -> MiMoV26FactoryError? {
        let session = try self.session(root)
        do {
            _ = try await prepare(root, session, tokenizer: tokenizer, configure: configure)
            return nil
        } catch let error as MiMoV26FactoryError {
            return error
        }
    }

    // MARK: - Preparation

    func testPreparationBindsRequestTemplateStopTokensAndThinkingContract() async throws {
        let root = try bundle()
        let session = try self.session(root)
        let tokenizer = Fixture.Tokenizer()
        let prepared = try await prepare(root, session, tokenizer: tokenizer) {
            $0.extraEOSTokens = ["<|extra_eos|>"]
        }
        XCTAssertEqual(prepared.request, session.request)
        XCTAssertEqual(prepared.processor.chatTemplate, Fixture.template)
        XCTAssertEqual(prepared.processor.vocabularySize, 128)
        XCTAssertEqual(prepared.processor.maximumSequenceLength, 256)
        XCTAssertEqual(prepared.templateSHA256.count, 64)
        // Checkpoint EOS 1, tokenizer EOS spelling 1, explicit extra EOS 15.
        XCTAssertEqual(prepared.stopTokenIDs, [1, 15])
        XCTAssertEqual(prepared.nativeConfiguration.hiddenSize, 64)
        let contexts = tokenizer.contexts
        XCTAssertEqual(contexts.count, 3)
        XCTAssertNil(contexts[0])
        XCTAssertEqual(contexts[1]?["enable_thinking"] as? Bool, true)
        XCTAssertEqual(contexts[2]?["enable_thinking"] as? Bool, false)
        XCTAssertEqual(
            try prepared.processor.renderTokens(input: UserInput(prompt: "a")),
            [11, 20 + 97 % 32, 12])
    }

    func testPreparationUsesGenerationEOSAndTheEmbeddedTemplateFallback() async throws {
        let root = try bundle()
        try Fixture.data(["eos_token_id": [4, 5]])
            .write(to: root.appendingPathComponent("generation_config.json"))
        try FileManager.default.removeItem(at: root.appendingPathComponent("chat_template.jinja"))
        let session = try self.session(root)
        let prepared = try await prepare(root, session)
        XCTAssertEqual(prepared.processor.chatTemplate, "embedded alternate")
        XCTAssertEqual(prepared.stopTokenIDs, [1, 4, 5])
        let context = try Fixture.withScope { work in
            try MiMoV26ModelFactory.load(
                session: session, reservation: Fixture.Permit(session.request),
                prepared: prepared, retaining: work)
        }
        XCTAssertEqual(context.configuration.eosTokenIds, [4, 5])
        XCTAssertTrue(context.model is MiMoV26LoadedModel)
        XCTAssertTrue(context.processor is MiMoV26TextProcessor)
    }

    func testPreparationRefusesEveryUnboundOrInconsistentAsset() async throws {
        let other = try Fixture.temporaryRoot("factory-other")
        addTeardownBlock { try? FileManager.default.removeItem(at: other) }
        var error = try await prepareError(try bundle()) { $0.tokenizerDirectory = other }
        XCTAssertEqual(
            error,
            .invalidMetadata("model and tokenizer must use the exact bound local checkpoint root"))
        error = try await prepareError(try bundle()) { $0.eosTokenIds = [2] }
        XCTAssertEqual(
            error,
            .invalidMetadata(
                "resolved EOS IDs conflict with checkpoint EOS; use extraEOSTokens for explicit additions"
            ))
        error = try await prepareError(try bundle()) { $0.extraEOSTokens = ["<|missing|>"] }
        XCTAssertEqual(error, .invalidMetadata("unresolved explicit EOS token"))
        error = try await prepareError(try bundle(), tokenizer: .init(eosToken: "<|unknown|>"))
        XCTAssertEqual(error, .invalidMetadata("tokenizer EOS spelling has no token ID"))
        error = try await prepareError(try bundle(), tokenizer: .init(ignoresThinking: true))
        XCTAssertEqual(
            error,
            .invalidMetadata(
                "tokenizer/template does not preserve MiMo's default-on Boolean thinking contract"))

        var root = try bundle()
        try FileManager.default.removeItem(at: root.appendingPathComponent("tokenizer.json"))
        error = try await prepareError(root)
        XCTAssertEqual(error, .invalidMetadata("native checkpoint tokenizer files are missing"))

        root = try bundle()
        try FileManager.default.removeItem(at: root.appendingPathComponent("chat_template.jinja"))
        try Fixture.data(["eos_token": "<|im_end|>"])
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        error = try await prepareError(root)
        XCTAssertEqual(
            error, .invalidMetadata("checkpoint has no unambiguous native chat template"))

        root = try bundle()
        try Data(" \n".utf8).write(to: root.appendingPathComponent("chat_template.jinja"))
        error = try await prepareError(root)
        XCTAssertEqual(error, .invalidMetadata("checkpoint chat template is empty"))

        root = try bundle()
        try Data("[]".utf8).write(to: root.appendingPathComponent("tokenizer_config.json"))
        error = try await prepareError(root)
        XCTAssertEqual(error, .invalidMetadata("tokenizer_config.json requires an object"))

        root = try bundle()
        try Fixture.data(["eos_token_id": [500]])
            .write(to: root.appendingPathComponent("generation_config.json"))
        error = try await prepareError(root)
        XCTAssertEqual(error, .invalidMetadata("checkpoint EOS outside native vocabulary"))

        root = try bundle()
        let tokenizerFile = root.appendingPathComponent("tokenizer.json")
        let moved = root.appendingPathComponent("tokenizer.source")
        try FileManager.default.moveItem(at: tokenizerFile, to: moved)
        try FileManager.default.createSymbolicLink(at: tokenizerFile, withDestinationURL: moved)
        error = try await prepareError(root)
        XCTAssertEqual(error, .invalidMetadata("unbound tokenizer asset: tokenizer.json"))

        // The bound configuration digest is checked before any decode.
        root = try bundle()
        let session = try self.session(root)
        let config = root.appendingPathComponent("config.json")
        var data = try Data(contentsOf: config)
        data.append(Data("\n".utf8))
        try data.write(to: config)
        do {
            _ = try await prepare(root, session)
            XCTFail("a changed configuration was accepted")
        } catch {
            XCTAssertEqual(error as? MiMoV26FactoryError, .incompatiblePreparation)
        }
    }

    // MARK: - Load

    func testLoadRefusesForeignShortCancelledAndChangedBeforeNativeWork() async throws {
        let root = try bundle()
        let session = try self.session(root)
        let prepared = try await prepare(root, session)
        let other = try self.session(root)
        XCTAssertThrowsError(
            try Fixture.withScope {
                try MiMoV26ModelFactory.load(
                    session: other, reservation: Fixture.Permit(other.request),
                    prepared: prepared, retaining: $0)
            }
        ) { XCTAssertEqual($0 as? MiMoV26FactoryError, .incompatiblePreparation) }
        XCTAssertThrowsError(
            try Fixture.withScope {
                try MiMoV26ModelFactory.load(
                    session: session, reservation: ShortPermit(session.request),
                    prepared: prepared, retaining: $0)
            }
        ) { XCTAssertEqual($0 as? MiMoV26SerialLoadError, .insufficientReservation) }
        XCTAssertThrowsError(
            try Fixture.withScope {
                try MiMoV26ModelFactory.load(
                    session: session, reservation: Fixture.Permit(session.request),
                    prepared: prepared, retaining: $0, isCancelled: { true })
            }
        ) { XCTAssertEqual($0 as? MiMoV26SerialLoadError, .cancelled) }
        try Data("changed template".utf8)
            .write(to: root.appendingPathComponent("chat_template.jinja"))
        XCTAssertThrowsError(
            try Fixture.withScope {
                try MiMoV26ModelFactory.load(
                    session: session, reservation: Fixture.Permit(session.request),
                    prepared: prepared, retaining: $0)
            }
        ) { XCTAssertEqual($0 as? MiMoV26FactoryError, .changedTokenizerAssets) }
    }

    func testContainerLoadAdoptsTheWrapperAndAllowsNativeConstructionAccess() async throws {
        let root = try bundle()
        let session = try self.session(root)
        let prepared = try await prepare(root, session)
        let construction = NativeConstructionWork()
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: session, reservation: Fixture.Permit(session.request), prepared: prepared,
            retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        let stops = try await MiMoV26ModelFactory.withNativeConstruction(
            container: container, retaining: construction
        ) { model, _ in model.stopTokenIDs }
        XCTAssertEqual(stops, [1])
        guard case .completed(let receipt) = construction.snapshot.disposition else {
            return XCTFail("the construction did not complete")
        }
        try construction.validate(receipt)
        // No audio sidecar was installed, so the release is a no-op.
        try await container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            try model.releaseInstalledAudioAfterConstructionCompletion(receipt)
            XCTAssertThrowsError(try model.validateGenericGeneration())
        }
    }

    // MARK: - Loaded wrapper

    func testLoadedWrapperRefusesGenericGenerationAndForwardsText() async throws {
        let loaded = try await Fixture.loaded()
        addTeardownBlock { try? FileManager.default.removeItem(at: loaded.root) }
        let model = loaded.model
        XCTAssertEqual(model.nativeConfiguration.hiddenSize, 64)
        XCTAssertEqual(model.loadReceipt.sourceTensorCount, 207)
        XCTAssertEqual(model.loadReceipt.parameterCount, 207)
        XCTAssertEqual(model.stopTokenIDs, [1])
        XCTAssertEqual(model.templateSHA256.count, 64)
        XCTAssertThrowsError(try model.validateGenericGeneration()) {
            XCTAssertEqual($0 as? MiMoV26FactoryError, .nativeCBv2Required)
        }
        XCTAssertThrowsError(
            try model.prepare(
                LMInput(tokens: MLXArray([Int32(1), 2])), cache: [], windowSize: nil)
        ) { XCTAssertEqual($0 as? MiMoV26FactoryError, .nativeCBv2Required) }
        let caches = model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 2)
        XCTAssertTrue(caches[0] is KVCacheSimple)
        XCTAssertTrue(caches[1] is RotatingKVCache)
        let ids = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let output = try model.forwardText(inputIDs: ids)
        let direct = model(ids, cache: nil)
        eval(output.logits, direct)
        XCTAssertEqual(output.logits.shape, [1, 3, 128])
        XCTAssertEqual(direct.shape, [1, 3, 128])
        XCTAssertTrue(output.logits.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        // Two runs of the same BF16 graph; 1e-3 allows for BF16 rounding only.
        XCTAssertLessThanOrEqual(
            abs(output.logits.asType(.float32) - direct.asType(.float32)).max().item(Float.self),
            1e-3)
        let cache = model.newCache(parameters: nil)
        let prefill = try model.forwardText(inputIDs: ids, cache: cache)
        let step = try model.forwardText(
            inputIDs: MLXArray([Int32(4)]).reshaped(1, 1), cache: cache)
        eval(prefill.logits, step.logits)
        XCTAssertEqual(prefill.firstPosition, 0)
        XCTAssertEqual(step.firstPosition, 3)
        XCTAssertEqual(step.logits.shape, [1, 1, 128])
    }

    func testBindingsAndMediaProcessorBelongToTheLoadedOwner() async throws {
        let loaded = try await Fixture.loaded()
        let foreign = try await Fixture.loaded()
        addTeardownBlock {
            try? FileManager.default.removeItem(at: loaded.root)
            try? FileManager.default.removeItem(at: foreign.root)
        }
        let model = loaded.model
        let binding = try model.makeCBv2Binding()
        XCTAssertNil(binding.assistant)
        XCTAssertEqual(binding.stopTokenIDs, [1])
        let mtp = try model.makeCBv2Binding(enableMTP: true)
        let assistant = try XCTUnwrap(mtp.assistant)
        XCTAssertEqual(assistant.maximumDraftTokens, 3)
        XCTAssertEqual(assistant.verificationMode, .serialTarget)
        XCTAssertThrowsError(try model.makeCBv2Binding(verificationMode: .rectangular))
        XCTAssertThrowsError(
            try model.makeMultimodalProcessor(
                binding: foreign.model.makeCBv2Binding(), limits: MiMoMediaFixture.limits)
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner) }
        let processor = try model.makeMultimodalProcessor(
            binding: binding, limits: MiMoMediaFixture.limits)
        let plan = try processor.plan(MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        // A 64 x 64 image resizes to 12 x 12: 6 x 6 patches, 9 merged tokens.
        XCTAssertEqual(plan.spans.map(\.length), [9])
        XCTAssertEqual(plan.promptTokens, [11, 4] + Array(repeating: 2, count: 9) + [5, 12])
        XCTAssertEqual(plan.decodedElements, 3 * 64 * 64)
        XCTAssertEqual(plan.patchElements, 36 * 24)
        XCTAssertEqual(plan.featureElements, 9 * 64)
        XCTAssertEqual(plan.logicalFeatureBytes, 9 * 64 * 2)
        XCTAssertEqual(plan.templateSHA256, model.templateSHA256)
        let geometry = try XCTUnwrap(plan.visionGeometryByMediaIndex[0])
        let expected =
            try MiMoV26Pixels.workingByteCount(
                inputElements: 3 * 64 * 64, frameCount: 1, plan: geometry)
            + 3 * 64 * 64 * 4 + 36 * 24 * 8 + 9 * 64 * 16
            + MiMoV26VisionWorkingSet.frameBytes(
                geometry, configuration: XCTUnwrap(model.nativeConfiguration.vision))
        XCTAssertEqual(try processor.managedVisualCommitmentBytes(plan), expected)
        XCTAssertThrowsError(try processor.managedAudioCommitmentBytes(plan)) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .missingAudioCodec)
        }
        let prepared = try processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        XCTAssertThrowsError(try prepared.makeRequest(binding: mtp, id: .init(1)))
        let request = try prepared.makeRequest(binding: binding, id: .init(2))
        XCTAssertEqual(request.promptTokens, plan.promptTokens)
        let features = try XCTUnwrap(request.multimodal).embeddings()
        XCTAssertEqual(features.count, 1)
        eval(features)
        XCTAssertEqual(features[0].shape, [9, 64])
        XCTAssertEqual(features[0].dtype, .bfloat16)
        XCTAssertTrue(features[0].asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        model.invalidateMultimodalPreparation()
        XCTAssertThrowsError(
            try processor.plan(MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .invalidatedOwner) }
        XCTAssertThrowsError(
            try model.makeMultimodalProcessor(binding: binding, limits: MiMoMediaFixture.limits))
    }

    func testManagedProfilesIssueOnceAndRequireAnInstalledAudioOwner() async throws {
        let loaded = try await Fixture.loaded()
        addTeardownBlock { try? FileManager.default.removeItem(at: loaded.root) }
        let model = loaded.model
        let binding = try model.makeCBv2Binding()
        func scope<T>(_ body: (NativeConstructionScope) throws -> T) throws -> T {
            try Fixture.withScope { work in
                try work.withPhase(.nativeSetup) {
                    try work.authorizeImmutableLoadedOwner(model.resources)
                    return try body(work)
                }
            }
        }
        XCTAssertThrowsError(
            try scope {
                try model.makeManagedAudioExecutionResources(
                    binding: binding, bytesCapacity: 32 << 20, limits: MiMoMediaFixture.limits,
                    retaining: $0)
            }
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .missingAudioCodec) }
        XCTAssertFalse(model.hasIssuedManagedMediaProfile)
        let issued = try scope { work in
            _ = try binding.adapter.probeNativeKVTypes(retaining: work)
            return try model.makeManagedMediaExecutionResources(
                binding: binding, bytesCapacity: 32 << 20, limits: MiMoMediaFixture.limits,
                retaining: work)
        }
        XCTAssertTrue(issued.contract.supportsManagedDecodedMedia)
        XCTAssertTrue(model.hasIssuedManagedMediaProfile)
        XCTAssertThrowsError(
            try scope {
                try model.makeManagedMediaExecutionResources(
                    binding: binding, bytesCapacity: 32 << 20, limits: MiMoMediaFixture.limits,
                    retaining: $0)
            }
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner) }
        // A supported wrapper mutation invalidates every issued preparation.
        try model.update(parameters: .unflattened([]), verify: .noUnusedKeys)
        XCTAssertThrowsError(
            try model.makeMultimodalProcessor(binding: binding, limits: MiMoMediaFixture.limits))
    }
}
