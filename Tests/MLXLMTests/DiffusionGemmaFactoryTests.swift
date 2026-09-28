import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXVLM

@Suite("DiffusionGemma native factory", .serialized)
struct DiffusionGemmaFactoryTests {
    struct Loader: TokenizerLoader {
        enum Failure: Error { case tokenizer }
        let fail: Bool
        func load(from directory: URL) async throws -> any Tokenizer {
            if fail { throw Failure.tokenizer }
            return TestTokenizer(vocabularySize: 128)
        }
    }

    func fixture() throws -> (URL, [String: MLXArray]) {
        let metadata = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-oracle", withExtension: "json"))
        let tensors = try #require(
            Bundle.module.url(
                forResource: "diffusiongemma-text-oracle", withExtension: "safetensors"))
        let source = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadata)) as? [String: Any])
        let root: [String: Any] = [
            "model_type": "diffusion_gemma", "text_config": source["model_config"]!,
            "canvas_length": 4, "tie_word_embeddings": true, "eos_token_id": [1],
        ]
        let arrays = try loadArrays(url: tensors)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "diffusion-factory-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        do {
            try JSONSerialization.data(withJSONObject: root)
                .write(to: directory.appendingPathComponent("config.json"))
            try JSONEncoder().encode(
                DiffusionGemmaGenerationConfiguration(maxNewTokens: 4, eosTokenIds: [1])
            )
            .write(to: directory.appendingPathComponent("generation_config.json"))
            try Data("{# comment -#}\n<bos>template".utf8)
                .write(to: directory.appendingPathComponent("chat_template.jinja"))
            try save(
                arrays: arrays.filter { $0.key.hasPrefix("model.") },
                url: directory.appendingPathComponent("model.safetensors"))
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        return (directory, arrays)
    }

    @Test func nativeFactoryLoadsExactStateAndReleasesItsModel() async throws {
        let (directory, arrays) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        var context: DiffusionGemmaContext? = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(fail: false))
        weak var owner = context?.model
        #expect(context?.generationConfiguration.eosTokenIds == [1])
        #expect(context?.chatTemplate == "{# comment #}<bos>template")
        #expect(context?.model.parameters().flattened().count == 51)
        let cache = try context!.model.makeCache(expectedPromptLength: 11)
        _ = try context!.model.encode(tokenIds: arrays["prompt11.tokens"]!, cache: cache)
        let actual = try context!.model.denoise(canvasIds: arrays["prompt11.canvas"]!, cache: cache)
        let expected = arrays["prompt11.logits0"]!
        eval(actual, expected)
        #expect(
            actual.asArray(Float.self).map(\.bitPattern)
                == expected.asArray(Float.self).map(\.bitPattern))
        context = nil
        #expect(owner == nil, "Committed state does not retain the model on unload")
    }

    @Test func textOnlyNativeArtifactRejectsMediaBeforeMaterialization() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: Loader(fail: false))
        let engine = try context.makeNativeEngine(kvBytesCapacity: 16 * 1024 * 1024)
        let media = CBv2MultimodalInput(spans: [.init(tokenOffset: 0, length: 1)]) {
            Issue.record("Unsupported media must reject before touching its feature provider")
            return []
        }
        #expect(throws: CBv2NativeBlockError.unsupportedRequest("model has no native vision tower")) {
            try engine.submit(.init(id: .init(817), promptTokens: [2, 3], maxTokens: 4, multimodal: media))
        }
        #expect(engine.capacity().kvBytesReserved == 0)
        await engine.shutdown()
    }

    @Test func nativeEvaluationFaultIsRecoverableAndCannotCommitOrPoisonNextRequest() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: Loader(fail: false))
        let tokens = MLXArray([Int32(2), 3]).reshaped(1, 2)
        let baseline = try context.model.generateNative(promptTokenIds: tokens,
            generation: context.generationConfiguration, seed: 47)
        let identity = try DiffusionGemmaPrefixIdentity(tenantScope: "fault-fixture", artifact: "fixture",
            template: "fixture", media: "text-only", numericalProfile: "strict", epoch: "one")
        let session = try DiffusionGemmaGenerationSession(model: context.model, promptTokenIds: tokens,
            generation: context.generationConfiguration, seed: 47, prefixIdentity: identity,
            onEncodedBoundary: { _, _ in
                // Deterministic C++/MLX broadcast error, not a Swift test throw.
                let invalid = MLXArray.ones([2, 3]) + MLXArray.ones([4, 5])
                eval(invalid)
            })
        #expect(throws: MLX.MLXError.self) { try session.advance() }
        #expect(session.phase == .failed && session.retainedStateBytes == 0)
        #expect(session.generatedTokenCount == 0 && session.result == nil)
        let next = try context.model.generateNative(promptTokenIds: tokens,
            generation: context.generationConfiguration, seed: 47)
        #expect(next.tokenIds == baseline.tokenIds && next.denoisingSteps == baseline.denoisingSteps)
    }

    @Test func nativeContainerSupportsRepeatedGenerationWithoutSharedRequestState() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let container = try await DiffusionGemmaModelFactory.shared.loadContainer(
            from: directory, using: Loader(fail: false))
        func run() async throws -> DiffusionGemmaGenerationResult {
            try await container.perform { context in
                try context.model.generateNative(
                    promptTokenIds: MLXArray([Int32(2), 3]).reshaped(1, 2),
                    generation: context.generationConfiguration, seed: 123)
            }
        }
        let first = try await run()
        let second = try await run()
        #expect(first.tokenIds == second.tokenIds)
        #expect(first.generatedTokenCount == second.generatedTokenCount)
        #expect(first.denoisingSteps == second.denoisingSteps)
        #expect(first.generatedTokenCount > 0)
    }

    @Test func failedLoadRemainsRecoverableAndARFactoryRejectsDiffusion() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        await #expect(throws: Loader.Failure.self) {
            try await DiffusionGemmaModelFactory.shared.load(
                from: directory, using: Loader(fail: true))
        }
        await #expect(throws: ModelFactoryError.self) {
            try await VLMModelFactory.shared.load(from: directory, using: Loader(fail: false))
        }
        // Incomplete weights must not return a randomly initialized model.
        try save(
            arrays: ["unexpected.weight": MLXArray.ones([1])],
            url: directory.appendingPathComponent("model.safetensors"))
        await #expect(throws: (any Error).self) {
            try await DiffusionGemmaModelFactory.shared.load(
                from: directory, using: Loader(fail: false))
        }
        await #expect(throws: DiffusionGemmaModelError.self) {
            try await DiffusionGemmaModelFactory.shared.load(
                from: directory.appendingPathComponent("missing"), using: Loader(fail: false))
        }
    }

    @Test func nativeGenerationPrefixHitPreservesSamplingAndActualReuseAccounting() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(fail: false))
        let identity = try DiffusionGemmaPrefixIdentity(
            tenantScope: "local-test", artifact: "generated-fixture", template: "fixture",
            media: "text-only", numericalProfile: "frozen-oracle", epoch: "0")
        let tokens = MLXArray([Int32(2), 3]).reshaped(1, 2)
        var checkpoint: DiffusionGemmaPrefixCheckpoint?
        let cold = try context.model.generateNative(
            promptTokenIds: tokens, generation: context.generationConfiguration, seed: 77,
            prefixIdentity: identity, onPromptCheckpoint: { checkpoint = $0 })
        let hit = try context.model.generateNative(
            promptTokenIds: tokens, generation: context.generationConfiguration, seed: 77,
            prefixCheckpoint: #require(checkpoint), prefixIdentity: identity)
        let unobserved = try context.model.generateNative(
            promptTokenIds: tokens, generation: context.generationConfiguration, seed: 77)
        #expect(cold.tokenIds == hit.tokenIds && hit.tokenIds == unobserved.tokenIds)
        #expect(cold.denoisingSteps == hit.denoisingSteps)
        #expect(hit.generatedTokenCount == cold.generatedTokenCount)
        #expect(hit.reusedPromptTokenCount == 2)
        #expect(cold.reusedPromptTokenCount == 0 && unobserved.reusedPromptTokenCount == 0)
        #expect(throws: DiffusionGemmaModelError.self) {
            try context.model.generateNative(
                promptTokenIds: tokens, generation: context.generationConfiguration, seed: 77,
                prefixCheckpoint: checkpoint)
        }
        #expect(throws: CancellationError.self) {
            try context.model.generateNative(
                promptTokenIds: tokens, generation: context.generationConfiguration, seed: 77,
                prefixCheckpoint: checkpoint, prefixIdentity: identity, isCancelled: { true })
        }
    }

    @Test func steppedCancellationNeverCommitsProvisionalTokensAndReleasesState() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(fail: false))
        let session = try DiffusionGemmaGenerationSession(
            model: context.model, promptTokenIds: MLXArray([Int32(2), 3]).reshaped(1, 2),
            generation: context.generationConfiguration, seed: 66)
        #expect(session.phase == .prefill && session.generatedTokenCount == 0)
        if case .prefill(let count, let complete) = try session.advance() {
            #expect(count == 2 && complete)
        } else {
            Issue.record("Expected prefill, not visible output")
        }
        #expect(session.phase == .denoising && session.retainedCacheBytes > 0)
        if case .refined = try session.advance() {
        } else {
            Issue.record("Stability1 cannot finalize its first denoising step")
        }
        #expect(session.denoisingSteps == 1 && session.generatedTokenCount == 0)
        session.cancel()
        #expect(session.phase == .cancelled && session.retainedCacheBytes == 0)
        #expect(session.result == nil)
        #expect(throws: CancellationError.self) { try session.advance() }
    }

    @Test func steppedMultipleCanvasesOnlyEncodeCompleteNonterminalBlocks() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(fail: false))
        let generation = try DiffusionGemmaGenerationConfiguration(
            maxNewTokens: 9, maxDenoisingSteps: 2, eosTokenIds: [])
        let session = try DiffusionGemmaGenerationSession(
            model: context.model, promptTokenIds: MLXArray([Int32(2), 3]).reshaped(1, 2),
            generation: generation, seed: 77)
        var blocks = [[Int32]]()
        var positions = [Int]()
        while session.phase != .finished {
            if case .committed(let tokens, let terminal) = try session.advance() {
                blocks.append(tokens)
                if !terminal { positions.append(session.committedCachePosition) }
            }
        }
        let result = try #require(session.result)
        #expect(blocks.map(\.count) == [4, 4, 1])
        #expect(positions == [2, 6], "Denoising must not append provisional or final unused K/V")
        #expect(result.tokenIds == blocks.flatMap { $0 })
        #expect(result.generatedTokenCount == 9 && result.committedCanvasCount == 3)
        #expect(result.denoisingSteps == 6 && result.workTokenCount == 24)
        #expect(session.retainedCacheBytes == 0 && session.phase == .finished)
        #expect(throws: DiffusionGemmaModelError.self) { try session.advance() }
    }

    @Test func interleavedSessionsMatchIsolatedRequestsWithoutSharingNoiseOrKV() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(fail: false))
        let generation = try DiffusionGemmaGenerationConfiguration(
            maxNewTokens: 8, maxDenoisingSteps: 2, eosTokenIds: [])
        let seeds: [UInt64] = [42, 43]
        let isolated = try seeds.map { seed in
            try context.model.generateNative(
                promptTokenIds: MLXArray([Int32(2), Int32(seed)]).reshaped(1, 2),
                generation: generation, seed: seed)
        }
        let sessions = try seeds.map { seed in
            try DiffusionGemmaGenerationSession(
                model: context.model,
                promptTokenIds: MLXArray([Int32(2), Int32(seed)]).reshaped(1, 2),
                generation: generation, seed: seed)
        }
        // Round-robin stepping tests isolation, not rectangular GPU batching.
        while sessions.contains(where: { $0.phase != .finished }) {
            for session in sessions.reversed() where session.phase != .finished {
                _ = try session.advance()
            }
        }
        for (index, session) in sessions.enumerated() {
            #expect(session.result?.tokenIds == isolated[index].tokenIds)
            #expect(session.result?.denoisingSteps == isolated[index].denoisingSteps)
        }
    }

    @Test func chunkedSessionMatchesIndependentNativeReferenceAndCanCancelAfterPrefill()
        async throws
    {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(fail: false))
        let url = try #require(
            Bundle.module.url(
                forResource: "diffusiongemma-chunked-text-oracle", withExtension: "safetensors"))
        let arrays = try loadArrays(url: url)
        let tokens = arrays["tokens"]!
        let identity = try DiffusionGemmaPrefixIdentity(
            tenantScope: "fixture", artifact: "frozen-tiny", template: "literal-tokens",
            media: "text-only", numericalProfile: "native-reference-chunks", epoch: "0")
        func exact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
            eval(actual, expected)
            #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
            let matches =
                actual.asArray(Float.self).map(\.bitPattern)
                == expected.asArray(Float.self).map(\.bitPattern)
            #expect(
                matches, "\(label): chunk policy must match the unchanged independent reference")
        }
        for width in [1, 3, 7] {
            var checkpoint: DiffusionGemmaPrefixCheckpoint?
            let session = try DiffusionGemmaGenerationSession(
                model: context.model, promptTokenIds: tokens,
                generation: context.generationConfiguration, seed: 9, prefillChunkSize: width,
                prefixIdentity: identity, onPromptCheckpoint: { checkpoint = $0 })
            var computed = 0
            while session.phase == .prefill {
                if case .prefill(let count, let complete) = try session.advance() {
                    #expect(count > 0 && count <= width)
                    computed += count
                    #expect(complete == (computed == tokens.dim(1)))
                } else {
                    Issue.record("Expected bounded prefill")
                }
            }
            #expect(
                computed == 20 && session.generatedTokenCount == 0 && session.denoisingSteps == 0)
            session.cancel()
            let cache = try context.model.model.decoder.restorePrefix(
                #require(checkpoint), identity: identity, promptTokenIds: tokens)
            for (index, layer) in cache.snapshots().enumerated() {
                exact(
                    layer.keys, arrays["width\(width).layer\(index).keys"]!,
                    "width\(width).layer\(index).keys")
                exact(
                    layer.values, arrays["width\(width).layer\(index).values"]!,
                    "width\(width).layer\(index).values")
            }
            exact(
                try context.model.denoise(canvasIds: arrays["canvas"]!, cache: cache),
                arrays["width\(width).logits"]!, "width\(width).logits")
            #expect(session.retainedCacheBytes == 0)
        }
    }

    @Test func realNativeEngineMatchesDirectGenerationAndRejectsUnsupportedControls() async throws {
        let (directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory, using: Loader(fail: false))
        let engine = try context.makeNativeEngine(kvBytesCapacity: 128 * 1024 * 1024)
        let prompt = [Int32(2), 3]
        let baseline = try context.model.generateNative(
            promptTokenIds: MLXArray(prompt).reshaped(1, 2),
            generation: context.generationConfiguration, seed: 123)
        let stream = try engine.submit(
            .init(
                id: .init(17), promptTokens: prompt.map(Int.init), sampling: .init(seed: 123),
                maxTokens: 4))
        var raw = [Int]()
        var text = ""
        var usage: CBv2Usage?
        for await event in stream {
            switch event {
            case .delta(let value, let tokens, _):
                raw += tokens
                text += value
            case .finished(let reason, let value):
                #expect(reason == (baseline.finishReason == "stop" ? .stop : .length))
                usage = value
            }
        }
        #expect(raw.prefix(baseline.tokenIds.count).map(Int32.init) == baseline.tokenIds)
        #expect(raw.count == baseline.generatedTokenCount)
        #expect(usage?.completionTokens == baseline.generatedTokenCount)
        #expect(
            text
                == context.tokenizer.decode(
                    tokenIds: baseline.tokenIds.map(Int.init), skipSpecialTokens: false))
        #expect(engine.capacity().kvBytesReserved == 0)
        #expect(throws: CBv2NativeBlockError.self) {
            try engine.submit(
                .init(
                    id: .init(18), promptTokens: [2, 3], sampling: .init(topP: 0.5), maxTokens: 4))
        }
        #expect(engine.capacity().activeRequests == 0)
        await engine.shutdown()
    }
}
