// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The tiny VLM through the CBv2 engine on segmented paged storage: image
/// features ride as causal embedding spans at scalar positions.
@Suite(.serialized, .tags(.integration))
struct Inkling2VLMEngineTests {
    static func engine(_ model: Inkling2VLM, chunk: Int = 16) throws -> (EngineV2, PagedKVBackend) {
        let kinds = model.cbv2LayerKinds
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, dtype: .float32, maxPrefillChunk: max(chunk, 32),
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: .float32, count: kinds.count)))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches, permitsCausalMedia: true),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: chunk, maxWaiting: 8, enablePrefixCache: false))
        return (engine, paged)
    }

    /// Greedy continuation by full recompute over spliced embeddings.
    static func noCacheGreedy(
        _ model: Inkling2VLM, prompt: [Int], image: LMInput.ProcessedImage, tokens: Int
    ) throws -> [Int] {
        var sequence = prompt
        var out: [Int] = []
        for _ in 0 ..< tokens {
            let logits = try model.forwardNoCache(
                tokens: MLXArray(sequence.map(Int32.init)).reshaped(1, -1), image: image)
            let next = argMax(logits[0, -1], axis: -1).item(Int.self)
            out.append(next)
            sequence.append(next)
        }
        return out
    }

    static func prompt(imageTokens: Int, prefix: Int = 5, suffix: Int = 7) -> (
        tokens: [Int], span: CBv2ImageSpan
    ) {
        let head = SyntheticModel.tokens(count: prefix, vocabularySize: 48, seed: 3)
        let tail = SyntheticModel.tokens(count: suffix, vocabularySize: 48, seed: 4)
        let tokens =
            head + Array(repeating: Inkling2TinyVLM.imageTokenId, count: imageTokens) + tail
        return (tokens, CBv2ImageSpan(tokenOffset: prefix, length: imageTokens))
    }

    static func generate(
        _ engine: EngineV2, id: UInt64, prompt: [Int], span: CBv2ImageSpan,
        features: MLXArray, tokens: Int
    ) async throws -> [Int] {
        let stream = try engine.submit(
            .init(
                id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0),
                maxTokens: tokens, prefixCacheEnabled: false,
                multimodal: CBv2MultimodalInput(spans: [span], attention: .causal) {
                    [features]
                }))
        return await cbv2SchedCollect(stream, timeoutSeconds: 120).tokens
    }

    /// One image whose 33 rows cross two 16-token prefill chunks.
    @Test(arguments: [8, 16, 64])
    func pagedImagePrefillMatchesTheFullRecompute(chunk: Int) async throws {
        let model = try Inkling2TinyVLM.model()
        let image = try Inkling2TinyVLM.image(height: 403, width: 95)
        let features = try model.perImageVisionFeatures(pixels: image.pixels, frames: image.frames)
        eval(features)
        let (prompt, span) = Self.prompt(imageTokens: features[0].dim(1))
        let expected = try Self.noCacheGreedy(model, prompt: prompt, image: image, tokens: 10)
        let (engine, backend) = try Self.engine(model, chunk: chunk)
        let tokens = try await Self.generate(
            engine, id: 1, prompt: prompt, span: span, features: features[0], tokens: 10)
        await engine.shutdown()
        #expect(tokens == expected)
        #expect(backend.bytesInUse == 0)
    }

    @Test func textOnlyRequestsThroughTheWrapperMatchTheTextModel() async throws {
        let model = try Inkling2TinyVLM.model()
        let prompt = SyntheticModel.tokens(count: 29, vocabularySize: 48, seed: 9)
        var sequence = prompt
        var expected: [Int] = []
        for _ in 0 ..< 8 {
            let logits = model.languageModel.forwardNoCache(
                MLXArray(sequence.map(Int32.init)).reshaped(1, -1)
            ).logits
            let next = argMax(logits[0, -1], axis: -1).item(Int.self)
            expected.append(next)
            sequence.append(next)
        }
        let (engine, _) = try Self.engine(model)
        let stream = try engine.submit(
            .init(
                id: .init(7), promptTokens: prompt, sampling: .init(temperature: 0),
                maxTokens: 8, prefixCacheEnabled: false))
        let tokens = await cbv2SchedCollect(stream, timeoutSeconds: 120).tokens
        await engine.shutdown()
        #expect(tokens == expected)
    }
}
