// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The tiny Inkling 2 VLM loaded through `VLMTypeRegistry` (the provider's
/// load path) with the random DSpark drafter of `Inkling2DSparkTargetEngineTests`
/// bound to its text model. Text requests draft and match the drafter-off
/// tokens; image requests stay target-only and match too.
@Suite(.serialized, .tags(.integration))
struct Inkling2VLMDSparkEngineTests {
    static let dtype = DType.bfloat16

    static func model() async throws -> Inkling2VLM {
        var configuration = Inkling2TinyVLM.configuration
        configuration["mtp_config"] =
            [
                "hidden_size": 128, "num_hidden_layers": 3, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 64, "intermediate_size": 64,
                "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0, "local_layer_ids": [0, 2],
                "sliding_window_size": 16, "block_size": 8, "markov_rank": 16,
                "markov_head_type": "rnn_hidden",
                "target_layer_ids": Inkling2DSparkTargetEngineTests.visits,
            ] as [String: Any]
        let loaded = try await VLMTypeRegistry.shared.createModel(
            configuration: try JSONSerialization.data(withJSONObject: configuration),
            modelType: "inkling2_loop")
        let model = try #require(loaded as? Inkling2VLM)
        SyntheticModel.randomize(model, seed: 11)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map {
                    (
                        $0.0,
                        $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                            ? $0.1.asType(dtype) : $0.1
                    )
                }))
        eval(model)
        return model
    }

    static func engine(_ model: Inkling2VLM, drafter: Inkling2DSparkDrafter?, rows: Int) throws
        -> EngineV2
    {
        let kinds = model.cbv2LayerKinds
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 128 << 20, dtype: dtype, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: dtype, count: kinds.count)))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches, permitsCausalMedia: true),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: drafter == nil
                ? .init() : Inkling2DSparkDrafter.defaultMTPConfig(adaptive: false))
    }

    /// One request per entry; `features` non-nil makes it an image request.
    struct Prompt {
        let tokens: [Int]
        let span: CBv2ImageSpan?
        let features: MLXArray?
    }

    static func run(_ engine: EngineV2, _ prompts: [Prompt], tokens: Int) async throws -> [[Int]] {
        let streams = try prompts.enumerated().map { index, prompt in
            try engine.submit(
                .init(
                    id: .init(UInt64(index + 1)), promptTokens: prompt.tokens,
                    sampling: .init(temperature: 0), maxTokens: tokens,
                    prefixCacheEnabled: false,
                    multimodal: prompt.features.map { features in
                        CBv2MultimodalInput(spans: [prompt.span!], attention: .causal) {
                            [features]
                        }
                    }))
        }
        var outputs: [[Int]] = []
        for stream in streams {
            outputs.append(await cbv2SchedCollect(stream, timeoutSeconds: 300).tokens)
        }
        return outputs
    }

    static func textPrompt(_ seed: Int, count: Int) -> Prompt {
        Prompt(
            tokens: SyntheticModel.tokens(count: count, vocabularySize: 48, seed: seed),
            span: nil, features: nil)
    }

    static func imagePrompt(_ model: Inkling2VLM) throws -> Prompt {
        let image = try Inkling2TinyVLM.image(height: 75, width: 130)
        let features = try model.perImageVisionFeatures(pixels: image.pixels, frames: image.frames)
        eval(features)
        let head = SyntheticModel.tokens(count: 5, vocabularySize: 48, seed: 3)
        let tail = SyntheticModel.tokens(count: 7, vocabularySize: 48, seed: 4)
        let count = features[0].dim(1)
        return Prompt(
            tokens: head + Array(repeating: Inkling2TinyVLM.imageTokenId, count: count) + tail,
            span: CBv2ImageSpan(tokenOffset: head.count, length: count),
            features: features[0][0])
    }

    /// Drafter off vs on over the same prompts; returns both outputs and the
    /// drafter-on metrics.
    static func compare(_ model: Inkling2VLM, _ prompts: [Prompt], tokens: Int) async throws
        -> (off: [[Int]], on: [[Int]], metrics: CBv2MTPMetrics)
    {
        let baseline = try engine(model, drafter: nil, rows: prompts.count)
        let off = try await run(baseline, prompts, tokens: tokens)
        await baseline.shutdown()
        let drafter = try Inkling2DSparkTargetEngineTests.drafter(
            target: model.languageModel, dtype: dtype)
        let speculative = try engine(model, drafter: drafter, rows: prompts.count)
        #expect(speculative.loopForTesting.mtp != nil, "the drafter binds to the VLM target")
        let on = try await run(speculative, prompts, tokens: tokens)
        let metrics = try #require(speculative.mtpMetricsSnapshot())
        #expect(speculative.loopForTesting.mtp?.requestStateCountForTesting == 0)
        await speculative.shutdown()
        return (off, on, metrics)
    }

    @Test(arguments: [1, 2])
    func textRequestsDraftThroughTheVLM(rows: Int) async throws {
        let model = try await Self.model()
        let prompts = (0 ..< rows).map { Self.textPrompt(40 + $0, count: 9 + 13 * $0) }
        let (off, on, metrics) = try await Self.compare(model, prompts, tokens: 24)
        #expect(on == off)
        #expect(off.allSatisfy { $0.count == 24 })
        #expect(metrics.rounds > 0)
        #expect(metrics.rectangularVerificationRounds > 0)
        print("vlm text rows=\(rows) rounds=\(metrics.rounds) accepted=\(metrics.acceptedTokens)")
    }

    @Test func imageRequestsStayTargetOnly() async throws {
        let model = try await Self.model()
        let prompt = try Self.imagePrompt(model)
        let (off, on, metrics) = try await Self.compare(model, [prompt], tokens: 16)
        #expect(on == off)
        #expect(off[0].count == 16)
        #expect(metrics.rounds == 0)
        #expect(metrics.draftedTokens == 0)
        print("vlm image rounds=\(metrics.rounds) skipped=\(metrics.skippedRows)")
    }

    /// A text row beside an image row: both match the drafter-off tokens.
    @Test func mixedTextAndImageRowsMatchTheDrafterOff() async throws {
        let model = try await Self.model()
        let prompts = [Self.textPrompt(41, count: 22), try Self.imagePrompt(model)]
        let (off, on, metrics) = try await Self.compare(model, prompts, tokens: 16)
        #expect(on == off)
        print("vlm mixed rounds=\(metrics.rounds) skipped=\(metrics.skippedRows)")
    }
}
