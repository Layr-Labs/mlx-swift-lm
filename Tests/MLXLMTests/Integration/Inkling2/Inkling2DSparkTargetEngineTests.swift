import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny Inkling 2 target (10 layer visits, paged sliding and full slots,
/// conv state as recurrent state) with a random DSpark drafter reading visits
/// 1, 5, 8 and the final norm. Greedy tokens with the drafter on must equal
/// the drafter off at every batch size: drafts only decide how many target
/// tokens one row-exact verify window commits.
@Suite(.serialized, .tags(.integration))
struct Inkling2DSparkTargetEngineTests {
    static let visits = [1, 5, 8, 10]

    static func target(dtype: DType) throws -> Inkling2Model {
        let config: [String: Any] = [
            "model_type": "inkling2_loop",
            "text_config": [
                "hidden_size": 128, "num_hidden_layers": 6, "vocab_size": 64,
                "unpadded_vocab_size": 60, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 64, "d_rel": 4, "rel_extent": 24, "log_scaling_n_floor": 8,
                "log_scaling_alpha": 0.5, "qk_l2_clip_threshold": 8.0,
                "rel_l2_clip_threshold": 2.0, "local_layer_ids": [0, 2, 4],
                "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
                "sliding_window_size": 24, "n_layers_per_attn_res_block": 2,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32,
                "loop_config": ["core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8],
            ] as [String: Any],
            "mtp_config": [
                "hidden_size": 128, "num_hidden_layers": 3, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 64, "intermediate_size": 64,
                "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0, "local_layer_ids": [0, 2],
                "sliding_window_size": 16, "block_size": 8, "markov_rank": 16,
                "markov_head_type": "rnn_hidden", "target_layer_ids": visits,
            ] as [String: Any],
        ]
        let model = Inkling2Model(
            try SyntheticModel.configuration(Inkling2Configuration.self, config))
        SyntheticModel.randomize(model, seed: 11)
        if dtype != .float32 {
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map {
                        (
                            $0.0,
                            $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                                ? $0.1.asType(dtype) : $0.1
                        )
                    }))
        }
        eval(model)
        return model
    }

    static func drafter(target: Inkling2Model, dtype: DType) throws -> Inkling2DSparkDrafter {
        #expect(target.dsparkVisits == visits)
        let configuration = Inkling2DSparkConfiguration(
            hiddenSize: 128, numHiddenLayers: 3, numAttentionHeads: 2, numKeyValueHeads: 1,
            headDim: 64, intermediateSize: 64, localLayerIds: [0, 2], slidingWindowSize: 16,
            blockSize: 8, markovRank: 16, targetLayerIds: visits, vocabularySize: 60,
            paddedVocabularySize: 64, finalLogitSoftcapping: 30, targetLayerVisits: 10)
        try configuration.validate()
        let model = Inkling2DSparkModel(configuration)
        SyntheticModel.randomize(model, seed: 5)
        if dtype != .float32 {
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
        }
        eval(model)
        model.prepareDerivedWeights()
        return Inkling2DSparkDrafter(
            model: model, target: target, requiredVerificationMode: .rectangularExact,
            commitsAcceptedPrefixPerRow: true)
    }

    static func engine(
        _ model: Inkling2Model, drafter: Inkling2DSparkDrafter?, dtype: DType, rows: Int
    ) throws -> EngineV2 {
        let kinds = model.cbv2LayerKinds
        let types = Array(repeating: dtype, count: kinds.count)
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 128 << 20, dtype: dtype, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20, layerDTypes: types))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: drafter == nil
                ? .init() : Inkling2DSparkDrafter.defaultMTPConfig(adaptive: false))
    }

    static func run(_ engine: EngineV2, prompts: [[Int]], tokens: Int) async throws -> [[Int]] {
        let streams = try prompts.enumerated().map { index, prompt in
            try engine.submit(
                .init(
                    id: .init(UInt64(index + 1)), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: tokens,
                    prefixCacheEnabled: false))
        }
        var outputs: [[Int]] = []
        for stream in streams {
            outputs.append(try await cbv2CollectFinished(stream, timeoutSeconds: 300).tokens)
        }
        return outputs
    }

    @Test(arguments: [1, 2, 4])
    func drafterOnEqualsOff(rows: Int) async throws {
        let dtype = DType.bfloat16
        let model = try Self.target(dtype: dtype)
        let drafter = try Self.drafter(target: model, dtype: dtype)
        let prompts = (0 ..< rows).map {
            SyntheticModel.tokens(count: 9 + 13 * $0, vocabularySize: 60, seed: 40 + $0)
        }
        let baseline = try Self.engine(model, drafter: nil, dtype: dtype, rows: rows)
        let expected = try await Self.run(baseline, prompts: prompts, tokens: 24)
        await baseline.shutdown()

        let engine = try Self.engine(model, drafter: drafter, dtype: dtype, rows: rows)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.verificationMode == .rectangularExact)
        let actual = try await Self.run(engine, prompts: prompts, tokens: 24)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(actual == expected)
        #expect(expected.allSatisfy { $0.count == 24 })
        #expect(metrics.rounds > 0)
        #expect(metrics.rectangularVerificationRounds > 0)
        #expect(metrics.serialVerificationRounds == 0)
        #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
        print(
            "rows=\(rows) rounds=\(metrics.rounds) drafted=\(metrics.draftedTokens) "
                + "accepted=\(metrics.acceptedTokens) skipped=\(metrics.skippedRows) "
                + "fallbacks=\(metrics.controllerFallbacks)")
    }
}
