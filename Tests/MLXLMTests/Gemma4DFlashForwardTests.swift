// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM

@Suite("Gemma4TextModel forwardForDFlash")
struct Gemma4DFlashForwardTests {
    @Test func autoVectorVerifySuffixUsesConservativeEnvelope() {
        #expect(
            (1 ... 3).map {
                gemma4DFlashAutoVectorSuffixLength(
                    sequenceLength: $0,
                    baseSafeVectorTokens: 1,
                    k16SafeVectorTokens: 1)
            } == Array(repeating: 1, count: 3))

        #expect(
            (4 ... 15).map {
                gemma4DFlashAutoVectorSuffixLength(
                    sequenceLength: $0,
                    baseSafeVectorTokens: 1,
                    k16SafeVectorTokens: 1,
                    perSequenceSafeVectorTokens: [:])
            } == Array(repeating: 1, count: 12))

        #expect(
            (9 ... 15).map {
                gemma4DFlashAutoVectorSuffixLength(
                    sequenceLength: $0,
                    baseSafeVectorTokens: 9,
                    k16SafeVectorTokens: 13,
                    perSequenceSafeVectorTokens: [:])
            } == Array(repeating: 9, count: 7))

        #expect(
            gemma4DFlashAutoVectorSuffixLength(
                sequenceLength: 16,
                baseSafeVectorTokens: 1,
                k16SafeVectorTokens: 1,
                perSequenceSafeVectorTokens: [:]) == 1)
    }

    private func tinyGemma4Config(
        hiddenLayers: Int = 4,
        sharedLayers: Int = 0,
        slidingWindow: Int = 16
    ) throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 16,
            "num_hidden_layers": \(hiddenLayers),
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "global_partial_rotary_factor": 1.0,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": \(sharedLayers),
            "hidden_size_per_layer_input": 0,
            "sliding_window": \(slidingWindow),
            "sliding_window_pattern": 2,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "attention_k_eq_v": false,
            "final_logit_softcapping": 30.0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyGemma4WrapperConfig() throws -> Gemma4Configuration {
        let json = """
        {
            "model_type": "gemma4",
            "vocab_size": 32,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 8,
                "global_partial_rotary_factor": 1.0,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 16,
                "sliding_window_pattern": 2,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128,
                "attention_k_eq_v": false,
                "final_logit_softcapping": 30.0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true
            }
        }
        """
        return try JSONDecoder.json5().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }

    @Test func logitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test func capturesRequestedHiddenStatesInRequestedOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [3, 1]
            )

            #expect(forward.hiddenStates.count == 2)
            #expect(forward.hiddenStates[0].shape == [1, 3, 16])
            #expect(forward.hiddenStates[1].shape == [1, 3, 16])
            #expect(forward.targetHidden.shape == [1, 3, 32])
        }
    }

    @Test func greedyTokensMatchSoftcappedLogitsArgmax() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )
            let greedy = try model.forwardGreedyTokensForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )

            #expect((greedy.tokens .== forward.logits.argMax(axis: -1)).all().item(Bool.self))
        }
    }

    @Test func capturesSharedKVLayerHiddenStates() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(
                try tinyGemma4Config(hiddenLayers: 4, sharedLayers: 2))
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [1, 3]
            )

            #expect(forward.logits.shape == [1, 3, 32])
            #expect(forward.targetHidden.shape == [1, 3, 32])
        }
    }

    @Test func cachedBlockForwardMatchesSequentialDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(
                try tinyGemma4Config(hiddenLayers: 6, sharedLayers: 2))
            eval(model)
            let prompt = MLXArray([Int32(1), 2, 3, 4])[.newAxis, .ellipsis]
            let block = MLXArray([Int32(5), 6, 7, 8])[.newAxis, .ellipsis]

            let blockCache = model.newCache(parameters: nil)
            _ = model(prompt, cache: blockCache)
            let blockLogits = model(block, cache: blockCache)

            let sequentialCache = model.newCache(parameters: nil)
            _ = model(prompt, cache: sequentialCache)
            let sequentialLogits = concatenated(
                (0 ..< block.dim(1)).map { i in
                    model(block[0..., i ..< (i + 1)], cache: sequentialCache)
                },
                axis: 1)

            eval(blockLogits, sequentialLogits)
            #expect(
                allClose(blockLogits, sequentialLogits, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    @Test func cachedBlockForwardMatchesSequentialDecodeAfterSlidingWindowSaturation() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(
                try tinyGemma4Config(hiddenLayers: 6, sharedLayers: 2, slidingWindow: 4))
            eval(model)
            let prompt = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7])[.newAxis, .ellipsis]
            let block = MLXArray([Int32(8), 9, 10, 11])[.newAxis, .ellipsis]

            let blockCache = model.newCache(parameters: nil)
            _ = model(prompt, cache: blockCache)
            let blockLogits = model(block, cache: blockCache)

            let sequentialCache = model.newCache(parameters: nil)
            _ = model(prompt, cache: sequentialCache)
            let sequentialLogits = concatenated(
                (0 ..< block.dim(1)).map { i in
                    model(block[0..., i ..< (i + 1)], cache: sequentialCache)
                },
                axis: 1)

            eval(blockLogits, sequentialLogits)
            #expect(
                allClose(blockLogits, sequentialLogits, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    @Test func dFlashBlockForwardMatchesSequentialAfterSlidingWindowSaturation() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(
                try tinyGemma4Config(hiddenLayers: 6, sharedLayers: 2, slidingWindow: 4))
            eval(model)
            let prompt = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7])[.newAxis, .ellipsis]
            let block = MLXArray([Int32(8), 9, 10, 11])[.newAxis, .ellipsis]
            let targetLayerIds = [1, 3]

            let blockCache = model.newCache(parameters: nil)
            _ = model(prompt, cache: blockCache)
            let blockForward = try model.forwardForDFlash(
                block, cache: blockCache, targetLayerIds: targetLayerIds)

            let sequentialCache = model.newCache(parameters: nil)
            _ = model(prompt, cache: sequentialCache)
            let sequentialForwards = try (0 ..< block.dim(1)).map { i in
                try model.forwardForDFlash(
                    block[0..., i ..< (i + 1)],
                    cache: sequentialCache,
                    targetLayerIds: targetLayerIds)
            }
            let sequentialLogits = concatenated(sequentialForwards.map(\.logits), axis: 1)
            let sequentialHidden = concatenated(sequentialForwards.map(\.targetHidden), axis: 1)

            eval(blockForward.logits, sequentialLogits, blockForward.targetHidden, sequentialHidden)
            #expect(
                allClose(blockForward.logits, sequentialLogits, rtol: 1e-4, atol: 1e-4)
                    .item(Bool.self))
            #expect(
                allClose(blockForward.targetHidden, sequentialHidden, rtol: 1e-4, atol: 1e-4)
                    .item(Bool.self))
        }
    }

    @Test func autoVerifyPathReportsSubphaseTimings() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MLX_GEMMA4_DFLASH_SEQUENTIAL_VERIFY"] == nil,
            environment["MLX_GEMMA4_DFLASH_AUTO_VERIFY"] == nil,
            environment["MLX_GEMMA4_DFLASH_BATCHED_VECTOR_VERIFY"] == nil
        else {
            return
        }

        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(
                try tinyGemma4Config(hiddenLayers: 6, sharedLayers: 2, slidingWindow: 4))
            eval(model)
            let prompt = MLXArray([Int32(1), 2, 3, 4, 5])[.newAxis, .ellipsis]
            let block = MLXArray([Int32(6), 7, 8, 9])[.newAxis, .ellipsis]
            let cache = model.newCache(parameters: nil)
            _ = model(prompt, cache: cache)

            let out = try model.forwardGreedyTokensForDFlash(
                block,
                cache: cache,
                targetLayerIds: [1, 3],
                collectVerifyTimings: true)

            let timings = try #require(out.verifyTimings)
            #expect(out.tokens.shape == [1, 4])
            #expect(out.targetHidden.shape == [1, 4, 32])
            #expect(timings.trunkSeconds.isFinite && timings.trunkSeconds >= 0)
            #expect(timings.lmHeadSeconds.isFinite && timings.lmHeadSeconds >= 0)
            #expect(timings.softcapArgmaxSeconds.isFinite && timings.softcapArgmaxSeconds >= 0)
        }
    }

    @Test func drafterEmbeddingUsesGemmaInputScale() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2])[.newAxis, .ellipsis]

            let dFlashEmbedding = model.embedTokensForDFlash(tokens)
            let drafterEmbedding = model.embedTokensForDrafter(tokens)

            #expect(
                allClose(dFlashEmbedding, drafterEmbedding, rtol: 1e-5, atol: 1e-5)
                    .item(Bool.self))
        }
    }

    @Test func dFlashHiddenLogitsAreRawHeadOutput() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let hidden = MLXArray.ones([1, 2, 16]) * 1_000_000

            let rawLogits = model.logitsForDFlashHidden(hidden)

            #expect(MLX.abs(rawLogits).max().item(Float.self) > 30.0)
        }
    }

    @Test func wrapperDelegatesDFlashTargetSurfaceToTextModel() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4Model(try tinyGemma4WrapperConfig())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            )
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(model.dFlashVocabularySize == 32)
            #expect(model.dFlashHiddenSize == 16)
            #expect(model.dFlashLayerCount == 4)
            #expect(forward.targetHidden.shape == [1, 3, 32])
            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test func rejectsInvalidLayerIds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            let tokens = MLXArray([Int32(1), 2])[.newAxis, .ellipsis]

            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens,
                    cache: model.newCache(parameters: nil),
                    targetLayerIds: []
                )
            }

            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens,
                    cache: model.newCache(parameters: nil),
                    targetLayerIds: [0, 0]
                )
            }

            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens,
                    cache: model.newCache(parameters: nil),
                    targetLayerIds: [4]
                )
            }
        }
    }
}
