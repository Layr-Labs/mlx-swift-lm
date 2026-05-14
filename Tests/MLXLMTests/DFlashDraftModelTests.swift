// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM

@Suite("DFlashDraftModel")
struct DFlashDraftModelTests {
    private func tinyQwen3Config() throws -> Qwen3Configuration {
        let json = """
        {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128
        }
        """
        return try JSONDecoder.json5().decode(Qwen3Configuration.self, from: Data(json.utf8))
    }

    private func dflashConfig(
        layerTypes: String = #""full_attention", "full_attention""#,
        numTargetLayers: Int = 3
    ) throws -> DFlashConfiguration
    {
        let slidingWindow =
            layerTypes.contains("sliding_attention")
            ? #""sliding_window": 64,"#
            : ""
        let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 128,
            "block_size": 4,
            "num_target_layers": \(numTargetLayers),
            "layer_types": [\(layerTypes)],
            \(slidingWindow)
            "tie_word_embeddings": true,
            "dflash_config": {
                "target_layer_ids": [0, 1],
                "mask_token_id": 4
            }
        }
        """
        return try JSONDecoder.json5().decode(DFlashConfiguration.self, from: Data(json.utf8))
    }

    @Test func makeCacheUsesConfiguredLayerTypes() throws {
        try Device.withDefaultDevice(.cpu) {
            let draft = DFlashDraftModel(
                config: try dflashConfig(layerTypes: #""sliding_attention", "full_attention""#))
            let cache = try draft.makeCache()
            #expect(cache.count == 2)
            #expect(cache[0] is RotatingKVCache)
            #expect(cache[1] is KVCacheSimple)
        }
    }

    @Test func forwardProducesLogitsAfterBindingTarget() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let draft = DFlashDraftModel(config: try dflashConfig())
            try draft.bind(target: target)
            eval(target, draft)

            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let targetOut = try target.forwardForDFlash(
                tokens,
                cache: target.newCache(parameters: nil),
                targetLayerIds: draft.config.targetLayerIds
            )
            let block = MLXArray([Int32(3), Int32(draft.config.maskTokenId)])[.newAxis, .ellipsis]
            let logits = try draft(
                block,
                targetHidden: targetOut.targetHidden,
                cache: try draft.makeCache(),
                logitsStart: 1
            )
            eval(logits)
            #expect(logits.shape == [1, 1, 32])
        }
    }

    @Test func draftBlockProducesConfiguredNumberOfTokens() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let draft = DFlashDraftModel(config: try dflashConfig())
            try draft.bind(target: target)
            eval(target, draft)

            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let targetOut = try target.forwardForDFlash(
                tokens,
                cache: target.newCache(parameters: nil),
                targetLayerIds: draft.config.targetLayerIds
            )
            let draftTokens = try draft.draftBlock(
                bonus: 3,
                targetHidden: targetOut.targetHidden,
                cache: try draft.makeCache(),
                blockSize: 4
            )
            eval(draftTokens)
            #expect(draftTokens.shape == [1, 3])
        }
    }

    @Test func rejectsForwardBeforeBinding() throws {
        try Device.withDefaultDevice(.cpu) {
            let draft = DFlashDraftModel(config: try dflashConfig())
            let block = MLXArray([Int32(1), Int32(4)])[.newAxis, .ellipsis]
            let hidden = MLXArray.zeros([1, 1, draft.config.targetHiddenSize])

            #expect(throws: DFlashError.self) {
                _ = try draft(
                    block,
                    targetHidden: hidden,
                    cache: try draft.makeCache(),
                    logitsStart: 1
                )
            }
        }
    }

    @Test func rejectsTargetLayerCountMismatchOnBind() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let draft = DFlashDraftModel(config: try dflashConfig(numTargetLayers: 4))

            #expect(throws: DFlashError.incompatibleDrafter(
                field: "numTargetLayers",
                drafter: "4",
                target: "3"
            )) {
                try draft.bind(target: target)
            }
        }
    }
}
