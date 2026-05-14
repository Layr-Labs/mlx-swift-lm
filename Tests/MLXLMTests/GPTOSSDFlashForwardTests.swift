// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM

@Suite("GPTOSSModel forwardForDFlash")
struct GPTOSSDFlashForwardTests {
    private func tinyGPTOSSConfig(hiddenLayers: Int = 4) throws -> GPTOSSConfiguration {
        let layerTypes = Array(
            repeating: #""sliding_attention", "full_attention""#,
            count: hiddenLayers / 2
        ).joined(separator: ", ")
        let json = """
        {
            "model_type": "gpt_oss",
            "num_hidden_layers": \(hiddenLayers),
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "vocab_size": 32,
            "rms_norm_eps": 1e-5,
            "hidden_size": 16,
            "intermediate_size": 16,
            "head_dim": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "sliding_window": 16,
            "rope_theta": 150000,
            "layer_types": [\(layerTypes)]
        }
        """
        return try JSONDecoder.json5().decode(GPTOSSConfiguration.self, from: Data(json.utf8))
    }

    @Test func logitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = GPTOSSModel(try tinyGPTOSSConfig())
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
            let model = GPTOSSModel(try tinyGPTOSSConfig())
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

    @Test func rawDFlashHeadHasExpectedShape() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = GPTOSSModel(try tinyGPTOSSConfig())
            eval(model)
            let hidden = MLXArray.ones([1, 2, 16])

            let logits = model.logitsForDFlashHidden(hidden)

            #expect(logits.shape == [1, 2, 32])
        }
    }

    @Test func rejectsInvalidLayerIds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = GPTOSSModel(try tinyGPTOSSConfig())
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
