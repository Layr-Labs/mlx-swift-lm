// Copyright © 2026 Eigen Labs.

import Foundation
import MLXLLM
import MLXLMCommon
import Testing

extension UnitTests {
    struct MuseGlimmerTextModelConfigurationTests {
        private func decode(_ object: [String: Any]) throws -> MuseGlimmerTextModelConfiguration {
            try JSONDecoder().decode(
                MuseGlimmerTextModelConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        @Test func defaultsDescribeThePublishedDecoder() throws {
            let c = try decode([:])
            #expect(c.hiddenLayers == 52 && c.hiddenSize == 6656 && c.intermediateSize == 19968)
            #expect(c.attentionHeads == 32 && c.kvHeads == 2 && c.headDimensions == 128)
            #expect(c.vocabularySize == 202048 && c.slidingWindow == 2048)
            #expect(c.rmsNormEps == 1e-5 && c.postNormEps == 1e-8 && c.qkScaleFactor == 3.87)
            #expect(c.finalLogitSoftcapping == 20 && c.outputMultiplier == 0.196_116_135_138_184_04)
            // Three sliding layers, then one full layer, thirteen times.
            #expect(c.layerTypes.count == 52)
            for layer in 0 ..< 52 {
                #expect(c.isSliding(layer) == (layer % 4 != 3))
                #expect(c.layerRopeTheta[layer] == (layer % 4 != 3 ? 500_000 : 0))
            }
            let kinds = c.cbv2LayerKinds
            #expect(kinds.count == 52)
            #expect(kinds.filter { $0.attention == .slidingWindow(2048) }.count == 39)
            #expect(kinds.filter { $0.attention == .full }.count == 13)
            #expect(
                kinds.allSatisfy {
                    $0.headDim == 128 && $0.valueHeadDim == 128 && $0.kvHeads == 2
                        && $0.queryHeads == 32 && $0.sharesKVWithLayer == nil && !$0.hasSinks
                        && !$0.isBidirectional
                })
            #expect(kinds[51].attention == .full && kinds[0].attention == .slidingWindow(2048))
        }

        @Test func explicitTablesAndTheNestedCheckpointConfigDecode() throws {
            let text: [String: Any] = [
                "num_hidden_layers": 3, "hidden_size": 16, "sliding_window": 4,
                "layer_types": ["full_attention", "sliding_attention", "full_attention"],
                "layer_rope_theta": [0, 1000.0, 0],
            ]
            let c = try decode(text)
            #expect(c.layerTypes == ["full_attention", "sliding_attention", "full_attention"])
            #expect(c.layerRopeTheta == [0, 1000, 0])
            let root = try JSONSerialization.data(
                withJSONObject: ["model_type": "muse_glimmer", "text_config": text])
            #expect(
                try MuseGlimmerTextModelConfiguration.textConfiguration(
                    checkpointConfiguration: root) == c)
            let again = try JSONDecoder().decode(
                MuseGlimmerTextModelConfiguration.self, from: JSONEncoder().encode(c))
            #expect(again == c)
        }

        @Test func refusesInconsistentArchitectures() throws {
            let cases: [[String: Any]] = [
                ["num_hidden_layers": 3, "layer_types": ["full_attention"]],
                ["num_hidden_layers": 1, "layer_types": ["window_attention"]],
                ["num_hidden_layers": 2, "layer_rope_theta": [0]],
                ["num_attention_heads": 6, "num_key_value_heads": 4],
                ["head_dim": 7], ["sliding_window": 0], ["num_hidden_layers": 0],
                ["attention_bias": true], ["tie_word_embeddings": true],
                ["num_hidden_layers": 1, "layer_rope_theta": [-1.0]],
            ]
            for object in cases {
                #expect(throws: (any Error).self) { try decode(object) }
            }
        }
    }
}
