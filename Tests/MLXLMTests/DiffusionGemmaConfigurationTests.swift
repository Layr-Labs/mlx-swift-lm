import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("DiffusionGemma root configuration and vision projection", .serialized)
struct DiffusionGemmaConfigurationTests {
    private func fixture() throws -> [String: Any] {
        let url = try #require(
            Bundle.module.url(forResource: "diffusiongemma-root-config", withExtension: "json"))
        return try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    private func decode(_ values: [String: Any]) throws -> DiffusionGemmaConfiguration {
        try JSONDecoder().decode(
            DiffusionGemmaConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
    }

    @Test func fullRootPreservesNondefaultGeometryEOSAndPrecision() throws {
        var root = try fixture()
        root["canvas_length"] = 128
        root["eos_token_id"] = [0, 106]
        root["video_token_id"] = 123
        var vision = root["vision_config"] as! [String: Any]
        vision["rope_parameters"] = ["rope_type": "default", "rope_theta": 137.0]
        root["vision_config"] = vision
        root["quantization"] =
            [
                "group_size": 64, "bits": 4, "mode": "affine",
                "model.encoder.embed_vision.embedding_projection": [
                    "group_size": 32, "bits": 8, "mode": "affine",
                ],
                "model.decoder.layers.0.experts.down_proj": false,
            ] as [String: Any]
        let config = try decode(root)
        let again = try JSONDecoder().decode(
            DiffusionGemmaConfiguration.self, from: JSONEncoder().encode(config))
        #expect(again.modelType == "diffusion_gemma" && again.canvasLength == 128)
        #expect(again.textConfig == config.textConfig)
        #expect(again.base.eosTokenIds?.values == [0, 106])
        #expect(again.videoTokenId == 123 && again.visionConfig?.ropeTheta == 137)
        #expect(again.visionConfig?.hiddenSize == 1152 && again.visionConfig?.numHiddenLayers == 27)
        let precision = try #require(again.base.perLayerQuantization)
        #expect(precision.quantization?.bits == 4 && precision.quantization?.groupSize == 64)
        #expect(
            precision.quantization(layer: "model.encoder.embed_vision.embedding_projection")?.bits
                == 8)
        #expect(
            precision.quantization(layer: "model.encoder.embed_vision.embedding_projection")?
                .groupSize == 32)
        #expect(precision.quantization(layer: "model.decoder.layers.0.experts.down_proj") == nil)
    }

    @Test(arguments: [
        "model_type", "hidden_activation", "attention_bias", "global_head_dim", "rope_parameters",
    ])
    func unsupportedVisionSemanticsFailBeforeConstruction(_ field: String) throws {
        var root = try fixture()
        var vision = root["vision_config"] as! [String: Any]
        let invalid: [String: Any] = [
            "model_type": "other_vision", "hidden_activation": "silu",
            "attention_bias": true, "global_head_dim": 128,
            "rope_parameters": ["rope_type": "linear", "rope_theta": 100],
        ]
        vision[field] = invalid[field]
        root["vision_config"] = vision
        #expect(throws: (any Error).self) { try decode(root) }
    }

    @Test func projectorNormalizesBeforeTheLearnedProjection() throws {
        let vision = try JSONDecoder().decode(
            Gemma4VisionConfig.self, from: Data(#"{"hidden_size":4}"#.utf8))
        let projection = DiffusionGemmaVisionProjection(vision: vision, hiddenSize: 2)
        let weight = MLXArray([Float(1), 2, 3, 4, -2, 1, 0, 3]).reshaped(2, 4)
        try projection.update(
            parameters: ModuleParameters.unflattened(["embedding_projection.weight": weight]),
            verify: .all)
        let input = MLXArray([Float(1), -2, 3, -4]).reshaped(1, 1, 4)
        let expected = matmul(MLXFast.rmsNorm(input, weight: .mlxNone, eps: 1e-6), weight.T)
        let actual = projection(input)
        let incorrectPostNorm = MLXFast.rmsNorm(
            matmul(input, weight.T), weight: .mlxNone, eps: 1e-6)
        eval(actual, expected, incorrectPostNorm)
        #expect(
            actual.asArray(Float.self).map(\.bitPattern)
                == expected.asArray(Float.self).map(\.bitPattern))
        #expect(
            actual.asArray(Float.self).map(\.bitPattern)
                != incorrectPostNorm.asArray(Float.self).map(\.bitPattern))
    }
}
