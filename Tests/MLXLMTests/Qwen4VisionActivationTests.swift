import Foundation
import MLX
import MLXNN
import MLXLLM
import XCTest
@testable import MLXVLM

/// Fusion's Qwen4 vision bridge inherits pinned mlx-vlm 78b96eb5462141447b9a6b4943ef553891da56dd,
/// mlx_vlm/models/qwen3_vl/vision.py: MLP uses nn.GELU(approx="tanh").
/// This is a reference-correctness change, not parity with the former fast approximation.
final class Qwen4VisionActivationTests: XCTestCase {
    func testVLMRootLMHeadMatchesTextWrapperNamespace() throws {
        var text = Qwen4ExpTextConfiguration()
        text.hiddenLayers = 0; text.hiddenSize = 8; text.vocabularySize = 16
        text.pleLayerIds = []; text.numExperts = 0; text.tieWordEmbeddings = false
        let model = MLXVLM.Qwen4Exp(Qwen4ExpVLMConfiguration(
            text: Qwen4ExpConfiguration(textConfig: text)))
        let weights = ["lm_head.weight": MLXArray.zeros([16, 8]),
                       "lm_head.scales": MLXArray.ones([16, 1]),
                       "lm_head.biases": MLXArray.zeros([16, 1])]
        let sanitized = model.sanitize(weights: weights)
        for suffix in ["weight", "scales", "biases"] {
            XCTAssertNil(sanitized["lm_head.\(suffix)"])
            XCTAssertEqual(sanitized["language_model.lm_head.\(suffix)"]?.shape,
                           weights["lm_head.\(suffix)"]?.shape)
        }
        let twice = model.sanitize(weights: sanitized)
        XCTAssertEqual(Set(twice.keys), Set(sanitized.keys))
    }

    private func configuration(_ modelType: String) throws -> Qwen3VLConfiguration.VisionConfiguration {
        let object: [String: Any] = ["model_type":modelType,"depth":1,"hidden_size":8,
            "intermediate_size":12,"out_hidden_size":8,"num_heads":2,"patch_size":2,
            "spatial_merge_size":2,"temporal_patch_size":2,"num_position_embeddings":4,
            "hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[]]
        return try JSONDecoder().decode(Qwen3VLConfiguration.VisionConfiguration.self,
            from: JSONSerialization.data(withJSONObject: object))
    }

    func testOnlyNativeQwen4JoinsExistingTanhPolicy() throws {
        for name in ["qwen4_exp", "qwen3_vl_moe"] {
            if case .tanh = try configuration(name).visionMLPApproximationForConstruction() {}
            else { XCTFail("Expected the reference tanh policy for \(name)") }
        }
        for name in ["qwen3_vl", "qwen3_5", "qwen3_5_moe_vision", "other"] {
            if case .fast = try configuration(name).visionMLPApproximationForConstruction() {}
            else { XCTFail("Unrelated model policy changed for \(name)") }
        }
    }

    func testNativeActivationMatchesTanhReferenceAndDiffersFromLegacyFast() throws {
        let values: [Float] = [-4, -3, -1.25, -0.2, 0, 0.8, 2, 4]
        let input = MLXArray(values)
        let actual = GELU(approximation: try configuration("qwen4_exp").visionMLPApproximationForConstruction())(input)
        let legacy = GELU(approximation: .fast)(input)
        eval(actual, legacy)
        let reference = values.map { value -> Float in
            let x = Double(value)
            return Float(0.5 * x * (1 + tanh(sqrt(2 / Double.pi) * (x + 0.044715 * x * x * x))))
        }
        for (got, expected) in zip(actual.asArray(Float.self), reference) {
            XCTAssertEqual(got, expected, accuracy: 1e-5)
        }
        XCTAssertGreaterThan(zip(actual.asArray(Float.self), legacy.asArray(Float.self))
            .map { abs($0 - $1) }.max() ?? 0, 0.005)
    }
}
