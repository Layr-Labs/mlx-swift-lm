// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXVLM

extension KernelTests {
    @Suite(.serialized)
    struct Qwen35VisionActivationTests {
        // The inspected 9B checkpoint uses the converted `qwen3_5` vision alias.
        // Keep its real geometry here; only the numerical fixture below is reduced.
        private let artifactVisionJSON = """
            {
                "model_type": "qwen3_5",
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 4096,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "deepstack_visual_indexes": []
            }
            """

        private func configuration(
            modelType: String = "qwen3_5", activation: String? = "gelu_pytorch_tanh",
            reduced: Bool = true
        ) throws -> Qwen3VLConfiguration.VisionConfiguration {
            var object = try #require(
                JSONSerialization.jsonObject(with: Data(artifactVisionJSON.utf8)) as? [String: Any])
            object["model_type"] = modelType
            object["hidden_act"] = activation
            if reduced {
                object["depth"] = 1
                object["hidden_size"] = 8
                object["intermediate_size"] = 8
                object["out_hidden_size"] = 8
                object["num_heads"] = 2
            }
            return try JSONDecoder().decode(
                Qwen3VLConfiguration.VisionConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        @Test
        func testMissingOrUnknownActivationPreservesLegacyFastPolicy() throws {
            let declarations: [String?] = [nil, "unrecognized_activation"]
            for modelType in ["qwen3_5", "qwen3_5_vision", "qwen3_5_moe_vision"] {
                for activation in declarations {
                    let config = try configuration(modelType: modelType, activation: activation)
                    if case .fast = config.visionMLPApproximationForConstruction() {
                    } else {
                        Issue.record(
                            Comment(
                                rawValue:
                                    "Unspecified or unsupported activation changed legacy behavior")
                        )
                    }
                }
            }
        }

        @Test
        func testArtifactVisionConfigurationSelectsDeclaredTanhActivation() throws {
            let config = try configuration(reduced: false)
            #expect((config.modelType) == ("qwen3_5"))
            #expect((config.hiddenAct) == ("gelu_pytorch_tanh"))
            #expect((config.depth) == (27))
            #expect((config.hiddenSize) == (1152))
            #expect((config.intermediateSize) == (4304))
            #expect((config.outHiddenSize) == (4096))
            #expect((config.numHeads) == (16))
            #expect((config.numPositionEmbeddings) == (2304))
            #expect((config.patchSize) == (16))
            #expect((config.spatialMergeSize) == (2))
            #expect((config.temporalPatchSize) == (2))
            #expect((config.inChannels) == (3))
            #expect(config.deepstackVisualIndexes.isEmpty)
            if case .tanh = config.visionMLPApproximationForConstruction() {
            } else {
                Issue.record(
                    Comment(rawValue: "The artifact declares tanh GELU for its vision MLP"))
            }
        }

        private let inputRow: [Float] = [-4, -3, -1.25, -0.2, 0, 0.8, 2, 4]

        private func diagonal(_ dimension: Int, scale: Float = 1) -> MLXArray {
            MLXArray(
                (0 ..< dimension * dimension).map {
                    $0 / dimension == $0 % dimension ? scale : Float(0)
                }
            ).reshaped(dimension, dimension)
        }

        private func normalized(_ row: [Float]) -> [Double] {
            let values = row.map(Double.init)
            let mean = values.reduce(0, +) / Double(values.count)
            let variance =
                values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
                / Double(values.count)
            return values.map { ($0 - mean) / sqrt(variance + 1e-6) }
        }

        private func tanhGELU(_ value: Double) -> Double {
            0.5 * value
                * (1
                    + tanh(
                        sqrt(2 / Double.pi)
                            * (value + 0.044715 * value * value * value)))
        }

        private func exactGELU(_ value: Double) -> Double {
            0.5 * value * (1 + erf(value / sqrt(2)))
        }

        private func blockOutput(
            modelType: String, activation: String
        ) throws -> ([Float], [Float], [Double]) {
            let block = Qwen3VLVision.VisionBlock(
                try configuration(modelType: modelType, activation: activation))
            // Zero attention isolates the MLP without bypassing block construction,
            // layer normalization, residuals, or the real attention/block call path.
            try block.update(
                parameters: .unflattened([
                    "norm1.weight": MLXArray.ones([8]),
                    "norm1.bias": MLXArray.zeros([8]),
                    "norm2.weight": MLXArray.ones([8]),
                    "norm2.bias": MLXArray.zeros([8]),
                    "attn.qkv.weight": MLXArray.zeros([24, 8]),
                    "attn.qkv.bias": MLXArray.zeros([24]),
                    "attn.proj.weight": MLXArray.zeros([8, 8]),
                    "attn.proj.bias": MLXArray.zeros([8]),
                    "mlp.linear_fc1.weight": diagonal(8, scale: 1.75),
                    "mlp.linear_fc1.bias": MLXArray.zeros([8]),
                    "mlp.linear_fc2.weight": diagonal(8, scale: 0.5),
                    "mlp.linear_fc2.bias": MLXArray.zeros([8]),
                ]), verify: [.all])
            let rows = [inputRow, Array(inputRow.reversed())]
            let input = rows.flatMap { $0 }
            let preactivations = rows.flatMap { normalized($0).map { 1.75 * $0 } }
            let output = block(
                MLXArray(input).reshaped(2, 8),
                cuSeqlens: MLXArray([Int32(0), 2]),
                rotaryPosEmb: MLXArray.zeros([2, 2]))
            eval(output)
            #expect((output.shape) == ([2, 8]))
            return (output.asArray(Float.self), input, preactivations)
        }

        @Test
        func testQwen35VisionBlockMatchesDeclaredTanhMath() throws {
            // Converted dense artifact alias plus official dense/MoE vision types.
            for modelType in ["qwen3_5", "qwen3_5_vision", "qwen3_5_moe_vision"] {
                let (actual, input, preactivations) = try blockOutput(
                    modelType: modelType, activation: "gelu_pytorch_tanh")
                let expected = zip(input, preactivations).map {
                    Float(Double($0.0) + 0.5 * tanhGELU($0.1))
                }
                for (got, want) in zip(actual, expected) {
                    #expect(abs((got) - (want)) <= 2e-5, Comment(rawValue: modelType))
                }
                // The scalar fast policy is x * sigmoid(1.702*x). These inputs
                // distinguish it from tanh by much more than the assertion tolerance.
                let legacy = zip(input, preactivations).map {
                    Float(Double($0.0) + 0.5 * $0.1 / (1 + exp(-1.702 * $0.1)))
                }
                #expect((zip(expected, legacy).map { abs($0 - $1) }.max() ?? 0) > (0.003))
            }
        }

        @Test
        func testQwen35VisionBlockHonorsExplicitExactGELU() throws {
            let (actual, input, preactivations) = try blockOutput(
                modelType: "qwen3_5", activation: "gelu")
            for ((got, residual), preactivation) in zip(zip(actual, input), preactivations) {
                #expect(
                    abs((got) - (Float(Double(residual) + 0.5 * exactGELU(preactivation)))) <= 2e-5)
            }
        }

        @Test
        func testPatchMergerRetainsExactGELUMath() throws {
            let config = try configuration()
            // Exercise both normal and deep-stack merger normalization paths.
            for postShuffle in [false, true] {
                let merger = Qwen3VLVision.PatchMerger(
                    config: config, usePostShuffleNorm: postShuffle)
                let normDimension = postShuffle ? 32 : 8
                let outputProjection = MLXArray(
                    (0 ..< 8 * 32).map { index -> Float in
                        index / 32 == index % 32 ? 1 : 0
                    }
                ).reshaped(8, 32)
                try merger.update(
                    parameters: .unflattened([
                        "norm.weight": MLXArray.ones([normDimension]),
                        "norm.bias": MLXArray.zeros([normDimension]),
                        "linear_fc1.weight": diagonal(32, scale: 1.75),
                        "linear_fc1.bias": MLXArray.zeros([32]),
                        "linear_fc2.weight": outputProjection,
                        "linear_fc2.bias": MLXArray.zeros([8]),
                    ]), verify: [.all])
                let input = Array(repeating: inputRow, count: 4).flatMap { $0 }
                let output = merger(MLXArray(input).reshaped(4, 8))
                eval(output)
                #expect((output.shape) == ([1, 8]))
                let preactivations = normalized(inputRow).map { 1.75 * $0 }
                let expected = preactivations.map { Float(exactGELU($0)) }
                for (got, want) in zip(output.asArray(Float.self), expected) {
                    #expect(abs((got) - (want)) <= 2e-5)
                }
                #expect(
                    (zip(expected, preactivations.map { Float(tanhGELU($0)) })
                        .map { abs($0 - $1) }.max() ?? 0) > (1e-4))
                if case .none = merger.activation.approximation {
                } else {
                    Issue.record(Comment(rawValue: "The patch merger must retain exact GELU"))
                }
            }
        }
    }
}
