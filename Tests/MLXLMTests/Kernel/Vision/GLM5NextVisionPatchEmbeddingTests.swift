// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXVLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextVisionPatchEmbeddingTests {
        private static var devices: [DeviceType] {
            #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
                [.cpu, .gpu]
            #else
                [.cpu]
            #endif
        }

        private func makeEmbedding(
            configuration: GLM5NextVisionConfiguration, weight: MLXArray, bias: MLXArray
        ) throws -> GLM5NextVisionPatchEmbedding {
            let embedding = GLM5NextVisionPatchEmbedding(configuration)
            try embedding.update(
                parameters: ModuleParameters.unflattened([
                    ("proj.weight", weight.transposed(0, 2, 3, 4, 1)),
                    ("proj.bias", bias),
                ]), verify: .all)
            return embedding
        }

        @Test(
            .enabled(
                if: GLM5NextReferenceFixture.enabled,
                "Requires the unchanged canonical official CPU oracle"))
        func pinnedPatchProjectionBeforeVisionNormalization() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let embedding = try makeEmbedding(
                    configuration: fixture.configuration.visionConfig,
                    weight: fixture.weights["model.visual.patch_embed.proj.weight"]!,
                    bias: fixture.weights["model.visual.patch_embed.proj.bias"]!)
                for name in ["image", "video"] {
                    let test = try fixture.caseObject(name)
                    let patches = try fixture.tensor(fixture.key(test, "pixel_values"))
                    try MLX.withError { error in
                        let output = embedding(patches)
                        eval(output)
                        try error.check()
                        // A new focused assertion, not a change to any frozen
                        // expected tensor or existing end-to-end tolerance.
                        try fixture.compare(
                            output,
                            key: name + ".model.visual.patch_embed.call0.output",
                            absolute: 2e-6, relative: 0)
                    }
                }
            }
        }

        @Test(arguments: devices)
        func sourceOrderBiasDtypesAndIndependentPatchBatches(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) {
                let object: [String: Any] = [
                    "model_type": "glm5_next_vision", "depth": 1, "hidden_size": 16,
                    "intermediate_size": 32, "num_heads": 2, "image_size": 28,
                    "patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2,
                    "in_channels": 3, "out_hidden_size": 16,
                    "projection_intermediate_size": 32, "hidden_act": "silu",
                    "swiglu_limit": 10.0, "rms_norm_eps": 1e-5,
                    "attention_bias": true, "attention_dropout": 0.0,
                    "initializer_range": 0.02,
                ]
                let configuration = try JSONDecoder().decode(
                    GLM5NextVisionConfiguration.self,
                    from: JSONSerialization.data(withJSONObject: object))
                let rows = 7
                let width = 3 * 2 * 14 * 14
                let hidden = 16
                let pixels = (0 ..< (rows * width)).map { (index: Int) -> Float in
                    let pattern: Int = (index * 11 + 3) % 23 - 11
                    return Float(pattern) / 32
                }
                let weights = (0 ..< (hidden * width)).map { (index: Int) -> Float in
                    let pattern: Int = (index * 7 + 5) % 15 - 7
                    return Float(pattern) / 64
                }
                let biases = (0 ..< hidden).map { (index: Int) -> Float in Float(index - 8) / 32 }
                // Dyadic inputs make this independent Float64 contraction
                // exactly representable in FP32 before the activation cast.
                let expected = (0 ..< (rows * hidden)).map { index -> Float in
                    let row = index / hidden
                    let output = index % hidden
                    var value = Double(biases[output])
                    for column in 0 ..< width {
                        value +=
                            Double(pixels[row * width + column])
                            * Double(weights[output * width + column])
                    }
                    return Float(value)
                }
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    let embedding = try makeEmbedding(
                        configuration: configuration,
                        weight: MLXArray(weights, [hidden, 3, 2, 14, 14]).asType(dtype),
                        bias: MLXArray(biases).asType(dtype))
                    let input = MLXArray(pixels, [rows, width]).asType(dtype)
                    let strided = stacked([input, input], axis: -1)
                        .reshaped(rows, 2 * width)[0..., .stride(by: 2)]
                    try MLX.withError { error in
                        let output = embedding(strided)
                        let separate = concatenated(
                            (0 ..< rows).map { embedding(strided[$0 ..< ($0 + 1)]) }, axis: 0)
                        let reference = MLXArray(expected, [rows, hidden]).asType(dtype)
                        eval(output, separate, reference)
                        StreamOrDevice.default.stream.synchronize()
                        try error.check()
                        #expect(output.shape == [rows, hidden] && output.dtype == dtype)
                        #expect(
                            output.asType(.float32).asArray(Float.self)
                                == reference.asType(.float32).asArray(Float.self))
                        #expect(
                            output.asType(.float32).asArray(Float.self)
                                == separate.asType(.float32).asArray(Float.self))
                    }
                }
            }
        }

        @Test(
            .enabled(
                if: GLM5NextReferenceFixture.enabled,
                "Requires the unchanged canonical official CPU oracle"))
        func pinnedVisionRMSNormalizationBeforeProjection() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                for name in ["image", "video"] {
                    for layer in 0 ..< 2 {
                        for normName in ["norm1", "norm2", "attn.q_norm", "attn.k_norm"] {
                            let path = "model.visual.blocks.\(layer).\(normName)"
                            let input = try fixture.tensor(name + "." + path + ".call0.input.0")
                            let output = try fixture.tensor(name + "." + path + ".call0.output")
                            let weight = fixture.weights[path + ".weight"]!
                            let norm = GLM5NextVisionRMSNorm(
                                dimensions: input.dim(-1),
                                eps: Float(fixture.configuration.visionConfig.rmsNormEps))
                            try norm.update(
                                parameters: ModuleParameters.unflattened([("weight", weight)]),
                                verify: .all)
                            try MLX.withError { error in
                                let actual = norm(input)
                                eval(actual, output)
                                try error.check()
                                try fixture.compare(
                                    actual, expected: output,
                                    label: name + "." + path, absolute: 0, relative: 0)
                            }
                        }
                    }
                }
            }
        }

        @Test(
            .enabled(
                if: GLM5NextReferenceFixture.enabled,
                "Requires the unchanged canonical official CPU oracle"))
        func pinnedMergeProjectionMatchesSourceLayout() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                for name in ["image", "video"] {
                    let input = try fixture.tensor(name + ".model.visual.downsample.call0.input.0")
                    let weight = fixture.weights["model.visual.downsample.weight"]!
                    let bias = fixture.weights["model.visual.downsample.bias"]!
                    let patches = input.transposed(0, 2, 3, 1).reshaped(-1, input.dim(1))
                    for groups in [1, 2, 7, 15, 16, 31] {
                        try MLX.withError { error in
                            let parts = stride(from: 0, to: input.dim(0), by: groups).map { start in
                                GLM5NextVisionTower.downsamplePatches(
                                    patches[
                                        (4 * start) ..< (4 * min(start + groups, input.dim(0)))],
                                    weight: weight.transposed(0, 2, 3, 1), bias: bias, merge: 2)
                            }
                            let result = concatenated(parts, axis: 0)
                            let reference = try fixture.tensor(
                                name + ".model.visual.downsample.call0.output"
                            )
                            .reshaped(result.shape)
                            eval(result, reference)
                            try error.check()
                            try fixture.compare(
                                result, expected: reference, label: name + ".downsample",
                                absolute: 0, relative: 0)
                        }
                    }
                }
            }
        }

        @Test(arguments: devices)
        func mergedProjectionDtypesStridesAndFrameIsolation(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) {
                let channels = 4
                let outputs = 8
                let weights = (0 ..< (outputs * channels * 4)).map { Float($0 % 11 - 5) / 32 }
                let bias = (0 ..< outputs).map { Float($0 - 4) / 32 }
                for groups in [1, 7, 15, 16, 17] {
                    let values = (0 ..< (groups * 4 * channels)).map { Float($0 % 13 - 6) / 16 }
                    let expected = (0 ..< (groups * outputs)).map { index -> Float in
                        let group = index / outputs
                        let output = index % outputs
                        var value = Double(bias[output])
                        for channel in 0 ..< channels {
                            for position in 0 ..< 4 {
                                value +=
                                    Double(values[(group * 4 + position) * channels + channel])
                                    * Double(
                                        weights[output * channels * 4 + channel * 4 + position])
                            }
                        }
                        return Float(value)
                    }
                    for dtype: DType in [.float32, .float16, .bfloat16] {
                        try MLX.withError { error in
                            let source = MLXArray(values, [groups * 4, channels]).asType(dtype)
                            let strided = stacked([source, source], axis: -1)
                                .reshaped(groups * 4, channels * 2)[0..., .stride(by: 2)]
                            let matrix = MLXArray(weights, [outputs, channels, 2, 2])
                                .transposed(0, 2, 3, 1).asType(dtype)
                            let bias = MLXArray(bias).asType(dtype)
                            let result = GLM5NextVisionTower.downsamplePatches(
                                strided, weight: matrix, bias: bias, merge: 2)
                            let pieces = (0 ..< groups).map { group in
                                GLM5NextVisionTower.downsamplePatches(
                                    strided[(4 * group) ..< (4 * group + 4)],
                                    weight: matrix, bias: bias, merge: 2)
                            }
                            let joined = concatenated(pieces, axis: 0)
                            let reference = MLXArray(expected, [groups, outputs]).asType(dtype)
                            eval(result, joined, reference)
                            StreamOrDevice.default.stream.synchronize()
                            try error.check()
                            #expect(result.shape == [groups, outputs] && result.dtype == dtype)
                            #expect(
                                result.asType(.float32).asArray(Float.self)
                                    == reference.asType(.float32).asArray(Float.self))
                            #expect(
                                result.asType(.float32).asArray(Float.self)
                                    == joined.asType(.float32).asArray(Float.self))
                        }
                    }
                }
            }
        }

    }
}
