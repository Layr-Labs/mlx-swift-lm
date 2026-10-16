// Copyright 2026 Eigen Labs. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    @Suite
    struct GLM5NextPackedLoadingTests {
        @Test(arguments: [4, 5, 6])
        func packedProjectionMatchesExistingQuantizer(bits: Int) throws {
            for dtype: DType in [.float16, .bfloat16, .float32] {
                let bias = fixed([2, 32], dtype: dtype)
                let reference = QuantizedSwitchLinear(
                    SwitchLinear(
                        inputDims: 64, outputDims: 32, numExperts: 2,
                        weight: fixed([2, 32, 64], dtype: dtype), bias: bias),
                    groupSize: 32, bits: bits)
                let offsets = try #require(reference.biases)
                let loaded = try QuantizedSwitchLinear(
                    inputDims: 64, outputDims: 32, numExperts: 2,
                    weight: reference.weight, scales: reference.scales, biases: offsets,
                    bias: bias, groupSize: 32, bits: bits)

                #expect(loaded.weight === reference.weight)
                #expect(loaded.scales === reference.scales)
                #expect(loaded.biases === offsets)
                #expect(loaded.bias === bias)
                #expect(loaded.shape == (2, 32, 64))
                #expect(loaded.weight.shape == [2, 32, 2 * bits])
                #expect(loaded.mode == .affine && loaded.bits == bits && loaded.groupSize == 32)
                #expect(loaded.trainableParameters().flattened().isEmpty)
                #expect(
                    loaded.parameters().flattened().map(\.0).sorted()
                        == ["bias", "biases", "scales", "weight"])

                let input = fixed([2, 1, 1, 64], dtype: dtype)
                let indices = MLXArray([UInt32(1), 0, 0, 1]).reshaped([2, 2])
                // Both paths must use the existing gather arithmetic, including bias.
                expectEqual(loaded(input, indices), reference(input, indices))
            }
        }

        @Test(arguments: [32, 64, 128])
        func supportedGroupsRetainPackedStorage(groupSize: Int) throws {
            let reference = QuantizedSwitchLinear(
                SwitchLinear(
                    inputDims: 128, outputDims: 8, numExperts: 2,
                    weight: fixed([2, 8, 128])), groupSize: groupSize, bits: 4)
            let loaded = try QuantizedSwitchLinear(
                inputDims: 128, outputDims: 8, numExperts: 2,
                weight: reference.weight, scales: reference.scales,
                biases: try #require(reference.biases), groupSize: groupSize, bits: 4)
            #expect(loaded.weight === reference.weight)
            #expect(loaded.bias == nil)
            #expect(loaded.scales.shape == [2, 8, 128 / groupSize])
        }

        @Test func invalidPackedGeometryAndPoliciesThrowBeforeConstructingBanks() throws {
            let fixture = PackedFixture()
            for dimension in [0, -1] {
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidDimensions) {
                    try fixture.load(input: dimension)
                }
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidDimensions) {
                    try fixture.load(output: dimension)
                }
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidDimensions) {
                    try fixture.load(experts: dimension)
                }
            }
            for bits in [0, 1, 7, 9, Int.max] {
                #expect(throws: QuantizedSwitchLinear.LoadError.unsupportedBits(bits)) {
                    try fixture.load(bits: bits)
                }
            }
            for group in [0, 16, 256, Int.max] {
                #expect(throws: QuantizedSwitchLinear.LoadError.unsupportedGroupSize(group)) {
                    try fixture.load(group: group)
                }
            }
            #expect(throws: QuantizedSwitchLinear.LoadError.nonDivisibleInput) {
                try fixture.load(input: 63)
            }
            #expect(throws: QuantizedSwitchLinear.LoadError.dimensionOverflow) {
                try fixture.load(input: Int.max - 31)
            }
            #expect(throws: QuantizedSwitchLinear.LoadError.dimensionOverflow) {
                try fixture.load(output: Int.max, experts: 2)
            }
            #expect(throws: QuantizedSwitchLinear.LoadError.dimensionOverflow) {
                try fixture.load(output: Int(Int32.max) + 1)
            }
            #expect(throws: QuantizedSwitchLinear.LoadError.dimensionOverflow) {
                try fixture.load(input: Int(Int32.max) - 31, output: Int(Int32.max), experts: 3)
            }
        }

        @Test func invalidPackedTensorsAndLinearBiasAreRejected() throws {
            let fixture = PackedFixture()
            for weight in [
                MLXArray.zeros([2, 32, 8]),
                MLXArray.zeros([2, 32, 9], dtype: .uint32),
                MLXArray.zeros([64, 8], dtype: .uint32),
                MLXArray.zeros([1, 32, 8], dtype: .uint32),
            ] {
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidWeight) {
                    try fixture.load(weight: weight)
                }
            }
            for scales in [
                MLXArray.zeros([2, 32, 2], dtype: .uint32),
                MLXArray.zeros([2, 32, 1]), MLXArray.zeros([64, 2]),
            ] {
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidScales) {
                    try fixture.load(scales: scales)
                }
            }
            for offsets in [
                MLXArray.zeros([2, 32, 2], dtype: .uint32),
                MLXArray.zeros([2, 32, 2], dtype: .float16),
                MLXArray.zeros([2, 32, 1]), MLXArray.zeros([64, 2]),
            ] {
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidOffsets) {
                    try fixture.load(offsets: offsets)
                }
            }
            for bias in [
                fixture.offsets, MLXArray.zeros([32]),
                MLXArray.zeros([2, 32], dtype: .uint32),
            ] {
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidBias) {
                    try fixture.load(bias: bias)
                }
            }
        }

        @Test func loadedMLPRetainsMixedPrecisionComponentsAndClamping() throws {
            let gate = packedLinear(input: 64, output: 32, bits: 4)
            let up = packedLinear(input: 64, output: 32, bits: 5)
            let down = packedLinear(input: 32, output: 64, bits: 6)
            let model = try GLM5NextMLP(
                hiddenSize: 64, intermediateSize: 32,
                gateProj: gate, upProj: up, downProj: down, swigluLimit: 0.5)
            #expect(model.gateProj === gate && model.upProj === up && model.downProj === down)
            let input = fixed([2, 64])
            let expected = down(
                silu(clip(gate(input), max: 0.5)) * clip(up(input), min: -0.5, max: 0.5))
            expectEqual(model(input), expected)
            #expect(model.parameters().flattened().count == 9)
        }

        @Test func loadedExpertsRetainMixedPackedBanksAndContributionOrder() throws {
            let gate = try packedExpert(input: 64, output: 32, bits: 4)
            let up = try packedExpert(input: 64, output: 32, bits: 5)
            let down = try packedExpert(input: 32, output: 64, bits: 6)
            let model = try GLM5NextExperts(
                hiddenSize: 64, intermediateSize: 32, numExperts: 2,
                gateProj: gate, upProj: up, downProj: down, swigluLimit: 0.5)
            #expect(model.gateProj === gate && model.upProj === up && model.downProj === down)
            #expect(model.parameters().flattened().count == 9)
            #expect(model.trainableParameters().flattened().isEmpty)
            let input = fixed([2, 64]).asType(.bfloat16)
            let sorted = MLXArray([UInt32(0), 1, 0, 1]).reshaped([2, 2])
            let weights = MLXArray([Float(0.25), 0.75, 0.625, 0.375]).reshaped([2, 2])
            let expanded = input.expandedDimensions(axes: [-2, -3])
            let projected = down(
                silu(clip(gate(expanded, sorted), max: 0.5))
                    * clip(up(expanded, sorted), min: -0.5, max: 0.5), sorted
            ).squeezed(axis: -2)
            let contributions = (projected.asType(.float32) * weights[.ellipsis, .newAxis])
                .asType(input.dtype)
            let expected = contributions[0..., 0, 0...] + contributions[0..., 1, 0...]
            let reversed = MLXArray([UInt32(1), 0, 1, 0]).reshaped([2, 2])
            expectEqual(
                model(input, indices: reversed, weights: weights[0..., .stride(by: -1)]),
                expected)
        }

        @Test func loadedRouterRetainsWeightsAndExistingRoutingSemantics() throws {
            let weight = MLXArray([Float(2), 0, 0, 0, -2, 0, -3, 0]).reshaped([4, 2])
                .asType(.float16)
            let bias = MLXArray([Float(0), 0, 1, 0]).asType(.bfloat16)
            let loaded = try GLM5NextRouter(
                hiddenSize: 2, numExperts: 4, topK: 2, weight: weight, correctionBias: bias)
            #expect(loaded.weight === weight && loaded.correctionBias === bias)
            #expect(loaded.trainableParameters().flattened().map(\.0) == ["weight"])
            let reference = GLM5NextRouter(hiddenSize: 2, numExperts: 4, topK: 2)
            try reference.update(
                parameters: .unflattened(["weight": weight, "e_score_correction_bias": bias]),
                verify: .all)
            let input = MLXArray([Float(1), 0, -1, 0]).reshaped([2, 2]).asType(.float16)
            let actual = loaded(input)
            let expected = reference(input)
            expectEqual(actual.logits, expected.logits)
            expectEqual(actual.indices, expected.indices)
            expectEqual(actual.weights, expected.weights)
        }

        @Test func loadedComponentsRejectIncorrectGeometryAndBias() throws {
            let gate = Linear(weight: fixed([32, 64]))
            let down = Linear(weight: fixed([64, 32]))
            #expect(throws: GLM5NextComponentLoadingError.invalidProjection("up_proj")) {
                try GLM5NextMLP(
                    hiddenSize: 64, intermediateSize: 32,
                    gateProj: gate, upProj: down, downProj: down)
            }
            let biased = Linear(weight: gate.weight, bias: fixed([32]))
            #expect(throws: GLM5NextComponentLoadingError.invalidProjection("gate_proj")) {
                try GLM5NextMLP(
                    hiddenSize: 64, intermediateSize: 32,
                    gateProj: biased, upProj: gate, downProj: down)
            }
            let expert = try packedExpert(input: 64, output: 32, bits: 4)
            let expertDown = try packedExpert(input: 32, output: 64, bits: 4)
            #expect(throws: GLM5NextComponentLoadingError.invalidProjection("gate_proj")) {
                try GLM5NextExperts(
                    hiddenSize: 64, intermediateSize: 32, numExperts: 3,
                    gateProj: expert, upProj: expert, downProj: expertDown)
            }
            #expect(throws: GLM5NextComponentLoadingError.invalidConfiguration) {
                try GLM5NextExperts(
                    hiddenSize: 64, intermediateSize: 32, numExperts: 2,
                    gateProj: expert, upProj: expert, downProj: expertDown, swigluLimit: .nan)
            }
            #expect(throws: GLM5NextComponentLoadingError.invalidRouterWeights) {
                try GLM5NextRouter(
                    hiddenSize: 2, numExperts: 4, topK: 2,
                    weight: fixed([4, 3]), correctionBias: fixed([4]))
            }
            #expect(throws: GLM5NextComponentLoadingError.invalidConfiguration) {
                try GLM5NextRouter(
                    hiddenSize: 2, numExperts: 4, topK: 3,
                    weight: fixed([4, 2]), correctionBias: fixed([4]), nGroup: 2, topkGroup: 1)
            }
        }

        private func fixed(_ shape: [Int], dtype: DType = .float32) -> MLXArray {
            let values = (0 ..< shape.reduce(1, *)).map { (index: Int) -> Float in
                Float((index * 7) % 29 - 14) / 128
            }
            return MLXArray(values).reshaped(shape).asType(dtype)
        }

        private func packedLinear(input: Int, output: Int, bits: Int) -> QuantizedLinear {
            let (weight, scales, offsets) = quantized(
                fixed([output, input]), groupSize: 32, bits: bits)
            return QuantizedLinear(
                weight: weight, scales: scales, biases: offsets, groupSize: 32, bits: bits)
        }

        private func packedExpert(input: Int, output: Int, bits: Int) throws
            -> QuantizedSwitchLinear
        {
            let (weight, scales, offsets) = quantized(
                fixed([2, output, input]), groupSize: 32, bits: bits)
            return try QuantizedSwitchLinear(
                inputDims: input, outputDims: output, numExperts: 2,
                weight: weight, scales: scales, biases: try #require(offsets),
                groupSize: 32, bits: bits)
        }

        private func expectEqual(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
            #expect((actual .== expected).all().item(Bool.self))
        }

        private struct PackedFixture {
            let weight = MLXArray.zeros([2, 32, 8], dtype: .uint32)
            let scales = MLXArray.ones([2, 32, 2])
            let offsets = MLXArray.zeros([2, 32, 2])

            func load(
                input: Int = 64, output: Int = 32, experts: Int = 2,
                weight: MLXArray? = nil, scales: MLXArray? = nil, offsets: MLXArray? = nil,
                bias: MLXArray? = nil, group: Int = 32, bits: Int = 4
            ) throws -> QuantizedSwitchLinear {
                try QuantizedSwitchLinear(
                    inputDims: input, outputDims: output, numExperts: experts,
                    weight: weight ?? self.weight, scales: scales ?? self.scales,
                    biases: offsets ?? self.offsets, bias: bias, groupSize: group, bits: bits)
            }
        }
    }
}
