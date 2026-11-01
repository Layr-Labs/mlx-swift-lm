import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

extension KernelTests {
    struct ModelOptCalibratedLinearTests {
        @Test func fp8ScalarEquationAndStrictParameterReload() throws {
            // E4M3 bytes 56,64,184,48 represent 1,2,-1,0.5.
            // x/inputScale is exactly representable, so calibrated x = x.
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray([UInt8(56), 64, 184, 48]).reshaped(2, 2), tensorScale: 0.5)
            let layer = try ModelOptCalibratedLinear(
                weights: weights, inputScale: 0.25, weightDType: .float32)
            let input = MLXArray([Float(1), 2, -1, 1]).reshaped(1, 2, 2)
            let actual = try layer.checkedForward(input)
            let expected = MLXArray([Float(2.5), 0, 0.5, 0.75]).reshaped(1, 2, 2)
            #expect(arrayEqual(actual, expected).item(Bool.self))
            var parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
            #expect(Set(parameters.keys) == Set(["weight", "weight_scale", "input_scale"]))
            parameters["weight_scale"] = MLXArray(Float(1))
            try layer.update(parameters: .unflattened(parameters), verify: .all)
            #expect(arrayEqual(try layer.checkedForward(input), expected * 2).item(Bool.self))
            #expect(quantizeSingle(layer: layer) == nil)
        }

        @Test func nvfp4PackedParametersAndHigherRankForward() throws {
            let weights = try ModelOptNVFP4Weights(
                packedBytes: MLXArray([UInt8](repeating: 0x22, count: 8)).reshaped(1, 8),
                e4m3ScaleBytes: MLXArray([UInt8(56)]).reshaped(1, 1), tensorScale: 0.5)
            let layer = try ModelOptCalibratedLinear(
                weights: weights, inputScale: 1, weightDType: .float32)
            // Each E2M1 weight nibble is 1; block scale 1, global multiplier .5.
            let input = MLXArray.ones([2, 1, 16]) * 6
            let actual = try layer.checkedForward(input)
            #expect(actual.shape == [2, 1, 1])
            #expect(abs(actual - 48).max().item(Float.self) < 0.0001)
            let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
            #expect(
                Set(parameters.keys)
                    == Set(["weight", "weight_scale", "weight_scale_2", "input_scale"]))
            #expect(parameters["weight"]?.dtype == .uint32)
            try layer.update(parameters: .unflattened(parameters), verify: .all)
            #expect(arrayEqual(actual, try layer.checkedForward(input)).item(Bool.self))
            #expect(quantizeSingle(layer: layer) == nil)
        }

        @Test func originalWeightDTypeIsPreservedAndMixedInputRejected() throws {
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray([UInt8(56)]).reshaped(1, 1), tensorScale: 0.5)
            let layer = try ModelOptCalibratedLinear(
                weights: weights, inputScale: 1, weightDType: .bfloat16)
            let output = try layer.checkedForward(MLXArray([Float(2)]).asType(.bfloat16))
            #expect(output.dtype == .bfloat16 && output.shape == [1])
            #expect(output.item(Float.self) == 1)
            #expect(throws: (any Error).self) { try layer.checkedForward(MLXArray([Float(2)])) }
            #expect(throws: (any Error).self) {
                try ModelOptCalibratedLinear(weights: weights, inputScale: 0, weightDType: .float32)
            }
            #expect(throws: (any Error).self) {
                try ModelOptCalibratedLinear(weights: weights, inputScale: 1, weightDType: .uint8)
            }
        }

        @Test func malformedUpdatedCalibrationAndWorkspaceFailBeforeCompute() throws {
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray([UInt8(56)]).reshaped(1, 1), tensorScale: 1)
            let layer = try ModelOptCalibratedLinear(
                weights: weights, inputScale: 1, weightDType: .float32)
            #expect(throws: (any Error).self) {
                try layer.checkedForward(MLXArray.ones([8193, 1]))
            }
            #expect(throws: (any Error).self) { try layer.checkedForward(MLXArray.ones([1, 2])) }
            // The update applies the parameters and reports the invalid calibration.
            for invalid in [MLXArray(Float.nan), MLXArray(UInt8(1))] {
                #expect(throws: ModelOptCalibratedLinear.Failure.invalidMetadata) {
                    try layer.update(
                        parameters: .unflattened(["input_scale": invalid]), verify: .none)
                }
                #expect(!layer.graphCalibrationValid)
                #expect(throws: (any Error).self) {
                    try layer.checkedForward(MLXArray.ones([1, 1]))
                }
            }
        }

        @Test func largeFP8ProjectionUsesBoundedTilesAndPreservesRows() throws {
            // 2.1M elements exceed the previous whole-matrix workspace guard.
            // Independent scalar oracle: E4M3(1) * .5 weight, input 1 => width/2.
            let width = 2048
            let outputs = 1030
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray([UInt8](repeating: 56, count: outputs * width)).reshaped(
                    outputs, width),
                tensorScale: 0.5)
            let layer = try ModelOptCalibratedLinear(
                weights: weights, inputScale: 1, weightDType: .float32)
            let actual = try layer.checkedForward(MLXArray.ones([1, width]))
            #expect(actual.shape == [1, outputs])
            #expect(
                arrayEqual(actual, MLXArray.ones([1, outputs]) * Float(width / 2)).item(Bool.self))
            // Live output reservation is bounded independently of weight tiling.
            #expect(throws: (any Error).self) {
                try layer.checkedForward(MLXArray.ones([2048, width]))
            }
        }

    }
}
