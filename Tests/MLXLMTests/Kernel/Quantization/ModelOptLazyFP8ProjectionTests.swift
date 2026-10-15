import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

extension KernelTests {
    struct ModelOptLazyFP8ProjectionTests {
        @Test(arguments: [DType.float32, .bfloat16])
        func matchesEagerAndObservesCalibrationUpdates(dtype: DType) throws {
            let layer = try ModelOptCalibratedLinear(weights: ModelOptFP8Weights(
                e4m3Bytes: MLXArray([UInt8(56), 64, 184, 48]).reshaped(2, 2), tensorScale: 0.75),
                inputScale: 0.5, weightDType: dtype)
            let input = MLXArray([Float(0.7), -1.2]).reshaped(1, 2).asType(dtype)
            let budget = try ModelOptLazyFP8Projection.requiredArrayWorkspaceBytes(rows: 1, inputDimensions: 2, outputDimensions: 2)
            for scale: Float in [0.75, 1.5] {
                try layer.update(parameters: .unflattened([("weight_scale", MLXArray(scale))]), verify: .noUnusedKeys)
                let lazy = try layer.lazyFP8Forward(input, admittedArrayWorkspaceBytes: budget)
                let eager = try layer.checkedForward(input)
                eval(lazy.value, lazy.calibrationValid, eager)
                #expect(lazy.calibrationValid.item(Bool.self))
                #expect(abs(lazy.value.asType(.float32) - eager.asType(.float32)).max().item(Float.self) < 1e-5)
            }
            try layer.update(parameters: .unflattened([("input_scale", MLXArray(Float(0)))]), verify: .noUnusedKeys)
            let invalid = try layer.lazyFP8Forward(input, admittedArrayWorkspaceBytes: budget)
            eval(invalid.value, invalid.calibrationValid)
            #expect(!invalid.calibrationValid.item(Bool.self))
            #expect(isNaN(invalid.value).all().item(Bool.self))
        }
        @Test func rejectsInsufficientAndOverflowBudget() throws {
            #expect(throws: ModelOptCalibratedLinear.Failure.self) {
                try ModelOptLazyFP8Projection.requiredArrayWorkspaceBytes(rows: 1, inputDimensions: Int.max, outputDimensions: 2)
            }
            let layer = try ModelOptCalibratedLinear(weights: ModelOptFP8Weights(
                e4m3Bytes: MLXArray([UInt8(56)]).reshaped(1, 1), tensorScale: 1), inputScale: 1, weightDType: .float32)
            #expect(throws: ModelOptCalibratedLinear.Failure.self) {
                try layer.lazyFP8Forward(MLXArray.ones([1, 1]), admittedArrayWorkspaceBytes: 0)
            }
        }
    }
}
