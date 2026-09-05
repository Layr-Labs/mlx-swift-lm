import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM
import MLXLMCommon

@Suite("GPTOSS on-demand expert dequantization", .serialized)
struct GPTOSSDequantizedPrefillTests {
    @Test("production gate excludes decode and non-GPTOSS geometry")
    func gate() {
        #expect(GPTOSSDequantizedPrefill.eligible(
            inputDims: 2880, hiddenDims: 2880, experts: 32, assignments: 2048, sequenceLength: 512, sorted: true))
        #expect(!GPTOSSDequantizedPrefill.eligible(
            inputDims: 2880, hiddenDims: 2880, experts: 32, assignments: 32, sequenceLength: 1, sorted: true))
        #expect(!GPTOSSDequantizedPrefill.eligible(
            inputDims: 2880, hiddenDims: 2880, experts: 32, assignments: 2048, sequenceLength: 512, sorted: false))
        #expect(!GPTOSSDequantizedPrefill.eligible(
            inputDims: 2880, hiddenDims: 2880, experts: 128, assignments: 8192, sequenceLength: 2048, sorted: true))
    }

    @Test("tiny MXFP4 retains FP32/BF16 output dtype, bias, and finite parity")
    func tiny() throws {
        for dtype in [DType.float32, .bfloat16] {
            try compare(experts: 4, inputDims: 64, outputDims: 64,
                        assignments: 64, dtype: dtype)
        }
    }

    @Test("actual GPTOSS projection shape", .enabled(if:
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_DEQUANT_SHAPE_TEST"] == "1"))
    func actualShape() throws {
        // One projection at a time; peak telemetry includes expanded weights.
        for outputDims in [2880, 5760] {
            try compare(experts: 32, inputDims: 2880, outputDims: outputDims,
                        assignments: 2048, dtype: .float32)
        }
    }

    private func compare(experts: Int, inputDims: Int, outputDims: Int,
                         assignments: Int, dtype: DType) throws {
        MLXRandom.seed(0xDEC0DE)
        let projection: QuantizedSwitchLinear
        do {
            let plain = SwitchLinear(inputDims: inputDims, outputDims: outputDims,
                                      numExperts: experts, bias: true)
            projection = QuantizedSwitchLinear(plain, groupSize: 32, bits: 4, mode: .mxfp4)
            eval(projection)
        }
        // Use a nonzero bias to detect omission or wrong gather/broadcast.
        var biasValues: [Float] = []
        biasValues.reserveCapacity(experts * outputDims)
        for expert in 0..<experts {
            for column in 0..<outputDims {
                biasValues.append(Float(expert + 1) * 0.125 + Float(column % 17) * 0.015625)
            }
        }
        let bias = MLXArray(biasValues).reshaped(experts, outputDims).asType(dtype)
        projection.update(parameters: ModuleParameters.unflattened(["bias": bias]))
        eval(projection)
        let x = (MLXRandom.normal([assignments, 1, inputDims]) * 0.25).asType(dtype)
        let perExpert = assignments / experts
        var expertIDs: [UInt32] = []
        expertIDs.reserveCapacity(assignments)
        for index in 0..<assignments {
            expertIDs.append(UInt32(min(experts - 1, index / perExpert)))
        }
        let indices = MLXArray(expertIDs)
        eval(x, indices)
        let baseline = projection(x, indices, sortedIndices: true)
        eval(baseline)
        let activeBefore = Memory.activeMemory
        Memory.peakMemory = 0
        let candidate = try #require(GPTOSSDequantizedPrefill.project(
            projection, x, indices, sortedIndices: true))
        eval(candidate)
        let peak = Memory.peakMemory
        #expect(candidate.dtype == baseline.dtype)
        #expect(candidate.shape == baseline.shape)
        let error = abs(candidate.asType(.float32) - baseline.asType(.float32)).max().item(Float.self)
        let tolerance: Float = dtype == .float32 ? 1e-4 : 0.015625
        #expect(error.isFinite && error <= tolerance)
        #expect(GPTOSSDequantizedPrefill.project(projection, x, indices, sortedIndices: false) == nil)
        print("[gptoss-dequant-shape] E=\(experts) K=\(inputDims) N=\(outputDims) A=\(assignments) dtype=\(dtype) maxDiff=\(error) peakBytes=\(peak) activeBeforeBytes=\(activeBefore) additionalPeakBytes=\(max(0, peak - activeBefore))")
    }
}
