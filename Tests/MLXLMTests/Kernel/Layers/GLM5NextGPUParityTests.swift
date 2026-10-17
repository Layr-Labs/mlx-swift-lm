// Copyright © 2026 Eigen Labs.

import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Complements the independent scalar oracles with actual Metal execution.
    /// These are tiny projected-tensor checks, not full-model qualification.
    @Suite(.serialized)
    struct GLM5NextGPUParityTests {
        private struct LinearResult {
            let values: [Float]
            let recurrent: [Float]
            let convolution: [Float]
        }

        private func linear(device: Device, dtype: DType, chunked: Bool) throws -> LinearResult {
            try Device.withDefaultDevice(device) {
                func tensor(_ values: [Float]) -> MLXArray {
                    MLXArray(values).reshaped([1, 4, 1, 2]).asType(dtype)
                }
                let q = tensor([0.7, 0.2, 0.4, 0.8, 0.5, 0.3, 0.9, 0.6])
                let k = tensor([0.3, 0.8, 0.7, 0.4, 0.9, 0.6, 0.2, 0.5])
                let v = tensor([0.2, -0.4, 0.8, 0.3, -0.5, 0.7, 0.6, -0.2])
                let forget = tensor([0.3, -0.2, 0.5, -0.4, 0.1, -0.3, 0.2, -0.1])
                let gate = tensor([0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.3])
                let beta = MLXArray([Float(0.2), -0.4, 0.6, -0.1])
                    .reshaped([1, 4, 1]).asType(dtype)
                let weights = GLM5NextLinearAttention.Weights(
                    convolution: MLXArray(
                        Array(repeating: [Float(0.1), 0.2, 0.3, 0.4], count: 6)
                            .flatMap { $0 }
                    ).reshaped([6, 4]),
                    aLog: MLXArray([Float(-0.5)]),
                    dtBias: MLXArray([Float(0.1), -0.2]),
                    outputNorm: MLXArray([Float(0.8), 1.2]))
                var state: GLM5NextLinearAttention.State?
                var outputs: [MLXArray] = []
                for range in chunked ? [0 ..< 2, 2 ..< 4] : [0 ..< 4] {
                    let result = try GLM5NextLinearAttention.apply(
                        query: q[0..., range, 0..., 0...],
                        key: k[0..., range, 0..., 0...],
                        value: v[0..., range, 0..., 0...],
                        forgetLogits: forget[0..., range, 0..., 0...],
                        betaLogits: beta[0..., range, 0...],
                        outputGate: gate[0..., range, 0..., 0...],
                        weights: weights, state: state)
                    outputs.append(result.values)
                    state = result.state
                }
                let final = try #require(state)
                return LinearResult(
                    values: concatenated(outputs, axis: 1).asType(.float32).asArray(Float.self),
                    recurrent: final.recurrent.asArray(Float.self),
                    convolution: final.convolution.asType(.float32).asArray(Float.self))
            }
        }

        @Test(arguments: [DType.float32, .float16, .bfloat16])
        func metalKDAAgreesWithCPUAndChunkContinuation(dtype: DType) throws {
            let cpu = try linear(device: .cpu, dtype: dtype, chunked: false)
            let gpu = try linear(device: .gpu, dtype: dtype, chunked: false)
            let chunked = try linear(device: .gpu, dtype: dtype, chunked: true)
            // Reduced-precision outputs permit one activation-scale rounding step;
            // recurrent state remains FP32 and uses the tighter bound in every case.
            let outputTolerance: Float =
                dtype == .float32 ? 2e-5 : dtype == .float16 ? 1 / 1024 : 1 / 128
            close(cpu.values, gpu.values, tolerance: outputTolerance, scaleByMagnitude: true)
            close(gpu.values, chunked.values, tolerance: outputTolerance, scaleByMagnitude: true)
            close(cpu.recurrent, gpu.recurrent, tolerance: 2e-5)
            close(gpu.recurrent, chunked.recurrent, tolerance: 2e-5)
            #expect(cpu.convolution == gpu.convolution)
            #expect(gpu.convolution == chunked.convolution)
        }

        private func sparse(device: Device) throws -> (indices: [Int32], values: [Float]) {
            try Device.withDefaultDevice(device) {
                let keys: [Float] = (0 ..< 12).flatMap { position in
                    position < 4 ? [Float(2), 0] : position < 8 ? [0, 1] : [-1, 0]
                }
                let selection = try GLM5NextSparseAttention.selectTokens(
                    indexQueries: MLXArray([Float(1), 0, 0, 0.1]).reshaped([1, 1, 2, 2]),
                    indexKeys: MLXArray(keys).reshaped([1, 12, 2]),
                    poolGates: MLXArray.zeros([1, 12, 2]),
                    poolPositionBias: MLXArray.zeros([4, 2]),
                    headWeights: MLXArray([Float(1), 0.2]).reshaped([1, 1, 2]),
                    keyValidity: MLXArray.ones([1, 12], dtype: .bool),
                    queryValidity: MLXArray.ones([1, 1], dtype: .bool),
                    queryStart: 11, tokenBudget: 4)
                let queries = MLXArray((0 ..< 512).map { Float($0 % 7 - 3) * 0.01 })
                    .reshaped([1, 1, 2, 256])
                let latents = MLXArray((0 ..< 48).map { Float($0 % 11 - 5) * 0.1 })
                    .reshaped([1, 12, 4])
                let keyWeights = MLXArray((0 ..< 2048).map { Float($0 % 13 - 6) * 0.01 })
                    .reshaped([2, 256, 4])
                let valueWeights = MLXArray((0 ..< 24).map { Float($0 % 5 - 2) * 0.1 })
                    .reshaped([2, 3, 4])
                let values = try GLM5NextSparseAttention.attend(
                    queries: queries, latents: latents, selection: selection,
                    keyProjection: keyWeights, valueProjection: valueWeights)
                return (selection.indices.asArray(Int32.self), values.asArray(Float.self))
            }
        }

        @Test func metalSparseSelectionAndNoPEAttentionAgreeWithCPU() throws {
            let cpu = try sparse(device: .cpu)
            let gpu = try sparse(device: .gpu)
            #expect(cpu.indices == [0, 1, 2, 3, -1, -1, -1])
            #expect(gpu.indices == cpu.indices)
            close(cpu.values, gpu.values, tolerance: 2e-5)
        }

        @Test func comparisonRejectsNonFiniteInputs() {
            for value in [Float.infinity, -.infinity, .nan] {
                #expect(!withinTolerance(1, value, tolerance: 1 / 1024, scaleByMagnitude: true))
                #expect(!withinTolerance(value, 1, tolerance: 1 / 128, scaleByMagnitude: true))
            }
        }

        private func withinTolerance(
            _ lhs: Float, _ rhs: Float, tolerance: Float, scaleByMagnitude: Bool
        ) -> Bool {
            guard lhs.isFinite && rhs.isFinite else { return false }
            let bound = tolerance * (scaleByMagnitude ? max(1, max(abs(lhs), abs(rhs))) : 1)
            return abs(lhs - rhs) <= bound
        }

        private func close(
            _ lhs: [Float], _ rhs: [Float], tolerance: Float, scaleByMagnitude: Bool = false
        ) {
            #expect(lhs.count == rhs.count)
            for (index, pair) in zip(lhs, rhs).enumerated() {
                #expect(
                    withinTolerance(
                        pair.0, pair.1, tolerance: tolerance, scaleByMagnitude: scaleByMagnitude),
                    "element \(index): \(pair.0) versus \(pair.1), tolerance \(tolerance)")
            }
        }
    }
}
