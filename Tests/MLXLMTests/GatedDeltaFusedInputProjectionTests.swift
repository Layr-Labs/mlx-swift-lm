// Copyright © 2026 Eigen Labs.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLX
@testable import MLXFast
@testable import MLXLLM
@testable import MLXNN

final class GatedDeltaFusedInputProjectionTests: XCTestCase {

    func testFusedInputProjectionParity() {
        let hiddenSize = 2048
        let qkvDim = 8192
        let zDim = 4096
        let bDim = 32
        let aDim = 32

        MLXRandom.seed(0)
        let separate = [qkvDim, zDim, bDim, aDim].map {
            QuantizedLinear(Linear(hiddenSize, $0, bias: false), groupSize: 64, bits: 4)
        }
        let fusedWeight = concatenated(separate.map(\.weight), axis: 0)
        let fusedScales = concatenated(separate.map(\.scales), axis: 0)
        let fusedBiases = concatenated(separate.map { $0.biases! }, axis: 0)
        let fusedLayer = QuantizedLinear(
            weight: fusedWeight,
            bias: nil,
            scales: fusedScales,
            biases: fusedBiases,
            groupSize: 64,
            bits: 4
        )

        // One decode token: the fused and the separate layers both use the
        // quantized matrix-vector kernel, which reduces each output column
        // in the same order for every N. The outputs are bit-identical.
        let token = MLXRandom.normal([1, 1, hiddenSize], type: Float.self)
        let decodeDiff = abs(
            concatenated(separate.map { $0(token) }, axis: -1) - fusedLayer(token)
        ).max().item(Float.self)
        XCTAssertEqual(decodeDiff, 0)

        // A prefill chunk: MLX can select different matrix-matrix kernels
        // for the same column. The b and a projections (N = 32) use the
        // float32 split-K kernel. The wide fused layer uses the NAX kernel
        // when the GPU has neural accelerators, and that kernel multiplies
        // float32 inputs in TF32 (MLX_ENABLE_TF32 is on by default). Thus
        // the outputs agree only to TF32 precision. TF32 rounds x and w with
        // unit roundoff u = 2^-11, and float32 accumulates K products with
        // unit roundoff 2^-24. Each output is within
        // (2u + K * 2^-24) * sum_k |x_k * w_k| of the exact product, so two
        // outputs differ by at most twice that bound.
        let chunk = MLXRandom.normal([1, 512, hiddenSize], type: Float.self)
        let separateOut = concatenated(separate.map { $0(chunk) }, axis: -1)
        let fusedOut = fusedLayer(chunk)
        let dequantizedWeight = dequantized(
            fusedWeight, scales: fusedScales, biases: fusedBiases,
            groupSize: 64, bits: 4)
        let unitRoundoffTF32: Float = 0x1p-11
        let unitRoundoffFloat32: Float = 0x1p-24
        let relativeBound =
            2 * (2 * unitRoundoffTF32 + Float(hiddenSize) * unitRoundoffFloat32)
        let bound = relativeBound * matmul(abs(chunk), abs(dequantizedWeight).T)
        let excess = (abs(separateOut - fusedOut) - bound).max().item(Float.self)
        XCTAssertLessThanOrEqual(excess, 0)
    }
}

private func smallQwenGDNConfiguration() throws -> Qwen35TextConfiguration {
    let json = """
        {
          "hidden_size": 64,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "linear_num_value_heads": 2,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "vocab_size": 128,
          "full_attention_interval": 4
        }
        """
    return try JSONDecoder().decode(
        Qwen35TextConfiguration.self, from: Data(json.utf8))
}

extension GatedDeltaFusedInputProjectionTests {
    func testFusionKeepsCheckpointTopologyAndFrozenCache() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])

        XCTAssertTrue(layer.prepareFusedInputProjection())
        XCTAssertTrue(layer.hasFusedInputProjection)
        XCTAssertNotNil(layer.inProjQKV)
        XCTAssertNotNil(layer.inProjZ)
        XCTAssertNotNil(layer.inProjB)
        XCTAssertNotNil(layer.inProjA)
        let keys = Set(layer.parameters().flattened().map(\.0))
        XCTAssertFalse(keys.contains("in_proj_fused.weight"))
        XCTAssertTrue(keys.contains("in_proj_qkv.weight"))
        XCTAssertFalse(
            layer.trainableParameters().flattened().contains {
                $0.0.hasPrefix("in_proj_fused")
            })
    }

    func testHeterogeneousQuantizationRetainsSeparateProjections() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 8)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])

        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertNotNil(layer.inProjQKV)
        XCTAssertNotNil(layer.inProjB)
    }

    func testAdapterBackedProjectionRetainsSeparateCalls() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        let adapted =
            LoRALinear.from(
                linear: layer.inProjQKV, rank: 4, scale: 1) as! Linear
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(adapted),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])

        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(layer.inProjQKV is LoRALinear)
    }
    func testBatchedMoEInputsFlattenTokensAndPreserveTopK() {
        let x = MLXArray.zeros([2, 7, 2048], dtype: .bfloat16)
        let indices = MLXArray.zeros([2, 7, 8], dtype: .uint32)
        let scores = MLXArray.zeros([2, 7, 8], dtype: .bfloat16)
        let flattened = qwen35FlattenMoEInputs(
            x: x, indices: indices, scores: scores)
        XCTAssertEqual(flattened.x.shape, [14, 2048])
        XCTAssertEqual(flattened.indices.shape, [14, 8])
        XCTAssertEqual(flattened.scores.shape, [14, 8])
    }

    func testUnfrozenQuantizedProjectionsRetainDifferentiablePath() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        let modules = ModuleChildren(values: [
            "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
            "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
            "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
            "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
        ])
        try layer.update(modules: modules, verify: [])
        layer.unfreeze(recursive: true)
        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
    }

    func testReplacingSourceProjectionInvalidatesInferenceCache() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])
        XCTAssertTrue(layer.prepareFusedInputProjection())
        XCTAssertTrue(layer.hasFusedInputProjection)

        let adapted =
            LoRALinear.from(
                linear: layer.inProjQKV, rank: 4, scale: 1) as! Linear
        try layer.update(
            modules: ModuleChildren(values: ["in_proj_qkv": .value(adapted)]),
            verify: [])
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(
            ObjectIdentifier(type(of: layer.inProjQKV))
                == ObjectIdentifier(QuantizedLinear.self))
    }

    func testReplacingSourceParametersInvalidatesInferenceCache() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])
        XCTAssertTrue(layer.prepareFusedInputProjection())
        let replacement =
            layer.inProjQKV.weight
            + MLXArray.zeros(
                layer.inProjQKV.weight.shape, dtype: layer.inProjQKV.weight.dtype)
        try layer.update(
            parameters: ModuleParameters.unflattened([
                "in_proj_qkv.weight": replacement
            ]), verify: [])
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(layer.prepareFusedInputProjection())
        XCTAssertTrue(layer.hasFusedInputProjection)
    }

}
