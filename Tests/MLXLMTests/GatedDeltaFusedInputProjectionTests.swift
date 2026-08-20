// Copyright © 2026 Eigen Labs.

import Foundation
import XCTest
@testable import MLX
@testable import MLXNN
@testable import MLXFast

final class GatedDeltaFusedInputProjectionTests: XCTestCase {

    func testFusedInputProjectionParity() {
        let hiddenSize = 2048
        let qkvDim = 8192
        let zDim = 4096
        let bDim = 32
        let aDim = 32
        let totalOut = qkvDim + zDim + bDim + aDim // 12352

        let linQKV = Linear(hiddenSize, qkvDim, bias: false)
        let linZ = Linear(hiddenSize, zDim, bias: false)
        let linB = Linear(hiddenSize, bDim, bias: false)
        let linA = Linear(hiddenSize, aDim, bias: false)

        // Quantize all to QuantizedLinear (4-bit, group 64)
        let qQKV = QuantizedLinear(linQKV, groupSize: 64, bits: 4)
        let qZ = QuantizedLinear(linZ, groupSize: 64, bits: 4)
        let qB = QuantizedLinear(linB, groupSize: 64, bits: 4)
        let qA = QuantizedLinear(linA, groupSize: 64, bits: 4)

        // Concatenate weights along axis 0
        let fusedWeight = concatenated([qQKV.weight, qZ.weight, qB.weight, qA.weight], axis: 0)
        let fusedScales = concatenated([qQKV.scales, qZ.scales, qB.scales, qA.scales], axis: 0)
        let fusedBiases = concatenated([qQKV.biases!, qZ.biases!, qB.biases!, qA.biases!], axis: 0)

        let fusedLayer = QuantizedLinear(
            weight: fusedWeight,
            bias: nil,
            scales: fusedScales,
            biases: fusedBiases,
            groupSize: 64,
            bits: 4
        )

        let S = 512
        let inputs = MLXRandom.normal([1, S, hiddenSize], type: Float.self).asType(Float.self)

        // 1. Evaluate separate
        let outQKV = qQKV(inputs)
        let outZ = qZ(inputs)
        let outB = qB(inputs)
        let outA = qA(inputs)
        eval(outQKV, outZ, outB, outA)

        // 2. Evaluate fused
        let outFused = fusedLayer(inputs)
        let fused_qkv = outFused[0..., 0..., 0 ..< qkvDim]
        let fused_z = outFused[0..., 0..., qkvDim ..< (qkvDim + zDim)]
        let fused_b = outFused[0..., 0..., (qkvDim + zDim) ..< (qkvDim + zDim + bDim)]
        let fused_a = outFused[0..., 0..., (qkvDim + zDim + bDim) ..< totalOut]
        eval(fused_qkv, fused_z, fused_b, fused_a)

        // Assert exact bitwise/numerical identity
        let diffQKV = abs(outQKV - fused_qkv).max().item(Float.self)
        let diffZ = abs(outZ - fused_z).max().item(Float.self)
        let diffB = abs(outB - fused_b).max().item(Float.self)
        let diffA = abs(outA - fused_a).max().item(Float.self)

        print("Max diff QKV:", diffQKV)
        print("Max diff Z:", diffZ)
        print("Max diff B:", diffB)
        print("Max diff A:", diffA)

        XCTAssertEqual(diffQKV, 0.0, accuracy: 1e-5)
        XCTAssertEqual(diffZ, 0.0, accuracy: 1e-5)
        XCTAssertEqual(diffB, 0.0, accuracy: 1e-5)
        XCTAssertEqual(diffA, 0.0, accuracy: 1e-5)
    }
}
