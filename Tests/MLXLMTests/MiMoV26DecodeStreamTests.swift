import Foundation
import MLX
import MLXNN
@testable import MLXLLM
import XCTest

final class MiMoV26DecodeStreamTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["MLX_TEST_MIMO_DECODE_KERNELS"] == "1" else {
            throw XCTSkip("Requires explicitly owned native lane")
        }
    }

    private func normalizer() throws -> RMSNorm {
        let norm = RMSNorm(dimensions: 128, eps: 1e-6)
        try norm.update(parameters: .unflattened([
            ("weight", MLXArray.ones([128], dtype: .bfloat16))]), verify: .all)
        return norm
    }

    func testTaskLocalCPURefusesBothKernelsWhenDefaultDeviceIsGPU() throws {
        try Device.withDefaultDevice(.gpu) {
            try Stream.withNewDefaultStream(device: .cpu) {
                XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
                XCTAssertEqual(MiMoV26DecodeStream.deviceType(of: .default), .cpu)
                let x = MLXArray.ones([1, 1, 128], dtype: .bfloat16)
                let norm = try normalizer()
                XCTAssertNil(MiMoV26DecodeKernels.addRMS(x, x, norm: norm))
                XCTAssertNil(MiMoV26DecodeKernels.combineRMS(
                    x, experts: MLXArray.ones([1, 1, 8, 128], dtype: .bfloat16),
                    weights: MLXArray.ones([1, 1, 8], dtype: .float32), norm: norm))
                XCTAssertNil(MiMoV26DecodeRouter.logits(
                    MLXArray.ones([1, 1, 1024], dtype: .bfloat16),
                    weight: MLXArray.ones([64, 1024], dtype: .bfloat16),
                    operandDType: .bfloat16, enabled: true))
                // The reference still evaluates on the selected CPU stream.
                let expected = norm(x + x)
                eval(expected)
                XCTAssertTrue(expected.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
            }
        }
    }

    func testOrdinaryGPUControlEngagesAndMatchesReference() throws {
        try Device.withDefaultDevice(.gpu) {
            XCTAssertEqual(MiMoV26DecodeStream.deviceType(of: .default), .gpu)
            let x = MLXArray.ones([1, 1, 128], dtype: .bfloat16), norm = try normalizer()
            let result = try XCTUnwrap(MiMoV26DecodeKernels.addRMS(x, x, norm: norm))
            let expected = norm(x + x)
            eval(result.normalized, expected)
            XCTAssertEqual(result.normalized.view(dtype: .uint16).asArray(UInt16.self),
                           expected.view(dtype: .uint16).asArray(UInt16.self))
            let vector = MLXArray.ones([1, 1, 1024], dtype: .bfloat16)
            let matrix = MLXArray.ones([64, 1024], dtype: .bfloat16)
            let logits = try XCTUnwrap(MiMoV26DecodeRouter.logits(
                vector, weight: matrix, operandDType: .bfloat16, enabled: true))
            let reference = matmul(vector.asType(.float32), matrix.asType(.float32).T)
            eval(logits, reference)
            XCTAssertEqual(logits.asArray(Float.self).map(\.bitPattern),
                           reference.asArray(Float.self).map(\.bitPattern))
        }
    }
}
