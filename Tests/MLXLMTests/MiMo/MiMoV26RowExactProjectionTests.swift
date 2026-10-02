// Copyright © 2026 Eigen Labs.
// Native-lane test: requires MLX_TEST_MIMO_DECODE_KERNELS=1 and a GPU stream.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class MiMoV26RowExactProjectionEligibilityTests: XCTestCase {
    func testOnlyShortAlignedAffineEightBitRowsAreEligible() {
        func supported(
            _ shape: [Int], _ weight: [Int], dtype: DType = .bfloat16, bits: Int = 8,
            group: Int = 64, mode: QuantizationMode = .affine, device: DeviceType = .gpu
        ) -> Bool {
            MiMoV26RowExactProjection.supports(
                shape: shape, dtype: dtype, weightShape: weight, weightDType: .uint32,
                scalesDType: dtype, biasesDType: dtype, groupSize: group, bits: bits,
                mode: mode, device: device)
        }
        for rows in 2 ... 7 { XCTAssertTrue(supported([1, rows, 4096], [12288, 1024])) }
        XCTAssertFalse(supported([1, 1, 4096], [12288, 1024]), "one row keeps stock qmv_fast")
        XCTAssertFalse(supported([1, 8, 4096], [12288, 1024]))
        XCTAssertFalse(supported([2, 2, 4096], [12288, 1024]))
        XCTAssertFalse(supported([1, 2, 4000], [12288, 1000]))
        XCTAssertFalse(supported([1, 2, 4096], [12284, 1024]))
        XCTAssertFalse(supported([1, 2, 4096], [12288, 512], bits: 4))
        XCTAssertFalse(supported([1, 2, 4096], [12288, 1024], group: 32))
        XCTAssertFalse(supported([1, 2, 4096], [12288, 1024], mode: .mxfp4))
        XCTAssertFalse(supported([1, 2, 4096], [12288, 1024], dtype: .float32))
        XCTAssertFalse(supported([1, 2, 4096], [12288, 1024], device: .cpu))
    }
}

final class MiMoV26RowExactProjectionTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["MLX_TEST_MIMO_DECODE_KERNELS"] == "1" else {
            throw XCTSkip("Requires explicit native-lane run: MLX_TEST_MIMO_DECODE_KERNELS=1")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The Metal dispatch cannot be qualified by a CPU fallback")
        }
    }

    private func layer(_ n: Int, _ k: Int, bias: Bool, seed: UInt64) -> QuantizedLinear {
        let key = MLXRandom.key(seed)
        let weight = MLXRandom.normal([n, k], key: key).asType(.bfloat16) * 0.02
        let offsets =
            bias ? MLXRandom.normal([n], key: MLXRandom.key(seed + 1)).asType(.bfloat16) : nil
        let result = QuantizedLinear(weight: weight, bias: offsets, groupSize: 64, bits: 8)
        eval(result)
        return result
    }

    private func bits(_ a: MLXArray) -> [UInt16] {
        a.view(dtype: .uint16).asArray(UInt16.self)
    }

    /// Every MiMo V2.6 affine 8-bit projection geometry: q/k/v/o, the dense
    /// layer-0 MLP and the readout head.
    func testEveryRowMatchesTheOneRowStockProjectionBitForBit() throws {
        let shapes: [(n: Int, k: Int)] = [
            (12288, 4096), (768, 4096), (512, 4096), (4096, 8192),
            (16384, 4096), (4096, 16384), (152_576, 4096),
        ]
        for (index, shape) in shapes.enumerated() {
            let projection = layer(shape.n, shape.k, bias: index == 1, seed: UInt64(17 + index))
            for rows in 2 ... 7 {
                let x =
                    (MLXRandom.normal(
                        [1, rows, shape.k], key: MLXRandom.key(UInt64(rows * 31 + index)))
                    * 3).asType(.bfloat16)
                eval(x)
                let actual = try XCTUnwrap(MiMoV26RowExactProjection.apply(projection, x))
                XCTAssertEqual(actual.shape, [1, rows, shape.n])
                XCTAssertEqual(actual.dtype, .bfloat16)
                for r in 0 ..< rows {
                    let expected = projection(x[0..., r ..< (r + 1), 0...].contiguous())
                    XCTAssertEqual(
                        bits(actual[0..., r ..< (r + 1), 0...]), bits(expected),
                        "n=\(shape.n) k=\(shape.k) rows=\(rows) row=\(r)")
                }
            }
        }
    }
}
