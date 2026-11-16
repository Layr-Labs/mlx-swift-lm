// Copyright © 2026 Eigen Labs / Darkbloom.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

extension KernelTests {
    /// `ResidualRMSNorm` writes the bits of MLX's unfused add and RMSNorm on
    /// the rows where the two ways of rounding the norm's mean differ (built
    /// by `RMSNormMeanForm.Calibration.rows`), in the single-row and looped
    /// kernels, and on random rows with a residual.
    @Suite(.serialized)
    struct ResidualRMSNormMeanFormTests {
        private static let eps: Float = 1e-6

        private var verified: Bool {
            Device.defaultDevice().deviceType == .gpu
                && ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family())
        }

        private func bytes(_ x: MLXArray) -> Data {
            eval(x)
            return x.asData(access: .copy).data
        }

        /// Qwen 3.5/3.6 A3B (2048), Gemma 4 26B (2816), Qwen 3.5 9B (4096, the
        /// forms coincide), Qwen 3.8 27B (5120, MLX's looped kernel), 7168.
        static let widths = [2048, 2816, 4096, 5120, 7168]

        @Test func meanFormIsMeasuredForLoopedWidths() throws {
            guard verified else {
                #expect(RMSNormMeanForm.measured(width: 5120, dtype: .bfloat16) == nil)
                return
            }
            for dtype in [DType.bfloat16, .float16, .float32] {
                for width in [5120, 7168] {
                    let m = try #require(
                        RMSNormMeanForm.Calibration.measurement(width: width, dtype: dtype))
                    #expect(m.formsDiffer, "width \(width) \(dtype)")
                    #expect(m.divideMatches != m.fusedMultiplyAddMatches, "width \(width) \(dtype)")
                    #expect(RMSNormMeanForm.measured(width: width, dtype: dtype) != nil)
                }
                for width in Self.widths {
                    // The weight product's order (MLX's float32 looped kernel
                    // reassociates its loop-invariant inverse).
                    #expect(
                        RMSNormMeanForm.measuredProduct(width: width, dtype: dtype) != nil,
                        "width \(width) \(dtype)")
                }
            }
        }

        @Test func separatingRowsKeepTheUnfusedBits() throws {
            guard verified else { return }
            for dtype in [DType.bfloat16, .float16, .float32] {
                for width in Self.widths {
                    let coincide = width & (width - 1) == 0
                    let (separating, plain) = RMSNormMeanForm.Calibration.rows(
                        width: width, dtype: dtype, separate: !coincide)
                    let all = separating + plain
                    let x = MLXArray(all.flatMap { $0 }, [all.count, width]).asType(dtype)
                    let weight = (MLXRandom.normal([width], key: MLXRandom.key(UInt64(width))) / 4 + 1)
                        .asType(dtype)
                    // A zero residual: the sum is the row itself.
                    let zero = MLXArray.zeros([all.count, width], dtype: dtype)
                    let fused = try #require(
                        ResidualRMSNorm.apply(x, zero, weight: weight, eps: Self.eps))
                    let sum = x + zero
                    #expect(bytes(fused.sum) == bytes(sum), "width \(width) \(dtype)")
                    #expect(
                        bytes(fused.norm) == bytes(MLXFast.rmsNorm(sum, weight: weight, eps: Self.eps)),
                        "width \(width) \(dtype)")
                }
            }
        }

        @Test func randomRowsWithAResidualKeepTheUnfusedBits() throws {
            guard verified else { return }
            for dtype in [DType.bfloat16, .float16, .float32] {
                for width in Self.widths {
                    for rows in [1, 3, 64] {
                        let seed = UInt64(width * 7 + rows)
                        let x = (MLXRandom.normal([rows, width], key: MLXRandom.key(seed)) * 3)
                            .asType(dtype)
                        let residual = MLXRandom.normal([rows, width], key: MLXRandom.key(seed + 1))
                        let weight = (MLXRandom.normal([width], key: MLXRandom.key(seed + 2)) / 4 + 1)
                            .asType(dtype)
                        for r in [residual.asType(dtype), residual] {
                            let fused = try #require(
                                ResidualRMSNorm.apply(x, r, weight: weight, eps: Self.eps))
                            let sum = x + r.asType(dtype)
                            #expect(bytes(fused.sum) == bytes(sum))
                            #expect(
                                bytes(fused.norm)
                                    == bytes(MLXFast.rmsNorm(sum, weight: weight, eps: Self.eps)),
                                "width \(width) \(dtype) rows \(rows) residual \(r.dtype)")
                        }
                    }
                }
            }
        }
    }
}
