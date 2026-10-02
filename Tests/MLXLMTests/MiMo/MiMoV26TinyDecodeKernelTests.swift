import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The short-forward residual and RMS norm kernels of MiMoV26DecodeKernels.
/// They are ordinary SIMD Metal kernels with no NAX or M5-specific dispatch.
/// Each result is compared with the same arithmetic in MLX operations.
final class MiMoV26TinyDecodeKernelTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint
    /// BF16 storage. One rounding step near 1.0 is 2^-7 = 0.0078; the kernel
    /// and the reference round the normalized value at different points, so
    /// a few steps are allowed.
    private let tolerance: Float = 0.05

    private func values(_ name: String, _ shape: [Int], scale: Float = 1) -> MLXArray {
        MLXArray(Fixture.floats(name, count: shape.reduce(1, *)).map { $0 * 20 * scale }, shape)
            .asType(.bfloat16)
    }
    private func difference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    private func norm(width: Int = 64) throws -> RMSNorm {
        let norm = RMSNorm(dimensions: width, eps: 1e-5)
        try norm.update(
            parameters: .unflattened(["weight": values("norm-scale", [width], scale: 0.5) + 1]),
            verify: .all)
        return norm
    }

    func testSupportedShapesAreOneRowSmallAndHalfPrecisionOnGPU() {
        let supports = MiMoV26DecodeKernels.supports(shape:dtype:device:)
        XCTAssertTrue(supports([1, 1, 64], .bfloat16, .gpu))
        XCTAssertTrue(supports([1, 7, 4096], .float16, .gpu))
        XCTAssertTrue(supports([1, 3, 4], .bfloat16, .gpu))
        XCTAssertFalse(supports([2, 1, 64], .bfloat16, .gpu))
        XCTAssertFalse(supports([1, 8, 64], .bfloat16, .gpu))
        XCTAssertFalse(supports([1, 0, 64], .bfloat16, .gpu))
        XCTAssertFalse(supports([1, 1, 66], .bfloat16, .gpu))
        XCTAssertFalse(supports([1, 1, 4100], .bfloat16, .gpu))
        XCTAssertFalse(supports([1, 64], .bfloat16, .gpu))
        XCTAssertFalse(supports([1, 1, 64], .float32, .gpu))
        XCTAssertFalse(supports([1, 1, 64], .bfloat16, .cpu))
        XCTAssertFalse(supports([1, 1, 64], .bfloat16, nil))
        XCTAssertEqual(MiMoV26DecodeStream.deviceType(of: .gpu), .gpu)
        XCTAssertEqual(MiMoV26DecodeStream.deviceType(of: .cpu), .cpu)
    }

    func testAddRMSMatchesAdditionThenNorm() throws {
        let norm = try self.norm()
        let x = values("x", [1, 3, 64])
        let y = values("y", [1, 3, 64])
        let result = try XCTUnwrap(MiMoV26DecodeKernels.addRMS(x, y, norm: norm))
        let residual = x + y
        let normalized = norm(residual)
        eval(result.residual, result.normalized, residual, normalized)
        XCTAssertEqual(result.residual.dtype, .bfloat16)
        XCTAssertEqual(result.normalized.shape, [1, 3, 64])
        // The sum is one BF16 addition on both sides.
        XCTAssertLessThanOrEqual(difference(result.residual, residual), tolerance)
        XCTAssertLessThanOrEqual(difference(result.normalized, normalized), tolerance)
        XCTAssertTrue(result.normalized.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        // Unsupported inputs return nil and build no kernel.
        XCTAssertNil(
            MiMoV26DecodeKernels.addRMS(x.asType(.float32), y.asType(.float32), norm: norm))
        XCTAssertNil(MiMoV26DecodeKernels.addRMS(x, y[0..., ..<2, 0...], norm: norm))
        XCTAssertNil(MiMoV26DecodeKernels.addRMS(x, y.asType(.float16), norm: norm))
        XCTAssertNil(MiMoV26DecodeKernels.addRMS(x, y, norm: try self.norm(width: 32)))
        let cpu = try Stream.withNewDefaultStream(device: .cpu) {
            MiMoV26DecodeKernels.addRMS(x, y, norm: norm)
        }
        XCTAssertNil(cpu)
    }

    func testCombineRMSKeepsFP32WeightingThenRoundsOnce() throws {
        let norm = try self.norm()
        let h = values("h", [1, 2, 64])
        let experts = values("experts", [1, 2, 3, 64])
        let weights = MLXArray([Float(0.5), 0.3, 0.2, 0.1, 0.6, 0.3], [1, 2, 3])
        let result = try XCTUnwrap(
            MiMoV26DecodeKernels.combineRMS(h, experts: experts, weights: weights, norm: norm))
        let output = (experts.asType(.float32) * weights[.ellipsis, .newAxis]).sum(axis: -2)
            .asType(.bfloat16)
        let residual = h + output
        let normalized = norm(residual)
        eval(result.residual, result.normalized, residual, normalized)
        // The expert sum is FP32 on both sides, then rounded to BF16 once.
        XCTAssertLessThanOrEqual(difference(result.residual, residual), tolerance)
        XCTAssertLessThanOrEqual(difference(result.normalized, normalized), tolerance)
        XCTAssertNil(
            MiMoV26DecodeKernels.combineRMS(
                h, experts: experts, weights: weights.asType(.bfloat16), norm: norm))
        XCTAssertNil(
            MiMoV26DecodeKernels.combineRMS(
                h, experts: experts[0..., 0..., ..<2, 0...], weights: weights, norm: norm))
        XCTAssertNil(
            MiMoV26DecodeKernels.combineRMS(
                h, experts: experts.asType(.float16), weights: weights, norm: norm))
    }

    func testFinishLayerMatchesTheOriginalDenseAndMoELayerArithmetic() throws {
        let f = try Fixture.mediaModels(dtype: "bfloat16")
        let model = f.target.model
        for index in 0 ..< 2 {
            let layer = model.layers[index]
            let hidden = values("hidden-\(index)", [1, 3, 64])
            let attention = values("attention-\(index)", [1, 3, 64], scale: 0.5)
            XCTAssertNil(
                MiMoV26DecodeKernels.finishLayer(
                    hidden, attentionOutput: attention, layer: layer, nextNorm: model.norm,
                    enabled: false))
            let result = try XCTUnwrap(
                MiMoV26DecodeKernels.finishLayer(
                    hidden, attentionOutput: attention, layer: layer, nextNorm: model.norm,
                    enabled: true))
            let post = hidden + attention
            let residual = post + layer.mlp(layer.postAttentionNorm(post))
            let normalized = model.norm(residual)
            eval(result.residual, result.normalized, residual, normalized)
            XCTAssertEqual(result.residual.shape, [1, 3, 64])
            XCTAssertLessThanOrEqual(difference(result.residual, residual), tolerance, "\(index)")
            XCTAssertLessThanOrEqual(
                difference(result.normalized, normalized), tolerance, "\(index)")
            // A float32 activation is outside the kernel contract.
            XCTAssertNil(
                MiMoV26DecodeKernels.finishLayer(
                    hidden.asType(.float32), attentionOutput: attention.asType(.float32),
                    layer: layer, nextNorm: model.norm, enabled: true))
        }
    }
}
