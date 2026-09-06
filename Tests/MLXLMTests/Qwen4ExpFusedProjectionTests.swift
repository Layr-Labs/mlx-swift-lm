// Copyright © 2026 Eigen Labs.
//
// The fused projections (weight concat at load) must be bit-identical to the
// separate projections they replace: `in_proj_qkv | in_proj_z | in_proj_b |
// in_proj_a` in the gated deltanet, and `input_mix_weight_down |
// block_inject_weight` in the hyper-connection mixer. Both paths are driven
// on the same module with the same input, quantized (as the checkpoint is:
// affine, 4 bits, group 32) and plain, and compared with exact equality --
// not a tolerance -- because stacking rows along the output axis changes no
// arithmetic.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpFusedProjectionTests: XCTestCase {
    /// The quantized matmul and the split are Metal kernels; the library the
    /// ordinary test build links carries them only in a complete metallib
    /// (see `Qwen4ExpForwardParityTests`), so the quantized cases opt in the
    /// same way. The plain-`Linear` cases run everywhere.
    private func requireCompleteMetallib() throws {
        let raw = ProcessInfo.processInfo.environment["MLXLM_FULL_AOT_METALLIB"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard ["1", "true", "yes", "on"].contains(raw ?? "") else {
            throw XCTSkip("Set MLXLM_FULL_AOT_METALLIB=1: the quantized cases need a complete mlx.metallib")
        }
    }

    private func assertExactlyEqual(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        XCTAssertEqual(a.shape, b.shape, "\(label): shape")
        XCTAssertEqual(a.dtype, b.dtype, "\(label): dtype")
        XCTAssertTrue((a .== b).all().item(Bool.self), "\(label): fused output differs from the separate projections")
    }

    private func separateGDN(_ gdn: Qwen4ExpGatedDeltaNet, _ x: MLXArray) -> [MLXArray] {
        [gdn.inProjQKV(x), gdn.inProjZ(x), gdn.inProjB(x), gdn.inProjA(x)]
    }

    private func checkGDN(_ gdn: Qwen4ExpGatedDeltaNet, _ x: MLXArray, _ label: String) {
        let fused = gdn.inProjections(x)
        let parts = separateGDN(gdn, x)
        assertExactlyEqual(fused.qkv, parts[0], "\(label) qkv")
        assertExactlyEqual(fused.z, parts[1], "\(label) z")
        assertExactlyEqual(fused.b, parts[2], "\(label) b")
        assertExactlyEqual(fused.a, parts[3], "\(label) a")
    }

    private func checkMixer(_ mixer: Qwen4ExpGatedResidual, _ hyper: MLXArray, _ label: String) {
        // The public surface: the fused path feeds mixWithInject; the separate
        // path is recomputed here from the module's own parts.
        let (input, residual, inject) = mixer.mixWithInject(hyper)
        let normed = mixer.hcNorm(hyper)
        let hc = Float(mixer.hcCount)
        var w = silu(mixer.mixDown(normed) / hc)
        w = sigmoid(mixer.mixUp(w))
        let lead = w.shape.dropLast()
        let expectedInput =
            (w.reshaped(lead + [mixer.hcCount, mixer.dimensions])
            * normed.reshaped(lead + [mixer.hcCount, mixer.dimensions])).mean(axis: -2)
        let expectedInject = 2 * sigmoid(mixer.blockInject!(normed) / hc)
        assertExactlyEqual(input, expectedInput, "\(label) block input")
        assertExactlyEqual(residual, hyper, "\(label) residual")
        assertExactlyEqual(inject, expectedInject, "\(label) inject")
    }

    /// Plain `Linear` modules are never fused: the dense matmul's tiling
    /// depends on the output width, so a fused dense projection would not be
    /// bit-identical. The module must fall back to the separate projections.
    func testPlainProjectionsKeepTheSeparatePath() throws {
        try requireCompleteMetallib()
        let args = try Qwen4ExpFixture.configuration()
        MLXRandom.seed(11)
        let gdn = Qwen4ExpGatedDeltaNet(args)
        let mixer = Qwen4ExpGatedResidual(args)
        XCTAssertNil(qwen4ExpFuseLinears([gdn.inProjQKV, gdn.inProjZ, gdn.inProjB, gdn.inProjA]))
        XCTAssertNil(qwen4ExpFuseLinears([mixer.mixDown, mixer.blockInject!]))
        let x = MLXRandom.normal([1, 3, args.hiddenSize]).asType(.bfloat16)
        let hyper = MLXRandom.normal([1, 3, args.hcCount * args.hiddenSize]).asType(.bfloat16)
        checkGDN(gdn, x, "plain gdn")
        checkMixer(mixer, hyper, "plain mixer")
    }

    /// Quantize like the checkpoint (affine, 4 bits, group 32) every module
    /// whose input width the group size divides; the tiny fixture's low-rank
    /// `mixUp` (input 8) stays plain, as `quantize(model:)` would leave it.
    private func quantizeLikeTheCheckpoint(_ module: Module) {
        quantize(model: module, groupSize: 32, bits: 4) { _, m in
            guard let linear = m as? Linear else { return false }
            return linear.weight.dim(-1) % 32 == 0
        }
    }

    func testQuantizedProjectionsFuseExactlyAtDecodeWidth() throws {
        try requireCompleteMetallib()
        let args = try Qwen4ExpFixture.configuration()
        MLXRandom.seed(12)
        let gdn = Qwen4ExpGatedDeltaNet(args)
        let mixer = Qwen4ExpGatedResidual(args)
        quantizeLikeTheCheckpoint(gdn)
        quantizeLikeTheCheckpoint(mixer)
        XCTAssertTrue(gdn.inProjQKV is QuantizedLinear, "the fixture must quantize the projections")
        XCTAssertNotNil(qwen4ExpFuseLinears([gdn.inProjQKV, gdn.inProjZ, gdn.inProjB, gdn.inProjA]))
        let x = MLXRandom.normal([1, 1, args.hiddenSize]).asType(.bfloat16)
        let hyper = MLXRandom.normal([1, 1, args.hcCount * args.hiddenSize]).asType(.bfloat16)
        checkGDN(gdn, x, "quantized gdn S=1")
        checkMixer(mixer, hyper, "quantized mixer S=1")
    }

    func testQuantizedProjectionsFuseExactlyAtPrefillWidth() throws {
        try requireCompleteMetallib()
        let args = try Qwen4ExpFixture.configuration()
        MLXRandom.seed(13)
        let gdn = Qwen4ExpGatedDeltaNet(args)
        let mixer = Qwen4ExpGatedResidual(args)
        quantizeLikeTheCheckpoint(gdn)
        quantizeLikeTheCheckpoint(mixer)
        let x = MLXRandom.normal([1, 7, args.hiddenSize]).asType(.bfloat16)
        let hyper = MLXRandom.normal([1, 7, args.hcCount * args.hiddenSize]).asType(.bfloat16)
        checkGDN(gdn, x, "quantized gdn S=7")
        checkMixer(mixer, hyper, "quantized mixer S=7")
    }

    func testMixedModulesFallBackToSeparateProjections() throws {
        // One quantized sibling among plain ones cannot be stacked exactly;
        // the helper must decline rather than approximate.
        let args = try Qwen4ExpFixture.configuration()
        let a = Linear(args.hiddenSize, 16, bias: false)
        let b = QuantizedLinear(Linear(args.hiddenSize, 16, bias: false), groupSize: 32, bits: 4)
        XCTAssertNil(qwen4ExpFuseLinears([a, b]), "mixed plain and quantized")
        XCTAssertNil(qwen4ExpFuseLinears([a, Linear(args.hiddenSize, 8, bias: false)]), "plain modules never fuse")
        let c = QuantizedLinear(Linear(args.hiddenSize, 8, bias: false), groupSize: 32, bits: 4)
        XCTAssertNotNil(qwen4ExpFuseLinears([b, c]), "two quantized siblings fuse")
        let other = QuantizedLinear(Linear(args.hiddenSize, 8, bias: false), groupSize: 32, bits: 8)
        XCTAssertNil(qwen4ExpFuseLinears([b, other]), "differing bits never fuse")
    }
}
