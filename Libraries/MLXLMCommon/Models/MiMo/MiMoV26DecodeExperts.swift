// SPDX-License-Identifier: Apache-2.0
// Adapted from jundot/omlx #3990 moe_decode.py at
// e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb.
// Swift adaptation Copyright © 2026 Eigen Labs.
// See docs/mimo-v26/DECODE-EXPERT-ATTRIBUTION.md.

import Cmlx
import Foundation
import MLX

/// Paired gate/up/SwiGLU and down projection. Sharing is per distinct expert
/// and token row; every routed slot is written, including duplicate slots.
enum MiMoV26DecodeExperts {
    static let requested =
        ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_DECODE_EXPERTS"] == "1"

    struct Matrix {
        let weight: MLXArray
        let scales: MLXArray
    }

    struct Result {
        let activation: MLXArray
        let output: MLXArray
    }

    static func gpuStream(_ stream: StreamOrDevice) -> Bool {
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        var type = MLX_CPU
        return mlx_stream_get_device(&device, stream.ctx) == 0
            && mlx_device_get_type(&type, device) == 0 && type == MLX_GPU
    }

    static func supports(rows: Int, hidden: Int, intermediate: Int, experts: Int,
                         topK: Int, dtype: DType) -> Bool {
        (1...7).contains(rows) && topK == 8 && (8...256).contains(experts)
            && (512...4096).contains(hidden) && hidden % 512 == 0
            && (512...4096).contains(intermediate) && intermediate % 512 == 0
            && (dtype == .bfloat16 || dtype == .float16)
    }

    private static func matrix(_ layer: SwitchLinear) -> Matrix? {
        guard let quantized = layer as? QuantizedSwitchLinear,
              ObjectIdentifier(type(of: quantized)) == ObjectIdentifier(QuantizedSwitchLinear.self),
              quantized.mode == .mxfp4, quantized.bits == 4, quantized.groupSize == 32,
              quantized.bias == nil, quantized.biases == nil else { return nil }
        return Matrix(weight: quantized.weight, scales: quantized.scales)
    }

    /// SwitchLayers calls this before its normal projection.
    /// No dtype conversion, CPU synchronization, tensor cache or layout memo.
    static func tryProject(_ x: MLXArray, indices: MLXArray, glu: SwitchGLU,
                           enabled: Bool) -> Result? {
        guard enabled, glu.weightedReductionProfile == .mimoV26FP32,
              ObjectIdentifier(type(of: glu)) == ObjectIdentifier(SwitchGLU.self),
              glu.activationProduct != nil, glu.isSiluActivation,
              MLXHardwareInfo.isCompiledDecodeSupported,
              let down = matrix(glu.downProj) else { return nil }
        if let fused = glu.gateUpProj {
            guard let gateUp = matrix(fused) else { return nil }
            return project(x, indices: indices, gate: gateUp, up: gateUp, down: down,
                           upOffset: glu.hiddenDims)
        }
        guard let gate = glu.gateProj, let up = glu.upProj,
              let gateMatrix = matrix(gate), let upMatrix = matrix(up) else { return nil }
        return project(x, indices: indices, gate: gateMatrix, up: upMatrix, down: down)
    }

    /// Internal raw packed-matrix seam also permits bounded synthetic fixtures.
    /// Production calls reach it only through the stock MiMo module gate above.
    static func project(_ x: MLXArray, indices: MLXArray, gate: Matrix, up: Matrix,
                        down: Matrix, upOffset: Int = 0) -> Result? {
        let stream = StreamOrDevice.default
        guard gpuStream(stream), x.ndim == 3, x.dim(0) == 1,
              indices.ndim == 3, indices.shape == [1, x.dim(1), 8],
              indices.dtype == .uint32, down.weight.ndim == 3,
              down.scales.ndim == 3 else { return nil }
        let rows = x.dim(1), hidden = x.dim(2), intermediate = down.weight.dim(2) * 8
        let count = down.weight.dim(0)
        guard supports(rows: rows, hidden: hidden, intermediate: intermediate,
                       experts: count, topK: 8, dtype: x.dtype),
              down.weight.shape == [count, hidden, intermediate / 8],
              down.scales.shape == [count, hidden, intermediate / 32],
              upOffset == 0 || upOffset == intermediate else { return nil }
        let gateRows = upOffset == 0 ? intermediate : 2 * intermediate
        guard gate.weight.shape == [count, gateRows, hidden / 8],
              up.weight.shape == [count, gateRows, hidden / 8],
              gate.scales.shape == [count, gateRows, hidden / 32],
              up.scales.shape == [count, gateRows, hidden / 32],
              [gate, up, down].allSatisfy({ $0.weight.dtype == .uint32 && $0.scales.dtype == .uint8 })
        else { return nil }
        let pairs = rows * 8
        let activation = gateUpKernel(
            [x, indices, gate.weight, gate.scales, up.weight, up.scales],
            template: [("T", x.dtype), ("KDIM", hidden), ("NOUT", intermediate),
                       ("GROWS", gateRows), ("UROWS", gateRows), ("UOFF", upOffset),
                       ("TOPK", 8), ("NPAIRS", pairs), ("MAXDUP", rows), ("NSG", 2), ("RPS", 4)],
            grid: (32 * pairs, 2 * (intermediate / 8), 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, rows, 8, intermediate]], outputDTypes: [x.dtype], stream: stream)[0]
        let output = downKernel(
            [activation, indices, down.weight, down.scales],
            template: [("T", x.dtype), ("KDIM", intermediate), ("NOUT", hidden),
                       ("TOPK", 8), ("NPAIRS", pairs), ("MAXDUP", rows), ("NSG", 2), ("RPS", 4)],
            grid: (32 * pairs, 2 * (hidden / 8), 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, rows, 8, hidden]], outputDTypes: [x.dtype], stream: stream)[0]
        return Result(activation: activation, output: output)
    }

    private static let gateUpKernel = MLXFast.metalKernel(
        name: "mimo_v26_mxfp4_decode_gate_up", inputNames: ["x", "inds", "wg", "sg", "wu", "su"],
        outputNames: ["act"], source: MiMoV26DecodeExpertMetal.gateUp,
        header: MiMoV26DecodeExpertMetal.header, ensureRowContiguous: true)

    private static let downKernel = MLXFast.metalKernel(
        name: "mimo_v26_mxfp4_decode_down", inputNames: ["a", "inds", "w", "s"],
        outputNames: ["y"], source: MiMoV26DecodeExpertMetal.down,
        header: MiMoV26DecodeExpertMetal.header, ensureRowContiguous: true)
}
