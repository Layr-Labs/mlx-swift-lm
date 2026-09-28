// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Host port of jundot/omlx PR #3995 at 47876fbc310fbb311cd382c4eba8572ec4368308.
// Experimental, OFF by default. See docs/mimo-v26/NAX-GATHER-PORT.md.

import Cmlx
import Foundation
import MLX
import MLXFast

/// Counts graph encodings, not completed GPU executions or benchmark samples.
public enum MiMoV26NAXGatherDiagnostics {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var encodings = 0

    public static func encodedCalls() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return encodings
    }

    fileprivate static func recordEncoding() {
        lock.lock()
        encodings += 1
        lock.unlock()
    }
}

enum MiMoV26NAXGatherQMM {
    static let requested: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_V26_NAX_GATHER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    struct Plan: Equatable {
        let rows: Int
        let experts: Int
        let input: Int
        let output: Int
        let tileRows: Int
        let doubleBuffer: Bool
        var maxTiles: Int { (rows + tileRows - 1) / tileRows + min(experts, rows) }

        init(rows: Int, experts: Int, input: Int, output: Int) {
            self.rows = rows
            self.experts = experts
            self.input = input
            self.output = output
            tileRows = rows > 64 * experts && rows <= 128 * experts && input >= 2048
                ? 128 : 64
            doubleBuffer = rows <= 128 * experts
        }

        /// Only exact native MiMo gate, up, optional gate/up and down shapes.
        static func production(rows: Int, experts: Int, input: Int, output: Int) -> Plan? {
            guard experts == 256, rows >= 1024, rows <= Int(Int32.max) - 128,
                (input == 4096 && (output == 2048 || output == 4096))
                    || (input == 2048 && output == 4096)
            else { return nil }
            let plan = Plan(rows: rows, experts: experts, input: input, output: output)
            guard plan.maxTiles <= Int(Int32.max) / 4 else { return nil }
            return plan
        }
    }

    static func gpuStream(_ stream: StreamOrDevice) -> Bool {
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        var type = MLX_CPU
        return mlx_stream_get_device(&device, stream.ctx) == 0
            && mlx_device_get_type(&type, device) == 0 && type == MLX_GPU
    }

    static let naxAvailable: Bool = {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // Uses the pinned core's real OS, architecture and NO_NAX-build gate.
        return GPU.gemma4ExpertQMMDiagnostics().naxAvailable
        #else
        return false
        #endif
    }()

    static func tryProjection(
        x: MLXArray, indices: MLXArray, weight: MLXArray, scales: MLXArray,
        biases: MLXArray?, sorted: Bool, groupSize: Int, bits: Int,
        mode: QuantizationMode
    ) -> MLXArray? {
        guard requested, sorted, x.ndim == 3, x.dim(1) == 1,
            x.dtype == .bfloat16 || x.dtype == .float16,
            indices.ndim == 1, indices.dtype == .uint32,
            indices.size == x.dim(0), weight.ndim == 3, weight.dtype == .uint32,
            mode == .mxfp4, bits == 4, groupSize == 32, biases == nil,
            scales.dtype == .uint8,
            let plan = Plan.production(
                rows: x.dim(0), experts: weight.dim(0),
                input: x.dim(2), output: weight.dim(1)),
            weight.shape == [plan.experts, plan.output, plan.input / 8],
            scales.shape == [plan.experts, plan.output, plan.input / 32]
        else { return nil }

        let stream = StreamOrDevice.default
        guard gpuStream(stream), naxAvailable else { return nil }
        // The caller's gatherSort owns sortedness and bounds. No eval, canary,
        // stream substitution, weight conversion or cached array enters forward.
        let result = launch(x: x, indices: indices, weight: weight, scales: scales,
                            plan: plan, stream: stream)
        MiMoV26NAXGatherDiagnostics.recordEncoding()
        return result
    }

    private static let scan = MLXFast.metalKernel(
        name: "mimo_v26_mxfp4_nax_tile_scan",
        inputNames: ["idx", "params"], outputNames: ["tiles", "tile_count"],
        source: MiMoV26NAXMetalSources.scanSource,
        header: MiMoV26NAXMetalSources.scanHeader + "\n",
        ensureRowContiguous: true)

    private static func makeMatmul(tileRows: Int) -> MLXFast.MLXFastKernel {
        let header = MiMoV26NAXMetalSources.matmulHeader
            .replacingOccurrences(of: "STEEL_CONST int kBM = 64;",
                                  with: "STEEL_CONST int kBM = \(tileRows);")
            .replacingOccurrences(of: "STEEL_CONST int kWM = 2;",
                                  with: "STEEL_CONST int kWM = \(tileRows / 32);")
        return MLXFast.metalKernel(
            name: "mimo_v26_mxfp4_nax_bm\(tileRows)",
            inputNames: ["x", "w", "scales", "tiles", "tile_count", "params"],
            outputNames: ["y"], source: MiMoV26NAXMetalSources.matmulSource,
            header: MiMoV26NAXMetalSources.mlxHeader + "\n" + header + "\n",
            ensureRowContiguous: true)
    }

    private static let matmul64 = makeMatmul(tileRows: 64)
    private static let matmul128 = makeMatmul(tileRows: 128)

    /// Internal test seam: synthetic aligned MXFP4 shapes use the same kernels.
    /// Callers must own the GPU lane; it never performs an evaluation itself.
    static func launch(
        x: MLXArray, indices: MLXArray, weight: MLXArray, scales: MLXArray,
        plan: Plan, stream: StreamOrDevice = .default,
        tileRows: Int? = nil, doubleBuffer: Bool? = nil
    ) -> MLXArray {
        let bm = tileRows ?? plan.tileRows
        let db = doubleBuffer ?? plan.doubleBuffer
        precondition(bm == 64 || bm == 128)
        precondition(plan.experts > 0 && plan.experts <= 256 && plan.rows >= 8)
        precondition(plan.input > 0 && plan.input % 64 == 0
                     && plan.output > 0 && plan.output % 64 == 0)
        precondition(x.shape == [plan.rows, 1, plan.input]
                     && (x.dtype == .bfloat16 || x.dtype == .float16))
        precondition(indices.shape == [plan.rows] && indices.dtype == .uint32)
        precondition(weight.shape == [plan.experts, plan.output, plan.input / 8]
                     && weight.dtype == .uint32)
        precondition(scales.shape == [plan.experts, plan.output, plan.input / 32]
                     && scales.dtype == .uint8)
        let maxTiles = (plan.rows + bm - 1) / bm + min(plan.experts, plan.rows)
        let scanParams = MLXArray([Int32(plan.rows), Int32(plan.experts), Int32(maxTiles)])
        let descriptors = scan(
            [indices, scanParams], template: [("BM", bm), ("MAXE", 256)],
            grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[maxTiles * 4], [1]], outputDTypes: [.uint32, .uint32],
            stream: stream)
        precondition(descriptors.count == 2)
        let matmul = bm == 128 ? matmul128 : matmul64
        let output = matmul(
            [x, weight, scales, descriptors[0], descriptors[1],
             MLXArray([Int32(plan.output), Int32(plan.input)])],
            template: [("T", x.dtype), ("GS", 32), ("SCHED", db ? 1 : 0),
                       ("ALIGN_N", true), ("ALIGN_K", true)],
            grid: ((plan.output / 64) * 32, maxTiles * 2, bm / 32),
            threadGroup: (32, 2, bm / 32),
            outputShapes: [[plan.rows, 1, plan.output]], outputDTypes: [x.dtype],
            stream: stream)
        precondition(output.count == 1)
        return output[0]
    }
}
