// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Joint native MXFP4 gate/up experiment; no checkpoint or module mutation.

import Foundation
import MLX
import MLXFast

enum MiMoV26NAXGateUp {
    static let requested: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_V26_NAX_GATE_UP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    /// Separate opt-in until the epilogue is qualified on the actual NAX host.
    static let activationRequested =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_MIMO_V26_NAX_SWIGLU"] == "1"
    static let rowMapRequested =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_MIMO_V26_NAX_ROW_MAP"] == "1"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var encodings = 0

    static func encodedCalls() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return encodings
    }

    static func tryProjection(
        x: MLXArray, indices: MLXArray, gate: SwitchLinear, up: SwitchLinear,
        sorted: Bool
    ) -> (gate: MLXArray, up: MLXArray)? {
        guard
            let (gate, up, plan, stream) = eligibleProjection(
                x: x, indices: indices, gate: gate, up: up, sorted: sorted)
        else { return nil }
        let result = launch(
            x: x, indices: indices,
            gateWeight: gate.weight, gateScales: gate.scales,
            upWeight: up.weight, upScales: up.scales, plan: plan, stream: stream)
        lock.lock()
        encodings += 1
        lock.unlock()
        return result
    }

    /// Only the caller's known default SiLU product may use this path.
    /// It never substitutes an unknown/custom activation or changes weights.
    static func tryActivation(
        x: MLXArray, indices: MLXArray, gate: SwitchLinear, up: SwitchLinear,
        sorted: Bool, rowMap: MLXArray? = nil
    ) -> MLXArray? {
        guard activationRequested, MLXHardwareInfo.isCompiledDecodeSupported,
            rowMap == nil || rowMapRequested,
            let (gate, up, plan, stream) = eligibleProjection(
                x: x, indices: indices, gate: gate, up: up, sorted: sorted, rowMap: rowMap)
        else { return nil }
        let result = launchActivation(
            x: x, indices: indices,
            gateWeight: gate.weight, gateScales: gate.scales,
            upWeight: up.weight, upScales: up.scales, plan: plan, stream: stream, rowMap: rowMap)
        lock.lock()
        encodings += 1
        lock.unlock()
        return result
    }

    private static func eligibleProjection(
        x: MLXArray, indices: MLXArray, gate: SwitchLinear, up: SwitchLinear,
        sorted: Bool, rowMap: MLXArray? = nil
    ) -> (QuantizedSwitchLinear, QuantizedSwitchLinear, MiMoV26NAXGatherQMM.Plan, StreamOrDevice)? {
        let sortedRows = rowMap?.size ?? (x.ndim > 0 ? x.dim(0) : 0)
        if let rowMap {
            guard rowMap.ndim == 1, rowMap.dtype == .uint32, x.ndim == 3, x.dim(0) > 0
            else { return nil }
        }
        guard requested, sorted,
            let gate = gate as? QuantizedSwitchLinear, let up = up as? QuantizedSwitchLinear,
            ObjectIdentifier(type(of: gate)) == ObjectIdentifier(QuantizedSwitchLinear.self),
            ObjectIdentifier(type(of: up)) == ObjectIdentifier(QuantizedSwitchLinear.self),
            gate.bias == nil, up.bias == nil, gate.biases == nil, up.biases == nil,
            gate.mode == .mxfp4, up.mode == .mxfp4,
            gate.bits == 4, up.bits == 4, gate.groupSize == 32, up.groupSize == 32,
            x.ndim == 3, x.dim(1) == 1, x.dim(2) == 4096,
            x.dtype == .bfloat16 || x.dtype == .float16,
            indices.shape == [sortedRows], indices.dtype == .uint32,
            gate.weight.shape == [256, 2048, 512], up.weight.shape == gate.weight.shape,
            gate.weight.dtype == .uint32, up.weight.dtype == .uint32,
            gate.scales.shape == [256, 2048, 128], up.scales.shape == gate.scales.shape,
            gate.scales.dtype == .uint8, up.scales.dtype == .uint8,
            let plan = MiMoV26NAXGatherQMM.Plan.production(
                rows: sortedRows, experts: 256, input: 4096, output: 2048)
        else { return nil }
        let stream = StreamOrDevice.default
        guard MiMoV26NAXGatherQMM.gpuStream(stream), MiMoV26NAXGatherQMM.naxAvailable
        else { return nil }
        return (gate, up, plan, stream)
    }

    private static let scan = MLXFast.metalKernel(
        name: "mimo_v26_mxfp4_gate_up_tile_scan",
        inputNames: ["idx", "params"], outputNames: ["tiles", "tile_count"],
        source: MiMoV26NAXMetalSources.scanSource,
        header: MiMoV26NAXMetalSources.scanHeader + "\n", ensureRowContiguous: true)

    private static func makeKernel(tileRows: Int, activation: Bool = false, rowMap: Bool = false)
        -> MLXFast.MLXFastKernel
    {
        let single = MiMoV26NAXMetalSources.matmulHeader
            .replacingOccurrences(
                of: "STEEL_CONST int kBM = 64;",
                with: "STEEL_CONST int kBM = \(tileRows);"
            )
            .replacingOccurrences(
                of: "STEEL_CONST int kWM = 2;",
                with: "STEEL_CONST int kWM = \(tileRows / 32);")
        var source =
            activation
            ? MiMoV26NAXGateUpMetalSources.source
                .replacingOccurrences(
                    of: "gather_gate_up<T, Q>(", with: "gather_gate_up<T, Q, true>("
                )
                .replacingOccurrences(of: "gate_y, up_y", with: "activation_y, activation_y")
            : MiMoV26NAXGateUpMetalSources.source
        if rowMap {
            precondition(activation)
            source =
                source
                .replacingOccurrences(
                    of: "gather_gate_up<T, Q, true>(", with: "gather_gate_up<T, Q, true, true>("
                )
                .replacingOccurrences(
                    of: "gate_q, up_q, tiles, tiles,", with: "gate_q, up_q, tiles, row_map,")
        }
        return MLXFast.metalKernel(
            name: "mimo_v26_mxfp4_gate_up_bm\(tileRows)" + (activation ? "_silu" : "")
                + (rowMap ? "_mapped" : ""),
            inputNames: [
                "x", "gate_w", "gate_scales", "up_w", "up_scales",
                "tiles", "tile_count", "params",
            ] + (rowMap ? ["row_map"] : []),
            outputNames: activation ? ["activation_y"] : ["gate_y", "up_y"], source: source,
            header: MiMoV26NAXMetalSources.mlxHeader + "\n" + single + "\n"
                + MiMoV26NAXGateUpMetalSources.header + "\n",
            ensureRowContiguous: true)
    }

    private static let kernel64 = makeKernel(tileRows: 64)
    private static let kernel128 = makeKernel(tileRows: 128)
    private static let activationKernel64 = makeKernel(tileRows: 64, activation: true)
    private static let activationKernel128 = makeKernel(tileRows: 128, activation: true)
    private static let mappedKernel64 = makeKernel(tileRows: 64, activation: true, rowMap: true)
    private static let mappedKernel128 = makeKernel(tileRows: 128, activation: true, rowMap: true)

    /// Internal test seam, aligned dimensions only; caller owns GPU execution.
    static func launch(
        x: MLXArray, indices: MLXArray,
        gateWeight: MLXArray, gateScales: MLXArray,
        upWeight: MLXArray, upScales: MLXArray,
        plan: MiMoV26NAXGatherQMM.Plan, stream: StreamOrDevice = .default,
        tileRows: Int? = nil
    ) -> (gate: MLXArray, up: MLXArray) {
        let result = launchOutputs(
            x: x, indices: indices, gateWeight: gateWeight,
            gateScales: gateScales, upWeight: upWeight, upScales: upScales,
            plan: plan, stream: stream, tileRows: tileRows, activation: false, rowMap: nil)
        return (gate: result[0], up: result[1])
    }

    /// Test seam shares the exact projection/tiling path with the separate outputs.
    static func launchActivation(
        x: MLXArray, indices: MLXArray,
        gateWeight: MLXArray, gateScales: MLXArray,
        upWeight: MLXArray, upScales: MLXArray,
        plan: MiMoV26NAXGatherQMM.Plan, stream: StreamOrDevice = .default,
        tileRows: Int? = nil, rowMap: MLXArray? = nil
    ) -> MLXArray {
        launchOutputs(
            x: x, indices: indices, gateWeight: gateWeight,
            gateScales: gateScales, upWeight: upWeight, upScales: upScales,
            plan: plan, stream: stream, tileRows: tileRows, activation: true, rowMap: rowMap)[0]
    }

    private static func launchOutputs(
        x: MLXArray, indices: MLXArray,
        gateWeight: MLXArray, gateScales: MLXArray,
        upWeight: MLXArray, upScales: MLXArray,
        plan: MiMoV26NAXGatherQMM.Plan, stream: StreamOrDevice,
        tileRows: Int?, activation: Bool, rowMap: MLXArray?
    ) -> [MLXArray] {
        let bm = tileRows ?? plan.tileRows
        precondition(bm == 64 || bm == 128)
        precondition(plan.experts > 0 && plan.experts <= 256 && plan.rows >= 8)
        precondition(
            plan.input > 0 && plan.input % 64 == 0
                && plan.output > 0 && plan.output % 64 == 0)
        precondition(
            x.ndim == 3 && x.dim(1) == 1 && x.dim(2) == plan.input
                && (rowMap == nil ? x.dim(0) == plan.rows : x.dim(0) > 0)
                && (x.dtype == .bfloat16 || x.dtype == .float16))
        if let rowMap {
            precondition(activation && rowMap.shape == [plan.rows] && rowMap.dtype == .uint32)
        }
        precondition(indices.shape == [plan.rows] && indices.dtype == .uint32)
        for weight in [gateWeight, upWeight] {
            precondition(
                weight.shape == [plan.experts, plan.output, plan.input / 8]
                    && weight.dtype == .uint32)
        }
        for scales in [gateScales, upScales] {
            precondition(
                scales.shape == [plan.experts, plan.output, plan.input / 32]
                    && scales.dtype == .uint8)
        }
        let maxTiles = (plan.rows + bm - 1) / bm + min(plan.experts, plan.rows)
        let descriptors = scan(
            [indices, MLXArray([Int32(plan.rows), Int32(plan.experts), Int32(maxTiles)])],
            template: [("BM", bm), ("MAXE", 256)],
            grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[maxTiles * 4], [1]], outputDTypes: [.uint32, .uint32],
            stream: stream)
        precondition(descriptors.count == 2)
        let kernel =
            rowMap != nil
            ? (bm == 128 ? mappedKernel128 : mappedKernel64)
            : activation
                ? (bm == 128 ? activationKernel128 : activationKernel64)
                : (bm == 128 ? kernel128 : kernel64)
        let result = kernel(
            [
                x, gateWeight, gateScales, upWeight, upScales, descriptors[0], descriptors[1],
                MLXArray([Int32(plan.output), Int32(plan.input)]),
            ] + (rowMap.map { [$0] } ?? []),
            template: [("T", x.dtype)],
            grid: ((plan.output / 64) * 32, maxTiles * 2, bm / 32),
            threadGroup: (32, 2, bm / 32),
            outputShapes: Array(repeating: [plan.rows, 1, plan.output], count: activation ? 1 : 2),
            outputDTypes: Array(repeating: x.dtype, count: activation ? 1 : 2), stream: stream)
        precondition(result.count == (activation ? 1 : 2))
        return result
    }
}
