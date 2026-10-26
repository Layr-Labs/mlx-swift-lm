// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Cmlx
import Foundation
import MLX

/// An opt-in ordinary-evaluation route; transformations keep the original graph.
enum Gemma4RowMappedNAX {
    static let requested =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_NAX_ROW_MAP"] == "1"

    private static func ordinary(_ array: MLXArray) -> Bool {
        var identity: UInt = 0
        var ordinary = false
        return _mlx_array_constant_cache_identity(&identity, &ordinary, array.ctx) == 0
            && ordinary
    }

    private static func supported(_ bank: QuantizedSwitchLinear) -> Bool {
        ObjectIdentifier(type(of: bank)) == ObjectIdentifier(QuantizedSwitchLinear.self)
            && !bank.training && bank.bias == nil
            && bank.mode == .affine && bank.bits == 4 && bank.groupSize == 64
            && bank.weight.shape == [128, 704, 352] && bank.weight.dtype == .uint32
            && bank.scales.shape == [128, 704, 44] && bank.scales.dtype == .bfloat16
            && bank.biases?.shape == [128, 704, 44] && bank.biases?.dtype == .bfloat16
    }

    static func supportsDownProjection(_ projection: SwitchLinear) -> Bool {
        guard let bank = projection as? QuantizedSwitchLinear,
            ObjectIdentifier(type(of: bank)) == ObjectIdentifier(QuantizedSwitchLinear.self),
            !bank.training, bank.bias == nil,
            bank.mode == .affine, bank.bits == 4, bank.groupSize == 64,
            bank.weight.shape == [128, 2816, 88], bank.weight.dtype == .uint32,
            bank.scales.shape == [128, 2816, 11], bank.scales.dtype == .bfloat16,
            let biases = bank.biases, biases.shape == [128, 2816, 11], biases.dtype == .bfloat16
        else { return false }
        return [bank.weight, bank.scales, biases].allSatisfy(ordinary)
    }

    /// All arrays remain live graph inputs. No weight, map or array identity is cached.
    /// The caller owns the stock-module and explicit-prefill contract.
    static func tryPair(
        _ x: MLXArray, indices: MLXArray, gate: SwitchLinear, up: SwitchLinear,
        enabled: Bool = requested
    ) -> (gate: MLXArray, up: MLXArray, sorted: MLXArray, inverseOrder: MLXArray)? {
        #if canImport(Metal)
        guard enabled, x.ndim == 2, x.dim(1) == 2816, x.dtype == .bfloat16,
            x.dim(0) >= 64, x.dim(0) <= 2048,
            indices.shape == [x.dim(0), 8], indices.dtype == .uint32,
            let gate = gate as? QuantizedSwitchLinear,
            let up = up as? QuantizedSwitchLinear,
            supported(gate), supported(up),
            let gateBiases = gate.biases, let upBiases = up.biases,
            MLXProcessStreams.isOnDefaultGPUStream
        else { return nil }
        // Read the core's actual OS/architecture/build gate after CPU refusal.
        guard GPU.gemma4ExpertQMMDiagnostics().naxAvailable,
            [
                x, indices, gate.weight, gate.scales, gateBiases,
                up.weight, up.scales, upBiases,
            ].allSatisfy(ordinary)
        else { return nil }

        let flat = indices.flattened()
        let order = argSort(flat)
        let sorted = flat[order]
        let rowMap = order.floorDivide(8)
        let tokenRows = x.reshaped([-1, 1, 2816])
        guard [order, sorted, rowMap, tokenRows].allSatisfy(ordinary) else { return nil }
        let pair = Gemma4RowMappedNAXKernel.pair(
            tokenRows: tokenRows, sortedExperts: sorted, rowMap: rowMap,
            gate: .init(weight: gate.weight, scales: gate.scales, biases: gateBiases),
            up: .init(weight: up.weight, scales: up.scales, biases: upBiases))
        CBv2StepProfiler.recordEvent("v2.gemma4.nax.row_map")
        return (pair.gate, pair.up, sorted, argSort(order))
        #else
        return nil
        #endif
    }
}
