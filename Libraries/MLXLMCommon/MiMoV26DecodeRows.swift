// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// oMLX #4033 multi-row vector attention, adapted to the pinned core's
// FP32 partials and this engine's SINGLE-ROW serial authority.
import Foundation
import MLX
import MLXFast

enum MiMoV26DecodeRows {
    static let requested = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MIMO_V26_DECODE_ROWS"] == "1"

    /// The source core consumes partials in 32-block groups. This is a
    /// refusal bound for this optional specialization, not an operator cap.
    static func blocks(keys: Int, simds: Int, deviceClass: Character, override: Int = 0) -> Int? {
        guard keys > 0, simds > 0, simds <= 32 else { return nil }
        let base: Int
        if deviceClass == "s" {
            base = keys > 1024 && simds > 4
                ? (keys <= 8192 ? 128 : keys <= 32768 ? 256 : keys <= 65536 ? 512 : 1024)
                : 64
        } else if deviceClass == "d" {
            base = simds <= 2 && keys > 8192 ? 256
                : simds >= 6 && keys >= 65536 ? 1024
                : simds >= 6 && keys >= 16384 ? 512 : 128
        } else {
            base = simds >= 4 ? 64 : 32
        }
        guard override <= 4096 else { return nil }
        return override > 0 ? ((override + 31) / 32) * 32 : base
    }

    /// Compare EVERY native scalar call, not oMLX's different two-row grouping.
    /// Crossing a single-pass or block-count boundary preserves the old path.
    static func serialBlocks(rows: Int, keys: Int, heads: Int = 64, kvHeads: Int = 4,
                             deviceClass: Character, override: Int = 0) -> Int? {
        guard (2...4).contains(rows), keys >= rows, heads == 64, kvHeads == 4 else { return nil }
        let gqa = heads / kvHeads
        var selected: Int?
        for row in 0..<rows {
            let visible = keys - rows + row + 1
            guard ((deviceClass == "d" || deviceClass == "s") && visible >= 1024)
                    || (kvHeads < heads && visible >= 4096),
                  let value = blocks(keys: visible, simds: gqa,
                                     deviceClass: deviceClass, override: override)
            else { return nil }
            if let selected, selected != value { return nil }
            selected = value
        }
        return selected
    }

    private static var deviceClass: Character? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let name = GPU.deviceInfo().architecture
        guard name.hasPrefix("applegpu_") else { return nil }
        return name.last
        #else
        return nil
        #endif
    }

    private static let first = MLXFast.metalKernel(name: "mimo_fp32partial_rows_pass1",
        inputNames: ["queries", "keys", "values", "scale", "mask", "sinks"],
        outputNames: ["partials", "sums", "maxs"],
        source: MiMoV26DecodeRowsMetalSources.pass1, ensureRowContiguous: false)
    private static let second = MLXFast.metalKernel(name: "mimo_fp32partial_rows_pass2",
        inputNames: ["partials", "sums", "maxs"], outputNames: ["out"],
        source: MiMoV26DecodeRowsMetalSources.pass2, ensureRowContiguous: true)

    /// Only genuine singleton, full-causal native MiMo verification calls.
    /// SWA, spans, softcap, arbitrary masks, batching and unknown devices use
    /// the original path. No host stride inspection or native evaluation here.
    static func tryAttention(queries q: MLXArray, keys k: MLXArray, values v: MLXArray,
                             scale: Float, sinks: MLXArray?) -> MLXArray? {
        guard requested, q.ndim == 4, k.ndim == 4, v.ndim == 4,
              q.dim(0) == 1, q.dim(1) == 64, q.dim(3) == 192,
              k.dim(0) == 1, k.dim(1) == 4, k.dim(3) == 192,
              v.shape == [1, 4, k.dim(2), 128], k.dim(2) <= 1_048_576,
              q.dtype == .bfloat16 || q.dtype == .float16,
              k.dtype == q.dtype, v.dtype == q.dtype, scale.isFinite, scale > 0,
              MiMoV26NAXGatherQMM.gpuStream(.default), let deviceClass
        else { return nil }
        if let sinks { guard sinks.shape == [64], sinks.dtype == q.dtype else { return nil } }
        let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard let blocks = serialBlocks(rows: q.dim(2), keys: k.dim(2),
                                        deviceClass: deviceClass, override: Int(raw) ?? 0)
        else { return nil }
        return launch(queries: q, keys: k, values: v, scale: scale, sinks: sinks, blocks: blocks)
    }

    /// Internal oracle seam. The caller owns actual evaluation/error handling.
    static func launch(queries q: MLXArray, keys k: MLXArray, values v: MLXArray,
                       scale: Float, sinks: MLXArray?, blocks: Int,
                       stream: StreamOrDevice = .default) -> MLXArray {
        let rows = q.dim(2), heads = q.dim(1), kvHeads = k.dim(1), batch = q.dim(0)
        precondition((2...4).contains(rows) && batch == 1 && heads == 64 && kvHeads == 4)
        precondition(blocks > 0 && blocks <= 4096 && blocks.isMultiple(of: 32))
        let preparedSinks = sinks.map { contiguous($0, stream: stream) }
            ?? MLXArray.zeros([1], dtype: q.dtype)
        let partials = first([q, k, v, MLXArray([scale]), MLXArray.zeros([1], dtype: .bool), preparedSinks],
            template: [("T", q.dtype), ("MT", DType.bool), ("D", 192), ("V", 128),
                       ("G", heads / kvHeads), ("NKV", kvHeads), ("ROWS", rows),
                       ("BLOCKS", blocks), ("CAUSAL", true), ("MASK_KIND", 0),
                       ("HAS_SINKS", sinks != nil)],
            grid: (32 * kvHeads, (heads / kvHeads) * batch, blocks),
            threadGroup: (32, heads / kvHeads, 1),
            outputShapes: [[batch * heads * rows, blocks, 128],
                           [batch * heads * rows, blocks], [batch * heads * rows, blocks]],
            outputDTypes: [.float32, .float32, .float32], stream: stream)
        let result = second(partials,
            template: [("T", q.dtype), ("V", 128), ("NQ", heads), ("ROWS", rows), ("BLOCKS", blocks)],
            grid: (1024 * batch * heads, rows, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[batch, rows, heads * 128]], outputDTypes: [q.dtype], stream: stream)
        return result[0].reshaped([batch, rows, heads, 128]).transposed(0, 2, 1, 3)
    }
}
