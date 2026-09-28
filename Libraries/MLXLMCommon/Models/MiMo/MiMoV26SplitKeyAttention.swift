// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0 AND MIT
// Narrow matrix-QK adaptation of oMLX #4033 @c10e4ad57b180fa1aa88bf40532cba4362b6851a.
// Source-only packet attribution: ATTRIBUTION.md, LICENSE-OMLX, LICENSE-MLX.
import Foundation
import MLX
import MLXFast

/// NOT wired into managed attention. A future caller must admit these exact
/// step temporaries, retain evaluationRoots before submission and keep its
/// native loan through the existing completion/fault boundary.
enum MiMoV26SplitKeyAttention {
    static let requested = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MIMO_V26_SPLITKEY_QK"] == "1"

    struct Plan: Equatable {
        let rows: Int
        let keys: Int
        let blocks: Int
        let elementBytes: Int
        let hasSinks: Bool
        var visibleKeys: [Int] { (0..<rows).map { keys - rows + $0 + 1 } }
        var threads: Int { rows * 64 } // G16 / eight-head band *32
        var threadgroupBytes: Int { (8 * 200 + 8 * 136) * elementBytes + rows * 2 * 64 * 4 }
        /// Independent allocations: FP32 numerator, sum, max; final native
        /// output; scalar scale; contiguous sink (or one-element placeholder).
        /// q/k/v are caller-owned borrowed inputs and must remain charged too.
        var allocationByteCounts: [Int] {
            let count = 64 * rows * blocks
            return [count * 128 * 4, count * 4, count * 4,
                    64 * rows * 128 * elementBytes, 4, (hasSinks ? 64 : 1) * elementBytes]
        }
        var temporaryLogicalBytes: Int { allocationByteCounts.prefix(3).reduce(0, +) }
        var totalLogicalBytes: Int { allocationByteCounts.reduce(0, +) }
        /// Apply the existing allocator policy independently to every buffer.
        /// This is admission metadata, NOT measured backing/materialization.
        func boundedAllocationBytes(_ bound: (Int) throws -> Int) throws -> Int {
            try allocationByteCounts.reduce(0) { total, bytes in
                let allocated = try bound(bytes)
                guard allocated >= bytes else { throw AllocationError.invalidBound }
                let sum = total.addingReportingOverflow(allocated)
                guard !sum.overflow else { throw AllocationError.overflow }
                return sum.partialValue
            }
        }
    }
    enum AllocationError: Error { case invalidBound, overflow }

    /// Metadata-only refusal checks, including EVERY scalar reference row's
    /// native block count. No new chunk-size heuristic or count-based widening.
    static func plan(rows: Int, keys: Int, elementBytes: Int, hasSinks: Bool,
                     deviceClass: Character, blockOverride: Int = 0) -> Plan? {
        guard (1...4).contains(rows), keys >= rows, keys <= 1_048_576,
              elementBytes == 2, Set<Character>(["g", "s", "d"]).contains(deviceClass),
              blockOverride >= 0, blockOverride <= 4096 else { return nil }
        let blocks: Int?
        if rows == 1 {
            guard ((deviceClass == "d" || deviceClass == "s") && keys >= 1024) || keys >= 4096
            else { return nil } // selected core would use one pass
            blocks = MiMoV26DecodeRows.blocks(keys: keys, simds: 16,
                deviceClass: deviceClass, override: blockOverride)
        } else {
            blocks = MiMoV26DecodeRows.serialBlocks(rows: rows, keys: keys,
                deviceClass: deviceClass, override: blockOverride)
        }
        guard let blocks else { return nil }
        return .init(rows: rows, keys: keys, blocks: blocks,
                     elementBytes: elementBytes, hasSinks: hasSinks)
    }

    static func makePlan(queries q: MLXArray, keys k: MLXArray, values v: MLXArray,
                         scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                         sinks: MLXArray?, deviceClass: Character, blockOverride: Int = 0) -> Plan? {
        guard q.ndim == 4, k.ndim == 4, v.ndim == 4,
              q.dim(0) == 1, q.dim(1) == 64, q.dim(3) == 192,
              k.dim(0) == 1, k.dim(1) == 4, k.dim(3) == 192,
              v.shape == [1, 4, k.dim(2), 128],
              q.dtype == .bfloat16 || q.dtype == .float16,
              k.dtype == q.dtype, v.dtype == q.dtype, scale.isFinite, scale > 0 else { return nil }
        switch mask {
        case .none: guard q.dim(2) == 1 else { return nil }
        case .causal: break
        case .array, .arrays: return nil // includes SWA/padding/bias/span masks
        }
        if let sinks { guard sinks.shape == [64], sinks.dtype == q.dtype else { return nil } }
        return plan(rows: q.dim(2), keys: k.dim(2), elementBytes: q.dtype.size,
                    hasSinks: sinks != nil, deviceClass: deviceClass, blockOverride: blockOverride)
    }

    struct Encoding {
        let output: MLXArray
        /// Explicit ownership, not merely a graph dependency assumption.
        /// No sync/eval/metadata probing/destructor refund occurs here.
        let evaluationRoots: [MLXArray]
        let plan: Plan
    }
    private static let first = MLXFast.metalKernel(name: "mimo_splitkey_matrix_qk",
        inputNames: ["queries", "keys", "values", "scale", "sinks"],
        outputNames: ["partials", "sums", "maxs"],
        source: MiMoV26SplitKeyAttentionMetalSources.pass1, ensureRowContiguous: false)
    private static let second = MLXFast.metalKernel(name: "mimo_splitkey_native_merge",
        inputNames: ["partials", "sums", "maxs"], outputNames: ["out"],
        // The already-qualified FP32-partial core merge is reused byte-for-byte.
        source: MiMoV26DecodeRowsMetalSources.pass2, ensureRowContiguous: true)

    /// Optional encoding seam, not model/row eligibility or a reservation.
    /// No production caller is installed by this packet.
    static func tryEncode(queries q: MLXArray, keys k: MLXArray, values v: MLXArray,
                          scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                          sinks: MLXArray?, stream: StreamOrDevice = .default) -> Encoding? {
        guard requested, MiMoV26NAXGatherQMM.gpuStream(stream),
              MiMoV26NAXGatherQMM.naxAvailable else { return nil }
        let architecture = GPU.deviceInfo().architecture
        guard architecture.hasPrefix("applegpu_"), let deviceClass = architecture.last else { return nil }
        let overrideText = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard overrideText.isEmpty || Int(overrideText) != nil else { return nil }
        guard let plan = makePlan(queries: q, keys: k, values: v, scale: scale,
            mask: mask, sinks: sinks, deviceClass: deviceClass,
            blockOverride: Int(overrideText) ?? 0) else { return nil }
        return encode(queries: q, keys: k, values: v, scale: scale,
                      sinks: sinks, plan: plan, stream: stream)
    }

    /// Native qualification seam. matrixScores=false uses the pinned scalar
    /// QK sum within the SAME recurrence/layout, discriminating changes outside
    /// matrix reduction. It is not a production fallback or a fake result.
    static func encode(queries q: MLXArray, keys k: MLXArray, values v: MLXArray,
                       scale: Float, sinks: MLXArray?, plan: Plan,
                       matrixScores: Bool = true, stream: StreamOrDevice = .default) -> Encoding {
        precondition(q.shape == [1, 64, plan.rows, 192] && k.shape == [1, 4, plan.keys, 192])
        precondition(v.shape == [1, 4, plan.keys, 128] && k.dtype == q.dtype && v.dtype == q.dtype)
        precondition((q.dtype == .bfloat16 || q.dtype == .float16) && (sinks != nil) == plan.hasSinks)
        precondition(sinks == nil || (sinks!.shape == [64] && sinks!.dtype == q.dtype))
        precondition(scale.isFinite && scale > 0 && plan.blocks > 0 && plan.blocks <= 4096
                     && plan.blocks.isMultiple(of: 32) && (1...4).contains(plan.rows))
        let scaleInput = MLXArray([scale])
        let sinkInput = sinks.map { contiguous($0, stream: stream) } ?? MLXArray.zeros([1], dtype: q.dtype, stream: stream)
        let partials = first([q, k, v, scaleInput, sinkInput],
            template: [("T", q.dtype), ("ROWS", plan.rows), ("BLOCKS", plan.blocks),
                       ("HAS_SINKS", plan.hasSinks), ("MATRIX_QK", matrixScores)],
            grid: (plan.threads * 4, 1, plan.blocks), threadGroup: (plan.threads, 1, 1),
            outputShapes: [[64 * plan.rows, plan.blocks, 128],
                           [64 * plan.rows, plan.blocks], [64 * plan.rows, plan.blocks]],
            outputDTypes: [.float32, .float32, .float32], stream: stream)
        let result = second(partials,
            template: [("T", q.dtype), ("V", 128), ("NQ", 64),
                       ("ROWS", plan.rows), ("BLOCKS", plan.blocks)],
            grid: (1024 * 64, plan.rows, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, plan.rows, 64 * 128]], outputDTypes: [q.dtype], stream: stream)[0]
        let output = result.reshaped([1, plan.rows, 64, 128]).transposed(0, 2, 1, 3)
        return .init(output: output, evaluationRoots: partials + [result, scaleInput, sinkInput], plan: plan)
    }
}
