// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Swift port of oMLX #3994, head 1c487861d1c1d4c82a2b2920dd69c064eec85e18.

import Foundation
import MLX
import MLXFast

/// Narrow MiMo prefill route. Results preserve the input/output dtype and
/// depend on the caller's existing step roots; this helper never evaluates.
public enum MiMoV26NAXAttention {
    // Default policy permits the existing native-rounded route; actual
    // geometry/device/dtype gates below remain authoritative. Explicit 0 rolls back.
    static let requested = MiMoV26PrefillPolicy.isEnabled(
        ProcessInfo.processInfo.environment[MiMoV26PrefillPolicy.attentionEnvironmentKey])

    private static let lock = NSLock()
    nonisolated(unsafe) private static var encodings = 0

    /// Graph encodings only, not a claim of completed kernel execution.
    public static func encodedCalls() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return encodings
    }

    struct Plan {
        let batch: Int
        let heads: Int
        let kvHeads: Int
        let queries: Int
        let keys: Int
        let causal: Bool
        let mask: MLXArray?
        let sinks: MLXArray?
    }

    static func makePlan(
        queries q: MLXArray, keys k: MLXArray, values v: MLXArray,
        scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sinks: MLXArray?, production: Bool
    ) -> Plan? {
        guard q.ndim == 4, k.ndim == 4, v.ndim == 4,
            q.dtype == .bfloat16 || q.dtype == .float16,
            k.dtype == q.dtype, v.dtype == q.dtype,
            q.dim(3) == 192, k.dim(3) == 192, v.dim(3) == 128,
            q.dim(0) > 0, q.dim(0) <= 64, q.dim(0) == k.dim(0),
            Array(v.shape.prefix(3)) == Array(k.shape.prefix(3)),
            q.dim(1) > 0, q.dim(1) <= 64, k.dim(1) > 0,
            q.dim(1) % k.dim(1) == 0,
            q.dim(2) > 8, k.dim(2) >= q.dim(2), k.dim(2) <= 1_048_576,
            scale.isFinite, scale > 0
        else { return nil }
        if production && (q.dim(1) != 64 || ![4, 8].contains(k.dim(1))) { return nil }
        if let sinks {
            guard sinks.shape == [q.dim(1)], sinks.dtype == q.dtype else { return nil }
        }
        let causal: Bool
        let arrayMask: MLXArray?
        switch mask {
        case .none:
            causal = false
            arrayMask = nil
        case .causal:
            causal = true
            arrayMask = nil
        case .array(let array):
            guard array.dtype == .bool, array.ndim >= 1, array.ndim <= 4 else { return nil }
            let padded = Array(repeating: 1, count: 4 - array.ndim) + array.shape
            let target = [q.dim(0), q.dim(1), q.dim(2), k.dim(2)]
            guard zip(padded, target).allSatisfy({ $0.0 == 1 || $0.0 == $0.1 }) else { return nil }
            causal = false
            arrayMask = array
        case .arrays:
            return nil
        }
        return Plan(
            batch: q.dim(0), heads: q.dim(1), kvHeads: k.dim(1),
            queries: q.dim(2), keys: k.dim(2), causal: causal,
            mask: arrayMask, sinks: sinks)
    }

    /// Called only by ordinary MiMo attention and MiMo-owned contiguous caches.
    /// Span overlays, softcaps, serialized MTP and <=8-row calls stay outside.
    public static func tryAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sinks: MLXArray?
    ) -> MLXArray? {
        guard requested,
            let plan = makePlan(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask, sinks: sinks, production: true)
        else { return nil }
        let stream = StreamOrDevice.default
        guard MiMoV26NAXGatherQMM.gpuStream(stream),
            MiMoV26NAXGatherQMM.naxAvailable
        else { return nil }
        let result = launch(
            queries: queries, keys: keys, values: values,
            scale: scale, plan: plan, stream: stream)
        lock.lock()
        encodings += 1
        lock.unlock()
        return result
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mimo_v26_nax_attention_192_128",
        inputNames: ["q", "k", "v", "mask", "sinks", "params"],
        outputNames: ["out"],
        source: MiMoV26NAXAttentionMetalSources.source,
        header: MiMoV26NAXMetalSources.mlxHeader + "\n"
            + MiMoV26NAXAttentionMetalSources.header + "\n",
        ensureRowContiguous: false)

    /// Internal numerical-oracle seam. The caller owns all execution and errors.
    static func launch(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, plan: Plan, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let nq = (plan.queries + 63) / 64
        let nk = (plan.keys + 31) / 32
        // Matches the 56-byte C struct: six int32, float32, seven int32.
        let words: [Int32] = [
            Int32(plan.batch), Int32(plan.heads), 192, Int32(plan.queries),
            Int32(plan.keys), Int32(plan.heads / plan.kvHeads),
            Int32(bitPattern: scale.bitPattern), Int32(nq), Int32(nk),
            Int32(plan.queries / 64), Int32(plan.keys / 32),
            Int32(plan.queries % 64), Int32(plan.keys % 32),
            Int32(plan.keys - plan.queries),
        ]
        let mask =
            plan.mask.map {
                broadcast($0, to: [plan.batch, plan.heads, plan.queries, plan.keys])
            } ?? MLXArray.zeros([1], dtype: .bool)
        // Reading sinks[head] requires contiguous storage; values are not cast.
        let sinks =
            plan.sinks.map { contiguous($0, stream: stream) }
            ?? MLXArray.zeros([1], dtype: queries.dtype)
        // Match fast.cpp: native-dtype scale constant and rounded scaled Q.
        let scaledQueries = queries * MLXArray(scale).asType(queries.dtype)
        let outputs = kernel(
            [scaledQueries, keys, values, mask, sinks, MLXArray(words)],
            template: [
                ("T", queries.dtype), ("ALIGN_Q", plan.queries % 64 == 0),
                ("ALIGN_K", plan.keys % 32 == 0), ("HAS_MASK", plan.mask != nil),
                ("DO_CAUSAL", plan.causal), ("HAS_SINKS", plan.sinks != nil),
            ],
            grid: (nq * 128, plan.heads, plan.batch), threadGroup: (128, 1, 1),
            outputShapes: [[plan.batch, plan.queries, plan.heads, 128]],
            outputDTypes: [queries.dtype], stream: stream)
        precondition(outputs.count == 1)
        return outputs[0].transposed(0, 2, 1, 3)
    }
}
