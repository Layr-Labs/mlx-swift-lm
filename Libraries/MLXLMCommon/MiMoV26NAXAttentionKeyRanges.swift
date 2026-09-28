// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Ordered ranges adapted from oMLX #4032 @12e5084f8cf888bb6e846147929419f044802343.
// Modified: three native-rounded score phases, no online softmax or head split.

import MLX
import MLXFast

/// Internal component candidate, NOT connected to tryAttention. A byte limit is
/// only a bounded component-call limit, not an admission ticket. Production
/// needs the existing engine/phase owner before this path may be wired.
enum MiMoV26NAXAttentionKeyRanges {
    struct Schedule: Equatable {
        let batch: Int
        let heads: Int
        let queries: Int
        let keys: Int
        let edges: [Int]
        let stateElements: Int
        let stateBufferBytes: Int
        let fullStateBuffers: Int
        let dispatches: Int
        /// All distinct array payloads, retaining every lazy dispatch result.
        /// Excludes caller-owned inputs, allocator padding and compiler/driver
        /// workspace; this is not measured peak or process admission authority.
        let arrayPayloadBytes: Int
    }

    struct Encoding {
        let output: MLXArray
        /// Retain through actual completion; on an uncertain drain the native
        /// fault owner must retain these and the corresponding charge.
        let retainedArrays: [MLXArray]
        let schedule: Schedule
    }

    // Match Python round-to-even for the reference's balanced ~8192-key
    // dispatch policy. All positions are absolute 32-key block indices.
    static func balancedEdges(batch: Int, heads: Int, queries: Int, keys: Int,
                              preferredKeys: Int = 8192,
                              minimumThreadgroups: Int = 512) -> [Int]? {
        guard validDimensions(batch, heads, queries, keys),
              preferredKeys >= 0, minimumThreadgroups >= 0 else { return nil }
        let nq = (queries + 63) / 64, nk = (keys + 31) / 32
        guard preferredKeys > 0, batch * heads * nq >= minimumThreadgroups else {
            return [0, nk]
        }
        let rounded = (Double(nk * 32) / Double(preferredKeys)).rounded(.toNearestOrEven)
        let count = max(1, min(nk, Int(rounded)))
        // Bound host graph size; unsupported fragmentation keeps old fallback.
        guard count <= 128 else { return nil }
        return (0...count).map { ($0 * nk) / count }
    }

    private static func validDimensions(_ b: Int, _ h: Int, _ q: Int, _ k: Int) -> Bool {
        b > 0 && b <= 64 && h > 0 && h <= 64 && q > 8 && q <= k && k <= 1_048_576
    }

    /// No arrays/strides/evaluation needed. Each of three passes has R
    /// dispatches; 3R-1 full states stay live until the caller's real fence.
    static func makeSchedule(batch: Int, heads: Int, queries: Int, keys: Int,
                             edges: [Int], maximumPayloadBytes: Int) -> Schedule? {
        guard validDimensions(batch, heads, queries, keys), maximumPayloadBytes > 0,
              edges.count >= 2, edges.count <= 129, edges.first == 0,
              edges.last == (keys + 31) / 32,
              zip(edges, edges.dropFirst()).allSatisfy({ $0.0 < $0.1 })
        else { return nil }
        func product(_ values: [Int]) -> Int? {
            var result = 1
            for value in values {
                let next = result.multipliedReportingOverflow(by: value)
                guard !next.overflow else { return nil }
                result = next.partialValue
            }
            return result
        }
        func sum(_ values: [Int]) -> Int? {
            var result = 0
            for value in values {
                let next = result.addingReportingOverflow(value)
                guard !next.overflow else { return nil }
                result = next.partialValue
            }
            return result
        }
        let dispatches = 3 * (edges.count - 1)
        let fullStates = dispatches - 1
        guard let stateElements = product([batch, heads, (queries + 63) / 64, 64, 130]),
              stateElements <= Int(Int32.max), // Flattened custom-output shape ABI.
              let stateBytes = product([stateElements, 4]),
              let allStates = product([stateBytes, fullStates]),
              let scaledQ = product([batch, heads, queries, 192, 2]),
              let finalOutput = product([batch, heads, queries, 128, 2]),
              let payload = sum([
                allStates, scaledQ, finalOutput,
                2 * fullStates, // Unused native-dtype out scalar per non-final dispatch.
                8, // Initial state + final unused state scalar (float32).
                64 * dispatches, // Distinct 16-word scalar parameter arrays.
                2 * heads, // Worst-case contiguous sinks (or smaller dummy).
                1, // Missing boolean-mask placeholder (broadcast masks are views).
                6, // float scale scalar + native-dtype cast.
              ]),
              payload <= maximumPayloadBytes
        else { return nil }
        return Schedule(batch: batch, heads: heads, queries: queries, keys: keys,
            edges: edges, stateElements: stateElements, stateBufferBytes: stateBytes,
            fullStateBuffers: fullStates, dispatches: dispatches, arrayPayloadBytes: payload)
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mimo_v26_nax_attention_key_ranges_192_128",
        inputNames: ["q", "k", "v", "mask", "sinks", "state", "params"],
        outputNames: ["out", "state_out"],
        source: #"""
          omlx_nax::attention_nax_bdv_impl<
              T, 64, 32, 192, 128, 4, 1,
              ALIGN_Q, ALIGN_K, HAS_MASK, DO_CAUSAL, HAS_SINKS,
              true, FIRST_RANGE, LAST_RANGE, SCORE_PASS, bool, float>(
              q, k, v, out,
              reinterpret_cast<const device omlx_nax::AttnParams*>(params),
              q_strides, k_strides, v_strides, mask_strides,
              mask, sinks, state, state_out, params[14], params[15],
              simdgroup_index_in_threadgroup,
              threadgroup_position_in_grid);
        """#,
        header: MiMoV26NAXMetalSources.mlxHeader + "\n"
            + MiMoV26NAXAttentionMetalSources.header + "\n",
        ensureRowContiguous: false)

    /// Never evaluates/synchronizes or changes stream/device. Not a production
    /// opt-in API: caller owns exclusive native execution and real completion.
    /// Returns nil before custom encoding for invalid geometry, CPU/non-NAX,
    /// schedule mismatch or a refused bounded payload.
    static func launch(queries: MLXArray, keys: MLXArray, values: MLXArray,
                       scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                       sinks: MLXArray?, schedule: Schedule,
                       maximumPayloadBytes: Int, stream: StreamOrDevice = .default,
                       retain: ([MLXArray]) -> Void) -> Encoding? {
        guard MiMoV26NAXGatherQMM.gpuStream(stream), MiMoV26NAXGatherQMM.naxAvailable,
              let plan = MiMoV26NAXAttention.makePlan(queries: queries, keys: keys, values: values,
                  scale: scale, mask: mask, sinks: sinks, production: false),
              let checked = makeSchedule(batch: plan.batch, heads: plan.heads,
                  queries: plan.queries, keys: plan.keys, edges: schedule.edges,
                  maximumPayloadBytes: maximumPayloadBytes),
              checked == schedule else { return nil }
        let nq = (plan.queries + 63) / 64, nk = (plan.keys + 31) / 32
        let mask = plan.mask.map {
            broadcast($0, to: [plan.batch, plan.heads, plan.queries, plan.keys])
        } ?? MLXArray.zeros([1], dtype: .bool)
        let sinks = plan.sinks.map { contiguous($0, stream: stream) }
            ?? MLXArray.zeros([1], dtype: queries.dtype)
        // Same native-dtype scale and multiplication as the original launcher.
        let scaledQueries = queries * MLXArray(scale).asType(queries.dtype)
        var state = MLXArray.zeros([1], dtype: .float32)
        var roots = [scaledQueries, mask, sinks, state]
        retain(roots)
        var output: MLXArray?
        // The state edge enforces complete phase 0 -> phase 1 -> phase 2.
        // No max, exponential, sum or PV operation is added at a boundary.
        for pass in 0..<3 {
            for index in 0..<(schedule.edges.count - 1) {
                let first = index == 0, last = index == schedule.edges.count - 2
                let final = pass == 2 && last
                let words: [Int32] = [
                    Int32(plan.batch), Int32(plan.heads), 192, Int32(plan.queries),
                    Int32(plan.keys), Int32(plan.heads / plan.kvHeads),
                    Int32(bitPattern: scale.bitPattern), Int32(nq), Int32(nk),
                    Int32(plan.queries / 64), Int32(plan.keys / 32),
                    Int32(plan.queries % 64), Int32(plan.keys % 32),
                    Int32(plan.keys - plan.queries),
                    Int32(schedule.edges[index]), Int32(schedule.edges[index + 1]),
                ]
                let params = MLXArray(words)
                retain([params])
                let outputs = kernel([scaledQueries, keys, values, mask, sinks, state, params],
                    template: [("T", queries.dtype), ("ALIGN_Q", plan.queries % 64 == 0),
                        ("ALIGN_K", plan.keys % 32 == 0), ("HAS_MASK", plan.mask != nil),
                        ("DO_CAUSAL", plan.causal), ("HAS_SINKS", plan.sinks != nil),
                        ("FIRST_RANGE", first), ("LAST_RANGE", last), ("SCORE_PASS", pass)],
                    grid: (nq * 128, plan.heads, plan.batch), threadGroup: (128, 1, 1),
                    outputShapes: [final ? [plan.batch, plan.queries, plan.heads, 128] : [1],
                                   final ? [1] : [schedule.stateElements]],
                    outputDTypes: [queries.dtype, .float32], stream: stream)
                precondition(outputs.count == 2)
                retain(outputs)
                roots.append(params)
                roots.append(contentsOf: outputs)
                state = outputs[1]
                if final { output = outputs[0].transposed(0, 2, 1, 3) }
            }
        }
        guard let output else { preconditionFailure("validated nonempty range schedule") }
        return Encoding(output: output, retainedArrays: roots, schedule: schedule)
    }
}
