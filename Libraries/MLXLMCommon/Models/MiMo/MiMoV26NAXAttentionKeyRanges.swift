// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Ordered ranges adapted from oMLX #4032 @12e5084f8cf888bb6e846147929419f044802343.
// Modified: three native-rounded score phases, no online softmax or head split.

import Foundation
import MLX
import MLXFast

/// Ordered component encoder plus the genuine grouped native prefill entry.
/// The legacy single-call byte limit remains a component bound, NEVER admission.
/// Managed grouping requires MiMoV26NAXKeyRangeNative's real phase/step owner.
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
    static func balancedEdges(
        batch: Int, heads: Int, queries: Int, keys: Int,
        preferredKeys: Int = 8192,
        minimumThreadgroups: Int = 512
    ) -> [Int]? {
        guard validDimensions(batch, heads, queries, keys),
            preferredKeys >= 0, minimumThreadgroups >= 0
        else { return nil }
        let nq = (queries + 63) / 64
        let nk = (keys + 31) / 32
        guard preferredKeys > 0, batch * heads * nq >= minimumThreadgroups else {
            return [0, nk]
        }
        let rounded = (Double(nk * 32) / Double(preferredKeys)).rounded(.toNearestOrEven)
        let count = max(1, min(nk, Int(rounded)))
        // Bound host graph size; unsupported fragmentation keeps old fallback.
        guard count <= 128 else { return nil }
        return (0 ... count).map { ($0 * nk) / count }
    }

    private static func validDimensions(_ b: Int, _ h: Int, _ q: Int, _ k: Int) -> Bool {
        b > 0 && b <= 64 && h > 0 && h <= 64 && q > 8 && q <= k && k <= 1_048_576
    }

    /// No arrays/strides/evaluation needed. Each of three passes has R
    /// dispatches; 3R-1 full states stay live until the caller's real fence.
    static func makeSchedule(
        batch: Int, heads: Int, queries: Int, keys: Int,
        edges: [Int], maximumPayloadBytes: Int
    ) -> Schedule? {
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
            stateElements <= Int(Int32.max),  // Flattened custom-output shape ABI.
            let stateBytes = product([stateElements, 4]),
            let allStates = product([stateBytes, fullStates]),
            let scaledQ = product([batch, heads, queries, 192, 2]),
            let finalOutput = product([batch, heads, queries, 128, 2]),
            let payload = sum([
                allStates, scaledQ, finalOutput,
                2 * fullStates,  // Unused native-dtype out scalar per non-final dispatch.
                8,  // Initial state + final unused state scalar (float32).
                64 * dispatches,  // Distinct 16-word scalar parameter arrays.
                2 * heads,  // Worst-case contiguous sinks (or smaller dummy).
                1,  // Missing boolean-mask placeholder (broadcast masks are views).
                6,  // float scale scalar + native-dtype cast.
            ]),
            payload <= maximumPayloadBytes
        else { return nil }
        return Schedule(
            batch: batch, heads: heads, queries: queries, keys: keys,
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
    static func launch(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sinks: MLXArray?, schedule: Schedule,
        maximumPayloadBytes: Int, stream: StreamOrDevice = .default,
        retain: ([MLXArray]) -> Void
    ) -> Encoding? {
        guard MiMoV26NAXGatherQMM.gpuStream(stream), MiMoV26NAXGatherQMM.naxAvailable,
            let plan = MiMoV26NAXAttention.makePlan(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask, sinks: sinks, production: false),
            let checked = makeSchedule(
                batch: plan.batch, heads: plan.heads,
                queries: plan.queries, keys: plan.keys, edges: schedule.edges,
                maximumPayloadBytes: maximumPayloadBytes),
            checked == schedule
        else { return nil }
        let nq = (plan.queries + 63) / 64
        let nk = (plan.keys + 31) / 32
        let mask =
            plan.mask.map {
                broadcast($0, to: [plan.batch, plan.heads, plan.queries, plan.keys])
            } ?? MLXArray.zeros([1], dtype: .bool)
        let sinks =
            plan.sinks.map { contiguous($0, stream: stream) }
            ?? MLXArray.zeros([1], dtype: queries.dtype)
        // Same native-dtype scale and multiplication as the original launcher.
        let scaledQueries = queries * MLXArray(scale).asType(queries.dtype)
        var state = MLXArray.zeros([1], dtype: .float32)
        var roots = [scaledQueries, mask, sinks, state]
        retain(roots)
        var output: MLXArray?
        // The state edge enforces complete phase 0 -> phase 1 -> phase 2.
        // No max, exponential, sum or PV operation is added at a boundary.
        for pass in 0 ..< 3 {
            for index in 0 ..< (schedule.edges.count - 1) {
                let first = index == 0
                let last = index == schedule.edges.count - 2
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
                let outputs = kernel(
                    [scaledQueries, keys, values, mask, sinks, state, params],
                    template: [
                        ("T", queries.dtype), ("ALIGN_Q", plan.queries % 64 == 0),
                        ("ALIGN_K", plan.keys % 32 == 0), ("HAS_MASK", plan.mask != nil),
                        ("DO_CAUSAL", plan.causal), ("HAS_SINKS", plan.sinks != nil),
                        ("FIRST_RANGE", first), ("LAST_RANGE", last), ("SCORE_PASS", pass),
                    ],
                    grid: (nq * 128, plan.heads, plan.batch), threadGroup: (128, 1, 1),
                    outputShapes: [
                        final ? [plan.batch, plan.queries, plan.heads, 128] : [1],
                        final ? [1] : [schedule.stateElements],
                    ],
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

// MARK: Native grouped descriptors (shared score body, independent state)
extension MiMoV26NAXAttentionKeyRanges {
    struct GroupedSchedule {
        let descriptors: [MiMoV26BlockBatchAttention.Descriptor]
        let schedules: [Schedule]
        let stateOffsets: [Int]
        let stateElements: Int
        let queryStart: Int
        let queryCount: Int
        var dispatches: Int { schedules[0].dispatches }
        var fullStateBuffers: Int { dispatches - 1 }
    }

    /// Exactly the reference rounding rule, before allocating an edge table.
    /// Group eligibility below supplies the real512-threadgroup geometry.
    private static func rangeCount(_ keys: Int) -> Int? {
        guard keys > 0, keys <= 1_048_576 else { return nil }
        let blocks = (keys + 31) / 32
        let rounded = (Double(blocks * 32) / 8192).rounded(.toNearestOrEven)
        let count = max(1, min(blocks, Int(rounded)))
        return count <= 128 ? count : nil
    }

    private static func groupedCount(
        _ group: [MiMoV26BlockBatchAttention.Descriptor],
        heads: Int
    ) -> Int? {
        guard group.count == 4, heads == 64, let first = group.first,
            group.allSatisfy({ $0.queryCount == 128 && $0.sameVariant(first) }),
            heads * group.reduce(0, { $0 + ($1.queryCount + 63) / 64 }) >= 512,
            let count = rangeCount(first.keyCount), count >= 2,
            group.allSatisfy({ rangeCount($0.keyCount) == count })
        else { return nil }
        return count
    }

    private static func validGroupedPlan(_ plan: MiMoV26BlockBatchAttention.Plan) -> Bool {
        guard (129 ... 8192).contains(plan.queryCount), plan.keyCount >= plan.queryCount,
            plan.keyCount <= 1_048_576, plan.heads == 64, [4, 8].contains(plan.kvHeads),
            plan.window == nil || plan.window == 128, !plan.groups.isEmpty,
            plan.groups.count <= (plan.queryCount + 127) / 128
        else { return false }
        let history = plan.keyCount - plan.queryCount
        var cursor = 0
        for group in plan.groups {
            guard (1 ... 4).contains(group.count), let first = group.first else { return false }
            for d in group {
                guard (9 ... 128).contains(d.queryCount), d.queryStart == cursor,
                    d.queryCount <= plan.queryCount - cursor, d.sameVariant(first)
                else { return false }
                let bounds = CBv2AttentionV1.queryBlockBounds(
                    historyCount: history,
                    offset: cursor, count: d.queryCount, window: plan.window)
                guard d.keyStart == bounds.visibleStart,
                    d.keyCount == bounds.visibleEnd - bounds.visibleStart,
                    d.causal == (plan.window == nil || d.keyCount <= plan.window!)
                else { return false }
                cursor += d.queryCount
            }
        }
        return cursor == plan.queryCount
    }

    /// Scalar, checked projection over the existing block descriptor arrays.
    /// It allocates no extra MLX arrays and no range tables before real reserve.
    /// Every actual group buffer has its own allocator bound and multiplicity.
    static func projectedGroupedBytes(
        plan: MiMoV26BlockBatchAttention.Plan,
        maximumBufferBytes: Int, upperBound: (Int) -> Int?
    ) -> Int? {
        guard maximumBufferBytes > 0, validGroupedPlan(plan) else { return nil }
        func add(_ a: Int, _ b: Int) -> Int? {
            let x = a.addingReportingOverflow(b)
            return a >= 0 && b >= 0 && !x.overflow ? x.partialValue : nil
        }
        func mul(_ a: Int, _ b: Int) -> Int? {
            let x = a.multipliedReportingOverflow(by: b)
            return a >= 0 && b >= 0 && !x.overflow ? x.partialValue : nil
        }
        var total = 0
        var accepted = 0
        func buffer(_ logical: Int, _ copies: Int = 1) -> Bool {
            guard logical > 0, copies > 0, let bound = upperBound(logical),
                bound >= logical, bound <= maximumBufferBytes,
                let all = mul(bound, copies), let next = add(total, all)
            else { return false }
            total = next
            return true
        }
        for group in plan.groups {
            guard let ranges = groupedCount(group, heads: plan.heads) else { continue }
            // Actual grid z is four independent original q128 descriptors.
            // Each carries its own K prefix; there is no full-key hoist.
            let queryCount = group.reduce(0) { $0 + $1.queryCount }
            let stateElements = group.count * plan.heads * 2 * 64 * 130
            guard stateElements <= Int(Int32.max), let stateBytes = mul(stateElements, 4) else {
                return nil
            }
            let dispatches = 3 * ranges
            let fullStates = dispatches - 1
            let scaledQ = queryCount * plan.heads * 192 * 2
            let output = queryCount * plan.heads * 128 * 2
            guard buffer(stateBytes, fullStates),
                buffer(scaledQ), buffer(output),
                buffer(group.count * 2, fullStates),  // separate unused out slot/descriptor
                buffer(group.count * 4, 2),  // initial/final unused state slots
                buffer(group.count * 20 * 4, dispatches),
                buffer(plan.heads * 2), buffer(1),  // sinks/absent-mask placeholder
                buffer(4), buffer(2)
            else { return nil }  // actual native scale inputs
            if !group[0].causal {
                // The real MiMo128 window has one range and declines above.
                // Keep a conservative exact packed-mask envelope for the helper.
                var maskBytes = 0
                for d in group {
                    guard let bytes = mul(d.queryCount, d.keyCount),
                        let next = add(maskBytes, bytes), buffer(bytes, 4)
                    else { return nil }
                    maskBytes = next
                }
                guard buffer(maskBytes) else { return nil }
            }
            // Bounded host/control storage: arrays/primitive descriptors,
            // all parameter words and separate edge/state-offset tables.
            // Conservative source allowance, not measured Swift/driver heap.
            guard let nodes = mul(dispatches * 3 + 24, 1024),
                let edges = mul(group.count * (ranges + 1), 16),
                let metadata = add(nodes, edges),
                let host = add(metadata, 65_536),
                let next = add(total, host)
            else { return nil }
            total = next
            accepted += 1
        }
        return accepted > 0 ? total : nil
    }

    /// Called only AFTER the exact whole-layer projection has been admitted.
    static func groupedSchedules(plan: MiMoV26BlockBatchAttention.Plan)
        -> [Int: GroupedSchedule]?
    {
        guard validGroupedPlan(plan) else { return nil }
        var result: [Int: GroupedSchedule] = [:]
        for (index, group) in plan.groups.enumerated() {
            guard groupedCount(group, heads: plan.heads) != nil else { continue }
            var schedules: [Schedule] = []
            var offsets: [Int] = []
            var elements = 0
            for descriptor in group {
                // B here is the REAL dispatch's independent descriptor count,
                // not a fictional increase to one query block's occupancy.
                guard
                    let edges = balancedEdges(
                        batch: group.count, heads: plan.heads,
                        queries: descriptor.queryCount, keys: descriptor.keyCount),
                    let item = makeSchedule(
                        batch: 1, heads: plan.heads,
                        queries: descriptor.queryCount, keys: descriptor.keyCount,
                        edges: edges, maximumPayloadBytes: Int.max)
                else { return nil }
                offsets.append(elements)
                let next = elements.addingReportingOverflow(item.stateElements)
                guard !next.overflow, next.partialValue <= Int(Int32.max) else { return nil }
                elements = next.partialValue
                schedules.append(item)
            }
            guard let first = group.first, let last = group.last else { return nil }
            result[index] = .init(
                descriptors: group, schedules: schedules, stateOffsets: offsets,
                stateElements: elements, queryStart: first.queryStart,
                queryCount: last.queryStart + last.queryCount - first.queryStart)
        }
        return result
    }

    private static let groupedKernel = MLXFast.metalKernel(
        name: "mimo_v26_native_grouped_key_ranges_192_128",
        inputNames: ["q", "k", "v", "mask", "sinks", "state", "params"],
        outputNames: ["out", "state_out"],
        source: MiMoV26NAXKeyRangeGroupedMetalSources.source,
        header: MiMoV26NAXMetalSources.mlxHeader + "\n"
            + MiMoV26NAXAttentionMetalSources.header + "\n",
        ensureRowContiguous: false)

    private static let groupedCounterLock = NSLock()
    nonisolated(unsafe) private static var groupedEncodings = 0
    static func encodedGroupedDispatches() -> Int {
        groupedCounterLock.withLock { groupedEncodings }
    }

    /// No byte-limit authority is accepted here. The exact admitted layer
    /// owner retains every returned array through its actual completion.
    static func launchGrouped(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, window: Int?,
        schedule: GroupedSchedule, layer: MiMoV26NAXKeyRangeLayer,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let descriptors = schedule.descriptors
        let first = descriptors[0]
        let heads = schedule.schedules[0].heads
        let scaleInput = MLXArray(scale)
        let nativeScale = scaleInput.asType(queries.dtype)
        let scaled =
            queries[
                0..., 0..., schedule.queryStart ..< (schedule.queryStart + schedule.queryCount),
                0...]
            * nativeScale
        let nativeSinks =
            sinks.map { contiguous($0, stream: stream) }
            ?? MLXArray.zeros([1], dtype: queries.dtype)
        var masks: [MLXArray] = []
        var maskOffsets: [Int] = []
        var maskElements = 0
        for d in descriptors {
            maskOffsets.append(maskElements)
            if case .array(let mask) = CBv2AttentionV1.maskMode(
                L: d.queryCount, kL: d.keyCount, window: window)
            {
                masks.append(mask.reshaped([-1]))
                maskElements += d.queryCount * d.keyCount
            }
        }
        let mask =
            masks.isEmpty
            ? MLXArray.zeros([1], dtype: .bool)
            : (masks.count == 1 ? masks[0] : concatenated(masks))
        var state = MLXArray.zeros([descriptors.count], dtype: .float32)
        layer.retain([scaleInput, nativeScale, scaled, nativeSinks, mask, state] + masks)
        // Borrowed K/V remain with the existing native request/step and
        // kernel dependency graph. Do not retain extra slice descriptors here:
        // extending them past normal evaluation can force full-KV COW on the
        // next append. This reservation prices newly emitted scratch only.
        var result: MLXArray?
        let ranges = schedule.schedules[0].edges.count - 1
        for pass in 0 ..< 3 {
            for range in 0 ..< ranges {
                let final = pass == 2 && range == ranges - 1
                var words: [Int32] = []
                words.reserveCapacity(descriptors.count * 20)
                for (block, d) in descriptors.enumerated() {
                    let item = schedule.schedules[block]
                    words += [
                        1, Int32(heads), 192, Int32(d.queryCount), Int32(d.keyCount),
                        Int32(heads / keys.dim(1)), Int32(bitPattern: scale.bitPattern),
                        Int32((d.queryCount + 63) / 64), Int32((d.keyCount + 31) / 32),
                        Int32(d.queryCount / 64), Int32(d.keyCount / 32),
                        Int32(d.queryCount % 64), Int32(d.keyCount % 32),
                        Int32(d.keyCount - d.queryCount),
                        Int32(d.queryStart - schedule.queryStart), Int32(d.keyStart),
                        Int32(maskOffsets[block]), Int32(schedule.stateOffsets[block]),
                        Int32(item.edges[range]), Int32(item.edges[range + 1]),
                    ]
                }
                let params = MLXArray(words)
                layer.retain([params])
                let output = groupedKernel(
                    [scaled, keys, values, mask, nativeSinks, state, params],
                    template: [
                        ("T", queries.dtype), ("ALIGN_Q", first.alignedQuery),
                        ("ALIGN_K", first.alignedKey), ("HAS_MASK", !first.causal),
                        ("DO_CAUSAL", first.causal), ("HAS_SINKS", sinks != nil),
                        ("FIRST_RANGE", range == 0), ("LAST_RANGE", range == ranges - 1),
                        ("SCORE_PASS", pass),
                    ],
                    grid: (2 * 128, heads, descriptors.count), threadGroup: (128, 1, 1),
                    outputShapes: [
                        final ? [1, schedule.queryCount, heads, 128] : [descriptors.count],
                        final ? [descriptors.count] : [schedule.stateElements],
                    ],
                    outputDTypes: [queries.dtype, .float32], stream: stream)
                layer.retain(output)
                precondition(output.count == 2)
                state = output[1]
                if final {
                    result = output[0].transposed(0, 2, 1, 3)
                    layer.retain([result!])
                }
                groupedCounterLock.withLock { groupedEncodings += 1 }
            }
        }
        return result!
    }
}
