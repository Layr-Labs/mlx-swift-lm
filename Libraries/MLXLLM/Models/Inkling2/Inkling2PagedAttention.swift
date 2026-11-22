// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXFast

/// Paged attention for Inkling 2 layers: `eager_attention_forward` of the
/// sliding and full attention layers, read straight from the engine's pages.
///
/// For query `i` of row `b` at absolute position `qpos = queryStart + i` and
/// query head `h` (kv head `h / gqa`):
///
///     logit[k] = scale * q.k + bias[h, i, qpos - k]   (bias only for distance < 1024)
///     out      = softmax([logits over k in [lo, qpos]] ++ [sink[h, i]])[:-1] @ V
///
/// with `lo = max(base, qpos - window + 1)` for sliding layers and `base` for
/// full ones. The sink is an extra logit with a zero value.
///
/// Keys before `queryStart` come from the pages; the call's own `L` tokens come
/// from `newKeys`/`newValues`, so a prefill chunk never reads a ring slot its own
/// write would overwrite. Each query's arithmetic is fixed by its own inputs:
/// the output bits do not depend on the batch, on `L`, on the chunking, or on
/// which kernel ran (split or sequential).
public enum Inkling2PagedAttention {

    public static let pageSize = 16
    /// Tokens per partition. Part of the arithmetic: changing it changes bits.
    public static let partitionTokens = 128
    /// Keys per online-softmax tile (one page). Part of the arithmetic.
    public static let tileKeys = 16
    /// Inkling 2's relative-bias extent (distances `0 ..< 1024` carry a bias)
    /// and relative-state width. Calls may use any extent and widths up to
    /// `maximumRelativeDim`; both come from `proj`'s shape.
    public static let relativeExtent = 1024
    public static let relativeDim = 16
    public static let maximumRelativeDim = 16
    /// Queries per row up to which `.automatic` picks the split kernel.
    public static let splitMaximumQueries = 16
    /// Largest float32 partial buffer `.automatic` lets the split kernel
    /// allocate; above it the sequential kernel (no partials, same bits) runs.
    public static let splitPartialsLimit = 1 << 30
    public static let supportedHeadDims: Set<Int> = [64, 128]

    /// Most segment buffers one call may bind. Calls bind one of two counts
    /// per pass whatever the pool holds: `fewSegmentBindings` when the call
    /// touches that few, else the pass's maximum (fused
    /// `fusedMaximumSegmentBindings`, every other pass `maximumSegments`);
    /// unused bindings repeat segment 0 and are never selected. So the kernels
    /// compiled do not depend on the pool's segment count, and a warm-up with
    /// one small and one large call covers them.
    public static let maximumSegments = 16
    /// Bindings of a call over at most this many segments (one-row calls:
    /// binding 8 or 16 there cost the M5 Max 10 % of a one-row step).
    public static let fewSegmentBindings = 2

    /// One layer group's paged K/V in either engine layout. Pages are written
    /// in place; reads and writes order through the group's write fence.
    public struct PagedKV {
        public enum Storage {
            /// Fixed slabs: keys and values `[pages, kvHeads, 16, headDim]`.
            /// Table entries are page ids.
            case slab(keys: MLXArray, values: MLXArray)
            /// Segments `[2, pages, kvHeads, 16, headDim]` (keys, then values),
            /// as `PagedKVBackend.segmented` stores them. Table entries are
            /// `binding << 24 | local page`, `binding` indexing `segments`.
            case segments([MLXArray])
            /// More segments than one dispatch binds, split into buckets of at
            /// most `maximumBucketSegments` (one split dispatch each, then one
            /// merge). Table entries are page ids. Decode and verification calls
            /// (`L <= splitMaximumQueries`) only.
            case segmentBuckets([SegmentBucket])
        }

        /// One dispatch's share of a bucketed call.
        public struct SegmentBucket {
            /// The bucket's segments `[2, pages, kvHeads, 16, headDim]`, at most
            /// `maximumBucketSegments`.
            public var segments: [MLXArray]
            /// int32, at least `2 * maximumBucketSegments`: each segment's first
            /// page id and page count, zero past `segments`.
            public var ranges: MLXArray
            /// int32 `[2 * itemCount]`: (row, partition) work items, the
            /// partition counted from the one holding the row's lowest key.
            /// Every page an item reads, and for partition 0 every page the
            /// row's tokens are written to, lies in `segments`.
            public var items: MLXArray
            public var itemCount: Int

            public init(segments: [MLXArray], ranges: MLXArray, items: MLXArray, itemCount: Int) {
                self.segments = segments
                self.ranges = ranges
                self.items = items
                self.itemCount = itemCount
            }
        }

        public var storage: Storage
        /// `[B, maxPages]` int32 table entries, at least 8 columns.
        public var tables: MLXArray
        /// `[1]` int32 latest fence of the group's write chain.
        public var writeFence: MLXArray

        public init(keys: MLXArray, values: MLXArray, tables: MLXArray, writeFence: MLXArray) {
            storage = .slab(keys: keys, values: values)
            self.tables = tables
            self.writeFence = writeFence
        }

        public init(segments: [MLXArray], tables: MLXArray, writeFence: MLXArray) {
            precondition(
                (1 ... Inkling2PagedAttention.maximumSegments).contains(segments.count),
                "bind 1 ... \(Inkling2PagedAttention.maximumSegments) segments per call")
            storage = .segments(segments)
            self.tables = tables
            self.writeFence = writeFence
        }

        /// Page-id tables over more segments than one dispatch binds, split
        /// into buckets (decode and verification calls).
        public init(buckets: [SegmentBucket], tables: MLXArray, writeFence: MLXArray) {
            precondition(
                !buckets.isEmpty
                    && buckets.allSatisfy {
                        (1 ... Inkling2PagedAttention.maximumBucketSegments).contains(
                            $0.segments.count)
                    },
                "bind 1 ... \(Inkling2PagedAttention.maximumBucketSegments) segments per bucket")
            storage = .segmentBuckets(buckets)
            self.tables = tables
            self.writeFence = writeFence
        }

        var dtype: DType {
            switch storage {
            case .slab(let keys, _): keys.dtype
            case .segments(let segments): segments[0].dtype
            case .segmentBuckets(let buckets): buckets[0].segments[0].dtype
            }
        }
    }

    /// Host description of one row of a call.
    public struct Row: Sendable, Equatable {
        /// Absolute position of the call's first query (and first new token).
        public var queryStart: Int
        /// Oldest position the row may attend (non-zero after prefix adoption).
        public var base: Int
        /// Length the logical page index wraps at: the ring length for a
        /// sliding row, the table capacity for a full row.
        public var tableLength: Int

        public init(queryStart: Int, base: Int = 0, tableLength: Int) {
            self.queryStart = queryStart
            self.base = base
            self.tableLength = tableLength
        }
    }

    /// Row metadata for every layer of one kind in one step. Build it once per
    /// step and reuse it across layers: the device copy is made here.
    public struct Plan {
        public let rows: [Row]
        public let queries: Int
        public let window: Int?
        public let scale: Float
        /// `[B, 8]` int32: query start, base, table length, window, scale bits,
        /// the plan's partitions per row and its row count.
        let rowInfo: MLXArray
        let maxPartitions: Int
        /// Arrival counters and partition states of the fused kernel, shared by
        /// the plan's calls (which run one after another on one stream).
        let workspace = FusedWorkspace()

        public init(rows: [Row], queries: Int, window: Int?, scale: Float) {
            precondition(!rows.isEmpty && queries > 0, "a plan needs rows and queries")
            precondition((window ?? 1) > 0, "window must be positive")
            for row in rows {
                precondition(row.queryStart >= row.base && row.base >= 0 && row.tableLength > 0)
            }
            let windowTokens = window ?? 0
            self.rows = rows
            self.queries = queries
            self.window = window
            self.scale = scale
            // Column 4 carries `scale * log2(e)` as float bits.
            let scaleBits = Int32(bitPattern: (scale * Float(M_LOG2E)).bitPattern)
            let maxPartitions = rows.map {
                Inkling2PagedAttention.partitionCount(
                    row: $0, queries: queries, window: windowTokens)
            }.max()!
            self.maxPartitions = maxPartitions
            rowInfo = MLXArray(
                rows.flatMap {
                    [
                        Int32($0.queryStart), Int32($0.base), Int32($0.tableLength),
                        Int32(windowTokens), scaleBits, Int32(maxPartitions), Int32(rows.count), 0,
                    ]
                }, [rows.count, 8])
        }
    }

    /// Axis order of the call's per-query and new-token tensors. Token-major
    /// is what the projections produce (`[B, L, heads, ...]`) and needs no
    /// transpose; head-major is `[B, heads, L, ...]`. Same bits either way.
    public enum Layout: Sendable {
        case headMajor
        case tokenMajor
    }

    /// Which kernel runs the call. All give the same bits.
    public enum Kernel: Sendable {
        /// Fused while `L <= splitMaximumQueries`, `B <= fusedMaximumBatch`, the
        /// plan has at most `fusedMaximumPartitions` partitions
        /// (`fusedMultiBlockMaximumPartitions` with two or more blocks) and its
        /// states fit `fusedWorkspaceLimit`; else split while its partials fit
        /// `splitPartialsLimit`; else sequential.
        case automatic
        /// Partitions in parallel and the merge in one dispatch: the last
        /// threadgroup of each (row, kv head, query group) merges its
        /// partitions. Up to `fusedMaximumPartitions` partitions.
        case fused
        /// Partitions in parallel, then a merge pass (decode, verification).
        case split
        /// Partitions in order inside one threadgroup (long calls; same bits).
        case sequential
        /// Prefill chunks on the M5 neural accelerators
        /// (`neuralAcceleratorsAvailable`): its own arithmetic, chunk- and
        /// batch-invariant, not bit-equal to the kernels above. Never picked by
        /// `.automatic`; prefill callers choose it so every prompt token takes
        /// the same arithmetic however the prompt is chunked.
        case prefill
    }

    /// The GPU has the neural accelerators the prefill kernel runs on: MLX's
    /// own NAX condition (OS, architecture generation, phone-class rule and
    /// build), read once per process. The kernel's fragment layout is MLX's
    /// NAX layout, so it runs exactly where MLX runs its NAX kernels; anywhere
    /// else prompt chunks take the decode arithmetic.
    public static let neuralAcceleratorsAvailable: Bool = {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // MLX exposes the device NAX probe only through this diagnostics record.
        return GPU.gemma4ExpertQMMDiagnostics().naxAvailable
        #else
        return false
        #endif
    }()

    /// Most partitions a row's queries may span for the fused kernel.
    public static let fusedMaximumPartitions = 16
    /// Largest batch `.automatic` runs fused: one dispatch saves a launch and
    /// its encoding, which pays at small batches (measured on M5 Max: sliding
    /// B=1 27 vs 36 us GPU per call); larger batches keep the two passes.
    public static let fusedMaximumBatch = 8
    /// Largest fused workspace (float32 partition states) `.automatic` uses.
    public static let fusedWorkspaceLimit = 64 << 20
    /// Most segment buffers the fused kernel binds: with its workspace and the
    /// sinks it needs, 16 segments would pass Metal's 31 buffer slots.
    public static let fusedMaximumSegmentBindings = 8

    /// The fused kernel's per-plan device state: zeroed arrival counters (each
    /// launch's last threadgroups clear theirs) and partition states, sized
    /// for the largest launch so far.
    final class FusedWorkspace: @unchecked Sendable {
        private let lock = NSLock()
        private var counters: MLXArray?
        private var states: MLXArray?

        func buffers(counters needCounters: Int, states needStates: Int) -> (MLXArray, MLXArray) {
            lock.withLock {
                if let c = counters, c.size >= needCounters, let w = states, w.size >= needStates {
                    return (c, w)
                }
                let c = MLXArray([UInt32](repeating: 0, count: max(8, needCounters)))
                let w = MLXArray.zeros([max(8, needStates)], dtype: .float32)
                counters = c
                states = w
                return (c, w)
            }
        }
    }

    public struct Output {
        /// `[B, L, QH * headDim]` in the slab dtype.
        public let output: MLXArray
        /// Next fence of the write chain. Later writers and readers of the
        /// slabs must consume it.
        public let writeFence: MLXArray
    }

    /// Attend `L` queries per row over the row's paged history plus the call's
    /// own tokens.
    ///
    /// Shapes are given head-major; with `layout: .tokenMajor` swap the head
    /// and `L` axes of `queries`, `newKeys`, `newValues`, `relativeStates` and
    /// `sinks`.
    ///
    /// - Parameters:
    ///   - queries: `[B, QH, L, D]`, normalised, not scaled.
    ///   - newKeys: `[B, KVH, L, D]` keys of positions `queryStart ..< queryStart + L`,
    ///     in the slab dtype.
    ///   - newValues: `[B, KVH, L, D]`.
    ///   - relativeStates: `[B, QH, L, 16]` normalised (and, for full layers,
    ///     log-scaled) relative states; the kernel adds `relativeStates @ proj`
    ///     for distances below 1024.
    ///   - proj: `[16, 1024]` the layer's `rel_logits_proj.proj`.
    ///   - sinks: `[B, QH, L]` sink logits.
    ///   - cache: slabs, tables and write fence of this layer's group.
    ///   - plan: rows, query count, window and scale of the call.
    ///   - write: the kernel also writes the new K/V into their pages (single
    ///     writer per byte): position `queryStart + i` to slot `pos % 16` of the
    ///     page that table entry `(pos / 16) % tableLength` names, which the
    ///     caller must have reserved. A sliding row must satisfy
    ///     `L + window - 1 <= tableLength * 16` so no attended slot is overwritten.
    public static func attend(
        queries: MLXArray,
        newKeys: MLXArray,
        newValues: MLXArray,
        relativeStates: MLXArray,
        proj: MLXArray,
        sinks: MLXArray,
        cache: PagedKV,
        plan: Plan,
        write: Bool = false,
        layout: Layout = .headMajor,
        kernel: Kernel = .automatic,
        subPartitions: Int? = nil,
        stream: StreamOrDevice = .default
    ) -> Output {
        // One shape read per array: each read copies the shape out of MLX.
        let qShape = queries.shape
        let kShape = newKeys.shape
        precondition(qShape.count == 4, "queries must be rank 4")
        let tokenMajor = layout == .tokenMajor
        let b = qShape[0]
        let queryHeads = qShape[tokenMajor ? 2 : 1]
        let l = qShape[tokenMajor ? 1 : 2]
        let headDim = qShape[3]
        let kvHeads = kShape.count == 4 ? kShape[tokenMajor ? 2 : 1] : 0
        // `[B, H, L, X]` head-major or `[B, L, H, X]` token-major.
        func shape(_ heads: Int, _ inner: Int?) -> [Int] {
            let order = tokenMajor ? [l, heads] : [heads, l]
            return [b] + order + (inner.map { [$0] } ?? [])
        }
        let dtype = cache.dtype
        precondition(b == plan.rows.count && l == plan.queries, "queries do not match the plan")
        precondition(supportedHeadDims.contains(headDim), "unsupported head dim \(headDim)")
        precondition(kvHeads > 0 && queryHeads % kvHeads == 0, "GQA requires QH % KVH == 0")
        // Slabs, or the call's segments (bound after the pass is chosen).
        var slabInputs: [MLXArray] = []
        var segmentList: [MLXArray] = []
        switch cache.storage {
        case .slab(let keys, let values):
            let slabShape = keys.shape
            precondition(slabShape == values.shape && values.dtype == dtype)
            precondition(
                slabShape.count == 4 && slabShape[1] == kvHeads && slabShape[2] == pageSize
                    && slabShape[3] == headDim, "slabs must be [pages, KVH, 16, D]")
            slabInputs = [keys, values]
        case .segments(let segments):
            segmentList = segments
        case .segmentBuckets:
            break
        }
        precondition(
            kShape == shape(kvHeads, headDim) && newValues.shape == kShape,
            "new K/V must be [B, KVH, L, D] (head-major) or [B, L, KVH, D] (token-major)")
        precondition(
            newKeys.dtype == dtype && newValues.dtype == dtype, "new K/V must match the slabs")
        precondition(sinks.shape == shape(queryHeads, nil), "sinks must match the layout")
        let tableShape = cache.tables.shape
        precondition(
            tableShape.count == 2 && tableShape[0] == b && tableShape[1] >= 8
                && cache.tables.dtype == .int32, "tables must be int32 [B, >= 8]")
        if write, let window = plan.window {
            for row in plan.rows {
                precondition(
                    l + window - 1 <= row.tableLength * pageSize,
                    "a fused ring write of \(l) tokens would overwrite attended history")
            }
        }

        let gqa = queryHeads / kvHeads
        let projShape = proj.shape
        precondition(
            projShape.count == 2 && (1 ... maximumRelativeDim).contains(projShape[0])
                && projShape[1] > 0, "proj must be [dRel <= \(maximumRelativeDim), extent]")
        let relativeWidth = projShape[0]
        let extent = projShape[1]
        precondition(relativeStates.shape == shape(queryHeads, relativeWidth))

        // Simdgroups per partition of the split family (same bits either way),
        // then queries per threadgroup; each simdgroup holds 8 (query, head) pairs.
        var psub = subPartitionCount(
            requested: subPartitions, b: b, kvHeads: kvHeads, queries: l, gqa: gqa,
            maxPartitions: plan.maxPartitions, headDim: headDim, elementSize: dtype.size)
        var splitQPT = queriesPerThreadgroup(
            split: true, queries: l, gqa: gqa, headDim: headDim, elementSize: dtype.size,
            simdgroupsPerBlock: psub)
        if case .segmentBuckets(let buckets) = cache.storage {
            precondition(
                l <= splitMaximumQueries && kernel != .prefill && kernel != .sequential,
                "bucketed segments serve decode and verification calls")
            return attendBuckets(
                buckets, queries: queries, relativeStates: relativeStates, proj: proj,
                sinks: sinks, newKeys: newKeys, newValues: newValues, cache: cache, plan: plan,
                variant: Variant(
                    dtype: dtype, headDim: headDim, gqa: gqa, qpt: splitQPT, storageBindings: 0,
                    tokenMajor: tokenMajor, extent: extent, relativeWidth: relativeWidth,
                    write: write, psub: psub),
                kvHeads: kvHeads, queryHeads: queryHeads, b: b, l: l, stream: stream)
        }
        var fusedStates =
            b * kvHeads * ((l + splitQPT - 1) / splitQPT) * splitQPT * gqa * plan.maxPartitions
            * (headDim + 2)
        let pass: Pass
        switch kernel {
        case .automatic:
            let partialsBytes = b * queryHeads * l * plan.maxPartitions * (headDim + 2) * 4
            if l > splitMaximumQueries {
                pass = .sequential
            } else if b <= fusedMaximumBatch && plan.maxPartitions <= fusedMaximumPartitions
                && fusedStates * 4 <= fusedWorkspaceLimit
                && segmentList.count <= fusedMaximumSegmentBindings
                && !((splitQPT * gqa + 7) / 8 > 1
                    && fusedMultiBlockCap.map { plan.maxPartitions > $0 } ?? false)
            {
                pass = .fused
            } else {
                pass = partialsBytes <= splitPartialsLimit ? .split : .sequential
            }
        case .fused:
            precondition(
                plan.maxPartitions <= fusedMaximumPartitions,
                "the fused kernel takes up to \(fusedMaximumPartitions) partitions")
            precondition(
                segmentList.count <= fusedMaximumSegmentBindings,
                "the fused kernel binds up to \(fusedMaximumSegmentBindings) segments")
            pass = .fused
        case .split: pass = .split
        case .sequential: pass = .sequential
        case .prefill:
            precondition(
                neuralAcceleratorsAvailable, "the prefill kernel needs the M5 neural accelerators")
            precondition(!write, "the prefill kernel does not write pages")
            precondition(
                dtype == .bfloat16 || dtype == .float16,
                "the prefill kernel takes bfloat16 or float16 K/V")
            precondition(headDim % 32 == 0 && relativeWidth <= 16)
            pass = .prefill
        }
        let storageInputs: [MLXArray]
        let storageBindings: Int
        if segmentList.isEmpty {
            (storageInputs, storageBindings) = (slabInputs, 0)
        } else {
            (storageInputs, storageBindings) = segmentInputs(
                segmentList,
                classes: [
                    fewSegmentBindings,
                    pass == .fused ? fusedMaximumSegmentBindings : maximumSegments,
                ],
                dtype: dtype, kvHeads: kvHeads, headDim: headDim)
        }
        // The one-dispatch kernel from `fusedOneSimdgroupRows` rows on runs one
        // simdgroup per block (M5 Max, sliding 8K: 4 rows 50.7 vs 63.3 us, 8 rows
        // 88.8 vs 97.5); same bits.
        if subPartitions == nil && pass == .fused && b >= fusedOneSimdgroupRows && psub == 2 {
            psub = 1
            splitQPT = queriesPerThreadgroup(
                split: true, queries: l, gqa: gqa, headDim: headDim, elementSize: dtype.size)
            fusedStates =
                b * kvHeads * ((l + splitQPT - 1) / splitQPT) * splitQPT * gqa * plan.maxPartitions
                * (headDim + 2)
        }
        let qpt: Int
        switch pass {
        case .sequential:
            qpt = queriesPerThreadgroup(
                split: false, queries: l, gqa: gqa, headDim: headDim, elementSize: dtype.size)
        case .prefill: qpt = min(l, prefillQueriesPerThreadgroup)
        default: qpt = splitQPT
        }
        let blocks = pass == .prefill ? (qpt * gqa + 15) / 16 : (qpt * gqa + 7) / 8
        let groups = (l + qpt - 1) / qpt
        // The split family runs `psub` simdgroups per block (one per half
        // partition); the sequential and prefill kernels one.
        let passPSUB = pass == .split || pass == .fused ? psub : 1
        let threads = 32 * blocks * passPSUB
        let variant = Variant(
            dtype: dtype, headDim: headDim, gqa: gqa, qpt: qpt, storageBindings: storageBindings,
            tokenMajor: tokenMajor, extent: extent, relativeWidth: relativeWidth,
            write: write, psub: passPSUB)
        if pass == .fused {
            let (counters, states) = plan.workspace.buffers(
                counters: b * kvHeads * groups, states: fusedStates)
            let launch = launches.launch(
                .init(
                    pass: .fused, variant: variant, b: b, l: l, heads: queryHeads,
                    parts: plan.maxPartitions)
            ) {
                $0.prepare(
                    template: variant.template(pass: .fused),
                    grid: (kvHeads * groups * threads, b, plan.maxPartitions),
                    threadGroup: (threads, 1, 1),
                    outputShapes: [[b, l, queryHeads * headDim], [1]],
                    outputDTypes: [dtype, .int32])
            }
            let outputs = launch(
                [queries, relativeStates, proj, sinks, newKeys, newValues] + storageInputs + [
                    cache.tables, plan.rowInfo, cache.writeFence, counters, states,
                ], stream: stream)
            return Output(output: outputs[0], writeFence: outputs[1])
        }
        if pass == .prefill {
            let launch = launches.launch(
                .init(pass: .prefill, variant: variant, b: b, l: l, heads: queryHeads, parts: 0)
            ) {
                $0.prepare(
                    template: variant.template(pass: .prefill),
                    grid: (kvHeads * groups * threads, b, 1), threadGroup: (threads, 1, 1),
                    outputShapes: [[b, l, queryHeads * headDim], [1]],
                    outputDTypes: [dtype, .int32])
            }
            let outputs = launch(
                [queries, relativeStates, proj, sinks, newKeys, newValues] + storageInputs + [
                    cache.tables, plan.rowInfo, cache.writeFence,
                ], stream: stream)
            return Output(output: outputs[0], writeFence: outputs[1])
        }
        if pass == .sequential {
            let launch = launches.launch(
                .init(pass: .sequential, variant: variant, b: b, l: l, heads: queryHeads, parts: 0)
            ) {
                $0.prepare(
                    template: variant.template(pass: .sequential),
                    grid: (kvHeads * groups * threads, b, 1), threadGroup: (threads, 1, 1),
                    outputShapes: [[b, l, queryHeads * headDim], [1]],
                    outputDTypes: [dtype, .int32])
            }
            let outputs = launch(
                [queries, relativeStates, proj, sinks, newKeys, newValues] + storageInputs + [
                    cache.tables, plan.rowInfo, cache.writeFence,
                ], stream: stream)
            return Output(output: outputs[0], writeFence: outputs[1])
        }

        let maxParts = plan.maxPartitions
        let split = launches.launch(
            .init(pass: .split, variant: variant, b: b, l: l, heads: queryHeads, parts: maxParts)
        ) {
            $0.prepare(
                template: variant.template(pass: .split),
                grid: (kvHeads * groups * threads, b, maxParts), threadGroup: (threads, 1, 1),
                outputShapes: [[b, queryHeads, l, maxParts, headDim + 2], [1]],
                outputDTypes: [.float32, .int32])
        }
        let parts = split(
            [queries, relativeStates, proj, newKeys, newValues] + storageInputs + [
                cache.tables, plan.rowInfo, cache.writeFence,
            ], stream: stream)
        let merge = launches.launch(
            .init(
                pass: .merge, variant: variant.mergeVariant, b: b, l: l, heads: queryHeads, parts: 0
            )
        ) {
            $0.prepare(
                template: variant.template(pass: .merge),
                grid: (queryHeads * 32, l, b), threadGroup: (32, 1, 1),
                outputShapes: [[b, l, queryHeads * headDim]], outputDTypes: [dtype])
        }
        let output = merge([parts[0], plan.rowInfo, sinks], stream: stream)
        return Output(output: output[0], writeFence: parts[1])
    }

    /// A bucketed split dispatch also binds its items, segment ranges and the
    /// shared partials; with its shape buffers it fits Metal's 31 buffer slots
    /// with up to 15 segments, and always binds 15 (one kernel whatever the
    /// bucket's segment count).
    static let bucketBindingClasses = [15]
    /// Most segments one bucket of a bucketed call binds.
    public static let maximumBucketSegments = 15

    /// Validated, padded segment inputs (segments then value offsets).
    static func segmentInputs(
        _ segments: [MLXArray], classes: [Int], dtype: DType, kvHeads: Int, headDim: Int
    ) -> ([MLXArray], Int) {
        var offsets: [Int64] = []
        for segment in segments {
            let segmentShape = segment.shape
            precondition(
                segment.dtype == dtype && segmentShape.count == 5 && segmentShape[0] == 2
                    && segmentShape[2] == kvHeads && segmentShape[3] == pageSize
                    && segmentShape[4] == headDim, "segments must be [2, pages, KVH, 16, D]")
            offsets.append(Int64(segmentShape[1] * kvHeads * pageSize * headDim))
        }
        guard let bindings = classes.first(where: { $0 >= segments.count }) else {
            preconditionFailure("\(segments.count) segments exceed one dispatch")
        }
        let padded = segments + Array(repeating: segments[0], count: bindings - segments.count)
        offsets += Array(repeating: offsets[0], count: bindings - segments.count)
        offsets += Array(repeating: 0, count: max(0, 8 - offsets.count))
        return (padded + [segmentOffsets.array(offsets)], bindings)
    }

    /// A bucketed call: one split dispatch per bucket (each (row, partition)
    /// item computed exactly as the split pass computes it), chained through
    /// the write fence, then the merge pass. No fill: the first bucket
    /// allocates the partials and later buckets write their items' slots in
    /// place; the merge reads only the slots of the plan's (row, partition)
    /// items, and every item is in exactly one bucket.
    static func attendBuckets(
        _ buckets: [PagedKV.SegmentBucket], queries: MLXArray, relativeStates: MLXArray,
        proj: MLXArray, sinks: MLXArray, newKeys: MLXArray, newValues: MLXArray, cache: PagedKV,
        plan: Plan, variant base: Variant, kvHeads: Int, queryHeads: Int, b: Int, l: Int,
        stream: StreamOrDevice
    ) -> Output {
        let dtype = base.dtype
        let headDim = base.headDim
        let groups = (l + base.qpt - 1) / base.qpt
        let threads = 32 * base.blocks * base.psub
        let maxParts = plan.maxPartitions
        precondition(
            buckets.reduce(0) { $0 + $1.itemCount }
                == plan.rows.reduce(0) {
                    $0 + partitionCount(row: $1, queries: l, window: plan.window ?? 0)
                }, "a bucketed call runs each (row, partition) of the plan once")
        var fence = cache.writeFence
        var partials: MLXArray?
        for bucket in buckets where bucket.itemCount > 0 {
            let (inputs, bindings) = segmentInputs(
                bucket.segments, classes: bucketBindingClasses, dtype: dtype, kvHeads: kvHeads,
                headDim: headDim)
            precondition(bucket.ranges.size >= 2 * bindings, "bucket ranges are too short")
            let variant = Variant(
                dtype: dtype, headDim: headDim, gqa: base.gqa, qpt: base.qpt,
                storageBindings: bindings, tokenMajor: base.tokenMajor, extent: base.extent,
                relativeWidth: base.relativeWidth, write: base.write, psub: base.psub)
            let pass: Pass = partials == nil ? .splitItemsFirst : .splitItems
            let launch = launches.launch(
                .init(
                    pass: pass, variant: variant, b: b, l: l, heads: queryHeads,
                    parts: bucket.itemCount << 24 | maxParts)
            ) {
                $0.prepare(
                    template: variant.template(pass: pass),
                    grid: (kvHeads * groups * threads, bucket.itemCount, 1),
                    threadGroup: (threads, 1, 1),
                    outputShapes: pass == .splitItemsFirst
                        ? [[b, queryHeads, l, maxParts, headDim + 2], [1]] : [[1]],
                    outputDTypes: pass == .splitItemsFirst ? [.float32, .int32] : [.int32])
            }
            let common =
                [queries, relativeStates, proj, newKeys, newValues] + inputs + [
                    cache.tables, plan.rowInfo, fence, bucket.items, bucket.ranges,
                ]
            if let partials {
                fence = launch(common + [partials], stream: stream)[0]
            } else {
                let outputs = launch(common, stream: stream)
                partials = outputs[0]
                fence = outputs[1]
            }
        }
        let merge = launches.launch(
            .init(
                pass: .mergeAfter, variant: base.mergeVariant, b: b, l: l, heads: queryHeads,
                parts: 0)
        ) {
            $0.prepare(
                template: base.template(pass: .merge),
                grid: (queryHeads * 32, l, b), threadGroup: (32, 1, 1),
                outputShapes: [[b, l, queryHeads * headDim]], outputDTypes: [dtype])
        }
        let output = merge([partials!, plan.rowInfo, sinks, fence], stream: stream)
        return Output(output: output[0], writeFence: fence)
    }

    /// Convenience form that builds the plan for this call only.
    public static func attend(
        queries: MLXArray,
        newKeys: MLXArray,
        newValues: MLXArray,
        relativeStates: MLXArray,
        proj: MLXArray,
        sinks: MLXArray,
        cache: PagedKV,
        rows: [Row],
        window: Int?,
        scale: Float,
        write: Bool = false,
        layout: Layout = .headMajor,
        kernel: Kernel = .automatic,
        subPartitions: Int? = nil,
        stream: StreamOrDevice = .default
    ) -> Output {
        attend(
            queries: queries, newKeys: newKeys, newValues: newValues,
            relativeStates: relativeStates, proj: proj, sinks: sinks,
            cache: cache,
            plan: Plan(
                rows: rows, queries: queries.dim(layout == .tokenMajor ? 1 : 2), window: window,
                scale: scale),
            write: write, layout: layout, kernel: kernel, subPartitions: subPartitions,
            stream: stream)
    }

    /// Metal's threadgroup memory limit, which every variant must fit.
    static let threadgroupMemoryLimit = 32 * 1024

    /// Threadgroup bytes of a variant, mirroring the kernels' allocations:
    /// a staged page per simdgroup (split: keys and values `[S][D + 4]`; one
    /// per half partition when several blocks share it) or per threadgroup
    /// (sequential: keys `[D][S + 2]`, values `[S][D + 4]`) in the slab dtype,
    /// 8 x 24 float bias scratch per simdgroup, and the sequential kernel's
    /// staged queries (8 x D per simdgroup, slab dtype).
    /// `simdgroupsPerBlock` (split only): 2 stage a page per simdgroup of each
    /// block; 1 also holds a block's first-half state (`(D / 4 + 2) x 32` floats).
    static func threadgroupBytes(
        split: Bool, blocks: Int, headDim: Int, elementSize: Int, simdgroupsPerBlock: Int = 1
    ) -> Int {
        if split {
            let simdgroups = blocks * simdgroupsPerBlock
            let hold = simdgroupsPerBlock == 1 ? blocks * (headDim / 4 + 2) * 32 * 4 : 0
            // Shared pages (several blocks): one per half, also holding every
            // block's hand-off between halves.
            let shared = blocks > 1
            var pages =
                (shared ? simdgroupsPerBlock : simdgroups) * 2 * pageSize
                * (headDim + 4) * elementSize
            if shared && simdgroupsPerBlock == 2 {
                pages = max(pages, blocks * (headDim / 4 + 2) * 32 * 4)
            }
            return pages + simdgroups * 8 * 24 * 4 + hold
        }
        let scratch = blocks * 8 * 24 * 4
        let stage = (headDim * (pageSize + 2) + pageSize * (headDim + 4)) * elementSize
        return stage + blocks * 8 * headDim * elementSize + scratch
    }

    /// Up to 8 (split) or 16 (sequential) queries per threadgroup, fewer when
    /// the budget needs it. How queries are grouped never changes a query's
    /// arithmetic.
    static func queriesPerThreadgroup(
        split: Bool, queries: Int, gqa: Int, headDim: Int, elementSize: Int,
        simdgroupsPerBlock: Int = 1
    ) -> Int {
        var qpt = min(queries, split ? 8 : 16)
        while qpt > 1 {
            let blocks = (qpt * gqa + 7) / 8
            if threadgroupBytes(
                split: split, blocks: blocks, headDim: headDim, elementSize: elementSize,
                simdgroupsPerBlock: simdgroupsPerBlock) <= threadgroupMemoryLimit
            {
                break
            }
            qpt -= 1
        }
        return qpt
    }

    /// Most partitions `.automatic` runs fused when the threadgroup holds more
    /// than one block (verification windows): past it the one-dispatch tail
    /// (each group's last threadgroup merging every partition) costs more
    /// than the split kernel's merge pass. 4 rows x 8 tokens, fused vs split
    /// us: M5 Ultra 123 vs 95 at 8 partitions, 132 vs 105 at 9, 263 vs 175 at
    /// 16; M5 Max 139 vs 121, 159 vs 132, 323 vs 222. Same bits either way.
    public static let fusedMultiBlockMaximumPartitions = 7
    /// The cap in effect: `DARKBLOOM_INKLING2_ATTN_FUSED_CAP=0` turns it off,
    /// another number sets it (A/B).
    static let fusedMultiBlockCap: Int? = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_ATTN_FUSED_CAP"],
            let value = Int(raw)
        else { return fusedMultiBlockMaximumPartitions }
        return value > 0 ? value : nil
    }()

    /// Rows from which the one-dispatch kernel runs one simdgroup per block.
    public static let fusedOneSimdgroupRows = 3

    /// Simdgroups per block for a split-family call: `requested` (1 or 2), else
    /// 2 (one stream: sliding 8K 38.1 -> 28.1 us, full 99.9 -> 92.7 on the M5
    /// Max; split launches at 16-32 rows equal or faster), and 1 when two
    /// simdgroups' staging does not fit. Either count gives the same bits.
    static func subPartitionCount(
        requested: Int?, b: Int, kvHeads: Int, queries: Int, gqa: Int, maxPartitions: Int,
        headDim: Int, elementSize: Int
    ) -> Int {
        precondition(requested.map { $0 == 1 || $0 == 2 } ?? true, "1 or 2 sub-partitions")
        // One query per threadgroup is the smallest a launch can take.
        let fits =
            threadgroupBytes(
                split: true, blocks: (gqa + 7) / 8, headDim: headDim, elementSize: elementSize,
                simdgroupsPerBlock: 2) <= threadgroupMemoryLimit
        guard fits else { return 1 }
        return requested ?? 2
    }

    /// Partitions the split kernel launches for a row: from the first query's
    /// lowest key to the last query.
    static func partitionCount(row: Row, queries: Int, window: Int) -> Int {
        let lowest = window > 0 ? max(row.base, row.queryStart - window + 1) : row.base
        return (row.queryStart + queries - 1) / partitionTokens - lowest / partitionTokens + 1
    }

    static func dtypeName(_ dtype: DType) -> String {
        switch dtype {
        case .float16: "f16"
        case .bfloat16: "bf16"
        case .float32: "f32"
        default: "dt\(dtype.size)"
        }
    }

    enum Pass: Hashable {
        case split
        /// The split pass over a bucket's (row, partition) items: the first
        /// bucket allocates the call's partials, later ones write them in place.
        case splitItemsFirst
        case splitItems
        case fused
        case merge
        /// The merge pass ordered behind a fence (bucketed calls).
        case mergeAfter
        case sequential
        case prefill
    }

    /// Queries per prefill threadgroup (16 x GQA rows, three simdgroups at GQA 3).
    static let prefillQueriesPerThreadgroup = 16

    /// Everything a kernel's source and template depend on.
    struct Variant: Hashable {
        let dtype: DType
        let headDim: Int
        let gqa: Int
        let qpt: Int
        /// 0 for slabs, else the number of segment buffers bound.
        let storageBindings: Int
        let tokenMajor: Bool
        let extent: Int
        let relativeWidth: Int
        let write: Bool
        /// Simdgroups per block of the split family: 2 take a partition's two
        /// halves in parallel, 1 takes them in order. Same bits.
        var psub: Int = 1

        /// The merge pass depends on the dtype, head dim and layout only.
        var mergeVariant: Variant {
            Variant(
                dtype: dtype, headDim: headDim, gqa: 1, qpt: 1, storageBindings: 0,
                tokenMajor: tokenMajor, extent: 0, relativeWidth: 0, write: false)
        }

        var blocks: Int { (qpt * gqa + 7) / 8 }
        /// Simdgroups of a prefill threadgroup (16 rows each).
        var prefillSimdgroups: Int { (qpt * gqa + 15) / 16 }

        func template(pass: Pass) -> [(String, any KernelTemplateArg)] {
            if pass == .merge || pass == .mergeAfter {
                return [
                    ("T", dtype), ("D", headDim), ("PTOK", partitionTokens), ("TM", tokenMajor),
                ]
            }
            if pass == .prefill {
                return [
                    ("T", dtype), ("D", headDim), ("S", pageSize), ("GQA", gqa), ("QPT", qpt),
                    ("NS", prefillSimdgroups), ("REL", extent), ("DREL", relativeWidth),
                    ("BIAS", 1), ("TM", tokenMajor),
                ]
            }
            var template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("D", headDim), ("S", pageSize), ("GQA", gqa), ("QPT", qpt),
                ("NB", blocks), ("PTOK", partitionTokens), ("REL", extent),
                ("DREL", relativeWidth), ("BIAS", 1), ("HAS_WRITE", write), ("TM", tokenMajor),
            ]
            if pass == .fused { template.append(("FUSED_PARTS", fusedMaximumPartitions)) }
            if pass != .sequential { template.append(("PSUB", psub)) }
            return template
        }

        /// Kernel name; the template arguments make each instantiation unique.
        func name(pass: Pass) -> String {
            let storage = storageBindings == 0 ? "slab" : "seg\(storageBindings)"
            switch pass {
            case .merge, .mergeAfter:
                return "inkling2_paged_\(pass)_\(dtypeName(dtype))_d\(headDim)_"
                    + (tokenMajor ? "tm" : "hm")
            case .split, .splitItemsFirst, .splitItems, .fused, .sequential, .prefill:
                return "inkling2_paged_\(pass)_\(dtypeName(dtype))_d\(headDim)_g\(gqa)_q\(qpt)_"
                    + "\(storage)_\(tokenMajor ? "tm" : "hm")" + (write ? "_write" : "")
                    + (psub > 1 ? "_p\(psub)" : "")
            }
        }
    }

    /// Input names and page accessor of a storage binding count (0: slabs);
    /// `mapped`: entries are page ids, located through the bindings' `segrange`.
    static func storageBinding(_ count: Int, mapped: Bool = false)
        -> (names: [String], declaration: String)
    {
        guard count > 0 else {
            return (
                ["kcache", "vcache"],
                "const ink2::SlabPages<T> pages{kcache, vcache, (size_t)KVH * S * D};"
            )
        }
        let names = (0 ..< count).map { "seg\($0)" }
        if mapped {
            return (
                names + ["segoff"],
                "const ink2::RangedSegmentPages<T, \(count)> pages{{"
                    + "\(names.joined(separator: ", "))}, segoff, segrange, (size_t)KVH * S * D};"
            )
        }
        return (
            names + ["segoff"],
            "const ink2::SegmentPages<T, \(count)> pages{{\(names.joined(separator: ", "))}, "
                + "segoff, (size_t)KVH * S * D};"
        )
    }

    /// The split core's row, partition and row count: from the grid, or from
    /// a bucket's (row, partition) items.
    static let gridItems = """
        const int b = threadgroup_position_in_grid.y;
        const int z = threadgroup_position_in_grid.z;
        const int maxparts = threadgroups_per_grid.z;
        const int nrows = threadgroups_per_grid.y;
        """
    static let bucketItems = """
        const int b = items[2 * threadgroup_position_in_grid.y];
        const int z = items[2 * threadgroup_position_in_grid.y + 1];
        const int maxparts = rowinfo[5];
        const int nrows = rowinfo[6];
        """

    /// Compiled kernels by name.
    final class KernelCache: @unchecked Sendable {
        private let lock = NSLock()
        private var kernels: [String: MLXFast.MLXFastKernel] = [:]

        /// Kernels built so far (one Metal library each): tests check that a
        /// workload compiles nothing new after its warm-up.
        var count: Int { lock.withLock { kernels.count } }

        func kernel(_ pass: Pass, _ variant: Variant) -> MLXFast.MLXFastKernel {
            let name = variant.name(pass: pass)
            return lock.withLock {
                if let made = kernels[name] { return made }
                let made: MLXFast.MLXFastKernel
                switch pass {
                case .split, .splitItemsFirst, .splitItems, .fused, .sequential, .prefill:
                    let items = pass == .splitItemsFirst || pass == .splitItems
                    let binding = storageBinding(variant.storageBindings, mapped: items)
                    let body: String
                    switch pass {
                    case .split, .splitItemsFirst, .splitItems:
                        body = Inkling2PagedAttentionMetal.splitBody
                    case .fused: body = Inkling2PagedAttentionMetal.fusedBody
                    case .prefill: body = Inkling2PagedAttentionMetal.prefillBody
                    default: body = Inkling2PagedAttentionMetal.sequentialBody
                    }
                    let pageInputs = binding.names.filter { $0 != "segoff" }
                    // Inputs after the tables, those the kernel writes in place,
                    // and the outputs.
                    let extra: [String]
                    let workspace: [String]
                    let outputs: [String]
                    switch pass {
                    case .fused:
                        (extra, workspace, outputs) = (
                            ["counters", "work"], ["counters", "work"], ["out", "fence"]
                        )
                    case .split: (extra, workspace, outputs) = ([], [], ["partials", "fence"])
                    case .splitItemsFirst:
                        (extra, workspace, outputs) = (
                            ["items", "segrange"], [], ["partials", "fence"]
                        )
                    case .splitItems:
                        (extra, workspace, outputs) = (
                            ["items", "segrange", "partials"], ["partials"], ["fence"]
                        )
                    default: (extra, workspace, outputs) = ([], [], ["out", "fence"])
                    }
                    // The split pass never reads the sinks (the merge applies them).
                    let perQuery =
                        pass == .split || items
                        ? ["q", "rel", "proj", "knew", "vnew"]
                        : ["q", "rel", "proj", "sinks", "knew", "vnew"]
                    made = MLXFast.metalKernel(
                        name: name,
                        inputNames: perQuery + binding.names
                            + ["tables", "rowinfo", "wfence"] + extra,
                        outputNames: outputs,
                        source: body.replacingOccurrences(
                            of: "PAGES_DECL", with: binding.declaration
                        ).replacingOccurrences(
                            of: "ITEM_DECL", with: items ? bucketItems : gridItems),
                        header: Inkling2PagedAttentionMetal.header
                            + (pass == .prefill ? Inkling2PagedAttentionMetal.prefillHeader : ""),
                        mutableInputs: (variant.write ? pageInputs : []) + workspace)
                case .merge, .mergeAfter:
                    // `wfence` only orders the merge behind a bucketed call's
                    // split dispatches, which write the partials in place.
                    made = MLXFast.metalKernel(
                        name: name,
                        inputNames: ["partials", "rowinfo", "sinks"]
                            + (pass == .mergeAfter ? ["wfence"] : []),
                        outputNames: ["out"], source: Inkling2PagedAttentionMetal.mergeBody,
                        header: Inkling2PagedAttentionMetal.header)
                }
                kernels[name] = made
                return made
            }
        }
    }

    static let kernels = KernelCache()

    /// A launch's kernel, template, grid and output shapes.
    struct LaunchKey: Hashable {
        let pass: Pass
        let variant: Variant
        let b: Int
        let l: Int
        let heads: Int
        let parts: Int
    }

    /// Prepared launches: the template, grid and outputs of a call shape are
    /// bound once, so a call only passes its inputs.
    final class LaunchCache: @unchecked Sendable {
        private let lock = NSLock()
        private var launches: [LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
        /// Shapes grow with batch composition and context; start over rather
        /// than keep every one.
        static let capacity = 4096

        func launch(
            _ key: LaunchKey,
            _ prepare: (MLXFast.MLXFastKernel) -> MLXFast.MLXFastKernel.Launch
        ) -> MLXFast.MLXFastKernel.Launch {
            lock.lock()
            if let made = launches[key] {
                lock.unlock()
                return made
            }
            lock.unlock()
            let made = prepare(kernels.kernel(key.pass, key.variant))
            lock.withLock {
                if launches.count >= Self.capacity { launches.removeAll(keepingCapacity: true) }
                launches[key] = made
            }
            return made
        }
    }

    static let launches = LaunchCache()

    /// Device copies of segment value offsets: a pool's segments keep their
    /// sizes, so calls reuse a handful of these.
    final class SegmentOffsets: @unchecked Sendable {
        private let lock = NSLock()
        private var arrays: [[Int64]: MLXArray] = [:]

        func array(_ offsets: [Int64]) -> MLXArray {
            lock.withLock {
                if let made = arrays[offsets] { return made }
                if arrays.count >= 64 { arrays.removeAll(keepingCapacity: true) }
                let made = MLXArray(offsets)
                arrays[offsets] = made
                return made
            }
        }
    }

    static let segmentOffsets = SegmentOffsets()
}
