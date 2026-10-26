// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXFast

/// How a GLM payload addresses paged history. Both copy identical bytes into
/// identically shaped outputs; the binding takes its target's choice for its
/// lifetime so step pricing and execution agree.
public enum GLM5NextPagedMetadataReads: Sendable, Hashable {
    /// Host-built (token, page, slot) records per metadata token.
    case hostRecords
    /// The metadata kernel also translates tokens through
    /// the row's device page table, so host work is O(pages).
    case deviceTable
}

/// One-sided transfers for the main GLM payload row. Neither path expands K/V
/// projections or gathers the complete latent plane. All roots join the engine's
/// existing paged step owner; no eval/synchronize occurs while building graphs.
enum GLM5NextPagedTransfers {
    private static let readV = MLXFast.metalKernel(
        name: "glm5_next_read_metadata", inputNames: ["storage", "records", "output", "previous"],
        outputNames: ["fence"],
        source: """
            const int d = int(thread_position_in_grid.x);
            const int r = int(thread_position_in_grid.y);
            const int token = records[r * 3], page = records[r * 3 + 1], slot = records[r * 3 + 2];
            const size_t source = ((size_t)page * S + slot) * D;
            device T* dst = output;
            dst[(size_t)token * D + d] = storage[VBASE + source + d];
            if (d == 0 && r == 0) fence[0] = previous[0] + 1;
            """, ensureRowContiguous: true, mutableInputs: ["output"])

    /// Segments bound by one launch: `PagedSegmentDispatchPlan`'s binding
    /// classes. Unused slots repeat the first storage with empty bounds.
    static let maximumBindings = PagedSegmentDispatchPlan.maximumBindings
    static let pageRunThreads = 256
    static let selectedRowLanes = 32
    static let selectedRowsPerGroup = 8

    private enum Kernel: Hashable {
        /// Page runs of a row's history; `first` allocates the output.
        case history(bindings: Int, first: Bool)
        case selected(bindings: Int)
    }

    private static let kernelLock = NSLock()
    nonisolated(unsafe) private static var kernels: [Kernel: MLXFast.MLXFastKernel] = [:]

    private static func kernel(_ kind: Kernel) -> MLXFast.MLXFastKernel {
        kernelLock.withLock {
            if let cached = kernels[kind] { return cached }
            let made: MLXFast.MLXFastKernel
            switch kind {
            case .history(let bindings, let first):
                let segments = (0 ..< bindings).map { "segment\($0)" }
                made = MLXFast.metalKernel(
                    name: "glm5_next_read_history_pages\(first ? "" : "_into")_bound\(bindings)",
                    inputNames: segments
                        + (first
                            ? ["table", "positions", "previous", "bounds", "bases"]
                            : ["table", "positions", "output", "previous", "bounds", "bases"]),
                    outputNames: first ? ["output", "fence"] : ["fence"],
                    source: historySource(bindings: bindings), ensureRowContiguous: true,
                    mutableInputs: first ? [] : ["output"])
            case .selected(let bindings):
                made = MLXFast.metalKernel(
                    name: "glm5_next_read_selected_latents_bound\(bindings)",
                    inputNames: (0 ..< bindings).map { "segment\($0)" }
                        + ["table", "indices", "output", "previous", "bounds"],
                    outputNames: ["fence"], source: selectedSource(bindings: bindings),
                    ensureRowContiguous: true, mutableInputs: ["output"])
            }
            kernels[kind] = made
            return made
        }
    }

    /// One branch per bound segment, testing `page` against its bounds.
    private static func branches(bindings: Int, body: (Int) -> String) -> String {
        (0 ..< bindings).map { index in
            let test = "page > bounds[\(2 * index)] && page < bounds[\(2 * index + 1)]"
            return "\(index == 0 ? "if" : "else if") (\(test)) {\n\(body(index))\n}"
        }.joined(separator: " ")
    }

    /// One threadgroup per listed page position. A page's rows are one run in
    /// its segment and in the output.
    private static func historySource(bindings: Int) -> String {
        let copies = branches(bindings: bindings) { index in
            """
            const size_t source =
                size_t(bases[\(index)]) + (size_t)(page - bounds[\(2 * index)]) * S * D;
            for (int e = lane; e < rows * D; e += THREADS) {
                dst[e] = static_cast<O>(segment\(index)[source + e]);
            }
            """
        }
        return """
            const int position = positions[threadgroup_position_in_grid.y];
            const int page = table[position];
            const int rows = min(S, table[table_shape[0] - 1] - position * S);
            const int lane = int(thread_position_in_threadgroup.x);
            device O* dst = output + (size_t)position * S * D;
            if (rows > 0) {
            \(copies)
            }
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            """
    }

    /// One SIMD-group copies each selection row whose page a bound segment holds.
    private static func selectedSource(bindings: Int) -> String {
        let copies = branches(bindings: bindings) { index in
            """
            const size_t source = ((size_t)(page - bounds[\(2 * index)]) * S + token % S) * D;
            for (int d = lane; d < D; d += LANES) { dst[d] = segment\(index)[source + d]; }
            """
        }
        return """
            const int r = int(thread_position_in_grid.y);
            const int token = r < int(indices_shape[0]) ? indices[r] : -1;
            if (token >= 0 && token < table[table_shape[0] - 1]) {
                const int page = table[token / S];
                const int lane = int(thread_position_in_threadgroup.x);
                device T* dst = output + (size_t)r * D;
            \(copies)
            }
            if (thread_position_in_grid.x == 0 && r == 0) fence[0] = previous[0] + 1;
            """
    }

    /// `segments` (at most `maximumBindings`) padded to their binding class:
    /// unused slots repeat the first storage with empty page bounds.
    private static func bindings(_ segments: [(storage: MLXArray, pages: Range<Int>)]) -> (
        storages: [MLXArray], bounds: MLXArray
    ) {
        precondition(!segments.isEmpty && segments.count <= maximumBindings)
        let size = PagedSegmentDispatchPlan.bindingClasses.first { $0 >= segments.count }!
        var storages = segments.map(\.storage)
        storages += repeatElement(storages[0], count: size - segments.count)
        var bounds = segments.flatMap { [Int32($0.pages.lowerBound), Int32($0.pages.upperBound)] }
        bounds += repeatElement(Int32(0), count: 2 * (size - segments.count))
        return (storages, MLXArray(bounds))
    }

    private static func batches<T>(_ items: [T]) -> [ArraySlice<T>] {
        stride(from: 0, to: items.count, by: maximumBindings).map {
            items[$0 ..< min($0 + maximumBindings, items.count)]
        }
    }

    private static let complete = MLXFast.metalKernel(
        name: "glm5_next_payload_read_complete", inputNames: ["previous"], outputNames: ["witness"],
        source: "witness[0] = previous[0] + 1;", ensureRowContiguous: true)

    /// The row's device page table with the history length appended. Kernels
    /// read the length at run time: MLX compiles one Metal library per
    /// distinct template tuple, so a templated length recompiled every step.
    static func table(row: PagedSequenceKV) -> MLXArray {
        MLXArray(row.table + [Int32(row.absoluteOffset)])
    }

    /// The row's first `count` metadata rows `[1, count, V]`, read through its
    /// device page table. Exactly one segment owns each token's (non-poison)
    /// page, so every row is written once, as by the record path.
    static func tableValues(
        row: PagedSequenceKV, table: MLXArray, count: Int, work: CBv2PagedAttentionStepOwner
    ) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        return historyRuns(
            row: row, table: table, count: count, width: group.key.valueHeadDim,
            base: \.valueOffset, dtype: group.dtype, work: work)
    }

    /// The row's first `count` latent rows `[1, count, D]` in `dtype`, read
    /// through its device page table like `tableValues`. The FP32 pages hold
    /// losslessly widened latents, so narrowing to their dtype is exact.
    static func latentHistory(
        row: PagedSequenceKV, table: MLXArray, count: Int, dtype: DType,
        work: CBv2PagedAttentionStepOwner
    ) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        return historyRuns(
            row: row, table: table, count: count, width: group.key.headDim, base: { _ in 0 },
            dtype: dtype, work: work)
    }

    /// One segment's share of a row's first pages: their table positions.
    struct SegmentRun: Equatable {
        let segment: Int
        let positions: [Int32]
    }

    /// The row's page positions below `pages`, grouped by owning segment in
    /// ascending segment order.
    static func segmentRuns(row: PagedSequenceKV, pages: Int) -> [SegmentRun] {
        let layout = row.pool.group(row.groupKey).segmentLayout!
        var bySegment: [Int: [Int32]] = [:]
        for position in 0 ..< pages {
            bySegment[layout.segmentIndex(page: row.table[position]), default: []]
                .append(Int32(position))
        }
        return bySegment.keys.sorted().map { SegmentRun(segment: $0, positions: bySegment[$0]!) }
    }

    /// One launch per `maximumBindings` segments the row's history occupies,
    /// each covering only those segments' pages.
    private static func historyRuns(
        row: PagedSequenceKV, table: MLXArray, count: Int, width: Int,
        base: (PagedKVSegment) -> Int, dtype: DType, work: CBv2PagedAttentionStepOwner
    ) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        // The record path's per-token page checks, once per page.
        let pages = (count + group.pageSize - 1) / group.pageSize
        precondition(
            count > 0 && row.table.count >= pages
                && row.table.prefix(pages).allSatisfy(group.isAllocatable))
        let threads = pageRunThreads
        let template: [(String, any KernelTemplateArg)] = [
            ("O", dtype), ("D", width), ("S", group.pageSize), ("THREADS", threads),
        ]
        var output: MLXArray?
        var fence = group.writeFence
        for batch in batches(segmentRuns(row: row, pages: pages)) {
            let segments = batch.map { run -> PagedKVSegment in
                guard let segment = group.segments[run.segment] else {
                    preconditionFailure("transfer names an uncommitted segment")
                }
                return segment
            }
            let (storages, bounds) = bindings(segments.map { ($0.storage, $0.pages) })
            let bases = MLXArray(
                segments.map { Int64(base($0)) }
                    + repeatElement(Int64(0), count: storages.count - segments.count))
            let positions = MLXArray(batch.flatMap(\.positions))
            let grid = (threads, positions.size, 1)
            if let output {
                fence =
                    kernel(.history(bindings: storages.count, first: false))(
                        storages + [table, positions, output, fence, bounds, bases],
                        template: template, grid: grid, threadGroup: (threads, 1, 1),
                        outputShapes: [[1]], outputDTypes: [.int32])[0]
            } else {
                let first = kernel(.history(bindings: storages.count, first: true))(
                    storages + [table, positions, fence, bounds, bases],
                    template: template, grid: grid, threadGroup: (threads, 1, 1),
                    outputShapes: [[1, count, width], [1]], outputDTypes: [dtype, .int32])
                output = first[0]
                fence = first[1]
            }
            work.retainRoots(segments.map(\.storage) + [positions, bounds, bases, fence])
        }
        return completeRead(output: output!, fence: fence, group: group, work: work)
    }

    private static func completeRead(
        output: MLXArray, fence: MLXArray, group: PagedKVGroup, work: CBv2PagedAttentionStepOwner,
        extraRoots: [MLXArray] = []
    ) -> MLXArray {
        let fence =
            complete(
                [fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
        group.writeFence = fence
        let result = depends(input: output, dependencies: [fence])
        work.retainRoots([output, result, fence] + extraRoots)
        return result
    }

    static func values(row: PagedSequenceKV, count: Int, work: CBv2PagedAttentionStepOwner)
        -> MLXArray
    {
        let group = row.pool.group(row.groupKey)
        let size = group.pageSize
        let slots = (0 ..< count).map { token in
            row.table[token / size] * Int32(size) + Int32(token % size)
        }
        let output = MLXArray.zeros([1, count, group.key.valueHeadDim], dtype: group.dtype)
        var fence = group.writeFence
        for (segment, triples) in PagedSegmentTransfers.buckets(group: group, slots: slots) {
            let records = PagedSegmentTransfers.records(triples)
            fence =
                readV(
                    [segment.storage, records, output, fence],
                    template: [
                        ("T", group.dtype), ("D", group.key.valueHeadDim), ("S", size),
                        ("VBASE", segment.valueOffset),
                    ],
                    grid: (group.key.valueHeadDim, triples.count / 3, 1),
                    threadGroup: (min(256, group.key.valueHeadDim), 1, 1),
                    outputShapes: [[1]], outputDTypes: [.int32])[0]
            work.retainRoots([segment.storage, records, fence])
        }
        return completeRead(output: output, fence: fence, group: group, work: work)
    }

    /// `table` comes from `table(row:)`. Ids outside `0 ..< length` stay zero.
    static func keys(
        row: PagedSequenceKV, indices: MLXArray, table: MLXArray,
        segmentIDs: [Int], work: CBv2PagedAttentionStepOwner
    ) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        let output = MLXArray.zeros([1, indices.size, group.key.headDim], dtype: group.dtype)
        var fence = group.writeFence
        for batch in batches(segmentIDs) {
            let segments = batch.map { group.segments[$0]! }
            fence = selectedLatents(
                segments: segments.map { ($0.storage, $0.pages) }, table: table, indices: indices,
                output: output, previous: fence, pageSize: group.pageSize)
            work.retainRoots(segments.map(\.storage) + [fence])
        }
        return completeRead(
            output: output, fence: fence, group: group, work: work, extraRoots: [table, indices])
    }

    /// Geometry only, never history length or segment identity: one pipeline
    /// per binding class serves every step.
    static func selectedLatentTemplate(dtype: DType, headDim: Int, pageSize: Int)
        -> [(String, any KernelTemplateArg)]
    {
        [("T", dtype), ("D", headDim), ("S", pageSize), ("LANES", selectedRowLanes)]
    }

    /// One launch copying the selection rows whose pages lie in `segments`
    /// (at most `maximumBindings`) into `output` `[1, W, D]`.
    static func selectedLatents(
        segments: [(storage: MLXArray, pages: Range<Int>)], table: MLXArray, indices: MLXArray,
        output: MLXArray, previous: MLXArray, pageSize: Int
    ) -> MLXArray {
        let (storages, bounds) = bindings(segments)
        return kernel(.selected(bindings: storages.count))(
            storages + [table, indices, output, previous, bounds],
            template: selectedLatentTemplate(
                dtype: output.dtype, headDim: output.dim(2), pageSize: pageSize),
            grid: (selectedRowLanes, indices.size, 1),
            threadGroup: (selectedRowLanes, selectedRowsPerGroup, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
    }
}

/// Graph-only access bounded by one already-admitted engine step. The step
/// owns the charged roots even after this short-lived model-side handle drops.
package final class GLM5NextPagedPayload {
    private let cache: PagedLayerCache
    private let work: CBv2PagedAttentionStepOwner
    private let rows: [PagedSequenceKV]
    private let metadata: [MLXArray]
    private let tables: [MLXArray]
    private let segments: [[Int]]
    private let lengths: [Int]
    private var reads: [Int]
    private var historyRead: [Bool]

    /// Tables and segments must be built after the step's writes (including
    /// any copy-on-write page replacement) and before any read.
    fileprivate init(
        cache: PagedLayerCache, work: CBv2PagedAttentionStepOwner,
        rows: [PagedSequenceKV], tables: [MLXArray], segments: [[Int]], metadata: [MLXArray]
    ) {
        self.cache = cache
        self.work = work
        self.rows = rows
        self.tables = tables
        self.segments = segments
        self.metadata = metadata
        lengths = rows.map(\.absoluteOffset)
        reads = Array(repeating: 0, count: rows.count)
        historyRead = Array(repeating: false, count: rows.count)
    }

    fileprivate static func segmentIDs(_ row: PagedSequenceKV) -> [Int] {
        Array(
            Set(
                row.table.map {
                    row.pool.group(row.groupKey).segmentLayout!.segmentIndex(page: $0)
                })
        ).sorted()
    }

    package func values(row: Int) -> MLXArray { metadata[row] }

    /// Whether this row's `sparseQueries` read its whole latent history once
    /// (`latentHistory`) rather than one selection row per query.
    package func readsLatentHistory(row: Int, sparseQueries: Int) -> Bool {
        guard rows.indices.contains(row), let binding = cache.glm5NextModelBinding else {
            return false
        }
        return plan(row: row) == .history
            && binding.readsLatentHistory(sparseQueries: sparseQueries, historyLength: lengths[row])
    }

    /// The reads this step priced for the row at this layer.
    private func plan(row: Int) -> GLM5NextPagedModelBinding.ReadPlan? {
        cache.glm5NextModelBinding?.readPlan(
            work: work, layer: cache.layerIndex, serial: rows[row].serial)
    }

    /// The row's whole latent history `[1, N, C]` in the latents' `dtype`
    /// (FP32 pages widened them losslessly), read once through its page
    /// table, so its sparse queries index it in place like a contiguous row
    /// and retain no per-query copy.
    package func latentHistory(row: Int, sparseQueries: Int, dtype: DType) throws -> MLXArray {
        guard rows.indices.contains(row), cache.pool.activeAttentionWork === work,
            !work.completed, !historyRead[row], reads[row] == 0,
            let binding = cache.glm5NextModelBinding, plan(row: row) == .history,
            [.float16, .bfloat16, .float32].contains(dtype),
            let descriptor = work.descriptors[
                .init(layer: cache.layerIndex, serial: rows[row].serial)],
            sparseQueries <= descriptor.range.count,
            binding.readsLatentHistory(sparseQueries: sparseQueries, historyLength: lengths[row]),
            rows[row].absoluteOffset == lengths[row]
        else {
            throw CBv2KVError.backendIneligible(reason: "stale or unpriced GLM latent history read")
        }
        historyRead[row] = true
        binding.recordHistoryRows(lengths[row])
        return GLM5NextPagedTransfers.latentHistory(
            row: rows[row], table: tables[row], count: lengths[row], dtype: dtype, work: work)
    }

    /// One query's full selection row, `selectionWidth` ids.
    package func selectedKeys(row: Int, indices: MLXArray) throws -> MLXArray {
        try selectedKeys(row: row, indices: indices) { $0.selectionWidth }
    }

    /// One query's selection row without its static padding columns, which the
    /// selector fills with -1 in every row. `staticPadding` is that selection's
    /// four-aligned budget suffix (`Selection.staticPaddingColumns`); the step
    /// still prices the full row.
    package func selectedKeys(row: Int, indices: MLXArray, staticPadding: Range<Int>) throws
        -> MLXArray
    {
        try selectedKeys(row: row, indices: indices) { binding in
            // The selection row is the four-aligned token budget, then a 3-id tail.
            let budget = binding.selectionWidth - 3
            guard staticPadding.lowerBound >= 0, staticPadding.lowerBound % 4 == 0,
                staticPadding.upperBound == budget
            else { return nil }
            return binding.selectionWidth - staticPadding.count
        }
    }

    private func selectedKeys(
        row: Int, indices: MLXArray, width: (GLM5NextPagedModelBinding) -> Int?
    ) throws -> MLXArray {
        guard rows.indices.contains(row), cache.pool.activeAttentionWork === work,
            !work.completed, indices.ndim == 2, indices.dim(0) == 1,
            indices.dtype == .int32, let binding = cache.glm5NextModelBinding,
            indices.size == width(binding), !historyRead[row],
            case .selectedRows(let priced)? = plan(row: row), reads[row] < priced,
            rows[row].absoluteOffset == lengths[row]
        else {
            throw CBv2KVError.backendIneligible(
                reason: "stale or oversized GLM selected latent read")
        }
        reads[row] += 1
        cache.glm5NextModelBinding?.recordSelectedRows(indices.size)
        return GLM5NextPagedTransfers.keys(
            row: rows[row], indices: indices.flattened(), table: tables[row],
            segmentIDs: segments[row], work: work)
    }

    package func retainResult(_ result: MLXArray) { work.retainRoots([result]) }
}

extension PagedLayerCache {
    package func matchesGLM5NextTarget(_ identity: ObjectIdentifier) -> Bool {
        glm5NextModelBinding?.matches(identity) == true
    }

    package func beginGLM5NextPayload(
        keys: MLXArray, values: MLXArray,
        targetIdentity: ObjectIdentifier
    ) throws -> GLM5NextPagedPayload {
        guard matchesGLM5NextTarget(targetIdentity), let binding = glm5NextModelBinding,
            let work = pool.activeAttentionWork,
            pool.activeAttentionLayer == nil, kind.attention == .full,
            keys.ndim == 4, values.ndim == 4, keys.dim(0) == rows.count,
            keys.shape == [rows.count, 1, keys.dim(2), kind.headDim],
            values.shape == [rows.count, 1, keys.dim(2), kind.valueHeadDim],
            keys.dtype == .float32, values.dtype == .float32,
            work.consume(
                layer: layerIndex, rows: rows.compactMap { $0 as? PagedSequenceKV },
                queries: keys.dim(2), softcap: false)
        else {
            throw CBv2KVError.backendIneligible(
                reason: "GLM payload lacks its exact admitted cache ticket")
        }
        pool.activeAttentionLayer = layerIndex
        defer { pool.activeAttentionLayer = nil }
        let native = rows.map { $0 as! PagedSequenceKV }
        for (index, row) in native.enumerated() {
            row.write(keys: keys[index].asType(.float32), values: values[index].asType(.float32))
        }
        try pool.writeValidation.check()
        let reads = binding.metadataReads
        let tables = native.map(GLM5NextPagedTransfers.table(row:))
        let segments = native.map(GLM5NextPagedPayload.segmentIDs)
        var metadata: [MLXArray] = []
        for (index, row) in native.enumerated() {
            guard
                work.authorizeRead(layer: layerIndex, row: row, start: 0, count: row.absoluteOffset)
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "GLM index read does not match the planned frontier")
            }
            switch reads {
            case .hostRecords:
                metadata.append(
                    GLM5NextPagedTransfers.values(row: row, count: row.absoluteOffset, work: work))
            case .deviceTable:
                metadata.append(
                    GLM5NextPagedTransfers.tableValues(
                        row: row, table: tables[index], count: row.absoluteOffset, work: work))
            }
            binding.recordMetadataRows(row.absoluteOffset)
        }
        advanceGLM5NextOffsets(by: keys.dim(2))
        work.retainRoots(tables)
        return .init(
            cache: self, work: work, rows: native, tables: tables, segments: segments,
            metadata: metadata)
    }

    package func recordGLM5NextFailure(_ error: any Error) { pool.writeValidation.record(error) }
}
