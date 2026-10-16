// Copyright © 2026 Eigen Labs.

import MLX
import MLXFast

/// How a GLM payload addresses paged history. Both copy identical bytes into
/// identically shaped outputs; the binding takes its target's choice for its
/// lifetime so step pricing and execution agree.
public enum GLM5NextPagedMetadataReads: Sendable, Hashable {
    /// Host-built (token, page, slot) records per metadata token.
    case hostRecords
    /// Opt-in candidate: the metadata kernel also translates tokens through
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

    private static let readVTable = MLXFast.metalKernel(
        name: "glm5_next_read_metadata_table",
        inputNames: ["storage", "table", "output", "previous"], outputNames: ["fence"],
        source: """
            const int d = int(thread_position_in_grid.x);
            const int token = int(thread_position_in_grid.y);
            if (token < table[table_shape[0] - 1]) {
                const int page = table[token / S];
                if (page > FIRST && page < END) {
                    const size_t source = ((size_t)(page - FIRST) * S + token % S) * D;
                    device T* dst = output;
                    dst[(size_t)token * D + d] = storage[VBASE + source + d];
                }
            }
            if (d == 0 && token == 0) fence[0] = previous[0] + 1;
            """, ensureRowContiguous: true, mutableInputs: ["output"])

    private static let readKTable = MLXFast.metalKernel(
        name: "glm5_next_read_selected_latents_table",
        inputNames: ["storage", "table", "indices", "output", "previous"], outputNames: ["fence"],
        source: """
            const int d = int(thread_position_in_grid.x);
            const int r = int(thread_position_in_grid.y);
            const int token = indices[r];
            if (token >= 0 && token < table[table_shape[0] - 1]) {
                const int page = table[token / S];
                if (page > FIRST && page < END) {
                    const size_t source = ((size_t)(page - FIRST) * S + token % S) * D;
                    device T* dst = output;
                    dst[(size_t)r * D + d] = storage[source + d];
                }
            }
            if (d == 0 && r == 0) fence[0] = previous[0] + 1;
            """, ensureRowContiguous: true, mutableInputs: ["output"])

    private static let complete = MLXFast.metalKernel(
        name: "glm5_next_payload_read_complete", inputNames: ["previous"], outputNames: ["witness"],
        source: "witness[0] = previous[0] + 1;", ensureRowContiguous: true)

    /// The row's device page table with the history length appended. Kernels
    /// read the length at run time: MLX compiles one Metal library per
    /// distinct template tuple, so a templated length recompiled every step.
    static func table(row: PagedSequenceKV) -> MLXArray {
        MLXArray(row.table + [Int32(row.absoluteOffset)])
    }

    /// Exactly one segment owns each token's (non-poison) page, so every
    /// metadata row is written once, as by the record path.
    static func tableValues(
        row: PagedSequenceKV, table: MLXArray, segmentIDs: [Int], count: Int,
        work: CBv2PagedAttentionStepOwner
    ) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        // The record path's per-token page checks, once per page.
        let pages = (count + group.pageSize - 1) / group.pageSize
        precondition(
            row.table.count >= pages && row.table.prefix(pages).allSatisfy(group.isAllocatable))
        let output = MLXArray.zeros([1, count, group.key.valueHeadDim], dtype: group.dtype)
        var fence = group.writeFence
        for id in segmentIDs {
            guard let segment = group.segments[id] else {
                preconditionFailure("transfer names an uncommitted segment")
            }
            fence =
                readVTable(
                    [segment.storage, table, output, fence],
                    template: [
                        ("T", group.dtype), ("D", group.key.valueHeadDim), ("S", group.pageSize),
                        ("VBASE", segment.valueOffset), ("FIRST", segment.pages.lowerBound),
                        ("END", segment.pages.upperBound),
                    ],
                    grid: (group.key.valueHeadDim, count, 1),
                    threadGroup: (min(256, group.key.valueHeadDim), 1, 1),
                    outputShapes: [[1]], outputDTypes: [.int32])[0]
            work.retainRoots([segment.storage, fence])
        }
        return completeRead(output: output, fence: fence, group: group, work: work)
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
        for id in segmentIDs {
            let segment = group.segments[id]!
            fence = selectedLatents(
                storage: segment.storage, table: table, indices: indices, output: output,
                previous: fence,
                template: selectedLatentTemplate(
                    dtype: group.dtype, headDim: group.key.headDim, pageSize: group.pageSize,
                    pages: segment.pages))
            work.retainRoots([segment.storage, fence])
        }
        return completeRead(
            output: output, fence: fence, group: group, work: work, extraRoots: [table, indices])
    }

    /// Segment identity, never history length: one pipeline per segment serves
    /// every step.
    static func selectedLatentTemplate(
        dtype: DType, headDim: Int, pageSize: Int, pages: Range<Int>
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("T", dtype), ("D", headDim), ("S", pageSize),
            ("FIRST", pages.lowerBound), ("END", pages.upperBound),
        ]
    }

    /// One segment's selected-latent copy into `output` `[1, W, D]`.
    static func selectedLatents(
        storage: MLXArray, table: MLXArray, indices: MLXArray, output: MLXArray,
        previous: MLXArray, template: [(String, any KernelTemplateArg)]
    ) -> MLXArray {
        let headDim = output.dim(2)
        return readKTable(
            [storage, table, indices, output, previous], template: template,
            grid: (headDim, indices.size, 1), threadGroup: (min(256, headDim), 1, 1),
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
            indices.size == width(binding),
            let descriptor = work.descriptors[
                .init(layer: cache.layerIndex, serial: rows[row].serial)],
            reads[row] < descriptor.range.count, rows[row].absoluteOffset == lengths[row]
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
                        row: row, table: tables[index], segmentIDs: segments[index],
                        count: row.absoluteOffset, work: work))
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
