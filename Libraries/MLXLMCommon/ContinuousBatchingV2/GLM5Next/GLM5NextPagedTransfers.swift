// Copyright © 2026 Eigen Labs.

import MLX
import MLXFast

/// One-sided transfers for the main GLM payload row. Neither path expands K/V
/// projections or gathers the complete latent plane. All roots join the engine's
/// existing paged step owner; no eval/synchronize occurs while building graphs.
enum GLM5NextPagedTransfers {
    private static let readV = MLXFast.metalKernel(
        name: "glm5_next_read_metadata", inputNames: ["storage", "records", "output", "previous"],
        outputNames: ["fence"], source: """
            const int d = int(thread_position_in_grid.x);
            const int r = int(thread_position_in_grid.y);
            const int token = records[r * 3], page = records[r * 3 + 1], slot = records[r * 3 + 2];
            const size_t source = ((size_t)page * S + slot) * D;
            device T* dst = output;
            dst[(size_t)token * D + d] = storage[VBASE + source + d];
            if (d == 0 && r == 0) fence[0] = previous[0] + 1;
            """, ensureRowContiguous: true, mutableInputs: ["output"])

    private static let readK = MLXFast.metalKernel(
        name: "glm5_next_read_selected_latents",
        inputNames: ["storage", "table", "indices", "output", "previous"], outputNames: ["fence"],
        source: """
            const int d = int(thread_position_in_grid.x);
            const int r = int(thread_position_in_grid.y);
            const int token = indices[r];
            if (token >= 0 && token < N) {
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

    static func values(row: PagedSequenceKV, count: Int, work: CBv2PagedAttentionStepOwner) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        let size = group.pageSize
        let slots = (0..<count).map { token in row.table[token / size] * Int32(size) + Int32(token % size) }
        let output = MLXArray.zeros([1, count, group.key.valueHeadDim], dtype: group.dtype)
        var fence = group.writeFence
        for (segment, triples) in PagedSegmentTransfers.buckets(group: group, slots: slots) {
            let records = PagedSegmentTransfers.records(triples)
            fence = readV([segment.storage, records, output, fence],
                template: [("T", group.dtype), ("D", group.key.valueHeadDim), ("S", size), ("VBASE", segment.valueOffset)],
                grid: (group.key.valueHeadDim, triples.count / 3, 1),
                threadGroup: (min(256, group.key.valueHeadDim), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
            work.retainRoots([segment.storage, records, fence])
        }
        fence = complete([fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                         outputShapes: [[1]], outputDTypes: [.int32])[0]
        group.writeFence = fence
        let result = depends(input: output, dependencies: [fence])
        work.retainRoots([output, result, fence])
        return result
    }

    static func keys(row: PagedSequenceKV, indices: MLXArray, table: MLXArray,
                     segmentIDs: [Int], count: Int, work: CBv2PagedAttentionStepOwner) -> MLXArray {
        let group = row.pool.group(row.groupKey)
        let output = MLXArray.zeros([1, indices.size, group.key.headDim], dtype: group.dtype)
        var fence = group.writeFence
        for id in segmentIDs {
            let segment = group.segments[id]!
            fence = readK([segment.storage, table, indices, output, fence],
                template: [("T", group.dtype), ("D", group.key.headDim), ("S", group.pageSize),
                           ("N", count), ("FIRST", segment.pages.lowerBound), ("END", segment.pages.upperBound)],
                grid: (group.key.headDim, indices.size, 1), threadGroup: (min(256, group.key.headDim), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
            work.retainRoots([segment.storage, fence])
        }
        fence = complete([fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                         outputShapes: [[1]], outputDTypes: [.int32])[0]
        group.writeFence = fence
        let result = depends(input: output, dependencies: [fence])
        work.retainRoots([output, result, fence, table, indices])
        return result
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

    fileprivate init(cache: PagedLayerCache, work: CBv2PagedAttentionStepOwner,
                     rows: [PagedSequenceKV], metadata: [MLXArray]) {
        self.cache = cache; self.work = work; self.rows = rows; self.metadata = metadata
        lengths = rows.map(\.absoluteOffset)
        tables = rows.map { MLXArray($0.table) }
        segments = rows.map { row in
            Array(Set(row.table.map { row.pool.group(row.groupKey).segmentLayout!.segmentIndex(page: $0) })).sorted()
        }
        reads = Array(repeating: 0, count: rows.count)
        work.retainRoots(tables)
    }

    package func values(row: Int) -> MLXArray { metadata[row] }

    package func selectedKeys(row: Int, indices: MLXArray) throws -> MLXArray {
        guard rows.indices.contains(row), cache.pool.activeAttentionWork === work,
            !work.completed, indices.ndim == 2, indices.dim(0) == 1,
            indices.dtype == .int32, let binding = cache.glm5NextModelBinding,
            indices.size == binding.selectionWidth,
            let descriptor = work.descriptors[.init(layer: cache.layerIndex, serial: rows[row].serial)],
            reads[row] < descriptor.range.count, rows[row].absoluteOffset == lengths[row]
        else { throw CBv2KVError.backendIneligible(reason: "stale or oversized GLM selected latent read") }
        reads[row] += 1
        cache.glm5NextModelBinding?.recordSelectedRows(indices.size)
        return GLM5NextPagedTransfers.keys(row: rows[row], indices: indices.flattened(), table: tables[row],
            segmentIDs: segments[row], count: lengths[row], work: work)
    }

    package func retainResult(_ result: MLXArray) { work.retainRoots([result]) }
}

extension PagedLayerCache {
    package func matchesGLM5NextTarget(_ identity: ObjectIdentifier) -> Bool {
        glm5NextModelBinding?.matches(identity) == true
    }

    package func beginGLM5NextPayload(keys: MLXArray, values: MLXArray,
                                     targetIdentity: ObjectIdentifier) throws -> GLM5NextPagedPayload {
        guard matchesGLM5NextTarget(targetIdentity), let work = pool.activeAttentionWork,
            pool.activeAttentionLayer == nil, kind.attention == .full,
            keys.ndim == 4, values.ndim == 4, keys.dim(0) == rows.count,
            keys.shape == [rows.count, 1, keys.dim(2), kind.headDim],
            values.shape == [rows.count, 1, keys.dim(2), kind.valueHeadDim],
            keys.dtype == .float32, values.dtype == .float32,
            work.consume(layer: layerIndex, rows: rows.compactMap { $0 as? PagedSequenceKV },
                         queries: keys.dim(2), softcap: false)
        else { throw CBv2KVError.backendIneligible(reason: "GLM payload lacks its exact admitted cache ticket") }
        pool.activeAttentionLayer = layerIndex
        defer { pool.activeAttentionLayer = nil }
        let native = rows.map { $0 as! PagedSequenceKV }
        for (index, row) in native.enumerated() {
            row.write(keys: keys[index].asType(.float32), values: values[index].asType(.float32))
        }
        try pool.writeValidation.check()
        var metadata: [MLXArray] = []
        for row in native {
            guard work.authorizeRead(layer: layerIndex, row: row, start: 0, count: row.absoluteOffset) else {
                throw CBv2KVError.backendIneligible(reason: "GLM index read does not match the planned frontier")
            }
            metadata.append(GLM5NextPagedTransfers.values(row: row, count: row.absoluteOffset, work: work))
            glm5NextModelBinding?.recordMetadataRows(row.absoluteOffset)
        }
        advanceGLM5NextOffsets(by: keys.dim(2))
        return .init(cache: self, work: work, rows: native, metadata: metadata)
    }

    package func recordGLM5NextFailure(_ error: any Error) { pool.writeValidation.record(error) }
}
