import MLX

extension PagedLayerCache {
    /// Full-history prefill reads packed pages directly. Query columns become
    /// independent decode rows with their own causal upper bound. This favors
    /// bounded memory over native SDPA's matrix throughput; benchmark separately.
    func quantizedPrefill(queries: MLXArray, row: PagedSequenceKV,
                         queryStart: Int, scale: Float, sinks: MLXArray?,
                         chunkKeys: MLXArray? = nil, chunkValues: MLXArray? = nil) -> MLXArray {
        precondition(row.windowSize == nil && queries.dim(0) == 1)
        let count = queries.dim(2)
        let group = pool.group(row.groupKey)
        let blockSize = 8
        let workspace: PagedQuantizedAttentionWorkspace
        do {
            workspace = try PagedQuantizedAttentionWorkspace(
                pool: pool, group: group, queryCount: count, blockSize: min(blockSize, count),
                queryHeads: queries.dim(1), maxAttendLength: queryStart + count - row.baseOffset,
                maximumSegmentCount: min(group.segments.count,
                    (queryStart + count - row.baseOffset - 1) / group.pageSize + 2),
                broadcastTopology: true, nativeOutputBytes: queries.nbytes)
        } catch {
            pool.writeValidation.record(error)
            return queries
        }
        if let chunkKeys, let chunkValues {
            row.write(keys: chunkKeys.squeezed(axis: 0), values: chunkValues.squeezed(axis: 0))
        }
        let output = MLXArray.zeros(queries.shape, dtype: queries.dtype)
        let prepared = PagedSegmentPreparedDispatch(
            rows: [.init(pages: row.table, info: row.seqInfoRow(
                attending: (start: row.baseOffset, length: queryStart + count - row.baseOffset)))],
            group: group, partitionTokens: workspace.partitionTokens, hasWrite: false)
        workspace.lease.retainAdditional([output] + prepared.allocationArrays)
        let context = PagedQuantizedPrefillDispatch(prepared: prepared, output: output)
        for offset in stride(from: 0, to: count, by: blockSize) {
            let length = min(blockSize, count - offset)
            let descriptors = (offset ..< offset + length).map { index in
                let position = queryStart + index
                var end = kind.isBidirectional ? queryStart + count : position + 1
                if let context = boundSpanContext {
                    for span in context.blocks where position >= span.tokenOffset && position < span.end {
                        end = max(end, span.end)
                    }
                }
                end = min(end, queryStart + count)
                var info = row.seqInfoRow(attending: (start: row.baseOffset, length: end - row.baseOffset))
                info.queryIndex = index
                return PagedSegmentDispatchPlan.Row(pages: row.table, info: info)
            }
            _ = PagedSegmentAttention.decode(
                queries: queries, newKeys: nil, newValues: nil, group: group, rows: descriptors,
                sinks: preparedSinks(sinks), params: params(scale: scale),
                softcap: attentionSoftcap != nil, source: pool.kernelSource,
                workspace: workspace, prefill: context)
        }
        return PagedQuantizedPrefillMerge.finish(output: output, group: group, workspace: workspace)
    }
}
