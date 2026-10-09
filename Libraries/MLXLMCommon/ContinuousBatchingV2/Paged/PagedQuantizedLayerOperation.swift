import MLX

struct PagedQuantizedBorrowedView {
    let row: PagedSequenceKV
    let queryStart: Int
    let visibleStart: Int
    let visibleEnd: Int
    let native: PagedQuantizedNativeView
}

/// Original window history can be overwritten by a large chunk's ring writes.
/// This bounded pre-write owner carries its charge through shared-layer readers.
final class PagedQuantizedWindowOwner {
    private var reservation: CBv2CheckpointReservation?
    private var roots: [MLXArray] = []
    init(reservation: CBv2CheckpointReservation?) { self.reservation = reservation }
    func retain(_ arrays: [MLXArray]) { roots = arrays }
    deinit {
        roots = []
        reservation?.release()
    }
}

enum PagedQuantizedLayerOperation {
    typealias WindowProducer = (PagedSequenceKV, MLXArray, MLXArray) -> PagedQuantizedNativeView

    static func write(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        rows: [PagedSequenceKV], kind: CBv2LayerKind,
        params: MLXArray, sinks: MLXArray?, softcap: Bool,
        spans: CBv2SpanChunkContext?, source: String,
        windowProducer: WindowProducer
    ) -> (output: MLXArray, views: [PagedQuantizedBorrowedView]) {
        var outputs: [MLXArray] = []
        var views: [PagedQuantizedBorrowedView] = []
        for (index, row) in rows.enumerated() {
            let q = queries[index ..< index + 1]
            let k = keys[index ..< index + 1]
            let v = values[index ..< index + 1]
            let queryStart = row.absoluteOffset
            let native: PagedQuantizedNativeView
            let visibleStart: Int
            if let window = row.windowSize, queries.dim(2) > 1 {
                do {
                    // Conservatively cover gather, overlap splice, concat,
                    // graph metadata and retained borrower aliases before IO.
                    let tokens = try PagedKVQuantizationConfig.multiply(
                        11,
                        try CBv2CheckpointAllocationFootprint.add(window, queries.dim(2)))
                    let raw = try PagedKVQuantizationConfig.multiply(
                        tokens,
                        try PagedKVQuantizationConfig.multiply(
                            kind.kvHeads,
                            try PagedKVQuantizationConfig.multiply(
                                kind.headDim, row.groupKey.dtype.size)))
                    let offsets = try PagedKVQuantizationConfig.multiply(
                        row.pool.group(row.groupKey).segments.count,
                        CBv2CheckpointAllocationFootprint.bound(MemoryLayout<Int64>.stride))
                    let bytes = try CBv2CheckpointAllocationFootprint.add(
                        CBv2CheckpointAllocationFootprint.add(
                            CBv2CheckpointAllocationFootprint.bound(raw), 1 << 20), offsets)
                    let owner = PagedQuantizedWindowOwner(
                        reservation: try row.pool.memoryAdmission?.reserveTransient(bytes: bytes))
                    let view = windowProducer(row, k, v)
                    guard !row.pool.writeValidation.isFaulted else { return (queries, []) }
                    owner.retain([view.keys, view.values])
                    native = .init(
                        start: view.start, keys: view.keys, values: view.values, owner: owner)
                    visibleStart = view.start
                } catch {
                    row.pool.writeValidation.record(error)
                    return (queries, [])
                }
            } else {
                row.write(keys: k.squeezed(axis: 0), values: v.squeezed(axis: 0))
                guard !row.pool.writeValidation.isFaulted else { return (queries, []) }
                guard let nk = row.nativeRecentKeys, let nv = row.nativeRecentValues else {
                    row.pool.writeValidation.refuse(
                        "packed attention requires an original native band",
                        expected: row.groupKey.dtype)
                    return (queries, [])
                }
                native = .init(
                    start: row.nativeRecentStart, keys: nk, values: nv,
                    owner: row.nativeRecentStorageOwner)
                visibleStart =
                    row.windowSize.map { max(row.baseOffset, queryStart - $0 + 1) }
                    ?? row.baseOffset
            }
            let view = PagedQuantizedBorrowedView(
                row: row, queryStart: queryStart,
                visibleStart: visibleStart, visibleEnd: row.absoluteOffset, native: native)
            outputs.append(
                attend(
                    queries: q, view: view, kind: kind,
                    params: params, sinks: sinks, softcap: softcap, spans: spans, source: source))
            views.append(view)
        }
        return (outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0), views)
    }

    static func attend(
        queries: MLXArray, view: PagedQuantizedBorrowedView,
        kind: CBv2LayerKind, params: MLXArray, sinks: MLXArray?, softcap: Bool,
        spans: CBv2SpanChunkContext?, source: String
    ) -> MLXArray {
        let row = view.row
        guard !row.pool.writeValidation.isFaulted else { return queries }
        let bounds = (0 ..< queries.dim(2)).map { index -> Range<Int> in
            let position = view.queryStart + index
            var low =
                row.windowSize.map { max(view.visibleStart, position - $0 + 1) }
                ?? view.visibleStart
            var high =
                kind.isBidirectional
                ? min(view.visibleEnd, row.windowSize.map { position + $0 } ?? view.visibleEnd)
                : min(view.visibleEnd, position + 1)
            if let spans {
                for span in spans.blocks where position >= span.tokenOffset && position < span.end {
                    low = min(low, max(view.visibleStart, span.tokenOffset))
                    high = min(view.visibleEnd, max(high, span.end))
                }
            }
            return low ..< high
        }
        do {
            let group = row.pool.group(row.groupKey)
            let workspace = try PagedQuantizedAttentionWorkspace(
                pool: row.pool, group: group,
                queryShape: queries.shape, dtype: queries.dtype,
                attendLength: view.visibleEnd - view.visibleStart,
                metadataRecordCounts: PagedQuantizedAttention.metadataRecordCounts(
                    row: row,
                    visibleStart: view.visibleStart, visibleEnd: view.visibleEnd))
            return PagedQuantizedAttention.attend(
                queries: queries, row: row, native: view.native,
                queryStart: view.queryStart, visibleStart: view.visibleStart,
                visibleEnd: view.visibleEnd,
                queryBounds: bounds, sinks: sinks, params: params, softcap: softcap,
                workspace: workspace, source: source)
        } catch {
            row.pool.writeValidation.record(error)
            return queries
        }
    }
}
