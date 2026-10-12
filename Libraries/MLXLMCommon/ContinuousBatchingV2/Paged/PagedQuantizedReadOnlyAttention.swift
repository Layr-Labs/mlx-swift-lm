import MLX
import MLXFast

private final class PagedQuantizedCanvasOwner {
    private var roots: [MLXArray] = []
    private var sourceOwner: AnyObject?
    private var reservation: CBv2CheckpointReservation?
    init(reservation: CBv2CheckpointReservation?, sourceOwner: AnyObject?) {
        self.reservation = reservation
        self.sourceOwner = sourceOwner
    }
    func retain(_ arrays: [MLXArray]) { roots = arrays }
    deinit {
        roots = []
        sourceOwner = nil
        reservation?.release()
    }
}

extension PagedSequenceKV {
    public var usesQuantizedStorage: Bool { groupKey.quantization != nil }

    /// The original ring divisor stays fixed even when the absolute lifetime
    /// is long or only trailing slots were adopted. Ephemeral native canvas
    /// positions may name any owned binding because they never read its codes.
    func quantizedReadOnlyPages(end: Int) throws -> [Int32] {
        let group = pool.group(groupKey)
        guard let fallback = table.last(where: { group.isAllocatable($0) }), !table.isEmpty else {
            throw CBv2KVError.backendIneligible(
                reason: "packed canvas has no committed prefix page")
        }
        let length =
            windowSize == nil ? max(table.count, (end - 1) / group.pageSize + 1) : decodeTableLength
        var result = [Int32](repeating: fallback, count: length)
        for (slot, page) in table.enumerated() where group.isAllocatable(page) {
            result[slot] = page
        }
        return result
    }

    /// Actual write/native generation completion roots for the owning model
    /// step. Observing committed storage never gathers/dequantizes history.
    public func quantizedEvaluationRoots() -> [MLXArray] {
        guard usesQuantizedStorage, !isReleased else { return [] }
        return [pool.group(groupKey).writeFence] + nativeRecentEvaluationRoots
    }

    /// Commit encoder/finalized-canvas rows without materializing the prefix.
    /// Denoising callers must use the read-only operation instead.
    public func appendQuantizedCommitted(keys: MLXArray, values: MLXArray) throws {
        guard usesQuantizedStorage, keys.ndim == 4, values.ndim == 4,
            keys.dim(0) == 1, values.dim(0) == 1
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid quantized committed append")
        }
        write(keys: keys.squeezed(axis: 0), values: values.squeezed(axis: 0))
        if pool.writeValidation.isFaulted {
            throw CBv2KVError.backendIneligible(reason: "quantized committed append was refused")
        }
    }

    /// Immutable packed prefix + an ephemeral original-precision canvas.
    /// Neither the cursor, page ownership nor native recent generation changes.
    /// A later committed append consumes this read fence before overwriting KV.
    public func attendQuantizedReadOnly(
        queries: MLXArray, currentKeys: MLXArray, currentValues: MLXArray,
        scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        prefixCount: Int? = nil, softcap: Float? = nil, sinks: MLXArray? = nil
    ) throws -> MLXArray? {
        guard usesQuantizedStorage else { return nil }
        let count = queries.ndim == 4 ? queries.dim(2) : 0
        let history = prefixCount ?? retainedCount
        guard !isReleased, !pool.writeValidation.isFaulted,
            queries.ndim == 4, currentKeys.ndim == 4, currentValues.ndim == 4,
            [.float16, .bfloat16, .float32].contains(queries.dtype),
            queries.dim(1) > 0, scale.isFinite, scale > 0,
            softcap == nil || (softcap!.isFinite && softcap! > 0),
            sinks == nil
                || (sinks!.size == queries.dim(1)
                    && [.float16, .bfloat16, .float32].contains(sinks!.dtype)),
            queries.dim(0) == 1, currentKeys.dim(0) == 1, currentValues.dim(0) == 1,
            queries.dim(3) == groupKey.headDim, queries.dim(1) % groupKey.kvHeads == 0,
            count > 0, currentKeys.shape == [1, groupKey.kvHeads, count, groupKey.headDim],
            currentValues.shape == [1, groupKey.kvHeads, count, groupKey.valueHeadDim],
            currentKeys.dtype == groupKey.dtype, currentValues.dtype == groupKey.dtype,
            groupKey.headDim == groupKey.valueHeadDim,
            history >= 0, history <= retainedCount,
            count <= pool.config.maxPrefillChunk,
            let end = CBv2KVGeometry.add(absoluteOffset, count), end <= Int(Int32.max)
        else {
            throw CBv2KVError.backendIneligible(
                reason: "invalid read-only packed attention geometry")
        }
        let start = absoluteOffset - history
        guard history == 0 || start >= oldestValidPosition else {
            throw CBv2KVError.backendIneligible(reason: "packed canvas names evicted prefix rows")
        }
        let booleanMask: MLXArray?
        let causal: Bool
        switch mask {
        case .none:
            booleanMask = nil
            causal = false
        case .causal:
            booleanMask = nil
            causal = true
        case .arrays:
            throw CBv2KVError.backendIneligible(
                reason: "packed canvas boolean mask lists are unsupported")
        case .array(let value):
            guard value.dtype == .bool, value.shape == [1, 1, count, history + count] else {
                throw CBv2KVError.backendIneligible(
                    reason: "packed canvas requires a canonical boolean mask")
            }
            booleanMask = value
            causal = false
        }
        let nativeStart = max(start, nativeRecentStart)
        let oldCount = max(0, absoluteOffset - nativeStart)
        let old = oldCount > 0 ? nativeRecentRange(start: nativeStart, count: oldCount) : nil
        if oldCount > 0, old == nil {
            throw CBv2KVError.backendIneligible(reason: "packed canvas native prefix is missing")
        }
        let tokens = try CBv2CheckpointAllocationFootprint.add(oldCount, count)
        let roleBytes = try PagedKVQuantizationConfig.multiply(
            tokens,
            try PagedKVQuantizationConfig.multiply(
                groupKey.kvHeads,
                try PagedKVQuantizationConfig.multiply(groupKey.headDim, groupKey.dtype.size)))
        let reserved = try CBv2CheckpointAllocationFootprint.add(
            try PagedKVQuantizationConfig.multiply(
                4, CBv2CheckpointAllocationFootprint.bound(roleBytes)), 64 << 10)
        let owner = PagedQuantizedCanvasOwner(
            reservation: try pool.memoryAdmission?.reserveTransient(bytes: reserved),
            sourceOwner: nativeRecentStorageOwner)
        let k = currentKeys.squeezed(axis: 0)
        let v = currentValues.squeezed(axis: 0)
        let nativeKeys = old.map { concatenated([$0.keys, k], axis: 1) } ?? k
        let nativeValues = old.map { concatenated([$0.values, v], axis: 1) } ?? v
        owner.retain([nativeKeys, nativeValues])
        let native = PagedQuantizedNativeView(
            start: nativeStart, keys: nativeKeys, values: nativeValues, owner: owner)
        let group = pool.group(groupKey)
        let virtualPages: [Int32]
        if history == 0 {
            virtualPages = []
        } else {
            virtualPages = try quantizedReadOnlyPages(end: end)
        }
        let bounds = (0 ..< count).map { index in
            start ..< (causal ? absoluteOffset + index + 1 : end)
        }
        let workspace = try PagedQuantizedAttentionWorkspace(
            pool: pool, group: group,
            queryShape: queries.shape, dtype: queries.dtype, attendLength: end - start,
            metadataRecordCounts: PagedQuantizedAttention.metadataRecordCounts(
                row: self,
                visibleStart: start, visibleEnd: end, virtualPages: virtualPages))
        let params = MLXArray([softcap ?? 0, scale, 0, 0, 0, 0, 0, 0])
        let sinkArray = sinks.map { value -> MLXArray in
            let flat = value.asType(.float32).reshaped([-1])
            return flat.size >= 8
                ? flat : concatenated([flat, MLXArray.zeros([8 - flat.size], dtype: .float32)])
        }
        return PagedQuantizedAttention.attend(
            queries: queries, row: self, native: native,
            queryStart: absoluteOffset, visibleStart: start, visibleEnd: end,
            queryBounds: bounds, sinks: sinkArray, params: params, softcap: softcap != nil,
            workspace: workspace, source: pool.kernelSource,
            booleanMask: booleanMask, virtualPages: virtualPages)
    }
}
