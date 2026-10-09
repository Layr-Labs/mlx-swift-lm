import MLX
import MLXFast

extension PagedQuantizedKernelSmoke {
    /// Run first in the isolated preflight child: Metal compiler failures can
    /// terminate its process. Each selected geometry is probed serially under
    /// a 64 MiB local admission ceiling, without model weights or user content.
    /// Native `PagedAttentionKernel.runtimeSmoke` retains its existing contract.
    @discardableResult
    public static func runtimeSmoke(
        shapes: [PagedQuantizedKernelSmokeShape],
        quantization: PagedKVQuantizationConfig
    ) throws
        -> PagedQuantizedKernelSmokeCoverage
    {
        try quantization.validateParameters()
        let unique = Set(shapes)
        guard unique.count <= 128 else {
            throw PagedAttentionKernelSmokeError.invalidShape("too many packed smoke shapes")
        }
        var coverage: PagedQuantizedKernelSmokeCoverage = [:]
        for shape in unique.sorted(by: { $0.argumentValue < $1.argumentValue }) {
            coverage[shape] = try probe(shape: shape, quantization: quantization)
        }
        return coverage
    }

    private static let capacity = 64 << 20

    private static func probe(
        shape: PagedQuantizedKernelSmokeShape,
        quantization: PagedKVQuantizationConfig
    ) throws -> Set<String> {
        let dtypes: [DType] = [.float16, .bfloat16, .float32]
        guard shape.kvHeads > 0, shape.kvHeads <= 64,
            shape.queryHeads > 0, shape.queryHeads <= 256,
            shape.queryHeads % shape.kvHeads == 0,
            dtypes.contains(shape.dtype), dtypes.contains(shape.queryDType),
            shape.windowSize == nil || shape.windowSize! > quantization.recentTokenCount
        else {
            throw PagedAttentionKernelSmokeError.invalidShape(shape.argumentValue)
        }
        try quantization.validate(headDim: shape.headDim)
        if let reason = PagedAttentionKernel.ineligibilityReason(
            headDim: shape.headDim, gqa: shape.queryHeads / shape.kvHeads)
        {
            throw PagedAttentionKernelSmokeError.ineligibleShape(reason)
        }
        let prime = max(160, quantization.recentTokenCount + 32)
        let maximum = prime + 32
        func multiply(_ values: [Int]) throws -> Int {
            try values.reduce(1) { try PagedKVQuantizationConfig.multiply($0, $1) }
        }
        let nativeBound = try multiply([8, prime, shape.kvHeads, shape.headDim, shape.dtype.size])
        let packedBound = try multiply([
            2, maximum,
            quantization.bytesPerToken(kvHeads: shape.kvHeads, headDim: shape.headDim),
        ])
        let scratchBound = try multiply([32, shape.queryHeads, shape.headDim, 4])
        let bound = try CBv2CheckpointAllocationFootprint.add(
            CBv2CheckpointAllocationFootprint.add(nativeBound, packedBound), scratchBound)
        guard bound <= capacity - (8 << 20) else {
            throw PagedAttentionKernelSmokeError.ineligibleShape(
                "packed smoke exceeds bounded scratch")
        }
        let kind = CBv2LayerKind(
            attention: shape.windowSize.map { .slidingWindow($0) } ?? .full,
            hasSinks: shape.hasSinks, isBidirectional: shape.isBidirectional,
            headDim: shape.headDim, kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)
        let pageSize = CBv2PagedDefaults.pageSize
        let pageBytes = try multiply([
            pageSize,
            quantization.bytesPerToken(kvHeads: shape.kvHeads, headDim: shape.headDim),
        ])
        // One usable page per segment deliberately exercises every binding
        // class without allocating a model-sized history or backing segment.
        let config = PagedKVPoolConfig(
            pageSize: pageSize, capacityBytes: capacity,
            dtype: shape.dtype, maxPrefillChunk: prime, nominalMaxSequenceLength: maximum,
            maxBufferLength: capacity, segmentSizeBytes: 2 * pageBytes,
            quantization: quantization)
        let backend = try PagedKVBackend(layerKinds: [kind], config: config)
        let pool = backend.pool
        let admission = AdmissionV2(
            layerKinds: [kind], bytesCapacity: capacity,
            config: try pool.admissionStorageConfig(
                .init(
                    watermarkFraction: 0,
                    elementBytes: shape.dtype.size)),
            residency: backend.kvResidency)
        pool.bindAdmission(admission)
        let id = CBv2RequestID(1)
        try admission.reserve(id: id, additionalTokens: maximum)
        defer { admission.releaseAll(id: id) }
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: prime,
            maxLength: maximum)
        let stream = StreamOrDevice.default
        defer {
            stream.stream.synchronize()
            pool.discardQuantizedStorageStepAfterSynchronization()
            backend.release(state)
        }
        guard let row = state[0] as? PagedSequenceKV else {
            throw PagedAttentionKernelSmokeError.ineligibleShape("missing packed smoke row")
        }
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([row])
        cache.setRetainsChunkForBorrowers(false)
        var completed = Set<String>()
        func finish(_ roots: [MLXArray], operation: String, attention: MLXArray? = nil) throws {
            try withError { fault in
                eval(roots + row.quantizedEvaluationRoots())
                try fault.check()
                stream.stream.synchronize()
                try fault.check()
                if let attention {
                    let values = attention.asArray(Float.self)
                    try fault.check()
                    guard !values.isEmpty,
                        values.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1.02 })
                    else {
                        throw PagedAttentionKernelSmokeError.ineligibleShape(
                            "non-finite or invalid packed smoke output: \(operation)")
                    }
                }
            }
            try pool.writeValidation.check()
            try pool.finishQuantizedStorageStep()
            completed.insert(operation)
        }
        let kv = MLXArray.ones([1, shape.kvHeads, prime, shape.headDim], dtype: shape.dtype)
        row.write(keys: kv[0], values: kv[0])
        try finish([], operation: "packed-write")
        guard row.nativeRecentStart > 0 else {
            throw PagedAttentionKernelSmokeError.ineligibleShape("smoke has no coded history")
        }
        let native = row.gatherRange(start: 0, count: 1)
        try finish([native.keys, native.values], operation: "native-gather")
        let group = pool.group(row.groupKey)
        let packed = PagedQuantizedTransfers.gatherPacked(
            group: group, pages: row.table,
            firstSlot: 0, count: 1, publishReadFence: false)
        try finish([packed.keys, packed.values], operation: "packed-gather")
        if shape.windowSize == nil {
            try selectedGatherProbes(
                backend: backend, admission: admission, kind: kind,
                dtype: shape.dtype, completed: &completed)
        }
        for history in [16, 32, 80, 144] {
            let prefix = min(history, row.retainedCount)
            for queries in [1, PagedQuantizedAttentionWorkspace.queryBlockSize] {
                let q = MLXArray.full(
                    [1, shape.queryHeads, queries, shape.headDim],
                    values: MLXArray(Float(0.03125)), dtype: shape.queryDType)
                let current = MLXArray.ones(
                    [1, shape.kvHeads, queries, shape.headDim], dtype: shape.dtype)
                let bindings = try readOnlyBindingClasses(
                    row: row, prefix: prefix, queries: queries)
                for softcap in [false, true] {
                    for masked in [false, true] {
                        let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                            masked
                            ? .array(
                                MLXArray.ones([1, 1, queries, prefix + queries], dtype: .bool))
                            : .none
                        guard
                            let output = try row.attendQuantizedReadOnly(
                                queries: q,
                                currentKeys: current, currentValues: current,
                                scale: 1 / Float(shape.headDim).squareRoot(), mask: mask,
                                prefixCount: prefix, softcap: softcap ? 1 : nil,
                                sinks: shape.hasSinks
                                    ? MLXArray.zeros([shape.queryHeads], dtype: .float32) : nil)
                        else {
                            throw PagedAttentionKernelSmokeError.ineligibleShape(
                                "packed read-only dispatch refused")
                        }
                        let tag =
                            "part-n\(bindings.map(String.init).sorted().joined(separator: "+"))-q\(queries)-mask\(masked ? 1 : 0)-cap\(softcap ? 1 : 0)"
                        try finish([output], operation: tag, attention: output)
                    }
                }
            }
        }
        for count in [1, PagedQuantizedAttentionWorkspace.queryBlockSize] {
            let queries = MLXArray.full(
                [1, shape.queryHeads, count, shape.headDim],
                values: MLXArray(Float(0.03125)), dtype: shape.queryDType)
            let keys = MLXArray.ones([1, shape.kvHeads, count, shape.headDim], dtype: shape.dtype)
            let output = cache.updateAndAttend(
                queries: queries, keys: keys, values: keys,
                scale: 1 / Float(shape.headDim).squareRoot(),
                sinks: shape.hasSinks ? MLXArray.zeros([shape.queryHeads], dtype: .float32) : nil)
            try finish(
                [output] + cache.innerState(), operation: "update-q\(count)", attention: output)
        }
        return completed
    }

    private static func readOnlyBindingClasses(
        row: PagedSequenceKV, prefix: Int,
        queries: Int
    ) throws -> Set<Int> {
        let pool = row.pool
        let group = pool.group(row.groupKey)
        let end = row.absoluteOffset + queries
        let pages = try row.quantizedReadOnlyPages(end: end)
        let partition = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: group.pageSize)
        let info = PagedAttentionKernel.SeqInfoRow(
            attendStart: row.absoluteOffset - prefix,
            attendLength: prefix + queries, tableLength: max(1, pages.count))
        let plan = PagedSegmentDispatchPlan(
            rows: [.init(pages: pages, info: info)],
            layout: group.segmentLayout!, pageSize: group.pageSize, partitionTokens: partition,
            hasWrite: false)
        return Set(plan.buckets.map(\.bindingClass))
    }

    private static func selectedGatherProbes(
        backend: PagedKVBackend, admission: AdmissionV2,
        kind: CBv2LayerKind, dtype: DType,
        completed: inout Set<String>
    ) throws {
        for tokens in [16, 32, 80, 144] {
            let id = CBv2RequestID(2)
            try admission.reserve(id: id, additionalTokens: tokens)
            defer { admission.releaseAll(id: id) }
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: tokens,
                maxLength: tokens)
            defer {
                StreamOrDevice.default.stream.synchronize()
                backend.pool.discardQuantizedStorageStepAfterSynchronization()
                backend.release(state)
            }
            guard let row = state[0] as? PagedSequenceKV else {
                throw PagedAttentionKernelSmokeError.ineligibleShape("missing selected smoke row")
            }
            let kv = MLXArray.ones([kind.kvHeads, tokens, kind.headDim], dtype: dtype)
            row.write(keys: kv, values: kv)
            try withError { fault in
                eval(row.quantizedEvaluationRoots())
                try fault.check()
            }
            try backend.pool.finishQuantizedStorageStep()
            let indices = MLXArray([Int32(0), Int32(tokens - 1)])
            let selected = try PagedQuantizedSelectedGather.gather(row: row, indices: indices)
            try withError { fault in
                eval(selected.keys, selected.values)
                try fault.check()
                StreamOrDevice.default.stream.synchronize()
                try fault.check()
            }
            try backend.pool.writeValidation.check()
            try backend.pool.finishQuantizedStorageStep()
            let segments = Set(
                row.table.map {
                    backend.pool.group(row.groupKey).segmentLayout!.segmentIndex(page: $0)
                })
            let binding =
                PagedSelectedGather.boundEnabled()
                ? PagedSegmentDispatchPlan.bindingClasses.first(where: { $0 >= segments.count })!
                : 1
            completed.insert("selected-n\(binding)")
        }
    }
}
