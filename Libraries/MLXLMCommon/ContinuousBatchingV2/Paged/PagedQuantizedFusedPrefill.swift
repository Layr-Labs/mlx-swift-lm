import MLX

extension PagedLayerCache {
    /// Optional acceleration with an additive permit. Any preflight refusal
    /// returns nil before constructing arrays, inserting an arena, or writing KV.
    func opportunisticQuantizedPrefill(queries: MLXArray, row: PagedSequenceKV,
                                       queryStart: Int, scale: Float, sinks: MLXArray?,
                                       chunkKeys: MLXArray?, chunkValues: MLXArray?) -> MLXArray? {
        guard pool.config.quantizedPrefillMode == .opportunisticSDPA else { return nil }
        let count = queries.dim(2), d = queries.dim(3)
        let (end, overflow) = queryStart.addingReportingOverflow(count)
        guard count >= 9, [64, 256].contains(d), attentionSoftcap == nil,
              [DType.float16, .bfloat16, .float32].contains(queries.dtype),
              sinks == nil || sinks!.size == queries.dim(1),
              !kind.isBidirectional, boundSpanContext == nil,
              !overflow, end <= Int(Int32.max),
              let admission = pool.memoryAdmission, let scope = pool.quantizedScratchScope else {
            pool.quantizedPrefillCounters.fallback(budget: false)
            return nil
        }
        let group = pool.group(row.groupKey)
        let geometry = PagedQuantizedFusedPrefillGeometry(
            kvHeads: group.key.kvHeads, queryHeads: queries.dim(1), headDim: d)
        let existing = scope.fusedPrefillArenas[geometry]
        let length = end - row.baseOffset
        let plan: PagedQuantizedFusedPrefillPlan
        do {
            plan = try .init(geometry: geometry, queryCount: count,
                             maximumTokens: scope.maximumAttendLength, attendLength: length,
                             pageSize: group.pageSize, outputElementBytes: queries.dtype.size,
                             needsArena: existing == nil, maxBufferLength: pool.config.maxBufferLength)
        } catch {
            pool.quantizedPrefillCounters.fallback(budget: false)
            return nil
        }
        let permit: CBv2CheckpointReservation
        do { permit = try admission.reserveOpportunisticWorkspace(bytes: plan.reservationBytes) }
        catch {
            pool.quantizedPrefillCounters.fallback(budget: true)
            return nil
        }
        // The first successful call owns the complete arena plus its own fresh
        // allocations. Every later call owns only its fresh outputs/metadata.
        let counters = pool.quantizedPrefillCounters
        counters.reserved(bytes: plan.reservationBytes)
        let lease = PagedQuantizedScratchLease(reservation: permit, bytes: plan.reservationBytes) {
            counters.retired(bytes: plan.reservationBytes)
        }
        let arena = existing ?? PagedQuantizedFusedPrefillArena(plan: plan, owner: lease)
        if existing == nil { scope.fusedPrefillArenas[geometry] = arena }
        pool.appendQuantizedScratch(lease)
        if let chunkKeys, let chunkValues {
            row.write(keys: chunkKeys.squeezed(axis: 0), values: chunkValues.squeezed(axis: 0))
        }
        group.writeFence = PagedQuantizedFusedPrefillKernels.acquire(
            previous: group.writeFence, lastUse: arena.lastUse)
        let prepared = PagedSegmentPreparedDispatch(
            rows: [.init(pages: row.table, info: row.seqInfoRow(
                attending: (start: row.baseOffset, length: length)))],
            group: group, partitionTokens: plan.partitionTokens, hasWrite: false)
        let output = MLXArray.zeros(queries.shape, dtype: queries.dtype)
        let callParameters = MLXArray([Int32(length), Int32(row.baseOffset), Int32(queryStart), 0, 0, Int32(count), 0, 0])
        lease.retainAdditional([output, callParameters] + prepared.allocationArrays)
        PagedQuantizedFusedPrefillKernels.dequantize(
            group: group, prepared: prepared, destination: arena.keysValues, parameters: callParameters)
        let preparedSinks = sinks.map { $0.asType(.float32).reshaped([-1]) }
        if let preparedSinks { lease.retainAdditional([preparedSinks]) }
        for block in plan.blocks {
            let parameters = MLXArray([Int32(length), Int32(row.baseOffset), Int32(queryStart),
                                       Int32(block.lowerBound), Int32(block.count), Int32(count), 0, 0])
            let visible = PagedQuantizedFusedPrefillKernels.prepareBlock(
                queries: queries, arena: arena, parameters: parameters, count: block.count,
                attendLength: length, quantization: group.key.quantization!, previous: group.writeFence)
            let q = depends(input: arena.queries[0..., 0..., 0 ..< block.count, 0...], dependencies: [visible])
            let k = depends(input: arena.keysValues[0, 0..., 0..., 0 ..< length, 0...], dependencies: [visible])
            let v = depends(input: arena.keysValues[1, 0..., 0..., 0 ..< length, 0...], dependencies: [visible])
            let mask = depends(input: arena.mask[0 ..< block.count, 0 ..< length], dependencies: [visible])
            let attention = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(mask),
                sinks: preparedSinks, forceFused: true)
            group.writeFence = PagedQuantizedFusedPrefillKernels.copyOutput(
                attention, output: output, parameters: parameters, count: block.count, previous: visible)
            lease.retainAdditional([parameters, attention])
            arena.recordCompletion(group.writeFence, lease: lease)
        }
        group.writeFence = PagedQuantizedFusedPrefillKernels.witness(group.writeFence)
        arena.recordCompletion(group.writeFence, lease: lease)
        counters.fused(tokens: count)
        return depends(input: output, dependencies: [group.writeFence])
    }
}
