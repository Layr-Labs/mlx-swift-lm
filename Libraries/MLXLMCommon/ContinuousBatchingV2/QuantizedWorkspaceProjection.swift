import MLX

private struct CBv2QuantizedWorkspaceGeometry: Hashable, Sendable {
    let queryHeads: Int
    let headDim: Int
}

extension CBv2RequestWorkspaceProjection {
    /// Each request prepays every overlap allowed by its caller. Borrowers
    /// execute attention and contribute even though they own no KV bytes.
    static func quantizedPaged(
        layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig, maximumChunk: Int,
        maximumBatch: Int = 1, maximumSerialDecodeCalls: Int = 1,
        overlapPolicy: CBv2WorkspaceOverlapPolicy = .twoArbitrarySteps,
        sharesStepArenas: Bool = false
    ) -> Self {
        let layers = layerKinds.filter { $0.attention == .full }
        let chunk = max(1, max(maximumChunk, CBv2PagedSpeculation.maxSpeculativeSpan))
        let batch = max(1, maximumBatch)
        let serialCalls = max(1, maximumSerialDecodeCalls)
        let arenaWidth = max(8, batch)
        let geometries = Set(layers.map {
            CBv2QuantizedWorkspaceGeometry(queryHeads: $0.queryHeads, headDim: $0.headDim)
        })
        let allocationPolicy = Memory.allocationFootprintPolicy()
        let aggregateEnvelope: CBv2WorkspaceCostEnvelope? = {
            guard let allocationPolicy else { return nil }
            var result = CBv2WorkspaceCostEnvelope()
            var prefill = CBv2WorkspaceCostEnvelope(), decode = CBv2WorkspaceCostEnvelope()
            for kind in layers {
                guard let layer = CBv2QuantizedWorkspaceCostEnvelope.layer(
                    queryHeads: kind.queryHeads, headDim: kind.headDim, batch: batch,
                    policy: allocationPolicy, includeArena: !sharesStepArenas)
                else { return nil }
                if sharesStepArenas {
                    guard let nextPrefill = prefill.adding(layer.prefill),
                        let nextDecode = decode.adding(layer.decode) else { return nil }
                    prefill = nextPrefill
                    decode = nextDecode
                } else {
                    guard let overlap = overlapPolicy.envelope(
                        prefill: layer.prefill, decode: layer.decode, serialDecodeCalls: serialCalls),
                        let next = result.adding(overlap) else { return nil }
                    result = next
                }
            }
            if sharesStepArenas {
                var arena = CBv2WorkspaceCostEnvelope()
                for geometry in geometries {
                    guard let value = CBv2QuantizedWorkspaceCostEnvelope.sharedArena(
                        queryHeads: geometry.queryHeads, headDim: geometry.headDim,
                        blockSize: arenaWidth, policy: allocationPolicy),
                        let next = arena.adding(value) else { return nil }
                    arena = next
                }
                return overlapPolicy.envelope(prefill: prefill, decode: decode,
                    serialDecodeCalls: serialCalls, sharedArena: arena)
            }
            return result
        }()
        let single: @Sendable (Int) -> Int? = { tokens in
            guard config.pageSize > 0, let allocationPolicy else { return nil }
            let (length, lengthOverflow) = tokens.addingReportingOverflow(
                CBv2PagedSpeculation.maxSpeculativeSpan)
            guard !lengthOverflow, length > 0 else { return nil }
            let pages = (length - 1) / config.pageSize + 1
            let (segments, segmentOverflow) = pages.addingReportingOverflow(1)
            guard !segmentOverflow else { return nil }
            let queries = min(tokens, chunk)
            var total = 0
            var allPrefill = 0, allDecode = 0
            for kind in layers {
                guard let vectorBytes = try? PagedKVQuantizationConfig.multiply(
                    kind.queryHeads, PagedKVQuantizationConfig.multiply(kind.headDim, 4)),
                    let prefillOutput = try? PagedKVQuantizationConfig.multiply(queries, vectorBytes),
                    let decodeOutput = try? PagedKVQuantizationConfig.multiply(batch, vectorBytes),
                    let prefill = try? PagedQuantizedAttentionWorkspace.reservationBytes(
                    queryCount: queries, blockSize: min(8, queries), queryHeads: kind.queryHeads,
                    headDim: kind.headDim, pageSize: config.pageSize,
                    maxAttendLength: length, maximumSegmentCount: segments,
                    allocationPolicy: allocationPolicy, broadcastTopology: true,
                    nativeOutputBytes: prefillOutput, includeArena: !sharesStepArenas),
                    let decode = try? PagedQuantizedAttentionWorkspace.reservationBytes(
                        // Rectangular partial buffers use B * max(P(N_i)), not
                        // sum(P(N_i)). The longest request must cover the entire
                        // supported batch even when every peer is very short.
                        queryCount: batch, blockSize: batch, queryHeads: kind.queryHeads,
                        headDim: kind.headDim, pageSize: config.pageSize,
                        maxAttendLength: length,
                        maximumSegmentCount: max(batch, PagedSegmentDispatchPlan.maximumBindings + 1),
                        allocationPolicy: allocationPolicy, nativeOutputBytes: decodeOutput,
                        includeArena: !sharesStepArenas)
                else { return nil }
                if sharesStepArenas {
                    let (nextPrefill, prefillOverflow) = allPrefill.addingReportingOverflow(prefill)
                    let (nextDecode, decodeOverflow) = allDecode.addingReportingOverflow(decode)
                    guard !prefillOverflow, !decodeOverflow else { return nil }
                    allPrefill = nextPrefill
                    allDecode = nextDecode
                    continue
                }
                // Rectangular MTP verification retains one decode workspace per
                // serialized verification column within the same submitted step.
                guard let overlapping = overlapPolicy.bytes(
                    prefill: prefill, decode: decode, serialDecodeCalls: serialCalls)
                else { return nil }
                let (next, totalOverflow) = total.addingReportingOverflow(overlapping)
                guard !totalOverflow else { return nil }
                total = next
            }
            if sharesStepArenas {
                var arena = 0
                for geometry in geometries {
                    guard let value = try? PagedQuantizedAttentionWorkspace.sharedArenaBytes(
                        blockSize: arenaWidth, queryHeads: geometry.queryHeads,
                        headDim: geometry.headDim, pageSize: config.pageSize,
                        maxAttendLength: length, allocationPolicy: allocationPolicy) else { return nil }
                    let (next, overflow) = arena.addingReportingOverflow(value)
                    guard !overflow else { return nil }
                    arena = next
                }
                return overlapPolicy.bytes(prefill: allPrefill, decode: allDecode,
                    serialDecodeCalls: serialCalls, sharedArena: arena)
            }
            return total
        }
        return Self(bytesForTokens: single, bytesForAggregate: { tokens, requests in
            if requests == 1 { return single(tokens) }
            guard config.pageSize > 0, let aggregateEnvelope else { return nil }
            let partitionTokens = PagedSegmentDispatchPlan.boundedPartitionTokens(
                PagedAttentionKernel.partitionTokens, pageSize: config.pageSize)
            guard let aggregate = aggregateEnvelope.bytes(
                totalTokens: tokens, maximumRequests: requests, maximumQueries: chunk,
                blockSize: 8, partitionTokens: partitionTokens,
                lookahead: CBv2PagedSpeculation.maxSpeculativeSpan)
            else { return nil }
            // Tiny requests can be dominated by fixed allocator padding. Both
            // independent bounds are valid, so keep whichever is tighter.
            guard let longest = single(tokens) else { return aggregate }
            let (fallback, overflow) = longest.multipliedReportingOverflow(by: requests)
            return overflow ? aggregate : min(aggregate, fallback)
        })
    }
}
