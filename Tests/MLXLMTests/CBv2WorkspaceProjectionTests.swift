import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 aggregate workspace projection", .serialized)
struct CBv2WorkspaceProjectionTests {
    @Test("engine overlap covers final prefill to decode, two decodes, and a nonchaining MTP round")
    func schedulingOverlapContract() {
        let engine = CBv2WorkspaceOverlapPolicy.engineSerialPrefill
        #expect(engine.bytes(prefill: 400, decode: 10, serialDecodeCalls: 1) == 410)
        #expect(engine.bytes(prefill: 10, decode: 40, serialDecodeCalls: 1) == 80)
        #expect(engine.bytes(prefill: 80, decode: 20, serialDecodeCalls: 8) == 160)
        #expect(CBv2WorkspaceOverlapPolicy.twoArbitrarySteps.bytes(
            prefill: 80, decode: 20, serialDecodeCalls: 8) == 320)
        #expect(engine.bytes(prefill: 40, decode: 10, serialDecodeCalls: 8, sharedArena: 8) == 88)
        #expect(engine.bytes(prefill: Int.max, decode: 1, serialDecodeCalls: 1) == nil)
    }

    @Test("aggregate polynomial dominates every small heterogeneous positive-token split")
    func exhaustiveAggregateSplits() throws {
        let envelope = CBv2WorkspaceCostEnvelope(
            constant: 13, partitions: 71, queries: 5, blocks: 11, partitionBlocks: 7)
        func exact(_ tokens: Int) -> Int {
            let p = (tokens + 2 + 6) / 7, q = min(tokens, 9), b = (q + 2) / 3
            return 13 + 71 * p + 5 * q + 11 * b + 7 * p * b
        }
        func check(_ remaining: Int, slots: Int, cost: Int, ceiling: Int) {
            if slots == 1 {
                #expect(cost + exact(remaining) <= ceiling)
                return
            }
            for next in 1 ... remaining - slots + 1 {
                check(remaining - next, slots: slots - 1, cost: cost + exact(next), ceiling: ceiling)
            }
        }
        for total in 1 ... 24 {
            for count in 1 ... min(total, 4) {
                let ceiling = try #require(envelope.bytes(
                    totalTokens: total, maximumRequests: count, maximumQueries: 9,
                    blockSize: 3, partitionTokens: 7, lookahead: 2))
                for slots in 1 ... count { check(total, slots: slots, cost: 0, ceiling: ceiling) }
            }
        }
        #expect(envelope.bytes(totalTokens: Int.max, maximumRequests: 4,
            maximumQueries: 9, blockSize: 3, partitionTokens: 7, lookahead: 2) == nil)
    }

    @Test("routing envelope dominates the shared runtime allocation bound across real geometries")
    func allocationEnvelopeCoversRuntime() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        for (heads, dimension) in [(1, 64), (16, 256), (64, 64), (16, 512)] {
            for batch in [1, 4, 24] {
                for includeArena in [true, false] {
                let envelopes = try #require(CBv2QuantizedWorkspaceCostEnvelope.layer(
                    queryHeads: heads, headDim: dimension, batch: batch, policy: policy,
                    includeArena: includeArena))
                for chunk in [8, 33, 2_048] {
                    for tokens in [1, 7, 8, 9, 128, 255, 256, 273, 512, 32_768, 131_072] {
                        let queries = min(tokens, chunk), length = tokens + 8
                        let prefill = try PagedQuantizedAttentionWorkspace.reservationBytes(
                            queryCount: queries, blockSize: min(8, queries), queryHeads: heads,
                            headDim: dimension, pageSize: 16, maxAttendLength: length,
                            maximumSegmentCount: (length - 1) / 16 + 2,
                            allocationPolicy: policy, broadcastTopology: true,
                            nativeOutputBytes: queries * heads * dimension * 4, includeArena: includeArena)
                        let decode = try PagedQuantizedAttentionWorkspace.reservationBytes(
                            queryCount: batch, blockSize: batch, queryHeads: heads,
                            headDim: dimension, pageSize: 16, maxAttendLength: length,
                            maximumSegmentCount: 100_000, allocationPolicy: policy,
                            nativeOutputBytes: batch * heads * dimension * 4, includeArena: includeArena)
                        let p = try #require(envelopes.prefill.bytes(
                            totalTokens: tokens, maximumRequests: 1, maximumQueries: chunk,
                            blockSize: 8, partitionTokens: 256, lookahead: 8))
                        let d = try #require(envelopes.decode.bytes(
                            totalTokens: tokens, maximumRequests: 1, maximumQueries: chunk,
                            blockSize: 8, partitionTokens: 256, lookahead: 8))
                        #expect(p >= prefill)
                        #expect(d >= decode)
                        let arena = try PagedQuantizedAttentionWorkspace.sharedArenaBytes(
                            blockSize: max(8, batch), queryHeads: heads, headDim: dimension,
                            pageSize: 16, maxAttendLength: length, allocationPolicy: policy)
                        let arenaEnvelope = try #require(CBv2QuantizedWorkspaceCostEnvelope.sharedArena(
                            queryHeads: heads, headDim: dimension, blockSize: max(8, batch), policy: policy))
                        #expect(try #require(arenaEnvelope.bytes(
                            totalTokens: tokens, maximumRequests: 1, maximumQueries: chunk,
                            blockSize: 8, partitionTokens: 256, lookahead: 8)) >= arena)
                    }
                }
                }
            }
        }
    }

    @Test("quantized aggregate bounds cover split, concentrated and mixed long-short traffic")
    func quantizedAggregateCoversRequestDistributions() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 16)
        let config = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 1 << 30, maxPrefillChunk: 2_048,
            maxBufferLength: 1 << 30, segmentSizeBytes: 64 << 20,
            layerDTypes: [.bfloat16], quantization: .init())
        for count in [1, 4, 24] {
            for (serial, shared) in [(1, false), (8, false), (1, true), (8, true)] {
                let projection = CBv2RequestWorkspaceProjection.quantizedPaged(
                    layerKinds: [kind], config: config, maximumChunk: 2_048,
                    maximumBatch: count, maximumSerialDecodeCalls: serial,
                    overlapPolicy: .engineSerialPrefill, sharesStepArenas: shared)
                for total in [count, max(count, 273), 32_768, 131_072] {
                    let aggregate = try #require(projection.bytes(totalTokens: total, maximumRequests: count))
                    var balanced = Array(repeating: total / count, count: count)
                    balanced[0] += total % count
                    for lengths in [
                        [total], balanced, [total - count + 1] + Array(repeating: 1, count: count - 1),
                    ] {
                        let exact = try lengths.reduce(0) { result, length in
                            result + (try #require(projection.bytes(forTokens: length)))
                        }
                        #expect(aggregate >= exact)
                    }
                    if total == 131_072 && count > 1 {
                        let longest = try #require(projection.bytes(forTokens: total))
                        #expect(aggregate < count * longest,
                            "routing must recover capacity from summing context growth once")
                    }
                }
                var previous = 0
                for total in 1 ... 1_024 {
                    let current = try #require(projection.bytes(totalTokens: total, maximumRequests: count))
                    #expect(current >= previous, "the provider solves this ceiling by binary search")
                    previous = current
                }
            }
        }
    }

    @Test("scoped projection charges one arena per geometry and all per-call owners")
    func scopedProjectionCountsGeometriesAndSerialOwners() throws {
        let repeated = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 16)
        let distinct = CBv2LayerKind(attention: .full, headDim: 512, kvHeads: 2, queryHeads: 16)
        let kinds = Array(repeating: repeated, count: 10) + [distinct]
        let config = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 1 << 30, maxPrefillChunk: 2_048,
            maxBufferLength: 1 << 30, segmentSizeBytes: 64 << 20,
            layerDTypes: Array(repeating: .bfloat16, count: kinds.count), quantization: .init())
        let tokens = 131_072, batch = 8, serial = 8, queries = 2_048, length = tokens + 8
        var prefill = 0, decode = 0, arena = 0
        for kind in kinds {
            prefill += try PagedQuantizedAttentionWorkspace.reservationBytes(
                queryCount: queries, blockSize: 8, queryHeads: kind.queryHeads, headDim: kind.headDim,
                pageSize: 16, maxAttendLength: length, maximumSegmentCount: (length - 1) / 16 + 2,
                broadcastTopology: true, nativeOutputBytes: queries * kind.queryHeads * kind.headDim * 4,
                includeArena: false)
            decode += try PagedQuantizedAttentionWorkspace.reservationBytes(
                queryCount: batch, blockSize: batch, queryHeads: kind.queryHeads, headDim: kind.headDim,
                pageSize: 16, maxAttendLength: length, maximumSegmentCount: 18,
                nativeOutputBytes: batch * kind.queryHeads * kind.headDim * 4, includeArena: false)
        }
        for kind in [repeated, distinct] {
            arena += try PagedQuantizedAttentionWorkspace.sharedArenaBytes(
                blockSize: 8, queryHeads: kind.queryHeads, headDim: kind.headDim,
                pageSize: 16, maxAttendLength: length)
        }
        let expected = try #require(CBv2WorkspaceOverlapPolicy.engineSerialPrefill.bytes(
            prefill: prefill, decode: decode, serialDecodeCalls: serial, sharedArena: arena))
        let shared = CBv2RequestWorkspaceProjection.quantizedPaged(
            layerKinds: kinds, config: config, maximumChunk: queries, maximumBatch: batch,
            maximumSerialDecodeCalls: serial, overlapPolicy: .engineSerialPrefill, sharesStepArenas: true)
        let unscoped = CBv2RequestWorkspaceProjection.quantizedPaged(
            layerKinds: kinds, config: config, maximumChunk: queries, maximumBatch: batch,
            maximumSerialDecodeCalls: serial, overlapPolicy: .engineSerialPrefill)
        #expect(shared.bytes(forTokens: tokens) == expected)
        #expect(expected >= max(prefill + decode + 2 * arena, serial * decode + arena))
        #expect(try #require(unscoped.bytes(forTokens: tokens)) > expected)
        let oneLongAndShortPeers = expected + (batch - 1) * (try #require(shared.bytes(forTokens: 1)))
        #expect(oneLongAndShortPeers >= serial * decode + arena,
            "the long row covers rectangular metadata and one shared arena, including serial MTP")
    }

    @Test("aggregate projections validate empty, invalid and overflowing inputs")
    func aggregateValidationAndFallback() throws {
        let generic = CBv2RequestWorkspaceProjection { tokens in
            let (result, overflow) = tokens.multipliedReportingOverflow(by: 16)
            return overflow ? nil : result
        }
        #expect(generic.bytes(totalTokens: 0, maximumRequests: 0) == 0)
        #expect(generic.bytes(totalTokens: 1, maximumRequests: 0) == nil)
        #expect(generic.bytes(totalTokens: -1, maximumRequests: 2) == nil)
        #expect(generic.bytes(totalTokens: 2, maximumRequests: -1) == nil)
        #expect(generic.bytes(totalTokens: 1, maximumRequests: 8) == 16)
        #expect(generic.bytes(totalTokens: 10, maximumRequests: 2) == 320)
        #expect(generic.bytes(totalTokens: Int.max / 16, maximumRequests: 2) == nil)
        let affine = CBv2RequestWorkspaceProjection(
            bytesForTokens: { 16 * $0 + 64 },
            bytesForAggregate: { 16 * $0 + 64 * $1 })
        #expect(affine.bytes(totalTokens: 10, maximumRequests: 2) == 288)
        #expect(affine.bytes(totalTokens: 1, maximumRequests: 8) == 80)
        let invalid = CBv2RequestWorkspaceProjection(
            bytesForTokens: { _ in 1 }, bytesForAggregate: { _, _ in -1 })
        #expect(invalid.bytes(totalTokens: 1, maximumRequests: 1) == nil)
    }

    @Test("production attention shapes have finite monotone workspace projections")
    func productionGeometryProjection() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        let diagnose = ProcessInfo.processInfo.environment["KV_WORKSPACE_DIAGNOSTICS"] == "1"
        if diagnose {
            print("WORKSPACE_POLICY small4=\(try #require(policy.upperBound(byteCount: 4)))"
                + " small96=\(try #require(policy.upperBound(byteCount: 96)))"
                + " small136=\(try #require(policy.upperBound(byteCount: 136)))"
                + " maximumExtra=\(try #require(policy.maximumExtraBytes))")
        }
        for (name, count, queryHeads, kvHeads, headDim) in [
            ("Qwen3.6", 10, 16, 2, 256), ("GPT-OSS", 12, 64, 8, 64),
            ("Gemma4", 5, 16, 2, 512),
        ] {
            let kinds = Array(repeating: CBv2LayerKind(
                attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads),
                count: count)
            let config = PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 1 << 30, maxPrefillChunk: 2_048,
                maxBufferLength: 1 << 30, segmentSizeBytes: 64 << 20,
                layerDTypes: Array(repeating: .bfloat16, count: count), quantization: .init())
            for batch in [1, 4, 8] {
                for serial in [1, 8] {
                    let projection = CBv2RequestWorkspaceProjection.quantizedPaged(
                        layerKinds: kinds, config: config, maximumChunk: 2_048,
                        maximumBatch: batch, maximumSerialDecodeCalls: serial,
                        overlapPolicy: .engineSerialPrefill, sharesStepArenas: true)
                    var previous = 0
                    for tokens in [32_768, 131_072] {
                        let bytes = try #require(projection.bytes(forTokens: tokens))
                        #expect(bytes >= previous)
                        previous = bytes
                        if diagnose {
                            print("WORKSPACE_MODEL name=\(name) tokens=\(tokens) batch=\(batch)"
                                + " serial=\(serial) bytes=\(bytes)")
                        }
                    }
                }
            }
        }
    }
}
