import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 prepaid attention workspace ownership", .serialized)
struct CBv2WorkspaceAdmissionTests {
    private func ledger(capacity: Int) -> AdmissionV2 {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
        var config = AdmissionV2.Config(watermarkFraction: 0, layerBytesPerToken: [80])
        // A checked fixture bound: 64 fixed bytes + 8/token covers two waves.
        config.workspaceProjection = .init { tokens in
            let (growth, overflow) = tokens.multipliedReportingOverflow(by: 8)
            let (total, sumOverflow) = growth.addingReportingOverflow(64)
            return overflow || sumOverflow ? nil : total
        }
        return AdmissionV2(layerKinds: [kind], bytesCapacity: capacity, config: config)
    }

    @Test("two waves consume their prepaid credit without a second capacity debit")
    func leasesShareOnlyWorkspaceCredit() throws {
        let admission = ledger(capacity: 1_472)
        #expect(admission.allocatedBytes(forTokens: 16) == 1_280 + 192)
        #expect(admission.canEverFit(promptTokens: 15, maxTokens: 1))
        #expect(!admission.canEverFit(promptTokens: 16, maxTokens: 1))
        try admission.reserve(id: .init(1), additionalTokens: 16)
        let first = try admission.reserveWorkspace(bytes: 96)
        let second = try admission.reserveWorkspace(bytes: 96)
        #expect(admission.bytesReserved == 1_472)
        #expect(throws: CBv2KVError.self) { try admission.reserveWorkspace(bytes: 1) }
        #expect(throws: CBv2KVError.self) { try admission.reserveTransient(bytes: 1) }
        first.release()
        second.release()
        #expect(admission.bytesReserved == 1_472, "future-step credit remains prepaid")
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 0)
    }

    @Test("finish before GPU completion cannot lend old workspace credit to a new request")
    func retiredCreditsCannotBeReusedPrematurely() throws {
        let admission = ledger(capacity: 1_472)
        try admission.reserve(id: .init(1), additionalTokens: 16)
        let first = try admission.reserveWorkspace(bytes: 96)
        let second = try admission.reserveWorkspace(bytes: 96)
        let next = CBv2ProjectedCapacityReservation(id: .init(2), additionalTokens: 16, additionalBytes: 0)
        #expect(!admission.canGuarantee(projectedOperations: [.release(.init(1)), .reserve(next)]))
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 192)
        #expect(throws: CBv2KVError.self) { try admission.reserve(id: .init(2), additionalTokens: 16) }
        admission.updateBytesCapacity(96)
        first.release()
        #expect(admission.bytesReserved == 96)
        #expect(throws: CBv2KVError.self) { try admission.reserveWorkspace(bytes: 1) }
        second.release()
        #expect(admission.bytesReserved == 0)
        admission.updateBytesCapacity(1_472)
        try admission.reserve(id: .init(2), additionalTokens: 16)
        admission.releaseAll(id: .init(2))
    }

    @Test("rollback preserves leased scratch and detached old generations cannot release a reused id")
    func rollbackAndDetachKeepDistinctOwners() throws {
        let admission = ledger(capacity: 2_944)
        try admission.reserve(id: .init(1), additionalTokens: 16)
        let oldScratch = try admission.reserveWorkspace(bytes: 192)
        let old = admission.detachReservation(id: .init(1))
        try admission.reserve(id: .init(1), additionalTokens: 16)
        #expect(admission.bytesReserved == 2_944)
        old.release()
        #expect(admission.bytesReserved == 1_664)
        let newScratch = try admission.reserveWorkspace(bytes: 192)
        admission.unreserve(id: .init(1), tokens: 16)
        #expect(admission.bytesReserved == 384)
        oldScratch.release()
        #expect(admission.bytesReserved == 192)
        newScratch.release()
        #expect(admission.bytesReserved == 0)
    }

    @Test("private teacher rows prepay workspaces and target physical slack cannot cover scratch")
    func unscheduledAndPhysicalOwnership() throws {
        let admission = ledger(capacity: 1_472)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 1_280)
        let teacher = try admission.reserveUnscheduledRequest(maximumTokens: 16, minimumTargetBytes: 1_280)
        #expect(admission.bytesReserved == 1_472)
        let scratch = try admission.reserveWorkspace(bytes: 192)
        teacher.release()
        #expect(admission.bytesReserved == 1_472)
        physical.release(to: 0)
        #expect(admission.bytesReserved == 192)
        scratch.release()
        #expect(admission.bytesReserved == 0)
        physical.close()
    }

    @Test("checkpoint transfer prepays fresh workspace without discounting imported auxiliary state")
    func checkpointTransferAndRollbackCarryWorkspaceCredit() throws {
        for commit in [false, true] {
            // The imported auxiliary state is 1 byte larger than the local
            // zero-auxiliary spec. Workspace allowance must not hide that byte.
            let admission = ledger(capacity: 1_473)
            let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
            let stage = try admission.reserveCheckpointStage(
                targetBytes: 1_280, auxiliaryBytes: 1, scratchBytes: 0)
            let ticket = try physical.transferCheckpoint(to: 1_280, admission: admission) { previous in
                try admission.transferCheckpointStage(
                    stage, requestID: .init(1), maximumTokens: 16,
                    previousPhysicalBytes: previous, physicalBytes: 1_280)
            }
            #expect(admission.bytesReserved == 1_473)
            let scratch = try admission.reserveWorkspace(bytes: 192)
            #expect(admission.bytesReserved == 1_473)
            if commit {
                ticket.commit()
                admission.releaseAll(id: .init(1))
                physical.release(to: 0)
            } else {
                ticket.rollbackAfterDroppingOwners()
            }
            stage.closeAfterDroppingOwners()
            #expect(admission.bytesReserved == 192)
            scratch.release()
            #expect(admission.bytesReserved == 0)
            physical.close()
        }
    }

    @Test("prospective bounds cover two live prefill or batched-decode workspaces")
    func projectionDominatesSupportedRuntimeShapes() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 8)
        let config = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 1 << 28, maxPrefillChunk: 512,
            maxBufferLength: 1 << 28, segmentSizeBytes: 1 << 20,
            layerDTypes: [.bfloat16], quantization: .init())
        let projection = CBv2RequestWorkspaceProjection.quantizedPaged(
            layerKinds: [kind], config: config, maximumChunk: 512)
        for tokens in [1, 16, 33, 273, 512, 4_096] {
            let prepaid = try #require(projection.bytes(forTokens: tokens))
            let queries = min(tokens, 512)
            let prefill = try PagedQuantizedAttentionWorkspace.reservationBytes(
                queryCount: queries, blockSize: min(8, queries), queryHeads: 8,
                headDim: 256, pageSize: 16, maxAttendLength: tokens,
                maximumSegmentCount: (tokens - 1) / 16 + 2,
                broadcastTopology: true, nativeOutputBytes: queries * 8 * 256 * 4)
            #expect(prepaid >= 2 * prefill)
            for batch in [1, 2, 3, 4, 8] {
                let decode = try PagedQuantizedAttentionWorkspace.reservationBytes(
                    queryCount: batch, blockSize: batch, queryHeads: 8,
                    headDim: 256, pageSize: 16, maxAttendLength: tokens,
                    maximumSegmentCount: PagedSegmentDispatchPlan.maximumBindings + 1,
                    nativeOutputBytes: batch * 8 * 256 * 4)
                #expect(batch * prepaid >= 2 * decode)
            }
        }
        var previous = 0
        for tokens in 1 ... 600 {
            let bytes = try #require(projection.bytes(forTokens: tokens))
            #expect(bytes >= previous, "routing binary search requires a monotone workspace ceiling")
            previous = bytes
        }
    }

    @Test("one long decode row prepays the whole rectangular batch partition width")
    func heterogeneousHistoriesCannotExceedPrepaidWorkspace() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 512, kvHeads: 8, queryHeads: 64)
        let config = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 1 << 30, maxPrefillChunk: 8,
            maxBufferLength: 1 << 30, segmentSizeBytes: 1 << 20,
            layerDTypes: [.bfloat16], quantization: .init())
        for batch in [8, 24] {
            let projection = CBv2RequestWorkspaceProjection.quantizedPaged(
                layerKinds: [kind], config: config, maximumChunk: 8, maximumBatch: batch,
                maximumSerialDecodeCalls: CBv2PagedSpeculation.maxSpeculativeSpan)
            for longest in [32_768, 131_072] {
                let lengths = [longest] + Array(repeating: 1, count: batch - 1)
                let prepaid = try lengths.reduce(0) {
                    $0 + (try #require(projection.bytes(forTokens: $1)))
                }
                let actual = try PagedQuantizedAttentionWorkspace.reservationBytes(
                    queryCount: batch, blockSize: batch, queryHeads: 64, headDim: 512,
                    pageSize: 16, maxAttendLength: longest,
                    maximumSegmentCount: 256, nativeOutputBytes: batch * 64 * 512 * 4)
                let twoVerificationWaves = 2 * CBv2PagedSpeculation.maxSpeculativeSpan * actual
                #expect(prepaid >= twoVerificationWaves)
                #expect(try #require(projection.bytes(forTokens: longest)) >= twoVerificationWaves,
                        "the longest row covers rectangular MTP columns and two live waves")
            }
        }
    }

    @Test("unrelated resident segments cannot inflate a short prefill's reserved workspace")
    func staleResidentSegmentsDoNotChangeRequestBound() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)
        let config = PagedKVPoolConfig(
            // Hundreds of tiny physical segments include allocator size-class
            // padding. This fixture intentionally funds that backing overhead.
            capacityBytes: 64 << 20, dtype: .bfloat16, maxPrefillChunk: 128,
            nominalMaxSequenceLength: 100_000, maxBufferLength: 16 << 20,
            segmentSizeBytes: 32_768, layerDTypes: [.bfloat16],
            quantization: .init(rotationBlockSize: 64))
        let backend = try PagedKVBackend(layerKinds: [kind], config: config)
        let unrelated = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 100_000)
        defer { backend.release(unrelated) }
        let short = try backend.makeSequenceState(layerKinds: [kind], promptLength: 128, maxLength: 128)
        defer { backend.release(short) }
        let group = backend.pool.group(backend.pool.groupKey(forLayer: 0))
        #expect(group.segments.count >= 256)
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([try #require(short[0])])
        let kv = MLXArray.ones([1, 1, 128, 64], dtype: .bfloat16)
        let output = cache.updateAndAttend(queries: kv, keys: kv, values: kv, scale: 0.125, sinks: nil)
        let leases = backend.pool.takePendingQuantizedScratch()
        #expect(leases.count == 1)
        eval([output] + leases.flatMap(\.evaluationTargets))
        defer { leases.forEach { $0.finishAfterSynchronization() } }
        let projection = CBv2RequestWorkspaceProjection.quantizedPaged(
            layerKinds: [kind], config: config, maximumChunk: 128)
        let prepaid = try #require(projection.bytes(forTokens: 128))
        #expect(prepaid >= 2 * leases.reduce(0) { $0 + $1.reservedBytes })
        #expect(abs(output.asType(.float32).mean().item(Float.self) - 1) < 0.01)
    }
}
