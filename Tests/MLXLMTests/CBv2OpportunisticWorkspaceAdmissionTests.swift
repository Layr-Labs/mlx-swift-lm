import Testing

@testable import MLXLMCommon

@Suite("CBv2 opportunistic workspace admission", .serialized)
struct CBv2OpportunisticWorkspaceAdmissionTests {
    private func ledger(capacity: Int) -> AdmissionV2 {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
        var config = AdmissionV2.Config(watermarkFraction: 0, layerBytesPerToken: [80])
        config.workspaceProjection = .init { tokens in
            let (growth, overflow) = tokens.multipliedReportingOverflow(by: 8)
            let (total, sumOverflow) = growth.addingReportingOverflow(64)
            return overflow || sumOverflow ? nil : total
        }
        return AdmissionV2(layerKinds: [kind], bytesCapacity: capacity, config: config)
    }

    @Test("unused direct credits and future KV cannot fund optional acceleration")
    func denialPreservesAllFutureDirectWork() throws {
        let admission = ledger(capacity: 2_944)
        try admission.reserve(id: .init(1), additionalTokens: 16)
        try admission.reserve(id: .init(2), additionalTokens: 16)
        let before = admission.bytesReserved
        #expect(throws: CBv2KVError.self) { try admission.reserveOpportunisticWorkspace(bytes: 192) }
        #expect(admission.bytesReserved == before)
        #expect(admission.transientBytesReserved == 0)
        // Both full direct allowances remain usable after the rejected fast
        // candidate, even though neither had materialized at candidate time.
        let first = try admission.reserveWorkspace(bytes: 192)
        let second = try admission.reserveWorkspace(bytes: 192)
        #expect(admission.bytesReserved == before)
        first.release()
        second.release()
        admission.releaseAll(id: .init(1))
        admission.releaseAll(id: .init(2))
        #expect(admission.bytesReserved == 0)
    }

    @Test("the complete fast candidate is additive and refunded before direct fallback")
    func completeCandidateHasNoPartialDebitOnFailure() throws {
        let admission = ledger(capacity: 1_572)
        try admission.reserve(id: .init(1), additionalTokens: 16)
        // Arena64 + output64 must be checked together. Checking only arena64
        // would succeed and strand a partial reservation before output fails.
        #expect(throws: CBv2KVError.self) { try admission.reserveOpportunisticWorkspace(bytes: 128) }
        #expect(admission.bytesReserved == 1_472)
        #expect(admission.transientBytesReserved == 0)
        let candidate = try admission.reserveOpportunisticWorkspace(bytes: 100)
        #expect(admission.bytesReserved == 1_572)
        candidate.release() // rejected shape before allocation/write
        candidate.release() // owner handoff/release is idempotent
        #expect(admission.bytesReserved == 1_472)
        let direct = try admission.reserveWorkspace(bytes: 192)
        #expect(admission.bytesReserved == 1_472)
        direct.release()
        admission.releaseAll(id: .init(1))
    }

    @Test("shared fast arena is charged once while each call keeps its output owner")
    func sharedArenaAndFreshOutputsRemainDistinctOwners() throws {
        let admission = ledger(capacity: 1_672)
        try admission.reserve(id: .init(1), additionalTokens: 16)
        let first = try admission.reserveOpportunisticWorkspace(bytes: 100 + 50)
        let second = try admission.reserveOpportunisticWorkspace(bytes: 50)
        #expect(admission.bytesReserved == 1_672)
        let direct = try admission.reserveWorkspace(bytes: 192)
        #expect(admission.bytesReserved == 1_672, "later direct layers retain their entire allowance")
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 392)
        second.release()
        #expect(admission.bytesReserved == 342)
        first.release()
        #expect(admission.bytesReserved == 192)
        direct.release()
        #expect(admission.bytesReserved == 0)
    }

    @Test("two fast steps survive cancellation and a capacity reduction until their own retirement")
    func cancellationResizeAndTwoSteps() throws {
        let admission = ledger(capacity: 1_772)
        try admission.reserve(id: .init(1), additionalTokens: 16)
        let first = try admission.reserveOpportunisticWorkspace(bytes: 150)
        let second = try admission.reserveOpportunisticWorkspace(bytes: 150)
        let direct = try admission.reserveWorkspace(bytes: 192)
        admission.updateBytesCapacity(1_000)
        #expect(admission.bytesReserved == 1_772)
        #expect(throws: CBv2KVError.self) { try admission.reserveOpportunisticWorkspace(bytes: 1) }
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 492)
        first.release()
        #expect(admission.bytesReserved == 342)
        second.release()
        #expect(admission.bytesReserved == 192)
        direct.release()
        #expect(admission.bytesReserved == 0)
    }

    @Test("physical target slack cannot hide optional workspace and private scoring keeps its lease")
    func physicalAndPrivateScoringOwners() throws {
        let admission = ledger(capacity: 1_772)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 1_280)
        let teacher = try admission.reserveUnscheduledRequest(maximumTokens: 16, minimumTargetBytes: 1_280)
        let fast = try admission.reserveOpportunisticWorkspace(bytes: 300)
        #expect(admission.bytesReserved == 1_772)
        #expect(throws: CBv2KVError.self) { try admission.reserveOpportunisticWorkspace(bytes: 1) }
        teacher.release()
        #expect(admission.bytesReserved == 1_580)
        physical.release(to: 0)
        #expect(admission.bytesReserved == 300)
        fast.release()
        #expect(admission.bytesReserved == 0)
        physical.close()
    }
}
