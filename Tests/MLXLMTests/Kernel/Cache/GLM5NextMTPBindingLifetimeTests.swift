import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextMTPBindingLifetimeTests {
        private final class Identity {}

        private func resolved() throws -> (CBv2ResolvedMTPAdmission, AllocationFootprintPolicy) {
            let policy = try #require(Memory.allocationFootprintPolicy())
            let resolution = CBv2MTPBoundedAdmission.resolve(
                spec: .init(resident: [.init(logicalBytes: 4096)],
                            working: [.init(logicalBytes: 8192)], hostBytes: 512),
                limits: .init(maximumPrefillTokens: 4, maximumDraftTokens: 1), policy: policy)
            guard case .bounded(let value) = resolution else {
                throw CBv2MTPBindingError.missingAdmission
            }
            return (value, policy)
        }

        @Test(arguments: [false, true])
        func detachedTargetAndAssistantRetireIndependently(assistantFirst: Bool) throws {
            let (plan, policy) = try resolved()
            let process = NativePageTestProcessOwner(maximum: 64 << 20)
            let ledger = AdmissionV2(
                layerKinds: [.init(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 1)],
                bytesCapacity: 64 << 20,
                config: .init(watermarkFraction: 0, elementBytes: 4,
                              fixedBytesPerRequest: plan.fixedBytesPerRequest),
                processMemoryOwner: process)
            let id = CBv2RequestID(918)
            let target = Identity(), drafter = Identity()
            try ledger.reserve(id: id, additionalTokens: 2)
            let original = ledger.bytesReserved
            #expect(original > plan.fixedBytesPerRequest)
            let old = try ledger.issueMTPBinding(
                id: id, maximumInputs: 4, target: ObjectIdentifier(target),
                drafter: ObjectIdentifier(drafter), resolved: plan, policy: policy)
            var allocation: CBv2MTPAllocationLease? = try old.reserve(
                .resident, buffers: [.init(logicalBytes: 4096)])
            #expect(allocation != nil)
            let targetRetirement = ledger.detachReservation(id: id)
            #expect(ledger.bytesReserved == original)
            #expect(throws: CBv2MTPBindingError.staleRequest) {
                try old.validate(target: ObjectIdentifier(target),
                                 drafter: ObjectIdentifier(drafter), maximumInputs: 4)
            }
            if assistantFirst {
                allocation = nil
                #expect(ledger.bytesReserved == original - plan.fixedBytesPerRequest)
            } else {
                targetRetirement.release()
                #expect(ledger.bytesReserved == plan.fixedBytesPerRequest)
            }

            try ledger.reserve(id: id, additionalTokens: 2)
            let fresh = try ledger.issueMTPBinding(
                id: id, maximumInputs: 4, target: ObjectIdentifier(target),
                drafter: ObjectIdentifier(drafter), resolved: plan, policy: policy)
            #expect(fresh.requestGeneration != old.requestGeneration)
            #expect(throws: CBv2MTPBindingError.staleRequest) {
                try old.reserve(.resident, buffers: [.init(logicalBytes: 1)])
            }
            ledger.releaseAll(id: id)
            if assistantFirst {
                targetRetirement.release()
            } else {
                allocation = nil
            }
            #expect(ledger.bytesReserved == 0)
            #expect(process.bytes == 0)
        }

        @Test func contextWithoutAllocationsIsStillInvalidAfterDetachAndReuse() throws {
            let (plan, policy) = try resolved()
            let process = NativePageTestProcessOwner(maximum: 64 << 20)
            let ledger = AdmissionV2(
                layerKinds: [], bytesCapacity: 64 << 20,
                config: .init(watermarkFraction: 0, fixedBytesPerRequest: plan.fixedBytesPerRequest),
                processMemoryOwner: process)
            let id = CBv2RequestID(919), target = Identity(), drafter = Identity()
            try ledger.reserve(id: id, additionalTokens: 1)
            let old = try ledger.issueMTPBinding(
                id: id, maximumInputs: 4, target: ObjectIdentifier(target),
                drafter: ObjectIdentifier(drafter), resolved: plan, policy: policy)
            let detached = ledger.detachReservation(id: id)
            detached.release()
            #expect(ledger.bytesReserved == 0)
            try ledger.reserve(id: id, additionalTokens: 1)
            let fresh = try ledger.issueMTPBinding(
                id: id, maximumInputs: 4, target: ObjectIdentifier(target),
                drafter: ObjectIdentifier(drafter), resolved: plan, policy: policy)
            #expect(fresh.requestGeneration != old.requestGeneration)
            #expect(throws: CBv2MTPBindingError.staleRequest) {
                try old.validate(target: ObjectIdentifier(target),
                                 drafter: ObjectIdentifier(drafter), maximumInputs: 4)
            }
            ledger.releaseAll(id: id)
            #expect(ledger.bytesReserved == 0 && process.bytes == 0)
        }
    }
}
