import MLX
import Testing

@testable import MLXLMCommon

private class AuxiliaryAdmissionModel: CBv2SteppableModel {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("Admission must not execute a model")
    }
}

private final class TargetAuxiliaryModel: AuxiliaryAdmissionModel,
    CBv2TargetAuxiliaryAllocationProviding
{
    let cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]?
    init(_ specs: [CBv2AuxiliaryAllocationSpec]?) {
        cbv2TargetAuxiliaryAllocationSpecs = specs
    }
}

@Suite("Target side-state admission", .serialized)
struct CBv2TargetAuxiliaryAdmissionTests {
    private let target = CBv2AuxiliaryAllocationSpec(
        bytesPerToken: 512, allocationCount: 2, tokenGranularity: 256, tokenPadding: 4)

    @Test func modelsWithoutSideStatePreserveExistingAdmission() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        for model: any CBv2SteppableModel in [AuxiliaryAdmissionModel(), TargetAuxiliaryModel([])] {
            var config = AdmissionV2.Config(watermarkFraction: 0, elementBytes: 2,
                layerElementBytes: nil, fixedBytesPerRequest: 123,
                auxiliaryBytesPerToken: 32, auxiliaryTokenGranularity: 16,
                auxiliaryTokenAllocationPadding: 5)
            CBv2TargetAuxiliaryAdmission.apply(model: model, config: &config,
                policy: policy, draftSpecs: nil)
            #expect(config.fixedBytesPerRequest == 123)
            #expect(config.auxiliaryBytesPerToken == 32)
            #expect(config.auxiliaryTokenGranularity == 16)
            #expect(config.auxiliaryTokenAllocationPadding == 5)
            #expect(config.auxiliaryAllocationProjection == nil)
        }
    }

    @Test func mtpOffStillChargesTargetAndReleasesItsReservation() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        var config = AdmissionV2.Config(watermarkFraction: 0, fixedBytesPerRequest: 64)
        CBv2TargetAuxiliaryAdmission.apply(model: TargetAuxiliaryModel([target]),
            config: &config, policy: policy,
            draftSpecs: [.init(bytesPerToken: 65536)]) // An inactive drafter is not charged.
        let projection = try #require(config.auxiliaryAllocationProjection)
        let expected = try #require(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [target]))
        #expect(config.auxiliaryBytesPerToken == 1024)
        #expect(config.fixedBytesPerRequest == 64)
        #expect(config.auxiliaryTokenGranularity == 1)
        #expect(config.auxiliaryTokenAllocationPadding == 0)
        for count in [0, 1, 252, 253, 256, 8192, 82000] {
            #expect(projection.bytes(forTokens: count) == expected.bytes(forTokens: count))
        }
        // Other suites can allocate concurrently. Validate request-owned
        // reservations, not a process-global before/after memory counter.
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        let total = 64 + 4 * 257 + (try #require(expected.bytes(forTokens: 257)))
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: total, config: config)
        try admission.reserve(id: .init(1), additionalTokens: 257)
        #expect(admission.bytesReserved == total)
        #expect(throws: CBv2KVError.self) { try admission.reserve(id: .init(2), additionalTokens: 1) }
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 0)
        #expect(admission.nonBackendBytesReserved == 0)
    }

    @Test func mtpOnCombinesRatherThanReplacesTargetState() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        let draft = CBv2AuxiliaryAllocationSpec(bytesPerToken: 32,
            tokenGranularity: 16, tokenPadding: 5, partitioned: true)
        var config = AdmissionV2.Config(watermarkFraction: 0, elementBytes: 2,
            layerElementBytes: nil, fixedBytesPerRequest: 123, auxiliaryBytesPerToken: 32)
        CBv2TargetAuxiliaryAdmission.apply(model: TargetAuxiliaryModel([target]),
            config: &config, policy: policy, draftSpecs: [draft])
        let actual = try #require(config.auxiliaryAllocationProjection)
        let expected = try #require(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [target, draft]))
        #expect(config.auxiliaryBytesPerToken == 1056)
        for count in [1, 16, 257, 8192, 82000] {
            #expect(actual.bytes(forTokens: count) == expected.bytes(forTokens: count))
        }
    }

    @Test func legacyDrafterUsesConservativePartitionedFallback() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        var config = AdmissionV2.Config(watermarkFraction: 0, elementBytes: 2,
            layerElementBytes: nil, fixedBytesPerRequest: 0, auxiliaryBytesPerToken: 32,
            auxiliaryTokenGranularity: 16, auxiliaryTokenAllocationPadding: 5)
        CBv2TargetAuxiliaryAdmission.apply(model: TargetAuxiliaryModel([target]),
            config: &config, policy: policy, draftSpecs: nil)
        let actual = try #require(config.auxiliaryAllocationProjection)
        let expected = try #require(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [
            target, .init(bytesPerToken: 1, allocationCount: 32,
                tokenGranularity: 16, tokenPadding: 5, partitioned: true)
        ]))
        #expect(actual.bytes(forTokens: 257) == expected.bytes(forTokens: 257))
    }

    @Test func malformedAndOverflowingStateRefusesAdmission() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        let cases: [[CBv2AuxiliaryAllocationSpec]?] = [
            nil, [.init(bytesPerToken: 0)], [.init(bytesPerToken: 1, allocationCount: 0)],
            [.init(bytesPerToken: Int.max, allocationCount: 2)],
            [.init(bytesPerToken: Int.max), .init(bytesPerToken: 1)],
            [.init(bytesPerToken: 1, tokenGranularity: 0)],
            [.init(bytesPerToken: 1, tokenPadding: -1)],
        ]
        for specs in cases {
            var config = AdmissionV2.Config(watermarkFraction: 0)
            CBv2TargetAuxiliaryAdmission.apply(model: TargetAuxiliaryModel(specs),
                config: &config, policy: policy, draftSpecs: nil)
            #expect(config.fixedBytesPerRequest == Int.max)
            let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
            let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 1 << 30, config: config)
            #expect(throws: CBv2KVError.self) { try admission.reserve(id: .init(1), additionalTokens: 1) }
        }
    }

    @Test func invalidDrafterCannotSubtractFromTargetCharge() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        for rate in [-1, 32, Int.max] {
            var config = AdmissionV2.Config(watermarkFraction: 0, elementBytes: 2,
                layerElementBytes: nil, fixedBytesPerRequest: 0, auxiliaryBytesPerToken: rate)
            CBv2TargetAuxiliaryAdmission.apply(model: TargetAuxiliaryModel([target]),
                config: &config, policy: policy, draftSpecs: [])
            #expect(config.fixedBytesPerRequest == Int.max)
        }
    }
}
