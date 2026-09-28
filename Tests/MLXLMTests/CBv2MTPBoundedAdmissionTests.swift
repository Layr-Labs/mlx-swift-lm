import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

private class BoundedAdmissionTarget: CBv2MTPSteppableModel {
    var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
    var supportsRequestStatefulMTP: Bool { true }
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("admission fixture must never forward")
    }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray) {
        preconditionFailure("admission fixture must never forward")
    }
}
private final class BoundedAdmissionTargetExtras: BoundedAdmissionTarget, CBv2TargetAuxiliaryAllocationProviding {
    let cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]?
    init(_ specs: [CBv2AuxiliaryAllocationSpec]?) { cbv2TargetAuxiliaryAllocationSpecs = specs }
}
private final class BoundedAdmissionRecurrentTarget: BoundedAdmissionTarget, CBv2RecurrentMTPSteppableModel {
    var cbv2Capabilities: CBv2ModelCapabilities { .init(supportsPrefixReuse: false, supportsMTP: true) }
    var recurrentStateSpec: CBv2RecurrentStateSpec? {
        .init(layers: [.init(modelLayerIndex: 0, convShape: [1, 8], convDType: .float32,
                            ssmShape: [1, 16], ssmDType: .float32)])
    }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
                 recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray { preconditionFailure("must not forward") }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
                           recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?)
        -> (logits: MLXArray, lastHidden: MLXArray) { preconditionFailure("must not forward") }
}
private class BoundedAdmissionLegacyDrafter: CBv2MTPRequestStatefulDrafter {
    final class Capture: CBv2MTPPreparedCapture {}
    final class State: CBv2MTPRequestState {
        var committedInputCount: Int { 0 }
        var stagedInputCount: Int { 0 }
    }
    let target: BoundedAdmissionTarget
    var stateCreations = 0
    init(_ target: BoundedAdmissionTarget) { self.target = target }
    var mtpTargetIdentity: ObjectIdentifier? { target.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { .serialTarget }
    var maximumDraftTokens: Int? { 3 }
    var requestStateBytesPerToken: Int { 188_480 }
    var requestStateTokenGranularity: Int { 256 }
    var requestStateTokenAllocationPadding: Int { 12 }
    func prepare(rows: [CBv2MTPRowCapture]) -> any CBv2MTPPreparedCapture { Capture() }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: any CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray) { preconditionFailure("must not draft") }
    func makeRequestState() -> any CBv2MTPRequestState { stateCreations += 1; return State() }
    func observeCommittedTarget(_ observation: CBv2MTPCommittedTargetObservation,
                                requestState: any CBv2MTPRequestState) {}
    func draftStep(tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
                   requestState: any CBv2MTPRequestState) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("must not draft")
    }
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }
    func finalizeRound(requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
                       committedDraftTokens: MLXArray, committedTargetHidden: MLXArray) {}
    func discardRound(requestState: any CBv2MTPRequestState) {}
    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {}
}
private final class BoundedAdmissionDrafter: BoundedAdmissionLegacyDrafter, CBv2MTPBoundedAllocationProviding {
    var declarations = 0
    var declaredSpec: CBv2MTPBoundedAllocationSpec? = .init(
        resident: [.init(logicalBytes: 4096, allocationCount: 4)],
        working: [.init(logicalBytes: 1024, allocationCount: 3)], hostBytes: 256)
    func boundedRequestAllocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec? {
        declarations += 1
        return declaredSpec
    }
}

/// Charge-only process owner: it never grants materialization credit for these
/// projections. ProviderCore must separately test its scalar bridge and shared
/// process gate against EngineV2.resolvedMTPAdmission.
private final class BoundedAdmissionProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
    enum Failure: Error { case exceeded }
    private let lock = NSLock()
    private let limit: UInt64
    private var charged: UInt64 = 0
    private var observations = 0
    init(limit: UInt64) { self.limit = limit }
    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= limit else { throw Failure.exceeded }
            charged = bytes
        }
    }
    func recordMaterialization(_ bytes: UInt64) throws { lock.withLock { observations += 1 } }
    func withdrawCoverage(_ bytes: UInt64) throws {}
    func retire() {}
    var charge: UInt64 { lock.withLock { charged } }
    var materializationCalls: Int { lock.withLock { observations } }
}

final class CBv2MTPBoundedAdmissionTests: XCTestCase {
    private let counts = [1, 2, 3, 127, 128, 129, 1024, 262_144, 1_048_576]
    private let limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 2048, maximumDraftTokens: 3)
    private var spec: CBv2MTPBoundedAllocationSpec {
        .init(resident: [.init(logicalBytes: 17, allocationCount: 4), .init(logicalBytes: 1, allocationCount: 2)],
              working: [.init(logicalBytes: 33, allocationCount: 3)], hostBytes: 64)
    }
    private func value(_ resolution: CBv2MTPAdmissionResolution,
                       file: StaticString = #filePath, line: UInt = #line) throws -> CBv2ResolvedMTPAdmission {
        guard case .bounded(let result) = resolution else {
            XCTFail("unexpected refusal: \(resolution)", file: file, line: line)
            throw BoundedAdmissionProcessOwner.Failure.exceeded
        }
        return result
    }

    func testIndependentAllocatorPaddingAndContextInvariantBothResidencyLedgers() throws {
        var calls: [Int] = []
        let resolution = CBv2MTPBoundedAdmission.resolve(spec: spec, limits: limits) { bytes in
            calls.append(bytes)
            return ((bytes + 15) / 16) * 16
        }
        let resolved = try value(resolution)
        XCTAssertEqual(calls, [17, 1, 33])
        XCTAssertEqual(resolved.residentBytes, 4 * 32 + 2 * 16)
        XCTAssertEqual(resolved.workingBytes, 3 * 48)
        XCTAssertEqual(resolved.fixedBytesPerRequest, 368)
        XCTAssertEqual(resolved.auxiliaryBytesPerToken, 0)
        XCTAssertEqual(resolved.auxiliaryTokenGranularity, 1)
        XCTAssertEqual(resolved.auxiliaryTokenAllocationPadding, 0)
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        let paged = CBv2PagedKVResidency(config: .init(capacityBytes: 1 << 30,
            maxBufferLength: 1 << 30, segmentSizeBytes: 1 << 16))
        for residency: any CBv2KVResidencyPolicy in [CBv2ContiguousKVResidency(), paged] {
            var config = AdmissionV2.Config(watermarkFraction: 0, fixedBytesPerRequest: 123)
            XCTAssertEqual(CBv2MTPBoundedAdmission.apply(resolution, to: &config), resolution)
            let owner = BoundedAdmissionProcessOwner(limit: 1 << 30)
            let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 1 << 30,
                config: config, residency: residency, processMemoryOwner: owner)
            XCTAssertEqual(admission.allocatedBytes(forTokens: 0), 0)
            for count in counts {
                let target = 4 * (try XCTUnwrap(residency.residentRows(layer: kind, tokens: count)))
                XCTAssertEqual(admission.allocatedBytes(forTokens: count) - target, 123 + resolved.fixedBytesPerRequest)
                try admission.reserve(id: .init(1), additionalTokens: count)
                XCTAssertEqual(admission.nonBackendBytesReserved, 491)
                XCTAssertEqual(owner.charge, UInt64(admission.bytesReserved))
                if count > 1 {
                    admission.unreserve(id: .init(1), tokens: count - 1)
                    XCTAssertEqual(admission.nonBackendBytesReserved, 491)
                }
                admission.releaseAll(id: .init(1))
                XCTAssertEqual(admission.bytesReserved, 0)
                XCTAssertEqual(owner.charge, 0)
            }
            XCTAssertEqual(owner.materializationCalls, 0)
            XCTAssertEqual(admission.allocatedBytes(forTokens: Int.max), Int.max)
        }
    }

    func testInvalidLimitsDeclarationPolicyAndCheckedOverflowRefuse() throws {
        XCTAssertEqual(CBv2MTPBoundedAdmission.resolve(spec: spec, limits: limits, policy: nil),
                       .unavailable(.allocatorPolicyUnavailable))
        for bad in [0, -1] {
            XCTAssertEqual(CBv2MTPBoundedAdmission.resolve(spec: spec,
                limits: .init(maximumPrefillTokens: bad, maximumDraftTokens: 3), upperBound: { $0 }),
                .unavailable(.invalidLimits))
            XCTAssertEqual(CBv2MTPBoundedAdmission.resolve(spec: spec,
                limits: .init(maximumPrefillTokens: 1, maximumDraftTokens: bad), upperBound: { $0 }),
                .unavailable(.invalidLimits))
        }
        let malformed: [CBv2MTPBoundedAllocationSpec?] = [nil,
            .init(resident: [], working: spec.working, hostBytes: 0),
            .init(resident: spec.resident, working: [], hostBytes: 0),
            .init(resident: [.init(logicalBytes: 0)], working: spec.working, hostBytes: 0),
            .init(resident: [.init(logicalBytes: 1, allocationCount: -1)], working: spec.working, hostBytes: 0),
            .init(resident: spec.resident, working: spec.working, hostBytes: -1),
            .init(resident: [.init(logicalBytes: Int.max, allocationCount: 2)], working: spec.working, hostBytes: 0),
            .init(resident: [.init(logicalBytes: Int.max)], working: spec.working, hostBytes: 0)]
        for declaration in malformed {
            let resolution = CBv2MTPBoundedAdmission.resolve(spec: declaration, limits: limits, upperBound: { $0 })
            guard case .unavailable = resolution else { return XCTFail("malformed declaration admitted") }
            var config = AdmissionV2.Config()
            CBv2MTPBoundedAdmission.apply(resolution, to: &config)
            XCTAssertEqual(config.fixedBytesPerRequest, Int.max)
        }
        for bound: (Int) -> Int? in [{ _ in nil }, { $0 - 1 }] {
            XCTAssertEqual(CBv2MTPBoundedAdmission.resolve(spec: spec, limits: limits, upperBound: bound),
                           .unavailable(.unrepresentableAllocation))
        }
        let resolved = CBv2MTPBoundedAdmission.resolve(spec: spec, limits: limits, upperBound: { $0 })
        var overflow = AdmissionV2.Config(fixedBytesPerRequest: Int.max)
        XCTAssertEqual(CBv2MTPBoundedAdmission.apply(resolved, to: &overflow), .unavailable(.chargeOverflow))
        var negative = AdmissionV2.Config(fixedBytesPerRequest: -1)
        XCTAssertEqual(CBv2MTPBoundedAdmission.apply(resolved, to: &negative), .unavailable(.invalidExistingCharge))
    }

    func testTightNativeAndProcessCapsRejectWithoutLeakingCharge() throws {
        let resolution = CBv2MTPBoundedAdmission.resolve(spec: spec, limits: limits, upperBound: { $0 })
        var config = AdmissionV2.Config(watermarkFraction: 0)
        CBv2MTPBoundedAdmission.apply(resolution, to: &config)
        let fixed = config.fixedBytesPerRequest
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        for nativeTight in [false, true] {
            let owner = BoundedAdmissionProcessOwner(limit: UInt64(nativeTight ? fixed + 4 : fixed + 3))
            let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: nativeTight ? fixed + 3 : fixed + 4,
                                       config: config, processMemoryOwner: owner)
            XCTAssertThrowsError(try admission.reserve(id: .init(1), additionalTokens: 1))
            XCTAssertEqual(owner.charge, 0)
            XCTAssertEqual(admission.bytesReserved, 0)
        }
    }

    func testCallerProjectionIsExactWithoutTargetAndConservativeSumWithTarget() throws {
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let caller = try XCTUnwrap(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [
            .init(bytesPerToken: 13, allocationCount: 2, tokenGranularity: 128, tokenPadding: 5)]))
        XCTAssertEqual(caller.bytes(forTokens: 0), 0) // telescoping proof's required base
        var previous = 0
        for n in 0...2049 {
            let current = try XCTUnwrap(caller.bytes(forTokens: n))
            XCTAssertGreaterThanOrEqual(current, previous)
            XCTAssertLessThanOrEqual(current - previous, caller.maximumGrowthBytes)
            XCTAssertLessThanOrEqual(current, n * caller.maximumGrowthBytes)
            previous = current
        }
        var original = AdmissionV2.Config(watermarkFraction: 0, elementBytes: 2,
            layerElementBytes: nil, fixedBytesPerRequest: 123, auxiliaryBytesPerToken: 17,
            auxiliaryTokenGranularity: 16, auxiliaryTokenAllocationPadding: 7)
        original.auxiliaryAllocationProjection = caller
        let resolution = CBv2MTPBoundedAdmission.resolve(spec: spec, limits: limits, policy: policy)
        let fixed = try value(resolution).fixedBytesPerRequest
        let extra = CBv2AuxiliaryAllocationSpec(bytesPerToken: 31, allocationCount: 2,
            tokenGranularity: 256, tokenPadding: 3)
        let targetProjection = try XCTUnwrap(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [extra]))
        for hasTarget in [false, true] {
            let target: BoundedAdmissionTarget = hasTarget ? BoundedAdmissionTargetExtras([extra]) : BoundedAdmissionTarget()
            var config = original
            CBv2MTPBoundedAdmission.apply(resolution, to: &config)
            CBv2MTPBoundedAdmission.preserveCallerProjectionForTarget(model: target, config: &config)
            CBv2TargetAuxiliaryAdmission.apply(model: target, config: &config, policy: policy, draftSpecs: nil)
            XCTAssertEqual(config.fixedBytesPerRequest, 123 + fixed)
            let ledger = AdmissionV2(layerKinds: [], bytesCapacity: Int.max, config: config)
            for n in counts + [120, 121, 122, 123, 124, 251, 252, 253, 255, 256, 257] {
                let scalar = ((n + 7 + 15) / 16) * 16 * 17
                let callerBytes = max(scalar, try XCTUnwrap(caller.bytes(forTokens: n)))
                let targetBytes = hasTarget ? try XCTUnwrap(targetProjection.bytes(forTokens: n)) : 0
                let expected = 123 + fixed + callerBytes + targetBytes
                if hasTarget { XCTAssertGreaterThanOrEqual(ledger.allocatedBytes(forTokens: n), expected) }
                else { XCTAssertEqual(ledger.allocatedBytes(forTokens: n), expected) }
            }
            if !hasTarget {
                XCTAssertEqual(config.auxiliaryBytesPerToken, 17)
                XCTAssertEqual(config.auxiliaryTokenGranularity, 16)
                XCTAssertEqual(config.auxiliaryTokenAllocationPadding, 7)
            }
        }
    }

    func testMiMoMetadataBoundRetainsFourBanksWithoutChangingContext() throws {
        let config = try MiMoV26MTPChecks.config()
        let small = try XCTUnwrap(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: config,
            limits: .init(maximumPrefillTokens: 8, maximumDraftTokens: 3)))
        let large = try XCTUnwrap(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: config, limits: limits))
        XCTAssertEqual(small.resident, large.resident)
        XCTAssertEqual(small.resident[0].allocationCount, 12)
        XCTAssertEqual(small.resident[1].allocationCount, 12)
        XCTAssertEqual(small.resident[0].logicalBytes, 1 * 3 * 4 * 4)
        XCTAssertEqual(small.resident[1].logicalBytes, 1 * 3 * 2 * 4)
        let a = try value(CBv2MTPBoundedAdmission.resolve(spec: small, limits: limits, upperBound: { $0 }))
        let b = try value(CBv2MTPBoundedAdmission.resolve(spec: large, limits: limits, upperBound: { $0 }))
        XCTAssertEqual(a.residentBytes, b.residentBytes)
        XCTAssertGreaterThan(b.workingBytes, a.workingBytes)
        XCTAssertEqual(config.maxPositionEmbeddings, 64)
        XCTAssertNil(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: config,
            limits: .init(maximumPrefillTokens: Int.max, maximumDraftTokens: 3)))
        XCTAssertNil(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: config,
            limits: .init(maximumPrefillTokens: 1, maximumDraftTokens: 4)))
    }

    func testSchedulerRetainsConstructionEnvelopeWhenCallerChangesConfig() {
        var caller = CBv2SchedulerConfig(maxBatchedTokensPerStep: 128,
            prefillChunkSize: 64, soloPrefillStripeTokens: 512)
        let scheduler = SchedulerV2(config: caller)
        caller.maxBatchedTokensPerStep = 4096
        caller.soloPrefillStripeTokens = 8192
        scheduler.mixedStepPrefillTokenCap = 32 // a lower secondary cap only
        XCTAssertEqual(scheduler.config.maxBatchedTokensPerStep, 128)
        XCTAssertEqual(scheduler.config.soloPrefillStripeTokens, 512)
    }

    // Prepared native cells: coordinator's exclusive lane only. No native
    // execution, allocator peak or speed claim is made by this source packet.
    private func requireNativeLane() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST"] == "1" else {
            throw XCTSkip("Requires coordinator's exclusive native lane")
        }
    }

    func testActualEngineConsumesBoundForContiguousAndSegmentedAndLeavesLegacyZeroAlone() async throws {
        try requireNativeLane()
        let target = BoundedAdmissionTarget(), bounded = BoundedAdmissionDrafter(target)
        let legacy = BoundedAdmissionLegacyDrafter(target)
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)]
        for paged in [false, true] {
            for (drafter, depth) in [(bounded as BoundedAdmissionLegacyDrafter, 3), (legacy, 3), (bounded, 0)] {
                // Each segmented pool owns one admission lease; never rebind
                // the same native pool to successive engine instances.
                let backend: any CBv2KVBackend
                if paged {
                    backend = try PagedKVBackend(layerKinds: kinds, config: .init(capacityBytes: 32 << 20,
                        maxPrefillChunk: 128, maxBufferLength: 32 << 20, segmentSizeBytes: 64 << 10))
                } else { backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 32 << 20)) }
                let oldDeclarations = bounded.declarations
                let engine = EngineV2(model: target, layerKinds: kinds, backend: backend,
                    cacheProvider: CBv2LayerCacheBank(caches: []),
                    schedulerConfig: .init(maxBatchedTokensPerStep: 64, prefillChunkSize: 32, soloPrefillStripeTokens: 128),
                    admissionConfig: .init(watermarkFraction: 0, fixedBytesPerRequest: 123),
                    mtpDrafter: drafter, mtpConfig: .init(enabled: true, maxDraftTokens: 3,
                        fixedDraftTokens: depth, verificationMode: .serialTarget))
                let admission = try XCTUnwrap(engine.loopForTesting.capacity as? AdmissionV2)
                let driver = try XCTUnwrap(engine.loopForTesting.mtp)
                if drafter === bounded && depth > 0 {
                    let resolved = try value(XCTUnwrap(engine.resolvedMTPAdmission))
                    XCTAssertEqual(resolved.limits.maximumPrefillTokens, 128)
                    XCTAssertEqual(resolved.limits.maximumDraftTokens, 3)
                    XCTAssertEqual(engine.resolvedFixedBytesPerRequest, 123 + resolved.fixedBytesPerRequest)
                    XCTAssertEqual(admission.auxiliaryBytesPerToken, 0)
                    XCTAssertEqual(bounded.declarations, oldDeclarations + 1)
                    XCTAssertTrue(driver.tracksPersistentHistory)
                    engine.updateKVBytesCapacity(16 << 20)
                    XCTAssertEqual(engine.resolvedMTPAdmission, .bounded(resolved))
                    try admission.reserve(id: .init(1), additionalTokens: 1)
                    XCTAssertEqual(admission.nonBackendBytesReserved, engine.resolvedFixedBytesPerRequest)
                    admission.releaseAll(id: .init(1))
                } else {
                    XCTAssertNil(engine.resolvedMTPAdmission)
                    XCTAssertEqual(engine.resolvedFixedBytesPerRequest, 123)
                    XCTAssertEqual(admission.auxiliaryBytesPerToken, depth == 0 ? 0 : legacy.requestStateBytesPerToken)
                    XCTAssertEqual(admission.auxiliaryTokenGranularity, depth == 0 ? 1 : 256)
                    XCTAssertEqual(admission.auxiliaryTokenAllocationPadding, depth == 0 ? 0 : 12)
                    XCTAssertEqual(driver.tracksPersistentHistory, depth > 0)
                    XCTAssertEqual(bounded.declarations, oldDeclarations)
                }
                XCTAssertEqual(drafter.stateCreations, 0)
                await engine.shutdown()
            }
        }
    }

    func testActualBoundedEnginePreservesRecurrentAndCallerFixedAndRejectsMissingDeclaration() async throws {
        try requireNativeLane()
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let target = BoundedAdmissionRecurrentTarget(), assistant = BoundedAdmissionDrafter(target)
        let recurrent = try XCTUnwrap(target.recurrentStateSpec)
        let generations = CBv2RecurrentStateSpec.maximumLiveGenerations + 2 // depth3 stateful serial target
        let targetBytes = try recurrent.allocationBytesPerGeneration(policy: policy) * generations
        for valid in [true, false] {
            if !valid { assistant.declaredSpec = nil }
            let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 32 << 20))
            let engine = EngineV2(model: target, layerKinds: [], backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: []),
                admissionConfig: .init(watermarkFraction: 0, fixedBytesPerRequest: 123),
                mtpDrafter: assistant, mtpConfig: .init(enabled: true, maxDraftTokens: 3,
                    fixedDraftTokens: 3, verificationMode: .serialTarget))
            if valid {
                let resolved = try value(XCTUnwrap(engine.resolvedMTPAdmission))
                XCTAssertEqual(engine.resolvedFixedBytesPerRequest, targetBytes + 123 + resolved.fixedBytesPerRequest)
            } else {
                XCTAssertEqual(engine.resolvedMTPAdmission, .unavailable(.missingDeclaration))
                XCTAssertEqual(engine.resolvedFixedBytesPerRequest, Int.max)
                let admission = try XCTUnwrap(engine.loopForTesting.capacity as? AdmissionV2)
                XCTAssertThrowsError(try admission.reserve(id: .init(1), additionalTokens: 1))
                XCTAssertEqual(admission.bytesReserved, 0)
            }
            await engine.shutdown()
        }
    }

    func testActualDisabledAndAbsentDraftersRetainDefaultAccounting() async throws {
        try requireNativeLane()
        let target = BoundedAdmissionTarget(), assistant = BoundedAdmissionDrafter(target)
        for supplied in [false, true] {
            let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20))
            let engine = EngineV2(model: target, layerKinds: [], backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: []),
                admissionConfig: .init(watermarkFraction: 0, fixedBytesPerRequest: 123),
                mtpDrafter: supplied ? assistant : nil, mtpConfig: .init(enabled: false))
            let admission = try XCTUnwrap(engine.loopForTesting.capacity as? AdmissionV2)
            XCTAssertNil(engine.resolvedMTPAdmission)
            XCTAssertNil(engine.loopForTesting.mtp)
            XCTAssertEqual(engine.resolvedFixedBytesPerRequest, 123)
            XCTAssertEqual(admission.auxiliaryBytesPerToken, 0)
            XCTAssertEqual(admission.auxiliaryTokenGranularity, 1)
            XCTAssertEqual(admission.auxiliaryTokenAllocationPadding, 0)
            XCTAssertEqual(assistant.declarations, 0)
            XCTAssertEqual(assistant.stateCreations, 0)
            await engine.shutdown()
        }
    }

    func testNativeMeasuredOldAndCurrentOwnersStayInsideFourBankBoundThroughRollbackAndCancel() throws {
        try requireNativeLane()
        let (target, predictor) = try MiMoV26MTPChecks.fixture()
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: predictor)
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let envelope = CBv2MTPAllocationLimits(maximumPrefillTokens: 16, maximumDraftTokens: 3)
        let resolved = try value(CBv2MTPBoundedAdmission.resolve(
            spec: assistant.boundedRequestAllocation(limits: envelope), limits: envelope, policy: policy))
        let state = assistant.makeRequestState() as! MiMoV26MTPState
        try assistant.configureRequestState(state, maximumSequenceLength: 64)
        let tokens = MLXArray((1...11).map(Int32.init), [1, 11])
        let output = try target.forward(inputIDs: tokens)
        assistant.observeCommittedTarget(.init(tokens: tokens, hidden: output.normalizedHiddenStates), requestState: state)
        eval(assistant.evaluationTargets(for: state))
        try assistant.requestStateDidFinishEvaluation(state)
        XCTAssertLessThanOrEqual(state.materializedBytes, resolved.residentBytes)
        var token = MLXArray([Int32(7)], [1, 1])
        var hidden = output.normalizedHiddenStates[0..., (-1)..., 0...]
        for _ in 0..<3 {
            let previous = state.materializedBytes
            let proposal = assistant.draftStep(tokens: token, hidden: hidden, shortlist: nil, requestState: state)
            XCTAssertTrue(state.hasUnmeasuredResidency)
            XCTAssertEqual(state.materializedBytes, previous)
            eval([proposal.tokens, proposal.hidden] + assistant.evaluationTargets(for: state))
            StreamOrDevice.default.stream.synchronize()
            // A sum of individual backing observations can double count shared
            // old/current storage, so this is conservative, never a credit.
            let current = try assistant.evaluationTargets(for: state).reduce(0) {
                $0 + (try XCTUnwrap($1.evaluatedBufferInfo())).allocatedBytes
            }
            XCTAssertLessThanOrEqual(previous + current, resolved.residentBytes)
            try assistant.requestStateDidFinishEvaluation(state) // actual uniqueness/compactness proof
            XCTAssertFalse(state.hasUnmeasuredResidency)
            XCTAssertLessThanOrEqual(state.materializedBytes, resolved.residentBytes)
            token = proposal.tokens.reshaped([1, 1]); hidden = proposal.hidden
        }
        assistant.discardRound(requestState: state)
        eval(assistant.evaluationTargets(for: state))
        try assistant.requestStateDidFinishEvaluation(state)
        XCTAssertEqual(state.stagedInputCount, 0)
        XCTAssertLessThanOrEqual(state.materializedBytes, resolved.residentBytes)
        assistant.releaseRequestState(state)
        XCTAssertTrue(state.isReleased)
        XCTAssertEqual(state.materializedBytes, 0)
        XCTAssertEqual(state.measuredRootCount, 0)
    }
}
