import MLX
@testable import MLXLMCommon
import XCTest

/// Metadata-only driver admission. No forward, tensor, state allocation or
/// token-quality claim; actual native engine parity remains a separate suite.
final class CBv2StatefulAttentionMTPActivationTests: XCTestCase {
    private final class Owner {}
    private class Target: CBv2MTPSteppableModel {
        let owner: Owner
        let supported: Bool
        var hasIdentity = true
        init(_ owner: Owner, supported: Bool) { self.owner = owner; self.supported = supported }
        var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
        var supportsRequestStatefulMTP: Bool { supported }
        var mtpTargetIdentity: ObjectIdentifier? { hasIdentity ? ObjectIdentifier(owner) : nil }
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            preconditionFailure("admission must not run target")
        }
        func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
            -> (logits: MLXArray, lastHidden: MLXArray) {
            preconditionFailure("admission must not run target")
        }
    }
    private final class MissingRecurrentSpec: Target, CBv2RecurrentMTPSteppableModel {
        var cbv2Capabilities: CBv2ModelCapabilities { .initialRecurrentTarget }
        var recurrentStateSpec: CBv2RecurrentStateSpec? { nil }
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
                     recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
            preconditionFailure("missing recurrent state cannot run")
        }
        func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?)
            -> (logits: MLXArray, lastHidden: MLXArray) {
            preconditionFailure("missing recurrent state cannot run")
        }
    }
    private final class State: CBv2MTPRequestState {
        var committedInputCount: Int { 0 }
        var stagedInputCount: Int { 0 }
    }
    private final class Capture: CBv2MTPPreparedCapture {}
    private final class Drafter: CBv2MTPRequestStatefulDrafter {
        let owner: Owner
        var hasIdentity = true
        var stateCreations = 0
        init(_ owner: Owner) { self.owner = owner }
        var mtpTargetIdentity: ObjectIdentifier? { hasIdentity ? ObjectIdentifier(owner) : nil }
        var requiredVerificationMode: CBv2MTPVerificationMode? { .serialTarget }
        var maximumDraftTokens: Int? { 3 }
        var maximumSpeculativeBatch: Int? { 1 }
        func prepare(rows: [CBv2MTPRowCapture]) -> any CBv2MTPPreparedCapture { Capture() }
        func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: any CBv2MTPPreparedCapture)
            -> (tokens: MLXArray, hidden: MLXArray) {
            preconditionFailure("admission must not draft")
        }
        func makeRequestState() -> any CBv2MTPRequestState { stateCreations += 1; return State() }
        func observeCommittedTarget(_ observation: CBv2MTPCommittedTargetObservation,
                                    requestState: any CBv2MTPRequestState) {}
        func draftStep(tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
                       requestState: any CBv2MTPRequestState)
            -> (tokens: MLXArray, hidden: MLXArray) {
            preconditionFailure("admission must not draft")
        }
        func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }
        func finalizeRound(requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
                           committedDraftTokens: MLXArray, committedTargetHidden: MLXArray) {}
        func discardRound(requestState: any CBv2MTPRequestState) {}
        func releaseRequestState(_ requestState: any CBv2MTPRequestState) {}
    }
    private var enabled: CBv2MTPConfig {
        .init(enabled: true, maxDraftTokens: 5, maxSpeculativeBatch: 4,
              fixedDraftTokens: 5, verificationMode: .automatic)
    }

    func testAttentionStatefulActivationRequiresExplicitNativeCapability() throws {
        let owner = Owner(), drafter = Drafter(owner)
        XCTAssertNil(CBv2MTPRoundDriver.build(model: Target(owner, supported: false),
            drafter: drafter, config: enabled))
        let driver = try XCTUnwrap(CBv2MTPRoundDriver.build(
            model: Target(owner, supported: true), drafter: drafter, config: enabled))
        XCTAssertTrue(driver.usesRequestStatefulDrafter)
        XCTAssertEqual(driver.config.verificationMode, .serialTarget)
        XCTAssertEqual(driver.config.maxDraftTokens, 3)
        XCTAssertEqual(driver.config.fixedDraftTokens, 3)
        XCTAssertEqual(driver.config.maxSpeculativeBatch, 1)
        XCTAssertEqual(drafter.stateCreations, 0)
    }

    func testForeignAndMissingTargetIdentityRemainRefused() {
        let owner = Owner(), target = Target(owner, supported: true), drafter = Drafter(owner)
        XCTAssertNil(CBv2MTPRoundDriver.build(model: target, drafter: Drafter(Owner()), config: enabled))
        target.hasIdentity = false
        XCTAssertNil(CBv2MTPRoundDriver.build(model: target, drafter: drafter, config: enabled))
        target.hasIdentity = true; drafter.hasIdentity = false
        XCTAssertNil(CBv2MTPRoundDriver.build(model: target, drafter: drafter, config: enabled))
        XCTAssertEqual(drafter.stateCreations, 0)
    }

    func testMissingRecurrentSpecCannotMasqueradeAsAttentionOnly() {
        let owner = Owner(), drafter = Drafter(owner)
        XCTAssertNil(CBv2MTPRoundDriver.build(model: MissingRecurrentSpec(owner, supported: true),
            drafter: drafter, config: enabled))
        XCTAssertEqual(drafter.stateCreations, 0)
    }

    func testDisabledOrMissingAssistantDoesNotActivate() {
        let owner = Owner(), target = Target(owner, supported: true), drafter = Drafter(owner)
        var disabled = enabled; disabled.enabled = false
        XCTAssertNil(CBv2MTPRoundDriver.build(model: target, drafter: drafter, config: disabled))
        XCTAssertNil(CBv2MTPRoundDriver.build(model: target, drafter: nil, config: enabled))
        XCTAssertEqual(drafter.stateCreations, 0)
    }

    func testExistingDraftersDoNotOptIntoObservationFencingByDefault() {
        XCTAssertFalse(Drafter(Owner()).requiresCommittedObservationFence)
        XCTAssertNoThrow(try Drafter(Owner()).requestStateDidFinishEvaluation(State()))
    }

    func testPartialAssistantResidencyRemainsExplicitAndDeduplicated() throws {
        final class Measured: CBv2MTPRequestResidencyReporting {
            var committedInputCount = 0, stagedInputCount = 0
            var materializedBytes = 4096
            var hasUnmeasuredResidency = true
        }
        let owner = Owner(), drafter = Drafter(owner)
        let driver = try XCTUnwrap(CBv2MTPRoundDriver.build(
            model: Target(owner, supported: true), drafter: drafter, config: enabled))
        let state = Measured()
        driver.restoreAssistantState(state, for: .init(1))
        XCTAssertEqual(driver.materializedAssistantBytes(detachedStates: [state, state]), 4096)
        XCTAssertTrue(driver.hasUnmeasuredAssistantResidency())
        state.hasUnmeasuredResidency = false
        XCTAssertFalse(driver.hasUnmeasuredAssistantResidency(detachedStates: [state]))
        _ = driver.takeAssistantState(for: .init(1))
        state.hasUnmeasuredResidency = true
        XCTAssertFalse(driver.hasUnmeasuredAssistantResidency())
        XCTAssertTrue(driver.hasUnmeasuredAssistantResidency(detachedStates: [state]))
    }

    func testCapacitySnapshotDoesNotSilentlyLabelPartialAssistantTotal() {
        let legacy = CBv2CapacitySnapshot(activeRequests: 0, waitingRequests: 0,
            kvBytesInUse: 0, kvBytesCapacity: 8192, activeTokens: 0)
        XCTAssertFalse(legacy.hasUnmeasuredAssistantResidency)
        let transition = CBv2CapacitySnapshot(activeRequests: 1, waitingRequests: 0,
            kvBytesInUse: 4096, kvBytesCapacity: 8192, kvBytesReserved: 8192,
            activeTokens: 1, hasUnmeasuredAssistantResidency: true)
        XCTAssertTrue(transition.hasUnmeasuredAssistantResidency)
        XCTAssertEqual(transition.kvBytesReserved, 8192)
        XCTAssertEqual(transition.kvBytesInUse, 4096)
    }
}
