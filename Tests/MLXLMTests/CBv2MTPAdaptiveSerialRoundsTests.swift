// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2MTP adaptive serial rounds")
struct CBv2MTPAdaptiveSerialRoundsTests {
    @Test func suppressedAdaptiveSerialPlansStayTargetOnlyWithLiveHistory() throws {
        let driver = try makeDriver(allowsAdaptiveSerialRounds: false)
        #expect(driver.suppressesAdaptiveSerialRounds)
        #expect(driver.usesMarginalPolicy)
        #expect(driver.tracksPersistentHistory)
        for _ in 0 ..< 48 {
            driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
            #expect(driver.planDepth == 0)
            #expect(driver.previewDecision(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
            recordTargetOnlyStep(driver)
        }
        let fallbacks = driver.metricsSnapshot().controllerFallbacks
        #expect(fallbacks["adaptive_serial_suppressed", default: 0] > 0)
        #expect(driver.metricsSnapshot().draftedTokens == 0)
    }

    @Test func defaultAdaptiveSerialPlansStillExploreDraftDepth() throws {
        let driver = try makeDriver(allowsAdaptiveSerialRounds: true)
        #expect(!driver.suppressesAdaptiveSerialRounds)
        var positive = false
        for _ in 0 ..< 8 {
            driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
            positive = positive || driver.planDepth > 0
            recordTargetOnlyStep(driver)
        }
        #expect(positive)
    }

    @Test func fixedDepthAndBatchedVerificationAreUnaffected() throws {
        let fixed = try makeDriver(allowsAdaptiveSerialRounds: false, fixedDraftTokens: 2)
        #expect(!fixed.suppressesAdaptiveSerialRounds)
        fixed.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
        #expect(fixed.planDepth == 2)

        let rectangular = try makeDriver(
            allowsAdaptiveSerialRounds: false, requiredVerificationMode: .rectangular)
        #expect(!rectangular.suppressesAdaptiveSerialRounds)
    }

    @Test func engineSuppressionKeepsEveryPlanTargetOnlyIncludingFixedDepth() throws {
        for fixed in [nil, 2] as [Int?] {
            let driver = try makeDriver(
                allowsAdaptiveSerialRounds: true, fixedDraftTokens: fixed,
                requiredVerificationMode: .rectangular)
            #expect(driver.speculativeRoundsSuppression == nil)
            driver.suppressSpeculativeRounds(reason: "rectangular_exact_scratch_unavailable")
            #expect(driver.tracksPersistentHistory == (fixed != 0))
            for _ in 0 ..< 16 {
                driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
                #expect(driver.planDepth == 0)
                #expect(
                    driver.previewDecision(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
                recordTargetOnlyStep(driver)
            }
            let fallbacks = driver.metricsSnapshot().controllerFallbacks
            #expect(fallbacks["rectangular_exact_scratch_unavailable", default: 0] > 0)
        }
    }

    private func makeDriver(
        allowsAdaptiveSerialRounds: Bool, fixedDraftTokens: Int? = nil,
        requiredVerificationMode: CBv2MTPVerificationMode = .serialTarget
    ) throws -> CBv2MTPRoundDriver {
        let model = AdaptiveSerialTestModel()
        return try #require(
            CBv2MTPRoundDriver.build(
                model: model,
                drafter: AdaptiveSerialTestDrafter(
                    target: model, requiredVerificationMode: requiredVerificationMode),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: 3, maxSpeculativeBatch: 1,
                    fixedDraftTokens: fixedDraftTokens,
                    verificationMode: requiredVerificationMode,
                    allowsAdaptiveSerialRounds: allowsAdaptiveSerialRounds)))
    }

    private func recordTargetOnlyStep(_ driver: CBv2MTPRoundDriver) {
        driver.recordStepCost(
            CBv2MTPStepMeasurement(
                decision: driver.planDecision, actualDepth: 0, costEligible: true,
                chained: false, seedOnly: false),
            wallTimeNanos: 24_000_000, finalizedPlainWork: true,
            finalizedSeedIDs: [], finalizedVerification: false, claimedSeedCostNanos: 0)
    }
}

private final class AdaptiveSerialTestModel: CBv2MTPSteppableModel, CBv2MTPPolicyTopTwoProviding {
    var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
    var supportsRequestStatefulMTP: Bool { true }
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("controller fixture must never forward")
    }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
    {
        preconditionFailure("controller fixture must never forward")
    }
    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        preconditionFailure("controller fixture must never score")
    }
}

private final class AdaptiveSerialTestDrafter: CBv2MTPRequestStatefulDrafter {
    final class Capture: CBv2MTPPreparedCapture {}
    final class State: CBv2MTPRequestState {
        var committedInputCount: Int { 0 }
        var stagedInputCount: Int { 0 }
    }
    let target: AdaptiveSerialTestModel
    let requiredVerificationMode: CBv2MTPVerificationMode?
    init(target: AdaptiveSerialTestModel, requiredVerificationMode: CBv2MTPVerificationMode) {
        self.target = target
        self.requiredVerificationMode = requiredVerificationMode
    }
    var mtpTargetIdentity: ObjectIdentifier? { target.mtpTargetIdentity }
    var maximumDraftTokens: Int? { 3 }
    var maximumSpeculativeBatch: Int? { 1 }
    var requestStateBytesPerToken: Int { 1024 }
    func prepare(rows: [CBv2MTPRowCapture]) -> any CBv2MTPPreparedCapture { Capture() }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: any CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    { preconditionFailure("controller fixture must never draft") }
    func makeRequestState() -> any CBv2MTPRequestState { State() }
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {}
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("controller fixture must never draft")
    }
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }
    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {}
    func discardRound(requestState: any CBv2MTPRequestState) {}
    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {}
}
