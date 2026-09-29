// Copyright © 2026 Eigen Labs.

import Foundation

/// A captured external-work epoch. The owner invalidates this token whenever
/// shared-machine ownership changes. Reading it on the engine queue prevents
/// a queued admission from using an obsolete cross-engine snapshot.
public final class CBv2FirstContentEvidenceGuard: @unchecked Sendable, Equatable {
    private let lock = NSLock()
    private var valid = true

    public init() {}
    public func invalidate() { lock.withLock { valid = false } }
    public var isValid: Bool { lock.withLock { valid } }
    public static func == (lhs: CBv2FirstContentEvidenceGuard, rhs: CBv2FirstContentEvidenceGuard)
        -> Bool
    {
        lhs === rhs
    }
}

/// Caller-reviewed rates and measured prediction-error envelope. Qualification
/// and artifact identity belong to the caller; the engine checks the actual
/// queue, adopted prefix and work before selecting any cell.
public struct CBv2FirstContentCalibrationCell: Sendable, Equatable {
    public var promptTokensMin: Int
    public var promptTokensMax: Int
    public var contextTokensMin: Int
    public var contextTokensMax: Int
    public var reusedPrefix: Bool
    public var contention: String
    public var prefillTokensPerSecond: Double
    public var decodeTokensPerSecond: Double
    public var maxPrefillWorkTokens: Int
    public var maxDecodeWorkTokens: Int
    public var maxActiveRequests: Int
    public var competitorProfileIDs: [String]
    public var maxOtherModelRequests: Int
    public var maxOtherModelServiceFraction: Double
    public var errorRatio: Double
    public var errorAdditiveMilliseconds: Double

    public init(
        promptTokensMin: Int, promptTokensMax: Int,
        contextTokensMin: Int, contextTokensMax: Int, reusedPrefix: Bool,
        contention: String, prefillTokensPerSecond: Double, decodeTokensPerSecond: Double,
        maxPrefillWorkTokens: Int, maxDecodeWorkTokens: Int, maxActiveRequests: Int,
        competitorProfileIDs: [String], maxOtherModelRequests: Int,
        maxOtherModelServiceFraction: Double, errorRatio: Double,
        errorAdditiveMilliseconds: Double
    ) {
        self.promptTokensMin = promptTokensMin
        self.promptTokensMax = promptTokensMax
        self.contextTokensMin = contextTokensMin
        self.contextTokensMax = contextTokensMax
        self.reusedPrefix = reusedPrefix
        self.contention = contention
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.maxPrefillWorkTokens = maxPrefillWorkTokens
        self.maxDecodeWorkTokens = maxDecodeWorkTokens
        self.maxActiveRequests = maxActiveRequests
        self.competitorProfileIDs = competitorProfileIDs
        self.maxOtherModelRequests = maxOtherModelRequests
        self.maxOtherModelServiceFraction = maxOtherModelServiceFraction
        self.errorRatio = errorRatio
        self.errorAdditiveMilliseconds = errorAdditiveMilliseconds
    }
}

public struct CBv2FirstContentCalibration: Sendable, Equatable {
    public var cells: [CBv2FirstContentCalibrationCell]
    public var evidenceGuard: CBv2FirstContentEvidenceGuard
    public var sameModelRequests: Int
    public var otherModelRequests: Int
    public var otherModelServiceFraction: Double
    public var competitorProfileIDs: [String]
    public var firstContentDecodeAllowance: Int
    public var existingContextTokensMax: Int
    public var otherModelPrefillTokens: Int
    public var otherModelDecodeTokens: Int
    public var validUntil: ContinuousClock.Instant?
    public var sameModelPrefillTokens: Int
    public var sameModelDecodeTokens: Int

    public init(
        cells: [CBv2FirstContentCalibrationCell], evidenceGuard: CBv2FirstContentEvidenceGuard,
        sameModelRequests: Int, otherModelRequests: Int, otherModelServiceFraction: Double,
        competitorProfileIDs: [String], firstContentDecodeAllowance: Int = 33,
        existingContextTokensMax: Int = 0, otherModelPrefillTokens: Int = 0,
        otherModelDecodeTokens: Int = 0, validUntil: ContinuousClock.Instant? = nil,
        sameModelPrefillTokens: Int = 0, sameModelDecodeTokens: Int = 0
    ) {
        self.cells = cells
        self.evidenceGuard = evidenceGuard
        self.sameModelRequests = sameModelRequests
        self.otherModelRequests = otherModelRequests
        self.otherModelServiceFraction = otherModelServiceFraction
        self.competitorProfileIDs = competitorProfileIDs
        self.firstContentDecodeAllowance = firstContentDecodeAllowance
        self.existingContextTokensMax = existingContextTokensMax
        self.otherModelPrefillTokens = otherModelPrefillTokens
        self.otherModelDecodeTokens = otherModelDecodeTokens
        self.validUntil = validUntil
        self.sameModelPrefillTokens = sameModelPrefillTokens
        self.sameModelDecodeTokens = sameModelDecodeTokens
    }

    /// Nil means unsupported or stale evidence; the caller's ordinary phase
    /// rates remain authoritative in that case. No deadline is rewritten.
    func serviceSeconds(
        work: CBv2FirstTokenScheduledWork, promptTokens: Int,
        reusedPrefix: Bool, activeRequests: Int, maxOutputTokens: Int,
        existingSchedulerContextTokensMax: Int = 0, targetComputedTokens: Int = 0,
        clock: CBv2Clock = .continuous
    ) -> Double? {
        guard evidenceGuard.isValid, validUntil.map({ clock.now() <= $0 }) ?? true,
            promptTokens > 0, activeRequests > 0,
            sameModelRequests > 0, otherModelRequests >= 0,
            otherModelServiceFraction.isFinite, otherModelServiceFraction >= 0,
            otherModelServiceFraction <= 1, firstContentDecodeAllowance >= 0,
            work.prefillTokens >= 0, work.decodeTokens >= 0,
            otherModelPrefillTokens >= 0, otherModelDecodeTokens >= 0,
            sameModelPrefillTokens >= 0, sameModelDecodeTokens >= 0,
            targetComputedTokens >= 0, targetComputedTokens <= promptTokens
        else { return nil }
        let (active, activeOverflow) = max(activeRequests, sameModelRequests)
            .addingReportingOverflow(otherModelRequests)
        let incomingDecode = min(max(0, maxOutputTokens), firstContentDecodeAllowance)
        let (incomingContext, contextOverflow) = promptTokens.addingReportingOverflow(
            incomingDecode)
        guard !contextOverflow else { return nil }
        let context = max(
            incomingContext, existingContextTokensMax, existingSchedulerContextTokensMax)
        let contention =
            otherModelRequests > 0 ? "other_model" : active > 1 ? "same_model" : "isolated"
        let (ownDecodeWork, overflow) = max(work.decodeTokens, sameModelDecodeTokens)
            .addingReportingOverflow(incomingDecode)
        let (decodeWork, decodeOverflow) = ownDecodeWork.addingReportingOverflow(
            otherModelDecodeTokens)
        let (leasePrefill, leaseOverflow) = sameModelPrefillTokens.addingReportingOverflow(
            promptTokens - targetComputedTokens)
        let (prefillWork, prefillOverflow) = max(work.prefillTokens, leasePrefill)
            .addingReportingOverflow(otherModelPrefillTokens)
        guard !overflow, !activeOverflow, !decodeOverflow, !prefillOverflow, !leaseOverflow else {
            return nil
        }
        var bound: Double?
        for cell in cells {
            guard cell.promptTokensMin > 0, cell.promptTokensMin <= promptTokens,
                promptTokens <= cell.promptTokensMax,
                cell.contextTokensMin > 0, cell.contextTokensMin <= context,
                context <= cell.contextTokensMax,
                cell.reusedPrefix == reusedPrefix, cell.contention == contention,
                active <= cell.maxActiveRequests,
                prefillWork <= cell.maxPrefillWorkTokens,
                decodeWork <= cell.maxDecodeWorkTokens,
                competitorProfileIDs == cell.competitorProfileIDs,
                otherModelRequests <= cell.maxOtherModelRequests,
                otherModelServiceFraction <= cell.maxOtherModelServiceFraction,
                cell.prefillTokensPerSecond.isFinite, cell.prefillTokensPerSecond > 0,
                cell.decodeTokensPerSecond.isFinite, cell.decodeTokensPerSecond > 0,
                cell.errorRatio.isFinite, cell.errorRatio >= 1,
                cell.errorAdditiveMilliseconds.isFinite, cell.errorAdditiveMilliseconds >= 0
            else { continue }
            let seconds =
                (Double(prefillWork) / cell.prefillTokensPerSecond
                    + Double(decodeWork) / cell.decodeTokensPerSecond) * cell.errorRatio
                + cell.errorAdditiveMilliseconds / 1_000
            guard seconds.isFinite, seconds >= 0, seconds <= Double(Int64.max) else { continue }
            // Overlapping reviewed envelopes must never select the optimistic
            // one merely because of catalog ordering.
            bound = max(bound ?? seconds, seconds)
        }
        return evidenceGuard.isValid && (validUntil.map({ clock.now() <= $0 }) ?? true)
            ? bound : nil
    }
}
