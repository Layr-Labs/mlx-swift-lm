// Copyright © 2026 Eigen Labs.

import Foundation

/// Presence requires native-media evidence for the incoming target. Missing or
/// stale observations never fall back to a text rate for that target.
public struct CBv2NativeTargetPrefillPolicy: Sendable, Equatable {
    public let observation: CBv2NativeTargetPrefillRate?
    public let bootstrap: CBv2NativeMediaBootstrap?

    public init(
        observation: CBv2NativeTargetPrefillRate? = nil,
        bootstrap: CBv2NativeMediaBootstrap? = nil
    ) {
        self.observation = observation
        self.bootstrap = bootstrap
    }
}

/// Caller-owned evidence acquisition. The engine independently requires its
/// only row to be this cold native target, with no in-flight step.
public struct CBv2NativeMediaBootstrap: Sendable, Equatable {
    public let promptTokens: Int
    public let validUntil: ContinuousClock.Instant
    public let evidenceGuard: CBv2FirstContentEvidenceGuard

    public init(
        promptTokens: Int, validUntil: ContinuousClock.Instant,
        evidenceGuard: CBv2FirstContentEvidenceGuard
    ) {
        self.promptTokens = promptTokens
        self.validUntil = validUntil
        self.evidenceGuard = evidenceGuard
    }

    func isValid(request: CBv2Request, clock: CBv2Clock) -> Bool {
        promptTokens > 0 && promptTokens == request.promptTokens.count
            && evidenceGuard.isValid && clock.now() <= validUntil
    }
}

/// A caller-observed rate for this prepared native media target only. It is
/// deliberately separate from the phase rate used to price pre-existing rows.
/// The caller owns model/runtime identity, sample eligibility and the measured
/// prompt domain; the engine checks that domain against its actual target.
public struct CBv2NativeTargetPrefillRate: Sendable, Equatable {
    public let tokensPerSecond: Double
    public let promptTokensMin: Int
    public let promptTokensMax: Int
    public let validUntil: ContinuousClock.Instant
    public let evidenceGuard: CBv2FirstContentEvidenceGuard

    public init(
        tokensPerSecond: Double, promptTokensMin: Int, promptTokensMax: Int,
        validUntil: ContinuousClock.Instant, evidenceGuard: CBv2FirstContentEvidenceGuard
    ) {
        self.tokensPerSecond = tokensPerSecond
        self.promptTokensMin = promptTokensMin
        self.promptTokensMax = promptTokensMax
        self.validUntil = validUntil
        self.evidenceGuard = evidenceGuard
    }

    /// A native seal has already been validated by the submission transaction.
    /// This additional check never turns raw/legacy media into a qualified
    /// target and never prices adopted-prefix work with a cold-prefill rate.
    func rate(
        request: CBv2Request, reusedPrefix: Bool, targetComputedTokens: Int,
        clock: CBv2Clock
    ) -> Double? {
        guard let media = request.multimodal, media.nativeMediaToken != nil,
            media.attention == .causal, media.positionState == nil,
            media.deepstackEmbeddings == nil, !reusedPrefix, targetComputedTokens == 0,
            promptTokensMin > 0, promptTokensMax >= promptTokensMin,
            request.promptTokens.count >= promptTokensMin,
            request.promptTokens.count <= promptTokensMax,
            tokensPerSecond.isFinite, tokensPerSecond > 0,
            evidenceGuard.isValid, clock.now() <= validUntil
        else { return nil }
        return tokensPerSecond
    }
}
