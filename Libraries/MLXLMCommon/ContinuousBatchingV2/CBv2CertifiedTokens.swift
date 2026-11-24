// CBv2CertifiedTokens.swift
//
// Opt-in seam for models that choose decode tokens from their final hidden
// states with a proof that each token equals the one the full vocabulary
// product and the default sampler would give, while reading only part of the
// unembedding.

import Foundation
import MLX

/// A model that can fulfill a `CBv2CertifiedTokenRequest`. Models that do not
/// conform are never asked, so their steps are unchanged.
public protocol CBv2CertifiedTokenModel: AnyObject {
    /// False keeps every step on the full product (the model's own switch).
    var cbv2SupportsCertifiedTokens: Bool { get }
    /// The vocabulary size of the model's logits (the sampler is configured for it before the
    /// forward).
    var cbv2CertifiedVocabularySize: Int { get }
    /// Builds whatever the certified path needs and compiles its kernels, so no request pays for
    /// it. EngineV2 calls it once at construction when the seam is on.
    func cbv2WarmCertifiedTokens()
}

extension CBv2CertifiedTokenModel {
    public func cbv2WarmCertifiedTokens() {}
}

/// One decode forward's offer, set as `current` around the forward. The default sampler makes
/// it (`CBv2DefaultSampler.planCertifiedTokens`) only when its configured draw for these rows
/// is `argmax(logits)` (greedy rows) or `argmax(softmax(logits / T) / noise)` (temperature-only
/// rows) with nothing else applied, and the engine only when no other reader needs the step's
/// logits (diagnostics, prompt-frontier capture, MTP history). A model may fulfill it and
/// return placeholder logits; if it does not, it returns real logits and the engine samples them
/// as before. `DARKBLOOM_CERTIFIED_TOKENS=0` disables the seam.
public final class CBv2CertifiedTokenRequest: @unchecked Sendable {
    @TaskLocal public static var current: CBv2CertifiedTokenRequest?

    public static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CERTIFIED_TOKENS"]?.lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Rows of the step, in batch order.
    public var rows: Int { greedy.count }
    public let vocabularySize: Int
    /// Per row: argmax of the logits (true) or the temperature-only Gumbel race (false).
    public let greedy: [Bool]
    /// The sampler pipeline's `[B, 1]` temperature divisor (greedy rows 1), nil when no row
    /// divides. A certified race must divide by exactly this tensor.
    public let temperature: MLXArray?
    /// The `[B, vocab]` Exp(1) noise the sampler would draw this step (greedy rows 1), nil when
    /// every row is greedy. The sampler's token is `argmax(softmax(logits / T) / noise)`.
    public let noise: MLXArray?

    public private(set) var tokens: MLXArray?
    public private(set) var nonFinite: MLXArray?
    public private(set) var candidateCounts: MLXArray?

    init(vocabularySize: Int, greedy: [Bool], temperature: MLXArray?, noise: MLXArray?) {
        self.vocabularySize = vocabularySize
        self.greedy = greedy
        self.temperature = temperature
        self.noise = noise
    }

    public var isFulfilled: Bool { tokens != nil }

    /// `tokens` and `nonFinite`: `[rows]` int32, the step's tokens and the engine's non-finite
    /// flags (`CBv2NonFiniteLogits.flags`) of the full logits. `candidateCounts`: model-defined
    /// telemetry, never read on the step path.
    public func fulfill(tokens: MLXArray, nonFinite: MLXArray, candidateCounts: MLXArray) {
        precondition(!isFulfilled, "certified token request fulfilled twice")
        precondition(
            tokens.shape == [rows] && nonFinite.shape == [rows],
            "certified tokens must be [rows]")
        self.tokens = tokens.asType(.int32)
        self.nonFinite = nonFinite.asType(.int32)
        self.candidateCounts = candidateCounts
    }
}

extension EngineLoopV2 {
    /// The sampler's offer for a decode step over `ids`, or nil when the model does not take
    /// offers or another consumer of the step's logits needs the full product. Configures the
    /// sampler for these rows (as `sample` would).
    func certifiedTokenRequest(
        ids: [CBv2RequestID], pendingSampledTokens: MLXArray?, otherLogitReaders: Bool
    ) -> CBv2CertifiedTokenRequest? {
        guard CBv2CertifiedTokenRequest.enabled, !otherLogitReaders, logitDiagnostic == nil,
            let sampler = sampler as? CBv2DefaultSampler,
            let certifying = model as? any CBv2CertifiedTokenModel,
            certifying.cbv2SupportsCertifiedTokens
        else { return nil }
        return sampler.planCertifiedTokens(
            vocab: certifying.cbv2CertifiedVocabularySize, requestIDs: ids,
            pendingSampledTokens: pendingSampledTokens,
            rowContext: { [scheduler] in ids.map { Self.samplerRow(scheduler.record(for: $0)!) } })
    }

    func noteCertifiedStep(_ request: CBv2CertifiedTokenRequest) {
        certifiedTokenSteps += 1
        certifiedTokenRows += request.rows
    }
}
