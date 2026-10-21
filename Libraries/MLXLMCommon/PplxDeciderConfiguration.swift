// Copyright © 2026 Eigen Labs.
// pplx-decider artifact contract: HF perplexity-ai/pplx-decider-v1-27b (decision_config.json,
// format_version 1), converted to the dense Qwen3.8-27B MLX layout with the 255-row readout
// installed as `language_model.lm_head`.
import Foundation

public enum PplxDeciderError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidLogprobs(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid pplx-decider configuration: \(detail)"
        case .invalidLogprobs(let detail):
            return "Invalid pplx-decider option logprobs: \(detail)"
        }
    }
}

/// The top-level `decision` block of a `pplx_decider_qwen35` `config.json`.
///
/// Option `i` is the output-head row `i`; `codes[i]` is its letter code and
/// `tokenIDs[i]` the tokenizer id of that code. The head has `options` rows.
public struct PplxDeciderConfiguration: Codable, Sendable, Equatable {
    public static let modelType = "pplx_decider_qwen35"
    public static let maximumOptions = 255

    public let formatVersion: Int
    public let options: Int
    public let temperature: Float
    public let codes: [String]
    public let tokenIDs: [Int]
    public let maxInputTokens: Int

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case options
        case temperature
        case codes
        case tokenIDs = "token_ids"
        case maxInputTokens = "max_input_tokens"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        formatVersion = try c.decode(Int.self, forKey: .formatVersion)
        options = try c.decode(Int.self, forKey: .options)
        temperature = try c.decode(Float.self, forKey: .temperature)
        codes = try c.decode([String].self, forKey: .codes)
        tokenIDs = try c.decode([Int].self, forKey: .tokenIDs)
        maxInputTokens = try c.decode(Int.self, forKey: .maxInputTokens)

        guard formatVersion == 1 else {
            throw PplxDeciderError.invalidConfiguration("unsupported format_version")
        }
        guard (1 ... Self.maximumOptions).contains(options) else {
            throw PplxDeciderError.invalidConfiguration("options must be in 1...255")
        }
        guard codes.count == options, tokenIDs.count == options else {
            throw PplxDeciderError.invalidConfiguration(
                "codes and token_ids must have one entry per option")
        }
        guard codes.allSatisfy(Self.isReferenceCode), Set(codes).count == codes.count else {
            throw PplxDeciderError.invalidConfiguration("codes must be unique [A-Z]{1,2}")
        }
        guard tokenIDs.allSatisfy({ $0 >= 0 }), Set(tokenIDs).count == tokenIDs.count else {
            throw PplxDeciderError.invalidConfiguration("token_ids must be unique and non-negative")
        }
        guard temperature.isFinite, temperature > 0 else {
            throw PplxDeciderError.invalidConfiguration("temperature must be finite and positive")
        }
        guard maxInputTokens > 0 else {
            throw PplxDeciderError.invalidConfiguration("max_input_tokens must be positive")
        }
    }

    private static func isReferenceCode(_ code: String) -> Bool {
        let bytes = code.utf8
        return (1 ... 2).contains(bytes.count) && bytes.allSatisfy { (65 ... 90).contains($0) }
    }

    /// Output-head width for a top-level `config.json`: `decision.options`
    /// for the decider model type, nil (the vocabulary head) otherwise.
    ///
    /// The decider model type requires the block and an untied head; a block
    /// under any other model type is rejected.
    public static func outputHeadSize(
        modelType: String, decision: PplxDeciderConfiguration?, tieWordEmbeddings: Bool
    ) throws -> Int? {
        guard modelType == Self.modelType else {
            guard decision == nil else {
                throw PplxDeciderError.invalidConfiguration(
                    "a decision block requires model_type \(Self.modelType)")
            }
            return nil
        }
        guard let decision else {
            throw PplxDeciderError.invalidConfiguration(
                "model_type \(Self.modelType) requires a decision block")
        }
        guard !tieWordEmbeddings else {
            throw PplxDeciderError.invalidConfiguration(
                "the decision head cannot be tied to the embeddings")
        }
        return decision.options
    }
}

/// Decision probabilities from the engine's output-head logprobs.
///
/// The engine returns `log_softmax` of the raw head logits at T=1. For the
/// first `count` options, `softmax(logprob / T)` equals the reference
/// `softmax(mask(logits) / T)[:count]`: the per-row logsumexp shift cancels
/// and masked rows contribute exactly zero.
public enum PplxDecision {

    /// `softmax(optionLogprobs / temperature)`, computed in Double.
    /// `temperature` must be finite and positive (see `PplxDeciderConfiguration`).
    public static func probabilities(optionLogprobs: [Float], temperature: Float) -> [Float] {
        precondition(temperature.isFinite && temperature > 0, "temperature must be positive")
        guard !optionLogprobs.isEmpty else { return [] }
        let t = Double(temperature)
        let scaled = optionLogprobs.map { Double($0) / t }
        let peak = scaled.max()!
        let weights = scaled.map { exp($0 - peak) }
        let total = weights.reduce(0, +)
        return weights.map { Float($0 / total) }
    }

    /// Ordered logprobs of options `0..<count` from a top-k list whose token
    /// ids are output-head rows. Rows at or above `count` are ignored; every
    /// row below `count` must appear exactly once with a finite logprob.
    public static func optionLogprobs(
        topLogprobs: [(token: Int, logprob: Float)], count: Int
    ) throws -> [Float] {
        guard (1 ... PplxDeciderConfiguration.maximumOptions).contains(count) else {
            throw PplxDeciderError.invalidLogprobs("option count must be in 1...255")
        }
        var ordered = [Float?](repeating: nil, count: count)
        for entry in topLogprobs where entry.token >= 0 && entry.token < count {
            guard ordered[entry.token] == nil else {
                throw PplxDeciderError.invalidLogprobs("option \(entry.token) appears twice")
            }
            guard entry.logprob.isFinite else {
                throw PplxDeciderError.invalidLogprobs("option \(entry.token) is not finite")
            }
            ordered[entry.token] = entry.logprob
        }
        guard let missing = ordered.firstIndex(where: { $0 == nil }) else {
            return ordered.map { $0! }
        }
        throw PplxDeciderError.invalidLogprobs("option \(missing) is missing")
    }
}
