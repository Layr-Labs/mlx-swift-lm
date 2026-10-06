// DiffusionGemma generation contract, based on the Apache-2.0 Transformers reference:
// https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py

import Foundation

/// Native block-diffusion controls. These are not autoregressive sampling settings.
/// In particular, the default output budget is not the model's context capacity.
/// Configuration decoding does not register or qualify a model implementation.
public struct DiffusionGemmaGenerationConfiguration: Codable, Sendable, Equatable {
    public struct EntropyBoundSampler: Codable, Sendable, Equatable {
        public let className: String
        public let entropyBound: Float

        enum CodingKeys: String, CodingKey, CaseIterable {
            case className = "_cls_name"
            case entropyBound = "entropy_bound"
        }

        public init(entropyBound: Float = 0.1) throws {
            guard entropyBound.isFinite, entropyBound > 0 else {
                throw DiffusionGemmaGenerationError.invalidConfiguration("entropy_bound")
            }
            self.className = "EntropyBoundSamplerConfig"
            self.entropyBound = entropyBound
        }

        public init(from decoder: Decoder) throws {
            let keys = try decoder.container(keyedBy: AnyKey.self)
            let supported = Set(CodingKeys.allCases.map(\.rawValue))
            for key in keys.allKeys where !supported.contains(key.stringValue) {
                throw DiffusionGemmaGenerationError.unsupportedField(
                    "sampler_config." + key.stringValue)
            }
            let values = try decoder.container(keyedBy: CodingKeys.self)
            let name = try values.decode(String.self, forKey: .className)
            guard name == "EntropyBoundSamplerConfig" else {
                throw DiffusionGemmaGenerationError.unsupportedSampler(name)
            }
            try self.init(entropyBound: values.decode(Float.self, forKey: .entropyBound))
        }
    }

    public let maxNewTokens: Int
    public let maxLength: Int?
    public let maxDenoisingSteps: Int
    public let sampler: EntropyBoundSampler
    public let minimumTemperature: Float
    public let maximumTemperature: Float
    public let stabilityThreshold: Int
    public let confidenceThreshold: Float
    public let bosTokenId: Int?
    public let padTokenId: Int?
    public let eosTokenIds: [Int]?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case maxNewTokens = "max_new_tokens"
        case maxLength = "max_length"
        case maxDenoisingSteps = "max_denoising_steps"
        case sampler = "sampler_config"
        case minimumTemperature = "t_min"
        case maximumTemperature = "t_max"
        case stabilityThreshold = "stability_threshold"
        case confidenceThreshold = "confidence_threshold"
        case bosTokenId = "bos_token_id"
        case padTokenId = "pad_token_id"
        case eosTokenIds = "eos_token_id"
    }

    public init(
        maxNewTokens: Int = 256,
        maxLength: Int? = nil,
        maxDenoisingSteps: Int = 48,
        sampler: EntropyBoundSampler? = nil,
        minimumTemperature: Float = 0.4,
        maximumTemperature: Float = 0.8,
        stabilityThreshold: Int = 1,
        confidenceThreshold: Float = 0.005,
        bosTokenId: Int? = nil,
        padTokenId: Int? = nil,
        eosTokenIds: [Int]? = nil
    ) throws {
        guard maxNewTokens > 0 else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("max_new_tokens")
        }
        if let maxLength, maxLength <= 0 {
            throw DiffusionGemmaGenerationError.invalidConfiguration("max_length")
        }
        guard maxDenoisingSteps > 0 else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("max_denoising_steps")
        }
        guard minimumTemperature.isFinite, minimumTemperature >= 0,
            maximumTemperature.isFinite, maximumTemperature > minimumTemperature
        else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("t_min/t_max")
        }
        guard stabilityThreshold >= 0 else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("stability_threshold")
        }
        guard confidenceThreshold.isFinite, confidenceThreshold > 0 else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("confidence_threshold")
        }
        let specialIds = [bosTokenId, padTokenId].compactMap { $0 } + (eosTokenIds ?? [])
        guard specialIds.allSatisfy({ $0 >= 0 }) else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("special_token_ids")
        }
        self.maxNewTokens = maxNewTokens
        self.maxLength = maxLength
        self.maxDenoisingSteps = maxDenoisingSteps
        self.sampler = try sampler ?? EntropyBoundSampler()
        self.minimumTemperature = minimumTemperature
        self.maximumTemperature = maximumTemperature
        self.stabilityThreshold = stabilityThreshold
        self.confidenceThreshold = confidenceThreshold
        self.bosTokenId = bosTokenId
        self.padTokenId = padTokenId
        self.eosTokenIds = eosTokenIds
    }

    public init(from decoder: Decoder) throws {
        let keys = try decoder.container(keyedBy: AnyKey.self)
        let supported = Set(CodingKeys.allCases.map(\.rawValue))
        let metadata: Set<String> = ["transformers_version", "_commit_hash", "_from_model_config"]
        for key in keys.allKeys
        where !supported.contains(key.stringValue)
            && !metadata.contains(key.stringValue)
        {
            // Unsupported runtime/cache/sampling controls must not disappear
            // silently. Add support together with its execution and tests.
            throw DiffusionGemmaGenerationError.unsupportedField(key.stringValue)
        }
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let eos: [Int]?
        if try !values.contains(.eosTokenIds) || values.decodeNil(forKey: .eosTokenIds) {
            eos = nil
        } else if let single = try? values.decode(Int.self, forKey: .eosTokenIds) {
            eos = [single]
        } else {
            eos = try values.decode([Int].self, forKey: .eosTokenIds)
        }
        try self.init(
            maxNewTokens: values.decodeIfPresent(Int.self, forKey: .maxNewTokens) ?? 256,
            maxLength: values.decodeIfPresent(Int.self, forKey: .maxLength),
            maxDenoisingSteps: values.decodeIfPresent(Int.self, forKey: .maxDenoisingSteps) ?? 48,
            sampler: values.decodeIfPresent(EntropyBoundSampler.self, forKey: .sampler),
            minimumTemperature: values.decodeIfPresent(Float.self, forKey: .minimumTemperature)
                ?? 0.4,
            maximumTemperature: values.decodeIfPresent(Float.self, forKey: .maximumTemperature)
                ?? 0.8,
            stabilityThreshold: values.decodeIfPresent(Int.self, forKey: .stabilityThreshold) ?? 1,
            confidenceThreshold: values.decodeIfPresent(Float.self, forKey: .confidenceThreshold)
                ?? 0.005,
            bosTokenId: values.decodeIfPresent(Int.self, forKey: .bosTokenId),
            padTokenId: values.decodeIfPresent(Int.self, forKey: .padTokenId),
            eosTokenIds: eos)
    }

    /// `remainingStep` counts down N...1, as in the native reference. The final
    /// evaluated step uses t_min + (t_max - t_min)/N, not exactly t_min.
    public func temperature(remainingStep: Int) throws -> Float {
        guard (1 ... maxDenoisingSteps).contains(remainingStep) else {
            throw DiffusionGemmaGenerationError.invalidRemainingStep(remainingStep)
        }
        return minimumTemperature
            + (maximumTemperature - minimumTemperature)
            * (Float(remainingStep) / Float(maxDenoisingSteps))
    }

    /// Match the pinned native reference's length precedence: max_length takes
    /// effect when max_new_tokens has its default value256, including explicit256.
    public func outputTokenLimit(promptTokenCount: Int) throws -> Int {
        guard promptTokenCount >= 0 else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("prompt length")
        }
        let limit =
            (maxLength != nil && maxNewTokens == 256)
            ? maxLength! - promptTokenCount : maxNewTokens
        guard limit > 0 else {
            throw DiffusionGemmaGenerationError.invalidConfiguration("output length")
        }
        return limit
    }

    private struct AnyKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
}

public enum DiffusionGemmaGenerationError: Error, Sendable, Equatable, LocalizedError {
    case invalidConfiguration(String)
    case unsupportedSampler(String)
    case unsupportedField(String)
    case invalidRemainingStep(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let field):
            "Invalid DiffusionGemma generation value for \(field)."
        case .unsupportedSampler(let name):
            "Unsupported DiffusionGemma sampler: \(name)."
        case .unsupportedField(let field):
            "Unsupported DiffusionGemma generation field: \(field)."
        case .invalidRemainingStep(let step):
            "DiffusionGemma denoising step \(step) is outside the configured countdown."
        }
    }
}
