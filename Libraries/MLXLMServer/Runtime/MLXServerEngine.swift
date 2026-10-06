// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

public struct MLXServerModel: Codable, Sendable, Equatable {
    public var id: String
    public var object: String
    public var created: Int?
    public var ownedBy: String
    /// Advertised listing context. Omitted when unknown so older clients
    /// keep the historical `{id,object,owned_by}` shape. Encoded as both
    /// `context_length` and `max_model_len` (vLLM-style) when set.
    public var contextLength: Int?

    private enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
        case contextLength = "context_length"
        case maxModelLen = "max_model_len"
    }

    public init(
        id: String,
        object: String = "model",
        created: Int? = nil,
        ownedBy: String = "local",
        contextLength: Int? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.ownedBy = ownedBy
        self.contextLength = contextLength
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(object, forKey: .object)
        try container.encodeIfPresent(created, forKey: .created)
        try container.encode(ownedBy, forKey: .ownedBy)
        try container.encodeIfPresent(contextLength, forKey: .contextLength)
        try container.encodeIfPresent(contextLength, forKey: .maxModelLen)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        object = try container.decodeIfPresent(String.self, forKey: .object) ?? "model"
        created = try container.decodeIfPresent(Int.self, forKey: .created)
        ownedBy = try container.decodeIfPresent(String.self, forKey: .ownedBy) ?? "local"
        contextLength =
            try container.decodeIfPresent(Int.self, forKey: .contextLength)
            ?? container.decodeIfPresent(Int.self, forKey: .maxModelLen)
    }
}

public struct ServerGenerationInfo: Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var promptTime: TimeInterval
    public var generationTime: TimeInterval
    public var stopReason: String
    /// Request-owned engine accounting; nil means the engine did not report it.
    public var cachedPromptTokens: Int?

    public init(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval,
        stopReason: String,
        cachedPromptTokens: Int? = nil
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.promptTime = promptTime
        self.generationTime = generationTime
        self.stopReason = stopReason
        self.cachedPromptTokens = cachedPromptTokens.map { min(max(0, $0), max(0, promptTokens)) }
    }

    public init(_ info: GenerateCompletionInfo) {
        self.init(
            promptTokens: info.promptTokenCount,
            completionTokens: info.generationTokenCount,
            promptTime: info.promptTime,
            generationTime: info.generateTime,
            stopReason: info.stopReason.openAIFinishReason
        )
    }
}

public enum MLXServerGenerationEvent: Sendable, Equatable {
    case content(String)
    /// Authoritative native channel separation. Do not parse its content again
    /// or reinterpret reasoning as ordinary text, including at end-of-stream.
    case parsed(ParsedReasoning)
    case toolCall(ToolCall)
    case info(ServerGenerationInfo)
}

public protocol MLXServerEngine: Sendable {
    func availableModels() async throws -> [MLXServerModel]
    func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse
}

extension GenerateStopReason {
    var openAIFinishReason: String {
        switch self {
        case .stop:
            return "stop"
        case .length:
            return "length"
        case .cancelled:
            return "stop"
        }
    }
}
