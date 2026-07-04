// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

public struct MLXServerModel: Codable, Sendable, Equatable {
    public var id: String
    public var object: String
    public var created: Int?
    public var ownedBy: String

    private enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
    }

    public init(
        id: String,
        object: String = "model",
        created: Int? = nil,
        ownedBy: String = "local"
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.ownedBy = ownedBy
    }
}

public struct ServerGenerationInfo: Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var promptTime: TimeInterval
    public var generationTime: TimeInterval
    public var stopReason: String

    public init(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval,
        stopReason: String
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.promptTime = promptTime
        self.generationTime = generationTime
        self.stopReason = stopReason
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

    /// Model-type-derived default reasoning-parser format for `modelId`,
    /// consulted by `MLXOpenAIService` only when a request doesn't specify
    /// `reasoning_parser` explicitly (an explicit request value always
    /// wins; this is a fallback, one step above the service-wide
    /// `defaultReasoningParser` and one step below it in priority... see
    /// `MLXOpenAIService`'s resolution order: request > this > service
    /// default > `.none`).
    ///
    /// `nil` means "this engine has no per-model opinion" — the safe
    /// default for any engine that doesn't track model types (embeddings,
    /// single-model servers that already pass a fixed
    /// `defaultReasoningParser` to `MLXOpenAIService`, test doubles).
    /// Engines that DO track a `modelType` per loaded model (e.g. a
    /// multi-model registry that also resolves `ToolCallFormat` per
    /// request via `ServerToolParser.resolve(requested:modelType:)`) can
    /// mirror that same model-type inference for reasoning parsers.
    ///
    /// Given a default (nil-returning) implementation below, this is
    /// purely additive — existing conformers need no changes.
    func defaultReasoningParser(for modelId: String) async -> ReasoningParserFormat?
}

extension MLXServerEngine {
    public func defaultReasoningParser(for modelId: String) async -> ReasoningParserFormat? {
        nil
    }
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
