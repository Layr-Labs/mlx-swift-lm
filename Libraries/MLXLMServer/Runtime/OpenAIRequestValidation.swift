// Copyright © 2026 Eigen Labs Inc.

import Foundation
import Hummingbird

public enum OpenAIRequestValidationError: Error, LocalizedError, Equatable, HTTPResponseError {
    case negativeOutputTokenLimit
    /// A message carries a content part whose `type` this server does not
    /// serve. Refused before the engine sees the request, so the part is never
    /// dropped from the prompt in silence. `type` is already bounded for
    /// display (`shownPartType`).
    case unsupportedContentPart(type: String)

    public var status: HTTPResponse.Status { .badRequest }
    public var errorDescription: String? {
        switch self {
        case .negativeOutputTokenLimit:
            return "Output token limits must not be negative."
        case .unsupportedContentPart(let type):
            return "Content part type \"\(type)\" is not supported. "
                + "Send text, image_url, video_url or input_audio parts."
        }
    }

    public func response(from request: Request, context: some RequestContext) throws -> Response {
        var response = try context.responseEncoder.encode(
            OpenAIErrorResponse(message: localizedDescription, type: "invalid_request_error"),
            from: request, context: context)
        response.status = status
        return response
    }

    /// A part type as the error names it: at most 64 of its ASCII letters,
    /// digits and `_.-:/`, so a request cannot place arbitrary text in the
    /// error it gets back.
    static func shownPartType(_ type: String) -> String {
        let allowed = Set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-:/".unicodeScalars)
        let kept = String(
            String.UnicodeScalarView(type.unicodeScalars.filter(allowed.contains).prefix(64)))
        return kept.isEmpty ? "(unnamed)" : kept
    }
}

/// Shared preparation before engine acquisition or streaming headers. Zero is
/// intentionally preserved: existing callers can request an empty completion.
enum OpenAIRequestValidation {
    static func preparedRequest(_ request: OpenAIChatCompletionRequest) throws
        -> OpenAIChatCompletionRequest
    {
        if let limit = request.maxTokens, limit < 0 {
            throw OpenAIRequestValidationError.negativeOutputTokenLimit
        }
        if let type = request.unsupportedContentPartType {
            throw OpenAIRequestValidationError.unsupportedContentPart(
                type: OpenAIRequestValidationError.shownPartType(type))
        }
        return try OpenAIResponseFormatSupport.preparedRequest(request)
    }
}
