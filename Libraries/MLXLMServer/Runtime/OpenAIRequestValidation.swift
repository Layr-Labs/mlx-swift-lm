// Copyright © 2026 Eigen Labs Inc.

import Foundation
import Hummingbird

public enum OpenAIRequestValidationError: Error, LocalizedError, Equatable, HTTPResponseError {
    case negativeOutputTokenLimit

    public var status: HTTPResponse.Status { .badRequest }
    public var errorDescription: String? { "Output token limits must not be negative." }

    public func response(from request: Request, context: some RequestContext) throws -> Response {
        var response = try context.responseEncoder.encode(
            OpenAIErrorResponse(message: localizedDescription, type: "invalid_request_error"),
            from: request, context: context)
        response.status = status
        return response
    }
}

/// Shared preparation before engine acquisition or streaming headers. Zero is
/// intentionally preserved: existing callers can request an empty completion.
enum OpenAIRequestValidation {
    static func preparedRequest(_ request: OpenAIChatCompletionRequest) throws -> OpenAIChatCompletionRequest {
        if let limit = request.maxTokens, limit < 0 {
            throw OpenAIRequestValidationError.negativeOutputTokenLimit
        }
        return try OpenAIResponseFormatSupport.preparedRequest(request)
    }
}
