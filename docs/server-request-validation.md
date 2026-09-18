# Server request validation

`MLXOpenAIService` validates output-token limits before invoking its engine or
returning streaming headers. This applies to Chat Completions, Completions and
Responses after their shared chat-request lowering, including chat batch items.

Negative limits return HTTP 400 with an OpenAI-shaped `invalid_request_error`.
The fixed error message does not echo request content. Explicit zero, positive
and omitted limits retain their existing semantics and are forwarded unchanged;
model context, capacity and tool-budget checks still apply independently.

Implementation: `OpenAIRequestValidation.preparedRequest` and
`OpenAIRequestValidationError` in
`Libraries/MLXLMServer/Runtime/OpenAIRequestValidation.swift`.
`OutputTokenLimitHTTPTests` covers negative values (including `Int.min`), both
streaming modes, batch items and unchanged zero/positive/omitted limits.
This is input validation, not a numerical or model-weight change. A coordinator
or another engine entry point retains its own validation contract.

## Failures after HTTP streaming begins

Chat/Completions HTTP handlers request `frameGenerationErrors: true`. A late
generation failure produces one sanitized SSE error chunk with the existing
generation identity and `finish_reason: "error"`, then closes the body normally.
It does not emit a success finish reason or `[DONE]`, fabricate a tool call,
change argument bytes, or expose the underlying error description. Usage is
included only when requested and actually observed. This follows the
[OpenRouter mid-stream error contract](https://openrouter.ai/docs/api_reference/streaming#errors-after-the-response-is-committed-mid-stream).

Direct `streamChatCompletionFrames` callers retain throwing behavior by default.
Errors before engine acquisition/headers still use the existing HTTP mapper;
cancellation still propagates and cancels the producer. Responses keeps its
existing `response.failed` framing. The provider's local upload interceptor must
explicitly select the same HTTP policy as the SDK routes.

`ChatStreamingFailureHTTPTests` covers both HTTP routes and failures before any
generated text, after reasoning, after content and after a tool/usage event.
These transport checks are separate from model success/argument-fidelity gates;
a well-framed error is not a successful tool call.
