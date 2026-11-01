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


## Reasoning and response object compatibility

For Qwen/DeepSeek think-tag parsers, an unclosed `<think>` block remains
reasoning when generation ends, including a token-limit stop. Any answer text
before the opening marker remains answer content. Explicit parser `none` keeps
literal output unchanged, and native `.parsed` channels remain authoritative.
Engines whose rendered prompt already opens a think block must carry that state
into both collected and streaming output; the SDK cannot infer an unseen prompt
from generated text alone. The provider prompt probe supplies this state.
`CollectedReasoningTests` covers channel separation, length stops, usage,
collection, retrieval and streamed snapshots without model execution.

Model listings retain a supplied `created` timestamp. When a local catalog entry
has none, the service uses one process-scoped fallback across service facades.
This describes the local catalog representation, not model training or artifact
publication time.

Responses objects include flat function descriptors in `tools`, `tool_choice`
and `parallel_tool_calls`, including collected/retrieved objects and created,
in-progress and completed/incomplete/failed streaming snapshots. With no tools,
the defaults are `[]`, `none` and `true`; with tools and no explicit choice,
`tool_choice` defaults to `auto`. Named choices retain the function name; schemas,
descriptions and strictness remain intact. Chat tool serialization is unchanged.
`OpenAIObjectCompatibilityTests` covers these object contracts with scripted
engines; actual HTTP and model qualification remain separate checks.
