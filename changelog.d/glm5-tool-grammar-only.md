### GLM5 tool calls are not refused for their schema

- `GLM5ToolCallParser` refused a well-formed frame that left out a `required`
  argument, carried an argument the schema does not declare (with
  `additionalProperties` absent or `false`), or filled a parameter without a
  usable type. A server then returned the frame as text instead of a call.
- The parser now checks the frame grammar and the declared function name only.
  Such a call is returned as generated, for the caller to validate, as vLLM and
  SGLang do. A value declared only as `string` stays raw text. A value of
  another declared type is JSON when it matches that type; any other value is
  JSON when its raw text parses (for a union with `string`, only as one of the
  other types), else that raw text.
- Frames that break the grammar (open at the end, misplaced or nested tags, a
  repeated argument, an undeclared function name) still fail, and one of them
  still refuses the whole batch at EOS.
