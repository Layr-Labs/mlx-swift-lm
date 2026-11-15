### Tool parsers no longer refuse a well-formed call for its schema

- Three parsers refused a call that kept their frame grammar but broke the
  caller's schema, so a server returned the frame as text or failed the
  response: `GLM5ToolCallParser` a call that left out a `required` argument,
  carried an argument the schema does not declare (with `additionalProperties`
  absent or `false`), or filled a parameter without a usable type, a union or a
  value of another type; `GemmaFunctionParser` an undeclared argument under
  `additionalProperties: false`; `MiMoV2ToolCallParser` a union or a value of
  another type.
- They now check their grammar and the declared function name only and return
  such a call as generated, for the caller to validate, as vLLM and SGLang do.
  GLM5 (types from `type`, `nullable` and `anyOf`/`oneOf` branches,
  `GLM5ToolCallParser.argumentTypes`): a value declared only as `string` stays
  raw text; a value that opens a JSON value of a declared non-string type, or
  any JSON value but a string where no `string` is declared, keeps that JSON
  reading (quoted delimiters inside are data); any other value is JSON when its
  raw text parses (for a union with `string`, only as one of the other types),
  else that raw text. MiMo V2: values it parsed before read the same; a union
  or a value of another type it refused is JSON when it parses as a declared
  non-string type (any JSON when no `string` is declared), else its raw text.
- Frames that break the grammar (open at the end, misplaced or nested tags, a
  repeated argument, an undeclared function name) still fail, and GLM5 still
  refuses the whole batch at EOS for one of them.
- `DARKBLOOM_GRAMMAR_ONLY_TOOL_PARSING=0` (or `off`, `false`, `no`) restores
  the three parsers' schema refusals (`GrammarOnlyToolParsing`).
