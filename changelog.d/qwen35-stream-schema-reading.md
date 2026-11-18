### Qwen 3.5 XML calls: streamed parameters agree with the complete parse; local `$ref`s are read

- A streamed Qwen 3.5 XML call exposed a parameter as a JSON string unless its
  schema's `type` was a plain number, Boolean, object or array type. A
  parameter typed only through `anyOf`/`oneOf`/`allOf`, a type list,
  `nullable` or a `$ref` (how Pydantic, FastMCP and zod write optional and
  nested arguments) then parsed as another JSON type when its frame closed,
  and the call failed as `diverged`.
- `Qwen35ArgumentStreamer` now streams a parameter as text only when the
  complete parse keeps every text of it as written
  (`ToolArgumentNormalization.keepsText`, the parse's own reading); any other
  parameter waits for its closing tag and leaves converted.
- Qwen XML values (`QwenXMLPayloadScanner`, streamed and complete) read a local
  `$ref` (`#/$defs/…`, `#/definitions/…`, chained up to 8 times) as the schema
  it names, or by the `type` beside an unresolved reference, so an object,
  array or number parameter declared by reference is no longer returned as a
  string. A reference that names nothing and has no `type` keeps its text, as
  before. Other dialects read references as before.
- A reading follows at most 64 `$ref` hops, so union branches that name the same
  definitions again (a cycle the coordinator keeps, or a diamond) cost bounded
  time; past the bound the value keeps its text. Pointers are read as the
  coordinator's resolver reads them: each token percent-decoded (a malformed
  escape names nothing), then `~1`/`~0` unescaped, and arrays walked by index.
