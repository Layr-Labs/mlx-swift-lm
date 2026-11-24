### Tool-call arguments keep exact numbers and read JSON containers under untyped parameters

- In the XML tool-call formats (Qwen 3.5 to 3.8, Nemotron 3.5), a value written
  as a number that neither an integer nor a double holds exactly, such as
  `12345678901234567890` or `0.12345678901234567891`, now keeps its digits as a
  string instead of becoming a rounded double. This applies wherever such a
  value would otherwise be read as a literal: a parameter with no declared type,
  or a union or combinator schema.
- In the same formats, under a parameter with no declared type (an empty or
  annotation-only schema), a value written as a complete JSON object or array is
  now read as that object or array instead of a string. Any other text, a value
  whose numbers would not decode exactly, a typed or combinator schema, and
  Gemma's tool calls keep the text as before.
