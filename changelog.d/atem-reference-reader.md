### Read Muse Glimmer invokes for a host that validates calls itself

- `ATEMReferenceReader` reads one Onyx tool frame's ATEM invokes as the
  reference servers return them: every complete invoke is a call, in order, a
  doubled name `X.X` is the declared tool `X`, and values are typed by the
  declared schema as `OnyxReferenceDecoder` types them.
- An undeclared name is returned as generated, with `Call.isDeclared` false,
  so the host's own validation decides whether the call is allowed. An invoke
  without a name, or one the text leaves open, is no call and is counted.
- Nothing existing changes: `ATEMToolCallParser` and the reference-parity
  scanner path read as before.
