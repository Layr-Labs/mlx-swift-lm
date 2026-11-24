### Prefix checkpoint row decode: asymmetric row targets tested, task-local test decoder

- `CBv2PagedCheckpointStorage.rowTargets` is now tested against `append` for
  layers whose value width differs from the key width (128/64 and 64/128,
  BF16 and FP32), which pins the value role's segment offset.
- The test-only import decoder override
  (`CBv2NativeCheckpointRowDecoder.importDecoderForTesting`) is task-local
  instead of a shared static, so a test's decoder can't reach an import that
  another suite runs at the same time. Production imports are unchanged: they
  read no override and use the default decoder.
