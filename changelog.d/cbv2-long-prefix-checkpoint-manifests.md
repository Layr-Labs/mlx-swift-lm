### Complete checkpoints of long prompts fit their manifest

- A complete checkpoint's manifest stores its prefix tokens bit-packed when
  the prefix is longer than 65,536 tokens
  (`CBv2CompleteCheckpointManifest.packedPrefixTokenThreshold`), as the native
  block layout already did. As a JSON array, about 150K–190K tokens of a
  200K-token vocabulary filled the 1 MiB manifest bound, so a provider refused
  to write the checkpoint of a longer prompt. Packed, 262,144 18-bit tokens
  take about 786 KB. Shorter prefixes keep the array, encoded exactly as
  before, and each length decodes from its one encoding only.
- Capture keeps a boundary only when its manifest, encoded as the store
  encodes it (`CBv2CompleteCheckpointManifest.storageEncoding()`), fits the
  store's new `maximumManifestBytes` (default: the 1 MiB bound). A donor keeps
  only its first, fork-target and deepest boundaries, so a deepest boundary
  the store then refused used to leave only the first: a repeated Inkling 2
  Mini prompt of about 256K tokens restored 2,048 tokens. Now the deepest
  boundary the store can write is kept.
