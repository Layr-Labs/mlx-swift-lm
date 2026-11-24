### MTP: a prompt-length limit for drafting

- `CBv2MTPConfig.maxDraftPromptTokens` (nil by default): a request whose prompt
  holds more tokens than the limit never drafts and keeps no assistant history,
  as a reasoning budget does. It decodes as ordinary target-only rows, so its
  tokens are the MTP-off engine's. Rows at or under the limit are unchanged. A
  value below one means no limit. An integrator sets it where a chip's verify
  rounds were measured to cost more than they gain on long prompts.
