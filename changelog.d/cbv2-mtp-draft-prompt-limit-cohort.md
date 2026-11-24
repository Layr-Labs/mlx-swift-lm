### MTP: the prompt-length limit applies per plan

- `CBv2MTPConfig.maxDraftPromptTokens` is now a plan-time gate instead of a
  whole-request exclusion. A plan whose eligible decode rows are all over the
  limit runs at depth zero (controller reason `prompt_limit`) and stays on the
  chained decode path, so a long-prompt row decoding alone still never drafts
  and keeps the MTP-off engine's tokens. A plan with at least one row at or
  under the limit keeps its usual depth for every row: a short row beside a
  long one is no longer stopped from drafting. On a depth table that never
  drafts at two or more rows nothing changes. Still nil (off) by default.
