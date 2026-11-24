### Inkling 2 Mini rows that don't draft save prefix checkpoints

- An Inkling 2 Mini checkpoint carries the DSpark drafter's prompt history,
  and only rows that drafted recorded it. A cache-enabled text request that
  cannot speculate (logprobs, a logit bias, a repetition penalty, a stop
  string or penalty without its opt-in, a token constraint, a reasoning
  budget) lost every checkpoint (`assistantCaptureRefused`), so a repeated
  prompt with those settings always prefilled cold.
- The drafter now opts in to `checkpointsNonSpeculatingRows`, as the Qwen 3.5,
  Qwen4, Nemotron Super and MiMo assistants do. Such a row runs its prompt
  through the hidden-returning prefill, records the drafter's history and
  saves the same checkpoints, byte for byte, that an unconstrained row saves.
  It still never drafts. Tokens and logprobs are unchanged: a restored
  request, drafting or not, emits the cold run's tokens and logprob bits,
  which equal those of an engine without the drafter. The store's manifest
  bound applies as before. Media rows are unchanged.
