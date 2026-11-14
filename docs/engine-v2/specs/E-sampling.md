# WS-E: Vectorized sampling, logprobs, incremental detok + stop holdback

## Deliverables (`Libraries/MLXLMCommon/ContinuousBatchingV2/`)

1. `LogitsPipelineV2.swift` — ordered transform pipeline over `[B, vocab]`
   logits (vLLM order, report 08 §6):
   (0) capture RAW logprobs first when any row requests them (log_softmax
   on the raw row; gather top-k lazily); (1) logit_bias scatter-add;
   (2) penalties — repetition (multiply/divide on presence over prompt ∪
   output window), frequency (α·count), presence (β·mask), computed from
   per-row token-count tensors maintained incrementally (scatter-add per
   step, NEVER re-binned from scratch); (3) temperature divide; (4) top-k /
   top-p / min-p via vocab sort + cumsum threshold. ALL per-row parameters
   are tensors `[B]` or `[B,1]` built once at batch-membership change —
   zero per-row Swift loops in the step path.
2. `SamplerV2.swift` —
   - All-greedy fast path: single `argMax` when every row is greedy
     (temperature < 1e-5) — matches legacy vectorized greedy.
   - Mixed batch: Gumbel-max via exponential noise
     (`argmax(probs / -log(u))` pattern) so there is no multinomial sync;
     per-row keyed RNG: `MLXRandom.key(seed ⊕ requestID ⊕ step)` per row,
     stacked — a row's stream depends only on (seed, requestID, step),
     NEVER on batchmates (batch-invariance requirement).
   - `where(temp < eps, greedy, sampled)` merge for mixed batches.
3. `DetokenizerV2.swift` — per-request incremental detokenization: hold
   back bytes until valid UTF-8 boundary (multi-byte chars split across
   tokens must never emit replacement chars mid-stream).
4. `StopHoldback.swift` — stop-string streaming matcher: emit only text
   that provably cannot be a prefix of any stop string (longest-suffix
   holdback); on match, truncate at match start and signal stop. Handles
   stop strings spanning token boundaries and overlapping candidates.
   This is REQUIRED for B's one-step-late stop detection to be invisible.

## Tests (`Tests/MLXLMTests/CBv2SamplingTests.swift`)
- Pipeline order: logprobs reflect raw logits even with penalties/bias on.
- Penalties parity vs a scalar reference implementation on random histories.
- Determinism: same (seed, requestID) → same tokens at B=1 and inside a
  B=4 mixed batch (greedy neighbors), across two runs.
- Greedy fast path bit-identical to argMax.
- Detok: multi-byte UTF-8 (emoji, CJK) split across tokens; no replacement
  chars; byte-exact final text.
- StopHoldback: stop string split across 3 tokens; overlapping stops
  ("</s>", "</section>"); prefix-ambiguity holdback released when
  disambiguated; nothing after stop ever emitted.
- Use a real tokenizer fixture if a tiny one is available in the repo's
  test resources; otherwise a hand-rolled byte-pair stub is fine.

## References
Report 08 §6 (vLLM sampler order, Gumbel trick, BatchUpdate churn safety),
critique items on seed/batch-invariance and detok (report 12 items 5, 6).
Corpus: `~/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.

## Opt-in constrained spans in MTP

`CBv2MTPConfig.freeSpanSpeculation` defaults off. Set
`DARKBLOOM_CONSTRAINT_SPANS=1` before process startup or pass the configuration
value explicitly. The environment also accepts `true`, `yes`, and `on`, ignoring
case and surrounding whitespace. Unset or any other value keeps the existing
whole-request constrained fallback. Restart without the flag to roll back.

A `CBv2FreeSpanTokenConstraint` declares committed states whose vocabulary mask
is independent of a positive remaining-token budget. Its pure continuation check
may return true only for a valid token that stays in exactly that state and
touches no protected boundary, including intermediate bytes within a token. The
check must not intern states or change caches, exhaustion, or subsequent results.

Only built-in samplers attest span verification. Custom samplers and constraints
that do not opt in stay target-only. Existing sampling, media, logprob, penalty,
stop-string, storage and native-lifecycle exclusions still apply. The SDK does
not interpret tool or JSON markers; the consumer's constraint owns that policy.

Every target verification position uses the current grammar mask, including
correction and bonus positions. The engine accepts only the exact verified
draft prefix whose tokens pass the pure continuation check. The first boundary
token commits as a target correction; the speculative suffix rolls back.
Grammar confirmation runs once per emitted token, before publication. Protected
spans use ordinary masked decoding, and drafting can resume after the request
returns to a free state. Raw nonfinite logits retain their existing failure path.

Persistent assistants retain trusted target history through protected spans but
never draft there. This covers requests that start protected as well as those
that enter a protected span later. Both ordinary hidden-returning decode and the
existing seed path feed the intervening transitions. A constraint that remains
protected for its entire request may retain history while the flag is enabled.

`CBv2FreeSpanTests` checks literal outputs, masked greedy verification in all four
modes, boundary rollback and explicit-off behavior. `QwenFreeSpanHistoryTests`
uses a tiny real hybrid target and inline assistant through public request/event
and assistant interfaces to check contiguous history across a protected span.
These fixtures do not qualify production artifacts or establish throughput gains.
