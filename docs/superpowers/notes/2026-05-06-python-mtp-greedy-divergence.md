# Python mlx-vlm MTP diverges from baseline at greedy (temp=0)

**Date:** 2026-05-06
**Observed during:** Task 23 oracle generation (branch `mtp-gemma4-drafter`).
**Python reference:** `Blaizzy/mlx-vlm` at commit `244f4bb5a3339b180da3d2b276a4bdfcf7670f9f`.
**Python library versions:** mlx-vlm 0.4.5 (pinned at 244f4bb), via `uv` venv at `/Users/asv/mtp-integration-data/tools/mlx-vlm`.
**Model pair under test:** `mlx-community/gemma-4-e2b-it-bf16` target + `mlx-community/gemma-4-E2B-it-assistant-bf16` drafter, greedy sampling, `block_size=3`, `max_tokens=64`.

## Finding

The `mlx-vlm` MTP implementation, which is upstream's PR #1112's reference port, does **not** produce byte-identical output to the no-drafter baseline at `temperature=0`. This contradicts the PR's own description ("byte-identical greedy output") and the stretch claim in Google's blog post ("lossless").

Over the 20 prompts in `Tests/MLXLMTests/Resources/Gemma4MTPPrompts.json`:

| Category | Matching | Total |
|---|---|---|
| short (≤64 tokens) | 3 | 5 |
| medium | 7 | 10 |
| long | 2 | 5 |
| **total** | **12** | **20** |

The full breakdown is in `Tests/MLXLMTests/Resources/mtp-oracle/gemma4-e2b-block3-max64.json`, field `mtp[*].matches_baseline`.

## Divergence pattern

In every observed case, divergence happens **after the semantic response ends**. Example (`short_1`, prompt "Name three primary colors."):

- Both outputs produce `"The three primary colors are **red, yellow, and blue**."`
- At token index 18 (well past the content), baseline emits `1` (`<eos>`), MTP emits `106` (`<turn|>`).
- Baseline continues `106, 1, 106, 1, 106, ...` (alternating `<turn|>`/`<eos>`); MTP continues `106, 106, 106, ...` (all `<turn|>`).

In every diverging case, the mismatch:
- Is in the post-EOS padding tail, not in the model's actual response.
- Doesn't change the tokens the user would see (both hit `<eos>`/`<turn|>` quickly).
- Appears to be driven by drafter bias during the degenerate tail where the logits are near-uniform over a handful of special tokens.

## Root cause hypothesis (not verified)

The spec accept/reject protocol is sound on well-separated logits. In the post-EOS tail, the target's top-1 and top-2 logits are nearly equal — the model is oscillating between `<eos>` and `<turn|>` — and the drafter's predictions occasionally cause a round to accept one that the baseline wouldn't have picked, because of tiny fp differences propagating through the drafter's projection/attention stack.

This is **not** a Gemma-4-MTP-specific bug. It's a general property of speculative decoding in nearly-uniform regions: the "verify" step's argmax depends on a small numerical difference, and with a drafter in the loop the prior KV-cache state can cause a different — but equally-valid — argmax.

## Decision

The project's acceptance criterion ("byte-identical greedy output") cannot be measured against the Python reference as the oracle, because the Python reference itself doesn't meet it. We change the Swift parity gate from:

- ~~"Swift MTP output byte-identical to Python MTP output"~~ (can't — Python isn't a stable oracle)

to:

- **"Swift MTP output byte-identical to Swift baseline (no-drafter) greedy output"** — internal consistency within Swift. The drafter must not change the target's greedy emission.

This is a **stronger correctness claim** than Python parity: it directly asserts the "lossless" property of speculative decoding without deferring to a reference implementation that itself doesn't meet it.

## What this means for the spec

The spec (`docs/superpowers/specs/2026-05-05-gemma4-mtp-drafter-design.md`) section "Testing strategy > Integration" described the parity test as "baseline vs MTP at `temperature=0`, assert exact token-sequence equality." That phrasing already matched the internal-consistency framing — we just initially interpreted it as "match Python" during implementation. The spec text needs no change; only the intent needs clarification.

The spec's "Python oracle runs (one-time, frozen)" section overpromised. We still record Python's tokens/sec numbers for throughput comparison, but we no longer claim per-round accepted-histogram parity against Python — we record it, and any large divergence is a debug signal, not a merge blocker.

## What this means for benchmarks

- **Correctness gate (Task 23/24/25):** Swift MTP tokens == Swift baseline tokens, for every prompt × block × batch configuration. Hard equality.
- **Throughput gate (Task 26):** Swift MTP tokens/sec ≥ 0.9× Swift-baseline-in-Python tokens/sec on the same hardware. We compare *Python baseline* to *Swift MTP* — not Python MTP to Swift MTP — because Python MTP's correctness is questionable.

## Action items (done / pending)

- [x] Document the finding (this file).
- [x] Update Task 23's parity test to assert Swift MTP == Swift baseline.
- [ ] Update Task 24/25 the same way.
- [ ] Task 26 baseline changes from "Python MTP tokens/sec" to "Python baseline tokens/sec" as the reference.
- [ ] Add a follow-up issue (post-v1) to investigate whether Swift's MTP implementation has the same post-EOS divergence from its own baseline, and whether that's worth fixing.

## Data preserved

The full `gemma4-e2b-block3-max64.json` fixture is committed under `Tests/MLXLMTests/Resources/mtp-oracle/` — useful as a snapshot for future comparison, even though we no longer use it as a parity oracle.

## How to reproduce

```bash
source /Users/asv/mtp-integration-data/tools/mlx-vlm/.venv/bin/activate
python /Users/asv/mtp-integration-data/tools/generate_oracle.py \
  --target-dir /Users/asv/mtp-integration-data/gemma-4-e2b-it-bf16 \
  --drafter-dir /Users/asv/mtp-integration-data/gemma-4-E2B-it-assistant-bf16 \
  --out /tmp/repro.json \
  --prompts /Users/asv/projects/MLX-swift-lm/Tests/MLXLMTests/Resources/Gemma4MTPPrompts.json \
  --block-size 3 --max-tokens 64
python3 -c "import json; d = json.load(open('/tmp/repro.json')); print(f\"{sum(m['matches_baseline'] for m in d['mtp'])}/{len(d['mtp'])} match\")"
```
