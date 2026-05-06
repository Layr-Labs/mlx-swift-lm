# MLXSpeculative

Speculative-decoding drafters and round-loops for `mlx-swift-lm`.

v1 ships one drafter family: Google's Gemma 4 Multi-Token Prediction
(MTP) "assistant" drafters, published at
`mlx-community/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant-bf16`.

## What is MTP

MTP is a tightly-coupled speculative-decoding scheme:

- The drafter is a small model (4 kv-shared decoder layers) that borrows
  the target's input embeddings and last-layer hidden state. It has no
  KV cache of its own; every drafter layer reads K/V directly from the
  target's last non-shared full- and sliding-attention layers.
- On each round, the drafter generates `blockSize - 1` candidate tokens
  autoregressively from `(bonusToken, lastHidden)`, holding RoPE
  positions constant at the bonus token's absolute position.
- The target verifies `[bonus, c_0, ..., c_{K-1}]` in a single forward
  pass and emits its greedy prediction at each position. The drafter's
  candidates are accepted up to the first mismatch; the target's
  prediction at the mismatch position becomes the next bonus.
- At greedy (`temperature=0`) this produces output that is
  token-identical to running the target alone, with speedup proportional
  to the drafter's accept rate.

## Public API

### Single-request generation (B=1)

```swift
import MLXSpeculative

let drafter = try Gemma4AssistantDraftModel.load(
    from: assistantModelDirectory)
let stream = try generateGemma4MTP(
    input: lmInput,
    parameters: .init(maxTokens: 512),
    target: targetModelContext,     // ModelContext from MLXLLM
    drafter: drafter,
    blockSize: 4
)
for try await generation in stream {
    switch generation {
    case .chunk(let text): print(text, terminator: "")
    case .info(let info): print("\n[tokens: \(info.tokensPerSecond) tok/s]")
    }
}
```

### Batched (B>1) generation

Continuous-batching MTP round loop, one `BatchedGeneration` per
generated step with per-row slot state:

```swift
let stream = try runGemma4MTPRoundsBatched(
    target: targetModel,
    drafter: drafter,
    targetCache: prefilledBatchCache,   // BatchKVCache / BatchRotatingKVCache
    firstBonus: bonusPerRow,            // [Int], length B
    firstHidden: lastHiddenBxLxH,
    firstSharedKV: capturedSharedKV,
    maxTokens: 256,
    blockSize: 4,
    eosTokenIds: [1, 106]
)
for await step in stream {
    for slot in step.slots where slot.token != nil {
        handleToken(row: slot.row, token: slot.token!)
    }
}
```

### Benchmark primitives

Two measurement helpers for wall-clock comparisons. Callers supply a
loaded target + bound drafter; the helpers return timings + accept
histogram. No model loading or CLI in the library.

```swift
let baseline = measureBaselineThroughput(
    target: target, promptTokens: prompt, maxTokens: 256)
let mtp = try measureMTPThroughput(
    target: target, drafter: drafter,
    promptTokens: prompt, maxTokens: 256, blockSize: 4)
print("MTP speedup: \(mtp.tokensPerSecond / baseline.tokensPerSecond)x")
print("Accept histogram: \(mtp.acceptLengths ?? [])")
```

## Measured throughput (M3, 36 GB)

Single-batch greedy, 3 chat-templated prompts, 64 max_tokens, 16-token
warmup. Block size swept per model; best configuration shown.

| Model           | Best K | Baseline tok/s | MTP tok/s | Speedup |
|-----------------|--------|----------------|-----------|---------|
| E2B-bf16        | 5      | 21.4           | 24.1      | 1.13×   |
| E4B-bf16        | 5      | 11.5           | 12.5      | 1.08×   |
| 26B-A4B-4bit    | 3      | 28.7           | 35.9      | 1.25×   |

Per-round accept rates improve with target size, as expected — the
drafter's predictions align more closely with a larger target's greedy
output:

| Model           | Accept/K at K=3 | Accept/K at K=4 |
|-----------------|-----------------|-----------------|
| E2B-bf16        | 0.64/2 (32%)    | 0.71/3 (24%)    |
| E4B-bf16        | 0.53/2 (27%)    | 0.61/3 (20%)    |
| 26B-A4B-4bit    | 1.25/2 (62%)    | 1.42/3 (47%)    |

Reproduce with the `realModelThroughputBenchmark` test — requires
`MTP_BENCH_DATA_DIR` and `MTP_BENCH_PROMPTS` env vars pointing to
local model directories and a pre-tokenized prompt fixture JSON.

## Correctness

At `temperature=0`, MTP output is bitwise equal to target-only greedy
output, with two caveats driven by bf16 numerics rather than the
algorithm:

1. **Near-uniform logit tails.** When the target starts repeating a
   single token (post-EOS padding, or degenerate prompts at small
   random-weight shapes), its top-1 and top-2 logits can be within bf16
   precision. In that regime the multi-position verify forward's argmax
   occasionally flips vs a serial single-position forward at the same
   position — both continuations are mathematically valid greedy
   choices, but not bitwise identical. The Python mlx-vlm reference
   exhibits the same behavior; see
   `docs/superpowers/notes/2026-05-06-python-mtp-greedy-divergence.md`.

2. **Batched attention.** Small-target batched (B>1) runs can diverge
   from B=1 baselines on the same prompt for the same reason — bf16
   scaled-dot-product attention's argmax is not bitwise stable across
   batch shapes at near-uniform logits.

Parity is enforced as a hard gate internally (Swift MTP tokens == Swift
baseline tokens) across four test suites covering E2B / E4B / MoE /
batched shapes. 143/143 tests pass deterministically.

## Architecture notes

- **Shared-KV capture** (`Gemma4TextModel.forwardForMTP`): the target
  forward exposes a snapshot of the last non-shared full- and
  sliding-attention K/V tensors via `Gemma4SharedKVCapture`. The drafter
  consumes them directly; it never allocates KV of its own.
- **Pre-norm hidden**: the drafter's `pre_projection` is trained against
  the LAST decoder-layer output BEFORE `model.norm`. `forwardForMTP`
  returns this pre-norm hidden alongside post-norm logits so the
  drafter gets what it was trained against.
- **Per-row cache rewind**: B>1 rewind uses
  `BatchKVCache.zeroTailPerRow(keepLengths:)` plus
  `Gemma4SharedKV.zeroTailPerRow(from:keepLengths:)` so rows that
  accepted different numbers of drafter tokens all get their own
  correct cache state.
- **MaskedEmbedder** (centroid-routed sparse LM head): used by E2B / E4B
  drafters (`use_ordered_embeddings=true`). Scores 2048 token clusters,
  materialises the top-K clusters' tokens (default 32 of 2048) and
  scatters logits back into the full vocab with sentinel values
  elsewhere.

## Model / drafter pairing

| Target | Drafter |
|---|---|
| `mlx-community/gemma-4-E2B-it-{bf16,4bit}` | `mlx-community/gemma-4-E2B-it-assistant-bf16` |
| `mlx-community/gemma-4-E4B-it-{bf16,4bit}` | `mlx-community/gemma-4-E4B-it-assistant-bf16` |
| `mlx-community/gemma-4-26B-A4B-it-{bf16,4bit}` | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` |
| `mlx-community/gemma-4-31B-it-{bf16,4bit}` | `mlx-community/gemma-4-31B-it-assistant-bf16` |

Drafter attention head dims (`head_dim`, `global_head_dim`) must match
the target for `scaled_dot_product_attention` to accept shared K/V —
real drafters are published this way; the pairing checks in
`Gemma4AssistantDraftModel.bind(target:)` enforce it at runtime.

## Known limitations

- Greedy (`temperature=0`) only. Stochastic sampling is not implemented.
- Text-only. Multimodal prefill (images / audio) runs through the target
  unchanged; MTP kicks in on the text-decode tail.
- Drafter weights must match the target's input embedding + attention
  head shapes. Cross-family pairings (e.g. E2B drafter with 26B-A4B
  target) are rejected by `bind(target:)`.
