# Laguna XS 2.1 DFlash → MLX Conversion & Integration Design

**Date:** 2026-07-25
**Status:** Approved (design review complete)
**Host repo:** `Layr-Labs/MLX-swift-lm`, branch `dflash-mlx-swift`

## Goal

Convert `poolside/Laguna-XS-2.1-DFlash-NVFP4` (the DFlash block-draft
speculator for the Laguna XS 2.1 NVFP4 target) into an MLX-format draft
checkpoint, and extend this fork so the pair runs end-to-end speculative
generation on Apple Silicon. Competition-track integration in
`mlxfast-challenge` is explicitly deferred; nothing in this design touches
that repo.

## Source checkpoint characterization

`poolside/Laguna-XS-2.1-DFlash-NVFP4`: single `model.safetensors`
(924 MB, all BF16, `format: pt`), 58 tensors, ~462 M params. Architecture
(from `config.json` + safetensors header + the vLLM reference
implementation):

- 5 Laguna-style decoder layers, all `sliding_attention` (window 512).
- hidden 2048, 64 attention heads / 8 KV heads, head_dim 128,
  intermediate 8192 (silu), rms_norm_eps 1e-6, rope_theta 500000 (plain
  full-dim RoPE; no rope_scaling), max_position_embeddings 262144,
  vocab 100352 (= target vocab; `draft_vocab_size` equal, no id mapping).
- Per-layer tensors: fused `self_attn.qkv_proj.weight [10240, 2048]`
  (row layout `[q(8192); k(1024); v(1024)]`, vLLM QKVParallelLinear order),
  `self_attn.g_proj.weight [64, 2048]` (per-head gating),
  `self_attn.{q,k}_norm.weight [128]`, `self_attn.o_proj.weight
  [2048, 8192]`, `mlp.{gate,up}_proj.weight [8192, 2048]`,
  `mlp.down_proj.weight [2048, 8192]`,
  `{input,post_attention}_layernorm.weight [2048]`.
- Root tensors: `fc.weight [2048, 10240]` (context projection from
  5×2048 concat), `aux_hidden_norms.{0..4}.weight [2048]`,
  `hidden_norm.weight [2048]`, `norm.weight [2048]`.
- **No `embed_tokens`, no `lm_head`** — shared with the target
  (`bind`-style, which this fork's `DFlashDraftModel` already implements).
- DFlash metadata: `block_size 16`, `mask_token_id 12` (reuses a vocab
  token; no separate mask embedding), `num_target_layers 40`,
  `target_layer_ids [1, 13, 25, 33, 39]` (0-indexed residual stream
  captured **after** each listed target layer;
  `eagle_aux_hidden_state_layer_ids [2,14,26,34,40]` is the 1-indexed twin
  and is ignored).

## Reference semantics (from vLLM PR #46853, `DFlashLagunaForCausalLM`)

Authoritative behaviors the MLX port must match; cross-check SGLang
PR #29446 / TRT-LLM PR #15666 only if ambiguity arises.

1. **Context combine** (`combine_hidden_states`): view target hidden
   concat `[.., 5·2048]` as `[.., 5, 2048]`; apply `aux_hidden_norms[i]`
   to slice i; re-flatten; `fc`; `hidden_norm`. Result is the DFlash
   context vector (draft hidden size).
2. **Per-layer context K/V normalization (Laguna-specific):** each draft
   layer projects context K/V from `input_layernorm_l(context)` — its own
   input layernorm applied to the shared context vector — then `k_norm`
   and RoPE at absolute context positions. (Qwen3 DFlash instead norms
   context once with `hidden_norm`; the existing Swift path implements
   the Qwen3 formulation and must remain unchanged for qwen3.)
3. **Per-head attention output gating:** `softplus(g_proj(x))` computed
   in float32, broadcast across head_dim, applied to attention output —
   identical semantics to the Laguna target's attention gating
   (`gating: "per-head"`, gateDim = nHeads = 64).
4. **Draft block forward:** unchanged from the framework's existing
   mask-token block drafting (`[bonus, mask×(block_size−1)]`, logits from
   position 1, argmax). Drafter applies its own final `norm`, then the
   **target's** lm_head; embeddings come from the target's embed_tokens
   (mask token id 12 embeds through the shared table).
5. **Sliding window:** reference keeps full KV allocation and treats SWA
   as a compute-time limit; the Swift framework's existing
   trim-context-to-`sliding_window − 1` behavior is the equivalent. One
   open check (plan item): confirm against vLLM's `llm_base_proposer.py`
   that the speculators `max_anchors: 256` field imposes no additional
   context-window constraint at inference.

## Approach (selected: A)

Normalize at conversion time; extend the existing draft model with
config-gated Laguna behaviors. Rejected alternatives: (B) teaching Swift
the nested speculators schema + fused keys (schema fork, no artifact);
(C) a separate `DFlashLagunaDraftModel` class (would force a protocol
extraction through the engine/iterator for no behavioral gain).

## 1. Conversion script and artifact

`scripts/convert_laguna_dflash.py`, Python 3 **stdlib-only**: a
safetensors file is a little-endian u64 header length + JSON header + raw
buffer; splitting a row-major tensor along axis 0 is contiguous byte
slicing, so no torch/numpy dependency.

- **Input:** local snapshot dir of the poolside repo (obtained via
  `hf download` or an explicit path). **Output:** an MLX draft model dir
  (path given by the operator) containing `model.safetensors` +
  `config.json`, loadable by `DFlashDraftModel.load(from:)`.
- **Weight mapping:** for each layer N, split
  `layers.N.self_attn.qkv_proj.weight` rows into
  `q_proj.weight [8192,2048]` (rows 0..<8192),
  `k_proj.weight [1024,2048]` (rows 8192..<9216),
  `v_proj.weight [1024,2048]` (rows 9216..<10240). Every other tensor
  copies through byte-identical, BF16 preserved. Result: 68 tensors.
- **Config output** (flat `DFlashConfiguration` schema):
  - Hoisted: top-level `block_size: 16`, `num_target_layers: 40`.
  - Kept nested: `dflash_config: {target_layer_ids: [1,13,25,33,39],
    mask_token_id: 12}`.
  - Carried over: `model_type: "laguna"`, hidden/heads/kv/head_dim/
    intermediate/vocab/eps/rope_theta/max_position_embeddings/
    `sliding_window: 512` / `layer_types` (5× `sliding_attention`) /
    `tie_word_embeddings: false`.
  - New switch: `decoder_layer_type: "laguna_xs"` (poolside's own
    `config.py` vocabulary) — the single field that activates the Laguna
    draft behaviors in Swift.
  - Provenance: `_mlx_conversion: {source_repo, source_revision,
    source_sha256, script_version, converted_at}` (lands in the decoder's
    `ignoredConfigKeys`, harmless).
- **Strictness:** the script asserts the exact expected source manifest —
  58 tensors with exact names, shapes, dtypes — and fails loudly on any
  drift. It records the source file's sha256 in the provenance block.
- **Artifact home:** local dir now; a documented, manual HF-upload step
  (org/name TBD) later. R2/competition pinning deferred with the track
  decision.

## 2. Draft-model extensions (MLXSpeculative)

All additive, gated on a new `DFlashConfiguration.decoderLayerType`
(`"qwen3"` default | `"laguna_xs"`); existing checkpoints and the qwen3
code path remain byte-identical in behavior.

- **`DFlashConfiguration`:** add `decoder_layer_type` (default qwen3) and
  optional `gating` (string, informational; `laguna_xs` requires
  `"per-head"` if present). Validation: `laguna_xs` with a non-per-head
  gating value is a decode error.
- **`DFlashAttention`:** optional `@ModuleInfo(key: "g_proj") gProj:
  Linear` (hidden → nHeads), built only for `laguna_xs`. Output gating per
  reference semantics #3. Gating applies to proposal rows only (the only
  rows that produce outputs), matching the reference.
- **Per-layer context normalization:** for `laguna_xs`, the decoder layer
  applies its own `input_layernorm` to the shared context before context
  K/V projection (reference semantics #2). The fused context-KV fast path
  applies the norm before the fused matmul; the qwen3 path is untouched.
- **`DFlashDraftModel`:** optional `@ModuleInfo(key: "aux_hidden_norms")
  auxHiddenNorms: [RMSNorm]` (count = `targetLayerIds.count`), built only
  for `laguna_xs`; forward implements reference semantics #1 before the
  existing `fc → hidden_norm`.
- Untouched: mask-token block drafting, `DFlashTokenIterator`,
  `DFlashBatchedEngine`, greedy verify loop, `sanitize` (still drops any
  `embed_tokens.`/`lm_head.` keys defensively).

## 3. Laguna target port (MLXLLM)

- Copy `Laguna.swift` from
  `mlxfast-challenge-dev/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/`
  into this fork's `Libraries/MLXLLM/Models/`, adapting to fork APIs
  (KV-cache constructors, factory registration, quantization plumbing).
  The fork's mlx-swift supports `QuantizationMode.nvfp4`, so the NVFP4
  group-16 target checkpoint loads natively.
- Register `model_type: "laguna"` in `LLMModelFactory`.
- `DFlashTargetModel` conformance (Gemma4 pattern, simple form):
  - `dFlashHiddenSize` = 2048, `dFlashVocabularySize` = 100352,
    `dFlashLayerCount` = 40 (all from config).
  - `forwardForDFlash`: run the trunk once, capture the residual stream
    after each layer id in `targetLayerIds`, return
    `DFlashTargetForward(logits:hiddenStates:)` (framework concatenates
    in order). Gemma4's per-token sequential-verify optimization is out
    of scope for v1.
  - `embedTokensForDFlash` = the target's embed_tokens (no extra scaling;
    match the target model's own embedding path exactly).
  - `logitsForDFlashHidden` = the target's lm_head applied to
    drafter-normed hidden (drafter applies its own `norm` first;
    reference semantics #4).
  - Cache rollback: default helpers (trim if trimmable, else copied-cache
    rollback). Optimized rollback is future work.

## 4. Wiring, tests, e2e validation

- **mlx-bench:** extend the dflash bench/generate commands so a Laguna
  target (local path to the NVFP4 checkpoint; the ~21.6 GB base is
  already on local disk) pairs with the converted draft dir.
- **Unit tests** (mirror existing suites):
  - `DFlashConfigurationTests`: parse a checked-in Laguna fixture config
    (converted flat schema); validation failures (bad gating value,
    laguna_xs invariants).
  - `DFlashDraftModelTests`: laguna_xs wiring — shape-level forward with
    synthetic weights; zeroed `g_proj` ⇒ constant softplus(0)=ln 2 gate
    check; aux-norm slice order check (distinct per-slice weights ⇒
    detectably different combine output).
  - `LagunaDFlashForwardTests`: target capture — correct layer ids,
    capture point (post-layer residual stream), concat order.
- **E2E acceptance criteria** (all three required):
  1. Speculative greedy output token-identical to target-only greedy
     decode on the test prompts (DFlash greedy verify is
     exactness-preserving).
  2. Mean accepted tokens per 16-block ≥ 4.0 on coding-style prompts
     (miswiring collapses this toward ~1; the reference implementations
     report substantially higher). Below 4.0 is treated as a wiring bug
     until proven otherwise; record the measured number either way.
  3. Wall-clock speedup vs. plain decode reported via `DFlashBenchmark`;
     record the number (target: >1, no hard floor for v1).
- **Deliverables:** conversion script; converted local artifact; fork
  branch commits; green `swift test`; short `docs/laguna-dflash.md`
  covering conversion rerun, bench invocation, and the deferred HF
  upload.

## Risks / open items

1. The vendored `Laguna.swift` may depend on challenge-modified
   `MLXLMCommon` APIs that drifted from this fork's — the port adapts;
   budgeted as the least-predictable step.
2. `max_anchors: 256` — one-time check against vLLM's
   `llm_base_proposer.py` (see reference semantics #5).
3. Float32 gating math vs. reference bf16 kernels can flip near-tie
   argmaxes; acceptance-rate and greedy-equivalence criteria are robust
   to this (greedy equivalence is target-side, unaffected by draft
   numerics).
4. Local RAM: target (21.6 GB) + draft (0.9 GB) + KV/buffers must fit;
   validated on the operator's local machine (the challenge already runs
   the same target locally).

## Deferred (explicitly out of scope)

- Any `mlxfast-challenge` repo changes; new ranked track design
  (serial-v2 excludes speculative decoding; an organizer-provided DFlash
  track needs its own trusted variable-length block protocol per the
  challenge contract).
- HF/R2 artifact publication (documented as a manual follow-up).
- Gemma4-style sequential-verify and optimized cache rollback for the
  Laguna target.
