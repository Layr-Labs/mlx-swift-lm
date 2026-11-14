# Native Muse Glimmer text decoder

`MuseGlimmerTextModel` (MLXLLM) is a fork-only text tower for Muse Glimmer 30B. It serves
NVIDIA's ModelOpt checkpoint `nvidia/Muse-Glimmer-30B-NVFP4` through ContinuousBatchingV2 on
paged KV. Upstream's `MuseGlimmer` in MLXVLM is unchanged and still loads the MLX 4-bit
checkpoint through the generic factory. The native tower is not registered in a factory: a host
stages it with `MuseGlimmerInstallation`.

Status: text only; vision is not wired. The DFlash block drafter and exact verify windows are
part of it (sections below).

## Architecture

52 layers, hidden 6,656, 32 query heads over 2 KV heads of 128, SwiGLU 19,968, vocabulary
202,048. Layers repeat three sliding-window layers (2,048 tokens, RoPE theta 500,000) and one
full-attention layer without position encoding.

The arithmetic is upstream's, operation for operation, so BF16 weights give the same bits in
both towers:

- block norms scale by `1 + weight` in FP32; the norms before a sublayer use `rms_norm_eps`,
  the ones after it `post_norm_eps`;
- `embed_norm` and `qk_norm` have no parameters;
- queries are multiplied by `qk_scale_factor` in FP32 before RoPE, in addition to the usual
  `1 / sqrt(head_dim)` softmax scale;
- the attention output is multiplied by `sigmoid(gate_proj(x))`, where `x` is the normed layer
  input, before `o_proj`;
- the final norm is a plain RMS norm; logits are `lm_head`, times `output_multiplier`, then
  `20 * tanh(x / 20)`.

`MuseGlimmerTextModelTests` holds the two towers to equal bytes on a tiny random model.

## Checkpoint

`MuseGlimmerCheckpointPin` names the revision and the SHA-256 of the index and of each shard.
`MuseGlimmerInstallation.stage(directory:)` hashes the files, then reads each projection's format
from the verified shard headers:

| `weight` dtype | Sidecars | Layer | Count |
|---|---|---|---|
| BF16 | none | `Linear` | 24 |
| F8_E4M3 | `weight_scale`, `input_scale` | `ModelOptCalibratedLinear` (FP8 weights, FP8 input activations) | 180 |
| U8 | `weight_scale`, `weight_scale_2` | `ModelOptWeightOnlyNVFP4Linear` (NVFP4 weights, activations untouched) | 213 |

`hf_quant_config.json` is compared with that reading and never drives it. A missing, extra or
misshapen tensor throws a `MuseGlimmerInstallation.Failure` that names the key. The 809 vision
tensors are verified and listed in the receipt but not loaded; `Installed.visionSources` hands
them to a later vision loader.

Numerics decisions that enter a checkpoint's numerics fingerprint:

- FP8 layers quantize activations with saturating overflow
  (`ModelOptActivationOverflow.museGlimmer`), as deployment runtimes do. An activation beyond the
  calibrated range clamps instead of becoming NaN.
- Weight-only NVFP4 products run MLX's `nvfp4` quantized matmul on the producer's bytes; the
  per-tensor `weight_scale_2` multiplies the output in FP32. With BF16 activations the product
  is rounded to BF16 once by the kernel and once after the scale.

The installer reserves no memory. A host holds its own load admission around `stage`.

## ContinuousBatchingV2

- Layer kinds: `.slidingWindow(2048)` or `.full` per layer, no KV sharing, no sinks.
- Capabilities: `supportsPagedKV`, `requiresNativePagedKV` and `requiresPagedKV`. The last one
  is new and off for every other model; with it `EngineV2` refuses a backend that is not
  `PagedKVBackend`. Build the pool with `segmentSizeBytes` and
  `layerDTypes: model.kvStorageDTypes`.
- Prompt chunks run the whole trunk and write every layer's K/V; only the vocabulary projection
  of unread rows is skipped. Packed prefill is off.
- Complete prefix checkpoints: the model declares historical attention checkpoints and its K/V
  dtypes. Restore replays no tokens.
- Plain decode steps use `MuseGlimmerDecodeCommandBuffers.limits`.

`CBv2SteppableLanguageModelAdapter` drives plain serving. `MuseGlimmerCBv2Adapter` is for a
request-stateful drafter: the round driver refuses such a drafter on the generic adapter, which
is always a recurrent MTP model.

## Target hooks for a block drafter

`mtpHiddenLayerIDs` (default 1, 13, 25, 37, 49) names the layers whose output residual the
drafter conditions on. `forwardWithMTPHidden` and the adapter's `forwardWithHidden` return the
plain logits and those residuals concatenated on the last axis; the prefill variants keep the
hidden of every prompt row and project logits for the last row only. `rawTokenEmbeddings` skips
`embed_norm`; `headProduct` is `lm_head` alone.

## DFlash block drafter

`DFlash/` holds the published drafter for this tower. `MuseGlimmerDFlashDraftModel.load` reads
the BF16 artifact (`MuseGlimmerDFlashArtifactPin.museGlimmer30BBF16`) or its quantized form
(`.museGlimmer30BQuantized`: 5-bit MLP, 8-bit elsewhere) and checks the SHA-256 of both files
against the pin. `MuseGlimmerDFlashDrafter` is a request-stateful CBv2 drafter:

- Each request holds the drafter keys and values of its last `sliding_window` committed target
  positions in a fixed ring, charged once at admission.
- One block forward per round proposes up to 7 tokens (decision D6) with the target's own
  `headProduct` and argmax.
- The engine keeps a proposal only when it equals the token the target chose, greedy or
  sampled, so the drafter changes speed and never the output. The drafter requires this exact
  acceptance: under `.typical` the round driver is not built and speculation stays off.
- The target's tapped layers must be the drafter's `target_layer_ids`; another target is refused.
- The ring is advisory: it is not part of a prefix checkpoint, and after a restore it starts at
  the first observed position.
- A drafter refusal (an observation, draft step or finalize it cannot complete) ends the
  request with an error instead of continuing target-only: it fails closed.

`Settings.verifying(target, maximumSpeculativeRows:)` picks the verification. On a tower with
exact windows it is `rectangularExact` with per-row commits, and more than one request may share
a round. Otherwise it is serial, one request at a time. Asking for `rectangularExact` on a
tower without exact windows throws `MuseGlimmerDFlashError.target`.

## Exact verify windows

`setRowExactWindows(true)` makes the projections and block norms of every call, at any row
count, give each row the bytes it gets alone. One forward over a draft window (up to 8 requests
of 1 + 7 tokens) then equals the decode steps it replaces, in logits and in the K/V written, and
a decode step's bytes do not depend on how many requests share it.

- Projections: weight-only NVFP4 through `ModelOptWeightOnlyNVFP4RowExact`, FP8 through
  `rowExactForward` with the window schedule, BF16 through `ModelOptRowExact.matmul`. They run
  in pieces of 16 rows. A one-row NVFP4 step also takes the row-exact kernel, so NVFP4 decode
  bytes differ from the plain tower; BF16 and FP8 decode bytes do not.
- Block norms: from 32 rows MLX's row reduction takes another kernel, so exact calls of 32 or
  more rows reduce in pieces of 16.
- Calls above 64 rows (a decode step of more than 64 requests, a prompt chunk) take the same
  pieces. Prompt chunks therefore read every projection's weights once per 16 rows, so prefill
  is slower with exact windows on. `numericsProfile` is `muse-glimmer-exact-windows-v2`; `v1`
  left calls above 64 rows to MLX's kernels.
- Attention, RoPE, the gate and the softcap already compute each row alone. Under
  `withMTPRectangularVerification` the paged cache runs attention once per query column, or,
  when the drafter asks for it (`Settings.verifying` does), once per layer for the whole round:
  each row first writes its columns' K/V, then every (row, column) pair is one row of a single
  paged decode dispatch over that column's own range. The decode kernels give a row the same
  bytes whatever its batchmates, so this equals the column loop.

The switch is off by default, so the BF16 tower keeps upstream's arithmetic. It checks every
projection first and throws `MuseGlimmerRowExactError.unsupportedProjection` without changing
anything if one has no row-exact product. The installer does not set it: call it after staging
and before serving, and add `numericsProfile` to the checkpoint numerics fingerprint, because
decode bytes change with it.

## Tests

Tiny synthetic models, no weights:

```bash
swift test --build-system native --skip-build \
  --filter 'MuseGlimmer|ModelOptWeightOnlyNVFP4|ModelOptRowExact|PagedColumnsAsDecodeRows'
```

`MuseGlimmerExactWindowTests` compare windows of 1 to 8 tokens, and 2 to 16 requests at once,
with one-row steps in BF16 and float32, across the sliding window edge and the ring wrap, with
attention per column and as decode rows. `PagedColumnsAsDecodeRowsTests` check the decode-row
attention alone at the production head shape: 1 to 16 requests, rounds of 1 to 8 columns,
prompts across the 2,048-token window and the ring wrap, and rolled-back columns.
`MuseGlimmerDFlashExactVerifyTests` run the engine with the drafter on exact windows: the
output equals the target alone, greedy and sampled, cold and restored, one or two requests.

Real checkpoint and oracle (skipped unless the variables are set):

```bash
MUSE_GLIMMER_NVFP4_DIR=<snapshot directory> \
MUSE_GLIMMER_ORACLE_DIR=<directory of oracle .npz fixtures> \
swift test --build-system native --skip-build --filter MuseGlimmerRealArtifactTests
```

With `MUSE_GLIMMER_DFLASH_QUANTIZED_DIR` (the quantized drafter) also set,
`MuseGlimmerExactWindowRealTests` check 8- and 3-token windows against steps on the oracle
prompts, and greedy and sampled output with the drafter against output without it.
`MuseGlimmerDFlashRealArtifactTests` check the drafter itself against the oracle and its traces
(`MUSE_GLIMMER_ASSISTANT_DIR`, `MUSE_GLIMMER_DFLASH_TRACE_DIR`).

The oracle is an independent float32 implementation. The FP8 layers round activations to an
8-bit grid, so a last-bit difference near a rounding boundary becomes a visible step: the
oracle's own single-pass and chunked runs differ by 2 to 11% in the hidden states of layers 13
to 49. The tests bound hidden states and logits at about twice the measured distance and
require the oracle's greedy tokens wherever its top-1 margin is not a near-tie.

`MUSE_GLIMMER_BENCH=1` adds a one-row decode and prefill timing on the paged backend.

## Not done

- Vision: upstream's vision classes are private, so loading into them needs a small addition to
  the upstream file.
- Exact windows set by the installer, and the numerics profile in the checkpoint fingerprint.
- A measured prefill activation bound, load residency accounting and factory registration.
