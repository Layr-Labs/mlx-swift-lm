# Native Recursive Flint

The `flint` registry entry loads a native Swift decoder and supplies its hybrid
attention geometry to ContinuousBatchingV2. Python is reference material only.

## Checkpoint contract

The public release has 49 layers: two dense MLPs followed by routed/shared MoE;
32 query heads, four KV heads, head dimension 128; 256 experts with six selected.
All geometry and full/sliding indices are decoded and validated from config.
Sliding layers retain a 513-token window including the current token and apply
default RoPE; full layers have no RoPE. Learned sinks remain active even at zero.
Routing uses sigmoid scores in FP32, correction bias for selection only, and
normalized selected probabilities. The weighted expert reduction and head input
are FP32, following the reviewed BF16 reference semantics.

`FlintCheckpoint.swift` owns the official export conversion: packed U8 is viewed
as U32, embedding weights use eight bits, projections/experts four bits, all with
group size 64. Norms/router and scales preserve BF16. Strict inventory, shape,
dtype and alias checks run before installation. The only synthesized parameter
is the documented absent expert correction bias (zero). BF16 per-expert tensors
must be complete before stacking. Generic loader defaults are unchanged.

## Serving prefill

Flint implements the existing CBv2 prompt-output contract used by GPTOSS and
Gemma: every prompt token traverses the full trunk and updates KV, intermediate
chunks omit unused vocabulary projections, and the frontier projects only its
last hidden row. Head input remains FP32. Full forward and decode keep their
original contracts; packed prefill remains disabled.

## Protocol

`.flint` parses complete `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`
frames. Strings preserve whitespace; structured arguments must be valid JSON.
Values are opaque to closing tool and reasoning tags. Declared string schemas
prevent JSON-looking text such as `{` or `[todo` from changing framing. A literal
`</arg_value>` inside a raw string is ambiguous in the published protocol and
cannot be represented reliably; callers must reject such replay input. Incomplete/malformed
frames never become executable calls. Higher-level schema/choice validation
remains the serving layer's responsibility. Flint chooses reasoning itself;
provider-specific rejection of ineffective effort/enabled controls belongs in
d-inference, as does text-only media admission.

## Validation and limits

`FlintTests` uses an independent NumPy model oracle and checks sliding-window
boundaries, single/chunked cache parity and heterogeneous CBv2 row positions.
`FlintPrefixTests` compares cold and serialized-prefix restores through real Flint
layers on contiguous and paged backends after window wrap, including shutdown
reservation cleanup. The foreign-salt case tests store lookup isolation.
`FlintToolTests` exercises split markers, opaque Unicode values, parallel frames
and malformed input. Run numerical checks with `MLX_ENABLE_TF32=0`; that is an
explicit diagnostic posture, not a change to MLX's production defaults.

The official INT4 release used for local testing is
[3b93f9053b530488405ae9fca51d4900d6ce21f0](https://huggingface.co/RecursiveSI/flint-int4/tree/3b93f9053b530488405ae9fca51d4900d6ce21f0).
An opt-in `FlintCaptureTests` run on M5 Max confirmed executed dense and routed
expert NAX kernels with official INT4 weights, TF32 off, and an FP32 head fallback.
It captures layers 0–2 and the final head; it does not measure whole-model
throughput. Metal includes loaded checkpoint buffers in the capture: allow at
least 25 GiB temporary disk space.

Tiny-model tests and short generation do not establish full-release BF16 parity,
all-context cache qualification, physical RAM tiers, NAX speedups or catalog
eligibility. The preview used by mlx-vlm was inaccessible during intake; its
acceptance is not inherited. MTP and reduced-precision KV are not qualified.

## Attribution

This is a Swift adaptation of Prince Canuma's MIT mlx-vlm Flint model and
Recursive's Apache-2.0 rec checkpoint conventions. Read the exact revisions,
adaptations, numerical differences and complete licenses in
[ThirdPartyNotices/Flint.md](../ThirdPartyNotices/Flint.md). The NumPy oracle
also credits Prince Canuma's reference test. No checkpoint weights are bundled.
