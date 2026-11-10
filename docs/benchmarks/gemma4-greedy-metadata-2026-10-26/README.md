# Identity-greedy sampler metadata

Plain greedy batches previously constructed sampling-parameter tensors and
penalty-history buffers that their forward path never used. This change skips
that setup and the unused all-greedy mask. On the standalone Gemma 4 artifact,
a matched warm 8K-context Darkbloom HTTP comparison measured a 1.15–1.97%
improvement in the post-first-frame decode proxy across the two ABBA orderings.
The candidate's B/B drift was -0.91%, and one reverse-order pair was slower.
This is a modest, bounded observation, not a pooled or universal speedup.

The measured artifact was SDK `aee6ebb6835c62feda80000c83f81f2788cc9822` plus
only these greedy metadata edits. It did not include the earlier GeGLU,
command-buffer or compiled feed-forward changes. This proposed branch stacks
the same source bytes on SDK `221304cd52fdd701b41edeae4ed0e4070855983e`
([compiled feed-forward report](../gemma4-compiled-feedforward-2026-10-26/README.md)).
These runs do not measure the combined stack. Its endpoint capability and speed
qualification remain pending.
The percentages here must not be added to the earlier feed-forward results.
[receipt.json](receipt.json) maps the unchanged sampler inputs and tested output
source hashes and records sanitized test, artifact and endpoint evidence.

## Change boundary

`LogitsPipelineV2.setRows` first keeps the current row count. It then uses the
existing state-clearing routine when every row has temperature below the
existing greedy threshold, requests no raw logprobs, has no logit bias, has no
active windowed repetition penalty, and has zero frequency and presence
penalties. All other configurations use the original builder. The predicate
matches the existing transform flags; it does not change their thresholds or
floating-point comparisons. Top-k, top-p and min-p keep their existing behavior
for greedy rows.

The forward path retains the same float32 conversion and hard token constraints.
Returning from an identity batch to an active transform rebuilds the original
state. `SamplerV2.setRows` still reconstructs every request ID, seed, greedy flag
and history-derived step index; it omits the mask only when the existing
all-greedy branch cannot read it. Mixed and stochastic masks are unchanged.
MTP commit still invalidates the request-row context through the existing hook.

This is shared host-side sampler code with no model or chipset gate and no new
public setting. Its applicability to other models does not establish their
performance. No model arithmetic, quantization, precision mode, NAX kernel,
MTP depth or acceptance, admission, paged attention, prefix caching or SSD-write
implementation or policy is changed.

## Standalone endpoint comparison

The target was `gemma-4-26b-qat-4bit`, revision `2026-06-08-r1`, on an M5 Max
with 40 GPU cores and 128GB unified memory, macOS 27.0 build 26A428, AC Automatic
and automatic fans. Provider revision was
`f9d1af3d0d3e1838974a5cfdff0a08142977b45e`, MLX Swift
`eedca444432296c52d02b2595564ada936131633`, and core
`ff4bb35b60c3aadafb9e0bd30b6180cef646fb1a`. Both artifacts used the same Metal
library and retained natural adaptive MTP with exact acceptance.

Each A1/B2/B3/A4 leg used a fresh owned provider and artifact-specific exact
prewarm, then at least 30 seconds of nominal idle posture before each measured
cohort. The same two C1 requests per leg reported 8,204 prompt tokens, 8,192
cached tokens and 256 completion tokens, temperature 0 and seed 1729. Request
counters, immutable artifacts, thermal boundaries and full cohort membership
were checked. No endpoint policy was disabled to create this comparison.

| Ordering | Baseline median tok/s | Greedy median tok/s | Median paired gain |
|---|---:|---:|---:|
| A1 → B2 | 98.586 | 100.525 | 1.967% |
| A4 → B3 | 98.497 | 99.607 | 1.153% |

Both pairs qualified in each ordering with no exclusions and exact observed
outputs; A/A and B/B output controls also matched. The paired A/A decode drift
was -0.095%; B/B was -0.910%. The reverse-order pair ratios were 1.02875 and
0.99431, so the aggregate observation does not imply every request improved.
The reported gain is the median of pair ratios, which can differ from a ratio
of displayed medians.

The metric is `(completion_tokens - 1) / (last generated SSE frame time - first
generated SSE frame time)`, including generated reasoning and tool deltas. It
is an HTTP proxy, not engine token timestamps or a one-frame-per-token claim.
TTFT was mixed: approximately 0.6436 → 0.6432 seconds forward and
0.6104 → 0.6163 seconds reverse. The receipt retains E2E, individual pair,
same-arm and memory observations. There is no cold-prefill or concurrency claim.

Process lifetime physical-footprint maxima include loading and prewarm; sampled
RSS is not an exact high-water mark. The endpoint exposes no MLX allocator peak.
The recorded process values do not establish an isolated allocation reduction.

## Correctness and remaining qualification

The standalone focused suite passed seven required tests with zero failures or
runtime warnings. Its optional configuration benchmark was disabled and counted
as one skipped test; it supplies no microbenchmark result. The required tests
cover exact float conversion bytes, clearing stale penalties/logprobs, restoring
transforms, mixed transform rows, required token constraints, seeded RNG/history
transitions and MTP row-context invalidation. Both production files remain byte-identical to
the validated files. The composed test has two whitespace-only wraps from the
installed formatter; the receipt maps its pre-format and formatted hashes. The
standalone numerical binary used the pre-format test bytes. The only initial
compiler repair explicitly typed a test fixture as `[Float]` and did not change
production code.

The standalone endpoint passed all 12 functional smoke cases and matched the
baseline's exact outputs: reasoning on/off, automatic and required tools, JSON
object/schema, image/video color steering and text interleave. This fixed corpus
does not establish every model capability or every numerical trajectory. No
result is claimed for other chipsets, cluster execution, or the composed stack.

For a later allocated test run, select
`MLXLMTests/KernelTests/CBv2GreedyMetadataTests` in the Xcode package test scheme
with the source-matched staged Metal library. Leave
`DARKBLOOM_GREEDY_METADATA_BENCH` unset for correctness-only qualification.
Run the relevant existing sampler, constraint and MTP tests as well; they have
not been silently counted as executed by this focused suite.

The installed formatter was idempotent, and independent review confirmed the
test-only formatting delta. Strict lint reported exactly two pre-existing
`NoAssignmentInExpressions` findings, reproduced on SDK `221304` at lines 417/421
and shifted to 430/434 by the 13-line insertion. No new finding or unrelated style
change is included. Normal CI formatting remains pending.

All 11 required library DocC targets passed on the final composed source with
warnings as errors. This compiled symbol graphs; it did not rebuild or repeat
the numerical test binaries. The standalone focused results are retained with
explicit source mapping, and no numerical repeat is claimed solely for the
formatting/checkout change. Full-package CI and combined endpoint capability
and performance qualification remain pending. Preliminary source/refactor
review found no additional production rewrite needed; publication and retention
remain subject to independent review and the remaining checks.
