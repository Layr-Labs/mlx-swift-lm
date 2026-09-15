# Qwen 3.8 Next (Flash-Next) qualification scope

The model implementation is native Qwen4. The numerical and speed qualification below covers the original declared
affine-Q4 target and runtime sampling contract; it is not BF16 equivalence or
universal answer-quality certification. The companion provider repository
contains the API and lifecycle qualification procedures.

## Selected-checkpoint qualification update

The September 15 follow-up preserves the production `Libraries/` and
`Package.swift` trees from public SDK `f3f0b235c4e98c9c8e9f3cd76b4ec1752c27960b`.
It adds explicit test-only artifact profiles and records the completed native
and companion-provider requalification. No replacement numerical baseline,
weight, tokenizer, MTP-head, sampler or serving-default change is included.

- Each SDK optimization posture reports 891 XCTest tests with 9 skips and no
  failures, plus 1,210 Swift Testing pass records and 14 skips. Framework
  aggregates and individual pass records are not interchangeable totals.
- The selected checkpoint passes 25 native retained/rejected state and logit
  cases, trained-head replay/discard, serialized suffix restore, and output
  budgets 1–9. The original affine-Q4 profile and its oracles remain separate.
- The companion's final real-coordinator local API matrix passes all 118
  required cells across MTP OFF/AUTO, including reasoning OFF/ON, tools and
  supported streaming/nonstreaming Chat/Responses. Standard API compatibility
  does not replace strict semantic-quality or hosted-route qualification.
- Default one-row long-prefix cache checks pass 24 waves/32 responses across
  MTP OFF/AUTO and cache OFF/ON, with observed 6,144-token warm reuse. A separate
  opt-in multirow recheck has 58 passing and 16 failing waves; it remains
  disabled by default and is not qualified by queued HTTP concurrency.
- A 36-request matched text/image/code comparison preserves requests, outputs
  and token counts. Candidate/reference wall-time ratios range from 0.971784
  to 1.008637. These client measurements do not certify sustained throughput.

Strict literal-copy and some visual/tool-history answer failures remain open.
Physical tiers, original BF16 equivalence, signed persistence/trust and final
release operations require their own evidence. The complete current provider
record is `docs/reports/2026-09-15-qwen38-native-api-qualification.md` in
[d-inference PR1030](https://github.com/Layr-Labs/d-inference/pull/1030).

## Historical original-checkpoint runtime checks

Performance checkpoint qualified September 14 and published for draft review September 15, 2026. The qualified
runtime delta from the initial Qwen4 implementation is four files,
with archived patch SHA256
`afdd5eb45e89ae547b8b28b34ce656b6cdfdd78f6836cd3655db95a2b542f0d1`.
Review packaging preserves that exact runtime patch; later submission-cadence
and selected-KV experiments are excluded. Qualification is attached to these
runtime inputs and artifacts, not an assertion of a fresh build of every
documentation-only commit or green remote CI.

- Full SDK suites, separately defaults and optimizations enabled: each has
  875 XCTest passes/9 skips and1209 Swift Testing passes/14 skips, no failures.
  Applicable native opt-ins also ran separately; unrelated artifact tests and
  optional diagnostics are not silently counted as executed.
- Full-KV attention:360 bit-exact FP16/BF16 layout/context/width/partition cells,
  with actual dispatch witnesses. Layer submission:4 safety tests,104 miniature
  native-state cells, plus25 actual full-weight retained/rejected-state cases
  and exact target/MTP output-budget boundaries1–9 passed.
- Twelve interleaved fixed-depth arms retain all192 golden output tokens and
  identical per-depth MTP proposal/acceptance traces. Native target-only decode
  improves27.2→37.0 tok/s; depth2 improves46.4→59.5–60.0 (89.7% acceptance),
  depth4 improves52.5→63.3–63.6 (74.0%). Prefill889–906 tok/s on this7K fixture.
  These are bounded native measurements, not a sustained workload guarantee.
- Companion provider full defaults/enabled suites, local Chat/Responses/tools/
  reasoning OFF/ON and24 actual encrypted loopback-WebSocket cases passed.
  Standard compatibility is not hosted OpenRouter certification or universal
  payload fidelity. Native state, PLE, cancellation/readmission and frame
  extraction have separate execution evidence.
- Media-cache mechanisms passed14 reference and18 cached checks; cached
  lifecycle passed9/9. Existing answer-quality failures remain open. Across
  old/new and OFF/AUTO runs,65 standard API/media outputs and18 successful
  strict-fidelity outputs match; two original generated histories replay exactly.

For provider-specific totals and limitations, consult its qualification record.
The assessed runtime artifact is bound by executable SHA256
`3ac76ef3e9bade476fb2e21d71cab25c4211bc49cc39681fced25678cea808b8`
and metallib SHA256
`2129f6132794d84243c02631bc7478417dba0e0dbd10467beab56c11bf20bdd2`.
Source-equivalence checks must accompany publication-only packaging changes.

## Opt-in performance contract

Clean-checkout follow-up: the generic packed-vision fixture previously relied
on a50ms enqueue delay to obtain one three-row batch. A full enabled run kept
all reference tokens but observed one-row/two-row batches and correctly failed
its shape assertion. The fixture now groups its already-materialized synthetic
submissions on the existing engine queue before stepping. The exact[3,16]
assertion and all independent output oracles are unchanged; no runtime scheduler
or model code changes. Preserve the original failure and require isolated and
full-suite checks in both optimization postures before promotion.

`DARKBLOOM_QWEN4_QSA_PARALLEL_FULL_KV=1` enables the existing independent QK
tiles followed by the unchanged ordered softmax/PV recurrence on the caller's
full materialized KV. It is separate from compact-KV parallel scores. The
qualified setting uses `DARKBLOOM_QWEN4_QSA_PARALLEL_VALUE_PARTITIONS=32`.
`DARKBLOOM_QWEN4_LAYER_ASYNC=1` submits valid singleton text layers early for
native paged widths1–6, excluding explicit embeddings/positions. It resolves
deferred host fills without closing the owning scope and checks the whole
native fault bank before and after fills. Existing final roots, commit/rollback,
retirement and fallbacks remain authoritative.

Both new switches default OFF. The qualification profile enables them
explicitly; including this code does not activate them fleet-wide. No weights,
precision, selector, trained head, sampler or MTP controller changes are included.
Disable the two switches to revert scheduling/attention dispatch without a
model conversion. Preserve paired SDK/provider pins when reverting commits.
The40/66–88 tok/s goal and complete deployment qualification remain open.

## Behavior that must remain explicit

- Qwen4 uses its own model/configuration and vision processor. Reuse of a
  shared XML/framed-JSON codec is not an architecture alias.
- Preserve all target, embedded trained-head and learned-table weights.
  SSD n-gram/PLE offload remains enabled; no quantization change is implied.
- Prefix snapshots carry complete QSA/GDN/PLE/position state and authenticated
  media identity, not ordinary attention-only KV.
- Constrained required/named text tools remain target-only under the current
  CBv2 safety gate. Ordinary text and auto-tool requests retain eligibility;
  media remains target-only. Do not bypass the constraint gate to claim MTP.
- Semantic copying, reasoning-budget exhaustion, brief-video recognition and
  strict format following remain quality limitations. Never repair arguments
  or weaken an oracle to mark these as passed.
- Signed persistence/trust, physical hardware tiers, original BF16 comparison,
  hosted routing and deployment require separate qualification.

## Reproduction

Tests and opt-in requirements are in `Tests/MLXLMTests/Qwen4RealStateTests.swift`,
`Qwen4RealStateFixture.swift`, and the companion provider's validation guides.
Record exact source, artifact, sampler, MTP/cache posture, binary/metallib and
pass/fail/skip counts. Supply approved environment configuration externally;
never commit endpoints, credentials or private diagnostics.

The full-weight state fixture binds the config and index SHA256, tensor/shard
counts and embedded MTP entry count to `Qwen4RealArtifactProfile`. The default
`affine-q4` retains the original target; explicitly set
`DARKBLOOM_QWEN4_REAL_ARTIFACT_PROFILE=selected-q4` for the separately selected
checkpoint. Unknown profiles and mismatched artifacts fail before weight load.
The same independent serial-state, rollback, serialization and output-budget
assertions apply to either profile; no original-target goldens or tolerances
are relabeled. This fixture change is not itself a passed full-weight run or
a whole-payload integrity proof.

Implementation provenance and excluded experiments: [composition](composition.md).

## Artifact and deployment boundary

A separately selected checkpoint uses predominantly 4-bit storage with selected
5-, 6- and 8-bit modules and retains its native vision and embedded MTP weights.
The original fixed-depth numerical/speed evidence above does not automatically
qualify that different target. See the companion provider's September 15 report
for its separate local compatibility results and unresolved literal-string,
visual-quality and release gates. No artifact is made public by this code PR.

## Mixed text/media position correction

Mixed decoding builds a rectangular positions tensor, including synthetic ramps
for text requests that originally had no explicit positions. Those filler rows
must not change the text request from its native offset-based rotary path to
explicit media rotary processing. Equal-valued genuine media planes remain
explicit; tensor values are not an absence test.

`CBv2Qwen4PositionScope` carries request-owned host booleans for one synchronous
forward. `EngineLoopV2.withQwen4PositionScope` binds ordinary target forwards and
direct hidden-returning MTP seed/decode/prefill/verification paths. This includes
stateful history maintenance when batch pressure reduces draft depth to zero.
`Qwen4ExpBatchedQSA.positions` honors that provenance without changing external
explicit-position callers, singleton calls or other model architectures.

The correction's release build and 43 targeted tests passed: seven XCTest
position/policy/PLE tests plus 36 stateful-MTP integration tests, with no failures
or skips. Actual EngineV2 route spies exercise both ordinary and hidden-returning
mixed forwarding; scope tests cover reordering, equal media planes, nested
throws and concurrent isolation. These targeted results do not replace a fresh
full-suite run or full-model exactness qualification.

On the selected checkpoint, MTP OFF/cache ON and MTP AUTO/cache OFF each passed
11 unchanged HTTP/lifecycle waves: 19 completed responses and one live-peer
cancellation. Content, reasoning, tool arguments, finish and usage matched the
original isolated references. Both arms observed native B4; the AUTO singleton
control proposed 238 draft tokens, while OFF proposed zero. This is scoped
mixed-position evidence, not a full cache-by-MTP matrix or native logit proof.

The companion provider report records the real-model reproduction and its
remaining boundaries. Longer-prefix batching already diverged on the prior
runtime, including cache-disabled cases; that separate failure is not fixed or
waived here. Default admission remains one active row, the batched-QSA path
remains opt-in, and native Qwen4 MTP speculation remains capped to one row.
Weights, quantization, trained heads, sampler, cache format and fleet defaults
are unchanged.
