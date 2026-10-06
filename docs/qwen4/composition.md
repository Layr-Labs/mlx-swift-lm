# Qwen 3.8 Next (Flash-Next) private native Qwen4 composition

Prepared September 11, 2026. This is a local review candidate, not a production
qualification, benchmark result, or publication authorization.

Parent verification after composition is recorded in
[the qualification scope](qualification.md).
The original composition-only checks below retain their historical scope.

## Source identity and scope

- Base: current merged SDK `ce446cc5f76e013855fe0bde9002b6db1ac091b7`.
- Branch: `codex/qwen38-next-prod-support-20260911`.
- Selected native model source: private SDK
  `db4d5b8823f133ee93937c47bc0c420013146c02`, read from the paused September 10
  finalization checkout. Selected current working-tree fixes are listed below.
- Dependency revision remains mlx-swift
  `6d6796d7a81b656d2749d39067e0a6bea2bc2986`. No local dependency path override
  or old Package.swift was copied.
- No commit, push, build, dependency resolution, model evaluation, service
  operation, or GPU experiment occurred during this composition.
- Changes were confined to this SDK candidate. The paused source and the
  separate Nemotron work were read-only.

## Native implementation retained

The native Qwen4 text model retains its QSA indexer and sparse attention,
GDN recurrence, hyper-connections, MoE, packed SSD PLE, native activation
precision, embedded Lightning MTP, and exact request-state interfaces.
Factories recognize `qwen4_exp` and `qwen4_exp_text`.

Complete prefix state includes native K/V, GDN and PLE convolution/SSM state,
QSA raw index keys, original int32/int64 position axes, the actual pooled-key
frontier, and the assistant's pre-mixer residual/token/frontier history.
Ordinary KV-only prefix reuse remains false. Complete recurrent checkpoints
and native paging remain model capabilities. Provider activation and its
authenticated SSD store are separate composition responsibilities.

The VLM wrapper, preprocessing, video helpers, and attribution are preserved
so existing typed consumers compile against the native wrapper. Their presence
does not qualify vision. The provider must retain the language-model-only
artifact/capability gate.

Useful native provenance anchors include:

- `0d8da5b`: native model, SSD PLE, GDN and model-state port.
- `b2a1076`: model registration and model-owned PLE directory lease.
- `9b93063`: exact trusted MTP prefix history.
- `a5e04e9`: complete QSA checkpoint tensors and transfer ownership.
- `a264a10`: observed checkpoint geometry and paged engine reuse.
- `d9de9fd`, `c3dd6f7`: deferred PLE scope and fill-before-evaluation boundaries.
- `e81641c`: canonical attention when a verify window crosses the sparse budget.
- `b325240`: malformed PLE metadata/geometry refusal before allocation.
- `ad8af82`: padded projection caches bound to owners and parameter identities.
- `fecc594`: singleton router arithmetic for opt-in batched decode.
- `1f23d0a`: mutable-input declarations for native paged copies. Already merged
  runtime declarations were retained; the added selected-gather implementation
  uses the same current mutable-input API.

These are provenance anchors, not a claim that their old branch was copied
wholesale or that every historical experiment was carried forward.

## Shared changes inspected and composed

- `GatedDelta.swift`: additive per-position fp32 state capture and Qwen-only
  blocked prefill dispatch. Existing callers keep `useBlockedSeq=false`.
- `Qwen35.swift`: Qwen4 L2 normalization, sigmoid output gate, activation
  precision, exact projection and captured-state paths are selected by the
  Qwen4 normalization/reduction profile. Qwen3.5 defaults retain their original
  RMS-scaled normalization and SiLU gate. Exact timewise router verification
  remains intact. None of the later dirty GDN/M-RoPE/MoE experiments was copied.
- `Qwen3Next.swift`: additive gated-normalization activation choice, with the
  existing SiLU default.
- `SwitchLayers.swift`: Qwen4 top-10 expert reduction and explicitly model-owned
  affine dispatch. `SwitchGLU.projectExpert` reads the immutable
  `qwen4ProductionSwiGLU` profile and calls the private Qwen projection path
  only for an ordinary quantized child. Direct `QuantizedSwitchLinear` calls
  always retain generic MLX behavior. Expert count alone cannot activate a
  Qwen kernel or BF16 conversion. Quantization and fused/split module replacement
  retain the owning profile; shared child objects carry no mutable family flag.
- `Load.swift`: model-owned key filtering before shard evaluation, and no
  whole-shard read-ahead when the model retains an SSD-only table.
- Layer/sequence caches: request-owned QSA sidecars survive row recomposition;
  cache views retain write/read and speculative rollback ownership. Target
  auxiliary admission accounts for sequence-growing side state independently
  of MTP activation.
- Complete checkpoint codec: QSA tensor roles, int64 position storage, native
  geometry checks, reserved capture/import and QSA restoration on adopted rows.
- `EngineLoopV2.swift`: only deferred host-fill scope and resolution were added.
  The merged committed-decode/goodput logic was retained.

No files under the existing shared `ContinuousBatchingV2/MTP/` directory were
changed. Existing Gemma4 and Nemotron model files are unchanged from the merged
base. Private diagnostic interfaces, older MTP controller versions, and
separate Nemotron experiments were not imported.

## Selected current correctness fixes

1. Official Q4 stacked affine experts are re-keyed with their scales and biases
   by `qwen35FuseSwitchMLPGateUp`. The embedded assistant uses fused SwitchGLU
   and applies that checked mapping after stripping the assistant prefix.
2. `injectMissingNgramWeightScale` supplies the existing identity scale only
   when the official checkpoint omits it and the corresponding PLE prefix is
   present. Existing checkpoint scale values are preserved.
3. Qwen paged decode writes native pages, gathers their visible contents, and
   uses the same mask-free SDPA terminal as contiguous decode. Rectangular
   verification retains the per-column canonical attention order. The failed
   segmented-kernel A/B override was not copied into this candidate.
4. The prefill `PagedLayerCache.attendQueryBlock` mask-mode correction is
   explicitly restricted to `kind.qwen4IndexerCompressRatio != nil`, no spans,
   and no softcap. Other models retain the original absolute Boolean mask.
   The copied multi-chunk and crossover tests identify Qwen geometry explicitly.
5. Additive listing context and cache-usage fields preserve omission when
   unknown. Cache counts flow from `ServerGenerationInfo` to chat usage and
   Responses usage; counts are clamped to the request's prompt/input length.
   Unrelated Responses streaming, conversation-store, and status changes were
   not copied.

The private assistant's diagnostic conformance, diagnostic field, metadata
getter, initializer argument, and release callback were removed because those
private instrumentation types are absent from merged main. No replacement
stubs were introduced. Trusted observations, state arrays, history counters,
draft/finalize/discard behavior, and ownership cleanup are retained.

## Deliberate differences from the latest Grok binary

The latest historical `808b7f70…` binary is not this candidate. This candidate
uses the committed canonical RMSNorm, environment getter, affine projection
and PLE gather implementations. It does not carry later eager-dispatch,
RMSNorm affine-cache/hot-path tuning, PLE prefetch/concurrent cold-pread additions,
MTP bootstrap-cost override, submit-every-draft alias, or rejected compile,
router, softmax, M-RoPE, reshape and projection-fusion experiments. Those
omissions are not throughput claims; fresh exact-output comparisons are needed.

Established native defaults retained: BF16 activation policy, canonical HC
compile/hybrid projection, GDN blocked/stacked recurrence, Qwen affine QMV/QMM,
native Steel sparse attention, native indexer/top-k and bounded index capacity.
The uncommitted fast RMSNorm and fused-HC experiments are absent.

Committed experimental state APIs remain present with their original inactive
defaults to avoid rewriting trusted snapshot/rollback contracts: assistant
prime chunk = 0, cold replay skip = false, Qwen multirow batching = false,
selected KV = false, strided KV = false, parallel scores = false, incremental
pooled index = false, Steel indexer = false, and output partitions = 1. These
are not promoted or qualified by this composition. The standalone optional
Qwen router-finalizer implementation and shared hook were omitted entirely.

## Native kernel resource portability

Three previous header readers derived a sibling mlx-swift checkout from
`#filePath`. They now read packaged resources through `Qwen4ExpMetalHeaders`:

- `Qwen4ExpAffineQMM.swift`.
- `Qwen4ExpGatherQMM.swift`.
- `Qwen4ExpQSAIndexerSteel.swift`.

`Package.swift` adds only `.copy("Resources/Qwen4Metal")` to the existing
MLXLMCommon resource bundle. Missing required resources fail explicitly.
The original preamble concatenation and kernel-body extraction are preserved.
No native Qwen runtime source-checkout readers remain.

The fragments were extracted exactly from the pinned merged mlx-swift revision
above, with core `3fa8f25e6451174d7b06be372c3a24272b77d88e`. The generated C++
source files were also SHA-256 identical to the paused qualified dependency.
Each packaged fragment was compared byte-for-byte with the exact original
`R"preamble(...)preamble"` payload; all three matched.

| Resource | Bytes | SHA-256 |
|---|---:|---|
| gemm.metal | 57,347 | cc2597fad25939505da77537ff15e78eba2fccbdf14a0e636fb2d2ac0bb4bb6c |
| quantized_utils.metal | 2,785 | 36893d020956fec4b63f37855035dfebc317bd06a966de14c6372c9a2971ef28 |
| quantized.metal | 99,331 | d39dae2a27d0352facbe0e07c97af1ba0782a923698ce3931a92417d9e527162 |

Original Apple notices remain in the fragments; the core MIT license is copied
beside them. Apache and model/preprocessing implementation notices are retained.
The resource change is packaging work; it is not a fresh numerical audit of
the upstream Metal templates.

## Validation performed and pending

Performed without model execution:

- No-output Swift frontend syntax parsing: 113 changed/new Swift source and
  test files passed, exit 0. This is not typechecking or linking.
- `git diff --check`: passed for tracked changes.
- All three packaged preamble payloads compared byte-for-byte: passed.
- Static scans found no references to the removed private diagnostics or later
  rejected-performance types, and no native runtime `#filePath` header lookup.
- The current dependency declares the constant-cache identity C API and
  mutable-input Metal kernel API used by this source.

Ported tests cover configuration, packed expert remapping, PLE raw-byte transport
and resource lifecycle, deferred-fill boundaries, assistant history/rollback,
QSA checkpoint tensors/frontiers, target auxiliary admission, native paged
restore, canonical rectangular state, sparse crossover, batch policy, native
tool format and preprocessing geometry. New tests cover packaged resource
digests and additive listing/cache accounting. These tests have not run.

Required next gates: composed-source typecheck/build and source-matched metallib;
isolated fresh-process PLE test on the official Q4; full-model T=0 serial/MTP
oracle comparison with prefix OFF; actual cache ON hit/miss and encrypted SSD
restart, tenant/cancellation/readmission/cleanup qualification; and Qwen3.5,
Gemma/GPT-OSS/Nemotron regression checks proportionate to shared changes.
The existing native checkpoint fixture omits PLE, so it is not full PLE-plus-
checkpoint qualification. Old binary receipts must not be relabeled as proof
for this new composition. Physical 128-GiB qualification remains separate.

## Follow-up shared-family and GDN review

Parent review identified an E512 family-scope defect in the initial composition:
the generic `QuantizedSwitchLinear` entry point attempted Qwen kernels and could
narrow a non-Qwen fallback to BF16. This is corrected by explicit per-call
authorization from the immutable owning `SwitchGLU` profile. No process-global,
thread-local, or child-module family state is used. Custom quantized subclasses
retain their own ordinary call behavior.

`Qwen4SwitchDispatchScopeTests` covers unrelated E512 affine and MXFP4
projections, original FP32 output, no Qwen candidate attempts, generic/Qwen
owners sharing identical child projections, and profile preservation through
quantization plus fused/split replacement. The Qwen-owned path is compared
byte-for-byte with the existing selected native primitive on bounded fixtures.
Run this filter alone for its process-counter assertions. Tests are pending.

The stacked GDN implementation was compared directly with the current
unstacked source, excluding whitespace/comments and the added captured-state
store. These blocks match exactly:

- State load: `033803fc57740f2b82e2910b9667ef6e57f8367c8820cabca958ec8b6298057c`.
- Decay/Kahan dot: `9bfbc2cffdce5ba28158ec783fbeb08024e2b661ebfe04469744ef60dbe6879c`.
- Delta/state/output update: `35b753e83ca1560d76ceebfca5fc99659da71c388994c4abe70a3ddf0b9b9162`.

Both dot blocks have `#pragma clang fp reassociate(off)` and
`#pragma clang fp contract(off)`. Both dispatches use grid `(32,Dv,B*Hv)` and
threadgroup `(32,4,1)`. The Qwen wrapper supplies FP32 recurrent state.
Source correspondence does not prove JIT numerical equality: the new
`Qwen4GDNStackedKernelParityTests` compares FP16/BF16 outputs, final FP32 state,
and every captured FP32 state against chained canonical kernels for widths
1–6, batch sizes 1/2, and Dk 64/128. It has not run.

Follow-up no-output syntax parsing and `git diff --check` passed. No build,
GPU execution, commit, or push was performed.

### Signed app resources

`Qwen4ExpMetalHeaders` resolves native preambles through `Qwen4ExpMetalResources`. Packaged executables, including symlinked CLI launches, load only from the app’s sealed `Contents/Resources/mlx-swift-lm_MLXLMCommon.bundle/Qwen4Metal` directory. Missing files cannot fall back to a developer checkout or the working directory. `validateResources()` provides a throwing preflight for distribution smoke checks; standalone development still discovers adjacent SwiftPM bundles.
