# Gemma catch-up host contracts

Run `python3 -B scripts/test_gemma_host_contracts.py` from the repository root.
The command compiles the actual `LayerCacheBankV2.swift` and
`MTPSampledTokenAssembly.swift` with minimal CPU-only interface doubles, plus
the Foundation-only dense/prefill admission helpers and forward-local
`Gemma4PrefillNormalizationCarry.swift`.
It tests both unoptimized and optimized Swift builds without importing MLX,
resolving packages, loading weights or initializing a GPU.

Twenty-one scenarios cover unchanged membership, row replacement/reorder/join/shrink,
invalidation, release/readmission, borrowed layers, independent banks, weak-row
retirement, all-decode pointer reuse without slice/concatenate calls, and mixed,
prefill-only and empty sample ordering, default-off B1/Q8 admission, and exact
language namespaces with complete six-parameter binding. Repeated checks are assertions,
not separate model tests or a performance benchmark.

Prefill cases cover the explicit scheduled-prefill/model gate (not rectangle
shape alone), BF16/epsilon/weight geometry, Int32 grid/host multiplication
overflow, conservative nonblocking vector-alignment admission, and one-shot
normalization handoff identity/retirement. They do not execute Metal or prove
that the model's full forward uses the intended route.

The separately default-off prefix/scatter flags and eight-slot sorted-scatter
extent checks are also covered. These host checks do not run the GPU sort or
validate expert projection results; native tests own those obligations.

The real `Gemma4DeferredExpertState` is also compiled: failed eligibility peeks
do not consume it, materialization calls its resolver once, fusion consumes it
once, and pending inputs retire. These are host-state assertions, not GPU buffer
lifetime or numerical proof. The separately default-OFF expert-tail flag cannot
enable scatter or a deferred tail when chaining is disabled.

GeGLU policy checks retain the existing compiled-activation opt-out, constrain
actual dense/expert widths and packed offsets, enforce the 1024-row floor and
check grid/shape overflow. Native tests for split/packed planes, scalar/vector
accesses and exhaustive BF16 words are authored separately and are not executed
by the host harness.

Counting-sort tests include a clearly separate CPU specification model for
histogram, serial/eight-range scan, local-rank and bitset scatter. They compare
row order, sorted keys and inverse permutation against stable sort for ties,
tail blocks and large tables. This does not execute Metal or prove native
synchronization; native tests compare the real score-derived producer to MLX.

Decode policy/coverage checks admit only one-token rows, distinguish target and
explicit assistant roles, and validate element/grid bounds and per-row address
coverage. Repeated address assertions are not separate model tests. Native
decode byte equality, parameter/capture-hook invalidation and stream behavior
are authored separately and remain unrun.

Router finalist checks include CPU models of the two-stage bitonic selection
and native-order encoding under both gradual and flushed-subnormal semantics.
They do not establish actual GPU FTZ, subgroup or softmax behavior; those are
separate native byte-equality gates.

Scaled-embedding checks cover explicit target/phase/dtype/quantization and grid
admission. Native tests retain valid negative token wrapping and compare bytes
against the unchanged quantized-embedding API; invalid gather indices are not
redefined or silently clamped.

QKV checks model bank/thread/head-major addressing, odd prefill tails, explicit
phase admission and integer/grid bounds. Millions of address assertions are
not model tests. Native barriers, RoPE bytes, changed rope instances, K=V input
identity and full-stack cache/multimodal behavior remain separate gates.

This is host-policy evidence only. It does not prove actual native cache writes,
sampler-buffer lifetime, tensor dtype/value parity, MTP acceptance, model quality,
Metal execution or throughput. Those gates must run with the full SDK/provider
and unchanged Gemma artifacts before promotion.

`Gemma4PrefillGlueTests.swift` contains separately authorized native synthetic
checks for the four fused operations, byte-exact results, vector/scalar paths,
unaligned/lazy inputs, CPU fallback and parameter changes. These are not run
by this script, and source compilation is not shader qualification.

Provenance: the two execution optimizations adapt
`LayerCacheBankV2.swift` and `MTP/EngineLoopV2+MTPExecution.swift` from
Layr-Labs/mlxfast-gemma4-26b-a4b-engine at
`27c821c466c9799e87162e9436618863b7d0a0ba` onto the current merged engine.
The challenge's process-global mutable cache memoizer, shutdown changes,
controller replacement and numerical-tolerance policy are not included.

The separately opt-in dense gate/up storage and model integration are NOT
compiled by this host-only harness. Their native source tests live in
`Tests/MLXLMTests/Gemma4DenseGateUpStorageTests.swift`; those tests require MLX
and a separately authorized native lane. CPU policy results are not storage
value, descriptor-lifetime, model-output or speed evidence.

B8 expert policy adds exact phase/geometry/opt-out and power-of-two run-cap
checks. These do not execute the paired expert kernels or validate their
272.25 MiB-per-layer extra storage. Native tests are authored separately.

B8 routing checks exercise the actual pure-Swift fused-source generator and
policy, including per-row scratch layout and retention of the current softmax
tape. This does not compile MSL or execute routing/weight kernels on a GPU.

Unified-position checks compile the actual host cycle and cache bank with
interface doubles. They cover ordering, widths, rebinding, invalidation and
release, not native tensor identity, attention or loaded-model behavior.

Compact-root policy checks cover independent decode/verify switches and exact
binding/update/width epochs, including counter wrap. Actual materialization
tests use non-blocking availability checks and require the native lane.
