# Complete-prefix checkpoint retention

> Last updated: 2026-10-09

Complete checkpoints retain computed state under the existing codec, scope,
admission and retirement contracts. Demand does not prove a cache hit, create
an execution profile or supply uncaptured state.

[Authenticated token-byte topology](checkpoint-byte-topology.md) exposes
target K/V byte intervals under the complete manifest. It preserves endpoint
RNN, convolution, index and MTP state, original native-band overlap and the
existing capture/ownership boundaries.

## Captured first, target and latest

`CBv2Request.prefixCheckpointTargetTokens` names an advisory repeated-prefix
length. `CBv2CheckpointRetention(stride: nil, hintTokens:resumedAt:)` retains at
most three committed checkpoints per recurrent or native historical donor:
the first actual capture for a cold donor, the deepest captured frontier at or
below usable demand, and the rolling latest. Roles can share one checkpoint.
Nil/zero demand preserves first/latest, subject to byte pressure. A resumed
donor does not recapture a new interior first; demand at or below its restored
boundary creates no new target. Absent/refused frontiers are never invented.

Native-contiguous and native-paged historical capture still require actual
uniform, query-aligned complete chunk ends below prompt end, the issued native
profile, assistant observation and existing no-preemption/no-pack geometry.
They cannot backdate target or assistant state to an interior position.
Ordinary historical attention instead uses its existing stride and proven
ring residency for supported interior windows. The retention roles are shared;
the capture geometries remain distinct.

`commitHistorical` and `commitContiguousHistorical` pass demand and restore
position to `stageHistorical`. Publication offers deepest, then target, then
first. Store refusal or writer contention can prevent a file; three retained
roles do not guarantee three durable files.

## Bytes and retirement

Reservations precede copy graphs. Contiguous preparation uses
`CBv2ContiguousHistoricalCheckpoint.reservationFootprint` to apply the existing
slot cap before constructing arrays, including native backing and assistant
bytes. Its actual reserved footprint enters the same staging policy as paged
historical windows.

Above admission-capacity/16 the per-donor policy sheds first, then target, keeping
an already admitted latest; this is not a hard /16 limit on that checkpoint
alone. The slot policy uses admission-capacity/8 for staged/publishing payloads
plus net in-flight replacement bytes. Detached retiring copies remain charged
to `AdmissionV2` until completion even when outside those policy counters.
Neither ratio measures physical peak or RSS. Rolling a latest can temporarily
own both the replacement and retiring copy.

`retireCaptured` keeps displaced captures with their native completion owner.
Export aliases retain backing until the batch finishes and consumers drop.
Failed required native completion retains actual owners and charges; it cannot
become a successful refund or ordinary capacity refusal. A demanded target can
add a retained copy and file with real cost; the count stays within three and
transient, full-load and assistant grants stay unchanged.

## Optional demanded range creation

Retention selects captured frontiers. A separate qualified recurrent COMPLETE
policy can create one aligned interior endpoint:

| `CBv2SchedulerConfig` field | Default | Meaning |
|---|---|---|
| `demandedShortCheckpointMinimumTokens` | `nil` | Arm short-prompt capture using the caller's unchanged effective-token floor. |
| `demandedCheckpointPartitionIncludesLongPrompts` | `false` | Explicit longer-prompt qualification extension; does not qualify an architecture. |

The SDK exposes the long flag with default `false`; the companion provider
uses it only in typed benchmark qualification construction. This mechanism
adds no production long-prompt eligibility. Architecture and artifact
qualification belongs to the provider, not a model-family rule in this SDK.

Actual scheduling and deadline projection share `demandedCheckpointRange`.
The count-only `demandedShortCheckpointChunk` wrapper preserves existing short
callers. The target floors `min(hint, promptLength - 1)` to compatible
hash/query alignment. Zero demand, cache-disabled/unscoped input, media/position
state, adopters, preemption and permanent capture disarm keep ordinary geometry.
Native historical capture is outside this recurrent opt-in and keeps its
exact-frontier rules.

For qualified longer prompts, the policy remembers the **actual proposed
range**, rather than deriving a global stripe grid. If a range `12288..<16384`
is split at demand `14336`, its request-local continuation records the start,
target, original end and armed solo stripe. The next compatible range reaches
`16384`, preserving the later ordinary endpoint. A 16,513-token donor with a
4,096-token solo stripe therefore uses these prefill counts:

| Policy | Counts |
|---|---|
| Ordinary | `4096, 4096, 4096, 4096, 129` |
| Qualified long partition | `4096, 4096, 4096, 2048, 2048, 129` |

One actual range becomes target plus its original end, adding at most one
prefill range under compatible uninterrupted geometry. This creates no state
at an uncomputed position. An original end equal to prompt end remains a
terminal range; it does not become an exportable interior checkpoint.

Only the final assigned range commits continuation state. A reservation
fallback or actual incompatible unarmed work clears it; pause or a plan with
no assigned work preserves it. Rollback restores the exact previous optional
state. Existing adoption, preemption and monotonic capture-disarm predicates
remain authoritative, so old carry cannot re-arm a refused donor. Under changed
resource/cohort geometry the endpoint may be lost instead of being forced.

Projection copies the same state, accounts for an in-flight optimistic token
cursor and charges the target and residual once. The extra projected step is
charged without raising a stripe, token budget, capacity cap or capture byte
grant. Additional captured state and writes remain charged to their actual
owners. Only actually shortened target/residual ranges avoid packed
prefill; ordinary packing is unchanged. Future ordinary packing can still
cause conservative forecast overprediction, as in the existing projection
contract; this is not a full future-packing simulator.

The actual recurrent capture/retention contract can then select first, target
and latest from the computed frontiers. A host geometry test proves selection,
not encrypted archive publication or model parity. A target below the first ordinary stripe
can become the first actual capture. Writer/refusal and admitted-owner bounds above still apply.

## Focused validation

From the SDK root, six retention/refusal/bookkeeping, three partition/projection
and seven continuation methods are XCTest host witnesses. The continuation
methods cover actual-origin endpoints, rollback, in-flight projection, real
capacity fallback, pause versus incompatible work, preemption and disarm.
After building and staging the matching Metal library below, run:

```sh
scripts/run-nested-suite.sh \
  'CBv2NativeHistoricalRetentionTests|CBv2DemandedCheckpointPartitionTests|CBv2DemandedCheckpointContinuationTests' \
  --disable-automatic-resolution --no-parallel
```

These host witnesses do not qualify a native codec, actual model output or
speed. The actual native-contiguous selector requires an existing external
strict fixture containing `provenance.json` and `tiny-bf16/`: context at least
1,024, vocabulary at least 32, hidden size at most 64, at most four layers and
the declared three MTP heads. Preserve actual asymmetric widths/dtypes and
default 128-token query geometry; do not fabricate a paged profile or override
query geometry. Reserve an exclusive Metal lane.

Use the pinned SDK toolchain or the recorded system Apple Swift 6.3.3
comparison toolchain, with a source-matched metallib. Build before the host or
native selectors. From the SDK root, its offline helper can create the bounded
asymmetric fixture; choose fresh temporary output paths:

```sh
/usr/bin/swift build --build-tests --disable-automatic-resolution -j 4
scripts/build-metallib.sh /tmp/checkpoint-qualification.metallib
scripts/stage-test-metallib.sh /tmp/checkpoint-qualification.metallib \
  "$(/usr/bin/swift build --show-bin-path)"
python3 scripts/prepare-mimo-native-retention-fixtures.py \
  --output /tmp/mimo-native-retention --asymmetric
MIMO_V26_SERIAL_NATIVE_TESTS=1 \
MIMO_V26_SERIAL_LOAD_FIXTURES=/tmp/mimo-native-retention \
scripts/run-nested-suite.sh \
  MiMoV26NativeHistoricalRetentionTests.testDemandedMiddleFrontierSurvivesNativePublicationAndReopenedFork \
  --disable-automatic-resolution --no-parallel
```

When vendored at `libs/mlx-swift-lm`, first verify that the SDK's actual Cmlx
checkout matches the companion builder's source before using the companion's
`../../scripts/stage-test-metallib.sh "$PWD/.build/arm64-apple-macosx/debug"`
helper. The SDK-local staging script takes two arguments as shown above; the
companion helper takes one. Neither substitutes a metallib from another source
revision.

The test captures a 897-token donor with hint 512, publishes actual frontiers
896/512/128, retires the donor and reopens encoded state for a 513-token fork.
It checks 512 saved tokens, zero replay and exact 24-token greedy continuation
against independent cold execution and reservation drain. Both MTP modes run:
every completed on-engine donor, warm fork and cold reference must execute
positive serial verification rounds and drafts; off has no MTP metrics. This
transport does not qualify provider encryption, SSD durability, full-model TPS
or native-paged runtime.

The closed source-matched original A4 control retained 896/128 and failed
its one owning MTP-off XCTest with two expected missing-512 assertions. The
first iteration throws at the absent archive; this establishes no original
MTP-on or warm-import behavior. Preserve its source/compiler/image/metallib
receipts. The earlier disk-aborted build attempts remain separate and are not
native before results.

The fork CI runs the retention6, partition3, continuation7 and native1 XCTest selectors
through the nonempty/no-skip wrapper after successful build/Metal staging. Its
MiMo opt-in applies only to the dedicated native selector. The generic full
suite may skip optional native/reference-hardware cases; those skips do not
replace the explicit required native witness. Keep actual source, compiler,
image, Metal, test counts and failures bound to each cut. None of these small
fixtures qualifies full-model long output, encrypted SSD cost or serving
promotion.

Source: [retention](../Libraries/MLXLMCommon/ContinuousBatchingV2/Prefix/CheckpointRetention.swift),
[staging](../Libraries/MLXLMCommon/ContinuousBatchingV2/Prefix/HistoricalCheckpointStaging.swift),
[native/historical capture](../Libraries/MLXLMCommon/ContinuousBatchingV2/Prefix/EngineLoopV2+HistoricalCheckpoint.swift),
[range policy](../Libraries/MLXLMCommon/ContinuousBatchingV2/Prefix/DemandedShortCheckpointRange.swift)
and [actual-range continuation](../Libraries/MLXLMCommon/ContinuousBatchingV2/Prefix/DemandedCheckpointContinuation.swift).
