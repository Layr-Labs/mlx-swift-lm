# Native MiMo fast-prefill policy

The existing native-rounding NAX attention and admitted block grouping are
permitted by default. Actual device, dtype, shape, causal visibility and owner
checks still select the path. `DARKBLOOM_MIMO_V26_NAX_ATTENTION=0` rolls back
both NAX attention and its dependent grouping. Independently set
`DARKBLOOM_MIMO_BLOCK_BATCH_PREFILL=0` to keep scalar query-block dispatch.
Unset permits the policy; explicit `1`, `true`, `yes`, `on` and `auto` permit
it, while other explicit values refuse it. M3/non-NAX retains the old kernels.

Kernel controls are latched once per process. An injected provider factory
`environment` dictionary cannot change a previously latched control. Before
native slot assembly, the provider compares every explicitly supplied attention
or grouping value with the exact cached values used by SDK dispatch and throws
`MiMoV26PrefillPolicy.ProcessControlMismatch` on disagreement. Missing injected
keys inherit the actual latches. This is a recoverable construction refusal;
it does not mutate the process environment or silently apply only a chunk-width
change. Set rollback controls at process startup for actual kernel rollback.

The provider's native contiguous factory may request a larger **default** solo
prefill stripe on a >=128GiB NAX host. It requests4096 with supported native
192/128 attention and8192 only when the already requested oversized gather
matches every loaded MXFP4 expert projection. This does not enable the gather
experiment itself. The actual failure-aware native KV probe must show BF16 or
FP16 throughout. Unknown shapes, dtypes and other model families keep their
existing profiles.

Width selection is provisional until the engine prices the wider grouped
workspace, re-resolves any bounded MTP declaration for that width and installs
the budget on the exact native model/backend/cache identities. It tries8192,
then4096 where eligible, then the original width. Any refusal retains the
original scheduler and original MTP/caller/target charges. The successful
grouped charge remains additive; OS, activation and KV reserves are unchanged.
No array allocation or model forward is used to choose this policy.

Hosts may supply `miMoPrefillMemoryBudget` with the minimum useful KV allowance
and the target's fixed sliding-window bytes per request. Before installing a
candidate, EngineV2 prices its complete fixed MTP/grouped workspace for every
configured concurrent request, plus target rings, the actual admission watermark
and that KV allowance. A candidate that only fits one request cannot silently
select a profile advertised for four. Rejected candidates fall through to narrower widths;
if none fits, the original ungrouped scheduler and MTP charges are retained.
Optional rectangular scratch must pass the same budget. This is profile
selection, not a reduction in any request's actual allocation charge.

`EngineV2.admissibleKVBytesCapacity` exposes the live ledger ceiling after the
watermark and external carve for host capacity reporting. A host that re-slices
grants must still reduce concurrency or refuse a new load when a retained
profile's fixed workspace no longer fits. The selected scratch reservation is
immutable; changing the grant does not make it smaller.

Explicit `DARKBLOOM_CBV2_SOLO_PREFILL_STRIPE` values, including0, take precedence;
a query-block override other than128 prevents automatic widening. The native
provider/standalone/benchmark construction paths share the same selector.
Direct SDK callers keep their explicit scheduler envelope unless they pass
`automaticMiMoPrefill: true` to the authentic EngineV2 construction. The flag
requests a policy and cannot assert budget ownership.

Paged and complete-prefix profiles keep their existing chunk envelopes. A
genuine contiguous managed-media engine may use the larger stripe for plain
text requests. Actual media requests retain the previous stripe/plain-chunk
ceiling through the actual request.multimodal field, including causal MiMo inputs whose
bidirectional-block list is empty. Real scheduling and first-token projection
use the same selector. The ceiling publishes only with the successful budget;
all failed candidates retain the original width and original media policy.
All media capabilities, sealed owners and ordinary admission remain intact.
The larger text profile also supports genuine bounded MTP after repricing;
an unknown/unbounded drafter retains the old width.

Grouping preserves the existing128-query visibility slices, at most four per
dispatch. Unsupported tails (including a final1–8 queries), shapes or ownership
fall back. Key-range and split-key experiments remain independently OFF; this
policy does not import online FP32 softmax or head-dimension splitting.

Inspect the actual engine's `groupedPrefillScratchBytes`,
`groupedPrefillInactiveReason`, resolved scheduler width and graph-encoding
counters. Encoding counters are not GPU completion, exactness or speed evidence.
This source policy needs exact native/model/state, memory, cancellation,
MTP/media/paging/prefix, normal API and matched physical-host qualification;
default selection by itself proves none of those gates.
