# Calibrated first-content deadline admission

The optional `CBv2FirstContentCalibration` policy lets an integrator supply
reviewed prediction-error bounds without changing the engine's absolute-clock,
memory or retirement contracts. The default remains the caller's existing
conservative prefill and decode rates.

## Qualification and matching

The integrator owns model, artifact, runtime, hardware, template and MTP identity,
measurement freshness, and independent validation of the prediction envelope.
The SDK does not certify those facts or infer a profile from live throughput.
Each cell bounds prompt/context sizes, actual cold or reused prefix state,
contention, work tokens, total request count, exact competing profile IDs and
their service fraction. Matching occurs on the serialized engine queue after
prefix metadata establishes the actual replay start and the scheduler projects
work through the incoming request's first token.

The calibrated work bound includes service owners that may be preparing or
retiring outside the scheduler's current rows:

```text
prefill = max(scheduled_prefill, existing_same_model_prefill + incoming_after_reuse)
          + existing_other_model_prefill
decode  = max(scheduled_decode, existing_same_model_decode)
          + min(incoming_max_output, first_content_decode_allowance)
          + existing_other_model_decode
seconds = (prefill / prefill_tps + decode / decode_tps) * error_ratio
          + error_additive_milliseconds / 1000
```

`firstContentDecodeAllowance` defaults to 33. The incoming full output limit
does not become a TTFT completion requirement. Existing owners retain their
conservative full-work bounds until actual retirement. If multiple cells match,
the largest bound wins. Memory projection must still succeed before either
calibrated or ordinary rate conversion.

The owner invalidates `CBv2FirstContentEvidenceGuard` when shared-machine
ownership changes. The engine rechecks this token and `validUntil` while matching
the actual workload. Missing, unsupported, expired or invalidated calibration
falls back to the original phase rates; a missing required fallback rate remains
unbounded. Neither path rewrites the absolute deadline. Cancellation and
`CBv2RequestRetirement` ownership are unchanged.

Implementation: `DeadlineAdmission/CalibratedFirstContentV2.swift` and
`DeadlineAdmission/EngineLoopV2+DeadlineProjection.swift` under
`Libraries/MLXLMCommon/ContinuousBatchingV2/`.

## Native-media target observations

`CBv2FirstTokenDeadlineAdmission.nativeTargetPrefill` is a separate opt-in for
already prepared, owner-sealed causal native media. Its observed rate prices
only this cold target's prompt. Existing queued prefill retains
`conservativePrefillTokensPerSecond`, and decode retains its own phase rate.
The supplied observation must match the actual prompt range, remain fresh,
and hold a live `CBv2FirstContentEvidenceGuard`. Invalid or missing native
evidence is unbounded; it does not silently substitute the text rate.
The caller must bound the observed range and fence hardware/posture and
whole-machine activity changes. An observation is not a certified error bound.

A caller can instead supply a `CBv2NativeMediaBootstrap` for bounded evidence
acquisition. The engine accepts it only for a validated cold native target,
after physical-capacity projection succeeds, when it is the sole scheduler
row and there is no in-flight step, decode work, mixed step or prefix reuse.
The exact prompt count, shared-machine guard, freshness and original absolute
deadline are checked on the engine queue. The caller must permit at most one
bootstrap until real retirement and rate-limit retries, and must continue
enforcing the original first-content timer and cancellation.

Accepted bootstrap work returns `.unmeasuredNativeMedia(work:)`, rather than a
fabricated zero-duration prediction. This is a source addition to the public
projected-work enum; exhaustive consumers must handle that case. Busy,
expired, cancelled, malformed, unowned and physically infeasible requests keep
the normal refusal and retirement paths. No default caller opts into this.

Implementation: `DeadlineAdmission/NativeTargetPrefillRate.swift` and
`DeadlineAdmission/EngineLoopV2+DeadlineProjection.swift`.

## Opt-in qualification timing

`EngineV2.beginForwardShapeObservation()` enables bounded, scalar-only receipts.
`CBv2ForwardShapeSnapshot.completedStepTimings` records the existing interval
from launch to confirmed readback, classified as prefill, decode or mixed.
MTP verification contributes to decode classification. These are step wall
times, not isolated kernel timings.

Each scope retains at most 8192 completed-step receipts. A nonzero
`droppedStepTimings` makes full-run percentile qualification incomplete. Reset
clears receipts; abandonment never asserts completion. The ordinary unobserved
path retains no recorder or step object and allocates no timing array. Recording
adds no clock read, tensor readback or evaluation. The optional JSON fields are
absent from disabled snapshots and decode as absent in older receipts.

## Compatibility and validation

`CBv2FirstTokenDeadlineAdmission` adds a defaulted optional `calibration`
argument. Existing source callers and their rate-based decisions remain valid.
Consumers rebuilding against this fork can opt into the new data types; this
change makes no binary-library ABI guarantee.

Focused regression suites are `CBv2CalibratedFirstContentTests`,
`CBv2FirstTokenDeadlineEngineTests`, and `CBv2ForwardShapeTests`. They cover
unchanged deadlines, stale fallback, work ownership beyond scheduler rows,
cache/context matching, overlapping cells, bounded receipt storage and the
disabled observer path.

## Unbounded projection diagnostics

`CBv2FirstTokenProjectedWork.unbounded(reason:)` carries the first failed guard
as a content-free `CBv2FirstTokenUnboundedReason`. Scheduler projection,
capacity proof and duration conversion preserve that reason in
`CBv2FirstTokenDeadlineResult.deadlineUnreachable`. The verdict still fails
closed, uses the original deadline and retires the same request resources.
No reason implies that a rejected request would have completed on time.

The public reason is optional for test doubles and adapters without reason
evidence. Such callers construct `.unbounded()`; switch patterns may ignore
the payload with `case .unbounded`. Production scheduler/engine paths supply a
reason. This source API addition requires callers that constructed the old bare
`.unbounded` value to add parentheses; it makes no binary ABI guarantee.

| Reason | First failed guard |
|---|---|
| `unsupported_scheduler` | Serialized-prefill projection is unsupported by the configuration or engine implementation. |
| `target_missing` | The incoming request has no scheduler record. |
| `invalid_in_flight_assignment` | A launched assignment has invalid counts or no matching projected work. |
| `inconsistent_token_cursor` | Confirmed token cursors cannot be reconstructed from launched work. |
| `unowned_pending_sample` | A pending sample has no owning launched assignment. |
| `multimodal_work` | Text-token counts do not bound the multimodal work. |
| `invalid_prefix_reservation` | Prefix preview or projected capacity reservation cannot be represented. |
| `invalid_projection_assignment` | A projected step assignment cannot be represented. |
| `invalid_projection_transition` | A projected row advance or finalization cannot be represented. |
| `projection_arithmetic` | Work or step accounting cannot be safely accumulated. |
| `chained_step_unprojectable` | The terminal chained-decode successor cannot be represented. |
| `iteration_limit` | Projection exceeds its existing 32,768-iteration guard. |
| `prefix_geometry_blocked` | Prefix geometry would require an unpriced cold restart. |
| `speculation_bound_missing` | Speculation is enabled without its draft-token upper bound. |
| `no_scheduling_progress` | No row can advance in the projected scheduler state, including paused-slot blockage. |
| `target_not_sampled` | The target leaves projected rows without a first sample. |
| `invalid_work_totals` | The projected work totals violate phase accounting invariants. |
| `capacity_model_unsupported` | The installed capacity implementation cannot prove projected operations. |
| `capacity_not_guaranteed` | The existing memory/capacity proof refuses projected operations. |
| `prefill_rate_unavailable` | Conversion needs a positive finite prefill rate and has none. |
| `decode_rate_unavailable` | Conversion needs a positive finite decode rate and has none. |
| `service_duration_invalid` | Conversion overflows or otherwise produces an invalid service duration. |
| `service_duration_underflow` | Positive work rounds to a zero duration. |

Scheduler reasons take precedence over capacity and conversion reasons.
Capacity is checked before timing conversion. Rate reasons are emitted only if
neither calibration nor the existing phase-rate fallback produces a duration;
if both phase rates are missing, prefill is reported first. Several low-level
invariants share a reason, so a reason identifies the failing guard family,
not the ultimate cause of that invalid state. Regression coverage lives in
`CBv2UnboundedReasonTests` and the existing first-token admission suites.
