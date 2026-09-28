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
