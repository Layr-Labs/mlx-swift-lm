# Qwen3.6 A3B output-parity mode design

Date: 2026-08-25

## Scope

Add an explicit benchmark CLI choice between the highest-throughput MTP
verification path and byte-exact agreement with the stock serial target path:

```text
--output-parity fast|byte-exact
```

The default is `fast`, preserving the optimized benchmark's current behavior.
The option applies only to decode-capable `decode` and `full` runs with an MTP
depth greater than zero.

This change does not alter the model implementation, kernel geometry, cache
layout, prefill route, or the arithmetic of either verification mode. It only
selects one of two already-defined MTP verification routes at construction.

## Interface and behavior

`fast` constructs `CBv2MTPConfig` with `.rectangular`. Its stable receipt label
is `rectangular-target-authoritative`. This is the optimized route used for the
published decode result.

`byte-exact` constructs `CBv2MTPConfig` with `.serialTarget`. Its stable receipt
label is `serial-byte-exact`. This route preserves the stock target evaluation
order and is intended for byte-for-byte output comparisons, with the known
throughput cost made explicit.

The parsed option is a typed enum. That enum is the single source of truth for
both the `CBv2MTPVerificationMode` selected during benchmark construction and
the label written to receipts. The enabled execution path contains no parity
mode checks, environment reads, eligibility checks, or fallback branches.

## Validation

Validation occurs once after CLI parsing and before model or benchmark
construction.

The command fails clearly when:

- `--output-parity` has a value other than `fast` or `byte-exact`.
- The flag is supplied for a `stock` or `prefill` run.
- The flag is supplied with an MTP depth of zero.

Omitting the flag selects `fast`. An explicitly requested `byte-exact` mode
must never downgrade to `fast`; if the serial route cannot be constructed, the
run fails before measurement.

## Data flow

1. The CLI parser converts the argument to the typed output-parity enum.
2. Construction-time validation checks the benchmark phase and MTP depth.
3. `campaignMTPConfig` maps the enum directly to `.rectangular` or
   `.serialTarget`.
4. The selected enum and stable route label are recorded in JSON output and
   the Markdown benchmark summary.
5. Decode executes the installed route directly. Only values that genuinely
   vary at runtime, such as logical M, remain runtime routing inputs.

## Receipt schema

Every decode-capable optimized receipt records:

- `outputParity`: `fast` or `byte-exact`
- `verificationRoute`: `rectangular-target-authoritative` or
  `serial-byte-exact`

The Markdown report displays the same values beside the decode result so the
performance and parity contract cannot be separated from the reported number.

## Tests

Tests are written before implementation and cover:

- Parsing both accepted values and rejecting an unknown value.
- Defaulting to `fast` when the flag is omitted.
- Rejecting `stock`, `prefill`, and MTP-depth-zero combinations when the flag
  is supplied.
- Mapping `fast` to `.rectangular` and `byte-exact` to `.serialTarget`.
- Emitting matching JSON and Markdown route labels from the selected enum.
- Preserving the existing structural guarantee that installed optimized hot
  paths contain no invariant validation or silent stock fallback.

After the focused test gates pass, one locked-GPU smoke benchmark exercises
each mode on the exact pinned model artifact. The byte-exact run must reproduce
the stock token stream; the fast run is evaluated under its declared
target-authoritative contract.

## Failure modes and safeguards

- **Receipt labels diverge from execution:** derive both the verification mode
  and receipt label from the same enum rather than maintaining parallel flags.
- **Byte-exact silently uses the fast route:** install `.serialTarget` directly
  at construction and assert that mapping in tests; do not provide a runtime
  fallback.
- **A meaningless phase reports a parity mode:** reject unsupported phase/depth
  combinations before benchmark construction.
- **Parity selection adds hot-path overhead:** keep selection entirely at
  construction time and pass the already-specialized configuration into the
  measured path.

## Non-goals

- Making `byte-exact` as fast as the rectangular route in this change.
- Changing output semantics of the existing `fast` route.
- Applying this option to stock-only or prefill-only measurements.
- Reworking Qwen3.6 A3B kernels, MTP scheduling, or prefill optimization.
