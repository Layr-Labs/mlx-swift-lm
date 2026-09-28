# Composed MiMo component execution — 2026-09-28

The selected composed cohort passed 114 methods, with zero failures and zero
skips. Each method ran in a fresh process; 147 methods were discovered per
process and 33 were unselected. This is not the full SDK suite or a release gate.

## Exact composition

- Runtime SDK: `0f6892c23f95fc6aae0a2d3e4dae5aa7603bddef`.
- Provider: `d06b528e40aaa47ff96169a65d31159d48f0258d`.
- Source fingerprint: `7fa34983a8a7e9efb434a2c25ec99e93644a0c4b252a8d6374aa95f38206bb4c`.
- Test-only amendment: `cce1032375217e1cb060455cfe08e3475cef1199`.
- Test executable SHA-256: `a981685254d510e136f841cb06d171cbb0c4781dd6cc62c8ace5112656161416`.
- Audited result SHA-256: `1f2a92129f184b1da4f88f8733dde7a80422926772765774d9fac1fc29588ba0`.

The optimized/testable runtime integrates the upstream cache-retention changes.
The test amendment changes one stale expected publication order to the actual
deepest-first contract. All runtime library objects are unchanged by that
amendment; this is not a rebuilt-runtime claim. The original run's 60 passes and
one ordering assertion failure remain preserved. The successor reran all 114.

## Executed scope

| Selected group | Methods | Evidence boundary |
|---|---:|---|
| Original cache/ownership cohort | 89 | Re-executed against the integrated runtime with the earlier corrections committed |
| Scalar-dense verification | 12 | Metadata, numerical and genuinely admitted tiny-engine checks; not complete full-model KV/rollback proof |
| Native paging | 9 | SDK owner, strict-loaded asymmetric fixture and retained-fault checks; not provider-wide factory qualification |
| Key-range planning | 4 | Host descriptor/admission arithmetic; not M5 kernel execution |

Inputs, copied fixtures, 36 dependencies, executable/resources and terminal
process identities were independently audited. Compilation used release
optimization and testability, not DEBUG. The original three corrected test
sources from the [earlier cohort](qualified-cohort.md) are now committed.

The synthetic paging fixture has Q64/K192/V128, full/window KV-head counts 4/8,
window128 and context512. It is SDK-only and carries truthful synthetic
provenance; it cannot substitute for a provider's authenticated artifact.
The scalar fixture is also synthetic, not the selected published checkpoint.

## Still open

- Full SDK/provider suites and their applicable opt-ins.
- Selected-artifact HTTP tools/reasoning, Chat/Responses and connected transport.
- Full target/head KV and actual rejection/rollback state under admitted
  scalar-dense verification; matching tokens/hidden rows alone does not close it.
- Full-artifact media plus text-prefix composition, encrypted restart and
  concurrent lifecycle.
- Real provider paging, and paging combined with MTP, prefix and media.
- Six M5-only key-range owner tests and observed full-model dispatch.
- Matched realistic/agentic performance, final production defaults and all
  end-to-end release gates.

Native paging and scalar-dense verification remain opt-in candidates. Neither
this cohort nor source review promotes a performance default, changes weights,
establishes hosted OpenRouter certification or authorizes deployment.
