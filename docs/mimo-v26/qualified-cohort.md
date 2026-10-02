# Selected MiMo component execution record — 2026-09-28

This is an audited selected cohort, not the full SDK suite, provider API or
full-checkpoint qualification. Product14 source fingerprint:
`bc1f135ef562a131a61ad4cf6d9163d578924d027ed2eef112dd447d6477724e`.
The fingerprint is not a signed Git commit.

89 selected methods passed,0 failed,0 skipped, each in a fresh process.
104 methods were discovered;15 were unselected, not skipped or passed.
Six retained-fault selectors were isolated; injected failures are not physical
backend-fault proof. The source/object/resource closure and terminal process
drain were independently audited.

## Corrected test inputs

The passing runner used these test-only afterimages, rather than the original
unmodified checkout tests:

| Test source | SHA-256 |
|---|---|
| CBv2NativeCompletePrefixOwnerTests.swift | `d69fe691649cb0fee95381a17228f4be3b9e0e2e034e0ba5ec79801a0492694d` |
| MiMoV26NativePrefixProducerTests.swift | `ba28dd9059fc9a5e0a078eb2cc121a4e46a4038a422107ffe20497f0a5be14d9` |
| CBv2NativeCompleteCheckpointTransferTests.swift | `7edba7497520b79cb59423434c2c211a50e60bcd622f65cea377023026085e75` |

Changes capture the real Sendable engine, use a validated nonescaping native
codec borrow, release the final foreign-stage host handle at the correct
lifetime boundary, supply a genuinely asymmetric positive fixture and observe
the real held completion gate before collecting its terminal result.
Original oracles were retained. A weak MLXArray/row-before-credit regression
is included. Publishing tests without these afterimages does not reproduce89.

The strict positive fixture is fully synthetic:193 tensors, four shards,
K64/V32,251,644 tensor bytes,298,445 total file bytes. It is not selected model
weight data and does not qualify paging's distinct Q64/K192/V128 geometry.
`MIMO_V26_SERIAL_LOAD_FIXTURES` must contain the original tiny-bf16 fixtures
and the additional tiny-asymmetric-bf16 directory/provenance. Native selectors
require their explicit environment gates and source-matched resources; missing
fixtures are not a pass. A public portable fixture/selector distribution remains
a separate reproduction task until those exact inputs are included.

## Retained failures and limits

Preserve the earlier non-Sendable test compile failure, same-engine-queue trap,
legacy-only codec getter trap, final host-metadata/fake-owner lifetime trap,
ineligible equal-width positive fixture and held-gate assertion-order failure.
Corrections and the final pass do not erase those observations.

Completed matched full-model OFF/serial-MTP output IDs agree; serial MTP was
slower. Earlier assistant-ownership failures remain in the history. Fast
rectangular MTP remains unqualified: the full-model mismatch at output token38
and raw K-projection shape attribution are retained. No newer scalar-dense or
native paging implementation is covered by this binary.

Still open: final public composition/full suite, full-model state and rollback,
actual managed media/API/coordinator flows, native paging activation, encrypted
restart and concurrent lifecycle, matched performance and final end-to-end gates.
