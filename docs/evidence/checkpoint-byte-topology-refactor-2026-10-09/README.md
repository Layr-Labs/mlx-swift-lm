# Checkpoint byte topology refactor receipts

> Last updated: 2026-10-09

These receipts preserve two separate fixture gates. The first working version
passed 7 XCTest functions and 43 Swift Testing functions. After the dedicated
refactor, the same selector passed 7 XCTest functions and 47 Swift Testing
functions, with zero failures or skips. The four added regressions cover a
malformed legacy owner map, oversized record decoding, metadata alias lifetime,
and auxiliary-only canonical omission and import planning.

The refactor also ran the nine topology unit functions with every metallib
temporarily absent from the SDK build tree. This narrower pass confirms the
new `Unit/Cache` suite does not require Metal. It is distinct from the staged
checkpoint lifecycle pass. No actual model weights or network transfer ran.

## Source and execution provenance

| Receipt | Meaning |
|---|---|
| `before-source.json`, `before-source.patch.gz` | Original first working source hashes and bounded patch from signed SDK parent `aee5d368ffe81ee409a6aa11cded221ca789eb98`. |
| `before-tests.log.gz` | Unchanged first-gate log; it is not the final refactor result. |
| `refactor-source.patch.gz` | Bounded source/test refactor patch applied after the original source patch. |
| `refactor-receipt.json` | Final source and canonical-doc hashes, exact commands, pinned dependency state, executable hash, and staged resource hashes. |
| `refactor-build.log.gz`, `refactor-unit-no-metal.log.gz`, `refactor-focused.log.gz`, `refactor-stage.log.gz` | Separate final build, pure unit, lifecycle and source-matched resource receipts. |

The refactor source archive includes the library README navigation edit but
excludes `docs/`, the fork narrative and receipts; canonical-doc hashes are
recorded separately. Both source patches are gzip-compressed text,
with uncompressed size and SHA-256 recorded. No executable or metallib is
included. Reproduce source in a disposable checkout of the recorded SDK parent
by applying the decompressed first patch, then the decompressed refactor patch.
The recorded Git tree hashes are local source identities, not published commits.

The final build used system Swift 6.4 and the original remote MLX Swift pin
`6923a80f624f5c91fbf456efe4e00e9698a72961`, with native core
`cb77239be31b1df7f5db895226af55c39fc4f093`. No temporary package edit was needed;
tracked package bindings stayed unchanged and no edited dependency remained.
Injected test resources were removed before building and restaged only after
successful completion. Every staged metallib path matched
`fb1ba8f90b9f1cc346246b7240c3986771464bd977c67f0ef03186811f16c85c`.

## Refactor and claim boundaries

Loaded-codec derivation now has a separate source owner from manifest validation.
Validation checks the whole legacy attention-owner map before deriving offsets,
and decoding bounds the new record array before reserving or decoding members.
The positive host allowance covers simultaneous bounded arrays and a provider
index; metadata aliases retain their permit until the last owner releases it.
Auxiliary-only producers preserve omitted topology while exports deliberately
keep the conservative positive allowance. This is an admission envelope, not a
measurement of actual heap usage.

Canonical target streams, coded/native overlap, numerical profiles, windows,
auxiliary state, and package-issued native asymmetric/MiMo omission stay intact.
The fixtures establish byte topology and checkpoint lifecycle behavior. They do
not establish network model quality, latency, transfer performance, or memory
savings. SDK fork CI already selects the new unit suite through `\.UnitTests/`
and the quantized round trips through its packed-storage step; outer provider CI
qualification is a separate gate.

See [the canonical topology contract](../../checkpoint-byte-topology.md) for the
API, positive host accounting and exact lifecycle selector.
