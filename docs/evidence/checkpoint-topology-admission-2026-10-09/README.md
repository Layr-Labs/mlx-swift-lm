# Auxiliary-only checkpoint admission regression

> Last updated: 2026-10-09

The new topology envelope was reserved for auxiliary-only checkpoints even
though their producer emits no target K/V records. This changed an existing
export admission boundary. The correction uses one loaded-codec predicate for
record production and all three export reservation paths: existing native
format exclusions still apply, and an owning attention layer must be present.
The positive allowance for actual topology tables is unchanged.

## Red and green implementation gates

The regression calls the actual complete-checkpoint export API for both
contiguous and paged recurrent-only state. At position 32, the original host
metadata budget is 2,359,808 bytes. Both cases fail on the unchanged
`ee1c9c41e55b84690f096ece2efd0c4ed2fc37a0` production code with
`capacityExhausted(needed: 6947328, available: 2359808)`. The extra 4,587,520
bytes are the topology allowance for a table that will not exist.

With the correction, both exports succeed at that same exact budget, omit the
topology field, preserve the convolution and recurrent bytes, reject an extra
one-byte reservation, and release the metadata permit. The focused gate passes
7 XCTest functions plus 48 Swift Testing functions across 12 suites, with zero
failures or skips. The two new export cases are arguments of one test function.

These are implementation gates, recorded before the required independent
refactor review. They do not claim remote CI completion or actual-model
qualification. The earlier topology/refactor and CI-format receipts remain
unchanged in their separate evidence directory.

## Receipts

`implementation-receipt.json` records the baseline and changed source hashes,
the exact commands, log hashes, source patch, test executable and matched
metallib hashes, and package/lease cleanup. The source patch is bounded,
gzip-compressed text applicable to the recorded SDK parent. No executable or
metallib is included.

The `red-*` logs preserve the successful regression build, matched staging and
expected old-code failure. The `green-*` logs preserve the corrected build,
matched staging and focused pass. No model weights, network transfers,
threshold changes or numerical-kernel changes were involved.

The empty attention layout remains ineligible for historical attention export;
the regression covers the two existing recurrent-only export paths. Historical,
quantized, shared-owner and elastic-window behavior remains covered by the
focused selector recorded in the receipt and the
[canonical topology contract](../../checkpoint-byte-topology.md).
