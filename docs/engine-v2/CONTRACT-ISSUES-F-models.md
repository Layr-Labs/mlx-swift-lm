# Contract issues — WS-F (models)

Places where `CBv2Contracts.swift` was insufficient for the F-models
deliverables, and the conforming shape chosen. For integration to resolve.

## 1. No factory API for constructing `CBv2AttendingLayerCache` instances

The spec asks for `newCacheV2(backend:)` producing `[CBv2AttendingLayerCache]`,
but the contract's `CBv2KVBackend` only creates **per-request** sequence state
(`[CBv2SequenceKV?]`); there is no contract-level way to construct the
batch-facing per-layer cache objects (WS-A's `LayerCacheV2`) from a backend.

**Shape chosen:** both models expose

```swift
func newCacheV2(
    makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
        any CBv2AttendingLayerCache
) rethrows -> [any CBv2AttendingLayerCache]
```

plus `var cbv2LayerKinds: [CBv2LayerKind]`. Integration (WS-B engine build)
supplies the closure wrapping the concrete WS-A layer-cache class and the
chosen backend. This keeps the model layer typed purely against the contract.

## 2. `positionOffsets` pre-update snapshot semantics are unstated

The contract exposes `CBv2AttendingLayerCache.positionOffsets` but does not
state whether the array observed by the model is stable across
`updateAndAttend` (which advances the rows within the same step). The models
assume: **`positionOffsets` read before `updateAndAttend` reflects the
pre-update absolute offsets**, and defensively snapshot with `+ 0` (graph-safe
copy, same convention as `gemma4CapturePositionOffset`) before calling
`updateAndAttend`. KV-shared Gemma layers reuse the source layer's snapshot —
threaded through the trunk — and never re-read `source.positionOffsets` after
the source's update. WS-A's implementation must not invalidate this (e.g. by
mutating the returned array in place before the step's graph is built).

## 3. No load-time preparation hook for models

GPT-OSS needs a one-time host probe (`(sinks * sinks).max().item()`) to decide
whether the learned sinks are active. The contract has no "engine will call X
at model load" hook, so this is folded into `GPTOSSModel.newCacheV2`, which
primes the per-layer Bool before returning caches. If the engine ever builds
layer caches without going through `newCacheV2`, the first forward would fall
back to a lazily-cached probe (one `.item()` per layer, once per process — not
per request, but still on a step path). Integration should keep `newCacheV2`
as the single cache-construction entry point, or add an explicit
`prepareForEngine()` hook to the contract.
