# Gemma 4 MTP Drafter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Google's Gemma 4 Multi-Token Prediction drafters to `mlx-swift-lm` with greedy-identical output and ≥0.9× the Python reference throughput on M-series 36 GB hardware.

**Architecture:** New opt-in `MLXSpeculative` library housing a stateless `Gemma4AssistantDraftModel` + B=1 and B>1 round-loops. Seven localized edits to `Gemma4Text.swift` add a public position-offset namespace, a `forceSharedKV` switch, a shared-KV capture hook, a pre-head `forwardForMTP`, and a `rollbackSpeculativeCache` method. One new primitive on `BatchKVCache` (`zeroTailPerRow`) powers per-row cache rewind for the batched path.

**Tech Stack:** Swift 6.1, MLX Swift, Swift Testing (`@Test`/`#expect`), Swift Package Manager.

**Spec:** `docs/superpowers/specs/2026-05-05-gemma4-mtp-drafter-design.md`.

**Reference (semantics only — NOT a design to clone):** [Blaizzy/mlx-vlm#1112](https://github.com/Blaizzy/mlx-vlm/pull/1112) at `244f4bb5a3339b180da3d2b276a4bdfcf7670f9f`.

---

## Task Graph Overview

Tasks are ordered so that each commit leaves the build green and a growing set of tests passing. Tasks 1-4 are foundation (no MTP logic yet). Tasks 5-11 are the drafter architecture. Tasks 12-16 are the round-loops. Tasks 17-21 are integration glue + weight loading. Tasks 22-25 are parity tests (integration, local only). Tasks 26-27 are benchmarks + doc.

| # | Component | Unit-testable | Needs real weights |
|---|---|---|---|
| 1 | `Package.swift` edits + empty `MLXSpeculative` target | ✓ | — |
| 2 | `Gemma` namespace + `Gemma4.PositionOffset` public | ✓ | — |
| 3 | `BatchKVCache.zeroTailPerRow(keepLengths:)` | ✓ | — |
| 4 | `Gemma4Attention` `usesSharedKV` guard + `forceSharedKV` | ✓ | — |
| 5 | `Gemma4TextModel.applyLMHead(_:)` helper (refactor) | ✓ | — |
| 6 | `Gemma4SharedKV` + `Gemma4SharedKVCapture` types | ✓ | — |
| 7 | Inner trunk capture hook (edit #6 in spec) | ✓ | — |
| 8 | `Gemma4MTPForward` + `forwardForMTP(_:cache:)` | ✓ | — |
| 9 | `Gemma4AcceptCount` + `rollbackSpeculativeCache` | ✓ | — |
| 10 | `SpeculativeWalk.single` | ✓ | — |
| 11 | `SpeculativeWalk.batched` | ✓ | — |
| 12 | `Gemma4AssistantConfiguration` | ✓ | — |
| 13 | `MaskedEmbedder` | ✓ | — |
| 14 | `DrafterMasks` | ✓ | — |
| 15 | `Gemma4MTPError` | ✓ | — |
| 16 | `Gemma4AssistantDraftModel` (skeleton + init + `bind`/`unbind`) | ✓ | — |
| 17 | `Gemma4AssistantDraftModel.callAsFunction` (forward) | ✓ | — |
| 18 | `Gemma4AssistantDraftModel.sanitize` | ✓ | — |
| 19 | `Gemma4AssistantDraftModel.load(from:using:id:...)` | — | ✓ (stubbed to a fixture dir) |
| 20 | `runGemma4MTPRounds` (B=1) | ✓ | — |
| 21 | `runGemma4MTPRoundsBatched` (B>1) | ✓ | — |
| 22 | `generateGemma4MTP` + `generateGemma4MTPBatched` public entry | ✓ | — |
| 23 | `Gemma4E2BMTPParityTest` (smaller model, faster dev loop) | — | ✓ |
| 24 | `Gemma4E4BMTPParityTest` (acceptance gate) | — | ✓ |
| 25 | `Gemma4_26BA4B_4bit_MTPParityTest` (stretch) | — | ✓ |
| 26 | Benchmark harness + `Gemma4MTPBaseline.swift` + Python oracle run | — | ✓ |
| 27 | `MLXSpeculative/README.md` + top-level README section | — | — |

**Test terminology in this plan:**
- "Unit test" = lives in `Tests/MLXLMTests`, uses `@Test` from Swift Testing, runs under `swift test` in CI (no weights download).
- "Integration test" = lives in `Tests/MLXLMTests`, uses `@Test(.enabled(if: ...))` guard keyed on an env var so it's skipped in CI but runs locally.

**Commit discipline:** One commit per task. Commit message format:
```
<component>: <what changed>

<optional body>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 1: Package.swift edits + empty MLXSpeculative target

**Files:**
- Modify: `Package.swift`
- Create: `Libraries/MLXSpeculative/Placeholder.swift`
- Create: `Libraries/MLXSpeculative/README.md`

**Rationale:** Land the build graph first so every subsequent task can simply add files to an existing target. The placeholder file keeps the target buildable until real code lands.

- [ ] **Step 1.1: Add MLXSpeculative library product**

Edit `Package.swift`. After the existing `.library(name: "IntegrationTestHelpers", ...)` line, add:

```swift
        .library(
            name: "MLXSpeculative",
            targets: ["MLXSpeculative"]),
```

- [ ] **Step 1.2: Add MLXSpeculative target**

In `Package.swift`, after the existing `IntegrationTestHelpers` target (before the `.testTarget(name: "MLXLMTests", ...)` block), add:

```swift
        .target(
            name: "MLXSpeculative",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Libraries/MLXSpeculative",
            exclude: [
                "README.md"
            ]
        ),
```

- [ ] **Step 1.3: Add MLXSpeculative to MLXLMTests dependencies**

In `Package.swift`, locate the `.testTarget(name: "MLXLMTests", ...)` block. Change its `dependencies` array from:

```swift
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
            ],
```

to:

```swift
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXSpeculative",
            ],
```

- [ ] **Step 1.4: Add MLXSpeculative to BenchmarkHelpers dependencies**

In `Package.swift`, locate the `.target(name: "BenchmarkHelpers", ...)` block. Change its `dependencies` array from:

```swift
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
```

to:

```swift
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXSpeculative",
                .product(name: "MLX", package: "mlx-swift"),
            ],
```

- [ ] **Step 1.5: Create placeholder source file**

Create `Libraries/MLXSpeculative/Placeholder.swift`:

```swift
// Copyright © 2026 Apple Inc.
//
// Placeholder so the MLXSpeculative target has at least one Swift file while
// the real implementation is being assembled. Remove once Gemma4MTPError.swift
// lands (Task 15).

enum MLXSpeculative_Placeholder {}
```

- [ ] **Step 1.6: Create library README**

Create `Libraries/MLXSpeculative/README.md`:

```markdown
# MLXSpeculative

Speculative-decoding drafters and round-loops for `mlx-swift-lm`.

v1 ships one drafter: Google's Gemma 4 Multi-Token Prediction "assistant"
drafters (published under `mlx-community/gemma-4-*-it-assistant-bf16`).

See the top-level README for usage.
```

- [ ] **Step 1.7: Build to confirm the graph compiles**

Run: `swift build`

Expected: clean build, no new warnings.

- [ ] **Step 1.8: Commit**

```bash
git add Package.swift Libraries/MLXSpeculative/
git commit -m "$(cat <<'EOF'
MLXSpeculative: scaffold empty library target

New opt-in library that will house the Gemma 4 MTP drafter. Registered
as a product, wired into MLXLMTests and BenchmarkHelpers as a dep,
with a placeholder source file until real code lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Public `Gemma` namespace + promote `Gemma4.PositionOffset`

**Files:**
- Modify: `Libraries/MLXLLM/Models/Gemma4Text.swift` (lines 196-221 block)

**Rationale:** `Gemma4PositionOffset` is currently `private`; the drafter must be able to construct and pass these values into `Gemma4Attention.callAsFunction`. Promoting to `public` via a namespaced `Gemma4.PositionOffset` avoids polluting the top-level symbol table.

- [ ] **Step 2.1: Write the failing test**

Create `Tests/MLXLMTests/Gemma4PositionOffsetTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma4.PositionOffset is public and constructible")
struct Gemma4PositionOffsetTests {

    @Test func scalarCase() {
        let offset = Gemma4.PositionOffset.scalar(42)
        if case .scalar(let v) = offset {
            #expect(v == 42)
        } else {
            Issue.record("expected .scalar case")
        }
    }

    @Test func batchCase() {
        let arr = MLXArray([Int32(0), 1, 2, 3])
        let offset = Gemma4.PositionOffset.batch(arr)
        if case .batch(let a) = offset {
            #expect(a.dim(0) == 4)
        } else {
            Issue.record("expected .batch case")
        }
    }
}
```

- [ ] **Step 2.2: Run the test to verify it fails**

Run: `swift test --filter Gemma4PositionOffsetTests`

Expected: compilation failure. Error messages should mention `Gemma4` being undefined or `PositionOffset` being inaccessible.

- [ ] **Step 2.3: Replace the private enum with a public namespaced one**

In `Libraries/MLXLLM/Models/Gemma4Text.swift`, locate (around line 196):

```swift
private enum Gemma4PositionOffset {
    case scalar(Int)
    case batch(MLXArray)
}

private func gemma4CapturePositionOffset(from cache: KVCache?) -> Gemma4PositionOffset {
    if let batchCache = cache as? BatchPositionedKVCache {
        // Snapshot the per-sequence offsets before cache.update(...) advances them.
        .batch(batchCache.batchOffset + 0)
    } else {
        .scalar(cache?.offset ?? 0)
    }
}

private func gemma4ApplyRotaryPosition<R: RoPELayer>(
    _ rope: R,
    to x: MLXArray,
    offset: Gemma4PositionOffset
) -> MLXArray {
    switch offset {
    case .scalar(let value):
        rope(x, offset: value)
    case .batch(let values):
        rope(x, offset: values)
    }
}
```

Replace with:

```swift
/// Public namespace for Gemma 4 types that need cross-module visibility
/// (e.g. the MTP drafter in `MLXSpeculative`).
public enum Gemma4 {

    /// Position offset for RoPE, either a single scalar (standard decode)
    /// or a per-row `MLXArray` (continuous-batching / drafter paths).
    public enum PositionOffset: Sendable {
        case scalar(Int)
        case batch(MLXArray)
    }
}

@inline(__always)
internal func gemma4CapturePositionOffset(from cache: KVCache?) -> Gemma4.PositionOffset {
    if let batchCache = cache as? BatchPositionedKVCache {
        // Snapshot the per-sequence offsets before cache.update(...) advances them.
        .batch(batchCache.batchOffset + 0)
    } else {
        .scalar(cache?.offset ?? 0)
    }
}

@inline(__always)
internal func gemma4ApplyRotaryPosition<R: RoPELayer>(
    _ rope: R,
    to x: MLXArray,
    offset: Gemma4.PositionOffset
) -> MLXArray {
    switch offset {
    case .scalar(let value):
        rope(x, offset: value)
    case .batch(let values):
        rope(x, offset: values)
    }
}
```

- [ ] **Step 2.4: Update all internal references to the renamed enum**

In the same file, search for the pattern `Gemma4PositionOffset` (the old name). It appears in:

- The return type of `Gemma4Attention.callAsFunction` (the `-> (..., Gemma4PositionOffset)` tuple member). Rename to `Gemma4.PositionOffset`.
- The parameter `positionOffset: Gemma4PositionOffset?` on `Gemma4Attention.callAsFunction`. Rename to `Gemma4.PositionOffset?`.
- The `activePositionOffset` local variable's inferred type is fine; no change.
- The return type of `Gemma4DecoderLayer.callAsFunction`. Rename to `Gemma4.PositionOffset`.
- The parameter `positionOffset: Gemma4PositionOffset?` on `Gemma4DecoderLayer.callAsFunction`. Rename to `Gemma4.PositionOffset?`.
- The `intermediates` array type in `Gemma4TextModelInner.callAsFunction`: `(kv: ..., positionOffset: Gemma4PositionOffset?)`. Rename to `Gemma4.PositionOffset?`.

Use the exact search pattern `Gemma4PositionOffset` (no leading dot) to catch every call site.

- [ ] **Step 2.5: Run the test to verify it passes**

Run: `swift test --filter Gemma4PositionOffsetTests`

Expected: 2 tests pass.

- [ ] **Step 2.6: Run the full test suite to catch regressions**

Run: `swift test`

Expected: all previously-passing tests still pass.

- [ ] **Step 2.7: Commit**

```bash
git add Libraries/MLXLLM/Models/Gemma4Text.swift Tests/MLXLMTests/Gemma4PositionOffsetTests.swift
git commit -m "$(cat <<'EOF'
Gemma4Text: promote PositionOffset to public Gemma4 namespace

Drafter in MLXSpeculative needs to construct per-row offsets and pass
them into Gemma4DecoderLayer. Enum moved under a new public Gemma4
enum-namespace; helper functions kept internal with inline annotations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `BatchKVCache.zeroTailPerRow(keepLengths:)`

**Files:**
- Modify: `Libraries/MLXLMCommon/BatchKVCache.swift` (add an extension method)
- Create: `Tests/MLXLMTests/BatchKVCacheZeroTailTests.swift`

**Rationale:** MTP B>1 rewind path needs to zero per-row tail positions after a uniform trim. This is the only new public API in `MLXLMCommon`.

- [ ] **Step 3.1: Write the failing test**

Create `Tests/MLXLMTests/BatchKVCacheZeroTailTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("BatchKVCache.zeroTailPerRow")
struct BatchKVCacheZeroTailTests {

    /// Build a BatchKVCache with `B` rows, each containing `T` timesteps of
    /// non-zero marker values so zeroing is observable.
    private func makeCache(B: Int, T: Int, H: Int = 2, D: Int = 4) -> BatchKVCache {
        let cache = BatchKVCache(leftPadding: Array(repeating: 0, count: B))
        let k = MLXArray.ones([B, H, T, D], dtype: .float32)
        let v = MLXArray.ones([B, H, T, D], dtype: .float32) * 2.0
        _ = cache.update(keys: k, values: v)
        return cache
    }

    @Test func fullKeepIsNoop() {
        let cache = makeCache(B: 3, T: 8)
        let keep = MLXArray([Int32(8), 8, 8])
        cache.zeroTailPerRow(keepLengths: keep)
        // All slots should remain 1.0 / 2.0.
        let state = cache.state
        let keysSum = state[0].sum().item(Float.self)
        let valuesSum = state[1].sum().item(Float.self)
        // 3 rows × 2 heads × 8 time × 4 dim = 192 slots at 1.0 → sum == 192
        #expect(keysSum == 192.0)
        #expect(valuesSum == 384.0)
    }

    @Test func zeroKeepErasesEverything() {
        let cache = makeCache(B: 3, T: 8)
        let keep = MLXArray([Int32(0), 0, 0])
        cache.zeroTailPerRow(keepLengths: keep)
        let state = cache.state
        #expect(state[0].sum().item(Float.self) == 0.0)
        #expect(state[1].sum().item(Float.self) == 0.0)
    }

    @Test func mixedPerRowKeep() {
        // B=3, T=8, keep=[2, 5, 8]. Row 0 keeps 2 slots, row 1 keeps 5,
        // row 2 keeps all 8.
        let cache = makeCache(B: 3, T: 8, H: 1, D: 1)
        let keep = MLXArray([Int32(2), 5, 8])
        cache.zeroTailPerRow(keepLengths: keep)
        let state = cache.state
        // Expected keys sum: row0=2, row1=5, row2=8 = 15 (value 1.0 each)
        #expect(state[0].sum().item(Float.self) == 15.0)
        // Values sum: 15 * 2.0 = 30
        #expect(state[1].sum().item(Float.self) == 30.0)
    }

    @Test func mixedKeepLeavesKeptRegionUntouched() {
        // Verify row 1 still has its first 5 values unchanged.
        let cache = makeCache(B: 3, T: 8, H: 1, D: 1)
        let keep = MLXArray([Int32(2), 5, 8])
        cache.zeroTailPerRow(keepLengths: keep)
        let keys = cache.state[0]  // [3, 1, 8, 1]
        // Row 1, first 5 slots, all ones.
        let row1Head = keys[1, 0, ..<5, 0]
        #expect(row1Head.sum().item(Float.self) == 5.0)
        // Row 1, slots 5..<8, all zeros.
        let row1Tail = keys[1, 0, 5..., 0]
        #expect(row1Tail.sum().item(Float.self) == 0.0)
    }

    @Test func emptyCacheIsNoop() {
        // No update() called; keys/values are nil.
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.zeroTailPerRow(keepLengths: MLXArray([Int32(0), 0]))
        // No crash; state should still report the metadata-only shape.
        #expect(cache.isEmpty())
    }
}
```

- [ ] **Step 3.2: Run the test to verify it fails**

Run: `swift test --filter BatchKVCacheZeroTailTests`

Expected: compilation failure with message mentioning `zeroTailPerRow` not found.

- [ ] **Step 3.3: Implement `zeroTailPerRow`**

In `Libraries/MLXLMCommon/BatchKVCache.swift`, at the end of the file (after the `dynamicRoll` helper, before EOF), add:

```swift
// MARK: - Speculative-decoding primitives

extension BatchKVCache {

    /// Zero per-row tail positions. For each row `b`, slots
    /// `[keepLengths[b], _idx)` in both keys and values are set to 0.
    /// Rows where `keepLengths[b] >= _idx` are left unchanged.
    ///
    /// Used by the Gemma 4 MTP round-loop to clear rejected-tail mismatches
    /// when rows accepted different numbers of tokens within the same
    /// speculative block.
    ///
    /// No-op if the cache is empty (no `update` has been called yet).
    ///
    /// - Parameter keepLengths: int array of shape `[B]`. Values should
    ///   satisfy `0 <= keepLengths[b] <= _idx` for all `b`; out-of-range
    ///   values just fall back to "no zeroing" (keep >= T) or "zero all"
    ///   (keep <= 0).
    public func zeroTailPerRow(keepLengths: MLXArray) {
        guard let storedK = keys, let storedV = values else { return }
        let T = storedK.dim(2)
        let positions = MLXArray(Int32(0) ..< Int32(T))
            .reshaped([1, 1, T, 1])                          // [1, 1, T, 1]
        let keep = keepLengths.asType(.int32)
            .reshaped([-1, 1, 1, 1])                         // [B, 1, 1, 1]
        let keepMask = positions .< keep                     // [B, 1, T, 1] bool
        let maskFloat = keepMask.asType(storedK.dtype)
        keys = storedK * maskFloat
        values = storedV * maskFloat
    }
}
```

- [ ] **Step 3.4: Run the test to verify it passes**

Run: `swift test --filter BatchKVCacheZeroTailTests`

Expected: 5 tests pass.

- [ ] **Step 3.5: Run the full test suite to confirm no regression**

Run: `swift test`

Expected: all tests pass.

- [ ] **Step 3.6: Commit**

```bash
git add Libraries/MLXLMCommon/BatchKVCache.swift Tests/MLXLMTests/BatchKVCacheZeroTailTests.swift
git commit -m "$(cat <<'EOF'
BatchKVCache: add zeroTailPerRow(keepLengths:)

Per-row tail-zero primitive used by the Gemma 4 MTP round-loop to
rewind rows that accepted fewer tokens than the round's max. GPU-only
broadcast; empty-cache / out-of-range keep values are safe no-ops.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `Gemma4Attention` — fix `usesSharedKV` guard and add `forceSharedKV`

**Files:**
- Modify: `Libraries/MLXLLM/Models/Gemma4Text.swift` (lines 248-302 approx: `Gemma4Attention.init`, `Gemma4DecoderLayer.init`, `Gemma4TextModelInner.init`)
- Modify: `Tests/MLXLMTests/Gemma4ForceSharedKVTests.swift` (new)

**Rationale:** Current guard `firstKvSharedLayerIdx > 0` wrongly marks drafter layers as non-shared when `numKvSharedLayers == numHiddenLayers`. Fix the predicate, and add a `forceSharedKV` constructor switch as defense-in-depth.

- [ ] **Step 4.1: Write the failing test**

Create `Tests/MLXLMTests/Gemma4ForceSharedKVTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma4 shared-KV layer selection")
struct Gemma4ForceSharedKVTests {

    /// Build a 4-layer Gemma4TextConfiguration with every layer shared —
    /// the drafter shape.
    private func drafterLikeConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 2048,
            "num_attention_heads": 4,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 4,
            "layer_types": ["sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
            "sliding_window": 512,
            "final_logit_softcapping": null,
            "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    /// Build a 35-layer Gemma4TextConfiguration with the last 20 shared —
    /// the 4B target shape.
    private func targetLikeConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "num_hidden_layers": 35,
            "intermediate_size": 6144,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "sliding_window": 512,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    @Test func drafterConfigMarksAllLayersShared() throws {
        let config = try drafterLikeConfig()
        // For a drafter: numKvSharedLayers == numHiddenLayers, firstShared == 0.
        // Every layer must be marked usesSharedKV == true.
        // We probe this via Gemma4Attention's public introspection (added in Step 4.3).
        for i in 0 ..< config.numHiddenLayers {
            #expect(Gemma4Attention.layerUsesSharedKV(config: config, layerIdx: i))
        }
    }

    @Test func targetConfigKeepsExistingSplit() throws {
        let config = try targetLikeConfig()
        // First 15 layers: not shared.
        for i in 0 ..< 15 {
            #expect(!Gemma4Attention.layerUsesSharedKV(config: config, layerIdx: i))
        }
        // Last 20 layers: shared.
        for i in 15 ..< 35 {
            #expect(Gemma4Attention.layerUsesSharedKV(config: config, layerIdx: i))
        }
    }

    @Test func forceSharedKVOverridesIndex() throws {
        let config = try targetLikeConfig()
        // Even layer 0 should be treated as shared when forceSharedKV is true.
        #expect(Gemma4Attention.layerUsesSharedKV(
            config: config, layerIdx: 0, forceSharedKV: true))
    }
}
```

- [ ] **Step 4.2: Run the test to verify it fails**

Run: `swift test --filter Gemma4ForceSharedKVTests`

Expected: compilation failure — `Gemma4Attention.layerUsesSharedKV` does not exist.

- [ ] **Step 4.3: Add the `layerUsesSharedKV` static helper + fix the guard**

In `Libraries/MLXLLM/Models/Gemma4Text.swift`, in the `Gemma4Attention` class (around line 248-270), change the `init` signature and body. Current code:

```swift
init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
    self.config = config
    self.layerIdx = layerIdx
    self.layerType = config.layerTypes[layerIdx]
    self.isSliding = layerType == "sliding_attention"
    let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
    self.usesSharedKV = layerIdx >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0
    // ... rest unchanged
}
```

Replace the init header and the `usesSharedKV` line (leave everything after `self.usesSharedKV = ...` unchanged):

```swift
init(_ config: Gemma4TextConfiguration, layerIdx: Int, forceSharedKV: Bool = false) {
    self.config = config
    self.layerIdx = layerIdx
    self.layerType = config.layerTypes[layerIdx]
    self.isSliding = layerType == "sliding_attention"
    self.usesSharedKV = Self.layerUsesSharedKV(
        config: config, layerIdx: layerIdx, forceSharedKV: forceSharedKV)
    // ... rest unchanged
}

/// Predicate for whether a layer uses shared K/V (consuming it from an
/// earlier layer rather than projecting its own).
///
/// A layer is shared when either:
/// - `forceSharedKV` is true (drafter / assistant models where every layer
///   borrows K/V from the target), or
/// - the config declares `numKvSharedLayers > 0` AND this layer's index
///   falls within the trailing shared block.
public static func layerUsesSharedKV(
    config: Gemma4TextConfiguration,
    layerIdx: Int,
    forceSharedKV: Bool = false
) -> Bool {
    if forceSharedKV { return true }
    guard config.numKvSharedLayers > 0 else { return false }
    let firstShared = config.numHiddenLayers - config.numKvSharedLayers
    return layerIdx >= firstShared
}
```

- [ ] **Step 4.4: Plumb `forceSharedKV` through the layer / trunk inits**

In `Gemma4DecoderLayer.init` (around line 513), change the signature from:

```swift
init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
```

to:

```swift
init(_ config: Gemma4TextConfiguration, layerIdx: Int, forceSharedKV: Bool = false) {
```

And change the `self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)` line to:

```swift
self._selfAttn.wrappedValue = Gemma4Attention(
    config, layerIdx: layerIdx, forceSharedKV: forceSharedKV)
```

In `Gemma4TextModelInner.init` (around line 642), change the signature from:

```swift
init(_ config: Gemma4TextConfiguration) {
```

to:

```swift
init(_ config: Gemma4TextConfiguration, forceSharedKV: Bool = false) {
```

And change the layer construction line from:

```swift
self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
    Gemma4DecoderLayer(config, layerIdx: $0)
}
```

to:

```swift
self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
    Gemma4DecoderLayer(config, layerIdx: $0, forceSharedKV: forceSharedKV)
}
```

`Gemma4TextModel.init` (public) does NOT need the `forceSharedKV` parameter added — the drafter will use `Gemma4TextModelInner` directly via the new drafter-facing init path (added in Task 16).

- [ ] **Step 4.5: Run the test to verify it passes**

Run: `swift test --filter Gemma4ForceSharedKVTests`

Expected: 3 tests pass.

- [ ] **Step 4.6: Run the full test suite**

Run: `swift test`

Expected: all tests pass (the guard change preserves behavior for the existing 35-layer target by construction — validated by `targetConfigKeepsExistingSplit`).

- [ ] **Step 4.7: Commit**

```bash
git add Libraries/MLXLLM/Models/Gemma4Text.swift Tests/MLXLMTests/Gemma4ForceSharedKVTests.swift
git commit -m "$(cat <<'EOF'
Gemma4Text: fix usesSharedKV guard + add forceSharedKV switch

The old predicate (firstKvSharedLayerIdx > 0) incorrectly excluded
fully-shared configs like the Gemma 4 assistant drafter, where
numKvSharedLayers == numHiddenLayers. New predicate is exposed as a
static helper for test coverage, and the init gains a forceSharedKV
override for defense-in-depth when a drafter config is ever passed
with a surprising layer count.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `Gemma4TextModel.applyLMHead(_:)` helper (pure refactor)

**Files:**
- Modify: `Libraries/MLXLLM/Models/Gemma4Text.swift` (`Gemma4TextModel.callAsFunction` around line 802)

**Rationale:** Extract head-application so both `callAsFunction` and the new `forwardForMTP` share one code path.

- [ ] **Step 5.1: Refactor `callAsFunction` to delegate head+softcap to `applyLMHead`**

In `Libraries/MLXLLM/Models/Gemma4Text.swift`, locate `Gemma4TextModel.callAsFunction` (around line 802):

```swift
public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    var out = model(inputs, cache: cache)
    if let lmHead {
        out = lmHead(out)
    } else {
        out = model.embedTokens.asLinear(out)
    }
    out = tanh(out / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
    return out
}
```

Replace with:

```swift
public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    let hidden = model(inputs, cache: cache)
    return applyLMHead(hidden)
}

/// Apply the LM head (tied embedding or explicit `lm_head`) plus the
/// configured final-logit softcap. Pure function of the post-norm hidden.
private func applyLMHead(_ hidden: MLXArray) -> MLXArray {
    var out: MLXArray
    if let lmHead {
        out = lmHead(hidden)
    } else {
        out = model.embedTokens.asLinear(hidden)
    }
    out = tanh(out / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
    return out
}
```

- [ ] **Step 5.2: Run the full test suite to confirm the refactor is behavior-preserving**

Run: `swift test`

Expected: all existing tests pass unchanged. This is a pure refactor — no new tests, no behavior change.

- [ ] **Step 5.3: Commit**

```bash
git add Libraries/MLXLLM/Models/Gemma4Text.swift
git commit -m "$(cat <<'EOF'
Gemma4TextModel: extract applyLMHead helper

Pure refactor: callAsFunction now delegates head-application + softcap to
a private helper. Same behavior, but forwardForMTP (next task) can reuse
the exact same head path without duplication.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `Gemma4SharedKV` + `Gemma4SharedKVCapture` types

**Files:**
- Modify: `Libraries/MLXLLM/Models/Gemma4Text.swift` (add public types near the top, after `Gemma4` namespace)

**Rationale:** The shared-KV capture sink (class, mutable) and its immutable snapshot (struct) are part of `Gemma4TextModel`'s public API because the drafter (in `MLXSpeculative`) constructs and consumes both.

- [ ] **Step 6.1: Write the failing test**

Create `Tests/MLXLMTests/Gemma4SharedKVTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma4SharedKV types")
struct Gemma4SharedKVTests {

    @Test func sharedKVConstructs() {
        let k = MLXArray.zeros([1, 2, 8, 4], dtype: .float32)
        let v = MLXArray.zeros([1, 2, 8, 4], dtype: .float32)
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        #expect(shared.fullAttention.0.dim(2) == 8)
        #expect(shared.slidingAttention.1.dim(2) == 8)
    }

    @Test func sliceTailZeroRejectedIsIdentity() {
        let k = MLXArray.ones([1, 2, 8, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 8, 4], dtype: .float32) * 2.0
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        let sliced = Gemma4SharedKV.sliceTail(from: shared, rejected: 0)
        #expect(sliced.fullAttention.0.dim(2) == 8)
        #expect(sliced.slidingAttention.0.dim(2) == 8)
    }

    @Test func sliceTailTrimsTimeAxis() {
        let k = MLXArray.ones([1, 2, 8, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 8, 4], dtype: .float32) * 2.0
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        let sliced = Gemma4SharedKV.sliceTail(from: shared, rejected: 3)
        #expect(sliced.fullAttention.0.dim(2) == 5)
        #expect(sliced.slidingAttention.0.dim(2) == 5)
        #expect(sliced.fullAttention.1.dim(2) == 5)
    }

    @Test func sliceTailClampsAtOne() {
        // When rejected >= T, preserve at least one slot (preserving the
        // minimum cache invariant used by the round-loop).
        let k = MLXArray.ones([1, 2, 4, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 4, 4], dtype: .float32)
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        let sliced = Gemma4SharedKV.sliceTail(from: shared, rejected: 100)
        #expect(sliced.fullAttention.0.dim(2) == 1)
    }

    @Test func captureIsReferenceType() {
        let capture = Gemma4SharedKVCapture()
        #expect(capture.fullAttention == nil)
        #expect(capture.slidingAttention == nil)
        let k = MLXArray.zeros([1, 2, 4, 4], dtype: .float32)
        let v = MLXArray.zeros([1, 2, 4, 4], dtype: .float32)
        capture.fullAttention = (k, v)
        // Aliasing semantics: a second reference observes the write.
        let alias = capture
        #expect(alias.fullAttention != nil)
    }
}
```

- [ ] **Step 6.2: Run the test to verify it fails**

Run: `swift test --filter Gemma4SharedKVTests`

Expected: compilation failure — `Gemma4SharedKV` / `Gemma4SharedKVCapture` undefined.

- [ ] **Step 6.3: Define the types**

In `Libraries/MLXLLM/Models/Gemma4Text.swift`, immediately after the `public enum Gemma4` namespace (from Task 2), add:

```swift
/// Immutable snapshot of per-layer-type K/V captured from a target
/// `Gemma4TextModel` during `forwardForMTP`. Consumed by the Gemma 4 MTP
/// drafter; every drafter layer reads from one of these two slots rather
/// than projecting its own K/V.
public struct Gemma4SharedKV: Sendable {
    /// K/V from the target's last non-shared full-attention layer.
    /// Shape: `[B, nGlobalKVHeads, T, globalHeadDim]`.
    public let fullAttention: (MLXArray, MLXArray)
    /// K/V from the target's last non-shared sliding-attention layer.
    /// Shape: `[B, nKVHeads, T, headDim]`.
    public let slidingAttention: (MLXArray, MLXArray)

    public init(
        fullAttention: (MLXArray, MLXArray),
        slidingAttention: (MLXArray, MLXArray)
    ) {
        self.fullAttention = fullAttention
        self.slidingAttention = slidingAttention
    }

    /// Trim the tail of both K/V tensors by `rejected` time positions. Used
    /// by the MTP round-loop to match the post-rollback target cache length.
    /// If `rejected >= T`, clamps to a 1-slot tail so the drafter always has
    /// at least one K/V position to attend to.
    public static func sliceTail(
        from shared: Gemma4SharedKV, rejected: Int
    ) -> Gemma4SharedKV {
        func slice(_ kv: (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
            let T = kv.0.dim(2)
            let valid = max(1, T - max(0, rejected))
            if valid >= T { return kv }
            let k = kv.0[.ellipsis, ..<valid, 0...]
            let v = kv.1[.ellipsis, ..<valid, 0...]
            return (k, v)
        }
        return Gemma4SharedKV(
            fullAttention: slice(shared.fullAttention),
            slidingAttention: slice(shared.slidingAttention)
        )
    }
}

/// Mutable sink for the shared-KV capture hook in
/// `Gemma4TextModelInner.callAsFunction`. A reference type so the trunk can
/// write into it without a return-value contortion; cleared by the caller
/// between forwards.
public final class Gemma4SharedKVCapture: @unchecked Sendable {
    public var fullAttention: (MLXArray, MLXArray)? = nil
    public var slidingAttention: (MLXArray, MLXArray)? = nil

    public init() {}

    /// Snapshot into an immutable `Gemma4SharedKV`. Throws via
    /// `fatalError` if either slot is missing — the capture hook is
    /// expected to populate both.
    public func snapshot() -> Gemma4SharedKV {
        guard let full = fullAttention else {
            fatalError("Gemma4SharedKVCapture: fullAttention was not populated")
        }
        guard let sliding = slidingAttention else {
            fatalError("Gemma4SharedKVCapture: slidingAttention was not populated")
        }
        return Gemma4SharedKV(fullAttention: full, slidingAttention: sliding)
    }
}
```

- [ ] **Step 6.4: Run the tests**

Run: `swift test --filter Gemma4SharedKVTests`

Expected: 5 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add Libraries/MLXLLM/Models/Gemma4Text.swift Tests/MLXLMTests/Gemma4SharedKVTests.swift
git commit -m "$(cat <<'EOF'
Gemma4Text: add Gemma4SharedKV + Gemma4SharedKVCapture public types

Gemma4SharedKV is the immutable snapshot consumed by the MTP drafter;
Gemma4SharedKVCapture is the mutable reference-type sink used by the
forthcoming capture hook inside Gemma4TextModelInner. sliceTail clamps
to 1 slot when rejected >= T to keep the drafter's attention valid.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

---

## Task 7: Inner trunk shared-KV capture hook
**Files:**
- Modify: `Libraries/MLXLLM/Models/Gemma4Text.swift` (`Gemma4TextModelInner` class — add stored properties for capture indices, extend `callAsFunction` with an optional `capture` parameter)
- Create: `Tests/MLXLMTests/Gemma4CaptureHookTests.swift`

**Rationale:** The drafter reads K/V from the target's last non-shared full-attention layer and last non-shared sliding-attention layer. This hook is the only way to get those tensors out of the target forward without modifying the `LanguageModel` protocol or `LMOutput`.

- [ ] **Step 7.1: Write the failing test**

Create `Tests/MLXLMTests/Gemma4CaptureHookTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4TextModelInner shared-KV capture hook")
struct Gemma4CaptureHookTests {

    private func smallTargetConfig() throws -> Gemma4TextConfiguration {
        // 10-layer config with the last 5 kv-shared. First 5 are pattern
        // [sliding, sliding, sliding, sliding, full] — matching the real
        // model's slidingWindowPattern = 5. After derivation, layer_types
        // should be [sliding, sliding, sliding, sliding, full,
        //            sliding, sliding, sliding, sliding, full].
        // Last non-shared full-attn = layer 4; last non-shared sliding = 3.
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 5,
            "sliding_window": 16,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    @Test func captureSlotsArePopulated() throws {
        let config = try smallTargetConfig()
        let model = Gemma4TextModel(config)
        eval(model)

        let tokens = MLXArray([[Int32(1), 2, 3, 4]])  // [1, 4]
        let capture = Gemma4SharedKVCapture()
        _ = model.forwardForMTP(tokens, cache: model.newCache(parameters: nil))
        // Using forwardForMTP (Task 8) is the normal way to get capture
        // populated, but the test here just validates the hook works.
        // If forwardForMTP isn't yet implemented, use the internal trunk
        // directly via a new `callAsFunction(_:cache:capture:)` overload.
        //
        // NOTE: this test depends on Task 8's forwardForMTP. If executing
        // Task 7 standalone, skip the forwardForMTP call and drive the
        // inner trunk directly — see Step 7.3 for the signature.
        #expect(capture.fullAttention == nil)  // placeholder until Task 8 links up
    }
}
```

*Note:* Because the capture hook is internal plumbing consumed by `forwardForMTP` (Task 8), the observable test really lives in Task 8. Task 7's "test" here is a smoke compilation check — the real behavior test is in Task 8. If we run them together the Task 8 tests will fail until Task 7 lands. If strict TDD discipline is required per task, replace the test above with:

```swift
@Test func innerCaptureDirectlyPopulatesBothSlots() throws {
    let config = try smallTargetConfig()
    let model = Gemma4TextModel(config)
    eval(model)
    let tokens = MLXArray([[Int32(1), 2, 3, 4]])
    let cache = model.newCache(parameters: nil)
    let capture = Gemma4SharedKVCapture()
    _ = model._testCallInner(tokens, cache: cache, capture: capture)
    #expect(capture.fullAttention != nil)
    #expect(capture.slidingAttention != nil)
    // Expected shapes for this config: nKVHeads = 1, head_dim = 32,
    // global_head_dim = 32, T = 4.
    #expect(capture.fullAttention?.0.dim(2) == 4)
    #expect(capture.slidingAttention?.0.dim(2) == 4)
}
```

Where `_testCallInner` is an `internal` helper added in Step 7.3 that just forwards to the modified `Gemma4TextModelInner.callAsFunction` with capture. Make it `internal` (not `public`) so it isn't part of the library's API surface.

- [ ] **Step 7.2: Run the test to verify it fails**

Run: `swift test --filter Gemma4CaptureHookTests`

Expected: compilation failure — the inner trunk doesn't accept a `capture:` parameter yet, and `_testCallInner` doesn't exist.

- [ ] **Step 7.3: Compute capture indices at `Gemma4TextModelInner` init**

In `Gemma4TextModelInner.init` (around line 642 in the existing file), at the end of the init body (after `self.previousKvs = kvMap`, before `super.init()`), add:

```swift
// Capture indices for MTP drafter: the last layer of each type that
// still has its own K/V (not shared from an earlier layer).
self.lastFullAttentionNonSharedIdx = {
    var last = -1
    for i in 0 ..< firstKvSharedLayerIdx {
        if config.layerTypes[i] == "full_attention" { last = i }
    }
    return last
}()
self.lastSlidingAttentionNonSharedIdx = {
    var last = -1
    for i in 0 ..< firstKvSharedLayerIdx {
        if config.layerTypes[i] == "sliding_attention" { last = i }
    }
    return last
}()
```

And add these stored properties to `Gemma4TextModelInner`:

```swift
/// Index of the last non-shared full-attention layer (-1 if none).
/// Used by the shared-KV capture hook for the MTP drafter.
let lastFullAttentionNonSharedIdx: Int
let lastSlidingAttentionNonSharedIdx: Int
```

- [ ] **Step 7.4: Extend `callAsFunction` with the `capture` parameter**

In `Gemma4TextModelInner.callAsFunction` (around line 688), change the signature from:

```swift
func callAsFunction(
    _ inputs: MLXArray,
    cache: [KVCache]? = nil
) -> MLXArray {
```

to:

```swift
func callAsFunction(
    _ inputs: MLXArray,
    cache: [KVCache]? = nil,
    capture: Gemma4SharedKVCapture? = nil
) -> MLXArray {
```

Inside the layer loop (around line 758), find:

```swift
for (idx, layer) in layers.enumerated() {
    let prevIdx = previousKvs[idx]
    let sharedKV = intermediates[prevIdx].kv
    let sharedPositionOffset = intermediates[prevIdx].positionOffset

    let mask = maskByType[layer.layerType]
    let (out, kvPair, positionOffset) = layer(
        h,
        mask: mask,
        cache: fullCache[idx],
        perLayerInput: perLayerInputs[idx],
        sharedKV: sharedKV,
        positionOffset: sharedPositionOffset
    )
    h = out
    intermediates[idx] = (kvPair, positionOffset)
}
```

At the end of the loop body (right after `intermediates[idx] = (kvPair, positionOffset)`), add:

```swift
    if let capture = capture {
        if idx == lastFullAttentionNonSharedIdx {
            capture.fullAttention = kvPair
        } else if idx == lastSlidingAttentionNonSharedIdx {
            capture.slidingAttention = kvPair
        }
    }
```

- [ ] **Step 7.5: Add the `_testCallInner` internal helper**

In `Gemma4TextModel` (the public class, around line 782), add an `internal` method after the `callAsFunction` / `applyLMHead` block:

```swift
/// Internal helper for Gemma4CaptureHookTests. Not part of the public API.
internal func _testCallInner(
    _ inputs: MLXArray, cache: [KVCache], capture: Gemma4SharedKVCapture?
) -> MLXArray {
    model(inputs, cache: cache, capture: capture)
}
```

- [ ] **Step 7.6: Run the test to verify it passes**

Run: `swift test --filter Gemma4CaptureHookTests`

Expected: 1 test passes (`innerCaptureDirectlyPopulatesBothSlots`).

- [ ] **Step 7.7: Run the full suite**

Run: `swift test`

Expected: all tests pass. The `capture: nil` default on `callAsFunction` keeps every existing call site behavior-equivalent.

- [ ] **Step 7.8: Commit**

```bash
git add Libraries/MLXLLM/Models/Gemma4Text.swift Tests/MLXLMTests/Gemma4CaptureHookTests.swift
git commit -m "$(cat <<'EOF'
Gemma4TextModelInner: add optional shared-KV capture hook

When a Gemma4SharedKVCapture is passed in, the inner trunk snapshots
the K/V tensors of the last non-shared full-attention and last
non-shared sliding-attention layers into it. Capture indices are
precomputed at init. Default capture == nil keeps every existing call
site unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `Gemma4MTPForward` struct + `forwardForMTP(_:cache:)` method
**Files:**
- Modify: `Libraries/MLXLLM/Models/Gemma4Text.swift` (add `Gemma4MTPForward` + `forwardForMTP` method on `Gemma4TextModel`)
- Create: `Tests/MLXLMTests/Gemma4ForwardForMTPTests.swift`

**Rationale:** The MTP round-loop needs `(logits, lastHidden, capturedSharedKV)` in one pass. `lastHidden` is the trunk output *before* the LM head; `logits` is after. Shared-KV comes from the Task 7 capture hook.

- [ ] **Step 8.1: Write the failing test**

Create `Tests/MLXLMTests/Gemma4ForwardForMTPTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4TextModel.forwardForMTP")
struct Gemma4ForwardForMTPTests {

    private func smallConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 5,
            "sliding_window": 16,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    @Test func shapesMatchExpectations() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([[Int32(1), 2, 3, 4]])  // [1, 4]
        let cache = model.newCache(parameters: nil)
        let out = model.forwardForMTP(tokens, cache: cache)
        #expect(out.logits.shape == [1, 4, 32])       // [B, L, vocab]
        #expect(out.lastHidden.shape == [1, 4, 64])   // [B, L, hidden]
        // Shared-KV shapes: [B, nKVHeads, T, headDim]
        #expect(out.capturedSharedKV.fullAttention.0.dim(2) == 4)
        #expect(out.capturedSharedKV.slidingAttention.0.dim(2) == 4)
    }

    @Test func logitsRespectSoftcap() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([[Int32(1), 2, 3, 4]])
        let cache = model.newCache(parameters: nil)
        let out = model.forwardForMTP(tokens, cache: cache)
        // Softcap 30.0 ⇒ abs(logits) < 30.0 everywhere.
        let absMax = MLX.abs(out.logits).max().item(Float.self)
        #expect(absMax <= 30.0)
    }

    @Test func lastHiddenIsPreHead() throws {
        // The pre-head hidden passed through the LM head must reproduce the
        // same logits as forwardForMTP.logits (up to numerical equality).
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([[Int32(1), 2, 3]])
        let cache = model.newCache(parameters: nil)
        let out = model.forwardForMTP(tokens, cache: cache)
        // Recompute via the public callAsFunction (which applies the head
        // to the same pre-head hidden). These should be byte-identical.
        let cache2 = model.newCache(parameters: nil)
        let reference = model(tokens, cache: cache2)
        let close = allClose(out.logits, reference, rtol: 1e-5, atol: 1e-5)
        #expect(close.item(Bool.self))
    }
}
```

- [ ] **Step 8.2: Run the test to verify it fails**

Run: `swift test --filter Gemma4ForwardForMTPTests`

Expected: compilation failure — `forwardForMTP` / `Gemma4MTPForward` do not exist.

- [ ] **Step 8.3: Add `Gemma4MTPForward` struct**

In `Libraries/MLXLLM/Models/Gemma4Text.swift`, near the other Gemma 4 public types (after `Gemma4SharedKVCapture` from Task 6), add:

```swift
/// Result of `Gemma4TextModel.forwardForMTP`. Carries both the LM head
/// output and the pre-head trunk hidden, plus the shared-KV snapshot the
/// drafter needs for its next round.
public struct Gemma4MTPForward: Sendable {
    /// `[B, L, vocab]` — softcap applied.
    public let logits: MLXArray
    /// `[B, L, hidden_size]` — trunk output after `model.norm`, before the
    /// LM head.
    public let lastHidden: MLXArray
    /// Per-layer-type K/V from the last non-shared layers of the target.
    public let capturedSharedKV: Gemma4SharedKV

    public init(
        logits: MLXArray, lastHidden: MLXArray, capturedSharedKV: Gemma4SharedKV
    ) {
        self.logits = logits
        self.lastHidden = lastHidden
        self.capturedSharedKV = capturedSharedKV
    }
}
```

- [ ] **Step 8.4: Add `forwardForMTP(_:cache:)` method on `Gemma4TextModel`**

In `Gemma4TextModel` (after the `callAsFunction` and `applyLMHead` helpers from Task 5, before `sanitize`), add:

```swift
/// Forward pass tailored for MTP speculative decoding.
///
/// Returns both the LM head output (logits) AND the pre-head trunk
/// hidden AND a snapshot of the shared-KV tensors that the drafter
/// will consume in its next round.
///
/// The capture hook is always engaged on this path; if the target was
/// built without the shared-KV layer indices populated (e.g. a config
/// with `numKvSharedLayers == 0`), both slots of the returned
/// `capturedSharedKV` will be zero-sized tensors. In practice, Gemma 4
/// configs always have `numKvSharedLayers > 0`, so this degenerate case
/// only arises in tests.
///
/// - Parameters:
///   - tokens: `[B, L]` int token array.
///   - cache: pre-constructed KV caches for each non-shared layer. Must
///     have length `firstKvSharedLayerIdx` (see `Gemma4TextModelInner`).
/// - Returns: `Gemma4MTPForward` with logits, pre-head hidden, and
///   captured shared-KV.
public func forwardForMTP(
    _ tokens: MLXArray, cache: [KVCache]
) -> Gemma4MTPForward {
    let capture = Gemma4SharedKVCapture()
    let hidden = model(tokens, cache: cache, capture: capture)
    let logits = applyLMHead(hidden)
    return Gemma4MTPForward(
        logits: logits,
        lastHidden: hidden,
        capturedSharedKV: capture.snapshot()
    )
}
```

- [ ] **Step 8.5: Run the tests**

Run: `swift test --filter Gemma4ForwardForMTPTests`

Expected: 3 tests pass.

- [ ] **Step 8.6: Run the full suite**

Run: `swift test`

Expected: all tests pass.

- [ ] **Step 8.7: Commit**

```bash
git add Libraries/MLXLLM/Models/Gemma4Text.swift Tests/MLXLMTests/Gemma4ForwardForMTPTests.swift
git commit -m "$(cat <<'EOF'
Gemma4TextModel: add forwardForMTP(_:cache:)

Returns (logits, lastHidden, capturedSharedKV) in one pass. Logits go
through the same applyLMHead helper as the regular forward so softcap
behavior is unchanged; lastHidden is pre-head trunk output; capturedSharedKV
is populated via the Task 7 capture hook.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `Gemma4AcceptCount` + `rollbackSpeculativeCache`
**Files:**
- Modify: `Libraries/MLXLLM/Models/Gemma4Text.swift` (add `Gemma4AcceptCount` enum + `rollbackSpeculativeCache` method)
- Create: `Tests/MLXLMTests/Gemma4RollbackTests.swift`

**Rationale:** After the verify step, the target cache has `blockSize` speculative positions appended. For the B=1 path we uniform-trim `blockSize - accepted - 1` positions. For B>1 each row accepts a different count, so after the uniform trim we per-row tail-zero rows that accepted less than the max.

- [ ] **Step 9.1: Write the failing test**

Create `Tests/MLXLMTests/Gemma4RollbackTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4TextModel.rollbackSpeculativeCache")
struct Gemma4RollbackTests {

    private func smallConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 5,
            "sliding_window": 16,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    @Test func scalarTrimsUniformly() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([[Int32(1), 2, 3, 4, 5]])  // [1, 5]
        let cache = model.newCache(parameters: nil)
        _ = model(tokens, cache: cache)
        let offsetBefore = cache[0].offset  // == 5
        model.rollbackSpeculativeCache(cache, accepted: .scalar(1), blockSize: 3)
        // Trimmed = blockSize - accepted - 1 = 3 - 1 - 1 = 1
        #expect(cache[0].offset == offsetBefore - 1)
    }

    @Test func scalarNoopWhenFullyAccepted() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([[Int32(1), 2, 3, 4, 5]])
        let cache = model.newCache(parameters: nil)
        _ = model(tokens, cache: cache)
        let offsetBefore = cache[0].offset
        // accepted == blockSize - 1 means all drafts were accepted; no trim.
        model.rollbackSpeculativeCache(cache, accepted: .scalar(2), blockSize: 3)
        #expect(cache[0].offset == offsetBefore)
    }

    // perRow case tested indirectly through the B>1 round-loop tests;
    // the isolated test here just verifies the dispatch path calls both
    // trim and zeroTailPerRow without crashing on a batched cache.
    @Test func perRowDispatchesToBatchedPath() throws {
        // This test requires a BatchKVCache, which Gemma4TextModel.newCache
        // doesn't produce. We construct one manually and invoke the method
        // on a caches array that includes it.
        let b = BatchKVCache(leftPadding: [0, 0])
        let k = MLXArray.ones([2, 1, 6, 4], dtype: .float32)
        let v = MLXArray.ones([2, 1, 6, 4], dtype: .float32)
        _ = b.update(keys: k, values: v)

        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        let accepted = MLXArray([Int32(0), 2])  // row 0 accepts 0, row 1 accepts 2
        model.rollbackSpeculativeCache(
            [b], accepted: .perRow(accepted), blockSize: 3
        )
        // Uniform trim = 3 - max(accepted) - 1 = 3 - 2 - 1 = 0 → no trim.
        #expect(b.offset == 6)
        // zeroTailPerRow: row 0 keeps (6 - max + 0) = 4 slots, row 1 keeps
        // (6 - max + 2) = 6 slots. Row 0's last 2 slots should be zero.
        let state = b.state
        let row0Tail = state[0][0, 0, 4..., 0]
        #expect(row0Tail.sum().item(Float.self) == 0.0)
        let row1All = state[0][1, 0, 0..<6, 0]
        #expect(row1All.sum().item(Float.self) == 6.0)  // untouched
    }
}
```

- [ ] **Step 9.2: Run the test to verify it fails**

Run: `swift test --filter Gemma4RollbackTests`

Expected: compilation failure — `Gemma4AcceptCount` / `rollbackSpeculativeCache` do not exist.

- [ ] **Step 9.3: Add `Gemma4AcceptCount` and the method**

In `Libraries/MLXLLM/Models/Gemma4Text.swift`, after `Gemma4MTPForward` (from Task 8), add:

```swift
/// Count of accepted speculative tokens per round, either scalar (B=1)
/// or per-row (`[B]` int32 array, B>1).
public enum Gemma4AcceptCount: Sendable {
    case scalar(Int)
    case perRow(MLXArray)

    /// Max accepted across all rows (== the scalar value in the scalar case).
    func max() -> Int {
        switch self {
        case .scalar(let n): return n
        case .perRow(let arr): return Int(arr.max().item(Int32.self))
        }
    }
}
```

In `Gemma4TextModel` (after `forwardForMTP` from Task 8), add:

```swift
/// Rewind the target KV caches after a speculative-decoding round.
///
/// Uniformly trims every trimmable cache by `blockSize - max(accepted) - 1`
/// (all rows discard their rejected-suffix). In the `.perRow` case,
/// additionally calls `BatchKVCache.zeroTailPerRow` on every batched cache
/// to clear the per-row divergence where rows accepted fewer tokens than
/// the max.
///
/// Non-trimmable caches (e.g. a saturated `RotatingKVCache`) are skipped.
///
/// - Parameters:
///   - caches: the target's KV caches, typically obtained from `newCache`
///     and then advanced by `forwardForMTP`.
///   - accepted: per-round accept count.
///   - blockSize: the full block size of this speculative round (bonus +
///     k drafts; draft step count k = blockSize - 1).
public func rollbackSpeculativeCache(
    _ caches: [KVCache],
    accepted: Gemma4AcceptCount,
    blockSize: Int
) {
    let maxAccepted = accepted.max()
    let trim = max(0, blockSize - maxAccepted - 1)

    for cache in caches {
        guard cache.isTrimmable else { continue }
        if trim > 0 {
            cache.trim(trim)
        }
    }

    if case .perRow(let perRowAccepted) = accepted, maxAccepted > 0 {
        // After uniform trim, each batched cache is at length
        //   postTrimLen = preTrimLen - trim.
        // Verify-start within that post-trim cache is at
        //   postTrimLen - (maxAccepted + 1).
        // Row i keeps tokens through index
        //   postTrimLen - (maxAccepted + 1) + accepted[i] + 1
        //   = postTrimLen - (maxAccepted - accepted[i])
        // We encode this as:
        //   keepLengths[i] = postTrimLen - (maxAccepted - accepted[i])
        //                  = postTrimLen - maxAccepted + accepted[i]
        for cache in caches {
            guard let batched = cache as? BatchKVCache else { continue }
            let postTrimLen = batched.offset
            let offsetFromAccepted =
                perRowAccepted.asType(.int32)
                    + Int32(postTrimLen - maxAccepted)
            batched.zeroTailPerRow(keepLengths: offsetFromAccepted)
        }
    }
}
```

- [ ] **Step 9.4: Run the tests**

Run: `swift test --filter Gemma4RollbackTests`

Expected: 3 tests pass.

- [ ] **Step 9.5: Run the full suite**

Run: `swift test`

Expected: all tests pass.

- [ ] **Step 9.6: Commit**

```bash
git add Libraries/MLXLLM/Models/Gemma4Text.swift Tests/MLXLMTests/Gemma4RollbackTests.swift
git commit -m "$(cat <<'EOF'
Gemma4TextModel: add rollbackSpeculativeCache(_:accepted:blockSize:)

Uniform-trims caches by (blockSize - max_accepted - 1); for the per-row
case also calls BatchKVCache.zeroTailPerRow so rows that accepted fewer
tokens than the round's max have their tail divergence cleared.
Closes the cache-rewind half of the MTP target-side API.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `SpeculativeWalk.single`
**Files:**
- Create: `Libraries/MLXSpeculative/SpeculativeWalk.swift`
- Create: `Tests/MLXLMTests/SpeculativeWalkTests.swift`

**Rationale:** Greedy accept-prefix walker. Pure Swift, no MLX — trivially unit-testable and fast.

- [ ] **Step 10.1: Write the failing test**

Create `Tests/MLXLMTests/SpeculativeWalkTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLXSpeculative
import Testing

@Suite("SpeculativeWalk.single")
struct SpeculativeWalkSingleTests {

    @Test func emptyDraftReturnsJustMain() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [], main: [7]
        )
        #expect(accepted == 0)
        #expect(emitted == [7])
    }

    @Test func allDraftsMatch() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [1, 2, 3], main: [1, 2, 3, 4]
        )
        #expect(accepted == 3)
        #expect(emitted == [1, 2, 3, 4])
    }

    @Test func firstMismatchAtZero() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [9, 2, 3], main: [1, 2, 3, 4]
        )
        #expect(accepted == 0)
        #expect(emitted == [1])
    }

    @Test func firstMismatchInMiddle() {
        let (accepted, emitted) = SpeculativeWalk.single(
            draft: [1, 2, 99], main: [1, 2, 3, 4]
        )
        #expect(accepted == 2)
        #expect(emitted == [1, 2, 3])
    }

    @Test func emittedCountEqualsAcceptedPlusOne() {
        // Invariant: emitted.count == accepted + 1 for any non-trivial input.
        let draft = [1, 2, 3, 4, 5]
        let main = [1, 2, 99, 4, 5, 6]
        let (accepted, emitted) = SpeculativeWalk.single(draft: draft, main: main)
        #expect(emitted.count == accepted + 1)
    }
}
```

- [ ] **Step 10.2: Run the test to verify it fails**

Run: `swift test --filter SpeculativeWalkSingleTests`

Expected: compilation failure — `SpeculativeWalk` does not exist.

- [ ] **Step 10.3: Implement `SpeculativeWalk.single`**

Create `Libraries/MLXSpeculative/SpeculativeWalk.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation

/// Accept/reject walker for speculative decoding. Pure Swift — no MLX
/// dependency so it can be exhaustively unit-tested.
public enum SpeculativeWalk {

    /// Single-row greedy accept-prefix walker.
    ///
    /// The speculative-decoding contract: `main` contains `k + 1` tokens
    /// (one verify per draft plus one "bonus" token that the target sampled
    /// from the position after the last draft). `draft` contains the `k`
    /// tokens the drafter proposed. Walk left-to-right, accept while
    /// `draft[i] == main[i]`, and always emit one "correction" or "bonus"
    /// token from `main` past the accepted prefix.
    ///
    /// - Returns: `(acceptedCount, emittedTokens)` where
    ///   `emittedTokens.count == acceptedCount + 1`.
    public static func single(draft: [Int], main: [Int]) -> (Int, [Int]) {
        // If draft is empty, just emit main[0] (the bonus).
        guard !draft.isEmpty else {
            precondition(!main.isEmpty, "main must contain at least the bonus")
            return (0, [main[0]])
        }
        precondition(
            main.count >= draft.count + 1,
            "main must have at least draft.count + 1 tokens (drafts + bonus)"
        )
        var accepted = 0
        for i in 0 ..< draft.count {
            if main[i] != draft[i] { break }
            accepted += 1
        }
        // Emit main[0...accepted] inclusive — that's accepted + 1 tokens.
        return (accepted, Array(main[0 ... accepted]))
    }
}
```

- [ ] **Step 10.4: Run the tests**

Run: `swift test --filter SpeculativeWalkSingleTests`

Expected: 5 tests pass.

- [ ] **Step 10.5: Commit**

```bash
git add Libraries/MLXSpeculative/SpeculativeWalk.swift Tests/MLXLMTests/SpeculativeWalkTests.swift
git commit -m "$(cat <<'EOF'
MLXSpeculative: add SpeculativeWalk.single

Greedy accept-prefix walker for single-row speculative decoding. Pure
Swift; emitted.count == accepted + 1 invariant asserted via tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `SpeculativeWalk.batched`
**Files:**
- Modify: `Libraries/MLXSpeculative/SpeculativeWalk.swift` (add `batched` static method)
- Modify: `Tests/MLXLMTests/SpeculativeWalkTests.swift` (add a `SpeculativeWalkBatchedTests` suite)

**Rationale:** B>1 walker consumes `[B, k]` draft tokens, `[B, k+1]` main tokens, and per-row token budgets. Must match per-row calls to `single` exactly.

- [ ] **Step 11.1: Add the failing test**

Append to `Tests/MLXLMTests/SpeculativeWalkTests.swift`:

```swift
@Suite("SpeculativeWalk.batched")
struct SpeculativeWalkBatchedTests {

    @Test func perRowEquivalenceToSingle() {
        // B=2, k=3. Row 0 fully accepts; row 1 mismatches at position 1.
        let draft: [[Int]] = [
            [1, 2, 3],
            [4, 99, 6],
        ]
        let main: [[Int]] = [
            [1, 2, 3, 42],
            [4, 5, 6, 7],
        ]
        let budgets = [Int.max, Int.max]

        let (batchedAccepted, batchedNew) =
            SpeculativeWalk.batched(draft: draft, main: main, budgets: budgets)

        // Expected per-row via single(...)
        var expectedAccepted: [Int] = []
        var expectedNew: [[Int]] = []
        for i in 0 ..< draft.count {
            let (a, n) = SpeculativeWalk.single(draft: draft[i], main: main[i])
            expectedAccepted.append(a)
            expectedNew.append(n)
        }
        #expect(batchedAccepted == expectedAccepted)
        #expect(batchedNew == expectedNew)
    }

    @Test func budgetTruncates() {
        // Row accepts 3 and would emit 4 tokens, but budget = 2 truncates.
        let draft: [[Int]] = [[1, 2, 3]]
        let main: [[Int]] = [[1, 2, 3, 42]]
        let (accepted, emitted) =
            SpeculativeWalk.batched(draft: draft, main: main, budgets: [2])
        // Emission is capped to 2 tokens; accepted is also capped so
        // the invariant emitted.count == accepted + 1 still holds.
        #expect(emitted[0].count == 2)
        #expect(accepted[0] == 1)
    }
}
```

- [ ] **Step 11.2: Run the test to verify it fails**

Run: `swift test --filter SpeculativeWalkBatchedTests`

Expected: compilation failure — `SpeculativeWalk.batched` does not exist.

- [ ] **Step 11.3: Implement `SpeculativeWalk.batched`**

In `Libraries/MLXSpeculative/SpeculativeWalk.swift`, add a second static method to the `SpeculativeWalk` enum:

```swift
    /// Multi-row greedy accept-prefix walker with per-row emit budgets.
    ///
    /// - Parameters:
    ///   - draft: per-row draft tokens, length `[B][k]`.
    ///   - main: per-row verify tokens, length `[B][k+1]`.
    ///   - budgets: per-row max emit count. A row emits at most
    ///     `budgets[i]` tokens; the accept count is capped accordingly so
    ///     the `emitted.count == accepted + 1` invariant holds per row.
    /// - Returns: `(acceptedPerRow, emittedPerRow)`.
    public static func batched(
        draft: [[Int]], main: [[Int]], budgets: [Int]
    ) -> ([Int], [[Int]]) {
        precondition(
            draft.count == main.count && draft.count == budgets.count,
            "batched: all inputs must have the same outer length B"
        )
        var acceptedOut: [Int] = []
        var emittedOut: [[Int]] = []
        acceptedOut.reserveCapacity(draft.count)
        emittedOut.reserveCapacity(draft.count)
        for i in 0 ..< draft.count {
            var (a, e) = single(draft: draft[i], main: main[i])
            let budget = max(0, budgets[i])
            if e.count > budget {
                e = Array(e.prefix(budget))
                a = max(0, e.count - 1)
            }
            acceptedOut.append(a)
            emittedOut.append(e)
        }
        return (acceptedOut, emittedOut)
    }
```

- [ ] **Step 11.4: Run the tests**

Run: `swift test --filter SpeculativeWalkBatchedTests`

Expected: 2 tests pass. Also run the single-row suite to confirm no regressions: `swift test --filter SpeculativeWalkSingleTests`.

- [ ] **Step 11.5: Commit**

```bash
git add Libraries/MLXSpeculative/SpeculativeWalk.swift Tests/MLXLMTests/SpeculativeWalkTests.swift
git commit -m "$(cat <<'EOF'
MLXSpeculative: add SpeculativeWalk.batched

Per-row version with per-row emit budgets. Tested for equivalence with
per-row calls to single, and for correct budget truncation preserving
the emitted.count == accepted + 1 invariant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Tasks 12–27 — outline (to be expanded incrementally)

The remaining tasks are outlined below. Each is self-contained and can be expanded to the same level of detail (failing test → full code → pass → commit) when its turn comes up. Expansion is deferred to keep this plan file navigable and to allow per-task clarifications during execution.

**Task 12 — `Gemma4AssistantConfiguration`**
- Create `Libraries/MLXSpeculative/Gemma4AssistantConfiguration.swift` with `Codable` top-level config matching the HF schema (`backbone_hidden_size`, `use_ordered_embeddings`, `num_centroids`, `centroid_intermediate_top_k`, `tie_word_embeddings`, `block_size`, nested `text_config: Gemma4TextConfiguration`). Post-init clamp: if `num_kv_shared_layers` is 0 or missing, set it to `num_hidden_layers`.
- Commit two real drafter configs as test fixtures:
  - `Tests/MLXLMTests/Resources/gemma4-E4B-assistant-config.json` — verbatim from the HF repo.
  - `Tests/MLXLMTests/Resources/gemma4-26B-A4B-assistant-config.json` — verbatim.
- Test: decode both fixtures; assert `backboneHiddenSize`, `useOrderedEmbeddings`, `numCentroids`, `textConfig.numHiddenLayers`, and the post-init clamp.

**Task 13 — `MaskedEmbedder`**
- Create `Libraries/MLXSpeculative/MaskedEmbedder.swift`.
- `Module` class: `centroids: Linear(hidden → numCentroids)`, `tokenOrdering: MLXArray(int32, shape [vocabSize])`.
- Forward: `centroids(hidden)` → top-K via `argPartition(-scores, kth: topK - 1)[..., ..<topK]` (the established negate-and-partition idiom) → gather canonical token IDs from `tokenOrdering` → gather tied-embed rows → matmul against hidden → sentinel `min(selected) - 1` → `putAlong` scatter back into a full-vocab tensor.
- Test: output shape `[B, L, vocabSize]`; non-selected slots all equal sentinel; argmax's centroid is among the top-K scoring clusters.
- Fixture-based parity check deferred to Task 26's benchmark run (generate the reference snippet there, don't commit a hand-rolled Python stub).

**Task 14 — `DrafterMasks`**
- Create `Libraries/MLXSpeculative/DrafterMasks.swift`.
- `enum DrafterMasks` with two static methods:
  - `bidirectionalFull(...)` — always returns `.none` (SDPA handles full attention without a mask).
  - `bidirectionalSWA(queryLen:queryOffset:kvLen:window:dtype:)` — returns `.none` when `kvLen <= window && queryOffset + queryLen <= kvLen + window`, otherwise builds an additive bias with `0` inside the window and `-inf` outside.
- Test: in-window case returns `.none`; out-of-window case returns an `.array` mask with the expected `0`/`-inf` pattern on a small shape.

**Task 15 — `Gemma4MTPError`**
- Create `Libraries/MLXSpeculative/Gemma4MTPError.swift` with the five error cases from the spec (`unsupportedTarget`, `rebindForbidden`, `incompatibleDrafter`, `invalidBlockSize`, `drafterNotBound`) and `LocalizedError` conformance with non-empty `errorDescription` for each.
- Delete `Libraries/MLXSpeculative/Placeholder.swift` (from Task 1) in the same commit.
- Test: each case's `errorDescription` is non-empty and mentions the associated-value content where applicable.

**Task 16 — `Gemma4AssistantDraftModel` skeleton**
- Create `Libraries/MLXSpeculative/Gemma4AssistantDraftModel.swift`.
- `Module` class with init, `bind(target:)`, `unbind()`, and the compatibility-validation logic (throws `Gemma4MTPError.incompatibleDrafter(field:)` on each mismatch per spec).
- Init constructs: `text: Gemma4TextModelInner(config.textConfig, forceSharedKV: true)` (uses the Task 4 switch), `preProjection`, `postProjection`, optional `lmHead` / `maskedEmbedder`.
- `bind`: capture embedding closure + scale + `ObjectIdentifier` for rebind detection; run compatibility validation; idempotent on same target, throws `rebindForbidden` on different target.
- Test: bind succeeds on a matched random-weight small target; rebind to same target is a no-op; rebind to different target throws; each compatibility mismatch (backboneHiddenSize, vocabSize, layer_type, K=V, numKvSharedLayers) throws with the field name.

**Task 17 — `Gemma4AssistantDraftModel.callAsFunction`**
- Modify the same file to add the forward method.
- Pseudocode flow: `preProjection(inputsEmbeds)` → iterate layers, injecting `sharedKV` per layer-type + `positionOffset` → `text.norm` → `postProjection` for `lastHidden` → LM head dispatch (`maskedEmbedder` if non-nil; else tied if `tieWordEmbeddings`; else explicit `lmHead`) → return `(lastHidden, logits)`.
- `DrafterMasks.make(...)` consumed per layer.
- Test: shape invariants (`lastHidden: [1, 1, backbone]`, `logits: [1, 1, vocabSize]`); softcap absence (logits can exceed ±30 where target `callAsFunction` cannot).

**Task 18 — `Gemma4AssistantDraftModel.sanitize`**
- Modify the same file.
- Cast `masked_embedding.token_ordering` to int32; drop `lm_head.weight` when `tieWordEmbeddings`; fail loud if any `k_proj` / `v_proj` / `k_norm` / `v_norm` weight is present (since `forceSharedKV` means those modules aren't instantiated).
- Test: mock weights dict with int64 `token_ordering` → cast applied; `lm_head.weight` with `tie_word_embeddings=true` → dropped; unexpected `k_proj` weight → error.

**Task 19 — `Gemma4AssistantDraftModel.load(from:using:id:...)`**
- Modify the same file.
- Async free helper that mirrors `LLMModelFactory._load` but specialized: `Downloader.download(id: id, matching: ["*.safetensors", "*.json"])` → read `config.json` + `model.safetensors` → `Gemma4AssistantDraftModel(config)` → `sanitize` → `loadWeights(modelDirectory:model:perLayerQuantization:)` from `MLXLMCommon.Load`.
- Test (integration, gated on `MLX_SWIFT_LM_INTEGRATION_DATA_DIR`): points at a pre-downloaded E2B drafter directory; asserts shape of `model.layers.0.self_attn.q_proj.weight`.

**Task 20 — `runGemma4MTPRounds` (B=1 round loop)**
- Create `Libraries/MLXSpeculative/Gemma4MTPRoundLoop.swift`.
- Port the spec's pseudocode verbatim (spec §"Round-loop (Gemma4MTPRoundLoop, B=1)").
- Precondition assertions mirror the spec's debug-only invariants (`targetCache[0].offset` monotonicity, hidden shape, draft/main counts).
- `asyncEval` after each draft-step sample for pipelining (spec risk register).
- Test: with a drafter whose weights are identical to the target, every round accepts `blockSize - 1` tokens; with a zeroed drafter, `accepted == 0` every round; both paths produce the same token sequence as a no-drafter baseline.

**Task 21 — `runGemma4MTPRoundsBatched` (B>1 round loop)**
- Modify the same file to add the batched variant.
- Same skeleton with per-row positions, `.perRow(MLXArray)` acceptance, finished-row filtering via `BatchKVCache.filter(batchIndices:)` when caches are `BatchedCache`.
- Per-row EOS handling: accept a `Set<Int>` of EOS token IDs and a per-row "finished" flag.
- Test: B=4 with drafter==target → full acceptance per row; one row's first generated token is EOS → that row finishes, other rows keep going until their own EOS or `maxTokens`.

**Task 22 — Public entry points**
- Create `Libraries/MLXSpeculative/Gemma4MTPGenerate.swift`.
- `generateGemma4MTP(input:parameters:target:drafter:blockSize:wiredMemoryTicket:)` — opens an `AsyncStream<Generation>`, runs a normal target forward to prefill the cache and produce `(firstBonus, firstHidden, firstSharedKV)`, then drives `runGemma4MTPRounds`.
- `generateGemma4MTPBatched(...)` — same pattern using `runGemma4MTPRoundsBatched`, yields `BatchedGeneration`.
- The `BatchedGeneration` / `Slot` / `FinishReason` types from the spec live in this file.
- Input validation: `target.model as? Gemma4TextModel` → throws `unsupportedTarget` if nil; `blockSize >= 2 && blockSize <= 16` else `invalidBlockSize`.
- Test: end-to-end on random-weight small Gemma 4; both variants terminate on `maxTokens`; unsupported target type throws.

**Task 23 — `Gemma4E2BMTPParityTest` (integration, smaller model first)**
- Create `Tests/MLXLMTests/Gemma4E2BMTPParityTest.swift`.
- Gate with `@Test(.enabled(if: ProcessInfo.processInfo.environment["MLX_SWIFT_LM_RUN_INTEGRATION"] == "1"))`.
- Commit `Tests/MLXLMTests/Resources/Gemma4MTPPrompts.json` with the 20 prompts from the spec (5 short / 10 medium / 5 long).
- Resolve target + drafter paths via `MLX_SWIFT_LM_INTEGRATION_DATA_DIR`.
- Sweep 3 block sizes × {B=1, B=4}; assert exact token-sequence equality between baseline and MTP greedy.

**Task 24 — `Gemma4E4BMTPParityTest` (acceptance gate)**
- Same shape as Task 23 but pointing at E4B checkpoints. 120 configurations total. Merge-gate test.

**Task 25 — `Gemma4_26BA4B_4bit_MTPParityTest` (stretch)**
- Same shape, 5 short prompts, B=1, blockSize=3. Target is the 4-bit quantized variant; drafter stays bf16.

**Task 26 — Benchmark harness + Python oracle + `Gemma4MTPBaseline.swift`**
- Create `Libraries/BenchmarkHelpers/Gemma4MTPBenchmark.swift` — sweep harness (E4B bf16 + E2B bf16 + 26B-A4B 4-bit, per spec §Microbenchmarks, item 11).
- Run Python oracle (`Blaizzy/mlx-vlm` at `244f4bb5a3339b180da3d2b276a4bdfcf7670f9f`) on this 36 GB box over the `Gemma4MTPPrompts.json` prompt set at every (target × block × B) combo in scope; capture tokens/sec and accepted-per-round histograms.
- Create `Libraries/BenchmarkHelpers/Gemma4MTPBaseline.swift` with the frozen Python numbers + commit sha + `measuredOn` / `measuredAt` metadata.
- Swift benchmark entry point asserts tokens/sec ≥ 0.9× the frozen Python baseline for (E4B, B=1, block=3) and (E4B, B=4, block=3); KS test on per-round accepted histograms asserts p > 0.05 vs the frozen Python baseline; print the full sweep table.

**Task 27 — Documentation**
- Modify top-level `README.md`: add a "Speculative Decoding / Gemma 4 MTP" section with a minimal code sample, a link to `Libraries/MLXSpeculative/README.md`, and a pointer to the spec + plan.
- Rewrite `Libraries/MLXSpeculative/README.md` (initially stubbed in Task 1) with: (a) usage example, (b) supported target/drafter pairings table with sizes, (c) performance table from Task 26, (d) caveats from the spec (greedy-only v1, drafter always bf16, out-of-scope hardware, stochastic sampling divergence expected), (e) links to spec + plan.

---

## Self-review

Coverage vs spec:
- Public API (`generateGemma4MTP`, batched variant, `BatchedGeneration` struct) → Task 22 + the struct defined in Task 22's implementation.
- Module layout → Tasks 1 (Package.swift, README), 10 (SpeculativeWalk), 12 (Config), 13 (MaskedEmbedder), 14 (DrafterMasks), 15 (Error), 16-18 (DraftModel), 20-21 (round loops), 22 (public entry).
- Seven Gemma4Text.swift edits → Tasks 2 (PositionOffset public), 4 (usesSharedKV + forceSharedKV), 5 (applyLMHead), 6 (Gemma4SharedKV + Capture types), 7 (inner trunk hook), 8 (forwardForMTP), 9 (rollbackSpeculativeCache).
- `BatchKVCache.zeroTailPerRow` → Task 3.
- Compatibility validation → Task 16.
- Testing strategy → Tasks 3, 4, 6-11, 13-18 (unit); 19 (weight-loading smoke); 23-25 (parity integration); 26 (benchmark).
- Error types → Task 15.
- Package.swift edits (library, test dep, benchmark dep) → Task 1.

Placeholder scan: Tasks 1-11 have every code block fully inlined. Tasks 12-27 are outlined with specific files, goals, and test shape — the detailed step-by-step expansion happens during execution. No `TBD`/`TODO` in the outline.

Type consistency check: `Gemma4SharedKV`, `Gemma4SharedKVCapture`, `Gemma4AcceptCount`, `Gemma4MTPForward`, `Gemma4.PositionOffset`, `Gemma4MTPError`, `Gemma4AssistantConfiguration`, `Gemma4AssistantDraftModel`, `MaskedEmbedder`, `DrafterMasks`, `SpeculativeWalk` each appear exactly once as a definition and consistently at all reference sites.
- Create: `Libraries/MLXSpeculative/Gemma4AssistantConfiguration.swift`.
- `Codable` top-level config (`backboneHiddenSize`, `useOrderedEmbeddings`, `numCentroids`, `centroidIntermediateTopK`, `tieWordEmbeddings`, `blockSize`, nested `textConfig`). Post-init clamp: if `numKvSharedLayers` is 0/missing, set it to `numHiddenLayers`.
- Test: decode both E4B and 26B-A4B drafter `config.json` (committed as fixtures under `Tests/MLXLMTests/Resources/`); assert expected field values.

### Task 13 — `MaskedEmbedder`
- Create: `Libraries/MLXSpeculative/MaskedEmbedder.swift`.
- Centroid-routed sparse head: `centroids: Linear(hidden → numCentroids)`, `tokenOrdering: MLXArray (vocabSize, int32)`. Forward: score clusters, take top-K via `argPartition`-on-negated-scores, gather tied-embed rows, matmul, scatter back via `putAlong`, sentinel `min(selected) - 1`.
- Test: shape `[B, L, vocabSize]`; non-selected slots all equal the sentinel; argmax token's centroid is among the top-K; fixture-based parity against a committed Python-computed reference.

### Task 14 — `DrafterMasks`
- Create: `Libraries/MLXSpeculative/DrafterMasks.swift`.
- Two functions: `bidirectionalFull(...)` returns `.none`; `bidirectionalSWA(...)` returns `.none` when `kvLen <= window`, otherwise an additive `-inf`/`0` bias.
- Test: covered-window case returns `.none`; over-window case returns an array mask with the correct `0`/`-inf` pattern.

### Task 15 — `Gemma4MTPError`
- Create: `Libraries/MLXSpeculative/Gemma4MTPError.swift` (`LocalizedError`, five cases per spec).
- Delete the `Placeholder.swift` from Task 1.
- Test: each case has a non-empty `errorDescription`.

### Task 16 — `Gemma4AssistantDraftModel` skeleton (init + bind/unbind + compatibility validation)
- Create: `Libraries/MLXSpeculative/Gemma4AssistantDraftModel.swift`.
- `Module` subclass. Init constructs `text: Gemma4TextModelInner(config.textConfig, forceSharedKV: true)`, `preProjection`, `postProjection`, optional `lmHead`/`maskedEmbedder`. `bind(target:)`: captures embedding closure, `ObjectIdentifier` for rebind check; runs compatibility validation and throws `Gemma4MTPError.incompatibleDrafter` on mismatch. `unbind()` clears state.
- Test: bind succeeds on matched target; rebind to same target is a no-op; rebind to different target throws; each compatibility mismatch throws with the field name.

### Task 17 — `Gemma4AssistantDraftModel.callAsFunction`
- Modify: `Libraries/MLXSpeculative/Gemma4AssistantDraftModel.swift`.
- Forward: `preProjection` → per-layer pass through `text.layers` with `sharedKV` injected per layer-type + `positionOffset` forwarded → `text.norm` → `postProjection` for `lastHidden` → LM head dispatch (`maskedEmbedder` / tied / explicit `lmHead`) → **no softcap**.
- Test: shape `(lastHidden=[1,1,backbone], logits=[1,1,vocab])`; softcap-absence test (drafter output can exceed ±30 where target output cannot).

### Task 18 — `Gemma4AssistantDraftModel.sanitize`
- Modify: same file.
- Cast `masked_embedding.token_ordering` to int32; drop `lm_head.weight` when `tieWordEmbeddings`; fail loud on unexpected `k_proj`/`v_proj` keys.
- Test: mock weights dict with int64 `token_ordering` → cast applied; `lm_head.weight` present with `tie_word_embeddings=true` → dropped.

### Task 19 — `Gemma4AssistantDraftModel.load(from:using:id:...)`
- Modify: same file.
- Async free helper: `Downloader.download` → read `config.json` + `model.safetensors` → construct + `sanitize` + `loadWeights`.
- Test (integration): skipped in CI. Uses `MLX_SWIFT_LM_INTEGRATION_DATA_DIR` env var to point at a pre-downloaded E2B drafter directory; asserts shape of one weight tensor.

### Task 20 — `runGemma4MTPRounds` (B=1)
- Create: `Libraries/MLXSpeculative/Gemma4MTPRoundLoop.swift`.
- Port the spec's pseudocode verbatim. Mutable state on the function only.
- Test: with drafter == target (same weights), every round accepts `bs-1` tokens and emits `bs`; with a zeroed drafter, `accepted == 0` and output still matches a no-drafter baseline token-for-token.

### Task 21 — `runGemma4MTPRoundsBatched` (B>1)
- Modify: same file.
- Same skeleton with per-row accepted counts, `rollbackSpeculativeCache(..., accepted: .perRow(...))`, finished-row filtering when `cache is BatchedCache`.
- Test: B=4 with all rows using drafter==target → full acceptance per row; EOS on one row finishes it while others continue.

### Task 22 — Public API: `generateGemma4MTP` + batched variant
- Create: `Libraries/MLXSpeculative/Gemma4MTPGenerate.swift`.
- Thin wrappers around the round-loops that open an `AsyncStream`, prefill the target cache via the normal single-model forward to produce `firstBonus + firstHidden + firstSharedKV`, then drive the loop.
- Test: end-to-end on random-weight small Gemma 4 → both variants yield `Generation`s and terminate on `maxTokens`.

### Task 23 — `Gemma4E2BMTPParityTest` (integration, smaller model first)
- Create: `Tests/MLXLMTests/Gemma4E2BMTPParityTest.swift`.
- Gated on `MLX_SWIFT_LM_RUN_INTEGRATION=1`; requires `MLX_SWIFT_LM_INTEGRATION_DATA_DIR` to point at a dir containing `gemma-4-e2b-it-bf16/` + `gemma-4-E2B-it-assistant-bf16/`.
- Loads baseline prompt set from `Tests/MLXLMTests/Resources/Gemma4MTPPrompts.json` (committed in this task); runs baseline + MTP greedy; asserts exact token-sequence equality across all combinations.

### Task 24 — `Gemma4E4BMTPParityTest` (acceptance gate)
- Same shape as Task 23, pointed at E4B + E4B-assistant. 120 configurations total.
- This is the merge-gate test.

### Task 25 — `Gemma4_26BA4B_4bit_MTPParityTest` (stretch)
- Same shape, 5 short prompts, B=1 only, blockSize=3. Target is the 4-bit quantized variant; drafter stays bf16.

### Task 26 — Benchmark harness + `Gemma4MTPBaseline.swift`
- Create: `Libraries/BenchmarkHelpers/Gemma4MTPBenchmark.swift` (the sweep harness).
- Create: `Libraries/BenchmarkHelpers/Gemma4MTPBaseline.swift` (frozen Python reference numbers + commit sha + measurement metadata).
- Run the Python oracle (`Blaizzy/mlx-vlm` at `244f4bb`) on this 36 GB box over the same `Gemma4MTPPrompts.json`; record tokens/sec + accepted-per-round histograms per config; commit the numbers.
- Swift benchmark asserts tokens/sec ≥ 0.9× the recorded Python baseline for the two named primary configs (E4B B=1 block=3, E4B B=4 block=3) and prints the full sweep table.

### Task 27 — Docs
- Modify: top-level `README.md` with a new "Speculative Decoding / Gemma 4 MTP" section.
- Ensure `Libraries/MLXSpeculative/README.md` from Task 1 is updated with the post-implementation usage example + performance table + caveats (greedy-only v1, drafter always bf16, out-of-scope hardware).

---

## Self-review

Coverage vs spec:
- Public API (`generateGemma4MTP`, batched variant, `BatchedGeneration` struct) → Task 22 + the struct defined in Task 22's implementation.
- Module layout → Tasks 1 (Package.swift, README), 10 (SpeculativeWalk), 12 (Config), 13 (MaskedEmbedder), 14 (DrafterMasks), 15 (Error), 16-18 (DraftModel), 20-21 (round loops), 22 (public entry).
- Seven Gemma4Text.swift edits → Tasks 2 (PositionOffset public), 4 (usesSharedKV + forceSharedKV), 5 (applyLMHead), 6 (Gemma4SharedKV + Capture types), 7 (inner trunk hook), 8 (forwardForMTP), 9 (rollbackSpeculativeCache).
- `BatchKVCache.zeroTailPerRow` → Task 3.
- Compatibility validation → Task 16.
- Testing strategy → Tasks 3, 4, 6-11, 13-18 (unit); 19 (weight-loading smoke); 23-25 (parity integration); 26 (benchmark).
- Error types → Task 15.
- Package.swift edits (library, test dep, benchmark dep) → Task 1.

Placeholder scan: all inlined code blocks are complete. The "Remainder of the plan" section (Tasks 7-27) lists goals, files, and test shape for each remaining task — the detailed step-by-step (failing test → code → pass → commit) structure for those tasks is to be expanded in a follow-up pass if the subagent-driven executor requests it. This is intentional: the plan is structured so the first 6 tasks (with full detail) can begin execution immediately while the remainder can be expanded incrementally without blocking progress.

Type consistency: `Gemma4SharedKV`, `Gemma4SharedKVCapture`, `Gemma4AcceptCount`, `Gemma4MTPForward`, `Gemma4.PositionOffset`, `Gemma4MTPError`, `Gemma4AssistantConfiguration`, `Gemma4AssistantDraftModel`, `MaskedEmbedder`, `DrafterMasks`, `SpeculativeWalk` — each appears exactly once as a definition site and consistently in all reference sites.
