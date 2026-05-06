# Gemma 4 Multi-Token Prediction (MTP) Drafter — Design Spec

**Date:** 2026-05-05
**Target repo:** `mlx-swift-lm`
**Target branch:** `main`
**Target audience:** public MLX Swift users; ultimately upstreamed to Apple.
**Reference (semantics only, not a design to clone):**
[Blaizzy/mlx-vlm#1112](https://github.com/Blaizzy/mlx-vlm/pull/1112) (merged at
`244f4bb5a3339b180da3d2b276a4bdfcf7670f9f`), Google's blog post
[Multi-Token Prediction for Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/),
docs at [ai.google.dev/gemma/docs/mtp](https://ai.google.dev/gemma/docs/mtp/mtp).

## Goal

Add production-grade support for Google's Gemma 4 "assistant" Multi-Token
Prediction drafters to `mlx-swift-lm`, so that a user pairing a Gemma 4 target
with its published assistant checkpoint gets greedy-identical output at a
measurable throughput speedup on Apple Silicon.

## Non-goals

- Generic MTP / self-speculative abstractions. We ship exactly one drafter
  (`Gemma4AssistantDraftModel`) and one round-loop (`Gemma4MTPRoundLoop`). No
  `DraftModel` protocol, no `SpeculativeCaptureHost` protocol, no
  `DrafterStrategy` enum. Abstractions wait for a second concrete case.
- Composition with `BatchGenerator` (the existing continuous-batching engine).
  MTP v1 ships its own B=1 and B>1 round-loops. Integrating MTP *into*
  `BatchGenerator` is a follow-up design.
- Drafters for other models that currently strip `mtp.*` weights (Qwen 3.5,
  Qwen 3-Next, MiMo, DeepSeek). Those remain filtered-and-dropped.
- Multimodal drafting. Image/audio prefill goes through the target unchanged;
  speculative decoding only engages on text decode.

## Hardware targets

| Tier | Target | Drafter | Fits on M-series 36 GB? |
|---|---|---|---|
| **Primary** | `mlx-community/gemma-4-e4b-it-bf16` (14.9 GB) | `mlx-community/gemma-4-E4B-it-assistant-bf16` (180 MB) | yes, comfortably |
| **Primary** | `mlx-community/gemma-4-e2b-it-bf16` (9.6 GB) | `mlx-community/gemma-4-E2B-it-assistant-bf16` (180 MB) | yes |
| **Stretch** | `mlx-community/gemma-4-26b-a4b-it-4bit` (14.6 GB) | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` (800 MB) | yes, tight |
| **Out of scope** | `gemma-4-26b-a4b-it-bf16` (48 GB), `gemma-4-31b-it-*` (58 GB / no 4-bit) | — | no — can't benchmark on this hardware |

`*-assistant-4bit` variants are gated on Hugging Face (401). All drafters
stay at bf16 — they're tiny and this matches Blaizzy's published recipe.

## Public API

Exactly one free function added to `MLXSpeculative`:

```swift
import MLXLLM
import MLXLMCommon
import MLXSpeculative

public func generateGemma4MTP(
    input: LMInput,
    parameters: GenerateParameters,
    target: ModelContext,                  // must contain Gemma4TextModel
    drafter: Gemma4AssistantDraftModel,
    blockSize: Int = 4,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation>
```

- `target.model` is cast to `Gemma4TextModel`. A mismatch throws
  `Gemma4MTPError.unsupportedTarget` at call time — no runtime surprises
  further down the pipeline.
- Drafter/target compatibility is checked in the drafter's init (see
  "Compatibility validation"). A bad pair throws before any tokens are
  generated.
- The existing `SpeculativeTokenIterator` path in `MLXLMCommon.Evaluate.swift`
  is **untouched**. It remains the right API for the two-independent-models
  case. MTP is a separate, named feature.

Batched variant (ships in v1 because the primary benchmark table includes
B>1 configs):

```swift
public func generateGemma4MTPBatched(
    inputs: [LMInput],
    parameters: GenerateParameters,
    target: ModelContext,
    drafter: Gemma4AssistantDraftModel,
    blockSize: Int = 4,
    eosTokenIds: Set<Int>? = nil,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<BatchedGeneration>
```

`BatchedGeneration` is a new struct (local to `MLXSpeculative`):

```swift
public struct BatchedGeneration: Sendable {
    public let slots: [Slot]
    public struct Slot: Sendable {
        public let row: Int                     // original index in the input batch
        public let token: Int?                  // nil when this row emitted nothing this step
        public let finishReason: FinishReason?  // .stop / .eos / .length
    }
    public enum FinishReason: Sendable { case stop, eos, length }
}
```

This is a simpler analog of `GenerationBatchResponse` scoped to the MTP
loop. `slots.count == B` (the originally-submitted batch size); rows that
have finished carry `finishReason != nil` and `token == nil`.

Loading a drafter goes through a new factory in `MLXSpeculative`:

```swift
let drafter = try await Gemma4AssistantDraftModel.load(
    from: HubClient.default,
    using: tokenizerLoader,
    id: "mlx-community/gemma-4-E4B-it-assistant-bf16"
)
```

No drafter registry / trampoline in v1. One model, one loader.

## Module layout

```
Libraries/
├── MLXLMCommon/                           (no changes to public surface)
├── MLXLLM/
│   └── Models/Gemma4Text.swift            (6 targeted edits — see below)
└── MLXSpeculative/                        (NEW product)
    ├── Gemma4MTPError.swift
    ├── Gemma4MTPRoundLoop.swift           (B=1 and B>1 in one file)
    ├── Gemma4AssistantConfiguration.swift
    ├── Gemma4AssistantDraftModel.swift
    ├── MaskedEmbedder.swift
    ├── DrafterMasks.swift
    ├── SpeculativeWalk.swift              (accept/reject, no MLX deps)
    └── README.md
```

`MLXSpeculative` depends on `MLXLLM` (for `Gemma4DecoderLayer`, `Gemma4TextModel`,
`Gemma4TextConfiguration`) and `MLXLMCommon`.

**Three edits to `Package.swift`:**

1. Add product declaration:
   ```swift
   .library(name: "MLXSpeculative", targets: ["MLXSpeculative"]),
   ```

2. Add target declaration:
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
       exclude: ["README.md"]
   ),
   ```

3. Add `"MLXSpeculative"` to the `MLXLMTests` test target's dependencies
   (parity + round-loop + masked embedder unit tests live under
   `Tests/MLXLMTests` to share the `TestTokenizer` and resource fixtures
   already there).

`BenchmarkHelpers` gains a runtime-only dependency on `MLXSpeculative` for
the Gemma 4 MTP throughput harness (edit #4 to the `BenchmarkHelpers`
target's `dependencies` array — same file).

No new test target — MTP tests live alongside existing tests in
`Tests/MLXLMTests`.

## Changes to `Gemma4Text.swift` (targeted, 7 edits)

All edits are additive or locally-scoped behavior changes. No public types
removed, no existing signatures changed.

1. **Promote `Gemma4PositionOffset` from `private` to `public`.**
   Rename to `Gemma4.PositionOffset` (introduce a public `enum Gemma4` namespace).
   Required so `Gemma4AssistantDraftModel` can construct per-row offsets and pass
   them into `Gemma4DecoderLayer.callAsFunction`.

2. **Fix `usesSharedKV` guard at `Gemma4Attention.init` (currently line ~254).**
   Replace
   ```swift
   self.usesSharedKV = layerIdx >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0
   ```
   with
   ```swift
   self.usesSharedKV = config.numKvSharedLayers > 0
       && layerIdx >= (config.numHiddenLayers - config.numKvSharedLayers)
   ```
   Semantic: current behavior preserved for the 35-layer target
   (`numKvSharedLayers=20, firstShared=15`: layers 0-14 → false, 15-34 → true).
   Drafter case now works correctly
   (`numKvSharedLayers=4, firstShared=0, numHiddenLayers=4`:
   all layers → true).

3. **Add `forceSharedKV: Bool = false` to `Gemma4Attention` /
   `Gemma4DecoderLayer` / `Gemma4TextModelInner` inits.** When `true`, treat
   every layer as kv-shared regardless of index. Redundant with (2) for the
   drafter built from the same config, but the explicit switch keeps drafter
   construction robust if the drafter config is ever loaded against the
   target's layer count by accident.

4. **Factor head-application into a private helper.**
   `Gemma4TextModel.callAsFunction` today inlines `lmHead` / tied-embed + softcap.
   Extract to a private `applyLMHead(_ h: MLXArray) -> MLXArray`. Used by the
   existing path and by `forwardForMTP`. Pure refactor, no behavior change.

5. **Add `Gemma4TextModel.forwardForMTP(...)`.**
   ```swift
   public struct Gemma4MTPForward: Sendable {
       public let logits: MLXArray            // [B, L, vocab]
       public let lastHidden: MLXArray        // [B, L, hidden_size] — pre-head trunk output
       public let capturedSharedKV: Gemma4SharedKV
   }

   public func forwardForMTP(
       _ tokens: MLXArray, cache: [KVCache]
   ) -> Gemma4MTPForward
   ```
   Runs the inner trunk with capture hooks enabled (see edit #6), returns
   `(logits, lastHidden, capturedSharedKV)`. `lastHidden` is the trunk output
   after `model.norm` and before the LM head; softcap IS applied to `logits`
   (same head as non-MTP path via the factored `applyLMHead` helper).

6. **Layer-hook inside `Gemma4TextModelInner.callAsFunction` for shared-KV
   capture.** Add an optional `capture: Gemma4SharedKVCapture?` parameter
   (ignored when nil — the default for the existing path). When non-nil:
   after each decoder layer runs, if `idx == lastFullAttnNonSharedIdx` snapshot
   `(K, V) = intermediates[idx].kv` into `capture.fullAttention`; similarly
   for `lastSlidingAttnNonSharedIdx`. Indices are computed once at init from
   `config.layerTypes` and `firstKvSharedLayerIdx`. This is a ~20-LOC
   additive change; no existing call site is modified because the default
   `capture: nil` keeps the current behavior.

   ```swift
   final class Gemma4SharedKVCapture {
       var fullAttention: (MLXArray, MLXArray)? = nil
       var slidingAttention: (MLXArray, MLXArray)? = nil
   }
   ```

   The capture object is a class (reference semantics) so the inner trunk
   can write into it without returning it. `forwardForMTP` constructs one,
   passes it in, reads it out, packs into `Gemma4SharedKV`.

7. **Add `Gemma4TextModel.rollbackSpeculativeCache(_:accepted:blockSize:)`.**
   ```swift
   public enum Gemma4AcceptCount: Sendable {
       case scalar(Int)       // B=1 path
       case perRow(MLXArray)  // B>1 path; int32, shape [B]
   }

   public func rollbackSpeculativeCache(
       _ caches: [KVCache],
       accepted: Gemma4AcceptCount,
       blockSize: Int
   )
   ```
   Delegates cache rewind to the KVCache methods. Uniform-trim every
   trimmable cache by `blockSize - max(accepted) - 1`. In the batched case,
   additionally call the new `BatchKVCache.zeroTailPerRow(keepLengths:)`
   primitive. See "`BatchKVCache` additions" below.

## `BatchKVCache` addition (in `MLXLMCommon`)

```swift
extension BatchKVCache {
    /// Zero per-row tail positions. For each row `b`, slots
    /// `[keepLengths[b], _idx)` are set to 0 in both keys and values.
    /// Used by MTP round-loop to clear rejected-tail mismatches when rows
    /// accepted different numbers of tokens in the same round.
    ///
    /// - Parameter keepLengths: `[B]` int array. Values must satisfy
    ///   `0 <= keepLengths[b] <= _idx` for all b.
    public func zeroTailPerRow(keepLengths: MLXArray)
}
```

Implementation pattern (GPU-only, no CPU sync):

```swift
guard let storedK = keys, let storedV = values else { return }
let T = storedK.dim(2)
let positions = MLXArray.arange(T).reshaped([1, 1, T, 1])       // [1,1,T,1]
let keep = keepLengths.reshaped([-1, 1, 1, 1]).asType(.int32)   // [B,1,1,1]
let mask = positions .< keep                                     // [B,1,T,1]
let maskFloat = mask.asType(storedK.dtype)
keys = storedK * maskFloat
values = storedV * maskFloat
```

Batch-free case for `KVCacheSimple` / `RotatingKVCache`: the uniform-trim
path on a single cache is equivalent to per-row tail-zero with
`keepLengths = [accepted+1]`. No new method needed there — just call
`trim(blockSize - accepted - 1)`.

## Drafter architecture

Concrete class, no protocol. Takes weights in constructor; borrows the
target's embedding via a closure set at bind time.

```swift
public final class Gemma4AssistantDraftModel: Module, @unchecked Sendable {
    public let config: Gemma4AssistantConfiguration
    public let text: Gemma4TextModel        // drafter's own 4-layer trunk
    public let preProjection: Linear        // 2 * backbone_hidden → drafter hidden
    public let postProjection: Linear       // drafter hidden → backbone hidden
    public let lmHead: Linear?              // only when !tieWordEmbeddings
    public let maskedEmbedder: MaskedEmbedder?   // E2B/E4B centroid head

    // Set by bind(target:); drafter is not usable before bind().
    private var targetEmbed: ((MLXArray) -> MLXArray)?
    private var targetEmbedScale: Float = 1.0

    public init(config: Gemma4AssistantConfiguration)

    public func bind(target: Gemma4TextModel) throws
    public func unbind()

    /// Run one drafter forward step. Inputs:
    ///   - inputsEmbeds: [B, 1, 2 * backbone_hidden]
    ///     = concat(targetEmbed(lastToken), lastHidden) along axis -1
    ///   - sharedKV: per-layer-type K/V from the target's last non-shared
    ///     full-attention and sliding-attention layers
    ///   - positionOffset: absolute position of the bonus token;
    ///     constant across all draft steps in a block
    /// Returns (newHidden: [B, 1, backbone_hidden], logits: [B, 1, vocab])
    public func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset
    ) -> (lastHidden: MLXArray, logits: MLXArray)

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}
```

**Key design choices (diverging from the Python reference on purpose):**

- **Stateless between steps.** Python's drafter carries `_shared_kv`,
  `_kv_offset`, `_position`, `accept_lens` as instance attrs. In Swift, those
  belong to the round-loop, not the model. Drafter is "weights + forward
  function." Simpler to reason about, test in isolation, and thread-safe by
  construction (modulo the bind-once pattern).
- **`Gemma4SharedKV` is a struct with two named slots**, not a dict keyed by
  string. The drafter only cares about the two Gemma 4 layer types:
  ```swift
  public struct Gemma4SharedKV: Sendable {
      public let fullAttention: (MLXArray, MLXArray)    // [B, nKVHeads, T, globalHeadDim]
      public let slidingAttention: (MLXArray, MLXArray) // [B, nKVHeads, T, headDim]
  }
  ```
- **Drafter trunk is built as a `Gemma4TextModelInner`** (via the
  `forceSharedKV: true` init path). Reuses all existing Gemma 4 building
  blocks. No new attention / MLP / decoder code in `MLXSpeculative`.
- **LM head dispatch at bind time.** If `config.useOrderedEmbeddings` →
  `MaskedEmbedder` over tied embeds. Elif `config.tieWordEmbeddings` → tied
  dense head over drafter's `text.model.embedTokens.asLinear`. Else →
  `self.lmHead`.
- **Softcap: not applied.** Drafter configs have
  `final_logit_softcapping: null`. We route through `applyLMHead` that is
  local to the drafter and doesn't call `tanh(x/s)*s`.

### `bind(target:)` semantics

- Store a closure: `targetEmbed = { ids in target.model.embedTokens(ids) }`.
- Capture `targetEmbedScale = sqrt(Float(target.config.hiddenSize))` to match
  `Gemma4TextModelInner.embedScale`.
- Capture an `ObjectIdentifier` of the target for the rebind check.
- Validate compatibility (see below).
- Idempotent: calling `bind()` twice with the same target (same
  `ObjectIdentifier`) is a no-op. Binding to a different target throws
  `Gemma4MTPError.rebindForbidden` — the drafter weights encode assumptions
  about the target's hidden size.
- `unbind()` clears the closure; exists for cache hygiene in long-running
  processes.
- **Tokenizer assumption:** the drafter and target must share the same
  tokenizer. The drafter checkpoint ships its own `tokenizer.json` for
  self-consistency but at generation time the round-loop uses the target's
  tokenizer only. Compatibility validation asserts `vocab_size` equality as a
  proxy — a full token-ID-by-token-ID equality check is too expensive and not
  necessary for the published Google pairings where the tokenizer is
  byte-identical by construction.

### Compatibility validation (at bind time, throws)

Fail-fast on every mismatch, with the field name in the error. Minimum set:

- `drafter.config.backboneHiddenSize == target.config.hiddenSize` — or
  `pre_projection` / `post_projection` shapes mismatch weights silently.
- `drafter.config.textConfig.vocabSize == target.config.vocabSize` —
  `targetEmbed` shape vs drafter LM head output vocab.
- `drafter.config.textConfig.layerTypes` — every entry must be one of
  `"full_attention"` / `"sliding_attention"`.
- K=V compatibility (applies only when BOTH of these are true):
  (a) drafter has at least one layer with `layer_type == "full_attention"`,
  (b) drafter `attention_k_eq_v == true`.
  Under those conditions: assert
  `drafter.numGlobalKeyValueHeads == target.numGlobalKeyValueHeads` AND
  `target.attention_k_eq_v == true`. Mismatch throws
  `incompatibleDrafter(field: "num_global_key_value_heads", ...)`.
- `drafter.config.textConfig.numKvSharedLayers == drafter.config.textConfig.numHiddenLayers`
  — drafter must be fully KV-shared by construction.

## Round-loop (`Gemma4MTPRoundLoop`, B=1)

Single-file implementation. Mutable state lives on the loop, not on drafter
or target. Pseudocode mirroring the Python flow but Swift-idiomatic:

```swift
public func runGemma4MTPRounds(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    targetCache: [KVCache],
    firstBonus: Int,
    firstHidden: MLXArray,               // [1, 1, hidden] from prefill, last slot only
    firstSharedKV: Gemma4SharedKV,
    parameters: GenerateParameters,
    blockSize: Int,
    continuation: AsyncStream<Generation>.Continuation
) throws {
    try drafter.bind(target: target)
    let sampler = parameters.sampler()

    // nil maxTokens is interpreted as "unlimited" — caller is responsible
    // for stream termination (e.g. task cancel).
    let maxTokens = parameters.maxTokens ?? Int.max

    var bonus = firstBonus
    var hidden = firstHidden
    var sharedKV = firstSharedKV
    var emitted = 1        // caller already yielded the prefill bonus

    while emitted < maxTokens {
        let remaining = maxTokens - emitted
        let bs = min(blockSize, remaining + 1)
        if bs <= 1 { break }
        let k = bs - 1                              // draft step count

        // --- Draft (k autoregressive steps) ---
        // Drafter queries use the bonus's absolute position, held constant
        // across all k steps. `forwardForMTP` already advanced target cache
        // offset past the prefill, so `driveOffset == targetCache[0].offset`
        // is the position of the bonus token that's about to be verified.
        let driveOffset = targetCache[0].offset
        var draftTokens: [MLXArray] = []
        draftTokens.reserveCapacity(k)
        var tok = MLXArray([[Int32(bonus)]])
        var h = hidden
        for _ in 0 ..< k {
            let tokEmbed = target.model.embedTokens(tok) * targetEmbedScale
            let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
            // Drafter forward returns logits of shape [B, 1, vocab] —
            // squeeze the L=1 axis before sampling.
            let (newH, logits) = drafter(
                inputsEmbeds: inputsEmbeds,
                sharedKV: sharedKV,
                positionOffset: .scalar(driveOffset)
            )
            tok = sampler.sample(logits: logits.squeezed(axis: 1))
            asyncEval(tok)
            draftTokens.append(tok)
            h = newH
        }

        // --- Verify (one target forward over [bonus, ...draftTokens]) ---
        let verifyTokens = concatenated(
            [MLXArray([[Int32(bonus)]])] + draftTokens, axis: 1
        )
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyOut = target.forwardForMTP(verifyInput.tokens, cache: targetCache)
        // verifyOut.logits: [B, bs, vocab]
        let mainTokens = sampler.sample(
            logits: verifyOut.logits.squeezed(axis: 0)
        )                                           // [bs]

        // --- Walk: accept prefix + emit correction ---
        eval(mainTokens)
        let draftInts = concatenated(draftTokens, axis: 1)
            .squeezed(axis: 0).asArray(Int.self)    // [k]
        let mainInts = mainTokens.asArray(Int.self) // [bs]
        let (accepted, newTokens) = SpeculativeWalk.single(
            draft: draftInts, main: mainInts
        )
        precondition(newTokens.count == accepted + 1)

        for tok in newTokens {
            continuation.yield(.chunk(String(tok)))  // tokenizer-decode at emission site
            emitted += 1
            if emitted >= maxTokens { return }
        }

        // --- Rewind target cache for rejected tail ---
        if accepted < k {
            target.rollbackSpeculativeCache(
                targetCache,
                accepted: .scalar(accepted),
                blockSize: bs
            )
        }

        // --- Prepare state for next round ---
        hidden = verifyOut.lastHidden[0..., accepted ..< accepted + 1, 0...]
        bonus = newTokens.last!                    // always non-empty (accepted+1 >= 1)

        // Slice shared-KV to match the rewound target cache length.
        // After rollback, targetCache[0].offset == (prev_offset + accepted + 1);
        // sharedKV is re-captured from verifyOut by trimming the rejected tail.
        let rejected = k - accepted
        sharedKV = Gemma4SharedKV.sliceTail(
            from: verifyOut.capturedSharedKV,
            rejected: rejected
        )

        if emitted % 256 == 0 { MLX.Memory.clearCache() }
    }
}
```

Types referenced in the pseudocode that are specified in this doc:
- `Gemma4SharedKV.sliceTail(from:rejected:)` — pure slicing helper on the
  struct; no state, no MLX ops beyond `[..., ..<n, :]`.
- `forwardForMTP(_:cache:)` overload that takes `MLXArray` directly (the
  spec's signature is on `LMInput.Text`; the round-loop uses the tokens
  array). Both overloads defined on `Gemma4TextModel`, sharing the same
  inner implementation.
- `verifyOut.capturedSharedKV: Gemma4SharedKV` — additional field on
  `Gemma4MTPForward`. The trunk captures after specific layer indices (see
  `Gemma4TextModel` edits §6).

**`SpeculativeWalk.single`** (pure Swift, unit-testable):
```swift
enum SpeculativeWalk {
    /// Greedy accept-prefix match.
    /// Returns (acceptedCount, emittedTokens) where
    /// emittedTokens.count == acceptedCount + 1.
    static func single(draft: [Int], main: [Int]) -> (Int, [Int]) {
        var accepted = 0
        for i in 0 ..< draft.count {
            if main[i] != draft[i] { break }
            accepted += 1
        }
        return (accepted, Array(main[0 ... accepted]))
    }

    /// Per-row version for B>1. Input main/draft are MLXArrays of shape
    /// [B, K]; returned acceptedCounts and newTokens are per row.
    static func batched(draft: MLXArray, main: MLXArray, budgets: [Int])
        -> (accepted: [Int], newTokens: [[Int]])
}
```

**Mutating-state invariants (spec-level, asserted in debug builds):**
- `targetCache[0].offset` increases monotonically, by exactly `accepted + 1`
  per round.
- `hidden.shape[1] == 1` at loop-top.
- `draftTokens.count == bs - 1` before walk.
- `mainTokens.size == bs` (verifyInput includes the bonus at position 0).

## Round-loop (B>1)

Same skeleton as B=1 with three additions:

- `firstBonus: [Int]`, `firstHidden: [B,1,hidden]`, per-row positions tracked
  as `[Int]` updated by `accepted[i] + 1` per round.
- `rollbackSpeculativeCache` called with `.perRow(MLXArray)` → triggers the
  new `BatchKVCache.zeroTailPerRow(keepLengths:)` after the uniform trim.
- Continuous-batching filter when all caches are `BatchedCache`: finished rows
  are removed from the active batch via `BatchKVCache.filter(batchIndices:)`.
  Rows that don't support filter (none today for the Gemma 4 path but listed
  for future-proofing) just stop emitting and stay in the batch.

## `MaskedEmbedder` (centroid-routed sparse LM head)

```swift
public final class MaskedEmbedder: Module {
    @ModuleInfo(key: "centroids") var centroids: Linear   // [hidden → numCentroids]
    @ModuleInfo(key: "token_ordering") var tokenOrdering: MLXArray   // [vocabSize], int32

    public let numCentroids: Int                // 2048
    public let topK: Int                        // 32
    public let vocabSize: Int                   // 262144
    public let vocabSizePerCentroid: Int        // vocabSize / numCentroids = 128

    /// - hiddenStates: [B, L, hidden]
    /// - lmHeadWeight: [vocabSize, hidden]  (tied to drafter embed_tokens.weight)
    /// Returns: [B, L, vocabSize] with non-selected slots set to min(selected)-1
    public func callAsFunction(
        hiddenStates: MLXArray, lmHeadWeight: MLXArray
    ) -> MLXArray
}
```

MLX Swift API notes (verified against existing code paths):
- `argPartition`: use the already-established negate-and-partition-from-front
  idiom (`Evaluate.swift:288`), not `kth=-topK`. Clearer and guaranteed to
  work.
- `putAlong`: used at `Evaluate.swift:269` (topP sampler). Same call shape
  here for the scatter-back-to-vocab step.
- `min(selected).item() - 1` requires a CPU sync. This happens once per
  drafter forward; acceptable for the tiny drafter, but the spec mandates we
  measure this cost in microbench #3. If it shows up >5% of drafter forward
  time, replace with a GPU-only `min - 1` (no `.item()`).

Sanitize:
- `masked_embedding.token_ordering` arrives as int64 in some checkpoints;
  cast to int32 in `sanitize`.
- `lm_head.weight` dropped when `tieWordEmbeddings == true` (matches the
  Python drafter's sanitize).

## `DrafterMasks` (bidirectional full / SWA)

```swift
enum DrafterMasks {
    /// Returns `.none` in the common case; `.array(...)` only when the SWA
    /// window doesn't cover the full KV seen by any query position.
    static func make(
        sharedKV: Gemma4SharedKV,
        queryLen: Int,
        queryOffset: Int,
        slidingWindow: Int,
        dtype: DType
    ) -> (full: MLXFast.ScaledDotProductAttentionMaskMode,
          sliding: MLXFast.ScaledDotProductAttentionMaskMode)
}
```

Port of Blaizzy's `make_drafter_masks`. The SWA path is effectively dead code
when the target uses `RotatingKVCache` (kv_len always ≤ sliding_window), but
the spec includes it for long-prompt correctness. The full-attention path
always returns `.none` (SDPA handles bidirectional).

## Weight loading

`Gemma4AssistantDraftModel.load(from:using:id:...)` is a minimal free
function wrapping `Downloader.download` + safetensors read + construct +
`sanitize`. No factory / registry.

Config parsing:
- Top level: `backbone_hidden_size`, `use_ordered_embeddings`,
  `num_centroids`, `centroid_intermediate_top_k`, `tie_word_embeddings`,
  `block_size`, nested `text_config` (which is a `Gemma4TextConfiguration`
  with a few drafter-specific defaults).
- Post-init clamp: if `text_config.num_kv_shared_layers` is 0 or missing,
  set it to `text_config.num_hidden_layers`. Matches HF behavior.

Sanitize rules (drafter-specific):
- Keep `pre_projection.weight`, `post_projection.weight`.
- Keep `centroids.weight`, cast `token_ordering` to int32.
- Drop `lm_head.weight` iff `tie_word_embeddings` is true.
- Every `model.layers.{i}.self_attn.{q_proj,q_norm,o_proj}.weight` stays.
- `k_proj`, `v_proj`, `k_norm`, `v_norm` must NOT be present — since every
  layer is kv-shared, these aren't instantiated. Present weights trigger
  `loadWeights` mismatch error → fail loud.

## Testing strategy

Tests live in `Tests/MLXLMTests` with `MLXSpeculative` added as a dependency.
Two tiers: **unit** (fast, CI-eligible, mock weights) and **integration**
(real weights, local-only, marked `.tags(.requiresWeights)` so they skip in CI).

### Unit (mock weights, random init, fast)

1. **`SpeculativeWalkTests`** — property-based on `single(draft:main:)`:
   - Empty draft → (0, [main[0]]).
   - All match → (`draft.count`, `draft + [bonus]`).
   - First mismatch at position `k` → (`k`, `draft[0..<k] + [main[k]]`).
   - Batched variant: per-row equivalence to calling `single` per row.

2. **`MaskedEmbedderTests`** — with small random weights:
   - Output shape is `[B, L, vocabSize]`.
   - Non-selected slots all equal `min(selected) - 1`.
   - Argmax selects a token whose centroid is in the top-K scoring
     centroids.
   - Gradient-free (no training); compare against a reference Python snippet
     run once offline and frozen as a test fixture (JSON of (hidden,
     lm_head_weight) → (selected_logits, argmax)).

3. **`Gemma4AssistantDraftForwardTests`** — random weights, small dims:
   - `bind(target:)` then one drafter forward produces finite logits.
   - Shape checks: hidden `[1,1,backbone]`, logits `[1,1,vocab]`.
   - Compatibility validation rejects: mismatched `backboneHiddenSize`,
     vocab size, layer type.
   - **Softcap absence test:** construct a drafter whose weights would produce
     logit magnitudes > 30 in at least one position. Verify output logits
     also exceed 30 (softcap NOT applied). Compare vs the target on the same
     hidden: target's `callAsFunction` output must be bounded by 30 in
     absolute value. This locks in the spec invariant that the drafter's
     `applyLMHead` is separate from the target's.

4. **`Gemma4MTPRoundLoopTests`** — random weights, small dims, end-to-end
   loop with a mock tokenizer:
   - `accepted == blockSize - 1` every round when drafter == target
     (same weights, same sampler) → all drafts accepted. This is the
     "oracle" case and should emit exactly `blockSize` new tokens per round.
   - `accepted == 0` when drafter is zeroed out / forced to wrong tokens →
     still correct output (falls back to target one-token-at-a-time).
   - `targetCache[0].offset` invariant after each round.

5. **`BatchKVCacheZeroTailTests`** — the one new primitive:
   - `keepLengths = [_idx, ..., _idx]` → no-op.
   - `keepLengths = [0, ..., 0]` → all keys/values zeroed.
   - Mixed per-row → slots ≥ `keepLengths[b]` zeroed, slots < kept unchanged.

### Integration (real weights, local only)

6. **`Gemma4E4BMTPParityTest`** — the ironclad correctness test.
   - Load `mlx-community/gemma-4-e4b-it-bf16` + its `-assistant-bf16`.
   - Prompt set, fixed and committed alongside the test under
     `Tests/MLXLMTests/Resources/Gemma4MTPPrompts.json`:
     - 5 **short** prompts (≤ 64 tokens): single-sentence questions.
     - 10 **medium** prompts (256–512 tokens): paragraph-length context +
       instruction.
     - 5 **long** prompts (1024–2048 tokens): multi-paragraph documents
       (stays under the E4B `sliding_window=512` long-tail regime so we
       exercise the SWA mask path).
   - Sweep: 3 block sizes (2, 3, 4) × {B=1, B=4}. Greedy only at v1
     (`temperature=0`, no sampling); stochastic parity is a follow-up.
   - For each configuration: run baseline (no drafter) at `temperature=0`,
     then MTP at `temperature=0`, assert **exact token-sequence equality**.
   - Total: 20 prompts × 3 block_sizes × 2 batch_sizes = 120 comparisons.
   - This is the acceptance gate for merging. A single byte-divergent case
     fails the suite.

   *What this does NOT test:* stochastic sampling parity. Greedy byte-equality
   is the only hard guarantee the drafter can make; at `temp > 0` the drafter
   and target see the same random state but sampled token draws diverge because
   each model's logits differ slightly — this is expected and matches the
   Python reference's stated behavior.

7. **`Gemma4E2BMTPParityTest`** — same shape, smaller model for
   faster iteration. Run first when developing.

8. **`Gemma4_26BA4B_4bit_MTPParityTest`** — stretch tier, fewer prompts
   (5), B=1 only (memory), still greedy-identical.

### Microbenchmarks

Live in `Libraries/BenchmarkHelpers`. Not unit tests — invoked via a CLI
harness locally. Results recorded in `BenchmarkHelpers/Gemma4MTPBaseline.swift`
as constants alongside the commit + date they were measured.

9. **Per-round cost breakdown.** Instrument one MTP round with `measure`
   around: drafter autoregressive loop, target verify forward, speculative
   walk, rollback + per-row zero-tail, shared-KV slicing. Table output.

10. **`MaskedEmbedder` microbench.** Wall-clock vs dense
    `hidden @ embed.T`. Sentinel: the centroid path must be faster than
    dense on the E4B drafter (otherwise it's net negative).

11. **End-to-end throughput sweep** — the "Apple-facing" number.
    Configurations:
    - E4B + E4B-assistant @ bf16, B ∈ {1, 4}, block ∈ {2, 3, 4}, max_tokens=128
    - E2B + E2B-assistant @ bf16, same sweep
    - 26B-A4B @ 4-bit + 26B-A4B-assistant @ bf16, B=1, block ∈ {3}, max_tokens=64
    Record: prompt tok/s, decode tok/s, speedup vs no-drafter baseline,
    mean accepted/round, std deviation across 5 runs.

### Oracle runs (one-time, frozen)

On this 36 GB M-series box, run the Python reference
(`Blaizzy/mlx-vlm` at `244f4bb`) over the exact same prompt set
(`Gemma4MTPPrompts.json`) and measure:

- Accepted-tokens-per-round histogram (primary correctness signal).
- End-to-end tokens/sec per configuration.

Python runs use the exact same checkpoints as Swift:
- Target in bf16 (primary) or 4-bit (stretch) — matching the Swift config.
- Drafter always in bf16 (matches Swift; 4-bit drafters are HF-gated).
- Greedy sampling (`temp=0`), no top-p / top-k / penalties.
- Same `blockSize` values (2, 3, 4).
- `draft_kind=mtp` on the Python CLI.

Freeze the Python numbers in
`BenchmarkHelpers/Gemma4MTPBaseline.swift` as the Python reference line,
alongside:
- `blaizzyMlxVlmCommit = "244f4bb5a3339b180da3d2b276a4bdfcf7670f9f"`
- `measuredOn = "M-series 36 GB, <Mac model>, <macOS version>, <MLX version>"`
- `measuredAt = "YYYY-MM-DD"` (recorded in git blame alongside the constants).

Swift must meet:

- **Correctness:** per-round accepted histogram matches Python's to within
  statistical noise (Kolmogorov–Smirnov two-sample test, p > 0.05 over
  ≥ 100 rounds per configuration) AND greedy-identical output (hard
  token-for-token equality).
- **Performance:** Swift end-to-end tokens/sec ≥ 0.9× Python's on the same
  box, measured as the mean over 5 runs per configuration after a 1-run
  warmup. Below this threshold, the benchmark suite fails and we hunt the
  regression (most likely suspects in order: async-eval pipelining in the
  draft loop, `MaskedEmbedder` CPU sync, rollback broadcast shape).

## Error types

```swift
public enum Gemma4MTPError: LocalizedError {
    case unsupportedTarget(String)                    // target isn't Gemma4TextModel
    case rebindForbidden                              // bind() called with different target
    case incompatibleDrafter(field: String, drafter: String, target: String)
    case invalidBlockSize(Int)                        // blockSize < 2 or > 16
    case drafterNotBound                              // forward called before bind
}
```

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Async-eval pipelining in drafter autoregressive loop | Silent 3× perf loss | Explicit `asyncEval()` after each draft `tok`. Instruments trace on first run. Perf bench #9. |
| `MaskedEmbedder` CPU sync on `.item()` | Per-step stall | Measure in #10; if >5% of drafter forward, rewrite to GPU-only `min - 1`. |
| `BatchKVCache.zeroTailPerRow` broadcast shape bug | Silent corruption in B>1 | Test #5 covers shape invariants. Property test with `keepLengths == _idx` must be exact no-op (no float drift). |
| Gemma 4 `usesSharedKV` guard change breaks target 35-layer models | Regression in non-MTP usage | Existing test suite must pass unchanged. Explicit test case for target layer counts. |
| Drafter 4-bit gated on HF → can't test fully-4-bit pair | Benchmark gap | Documented in hardware-targets table. Primary benchmark uses bf16 drafter; stretch uses 4-bit target + bf16 drafter. No fully-4-bit config claimed. |
| Centroid head numerical drift (bf16 `matmul` + `min().item()` round-off) | Greedy output divergence at temp=0 | Parity test asserts hard equality on token IDs, not logit values. If logits diverge but arg-max stays, we're fine. |
| Softcap accidentally applied in drafter path | Wrong logits | Drafter has its own `applyLMHead`, not shared with target. Test #3 asserts no softcap by comparing to a reference computation. |

## Out-of-scope (explicit, for reviewer clarity)

- Drafter for Qwen 3.5 / DeepSeek-V4 / MiMo. Their `mtp.*` weights continue
  to be dropped by existing `sanitize` calls.
- A `DraftModel` protocol. When the second MTP drafter arrives, we extract
  the protocol from two concrete cases, not one.
- `SpeculativeCaptureHost` protocol. Capture lives as a public method on
  `Gemma4TextModel`, nothing more.
- Integration with `BatchGenerator`'s continuous-batching engine.
  `Gemma4MTPRoundLoop` is a parallel path.
- `ChatSession` wrapper. Users wanting multi-turn + MTP compose the two:
  `ChatSession` around a `ModelContainer` works unchanged for the prefill
  side; MTP activates on the decode stream via a future
  `ChatSession.respondWithMTP(...)` convenience — not v1.
- 31B and 26B-A4B @ bf16 targets. Can't benchmark on 36 GB hardware.

## Acceptance criteria for merge

All of the following must be green before this ships:

1. Every unit test in §Testing passes under `swift test` (clean, no flakes).
2. `Gemma4E4BMTPParityTest` passes on the 36 GB M-series box. Byte-identical
   token streams across 20 prompts × 3 block sizes × 2 batch sizes = 120
   configurations.
3. Python oracle baselines recorded in
   `BenchmarkHelpers/Gemma4MTPBaseline.swift` alongside the commit sha of
   `Blaizzy/mlx-vlm` used.
4. Swift throughput ≥ 0.9× the recorded Python baseline for E4B + E4B-assistant
   at B=1 block=3 and B=4 block=3.
5. `swift build -c release` succeeds with no new warnings in `MLXSpeculative`
   or the touched section of `Gemma4Text.swift`.
6. Existing test suite (`Tests/MLXLMTests` minus the new MTP tests) passes
   unchanged, verifying the `Gemma4Text.swift` edits are non-regressive.
7. No new public types in `MLXLMCommon`. Everything MTP-specific lives in
   `MLXSpeculative`.

## Follow-ups (tracked, explicitly not v1)

- `ChatSession.respondWithMTP(...)` convenience.
- Extract `DraftModel` protocol once a second concrete drafter lands.
- Integrate MTP into `BatchGenerator` as a drafting mode.
- Support for non-Gemma-4 MTP drafters (Qwen 3.5, DeepSeek-V4).
- CLI tool surfacing MTP via `llm-tool` in mlx-swift-examples.
