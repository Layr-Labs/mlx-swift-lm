// CompiledDecode: whole-step compiled decode helper.
//
// Ported (trimmed) from osaurus-ai/vmlx-swift-lm
// (Libraries/MLXLMCommon/BatchEngine/BatchCompile.swift `compileForward`).
//
// This wraps MLX `compile(inputs:outputs:)` so a single decode step — the model
// forward plus the per-layer KV-cache write — is captured as one compiled graph,
// collapsing hundreds of FFI crossings into a single compiled call. The tracer
// captures each cache layer's `innerState()`; subsequent invocations mutate the
// captured cache objects in place via `_updateInternal`.
//
// REQUIREMENTS / CONSTRAINTS (why this is currently UNWIRED):
// - Every layer must be a ``CompilableKVCache`` (fixed-shape, MLXArray offset).
//   Standard `KVCacheSimple` / `RotatingKVCache` change state shape per step and
//   cannot be compile-traced (that is the whole reason ``CompilableKVCache``
//   exists). Sliding-window (SWA) models therefore additionally need a
//   `CompilableRotatingKVCache` — see the port plan — before they can use this.
// - The trace specialises on the token-batch shape it first sees (typically
//   `[B, 1]`). A changing batch size forces a recompile, so the batched decode
//   path needs fixed-size buckets (see the port plan for `GenerationBatch`).
//
// This helper is dependency-free w.r.t. the continuous-batching engine: it can
// be exercised in isolation (see CompilableKVCacheTests) and reused by either a
// single-stream or a batched decode loop once the cache-promotion + bucketing
// plumbing lands.

import Foundation
import MLX

public enum CompiledDecode {

    /// True iff every layer is a ``CompilableKVCache`` and thus compile-traceable
    /// by ``compileForward(model:cacheRef:)``.
    public static func eligible(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy { $0 is CompilableKVCache }
    }

    /// Build a compiled forward closure for a decode step.
    ///
    /// The returned closure accepts `[tokens]` (a single `[B, L]` int token
    /// array wrapped in a one-element array) and returns `[logits]` (a single
    /// `[B, L, V]` array). The captured cache layers are mutated in place.
    ///
    /// - Precondition: `cacheRef` is non-empty and every element is a
    ///   ``CompilableKVCache`` (see ``eligible(_:)``). Call `eval(cacheRef)`
    ///   before this so no pending tracer ops corrupt state identity.
    ///
    /// - Parameters:
    ///   - model: The language model to trace through.
    ///   - cacheRef: Per-layer ``CompilableKVCache`` instances. Captured by the
    ///     returned closure; must not be empty.
    /// - Returns: A `@Sendable` closure mapping `[tokens]` → `[logits]`.
    public static func compileForward(
        model: any LanguageModel,
        cacheRef: [KVCache]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        precondition(
            eligible(cacheRef),
            "CompiledDecode.compileForward requires a non-empty cache where every "
                + "layer is a CompilableKVCache.")

        let capturedModel = model
        let captured = cacheRef

        return compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0]),
                cache: captured.isEmpty ? nil : captured,
                state: nil
            )
            return [result.logits]
        }
    }
}
