// SamplerV2.swift
//
// Vectorized token selection for ContinuousBatchingV2 (workstream E).
// Consumes the transformed [B, vocab] logits produced by
// `LogitsPipelineV2.process` and returns one token per row [B].
//
//  - All-greedy fast path: a single `argMax` when every row is greedy
//    (temperature < 1e-5), matching the legacy vectorized greedy path
//    bit for bit.
//  - Mixed batches: Gumbel-max via the exponential-noise race
//    (`argmax(probs / e)`, e ~ Exp(1)) so there is no multinomial and no
//    host sync. Greedy and sampled picks are merged with a per-row
//    `where(greedy, argmax, sampled)`.
//  - Per-row keyed RNG: each row's noise is generated from its own key,
//    derived ONLY from (seed, requestID, per-request step index). A row's
//    random stream therefore never depends on its batchmates or on when it
//    joined the batch — the batch-composition-invariance requirement
//    (research report 12, item 5). Reproducibility is best-effort under
//    batching, per the contract: fixed (seed, requestID) reproduces the
//    same stream across runs given the same per-row logits.
//
// Threading: `sample` builds graph nodes only (no eval, no `.item()`).
// `setRows` runs at batch-membership change; `commit` is O(B) host counter
// bookkeeping (no device work).

import Cmlx
import Foundation
import MLX
import MLXRandom

public final class SamplerV2 {

    /// Temperatures below this are treated as greedy (matches vLLM and
    /// `LogitsPipelineV2.greedyEpsilon`).
    public static let greedyEpsilon: Float = LogitsPipelineV2.greedyEpsilon

    /// Engine-level fallback used for rows that did not supply a seed.
    /// Fixed at init so nil-seed rows are still batch-invariant within a
    /// process; across runs they are intentionally non-deterministic.
    private let fallbackSeed: UInt64

    private struct RowState {
        var id: CBv2RequestID
        var seed: UInt64
        var greedy: Bool
        /// Per-request decode step index (number of tokens sampled so far
        /// for this request). Keying noise to the per-request index — not
        /// the global engine step — is what keeps a row's stream
        /// independent of when it joined the batch.
        var step: UInt64
    }

    private var rows: [RowState] = []
    private var allGreedy = true
    /// True when at least one row is greedy (the per-row merge is needed).
    private var anyGreedy = false
    /// [B] bool, built once per membership change.
    private var greedyMask: MLXArray?

    public init(fallbackSeed: UInt64? = nil) {
        self.fallbackSeed = fallbackSeed ?? UInt64.random(in: .min ... .max)
    }

    // MARK: Membership change

    /// Rebuild per-row sampling state. Row order must match the logits row
    /// order passed to `sample`. Mid-flight rows resume their step index
    /// from `outputTokens.count`, so re-vectorization after a membership
    /// change does not perturb their random streams.
    public func setRows(_ rows: [CBv2SamplerRow]) {
        self.rows = rows.map { row in
            RowState(
                id: row.id,
                seed: row.params.seed ?? fallbackSeed,
                greedy: row.params.temperature < Self.greedyEpsilon,
                step: UInt64(row.outputTokens.count)
            )
        }
        allGreedy = self.rows.allSatisfy(\.greedy)
        anyGreedy = self.rows.contains(where: \.greedy)
        greedyMask = self.rows.isEmpty ? nil : MLXArray(self.rows.map(\.greedy))
    }

    // MARK: Step path

    /// Select one token per row from transformed logits [B, vocab].
    /// Returns [B] int32. Pure graph construction — call `commit()` once
    /// per step afterwards to advance the per-row step indices.
    public func sample(from logits: MLXArray) -> MLXArray {
        precondition(
            logits.dim(0) == rows.count,
            "logits rows (\(logits.dim(0))) != configured rows (\(rows.count)) — call setRows")

        // Fast path: every row greedy ⇒ one argMax, bit-identical to the
        // legacy vectorized greedy decode.
        if allGreedy {
            return argMax(logits, axis: -1).asType(.int32)
        }

        // Mixed batch: exponential-race Gumbel-max. probs/e keeps masked
        // tokens (-inf logits → probability 0) unreachable, and argmax of
        // the ratio is an exact categorical draw.
        let vocab = logits.dim(-1)
        let probs = softmax(logits, axis: -1)
        let noise = exponentialNoise(vocab: vocab)
        let sampledTokens = argMax(probs / noise, axis: -1).asType(.int32)

        // Without greedy rows the merge selects every sampled pick, so the
        // greedy argmax is never built.
        guard anyGreedy, let greedyMask else { return sampledTokens }
        let greedyTokens = argMax(logits, axis: -1).asType(.int32)
        return which(greedyMask, greedyTokens, sampledTokens)
    }

    /// Advance every row's per-request step index. Call exactly once per
    /// sampled step, after `sample` (alongside `LogitsPipelineV2.commit`).
    public func commit() {
        for i in rows.indices {
            rows[i].step &+= 1
        }
    }

    /// Per-row step indices (RNG key inputs), row-aligned with `setRows`.
    var stepIndices: [UInt64] { rows.map(\.step) }

    /// Restore indices captured under the SAME `setRows` configuration, so
    /// a redrawn step uses exactly the keys of the draw it replaces.
    func restoreStepIndices(_ steps: [UInt64]) {
        precondition(steps.count == rows.count, "step snapshot does not match configured rows")
        for i in rows.indices { rows[i].step = steps[i] }
    }

    // MARK: - Per-row keyed noise

    /// Exp(1) noise [B, vocab]; row r is generated from key
    /// mix(seed_r, id_r, step_r), so each row's draw is a pure function of
    /// that row's identity — never of batchmates. Greedy rows receive a
    /// placeholder (their sampled pick is discarded by the `where` merge).
    private func exponentialNoise(vocab: Int) -> MLXArray {
        Self.exponentialNoise(
            keys: rows.map {
                $0.greedy ? nil : Self.mix(seed: $0.seed, id: $0.id.raw, step: $0.step)
            },
            vocab: vocab)
    }

    /// MTP verify pre-sampling noise: one Exp(1) row per (request, window
    /// position), keyed EXACTLY like the step path —
    /// `mix(seed ?? fallbackSeed, id, step)` with the same uniform→-log(1-u)
    /// transform and zero clamp — so a verify position's draw is bitwise the
    /// draw the ordinary path would have made at that per-request step.
    /// Greedy rows receive the same placeholder as `sample`.
    func verifyExponentialNoise(
        rows: [(seed: UInt64?, id: UInt64, step: UInt64, greedy: Bool)], vocab: Int
    ) -> MLXArray {
        Self.exponentialNoise(
            keys: rows.map {
                $0.greedy ? nil : Self.mix(seed: $0.seed ?? fallbackSeed, id: $0.id, step: $0.step)
            },
            vocab: vocab)
    }

    /// Row r (non-nil key) is `max(-log(1 - u), 1e-20)` with
    /// `u = MLXRandom.uniform(low: 0, high: 1, [1, vocab], key: key(keys[r]))`,
    /// bit for bit. MLX's uniform is `bits / UInt32.max` clamped below 1 and
    /// then mapped through `range * u + low` with range 1 and low 0 — exact
    /// identities on u ∈ [0, 1), so they are skipped. Only the random bits
    /// are per row; the rest runs once over the whole batch (elementwise
    /// results do not depend on the row they sit in). A nil key (greedy row)
    /// gets zero bits: its sampled pick is discarded.
    ///
    /// The cast, divide, clamp and `1 - u` run as one kernel, and the
    /// negate and clamp as another, each repeating MLX's own per-element
    /// operation: uint→float conversion, a divide by 2^32 (exact), MLX's
    /// `Minimum`/`Maximum` selects, an IEEE subtraction and a sign flip —
    /// none of which can round differently in another kernel. The log stays
    /// MLX's precompiled kernel (`precise::log` is not correctly rounded, so
    /// a recompiled copy could differ).
    static func exponentialNoise(keys: [UInt64?], vocab: Int) -> MLXArray {
        var bits = [MLXArray]()
        bits.reserveCapacity(keys.count)
        for key in keys {
            if let key {
                bits.append(randomBits(shape: [1, vocab], key: MLXRandom.key(key)))
            } else {
                bits.append(MLXArray.zeros([1, vocab], type: UInt32.self))
            }
        }
        let raw = bits.count == 1 ? bits[0] : concatenated(bits, axis: 0)
        // e = -log(1 - u) ~ Exp(1); u ∈ [0, 1) keeps the log argument in
        // (0, 1]. Clamp away e == 0 (u == 0) so probs/e never divides by
        // zero.
        let complement = elementwise(oneMinusUniformKernel, raw)
        return elementwise(negatedFloorKernel, log(complement))
    }

    private static func elementwise(_ kernel: MLXFast.MLXFastKernel, _ input: MLXArray)
        -> MLXArray
    {
        kernel(
            [input], grid: (input.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [input.shape], outputDTypes: [.float32]
        )[0]
    }

    /// `1 - minimum(float(bits) / Float(UInt32.max), Float(1).nextDown)`;
    /// `Float(UInt32.max)` rounds to 2^32 (0x4f800000).
    private static let oneMinusUniformKernel = MLXFast.metalKernel(
        name: "cbv2_sampler_one_minus_uniform",
        inputNames: ["bits"],
        outputNames: ["out"],
        source: """
            uint elem = thread_position_in_grid.x;
            float u = static_cast<float>(bits[elem]) / as_type<float>(0x4f800000u);
            float upper = as_type<float>(0x3f7fffffu);
            u = metal::isnan(u) ? u : (u < upper ? u : upper);
            out[elem] = 1.0f - u;
            """)

    /// `maximum(-x, Float(1e-20))` with MLX's NaN-first `Maximum` select;
    /// `Float(1e-20)` is 0x1e3ce508.
    private static let negatedFloorKernel = MLXFast.metalKernel(
        name: "cbv2_sampler_negated_floor",
        inputNames: ["x"],
        outputNames: ["out"],
        source: """
            uint elem = thread_position_in_grid.x;
            float e = -x[elem];
            float floor_value = as_type<float>(0x1e3ce508u);
            out[elem] = metal::isnan(e) ? e : (e > floor_value ? e : floor_value);
            """)

    /// `mlx::core::random::bits(shape, 4, key)`: the uint32 stream MLX's
    /// `uniform` draws from.
    static func randomBits(shape: [Int], key: MLXArray) -> MLXArray {
        var result = mlx_array_new()
        let rc = mlx_random_bits(
            &result, shape.map(Int32.init), shape.count, 4, key.ctx,
            StreamOrDevice.default.ctx)
        precondition(rc == 0, "mlx_random_bits failed with rc=\(rc)")
        return MLXArray(result)
    }

    /// SplitMix64-style mix of (seed, requestID, step) into one RNG key
    /// seed. Deterministic across processes (no `Hasher`), and a pure
    /// function of exactly the three inputs the contract allows.
    static func mix(seed: UInt64, id: UInt64, step: UInt64) -> UInt64 {
        func splitmix(_ value: UInt64) -> UInt64 {
            var z = value &+ 0x9E37_79B9_7F4A_7C15
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        return splitmix(splitmix(splitmix(seed) ^ id) ^ step)
    }
}
