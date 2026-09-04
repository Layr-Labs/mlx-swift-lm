// GPU tensor-level parity harness for the D=512 decode attention cell (C1).
//
// WHY THIS EXISTS. Rung 2 (the D512 two-pass vector SDPA) diverged from the
// control on all 7 greedy prompts, two of them inside the first ~40 tokens.
// That is either (i) the ordinary bf16-level difference between two correct
// attention algorithms, or (ii) a bug in the 512 instantiation -- a wrong
// scale, a wrong GQA head mapping, or a wrong cross-block max/denominator
// merge. A token diff cannot tell those apart. This measures the kernel
// directly against an fp32 reference, and -- the part that actually decides
// it -- against the SAME metric for MLX's own SHIPPED head_dim-256 two-pass
// kernel versus its own unfused path. If our 512 error sits at the 256
// kernel's level, MLX ships this behaviour for every long-context GQA decode
// and the 512 cell is not special.
//
// This test needs a GPU and must run under the flock. It is inert unless
// DARKBLOOM_D512_PARITY=1, so the ordinary CPU-only suite is unaffected.
//
// One run, with DARKBLOOM_GEMMA4_D512_DECODE_2PASS at its default. Setting it
// to 0 is still meaningful -- it makes MLX decline head dim 512 and every
// two-pass cell reports SKIPPED rather than silently measuring the fallback
// against itself -- but it yields no extra information now that the composed
// batch-1 chain is not part of this PR.

import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest

final class Gemma4D512AttentionParityTests: XCTestCase {

    // MARK: - Harness configuration

    private static func setting(_ name: String) -> String? {
        ProcessInfo.processInfo.environment[name]
    }

    private var enabled: Bool { Self.setting("DARKBLOOM_D512_PARITY") == "1" }

    /// Whether MLX will admit head_dim 512 to its fused vector SDPA in THIS
    /// process. Mirrors `d512_vector_sdpa_enabled()` in
    /// `mlx/backend/metal/scaled_dot_product_attention.cpp`, which is a
    /// memoized `static bool` -- so on the rung-1 pass every D=512 SDPA call
    /// silently falls back to the unfused graph. Cells (a), GQA and scale
    /// would then be measuring the fallback against itself while labelled
    /// "two-pass", which is worse than not measuring them: they must be
    /// SKIPPED, not reported.
    private var twoPassAdmitted: Bool {
        guard let raw = Self.setting("DARKBLOOM_GEMMA4_D512_DECODE_2PASS") else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }

    /// Key length. Defaults to THE TEST's 17,408.
    private var keyLength: Int {
        Int(Self.setting("DARKBLOOM_D512_PARITY_KEYS") ?? "") ?? 17408
    }

    private var outputPath: String {
        Self.setting("DARKBLOOM_D512_PARITY_OUT") ?? "/tmp/d512-parity.json"
    }

    // MARK: - Metrics

    private struct Err {
        var maxAbs: Double
        var maxRel: Double
        var refMaxAbs: Double
        var nonFinite: Int
    }

    /// max |a - b|, and that divided by the reference's own magnitude SCALE.
    /// Relative error is taken against the scale rather than element-wise so a
    /// near-zero output element cannot manufacture an enormous ratio.
    /// Computed on the host from fp32 copies, so no MLX reduction can hide a
    /// NaN and a shape mismatch traps instead of broadcasting.
    private func compare(_ candidate: MLXArray, _ reference: MLXArray) -> Err {
        let a = candidate.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        precondition(a.count == b.count, "shape mismatch \(a.count) vs \(b.count)")
        var maxAbs = 0.0
        var maxRel = 0.0
        var refMax = 0.0
        var nonFinite = 0
        for value in b where value.isFinite {
            refMax = Swift.max(refMax, Double(abs(value)))
        }
        let denom = refMax > 0 ? refMax : 1.0
        for i in 0 ..< a.count {
            if !a[i].isFinite || !b[i].isFinite {
                nonFinite += 1
                continue
            }
            let d = Double(abs(a[i] - b[i]))
            maxAbs = Swift.max(maxAbs, d)
            maxRel = Swift.max(maxRel, d / denom)
        }
        return Err(maxAbs: maxAbs, maxRel: maxRel, refMaxAbs: refMax, nonFinite: nonFinite)
    }

    private func encode(_ e: Err) -> [String: Any] {
        [
            "maxAbs": e.maxAbs, "maxRel": e.maxRel,
            "refMaxAbs": e.refMaxAbs, "nonFinite": e.nonFinite,
        ]
    }

    // MARK: - Reference implementations

    /// MLX's UNFUSED fallback, transcribed from the `fallback` lambda in
    /// `mlx/fast.cpp` (`scaled_dot_product_attention`). This is the graph the
    /// engine runs today for the global layers, so it is the production
    /// control -- and because it is hand-built it never calls SDPA, so it is
    /// available in both runs regardless of the switches.
    ///
    /// `kvIndexOverride` replaces the implicit `kv = head / nRepeats` GQA
    /// mapping with an explicit per-query-head KV index, which is how the
    /// mapping gets tested rather than assumed.
    private func unfusedGraph(
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float,
        dtype: DType, stream: StreamOrDevice, kvIndexOverride: [Int]? = nil
    ) -> MLXArray {
        let B = q.dim(0), H = q.dim(1), L = q.dim(2), D = q.dim(3)
        let kvHeads = k.dim(1)
        let scaleArray = MLXArray(scale).asType(dtype, stream: stream)
        let qs = q.asType(dtype, stream: stream) * scaleArray

        if let kvIndexOverride {
            // Explicit mapping: gather K/V per query head, then one plain
            // 3-D batched matmul. Same arithmetic, no GQA assumption.
            precondition(kvIndexOverride.count == H)
            let idx = MLXArray(kvIndexOverride.map { Int32($0) })
            let kt = k.asType(dtype, stream: stream)[0].take(idx, axis: 0, stream: stream)
            let vt = v.asType(dtype, stream: stream)[0].take(idx, axis: 0, stream: stream)
            let scores = matmul(
                qs.reshaped([H, L, D], stream: stream),
                kt.swappedAxes(-1, -2, stream: stream), stream: stream)
            let probs = softmax(scores, axis: -1, precise: true, stream: stream)
            return matmul(probs, vt, stream: stream).reshaped([B, H, L, D], stream: stream)
        }

        let nRepeats = H / kvHeads
        let q5 = qs.reshaped([B, kvHeads, nRepeats, L, D], stream: stream)
        let k5 = k.asType(dtype, stream: stream).expandedDimensions(axis: 2, stream: stream)
        let v5 = v.asType(dtype, stream: stream).expandedDimensions(axis: 2, stream: stream)
        let scores = matmul(q5, k5.swappedAxes(-1, -2, stream: stream), stream: stream)
        let probs = softmax(scores, axis: -1, precise: true, stream: stream)
        let out = matmul(probs, v5, stream: stream)
        return out.reshaped([B, H, L, D], stream: stream)
    }

    /// (d) the fp32 reference, on the CPU stream so it shares no kernel with
    /// anything under test. The inputs are bf16 and therefore exact in fp32,
    /// so this is ~1e-7 relative against exact arithmetic -- four orders below
    /// bf16's ~4e-3 ulp, which is all the headroom the comparison needs.
    private func referenceFP32(
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float, kvIndexOverride: [Int]? = nil
    ) -> MLXArray {
        let out = unfusedGraph(
            q: q, k: k, v: v, scale: scale, dtype: .float32, stream: .cpu,
            kvIndexOverride: kvIndexOverride)
        eval(out)
        return out
    }

    // MARK: - Inputs

    /// `qScale` sets the softmax temperature, which controls how much
    /// cancellation the AV sum carries and therefore how far two orderings can
    /// diverge. Both regimes are reported:
    ///   "diffuse" -- scores ~ N(0, 1): a broad softmax, many comparable terms.
    ///   "peaked"  -- scores ~ N(0, sqrt(D)): the regime the engine is in,
    ///                since Gemma folds its scale into the q norm and passes
    ///                scale = 1.0 (Gemma4Text.swift:1934).
    private func makeInputs(headDim: Int, keys: Int, qScale: Float)
        -> (q: MLXArray, k: MLXArray, v: MLXArray)
    {
        MLXRandom.seed(20260903)
        let q = (MLXRandom.normal([1, 16, 1, headDim]) * qScale).asType(.bfloat16)
        let k = MLXRandom.normal([1, 2, keys, headDim]).asType(.bfloat16)
        let v = MLXRandom.normal([1, 2, keys, headDim]).asType(.bfloat16)
        eval(q, k, v)
        return (q, k, v)
    }

    // MARK: - The harness

    func testD512AttentionParity() throws {
        try XCTSkipUnless(enabled, "set DARKBLOOM_D512_PARITY=1 (GPU, under the flock)")

        var report: [String: Any] = [:]
        report["keyLength"] = keyLength
        report["twoPassSwitch"] =
            Self.setting("DARKBLOOM_GEMMA4_D512_DECODE_2PASS") ?? "(unset -> on)"
        report["composedSwitch"] =
            Self.setting("DARKBLOOM_GEMMA4_D512_DECODE_COMPOSED") ?? "(unset -> on)"
        report["twoPassAdmitted"] = twoPassAdmitted

        let regimes: [(String, Float)] = [
            ("diffuse", 1.0 / Float(512).squareRoot()),
            ("peaked", 1.0),
        ]

        for (regime, qScale) in regimes {
            var cell: [String: Any] = [:]

            // ---------- head_dim 512: the cell under test ----------
            let (q, k, v) = makeInputs(headDim: 512, keys: keyLength, qScale: qScale)
            let ref = referenceFP32(q: q, k: k, v: v, scale: 1.0)

            // (c) the production unfused path, in bf16 on the GPU.
            let unfused = unfusedGraph(
                q: q, k: k, v: v, scale: 1.0, dtype: .bfloat16, stream: .gpu)
            eval(unfused)
            cell["c_unfused_vs_fp32"] = encode(compare(unfused, ref))

            // (a) our D512 two-pass, at several split-K partitions. The
            // partition IS the merge the coordinator asked about: if the
            // cross-block max/denominator fold were wrong, the error would
            // TRACK `blocks` instead of staying flat.
            if twoPassAdmitted {
                var blockCells: [String: Any] = [:]
                for blocks in ["default", "32", "256", "512", "1024"] {
                    if blocks == "default" {
                        unsetenv("MLX_SDPA_BLOCKS")
                    } else {
                        setenv("MLX_SDPA_BLOCKS", blocks, 1)
                    }
                    let fused = MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v, scale: 1.0, mask: .none)
                    eval(fused)
                    var entry = encode(compare(fused, ref))
                    entry["vs_unfused"] = encode(compare(fused, unfused))
                    blockCells[blocks] = entry
                }
                unsetenv("MLX_SDPA_BLOCKS")
                cell["a_twoPass_vs_fp32_byBlocks"] = blockCells
            } else {
                cell["a_twoPass_vs_fp32_byBlocks"] =
                    "SKIPPED: head dim 512 is not admitted to the fused vector path in "
                    + "this process, so MLXFast.scaledDotProductAttention would return "
                    + "the unfused fallback"
            }

            // ---------- the control: MLX's SHIPPED head_dim 256 ----------
            // Same kernel family, same routing (GQA + K >= 4096 -> two-pass),
            // same query shape, untouched by this port. Its fused-vs-unfused
            // error IS the expected error level for a two-pass vector SDPA.
            let (q2, k2, v2) = makeInputs(
                headDim: 256, keys: keyLength,
                qScale: regime == "diffuse" ? 1.0 / Float(256).squareRoot() : 1.0)
            let ref2 = referenceFP32(q: q2, k: k2, v: v2, scale: 1.0)
            let unfused2 = unfusedGraph(
                q: q2, k: k2, v: v2, scale: 1.0, dtype: .bfloat16, stream: .gpu)
            let fused2 = MLXFast.scaledDotProductAttention(
                queries: q2, keys: k2, values: v2, scale: 1.0, mask: .none)
            eval(unfused2, fused2)
            cell["control_headDim256"] = [
                "stock256_twoPass_vs_fp32": encode(compare(fused2, ref2)),
                "stock256_unfused_vs_fp32": encode(compare(unfused2, ref2)),
                "stock256_twoPass_vs_unfused": encode(compare(fused2, unfused2)),
            ]

            // ---------- GQA head mapping ----------
            // Which query heads read which KV head? Answered, not assumed:
            // build the reference under both candidate mappings and see which
            // one the kernel reproduces. `contiguous` is MLX's convention
            // (kv = h / nRepeats); `interleaved` is the classic way to get it
            // wrong (kv = h % kvHeads).
            //
            // Deliberately at a SHORT key length: the mapping is a structural
            // property, identical at any K, and the override arm has to gather
            // K/V per query head -- which at 17,408 keys would materialise
            // ~1.1 GB of fp32 on the CPU for no extra evidence. The routing is
            // unchanged (head dim 512 is forced to the two-pass kernel at every
            // key length), so this exercises the same kernel.
            if twoPassAdmitted {
                let mapKeys = 1024
                let (qm, km, vm) = makeInputs(headDim: 512, keys: mapKeys, qScale: qScale)
                let contiguous = (0 ..< 16).map { $0 / 8 }
                let interleaved = (0 ..< 16).map { $0 % 2 }
                let refContig = referenceFP32(
                    q: qm, k: km, v: vm, scale: 1.0, kvIndexOverride: contiguous)
                let refInter = referenceFP32(
                    q: qm, k: km, v: vm, scale: 1.0, kvIndexOverride: interleaved)
                let fusedForMap = MLXFast.scaledDotProductAttention(
                    queries: qm, keys: km, values: vm, scale: 1.0, mask: .none)
                eval(fusedForMap)
                cell["gqa_mapKeys"] = mapKeys
                cell["gqa_map_contiguous_kv_eq_h_div_8"] = encode(compare(fusedForMap, refContig))
                cell["gqa_map_interleaved_kv_eq_h_mod_2"] = encode(compare(fusedForMap, refInter))
            } else {
                cell["gqa_map_contiguous_kv_eq_h_div_8"] = "SKIPPED: 512 not fused here"
                cell["gqa_map_interleaved_kv_eq_h_mod_2"] = "SKIPPED: 512 not fused here"
            }

            // ---------- scale ----------
            // Each path must apply the given scale EXACTLY ONCE. Run both at a
            // non-unit scale against a reference built with the same scale;
            // then confirm a reference built with the WRONG scale does NOT
            // match, so "matches" is not vacuous.
            let s = 1.0 / Float(512).squareRoot()
            let refScaled = referenceFP32(q: q, k: k, v: v, scale: s)
            let unfusedScaled = unfusedGraph(
                q: q, k: k, v: v, scale: s, dtype: .bfloat16, stream: .gpu)
            eval(unfusedScaled)
            cell["scale_unfused_s_vs_ref_s"] = encode(compare(unfusedScaled, refScaled))
            if twoPassAdmitted {
                let fusedScaled = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: s, mask: .none)
                eval(fusedScaled)
                cell["scale_fused_s_vs_ref_s"] = encode(compare(fusedScaled, refScaled))
                cell["scale_fused_s_vs_ref_ONE_should_be_large"] =
                    encode(compare(fusedScaled, ref))
            } else {
                cell["scale_fused_s_vs_ref_s"] = "SKIPPED: 512 not fused here"
            }
            cell["scaleUsed"] = Double(s)

            report[regime] = cell
        }

        let data = try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outputPath))
        FileHandle.standardError.write(
            Data(
                ("\n=== D512 parity written to \(outputPath) ===\n"
                    + (String(data: data, encoding: .utf8) ?? "") + "\n").utf8))
    }
}
