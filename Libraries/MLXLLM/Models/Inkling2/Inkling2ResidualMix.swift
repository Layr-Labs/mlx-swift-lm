// Inkling 2 block-attention residual mix for prefill as one kernel per row.
import Foundation
import MLX

/// `inkling2AttentionResidual` for prefill: per token, RMS-normalize each of
/// the S states, score them against every learned query, softmax over S and
/// mix, all in FP32 in one threadgroup that reads each state once. The plain
/// graph stacks the states into an FP32 copy, normalizes it, then reads it
/// again for the scores and once more per query for the mix.
///
/// Every reduction runs in a fixed order inside the row's threadgroup (lane
/// sums, then `simd_sum`, then simdgroups in index order), so a token's result
/// depends only on its own states: not on chunking, row company or device
/// load. It is not bit-identical to the plain graph (different reduction
/// order; prefill == decode bits are not required by the gates); decode and
/// verify steps keep the plain graph.
enum Inkling2ResidualMix {
    /// `DARKBLOOM_INKLING2_PREFILL_MIX_KERNEL=0` keeps the plain graph in prefill.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFILL_MIX_KERNEL"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `enabled` for the current task.
    @TaskLocal static var override: Bool? = nil
    static var active: Bool { override ?? enabled }

    static let threads = 256
    static let maximumStates = 8
    static let maximumQueries = 48

    private final class Kernels: @unchecked Sendable {
        let lock = NSLock()
        var byStates: [Int: MLXFast.MLXFastKernel] = [:]
        func kernel(states: Int) -> MLXFast.MLXFastKernel {
            lock.withLock {
                if let k = byStates[states] { return k }
                let k = Inkling2ResidualMix.make(states: states)
                byStates[states] = k
                return k
            }
        }
    }
    private static let kernels = Kernels()

    private static func make(states: Int) -> MLXFast.MLXFastKernel {
        let names = (0 ..< states).map { "s\($0)" }
        let load = names.enumerated().map { i, name in
            "x[\(i)][j] = float(\(name)[base + h]);"
        }.joined(separator: "\n                    ")
        return MLXFast.metalKernel(
            name: "inkling2_residual_mix_s\(states)",
            inputNames: names + ["queries"],
            outputNames: ["out"],
            source: """
                constexpr int S = \(states);
                constexpr int T = \(threads);
                constexpr int PER = (H + T - 1) / T;
                constexpr int SIMDS = T / 32;
                constexpr float EPS = 1e-6f;
                const float SCALE = 1.0f / sqrt(float(H));
                threadgroup float partial[NQ * S * SIMDS];
                threadgroup float stat[S * SIMDS];
                threadgroup float weights[NQ * S];
                const uint row = threadgroup_position_in_grid.y;
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint simd = simdgroup_index_in_threadgroup;
                const size_t base = size_t(row) * H;
                float x[S][PER];
                for (int j = 0; j < PER; ++j) {
                    const int h = int(tid) + j * T;
                    if (h < H) {
                    \(load)
                    } else {
                        for (int s = 0; s < S; ++s) { x[s][j] = 0.0f; }
                    }
                }
                // RMS statistics per state: lane sums, simd_sum, simdgroups in order.
                for (int s = 0; s < S; ++s) {
                    float acc = 0.0f;
                    for (int j = 0; j < PER; ++j) { acc = fma(x[s][j], x[s][j], acc); }
                    acc = simd_sum(acc);
                    if (lane == 0) { stat[s * SIMDS + simd] = acc; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float inv[S];
                for (int s = 0; s < S; ++s) {
                    float total = 0.0f;
                    for (int g = 0; g < SIMDS; ++g) { total += stat[s * SIMDS + g]; }
                    inv[s] = rsqrt(total / float(H) + EPS);
                    for (int j = 0; j < PER; ++j) { x[s][j] *= inv[s]; }
                }
                // Scores: every (query, state) dot product.
                for (int q = 0; q < NQ; ++q) {
                    float qv[PER];
                    for (int j = 0; j < PER; ++j) {
                        const int h = int(tid) + j * T;
                        qv[j] = h < H ? queries[q * H + h] : 0.0f;
                    }
                    for (int s = 0; s < S; ++s) {
                        float acc = 0.0f;
                        for (int j = 0; j < PER; ++j) { acc = fma(x[s][j], qv[j], acc); }
                        acc = simd_sum(acc);
                        if (lane == 0) { partial[(q * S + s) * SIMDS + simd] = acc; }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Softmax over the states, one query per thread.
                for (int q = int(tid); q < NQ; q += T) {
                    float score[S];
                    float top = -INFINITY;
                    for (int s = 0; s < S; ++s) {
                        float total = 0.0f;
                        for (int g = 0; g < SIMDS; ++g) { total += partial[(q * S + s) * SIMDS + g]; }
                        score[s] = total * SCALE;
                        top = max(top, score[s]);
                    }
                    float sum = 0.0f;
                    for (int s = 0; s < S; ++s) {
                        score[s] = exp(score[s] - top);
                        sum += score[s];
                    }
                    for (int s = 0; s < S; ++s) { weights[q * S + s] = score[s] / sum; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Mix: states in order for every query.
                const size_t outBase = size_t(row) * NQ * H;
                for (int q = 0; q < NQ; ++q) {
                    for (int j = 0; j < PER; ++j) {
                        const int h = int(tid) + j * T;
                        if (h < H) {
                            float acc = 0.0f;
                            for (int s = 0; s < S; ++s) { acc = fma(weights[q * S + s], x[s][j], acc); }
                            out[outBase + size_t(q) * H + h] = acc;
                        }
                    }
                }
                """,
            ensureRowContiguous: true
        )
    }

    /// Whether the kernel covers this mix.
    static func supported(queries: MLXArray, states: [MLXArray]) -> Bool {
        let hidden = queries.dim(-1)
        let count = queries.size / hidden
        return active && !states.isEmpty && states.count <= maximumStates
            && count <= maximumQueries && queries.dtype == .float32
            && states.allSatisfy { $0.dim(-1) == hidden && $0.shape == states[0].shape }
    }

    /// `queries` `[nq, hidden]` or `[..., hidden]` FP32; `states` the S
    /// arrays `[..., hidden]` (BF16 or FP32). Returns `[..., nq, hidden]` FP32.
    static func mix(queries: MLXArray, states: [MLXArray]) -> MLXArray {
        let hidden = queries.dim(-1)
        let count = queries.size / hidden
        var outer = states[0].shape
        outer.removeLast()
        let rows = outer.reduce(1, *)
        let kernel = kernels.kernel(states: states.count)
        return kernel(
            states.map { $0.reshaped(rows, hidden) } + [queries.reshaped(count, hidden)],
            template: [("H", hidden), ("NQ", count)],
            grid: (threads, rows, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, count, hidden]], outputDTypes: [.float32]
        )[0].reshaped(outer + [count, hidden])
    }
}

/// A message gate's input for prefill: `[rms(h), rms(m)]` concatenated and
/// cast to the down projection's dtype, one threadgroup per token reading `h`
/// and `m` once (the plain graph runs two normalizations, a concatenation and
/// a cast through FP32 intermediates). Fixed-order reductions inside the
/// token's threadgroup: a token never depends on chunking or row company.
/// Decode and verify steps keep the plain graph.
enum Inkling2GateInput {
    /// `DARKBLOOM_INKLING2_PREFILL_GATE_KERNEL=0` keeps the plain graph in prefill.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFILL_GATE_KERNEL"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "inkling2_gate_input",
        inputNames: ["h", "m"],
        outputNames: ["out"],
        source: """
            constexpr int T = 256;
            constexpr int PER = (H + T - 1) / T;
            constexpr int SIMDS = T / 32;
            threadgroup float stat[2 * SIMDS];
            const uint row = threadgroup_position_in_grid.y;
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            const size_t base = size_t(row) * H;
            float xh[PER];
            float xm[PER];
            float sh = 0.0f;
            float sm = 0.0f;
            for (int j = 0; j < PER; ++j) {
                const int i = int(tid) + j * T;
                xh[j] = i < H ? float(h[base + i]) : 0.0f;
                xm[j] = i < H ? float(m[base + i]) : 0.0f;
                sh = fma(xh[j], xh[j], sh);
                sm = fma(xm[j], xm[j], sm);
            }
            sh = simd_sum(sh);
            sm = simd_sum(sm);
            if (lane == 0) {
                stat[simd] = sh;
                stat[SIMDS + simd] = sm;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float th = 0.0f;
            float tm = 0.0f;
            for (int g = 0; g < SIMDS; ++g) {
                th += stat[g];
                tm += stat[SIMDS + g];
            }
            const float ih = rsqrt(th / float(H) + 1e-6f);
            const float im = rsqrt(tm / float(H) + 1e-6f);
            const size_t outBase = size_t(row) * 2 * H;
            for (int j = 0; j < PER; ++j) {
                const int i = int(tid) + j * T;
                if (i < H) {
                    out[outBase + i] = static_cast<OutT>(xh[j] * ih);
                    out[outBase + H + i] = static_cast<OutT>(xm[j] * im);
                }
            }
            """,
        ensureRowContiguous: true
    )

    static func supported(_ h: MLXArray, _ m: MLXArray, dtype: DType) -> Bool {
        enabled && h.shape == m.shape && [DType.bfloat16, .float16].contains(dtype)
    }

    /// `h`, `m` `[..., H]`. Returns `[..., 2H]` in `dtype`.
    static func normalized(_ h: MLXArray, _ m: MLXArray, dtype: DType) -> MLXArray {
        let hidden = h.dim(-1)
        var outer = h.shape
        outer.removeLast()
        let rows = outer.reduce(1, *)
        return kernel(
            [h.reshaped(rows, hidden), m.reshaped(rows, hidden)],
            template: [("H", hidden), ("OutT", dtype)],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, 2 * hidden]], outputDTypes: [dtype]
        )[0].reshaped(outer + [2 * hidden])
    }
}
