// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion `gated_delta_blocked_seq` (omlx/custom_kernels/qwen35_prefill/gdn.py)
// for Qwen4 GDN prefill. Same sequential recurrence as mlx-lm, restructured
// for Apple GPUs: TB-token threadgroup staging of k/q/v, register-resident
// fp32 state, Dv/32 split. Fusion measured ~2× vs the stock T-loop kernel
// at 16K per layer. Qwen4-only (`qwen4L2`); 27B `qwen3_5` stays on the
// existing kernel until this wins on Flash-Next.
//
// Geometry: Dk == 128, Dv % 32 == 0, T >= 64, no mask.
// TB=32 for bf16 (default Flash-Next), TB=16 for float32 (32 KiB limit).
// Kill: DARKBLOOM_QWEN4_GDN_BLOCKED=0.

import Foundation
import MLX
import MLXLMCommon

enum Qwen4ExpGDNBlockedSeq: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_GDN_BLOCKED"
    static let minT = 64
    static let keyHeadDim = 128
    static let valueBlock = 32
    static let threads = 256

    static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    static func matchesGeometry(T: Int, keyHeadDim: Int, valueHeadDim: Int) -> Bool {
        T >= minT && keyHeadDim == Self.keyHeadDim && valueHeadDim > 0
            && valueHeadDim % valueBlock == 0
    }

    static func shouldDispatch(
        T: Int, keyHeadDim: Int, valueHeadDim: Int, hasMask: Bool,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        !hasMask && isEnabled(environment: environment)
            && matchesGeometry(T: T, keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim)
    }

    static func blockT(for dtype: DType) -> Int {
        dtype == .float32 ? 16 : 32
    }
}

private let gdnBlockedHeader = """
#include <metal_stdlib>
using namespace metal;
"""

private func gdnBlockedSeqSource(blockT: Int) -> String {
    """
        constexpr int TB = \(blockT);
        constexpr int DB = 32;
        const int tid = thread_position_in_threadgroup.x;
        const int blk = threadgroup_position_in_grid.x;
        const int hv  = threadgroup_position_in_grid.y;
        const int b   = threadgroup_position_in_grid.z;
        const int hk  = hv / (Hv / Hk);
        const int dv0 = blk * DB;

        // Preserve the canonical gated_delta_step reduction exactly: one
        // 32-lane SIMD group per value row, with each lane owning four
        // contiguous Dk values. Eight SIMD groups cover four value rows each
        // so the threadgroup still processes the same DB=32 block. The old
        // 8-lane/16-value partition was mathematically close but rounded
        // differently, making a long blocked prefill diverge when a restored
        // suffix fell through to the canonical short-window kernel.
        constexpr int SIMD_GROUPS = 8;
        constexpr int DV_PER_SIMD = DB / SIMD_GROUPS;
        const int simd = tid / 32;
        const int lane = tid % 32;
        const int d0 = lane * 4;

        threadgroup InT k_s[TB][Dk + 8];
        threadgroup InT q_s[TB][Dk + 8];
        threadgroup InT v_s[TB][DB + 8];
        threadgroup float g_s[TB];
        threadgroup float b_s[TB];

        const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
        const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
        const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
        const size_t krow = (size_t)Hk * Dk;

        float st[DV_PER_SIMD][4];
        for (int j = 0; j < DV_PER_SIMD; ++j) {
            const int dv = simd * DV_PER_SIMD + j;
            const device float* S_in =
                state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
            for (int i = 0; i < 4; ++i) st[j][i] = S_in[i];
        }

        device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

        for (int t0 = 0; t0 < T; t0 += TB) {
            const int tt = min(TB, T - t0);
            for (int p = tid; p < tt * Dk; p += 256) {
                const int r = p / Dk, d = p % Dk;
                k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
                q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
            }
            for (int p = tid; p < tt * DB; p += 256) {
                const int r = p / DB, d = p % DB;
                v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
            }
            for (int p = tid; p < tt; p += 256) {
                g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];
                b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int t = 0; t < tt; ++t) {
                const float gt = g_s[t];
                const float bt = b_s[t];
                for (int j = 0; j < DV_PER_SIMD; ++j) {
                    const int dv = simd * DV_PER_SIMD + j;
                    float kv_mem = 0.0f;
                    {
                        #pragma clang fp reassociate(off)
                        #pragma clang fp contract(off)
                        float kv_compensation = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            st[j][i] = st[j][i] * gt;
                            const float product =
                                st[j][i] * (float)k_s[t][d0 + i];
                            const float corrected = product - kv_compensation;
                            const float next_sum = kv_mem + corrected;
                            kv_compensation = (next_sum - kv_mem) - corrected;
                            kv_mem = next_sum;
                        }
                    }
                    kv_mem = simd_sum(kv_mem);
                    const float delta =
                        ((float)v_s[t][dv] - kv_mem) * bt;

                    float out = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        st[j][i] =
                            st[j][i] + (float)k_s[t][d0 + i] * delta;
                        out += st[j][i] * (float)q_s[t][d0 + i];
                    }
                    out = simd_sum(out);
                    if (lane == 0) {
                        y_base[(size_t)(t0 + t) * Hv * Dv + dv] =
                            (InT)out;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int j = 0; j < DV_PER_SIMD; ++j) {
            const int dv = simd * DV_PER_SIMD + j;
            device float* S_out =
                state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
            for (int i = 0; i < 4; ++i) S_out[i] = st[j][i];
        }
    """
}

private final class Qwen4ExpGDNBlockedKernelManager: Sendable {
    static let shared = Qwen4ExpGDNBlockedKernelManager()

    private let lock = NSLock()
    nonisolated(unsafe) private var kernels: [Int: MLXFast.MLXFastKernel] = [:]

    func kernel(blockT: Int) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let existing = kernels[blockT] { return existing }
        let built = MLXFast.metalKernel(
            name: "darkbloom_qwen4_gdn_blocked_seq_tb\(blockT)",
            inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
            outputNames: ["y", "state_out"],
            source: gdnBlockedSeqSource(blockT: blockT),
            header: gdnBlockedHeader)
        kernels[blockT] = built
        return built
    }
}

func gatedDeltaBlockedSeq(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let blockT = Qwen4ExpGDNBlockedSeq.blockT(for: q.dtype)
    let dvBlocks = Dv / Qwen4ExpGDNBlockedSeq.valueBlock
    let outputs = Qwen4ExpGDNBlockedKernelManager.shared.kernel(blockT: blockT)(
        [q, k, v, g, beta, state, MLXArray(T)],
        template: [
            ("InT", q.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (Qwen4ExpGDNBlockedSeq.threads * dvBlocks, Hv, B),
        threadGroup: (Qwen4ExpGDNBlockedSeq.threads, 1, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [q.dtype, .float32])
    return (outputs[0], outputs[1])
}
