// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion `hc_projection.py` (`omlx_qwen4_hc_hybrid_qmv`, Apache-2.0): the
// lossless one-dispatch Qwen4 hyper-connection input projection for the
// single-token decode shape. MLX's stock `quantized_matmul` runs the
// `block_inject_weight` bank (N=4, K=10240) as one latency-bound simdgroup
// (~95 µs in situ on M3 Ultra — the single most expensive kernel in every
// hyper-connection read, twice per layer). Fusion's kernel keeps MLX's
// literal traversals — the 320 `input_mix_weight_down` rows use
// `qmv_fast`, the 4 inject rows use the standalone-N=4 general `qmv` — in
// one dispatch, so both raw projections are bit-identical to the two
// canonical dispatches while the inject simdgroups overlap the down rows.
//
// Darkbloom extension: `DOWN_BITS` / `INJECT_BITS` are separate template
// arguments because 5 of the 96 oQ4e hyper-connection modules carry
// (down 4-bit, inject 5-bit); the two branches were already independent.
// Verified bit-identical against `quantized_matmul` for (4,4) (4,5) (5,5)
// (6,6) (8,8) × 30 random banks each (Python prototype), and pinned by
// `Qwen4ExpHCHybridTests`.
//
// Metal arithmetic transcribed from MLX core at pinned commit `ceab91938`
// (`mlx/backend/metal/kernels/quantized.h`). MLX is Copyright © 2023 Apple
// Inc. and licensed under the MIT License (notice preserved in the header
// string below). Kill: `DARKBLOOM_QWEN4_HC_HYBRID=0` restores the two
// canonical dispatches (Fusion `OMLX_QWEN4_HC_HYBRID`).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum Qwen4ExpHCHybridInvocation: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var native = 0
    nonisolated(unsafe) private static var fallback = 0

    public struct Snapshot: Sendable, Equatable {
        public var native: Int
        public var fallback: Int

        public init(native: Int, fallback: Int) {
            self.native = native
            self.fallback = fallback
        }

        public var line: String { "hcHybrid native=\(native) fallback=\(fallback)" }
    }

    static func recordNative() { lock.withLock { native += 1 } }
    static func recordFallback() { lock.withLock { fallback += 1 } }

    public static func snapshot() -> Snapshot {
        lock.withLock { Snapshot(native: native, fallback: fallback) }
    }

    public static func resetForTesting() {
        lock.withLock {
            native = 0
            fallback = 0
        }
    }
}

public enum Qwen4ExpHCHybrid: Sendable {
    public static let envFlag = "DARKBLOOM_QWEN4_HC_HYBRID"

    /// Fusion's geometry contract: the kernel hard-codes 40 down
    /// threadgroups × 8 rows and the `combined[320 + row]` inject offset.
    public static let hcCount = 4
    public static let hiddenSize = 2560
    public static let streamWidth = hcCount * hiddenSize
    public static let lowRank = 320
    public static let groupSize = 64
    public static let supportedBits: Set<Int> = [4, 5, 6, 8]

    public static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        guard let raw = environment[envFlag]?.trimmingCharacters(in: .whitespaces).lowercased()
        else { return true }
        return !["0", "false", "off", "no"].contains(raw)
    }

    /// Fusion `compatible_projections`: both banks are plain affine
    /// `QuantizedLinear` (no bias), group 64, supported bits, bf16 scales,
    /// and the exact `[320, packed]` / `[4, packed]` shapes.
    static func compatible(down: Linear, injection: Linear) -> (QuantizedLinear, QuantizedLinear)? {
        guard let d = down as? QuantizedLinear, let i = injection as? QuantizedLinear,
            ObjectIdentifier(type(of: d)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: i)) == ObjectIdentifier(QuantizedLinear.self),
            d.bias == nil, i.bias == nil,
            d.mode == .affine, i.mode == .affine,
            d.groupSize == groupSize, i.groupSize == groupSize,
            supportedBits.contains(d.bits), supportedBits.contains(i.bits),
            let dBiases = d.biases, let iBiases = i.biases
        else { return nil }
        let scaleCols = streamWidth / groupSize
        guard d.weight.dtype == .uint32, i.weight.dtype == .uint32,
            d.weight.shape == [lowRank, streamWidth * d.bits / 32],
            i.weight.shape == [hcCount, streamWidth * i.bits / 32],
            d.scales.shape == [lowRank, scaleCols], i.scales.shape == [hcCount, scaleCols],
            dBiases.shape == d.scales.shape, iBiases.shape == i.scales.shape,
            d.scales.dtype == .bfloat16, dBiases.dtype == .bfloat16,
            i.scales.dtype == .bfloat16, iBiases.dtype == .bfloat16
        else { return nil }
        return (d, i)
    }

    /// Widest window the one-dispatch hybrid serves (Lightning verify 1+k).
    static let maxTokens = 16

    static func eligible(_ x: MLXArray) -> Bool {
        x.ndim == 3 && x.dim(0) == 1 && (1 ... maxTokens).contains(x.dim(1))
            && x.dim(2) == streamWidth && x.dtype == .bfloat16
    }

    /// Fusion `hybrid_projection`: `[1, T, 324]` = `down(x) ++ inject(x)` for
    /// B1 bf16 windows up to `maxTokens` (one threadgroup grid slice per
    /// token), or nil (caller keeps the two canonical dispatches). Column t
    /// of a window equals the T=1 launch on row t bit for bit.
    static func apply(
        _ x: MLXArray, down: Linear, injection: Linear,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        guard isEnabled(environment: environment), eligible(x),
            let (d, i) = compatible(down: down, injection: injection),
            let dBiases = d.biases, let iBiases = i.biases
        else {
            Qwen4ExpHCHybridInvocation.recordFallback()
            return nil
        }
        let outputs = qwen4HCHybridKernel(
            [x, d.weight, d.scales, dBiases, i.weight, i.scales, iBiases],
            template: [
                ("T", x.dtype),
                ("DOWN_BITS", d.bits),
                ("INJECT_BITS", i.bits),
                ("K", streamWidth),
            ],
            grid: (32, 82, x.dim(1)),
            threadGroup: (32, 2, 1),
            outputShapes: [[1, x.dim(1), lowRank + hcCount]],
            outputDTypes: [x.dtype])
        guard let combined = outputs.first else {
            Qwen4ExpHCHybridInvocation.recordFallback()
            return nil
        }
        Qwen4ExpHCHybridInvocation.recordNative()
        return combined
    }
}

private let qwen4HCHybridKernel = MLXFast.metalKernel(
    name: "qwen4_hc_hybrid_qmv",
    inputNames: ["x", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"],
    outputNames: ["combined"],
    source: qwen4HCHybridSource,
    header: qwen4HCHybridHeader,
    ensureRowContiguous: true)

// MLX is Copyright © 2023 Apple Inc. and licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions: The above copyright notice and this permission
// notice shall be included in all copies or substantial portions of the
// Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
private let qwen4HCHybridHeader = """
    using namespace metal;

    template <int bits>
    inline constexpr short hc_pack_factor() {
        return bits == 5 ? 8 : (bits == 6 ? 4 : 32 / bits);
    }

    template <int bits>
    inline constexpr short hc_bytes_per_pack() {
        constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
        return power_of_2_bits ? 4 : (bits == 5 ? 5 : 3);
    }

    template <typename T, int N, int bits>
    inline float hc_load_vector(const device T* x, thread float* xt) {
        float sum = 0.0f;
        if (bits == 4) {
            for (int i = 0; i < N; i += 4) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                xt[i] = x[i];
                xt[i + 1] = x[i + 1] / 16.0f;
                xt[i + 2] = x[i + 2] / 256.0f;
                xt[i + 3] = x[i + 3] / 4096.0f;
            }
        } else if (bits == 5) {
            for (int i = 0; i < N; i += 8) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                    + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
                xt[i] = x[i];
                xt[i + 1] = x[i + 1] / 32.0f;
                xt[i + 2] = x[i + 2] / 4.0f;
                xt[i + 3] = x[i + 3] / 128.0f;
                xt[i + 4] = x[i + 4] / 16.0f;
                xt[i + 5] = x[i + 5] / 2.0f;
                xt[i + 6] = x[i + 6] / 64.0f;
                xt[i + 7] = x[i + 7] / 8.0f;
            }
        } else if (bits == 6) {
            for (int i = 0; i < N; i += 4) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                xt[i] = x[i];
                xt[i + 1] = x[i + 1] / 64.0f;
                xt[i + 2] = x[i + 2] / 16.0f;
                xt[i + 3] = x[i + 3] / 4.0f;
            }
        } else if (bits == 8) {
            for (int i = 0; i < N; ++i) {
                sum += x[i];
                xt[i] = x[i];
            }
        }
        return sum;
    }

    template <int N, int bits>
    inline float hc_qdot(
        const device uint8_t* w,
        const thread float* xt,
        float scale,
        float bias,
        float sum) {
        float accum = 0.0f;
        if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < N / 4; ++i) {
                accum +=
                    (xt[4 * i] * (ws[i] & 0x000f)
                     + xt[4 * i + 1] * (ws[i] & 0x00f0)
                     + xt[4 * i + 2] * (ws[i] & 0x0f00)
                     + xt[4 * i + 3] * (ws[i] & 0xf000));
            }
        } else if (bits == 5) {
            for (int i = 0; i < N / 8; ++i) {
                xt += 8 * i;
                w += 5 * i;
                accum += (w[0] & 0x1f) * xt[0];
                accum += (w[0] & 0xe0) * xt[1];
                accum += (w[1] & 0x3) * (xt[1] * 256.0f);
                accum += (w[1] & 0x7c) * xt[2];
                accum += (w[1] & 0x80) * xt[3];
                accum += (w[2] & 0xf) * (xt[3] * 256.0f);
                accum += (w[2] & 0xf0) * xt[4];
                accum += (w[3] & 0x1) * (xt[4] * 256.0f);
                accum += (w[3] & 0x3e) * xt[5];
                accum += (w[3] & 0xc0) * xt[6];
                accum += (w[4] & 0x7) * (xt[6] * 256.0f);
                accum += (w[4] & 0xf8) * xt[7];
            }
        } else if (bits == 6) {
            for (int i = 0; i < N / 4; ++i) {
                xt += 4 * i;
                w += 3 * i;
                accum += (w[0] & 0x3f) * xt[0];
                accum += (w[0] & 0xc0) * xt[1];
                accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
                accum += (w[1] & 0xf0) * xt[2];
                accum += (w[2] & 0x03) * (xt[2] * 256.0f);
                accum += (w[2] & 0xfc) * xt[3];
            }
        } else if (bits == 8) {
            for (int i = 0; i < N; ++i) accum += xt[i] * w[i];
        }
        return scale * accum + sum * bias;
    }

    template <int N, int bits>
    inline float hc_qdot_safe(
        const device uint8_t* w,
        const thread float* xt,
        float scale,
        float bias,
        float sum) {
        float accum = 0.0f;
        if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < N / 4; ++i) {
                accum +=
                    (xt[4 * i] * (ws[i] & 0x000f)
                     + xt[4 * i + 1] * (ws[i] & 0x00f0)
                     + xt[4 * i + 2] * (ws[i] & 0x0f00)
                     + xt[4 * i + 3] * (ws[i] & 0xf000));
            }
        } else if (bits == 5) {
            for (int i = 0; i < N / 8; ++i) {
                xt += 8 * i;
                w += 5 * i;
                accum += (w[0] & 0x1f) * xt[0];
                accum += (w[0] & 0xe0) * xt[1];
                accum += (w[1] & 0x3) * (xt[1] * 256.0f);
                accum += (w[1] & 0x7c) * xt[2];
                accum += (w[1] & 0x80) * xt[3];
                accum += (w[2] & 0xf) * (xt[3] * 256.0f);
                accum += (w[2] & 0xf0) * xt[4];
                accum += (w[3] & 0x1) * (xt[4] * 256.0f);
                accum += (w[3] & 0x3e) * xt[5];
                accum += (w[3] & 0xc0) * xt[6];
                accum += (w[4] & 0x7) * (xt[6] * 256.0f);
                accum += (w[4] & 0xf8) * xt[7];
            }
        } else if (bits == 6) {
            for (int i = 0; i < N / 4; ++i) {
                xt += 4 * i;
                w += 3 * i;
                accum += (w[0] & 0x3f) * xt[0];
                accum += (w[0] & 0xc0) * xt[1];
                accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
                accum += (w[1] & 0xf0) * xt[2];
                accum += (w[2] & 0x03) * (xt[2] * 256.0f);
                accum += (w[2] & 0xfc) * xt[3];
            }
        } else if (bits == 8) {
            for (int i = 0; i < N; ++i) accum += xt[i] * w[i];
        }
        return scale * accum + sum * bias;
    }
    """

private let qwen4HCHybridSource = """
    const uint tg = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    // Verify columns ride the grid z-axis: every column executes this exact
    // body on its own row of x, so a [1, T, K] window is bit-identical to T
    // single-token launches (the T=1 decode path).
    const uint tok = threadgroup_position_in_grid.z;
    x += tok * K;
    combined += tok * 324;
    constexpr int PF = hc_pack_factor<DOWN_BITS>();
    constexpr int BP = hc_bytes_per_pack<DOWN_BITS>();
    constexpr int GROUPS = K / 64;
    constexpr int ROW_BYTES = K * BP / PF;

    if (tg < 40) {
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int out_row = int(tg) * 8 + int(sg) * 4;
        const device uint8_t* wp = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS
            + int(lane) / SCALE_STEP;
        const device T* xp = x + int(lane) * VPT;
        float result[4] = {0.0f};
        float xv[VPT];
        for (int k = 0; k < K; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, DOWN_BITS>(xp, xv);
            for (int row = 0; row < 4; ++row) {
                result[row] += hc_qdot<VPT, DOWN_BITS>(
                    wp + row * ROW_BYTES,
                    xv,
                    float(sp[row * GROUPS]),
                    float(bp[row * GROUPS]),
                    sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp += BLOCK;
        }
        for (int row = 0; row < 4; ++row) {
            result[row] = simd_sum(result[row]);
            if (lane == 0) combined[out_row + row] = T(result[row]);
        }
        return;
    }

    if (sg != 0) return;
    constexpr int IPF = hc_pack_factor<INJECT_BITS>();
    constexpr int IBP = hc_bytes_per_pack<INJECT_BITS>();
    constexpr int IROW_BYTES = K * IBP / IPF;
    constexpr int PPT = 1;
    constexpr int VPT = IPF;
    constexpr int BLOCK = VPT * 32;
    constexpr int SCALE_STEP = 64 / VPT;
    const device uint8_t* wp = (const device uint8_t*)inject_w
        + int(lane) * IBP;
    const device T* sp = inject_s + int(lane) / SCALE_STEP;
    const device T* bp = inject_b + int(lane) / SCALE_STEP;
    const device T* xp = x + int(lane) * VPT;
    float result[4] = {0.0f};
    float xv[VPT];
    int k = 0;
    for (; k < K - BLOCK; k += BLOCK) {
        float sum = hc_load_vector<T, VPT, INJECT_BITS>(xp, xv);
        for (int row = 0; row < 4; ++row) {
            result[row] += hc_qdot<VPT, INJECT_BITS>(
                wp + row * IROW_BYTES,
                xv,
                float(sp[row * GROUPS]),
                float(bp[row * GROUPS]),
                sum);
        }
        wp += BLOCK * IBP / IPF;
        sp += BLOCK / 64;
        bp += BLOCK / 64;
        xp += BLOCK;
    }
    float sum = hc_load_vector<T, VPT, INJECT_BITS>(xp, xv);
    for (int row = 0; row < 4; ++row) {
        result[row] += hc_qdot_safe<VPT, INJECT_BITS>(
            wp + row * IROW_BYTES,
            xv,
            float(sp[row * GROUPS]),
            float(bp[row * GROUPS]),
            sum);
        result[row] = simd_sum(result[row]);
        if (lane == 0) combined[320 + row] = T(result[row]);
    }
    """
