// DIAGNOSTIC (scratch): the GDN chain as custom Metal kernels, re-measured
// on an idle machine after the first measurement was taken under load.
import Foundation
import MLX
import MLXFast
import MLXNN

enum Qwen4ExpGDNScratchKernels {
    static let conv: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_gdn_conv_scratch",
        inputNames: ["state", "x", "w"],
        outputNames: ["qk", "v", "new_state"],
        source: """
                auto c = thread_position_in_grid.x;
                auto s = thread_position_in_grid.y;
                auto b = thread_position_in_grid.z;
                if (c >= C) return;
                constexpr int P = K - 1;
                auto at = [&](int p) -> float {
                    if (p < P) return static_cast<float>(state[(b * P + p) * C + c]);
                    return static_cast<float>(x[(b * S + (p - P)) * C + c]);
                };
                if (s < S) {
                    float acc = 0.0f;
                    for (int t = 0; t < K; ++t) { acc += static_cast<float>(w[c * K + t]) * at(s + t); }
                    float y = acc / (1.0f + metal::exp(-acc));
                    if (c < QK) { qk[(b * S + s) * QK + c] = static_cast<InT>(y); }
                    else { v[(b * S + s) * (C - QK) + (c - QK)] = static_cast<InT>(y); }
                }
                if (s < P) { new_state[(b * P + s) * C + c] = static_cast<InT>(at(S + s)); }
            """)

    static func conv(state: MLXArray, x: MLXArray, weight: MLXArray, kernelSize: Int, keyDim: Int, valueHeads: Int, valueHeadDim: Int) -> (qk: MLXArray, v: MLXArray, newState: MLXArray) {
        let B = x.dim(0), S = x.dim(1), C = x.dim(2)
        let rows = max(S, kernelSize - 1)
        let outputs = conv([state, x, weight.reshaped([C, kernelSize])],
            template: [("InT", x.dtype), ("C", C), ("K", kernelSize), ("S", S), ("QK", 2 * keyDim)],
            grid: (C, rows, B), threadGroup: (min(C, 256), 1, 1),
            outputShapes: [[B, S, 2 * keyDim], [B, S, valueHeads, valueHeadDim], [B, kernelSize - 1, C]],
            outputDTypes: [x.dtype, x.dtype, x.dtype])
        return (outputs[0], outputs[1], outputs[2])
    }

    static let qkNorm: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_gdn_qk_norm_scratch",
        inputNames: ["qk", "scales", "eps"],
        outputNames: ["q", "k"],
        source: """
                auto lane = thread_position_in_threadgroup.x;
                auto which = thread_position_in_grid.y;
                auto row = thread_position_in_grid.z;
                constexpr int per = Dk / 32;
                auto head = which % Hk;
                auto isK = which / Hk;
                auto base = row * (2 * KD) + isK * KD + head * Dk;
                float vals[per]; float ss = 0.0f;
                for (int i = 0; i < per; ++i) { vals[i] = static_cast<float>(qk[base + lane * per + i]); ss += vals[i] * vals[i]; }
                ss = simd_sum(ss);
                float inv = metal::rsqrt(ss / float(Dk) + eps[0]) * scales[isK];
                for (int i = 0; i < per; ++i) {
                    auto o = row * KD + head * Dk + lane * per + i;
                    if (isK) { k[o] = static_cast<InT>(vals[i] * inv); } else { q[o] = static_cast<InT>(vals[i] * inv); }
                }
            """)

    static func qkNorm(qk: MLXArray, keyHeads: Int, keyHeadDim: Int, eps: Float, scales: MLXArray) -> (q: MLXArray, k: MLXArray) {
        let B = qk.dim(0), S = qk.dim(1)
        let outputs = qkNorm([qk, scales, MLXArray([eps])],
            template: [("InT", qk.dtype), ("Dk", keyHeadDim), ("Hk", keyHeads), ("KD", keyHeads * keyHeadDim)],
            grid: (32, 2 * keyHeads, B * S), threadGroup: (32, 1, 1),
            outputShapes: [[B, S, keyHeads, keyHeadDim], [B, S, keyHeads, keyHeadDim]], outputDTypes: [qk.dtype, qk.dtype])
        return (outputs[0], outputs[1])
    }

    static let gate: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_gdn_gate_scratch",
        inputNames: ["x", "gate", "w", "eps"],
        outputNames: ["y"],
        source: """
                auto lane = thread_position_in_threadgroup.x;
                auto head = thread_position_in_grid.y;
                auto row = thread_position_in_grid.z;
                constexpr int per = Dv / 32;
                auto base = (row * Hv + head) * Dv;
                float vals[per]; float ss = 0.0f;
                for (int i = 0; i < per; ++i) { vals[i] = static_cast<float>(x[base + lane * per + i]); ss += vals[i] * vals[i]; }
                ss = simd_sum(ss);
                float inv = metal::rsqrt(ss / float(Dv) + eps[0]);
                for (int i = 0; i < per; ++i) {
                    auto d = lane * per + i;
                    float g = static_cast<float>(gate[base + d]);
                    float sg = 1.0f / (1.0f + metal::exp(-g));
                    float act = SIGMOID ? sg : g * sg;
                    y[base + d] = static_cast<InT>(vals[i] * inv * static_cast<float>(w[d]) * act);
                }
            """)

    static func gate(x: MLXArray, gate g: MLXArray, weight: MLXArray, eps: Float, sigmoid: Bool) -> MLXArray {
        let B = x.dim(0), S = x.dim(1), Hv = x.dim(2), Dv = x.dim(3)
        return gate([x, g, weight, MLXArray([eps])],
            template: [("InT", x.dtype), ("Dv", Dv), ("Hv", Hv), ("SIGMOID", sigmoid)],
            grid: (32, Hv, B * S), threadGroup: (32, 1, 1), outputShapes: [[B, S, Hv, Dv]], outputDTypes: [x.dtype])[0]
    }
}
