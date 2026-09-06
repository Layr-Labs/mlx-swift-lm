// DIAGNOSTIC (scratch): the attention layer's non-matmul ops as kernels.
//   ropeNorm: per-head RMS norm with weight+offset, then the partial rope
//             with cos/sin computed in-kernel from the positions and the
//             inverse frequencies. One launch per tensor (q, k) for the
//             norm, scale, cos/sin table and rotation (≈ 12 launches).
//   gate:     out * sigmoid(gate), one launch.
import Foundation
import MLX
import MLXFast
import MLXNN

enum Qwen4ExpAttnScratchKernels {
    /// x: [B, S, H, D] (pre-transpose layout). positions: [S] int32.
    /// invFreq: [ROT/2] float32. Output in the same layout.
    static let ropeNorm: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_attn_rope_norm_scratch",
        inputNames: ["x", "w", "eps", "positions", "inv_freq"], outputNames: ["y"],
        source: """
                auto lane = thread_position_in_threadgroup.x;
                auto h = thread_position_in_grid.y;
                auto row = thread_position_in_grid.z;   // b * S + s
                constexpr int per = D / 32;
                constexpr int rotHalf = ROT / 2;
                auto base = (row * H + h) * D;
                float vals[per];
                float ss = 0.0f;
                for (int i = 0; i < per; ++i) { vals[i] = static_cast<float>(x[base + lane * per + i]); ss += vals[i] * vals[i]; }
                ss = simd_sum(ss);
                float inv = metal::rsqrt(ss / float(D) + eps[0]);
                float pos = float(positions[row % S]);
                for (int i = 0; i < per; ++i) {
                    int d = lane * per + i;
                    float v = vals[i] * inv * (static_cast<float>(w[d]) + OFFSET);
                    float out = v;
                    if (d < ROT) {
                        int j = d < rotHalf ? d : d - rotHalf;
                        // partner element, normed and scaled the same way
                        int pd = d < rotHalf ? d + rotHalf : d - rotHalf;
                        float pv = static_cast<float>(x[base + pd]) * inv * (static_cast<float>(w[pd]) + OFFSET);
                        float ang = pos * inv_freq[j];
                        float c = metal::cos(ang), s = metal::sin(ang);
                        out = d < rotHalf ? (v * c - pv * s) : (v * c + pv * s);
                    }
                    y[base + d] = static_cast<InT>(out);
                }
            """)

    static func ropeNorm(_ x: MLXArray, weight: MLXArray, eps: Float, offsetZero: Bool, positions: MLXArray, invFreq: MLXArray, rotaryDims: Int) -> MLXArray {
        let B = x.dim(0), S = x.dim(1), H = x.dim(2), D = x.dim(3)
        precondition(D % 32 == 0 && rotaryDims % 2 == 0)
        return ropeNorm([x, weight, MLXArray([eps]), positions.asType(.int32).reshaped([-1]), invFreq],
            template: [("InT", x.dtype), ("D", D), ("H", H), ("S", S), ("ROT", rotaryDims), ("OFFSET", offsetZero ? 0 : 1)],
            grid: (32, H, B * S), threadGroup: (32, 1, 1), outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }

    static let gate: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_attn_gate_scratch", inputNames: ["out", "gate"], outputNames: ["y"],
        source: """
                auto i = thread_position_in_grid.x;
                float g = static_cast<float>(gate[i]);
                y[i] = static_cast<InT>(static_cast<float>(out[i]) / (1.0f + metal::exp(-g)));
            """)
    static func gate(_ out: MLXArray, gate g: MLXArray) -> MLXArray {
        gate([out, g], template: [("InT", out.dtype)], grid: (out.size, 1, 1), threadGroup: (min(out.size, 256), 1, 1),
            outputShapes: [out.shape], outputDTypes: [out.dtype])[0]
    }
}

extension Qwen4ExpRotary {
    /// Inverse frequencies `[dimensions / 2]`, built once per (dimensions, base).
    var scratchInvFreq: MLXArray {
        Qwen4ExpRotaryInvFreqCache.shared.value(dimensions: dimensions, base: base)
    }
}

final class Qwen4ExpRotaryInvFreqCache: @unchecked Sendable {
    static let shared = Qwen4ExpRotaryInvFreqCache()
    private var cache: [String: MLXArray] = [:]
    private let lock = NSLock()
    func value(dimensions: Int, base: Float) -> MLXArray {
        let key = "\(dimensions)/\(base)"
        lock.lock(); defer { lock.unlock() }
        if let v = cache[key] { return v }
        let even = MLXArray(stride(from: Int32(0), to: Int32(dimensions), by: 2).map { $0 }).asType(.float32)
        let v = MLX.exp(even * (-Foundation.log(base) / Float(dimensions)))
        eval(v)
        cache[key] = v
        return v
    }
}
