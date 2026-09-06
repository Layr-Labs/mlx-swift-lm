// DIAGNOSTIC (scratch): the MoE block's non-matmul ops as custom kernels.
//   router: top-k over E logits per row + softmax over the selected k, one
//           kernel (replaces negate, argPartition, slice, takeAlong, softmax).
//   tail:   sum_k(experts[k] * weight[k]) + sigmoid(gate) * shared, one
//           kernel (replaces multiply, sum, cast, sigmoid, multiply, add).
import Foundation
import MLX
import MLXFast
import MLXNN

enum Qwen4ExpMoEScratchKernels {
    /// One simdgroup per row: each lane owns E/32 logits, the top-k is found
    /// by k rounds of simd_max over the remaining values (E ≤ 1024, k ≤ 16).
    /// Ties resolve to the lowest index, as argPartition's descending order
    /// followed by the softmax over the selected logits is order-agnostic
    /// except for which of equal logits is chosen.
    static let router: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_moe_router_scratch", inputNames: ["logits"], outputNames: ["indices", "weights"],
        source: """
                auto lane = thread_position_in_threadgroup.x;
                auto row = thread_position_in_grid.y;
                constexpr int per = E / 32;
                float vals[per];
                for (int i = 0; i < per; ++i) { vals[i] = logits[row * E + lane * per + i]; }
                float picked[KTOP];
                for (int r = 0; r < KTOP; ++r) {
                    float best = -INFINITY; int bestIdx = 0x7fffffff;
                    for (int i = 0; i < per; ++i) {
                        if (vals[i] > best) { best = vals[i]; bestIdx = lane * per + i; }
                    }
                    float gbest = simd_max(best);
                    // lowest index among lanes holding gbest
                    int cand = (best == gbest) ? bestIdx : 0x7fffffff;
                    int gidx = simd_min(cand);
                    if (lane == gidx / per) { vals[gidx - lane * per] = -INFINITY; }
                    picked[r] = gbest;
                    if (lane == 0) { indices[row * KTOP + r] = uint(gidx); }
                }
                if (lane == 0) {
                    float m = picked[0];
                    float sum = 0.0f;
                    for (int r = 0; r < KTOP; ++r) { sum += metal::exp(picked[r] - m); }
                    for (int r = 0; r < KTOP; ++r) { weights[row * KTOP + r] = metal::exp(picked[r] - m) / sum; }
                }
            """)

    /// - Parameter logits: `[rows, E]` float32.
    static func router(logits: MLXArray, topK: Int) -> (indices: MLXArray, weights: MLXArray) {
        let E = logits.dim(-1)
        precondition(E % 32 == 0 && topK <= 16, "router kernel: E multiple of 32, k <= 16")
        let rows = logits.size / E
        let lead = Array(logits.shape.dropLast())
        let outputs = router([logits], template: [("E", E), ("KTOP", topK)],
            grid: (32, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [lead + [topK], lead + [topK]], outputDTypes: [.uint32, .float32])
        return (outputs[0], outputs[1])
    }

    /// One thread per (row, d).
    static let tail: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_moe_tail_scratch", inputNames: ["experts", "weights", "gate", "shared"], outputNames: ["y"],
        source: """
                auto d = thread_position_in_grid.x;
                auto row = thread_position_in_grid.y;
                if (d >= D) return;
                float acc = 0.0f;
                for (int k = 0; k < KTOP; ++k) {
                    acc += static_cast<float>(experts[(row * KTOP + k) * D + d]) * weights[row * KTOP + k];
                }
                float g = static_cast<float>(gate[row]);
                float sg = 1.0f / (1.0f + metal::exp(-g));
                y[row * D + d] = static_cast<InT>(acc + sg * static_cast<float>(shared[row * D + d]));
            """)

    static func tail(experts: MLXArray, weights: MLXArray, gate: MLXArray, shared: MLXArray) -> MLXArray {
        let D = shared.dim(-1)
        let topK = weights.dim(-1)
        let rows = shared.size / D
        return tail([experts, weights.asType(.float32), gate, shared],
            template: [("InT", shared.dtype), ("D", D), ("KTOP", topK)],
            grid: (D, rows, 1), threadGroup: (min(D, 256), 1, 1),
            outputShapes: [shared.shape], outputDTypes: [shared.dtype])[0]
    }
}
