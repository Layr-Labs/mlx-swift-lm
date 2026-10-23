// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Eigen Labs.
// Adapted from oMLX v0.7.1.dev2 (33e0b0c20811b41928cfd85b7e0fe346c7b7c1ed),
// omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py: `_QMV_HEADER`
// (glm_qmv_rows, glm_router_topk, glm_clamped_swiglu), `_GATE_UP_SOURCE`,
// `_DOWN_SOURCE` and `_ROUTER_LOGITS_SOURCE`, Copyright oMLX contributors,
// Apache-2.0 (notice: docs/mimo-v26/LICENSE-OMLX).
// Modified: affine 4/8-bit banks only; gate and up share one activation load;
// every token of a step of up to eight selects in the gate/up launch; the down
// launch combines in ascending expert-ID order with the reference's
// per-contribution rounding; the router logits read the stored weight in place
// with eight-wide lanes and the existing native router's sigmoid and sums;
// the router splits each expert row across simdgroups, the top-K replay uses
// simdgroup reductions and NVFP4 lanes use vector loads, all bitwise
// identical to the earlier kernels (`Kernels.reference`).
// glm5d_load_vector and glm5d_qdot copy MLX quantized.h load_vector and qdot
// (MIT, Copyright © 2023-2024 Apple Inc.).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The whole `GLM5NextMoE` of a step of at most `maximumTokens` tokens in three
/// launches, for `GLM5NextExecutionPolicy.RoutedExperts.fusedDecode`:
///
/// 1. Router: FP32 logits from the stored router weight (widened in
///    registers), MLX's sigmoid and the correction bias, as `[tokens, E]`.
/// 2. Gate/up: one threadgroup slice per (token, slot), slots `0..<K` routed
///    and slot `K` the shared expert. Every routed simdgroup replays its token's
///    top-K selection (descending corrected score, lowest ID on a tie); one
///    publishes the indices and the normalized, scaled uncorrected scores. Each
///    simdgroup projects four gate and up rows with MLX's `qmv_fast` lane
///    mapping, rounds both to the activation dtype and applies
///    `glm5NextClampedGLUOperations` with every operation rounded to the
///    activation dtype, as its separate MLX launches do.
/// 3. Down: per token and eight output rows, the routed down rows in ascending
///    expert-ID order, `T(float(T(down)) * w)` added in the activation dtype
///    from zero (the `GLM5NextMoEKernel.ascendingCombine` sequence), then the
///    shared down projection added with one rounding of `routed + shared`.
///
/// NVFP4 banks (`QuantizedSwitchLinear` nvfp4 with per-expert global scales,
/// `GLM5NextNVFP4Linear`) use MLX's `fp_qmv_fast` lane mapping (16 E2M1
/// values and one E4M3 scale per lane and 512-value block); each projection
/// is rounded to the activation dtype, multiplied by its FP32 global scale and
/// rounded again, exactly as the composed modules do. Affine and NVFP4 banks
/// mix freely (routed NVFP4 with an affine shared expert).
///
/// `Kernels.current` (the default) and `Kernels.reference` (the first
/// version, kept for bitwise comparison) produce identical bits: the router
/// splits each expert's 256-column blocks across eight simdgroups and adds
/// the per-lane partials back in block order; the top-K replay finds each
/// pick with a value maximum and a lowest-ID minimum over the simdgroup (the
/// same total order); NVFP4 lanes read their 16 activations and each row's
/// eight packed bytes with vector loads.
///
/// Rounding points are those of `RoutedExperts.nativeCombine`. The router's
/// FP32 dot products and the projections' lane split use their own reduction
/// order, so results are not bitwise and a near-tie expert can change.
/// Weights are read in place; nothing is copied, cast or dequantized. Steps of
/// several tokens use a slot-major grid so tokens that share an expert read its
/// rows back to back.
enum GLM5NextMoEDecodeKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry(String)
        case unsupportedDeviceOrStream
    }

    static let maximumTokens = 8
    static let maximumExperts = 32 * 16
    static let maximumTopK = 16
    /// Output rows per simdgroup and simdgroups per projection threadgroup.
    static let rowsPerSimdgroup = 4
    static let simdgroups = 2
    static let rowsPerThreadgroup = rowsPerSimdgroup * simdgroups
    /// Router rows per logits threadgroup, one simdgroup each (reference).
    static let routerExpertsPerGroup = 4
    /// Simdgroups sharing one expert's router row (current).
    static let routerSplit = 8

    /// Which launches `decode` encodes; both give bitwise-identical results.
    enum Kernels: Sendable {
        /// Split router rows, reduction top-K, vector NVFP4 loads.
        case current
        /// The first version: four experts per router threadgroup, the
        /// shuffle-tree top-K and scalar loads.
        case reference
    }

    static func supports(tokens: Int) -> Bool { (1 ... maximumTokens).contains(tokens) }

    /// One quantized matrix `[rows, inputs]` or expert stack `[E, rows,
    /// inputs]` in a layout the qmv lanes read: affine 4/8-bit with
    /// activation-dtype scales and offsets, or NVFP4 (E2M1 values, one E4M3
    /// scale per 16, FP32 global scale per matrix).
    struct Bank {
        let weight: MLXArray
        let scales: MLXArray
        /// Affine offsets; an unread placeholder for NVFP4.
        let biases: MLXArray
        /// NVFP4 global scales (`[E]` or `[1]`); an unread placeholder for affine.
        let globalScale: MLXArray
        let nvfp4: Bool
        let bits: Int
        let groupSize: Int

        /// nil unless the layout is one of the above, with whole 8-row tiles
        /// and whole `32 * valuesPerLane` blocks.
        init?(
            weight: MLXArray, scales: MLXArray, biases: MLXArray?, globalScale: MLXArray?,
            bits: Int, groupSize: Int, mode: QuantizationMode, leading: [Int], rows: Int,
            inputs: Int, dtype: DType
        ) {
            guard rows % GLM5NextMoEDecodeKernel.rowsPerThreadgroup == 0,
                weight.dtype == .uint32, weight.shape == leading + [rows, inputs * bits / 32]
            else { return nil }
            switch mode {
            case .affine:
                guard let biases, globalScale == nil, [4, 8].contains(bits),
                    [32, 64, 128].contains(groupSize)
                else { return nil }
                let valuesPerLane = 2 * 32 / bits
                guard inputs % (32 * valuesPerLane) == 0, groupSize % valuesPerLane == 0,
                    scales.dtype == dtype, biases.dtype == dtype,
                    scales.shape == leading + [rows, inputs / groupSize],
                    biases.shape == scales.shape
                else { return nil }
                self.biases = biases
                self.globalScale = MLXArray([Float(1), Float(1)])
                nvfp4 = false
            case .nvfp4:
                let scaleCount = max(1, leading.first ?? 1)
                guard biases == nil, bits == 4, groupSize == 16, inputs % 512 == 0,
                    scales.dtype == .uint8, scales.shape == leading + [rows, inputs / 16],
                    globalScale.map({ $0.dtype == .float32 && $0.size == scaleCount }) ?? true
                else { return nil }
                self.biases = MLXArray.zeros([2], dtype: dtype)
                // Kernels receive one-element arrays in constant memory; keep
                // every array a device buffer (the scalar is duplicated).
                let scales = globalScale?.reshaped([scaleCount]) ?? MLXArray.ones([scaleCount])
                self.globalScale = scaleCount == 1 ? concatenated([scales, scales]) : scales
                nvfp4 = true
            default:
                return nil
            }
            (self.weight, self.scales) = (weight, scales)
            (self.bits, self.groupSize) = (bits, groupSize)
        }

        init?(_ linear: Linear, rows: Int, inputs: Int, dtype: DType) {
            guard let packed = linear as? QuantizedLinear, linear.bias == nil else { return nil }
            self.init(
                weight: packed.weight, scales: packed.scales, biases: packed.biases,
                globalScale: (linear as? GLM5NextNVFP4Linear)?.weightScale2,
                bits: packed.bits, groupSize: packed.groupSize,
                mode: packed.mode, leading: [], rows: rows, inputs: inputs, dtype: dtype)
        }

        init?(_ stack: SwitchLinear, experts: Int, rows: Int, inputs: Int, dtype: DType) {
            // `GLM5NextExperts` admits only bias-free banks.
            guard let packed = stack as? QuantizedSwitchLinear else { return nil }
            self.init(
                weight: packed.weight, scales: packed.scales, biases: packed.biases,
                globalScale: packed.globalScales, bits: packed.bits, groupSize: packed.groupSize,
                mode: packed.mode, leading: [experts], rows: rows, inputs: inputs, dtype: dtype)
        }

        var arrays: [MLXArray] { [weight, scales, biases, globalScale] }
    }

    /// Everything one MoE layer's three launches read, for one activation dtype.
    struct Plan {
        let dtype: DType
        let hidden: Int
        let intermediate: Int
        let sharedIntermediate: Int
        let experts: Int
        let topK: Int
        let normalize: Bool
        let routerWeight: MLXArray
        let correctionBias: MLXArray
        /// FP32 `[routed scaling, routed limit, shared limit]`.
        let parameters: MLXArray
        let gate: Bank
        let up: Bank
        let down: Bank
        let sharedGate: Bank
        let sharedUp: Bank
        let sharedDown: Bank

        /// nil when the kernel cannot read this layer: grouped routing, banks
        /// that are neither affine 4/8-bit nor NVFP4 (or whose gate and up
        /// differ), widths that are not whole 512-value (4-bit) or 256-value
        /// (8-bit) blocks.
        init?(_ moe: GLM5NextMoE, dtype: DType) {
            let router = moe.gate
            let experts = moe.experts
            let shared = moe.sharedExperts
            let (d, i, s, e) = (
                experts.hiddenSize, experts.intermediateSize, shared.intermediateSize,
                experts.numExperts
            )
            guard [.float16, .bfloat16, .float32].contains(dtype),
                router.nGroup == router.topkGroup,
                (1 ... GLM5NextMoEDecodeKernel.maximumExperts).contains(e),
                (1 ... GLM5NextMoEDecodeKernel.maximumTopK).contains(router.topK),
                router.topK <= e, router.weight.shape == [e, d],
                router.weight.dtype.isFloatingPoint, router.correctionBias.shape == [e],
                router.correctionBias.dtype.isFloatingPoint,
                let gate = Bank(
                    experts.gateProj, experts: e,
                    rows: i, inputs: d, dtype: dtype),
                let up = Bank(
                    experts.upProj, experts: e, rows: i,
                    inputs: d, dtype: dtype),
                let down = Bank(
                    experts.downProj, experts: e,
                    rows: d, inputs: i, dtype: dtype),
                let sharedGate = Bank(
                    shared.gateProj, rows: s, inputs: d,
                    dtype: dtype),
                let sharedUp = Bank(
                    shared.upProj, rows: s, inputs: d,
                    dtype: dtype),
                let sharedDown = Bank(
                    shared.downProj, rows: d, inputs: s,
                    dtype: dtype),
                (gate.nvfp4, gate.bits, gate.groupSize) == (up.nvfp4, up.bits, up.groupSize),
                (sharedGate.nvfp4, sharedGate.bits, sharedGate.groupSize)
                    == (sharedUp.nvfp4, sharedUp.bits, sharedUp.groupSize)
            else { return nil }
            self.dtype = dtype
            (hidden, intermediate, sharedIntermediate, self.experts) = (d, i, s, e)
            topK = router.topK
            normalize = router.normTopkProb
            routerWeight = router.weight
            correctionBias = router.correctionBias
            parameters = MLXArray([
                router.routedScalingFactor, experts.swigluLimit, shared.swigluLimit,
            ])
            (self.gate, self.up, self.down) = (gate, up, down)
            (self.sharedGate, self.sharedUp, self.sharedDown) = (sharedGate, sharedUp, sharedDown)
        }
    }

    /// Each MoE's plans, built on its first eligible step per activation
    /// dtype. Expert banks are fixed after load, so a plan is never rebuilt.
    final class PlanCache: @unchecked Sendable {
        private let lock = NSLock()
        private var plans: [DType: Plan?] = [:]

        func plan(_ moe: GLM5NextMoE, dtype: DType) -> Plan? {
            lock.lock()
            defer { lock.unlock() }
            if let plan = plans[dtype] { return plan }
            let plan = Plan(moe, dtype: dtype)
            plans[dtype] = plan
            return plan
        }
    }

    struct Decoded {
        /// `[tokens, D]`: routed plus shared.
        let output: MLXArray
        /// UInt32 expert IDs and FP32 scaled weights `[tokens, K]`, in selection order.
        let indices: MLXArray
        let weights: MLXArray
        /// Clamped-GLU activations: routed `[tokens * K, I]` in selection
        /// order, shared `[tokens, S]`.
        let activations: MLXArray
        let sharedActivations: MLXArray
    }

    /// `x` is `[tokens, D]` in the plan's dtype.
    static func decode(_ x: MLXArray, plan: Plan, kernels: Kernels = .current) throws -> Decoded {
        guard x.ndim == 2, supports(tokens: x.dim(0)), x.dim(1) == plan.hidden,
            x.dtype == plan.dtype
        else {
            throw Failure.invalidGeometry("x must be [1...\(maximumTokens), D] in the plan dtype")
        }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let tokens = x.dim(0)
            let (d, e, k) = (plan.hidden, plan.experts, plan.topK)
            let current = kernels == .current
            let scored: [MLXArray]
            if current {
                scored = splitRouterKernel(
                    [x, plan.routerWeight, plan.correctionBias],
                    template: [
                        ("Tokens", tokens), ("Width", d), ("Experts", e), ("Split", routerSplit),
                    ],
                    grid: (32 * routerSplit * e, 1, 1), threadGroup: (32 * routerSplit, 1, 1),
                    outputShapes: [[tokens, e], [tokens, e]], outputDTypes: [.float32, .float32])
            } else {
                let routerGroups = (e + routerExpertsPerGroup - 1) / routerExpertsPerGroup
                scored = routerKernel(
                    [x, plan.routerWeight, plan.correctionBias],
                    template: [
                        ("Tokens", tokens), ("Width", d), ("Experts", e),
                        ("GroupExperts", routerExpertsPerGroup),
                    ],
                    grid: (32 * routerExpertsPerGroup * routerGroups, 1, 1),
                    threadGroup: (32 * routerExpertsPerGroup, 1, 1),
                    outputShapes: [[tokens, e], [tokens, e]], outputDTypes: [.float32, .float32])
            }

            let slotMajor = tokens > 1
            let slots = tokens * (k + 1)
            let gateTiles = max(plan.intermediate, plan.sharedIntermediate) / rowsPerThreadgroup
            let projected = gateUpKernel(
                [x, scored[0], scored[1], plan.parameters] + plan.gate.arrays + plan.up.arrays
                    + plan.sharedGate.arrays + plan.sharedUp.arrays,
                template: [
                    ("T", plan.dtype), ("Hidden", d), ("Inter", plan.intermediate),
                    ("SharedInter", plan.sharedIntermediate), ("Experts", e), ("TopK", k),
                    ("GateBits", plan.gate.bits), ("GateGroup", plan.gate.groupSize),
                    ("SharedBits", plan.sharedGate.bits),
                    ("SharedGroup", plan.sharedGate.groupSize), ("Normalize", plan.normalize),
                    ("SlotMajor", slotMajor), ("GateFP", plan.gate.nvfp4),
                    ("SharedFP", plan.sharedGate.nvfp4), ("FastTopK", current), ("Vec", current),
                ],
                grid: slotMajor
                    ? (32, slots * simdgroups, gateTiles) : (32, gateTiles * simdgroups, slots),
                threadGroup: (32, simdgroups, 1),
                outputShapes: [
                    [tokens * k, plan.intermediate], [tokens, plan.sharedIntermediate],
                    [tokens, k], [tokens, k],
                ],
                outputDTypes: [plan.dtype, plan.dtype, .uint32, .float32])

            let downTiles = d / rowsPerThreadgroup
            let output = downKernel(
                Array(projected) + plan.down.arrays + plan.sharedDown.arrays,
                template: [
                    ("T", plan.dtype), ("Hidden", d), ("Inter", plan.intermediate),
                    ("SharedInter", plan.sharedIntermediate), ("TopK", k),
                    ("DownBits", plan.down.bits), ("DownGroup", plan.down.groupSize),
                    ("SharedDownBits", plan.sharedDown.bits),
                    ("SharedDownGroup", plan.sharedDown.groupSize), ("SlotMajor", slotMajor),
                    ("DownFP", plan.down.nvfp4), ("SharedDownFP", plan.sharedDown.nvfp4),
                    ("Vec", current),
                ],
                grid: slotMajor
                    ? (32, tokens * simdgroups, downTiles) : (32, downTiles * simdgroups, tokens),
                threadGroup: (32, simdgroups, 1),
                outputShapes: [[tokens, d]], outputDTypes: [plan.dtype])[0]
            return Decoded(
                output: output, indices: projected[2], weights: projected[3],
                activations: projected[0], sharedActivations: projected[1])
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let header = """
            #define GLM5D_RPS 4
            #define GLM5D_NSG 2

            // MLX quantized.h load_vector (U = float), 4 and 8 bits.
            template <typename T, int values_per_thread, int bits>
            inline float glm5d_load_vector(const device T* x, thread float* x_thread) {
              float sum = 0;
              if (bits == 4) {
                for (int i = 0; i < values_per_thread; i += 4) {
                  sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
                  x_thread[i] = x[i];
                  x_thread[i + 1] = x[i + 1] / 16.0f;
                  x_thread[i + 2] = x[i + 2] / 256.0f;
                  x_thread[i + 3] = x[i + 3] / 4096.0f;
                }
              } else if (bits == 8) {
                for (int i = 0; i < values_per_thread; i++) {
                  sum += x[i];
                  x_thread[i] = x[i];
                }
              }
              return sum;
            }

            // MLX quantized.h qdot (U = float), 4 and 8 bits.
            template <int values_per_thread, int bits>
            inline float glm5d_qdot(
                const device uint8_t* w, const thread float* x_thread, float scale, float bias,
                float sum) {
              float accum = 0;
              if (bits == 4) {
                const device uint16_t* ws = (const device uint16_t*)w;
                for (int i = 0; i < (values_per_thread / 4); i++) {
                  accum +=
                      (x_thread[4 * i] * (ws[i] & 0x000f) +
                       x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                       x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                       x_thread[4 * i + 3] * (ws[i] & 0xf000));
                }
              } else if (bits == 8) {
                for (int i = 0; i < values_per_thread; i++) {
                  accum += x_thread[i] * w[i];
                }
              }
              return scale * accum + sum * bias;
            }

            // RPS consecutive rows of one [N, K] affine matrix (pointers at the
            // first row); per-lane partial sums are left for simd_sum.
            template <typename T, int K, int GS, int BITS, int RPS>
            inline void glm5d_qmv_rows(
                const device uint8_t* ws, const device T* scales, const device T* biases,
                const device T* x, uint simd_lid, thread float* result) {
              // qmv_fast lane mapping: two packs per lane, 32 lanes per block.
              constexpr int vpt = 2 * 32 / BITS;
              constexpr int block = 32 * vpt;
              constexpr int row_bytes = K * BITS / 8;
              constexpr int row_groups = K / GS;
              thread float x_thread[vpt];
              ws += simd_lid * vpt * BITS / 8;
              scales += simd_lid / (GS / vpt);
              biases += simd_lid / (GS / vpt);
              x += simd_lid * vpt;
              for (int k = 0; k < K; k += block) {
                const float sum = glm5d_load_vector<T, vpt, BITS>(x, x_thread);
                for (int row = 0; row < RPS; row++) {
                  result[row] += glm5d_qdot<vpt, BITS>(
                      ws + row * row_bytes, x_thread, float(scales[row * row_groups]),
                      float(biases[row * row_groups]), sum);
                }
                ws += block * BITS / 8;
                scales += block / GS;
                biases += block / GS;
                x += block;
              }
            }

            // glm5d_qmv_rows for two matrices of one input (gate and up),
            // sharing each activation load; per-row arithmetic is unchanged.
            template <typename T, int K, int GS, int BITS, int RPS>
            inline void glm5d_qmv_rows2(
                const device uint8_t* gw, const device T* gs, const device T* gb,
                const device uint8_t* uw, const device T* us, const device T* ub,
                const device T* x, uint simd_lid, thread float* g_res, thread float* u_res) {
              // qmv_fast lane mapping: two packs per lane, 32 lanes per block.
              constexpr int vpt = 2 * 32 / BITS;
              constexpr int block = 32 * vpt;
              constexpr int row_bytes = K * BITS / 8;
              constexpr int row_groups = K / GS;
              thread float x_thread[vpt];
              const int w_lane = simd_lid * vpt * BITS / 8;
              const int g_lane = simd_lid / (GS / vpt);
              gw += w_lane;
              uw += w_lane;
              gs += g_lane;
              gb += g_lane;
              us += g_lane;
              ub += g_lane;
              x += simd_lid * vpt;
              for (int k = 0; k < K; k += block) {
                const float sum = glm5d_load_vector<T, vpt, BITS>(x, x_thread);
                for (int row = 0; row < RPS; row++) {
                  g_res[row] += glm5d_qdot<vpt, BITS>(
                      gw + row * row_bytes, x_thread, float(gs[row * row_groups]),
                      float(gb[row * row_groups]), sum);
                  u_res[row] += glm5d_qdot<vpt, BITS>(
                      uw + row * row_bytes, x_thread, float(us[row * row_groups]),
                      float(ub[row * row_groups]), sum);
                }
                gw += block * BITS / 8;
                uw += block * BITS / 8;
                gs += block / GS;
                gb += block / GS;
                us += block / GS;
                ub += block / GS;
                x += block;
              }
            }

            // MLX's Sigmoid and Minimum/Maximum functors (unary_ops.h,
            // binary_ops.h), as the compiled clamped GLU applies them.
            template <typename T>
            inline T glm5d_sigmoid(T x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }
            template <typename T>
            inline T glm5d_minimum(T x, T y) {
              if (metal::isnan(x)) {
                return x;
              }
              return x < y ? x : y;
            }
            template <typename T>
            inline T glm5d_maximum(T x, T y) {
              if (metal::isnan(x)) {
                return x;
              }
              return x > y ? x : y;
            }

            // silu(minimum(gate, limit)) * minimum(maximum(up, -limit), limit),
            // every operation rounded to T as separate MLX launches do (MLXNN's
            // compiled silu rounds its sigmoid to T as well).
            template <typename T>
            inline T glm5d_clamped_glu(T gate, T up, T limit, T neg_limit) {
              T g = glm5d_minimum(gate, limit);
              T s = g * glm5d_sigmoid(g);
              T u = glm5d_minimum(glm5d_maximum(up, neg_limit), limit);
              return s * u;
            }

            // The projection's activation-dtype output; NVFP4 then applies the
            // FP32 global scale and rounds again (the reference composition).
            template <typename T, bool FP>
            inline T glm5d_output(float acc, float global_scale) {
              const T y = static_cast<T>(acc);
              if constexpr (FP) {
                return static_cast<T>(static_cast<float>(y) * global_scale);
              }
              return y;
            }

            template <typename T, bool FP>
            inline void glm5d_store_glu(
                device T* out, thread float* g_res, thread float* u_res, float limit,
                float gate_scale, float up_scale, uint simd_lid) {
              const T lim = static_cast<T>(limit);
              const T neg_lim = static_cast<T>(-limit);
              for (int row = 0; row < GLM5D_RPS; row++) {
                const float g = simd_sum(g_res[row]);
                const float u = simd_sum(u_res[row]);
                if (simd_lid == 0) {
                  out[row] = glm5d_clamped_glu<T>(
                      glm5d_output<T, FP>(g, gate_scale), glm5d_output<T, FP>(u, up_scale), lim,
                      neg_lim);
                }
              }
            }

            // MLX fp4.h / fp8.h conversions (E2M1 and E4M3 through half; exact).
            inline float glm5d_e2m1(uint8_t bits) {
              half converted = as_type<half>(ushort((bits & 7) << 9));
              converted *= 16384.0;
              return static_cast<float>(bits & 8 ? -converted : converted);
            }
            inline float glm5d_e4m3(uint8_t bits) {
              uint16_t v = (bits & 127) << 7;
              half converted = as_type<half>(v);
              converted *= 256.0;
              return static_cast<float>((bits & 128) ? -converted : converted);
            }

            // MLX fp_quantized.h qdot (U = float, 4 bits).
            template <int values_per_thread>
            inline float glm5d_fp_qdot(const device uint8_t* w, const thread float* x_thread, float scale) {
              float accum = 0;
              const device uint16_t* ws = (const device uint16_t*)w;
              for (int i = 0; i < (values_per_thread / 4); i++) {
                accum +=
                    (x_thread[4 * i] * glm5d_e2m1(uint8_t(ws[i])) +
                     x_thread[4 * i + 1] * glm5d_e2m1(uint8_t(ws[i] >> 4)) +
                     x_thread[4 * i + 2] * glm5d_e2m1(uint8_t(ws[i] >> 8)) +
                     x_thread[4 * i + 3] * glm5d_e2m1(uint8_t(ws[i] >> 12)));
              }
              return scale * accum;
            }

            // MLX fp_qmv_fast_impl (NVFP4: two packs, 16 values, one E4M3 scale per
            // lane and 512-value block) for RPS rows; one or two matrices of one input.
            template <typename T, int K, int RPS, int NM>
            inline void glm5d_fp_qmv_rows(
                const device uint8_t* w0, const device uint8_t* s0,
                const device uint8_t* w1, const device uint8_t* s1,
                const device T* x, uint simd_lid, thread float* r0, thread float* r1) {
              constexpr int vpt = 16;
              constexpr int block = 512;
              constexpr int row_bytes = K / 2;
              constexpr int row_groups = K / 16;
              thread float x_thread[vpt];
              w0 += simd_lid * 8;
              s0 += simd_lid;
              w1 += simd_lid * 8;
              s1 += simd_lid;
              x += simd_lid * vpt;
              for (int k = 0; k < K; k += block) {
                for (int i = 0; i < vpt; i++) {
                  x_thread[i] = x[i];
                }
                for (int row = 0; row < RPS; row++) {
                  r0[row] += glm5d_fp_qdot<vpt>(
                      w0 + row * row_bytes, x_thread, glm5d_e4m3(s0[row * row_groups]));
                  if (NM == 2) {
                    r1[row] += glm5d_fp_qdot<vpt>(
                        w1 + row * row_bytes, x_thread, glm5d_e4m3(s1[row * row_groups]));
                  }
                }
                w0 += block / 2;
                w1 += block / 2;
                s0 += block / 16;
                s1 += block / 16;
                x += block;
              }
            }

            // glm5d_fp_qdot on one lane's eight packed bytes, read as one uint2.
            inline float glm5d_fp_qdot_packed(uint2 w, const thread float* x_thread, float scale) {
              const ushort ws[4] = {
                  ushort(w.x & 0xffff), ushort(w.x >> 16), ushort(w.y & 0xffff), ushort(w.y >> 16)};
              float accum = 0;
              for (int i = 0; i < 4; i++) {
                accum +=
                    (x_thread[4 * i] * glm5d_e2m1(uint8_t(ws[i])) +
                     x_thread[4 * i + 1] * glm5d_e2m1(uint8_t(ws[i] >> 4)) +
                     x_thread[4 * i + 2] * glm5d_e2m1(uint8_t(ws[i] >> 8)) +
                     x_thread[4 * i + 3] * glm5d_e2m1(uint8_t(ws[i] >> 12)));
              }
              return scale * accum;
            }

            // glm5d_fp_qmv_rows with vector loads: each lane's 16 activations
            // as four 4-wide loads and each row's eight packed bytes as one
            // uint2; the same values in the same order.
            template <typename T, int K, int RPS, int NM>
            inline void glm5d_fp_qmv_rows_vec(
                const device uint8_t* w0, const device uint8_t* s0,
                const device uint8_t* w1, const device uint8_t* s1,
                const device T* x, uint simd_lid, thread float* r0, thread float* r1) {
              constexpr int vpt = 16;
              constexpr int block = 512;
              constexpr int row_bytes = K / 2;
              constexpr int row_groups = K / 16;
              thread float x_thread[vpt];
              w0 += simd_lid * 8;
              s0 += simd_lid;
              w1 += simd_lid * 8;
              s1 += simd_lid;
              x += simd_lid * vpt;
              for (int k = 0; k < K; k += block) {
                for (int i = 0; i < vpt; i += 4) {
                  const vec<T, 4> xv = *(const device vec<T, 4>*)(x + i);
                  x_thread[i] = xv[0];
                  x_thread[i + 1] = xv[1];
                  x_thread[i + 2] = xv[2];
                  x_thread[i + 3] = xv[3];
                }
                uint2 p0[RPS];
                uint2 p1[RPS];
                for (int row = 0; row < RPS; row++) {
                  p0[row] = *(const device uint2*)(w0 + row * row_bytes);
                  if (NM == 2) {
                    p1[row] = *(const device uint2*)(w1 + row * row_bytes);
                  }
                }
                for (int row = 0; row < RPS; row++) {
                  r0[row] += glm5d_fp_qdot_packed(
                      p0[row], x_thread, glm5d_e4m3(s0[row * row_groups]));
                  if (NM == 2) {
                    r1[row] += glm5d_fp_qdot_packed(
                        p1[row], x_thread, glm5d_e4m3(s1[row * row_groups]));
                  }
                }
                w0 += block / 2;
                w1 += block / 2;
                s0 += block / 16;
                s1 += block / 16;
                x += block;
              }
            }

            // Either format: FP selects NVFP4 (scales are E4M3 bytes, no offsets).
            // S and B are pointer types: an unread placeholder may arrive in
            // constant memory.
            template <typename T, int K, int GS, int BITS, bool FP, int RPS, typename S, typename B>
            inline void glm5d_bank_rows(
                const device uint8_t* w, S s, B b,
                const device T* x, uint lane, thread float* result) {
              if constexpr (FP) {
                glm5d_fp_qmv_rows<T, K, RPS, 1>(
                    w, (const device uint8_t*)s, w, (const device uint8_t*)s, x, lane, result,
                    result);
              } else {
                glm5d_qmv_rows<T, K, GS, BITS, RPS>(
                    w, (const device T*)s, (const device T*)b, x, lane, result);
              }
            }

            template <typename T, int K, int GS, int BITS, bool FP, int RPS, typename S, typename B>
            inline void glm5d_bank_rows2(
                const device uint8_t* gw, S gs, B gb, const device uint8_t* uw, S us, B ub,
                const device T* x, uint lane, thread float* g_res, thread float* u_res) {
              if constexpr (FP) {
                glm5d_fp_qmv_rows<T, K, RPS, 2>(
                    gw, (const device uint8_t*)gs, uw, (const device uint8_t*)us, x, lane, g_res,
                    u_res);
              } else {
                glm5d_qmv_rows2<T, K, GS, BITS, RPS>(
                    gw, (const device T*)gs, (const device T*)gb, uw, (const device T*)us,
                    (const device T*)ub, x, lane, g_res, u_res);
              }
            }

            template <typename T, int K, int GS, int BITS, bool FP, int RPS, bool VEC, typename S, typename B>
            inline void glm5d_bank_rows_v(
                const device uint8_t* w, S s, B b,
                const device T* x, uint lane, thread float* result) {
              if constexpr (FP && VEC) {
                glm5d_fp_qmv_rows_vec<T, K, RPS, 1>(
                    w, (const device uint8_t*)s, w, (const device uint8_t*)s, x, lane, result,
                    result);
              } else {
                glm5d_bank_rows<T, K, GS, BITS, FP, RPS>(w, s, b, x, lane, result);
              }
            }

            template <typename T, int K, int GS, int BITS, bool FP, int RPS, bool VEC, typename S, typename B>
            inline void glm5d_bank_rows2_v(
                const device uint8_t* gw, S gs, B gb, const device uint8_t* uw, S us, B ub,
                const device T* x, uint lane, thread float* g_res, thread float* u_res) {
              if constexpr (FP && VEC) {
                glm5d_fp_qmv_rows_vec<T, K, RPS, 2>(
                    gw, (const device uint8_t*)gs, uw, (const device uint8_t*)us, x, lane, g_res,
                    u_res);
              } else {
                glm5d_bank_rows2<T, K, GS, BITS, FP, RPS>(gw, gs, gb, uw, us, ub, x, lane, g_res, u_res);
              }
            }

            // Top-K of one token's corrected scores, one simdgroup: descending
            // value, lowest expert ID on a tie, NaNs last. Every lane ends with
            // the same picked[].
            template <int E, int TOPK>
            inline void glm5d_router_topk(const device float* bz, uint lane, thread int* picked) {
              constexpr int PER = (E + 31) / 32;
              float vals[PER];
              bool taken[PER];
              for (int j = 0; j < PER; j++) {
                const int e = j * 32 + int(lane);
                vals[j] = e < E ? bz[e] : -INFINITY;
                taken[j] = e >= E;
              }
              for (int r = 0; r < TOPK; r++) {
                float best = -INFINITY;
                int best_e = 0x7fffffff;
                for (int j = 0; j < PER; j++) {
                  const int e = j * 32 + int(lane);
                  if (!taken[j] && !isnan(vals[j]) &&
                      (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
                    best = vals[j];
                    best_e = e;
                  }
                }
                for (ushort off = 16; off >= 1; off >>= 1) {
                  float ob = simd_shuffle_xor(best, off);
                  int oe = simd_shuffle_xor(best_e, off);
                  const bool other_better = oe != 0x7fffffff &&
                      (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
                  if (other_better) {
                    best = ob;
                    best_e = oe;
                  }
                }
                if (best_e == 0x7fffffff) {
                  for (int j = 0; j < PER; j++) {
                    const int e = j * 32 + int(lane);
                    if (!taken[j] && e < best_e) {
                      best_e = e;
                    }
                  }
                  for (ushort off = 16; off >= 1; off >>= 1) {
                    best_e = min(best_e, simd_shuffle_xor(best_e, off));
                  }
                }
                picked[r] = best_e;
                if ((best_e % 32) == int(lane)) {
                  taken[best_e / 32] = true;
                }
              }
            }

            // glm5d_router_topk with the same order (descending value, lowest
            // expert ID on a tie, NaNs last): each lane's best untaken entry,
            // then the value maximum and the lowest ID holding it as two
            // simdgroup reductions; the taken flags keep static indices.
            template <int E, int TOPK>
            inline void glm5d_router_topk_fast(
                const device float* bz, uint lane, thread int* picked) {
              constexpr int PER = (E + 31) / 32;
              constexpr int NONE = 0x7fffffff;
              float vals[PER];
              bool taken[PER];
              for (int j = 0; j < PER; j++) {
                const int e = j * 32 + int(lane);
                vals[j] = e < E ? bz[e] : -INFINITY;
                taken[j] = e >= E;
              }
              for (int r = 0; r < TOPK; r++) {
                float best = -INFINITY;
                int best_e = NONE;
                for (int j = 0; j < PER; j++) {
                  const int e = j * 32 + int(lane);
                  if (!taken[j] && !isnan(vals[j]) &&
                      (best_e == NONE || vals[j] > best || (vals[j] == best && e < best_e))) {
                    best = vals[j];
                    best_e = e;
                  }
                }
                const float top = simd_max(best);
                int winner = simd_min(best_e != NONE && best == top ? best_e : NONE);
                if (winner == NONE) {
                  int lowest = NONE;
                  for (int j = 0; j < PER; j++) {
                    const int e = j * 32 + int(lane);
                    if (!taken[j] && e < lowest) {
                      lowest = e;
                    }
                  }
                  winner = simd_min(lowest);
                }
                picked[r] = winner;
                for (int j = 0; j < PER; j++) {
                  if (winner == j * 32 + int(lane)) {
                    taken[j] = true;
                  }
                }
              }
            }
            """

        // SIMD group g of threadgroup b owns expert b*GroupExperts+g; lane l
        // reads eight contiguous columns of every 256-column block.
        private static let routerKernel = MLXFast.metalKernel(
            name: "glm5_next_moe_decode_router",
            inputNames: ["x", "weight", "bias"], outputNames: ["scores", "corrected"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                const uint expert = threadgroup_position_in_grid.x * GroupExperts
                    + simdgroup_index_in_threadgroup;
                if (expert >= uint(Experts)) return;
                const uint lane = thread_index_in_simdgroup;
                const size_t row = size_t(expert) * Width;
                float totals[Tokens];
                for (int t = 0; t < Tokens; ++t) { totals[t] = 0.0f; }
                for (int column = int(lane) * 8; column < Width; column += 256) {
                    float w[8];
                    for (int i = 0; i < 8; ++i) { w[i] = static_cast<float>(weight[row + column + i]); }
                    for (int t = 0; t < Tokens; ++t) {
                        const size_t base = size_t(t) * Width + column;
                        float partial = 0.0f;
                        for (int i = 0; i < 8; ++i) {
                            const float product = static_cast<float>(x[base + i]) * w[i];
                            partial += product;
                        }
                        totals[t] += partial;
                    }
                }
                for (int t = 0; t < Tokens; ++t) {
                    const float logit = simd_sum(totals[t]);
                    if (lane == 0) {
                        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(logit)));
                        const float score = logit < 0.0f ? y : 1.0f - y;
                        scores[size_t(t) * Experts + expert] = score;
                        corrected[size_t(t) * Experts + expert] =
                            score + static_cast<float>(bias[expert]);
                    }
                }
                """,
            ensureRowContiguous: true)

        // Split simdgroups of one threadgroup own one expert: simdgroup s
        // computes the eight-column partials of its share of the 256-column
        // blocks; simdgroup 0 adds them per lane in block order from zero (the
        // legacy kernel's per-lane sequence), then sums the lanes.
        private static let splitRouterKernel = MLXFast.metalKernel(
            name: "glm5_next_moe_decode_router_split",
            inputNames: ["x", "weight", "bias"], outputNames: ["scores", "corrected"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr int NB = (Width + 255) / 256;
                constexpr int PER = (NB + Split - 1) / Split;
                threadgroup float part[Tokens * NB * 32];
                const uint expert = threadgroup_position_in_grid.x;
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const size_t row = size_t(expert) * Width;
                for (int jj = 0; jj < PER; ++jj) {
                    const int j = int(sg) * PER + jj;
                    const int column = int(lane) * 8 + j * 256;
                    if (j < NB && column < Width) {
                        float w[8];
                        for (int i = 0; i < 8; ++i) { w[i] = static_cast<float>(weight[row + column + i]); }
                        for (int t = 0; t < Tokens; ++t) {
                            const size_t base = size_t(t) * Width + column;
                            float partial = 0.0f;
                            for (int i = 0; i < 8; ++i) {
                                const float product = static_cast<float>(x[base + i]) * w[i];
                                partial += product;
                            }
                            part[(t * NB + j) * 32 + lane] = partial;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg != 0) return;
                float totals[Tokens];
                for (int t = 0; t < Tokens; ++t) { totals[t] = 0.0f; }
                for (int j = 0; j < NB; ++j) {
                    if (int(lane) * 8 + j * 256 < Width) {
                        for (int t = 0; t < Tokens; ++t) { totals[t] += part[(t * NB + j) * 32 + lane]; }
                    }
                }
                for (int t = 0; t < Tokens; ++t) {
                    const float logit = simd_sum(totals[t]);
                    if (lane == 0) {
                        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(logit)));
                        const float score = logit < 0.0f ? y : 1.0f - y;
                        scores[size_t(t) * Experts + expert] = score;
                        corrected[size_t(t) * Experts + expert] =
                            score + static_cast<float>(bias[expert]);
                    }
                }
                """,
            ensureRowContiguous: true)

        private static let gateUpKernel = MLXFast.metalKernel(
            name: "glm5_next_moe_decode_gate_up",
            inputNames: [
                "x", "scores", "corrected", "parameters", "gate_w", "gate_s", "gate_b", "gate_g",
                "up_w", "up_s", "up_b", "up_g", "sh_gate_w", "sh_gate_s", "sh_gate_b",
                "sh_gate_g", "sh_up_w", "sh_up_s", "sh_up_b", "sh_up_g",
            ],
            outputNames: ["act", "shared_act", "sel_indices", "sel_weights"],
            source: """
                const uint simd_lid = thread_index_in_simdgroup;
                const uint simd_gid = simdgroup_index_in_threadgroup;
                // Slot-major: the slots of a tile, for every token, run back to back.
                const int tile = int(SlotMajor ? threadgroup_position_in_grid.z
                                               : threadgroup_position_in_grid.y);
                const int z = int(SlotMajor ? threadgroup_position_in_grid.y
                                            : threadgroup_position_in_grid.z);
                constexpr int RT = TopK + 1;
                const int token = z / RT;
                const int r = z - token * RT;
                const int out_row = (tile * GLM5D_NSG + int(simd_gid)) * GLM5D_RPS;
                const device T* xr = x + size_t(token) * Hidden;
                float g_res[GLM5D_RPS] = {0};
                float u_res[GLM5D_RPS] = {0};
                if (r < TopK) {
                    int picked[TopK];
                    if (FastTopK) {
                        glm5d_router_topk_fast<Experts, TopK>(
                            corrected + size_t(token) * Experts, simd_lid, picked);
                    } else {
                        glm5d_router_topk<Experts, TopK>(
                            corrected + size_t(token) * Experts, simd_lid, picked);
                    }
                    if (r == 0 && tile == 0 && simd_gid == 0 && simd_lid == 0) {
                        // The native router's weights: uncorrected scores summed in
                        // selection order.
                        float chosen[TopK];
                        float total = 0.0f;
                        for (int q = 0; q < TopK; ++q) {
                            chosen[q] = scores[size_t(token) * Experts + picked[q]];
                            if (Normalize) { total += chosen[q]; }
                        }
                        total = total + 1e-20f;
                        for (int q = 0; q < TopK; ++q) {
                            const float normalized = Normalize ? chosen[q] / total : chosen[q];
                            sel_weights[size_t(token) * TopK + q] = normalized * parameters[0];
                            sel_indices[size_t(token) * TopK + q] = uint(picked[q]);
                        }
                    }
                    if (out_row >= Inter) return;
                    constexpr int WB = Hidden * GateBits / 8;
                    constexpr int G = Hidden / GateGroup;
                    const int expert = picked[r];
                    const size_t row0 = size_t(expert) * Inter + out_row;
                    glm5d_bank_rows2_v<T, Hidden, GateGroup, GateBits, GateFP, GLM5D_RPS, Vec>(
                        (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
                        gate_b + row0 * G, (const device uint8_t*)up_w + row0 * WB,
                        up_s + row0 * G, up_b + row0 * G, xr, simd_lid, g_res, u_res);
                    glm5d_store_glu<T, GateFP>(
                        act + (size_t(token) * TopK + r) * Inter + out_row, g_res, u_res,
                        parameters[1], GateFP ? gate_g[expert] : 1.0f,
                        GateFP ? up_g[expert] : 1.0f, simd_lid);
                } else {
                    if (out_row >= SharedInter) return;
                    constexpr int WB = Hidden * SharedBits / 8;
                    constexpr int G = Hidden / SharedGroup;
                    const size_t row0 = size_t(out_row);
                    glm5d_bank_rows2_v<T, Hidden, SharedGroup, SharedBits, SharedFP, GLM5D_RPS, Vec>(
                        (const device uint8_t*)sh_gate_w + row0 * WB, sh_gate_s + row0 * G,
                        sh_gate_b + row0 * G, (const device uint8_t*)sh_up_w + row0 * WB,
                        sh_up_s + row0 * G, sh_up_b + row0 * G, xr, simd_lid, g_res, u_res);
                    glm5d_store_glu<T, SharedFP>(
                        shared_act + size_t(token) * SharedInter + out_row, g_res, u_res,
                        parameters[2], SharedFP ? sh_gate_g[0] : 1.0f,
                        SharedFP ? sh_up_g[0] : 1.0f, simd_lid);
                }
                """,
            header: header,
            ensureRowContiguous: true)

        private static let downKernel = MLXFast.metalKernel(
            name: "glm5_next_moe_decode_down",
            inputNames: [
                "act", "shared_act", "indices", "weights", "down_w", "down_s", "down_b",
                "down_g", "sh_down_w", "sh_down_s", "sh_down_b", "sh_down_g",
            ],
            outputNames: ["y"],
            source: """
                #pragma clang fp contract(off)
                const uint simd_lid = thread_index_in_simdgroup;
                const uint simd_gid = simdgroup_index_in_threadgroup;
                // Slot-major: the tokens of a tile run back to back.
                const int tile = int(SlotMajor ? threadgroup_position_in_grid.z
                                               : threadgroup_position_in_grid.y);
                const int token = int(SlotMajor ? threadgroup_position_in_grid.y
                                                : threadgroup_position_in_grid.z);
                const int out_row = (tile * GLM5D_NSG + int(simd_gid)) * GLM5D_RPS;
                const size_t first = size_t(token) * TopK;
                // Stable insertion sort of this token's slots by expert ID.
                uint ids[TopK];
                int order[TopK];
                for (int slot = 0; slot < TopK; ++slot) {
                    const uint id = indices[first + slot];
                    int position = slot;
                    while (position > 0 && ids[position - 1] > id) {
                        ids[position] = ids[position - 1];
                        order[position] = order[position - 1];
                        --position;
                    }
                    ids[position] = id;
                    order[position] = slot;
                }
                T total[GLM5D_RPS];
                for (int row = 0; row < GLM5D_RPS; ++row) { total[row] = static_cast<T>(0.0f); }
                constexpr int WB = Inter * DownBits / 8;
                constexpr int G = Inter / DownGroup;
                for (int rank = 0; rank < TopK; ++rank) {
                    const int slot = order[rank];
                    const size_t row0 = size_t(ids[rank]) * Hidden + out_row;
                    float res[GLM5D_RPS] = {0};
                    glm5d_bank_rows_v<T, Inter, DownGroup, DownBits, DownFP, GLM5D_RPS, Vec>(
                        (const device uint8_t*)down_w + row0 * WB, down_s + row0 * G,
                        down_b + row0 * G, act + (first + slot) * Inter, simd_lid, res);
                    const float weight = weights[first + slot];
                    const float down_scale = DownFP ? down_g[ids[rank]] : 1.0f;
                    for (int row = 0; row < GLM5D_RPS; ++row) {
                        const float projected = simd_sum(res[row]);
                        const float weighted = static_cast<float>(
                            glm5d_output<T, DownFP>(projected, down_scale)) * weight;
                        const T contribution = static_cast<T>(weighted);
                        total[row] = static_cast<T>(total[row] + contribution);
                    }
                }
                constexpr int SWB = SharedInter * SharedDownBits / 8;
                constexpr int SG = SharedInter / SharedDownGroup;
                float shared[GLM5D_RPS] = {0};
                glm5d_bank_rows_v<T, SharedInter, SharedDownGroup, SharedDownBits, SharedDownFP,
                                GLM5D_RPS, Vec>(
                    (const device uint8_t*)sh_down_w + size_t(out_row) * SWB,
                    sh_down_s + size_t(out_row) * SG, sh_down_b + size_t(out_row) * SG,
                    shared_act + size_t(token) * SharedInter, simd_lid, shared);
                const float shared_scale = SharedDownFP ? sh_down_g[0] : 1.0f;
                for (int row = 0; row < GLM5D_RPS; ++row) {
                    const float projected = simd_sum(shared[row]);
                    if (simd_lid == 0) {
                        y[size_t(token) * Hidden + out_row + row] = static_cast<T>(
                            total[row] + glm5d_output<T, SharedDownFP>(projected, shared_scale));
                    }
                }
                """,
            header: header,
            ensureRowContiguous: true)
    #endif
}
