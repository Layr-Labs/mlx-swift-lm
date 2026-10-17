// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Eigen Labs.
// Adapted from oMLX (Apache-2.0, Copyright oMLX contributors; notice:
// docs/mimo-v26/LICENSE-OMLX): #3995 47876fbc310fbb311cd382c4eba8572ec4368308
// (segmented single-expert NAX tiles), #3993 a68cab68ae40eeab7062ba7c1eff4ddffa06e373
// (joint gate/up), #4022 d6dbc9970c6df99b6a8589746e6ff8ddc38c33ff (SwiGLU
// epilogue), #4029 498685bae7b536cbd0d7d5e145b0327544e2fb06 (row-map reads),
// through the pinned MiMo port in `MiMoV26NAXMetalSources` (Apple MLX MIT
// primitives, notice retained there). Modified: affine 4-bit group dequant
// with activation-dtype scales and offsets; the clamped-GLU epilogue; token
// rows read through the sort order (`order / K`); inverse by scatter.

import Foundation
import MLX
import MLXLMCommon

/// Routed experts of a long step for
/// `GLM5NextExecutionPolicy.RoutedExperts.fusedDecodeNAXPrefill`:
///
/// 1. Sort the assignments by expert (MLX argsort), cut the sorted run into
///    single-expert tiles of at most 64/128 rows (one scan launch).
/// 2. Gate and up on the Neural Accelerators in one launch: both weight tiles
///    are dequantized to the activation dtype, each token row is read once
///    through the sort order (no replicated `[T·K, D]` copy), the FP32
///    accumulators are rounded to the activation dtype and the clamped GLU is
///    applied in that dtype in the epilogue; only `[T·K, I]` is written.
/// 3. Down on the same tiles, rounded to the activation dtype.
/// 4. The inverse permutation by scatter, then `GLM5NextMoEKernel`'s
///    ascending-expert-ID combine adding the shared experts.
///
/// Rounding points are the reference's. The tensor-op K order and the
/// dequantized-weight rounding differ from MLX's gathered matmul, so results
/// are not bitwise. No weights are copied or concatenated.
/// Whether this GPU has Neural Accelerators (MLX's NAX kernels' condition:
/// OS, architecture and build). Evaluated once per process.
let glm5NextNAXAvailable: Bool = {
    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // MLX exposes the device NAX probe only through this diagnostics record.
        return GPU.gemma4ExpertQMMDiagnostics().naxAvailable
    #else
        return false
    #endif
}()

enum GLM5NextMoEPrefillKernel {
    static let maximumExperts = 512

    static var naxAvailable: Bool { glm5NextNAXAvailable }

    /// Banks the kernels read: all affine 4-bit (groups dividing the
    /// loader's 16-value runs) or all NVFP4, widths in whole 64-value tiles.
    static func admits(_ plan: GLM5NextMoEDecodeKernel.Plan) -> Bool {
        naxAvailable && plan.experts <= maximumExperts
            && [.bfloat16, .float16].contains(plan.dtype)
            && plan.hidden % 64 == 0 && plan.intermediate % 64 == 0
            && ([plan.gate, plan.up, plan.down].allSatisfy(\.nvfp4)
                || [plan.gate, plan.up, plan.down].allSatisfy {
                    !$0.nvfp4 && $0.bits == 4 && [32, 64, 128].contains($0.groupSize)
                })
    }

    /// `x` `[T, D]`; indices and FP32 weights `[T, K]`; `addend` `[T, D]`.
    static func routedExperts(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray, addend: MLXArray,
        plan: GLM5NextMoEDecodeKernel.Plan, limit: Float
    ) throws -> MLXArray {
        let stages = try project(x, indices: indices, plan: plan, limit: limit)
        return try GLM5NextMoEKernel.ascendingCombine(
            projected: stages.projected, rowMap: stages.inverse, indices: indices,
            weights: weights, addend: addend)
    }

    /// Expert-sorted intermediates: clamped-GLU activations `[T·K, I]` and
    /// down projections `[T·K, D]` in sorted order, and the sorted position
    /// of each `token·K + slot`.
    struct Stages {
        let activations: MLXArray
        let projected: MLXArray
        let inverse: MLXArray
    }

    /// `gateUpSchedule` and `downSchedule` override the double-buffered
    /// schedule choices (diagnostics only).
    static func project(
        _ x: MLXArray, indices: MLXArray, plan: GLM5NextMoEDecodeKernel.Plan, limit: Float,
        gateUpSchedule: Bool? = nil, downSchedule: Bool? = nil
    ) throws -> Stages {
        let tokens = x.dim(0)
        let topK = indices.dim(1)
        let rows = tokens * topK
        guard x.shape == [tokens, plan.hidden], x.dtype == plan.dtype, rows >= 8,
            rows <= Int(Int32.max) - 128, indices.dtype == .uint32
        else { throw GLM5NextMoEDecodeKernel.Failure.invalidGeometry("prefill experts") }
        guard glm5NextOnDefaultGPUStream() else {
            throw GLM5NextMoEDecodeKernel.Failure.unsupportedDeviceOrStream
        }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let flattened = indices.flattened()
            let order = argSort(flattened).asType(.uint32)
            let sorted = flattened[order]
            let (gateUpTiles, gateUpCount, gateUpRows) = tiles(
                sorted, rows: rows, experts: plan.experts, input: plan.hidden)
            let fp = plan.gate.nvfp4
            let interleaved = fp || (gateUpSchedule ?? true)
            let shape = MLXArray([Int32(plan.intermediate), Int32(plan.hidden)])
            let gateUpInputs: [MLXArray] =
                interleaved
                ? [
                    x, plan.gate.weight, plan.gate.scales, plan.gate.biases,
                    plan.gate.globalScale, plan.up.weight, plan.up.scales, plan.up.biases,
                    plan.up.globalScale, gateUpTiles, gateUpCount, order, shape,
                    MLXArray([limit]),
                ]
                : [
                    x, plan.gate.weight, plan.gate.scales, plan.gate.biases, plan.up.weight,
                    plan.up.scales, plan.up.biases, gateUpTiles, gateUpCount, order, shape,
                    MLXArray([limit]),
                ]
            var gateUpTemplate: [(String, any KernelTemplateArg)] = [
                ("T", plan.dtype), ("GS", plan.gate.groupSize), ("TopK", topK),
            ]
            if interleaved { gateUpTemplate.append(("FP", fp)) }
            let gateUp =
                interleaved
                ? (gateUpRows == 128 ? gateUpDB128 : gateUpDB64) : gateUpKernel(gateUpRows)
            let activations = gateUp(
                gateUpInputs,
                template: gateUpTemplate,
                grid: (
                    (plan.intermediate / (interleaved ? 32 : 64)) * 32,
                    gateUpTiles.size / 4 * 2, gateUpRows / 32
                ),
                threadGroup: (32, 2, gateUpRows / 32),
                outputShapes: [[rows, plan.intermediate]], outputDTypes: [plan.dtype])[0]

            let (downTiles, downCount, downRows) = tiles(
                sorted, rows: rows, experts: plan.experts, input: plan.intermediate)
            let doubleBuffer = downSchedule ?? (rows <= 128 * plan.experts)
            let downShape = MLXArray([Int32(plan.hidden), Int32(plan.intermediate)])
            let downGrid = ((plan.hidden / 64) * 32, downTiles.size / 4 * 2, downRows / 32)
            let projected: MLXArray
            if fp {
                projected =
                    (downRows == 128 ? downFP128 : downFP64)(
                        [
                            activations, plan.down.weight, plan.down.scales, plan.down.globalScale,
                            downTiles, downCount, downShape,
                        ],
                        template: [("T", plan.dtype)], grid: downGrid,
                        threadGroup: (32, 2, downRows / 32),
                        outputShapes: [[rows, plan.hidden]], outputDTypes: [plan.dtype])[0]
            } else {
                projected =
                    downKernel(downRows)(
                        [
                            activations, plan.down.weight, plan.down.scales, plan.down.biases,
                            downTiles, downCount, downShape,
                        ],
                        template: [
                            ("T", plan.dtype), ("GS", plan.down.groupSize),
                            ("SCHED", doubleBuffer ? 1 : 0),
                        ],
                        grid: downGrid, threadGroup: (32, 2, downRows / 32),
                        outputShapes: [[rows, plan.hidden]], outputDTypes: [plan.dtype])[0]
            }

            let inverse = inverseKernel(
                [order], grid: (rows, 1, 1), threadGroup: (min(rows, 256), 1, 1),
                outputShapes: [[rows]], outputDTypes: [.uint32])[0]
            return Stages(activations: activations, projected: projected, inverse: inverse)
        #else
            throw GLM5NextMoEDecodeKernel.Failure.unsupportedDeviceOrStream
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// The pinned port's tile height: 128 rows once every expert averages
        /// more than 64 (and at most 128) rows over a wide input, else 64.
        private static func tileRows(rows: Int, experts: Int, input: Int) -> Int {
            rows > 64 * experts && rows <= 128 * experts && input >= 2048 ? 128 : 64
        }

        private static func tiles(_ sorted: MLXArray, rows: Int, experts: Int, input: Int)
            -> (MLXArray, MLXArray, Int)
        {
            let bm = tileRows(rows: rows, experts: experts, input: input)
            let maxTiles = (rows + bm - 1) / bm + min(experts, rows)
            let result = scanKernel(
                [sorted, MLXArray([Int32(rows), Int32(experts), Int32(maxTiles)])],
                template: [("BM", bm), ("MAXE", maximumExperts)],
                grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[maxTiles * 4], [1]], outputDTypes: [.uint32, .uint32])
            return (result[0], result[1], bm)
        }

        private static let scanKernel = MLXFast.metalKernel(
            name: "glm5_next_moe_prefill_tile_scan",
            inputNames: ["idx", "params"], outputNames: ["tiles", "tile_count"],
            source: MiMoV26NAXMetalSources.scanSource,
            header: MiMoV26NAXMetalSources.scanHeader + "\n", ensureRowContiguous: true)

        private static let inverseKernel = MLXFast.metalKernel(
            name: "glm5_next_moe_prefill_inverse",
            inputNames: ["order"], outputNames: ["inverse"],
            source: """
                const uint i = thread_position_in_grid.x;
                inverse[order[i]] = i;
                """,
            ensureRowContiguous: true)

        private static func matmulHeader(tileRows: Int) -> String {
            MiMoV26NAXMetalSources.mlxHeader + "\n"
                + MiMoV26NAXMetalSources.matmulHeader
                .replacingOccurrences(
                    of: "STEEL_CONST int kBM = 64;", with: "STEEL_CONST int kBM = \(tileRows);"
                )
                .replacingOccurrences(
                    of: "STEEL_CONST int kWM = 2;", with: "STEEL_CONST int kWM = \(tileRows / 32);")
                + "\n" + affineHeader + "\n"
        }

        private static let gateUp64 = makeGateUp(tileRows: 64)
        private static let gateUp128 = makeGateUp(tileRows: 128)
        private static let down64 = makeDown(tileRows: 64)
        private static let down128 = makeDown(tileRows: 128)

        private static let gateUpDB64 = makeGateUpDB(tileRows: 64)
        private static let gateUpDB128 = makeGateUpDB(tileRows: 128)

        private static func gateUpKernel(_ rows: Int) -> MLXFast.MLXFastKernel {
            rows == 128 ? gateUp128 : gateUp64
        }

        private static func makeGateUpDB(tileRows: Int) -> MLXFast.MLXFastKernel {
            MLXFast.metalKernel(
                name: "glm5_next_moe_prefill_gate_up_db_bm\(tileRows)",
                inputNames: [
                    "x", "gate_w", "gate_s", "gate_b", "gate_g", "up_w", "up_s", "up_b",
                    "up_g", "tiles", "tile_count", "order", "params", "limit",
                ],
                outputNames: ["act"],
                source: """
                    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
                    threadgroup T Ws[2 * omlx_gqmm::kBN * BKP];
                    if (threadgroup_position_in_grid.y >= tile_count[0]) {
                        return;
                    }
                    if constexpr (FP) {
                        using Q = omlx_gqmm::GlmNvfp4Q<T>;
                        const Q gate_q{(const device uint8_t*)gate_s};
                        const Q up_q{(const device uint8_t*)up_s};
                        omlx_gqmm::glm_gather_gate_up_db<T, Q, TopK, true>(
                            x, (const device uint8_t*)gate_w, (const device uint8_t*)up_w,
                            gate_q, up_q, gate_g, up_g, tiles, order, act, params[0],
                            params[1], limit[0], Ws, threadgroup_position_in_grid,
                            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                    } else {
                        using Q = omlx_gqmm::GlmAffineQ<T, GS>;
                        const Q gate_q{(const device T*)gate_s, (const device T*)gate_b, 0.0f, 0.0f};
                        const Q up_q{(const device T*)up_s, (const device T*)up_b, 0.0f, 0.0f};
                        omlx_gqmm::glm_gather_gate_up_db<T, Q, TopK, false>(
                            x, (const device uint8_t*)gate_w, (const device uint8_t*)up_w,
                            gate_q, up_q, gate_g, up_g, tiles, order, act, params[0],
                            params[1], limit[0], Ws, threadgroup_position_in_grid,
                            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                    }
                    """,
                header: matmulHeader(tileRows: tileRows), ensureRowContiguous: true)
        }

        private static func downKernel(_ rows: Int) -> MLXFast.MLXFastKernel {
            rows == 128 ? down128 : down64
        }

        private static func makeGateUp(tileRows: Int) -> MLXFast.MLXFastKernel {
            MLXFast.metalKernel(
                name: "glm5_next_moe_prefill_gate_up_bm\(tileRows)",
                inputNames: [
                    "x", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b", "tiles",
                    "tile_count", "order", "params", "limit",
                ],
                outputNames: ["act"],
                source: """
                    if (threadgroup_position_in_grid.y >= tile_count[0]) {
                        return;
                    }
                    using Q = omlx_gqmm::GlmAffineQ<T, GS>;
                    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
                    threadgroup T gate_ws[omlx_gqmm::kBN * BKP];
                    threadgroup T up_ws[omlx_gqmm::kBN * BKP];
                    Q gate_q{gate_s, gate_b, 0.0f, 0.0f};
                    Q up_q{up_s, up_b, 0.0f, 0.0f};
                    omlx_gqmm::glm_gather_gate_up_glu<T, Q, TopK>(
                        x, (const device uint8_t*)gate_w, (const device uint8_t*)up_w,
                        gate_q, up_q, tiles, order, act, params[0], params[1], limit[0],
                        gate_ws, up_ws, threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                    """,
                header: matmulHeader(tileRows: tileRows), ensureRowContiguous: true)
        }

        private static let downFP64 = makeDownFP(tileRows: 64)
        private static let downFP128 = makeDownFP(tileRows: 128)

        private static func makeDownFP(tileRows: Int) -> MLXFast.MLXFastKernel {
            MLXFast.metalKernel(
                name: "glm5_next_moe_prefill_down_fp_bm\(tileRows)",
                inputNames: ["x", "w", "scales", "g", "tiles", "tile_count", "params"],
                outputNames: ["y"],
                source: """
                    using Q = omlx_gqmm::GlmNvfp4Q<T>;
                    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
                    threadgroup T Ws[2 * omlx_gqmm::kBN * BKP];
                    if (threadgroup_position_in_grid.y >= tile_count[0]) {
                        return;
                    }
                    const Q q{scales};
                    omlx_gqmm::glm_gather_down_db<T, Q, true>(
                        x, (const device uint8_t*)w, q, g, tiles, y, params[0], params[1], Ws,
                        threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
                    """,
                header: matmulHeader(tileRows: tileRows), ensureRowContiguous: true)
        }

        private static func makeDown(tileRows: Int) -> MLXFast.MLXFastKernel {
            MLXFast.metalKernel(
                name: "glm5_next_moe_prefill_down_bm\(tileRows)",
                inputNames: ["x", "w", "scales", "biases", "tiles", "tile_count", "params"],
                outputNames: ["y"],
                source: """
                    using Q = omlx_gqmm::GlmAffineQ<T, GS>;
                    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
                    threadgroup T Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP];
                    if (threadgroup_position_in_grid.y >= tile_count[0]) {
                        return;
                    }
                    Q q{scales, biases, 0.0f, 0.0f};
                    if constexpr (SCHED == 1) {
                        omlx_gqmm::gather_db<T, Q>(
                            x, (const device uint8_t*)w, q, tiles, y, params[0], params[1],
                            Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
                            thread_index_in_simdgroup);
                    } else {
                        omlx_gqmm::gather_seg<T, Q, true, true>(
                            x, (const device uint8_t*)w, q, tiles, y, params[0], params[1],
                            Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
                            thread_index_in_simdgroup);
                    }
                    """,
                header: matmulHeader(tileRows: tileRows), ensureRowContiguous: true)
        }

        private static let affineHeader = #"""
            namespace omlx_gqmm {

            // Affine group dequantization (scale * q + offset), rounded once to
            // the activation dtype the tensor ops read.
            template <typename T, int GS>
            struct GlmAffineQ {
              using WT = T;
              STEEL_CONST int kBits = 4;
              STEEL_CONST int kGroup = GS;
              const device T* scales;
              const device T* biases;
              float s;
              float b;

              METAL_FUNC void advance(const size_t n) thread {
                scales += n;
                biases += n;
              }
              METAL_FUNC void load_params(const int g) thread {
                s = float(scales[g]);
                b = float(biases[g]);
              }
              METAL_FUNC WT dq(const uint32_t q) const thread {
                return static_cast<WT>(s * float(q) + b);
              }
              METAL_FUNC void params(const int g, thread float& s_, thread float& b_) const thread {
                s_ = float(scales[g]);
                b_ = float(biases[g]);
              }
              static METAL_FUNC WT dqp(const uint32_t q, const float s_, const float b_) {
                return static_cast<WT>(s_ * float(q) + b_);
              }
            };

            // MLX fp4.h / fp8.h conversions (exact through half).
            inline float glm5p_e2m1(uint8_t bits) {
              half converted = as_type<half>(ushort((bits & 7) << 9));
              converted *= 16384.0;
              return static_cast<float>(bits & 8 ? -converted : converted);
            }
            inline float glm5p_e4m3(uint8_t bits) {
              uint16_t v = (bits & 127) << 7;
              half converted = as_type<half>(v);
              converted *= 256.0;
              return static_cast<float>((bits & 128) ? -converted : converted);
            }

            // NVFP4: E2M1 codes times the E4M3 group-16 scale, rounded once to the
            // activation dtype like MLX's fp QuantizedBlockLoader; the FP32 global
            // scale is applied to the rounded output in the epilogue.
            template <typename T>
            struct GlmNvfp4Q {
              using WT = T;
              STEEL_CONST int kBits = 4;
              STEEL_CONST int kGroup = 16;
              const device uint8_t* scales;

              METAL_FUNC void advance(const size_t n) thread {
                scales += n;
              }
              METAL_FUNC void params(const int g, thread float& s_, thread float& b_) const thread {
                s_ = glm5p_e4m3(scales[g]);
                b_ = 0.0f;
              }
              static METAL_FUNC WT dqp(const uint32_t q, const float s_, const float) {
                return static_cast<WT>(s_ * glm5p_e2m1(uint8_t(q)));
              }
            };

            // The projection's activation-dtype output; NVFP4 then applies its
            // FP32 global scale and rounds again (the composed module's order).
            template <typename T, bool FP>
            inline T glm5p_out(const float acc, const float global_scale) {
              const T y = static_cast<T>(acc);
              if constexpr (FP) {
                return static_cast<T>(static_cast<float>(y) * global_scale);
              }
              return y;
            }

            // MLX's Sigmoid / Minimum / Maximum functors (unary_ops.h,
            // binary_ops.h), as glm5NextClampedGLUOperations' launches apply them.
            template <typename T>
            inline T glm5p_sigmoid(T x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }
            template <typename T>
            inline T glm5p_minimum(T x, T y) {
              if (metal::isnan(x)) {
                return x;
              }
              return x < y ? x : y;
            }
            template <typename T>
            inline T glm5p_maximum(T x, T y) {
              if (metal::isnan(x)) {
                return x;
              }
              return x > y ? x : y;
            }

            // Joint gate/up over single-expert sorted tiles: one activation
            // fragment read (through the sort order) feeds both independent
            // FP32 accumulators; each is rounded to T, then the clamped GLU
            // runs in T. Only the activation [rows, N] is written.
            template <typename T, typename Q, int TOPK>
            METAL_FUNC void glm_gather_gate_up_glu(
                const device T* x,
                const device uint8_t* gate_w,
                const device uint8_t* up_w,
                thread Q& gate_q,
                thread Q& up_q,
                const device uint32_t* tiles,
                const device uint32_t* order,
                device T* act,
                const int N,
                const int K,
                const float limit,
                threadgroup typename Q::WT* gate_ws,
                threadgroup typename Q::WT* up_ws,
                const uint3 tid,
                const uint sgid,
                const uint lane) {
              using WT = typename Q::WT;
              constexpr int BKP = kBK + 16 / sizeof(WT);
              const uint4 desc = *((const device uint4*)tiles + tid.y);
              const int row_start = int(desc.x);
              const uint32_t expert = desc.y;
              const int rows = int(desc.z);
              const int K_w = K * Q::kBits / 8;
              const int K_g = K / Q::kGroup;
              const int y_col = int(tid.x) * kBN;
              const size_t w_row = size_t(expert) * N + y_col;

              gate_q.advance(w_row * K_g);
              up_q.advance(w_row * K_g);
              TileLoader<Q> gate_loader(gate_w + w_row * K_w, K, gate_q, sgid * 32 + lane);
              TileLoader<Q> up_loader(up_w + w_row * K_w, K, up_q, sgid * 32 + lane);

              act += size_t(row_start) * N + y_col;
              const short tm = kSM * short(sgid / kWN);
              const short tn = kSN * short(sgid % kWN);
              const short live_rows = short(min(int(kSM), max(0, rows - int(tm))));
              size_t mapped_offsets[kTM][2];
              const short2 sc = BaseNAXFrag::get_coord();
              STEEL_PRAGMA_UNROLL
              for (short mm = 0; mm < kTM; ++mm) {
                STEEL_PRAGMA_UNROLL
                for (short h = 0; h < 2; ++h) {
                  const int local_row = min(rows - 1, int(tm) + mm * 16 + h * 8 + sc.y);
                  mapped_offsets[mm][h] =
                      size_t(order[row_start + local_row] / uint32_t(TOPK)) * K + sc.x;
                }
              }
              const device T* xn = x;

              NAXTile<float, kTM, kTN> gate_d;
              NAXTile<float, kTM, kTN> up_d;
              gate_d.clear();
              up_d.clear();

              for (int kb = 0; kb < K / kBK; ++kb) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                gate_loader.fetch(kb);
                up_loader.fetch(kb);
                gate_loader.store(gate_ws);
                up_loader.store(up_ws);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                STEEL_PRAGMA_NO_UNROLL
                for (short kk1 = 0; kk1 < kBK; kk1 += kSK) {
                  if (live_rows > 0) {
                    NAXTile<WT, kTN, kTK> gate_b;
                    NAXTile<WT, kTN, kTK> up_b;
                    gate_b.template load<WT, BKP, 1>(gate_ws + tn * BKP + kk1);
                    up_b.template load<WT, BKP, 1>(up_ws + tn * BKP + kk1);
                    STEEL_PRAGMA_UNROLL
                    for (short mm = 0; mm < kTM; ++mm) {
                      if (mm * 16 < live_rows) {
                        NAXTile<T, 1, kTK> a;
                        STEEL_PRAGMA_UNROLL
                        for (short h = 0; h < 2; ++h) {
                          STEEL_PRAGMA_UNROLL
                          for (short kk = 0; kk < kTK; ++kk) {
                            const vec<T, 4> v = *(const device vec<T, 4>*)(
                                xn + mapped_offsets[mm][h] + kk1 + kk * 16);
                            STEEL_PRAGMA_UNROLL
                            for (short c = 0; c < 4; ++c) {
                              a.frag_at(0, kk)[h * 4 + c] =
                                  mm * 16 + h * 8 + sc.y < live_rows ? v[c] : T(0);
                            }
                          }
                        }
                        STEEL_PRAGMA_UNROLL
                        for (short nn = 0; nn < kTN; nn += 2) {
                          STEEL_PRAGMA_UNROLL
                          for (short kk = 0; kk < kTK; ++kk) {
                            BaseNAXFrag::mma(
                                gate_d.frag_at(mm, nn), gate_d.frag_at(mm, nn + 1),
                                a.frag_at(0, kk), metal::bool_constant<false>{},
                                gate_b.frag_at(nn, kk), gate_b.frag_at(nn + 1, kk),
                                metal::bool_constant<true>{});
                            BaseNAXFrag::mma(
                                up_d.frag_at(mm, nn), up_d.frag_at(mm, nn + 1),
                                a.frag_at(0, kk), metal::bool_constant<false>{},
                                up_b.frag_at(nn, kk), up_b.frag_at(nn + 1, kk),
                                metal::bool_constant<true>{});
                          }
                        }
                      }
                    }
                  }
                }
                xn += kBK;
              }

              const T lim = static_cast<T>(limit);
              const T neg_lim = static_cast<T>(-limit);
              NAXTile<T, kTM, kTN> activated;
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < decltype(activated)::kNumFrags; ++i) {
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < decltype(activated)::kElemsPerFrag; ++e) {
                  const T gate = static_cast<T>(gate_d.val_frags[i][e]);
                  const T up = static_cast<T>(up_d.val_frags[i][e]);
                  const T g = glm5p_minimum(gate, lim);
                  const T s = g * glm5p_sigmoid(g);
                  const T u = glm5p_minimum(glm5p_maximum(up, neg_lim), lim);
                  activated.val_frags[i][e] = s * u;
                }
              }
              if (live_rows == kSM) {
                activated.store(act + tm * N + tn, N);
              } else if (live_rows > 0) {
                activated.store_safe(act + tm * N + tn, N, short2(kSN, live_rows));
              }
            }

            // TileLoader with an explicit per-thread weight row, so one tile can
            // interleave rows of two matrices.
            template <typename Q>
            struct GlmRowLoader {
              using WT = typename Q::WT;
              STEEL_CONST int kBits = Q::kBits;
              STEEL_CONST int kWords = kVPT * kBits / 32;
              STEEL_CONST int kPer = 32 / kBits;
              STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
              static_assert(Q::kGroup % kPer == 0, "whole words per group");

              const device uint32_t* src;
              Q q;
              const short row;
              const short col;
              uint32_t raw[kWords];
              float sc[kWords];
              float bi[kWords];

              METAL_FUNC GlmRowLoader(
                  const device uint8_t* w_row, thread const Q& q_, const uint lid) thread
                  : q(q_), row(short(lid / kTPR)), col(short((lid % kTPR) * kVPT)) {
                src = (const device uint32_t*)(w_row + col * kBits / 8);
              }

              METAL_FUNC void fetch(const int kb) thread {
                const device uint32_t* p = src + kb * (kBK * kBits / 32);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < kWords; i++) {
                  raw[i] = p[i];
                  q.params((kb * kBK + col + i * kPer) / Q::kGroup, sc[i], bi[i]);
                }
              }

              METAL_FUNC void store(threadgroup WT* Ws) const thread {
                threadgroup WT* dst = Ws + row * (kBK + 16 / sizeof(WT)) + col;
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < kWords; i++) {
                  vec<WT, kPer> v;
                  STEEL_PRAGMA_UNROLL
                  for (short j = 0; j < kPer; j++) {
                    v[j] = Q::dqp((raw[i] >> (kBits * j)) & kMask, sc[i], bi[i]);
                  }
                  *(threadgroup vec<WT, kPer>*)(dst + i * kPer) = v;
                }
              }
            };

            // #4022's interleaved schedule on the double-buffered tiles of
            // gather_db: weight-tile rows alternate 16 gate and 16 up rows of
            // the same 16 outputs, so each simdgroup's two 16-column fragments
            // hold gate and up of identical elements and the clamped GLU needs
            // no exchange. One barrier per K step; activation fragments are
            // read from device memory through the sort order. Requires K and N
            // in whole 64- and 32-value tiles.
            template <typename T, typename Q, int TOPK, bool FP, typename G>
            METAL_FUNC void glm_gather_gate_up_db(
                const device T* x,
                const device uint8_t* gate_w,
                const device uint8_t* up_w,
                thread const Q& gate_q,
                thread const Q& up_q,
                G gate_g,
                G up_g,
                const device uint32_t* tiles,
                const device uint32_t* order,
                device T* act,
                const int N,
                const int K,
                const float limit,
                threadgroup typename Q::WT* Ws,
                const uint3 tid,
                const uint sgid,
                const uint lane) {
              using WT = typename Q::WT;
              constexpr int BKP = kBK + 16 / sizeof(WT);
              constexpr int kTile = kBN * BKP;
              static_assert(kTN == 2, "one gate and one up fragment per simdgroup");
              const uint4 desc = *((const device uint4*)tiles + tid.y);
              const int row_start = int(desc.x);
              const uint32_t expert = desc.y;
              const int tile_rows = int(desc.z);

              const int K_w = K * Q::kBits / 8;
              const int K_g = K / Q::kGroup;
              const int K_it = K / kBK;
              const int y_col = int(tid.x) * (kBN / 2);

              const uint lid = sgid * 32 + lane;
              const int lrow = int(lid) / kTPR;
              const int blk = lrow / 16;
              const bool is_up = (blk & 1) != 0;
              const size_t w_row = size_t(expert) * N + y_col + (blk >> 1) * 16 + (lrow & 15);
              Q q = is_up ? up_q : gate_q;
              q.advance(w_row * K_g);
              GlmRowLoader<Q> loader((is_up ? up_w : gate_w) + w_row * K_w, q, lid);

              const int m0 = kSM * int(sgid / kWN);
              const int rows = min(int(kSM), tile_rows - m0);
              const short2 sc = BaseNAXFrag::get_coord();
              size_t x_off[kTM][2];
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < kTM; i++) {
                STEEL_PRAGMA_UNROLL
                for (short h = 0; h < 2; h++) {
                  const int r = min(m0 + int(i * 16 + sc.y + h * 8), tile_rows - 1);
                  x_off[i][h] = size_t(order[row_start + r] / uint32_t(TOPK)) * K + sc.x;
                }
              }
              const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
              const threadgroup WT* wsg = Ws + (sgid % kWN) * kSN * BKP;

              NAXTile<float, kTM, kTN> D;
              D.clear();

              loader.fetch(0);
              loader.store(Ws);
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int kb = 0; kb < K_it; kb++) {
                const bool more = kb + 1 < K_it;
                if (more) {
                  loader.fetch(kb + 1);
                }
                const threadgroup WT* wb = wsg + (kb & 1) * kTile;
                STEEL_PRAGMA_UNROLL
                for (short kk1 = 0; kk1 < kBK; kk1 += kSK) {
                  NAXTile<WT, kTN, 2> Btile;
                  Btile.template load<WT, BKP, 1>(wb + kk1);
                  const int k = kb * kBK + kk1;
                  STEEL_PRAGMA_UNROLL
                  for (short i = 0; i < kTM; i++) {
                    if (i < m_frags) {
                      NAXTile<T, 1, 2> Atile;
                      STEEL_PRAGMA_UNROLL
                      for (short h = 0; h < 2; h++) {
                        const device T* xp = x + x_off[i][h] + k;
                        const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
                        const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
                        STEEL_PRAGMA_UNROLL
                        for (short c = 0; c < 4; c++) {
                          Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
                          Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
                        }
                      }
                      STEEL_PRAGMA_UNROLL
                      for (short kk = 0; kk < 2; kk++) {
                        BaseNAXFrag::mma(
                            D.frag_at(i, 0),
                            D.frag_at(i, 1),
                            Atile.frag_at(0, kk),
                            metal::bool_constant<false>{},
                            Btile.frag_at(0, kk),
                            Btile.frag_at(1, kk),
                            metal::bool_constant<true>{});
                      }
                    }
                  }
                }
                if (more) {
                  loader.store(Ws + ((kb + 1) & 1) * kTile);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
              }

              const T lim = static_cast<T>(limit);
              const T neg_lim = static_cast<T>(-limit);
              const float gate_scale = FP ? gate_g[expert] : 1.0f;
              const float up_scale = FP ? up_g[expert] : 1.0f;
              NAXTile<T, kTM, 1> activated;
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < kTM; i++) {
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < decltype(activated)::kElemsPerFrag; e++) {
                  const T gate = glm5p_out<T, FP>(D.frag_at(i, 0)[e], gate_scale);
                  const T up = glm5p_out<T, FP>(D.frag_at(i, 1)[e], up_scale);
                  const T g = glm5p_minimum(gate, lim);
                  const T s = g * glm5p_sigmoid(g);
                  const T u = glm5p_minimum(glm5p_maximum(up, neg_lim), lim);
                  activated.frag_at(i, 0)[e] = s * u;
                }
              }
              device T* yb = act + size_t(row_start + m0) * N + y_col + 16 * (sgid % kWN);
              if (rows >= kSM) {
                activated.store(yb, N);
              } else if (rows > 0) {
                activated.store_safe(yb, N, short2(16, short(rows)));
              }
            }

            // gather_db with GlmRowLoader (any group size, NVFP4 included) and the
            // global-scale epilogue: down projection of the sorted activation rows.
            template <typename T, typename Q, bool FP, typename G>
            METAL_FUNC void glm_gather_down_db(
                const device T* x,
                const device uint8_t* w,
                thread const Q& q0,
                G global_g,
                const device uint32_t* tiles,
                device T* y,
                const int N,
                const int K,
                threadgroup typename Q::WT* Ws,
                const uint3 tid,
                const uint sgid,
                const uint lane) {
              using WT = typename Q::WT;
              constexpr int BKP = kBK + 16 / sizeof(WT);
              constexpr int kTile = kBN * BKP;
              const uint4 desc = *((const device uint4*)tiles + tid.y);
              const int row_start = int(desc.x);
              const uint32_t expert = desc.y;
              const int tile_rows = int(desc.z);

              const int K_w = K * Q::kBits / 8;
              const int K_g = K / Q::kGroup;
              const int K_it = K / kBK;
              const int y_col = int(tid.x) * kBN;
              const uint lid = sgid * 32 + lane;
              const size_t w_row = size_t(expert) * N + y_col + lid / kTPR;
              Q q = q0;
              q.advance(w_row * K_g);
              GlmRowLoader<Q> loader(w + w_row * K_w, q, lid);

              const int m0 = kSM * int(sgid / kWN);
              const int rows = min(int(kSM), tile_rows - m0);
              const device T* xs =
                  x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;
              const short2 sc = BaseNAXFrag::get_coord();
              int x_off[kTM][2];
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < kTM; i++) {
                STEEL_PRAGMA_UNROLL
                for (short h = 0; h < 2; h++) {
                  const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
                  x_off[i][h] = r * K + sc.x;
                }
              }
              const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
              const threadgroup WT* wsg = Ws + (sgid % kWN) * kSN * BKP;

              NAXTile<float, kTM, kTN> D;
              D.clear();

              loader.fetch(0);
              loader.store(Ws);
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int kb = 0; kb < K_it; kb++) {
                const bool more = kb + 1 < K_it;
                if (more) {
                  loader.fetch(kb + 1);
                }
                const threadgroup WT* wb = wsg + (kb & 1) * kTile;
                STEEL_PRAGMA_UNROLL
                for (short kk1 = 0; kk1 < kBK; kk1 += kSK) {
                  NAXTile<WT, kTN, 2> Btile;
                  Btile.template load<WT, BKP, 1>(wb + kk1);
                  const int k = kb * kBK + kk1;
                  STEEL_PRAGMA_UNROLL
                  for (short i = 0; i < kTM; i++) {
                    if (i < m_frags) {
                      NAXTile<T, 1, 2> Atile;
                      STEEL_PRAGMA_UNROLL
                      for (short h = 0; h < 2; h++) {
                        const device T* xp = xs + x_off[i][h] + k;
                        const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
                        const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
                        STEEL_PRAGMA_UNROLL
                        for (short c = 0; c < 4; c++) {
                          Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
                          Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
                        }
                      }
                      STEEL_PRAGMA_UNROLL
                      for (short kk = 0; kk < 2; kk++) {
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < kTN; j += 2) {
                          BaseNAXFrag::mma(
                              D.frag_at(i, j),
                              D.frag_at(i, j + 1),
                              Atile.frag_at(0, kk),
                              metal::bool_constant<false>{},
                              Btile.frag_at(j, kk),
                              Btile.frag_at(j + 1, kk),
                              metal::bool_constant<true>{});
                        }
                      }
                    }
                  }
                }
                if (more) {
                  loader.store(Ws + ((kb + 1) & 1) * kTile);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
              }

              const float global_scale = FP ? global_g[expert] : 1.0f;
              NAXTile<T, kTM, kTN> out;
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < kTM; i++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kTN; j++) {
                  STEEL_PRAGMA_UNROLL
                  for (short e = 0; e < decltype(out)::kElemsPerFrag; e++) {
                    out.frag_at(i, j)[e] = glm5p_out<T, FP>(D.frag_at(i, j)[e], global_scale);
                  }
                }
              }
              device T* yb = y + size_t(row_start + m0) * N + y_col + kSN * (sgid % kWN);
              if (rows >= kSM) {
                out.store(yb, N);
              } else if (rows > 0) {
                out.store_safe(yb, N, short2(kSN, short(rows)));
              }
            }

            } // namespace omlx_gqmm
            """#
    #endif
}
