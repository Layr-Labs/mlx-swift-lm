// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Eigen Labs.
// Adapted from oMLX (Apache-2.0, Copyright oMLX contributors; notice:
// docs/mimo-v26/LICENSE-OMLX): #3995 47876fbc310fbb311cd382c4eba8572ec4368308
// (segmented single-expert NAX tiles), #3993 a68cab68ae40eeab7062ba7c1eff4ddffa06e373
// (joint gate/up), #4022 d6dbc9970c6df99b6a8589746e6ff8ddc38c33ff (SwiGLU
// epilogue), #4029 498685bae7b536cbd0d7d5e145b0327544e2fb06 (row-map reads),
// 97e473efe6836d6b44a01a5e0e63358ba0611db8 (tile plan: 64/96/128-row tiles,
// 128-deep K steps, tile-on-x layout, threadgroup pad; merged in #3995
// 8bb1fb9ca00c1b07f91283a7361a1a00ed8f46a2), through the pinned MiMo port in
// `MiMoV26NAXMetalSources` (Apple MLX MIT primitives, notice retained there).
// Modified: affine 4-bit group dequant with activation-dtype scales and
// offsets; NVFP4; the clamped-GLU epilogue; token rows read through the sort
// order (`order / K`); inverse by scatter.

import Foundation
import MLX
import MLXLMCommon

/// Routed experts of a long step for
/// `GLM5NextExecutionPolicy.RoutedExperts.fusedDecodeNAXPrefill`:
///
/// 1. Sort the assignments by expert (MLX argsort), cut the sorted run into
///    single-expert tiles of at most 64, 96 or 128 rows (one scan launch;
///    `TilePlan.plan` picks the height per projection from the mean rows per
///    expert and the projection's input width).
/// 2. Gate and up on the Neural Accelerators in one launch: both weight tiles
///    are dequantized to the activation dtype, each token row is read once
///    through the sort order (no replicated `[T·K, D]` copy), the FP32
///    accumulators are rounded to the activation dtype and the clamped GLU is
///    applied in that dtype in the epilogue; only `[T·K, I]` is written.
/// 3. Down on tiles of its own plan (the gate/up tile list when the heights
///    match), rounded to the activation dtype.
/// 4. The inverse permutation by scatter, then `GLM5NextMoEKernel`'s
///    ascending-expert-ID combine adding the shared experts.
///
/// Rounding points are the reference's. The tensor-op K order and the
/// dequantized-weight rounding differ from MLX's gathered matmul, so results
/// are not bitwise. Every `TilePlan` issues the same tensor ops per output
/// element in the same K order, so plans are bitwise identical to each other.
/// No weights are copied or concatenated.
enum GLM5NextMoEPrefillKernel {
    static let maximumExperts = 512

    static var naxAvailable: Bool { glm5NextNAXAvailable }

    /// Rows of one single-expert tile and the depth of one K step. Both are
    /// compile-time constants of the matmul header, so each shape is its own
    /// kernel. `loaderThreads` dequantize the 64 x `depth` weight tile: every
    /// thread when they split it evenly, else 128 (96-row tiles: 128 of 192).
    struct TileShape: Hashable, Sendable, CustomStringConvertible {
        let rows: Int
        let depth: Int

        static let all: [TileShape] = [
            .init(rows: 64, depth: 64), .init(rows: 96, depth: 64), .init(rows: 128, depth: 64),
            .init(rows: 96, depth: 128), .init(rows: 128, depth: 128),
        ]

        var loaderThreads: Int { rows == 96 ? 128 : 2 * rows }
        var description: String { "\(rows)x64 bk\(depth)" }
    }

    /// One launch configuration (oMLX 97e473ef's `Plan`). `doubleBuffered`:
    /// 64-deep weight tiles in two buffers, one barrier per K step; otherwise
    /// (seg) each `depth`-deep step is dequantized into one buffer between two
    /// barriers. `groupX` > 0 is the tile-on-x layout: tile t on grid x
    /// `t % groupX`, its column blocks on y `groupX` threadgroups apart, so
    /// every threadgroup of a row tile shares one x coordinate (0: MLX's
    /// (column, tile) grid). `pad` bytes of unused threadgroup memory lower
    /// residency.
    struct TilePlan: Hashable, Sendable, CustomStringConvertible {
        let shape: TileShape
        let doubleBuffered: Bool
        let groupX: Int
        let pad: Int

        init(rows: Int, depth: Int, doubleBuffered: Bool, groupX: Int, pad: Int) {
            shape = TileShape(rows: rows, depth: depth)
            self.doubleBuffered = doubleBuffered
            self.groupX = groupX
            self.pad = pad
        }

        var description: String {
            "\(doubleBuffered ? "db" : "seg") \(shape)"
                + (groupX > 0 ? " gx\(groupX)" : "") + (pad > 0 ? " pad\(pad)" : "")
        }

        /// Whether this plan's kernels read a projection of `input` columns.
        func admits(input: Int) -> Bool {
            TileShape.all.contains(shape) && input % shape.depth == 0
                && (!doubleBuffered || shape.depth == 64) && groupX >= 0
                && (0 ... 8192).contains(pad) && pad % 16 == 0
        }

        /// oMLX 97e473ef's `_plan` (measured on M5 Ultra with affine and
        /// MXFP4 banks at Qwen3.8, GLM-5.3 and MiMo-V2.6 expert shapes,
        /// 1K-8K-token chunks): weight streaming dominates below 36 rows per
        /// expert and for short inputs below 120; from 36 rows activation
        /// reuse across a tile's columns does (tile-on-x); 48-95 rows fill
        /// 96-row tiles; from 96 rows 128-row tiles stream each expert's
        /// weights once, with 128-deep K steps. A 128-deep plan needs
        /// `input % 128 == 0` and otherwise keeps 64-deep double buffering.
        static func plan(rows: Int, experts: Int, input: Int) -> TilePlan {
            let perExpert = Double(rows) / Double(max(1, experts))
            let deep = input % 128 == 0
            if perExpert < 36 || (input < 1024 && perExpert < 120) {
                return TilePlan(rows: 64, depth: 64, doubleBuffered: true, groupX: 0, pad: 0)
            }
            if input < 1024 {
                return deep
                    ? TilePlan(rows: 96, depth: 128, doubleBuffered: false, groupX: 32, pad: 0)
                    : TilePlan(rows: 96, depth: 64, doubleBuffered: true, groupX: 32, pad: 0)
            }
            if perExpert < 48 {
                return TilePlan(rows: 64, depth: 64, doubleBuffered: true, groupX: 32, pad: 0)
            }
            if perExpert < 96 {
                return TilePlan(rows: 96, depth: 64, doubleBuffered: true, groupX: 32, pad: 0)
            }
            return deep
                ? TilePlan(rows: 128, depth: 128, doubleBuffered: false, groupX: 32, pad: 8192)
                : TilePlan(rows: 128, depth: 64, doubleBuffered: true, groupX: 32, pad: 0)
        }
    }

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

    /// `gateUpSchedule: false` selects the separate-tile affine gate/up kernel
    /// on the legacy 64/128-row tiles; `gateUpPlan` and `downPlan` pin the
    /// interleaved gate/up and the down launch configurations (diagnostics
    /// only; every plan is bitwise identical).
    static func project(
        _ x: MLXArray, indices: MLXArray, plan: GLM5NextMoEDecodeKernel.Plan, limit: Float,
        gateUpSchedule: Bool? = nil, gateUpPlan: TilePlan? = nil, downPlan: TilePlan? = nil
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
        let fp = plan.gate.nvfp4
        let interleaved = fp || (gateUpSchedule ?? true)
        let gateUpPlan =
            gateUpPlan ?? TilePlan.plan(rows: rows, experts: plan.experts, input: plan.hidden)
        let downPlan =
            downPlan ?? TilePlan.plan(rows: rows, experts: plan.experts, input: plan.intermediate)
        guard gateUpPlan.admits(input: plan.hidden), downPlan.admits(input: plan.intermediate)
        else { throw GLM5NextMoEDecodeKernel.Failure.invalidGeometry("prefill tile plan") }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let flattened = indices.flattened()
            let order = argSort(flattened).asType(.uint32)
            let sorted = flattened[order]
            let shape = MLXArray([Int32(plan.intermediate), Int32(plan.hidden)])
            let limits = MLXArray([limit])
            let activations: MLXArray
            let gateUpTiles: Tiles
            if interleaved {
                gateUpTiles = tiles(
                    sorted, rows: rows, experts: plan.experts, tileRows: gateUpPlan.shape.rows)
                // Each threadgroup computes 32 outputs from a 64-row tile of
                // 16-row gate/up blocks.
                let columns = plan.intermediate / 32
                activations = gateUpKernels[gateUpPlan.shape]!(
                    [
                        x, plan.gate.weight, plan.gate.scales, plan.gate.biases,
                        plan.gate.globalScale, plan.up.weight, plan.up.scales, plan.up.biases,
                        plan.up.globalScale, gateUpTiles.list, gateUpTiles.count, order, shape,
                        limits,
                    ],
                    template: [
                        ("T", plan.dtype), ("GS", plan.gate.groupSize), ("TopK", topK),
                        ("FP", fp), ("DB", gateUpPlan.doubleBuffered),
                        ("GX", gateUpPlan.groupX), ("NC", columns), ("PAD", gateUpPlan.pad),
                    ],
                    grid: grid(gateUpPlan, columns: columns, maximumTiles: gateUpTiles.maximum),
                    threadGroup: (32, 2, gateUpPlan.shape.rows / 32),
                    outputShapes: [[rows, plan.intermediate]], outputDTypes: [plan.dtype])[0]
            } else {
                let tileRows = legacyTileRows(rows: rows, experts: plan.experts, input: plan.hidden)
                gateUpTiles = tiles(sorted, rows: rows, experts: plan.experts, tileRows: tileRows)
                activations = gateUpKernel(tileRows)(
                    [
                        x, plan.gate.weight, plan.gate.scales, plan.gate.biases, plan.up.weight,
                        plan.up.scales, plan.up.biases, gateUpTiles.list, gateUpTiles.count,
                        order, shape, limits,
                    ],
                    template: [
                        ("T", plan.dtype), ("GS", plan.gate.groupSize), ("TopK", topK),
                    ],
                    grid: ((plan.intermediate / 64) * 32, gateUpTiles.maximum * 2, tileRows / 32),
                    threadGroup: (32, 2, tileRows / 32),
                    outputShapes: [[rows, plan.intermediate]], outputDTypes: [plan.dtype])[0]
            }

            // The tile list depends only on the tile height.
            let downTiles =
                gateUpTiles.rows == downPlan.shape.rows
                ? gateUpTiles
                : tiles(sorted, rows: rows, experts: plan.experts, tileRows: downPlan.shape.rows)
            let downColumns = plan.hidden / 64
            let projected = downKernels[downPlan.shape]!(
                [
                    activations, plan.down.weight, plan.down.scales, plan.down.biases,
                    plan.down.globalScale, downTiles.list, downTiles.count,
                    MLXArray([Int32(plan.hidden), Int32(plan.intermediate)]),
                ],
                template: [
                    ("T", plan.dtype), ("GS", plan.down.groupSize), ("FP", fp),
                    ("DB", downPlan.doubleBuffered), ("GX", downPlan.groupX),
                    ("NC", downColumns), ("PAD", downPlan.pad),
                ],
                grid: grid(downPlan, columns: downColumns, maximumTiles: downTiles.maximum),
                threadGroup: (32, 2, downPlan.shape.rows / 32),
                outputShapes: [[rows, plan.hidden]], outputDTypes: [plan.dtype])[0]

            let inverse = inverseKernel(
                [order], grid: (rows, 1, 1), threadGroup: (min(rows, 256), 1, 1),
                outputShapes: [[rows]], outputDTypes: [.uint32])[0]
            return Stages(activations: activations, projected: projected, inverse: inverse)
        #else
            throw GLM5NextMoEDecodeKernel.Failure.unsupportedDeviceOrStream
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// The separate-tile gate/up kernel's tile height (#3995 47876fbc):
        /// 128 rows once every expert averages more than 64 (and at most 128)
        /// rows over a wide input, else 64.
        private static func legacyTileRows(rows: Int, experts: Int, input: Int) -> Int {
            rows > 64 * experts && rows <= 128 * experts && input >= 2048 ? 128 : 64
        }

        /// `(row_start, expert, rows, 0)` tile descriptors (`maximum` slots,
        /// `count` written) of at most `rows` rows each.
        private struct Tiles {
            let list: MLXArray
            let count: MLXArray
            let maximum: Int
            let rows: Int
        }

        private static func tiles(_ sorted: MLXArray, rows: Int, experts: Int, tileRows: Int)
            -> Tiles
        {
            let maximum = (rows + tileRows - 1) / tileRows + min(experts, rows)
            let result = scanKernel(
                [sorted, MLXArray([Int32(rows), Int32(experts), Int32(maximum)])],
                template: [("BM", tileRows), ("MAXE", maximumExperts)],
                grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[maximum * 4], [1]], outputDTypes: [.uint32, .uint32])
            return Tiles(list: result[0], count: result[1], maximum: maximum, rows: tileRows)
        }

        /// Launch threads: (column, tile) threadgroups, or with tile-on-x
        /// `groupX` tiles on x and (tile group, column) on y, matching
        /// `glm_tile_of`.
        private static func grid(_ plan: TilePlan, columns: Int, maximumTiles: Int)
            -> (Int, Int, Int)
        {
            let (x, y) =
                plan.groupX > 0
                ? (plan.groupX, (maximumTiles + plan.groupX - 1) / plan.groupX * columns)
                : (columns, maximumTiles)
            return (x * 32, y * 2, plan.shape.rows / 32)
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

        /// The pinned header with this shape's compile-time tile constants.
        /// Every replaced line must be present (a pinned-header update that
        /// renames one fails here, at kernel construction, not at run time).
        private static func matmulHeader(_ shape: TileShape) -> String {
            var header = MiMoV26NAXMetalSources.matmulHeader
            for (pinned, local) in [
                ("STEEL_CONST int kBM = 64;", "STEEL_CONST int kBM = \(shape.rows);"),
                ("STEEL_CONST int kBK = 64;", "STEEL_CONST int kBK = \(shape.depth);"),
                ("STEEL_CONST int kWM = 2;", "STEEL_CONST int kWM = \(shape.rows / 32);"),
                (
                    "STEEL_CONST int kVPT = kBN * kBK / kThreads;",
                    "STEEL_CONST int kLT = \(shape.loaderThreads);\n"
                        + "STEEL_CONST int kVPT = kBN * kBK / kLT;"
                ),
            ] {
                precondition(header.contains(pinned), "pinned NAX header lacks \(pinned)")
                header = header.replacingOccurrences(of: pinned, with: local)
            }
            return MiMoV26NAXMetalSources.mlxHeader + "\n" + header + "\n" + affineHeader + "\n"
        }

        private static let gateUp64 = makeGateUp(tileRows: 64)
        private static let gateUp128 = makeGateUp(tileRows: 128)

        private static func gateUpKernel(_ rows: Int) -> MLXFast.MLXFastKernel {
            rows == 128 ? gateUp128 : gateUp64
        }

        private static let gateUpKernels = Dictionary(
            uniqueKeysWithValues: TileShape.all.map { ($0, makeInterleavedGateUp($0)) })
        private static let downKernels = Dictionary(
            uniqueKeysWithValues: TileShape.all.map { ($0, makeDown($0)) })

        /// The interleaved gate/up for both bank formats on one tile shape;
        /// `DB`, `GX`, `NC` (column blocks) and `PAD` are template arguments.
        private static func makeInterleavedGateUp(_ shape: TileShape) -> MLXFast.MLXFastKernel {
            MLXFast.metalKernel(
                name: "glm5_next_moe_prefill_gate_up_bm\(shape.rows)_bk\(shape.depth)",
                inputNames: [
                    "x", "gate_w", "gate_s", "gate_b", "gate_g", "up_w", "up_s", "up_b",
                    "up_g", "tiles", "tile_count", "order", "params", "limit",
                ],
                outputNames: ["act"],
                source: """
                    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
                    threadgroup T Ws[(DB ? 2 : 1) * omlx_gqmm::kBN * BKP + PAD / sizeof(T)];
                    uint4 desc;
                    int col;
                    if (!omlx_gqmm::glm_tile_of<GX, NC>(
                            tiles, tile_count[0], threadgroup_position_in_grid, desc, col)) {
                        return;
                    }
                    if constexpr (FP) {
                        using Q = omlx_gqmm::GlmNvfp4Q<T>;
                        const Q gate_q{(const device uint8_t*)gate_s};
                        const Q up_q{(const device uint8_t*)up_s};
                        omlx_gqmm::glm_gather_gate_up<DB, T, Q, TopK, true>(
                            x, (const device uint8_t*)gate_w, (const device uint8_t*)up_w,
                            gate_q, up_q, gate_g, up_g, desc, col, order, act, params[0],
                            params[1], limit[0], Ws, simdgroup_index_in_threadgroup,
                            thread_index_in_simdgroup);
                    } else {
                        using Q = omlx_gqmm::GlmAffineQ<T, GS>;
                        const Q gate_q{(const device T*)gate_s, (const device T*)gate_b, 0.0f, 0.0f};
                        const Q up_q{(const device T*)up_s, (const device T*)up_b, 0.0f, 0.0f};
                        omlx_gqmm::glm_gather_gate_up<DB, T, Q, TopK, false>(
                            x, (const device uint8_t*)gate_w, (const device uint8_t*)up_w,
                            gate_q, up_q, gate_g, up_g, desc, col, order, act, params[0],
                            params[1], limit[0], Ws, simdgroup_index_in_threadgroup,
                            thread_index_in_simdgroup);
                    }
                    """,
                header: matmulHeader(shape), ensureRowContiguous: true)
        }

        /// The separate-tile affine gate/up (`gateUpSchedule: false`), on the
        /// legacy 64-deep tiles of `tileRows` rows.
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
                header: matmulHeader(TileShape(rows: tileRows, depth: 64)),
                ensureRowContiguous: true)
        }

        /// Down for both bank formats on one tile shape (affine: `g` is an
        /// unread placeholder; NVFP4: `biases` is).
        private static func makeDown(_ shape: TileShape) -> MLXFast.MLXFastKernel {
            MLXFast.metalKernel(
                name: "glm5_next_moe_prefill_down_bm\(shape.rows)_bk\(shape.depth)",
                inputNames: [
                    "x", "w", "scales", "biases", "g", "tiles", "tile_count", "params",
                ],
                outputNames: ["y"],
                source: """
                    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
                    threadgroup T Ws[(DB ? 2 : 1) * omlx_gqmm::kBN * BKP + PAD / sizeof(T)];
                    uint4 desc;
                    int col;
                    if (!omlx_gqmm::glm_tile_of<GX, NC>(
                            tiles, tile_count[0], threadgroup_position_in_grid, desc, col)) {
                        return;
                    }
                    if constexpr (FP) {
                        using Q = omlx_gqmm::GlmNvfp4Q<T>;
                        const Q q{(const device uint8_t*)scales};
                        omlx_gqmm::glm_gather_down<DB, T, Q, true>(
                            x, (const device uint8_t*)w, q, g, desc, col, y, params[0],
                            params[1], Ws, simdgroup_index_in_threadgroup,
                            thread_index_in_simdgroup);
                    } else {
                        using Q = omlx_gqmm::GlmAffineQ<T, GS>;
                        const Q q{(const device T*)scales, (const device T*)biases, 0.0f, 0.0f};
                        omlx_gqmm::glm_gather_down<DB, T, Q, false>(
                            x, (const device uint8_t*)w, q, g, desc, col, y, params[0],
                            params[1], Ws, simdgroup_index_in_threadgroup,
                            thread_index_in_simdgroup);
                    }
                    """,
                header: matmulHeader(shape), ensureRowContiguous: true)
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

            // The row tile and column block of this threadgroup. GX == 0:
            // (column, tile) as MLX lays the grid out. GX > 0 (oMLX 97e473ef's
            // tile-on-x layout): tile t on grid x t % GX and (t / GX, column)
            // on y, so all NC column blocks of a row tile share one x
            // coordinate and run GX threadgroups apart.
            template <int GX, int NC>
            METAL_FUNC bool glm_tile_of(
                const device uint32_t* tiles,
                const uint tile_count,
                const uint3 tg,
                thread uint4& desc,
                thread int& col) {
              const uint t = GX > 0 ? (tg.y / uint(NC)) * uint(GX) + tg.x : tg.y;
              col = int(GX > 0 ? tg.y % uint(NC) : tg.x);
              if (t >= tile_count) {
                return false;
              }
              desc = *((const device uint4*)tiles + t);
              return true;
            }

            // One kBK-deep K step of a simdgroup's 32 x 32 block: kBK / 32
            // sub-steps of 16x32x16 tensor ops in ascending K over the
            // dequantized weight rows at wb (this simdgroup's 32 columns), with
            // activation fragments read from device memory at xs + x_off. The
            // 16-row fragments past m_frags are skipped. Every schedule and
            // tile height issues these ops in this order per output element.
            template <typename T, typename WT, typename XO>
            METAL_FUNC void glm_k_step(
                thread NAXTile<float, kTM, kTN>& D,
                const device T* xs,
                const thread XO (&x_off)[kTM][2],
                const threadgroup WT* wb,
                const int k0,
                const short m_frags) {
              constexpr int BKP = kBK + 16 / sizeof(WT);
              STEEL_PRAGMA_UNROLL
              for (short kk1 = 0; kk1 < kBK; kk1 += kSK) {
                NAXTile<WT, kTN, 2> Btile;
                Btile.template load<WT, BKP, 1>(wb + kk1);
                const int k = k0 + kk1;
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
            }

            // The K loop of one tile. DB: 64-deep weight tiles in two buffers;
            // step k + 1's packed words are fetched before step k's tensor ops
            // and dequantized into the other buffer after them (one barrier per
            // step). Otherwise (seg): each kBK-deep step is dequantized into one
            // buffer between two barriers. Only loader threads touch the weight
            // tile; every thread reaches every barrier.
            template <bool DB, typename T, typename L, typename XO>
            METAL_FUNC void glm_k_loop(
                thread NAXTile<float, kTM, kTN>& D,
                thread L& loader,
                const bool loads,
                const device T* xs,
                const thread XO (&x_off)[kTM][2],
                threadgroup typename L::WT* Ws,
                const int sg_col,
                const int K,
                const short m_frags) {
              using WT = typename L::WT;
              static_assert(!DB || kBK == 64, "double-buffered tiles are 64 deep");
              constexpr int kTile = kBN * (kBK + 16 / sizeof(WT));
              const int K_it = K / kBK;
              if constexpr (DB) {
                if (loads) {
                  loader.fetch(0);
                  loader.store(Ws);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int kb = 0; kb < K_it; kb++) {
                  const bool more = kb + 1 < K_it;
                  if (more && loads) {
                    loader.fetch(kb + 1);
                  }
                  glm_k_step<T, WT, XO>(
                      D, xs, x_off, Ws + (kb & 1) * kTile + sg_col, kb * kBK, m_frags);
                  if (more && loads) {
                    loader.store(Ws + ((kb + 1) & 1) * kTile);
                  }
                  threadgroup_barrier(mem_flags::mem_threadgroup);
                }
              } else {
                for (int kb = 0; kb < K_it; kb++) {
                  threadgroup_barrier(mem_flags::mem_threadgroup);
                  if (loads) {
                    loader.fetch(kb);
                    loader.store(Ws);
                  }
                  threadgroup_barrier(mem_flags::mem_threadgroup);
                  glm_k_step<T, WT, XO>(D, xs, x_off, Ws + sg_col, kb * kBK, m_frags);
                }
              }
            }

            // #4022's interleaved gate/up on one single-expert tile: weight-tile
            // rows alternate 16 gate and 16 up rows of the same 16 outputs, so
            // each simdgroup's two 16-column fragments hold gate and up of
            // identical elements and the clamped GLU needs no exchange.
            // Activation fragments are read through the sort order. Requires K
            // in whole kBK steps and N in whole 32-value blocks.
            template <bool DB, typename T, typename Q, int TOPK, bool FP, typename G>
            METAL_FUNC void glm_gather_gate_up(
                const device T* x,
                const device uint8_t* gate_w,
                const device uint8_t* up_w,
                thread const Q& gate_q,
                thread const Q& up_q,
                G gate_g,
                G up_g,
                const uint4 desc,
                const int col,
                const device uint32_t* order,
                device T* act,
                const int N,
                const int K,
                const float limit,
                threadgroup typename Q::WT* Ws,
                const uint sgid,
                const uint lane) {
              using WT = typename Q::WT;
              constexpr int BKP = kBK + 16 / sizeof(WT);
              static_assert(kTN == 2, "one gate and one up fragment per simdgroup");
              const int row_start = int(desc.x);
              const uint32_t expert = desc.y;
              const int tile_rows = int(desc.z);

              const int K_w = K * Q::kBits / 8;
              const int K_g = K / Q::kGroup;
              const int y_col = col * (kBN / 2);

              // Threads past kLT (96-row tiles) do not load; their loader
              // addresses a valid row and is never used.
              const uint lid = sgid * 32 + lane;
              const bool loads = lid < uint(kLT);
              const uint llid = min(lid, uint(kLT - 1));
              const int lrow = int(llid) / kTPR;
              const int blk = lrow / 16;
              const bool is_up = (blk & 1) != 0;
              const size_t w_row = size_t(expert) * N + y_col + (blk >> 1) * 16 + (lrow & 15);
              Q q = is_up ? up_q : gate_q;
              q.advance(w_row * K_g);
              GlmRowLoader<Q> loader((is_up ? up_w : gate_w) + w_row * K_w, q, llid);

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

              NAXTile<float, kTM, kTN> D;
              D.clear();
              glm_k_loop<DB, T>(
                  D, loader, loads, x, x_off, Ws, (sgid % kWN) * kSN * BKP, K, m_frags);

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

            // Down projection of the sorted activation rows on one single-expert
            // tile, with GlmRowLoader (any group size, NVFP4 included) and the
            // global-scale epilogue. Requires K in whole kBK steps and N in
            // whole 64-value blocks.
            template <bool DB, typename T, typename Q, bool FP, typename G>
            METAL_FUNC void glm_gather_down(
                const device T* x,
                const device uint8_t* w,
                thread const Q& q0,
                G global_g,
                const uint4 desc,
                const int col,
                device T* y,
                const int N,
                const int K,
                threadgroup typename Q::WT* Ws,
                const uint sgid,
                const uint lane) {
              using WT = typename Q::WT;
              constexpr int BKP = kBK + 16 / sizeof(WT);
              const int row_start = int(desc.x);
              const uint32_t expert = desc.y;
              const int tile_rows = int(desc.z);

              const int K_w = K * Q::kBits / 8;
              const int K_g = K / Q::kGroup;
              const int y_col = col * kBN;
              const uint lid = sgid * 32 + lane;
              const bool loads = lid < uint(kLT);
              const uint llid = min(lid, uint(kLT - 1));
              const size_t w_row = size_t(expert) * N + y_col + llid / kTPR;
              Q q = q0;
              q.advance(w_row * K_g);
              GlmRowLoader<Q> loader(w + w_row * K_w, q, llid);

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

              NAXTile<float, kTM, kTN> D;
              D.clear();
              glm_k_loop<DB, T>(
                  D, loader, loads, xs, x_off, Ws, (sgid % kWN) * kSN * BKP, K, m_frags);

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
