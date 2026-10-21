// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Eigen Labs.
// Metal adapted from GLM5NextMoEPrefillKernel (mlx-swift-lm
// work/glm53-flash-release 97b2ca6d), itself adapted from oMLX (Apache-2.0,
// Copyright oMLX contributors; notice: docs/mimo-v26/LICENSE-OMLX): #3995
// 47876fbc310fbb311cd382c4eba8572ec4368308 (segmented single-expert NAX
// tiles), 97e473efe6836d6b44a01a5e0e63358ba0611db8 (tile plan: 64/96/128-row
// tiles, 128-deep K steps, tile-on-x layout, threadgroup pad), through the
// pinned MiMo port in `MiMoV26NAXMetalSources` (Apple MLX MIT primitives,
// notice retained there). Modified: ModelOpt NVFP4 expert rows already sorted
// by `graphForward`; the ModelOpt weight-scale and relu2 epilogue.

import Foundation
import MLX

/// The sorted product of `ModelOptNVFP4SwitchLinear.graphForward`'s tiled
/// route on the Neural Accelerators, in place of MLX's sorted
/// `gatherQuantizedMM` plus `scaledCast`.
///
/// The rows (already rounded to the FP4 grid and sorted by expert) are cut
/// into single-expert tiles of 64, 96 or 128 rows by one scan launch. One
/// launch then multiplies each tile by its expert's weights: 64-value-deep
/// weight tiles are decoded a packed word at a time into threadgroup memory
/// (double-buffered, one barrier per K step), the FP32 accumulators run on
/// 16x32x16 tensor ops in ascending K, and 16-row fragments without rows are
/// skipped. The epilogue rounds each sum to the activation dtype, applies the
/// expert's weight scale and rounds again (MLX's gathered output, then
/// `scaledCast`), with relu2 for fc1.
///
/// Weights are decoded exactly (E2M1 times E4M3 is exact in BF16), so
/// only the summation order inside a 16-deep tensor op can differ from MLX's
/// sorted NAX gather. Every `TilePlan` issues the same tensor ops per output
/// element in the same K order, so plans are bitwise identical to each other.
public enum ModelOptNVFP4NAXExperts {
    /// On wherever NAX is available. On an M5 Ultra at Super's shapes (45,056
    /// routed rows, 512 experts) it matches MLX's sorted NAX gather bit for bit
    /// and takes 2.98 instead of 4.7 ms for fc1 and 2.96 instead of 3.75 ms
    /// for fc2.
    nonisolated(unsafe) public static var isEnabled = naxAvailable && runtimeCompilerSupported

    /// Tile plans pinned by projection input width; a plan that does not admit
    /// the width is ignored. The defaults are the fastest measured on an M5
    /// Ultra for Super's fc1 (K = 1024) and fc2 (K = 2688).
    nonisolated(unsafe) public static var planOverrides: [Int: TilePlan] = [
        1024: TilePlan(rows: 64, depth: 64, doubleBuffered: true, groupX: 32, pad: 0),
        2688: TilePlan(rows: 96, depth: 128, doubleBuffered: false, groupX: 32, pad: 0),
    ]

    /// The tile scan holds every expert's row range in threadgroup memory.
    public static let maximumExperts = 1024

    public static var naxAvailable: Bool { MiMoV26NAXGatherQMM.naxAvailable }

    /// Whether the OS's Metal compiler builds the tensor-op kernels compiled
    /// here at first use (these, ``ModelOptNAXDenseLinear`` and Super's
    /// chunked SSD). Validated on macOS 27; macOS 26's compiler rejects some
    /// of them ("unsupported deferred-static-alloca-size"), which MLX reports
    /// as a fatal error, so earlier systems keep MLX's prebuilt NAX kernels.
    public static let runtimeCompilerSupported = ProcessInfo.processInfo.isOperatingSystemAtLeast(
        OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0))

    public enum Failure: Error, CustomStringConvertible {
        case unsupported(String)

        public var description: String {
            switch self {
            case .unsupported(let reason): "ModelOptNVFP4NAXExperts unsupported: \(reason)"
            }
        }
    }

    /// Rows of one single-expert tile and the depth of one K step. Both are
    /// compile-time constants of the matmul header, so each shape is its own
    /// kernel. `loaderThreads` decode the 64 x `depth` weight tile: every
    /// thread when they split it evenly, else 128 (96-row tiles: 128 of 192).
    public struct TileShape: Hashable, Sendable, CustomStringConvertible {
        public let rows: Int
        public let depth: Int

        public static let all: [TileShape] = [
            .init(rows: 64, depth: 64), .init(rows: 96, depth: 64), .init(rows: 128, depth: 64),
            .init(rows: 96, depth: 128), .init(rows: 128, depth: 128),
        ]

        public init(rows: Int, depth: Int) {
            self.rows = rows
            self.depth = depth
        }

        var loaderThreads: Int { rows == 96 ? 128 : 2 * rows }
        public var description: String { "\(rows)x64 bk\(depth)" }
    }

    /// One launch configuration (oMLX 97e473ef's `Plan`). `doubleBuffered`:
    /// 64-deep weight tiles in two buffers, one barrier per K step; otherwise
    /// (seg) each `depth`-deep step is decoded into one buffer between two
    /// barriers. `groupX` > 0 is the tile-on-x layout: tile t on grid x
    /// `t % groupX`, its column blocks on y `groupX` threadgroups apart, so
    /// every threadgroup of a row tile shares one x coordinate (0: MLX's
    /// (column, tile) grid). `pad` bytes of unused threadgroup memory lower
    /// residency. `prefetch` (seg only) issues step k + 1's packed weight
    /// words right after step k is decoded.
    public struct TilePlan: Hashable, Sendable, CustomStringConvertible {
        public let shape: TileShape
        public let doubleBuffered: Bool
        public let groupX: Int
        public let pad: Int
        public let prefetch: Bool

        public init(
            rows: Int, depth: Int, doubleBuffered: Bool, groupX: Int, pad: Int,
            prefetch: Bool = false
        ) {
            shape = TileShape(rows: rows, depth: depth)
            self.doubleBuffered = doubleBuffered
            self.groupX = groupX
            self.pad = pad
            self.prefetch = prefetch
        }

        /// Parses `description`, e.g. "db 96x64 bk64 gx32" or
        /// "seg+pf 128x64 bk128 gx32 pad8192".
        public init?(_ text: String) {
            let tokens = text.split(separator: " ").map(String.init)
            guard let mode = tokens.first, ["db", "seg", "seg+pf"].contains(mode) else {
                return nil
            }
            var rows: Int?
            var depth: Int?
            var groupX = 0
            var pad = 0
            for token in tokens.dropFirst() {
                if token.hasSuffix("x64"), let value = Int(token.dropLast(3)) {
                    rows = value
                } else if token.hasPrefix("bk"), let value = Int(token.dropFirst(2)) {
                    depth = value
                } else if token.hasPrefix("gx"), let value = Int(token.dropFirst(2)) {
                    groupX = value
                } else if token.hasPrefix("pad"), let value = Int(token.dropFirst(3)) {
                    pad = value
                } else {
                    return nil
                }
            }
            guard let rows, let depth else { return nil }
            self.init(
                rows: rows, depth: depth, doubleBuffered: mode == "db", groupX: groupX, pad: pad,
                prefetch: mode == "seg+pf")
            guard TileShape.all.contains(shape) else { return nil }
        }

        public var description: String {
            "\(doubleBuffered ? "db" : prefetch ? "seg+pf" : "seg") \(shape)"
                + (groupX > 0 ? " gx\(groupX)" : "") + (pad > 0 ? " pad\(pad)" : "")
        }

        /// Whether this plan's kernels read a projection of `input` columns.
        public func admits(input: Int) -> Bool {
            TileShape.all.contains(shape) && input % shape.depth == 0
                && (!doubleBuffered || shape.depth == 64) && !(doubleBuffered && prefetch)
                && groupX >= 0 && (0 ... 8192).contains(pad) && pad % 16 == 0
        }

        /// oMLX 97e473ef's `_plan` (measured on M5 Ultra with affine and
        /// MXFP4 banks at Qwen3.8, GLM-5.3 and MiMo-V2.6 expert shapes,
        /// 1K-8K-token chunks): weight streaming dominates below 36 rows per
        /// expert and for short inputs below 120; from 36 rows activation
        /// reuse across a tile's columns does (tile-on-x); 48-95 rows fill
        /// 96-row tiles; from 96 rows 128-row tiles stream each expert's
        /// weights once, with 128-deep K steps.
        public static func plan(rows: Int, experts: Int, input: Int) -> TilePlan {
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
                ? TilePlan(
                    rows: 128, depth: 128, doubleBuffered: false, groupX: 32, pad: 8192,
                    prefetch: true)
                : TilePlan(rows: 128, depth: 64, doubleBuffered: true, groupX: 32, pad: 0)
        }
    }

    /// Whether `product` runs `rows` sorted rows of `dtype` through a bank of
    /// `experts` `[outputs, inputs]` NVFP4 matrices on this device and stream.
    /// BF16 only: with FP16, the weight-tile vector stores gave wrong products
    /// under an M3 tensor-op emulation, so FP16 is not qualified.
    public static func supports(
        rows: Int, inputs: Int, outputs: Int, experts: Int, dtype: DType
    ) -> Bool {
        naxAvailable && dtype == .bfloat16
            && (1 ... maximumExperts).contains(experts)
            && inputs > 0 && inputs % 64 == 0 && outputs > 0 && outputs % 64 == 0
            && (16 ... ModelOptNVFP4SwitchLinear.maximumGraphRows).contains(rows)
            && MiMoV26NAXGatherQMM.gpuStream(.default)
    }

    /// The plan `product` uses for `rows` rows of `input` columns.
    public static func plan(rows: Int, experts: Int, input: Int) -> TilePlan {
        if let pinned = planOverrides[input], pinned.admits(input: input) { return pinned }
        return TilePlan.plan(rows: rows, experts: experts, input: input)
    }

    /// `[R, N]` in `x.dtype` (`supports`): row r of `x` (`[R, K]`) times
    /// expert `experts[r]` (uint32, sorted ascending, all below E) of
    /// `weight` (`[E, N, K/8]` packed E2M1) with `scales` (`[E, N, K/16]`
    /// E4M3). With `expertScale` (`[E]` float32) each sum is rounded to T,
    /// multiplied by its expert's scale and rounded again, then relu2'd when
    /// `squaredReLU`; without it the rounded sums (MLX's gathered output).
    /// Unsorted IDs give undefined rows but no access outside the arrays.
    static func product(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, experts: MLXArray,
        expertScale: MLXArray?, squaredReLU: Bool, plan pinned: TilePlan? = nil
    ) -> MLXArray {
        let (rows, inputs) = (x.dim(0), x.dim(1))
        let (count, outputs) = (weight.dim(0), weight.dim(1))
        precondition(
            x.ndim == 2 && x.dtype == .bfloat16 && rows >= 16
                && experts.shape == [rows] && experts.dtype == .uint32
                && (1 ... maximumExperts).contains(count) && weight.dtype == .uint32
                && weight.shape == [count, outputs, inputs / 8] && scales.dtype == .uint8
                && scales.shape == [count, outputs, inputs / 16] && outputs % 64 == 0
                && (expertScale == nil || expertScale!.shape == [count])
                && (expertScale != nil || !squaredReLU),
            "NVFP4 NAX expert geometry")
        let plan = pinned ?? Self.plan(rows: rows, experts: count, input: inputs)
        precondition(plan.admits(input: inputs), "tile plan \(plan) for K \(inputs)")
        let maximumTiles = (rows + plan.shape.rows - 1) / plan.shape.rows + min(count, rows)
        let tiles = scanKernel(
            [experts, MLXArray([Int32(rows), Int32(count), Int32(maximumTiles)])],
            template: [("BM", plan.shape.rows), ("MAXE", maximumExperts)],
            grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[maximumTiles * 4], [1]], outputDTypes: [.uint32, .uint32])
        return gemm(
            x, weight: weight, scales: scales, expertScale: expertScale,
            squaredReLU: squaredReLU, tiles: tiles[0], tileCount: tiles[1],
            maximumTiles: maximumTiles, plan: plan)
    }

    /// `product`'s matrix launch over precomputed `(row_start, expert, rows,
    /// 0)` tiles of `plan`'s height (`maximumTiles` slots, `tileCount`
    /// written). With `roundingScale` (`[E]` float32, the next projection's
    /// NVFP4 global activation scales) the epilogue also rounds each row to
    /// that projection's activation grid (QOUT; requires `expertScale`).
    static func gemm(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, expertScale: MLXArray?,
        squaredReLU: Bool, tiles: MLXArray, tileCount: MLXArray, maximumTiles: Int,
        plan: TilePlan, roundingScale: MLXArray? = nil
    ) -> MLXArray {
        let (rows, inputs) = (x.dim(0), x.dim(1))
        let outputs = weight.dim(1)
        let columns = outputs / 64
        var arrays = [
            x, weight, scales, expertScale ?? MLXArray([Float(1)]), tiles, tileCount,
            MLXArray([Int32(outputs), Int32(inputs)]),
        ]
        var template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("DB", plan.doubleBuffered), ("PF", plan.prefetch),
            ("GX", plan.groupX), ("NC", columns), ("PAD", plan.pad),
            ("SCALED", expertScale != nil), ("RELU2", squaredReLU),
        ]
        var kernel = gemmKernels[plan.shape]!
        if let roundingScale {
            precondition(expertScale != nil, "QOUT rounds the scaled output")
            arrays.append(roundingScale)
            template.append(("QOUT", true))
            kernel = roundingGemmKernels[plan.shape]!
        }
        return kernel(
            arrays, template: template,
            grid: grid(plan, columns: columns, maximumTiles: maximumTiles),
            threadGroup: (32, 2, plan.shape.rows / 32),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
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

    // MARK: - Diagnostic

    /// Runs `rows` sorted rows of `k` columns through a random NVFP4 bank of
    /// `experts` `[n, k]` matrices with this kernel and with MLX's sorted
    /// `gatherQuantizedMM` (plus `scaledCast` when `fused`, the tiled route's
    /// two launches), on the default GPU stream.
    ///
    /// Rows per expert follow a log-normal of `skew` (1: median about 0.6 of
    /// the mean, as Super's routing; 0: uniform). Activations are BF16
    /// standard normals rounded to the FP4 grid with `nvfp4Routed`, so inputs
    /// are the tiled route's. `fused: false` compares the bare sums (both
    /// GEMMs only).
    ///
    /// Returns the largest absolute difference, the largest difference over
    /// its reference row's largest magnitude (the tiled route's documented
    /// bound is 2^-7), both infinite when NaN positions differ, and the
    /// median wall time per call over `iterations` evaluated calls.
    public static func compareWithMLX(
        rows: Int, k: Int, n: Int, experts: Int, seed: UInt64,
        squaredReLU: Bool = false, fused: Bool = true, skew: Double = 1,
        plan: TilePlan? = nil, iterations: Int = 10
    ) throws -> (maxAbs: Float, maxRel: Float, seconds: (nax: Double, mlx: Double)) {
        let dtype = DType.bfloat16
        guard supports(rows: rows, inputs: k, outputs: n, experts: experts, dtype: dtype) else {
            throw Failure.unsupported(
                "rows \(rows), k \(k), n \(n), experts \(experts), NAX \(naxAvailable)")
        }
        if let plan, !plan.admits(input: k) {
            throw Failure.unsupported("tile plan \(plan) for k \(k)")
        }
        guard iterations > 0, skew >= 0, experts * n * (k / 8) <= Int(Int32.max) else {
            throw Failure.unsupported("iterations \(iterations), skew \(skew), bank size")
        }
        return try withError {
            var random = DiagnosticRandom(seed: seed)
            let counts = random.skewedCounts(rows: rows, experts: experts, sigma: skew)
            let ids = MLXArray(
                counts.enumerated().flatMap {
                    Array(repeating: UInt32($0.offset), count: $0.element)
                })
            let weight = randomBits(
                count: experts * n * k / 8, dtype: .uint32, scales: false, salt: random.word()
            ).reshaped(experts, n, k / 8)
            let scales = randomBits(
                count: experts * n * k / 16, dtype: .uint8, scales: true, salt: random.word()
            ).reshaped(experts, n, k / 16)
            let tables = ModelOptNVFP4RouteTables(
                inputScales: (0 ..< experts).map { _ in Float(4.5 / 2688 * (1 + random.unit())) },
                tensorScales: (0 ..< experts).map { _ in Float(0.001 + 0.01 * random.unit()) },
                activationOverflow: .saturating)
            let source = MLXRandom.normal([rows, k], dtype: dtype, key: MLXRandom.key(seed))
            let quantized = ModelOptActivationKernels.nvfp4Routed(
                source, sources: MLXArray((0 ..< rows).map { Int32($0) }), experts: ids,
                globalScale: tables.globalScale, overflowDivisor: tables.overflowDivisor,
                inverseWeightScale: tables.inverseWeightScale, activationDType: dtype)
            try checkedEval(
                weight, scales, quantized.activation, quantized.safeExperts, quantized.rowScale,
                tables.inverseWeightScale)

            func nax() -> MLXArray {
                product(
                    quantized.activation, weight: weight, scales: scales,
                    experts: quantized.safeExperts,
                    expertScale: fused ? tables.inverseWeightScale : nil,
                    squaredReLU: fused && squaredReLU, plan: plan)
            }
            func mlx() -> MLXArray {
                let sums = gatherQuantizedMM(
                    quantized.activation.reshaped(rows, 1, k), weight, scales: scales,
                    biases: nil, rhsIndices: quantized.safeExperts, transpose: true,
                    groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: true
                ).reshaped(rows, n)
                guard fused else { return sums }
                return ModelOptActivationKernels.scaledCast(
                    sums, scale: quantized.rowScale, dtype: dtype, squaredReLU: squaredReLU)
            }
            func seconds(_ body: () -> MLXArray) throws -> Double {
                for _ in 0 ..< 2 { try checkedEval(body()) }
                var times: [Double] = []
                for _ in 0 ..< iterations {
                    let start = DispatchTime.now().uptimeNanoseconds
                    try checkedEval(body())
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
                }
                return times.sorted()[times.count / 2]
            }

            let actual = nax().asType(.float32)
            let expected = mlx().asType(.float32)
            let nanActual = isNaN(actual)
            let nanExpected = isNaN(expected)
            let either = logicalOr(nanActual, nanExpected)
            let difference = which(either, MLXArray(Float(0)), abs(actual - expected))
            let rowMax = which(nanExpected, MLXArray(Float(0)), abs(expected))
                .max(axis: -1, keepDims: true)
            let maxAbs = difference.max()
            let maxRel = (difference / maximum(rowMax, MLXArray(Float(1e-30)))).max()
            let nanMismatch = notEqual(nanActual, nanExpected).any()
            try checkedEval(maxAbs, maxRel, nanMismatch)
            let mismatch = nanMismatch.item(Bool.self)
            let timeNAX = try seconds(nax)
            let timeMLX = try seconds(mlx)
            return (
                maxAbs: mismatch ? .infinity : maxAbs.item(Float.self),
                maxRel: mismatch ? .infinity : maxRel.item(Float.self),
                seconds: (nax: timeNAX, mlx: timeMLX)
            )
        }
    }

    /// `count` hashed words (`scales`: E4M3 codes 0x28-0x48, 0.25 to 4).
    static func randomBits(count: Int, dtype: DType, scales: Bool, salt: UInt32)
        -> MLXArray
    {
        randomKernel(
            [MLXArray([salt, 0])], template: [("O", dtype), ("SCALES", scales)],
            grid: (count, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[count]], outputDTypes: [dtype])[0]
    }

    struct DiagnosticRandom {
        var state: UInt64

        init(seed: UInt64) { state = seed }

        /// SplitMix64.
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }

        mutating func word() -> UInt32 { UInt32(truncatingIfNeeded: next() >> 32) }

        mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }

        mutating func gaussian() -> Double {
            let u = max(unit(), .leastNormalMagnitude)
            return (-2 * log(u)).squareRoot() * cos(2 * .pi * unit())
        }

        /// Log-normal expert weights, floored to rows; the remainder goes
        /// one row each to the first experts.
        mutating func skewedCounts(rows: Int, experts: Int, sigma: Double) -> [Int] {
            let weights = (0 ..< experts).map { _ in exp(sigma * gaussian()) }
            let total = weights.reduce(0, +)
            var counts = weights.map { Int(Double(rows) * $0 / total) }
            for e in 0 ..< rows - counts.reduce(0, +) { counts[e % experts] += 1 }
            return counts
        }
    }

    // MARK: - Kernels

    private static let randomKernel = MLXFast.metalKernel(
        name: "modelopt_nvfp4_nax_random_bits",
        inputNames: ["salt"], outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            uint h = (i + 1u) * 0x9E3779B9u ^ salt[0];
            h ^= h >> 16;
            h *= 0x85EBCA6Bu;
            h ^= h >> 13;
            h *= 0xC2B2AE35u;
            h ^= h >> 16;
            out[i] = SCALES ? static_cast<O>(0x28u + h % 33u) : static_cast<O>(h);
            """)

    /// Sorted expert IDs into `(row_start, expert, rows, 0)` tiles of at most
    /// BM rows, expert-major (`MiMoV26NAXMetalSources.scanHeader`).
    private static let scanKernel = MLXFast.metalKernel(
        name: "modelopt_nvfp4_nax_tile_scan",
        inputNames: ["idx", "params"], outputNames: ["tiles", "tile_count"],
        source: MiMoV26NAXMetalSources.scanSource,
        header: MiMoV26NAXMetalSources.scanHeader + "\n", ensureRowContiguous: true)

    private static let gemmKernels = Dictionary(
        uniqueKeysWithValues: TileShape.all.map { ($0, makeKernel($0, rounding: false)) })

    /// The kernels with an `out_global` input for the QOUT epilogue.
    private static let roundingGemmKernels = Dictionary(
        uniqueKeysWithValues: TileShape.all.map { ($0, makeKernel($0, rounding: true)) })

    /// One tile shape; `DB`, `PF`, `GX`, `NC` (column blocks), `PAD`,
    /// `SCALED` and `RELU2` (and with `rounding`, `QOUT`) are template
    /// arguments. Without `SCALED`, `expert_scale` is an unread placeholder,
    /// as is `out_global` without `QOUT`.
    private static func makeKernel(_ shape: TileShape, rounding: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "modelopt_nvfp4_nax_experts\(rounding ? "_q" : "")"
                + "_bm\(shape.rows)_bk\(shape.depth)",
            inputNames: ["x", "w", "scales", "expert_scale", "tiles", "tile_count", "params"]
                + (rounding ? ["out_global"] : []),
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
                using Q = omlx_gqmm::GlmNvfp4Q<T>;
                const Q q{(const device uint8_t*)scales};
                """ + "\n"
                    + (rounding
                        ? """
                        omlx_gqmm::modelopt_nax_gather<DB, PF, T, Q, SCALED, RELU2, QOUT>(
                            x, (const device uint8_t*)w, q, expert_scale, desc, col, y, params[0],
                            params[1], Ws, simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
                            out_global);
                        """
                        : """
                        omlx_gqmm::modelopt_nax_gather<DB, PF, T, Q, SCALED, RELU2>(
                            x, (const device uint8_t*)w, q, expert_scale, desc, col, y, params[0],
                            params[1], Ws, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                        """),
            header: matmulHeader(shape), ensureRowContiguous: true)
    }

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
        return MiMoV26NAXMetalSources.mlxHeader + "\n" + header + "\n"
            + ModelOptActivationKernels.header + "\n" + expertHeader + "\n"
    }

    /// GLM5NextMoEPrefillKernel's `affineHeader`, NVFP4 parts verbatim
    /// (`glm5p_e2m1`/`glm5p_e4m3`, `GlmNvfp4Q`, `glm5p_out`, `GlmRowLoader`,
    /// `glm_tile_of`, `glm_k_step`, `glm_k_loop`), and `glm_gather_down` as
    /// `modelopt_nax_gather` with the ModelOpt epilogue.
    private static let expertHeader = #"""
        namespace omlx_gqmm {

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
        //
        // One packed word's eight codes are decoded together: codes j and
        // j + 4 become the two halves of one half2 bit pattern (sign to bit
        // 15, magnitude to bits 9-11), which is exactly e2m1 * 2^-14, and
        // the group scale is pre-multiplied by 2^14 (exact). The FP32
        // product is then the same exact value as glm5p_e2m1(q) * scale
        // (at most 6 significant bits), so every rounded weight is bitwise
        // the per-code decode's, in fewer instructions.
        template <typename T>
        struct GlmNvfp4Q {
          using WT = T;
          STEEL_CONST int kBits = 4;
          STEEL_CONST int kGroup = 16;
          STEEL_CONST bool kWord = true;
          const device uint8_t* scales;

          METAL_FUNC void advance(const size_t n) thread {
            scales += n;
          }
          // The scale times 2^14, for dq_word.
          METAL_FUNC void params(const int g, thread float& s_, thread float& b_) const thread {
            s_ = glm5p_e4m3(scales[g]) * 16384.0f;
            b_ = 0.0f;
          }
          static METAL_FUNC vec<WT, 8> dq_word(const uint32_t w, const float s16) {
            vec<WT, 8> v;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
              const uint32_t u = w >> (4 * j);
              const half2 h =
                  as_type<half2>(((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12));
              v[j] = static_cast<WT>(static_cast<float>(h.x) * s16);
              v[j + 4] = static_cast<WT>(static_cast<float>(h.y) * s16);
            }
            return v;
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

        // TileLoader with an explicit per-thread weight row, so one tile can
        // interleave rows of two matrices. A thread's kVPT values lie in
        // one group or span whole groups (power-of-two sizes, aligned
        // runs), so each group's parameters are read and decoded once.
        template <typename Q>
        struct GlmRowLoader {
          using WT = typename Q::WT;
          STEEL_CONST int kBits = Q::kBits;
          STEEL_CONST int kWords = kVPT * kBits / 32;
          STEEL_CONST int kPer = 32 / kBits;
          STEEL_CONST int kGroups = kVPT > Q::kGroup ? kVPT / Q::kGroup : 1;
          STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
          static_assert(Q::kGroup % kPer == 0, "whole words per group");
          static_assert(kVPT % Q::kGroup == 0 || Q::kGroup % kVPT == 0, "aligned groups");

          const device uint32_t* src;
          Q q;
          const short row;
          const short col;
          uint32_t raw[kWords];
          float sc[kGroups];
          float bi[kGroups];

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
            }
            const int g0 = (kb * kBK + col) / Q::kGroup;
            STEEL_PRAGMA_UNROLL
            for (short g = 0; g < kGroups; g++) {
              q.params(g0 + g, sc[g], bi[g]);
            }
          }

          METAL_FUNC void store(threadgroup WT* Ws) const thread {
            threadgroup WT* dst = Ws + row * (kBK + 16 / sizeof(WT)) + col;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kWords; i++) {
              const short g = short((i * kPer) / Q::kGroup);
              vec<WT, kPer> v;
              if constexpr (Q::kWord) {
                v = Q::dq_word(raw[i], sc[g]);
              } else {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kPer; j++) {
                  v[j] = Q::dqp((raw[i] >> (kBits * j)) & kMask, sc[g], bi[g]);
                }
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
        // buffer between two barriers; PF fetches step k + 1's packed words
        // right after step k's dequantization, so the load overlaps step k's
        // tensor ops. Only loader threads touch the weight tile; every
        // thread reaches every barrier.
        template <bool DB, bool PF, typename T, typename L, typename XO>
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
          static_assert(!(DB && PF), "double buffering already prefetches");
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
            if (PF && loads) {
              loader.fetch(0);
            }
            for (int kb = 0; kb < K_it; kb++) {
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (loads) {
                if (!PF) {
                  loader.fetch(kb);
                }
                loader.store(Ws);
                if (PF && kb + 1 < K_it) {
                  loader.fetch(kb + 1);
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              glm_k_step<T, WT, XO>(D, xs, x_off, Ws + sg_col, kb * kBK, m_frags);
            }
          }
        }

        // glm_gather_down on one single-expert tile of sorted rows, with the
        // ModelOpt epilogue: SCALED rounds each FP32 sum to T, multiplies it
        // by the expert's weight scale and rounds again (MLX's gathered T
        // output, then scaledCast); RELU2 then squares max(y, 0) in T as
        // NemotronH's relu2, keeping NaN. QOUT then rounds the tile to the
        // next projection's NVFP4 activation grid with that projection's
        // global scale `out_global[expert]`, as nvfp4Body does under
        // .saturating, except that NaN spreads over a 16-column block rather
        // than the row. Requires K in whole kBK steps and N in whole
        // 64-value blocks.
        template <
            bool DB, bool PF, typename T, typename Q, bool SCALED, bool RELU2,
            bool QOUT = false, typename G, typename G2 = const device float*>
        METAL_FUNC void modelopt_nax_gather(
            const device T* x,
            const device uint8_t* w,
            thread const Q& q0,
            G expert_scale,
            const uint4 desc,
            const int col,
            device T* y,
            const int N,
            const int K,
            threadgroup typename Q::WT* Ws,
            const uint sgid,
            const uint lane,
            G2 out_global = G2()) {
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
          glm_k_loop<DB, PF, T>(
              D, loader, loads, xs, x_off, Ws, (sgid % kWN) * kSN * BKP, K, m_frags);

          float scale = 1.0f;
          if constexpr (SCALED) {
            scale = expert_scale[expert];
          }
          NAXTile<T, kTM, kTN> out;
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kTM; i++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j++) {
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < decltype(out)::kElemsPerFrag; e++) {
                T value = glm5p_out<T, SCALED>(D.frag_at(i, j)[e], scale);
                if constexpr (RELU2) {
                  const T zero = static_cast<T>(0);
                  const T r = metal::isnan(static_cast<float>(value))
                      ? value
                      : (value > zero ? value : zero);
                  value = r * r;
                }
                out.frag_at(i, j)[e] = value;
              }
            }
          }
          if constexpr (QOUT) {
            // A fragment is 16 aligned columns, one NVFP4 block: lane l holds
            // columns 4 * ((l & 1) | ((l >> 2) & 2)) + c (BaseNAXFrag's
            // get_coord) of rows fm (elements 0-3) and fm + 8 (4-7), so the
            // block of a row is the 4 lanes that differ in lane bits 0 and 3.
            // Then nvfp4Body's per-block arithmetic, verbatim.
            const float scale_enc = (448.0f * 6.0f) / out_global[expert];
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kTM; i++) {
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < kTN; j++) {
                STEEL_PRAGMA_UNROLL
                for (short r = 0; r < 2; r++) {
                  float v[4];
                  float block_max = 0.0f;
                  uint block_nan = 0u;
                  STEEL_PRAGMA_UNROLL
                  for (short c = 0; c < 4; c++) {
                    v[c] = static_cast<float>(out.frag_at(i, j)[r * 4 + c]);
                    block_nan |= metal::isnan(v[c]) ? 1u : 0u;
                    block_max = metal::max(block_max, metal::abs(v[c]));
                  }
                  block_max = metal::max(block_max, simd_shuffle_xor(block_max, ushort(1)));
                  block_max = metal::max(block_max, simd_shuffle_xor(block_max, ushort(8)));
                  block_nan |= simd_shuffle_xor(block_nan, ushort(1));
                  block_nan |= simd_shuffle_xor(block_nan, ushort(8));
                  float scale_dec = block_max;
                  scale_dec /= 6.0f;
                  scale_dec *= scale_enc;
                  const float scale = modelopt_e4m3_value(modelopt_e4m3_bits(scale_dec));
                  const float scale_enc_b = (scale == 0) ? 0.0f : (scale_enc / scale);
                  STEEL_PRAGMA_UNROLL
                  for (short c = 0; c < 4; c++) {
                    const float q = modelopt_e2m1_value(modelopt_e2m1_bits(v[c] * scale_enc_b));
                    const float value = static_cast<float>(static_cast<T>((scale / scale_enc) * q));
                    out.frag_at(i, j)[r * 4 + c] =
                        static_cast<T>(block_nan != 0u ? nan_value : value);
                  }
                }
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
}
