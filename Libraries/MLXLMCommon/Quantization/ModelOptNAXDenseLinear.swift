// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Eigen Labs.
// Metal adapted from ModelOptNVFP4NAXExperts (the GLM5NextMoEPrefillKernel
// port: NVFP4 word decode, row loader, double-buffered and prefetching K
// loops), itself from oMLX (Apache-2.0, Copyright oMLX contributors; notice:
// docs/mimo-v26/LICENSE-OMLX) through the pinned MiMo port in
// `MiMoV26NAXMetalSources` (Apple MLX MIT primitives, notice retained there).
// Modified: one dense matrix instead of expert tiles, the ModelOpt FP8
// per-tensor weight decode, and the dense NVFP4 scale and relu2 epilogue.

import Foundation
import MLX

/// The many-row (prefill) product of `ModelOptCalibratedLinear.graphForward`
/// on the Neural Accelerators with the weight decode fused into the matrix
/// kernel, in place of
/// - FP8: MLX's NAX GEMM on a BF16 copy of the weight that the graph path
///   decodes on every call (plus, beyond 32 Mi weights, a concatenation of the
///   output tiles), and
/// - NVFP4: MLX's NVFP4 NAX product with float32 sums followed by the separate
///   `scaledCast` (weight scale, relu2) pass.
///
/// The activation quantization stays a separate pass (`ModelOptActivationKernels`),
/// as every output column block would repeat it otherwise.
///
/// Weights are decoded per K step into threadgroup memory with the graph
/// path's rounding (FP8: `T(e4m3 * tensorScale)`; NVFP4: E2M1 times E4M3,
/// exact in BF16), and the FP32 accumulators run the same 16x32x16 tensor ops
/// in ascending K as MLX's NAX kernels, on the same operand types (FP8: BF16 x
/// BF16; NVFP4: float32 x BF16, as MLX's product with float32 activations).
/// The epilogue rounds once (FP8: the sum; NVFP4: the sum times the inverse
/// weight scale, as `scaledCast`). So the result should be bit-identical to
/// the graph path wherever that path runs MLX's regular NAX kernels, which
/// `takesFP8` and `takesNVFP4` require (no split-K, TF32 on for NVFP4). Under
/// an M3 tensor-op emulation it equals MLX's non-split-K products bit for bit;
/// `compareWithReference` checks it on the device.
public enum ModelOptNAXDenseLinear {
    /// On wherever NAX is available: on an M5 Ultra it matches MLX's products
    /// bit for bit at every Super prefill shape, and with the fused expert
    /// route takes Super prefill from 2,368 to 2,471 tokens/s. Clearing it
    /// keeps MLX's products.
    nonisolated(unsafe) public static var isEnabled = naxAvailable

    /// Tile plans pinned by projection input width (tuning only); other
    /// widths use ``TilePlan/plan(inputs:)``. A plan that does not admit the
    /// width is ignored.
    nonisolated(unsafe) public static var planOverrides: [Int: TilePlan] = [:]

    /// NVFP4 activations in BF16 instead of float32: the same values (they
    /// are rounded to BF16 before they are stored), half the activation
    /// traffic, but BF16 x BF16 tensor ops instead of MLX's float32 x BF16 ones.
    /// Off by default; bit identity with the graph path is not established.
    nonisolated(unsafe) public static var nvfp4BFloat16Activations = false

    /// Fewest rows the graph path hands to the kernel: prefill chunks only,
    /// never decode, row-exact batches or verify windows.
    public static let minimumRows = 128

    public static var naxAvailable: Bool { MiMoV26NAXGatherQMM.naxAvailable }

    /// MLX's `MLX_ENABLE_TF32` (default on): its NVFP4 product with float32
    /// activations runs on the Neural Accelerators only with TF32 enabled.
    public static let mlxTF32Enabled: Bool = {
        guard let value = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] else {
            return true
        }
        return (Int(value.trimmingCharacters(in: .whitespaces)) ?? 0) != 0
    }()

    public enum Failure: Error, CustomStringConvertible {
        case unsupported(String)

        public var description: String {
            switch self {
            case .unsupported(let reason): "ModelOptNAXDenseLinear unsupported: \(reason)"
            }
        }
    }

    /// Rows of one tile and the depth of one K step (compile-time constants of
    /// the matmul header, so each shape is its own kernel). Columns are 64 per
    /// threadgroup; every thread decodes weights.
    public struct TileShape: Hashable, Sendable, CustomStringConvertible {
        public let rows: Int
        public let depth: Int

        public static let all: [TileShape] = [
            .init(rows: 64, depth: 64), .init(rows: 128, depth: 64),
            .init(rows: 64, depth: 128), .init(rows: 128, depth: 128),
        ]

        public init(rows: Int, depth: Int) {
            self.rows = rows
            self.depth = depth
        }

        public var description: String { "\(rows)x64 bk\(depth)" }
    }

    /// `doubleBuffered`: 64-deep weight tiles in two buffers, one barrier per
    /// K step; otherwise each `depth`-deep step is decoded into one buffer
    /// between two barriers, and `prefetch` reads step k + 1's weights while
    /// step k multiplies. `pad` bytes of unused threadgroup memory lower
    /// residency. Every plan issues the same tensor ops per output element in
    /// the same K order, so plans are bitwise identical to each other.
    public struct TilePlan: Hashable, Sendable, CustomStringConvertible {
        public let shape: TileShape
        public let doubleBuffered: Bool
        public let prefetch: Bool
        public let pad: Int

        public init(
            rows: Int, depth: Int, doubleBuffered: Bool, prefetch: Bool = false, pad: Int = 0
        ) {
            shape = TileShape(rows: rows, depth: depth)
            self.doubleBuffered = doubleBuffered
            self.prefetch = prefetch
            self.pad = pad
        }

        /// Parses `description`, e.g. "db 128x64 bk64" or "seg+pf 128x64 bk128 pad8192".
        public init?(_ text: String) {
            let tokens = text.split(separator: " ").map(String.init)
            guard let mode = tokens.first, ["db", "seg", "seg+pf"].contains(mode) else {
                return nil
            }
            var rows: Int?
            var depth: Int?
            var pad = 0
            for token in tokens.dropFirst() {
                if token.hasSuffix("x64"), let value = Int(token.dropLast(3)) {
                    rows = value
                } else if token.hasPrefix("bk"), let value = Int(token.dropFirst(2)) {
                    depth = value
                } else if token.hasPrefix("pad"), let value = Int(token.dropFirst(3)) {
                    pad = value
                } else {
                    return nil
                }
            }
            guard let rows, let depth else { return nil }
            self.init(
                rows: rows, depth: depth, doubleBuffered: mode == "db",
                prefetch: mode == "seg+pf", pad: pad)
            guard TileShape.all.contains(shape) else { return nil }
        }

        public var description: String {
            "\(doubleBuffered ? "db" : prefetch ? "seg+pf" : "seg") \(shape)"
                + (pad > 0 ? " pad\(pad)" : "")
        }

        /// The plans worth sweeping at Super's shapes.
        public static let sweep: [TilePlan] = [
            .init(rows: 64, depth: 64, doubleBuffered: true),
            .init(rows: 128, depth: 64, doubleBuffered: true),
            .init(rows: 128, depth: 128, doubleBuffered: false, prefetch: true),
            .init(rows: 128, depth: 128, doubleBuffered: false, prefetch: true, pad: 8192),
            .init(rows: 64, depth: 128, doubleBuffered: false, prefetch: true),
        ]

        /// Whether this plan's kernels read a projection of `inputs` columns.
        /// Single-buffered steps need the prefetch: without it, 128x64 bk128
        /// with relu2 gave wrong products under the M3 tensor-op emulation.
        public func admits(inputs: Int) -> Bool {
            TileShape.all.contains(shape) && inputs % shape.depth == 0
                && (doubleBuffered ? shape.depth == 64 && !prefetch : prefetch)
                && (0 ... 8192).contains(pad) && pad % 16 == 0
        }

        /// The expert kernel's plan for runs of 96 or more rows (measured on
        /// the M5 Ultra): 128-row tiles stream each weight once, 128 deep with
        /// prefetch when the width allows.
        public static func plan(inputs: Int) -> TilePlan {
            inputs % 128 == 0
                ? TilePlan(rows: 128, depth: 128, doubleBuffered: false, prefetch: true, pad: 8192)
                : TilePlan(rows: 128, depth: 64, doubleBuffered: true)
        }
    }

    /// The plan the product uses for `inputs` columns.
    public static func plan(inputs: Int) -> TilePlan {
        if let pinned = planOverrides[inputs], pinned.admits(inputs: inputs) { return pinned }
        return TilePlan.plan(inputs: inputs)
    }

    /// Geometry the kernel runs on this device and the default stream. BF16
    /// only (FP16 is not qualified: the expert kernel's weight-tile vector
    /// stores gave wrong products under an M3 tensor-op emulation).
    public static func supports(rows: Int, inputs: Int, outputs: Int, dtype: DType) -> Bool {
        naxAvailable && dtype == .bfloat16 && rows >= 1 && inputs > 0 && inputs % 64 == 0
            && outputs > 0 && outputs % 64 == 0 && rows <= 1 << 16
            && outputs.multipliedReportingOverflow(by: inputs).partialValue < Int(Int32.max)
            && MiMoV26NAXGatherQMM.gpuStream(.default)
    }

    /// MLX's regular (not split-K) NAX GEMM for one `[rows, inputs] x
    /// [inputs, outputs]` BF16 product (mlx 81f58d79 `matmul.cpp`).
    static func mlxRegularNAXGEMM(rows: Int, inputs: Int, outputs: Int) -> Bool {
        let largest = max(rows, outputs)
        return !(inputs >= 3 * largest || (largest <= 1024 && inputs > 2 * largest))
    }

    /// Whether the FP8 graph path hands `rows` rows to the kernel: enabled,
    /// supported, a prefill chunk, and every eager output tile (`tileOutputs`)
    /// a regular NAX GEMM in MLX, so the result keeps the graph path's bits.
    static func takesFP8(
        rows: Int, inputs: Int, outputs: Int, tileOutputs: [Int], dtype: DType
    ) -> Bool {
        isEnabled && rows >= minimumRows
            && supports(rows: rows, inputs: inputs, outputs: outputs, dtype: dtype)
            && tileOutputs.allSatisfy {
                mlxRegularNAXGEMM(rows: rows, inputs: inputs, outputs: $0)
            }
    }

    /// Whether the NVFP4 tiled graph path hands `rows` rows to the kernel:
    /// enabled, supported, a prefill chunk, and MLX's product there is its
    /// regular NAX kernel (`qmm_splitk` keeps one partition from 257 32x32
    /// tiles; float32 activations need TF32).
    static func takesNVFP4(rows: Int, inputs: Int, outputs: Int, dtype: DType) -> Bool {
        isEnabled && rows >= minimumRows && mlxTF32Enabled
            && ((rows + 31) / 32) * ((outputs + 31) / 32) > 256
            && supports(rows: rows, inputs: inputs, outputs: outputs, dtype: dtype)
    }

    /// The activation dtype the NVFP4 route quantizes into.
    static var nvfp4ActivationDType: DType { nvfp4BFloat16Activations ? .bfloat16 : .float32 }

    /// `[R, N]` BF16: `activation` (`[R, K]` BF16, already FP8-rounded) times
    /// the E4M3 `weight` (`[N, K]` uint8) decoded as `T(e4m3 * scale)`
    /// (`scale` `[1]` float32), one rounding of the FP32 sum.
    static func fp8Product(
        _ activation: MLXArray, weight: MLXArray, scale: MLXArray, plan: TilePlan? = nil
    ) -> MLXArray {
        precondition(
            weight.dtype == .uint8 && weight.ndim == 2 && weight.dim(1) == activation.dim(1),
            "FP8 NAX dense geometry")
        // Unread; 8 elements keep it in device memory like real block scales
        // (MLX passes smaller inputs in the constant address space).
        return launch(
            activation, weight: weight, blockScales: MLXArray.zeros([8], type: UInt8.self),
            scale: scale, nvfp4: false, squaredReLU: false, dtype: activation.dtype, plan: plan)
    }

    /// `[R, N]` in `dtype`: `activation` (`[R, K]`, float32 or BF16, on the
    /// FP4 grid) times the NVFP4 `weight` (`[N, K/8]` uint32) with `blockScales`
    /// (`[N, K/16]` E4M3), each FP32 sum times `outputScale` (`[1]`) rounded
    /// once to `dtype`, then relu2'd when `squaredReLU` (as `scaledCast`).
    static func nvfp4Product(
        _ activation: MLXArray, weight: MLXArray, blockScales: MLXArray, outputScale: MLXArray,
        squaredReLU: Bool, dtype: DType, plan: TilePlan? = nil
    ) -> MLXArray {
        let inputs = activation.dim(1)
        precondition(
            weight.dtype == .uint32 && weight.shape == [weight.dim(0), inputs / 8]
                && blockScales.dtype == .uint8
                && blockScales.shape == [weight.dim(0), inputs / 16],
            "NVFP4 NAX dense geometry")
        return launch(
            activation, weight: weight, blockScales: blockScales, scale: outputScale,
            nvfp4: true, squaredReLU: squaredReLU, dtype: dtype, plan: plan)
    }

    private static func launch(
        _ activation: MLXArray, weight: MLXArray, blockScales: MLXArray, scale: MLXArray,
        nvfp4: Bool, squaredReLU: Bool, dtype: DType, plan pinned: TilePlan?
    ) -> MLXArray {
        let (rows, inputs) = (activation.dim(0), activation.dim(1))
        let outputs = weight.dim(0)
        precondition(
            activation.ndim == 2 && dtype == .bfloat16
                && (activation.dtype == dtype || (nvfp4 && activation.dtype == .float32))
                && inputs % 64 == 0 && outputs % 64 == 0 && scale.size == 1
                && scale.dtype == .float32,
            "NAX dense operands")
        let plan = pinned ?? Self.plan(inputs: inputs)
        precondition(plan.admits(inputs: inputs), "tile plan \(plan) for K \(inputs)")
        let tiles = (rows + plan.shape.rows - 1) / plan.shape.rows
        return kernels[plan.shape]!(
            [
                activation, weight, blockScales, scale.reshaped(1),
                MLXArray([Int32(rows), Int32(outputs), Int32(inputs)]),
            ],
            template: [
                ("T", dtype), ("TA", activation.dtype), ("NVFP4", nvfp4), ("RELU2", squaredReLU),
                ("DB", plan.doubleBuffered), ("PF", plan.prefetch), ("PAD", plan.pad),
            ],
            grid: (tiles * 32, outputs / 64 * 2, plan.shape.rows / 32),
            threadGroup: (32, 2, plan.shape.rows / 32),
            outputShapes: [[rows, outputs]], outputDTypes: [dtype])[0]
    }

    // MARK: - Comparison

    /// The kernel against the current graph path on one random projection.
    public struct Comparison: Sendable, CustomStringConvertible {
        public let plan: TilePlan
        /// Largest absolute difference; infinite when NaN positions differ.
        public let maxAbs: Float
        /// Largest difference over its reference row's largest magnitude.
        public let maxRel: Float
        /// Every output bit equal (NaN payloads included).
        public let bitIdentical: Bool
        /// Median seconds per evaluated `graphForward` call (activation
        /// quantization included), with the kernel and with the current path.
        public let kernelSeconds: Double
        public let referenceSeconds: Double

        public var description: String {
            "\(plan): bitIdentical \(bitIdentical) maxAbs \(maxAbs) maxRel \(maxRel) kernel "
                + String(
                    format: "%.3f ms, current %.3f ms (%.2fx)", kernelSeconds * 1e3,
                    referenceSeconds * 1e3, referenceSeconds / max(kernelSeconds, 1e-12))
        }
    }

    /// Runs `graphForward` of a random BF16 `ModelOptCalibratedLinear`
    /// (`inputs` -> `outputs`, `format`, Super's saturating activation policy,
    /// tiled accumulation) on `rows` random rows, once through the current
    /// path and once through this kernel, on the default GPU stream. Throws
    /// when the graph path would not take the kernel at this geometry.
    /// `nvfp4BFloat16Activations` nil keeps the global setting.
    ///
    /// Super's shapes at a 2,048-token chunk: FP8 in_proj 4096 -> 18560,
    /// out_proj 8192 -> 4096, q/o 4096 -> 4096, k/v 4096 -> 256; NVFP4 shared
    /// expert 4096 -> 5376 (`squaredReLU`) and 5376 -> 4096.
    public static func compareWithReference(
        rows: Int, inputs: Int, outputs: Int, format: ModelOptCalibratedLinear.Format,
        seed: UInt64, squaredReLU: Bool = false, plan: TilePlan? = nil,
        nvfp4BFloat16Activations bf16Activations: Bool? = nil, iterations: Int = 10
    ) throws -> Comparison {
        let dtype = DType.bfloat16
        guard iterations > 0, supports(rows: rows, inputs: inputs, outputs: outputs, dtype: dtype)
        else {
            throw Failure.unsupported(
                "rows \(rows), inputs \(inputs), outputs \(outputs), NAX \(naxAvailable)")
        }
        let chosen = plan ?? TilePlan.plan(inputs: inputs)
        guard chosen.admits(inputs: inputs) else {
            throw Failure.unsupported("tile plan \(chosen) for inputs \(inputs)")
        }
        let saved = (isEnabled, planOverrides, nvfp4BFloat16Activations)
        defer { (isEnabled, planOverrides, nvfp4BFloat16Activations) = saved }
        planOverrides[inputs] = chosen
        if let bf16Activations { nvfp4BFloat16Activations = bf16Activations }

        return try withError { _ in
            var random = UInt64(truncatingIfNeeded: seed) &* 0x9E37_79B9_7F4A_7C15 | 1
            func unit() -> Float {
                random ^= random << 13
                random ^= random >> 7
                random ^= random << 17
                return Float(random >> 40) / Float(1 << 24)
            }
            func bytes(_ count: Int, mode: Int) -> MLXArray {
                randomKernel(
                    [MLXArray([UInt32(truncatingIfNeeded: random >> 32) ^ UInt32(mode)])],
                    template: [("MODE", mode)], grid: (count, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [[count]], outputDTypes: [.uint8])[0]
            }
            let layer: ModelOptCalibratedLinear
            switch format {
            case .fp8:
                _ = unit()
                let weight = bytes(outputs * inputs, mode: 2).reshaped(outputs, inputs)
                try checkedEval(weight)
                layer = try ModelOptCalibratedLinear(
                    weights: ModelOptFP8Weights(
                        e4m3Bytes: weight, tensorScale: 0.0005 + 0.002 * unit()),
                    inputScale: 8 / 448 * (1 + unit()), weightDType: dtype,
                    activationOverflow: .saturating)
            case .nvfp4:
                _ = unit()
                let packed = bytes(outputs * inputs / 2, mode: 0).reshaped(outputs, inputs / 2)
                _ = unit()
                let scales = bytes(outputs * inputs / 16, mode: 1).reshaped(outputs, inputs / 16)
                try checkedEval(packed, scales)
                layer = try ModelOptCalibratedLinear(
                    weights: ModelOptNVFP4Weights(
                        packedBytes: packed, e4m3ScaleBytes: scales,
                        tensorScale: 0.001 + 0.01 * unit()),
                    inputScale: 4.5 / 2688 * (1 + unit()), weightDType: dtype,
                    activationOverflow: .saturating)
            }
            layer.accumulation = .tiled
            let x = (MLXRandom.normal([rows, inputs], key: MLXRandom.key(seed)) * 2)
                .asType(dtype)
            try checkedEval(x)

            isEnabled = true
            let taken =
                switch format {
                case .fp8:
                    takesFP8(
                        rows: rows, inputs: inputs, outputs: outputs,
                        tileOutputs: layer.fp8EagerTiles(rows: rows).map(\.count), dtype: dtype)
                case .nvfp4:
                    takesNVFP4(rows: rows, inputs: inputs, outputs: outputs, dtype: dtype)
                }
            guard taken else {
                throw Failure.unsupported(
                    "graphForward keeps MLX's product at rows \(rows) (split-K or TF32 off)")
            }
            func forward(_ enabled: Bool) throws -> MLXArray {
                isEnabled = enabled
                return try layer.graphForward(x, squaredReLU: squaredReLU)
            }
            func seconds(_ enabled: Bool) throws -> Double {
                for _ in 0 ..< 2 { try checkedEval(forward(enabled)) }
                var times: [Double] = []
                for _ in 0 ..< iterations {
                    let start = DispatchTime.now().uptimeNanoseconds
                    try checkedEval(forward(enabled))
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
                }
                return times.sorted()[times.count / 2]
            }

            let actual = try forward(true)
            let expected = try forward(false)
            let bits = all(
                equal(actual.view(dtype: .uint16), expected.view(dtype: .uint16)))
            let a = actual.asType(.float32)
            let b = expected.asType(.float32)
            let nanActual = isNaN(a)
            let nanExpected = isNaN(b)
            let difference = which(
                logicalOr(nanActual, nanExpected), MLXArray(Float(0)), abs(a - b))
            let rowMax = which(nanExpected, MLXArray(Float(0)), abs(b))
                .max(axis: -1, keepDims: true)
            let maxAbs = difference.max()
            let maxRel = (difference / maximum(rowMax, MLXArray(Float(1e-30)))).max()
            let nanMismatch = notEqual(nanActual, nanExpected).any()
            try checkedEval(bits, maxAbs, maxRel, nanMismatch)
            let mismatch = nanMismatch.item(Bool.self)
            let kernelTime = try seconds(true)
            let referenceTime = try seconds(false)
            return Comparison(
                plan: chosen,
                maxAbs: mismatch ? .infinity : maxAbs.item(Float.self),
                maxRel: mismatch ? .infinity : maxRel.item(Float.self),
                bitIdentical: bits.item(Bool.self),
                kernelSeconds: kernelTime, referenceSeconds: referenceTime)
        }
    }

    // MARK: - Kernels

    /// Hashed bytes. MODE 0: raw; 1: E4M3 block scales 0x28-0x48 (0.25 to 4);
    /// 2: every finite E4M3FN code.
    private static let randomKernel = MLXFast.metalKernel(
        name: "modelopt_nax_dense_random_bytes",
        inputNames: ["salt"], outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            uint h = (i + 1u) * 0x9E3779B9u ^ salt[0];
            h ^= h >> 16;
            h *= 0x85EBCA6Bu;
            h ^= h >> 13;
            h *= 0xC2B2AE35u;
            h ^= h >> 16;
            uint8_t code = static_cast<uint8_t>(h);
            if (MODE == 1) {
                code = static_cast<uint8_t>(0x28u + h % 33u);
            } else if (MODE == 2 && (code & 0x7F) == 0x7F) {
                code ^= 1;
            }
            out[i] = code;
            """)

    private static let kernels = Dictionary(
        uniqueKeysWithValues: TileShape.all.map { ($0, makeKernel($0)) })

    /// One tile shape; `T` (output), `TA` (activation), `NVFP4`, `RELU2`,
    /// `DB`, `PF` and `PAD` are template arguments. FP8 reads `scale` as its
    /// tensor weight scale and ignores `block_scales`; NVFP4 reads `scale` as
    /// its inverse weight scale.
    private static func makeKernel(_ shape: TileShape) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "modelopt_nax_dense_bm\(shape.rows)_bk\(shape.depth)",
            inputNames: ["x", "w", "block_scales", "scale", "params"],
            outputNames: ["y"],
            source: """
                constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
                threadgroup T Ws[(DB ? 2 : 1) * omlx_gqmm::kBN * BKP + PAD / sizeof(T)];
                using Q = typename omlx_gqmm::mdn_select<
                    NVFP4, omlx_gqmm::DenseNvfp4Q<T>, omlx_gqmm::DenseFp8Q<T>>::type;
                const Q q{(const device uint8_t*)block_scales, scale[0]};
                omlx_gqmm::modelopt_nax_dense<DB, PF, T, TA, Q, NVFP4, RELU2>(
                    x, (const device uint8_t*)w, q, scale[0],
                    int(threadgroup_position_in_grid.x), int(threadgroup_position_in_grid.y),
                    y, params[0], params[1], params[2], Ws, simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
                """,
            header: matmulHeader(shape), ensureRowContiguous: true)
    }

    /// The pinned header with this shape's compile-time tile constants. Every
    /// replaced line must be present (a pinned-header update that renames one
    /// fails here, at kernel construction, not at run time).
    private static func matmulHeader(_ shape: TileShape) -> String {
        var header = MiMoV26NAXMetalSources.matmulHeader
        for (pinned, local) in [
            ("STEEL_CONST int kBM = 64;", "STEEL_CONST int kBM = \(shape.rows);"),
            ("STEEL_CONST int kBK = 64;", "STEEL_CONST int kBK = \(shape.depth);"),
            ("STEEL_CONST int kWM = 2;", "STEEL_CONST int kWM = \(shape.rows / 32);"),
            (
                "STEEL_CONST int kVPT = kBN * kBK / kThreads;",
                "STEEL_CONST int kLT = kThreads;\nSTEEL_CONST int kVPT = kBN * kBK / kLT;"
            ),
        ] {
            precondition(header.contains(pinned), "pinned NAX header lacks \(pinned)")
            header = header.replacingOccurrences(of: pinned, with: local)
        }
        return MiMoV26NAXMetalSources.mlxHeader + "\n" + header + "\n" + denseHeader + "\n"
    }

    /// `ModelOptNVFP4NAXExperts.expertHeader`'s NVFP4 decode, row loader and
    /// K loops (activation type `TA` instead of `T`), an FP8 decode with the
    /// graph path's rounding, and the dense driver with the ModelOpt epilogue.
    private static let denseHeader = #"""
        namespace omlx_gqmm {

        template <bool B, typename A, typename C>
        struct mdn_select {
          using type = A;
        };
        template <typename A, typename C>
        struct mdn_select<false, A, C> {
          using type = C;
        };

        // MLX fp8.h E4M3 conversion (exact through half), as ModelOptActivationKernels.
        inline float mdn_e4m3(uint8_t bits) {
          uint16_t v = (bits & 127) << 7;
          half converted = as_type<half>(v);
          converted *= 256.0;
          return static_cast<float>((bits & 128) ? -converted : converted);
        }

        // NVFP4: E2M1 codes times the E4M3 group-16 scale, rounded once to the
        // activation dtype like MLX's fp QuantizedBlockLoader (exact in BF16).
        // One packed word's eight codes are decoded together: codes j and
        // j + 4 become the two halves of one half2 bit pattern (sign to bit
        // 15, magnitude to bits 9-11), exactly e2m1 * 2^-14, against the
        // group scale times 2^14 (exact).
        template <typename T>
        struct DenseNvfp4Q {
          using WT = T;
          STEEL_CONST int kBits = 4;
          STEEL_CONST int kGroup = 16;
          const device uint8_t* scales;
          float tensor_scale;

          METAL_FUNC void advance(const size_t n) thread {
            scales += n;
          }
          METAL_FUNC float params(const int g) const thread {
            return mdn_e4m3(scales[g]) * 16384.0f;
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

        // ModelOpt per-tensor FP8: T(e4m3 * tensor_scale) with E4M3FN NaN
        // codes, as the graph path's decoded weight (fp8DenseWeight).
        template <typename T>
        struct DenseFp8Q {
          using WT = T;
          STEEL_CONST int kBits = 8;
          STEEL_CONST int kGroup = 32;
          const device uint8_t* scales;
          float tensor_scale;

          METAL_FUNC void advance(const size_t) thread {}
          METAL_FUNC float params(const int) const thread {
            return tensor_scale;
          }
          static METAL_FUNC vec<WT, 4> dq_word(const uint32_t w, const float s) {
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            vec<WT, 4> v;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
              const uint8_t code = static_cast<uint8_t>(w >> (8 * j));
              const float decoded = (code & 0x7F) == 0x7F ? nan_value : mdn_e4m3(code);
              v[j] = static_cast<WT>(decoded * s);
            }
            return v;
          }
        };

        // Thread lid decodes kVPT consecutive values of weight row lid / kTPR
        // per K step; its values lie in one group or span whole groups.
        template <typename Q>
        struct DenseRowLoader {
          using WT = typename Q::WT;
          STEEL_CONST int kBits = Q::kBits;
          STEEL_CONST int kWords = kVPT * kBits / 32;
          STEEL_CONST int kPer = 32 / kBits;
          STEEL_CONST int kGroups = kVPT > Q::kGroup ? kVPT / Q::kGroup : 1;
          static_assert(Q::kGroup % kPer == 0, "whole words per group");
          static_assert(kVPT % Q::kGroup == 0 || Q::kGroup % kVPT == 0, "aligned groups");

          const device uint32_t* src;
          Q q;
          const short row;
          const short col;
          uint32_t raw[kWords];
          float sc[kGroups];

          METAL_FUNC DenseRowLoader(
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
              sc[g] = q.params(g0 + g);
            }
          }

          METAL_FUNC void store(threadgroup WT* Ws) const thread {
            threadgroup WT* dst = Ws + row * (kBK + 16 / sizeof(WT)) + col;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kWords; i++) {
              const short g = short((i * kPer) / Q::kGroup);
              *(threadgroup vec<WT, kPer>*)(dst + i * kPer) = Q::dq_word(raw[i], sc[g]);
            }
          }
        };

        // One kBK-deep K step of a simdgroup's 32 x 32 block: kBK / 32
        // sub-steps of 16x32x16 tensor ops in ascending K over the decoded
        // weight rows at wb (this simdgroup's 32 columns), with activation
        // fragments read from device memory at xs + x_off. 16-row fragments
        // past m_frags are skipped.
        template <typename TA, typename WT>
        METAL_FUNC void dense_k_step(
            thread NAXTile<float, kTM, kTN>& D,
            const device TA* xs,
            const thread int (&x_off)[kTM][2],
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
                NAXTile<TA, 1, 2> Atile;
                STEEL_PRAGMA_UNROLL
                for (short h = 0; h < 2; h++) {
                  const device TA* xp = xs + x_off[i][h] + k;
                  const vec<TA, 4> a0 = *(const device vec<TA, 4>*)(xp);
                  const vec<TA, 4> a1 = *(const device vec<TA, 4>*)(xp + 16);
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

        // The K loop. DB: 64-deep weight tiles in two buffers; step k + 1's
        // words are fetched before step k's tensor ops and decoded into the
        // other buffer after them (one barrier per step). Otherwise each
        // kBK-deep step is decoded into one buffer between two barriers; PF
        // fetches step k + 1's words right after step k's decode.
        template <bool DB, bool PF, typename TA, typename L>
        METAL_FUNC void dense_k_loop(
            thread NAXTile<float, kTM, kTN>& D,
            thread L& loader,
            const device TA* xs,
            const thread int (&x_off)[kTM][2],
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
            loader.fetch(0);
            loader.store(Ws);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kb = 0; kb < K_it; kb++) {
              const bool more = kb + 1 < K_it;
              if (more) {
                loader.fetch(kb + 1);
              }
              dense_k_step<TA, WT>(D, xs, x_off, Ws + (kb & 1) * kTile + sg_col, kb * kBK, m_frags);
              if (more) {
                loader.store(Ws + ((kb + 1) & 1) * kTile);
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
          } else {
            if (PF) {
              loader.fetch(0);
            }
            for (int kb = 0; kb < K_it; kb++) {
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (!PF) {
                loader.fetch(kb);
              }
              loader.store(Ws);
              if (PF && kb + 1 < K_it) {
                loader.fetch(kb + 1);
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              dense_k_step<TA, WT>(D, xs, x_off, Ws + sg_col, kb * kBK, m_frags);
            }
          }
        }

        // Row tile `tile` and column block `col` of y = x . W^T ([M, K] x
        // [N, K]). SCALED rounds each FP32 sum times `scale` once to T (NVFP4,
        // as scaledCast); otherwise the sum alone (FP8: the weight already
        // carries its scale). RELU2 then squares max(y, 0) in T as NemotronH's
        // relu2, keeping NaN. Requires K in whole kBK steps, N in whole
        // 64-column blocks and every thread a loader.
        template <
            bool DB, bool PF, typename T, typename TA, typename Q, bool SCALED, bool RELU2>
        METAL_FUNC void modelopt_nax_dense(
            const device TA* x,
            const device uint8_t* w,
            thread const Q& q0,
            const float scale,
            const int tile,
            const int col,
            device T* y,
            const int M,
            const int N,
            const int K,
            threadgroup typename Q::WT* Ws,
            const uint sgid,
            const uint lane) {
          using WT = typename Q::WT;
          static_assert(kLT == kThreads, "every thread decodes weights");
          constexpr int BKP = kBK + 16 / sizeof(WT);
          const int row_start = tile * kBM;
          const int tile_rows = min(int(kBM), M - row_start);

          const int K_w = K * Q::kBits / 8;
          const int K_g = K / Q::kGroup;
          const int y_col = col * kBN;
          const uint lid = sgid * 32 + lane;
          const size_t w_row = size_t(y_col) + lid / kTPR;
          Q q = q0;
          q.advance(w_row * K_g);
          DenseRowLoader<Q> loader(w + w_row * K_w, q, lid);

          const int m0 = kSM * int(sgid / kWN);
          const int rows = min(int(kSM), tile_rows - m0);
          const device TA* xs = x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;
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
          dense_k_loop<DB, PF, TA>(D, loader, xs, x_off, Ws, (sgid % kWN) * kSN * BKP, K, m_frags);

          NAXTile<T, kTM, kTN> out;
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kTM; i++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j++) {
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < decltype(out)::kElemsPerFrag; e++) {
                const float acc = D.frag_at(i, j)[e];
                T value = SCALED ? static_cast<T>(acc * scale) : static_cast<T>(acc);
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
