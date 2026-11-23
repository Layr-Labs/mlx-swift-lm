// Inkling 2 attention projection stage after the products, fused for prefill.
// Reference: transformers PR 51050 `modeling_inkling2_loop.py` (Apache-2.0).
import Foundation
import MLX
import MLXNN

/// The attention projection stage between the products and attention, one
/// kernel per output: the query and relative scaled L2 clips with the log
/// scaling of full layers, the key convolution with its L2 clip, and the value
/// convolution with its RMS norm. Each replaces a chain of compiled elementwise
/// kernels, reductions and casts that passes over the products several times.
///
/// Bit-identical to those chains (`Inkling2Fused`, and the plain graphs they
/// trace): the kernels perform the same FP32 operations on the same operands
/// with the same constants, round to the model dtype where the chains do, and
/// sum a row in MLX's order. A 128-wide row is summed by one simdgroup, four
/// consecutive elements sequentially per lane, then `simd_sum`, as MLX's
/// `row_reduce_simple`, which MLX takes from 32 rows up; a row of at most 64
/// sequentially in one thread, as `row_reduce_small`. The squares are rounded
/// to FP32 before they are summed, as MLX stores them. Calls that MLX would
/// sum another way (a 128-wide reduction over fewer than 32 rows) keep the
/// chains.
enum Inkling2ProjectionFusion {
    /// `DARKBLOOM_INKLING2_PROJECTION_FUSION=0` keeps the compiled chains.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PROJECTION_FUSION"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `enabled` for the current task.
    @TaskLocal static var override: Bool? = nil

    /// Off with the plain op-by-op graphs (`DARKBLOOM_INKLING2_FUSED=0`).
    static var active: Bool { (override ?? enabled) && Inkling2Fused.active }

    /// Width of a row summed by one simdgroup (32 lanes of 4 elements).
    static let wideRow = 128
    /// MLX sums contiguous rows of 65 or more with `row_reduce_simple` from
    /// this many rows (`row_reduce_general_dispatch`).
    static let simpleReductionRows = 32
    /// Rows of at most this width MLX sums sequentially per thread.
    static let smallRow = 64

    static let floatTypes: Set<DType> = [.bfloat16, .float16, .float32]

    /// What follows the convolution.
    enum Tail {
        /// `l2Clip(y, threshold:)`.
        case clip(Float)
        /// `unweightedRMSNorm(y, eps:)`.
        case norm(Float)
    }

    /// `logScaled(module(products))`: `products` `[B, L, heads * D]` (rows may
    /// be strided), `tau` `[1 or B, L, 1, 1]` FP32 or nil. Returns
    /// `[B, L, heads, D]` in the products' dtype, or nil when not covered.
    static func clip(_ products: MLXArray, _ module: Inkling2ScaledL2Clip, tau: MLXArray?)
        -> MLXArray?
    {
        let scale = module.scale
        guard active, products.ndim == 3, scale.ndim == 2, floatTypes.contains(products.dtype),
            floatTypes.contains(scale.dtype), products.strides[2] == 1
        else { return nil }
        let (b, l) = (products.dim(0), products.dim(1))
        let (heads, d) = (scale.dim(0), scale.dim(1))
        let rows = b * l * heads
        let wide = d == wideRow
        guard products.dim(2) == heads * d, rows > 0,
            wide ? rows >= simpleReductionRows : d <= smallRow
        else { return nil }
        var tauMode = 0
        var tauValues = constants.floats
        if let tau {
            guard tau.dtype == .float32, tau.size == l || tau.size == b * l else { return nil }
            tauMode = tau.size == l ? 1 : 2
            tauValues = tau.reshaped(-1)
        }
        let template: [(String, any KernelTemplateArg)] = [
            ("T", products.dtype), ("S", scale.dtype), ("H", heads), ("D", d), ("TAU", tauMode),
            ("THR", bits(module.threshold)), ("CEPS", bits(clipEpsilon)),
        ]
        let key = Key(
            kind: wide ? 0 : 1, types: [products.dtype, scale.dtype],
            values: [heads, d, b, l, tauMode, bits(module.threshold)])
        let launch = launches.launch(key) {
            wide
                ? clipWideKernel.prepare(
                    template: template, grid: ((rows + 7) / 8 * 256, 1, 1),
                    threadGroup: (256, 1, 1), outputShapes: [[b, l, heads, d]],
                    outputDTypes: [products.dtype])
                : clipSmallKernel.prepare(
                    template: template, grid: ((rows + 255) / 256 * 256, 1, 1),
                    threadGroup: (256, 1, 1), outputShapes: [[b, l, heads, d]],
                    outputDTypes: [products.dtype])
        }
        return launch([products, scale, tauValues, constants.zeros])[0]
    }

    /// The convolution of `products` `[B, L, heads * 128]` (rows may be
    /// strided) with its carried state, then `tail`, in `dtype`: what
    /// `convolve` then `l2Clip(...).asType(dtype)` or
    /// `unweightedRMSNorm(...).asType(dtype)` return, `[B, L, heads, 128]`.
    /// Stores the next state as `convolve` does. Nil, with nothing stored,
    /// when not covered.
    static func convolution(
        _ module: Module, _ products: MLXArray, input: MLXArray, projected: MLXArray?,
        heads: Int, tail: Tail, dtype: DType, slot: Int, kind: Inkling2ConvKind,
        state: Inkling2StepState
    ) -> MLXArray? {
        guard active, products.ndim == 3, products.dtype == dtype, floatTypes.contains(dtype),
            products.strides[2] == 1
        else { return nil }
        let (b, l, c) = (products.dim(0), products.dim(1), products.dim(2))
        let rows = b * l * heads
        guard c == heads * wideRow, rows >= simpleReductionRows else { return nil }

        let kernel: Int
        let taps: MLXArray
        var dynamic: MLXArray? = nil
        if let module = module as? Inkling2DynamicShortConvolution {
            kernel = module.kernel
            guard module.heads == heads, module.staticWeight.shape == [kernel, c] else {
                return nil
            }
            let p =
                projected
                ?? Inkling2RowExact.matmul(input, transposedWeight: module.weightProjection)
            guard p.ndim == 3, p.dim(0) == b, p.dim(1) == l, p.dim(2) == heads * kernel,
                floatTypes.contains(p.dtype)
            else { return nil }
            // `[kernel, C]`: lag-major, as stored.
            taps = module.staticWeight.asType(.float32)
            dynamic = p
        } else if let module = module as? Inkling2ShortConvolution {
            kernel = module.kernel
            guard module.weight.size == c * kernel else { return nil }
            // `[C, kernel]`: index `kernel - 1` weights the current token.
            taps = module.weight.reshaped(c, kernel).asType(.float32)
        } else {
            return nil
        }
        guard kernel >= 2 else { return nil }

        let stored = state.convState(slot: slot, kind)
        let previous = stored ?? MLXArray.zeros([b, kernel - 1, c], dtype: dtype)
        guard previous.shape == [b, kernel - 1, c], floatTypes.contains(previous.dtype) else {
            return nil
        }

        // The constants the chains trace: `sqrt(Float(kernel + 1))` (static)
        // or `sqrt(Float(2 * kernel + 1))` (dynamic), the clip's `1e-6`, the
        // mean's `1 / 128`.
        let divisor = sqrt(Float(dynamic == nil ? kernel + 1 : 2 * kernel + 1))
        let (tailCode, tailValue): (Int, Float) =
            switch tail {
            case .clip(let threshold): (0, threshold)
            case .norm(let eps): (1, eps)
            }
        let dynamicValues = dynamic ?? constants.floats
        let template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("P", previous.dtype), ("Q", dynamicValues.dtype), ("H", heads),
            ("K", kernel), ("DYN", dynamic != nil), ("TAIL", tailCode),
            ("DIV", bits(divisor)), ("TV", bits(tailValue)), ("CEPS", bits(clipEpsilon)),
            ("INV", bits(1 / Float(wideRow))),
        ]
        // With the stage kernels on, the kernel also writes the carried state
        // (the same rows of `previous ++ products`, copied).
        let writesState = Inkling2StageFusion.active && previous.dtype == dtype
        let key = Key(
            kind: writesState ? 3 : 2, types: [dtype, previous.dtype, dynamicValues.dtype],
            values: [heads, kernel, b, l, dynamic != nil ? 1 : 0, tailCode, bits(tailValue)])
        let launch = launches.launch(key) {
            let grid = ((rows + 7) / 8 * 256, 1, 1)
            return writesState
                ? convolutionStateKernel.prepare(
                    template: template, grid: grid, threadGroup: (256, 1, 1),
                    outputShapes: [[b, l, heads, wideRow], [b, kernel - 1, c]],
                    outputDTypes: [dtype, dtype])
                : convolutionKernel.prepare(
                    template: template, grid: grid, threadGroup: (256, 1, 1),
                    outputShapes: [[b, l, heads, wideRow]], outputDTypes: [dtype])
        }
        let out = launch([products, previous, taps, dynamicValues, constants.zeros])

        // The carried state exactly as `convolve` builds it.
        let next =
            writesState
            ? out[1]
            : concatenated([previous, products], axis: 1)[0..., l ..< (l + kernel - 1), 0...]
        state.setConvState(slot: slot, kind, next, input: products)
        return out[0]
    }

    /// `plainL2Clip`'s `1e-6`: a `Float` literal.
    static let clipEpsilon: Float = 1e-6

    static func bits(_ value: Float) -> Int { Int(Int32(bitPattern: value.bitPattern)) }

    // MARK: - Launches

    struct Key: Hashable {
        let kind: Int
        let types: [DType]
        let values: [Int]
    }

    final class Launches: @unchecked Sendable {
        private let lock = NSLock()
        private var launches: [Key: MLXFast.MLXFastKernel.Launch] = [:]
        /// Chunk lengths vary; start over rather than keep every one.
        static let capacity = 1024

        func launch(_ key: Key, _ make: () -> MLXFast.MLXFastKernel.Launch)
            -> MLXFast.MLXFastKernel.Launch
        {
            lock.withLock {
                if let l = launches[key] { return l }
                if launches.count >= Self.capacity { launches.removeAll() }
                let l = make()
                launches[key] = l
                return l
            }
        }
    }

    static let launches = Launches()

    final class Constants: @unchecked Sendable {
        // Eight elements: smaller inputs bind in the constant address space.
        let zeros: MLXArray = {
            let a = MLXArray([UInt32](repeating: 0, count: 8))
            eval(a)
            return a
        }()
        let floats: MLXArray = {
            let a = MLXArray([Float](repeating: 0, count: 8))
            eval(a)
            return a
        }()
    }

    static let constants = Constants()

    // MARK: - Kernels

    /// Helpers shared by the kernels. `materialized` rounds a product to FP32
    /// where MLX stores it (the squares before the row sum), so the compiler
    /// cannot fuse it into the following add; `zero` is 0 at run time.
    /// `clipFactor` is `minimum(threshold / (sqrt(sum) + eps), 1)` with MLX's
    /// `Minimum` (a NaN first operand wins).
    static let header = #"""
        inline float inkling2_pf_materialized(float v, uint zero) {
            return as_type<float>(as_type<uint>(v) | zero);
        }

        inline float inkling2_pf_clip_factor(float total, float threshold, float eps) {
            const float norm = metal::precise::sqrt(total);
            const float shifted = norm + eps;
            const float ratio = threshold / shifted;
            if (metal::isnan(ratio)) {
                return ratio;
            }
            return ratio < 1.0f ? ratio : 1.0f;
        }

        """#

    /// Scaled L2 clip of 128-wide rows, then the log scaling: one simdgroup
    /// per (token, head) row, lane `i` holds elements `4i ..< 4i + 4`.
    static let clipWideKernel = MLXFast.metalKernel(
        name: "inkling2_projection_clip_wide", inputNames: ["x", "scale", "tau", "zeros"],
        outputNames: ["y"],
        source: #"""
            const int lane = int(thread_index_in_simdgroup);
            const int row = int(threadgroup_position_in_grid.x) * 8
                + int(simdgroup_index_in_threadgroup);
            const int L = x_shape[1];
            if (row >= x_shape[0] * L * H) {
                return;
            }
            const int token = row / H;
            const int head = row - token * H;
            const int b = token / L;
            const int t = token - b * L;
            const int first = lane * 4;
            const int64_t base = b * x_strides[0] + t * x_strides[1] + head * 128 + first;
            const uint zero = zeros[0];
            float v[4];
            for (int i = 0; i < 4; ++i) {
                const float xv = static_cast<float>(x[base + i]);
                const float sv = static_cast<float>(
                    scale[head * scale_strides[0] + (first + i) * scale_strides[1]]);
                v[i] = xv * sv;
            }
            float total = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const float square = inkling2_pf_materialized(v[i] * v[i], zero);
                total = square + total;
            }
            total = simd_sum(total);
            const float factor =
                inkling2_pf_clip_factor(total, as_type<float>(THR), as_type<float>(CEPS));
            const int64_t out = int64_t(token) * (H * 128) + head * 128 + first;
            for (int i = 0; i < 4; ++i) {
                T r = static_cast<T>(v[i] * factor);
                if (TAU != 0) {
                    const float scaled = static_cast<float>(r)
                        * tau[(TAU == 1 ? t : token) * tau_strides[0]];
                    r = static_cast<T>(scaled);
                }
                y[out + i] = r;
            }
            """#,
        header: header, ensureRowContiguous: false)

    /// Scaled L2 clip of rows of at most 64, then the log scaling: one thread
    /// per (token, head) row.
    static let clipSmallKernel = MLXFast.metalKernel(
        name: "inkling2_projection_clip_small", inputNames: ["x", "scale", "tau", "zeros"],
        outputNames: ["y"],
        source: #"""
            const int row = int(thread_position_in_grid.x);
            const int L = x_shape[1];
            if (row >= x_shape[0] * L * H) {
                return;
            }
            const int token = row / H;
            const int head = row - token * H;
            const int b = token / L;
            const int t = token - b * L;
            const int64_t base = b * x_strides[0] + t * x_strides[1] + head * D;
            const uint zero = zeros[0];
            float v[D];
            for (int i = 0; i < D; ++i) {
                const float xv = static_cast<float>(x[base + i]);
                const float sv =
                    static_cast<float>(scale[head * scale_strides[0] + i * scale_strides[1]]);
                v[i] = xv * sv;
            }
            float total = 0.0f;
            for (int i = 0; i < D; ++i) {
                const float square = inkling2_pf_materialized(v[i] * v[i], zero);
                total = square + total;
            }
            const float factor =
                inkling2_pf_clip_factor(total, as_type<float>(THR), as_type<float>(CEPS));
            const int64_t out = int64_t(token) * (H * D) + head * D;
            for (int i = 0; i < D; ++i) {
                T r = static_cast<T>(v[i] * factor);
                if (TAU != 0) {
                    const float scaled = static_cast<float>(r)
                        * tau[(TAU == 1 ? t : token) * tau_strides[0]];
                    r = static_cast<T>(scaled);
                }
                y[out + i] = r;
            }
            """#,
        header: header, ensureRowContiguous: false)

    /// Short convolution (static taps `[C, K]`, or dynamic: static `[K, C]`
    /// plus per-token head weights `[B, L, H * K]`) rounded to `T`, then the
    /// L2 clip (TAIL 0) or the unweighted RMS norm (TAIL 1) of each 128-wide
    /// head row: one simdgroup per (token, head), lane `i` holds channels
    /// `4i ..< 4i + 4`. The context is `previous ++ x` along the tokens.
    static let convolutionKernel = MLXFast.metalKernel(
        name: "inkling2_projection_convolution",
        inputNames: ["x", "prev", "taps", "dyn", "zeros"], outputNames: ["y"],
        source: convolutionSource, header: header, ensureRowContiguous: false)

    /// The same kernel with the next state as a second output: the last
    /// token's simdgroups copy rows `L ..< L + K - 1` of `prev ++ x`.
    static let convolutionStateKernel = MLXFast.metalKernel(
        name: "inkling2_projection_convolution_state",
        inputNames: ["x", "prev", "taps", "dyn", "zeros"], outputNames: ["y", "next"],
        source: convolutionSource + #"""

            if (t == L - 1) {
                for (int i = 0; i < 4; ++i) {
                    const int c = first + i;
                    for (int r = 0; r < K - 1; ++r) {
                        const int q = L + r;
                        const T carried = q < K - 1
                            ? prev[b * prev_strides[0] + q * prev_strides[1] + c * prev_strides[2]]
                            : x[b * x_strides[0] + (q - (K - 1)) * x_strides[1] + c];
                        next[(int64_t(b) * (K - 1) + r) * (H * 128) + c] = carried;
                    }
                }
            }
            """#, header: header, ensureRowContiguous: false)

    static let convolutionSource = #"""
        const int lane = int(thread_index_in_simdgroup);
        const int row = int(threadgroup_position_in_grid.x) * 8
            + int(simdgroup_index_in_threadgroup);
        const int L = x_shape[1];
        if (row >= x_shape[0] * L * H) {
            return;
        }
        const int token = row / H;
        const int head = row - token * H;
        const int b = token / L;
        const int t = token - b * L;
        const int first = head * 128 + lane * 4;
        const uint zero = zeros[0];
        float v[4];
        for (int i = 0; i < 4; ++i) {
            const int c = first + i;
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                // Static tap j reads context row t + j; dynamic lag j row t + K - 1 - j.
                const int r = DYN ? t + K - 1 - j : t + j;
                const float window = r < K - 1
                    ? static_cast<float>(
                        prev[b * prev_strides[0] + r * prev_strides[1] + c * prev_strides[2]])
                    : static_cast<float>(
                        x[b * x_strides[0] + (r - (K - 1)) * x_strides[1] + c]);
                float tap;
                if (DYN) {
                    const float s = taps[j * taps_strides[0] + c * taps_strides[1]];
                    const float d = static_cast<float>(dyn[b * dyn_strides[0]
                        + t * dyn_strides[1] + (head * K + j) * dyn_strides[2]]);
                    tap = s + d;
                } else {
                    tap = taps[c * taps_strides[0] + j * taps_strides[1]];
                }
                const float term = window * tap;
                if (j == 0) {
                    acc = term;
                } else {
                    acc = acc + term;
                }
            }
            const float xv = static_cast<float>(x[b * x_strides[0] + t * x_strides[1] + c]);
            const float sum = acc + xv;
            const float conv = sum / as_type<float>(DIV);
            v[i] = static_cast<float>(static_cast<T>(conv));
        }
        float total = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float square = inkling2_pf_materialized(v[i] * v[i], zero);
            total = square + total;
        }
        total = simd_sum(total);
        float factor;
        if (TAIL == 0) {
            factor = inkling2_pf_clip_factor(total, as_type<float>(TV), as_type<float>(CEPS));
        } else {
            const float mean = total * as_type<float>(INV);
            const float shifted = mean + as_type<float>(TV);
            factor = metal::precise::rsqrt(shifted);
        }
        const int64_t out = int64_t(token) * (H * 128) + first;
        for (int i = 0; i < 4; ++i) {
            y[out + i] = static_cast<T>(v[i] * factor);
        }
        """#
}
