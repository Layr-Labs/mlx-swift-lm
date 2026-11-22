// Inkling 2 layer stages between the matmuls as single kernels.
// Reference: transformers PR 51050 `modeling_inkling2_loop.py` (Apache-2.0).
import Foundation
import MLX
import MLXNN

/// Chains of a layer application that are not matmuls, one kernel each: the
/// post-attention and post-MLP short convolutions with the residual add (and
/// cast) that follows them, and a message gate's FP32 products with their
/// tail. Each replaces casts, concatenates and compiled elementwise kernels
/// that pass over `[tokens, hidden]` arrays several times.
///
/// Bit-identical to those chains: the kernels perform the same FP32
/// operations on the same operands in the same order and round to the model
/// dtype where the chains do. The gate products keep the fold of
/// `Inkling2ExactFP32` (from zero, one `fma` per term, k ascending).
enum Inkling2StageFusion {
    /// `DARKBLOOM_INKLING2_STAGE_FUSION=0` keeps the chains.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_STAGE_FUSION"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `enabled` for the current task.
    @TaskLocal static var override: Bool? = nil

    /// Off with the plain op-by-op graphs (`DARKBLOOM_INKLING2_FUSED=0`).
    static var active: Bool { (override ?? enabled) && Inkling2Fused.active }

    static let floatTypes: Set<DType> = [.bfloat16, .float16, .float32]

    /// What follows the convolution output `y` (in the input's dtype).
    enum Tail {
        /// `y.asType(.float32)`.
        case widened
        /// `other.asType(.float32) + y.asType(.float32)`, FP32.
        case added(MLXArray)
        /// `(residual + y.asType(.float32)).asType(dtype)`; `residual` FP32.
        case residual(MLXArray, DType)
    }

    /// `module(x, state:)` then `tail`, for `x` `[B, L, C]`: the tail's output
    /// `[B, L, C]` and the next state `[B, kernel - 1, C]`, which is stored as
    /// the layer stores it. Nil, with nothing stored, when not covered.
    static func convolution(
        _ module: Inkling2ShortConvolution, _ x: MLXArray, tail: Tail, slot: Int,
        kind: Inkling2ConvKind, state: Inkling2StepState
    ) -> (output: MLXArray, next: MLXArray)? {
        guard active, x.ndim == 3, floatTypes.contains(x.dtype) else { return nil }
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        let kernel = module.kernel
        guard kernel >= 2, b * l > 0, module.weight.size == c * kernel else { return nil }

        let other: MLXArray
        let outputType: DType
        let tailCode: Int
        switch tail {
        case .widened:
            (other, outputType, tailCode) = (constants.floats, .float32, 0)
        case .added(let value):
            guard value.shape == x.shape, floatTypes.contains(value.dtype) else { return nil }
            (other, outputType, tailCode) = (value, .float32, 1)
        case .residual(let value, let dtype):
            guard value.shape == x.shape, value.dtype == .float32, floatTypes.contains(dtype)
            else { return nil }
            (other, outputType, tailCode) = (value, dtype, 2)
        }

        let stored = state.convState(slot: slot, kind)
        let previous = stored ?? MLXArray.zeros([b, kernel - 1, c], dtype: x.dtype)
        guard previous.shape == [b, kernel - 1, c], previous.dtype == x.dtype else { return nil }

        // `[C, kernel]`: index `kernel - 1` weights the current token.
        let taps = module.weight.reshaped(c, kernel).asType(.float32)
        // The constant the chain traces: `sqrt(Float(kernel + 1))`.
        let divisor = sqrt(Float(kernel + 1))
        let template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("O", other.dtype), ("Y", outputType), ("K", kernel),
            ("TAIL", tailCode), ("DIV", bits(divisor)),
        ]
        let key = Key(
            kind: 0, types: [x.dtype, other.dtype, outputType],
            values: [kernel, b, l, c, tailCode])
        let launch = launches.launch(key) {
            convolutionKernel.prepare(
                template: template,
                grid: (((c + 3) / 4 + 63) / 64 * 64, (b * l + 3) / 4 * 4, 1),
                threadGroup: (64, 4, 1),
                outputShapes: [[b, l, c], [b, kernel - 1, c]],
                outputDTypes: [outputType, x.dtype])
        }
        let out = launch([x, previous, taps, other])
        state.setConvState(slot: slot, kind, out[1], input: x)
        return (out[0], out[1])
    }

    /// A message gate after its rank-`r` input `g`: the FP32 products with
    /// `fused` = `[forget; gain]` (`[2 * hidden, r]`), then
    /// `(1 - tanh(f)) * h + tanh(g) * m`. `g` `[..., r]`, `h` and `m`
    /// `[..., hidden]`, all FP32. Returns `h`'s shape in FP32, or nil when the
    /// chain would not take `Inkling2ExactFP32`'s kernel and the compiled tail.
    static func messageGate(_ g: MLXArray, fused: MLXArray, h: MLXArray, m: MLXArray)
        -> MLXArray?
    {
        guard active, !Inkling2RowExact.active, Inkling2ExactFP32.active, fused.ndim == 2,
            g.dtype == .float32, fused.dtype == .float32, h.dtype == .float32,
            m.dtype == .float32, h.shape == m.shape
        else { return nil }
        let (rank, hidden) = (fused.dim(1), fused.dim(0) / 2)
        let rows = h.size / max(h.dim(-1), 1)
        guard fused.dim(0) == 2 * hidden, fused.dim(0) >= 16, h.dim(-1) == hidden,
            g.dim(-1) == rank, rank > 0, rows > 0, g.size == rows * rank
        else { return nil }
        let key = Key(kind: 1, types: [.float32], values: [rows, hidden, rank])
        let launch = launches.launch(key) {
            messageGateKernel.prepare(
                grid: (((hidden + 63) / 64) * 16, ((rows + 63) / 64) * 16, 1),
                threadGroup: (16, 16, 1), outputShapes: [[rows, hidden]],
                outputDTypes: [.float32])
        }
        let y = launch([
            g.reshaped(rows, rank), fused, h.reshaped(rows, hidden), m.reshaped(rows, hidden),
        ])[0]
        return y.reshaped(h.shape)
    }

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
        let floats: MLXArray = {
            let a = MLXArray([Float](repeating: 0, count: 8))
            eval(a)
            return a
        }()
    }

    static let constants = Constants()

    // MARK: - Kernels

    /// Static short convolution of `prev ++ x` (taps `[C, K]`) rounded to `T`,
    /// then the tail: TAIL 0 the value in FP32, TAIL 1 `float(other) + value`,
    /// TAIL 2 `other + value` rounded to `Y`. One thread per token and four
    /// channels. The threads of each row's last token also write the next
    /// state: rows `L ..< L + K - 1` of `prev ++ x`.
    static let convolutionKernel = MLXFast.metalKernel(
        name: "inkling2_stage_convolution",
        inputNames: ["x", "prev", "taps", "other"], outputNames: ["y", "next"],
        source: #"""
            const int L = x_shape[1];
            const int C = x_shape[2];
            const int token = int(thread_position_in_grid.y);
            const int c0 = int(thread_position_in_grid.x) * 4;
            if (token >= x_shape[0] * L || c0 >= C) {
                return;
            }
            const int b = token / L;
            const int t = token - b * L;
            const int64_t xb = int64_t(b) * x_strides[0];
            const int64_t pb = int64_t(b) * prev_strides[0];
            for (int i = 0; i < 4 && c0 + i < C; ++i) {
                const int c = c0 + i;
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) {
                    // Tap j reads row t + j of the context.
                    const int r = t + j;
                    const float window = r < K - 1
                        ? static_cast<float>(prev[pb + r * prev_strides[1] + c * prev_strides[2]])
                        : static_cast<float>(
                            x[xb + int64_t(r - (K - 1)) * x_strides[1] + c * x_strides[2]]);
                    const float tap = taps[c * taps_strides[0] + j * taps_strides[1]];
                    const float term = window * tap;
                    if (j == 0) {
                        acc = term;
                    } else {
                        acc = acc + term;
                    }
                }
                const float xv = static_cast<float>(
                    x[xb + int64_t(t) * x_strides[1] + c * x_strides[2]]);
                const float sum = acc + xv;
                const float conv = sum / as_type<float>(DIV);
                float value = static_cast<float>(static_cast<T>(conv));
                if (TAIL != 0) {
                    const float ov = static_cast<float>(other[int64_t(b) * other_strides[0]
                        + int64_t(t) * other_strides[1] + c * other_strides[2]]);
                    value = ov + value;
                }
                y[int64_t(token) * C + c] = static_cast<Y>(value);
                if (t == L - 1) {
                    for (int r = 0; r < K - 1; ++r) {
                        const int q = L + r;
                        const T carried = q < K - 1
                            ? prev[pb + q * prev_strides[1] + c * prev_strides[2]]
                            : x[xb + int64_t(q - (K - 1)) * x_strides[1] + c * x_strides[2]];
                        next[(int64_t(b) * (K - 1) + r) * C + c] = carried;
                    }
                }
            }
            """#,
        ensureRowContiguous: false)

    /// `Inkling2ExactFP32`'s tile with two folds per output (forget row `n`
    /// and gain row `N + n` of `w`), then the tail on the thread's 4 x 4
    /// outputs with MLX's `tanh`.
    static let messageGateKernel = MLXFast.metalKernel(
        name: "inkling2_stage_message_gate", inputNames: ["x", "w", "h", "m"],
        outputNames: ["y"],
        source: #"""
            constexpr int BM = 64;
            constexpr int BN = 64;
            constexpr int BK = 16;
            threadgroup float xs[BK][BM];
            threadgroup float wf[BK][BN];
            threadgroup float wg[BK][BN];
            const int M = x_shape[0];
            const int K = x_shape[1];
            const int N = w_shape[0] / 2;
            const int tx = thread_position_in_threadgroup.x;
            const int ty = thread_position_in_threadgroup.y;
            const int tid = ty * 16 + tx;
            const int m0 = threadgroup_position_in_grid.y * BM;
            const int n0 = threadgroup_position_in_grid.x * BN;
            float af[4][4];
            float ag[4][4];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    af[i][j] = 0.0f;
                    ag[i][j] = 0.0f;
                }
            }
            for (int k0 = 0; k0 < K; k0 += BK) {
                for (int e = tid; e < BM * BK; e += 256) {
                    const int r = e / BK;
                    const int c = e % BK;
                    const int gm = m0 + r;
                    const int gk = k0 + c;
                    xs[c][r] = (gm < M && gk < K) ? x[size_t(gm) * K + gk] : 0.0f;
                }
                for (int e = tid; e < BN * BK; e += 256) {
                    const int r = e / BK;
                    const int c = e % BK;
                    const int gn = n0 + r;
                    const int gk = k0 + c;
                    wf[c][r] = (gn < N && gk < K) ? w[size_t(gn) * K + gk] : 0.0f;
                    wg[c][r] = (gn < N && gk < K) ? w[size_t(N + gn) * K + gk] : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const int kmax = min(BK, K - k0);
                for (int kk = 0; kk < kmax; ++kk) {
                    float a[4];
                    float bf[4];
                    float bg[4];
                    for (int i = 0; i < 4; ++i) {
                        a[i] = xs[kk][ty * 4 + i];
                        bf[i] = wf[kk][tx * 4 + i];
                        bg[i] = wg[kk][tx * 4 + i];
                    }
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            af[i][j] = fma(a[i], bf[j], af[i][j]);
                            ag[i][j] = fma(a[i], bg[j], ag[i][j]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int i = 0; i < 4; ++i) {
                const int gm = m0 + ty * 4 + i;
                for (int j = 0; j < 4; ++j) {
                    const int gn = n0 + tx * 4 + j;
                    if (gm < M && gn < N) {
                        const float hv = h[size_t(gm) * N + gn];
                        const float mv = m[size_t(gm) * N + gn];
                        const float keep = 1.0f - metal::precise::tanh(af[i][j]);
                        const float gain = metal::precise::tanh(ag[i][j]);
                        const float left = keep * hv;
                        const float right = gain * mv;
                        y[size_t(gm) * N + gn] = left + right;
                    }
                }
            }
            """#,
        ensureRowContiguous: true)
}
