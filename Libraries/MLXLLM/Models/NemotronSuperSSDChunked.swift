import Foundation
import MLX

/// Super's prefill SSD as a chunked scan with Neural Accelerator products.
///
/// One threadgroup per (batch row, head), four simdgroups of sixteen head-dim
/// rows each. The threadgroup walks the prompt in chunks of `Q` tokens with
/// the head's float32 state `S[p, n]` (64 x 128) held in registers as the
/// accumulator fragments of its products, the way the GLM 5.3 prefill
/// recurrence keeps each core's state resident across tokens. With `cum_t`
/// the inclusive in-chunk sum of `dt * A` and `S0` the state entering the
/// chunk, each chunk computes
///
///     Wt[t, s] = (C B^T)[t, s] exp(cum_t - cum_s) dt_s      (s <= t, else 0)
///     y[t, p]  = exp(cum_t) (C S0^T)[t, p] + (Wt X)[t, p] + D x[t, p]
///     S'[p, n] = exp(cum_last) S0[p, n] + sum_s exp(cum_last - cum_s) dt_s x[s, p] B[s, n]
///
/// which is `ssmAttn`'s window algebra (the quadratic form inside a window,
/// the carried state between windows) with `Q`-token windows. Every product
/// runs as MLX's 16 x 32 x 16 NAX tensor op (`mpp::tensor_ops::matmul2d`,
/// float32 accumulation) on fragments in MLX's `BaseNAXFrag` layout; the
/// decay factors, masks, `computeDt`, the state scaling and the `D` skip are
/// float32 elementwise work. The scan is rounded to the activation dtype, as
/// `ssmAttn` rounds its window outputs, before `x * D` is added in the
/// promoted dtype of x and D; the sum is then cast to `outputDType`.
///
/// Not bit-identical to `ssmAttn` or `NemotronSuperSSD`: the summation
/// order, the window length and the operand precision of the tensor ops
/// differ (`operandType`: float32 by default, which NAX may run at reduced
/// internal precision, as MLX's TF32 matrix products do).
public enum NemotronSuperSSDChunked {
    /// How fragments reach the tensor ops.
    public enum Layout: Sendable, Equatable {
        /// Element-for-element copies in MLX's NAX fragment layout, as MLX's
        /// own NAX kernels do. Correct only on NAX GPUs (M5 and later).
        case native
        /// Each operand passes through threadgroup memory by the cooperative
        /// tensors' own coordinates. Correct on any GPU with Metal
        /// Performance Primitives; slow, for checking the kernel on GPUs
        /// without Neural Accelerators.
        case generic
    }

    /// Operand dtype of the tensor ops: `.float32` (default), `.float16` or
    /// `.bfloat16`. The state and every accumulation stay float32.
    nonisolated(unsafe) public static var operandType: DType = .float32

    /// Tokens per chunk: 32 (default) or 64. 32 does fewer products per
    /// token (the in-chunk quadratic terms halve) and holds fewer
    /// accumulators; 64 halves the chunk steps and barriers.
    nonisolated(unsafe) public static var chunkSize = 32

    /// Whether this GPU has Neural Accelerators (the `.native` layout).
    public static let naxAvailable: Bool = {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            return GPU.gemma4ExpertQMMDiagnostics().naxAvailable
        #else
            return false
        #endif
    }()

    /// Head width, state width and group geometry the kernel reads.
    static func supports(heads: Int, headDim: Int, groups: Int, stateSize: Int) -> Bool {
        headDim == 64 && stateSize > 0 && stateSize % 32 == 0 && groups > 0
            && heads % groups == 0
    }

    /// The scan of one chunk of tokens.
    ///
    /// - `xbc`: `[b, L, >= H*P + 2*G*N]`, the activated convolution output:
    ///   x at columns `0 ..< H*P` (head-major), B at `H*P ..< H*P + G*N` and C
    ///   after it (group-major), one dtype.
    /// - `dt`: the projected time step before `dtBias`, `[b, L, >= dtOffset + H]`.
    /// - `state`: float32 `[b, H, P, N]`.
    ///
    /// Returns `y` `[b, L, H, P]` in `outputDType` (`ssmAttn`'s output: the
    /// rounded scan plus `x * D`) and the float32 state after the chunk.
    static func scan(
        xbc: MLXArray, dt: MLXArray, dtOffset: Int, heads: Int, headDim: Int, groups: Int,
        stateSize: Int, aLog: MLXArray, dtBias: MLXArray, D: MLXArray, state: MLXArray,
        timeStepLimit: (Float, Float), outputDType: DType, layout: Layout? = nil,
        operandType: DType? = nil, chunkSize: Int? = nil
    ) -> (y: MLXArray, state: MLXArray) {
        let (batch, tokens) = (xbc.dim(0), xbc.dim(1))
        let width = heads * headDim
        let operand = operandType ?? Self.operandType
        let chunk = chunkSize ?? Self.chunkSize
        precondition(
            supports(heads: heads, headDim: headDim, groups: groups, stateSize: stateSize)
                && xbc.ndim == 3 && xbc.dim(2) >= width + 2 * groups * stateSize
                && dt.ndim == 3 && dt.dim(0) == batch && dt.dim(1) == tokens
                && dt.dim(2) >= dtOffset + heads
                && state.shape == [batch, heads, headDim, stateSize] && state.dtype == .float32
                && [DType.float32, .float16, .bfloat16].contains(operand)
                && (chunk == 32 || chunk == 64) && tokens > 0,
            "chunked SSD geometry")
        let native = (layout ?? (naxAvailable ? .native : .generic)) == .native
        // `ssmAttn` adds `x * D` in the promoted dtype of x and D.
        let skipDType: DType = xbc.dtype == .float32 || D.dtype == .float32 ? .float32 : xbc.dtype
        let outputs = kernel(
            [
                xbc, dt, aLog, dtBias, D, MLXArray([timeStepLimit.0, timeStepLimit.1]), state,
                MLXArray(Int32(tokens)),
            ],
            template: [
                ("T", xbc.dtype), ("PT", skipDType), ("OT", outputDType), ("OP", operand),
                ("H", heads),
                ("P", headDim), ("N", stateSize), ("G", groups), ("Q", chunk),
                ("NATIVE", native), ("XS", xbc.dim(2)), ("DS", dt.dim(2)), ("DO", dtOffset),
            ],
            grid: (128 * heads, 1, batch), threadGroup: (128, 1, 1),
            outputShapes: [[batch, tokens, heads, headDim], state.shape],
            outputDTypes: [outputDType, .float32])
        return (outputs[0], outputs[1])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_super_ssd_chunked",
        inputNames: [
            "xbc", "dtp", "a_log", "dt_bias", "d_skip", "limits", "state_in", "token_count",
        ],
        outputNames: ["y", "state_out"],
        source: source, header: header)

    static let header = """
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        // 16x16 fragment in MLX's NAX layout (BaseNAXFrag): the lane holds rows
        // fm and fm + 8, columns fn .. fn + 3; element 4 i + j is (fm + 8 i, fn + j).
        inline short2 nemotron_ssd_coord(ushort lane) {
            const short qid = lane >> 2;
            const short fm = ((qid & 4) | ((lane >> 1) & 3));
            const short fn = ((qid & 2) | (lane & 1)) * 4;
            return short2(fn, fm);
        }

        // C[16 x 32] += op(A) op(B): MLX's BaseNAXFrag::mma (output fragments
        // for columns 0..15 and 16..31). Fragments hold operands as stored: A is
        // [16 m x 16 k] ([16 k x 16 m] when TA); B is [32 n x 16 k] as two
        // fragments of 16 rows (TB) or [16 k x 32 n] as two of 16 columns.
        // NATIVE copies fragments into the cooperative tensors element for
        // element, as MLX's NAX kernels do. Otherwise every operand goes
        // through threadgroup scratch by the tensors' own coordinates.
        template <typename OP, bool NATIVE, bool TA, bool TB>
        inline void nemotron_ssd_mma(
            thread metal::vec<float, 8>& c0, thread metal::vec<float, 8>& c1,
            thread const metal::vec<OP, 8>& a,
            thread const metal::vec<OP, 8>& b0, thread const metal::vec<OP, 8>& b1,
            threadgroup float* scratch, short2 sc) {
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                16, 32, 16, TA, TB, true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
            auto ct_a = op.template get_left_input_cooperative_tensor<OP, OP, float>();
            auto ct_b = op.template get_right_input_cooperative_tensor<OP, OP, float>();
            auto ct_c = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ct_a)>,
                metal::remove_addrspace_t<decltype(ct_b)>, float>();
            if (NATIVE) {
                for (short i = 0; i < 8; i++) {
                    ct_a[i] = a[i];
                    ct_b[i] = b0[i];
                    ct_b[8 + i] = b1[i];
                    ct_c[i] = c0[i];
                    ct_c[8 + i] = c1[i];
                }
                op.run(ct_a, ct_b, ct_c);
                for (short i = 0; i < 8; i++) {
                    c0[i] = ct_c[i];
                    c1[i] = ct_c[8 + i];
                }
            } else {
                constexpr short bld = TB ? 16 : 32;
                for (short i = 0; i < 2; i++) {
                    for (short j = 0; j < 4; j++) {
                        const short r = sc.y + 8 * i;
                        const short k = sc.x + j;
                        scratch[r * 16 + k] = static_cast<float>(a[4 * i + j]);
                        scratch[256 + r * bld + k] = static_cast<float>(b0[4 * i + j]);
                        scratch[256 + (TB ? (16 + r) * bld + k : r * bld + 16 + k)] =
                            static_cast<float>(b1[4 * i + j]);
                    }
                }
                metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (short s = 0; s < ct_a.get_capacity(); s++) {
                    auto ix = ct_a.get_multidimensional_index(s);
                    ct_a[s] = static_cast<OP>(scratch[ix[1] * 16 + ix[0]]);
                }
                for (short s = 0; s < ct_b.get_capacity(); s++) {
                    auto ix = ct_b.get_multidimensional_index(s);
                    ct_b[s] = static_cast<OP>(scratch[256 + ix[1] * bld + ix[0]]);
                }
                metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (short i = 0; i < 2; i++) {
                    for (short j = 0; j < 4; j++) {
                        const short r = sc.y + 8 * i;
                        scratch[r * 32 + sc.x + j] = c0[4 * i + j];
                        scratch[r * 32 + 16 + sc.x + j] = c1[4 * i + j];
                    }
                }
                metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (short s = 0; s < ct_c.get_capacity(); s++) {
                    auto ix = ct_c.get_multidimensional_index(s);
                    ct_c[s] = scratch[ix[1] * 32 + ix[0]];
                }
                op.run(ct_a, ct_b, ct_c);
                metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (short s = 0; s < ct_c.get_capacity(); s++) {
                    auto ix = ct_c.get_multidimensional_index(s);
                    scratch[ix[1] * 32 + ix[0]] = ct_c[s];
                }
                metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (short i = 0; i < 2; i++) {
                    for (short j = 0; j < 4; j++) {
                        const short r = sc.y + 8 * i;
                        c0[4 * i + j] = scratch[r * 32 + sc.x + j];
                        c1[4 * i + j] = scratch[r * 32 + 16 + sc.x + j];
                    }
                }
                metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
            }
        }

        // Fragment of 16 token rows of a [token, ld] array: rows first + r,
        // clamped to last (a partial chunk reads finite values it never keeps),
        // columns column .. column + 15.
        template <typename OP, typename T>
        inline void nemotron_ssd_load_rows(
            thread metal::vec<OP, 8>& dst, const device T* src, int first, int last, size_t ld,
            int column, short2 sc) {
            for (short i = 0; i < 2; i++) {
                const int r = metal::min(first + sc.y + 8 * i, last);
                const device T* row = src + size_t(r) * ld + column + sc.x;
                for (short j = 0; j < 4; j++) {
                    dst[4 * i + j] = static_cast<OP>(row[j]);
                }
            }
        }
        """

    static let source = """
        constexpr int W = H * P;
        constexpr int GN = G * N;
        constexpr int NB = N / 16;
        constexpr int QB = Q / 16;
        constexpr int QJ = Q / 32;
        constexpr int WLD = Q + 4;
        static_assert(P == 64, "four simdgroups of sixteen head-dim rows");
        static_assert((Q == 32 || Q == 64) && N % 32 == 0, "whole 16 x 32 products");

        threadgroup float wt[Q * WLD];
        threadgroup float cum_s[Q];
        threadgroup float ecum_s[Q];
        threadgroup float dt_s[Q];
        threadgroup float w_s[Q];
        threadgroup float elast_s[1];
        threadgroup float scratch_all[NATIVE ? 1 : 4 * 768];

        const ushort lane = ushort(thread_index_in_simdgroup);
        const ushort sg = ushort(simdgroup_index_in_threadgroup);
        threadgroup float* scratch = scratch_all + (NATIVE ? 0 : 768 * sg);
        const int h = int(threadgroup_position_in_grid.x);
        const int b = int(threadgroup_position_in_grid.z);
        const int g = h / (H / G);
        const int tokens = token_count;
        const size_t row0 = size_t(b) * size_t(tokens);
        const short2 sc = nemotron_ssd_coord(lane);
        const int p0 = 16 * int(sg);

        typedef metal::vec<float, 8> facc;
        typedef metal::vec<OP, 8> fop;

        // State rows p0 .. p0 + 15 of this head, all N columns: S[p, n].
        facc S[NB];
        {
            const device float* src = state_in + (size_t(b) * H + h) * P * N + size_t(p0) * N;
            for (int kb = 0; kb < NB; ++kb) {
                for (short i = 0; i < 2; i++) {
                    for (short j = 0; j < 4; j++) {
                        S[kb][4 * i + j] = src[(sc.y + 8 * i) * N + 16 * kb + sc.x + j];
                    }
                }
            }
        }
        const float A = -metal::precise::exp(static_cast<float>(a_log[h]));
        const float bias = static_cast<float>(dt_bias[h]);
        const float dskip = static_cast<float>(d_skip[h]);
        const float lower = limits[0];
        const float upper = limits[1];

        float dt_next[QJ];
        for (int part = 0; part < QJ; ++part) {
            const int t = 32 * part + int(lane);
            dt_next[part] = sg == 0 && t < tokens ? static_cast<float>(dtp[(row0 + t) * DS + DO + h]) : 0.0f;
        }

        for (int t0 = 0; t0 < tokens; t0 += Q) {
            const int last = min(Q, tokens - t0) - 1;
            const size_t base = row0 + size_t(t0);
            const device T* xc = xbc + base * XS + size_t(h) * P;
            const device T* bc = xbc + base * XS + W + size_t(g) * N;
            const device T* cc = xbc + base * XS + W + GN + size_t(g) * N;

            // Phase 1: dt and the in-chunk cumulative log decay (simdgroup 0;
            // the raw dt of the next chunk is fetched here for the next pass).
            if (sg == 0) {
                float carry = 0.0f;
                for (int part = 0; part < QJ; ++part) {
                    const int t = 32 * part + lane;
                    float dtv = 0.0f;
                    if (t <= last) {
                        // computeDt: softplus(dt + bias) as MLX's LogAddExp(x, 0), then the clip.
                        const float shifted = dt_next[part] + bias;
                        float softplus_value;
                        if (metal::isnan(shifted)) {
                            softplus_value = shifted;
                        } else {
                            constexpr float inf = metal::numeric_limits<float>::infinity();
                            const float maxval = metal::max(shifted, 0.0f);
                            const float minval = metal::min(shifted, 0.0f);
                            softplus_value = (minval == -inf || maxval == inf)
                                ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
                        }
                        const float raised = metal::isnan(softplus_value)
                            ? softplus_value : (softplus_value > lower ? softplus_value : lower);
                        dtv = metal::isnan(raised) ? raised : (raised < upper ? raised : upper);
                    }
                    const float c = simd_prefix_inclusive_sum(dtv * A) + carry;
                    carry = simd_shuffle(c, ushort(31));
                    cum_s[t] = c;
                    dt_s[t] = dtv;
                }
                for (int part = 0; part < QJ; ++part) {
                    const int t = 32 * part + lane;
                    ecum_s[t] = metal::precise::exp(cum_s[t]);
                    w_s[t] = dt_s[t] * metal::precise::exp(carry - cum_s[t]);
                    dt_next[part] = t0 + Q + t < tokens
                        ? static_cast<float>(dtp[(base + Q + t) * DS + DO + h]) : 0.0f;
                }
                if (lane == 0) {
                    elast_s[0] = metal::precise::exp(carry);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2: Wt[t, s] = (C B^T)[t, s] exp(cum_t - cum_s) dt_s for s <= t,
            // in tiles of 16 t rows by 32 s columns; simdgroup sg owns tiles sg
            // and sg + 4, accumulated as two independent chains.
            {
                constexpr int TILES = QB * QJ;
                const int tiles[2] = {int(sg), int(sg) + 4};
                bool live[2];
                facc g[2][2];
                for (int u = 0; u < 2; ++u) {
                    const int ti = tiles[u] / QJ;
                    const int sj = tiles[u] % QJ;
                    live[u] = tiles[u] < TILES && 32 * sj <= 16 * ti + 15;
                    g[u][0] = facc(0.0f);
                    g[u][1] = facc(0.0f);
                }
                for (int kb = 0; kb < NB; ++kb) {
                    for (int u = 0; u < 2; ++u) {
                        if (live[u]) {
                            const int ti = tiles[u] / QJ;
                            const int sj = tiles[u] % QJ;
                            fop af, bf0, bf1;
                            nemotron_ssd_load_rows(af, cc, 16 * ti, last, XS, 16 * kb, sc);
                            nemotron_ssd_load_rows(bf0, bc, 32 * sj, last, XS, 16 * kb, sc);
                            nemotron_ssd_load_rows(bf1, bc, 32 * sj + 16, last, XS, 16 * kb, sc);
                            nemotron_ssd_mma<OP, NATIVE, false, true>(
                                g[u][0], g[u][1], af, bf0, bf1, scratch, sc);
                        }
                    }
                }
                for (int u = 0; u < 2; ++u) {
                    if (tiles[u] >= TILES) {
                        continue;
                    }
                    const int ti = tiles[u] / QJ;
                    const int sj = tiles[u] % QJ;
                    for (short i = 0; i < 2; i++) {
                        const int t = 16 * ti + sc.y + 8 * i;
                        const float ct = cum_s[t];
                        for (short j = 0; j < 4; j++) {
                            const int s0 = 32 * sj + sc.x + j;
                            const int s1 = s0 + 16;
                            wt[t * WLD + s0] = s0 <= t
                                ? g[u][0][4 * i + j] * metal::precise::exp(ct - cum_s[s0]) * dt_s[s0]
                                : 0.0f;
                            wt[t * WLD + s1] = s1 <= t
                                ? g[u][1][4 * i + j] * metal::precise::exp(ct - cum_s[s1]) * dt_s[s1]
                                : 0.0f;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3: Y^T[p, t] = exp(cum_t) (S C^T)[p, t] + (X^T Wt^T)[p, t] + D x,
            // every 32-token block of the chunk as its own accumulator chain.
            {
                facc yacc[QJ][2];
                for (int J = 0; J < QJ; ++J) {
                    yacc[J][0] = facc(0.0f);
                    yacc[J][1] = facc(0.0f);
                }
                for (int kb = 0; kb < NB; ++kb) {
                    fop sa;
                    for (short e = 0; e < 8; e++) {
                        sa[e] = static_cast<OP>(S[kb][e]);
                    }
                    for (int J = 0; J < QJ; ++J) {
                        fop bf0, bf1;
                        nemotron_ssd_load_rows(bf0, cc, 32 * J, last, XS, 16 * kb, sc);
                        nemotron_ssd_load_rows(bf1, cc, 32 * J + 16, last, XS, 16 * kb, sc);
                        nemotron_ssd_mma<OP, NATIVE, false, true>(
                            yacc[J][0], yacc[J][1], sa, bf0, bf1, scratch, sc);
                    }
                }
                for (int J = 0; J < QJ; ++J) {
                    for (short e = 0; e < 8; e++) {
                        yacc[J][0][e] *= ecum_s[32 * J + sc.x + (e & 3)];
                        yacc[J][1][e] *= ecum_s[32 * J + 16 + sc.x + (e & 3)];
                    }
                }
                for (int sb = 0; sb < QB; ++sb) {
                    fop xf;
                    // X as stored, [s, p]: the transposed left operand.
                    nemotron_ssd_load_rows(xf, xc, 16 * sb, last, XS, p0, sc);
                    for (int J = sb / 2; J < QJ; ++J) {
                        fop bf0, bf1;
                        for (short i = 0; i < 2; i++) {
                            const threadgroup float* r0 =
                                wt + (32 * J + sc.y + 8 * i) * WLD + 16 * sb + sc.x;
                            const threadgroup float* r1 = r0 + 16 * WLD;
                            for (short j = 0; j < 4; j++) {
                                bf0[4 * i + j] = static_cast<OP>(r0[j]);
                                bf1[4 * i + j] = static_cast<OP>(r1[j]);
                            }
                        }
                        nemotron_ssd_mma<OP, NATIVE, true, true>(
                            yacc[J][0], yacc[J][1], xf, bf0, bf1, scratch, sc);
                    }
                }
                for (int J = 0; J < QJ; ++J) {
                    for (short f = 0; f < 2; f++) {
                        const int tb = 32 * J + 16 * f;
                        for (short i = 0; i < 2; i++) {
                            for (short j = 0; j < 4; j++) {
                                const int t = tb + sc.x + j;
                                const int p = p0 + sc.y + 8 * i;
                                if (t <= last) {
                                    const float xv = static_cast<float>(xc[size_t(t) * XS + p]);
                                    const float xd = static_cast<float>(static_cast<PT>(xv * dskip));
                                    y[(base + t) * W + size_t(h) * P + p] = static_cast<OT>(
                                        static_cast<PT>(
                                            static_cast<float>(static_cast<T>(yacc[J][f][4 * i + j]))
                                            + xd));
                                }
                            }
                        }
                    }
                }
            }

            // Phase 4: S = exp(cum_last) S + (X w)^T B.
            const float elast = elast_s[0];
            for (int kb = 0; kb < NB; ++kb) {
                S[kb] *= elast;
            }
            for (int sb = 0; sb < QB; ++sb) {
                fop xf;
                nemotron_ssd_load_rows(xf, xc, 16 * sb, last, XS, p0, sc);
                for (short i = 0; i < 2; i++) {
                    const float ws = w_s[16 * sb + sc.y + 8 * i];
                    for (short j = 0; j < 4; j++) {
                        xf[4 * i + j] = static_cast<OP>(static_cast<float>(xf[4 * i + j]) * ws);
                    }
                }
                for (int nb = 0; nb < N / 32; ++nb) {
                    fop bf0, bf1;
                    // B as stored, [s, n]: the untransposed right operand.
                    nemotron_ssd_load_rows(bf0, bc, 16 * sb, last, XS, 32 * nb, sc);
                    nemotron_ssd_load_rows(bf1, bc, 16 * sb, last, XS, 32 * nb + 16, sc);
                    nemotron_ssd_mma<OP, NATIVE, true, false>(
                        S[2 * nb], S[2 * nb + 1], xf, bf0, bf1, scratch, sc);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        {
            device float* dst = state_out + (size_t(b) * H + h) * P * N + size_t(p0) * N;
            for (int kb = 0; kb < NB; ++kb) {
                for (short i = 0; i < 2; i++) {
                    for (short j = 0; j < 4; j++) {
                        dst[(sc.y + 8 * i) * N + 16 * kb + sc.x + j] = S[kb][4 * i + j];
                    }
                }
            }
        }
        """
}
