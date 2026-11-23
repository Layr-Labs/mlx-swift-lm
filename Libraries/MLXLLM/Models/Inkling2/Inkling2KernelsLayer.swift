// Fused decode/verify layer application for Inkling 2: the block-residual
// mixes, message gates, norms, clips, short convolutions, output gate and
// expert combine of one layer application in a handful of row-exact kernels.
// FP32 arithmetic, rounded to the model dtype at the module graph's rounding
// points (the reference's), or only at interface tensors with
// DARKBLOOM_INKLING2_REFERENCE_ROUNDING=0.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension Inkling2Kernels {
    /// Threads per row of the row kernels (mixes, gate tails, post-MLP).
    static let rowThreads = 1024
    /// Threads per threadgroup of the split gate tail (1024 / TA groups per row;
    /// `DARKBLOOM_INKLING2_TAIL_TA`: 32, 64, 128 or 256; default 128).
    static let gateTailThreads: Int = {
        let v = Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_TAIL_TA"] ?? "") ?? 128
        return [32, 64, 128, 256].contains(v) ? v : 128
    }()

    private final class LayerConstants: @unchecked Sendable {
        private let lock = NSLock()
        private var dummies: [DType: MLXArray] = [:]
        private var params: [[UInt32]: MLXArray] = [:]

        /// Eight elements of `dtype` (smaller inputs bind as `constant`).
        func dummy(_ dtype: DType) -> MLXArray {
            lock.withLock {
                if let a = dummies[dtype] { return a }
                let a = MLXArray.zeros([8], dtype: dtype)
                eval(a)
                dummies[dtype] = a
                return a
            }
        }

        /// Eight FP32 parameters (padded with zeros).
        func params(_ values: [Float]) -> MLXArray {
            let padded = values + Array(repeating: 0, count: max(0, 8 - values.count))
            let key = padded.map(\.bitPattern)
            return lock.withLock {
                if let a = params[key] { return a }
                let a = MLXArray(padded)
                eval(a)
                params[key] = a
                return a
            }
        }
    }

    private static let layerConstants = LayerConstants()

    /// Plane `i` of a `[n, ...]` array as a view (an integer subscript gathers).
    static func plane(_ x: MLXArray, _ i: Int) -> MLXArray {
        x[i ..< i + 1].squeezed(axis: 0)
    }

    /// BF16 copies of FP32 weights whose values are BF16 (the checkpoint
    /// stores router and message-gate weights in BF16; the loader widens them):
    /// reading half the bytes gives the same values. Weights that are not
    /// exactly BF16 are used as they are.
    private final class ExactHalfCopies: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [ObjectIdentifier: (source: MLXArray, copy: MLXArray)] = [:]

        func copy(_ w: MLXArray) -> MLXArray {
            guard w.dtype == .float32 else { return w }
            let id = ObjectIdentifier(w)
            if let hit = lock.withLock({ entries[id] }) { return hit.copy }
            let half = w.asType(.bfloat16)
            eval(half)
            let exact = arrayEqual(half.asType(.float32), w).item(Bool.self)
            let chosen = exact ? half : w
            lock.withLock { entries[id] = (w, chosen) }
            return chosen
        }
    }

    private static let halfCopies = ExactHalfCopies()

    /// The router weight as read by the kernels (BF16 copy when exact).
    static func halfCopiesForRouter(_ w: MLXArray) -> MLXArray { halfCopies.copy(w) }

    /// Row reductions: per-thread partials, the fixed lane ladder, then the
    /// simdgroups in index order. Depends only on the row length.
    static let rowHeader =
        header + """

            template <int NV, int NSG>
            inline void ink2_row_sums(
                thread float* v, threadgroup float* red, int lane, int sgi) {
                for (int k = 0; k < NV; k++) {
                    v[k] = ink2_simd_total(v[k]);
                }
                if (lane == 0) {
                    for (int k = 0; k < NV; k++) {
                        red[k * NSG + sgi] = v[k];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Thread k finishes value k (simdgroups in index order) and
                // shares it; `red` holds NV * NSG + NV floats.
                const int tid = sgi * 32 + lane;
                if (tid < NV) {
                    float t = 0.0f;
                    for (int s = 0; s < NSG; s++) {
                        t += red[tid * NSG + s];
                    }
                    red[NV * NSG + tid] = t;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int k = 0; k < NV; k++) {
                    v[k] = red[NV * NSG + k];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            inline float ink2_sigmoid(float x) {
                const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                return x < 0.0f ? y : 1.0f - y;
            }
            """

    // MARK: Segmented products

    /// Several products in one launch: segment `s` multiplies input
    /// `X<s>` by weight `w<s>` (`[N<s>, K]`); the output `[R, NTOT]` holds the
    /// segments side by side. Every output uses the `ink2_mma_tile` sequence
    /// with K split `segmentedSplitK` ways (K only), so a weight's outputs do
    /// not depend on the rows or on the other segments.
    private static let segmentedKernel = MLXFast.metalKernel(
        name: "inkling2_segmented_product",
        inputNames: ["x0", "x1", "x2", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7"],
        outputNames: ["out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const int ks = int(simdgroup_index_in_threadgroup);
            const int rows = x0_shape[0];
            const int groups = (rows + 63) / 64;
            const int group = int(threadgroup_position_in_grid.x) % groups;
            int tile = int(threadgroup_position_in_grid.x) / groups;
            const int sizes[8] = {N0, N1, N2, N3, N4, N5, N6, N7};
            const int sources[8] = {X0, X1, X2, X3, X4, X5, X6, X7};
            int seg = 0;
            int col = 0;
            for (; seg < NSEG - 1; seg++) {
                const int tiles = (sizes[seg] + 7) / 8;
                if (tile < tiles) {
                    break;
                }
                tile -= tiles;
                col += sizes[seg];
            }
            const int nseg = sizes[seg];
            const int first = tile * 8;
            const int nb = nseg >= 8 ? min(first, nseg - 8) : 0;
            const int valid = min(8, nseg);
            const device WT* wsel = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : seg == 3 ? w3
                : seg == 4 ? w4 : seg == 5 ? w5 : seg == 6 ? w6 : w7;
            const device T* xsel = sources[seg] == 0 ? x0 : sources[seg] == 1 ? x1 : x2;
            const int start = group * 64;
            const int count = min(64, rows - start);
            threadgroup float buf[SK * 64 * MAXRB];
            threadgroup float* mine = buf + ks * 64 * MAXRB;
            const ink2_rows_contiguous src{start};
            const device WT* row0 = wsel + size_t(nb) * K;
            const int rb = count <= 8 ? 1 : count <= 16 ? 2 : count <= 32 ? 4 : 8;
            if (rb == 1) {
                ink2_mma_tile<T, WT, 0, K, SK, 1, false, OPT, PF>(xsel, src, count, row0, nullptr, K, 0, lane, ks, mine, valid);
            } else if (rb == 2) {
                ink2_mma_tile<T, WT, 0, K, SK, 2, false, OPT, PF>(xsel, src, count, row0, nullptr, K, 0, lane, ks, mine, valid);
            } else if (rb == 4) {
                ink2_mma_tile<T, WT, 0, K, SK, 4, false, OPT, PF>(xsel, src, count, row0, nullptr, K, 0, lane, ks, mine, valid);
            } else {
                ink2_mma_tile<T, WT, 0, K, SK, 8, false, OPT, PF>(xsel, src, count, row0, nullptr, K, 0, lane, ks, mine, valid);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int cols = rb * 8;
            for (int e = int(thread_index_in_threadgroup); e < cols * 8;
                 e += int(threads_per_threadgroup.x)) {
                const int nl = e % 8;
                const int c = e / 8;
                const int n = nb + nl;
                if (c >= count || nl >= valid || n < first || n >= nseg) {
                    continue;
                }
                const int at = ((c / 8) * 8 + nl) * 8 + (c % 8);
                float v = buf[at];
                for (int s = 1; s < SK; s++) {
                    v += buf[s * 64 * MAXRB + at];
                }
                out[size_t(start + c) * NTOT + col + n] = static_cast<OT>(v);
            }
            """,
        header: header)

    /// One-row segmented products, one thread per output (`ink2_row_dot`):
    /// the same bits as `segmentedKernel`.
    private static let segmentedGemvKernel = MLXFast.metalKernel(
        name: "inkling2_segmented_gemv",
        inputNames: ["x0", "x1", "x2", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7"],
        outputNames: ["out"],
        source: """
            // SPLIT: SK adjacent lanes per output, one split-K chain each, added
            // in chain order (small grids); else one thread per output.
            const int g0 = int(thread_position_in_grid.x);
            const int sub = SPLIT ? g0 % SK : 0;
            const int gid = SPLIT ? g0 / SK : g0;
            const int rows = x0_shape[0];
            const bool valid = gid < rows * NTOT;
            const int r = valid ? gid / NTOT : 0;
            int col = valid ? gid % NTOT : 0;
            const int sizes[8] = {N0, N1, N2, N3, N4, N5, N6, N7};
            const int sources[8] = {X0, X1, X2, X3, X4, X5, X6, X7};
            int seg = 0;
            while (seg < NSEG - 1 && col >= sizes[seg]) {
                col -= sizes[seg];
                seg++;
            }
            const device WT* wsel = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : seg == 3 ? w3
                : seg == 4 ? w4 : seg == 5 ? w5 : seg == 6 ? w6 : w7;
            const device T* xsel = sources[seg] == 0 ? x0 : sources[seg] == 1 ? x1 : x2;
            if (SPLIT) {
                const float v = ink2_row_chain<T, WT, 0, K, SK, OPT>(
                    xsel + size_t(r) * K, wsel + size_t(col) * K, nullptr, sub);
                float total = v;
                for (int i = 1; i < SK; i++) {
                    total += simd_shuffle_down(v, ushort(i));
                }
                if (valid && sub == 0) {
                    out[size_t(r) * NTOT + (gid % NTOT)] = static_cast<OT>(total);
                }
            } else if (valid) {
                const float v = ink2_row_dot<T, WT, 0, K, SK, OPT>(
                    xsel + size_t(r) * K, wsel + size_t(col) * K, nullptr);
                out[size_t(r) * NTOT + (gid % NTOT)] = static_cast<OT>(v);
            }
            """,
        header: header)

    /// Input values staged in threadgroup memory, read like `ink2_load8`.
    static let stagedLoaderHeader = """

            // Input values staged in threadgroup memory (T, XPAD elements of
            // padding after every 32), widened to FP32 like ink2_load8.
            template <typename T, int XPAD>
            struct ink2_xl_staged {
                const threadgroup T* xs;
                inline ink2_v8 load8(int k0) const {
                    const threadgroup T* p = xs + k0 + (k0 / 32) * XPAD;
                    ink2_v8 r;
                    if constexpr (metal::is_same_v<T, float>) {
                        r.lo = ((const threadgroup float4*)p)[0];
                        r.hi = ((const threadgroup float4*)p)[1];
                    } else if constexpr (metal::is_same_v<T, half>) {
                        r.lo = float4(((const threadgroup half4*)p)[0]);
                        r.hi = float4(((const threadgroup half4*)p)[1]);
                    } else {
                        const uint4 u = *((const threadgroup uint4*)p);
                        r.lo = float4(
                            as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u),
                            as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
                        r.hi = float4(
                            as_type<float>(u.z << 16), as_type<float>(u.z & 0xFFFF0000u),
                            as_type<float>(u.w << 16), as_type<float>(u.w & 0xFFFF0000u));
                    }
                    return r;
                }
            };
        """

    /// One-row segmented products whose input is computed on load (same
    /// values as the kernels that would have produced it): XMODE 1, a gate
    /// tail's output `T(nw * (vals * scale))` with the scale from its 32
    /// simdgroup totals (input index = gate plane); XMODE 2, the output gate
    /// `T(att * sigmoid(gate))` with the gate at column GO of `prod`. SK lanes
    /// per output, chains added in order. Each threadgroup computes its input
    /// values once into threadgroup memory (in T: the values are T-rounded, so
    /// the chains read the same values; XPAD elements of padding after every
    /// 32 spread the lanes' reads over the banks); a threadgroup whose outputs
    /// read more than one gate plane computes them per lane on load.
    private static let transformedGemvKernel = MLXFast.metalKernel(
        name: "inkling2_segmented_xgemv",
        inputNames: [
            "vals", "parts", "nw0", "nw1", "nw2", "att", "prod", "w0", "w1", "w2", "w3", "w4",
            "w5", "w6", "w7", "prm",
        ],
        outputNames: ["out"],
        source: """
            const int g0 = int(thread_position_in_grid.x);
            const int tid = int(thread_position_in_threadgroup.x);
            const int sub = g0 % SK;
            const int gid = g0 / SK;
            const bool valid = gid < NTOT;
            int col = valid ? gid : 0;
            const int sizes[8] = {N0, N1, N2, N3, N4, N5, N6, N7};
            const int sources[8] = {X0, X1, X2, X3, X4, X5, X6, X7};
            int seg = 0;
            while (seg < NSEG - 1 && col >= sizes[seg]) {
                col -= sizes[seg];
                seg++;
            }
            const device WT* wsel = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : seg == 3 ? w3
                : seg == 4 ? w4 : seg == 5 ? w5 : seg == 6 ? w6 : w7;
            // The one gate plane every output of this threadgroup reads (-1: several).
            int tgPlane = 0;
            if (XMODE == 1) {
                const int first = int(threadgroup_position_in_grid.x) * (TG / SK);
                const int last = min(first + TG / SK, NTOT) - 1;
                int lo = -1;
                bool mixed = false;
                int start = 0;
                for (int s = 0; s < NSEG; s++) {
                    const int end = start + sizes[s];
                    if (first < end && last >= start) {
                        if (lo < 0) {
                            lo = sources[s];
                        } else if (sources[s] != lo) {
                            mixed = true;
                        }
                    }
                    start = end;
                }
                tgPlane = mixed ? -1 : lo;
            }
            const int plane = XMODE == 1 ? (tgPlane >= 0 ? tgPlane : sources[seg]) : 0;
            const device NWT* nw = plane == 0 ? nw0 : plane == 1 ? nw1 : nw2;
            float scale = 0.0f;
            if (XMODE == 1) {
                float t = 0.0f;
                for (int s = 0; s < 32; s++) {
                    t += parts[plane * 32 + s];
                }
                scale = metal::precise::rsqrt(t / float(K) + prm[0]);
            }
            constexpr int KP = K / 32 * (32 + XPAD);
            threadgroup uint4 xs4[STAGE ? (KP * sizeof(T) + 15) / 16 : 1];
            threadgroup T* xs = (threadgroup T*)xs4;
            float v;
            if (STAGE && (XMODE == 2 || tgPlane >= 0)) {
                for (int e = tid * 8; e < K; e += TG * 8) {
                    const ink2_v8 u = XMODE == 1
                        ? ink2_xl_tail<T, NWT>{vals + size_t(plane) * K, nw, scale}.load8(e)
                        : ink2_xl_gate<T>{att, prod + GO}.load8(e);
                    threadgroup T* p = xs + e + (e / 32) * XPAD;
                    p[0] = static_cast<T>(u.lo.x);
                    p[1] = static_cast<T>(u.lo.y);
                    p[2] = static_cast<T>(u.lo.z);
                    p[3] = static_cast<T>(u.lo.w);
                    p[4] = static_cast<T>(u.hi.x);
                    p[5] = static_cast<T>(u.hi.y);
                    p[6] = static_cast<T>(u.hi.z);
                    p[7] = static_cast<T>(u.hi.w);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                v = ink2_row_chain_x<WT, 0, K, SK, OPT>(
                    ink2_xl_staged<T, XPAD>{xs}, wsel + size_t(col) * K, nullptr, sub);
            } else if (XMODE == 1) {
                v = ink2_row_chain_x<WT, 0, K, SK, OPT>(
                    ink2_xl_tail<T, NWT>{vals + size_t(plane) * K, nw, scale},
                    wsel + size_t(col) * K, nullptr, sub);
            } else {
                v = ink2_row_chain_x<WT, 0, K, SK, OPT>(
                    ink2_xl_gate<T>{att, prod + GO}, wsel + size_t(col) * K, nullptr, sub);
            }
            float total = v;
            for (int i = 1; i < SK; i++) {
                total += simd_shuffle_down(v, ushort(i));
            }
            if (valid && sub == 0) {
                out[gid] = static_cast<OT>(total);
            }
            """,
        header: rowHeader + stagedLoaderHeader + """

            template <typename T, typename NWT>
            struct ink2_xl_tail {
                const device float* vals;
                const device NWT* nw;
                float scale;
                inline ink2_v8 load8(int k0) const {
                    const ink2_v8 v = ink2_load8<float>(vals + k0);
                    const ink2_v8 w = ink2_load8<NWT>(nw + k0);
                    ink2_v8 r;
                    r.lo = float4(
                        float(static_cast<T>(w.lo.x * (v.lo.x * scale))),
                        float(static_cast<T>(w.lo.y * (v.lo.y * scale))),
                        float(static_cast<T>(w.lo.z * (v.lo.z * scale))),
                        float(static_cast<T>(w.lo.w * (v.lo.w * scale))));
                    r.hi = float4(
                        float(static_cast<T>(w.hi.x * (v.hi.x * scale))),
                        float(static_cast<T>(w.hi.y * (v.hi.y * scale))),
                        float(static_cast<T>(w.hi.z * (v.hi.z * scale))),
                        float(static_cast<T>(w.hi.w * (v.hi.w * scale))));
                    return r;
                }
            };

            template <typename T>
            struct ink2_xl_gate {
                const device T* att;
                const device T* gate;
                inline ink2_v8 load8(int k0) const {
                    const ink2_v8 a = ink2_load8<T>(att + k0);
                    const ink2_v8 g = ink2_load8<T>(gate + k0);
                    ink2_v8 r;
                    r.lo = float4(
                        float(static_cast<T>(a.lo.x * ink2_sigmoid(g.lo.x))),
                        float(static_cast<T>(a.lo.y * ink2_sigmoid(g.lo.y))),
                        float(static_cast<T>(a.lo.z * ink2_sigmoid(g.lo.z))),
                        float(static_cast<T>(a.lo.w * ink2_sigmoid(g.lo.w))));
                    r.hi = float4(
                        float(static_cast<T>(a.hi.x * ink2_sigmoid(g.hi.x))),
                        float(static_cast<T>(a.hi.y * ink2_sigmoid(g.hi.y))),
                        float(static_cast<T>(a.hi.z * ink2_sigmoid(g.hi.z))),
                        float(static_cast<T>(a.hi.w * ink2_sigmoid(g.hi.w))));
                    return r;
                }
            };
            """)

    /// One-row segmented GEMVs with at most this many threads (router logits,
    /// message-gate down) launch `smallGemvThreadgroup`-thread threadgroups
    /// (`DARKBLOOM_INKLING2_SMALL_TG`: 32, 64, 128 or 256; default 32).
    static let smallGemvThreads = 8192
    static let smallGemvThreadgroup: Int = {
        let v = Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_SMALL_TG"] ?? "") ?? 32
        return [32, 64, 128, 256].contains(v) ? v : 32
    }()

    /// Threads per threadgroup of the post-MLP kernel (one thread per element;
    /// `DARKBLOOM_INKLING2_POST_TG`: 64, 128 or 256; default 256).
    static let postMLPThreadgroup: Int = {
        let v = Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_POST_TG"] ?? "") ?? 256
        return [64, 128, 256].contains(v) ? v : 256
    }()

    /// Padding (elements after every 32) of the staged inputs of
    /// `transformedGemvKernel`: `DARKBLOOM_INKLING2_XPAD` (0 or 8, default 8).
    static let stagedInputPadding: Int =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_XPAD"] == "0" ? 0 : 8
    /// `DARKBLOOM_INKLING2_XSTAGE=0` computes the inputs per lane on load
    /// instead (the same values).
    static let stagedInputs: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_XSTAGE"] != "0"

    enum TransformedInput {
        /// Gate-tail values `[nn, 1, H]`, totals `[nn, 1, 32]`, norm weights, eps.
        case gateTails(vals: MLXArray, parts: MLXArray, norms: [MLXArray], eps: Float)
        /// The attention output `[1, W]` and the products row holding the gate at `offset`.
        case outputGate(attended: MLXArray, products: MLXArray, offset: Int)
    }

    /// The one-row segmented product of `segments` with the input computed on
    /// load; nil unless one row, exact operands, and the seg GEMV site is on.
    static func transformedProduct(
        _ input: TransformedInput, _ segments: [Segment], k: Int, dtype: DType, outType: DType
    ) -> MLXArray? {
        guard let wt = segments.first?.weight.dtype,
            segments.allSatisfy({
                $0.weight.ndim == 2 && $0.weight.dim(1) == k && $0.weight.dtype == wt
            }), segments.count <= 8
        else { return nil }
        let sizes = segments.map { $0.weight.dim(0) }
        let total = sizes.reduce(0, +)
        let sk = segmentedSplitK(k, outputs: total)
        let opt = operandType(x: dtype, weight: wt, dense: true)
        guard gemvSite("seg"), opt != .float32, k % 32 == 0, 32 % sk == 0 else { return nil }
        let mode: Int
        let normType: DType
        let go: Int
        var arrays: [MLXArray]
        let dummyF = layerConstants.dummy(.float32)
        let dummyT = layerConstants.dummy(dtype)
        switch input {
        case .gateTails(let vals, let parts, let norms, _):
            guard vals.dim(1) == 1, vals.dim(2) == k else { return nil }
            mode = 1
            normType = norms[0].dtype
            go = 0
            arrays =
                [vals, parts] + (0 ..< 3).map { norms[min($0, norms.count - 1)] }
                + [dummyT, dummyT]
        case .outputGate(let attended, let products, let offset):
            guard attended.dim(0) == 1, attended.dim(1) == k, products.dim(0) == 1,
                offset % 8 == 0, offset + k <= products.dim(1)
            else { return nil }
            mode = 2
            normType = dtype
            go = offset
            arrays = [dummyF, dummyF, dummyT, dummyT, dummyT, attended, products]
        }
        let eps: Float = {
            if case .gateTails(_, _, _, let e) = input { return e }
            return 0
        }()
        var key = LaunchKey(kernel: 44, dtype: dtype, weightType: wt, a: k, b: total, c: mode)
        key.d = sizes.enumerated().reduce(0) { $0 &* 131 &+ $1.element &* 7 &+ $1.offset }
        key.e = segments.reduce(0) { $0 * 4 + $1.input } &* 31 &+ go
        key.f = (outType == .float32 ? 1 : 0) + (normType == .float32 ? 2 : 0)
        let launch = launches.launch(key) {
            var template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("WT", wt), ("OT", outType), ("NWT", normType), ("K", k),
                ("SK", sk), ("NSEG", segments.count), ("NTOT", total), ("OPT", opt),
                ("XMODE", mode), ("GO", go), ("TG", 256), ("XPAD", stagedInputPadding),
                ("STAGE", stagedInputs),
            ]
            for i in 0 ..< 8 {
                template.append(("N\(i)", i < sizes.count ? sizes[i] : 8))
                template.append(("X\(i)", i < segments.count ? segments[i].input : 0))
            }
            return transformedGemvKernel.prepare(
                template: template, grid: ((total * sk + 255) / 256 * 256, 1, 1),
                threadGroup: (256, 1, 1), outputShapes: [[1, total]], outputDTypes: [outType])
        }
        let ws = (0 ..< 8).map {
            $0 < segments.count ? segments[$0].weight : layerConstants.dummy(wt)
        }
        return launch(arrays + ws + [layerConstants.params([eps])])[0]
    }

    struct Segment {
        let input: Int
        let weight: MLXArray
    }

    /// `[R, ΣN]` in `outType`: segment products of `inputs` (each `[R, K]`, one
    /// dtype) with `[N, K]` weights of one dtype. Nil when not covered.
    static func segmentedProduct(_ inputs: [MLXArray], _ segments: [Segment], outType: DType)
        -> MLXArray?
    {
        guard let x0 = inputs.first, x0.ndim == 2, !segments.isEmpty, segments.count <= 8,
            inputs.count <= 3, supportsActivation(x0.dtype)
        else { return nil }
        let (rows, k) = (x0.dim(0), x0.dim(1))
        let wt = segments[0].weight.dtype
        guard k % 32 == 0, k >= 32, rows > 0,
            inputs.allSatisfy({ $0.shape == x0.shape && $0.dtype == x0.dtype }),
            segments.allSatisfy({
                $0.weight.ndim == 2 && $0.weight.dim(1) == k && $0.weight.dtype == wt
                    && $0.input < inputs.count && $0.weight.dim(0) > 0
            }),
            supportsActivation(wt)
        else { return nil }
        let sizes = segments.map { $0.weight.dim(0) }
        let total = sizes.reduce(0, +)
        let tiles = sizes.map { ($0 + 7) / 8 }.reduce(0, +)
        let groups = (rows + 63) / 64
        let sk = segmentedSplitK(k, outputs: total)
        var key = LaunchKey(kernel: 20, dtype: x0.dtype, weightType: wt, a: k, b: rows, c: total)
        key.d = sizes.enumerated().reduce(0) { $0 &* 131 &+ $1.element &* 7 &+ $1.offset }
        key.e = segments.reduce(0) { $0 * 4 + $1.input }
        let opt = operandType(x: x0.dtype, weight: wt, dense: true)
        // One-row segmented products: SK lanes per output (split chains);
        // small ones always, large ones on the `seg` GEMV site.
        let split =
            gemvSplitCovers(rows: rows, outputs: total, inputs: k, operand: opt)
            || (gemvSite("seg") && gemvCovers(rows: rows, outputs: total, inputs: k, operand: opt)
                && 32 % sk == 0)
        if split {
            var gemvKey = LaunchKey(
                kernel: 21, dtype: x0.dtype, weightType: wt, a: k, b: rows, c: total)
            gemvKey.d = sizes.enumerated().reduce(0) { $0 &* 131 &+ $1.element &* 7 &+ $1.offset }
            gemvKey.e = segments.reduce(0) { $0 * 4 + $1.input }
            gemvKey.f = (outType == .float32 ? 1 : 0) + (opt == .bfloat16 ? 2 : 4) + (split ? 8 : 0)
            let gemv = launches.launch(gemvKey) {
                var template: [(String, any KernelTemplateArg)] = [
                    ("T", x0.dtype), ("WT", wt), ("OT", outType), ("K", k), ("SK", sk),
                    ("NSEG", segments.count), ("NTOT", total), ("OPT", opt), ("SPLIT", split),
                ]
                for i in 0 ..< 8 {
                    template.append(("N\(i)", i < sizes.count ? sizes[i] : 8))
                    template.append(("X\(i)", i < segments.count ? segments[i].input : 0))
                }
                let threads = rows * total * (split ? sk : 1)
                // Small grids spread over more GPU cores in smaller threadgroups
                // (whole simdgroups: an output's SK lanes stay in one).
                let tg = threads <= smallGemvThreads ? smallGemvThreadgroup : 256
                return segmentedGemvKernel.prepare(
                    template: template, grid: ((threads + tg - 1) / tg * tg, 1, 1),
                    threadGroup: (tg, 1, 1), outputShapes: [[rows, total]],
                    outputDTypes: [outType])
            }
            let xs = (0 ..< 3).map { $0 < inputs.count ? inputs[$0] : x0 }
            let ws = (0 ..< 8).map {
                $0 < segments.count ? segments[$0].weight : layerConstants.dummy(wt)
            }
            return gemv(xs + ws)[0]
        }
        let pf = prefetch(threadgroups: tiles * groups)
        key.f =
            (outType == .float32 ? 1 : 0) + (opt == .float32 ? 0 : opt == .bfloat16 ? 2 : 4)
            + (pf > 1 ? 8 : 0)
        let launch = launches.launch(key) {
            var template: [(String, any KernelTemplateArg)] = [
                ("T", x0.dtype), ("WT", wt), ("OT", outType), ("K", k), ("SK", sk),
                ("NSEG", segments.count), ("NTOT", total), ("OPT", opt), ("PF", pf),
                ("MAXRB", maxRB(rows)),
            ]
            for i in 0 ..< 8 {
                template.append(("N\(i)", i < sizes.count ? sizes[i] : 8))
                template.append(("X\(i)", i < segments.count ? segments[i].input : 0))
            }
            return segmentedKernel.prepare(
                template: template, grid: (tiles * groups * 32 * sk, 1, 1),
                threadGroup: (32 * sk, 1, 1), outputShapes: [[rows, total]],
                outputDTypes: [outType])
        }
        let xs = (0 ..< 3).map { $0 < inputs.count ? inputs[$0] : x0 }
        let ws = (0 ..< 8).map {
            $0 < segments.count ? segments[$0].weight : layerConstants.dummy(wt)
        }
        return launch(xs + ws)[0]
    }
}

// MARK: Row kernels

extension Inkling2Kernels {
    /// The partial state of a mix: given (`p`), or (PRE) the post-attention
    /// convolution of `a` over its carried state `ap` plus `hid` unless the
    /// layer is a block boundary (written to `res`, next state to `an`).
    private static let mixPartial = """
        constexpr int NT = 1024;
        constexpr int NSGP = NT / 32;
        constexpr int EPT = (H + NT - 1) / NT;
        threadgroup float red[NSGP * 32];
        const int tid = int(thread_index_in_threadgroup);
        const int lane = int(thread_index_in_simdgroup);
        const int sgi = int(simdgroup_index_in_threadgroup);
        const int r = int(threadgroup_position_in_grid.x);
        const int rowsTotal = int(threadgroups_per_grid.x);
        const float eps = prm[0];
        const float rinv = 1.0f / float(H);
        float st[S][EPT];
        """

    private static let mixPartialGiven = """
        for (int i = 0; i < EPT; i++) {
            const int e = tid + NT * i;
            st[S - 1][i] = e < H ? float(p[size_t(r) * H + e]) : 0.0f;
        }
        """

    private static let mixPartialConvolved = """
        {
            const int b = r / L;
            const int t = r % L;
            const float norm = 1.0f / metal::precise::sqrt(float(KS + 1));
            for (int i = 0; i < EPT; i++) {
                const int e = tid + NT * i;
                if (e >= H) {
                    st[S - 1][i] = 0.0f;
                    continue;
                }
                float acc = 0.0f;
                float now = 0.0f;
                for (int j = 0; j < KS; j++) {
                    const int src = t - (KS - 1) + j;
                    const float xv = src >= 0 ? float(a[size_t(b * L + src) * H + e])
                        : float(ap[b * ap_strides[0] + (KS - 1 + src) * ap_strides[1] + e * ap_strides[2]]);
                    acc = fma(xv, cw[size_t(e) * KS + j], acc);
                    now = xv;
                }
                float y = (acc + now) * norm;
                if (RR) {
                    y = float(static_cast<T>(y));
                }
                const float v = BOUNDARY ? y : float(hid[size_t(r) * H + e]) + y;
                st[S - 1][i] = v;
                res[size_t(r) * H + e] = v;
                if (CAP || t == L - 1) {
                    const size_t base = size_t(CAP ? r : b) * (KS - 1);
                    for (int j = 0; j < KS - 1; j++) {
                        const int src = t - (KS - 1) + 1 + j;
                        an[(base + j) * H + e] = src >= 0
                            ? a[size_t(b * L + src) * H + e]
                            : ap[b * ap_strides[0] + (KS - 1 + src) * ap_strides[1] + e * ap_strides[2]];
                    }
                }
            }
        }
        """

    /// The attention residual over the S states (blocks then the partial):
    /// unweighted RMS norm per state, scores with the NQ learned queries
    /// times H^-1/2, softmax over states, weighted sum. NQ 0: the partial.
    private static let mixBody = """
        for (int s = 0; s < S - 1; s++) {
            const device T* bs = s == 0 ? b0 : s == 1 ? b1 : s == 2 ? b2 : s == 3 ? b3
                : s == 4 ? b4 : b5;
            for (int i = 0; i < EPT; i++) {
                const int e = tid + NT * i;
                st[s][i] = e < H ? float(bs[size_t(r) * H + e]) : 0.0f;
            }
        }
        constexpr int MQ = NQ > 0 ? NQ : 1;
        float mix[MQ][EPT];
        if (NQ == 0) {
            for (int i = 0; i < EPT; i++) {
                mix[0][i] = st[S - 1][i];
            }
        } else {
            float ss[S];
            for (int s = 0; s < S; s++) {
                ss[s] = 0.0f;
                for (int i = 0; i < EPT; i++) {
                    ss[s] = fma(st[s][i], st[s][i], ss[s]);
                }
            }
            ink2_row_sums<S, NSGP>(ss, red, lane, sgi);
            for (int s = 0; s < S; s++) {
                const float scale = metal::precise::rsqrt(ss[s] * rinv + 1e-6f);
                for (int i = 0; i < EPT; i++) {
                    st[s][i] *= scale;
                }
            }
            float sc[MQ * S];
            for (int q = 0; q < MQ; q++) {
                for (int s = 0; s < S; s++) {
                    float d = 0.0f;
                    for (int i = 0; i < EPT; i++) {
                        const int e = tid + NT * i;
                        d = e < H ? fma(st[s][i], qy[q * H + e], d) : d;
                    }
                    sc[q * S + s] = d;
                }
            }
            ink2_row_sums<MQ * S, NSGP>(sc, red, lane, sgi);
            const float qscale = metal::precise::rsqrt(float(H));
            for (int q = 0; q < MQ; q++) {
                float mx = -INFINITY;
                for (int s = 0; s < S; s++) {
                    sc[q * S + s] *= qscale;
                    mx = max(mx, sc[q * S + s]);
                }
                float total = 0.0f;
                for (int s = 0; s < S; s++) {
                    sc[q * S + s] = metal::precise::exp(sc[q * S + s] - mx);
                    total += sc[q * S + s];
                }
                for (int i = 0; i < EPT; i++) {
                    float v = 0.0f;
                    for (int s = 0; s < S; s++) {
                        v = fma(sc[q * S + s] / total, st[s][i], v);
                    }
                    mix[q][i] = v;
                }
            }
        }
        """

    /// NN weighted RMS norms (weights nw0..2) of the mix outputs, in T.
    private static let mixNorms = """
        float nss[NN];
        for (int n = 0; n < NN; n++) {
            const int q = NQ == 3 ? n : 0;
            nss[n] = 0.0f;
            for (int i = 0; i < EPT; i++) {
                nss[n] = fma(mix[q][i], mix[q][i], nss[n]);
            }
        }
        ink2_row_sums<NN, NSGP>(nss, red, lane, sgi);
        for (int n = 0; n < NN; n++) {
            const int q = NQ == 3 ? n : 0;
            const device NWT* nw = n == 0 ? nw0 : n == 1 ? nw1 : nw2;
            const float scale = metal::precise::rsqrt(nss[n] * rinv + eps);
            for (int i = 0; i < EPT; i++) {
                const int e = tid + NT * i;
                if (e < H) {
                    out[(size_t(n) * rowsTotal + r) * H + e] =
                        static_cast<T>(float(nw[e]) * (mix[q][i] * scale));
                }
            }
        }
        """

    /// Message-gate inputs of NN gates: [norm(h), norm(m)] (FP32, unweighted
    /// RMS norms, eps 1e-6) with h the gate's mix output and m message
    /// MSG0 + g; h also kept for the gate tail.
    private static let mixGatePrep = """
        float mvs[NN][EPT];
        float gss[2 * NN];
        for (int g = 0; g < NN; g++) {
            const int q = NQ == 3 ? g : 0;
            gss[2 * g] = 0.0f;
            gss[2 * g + 1] = 0.0f;
            for (int i = 0; i < EPT; i++) {
                const int e = tid + NT * i;
                mvs[g][i] = e < H ? float(m[r * m_strides[0] + (MSG0 + g) * m_strides[1] + e * m_strides[2]]) : 0.0f;
                gss[2 * g] = fma(mix[q][i], mix[q][i], gss[2 * g]);
                gss[2 * g + 1] = fma(mvs[g][i], mvs[g][i], gss[2 * g + 1]);
            }
        }
        ink2_row_sums<2 * NN, NSGP>(gss, red, lane, sgi);
        for (int g = 0; g < NN; g++) {
            const int q = NQ == 3 ? g : 0;
            const float hs = metal::precise::rsqrt(gss[2 * g] * rinv + 1e-6f);
            const float ms = metal::precise::rsqrt(gss[2 * g + 1] * rinv + 1e-6f);
            for (int i = 0; i < EPT; i++) {
                const int e = tid + NT * i;
                if (e < H) {
                    const size_t row = size_t(g) * rowsTotal + r;
                    pre[row * 2 * H + e] = static_cast<PO>(mix[q][i] * hs);
                    pre[row * 2 * H + H + e] = static_cast<PO>(mvs[g][i] * ms);
                    hg[row * H + e] = mix[q][i];
                }
            }
        }
        """

    /// Loop messages of every loop layer of a pass in one launch: for row r
    /// and layer `l` (grid y), the attention residual of the layer's NQ (4)
    /// learned queries over the previous pass's S states (blocks then the
    /// partial): unweighted RMS norms (eps 1e-6), scores times H^-1/2,
    /// softmax over states, weighted sums; FP32 `[layers, rows, NQ, H]`.
    private static let loopMessagesKernel = MLXFast.metalKernel(
        name: "inkling2_loop_messages",
        inputNames: ["p", "b0", "b1", "b2", "b3", "b4", "b5", "qall", "prm"],
        outputNames: ["out"],
        source: mixPartial + mixPartialGiven + """
            const int layer = int(threadgroup_position_in_grid.y);
            const device float* qy = qall + size_t(layer) * NQ * H;
            """ + mixBody + """
                for (int q = 0; q < NQ; q++) {
                    for (int i = 0; i < EPT; i++) {
                        const int e = tid + NT * i;
                        if (e < H) {
                            out[((size_t(layer) * rowsTotal + r) * NQ + q) * H + e] = mix[q][i];
                        }
                    }
                }
                """,
        header: rowHeader)

    /// Messages `[layers, B, L, 4, H]` (FP32) of all loop layers for one pass
    /// from the previous pass's block states and final partial; nil when the
    /// kernels are off or the shapes are not covered.
    public static func loopMessages(
        queries: MLXArray, blocks: [MLXArray], partial: MLXArray
    ) -> MLXArray? {
        guard active, fusedLayerActive, queries.ndim == 3, queries.dtype == .float32,
            partial.ndim == 3, blocks.count <= 6,
            blocks.allSatisfy({ $0.shape == partial.shape && $0.dtype == partial.dtype }),
            supportsActivation(partial.dtype)
        else { return nil }
        let (b, l, h) = (partial.dim(0), partial.dim(1), partial.dim(2))
        let (layers, nq) = (queries.dim(0), queries.dim(1))
        guard queries.dim(2) == h, nq <= 4, l <= 16 else { return nil }
        let rows = b * l
        let s = blocks.count + 1
        let dtype = partial.dtype
        var key = LaunchKey(kernel: 43, dtype: dtype, a: h, b: rows, c: layers)
        key.d = s * 10 + nq
        let launch = launches.launch(key) {
            loopMessagesKernel.prepare(
                template: [("T", dtype), ("PT", dtype), ("H", h), ("S", s), ("NQ", nq)],
                grid: (rowThreads * rows, layers, 1), threadGroup: (rowThreads, 1, 1),
                outputShapes: [[layers, rows, nq, h]], outputDTypes: [.float32])
        }
        let dummy = layerConstants.dummy(dtype)
        // The launch (keyed by rows) is shared by every B x L with the same
        // product; the batch and position axes are restored by a view.
        return launch(
            [partial.reshaped(rows, h)]
                + (0 ..< 6).map { $0 < blocks.count ? blocks[$0].reshaped(rows, h) : dummy }
                + [queries, layerConstants.params([1e-6])])[0].reshaped(layers, b, l, nq, h)
    }

    private static func mixKernel(convolved: Bool, gates: Bool) -> MLXFast.MLXFastKernel {
        let partialInputs = convolved ? ["a", "ap", "cw", "hid"] : ["p"]
        let outputs = (gates ? ["pre", "hg"] : ["out"]) + (convolved ? ["res", "an"] : [])
        return MLXFast.metalKernel(
            name: "inkling2_layer_mix_\(convolved ? "conv" : "given")_\(gates ? "gates" : "norms")",
            inputNames: partialInputs + [
                "b0", "b1", "b2", "b3", "b4", "b5", "qy", "nw0", "nw1", "nw2", "m", "prm",
            ],
            outputNames: outputs,
            source: mixPartial + (convolved ? mixPartialConvolved : mixPartialGiven) + mixBody
                + (gates ? mixGatePrep : mixNorms),
            header: rowHeader,
            // The post-attention mix reads its conv state (a column view of the
            // packed state) and the messages through strides; every other input
            // is a kernel output, the layer input or a stored parameter.
            ensureRowContiguous: !convolved)
    }

    private static let mixKernels: [[MLXFast.MLXFastKernel]] = [
        [mixKernel(convolved: false, gates: false), mixKernel(convolved: false, gates: true)],
        [mixKernel(convolved: true, gates: false), mixKernel(convolved: true, gates: true)],
    ]

    /// `s` with `old` replaced once (the kernel-source variants below).
    private static func replacingOnce(_ s: String, _ old: String, _ new: String) -> String {
        precondition(s.components(separatedBy: old).count == 2, "Inkling2 kernel source changed")
        return s.replacingOccurrences(of: old, with: new)
    }

    /// One-row mix + message-gate down projections in one launch: every
    /// threadgroup runs the whole row mix (the same code and reduction order
    /// as the mix kernel, so the same values), keeps the gate inputs of its
    /// gate in threadgroup memory, and computes OUTS down-projection outputs
    /// with SK lanes each (the split GEMV's chains, added in order).
    /// Threadgroup 0 writes the row outputs (h for the gate tails, and the
    /// residual and conv states after a convolved mix).
    private static func mixDownKernel(convolved: Bool) -> MLXFast.MLXFastKernel {
        let partialInputs = convolved ? ["a", "ap", "cw", "hid"] : ["p"]
        let outputs = ["hg", "dn"] + (convolved ? ["res", "an"] : [])
        var head = replacingOnce(
            mixPartial, "const int r = int(threadgroup_position_in_grid.x);", "const int r = 0;")
        head = replacingOnce(
            head, "const int rowsTotal = int(threadgroups_per_grid.x);",
            "const int rowsTotal = 1;\n    const bool writer = threadgroup_position_in_grid.x == 0;"
        )
        var conv = replacingOnce(
            mixPartialConvolved, "res[size_t(r) * H + e] = v;",
            "if (writer) {\n                res[size_t(r) * H + e] = v;\n            }")
        conv = replacingOnce(
            conv, "if (CAP || t == L - 1) {", "if (writer && (CAP || t == L - 1)) {")
        let body = """
            constexpr int KD = 2 * H;
            threadgroup uint4 xs4[(KD * sizeof(PO) + 15) / 16];
            threadgroup PO* xs = (threadgroup PO*)xs4;
            const int tgi = int(threadgroup_position_in_grid.x);
            const int gsel = (tgi * OUTS) / RANK;
            float mvs[NN][EPT];
            float gss[2 * NN];
            for (int g = 0; g < NN; g++) {
                const int q = NQ == 3 ? g : 0;
                gss[2 * g] = 0.0f;
                gss[2 * g + 1] = 0.0f;
                for (int i = 0; i < EPT; i++) {
                    const int e = tid + NT * i;
                    mvs[g][i] = e < H ? float(m[r * m_strides[0] + (MSG0 + g) * m_strides[1] + e * m_strides[2]]) : 0.0f;
                    gss[2 * g] = fma(mix[q][i], mix[q][i], gss[2 * g]);
                    gss[2 * g + 1] = fma(mvs[g][i], mvs[g][i], gss[2 * g + 1]);
                }
            }
            ink2_row_sums<2 * NN, NSGP>(gss, red, lane, sgi);
            for (int g = 0; g < NN; g++) {
                const int q = NQ == 3 ? g : 0;
                const float hs = metal::precise::rsqrt(gss[2 * g] * rinv + 1e-6f);
                const float ms = metal::precise::rsqrt(gss[2 * g + 1] * rinv + 1e-6f);
                for (int i = 0; i < EPT; i++) {
                    const int e = tid + NT * i;
                    if (e < H) {
                        if (writer) {
                            hg[size_t(g) * H + e] = mix[q][i];
                        }
                        if (g == gsel) {
                            xs[e] = static_cast<PO>(mix[q][i] * hs);
                            xs[H + e] = static_cast<PO>(mvs[g][i] * ms);
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < OUTS * SK) {
                const int o = tgi * OUTS + tid / SK;
                const int sub = tid % SK;
                const int ol = o - gsel * RANK;
                const device DWT* w = gsel == 0 ? dw0 : gsel == 1 ? dw1 : dw2;
                const float v = ink2_row_chain_x<DWT, 0, KD, SK, OPT>(
                    ink2_xl_staged<PO, 0>{xs}, w + size_t(ol) * KD, nullptr, sub);
                float total = v;
                for (int i = 1; i < SK; i++) {
                    total += simd_shuffle_down(v, ushort(i));
                }
                if (sub == 0) {
                    dn[o] = static_cast<DOT>(total);
                }
            }
            """
        return MLXFast.metalKernel(
            name: "inkling2_layer_mix_down_\(convolved ? "conv" : "given")",
            inputNames: partialInputs + [
                "b0", "b1", "b2", "b3", "b4", "b5", "qy", "nw0", "nw1", "nw2", "m", "prm",
                "dw0", "dw1", "dw2",
            ],
            outputNames: outputs,
            source: head + (convolved ? conv : mixPartialGiven) + mixBody + body,
            header: rowHeader + stagedLoaderHeader,
            ensureRowContiguous: !convolved)
    }

    private static let mixDownKernels = [
        mixDownKernel(convolved: false), mixDownKernel(convolved: true),
    ]

    /// Down-projection outputs per threadgroup of the fused one-row mix + down
    /// (`DARKBLOOM_INKLING2_MIXDOWN_OUTS`: 8, 16 or 32; default 16; the gate
    /// rank when it does not divide it);
    /// `DARKBLOOM_INKLING2_MIXDOWN=0` keeps the separate launches.
    static let mixDownOutputs: Int = {
        let v =
            Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_MIXDOWN_OUTS"] ?? "") ?? 16
        return [8, 16, 32].contains(v) ? v : 16
    }()
    static let mixDownEnabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_MIXDOWN"] != "0"

    /// One row: the gated row mix and its message-gate down projections in one
    /// launch. Returns `[hg, down]` (+ residual and conv states when
    /// convolved), the same values as `rowMix(gates: true)` followed by the
    /// down product; nil when not covered (then use those).
    static func rowMixDown(
        partial: MLXArray?,
        convolved: (
            a: MLXArray, state: MLXArray, weight: MLXArray,
            hidden: MLXArray, boundary: Bool, tokens: Int
        )?,
        blocks: [MLXArray], queries: MLXArray?, norms: [MLXArray], gates: [Inkling2MessageGate],
        messages: MLXArray, msg0: Int, eps: Float, dtype: DType, rows: Int, hidden: Int,
        capture: Bool = false
    ) -> [MLXArray]? {
        let nn = norms.count
        guard mixDownEnabled, rows == 1, !capture, gates.count == nn, nn <= 3,
            let dw = gates.first?.down, dw.ndim == 2, dw.dim(1) == 2 * hidden,
            gates.allSatisfy({ $0.down.shape == dw.shape && $0.down.dtype == dw.dtype })
        else { return nil }
        let rank = dw.dim(0)
        let outs = rank % mixDownOutputs == 0 ? mixDownOutputs : rank
        let gateInputType: DType = referenceRounding ? dtype : .float32
        let k = 2 * hidden
        let opt = operandType(x: gateInputType, weight: dw.dtype, dense: true)
        let sk = segmentedSplitK(k, outputs: nn * rank)
        guard rank % outs == 0, (outs * sk) % 32 == 0, outs * sk <= rowThreads,
            gemvSplitCovers(rows: 1, outputs: nn * rank, inputs: k, operand: opt)
        else { return nil }
        let s = blocks.count + 1
        let nq = queries.map { $0.ndim == 1 ? 1 : $0.dim(0) } ?? 0
        let normType = norms[0].dtype
        let partialType = partial?.dtype ?? .float32
        let downType: DType = referenceRounding ? dtype : .float32
        var key = LaunchKey(kernel: 45, dtype: dtype, weightType: dw.dtype, a: hidden, b: rank)
        key.c = s * 1000 + nq * 100 + nn * 10 + msg0
        key.d =
            (convolved == nil ? 0 : 1) + (convolved?.boundary == true ? 4 : 0)
            + (partialType == .float32 ? 8 : 0) + (normType == .float32 ? 16 : 0)
            + (messages.dtype == .float32 ? 0 : messages.dtype == .bfloat16 ? 32 : 64)
        key.e = (convolved?.tokens ?? 0) + 1000 * (convolved?.weight.dim(-1) ?? 0)
        key.f = outs
        let kernel = mixDownKernels[convolved == nil ? 0 : 1]
        let launch = launches.launch(key) {
            var template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("NWT", normType), ("H", hidden), ("S", s), ("NQ", nq),
                ("NN", nn), ("MSG0", msg0), ("MT", messages.dtype), ("PO", gateInputType),
                ("DWT", dw.dtype), ("RANK", rank), ("OUTS", outs), ("SK", sk), ("OPT", opt),
                ("DOT", downType),
            ]
            if let convolved {
                template += [
                    ("L", convolved.tokens), ("KS", convolved.weight.dim(-1)),
                    ("RR", referenceRounding),
                    ("BOUNDARY", convolved.boundary), ("CAP", false),
                ]
            } else {
                template.append(("PT", partialType))
            }
            let outputs: [[Int]] =
                [[nn, 1, hidden], [1, nn * rank]]
                + (convolved.map { [[1, hidden], [1, $0.state.dim(1), hidden]] } ?? [])
            let types: [DType] =
                [.float32, downType] + (convolved == nil ? [] : [.float32, dtype])
            return kernel.prepare(
                template: template, grid: (rowThreads * (nn * rank / outs), 1, 1),
                threadGroup: (rowThreads, 1, 1), outputShapes: outputs, outputDTypes: types)
        }
        let dummyT = layerConstants.dummy(dtype)
        var inputs: [MLXArray]
        if let convolved {
            inputs = [
                convolved.a, convolved.state, convolved.weight.reshaped(-1), convolved.hidden,
            ]
        } else {
            inputs = [partial!]
        }
        inputs += (0 ..< 6).map { $0 < blocks.count ? blocks[$0] : dummyT }
        inputs.append(queries ?? layerConstants.dummy(.float32))
        inputs += (0 ..< 3).map { norms[min($0, nn - 1)] }
        inputs.append(messages)
        inputs.append(layerConstants.params([eps]))
        inputs += (0 ..< 3).map { gates[min($0, nn - 1)].down }
        return launch(inputs)
    }

    /// The message-gate tail, `(1 - tanh(d . forget)) * h + tanh(d . gain) * m`
    /// per gate (FP32) and its weighted norm in T, spread over many
    /// threadgroups: each 1024-thread row's logical threads are split into
    /// TA-thread groups; a logical thread computes the same elements and the
    /// same partial as before, each simdgroup writes its ladder total, and
    /// `gateTailScaleKernel` sums the 32 totals in simdgroup order and scales.
    private static let gateTailPartialKernel = MLXFast.metalKernel(
        name: "inkling2_gate_tail_partial",
        inputNames: ["d", "hg", "m", "f0", "g0", "f1", "g1", "f2", "g2"],
        outputNames: ["vals", "parts"],
        source: """
            constexpr int NT = 1024;
            constexpr int P = NT / TA;
            constexpr int EPT = (H + NT - 1) / NT;
            threadgroup float dv[RANK];
            const int local = int(thread_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);
            const int part = int(threadgroup_position_in_grid.x) % P;
            const int r = int(threadgroup_position_in_grid.x) / P;
            const int g = int(threadgroup_position_in_grid.y);
            const int rowsTotal = int(threadgroups_per_grid.x) / P;
            const int tid = part * TA + local;
            const device GWT* fw = g == 0 ? f0 : g == 1 ? f1 : f2;
            const device GWT* gw = g == 0 ? g0 : g == 1 ? g1 : g2;
            for (int j = local; j < RANK; j += TA) {
                dv[j] = float(d[size_t(r) * NDT + g * RANK + j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const size_t base = (size_t(g) * rowsTotal + r) * H;
            float ss = 0.0f;
            for (int i = 0; i < EPT; i++) {
                const int e = tid + NT * i;
                if (e >= H) {
                    continue;
                }
                float fl = 0.0f;
                float gl = 0.0f;
                for (int j = 0; j < RANK; j++) {
                    fl = fma(dv[j], float(fw[size_t(e) * RANK + j]), fl);
                    gl = fma(dv[j], float(gw[size_t(e) * RANK + j]), gl);
                }
                const float h = hg[base + e];
                const float mm = float(m[(size_t(r) * 4 + MSG0 + g) * H + e]);
                const float v = (1.0f - metal::precise::tanh(fl)) * h + metal::precise::tanh(gl) * mm;
                vals[base + e] = v;
                ss = fma(v, v, ss);
            }
            ss = ink2_simd_total(ss);
            if (lane == 0) {
                parts[(size_t(g) * rowsTotal + r) * 32 + tid / 32] = ss;
            }
            """,
        header: rowHeader)

    private static let gateTailScaleKernel = MLXFast.metalKernel(
        name: "inkling2_gate_tail_scale",
        inputNames: ["vals", "parts", "nw0", "nw1", "nw2", "prm"],
        outputNames: ["out"],
        source: """
            constexpr int HP = (H + 255) / 256 * 256;
            const int gid = int(thread_position_in_grid.x);
            const int r = gid / HP;
            const int e = gid % HP;
            const int g = int(thread_position_in_grid.y);
            const int rowsTotal = int(threads_per_grid.x) / HP;
            if (e >= H) {
                return;
            }
            const device NWT* nw = g == 0 ? nw0 : g == 1 ? nw1 : nw2;
            const size_t row = size_t(g) * rowsTotal + r;
            float t = 0.0f;
            for (int s = 0; s < 32; s++) {
                t += parts[row * 32 + s];
            }
            const float scale = metal::precise::rsqrt(t / float(H) + prm[0]);
            out[row * H + e] = static_cast<T>(float(nw[e]) * (vals[row * H + e] * scale));
            """,
        header: rowHeader)

    /// Attention inputs of one layer application from its projections `p`
    /// (`[R, NTOT]`): scaled L2-clipped queries and relative states (times
    /// the log scale on full layers), K/V short convolutions (static, or
    /// dynamic with the projected per-token taps) with the key L2 clip and
    /// the value RMS norm per head, sinks, and the K/V conv states after the
    /// step. One simdgroup per (token, unit): query heads, key heads, value
    /// heads, then relative-state heads (the first also copies the sinks).
    private static let attentionPrepKernel = MLXFast.metalKernel(
        name: "inkling2_attention_prep",
        inputNames: ["p", "qs", "rsc", "kw", "vw", "kp", "vp", "pos", "prm"],
        outputNames: ["q", "k", "v", "rs", "sk", "cn"],
        source: """
            constexpr int DPL = (D + 31) / 32;
            constexpr int RPL = (DR + 31) / 32;
            constexpr int C = KVH * D;
            const int lane = int(thread_position_in_threadgroup.x);
            const int u = int(threadgroup_position_in_grid.y);
            const int r = int(threadgroup_position_in_grid.z);
            const int b = r / L;
            const int t = r % L;
            const float qkThr = prm[0];
            const float relThr = prm[1];
            const float eps = prm[2];
            float tau = 1.0f;
            if (FULL) {
                const int pr0 = pos_shape[0] == 1 ? 0 : b;
                const float ratio =
                    (float(pos[pr0 * pos_strides[0] + t * pos_strides[1]]) + 1.0f) / prm[3];
                tau = 1.0f + prm[4] * metal::precise::log(max(ratio, 1.0f));
            }
            const device T* pr = p + size_t(r) * NTOT;
            if (u < HEADS) {
                const int h = u;
                float qv[DPL];
                float ss = 0.0f;
                for (int i = 0; i < DPL; i++) {
                    const int d = lane + 32 * i;
                    qv[i] = d < D ? float(pr[QO + h * D + d]) * float(qs[h * D + d]) : 0.0f;
                    ss = fma(qv[i], qv[i], ss);
                }
                ss = simd_shuffle(ink2_simd_total(ss), 0);
                const float f = min(qkThr / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
                for (int i = 0; i < DPL; i++) {
                    const int d = lane + 32 * i;
                    if (d < D) {
                        const float c = RR ? float(static_cast<T>(qv[i] * f)) : qv[i] * f;
                        q[(size_t(r) * HEADS + h) * D + d] = static_cast<T>(c * tau);
                    }
                }
            } else if (u < HEADS + 2 * KVH) {
                const bool isKey = u < HEADS + KVH;
                const int kh = isKey ? u - HEADS : u - HEADS - KVH;
                const int off = isKey ? KO : VO;
                const int doff = isKey ? KDO : VDO;
                const device float* cw = isKey ? kw : vw;
                const device T* prev = isKey ? kp : vp;
                // States may be column views of a packed [B, KS - 1, 2C] array.
                const int64_t ps0 = isKey ? kp_strides[0] : vp_strides[0];
                const int64_t ps1 = isKey ? kp_strides[1] : vp_strides[1];
                const int64_t ps2 = isKey ? kp_strides[2] : vp_strides[2];
                float y[DPL];
                float ss = 0.0f;
                for (int i = 0; i < DPL; i++) {
                    const int d = lane + 32 * i;
                    if (d >= D) {
                        y[i] = 0.0f;
                        continue;
                    }
                    const int c = kh * D + d;
                    const float now = float(pr[off + c]);
                    float acc = 0.0f;
                    if (FULL) {
                        // Dynamic taps: static_w[lag][c] + projected[head][lag];
                        // lag 0 is the current token.
                        for (int lag = 0; lag < KS; lag++) {
                            const int src = t - lag;
                            const float xv = src >= 0
                                ? float(p[size_t(b * L + src) * NTOT + off + c])
                                : float(prev[b * ps0 + (KS - 1 + src) * ps1 + c * ps2]);
                            const float tap = cw[size_t(lag) * C + c] + float(pr[doff + kh * KS + lag]);
                            acc = fma(xv, tap, acc);
                        }
                        y[i] = (acc + now) / metal::precise::sqrt(float(2 * KS + 1));
                        if (RR) {
                            y[i] = float(static_cast<T>(y[i]));
                        }
                    } else {
                        // Static taps [C, KS]; index KS - 1 weights the current token.
                        for (int j = 0; j < KS; j++) {
                            const int src = t - (KS - 1) + j;
                            const float xv = src >= 0
                                ? float(p[size_t(b * L + src) * NTOT + off + c])
                                : float(prev[b * ps0 + (KS - 1 + src) * ps1 + c * ps2]);
                            acc = fma(xv, cw[size_t(c) * KS + j], acc);
                        }
                        y[i] = (acc + now) / metal::precise::sqrt(float(KS + 1));
                        if (RR) {
                            y[i] = float(static_cast<T>(y[i]));
                        }
                    }
                    ss = fma(y[i], y[i], ss);
                    // Packed key || value state rows: after the row's last token, or
                    // (CAP) after every token of a verify window.
                    if (CAP || t == L - 1) {
                        const size_t base = size_t(CAP ? r : b) * (KS - 1);
                        for (int j = 0; j < KS - 1; j++) {
                            const int src = t - (KS - 1) + 1 + j;
                            const T val = src >= 0 ? p[size_t(b * L + src) * NTOT + off + c]
                                : prev[b * ps0 + (KS - 1 + src) * ps1 + c * ps2];
                            cn[(base + j) * (2 * C) + (isKey ? 0 : C) + c] = val;
                        }
                    }
                }
                ss = simd_shuffle(ink2_simd_total(ss), 0);
                const float f = isKey
                    ? min(qkThr / (metal::precise::sqrt(ss) + 1e-6f), 1.0f)
                    : metal::precise::rsqrt(ss / float(D) + eps);
                for (int i = 0; i < DPL; i++) {
                    const int d = lane + 32 * i;
                    if (d < D) {
                        const size_t at = (size_t(r) * KVH + kh) * D + d;
                        if (isKey) {
                            k[at] = static_cast<T>(y[i] * f);
                        } else {
                            v[at] = static_cast<T>(y[i] * f);
                        }
                    }
                }
            } else {
                // One unit per relative-state head; the first also copies the sinks.
                const int h = u - HEADS - 2 * KVH;
                {
                    float rv[RPL];
                    float ss = 0.0f;
                    for (int i = 0; i < RPL; i++) {
                        const int d = lane + 32 * i;
                        rv[i] = d < DR ? float(pr[RO + h * DR + d]) * float(rsc[h * DR + d]) : 0.0f;
                        ss = fma(rv[i], rv[i], ss);
                    }
                    ss = simd_shuffle(ink2_simd_total(ss), 0);
                    const float f = min(relThr / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
                    for (int i = 0; i < RPL; i++) {
                        const int d = lane + 32 * i;
                        if (d < DR) {
                            const float c = RR ? float(static_cast<T>(rv[i] * f)) : rv[i] * f;
                            rs[(size_t(r) * HEADS + h) * DR + d] = static_cast<T>(c * tau);
                        }
                    }
                }
                if (h == 0) {
                    for (int j = lane; j < HEADS; j += 32) {
                        sk[size_t(r) * HEADS + j] = pr[SO + j];
                    }
                }
            }
            """,
        header: rowHeader,
        // Strided reads for the conv states and positions; every other input is
        // a kernel output or a stored parameter (row-contiguous).
        ensureRowContiguous: false)

    /// `T(attended * sigmoid(gate))` (FP32 product), the output projection's input.
    private static let outputGateKernel = MLXFast.metalKernel(
        name: "inkling2_output_gate",
        inputNames: ["att", "p"],
        outputNames: ["out"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const int rows = att_shape[0];
            if (int(gid) >= rows * W) {
                return;
            }
            const int r = int(gid) / W;
            const int j = int(gid) % W;
            const float g = float(p[size_t(r) * NTOT + GO + j]);
            out[gid] = static_cast<T>(float(att[gid]) * ink2_sigmoid(g));
            """,
        header: rowHeader)

    /// The expert combine (FP32, ascending expert order) of this token and the
    /// three before it in its row, rounded to T (the post-MLP conv input), the
    /// post-MLP conv over them, the residual add, and the conv state after the
    /// step. Recomputing earlier tokens' combines keeps one launch for windows.
    private static let postMLPKernel = MLXFast.metalKernel(
        name: "inkling2_post_mlp",
        inputNames: ["y", "idx", "wts", "mp", "cw", "res", "an"],
        outputNames: ["out", "rt", "sn"],
        source: """
            // One thread per (row, element): no reductions, so the work spreads
            // over the whole grid (H padded to whole threadgroups).
            constexpr int HP = (H + 255) / 256 * 256;
            constexpr int NT = HP;
            constexpr int EPT = 1;
            const int tid = int(thread_position_in_grid.x) % HP;
            const int r = int(thread_position_in_grid.x) / HP;
            const int b = r / L;
            const int t = r % L;
            const float norm = 1.0f / metal::precise::sqrt(float(KS + 1));
            int order[KS][TOPK];
            bool live[KS];
            for (int j = 0; j < KS; j++) {
                const int src = t - (KS - 1) + j;
                live[j] = src >= 0;
                const int row = b * L + max(src, 0);
                uint ids[TOPK];
                for (int u = 0; u < TOPK; u++) {
                    ids[u] = idx[row * TOPK + u];
                }
                for (int s = 0; s < TOPK; s++) {
                    int rank = 0;
                    for (int u = 0; u < TOPK; u++) {
                        rank += (ids[u] < ids[s] || (ids[u] == ids[s] && u < s)) ? 1 : 0;
                    }
                    order[j][rank] = s;
                }
            }
            for (int i = 0; i < EPT; i++) {
                const int e = tid + NT * i;
                if (e >= H) {
                    continue;
                }
                float acc = 0.0f;
                float now = 0.0f;
                for (int j = 0; j < KS; j++) {
                    const int src = t - (KS - 1) + j;
                    float xv;
                    if (live[j]) {
                        const int row = b * L + src;
                        float sum = 0.0f;
                        for (int k = 0; k < TOPK; k++) {
                            const int s = order[j][k];
                            sum += float(y[size_t(row * TOPK + s) * H + e]) * wts[row * TOPK + s];
                        }
                        xv = float(static_cast<T>(sum));
                    } else {
                        xv = float(mp[b * mp_strides[0] + (KS - 1 + src) * mp_strides[1] + e * mp_strides[2]]);
                    }
                    acc = fma(xv, cw[size_t(e) * KS + j], acc);
                    now = xv;
                }
                float yv = (acc + now) * norm;
                if (RR) {
                    yv = float(static_cast<T>(yv));
                }
                out[size_t(r) * H + e] = static_cast<T>(res[size_t(r) * H + e] + yv);
                rt[size_t(r) * H + e] = static_cast<T>(now);
                // Packed postAttention || postMLP state rows (the last KS - 1 of
                // this token's context): after the row's last token, or (CAP)
                // after every token of a verify window. The first half copies
                // the post-attention rows `an` (same row layout).
                if (CAP || t == L - 1) {
                    const size_t base = size_t(CAP ? r : b) * (KS - 1);
                    for (int j = 0; j < KS - 1; j++) {
                        const int jj = j + 1;
                        const int src = t - (KS - 1) + jj;
                        T val;
                        if (src >= 0) {
                            const int row = b * L + src;
                            float sum = 0.0f;
                            for (int k = 0; k < TOPK; k++) {
                                const int s = order[jj][k];
                                sum += float(y[size_t(row * TOPK + s) * H + e]) * wts[row * TOPK + s];
                            }
                            val = static_cast<T>(sum);
                        } else {
                            val = mp[b * mp_strides[0] + (KS - 1 + src) * mp_strides[1]
                                + e * mp_strides[2]];
                        }
                        sn[(base + j) * (2 * H) + e] = an[(base + j) * H + e];
                        sn[(base + j) * (2 * H) + H + e] = val;
                    }
                }
            }
            """,
        header: rowHeader,
        // Strided reads for the conv state; every other input is a kernel output
        // or a stored parameter (row-contiguous).
        ensureRowContiguous: false)
}

// MARK: Layer application

extension Inkling2Kernels {
    /// `DARKBLOOM_INKLING2_FUSED_LAYER=0` keeps the module graph for the layer
    /// application (the projection and expert kernels still apply).
    public static let fusedLayerEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_FUSED_LAYER"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Round to the model dtype at the module graph's (and the reference's)
    /// rounding points: the default. `DARKBLOOM_INKLING2_REFERENCE_ROUNDING=0`
    /// keeps FP32 intermediates between interface tensors instead.
    static let referenceRounding: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_REFERENCE_ROUNDING"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Overrides `fusedLayerEnabled` for the current task (tests and benchmarks).
    @TaskLocal public static var fusedLayerOverride: Bool? = nil
    static var fusedLayerActive: Bool { active && (fusedLayerOverride ?? fusedLayerEnabled) }

    private static func plainLinear(_ layer: Linear, dtype: DType) -> MLXArray? {
        guard ObjectIdentifier(type(of: layer)) == ObjectIdentifier(Linear.self), layer.bias == nil,
            layer.weight.dtype == dtype, layer.weight.ndim == 2
        else { return nil }
        return layer.weight
    }

    /// The row mix kernel: block-residual mix over `blocks ++ [partial]`, then
    /// `norms.count` weighted RMS norms (T) or, with `gates`, message-gate
    /// inputs (FP32) for messages `msg0 ..< msg0 + norms.count`.
    static func rowMix(
        partial: MLXArray?,
        convolved: (
            a: MLXArray, state: MLXArray, weight: MLXArray,
            hidden: MLXArray, boundary: Bool, tokens: Int
        )?,
        blocks: [MLXArray], queries: MLXArray?, norms: [MLXArray], gates: Bool,
        messages: MLXArray?, msg0: Int, eps: Float, dtype: DType, rows: Int, hidden: Int,
        capture: Bool = false
    ) -> [MLXArray] {
        let s = blocks.count + 1
        let nq = queries.map { $0.ndim == 1 ? 1 : $0.dim(0) } ?? 0
        let nn = norms.count
        let normType = norms[0].dtype
        let partialType = partial?.dtype ?? .float32
        let messageType = messages?.dtype ?? .float32
        // Gate inputs: the model dtype at reference-rounding points (the
        // module graph rounds them there), else FP32.
        let gateInputType: DType = referenceRounding ? dtype : .float32
        let kernel = mixKernels[convolved == nil ? 0 : 1][gates ? 1 : 0]
        var key = LaunchKey(kernel: 30, dtype: dtype, weightType: normType, a: hidden, b: rows)
        key.c = s * 1000 + nq * 100 + nn * 10 + msg0
        key.d =
            (convolved == nil ? 0 : 1) + (gates ? 2 : 0) + (convolved?.boundary == true ? 4 : 0)
            + (partialType == .float32 ? 8 : 0) + (capture ? 16 : 0)
        // Every template value and output shape is a function of the key (a
        // launch is shared by all callers, on any thread).
        key.e = (convolved?.tokens ?? 0) + 1000 * (convolved?.weight.dim(-1) ?? 0)
        key.f = messageType == .float32 ? 0 : messageType == .bfloat16 ? 1 : 2
        let outputs: [[Int]] =
            (gates ? [[nn, rows, 2 * hidden], [nn, rows, hidden]] : [[nn, rows, hidden]])
            + (convolved.map {
                [[rows, hidden], [capture ? rows : rows / $0.tokens, $0.state.dim(1), hidden]]
            } ?? [])
        let outputTypes: [DType] =
            (gates ? [gateInputType, .float32] : [dtype])
            + (convolved == nil ? [] : [.float32, dtype])
        let launch = launches.launch(key) {
            var template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("NWT", normType), ("H", hidden), ("S", s), ("NQ", nq),
                ("NN", nn), ("MSG0", msg0), ("MT", messageType), ("PO", gateInputType),
            ]
            if let convolved {
                template += [
                    ("L", convolved.tokens), ("KS", convolved.weight.dim(-1)),
                    ("RR", referenceRounding),
                    ("BOUNDARY", convolved.boundary), ("CAP", capture),
                ]
            } else {
                template.append(("PT", partialType))
            }
            return kernel.prepare(
                template: template, grid: (rowThreads * rows, 1, 1),
                threadGroup: (rowThreads, 1, 1), outputShapes: outputs, outputDTypes: outputTypes)
        }
        let dummyT = layerConstants.dummy(dtype)
        var inputs: [MLXArray]
        if let convolved {
            inputs = [
                convolved.a, convolved.state, convolved.weight.reshaped(-1), convolved.hidden,
            ]
        } else {
            inputs = [partial!]
        }
        inputs += (0 ..< 6).map { $0 < blocks.count ? blocks[$0] : dummyT }
        inputs.append(queries ?? layerConstants.dummy(.float32))
        inputs += (0 ..< 3).map { norms[min($0, nn - 1)] }
        inputs.append(messages ?? layerConstants.dummy(.float32))
        inputs.append(layerConstants.params([eps]))
        return launch(inputs)
    }

    /// Message gates after the mix: the down projections (one launch), then
    /// `(1 - tanh(f)) h + tanh(g) m` and the weighted norms (one launch).
    /// Values `[nn, rows, H]` (FP32) and per-simdgroup totals `[nn, rows, 32]`
    /// of the gate tails, before the scale (`messageGates` = these + scale).
    private static func messageGatesPartial(
        _ gates: [Inkling2MessageGate], pre: MLXArray?, down precomputed: MLXArray? = nil,
        hg: MLXArray, messages: MLXArray,
        msg0: Int, norms: [MLXArray], dtype: DType, rows: Int, hidden: Int
    ) -> [MLXArray]? {
        let nn = gates.count
        let rank = gates[0].down.dim(0)
        let fg = (0 ..< 3).flatMap { i -> [MLXArray] in
            let g = gates[min(i, nn - 1)]
            return [halfCopies.copy(g.forget), halfCopies.copy(g.gain)]
        }
        let gateType = fg[0].dtype
        guard fg.allSatisfy({ $0.dtype == gateType }),
            let down = precomputed
                ?? pre.flatMap({ pre in
                    segmentedProduct(
                        (0 ..< nn).map { plane(pre, $0) },
                        gates.enumerated().map {
                            Segment(input: $0.offset, weight: $0.element.down)
                        },
                        outType: referenceRounding ? dtype : .float32)
                })
        else { return nil }
        // Values and per-simdgroup partial sums over many threadgroups, then
        // the scale (bit-identical to the one-threadgroup-per-row gate tail).
        let ta = gateTailThreads
        var key = LaunchKey(
            kernel: 31, dtype: dtype, weightType: norms[0].dtype, a: hidden, b: rows)
        key.c = nn * 100 + msg0
        key.d = rank
        key.e = gateType == .float32 ? 1 : 0
        key.f = (messages.dtype == .float32 ? 0 : messages.dtype == .bfloat16 ? 1 : 2) + 4 * ta
        let partialLaunch = launches.launch(key) {
            gateTailPartialKernel.prepare(
                template: [
                    ("T", dtype), ("GWT", gateType), ("H", hidden), ("RANK", rank),
                    ("NDT", nn * rank), ("MSG0", msg0), ("MT", messages.dtype), ("TA", ta),
                ],
                grid: (1024 * rows, nn, 1), threadGroup: (ta, 1, 1),
                outputShapes: [[nn, rows, hidden], [nn, rows, 32]],
                outputDTypes: [.float32, .float32])
        }
        return partialLaunch([down, hg, messages] + fg)
    }

    private static func messageGates(
        _ gates: [Inkling2MessageGate], pre: MLXArray?, down: MLXArray? = nil, hg: MLXArray,
        messages: MLXArray,
        msg0: Int, norms: [MLXArray], eps: Float, dtype: DType, rows: Int, hidden: Int
    ) -> MLXArray? {
        guard
            let partial = messageGatesPartial(
                gates, pre: pre, down: down, hg: hg, messages: messages, msg0: msg0, norms: norms,
                dtype: dtype, rows: rows, hidden: hidden)
        else { return nil }
        return gateTailScale(
            partial, norms: norms, eps: eps, dtype: dtype, rows: rows, hidden: hidden)
    }

    /// The gate tails' weighted norms in T from their values and totals.
    private static func gateTailScale(
        _ partial: [MLXArray], norms: [MLXArray], eps: Float, dtype: DType, rows: Int, hidden: Int
    ) -> MLXArray {
        let nn = partial[0].dim(0)
        var scaleKey = LaunchKey(
            kernel: 39, dtype: dtype, weightType: norms[0].dtype, a: hidden, b: rows)
        scaleKey.c = nn
        let scaleLaunch = launches.launch(scaleKey) {
            gateTailScaleKernel.prepare(
                template: [("T", dtype), ("NWT", norms[0].dtype), ("H", hidden)],
                grid: ((hidden + 255) / 256 * 256 * rows, nn, 1), threadGroup: (256, 1, 1),
                outputShapes: [[nn, rows, hidden]], outputDTypes: [dtype])
        }
        return scaleLaunch(
            partial + (0 ..< 3).map { norms[min($0, nn - 1)] } + [layerConstants.params([eps])])[0]
    }

    /// The decode/verify application of `layer` through the fused kernels;
    /// nil (before touching `state`) when the layer or step is not covered.
    static func decoderLayer(
        _ layer: Inkling2DecoderLayer, _ hidden: MLXArray, blocks: [MLXArray],
        messages: MLXArray?, slot: Int, state: Inkling2StepState
    ) -> MLXArray? {
        guard fusedLayerActive, hidden.ndim == 3 else { return nil }
        let dtype = hidden.dtype
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let rows = b * l
        let attention = layer.attention
        let isFull = layer.isFull
        let controllers = messages == nil ? nil : layer.controllers
        let mlpBlocks = layer.isBlockBoundary ? blocks + [hidden] : blocks
        guard supportsActivation(dtype), h % 32 == 0, rows > 0, blocks.count <= 6,
            mlpBlocks.count <= 6,
            let wq = plainLinear(attention.queryProjection, dtype: dtype),
            let wr = plainLinear(attention.relativeLinear, dtype: dtype),
            let wsink = plainLinear(attention.sinkProjection, dtype: dtype),
            let wgate = plainLinear(attention.outputGate, dtype: dtype),
            let wk = plainLinear(attention.keyProjection, dtype: dtype),
            let wv = plainLinear(attention.valueProjection, dtype: dtype),
            let wo = plainLinear(attention.outputProjection, dtype: dtype),
            let gateUp = weights(of: layer.mlp.experts.gateUp),
            let downBank = weights(of: layer.mlp.experts.down),
            covers(
                rows: rows, topK: layer.mlp.router.topK, hidden: h, dtype: dtype,
                gateUp: gateUp, down: downBank),
            layer.postAttentionConvolution.weight.dtype == .float32,
            layer.postMLPConvolution.weight.dtype == .float32,
            messages == nil || messages!.shape == [b, l, 4, h]
        else { return nil }
        let heads = attention.heads
        let kvHeads = attention.kvHeads
        let headDim = attention.headDim
        let rel = wr.dim(0) / heads
        let kvWidth = kvHeads * headDim
        var dynamicWeights: (MLXArray, MLXArray, MLXArray, MLXArray)? = nil
        let keyStatic: MLXArray
        let valueStatic: MLXArray
        if isFull {
            guard let kc = attention.keyConvolution as? Inkling2DynamicShortConvolution,
                let vc = attention.valueConvolution as? Inkling2DynamicShortConvolution,
                kc.weightProjection.dtype == dtype, vc.weightProjection.dtype == dtype,
                kc.staticWeight.dtype == .float32, vc.staticWeight.dtype == .float32
            else { return nil }
            dynamicWeights = (
                kc.weightProjection, vc.weightProjection, kc.staticWeight, vc.staticWeight
            )
            keyStatic = kc.staticWeight
            valueStatic = vc.staticWeight
        } else {
            guard let kc = attention.keyConvolution as? Inkling2ShortConvolution,
                let vc = attention.valueConvolution as? Inkling2ShortConvolution,
                kc.weight.dtype == .float32, vc.weight.dtype == .float32
            else { return nil }
            keyStatic = kc.weight.reshaped(-1)
            valueStatic = vc.weight.reshaped(-1)
        }
        let convKernel = layer.postAttentionConvolution.kernel
        let attentionKernels = [attention.keyConvolution, attention.valueConvolution].map {
            ($0 as? Inkling2ShortConvolution)?.kernel ?? ($0 as? Inkling2DynamicShortConvolution)?
                .kernel ?? 0
        }
        guard attentionKernels.allSatisfy({ $0 == convKernel }),
            layer.postMLPConvolution.kernel == convKernel, convKernel >= 2
        else { return nil }
        let eps = attention.eps
        let flat = hidden.reshaped(rows, h)
        let msg = messages.map { $0.reshaped(rows, 4, h) }

        // 1. Attention inputs: mix, then norms or message gates.
        let inputNorms: [MLXArray] =
            isFull
            ? [layer.queryNorm!.weight, layer.keyNorm!.weight, layer.valueNorm!.weight]
            : [layer.inputNorm!.weight]
        let gateModules = controllers.map { [$0.query, $0.key, $0.value] }
        let mixQueries = layer.attentionInput.map { $0.query.asType(.float32) }
        let normed: MLXArray
        var tailInputs: (partial: [MLXArray], norms: [MLXArray])?
        if let gateModules, let msg {
            let tailNorms = isFull ? inputNorms : Array(repeating: inputNorms[0], count: 3)
            let mixNorms = Array(repeating: inputNorms[0], count: 3)
            // One row: the mix and the down projections in one launch.
            let fusedDown = rowMixDown(
                partial: flat, convolved: nil, blocks: blocks, queries: mixQueries,
                norms: mixNorms, gates: gateModules, messages: msg, msg0: 0, eps: eps,
                dtype: dtype, rows: rows, hidden: h)
            let prep =
                fusedDown
                ?? rowMix(
                    partial: flat, convolved: nil, blocks: blocks, queries: mixQueries,
                    norms: mixNorms, gates: true, messages: msg,
                    msg0: 0, eps: eps, dtype: dtype, rows: rows, hidden: h)
            guard
                let partial = messageGatesPartial(
                    gateModules, pre: fusedDown == nil ? prep[0] : nil,
                    down: fusedDown?[1], hg: fusedDown?[0] ?? prep[1], messages: msg, msg0: 0,
                    norms: tailNorms, dtype: dtype, rows: rows, hidden: h)
            else { return nil }
            if rows == 1 {
                // One row: the projections compute the gate tails' outputs on load.
                tailInputs = (partial, tailNorms)
                normed = partial[0]
            } else {
                normed = gateTailScale(
                    partial, norms: tailNorms, eps: eps, dtype: dtype, rows: rows, hidden: h)
            }
        } else {
            normed =
                rowMix(
                    partial: flat, convolved: nil, blocks: blocks, queries: mixQueries,
                    norms: inputNorms, gates: false, messages: nil, msg0: 0, eps: eps, dtype: dtype,
                    rows: rows, hidden: h)[0]
        }
        let distinct = normed.dim(0) > 1
        Inkling2Diagnostics.section?("inputs", [normed])

        // 2. Projections, one launch.
        var segments = [
            Segment(input: 0, weight: wq), Segment(input: 0, weight: wr),
            Segment(input: 0, weight: wsink), Segment(input: 0, weight: wgate),
            Segment(input: distinct ? 1 : 0, weight: wk),
        ]
        if let dynamicWeights {
            segments.append(Segment(input: distinct ? 1 : 0, weight: dynamicWeights.0))
        }
        segments.append(Segment(input: distinct ? 2 : 0, weight: wv))
        if let dynamicWeights {
            segments.append(Segment(input: distinct ? 2 : 0, weight: dynamicWeights.1))
        }
        var transformed: MLXArray?
        if let tail = tailInputs {
            transformed = transformedProduct(
                .gateTails(
                    vals: tail.partial[0], parts: tail.partial[1], norms: tail.norms, eps: eps),
                segments, k: h, dtype: dtype, outType: dtype)
        }
        let viaScale =
            tailInputs.map {
                gateTailScale(
                    $0.partial, norms: $0.norms, eps: eps, dtype: dtype, rows: rows, hidden: h)
            } ?? normed
        let inputs = distinct ? (0 ..< 3).map { plane(viaScale, $0) } : [plane(viaScale, 0)]
        guard let products = transformed ?? segmentedProduct(inputs, segments, outType: dtype)
        else {
            preconditionFailure("Inkling2 fused layer: projection shapes changed after the check")
        }
        let qo = 0
        let ro = wq.dim(0)
        let so = ro + wr.dim(0)
        let go = so + wsink.dim(0)
        let ko = go + wgate.dim(0)
        let kdo = ko + wk.dim(0)
        let vo = dynamicWeights == nil ? kdo : kdo + dynamicWeights!.0.dim(0)
        let vdo = vo + wv.dim(0)
        let ntot = products.dim(1)

        // 3. Attention inputs and K/V conv states. The engine stages packed
        // states (key || value, postAttention || postMLP) once per layer
        // application; verify windows that capture positions get every
        // position's state from the kernels.
        let engine = state as? Inkling2EngineStepState
        let capture = engine?.capturesPositions ?? false
        let keyPrevious =
            state.convState(slot: slot, .key)
            ?? MLXArray.zeros([b, convKernel - 1, kvWidth], dtype: dtype)
        let valuePrevious =
            state.convState(slot: slot, .value)
            ?? MLXArray.zeros([b, convKernel - 1, kvWidth], dtype: dtype)
        let positions = state.positions
        var prepKey = LaunchKey(
            kernel: 32, dtype: dtype, weightType: attention.queryClip.scale.dtype)
        prepKey.a = ntot
        prepKey.b = rows
        prepKey.c = l
        prepKey.d = (isFull ? 1 : 0) + 2 * ro + 1_000_000 * heads + (capture ? 1 << 40 : 0)
        prepKey.e = ko * 10_000 + vo
        prepKey.f = rel * 1000 + headDim + 1_000_000 * kvHeads + 1_000_000_000 * convKernel
        let prepLaunch = launches.launch(prepKey) {
            attentionPrepKernel.prepare(
                template: [
                    ("T", dtype), ("HEADS", heads), ("KVH", kvHeads), ("D", headDim), ("DR", rel),
                    ("NTOT", ntot), ("QO", qo), ("RO", ro), ("SO", so), ("KO", ko), ("VO", vo),
                    ("KDO", isFull ? kdo : 0), ("VDO", isFull ? vdo : 0), ("L", l),
                    ("FULL", isFull), ("KS", convKernel), ("RR", referenceRounding),
                    ("CAP", capture),
                ],
                grid: (32, 2 * heads + 2 * kvHeads, rows), threadGroup: (32, 1, 1),
                outputShapes: [
                    [rows, heads, headDim], [rows, kvHeads, headDim], [rows, kvHeads, headDim],
                    [rows, heads, rel], [rows, heads],
                    [capture ? rows : b, convKernel - 1, 2 * kvWidth],
                ],
                outputDTypes: [dtype, dtype, dtype, dtype, dtype, dtype])
        }
        let prepared = prepLaunch([
            products, attention.queryClip.scale, attention.relativeClip.scale, keyStatic,
            valueStatic, keyPrevious, valuePrevious, positions.asType(.int32),
            layerConstants.params([
                attention.qkThreshold, attention.relativeClip.threshold, eps,
                Float(attention.logScalingFloor), attention.logScalingAlpha,
            ]),
        ])
        if engine == nil {
            state.setConvState(
                slot: slot, .key, prepared[5][.ellipsis, ..<kvWidth],
                input: products[0..., ko ..< ko + kvWidth].reshaped(b, l, kvWidth))
            state.setConvState(
                slot: slot, .value, prepared[5][.ellipsis, kvWidth...],
                input: products[0..., vo ..< vo + kvWidth].reshaped(b, l, kvWidth))
        }
        let relativeStates = prepared[3].reshaped(b, l, heads, rel)
        let attended = state.attend(
            slot: slot,
            Inkling2AttentionInputs(
                queries: prepared[0].reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3),
                keys: prepared[1].reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3),
                values: prepared[2].reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3),
                relativeLogits: Inkling2RowExact.matmul(
                    relativeStates, transposedWeight: attention.relativeProjection.proj.T
                ).transposed(0, 2, 1, 3),
                relativeStates: relativeStates,
                relativeProjection: attention.relativeProjection.proj,
                sinks: prepared[4].reshaped(b, l, heads).transposed(0, 2, 1),
                positions: state.positions, window: attention.window, scale: attention.scale))

        // 4. Output gate and projection.
        let width = heads * headDim
        var gateKey = LaunchKey(kernel: 33, dtype: dtype, a: width, b: rows, c: ntot)
        gateKey.d = go
        let gateLaunch = launches.launch(gateKey) {
            outputGateKernel.prepare(
                template: [("T", dtype), ("W", width), ("NTOT", ntot), ("GO", go)],
                grid: ((rows * width + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, width]], outputDTypes: [dtype])
        }
        // One row: the output projection computes the gate on load.
        let oneRowProjected =
            rows == 1
            ? transformedProduct(
                .outputGate(
                    attended: attended.reshaped(rows, width), products: products, offset: go),
                [Segment(input: 0, weight: wo)], k: width, dtype: dtype, outType: dtype)
            : nil
        let projected: MLXArray
        if let oneRowProjected {
            projected = oneRowProjected
        } else {
            let gated = gateLaunch([attended.reshaped(rows, width), products])[0]
            // The output projection as a one-segment product (split K eight ways).
            guard
                let segmented = segmentedProduct(
                    [gated], [Segment(input: 0, weight: wo)], outType: dtype)
            else {
                preconditionFailure("Inkling2 fused layer: output projection not covered")
            }
            projected = segmented
        }
        Inkling2Diagnostics.section?("attention", [projected])

        // 5. Post-attention conv, residual, MLP-input mix, norm or message gate.
        let postAttentionPrevious =
            state.convState(slot: slot, .postAttention)
            ?? MLXArray.zeros([b, convKernel - 1, h], dtype: dtype)
        let mlpGate = controllers?.mlp
        let postConvolved = (
            a: projected, state: postAttentionPrevious,
            weight: layer.postAttentionConvolution.weight, hidden: flat,
            boundary: layer.isBlockBoundary, tokens: l
        )
        let mlpQuery = layer.mlpInput.query.asType(.float32)
        // One row with a message gate: the mix and the down projection in one launch.
        let postDown = mlpGate.flatMap { gate in
            msg.flatMap { msg in
                rowMixDown(
                    partial: nil, convolved: postConvolved, blocks: mlpBlocks, queries: mlpQuery,
                    norms: [layer.mlpNorm.weight], gates: [gate], messages: msg, msg0: 3,
                    eps: layer.mlpNorm.eps, dtype: dtype, rows: rows, hidden: h,
                    capture: capture)
            }
        }
        let post =
            postDown
            ?? rowMix(
                partial: nil, convolved: postConvolved,
                blocks: mlpBlocks, queries: mlpQuery,
                norms: [layer.mlpNorm.weight], gates: mlpGate != nil, messages: msg, msg0: 3,
                eps: layer.mlpNorm.eps, dtype: dtype, rows: rows, hidden: h, capture: capture)
        let residual = post[post.count - 2]
        let postAttentionState = post[post.count - 1]
        if engine == nil {
            state.setConvState(
                slot: slot, .postAttention, postAttentionState, input: projected.reshaped(b, l, h))
        }
        var mlpInput = plane(post[0], 0)
        if let mlpGate, let msg {
            guard
                let tail = messageGates(
                    [mlpGate], pre: postDown == nil ? post[0] : nil,
                    down: postDown?[1], hg: postDown?[0] ?? post[1], messages: msg, msg0: 3,
                    norms: [layer.mlpNorm.weight], eps: layer.mlpNorm.eps, dtype: dtype,
                    rows: rows, hidden: h)
            else { preconditionFailure("Inkling2 fused layer: message gate not covered") }
            mlpInput = plane(tail, 0)
        }
        Inkling2Diagnostics.section?("post-attention", [mlpInput, residual])

        // 6. Routed experts: FP32 router logits (MMA), select, gate/up, down.
        let router = layer.mlp.router
        let indices: MLXArray
        let routeWeights: MLXArray
        let selection: MLXArray
        if rows == 1, let fused = routerSelectOneRow(router, mlpInput) {
            (indices, routeWeights, selection) = fused
        } else {
            guard
                let logits = segmentedProduct(
                    [mlpInput], [Segment(input: 0, weight: halfCopies.copy(router.weight))],
                    outType: .float32)
            else { preconditionFailure("Inkling2 fused layer: router not covered") }
            (indices, routeWeights, selection) = select(router, logits: logits)
        }
        Inkling2Diagnostics.routerIndices?(indices)
        Inkling2Diagnostics.routerSelection?(selection)
        Inkling2Diagnostics.section?("router", [indices, routeWeights])
        guard
            let expertOutputs = expertsUpDown(
                mlpInput, indices: indices, gateUp: gateUp, down: downBank,
                gateCap: layer.mlp.gateSoftcap, upCap: layer.mlp.upSoftcap)
        else { preconditionFailure("Inkling2 fused layer: experts not covered") }

        // 7. Combine, post-MLP conv, residual.
        let postMLPPrevious =
            state.convState(slot: slot, .postMLP)
            ?? MLXArray.zeros([b, convKernel - 1, h], dtype: dtype)
        let topK = router.topK
        var postKey = LaunchKey(kernel: 34, dtype: dtype, a: h, b: rows, c: l)
        postKey.d = topK * 100 + convKernel
        postKey.e = capture ? 1 : 0
        let postLaunch = launches.launch(postKey) {
            postMLPKernel.prepare(
                template: [
                    ("T", dtype), ("H", h), ("TOPK", topK), ("L", l), ("KS", convKernel),
                    ("RR", referenceRounding), ("CAP", capture),
                ],
                grid: ((h + 255) / 256 * 256 * rows, 1, 1),
                threadGroup: (postMLPThreadgroup, 1, 1),
                outputShapes: [
                    [rows, h], [rows, h], [capture ? rows : b, convKernel - 1, 2 * h],
                ],
                outputDTypes: [dtype, dtype, dtype])
        }
        let finished = postLaunch([
            expertOutputs, indices, routeWeights, postMLPPrevious,
            layer.postMLPConvolution.weight.reshaped(-1), residual, postAttentionState,
        ])
        Inkling2Diagnostics.section?("experts", [finished[1]])
        if let engine {
            engine.stagePackedConvStates(slot: slot, conv: prepared[5], ssm: finished[2], length: l)
        } else {
            state.setConvState(
                slot: slot, .postMLP, finished[2][.ellipsis, h...],
                input: finished[1].reshaped(b, l, h))
        }
        let output = finished[0].reshaped(b, l, h)
        Inkling2Diagnostics.section?("post-mlp", [output, finished[2]])
        return output
    }
}
