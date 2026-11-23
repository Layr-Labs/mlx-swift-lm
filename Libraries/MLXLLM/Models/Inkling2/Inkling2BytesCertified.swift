// Certified-exact greedy token selection over the unembedding.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Picks each row's greedy token without the full `[R, vocab]` product, and proves it equal to the
/// token of the full product (`softcap(rowProduct(h, W))`, first index of the maximum).
///
/// 1. A symmetric int6 proxy of `W` gives every vocabulary row an interval for the engine's FP32
///    accumulator: `|a_j - q_j·h| <= c_j ‖h‖` with `c_j = ‖w_j - q_j‖ + γ(‖q_j‖ + ‖w_j‖)` (rounding
///    of either sum in any order). The interval's bf16 ends pass through monotone envelopes of the
///    softcap table, because the GPU's tanh is not monotone everywhere.
/// 2. Rows whose upper end reaches the best lower end are re-bounded with an int8 proxy.
/// 3. The rows still in reach get exact logits from the engine's own row-exact product (the
///    `rowProduct` MMA tile, reading rows through an index list), and the first maximum among them
///    is the token. Every row left out is below that maximum by proof.
///
/// Work after stage 1 runs over device-side lists of any length up to the vocabulary, so there is
/// no host round trip and no fallback: a non-finite or huge hidden row makes every row a candidate
/// and stage 3 becomes the full product. The proxies are derived in memory from the verified
/// weight; nothing is written to disk.
public final class Inkling2CertifiedHead: @unchecked Sendable {
    public let rows: Int
    public let columns: Int
    let weight: MLXArray
    let codes6: MLXArray
    let scales6: MLXArray
    let bound6: MLXArray
    let codes8: MLXArray
    let scales8: MLXArray
    let bound8: MLXArray
    /// The engine's softcap of every bf16 logit, and its monotone envelopes.
    public let softcapEnvelope: Envelope
    let softcapTable: MLXArray
    let lowerEnvelope: MLXArray
    let upperEnvelope: MLXArray
    /// Number of bf16 grid steps where the GPU softcap decreases (8 on M5: |x| in 236 ... 249).
    public let softcapDecreasingSteps: Int

    public static let group = 64
    /// Exact-stage threadgroups; tiles beyond them loop (grid stride).
    static let exactThreadgroups = 64
    /// Stage-2 simdgroups; list rows beyond them loop.
    static let listSimdgroups = 1024
    /// Stage-2 threadgroups of the MMA proxy; list tiles beyond them loop.
    static let listTileGroups = 256

    public struct Selection {
        /// `[R]` int32 tokens.
        public let tokens: MLXArray
        /// `[R]` int32: 1 where the row's full logits would hold a NaN (the engine's non-finite rule).
        public let nonFinite: MLXArray
        /// `[2]` uint32: rows re-bounded by the int8 proxy, rows given exact logits (union over R).
        public let candidateCounts: MLXArray
    }

    /// A map of bf16 values tabulated on all 65,536 bf16 bit patterns, with monotone envelopes, so
    /// bounds `l <= x <= u` on a bf16 value become bounds `lower[l] <= map(x) <= upper[u]` even
    /// where the map (as the GPU evaluates it) is not monotone.
    public struct Envelope: @unchecked Sendable {
        /// `[65536]` float32: the map at every bf16 bit pattern.
        public let table: MLXArray
        /// `[65536]` float32, non-decreasing along the bf16 order (±infinity included): the
        /// minimum of the map over every non-NaN value at or above the pattern's value. NaN at
        /// NaN patterns.
        public let lower: MLXArray
        /// `[65536]` float32, non-decreasing: the maximum of the map over every non-NaN value at or
        /// below the pattern's value. NaN at NaN patterns.
        public let upper: MLXArray
        /// Steps along the bf16 order where the map decreases.
        public let decreasingSteps: Int

        /// Tabulates `transform`, which receives all 65,536 bf16 values as one bf16 `[65536]`
        /// array and must apply the consumer's own ops (same dtypes, same order), so the table
        /// holds exactly what the consumer computes. The result is stored as float32 (exact for
        /// bf16, f16 and f32 results).
        public init(_ transform: (MLXArray) -> MLXArray) {
            let grid = MLXArray(Array(UInt16(0) ... UInt16.max), [65536]).view(dtype: .bfloat16)
            let table = transform(grid).asType(.float32)
            eval(table)
            let f = table.asArray(Float.self)
            let values = grid.asType(.float32).asArray(Float.self)
            // Every non-NaN value, ±infinity included: a bound that rounds to ±infinity still
            // gets a finite or infinite envelope value, never NaN.
            let ordered = (0 ..< 65536).filter { !values[$0].isNaN }
                .sorted { values[$0] < values[$1] || (values[$0] == values[$1] && $0 < $1) }
            var down = [Float](repeating: .nan, count: 65536)
            var up = [Float](repeating: .nan, count: 65536)
            var running = -Float.infinity
            var decreasing = 0
            for (i, index) in ordered.enumerated() {
                if i > 0, f[index] < f[ordered[i - 1]] { decreasing += 1 }
                running = max(running, f[index])
                up[index] = running
            }
            running = .infinity
            for index in ordered.reversed() {
                running = min(running, f[index])
                down[index] = running
            }
            self.table = table
            self.lower = MLXArray(down, [65536])
            self.upper = MLXArray(up, [65536])
            self.decreasingSteps = decreasing
            eval(self.lower, self.upper)
        }

        /// The identity on bf16 values: bounds of the bf16-rounded product itself.
        public static func identity() -> Envelope { Envelope { $0 } }
    }

    /// `weight`: the unembedding `[N, K]` bf16 the engine multiplies (K a multiple of 256).
    /// `softcap`: the engine's FP32 logit transform, evaluated here once on every bf16 value.
    public init(weight: MLXArray, softcap: (MLXArray) -> MLXArray) throws {
        guard weight.ndim == 2, weight.dtype == .bfloat16, weight.dim(1) % 256 == 0 else {
            throw Inkling2Bytes.Failure.unsupported(
                "certified head needs bf16 [N, K], K % 256 == 0")
        }
        rows = weight.dim(0)
        columns = weight.dim(1)
        self.weight = weight
        let bits = weight.view(dtype: .uint16).asArray(UInt16.self)
        let built = bits.withUnsafeBufferPointer {
            Self.buildProxies($0, rows: weight.dim(0), columns: weight.dim(1))
        }
        (codes6, scales6, bound6) = built.p6
        (codes8, scales8, bound8) = built.p8

        let envelope = Envelope { softcap($0.asType(.float32)) }
        softcapEnvelope = envelope
        softcapTable = envelope.table
        lowerEnvelope = envelope.lower
        upperEnvelope = envelope.upper
        softcapDecreasingSteps = envelope.decreasingSteps
        eval(codes6, scales6, bound6, codes8, scales8, bound8, lowerEnvelope, upperEnvelope)
    }

    // MARK: Proxies

    /// Symmetric per-64 proxies with FP16 scales and their per-row bound constants (rounded up).
    static func buildProxies(_ bits: UnsafeBufferPointer<UInt16>, rows n: Int, columns k: Int) -> (
        p6: (MLXArray, MLXArray, MLXArray), p8: (MLXArray, MLXArray, MLXArray)
    ) {
        let groups = k / group
        var codes6 = [UInt16](repeating: 0, count: n * k / 8 * 3)
        var codes8 = [Int8](repeating: 0, count: n * k)
        var scales6 = [Float16](repeating: 1, count: n * groups)
        var scales8 = [Float16](repeating: 1, count: n * groups)
        var bound6 = [Float](repeating: 0, count: n)
        var bound8 = [Float](repeating: 0, count: n)
        // Proxy sums: own FMA chains (RNE). Exact sums: MMA accumulation, budgeted at 4 ulp-halves.
        let terms = Double(k + 64)
        let gammaProxy = terms * 0x1p-22 / (1 - terms * 0x1p-22)
        let gammaExact = terms * 0x1p-22 / (1 - terms * 0x1p-22)
        codes6.withUnsafeMutableBufferPointer { c6 in
            codes8.withUnsafeMutableBufferPointer { c8 in
                scales6.withUnsafeMutableBufferPointer { s6 in
                    scales8.withUnsafeMutableBufferPointer { s8 in
                        bound6.withUnsafeMutableBufferPointer { b6 in
                            bound8.withUnsafeMutableBufferPointer { b8 in
                                let pass = ProxyPass(
                                    bits: bits, rows: n, columns: k, gammaProxy: gammaProxy,
                                    gammaExact: gammaExact, codes6: c6, codes8: c8, scales6: s6,
                                    scales8: s8, bound6: b6, bound8: b8)
                                DispatchQueue.concurrentPerform(iterations: (n + 255) / 256) {
                                    pass.run(rowsFrom: $0 * 256, to: min(n, $0 * 256 + 256))
                                }
                            }
                        }
                    }
                }
            }
        }
        return (
            (
                MLXArray(codes6, [n, k / 8 * 3]), MLXArray(scales6, [n, groups]),
                MLXArray(bound6, [n])
            ),
            (MLXArray(codes8, [n, k]), MLXArray(scales8, [n, groups]), MLXArray(bound8, [n]))
        )
    }

    private struct ProxyPass: @unchecked Sendable {
        let bits: UnsafeBufferPointer<UInt16>
        let rows: Int
        let columns: Int
        let gammaProxy: Double
        let gammaExact: Double
        let codes6: UnsafeMutableBufferPointer<UInt16>
        let codes8: UnsafeMutableBufferPointer<Int8>
        let scales6: UnsafeMutableBufferPointer<Float16>
        let scales8: UnsafeMutableBufferPointer<Float16>
        let bound6: UnsafeMutableBufferPointer<Float>
        let bound8: UnsafeMutableBufferPointer<Float>

        func run(rowsFrom start: Int, to end: Int) {
            let k = columns
            var w = [Double](repeating: 0, count: k)
            var q6 = [Int](repeating: 0, count: k)
            for row in start ..< end {
                var wn = 0.0
                for i in 0 ..< k {
                    let value = Double(Float(bitPattern: UInt32(bits[row * k + i]) << 16))
                    w[i] = value
                    wn += value * value
                }
                var err6 = 0.0
                var qn6 = 0.0
                var err8 = 0.0
                var qn8 = 0.0
                for g in 0 ..< k / group {
                    var amax = 0.0
                    for i in g * group ..< (g + 1) * group { amax = max(amax, abs(w[i])) }
                    for (qmax, isSix) in [(31.0, true), (127.0, false)] {
                        var s = Float16(Float(amax / qmax))
                        if !(s > 0) || !s.isFinite { s = 1 }
                        let sd = Double(s)
                        for i in g * group ..< (g + 1) * group {
                            let q = max(-qmax, min(qmax, (w[i] / sd).rounded()))
                            let d = q * sd
                            if isSix {
                                q6[i] = Int(q)
                                err6 += (w[i] - d) * (w[i] - d)
                                qn6 += d * d
                            } else {
                                codes8[row * k + i] = Int8(q)
                                err8 += (w[i] - d) * (w[i] - d)
                                qn8 += d * d
                            }
                        }
                        if isSix {
                            scales6[row * (k / group) + g] = s
                        } else {
                            scales8[row * (k / group) + g] = s
                        }
                    }
                }
                for block in 0 ..< k / 8 {
                    var v: UInt64 = 0
                    for t in 0 ..< 8 { v |= UInt64(q6[block * 8 + t] + 32) << (6 * t) }
                    let o = row * (k / 8 * 3) + block * 3
                    codes6[o] = UInt16(v & 0xFFFF)
                    codes6[o + 1] = UInt16((v >> 16) & 0xFFFF)
                    codes6[o + 2] = UInt16((v >> 32) & 0xFFFF)
                }
                func constant(_ err: Double, _ qn: Double) -> Float {
                    let c =
                        (err.squareRoot() + gammaProxy * qn.squareRoot()
                            + gammaExact * wn.squareRoot()) * (1 + 1e-6)
                    return Float(c).nextUp
                }
                bound6[row] = constant(err6, qn6)
                bound8[row] = constant(err8, qn8)
            }
        }
    }

    // MARK: Kernels

    /// Bounds from a proxy: all rows (LIST 0) or the rows of a device list (LIST 1). Writes the
    /// lower and upper envelope values at `[v, row]`.
    private static let proxyKernel = MLXFast.metalKernel(
        name: "inkling2_certified_proxy",
        inputNames: ["codes", "scales", "bound", "x", "norm", "lower", "upper", "list", "count"],
        outputNames: ["lo", "hi"],
        source: """
            constexpr int TM = 4;
            constexpr int NI = K / 256;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint total = LIST ? count[0] : uint(N);
            const uint stride = LIST ? uint(SIMDGROUPS * TM) : uint(N);
            for (uint base = (threadgroup_position_in_grid.x * BM + sg) * TM; base < total;
                 base += stride) {
                uint r[TM];
                for (int tm = 0; tm < TM; tm++) {
                    const uint at = min(base + uint(tm), total - 1);
                    r[tm] = LIST ? list[at] : at;
                }
                float acc[V][TM];
                for (int v = 0; v < V; v++) {
                    for (int tm = 0; tm < TM; tm++) {
                        acc[v][tm] = 0.0f;
                    }
                }
                for (int i = 0; i < NI; i++) {
                    const int k0 = i * 256 + int(lane) * 8;
                    float xs[V][8];
                    for (int v = 0; v < V; v++) {
                        const uint4 u = *((const device uint4*)(x + size_t(v) * K + k0));
                        xs[v][0] = as_type<float>(u.x << 16);
                        xs[v][1] = as_type<float>(u.x & 0xFFFF0000u);
                        xs[v][2] = as_type<float>(u.y << 16);
                        xs[v][3] = as_type<float>(u.y & 0xFFFF0000u);
                        xs[v][4] = as_type<float>(u.z << 16);
                        xs[v][5] = as_type<float>(u.z & 0xFFFF0000u);
                        xs[v][6] = as_type<float>(u.w << 16);
                        xs[v][7] = as_type<float>(u.w & 0xFFFF0000u);
                    }
                    for (int tm = 0; tm < TM; tm++) {
                        float q[8];
                        if (QB == 6) {
                            const device ushort* p =
                                (const device ushort*)codes + size_t(r[tm]) * (K / 8 * 3) + (k0 / 8) * 3;
                            const ulong bits = ulong(p[0]) | (ulong(p[1]) << 16) | (ulong(p[2]) << 32);
                            for (int t = 0; t < 8; t++) {
                                q[t] = float(int((bits >> (6 * t)) & 63ul) - 32);
                            }
                        } else {
                            const uint2 b = *((const device uint2*)((const device char*)codes
                                + size_t(r[tm]) * K + k0));
                            for (int t = 0; t < 4; t++) {
                                q[t] = float(int(b.x << (24 - 8 * t)) >> 24);
                                q[t + 4] = float(int(b.y << (24 - 8 * t)) >> 24);
                            }
                        }
                        const float s = float(scales[size_t(r[tm]) * (K / 64) + k0 / 64]);
                        for (int v = 0; v < V; v++) {
                            float p = q[0] * xs[v][0];
                            for (int t = 1; t < 8; t++) {
                                p = fma(q[t], xs[v][t], p);
                            }
                            acc[v][tm] = fma(s, p, acc[v][tm]);
                        }
                    }
                }
                for (int v = 0; v < V; v++) {
                    for (int tm = 0; tm < TM; tm++) {
                        acc[v][tm] = simd_sum(acc[v][tm]);
                    }
                }
                if (lane == 0) {
                    for (int tm = 0; tm < TM; tm++) {
                        if (base + uint(tm) >= total) {
                            continue;
                        }
                        const uint row = r[tm];
                        for (int v = 0; v < V; v++) {
                            float l = -INFINITY;
                            float h = INFINITY;
                            const float hn = norm[v];
                            if (isfinite(hn)) {
                                const float a = acc[v][tm];
                                const float b = bound[row] * hn;
                                const float m = (fabs(a) + b) * 1e-6f + 1e-30f;
                                l = (a - b) - m;
                                h = (a + b) + m;
                            }
                            uint lb = as_type<uint>(l);
                            uint hb = as_type<uint>(h);
                            lb = (lb + 0x7FFFu + ((lb >> 16) & 1u)) >> 16;
                            hb = (hb + 0x7FFFu + ((hb >> 16) & 1u)) >> 16;
                            lo[size_t(v) * N + row] = lower[lb];
                            hi[size_t(v) * N + row] = upper[hb];
                        }
                    }
                }
            }
            """)

    /// Appends every row j with `hi[r, j] >= tau[r]` for some r to `list` (any order).
    private static let compactKernel = MLXFast.metalKernel(
        name: "inkling2_certified_compact",
        inputNames: ["hi", "tau"],
        outputNames: ["list", "count"],
        source: """
            const uint j = thread_position_in_grid.x;
            if (j >= uint(N)) {
                return;
            }
            const int R = int(tau_shape[0]);
            bool keep = false;
            for (int r = 0; r < R; r++) {
                keep = keep || hi[size_t(r) * N + j] >= tau[r];
            }
            if (keep) {
                const uint p = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
                atomic_store_explicit(&list[p], j, memory_order_relaxed);
            }
            """,
        atomicOutputs: true)

    /// Proxy bounds through 8x8 MMA tiles: each proxy row is read once for up to 64 input rows.
    /// Products of integer codes and bf16 inputs are exact; a superblock's partial sum is scaled by
    /// the row's group scale (the lane's own fragment row) and added. All rows (LIST 0, one tile per
    /// threadgroup) or the rows of a device list (LIST 1, grid stride). Writes `[c, row]`.
    private static let proxyTileKernel = MLXFast.metalKernel(
        name: "inkling2_certified_proxy_tile",
        inputNames: ["codes", "scales", "bound", "x", "norm", "lower", "upper", "list", "count"],
        outputNames: ["lo", "hi"],
        source: """
            const int R = int(x_shape[0]);
            const uint lane = thread_index_in_simdgroup;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int ks = sg % SK;
            threadgroup float buf[SK * 8 * 64];
            threadgroup float* mine = buf + ks * 8 * 64;
            const uint total = LIST ? count[0] : uint(N);
            const uint tiles = (total + 7) / 8;
            const uint step = LIST ? uint(TILEGROUPS) : tiles;
            const int qid = int(lane) / 4;
            const int fm = (qid & 4) + ((int(lane) / 2) % 4);
            const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
            const int pick = fm & 1;
            for (uint tile = threadgroup_position_in_grid.x; tile < tiles; tile += step) {
                const int valid = int(min(8u, total - tile * 8));
                uint xo[2 * RB];
                for (int t = 0; t < RB; t++) {
                    xo[2 * t] = uint(min(t * 8 + fn, R - 1) * K);
                    xo[2 * t + 1] = uint(min(t * 8 + fn + 1, R - 1) * K);
                }
                const uint at_row = tile * 8 + uint(min(fm, valid - 1));
                const uint row = LIST ? list[at_row] : at_row;
                simdgroup_float8x8 acc[RB];
                for (int t = 0; t < RB; t++) {
                    acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
                for (int s = ks; s < K / 32; s += SK) {
                    const int k0 = s * 32 + (fn / 2) * 8;
                    float q[8];
                    if (QB == 6) {
                        const device ushort* p =
                            (const device ushort*)codes + size_t(row) * (K / 8 * 3) + (k0 / 8) * 3;
                        const ulong bits = ulong(p[0]) | (ulong(p[1]) << 16) | (ulong(p[2]) << 32);
                        for (int t = 0; t < 8; t++) {
                            q[t] = float(int((bits >> (6 * t)) & 63ul) - 32);
                        }
                    } else {
                        const uint2 b = *((const device uint2*)((const device char*)codes
                            + size_t(row) * K + k0));
                        for (int t = 0; t < 4; t++) {
                            q[t] = float(int(b.x << (24 - 8 * t)) >> 24);
                            q[t + 4] = float(int(b.y << (24 - 8 * t)) >> 24);
                        }
                    }
                    const float sc = float(scales[size_t(row) * (K / 64) + (s * 32) / 64]);
                    simdgroup_float8x8 A0, A1, A2, A3;
                    A0.thread_elements()[0] = q[0];
                    A0.thread_elements()[1] = q[1];
                    A1.thread_elements()[0] = q[2];
                    A1.thread_elements()[1] = q[3];
                    A2.thread_elements()[0] = q[4];
                    A2.thread_elements()[1] = q[5];
                    A3.thread_elements()[0] = q[6];
                    A3.thread_elements()[1] = q[7];
                    const int kx = s * 32 + (fm / 2) * 8;
                    for (int t = 0; t < RB; t++) {
                        const ink2_v8 u = ink2_load8<T>(x + xo[2 * t] + kx);
                        const ink2_v8 v = ink2_load8<T>(x + xo[2 * t + 1] + kx);
                        simdgroup_float8x8 B0, B1, B2, B3;
                        B0.thread_elements()[0] = pick ? u.lo.y : u.lo.x;
                        B0.thread_elements()[1] = pick ? v.lo.y : v.lo.x;
                        B1.thread_elements()[0] = pick ? u.lo.w : u.lo.z;
                        B1.thread_elements()[1] = pick ? v.lo.w : v.lo.z;
                        B2.thread_elements()[0] = pick ? u.hi.y : u.hi.x;
                        B2.thread_elements()[1] = pick ? v.hi.y : v.hi.x;
                        B3.thread_elements()[0] = pick ? u.hi.w : u.hi.z;
                        B3.thread_elements()[1] = pick ? v.hi.w : v.hi.z;
                        simdgroup_float8x8 part = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                        simdgroup_multiply_accumulate(part, A0, B0, part);
                        simdgroup_multiply_accumulate(part, A1, B1, part);
                        simdgroup_multiply_accumulate(part, A2, B2, part);
                        simdgroup_multiply_accumulate(part, A3, B3, part);
                        acc[t].thread_elements()[0] =
                            fma(sc, part.thread_elements()[0], acc[t].thread_elements()[0]);
                        acc[t].thread_elements()[1] =
                            fma(sc, part.thread_elements()[1], acc[t].thread_elements()[1]);
                    }
                }
                for (int t = 0; t < RB; t++) {
                    simdgroup_store(acc[t], mine + t * 64, 8);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const int cols = RB * 8;
                for (int e = int(thread_index_in_threadgroup); e < cols * 8;
                     e += int(threads_per_threadgroup.x)) {
                    const int nl = e % 8;
                    const int c = (e / 8) % cols;
                    if (c >= R || nl >= valid) {
                        continue;
                    }
                    const int at = ((c / 8) * 8 + nl) * 8 + (c % 8);
                    float a = buf[at];
                    for (int s = 1; s < SK; s++) {
                        a += buf[s * 8 * 64 + at];
                    }
                    const uint orow = LIST ? list[tile * 8 + uint(nl)] : tile * 8 + uint(nl);
                    float l = -INFINITY;
                    float h = INFINITY;
                    const float hn = norm[c];
                    if (isfinite(hn)) {
                        const float b = bound[orow] * hn;
                        const float m = (fabs(a) + b) * 1e-6f + 1e-30f;
                        l = (a - b) - m;
                        h = (a + b) + m;
                    }
                    uint lb = as_type<uint>(l);
                    uint hb = as_type<uint>(h);
                    lb = (lb + 0x7FFFu + ((lb >> 16) & 1u)) >> 16;
                    hb = (hb + 0x7FFFu + ((hb >> 16) & 1u)) >> 16;
                    lo[size_t(c) * N + orow] = lower[lb];
                    hi[size_t(c) * N + orow] = upper[hb];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: Inkling2Kernels.header)

    /// Exact logits of the listed rows: `rowProduct`'s MMA tile with rows read through the list
    /// (`ink2_mma_tile` with a per-row pointer), split-K partials added in index order, rounded
    /// to T. Writes `[c, row]`.
    private static let exactKernel = MLXFast.metalKernel(
        name: "inkling2_certified_exact",
        inputNames: ["x", "w", "list", "count"],
        outputNames: ["out"],
        source: """
            const int R = int(x_shape[0]);
            const uint lane = thread_index_in_simdgroup;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int ks = sg % SK;
            threadgroup float buf[SK * 8 * 64];
            threadgroup float* mine = buf + ks * 8 * 64;
            const uint total = count[0];
            const uint tiles = (total + 7) / 8;
            const int qid = int(lane) / 4;
            const int fm = (qid & 4) + ((int(lane) / 2) % 4);
            const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
            const int pick = fm & 1;
            for (uint tile = threadgroup_position_in_grid.x; tile < tiles; tile += TILEGROUPS) {
                const int valid = int(min(8u, total - tile * 8));
                uint xo[2 * RB];
                for (int t = 0; t < RB; t++) {
                    xo[2 * t] = uint(min(t * 8 + fn, R - 1) * K);
                    xo[2 * t + 1] = uint(min(t * 8 + fn + 1, R - 1) * K);
                }
                simdgroup_float8x8 acc[RB];
                for (int t = 0; t < RB; t++) {
                    acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
                const int wr = min(fm, valid - 1);
                const device T* wrow = w + size_t(list[tile * 8 + uint(wr)]) * K;
                for (int s = ks; s < K / 32; s += SK) {
                    const ink2_v8 a = ink2_dequant8<T, 0, false>(wrow, nullptr, s * 32 + (fn / 2) * 8, nullptr);
                    simdgroup_float8x8 A0, A1, A2, A3;
                    A0.thread_elements()[0] = a.lo.x;
                    A0.thread_elements()[1] = a.lo.y;
                    A1.thread_elements()[0] = a.lo.z;
                    A1.thread_elements()[1] = a.lo.w;
                    A2.thread_elements()[0] = a.hi.x;
                    A2.thread_elements()[1] = a.hi.y;
                    A3.thread_elements()[0] = a.hi.z;
                    A3.thread_elements()[1] = a.hi.w;
                    const int kx = s * 32 + (fm / 2) * 8;
                    for (int t = 0; t < RB; t++) {
                        const ink2_v8 u = ink2_load8<T>(x + xo[2 * t] + kx);
                        const ink2_v8 v = ink2_load8<T>(x + xo[2 * t + 1] + kx);
                        simdgroup_float8x8 B0, B1, B2, B3;
                        B0.thread_elements()[0] = pick ? u.lo.y : u.lo.x;
                        B0.thread_elements()[1] = pick ? v.lo.y : v.lo.x;
                        B1.thread_elements()[0] = pick ? u.lo.w : u.lo.z;
                        B1.thread_elements()[1] = pick ? v.lo.w : v.lo.z;
                        B2.thread_elements()[0] = pick ? u.hi.y : u.hi.x;
                        B2.thread_elements()[1] = pick ? v.hi.y : v.hi.x;
                        B3.thread_elements()[0] = pick ? u.hi.w : u.hi.z;
                        B3.thread_elements()[1] = pick ? v.hi.w : v.hi.z;
                        simdgroup_multiply_accumulate(acc[t], A0, B0, acc[t]);
                        simdgroup_multiply_accumulate(acc[t], A1, B1, acc[t]);
                        simdgroup_multiply_accumulate(acc[t], A2, B2, acc[t]);
                        simdgroup_multiply_accumulate(acc[t], A3, B3, acc[t]);
                    }
                }
                for (int t = 0; t < RB; t++) {
                    simdgroup_store(acc[t], mine + t * 64, 8);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const int cols = RB * 8;
                for (int e = int(thread_index_in_threadgroup); e < cols * 8;
                     e += int(threads_per_threadgroup.x)) {
                    const int nl = e % 8;
                    const int c = (e / 8) % cols;
                    if (c >= R || nl >= valid) {
                        continue;
                    }
                    const int at = ((c / 8) * 8 + nl) * 8 + (c % 8);
                    float value = buf[at];
                    for (int s = 1; s < SK; s++) {
                        value += buf[s * 8 * 64 + at];
                    }
                    out[size_t(c) * N + list[tile * 8 + uint(nl)]] = static_cast<T>(value);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: Inkling2Kernels.header)

    /// Per row r over the listed rows: the non-finite flag (a NaN softcapped logit), and the token
    /// MLX's `argMax` gives: the first NaN if any, else the first index of the maximum. One
    /// threadgroup per row.
    private static let selectKernel = MLXFast.metalKernel(
        name: "inkling2_certified_select",
        inputNames: ["logits", "table", "list", "count"],
        outputNames: ["token", "flag"],
        source: """
            const uint r = threadgroup_position_in_grid.x;
            const uint tid = thread_index_in_threadgroup;
            const uint total = count[0];
            float best = -INFINITY;
            uint index = 0xFFFFFFFFu;
            uint firstNaN = 0xFFFFFFFFu;
            for (uint i = tid; i < total; i += THREADS) {
                const uint j = list[i];
                const ushort b = as_type<ushort>(logits[size_t(r) * N + j]);
                const float f = table[b];
                if (isnan(f)) {
                    firstNaN = min(firstNaN, j);
                } else if (f > best || (f == best && j < index)) {
                    best = f;
                    index = j;
                }
            }
            threadgroup float bests[THREADS];
            threadgroup uint indices[THREADS];
            threadgroup uint nans[THREADS];
            bests[tid] = best;
            indices[tid] = index;
            nans[tid] = firstNaN;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint half_ = THREADS / 2; half_ > 0; half_ >>= 1) {
                if (tid < half_) {
                    const float ob = bests[tid + half_];
                    const uint oi = indices[tid + half_];
                    if (ob > bests[tid] || (ob == bests[tid] && oi < indices[tid])) {
                        bests[tid] = ob;
                        indices[tid] = oi;
                    }
                    nans[tid] = min(nans[tid], nans[tid + half_]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                // MLX's argMax takes the first NaN when there is one (the engine then fails the
                // row on its non-finite flag); otherwise the first maximum.
                const bool bad = nans[0] != 0xFFFFFFFFu;
                token[r] = int(bad ? nans[0] : indices[0]);
                flag[r] = bad ? 1 : 0;
            }
            """)

    // MARK: Launches

    /// Research and tests: when true, every launch records the MLX library it needs in
    /// `censusVariants`, keyed the way MLX names a custom kernel library (kernel, template
    /// arguments, each input's dtype and whether it is small enough for the constant address
    /// space, output dtypes). Row counts are runtime values in every kernel, so the set is fixed
    /// for a head: RB (1, 2, 4, 8), QB (6, 8), LIST, the exact stage's two tile-group counts, and
    /// R-sized inputs below or from 8 elements.
    public static var censusEnabled: Bool {
        get { censusLock.withLock { censusOn } }
        set { censusLock.withLock { censusOn = newValue } }
    }
    private static let censusLock = NSLock()
    // Both guarded by `censusLock`: engines on one model launch from several threads while a
    // test may switch the census on or off.
    nonisolated(unsafe) private static var censusOn = false
    nonisolated(unsafe) private static var census = Set<String>()
    public static var censusVariants: Set<String> { censusLock.withLock { census } }

    private static func launch(
        _ kernel: MLXFast.MLXFastKernel, _ name: String, _ inputs: [MLXArray],
        template: [(String, any KernelTemplateArg)], grid: (Int, Int, Int),
        threadGroup: (Int, Int, Int), outputShapes: [[Int]], outputDTypes: [DType],
        initValue: Float? = nil
    ) -> [MLXArray] {
        if censusEnabled {
            var key = name
            for (k, v) in template { key += "_\(k)=\(v)" }
            for a in inputs { key += "_\(a.dtype)" + (a.ndim == 0 ? "s" : a.size < 8 ? "c" : "") }
            for d in outputDTypes { key += "_\(d)" }
            censusLock.withLock { _ = census.insert(key) }
        }
        return kernel(
            inputs, template: template, grid: grid, threadGroup: threadGroup,
            outputShapes: outputShapes, outputDTypes: outputDTypes, initValue: initValue)
    }

    /// Compiles every kernel variant this head can use for 1 ... `maxRows` rows (1 <= maxRows <=
    /// 64), greedy, temperature-only and through the public primitives, so no later call of any
    /// row count builds a library mid-request: row counts 1 (one-row proxy), 2 and 8 (R-sized
    /// inputs in constant and device memory), 9, 17 and 33 (the tile widths RB 2, 4, 8).
    public func warm(maxRows: Int = 64) {
        precondition((1 ... 64).contains(maxRows), "warm-up covers 1 ... 64 rows")
        for r in [1, 2, 8, 9, 17, 33] where r <= maxRows {
            let hidden = MLXRandom.normal([r, columns], key: MLXRandom.key(UInt64(r)))
                .asType(.bfloat16)
            let u = MLXRandom.uniform(
                low: Float(0), high: Float(1), [r, rows], key: MLXRandom.key(UInt64(100 + r)))
            let noise = maximum(-log(1 - u), MLXArray(Float(1e-20)))
            let greedyRows = (0 ..< r).map { $0 % 2 == 1 }
            let g = greedy(hidden)
            let race = select(
                hidden, greedy: greedyRows,
                temperature: MLXArray.full([r, 1], values: MLXArray(Float(0.8))), noise: noise)
            // The two proxy / list combinations `select` does not use.
            let all = MLXArray(Array(UInt32(0) ..< UInt32(rows)))
            let n = MLXArray([UInt32(rows)])
            let b8 = proxyBounds(hidden, proxy: .int8)
            let b6 = proxyBounds(hidden, proxy: .int6, rows: (all, n))
            eval(g.tokens, g.nonFinite, race.tokens, race.nonFinite, b8.lower, b6.upper)
        }
    }

    // MARK: Selection

    /// `hidden`: `[R, K]` bf16 final-norm rows (1 <= R <= 64). Greedy tokens of the full product.
    public func greedy(_ hidden: MLXArray) -> Selection { greedyDetailed(hidden).selection }

    /// `greedy` plus the exact-stage logits `[R, N]` (valid at the listed rows) and that list.
    func greedyDetailed(_ hidden: MLXArray) -> (
        selection: Selection, exact: MLXArray, list: MLXArray, count: MLXArray
    ) {
        precondition(
            hidden.ndim == 2 && hidden.dim(1) == columns && hidden.dtype == .bfloat16
                && (1 ... 64).contains(hidden.dim(0)),
            "certified head needs bf16 [1...64, K]")
        let r = hidden.dim(0)
        let norm = Self.upperNorm(hidden)
        let empty = MLXArray([UInt32(0)])

        // Stage 1: int6 bounds for every row.
        let (lo6, hi6) = bounds(
            codes6, scales6, bound6, bits: 6, hidden, norm, list: empty, count: empty, listed: false
        )
        let tau6 = lo6.max(axis: -1)
        let (list1, count1) = compact(hi6, tau6)

        // Stage 2: int8 bounds for the stage-1 rows.
        let (lo8, hi8) = bounds(
            codes8, scales8, bound8, bits: 8, hidden, norm, list: list1, count: count1, listed: true
        )
        let tau8 = maximum(tau6, lo8.max(axis: -1))
        let (list2, count2) = compact(hi8, tau8)

        // Stage 3: exact logits of the remaining rows; the first maximum.
        let exact = exactLogits(hidden, list: list2, count: count2)
        let picked = firstMaximum(exact, list: list2, count: count2)
        let selection = Selection(
            tokens: picked.tokens, nonFinite: picked.flags,
            candidateCounts: concatenated([count1, count2], axis: 0))
        return (selection, exact, list2, count2)
    }

    /// Exact bf16 logits `[R, N]` of the listed rows (other entries are not written).
    private func exactLogits(
        _ hidden: MLXArray, list: MLXArray, count: MLXArray,
        tileGroups: Int = Inkling2CertifiedHead.exactThreadgroups
    ) -> MLXArray {
        let r = hidden.dim(0)
        let rb = r <= 8 ? 1 : r <= 16 ? 2 : r <= 32 ? 4 : 8
        let sk = Inkling2Kernels.splitK(columns)
        return Self.launch(
            Self.exactKernel, "exactKernel",
            [hidden, weight, list, count],
            template: [
                ("T", DType.bfloat16), ("K", columns), ("N", rows), ("RB", rb),
                ("SK", sk), ("TILEGROUPS", tileGroups),
            ],
            grid: (tileGroups * 32 * sk, 1, 1), threadGroup: (32 * sk, 1, 1),
            outputShapes: [[r, rows]], outputDTypes: [.bfloat16])[0]
    }

    /// Per row: the first maximum of the softcapped listed logits and the non-finite flag.
    private func firstMaximum(_ exact: MLXArray, list: MLXArray, count: MLXArray) -> (
        tokens: MLXArray, flags: MLXArray
    ) {
        let r = exact.dim(0)
        let out = Self.launch(
            Self.selectKernel, "selectKernel",
            [exact, softcapTable, list, count],
            template: [("N", rows), ("THREADS", 256)],
            grid: (r * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[r], [r]], outputDTypes: [.int32, .int32])
        return (out[0], out[1])
    }

    /// `‖h‖₂` per row, scaled against overflow and rounded up generously; +inf when a row is
    /// non-finite or larger than 1e30 in magnitude (then every vocabulary row is a candidate).
    static func upperNorm(_ hidden: MLXArray) -> MLXArray {
        let h = hidden.asType(.float32)
        let peak = abs(h).max(axis: -1)
        let scale = which(peak .> 0, peak, MLXArray(Float(1)))
        let norm =
            scale * sqrt(sum(square(h / scale.expandedDimensions(axis: -1)), axis: -1))
            * Float(1.001)
        let bad = logicalOr(logicalOr(isNaN(norm), isInf(norm)), peak .> Float(1e30))
        return which(bad, MLXArray(Float.infinity), norm)
    }

    /// One row: the gemv-style proxy kernel. Two to 64 rows: the MMA tile kernel, which reads each
    /// proxy row once for all of them.
    private func bounds(
        _ codes: MLXArray, _ scales: MLXArray, _ bound: MLXArray, bits: Int, _ x: MLXArray,
        _ norm: MLXArray, list: MLXArray, count: MLXArray, listed: Bool, envelope: Envelope? = nil
    ) -> (MLXArray, MLXArray) {
        let r = x.dim(0)
        let lowerEnvelope = envelope?.lower ?? self.lowerEnvelope
        let upperEnvelope = envelope?.upper ?? self.upperEnvelope
        if r == 1 {
            return gemvBounds(
                codes, scales, bound, bits: bits, x, norm, list: list, count: count,
                listed: listed, lower: lowerEnvelope, upper: upperEnvelope)
        }
        let rb = r <= 8 ? 1 : r <= 16 ? 2 : r <= 32 ? 4 : 8
        let sk = Inkling2Kernels.splitK(columns)
        let groups = listed ? Self.listTileGroups : (rows + 7) / 8
        let out = Self.launch(
            Self.proxyTileKernel, "proxyTileKernel",
            [codes, scales, bound, x, norm, lowerEnvelope, upperEnvelope, list, count],
            template: [
                ("T", DType.bfloat16), ("QB", bits), ("K", columns), ("N", rows), ("RB", rb),
                ("SK", sk), ("LIST", listed), ("TILEGROUPS", Self.listTileGroups),
            ],
            grid: (groups * 32 * sk, 1, 1), threadGroup: (32 * sk, 1, 1),
            outputShapes: [[r, rows], [r, rows]], outputDTypes: [.float32, .float32],
            initValue: listed ? -Float.infinity : nil)
        return (out[0], out[1])
    }

    private func gemvBounds(
        _ codes: MLXArray, _ scales: MLXArray, _ bound: MLXArray, bits: Int, _ x: MLXArray,
        _ norm: MLXArray, list: MLXArray, count: MLXArray, listed: Bool, lower: MLXArray,
        upper: MLXArray
    ) -> (MLXArray, MLXArray) {
        let bm = 4
        let groups =
            listed ? Self.listSimdgroups / bm : (rows + 4 * bm - 1) / (4 * bm)
        let out = Self.launch(
            Self.proxyKernel, "proxyKernel",
            [codes, scales, bound, x, norm, lower, upper, list, count],
            template: [
                ("QB", bits), ("V", x.dim(0)), ("K", columns), ("N", rows), ("BM", bm),
                ("LIST", listed), ("SIMDGROUPS", Self.listSimdgroups),
            ],
            grid: (groups * 32, bm, 1), threadGroup: (32, bm, 1),
            outputShapes: [[x.dim(0), rows], [x.dim(0), rows]],
            outputDTypes: [.float32, .float32],
            initValue: listed ? -Float.infinity : nil)
        return (out[0], out[1])
    }

    private func compact(_ hi: MLXArray, _ tau: MLXArray) -> (MLXArray, MLXArray) {
        let out = Self.launch(
            Self.compactKernel, "compactKernel",
            [hi, tau],
            template: [("N", rows)],
            grid: ((rows + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows], [1]], outputDTypes: [.uint32, .uint32], initValue: 0)
        return (out[0], out[1])
    }

    /// Tests only: every temperature-only row takes the full-vocabulary fallback
    /// (`DARKBLOOM_INKLING2_CERTIFIED_FORCE_FALLBACK=1`).
    static let forceRaceFallback: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_CERTIFIED_FORCE_FALLBACK"] == "1"

    /// Score bounds `F / T - ln(noise)` from F bounds, widened for the float error of this
    /// arithmetic and of the engine's own `F / T` (an upper bound of -inf stays -inf).
    private static let scoreKernel = MLXFast.metalKernel(
        name: "inkling2_certified_scores",
        inputNames: ["lo", "hi", "divisor", "noise"],
        outputNames: ["slo", "shi"],
        source: """
            const uint i = thread_position_in_grid.x;
            const uint r = i / uint(N);
            const float t = divisor[r];
            const float ln = log(noise[i]);
            const float a = lo[i] / t;
            const float b = hi[i] / t;
            slo[i] = (a - ln) - (fabs(a) + fabs(ln)) * 1e-5f - 1e-6f;
            shi[i] = hi[i] == -INFINITY
                ? -INFINITY : (b - ln) + (fabs(b) + fabs(ln)) * 1e-5f + 1e-6f;
            """)

    /// Threadgroups per row in the race certification's first pass.
    static let raceGroups = 32

    /// Race certification, first pass (`raceGroups` threadgroups of 256 per row): over its share of
    /// the vocabulary, each threadgroup keeps the best exact candidate score `F / T - ln(noise)`
    /// (first index on ties), the runner-up among candidates, the largest upper bound of every
    /// other row and whether any of those was NaN.
    private static let racePartialKernel = MLXFast.metalKernel(
        name: "inkling2_certified_race_partial",
        inputNames: ["logits", "table", "mask", "noise", "divisor", "shi1", "shi2", "hi8"],
        outputNames: ["pb1", "pi1", "pb2", "pout", "pbad"],
        source: """
            const uint tg = threadgroup_position_in_grid.x;
            const uint r = tg / uint(G);
            const uint g = tg % uint(G);
            const uint tid = thread_index_in_threadgroup;
            const float t = divisor[r];
            float b1 = -INFINITY;
            uint i1 = 0xFFFFFFFFu;
            float b2 = -INFINITY;
            float out = -INFINITY;
            uint bad = 0u;
            for (uint j = g * 256u + tid; j < uint(N); j += uint(G) * 256u) {
                const size_t at = size_t(r) * N + j;
                if (mask[j] != 0u) {
                    const float f = table[as_type<ushort>(logits[at])];
                    const float sc = f / t - log(noise[at]);
                    bad |= isnan(sc) ? 1u : 0u;
                    if (sc > b1) {
                        b2 = max(b2, b1);
                        b1 = sc;
                        i1 = j;
                    } else {
                        b2 = max(b2, sc);
                    }
                } else {
                    const float h = hi8[at] == -INFINITY ? shi1[at] : min(shi1[at], shi2[at]);
                    bad |= isnan(h) ? 1u : 0u;
                    out = max(out, h);
                }
            }
            threadgroup float tb1[256];
            threadgroup uint ti1[256];
            threadgroup float tb2[256];
            threadgroup float tout[256];
            threadgroup uint tbad[256];
            tb1[tid] = b1;
            ti1[tid] = i1;
            tb2[tid] = b2;
            tout[tid] = out;
            tbad[tid] = bad;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint h = 128u; h > 0u; h >>= 1) {
                if (tid < h) {
                    const float ob1 = tb1[tid + h];
                    const uint oi1 = ti1[tid + h];
                    const float ob2 = tb2[tid + h];
                    if (ob1 > tb1[tid] || (ob1 == tb1[tid] && oi1 < ti1[tid])) {
                        tb2[tid] = max(max(tb2[tid], tb1[tid]), ob2);
                        tb1[tid] = ob1;
                        ti1[tid] = oi1;
                    } else {
                        tb2[tid] = max(tb2[tid], max(ob1, ob2));
                    }
                    tout[tid] = max(tout[tid], tout[tid + h]);
                    tbad[tid] |= tbad[tid + h];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                pb1[tg] = tb1[0];
                pi1[tg] = ti1[0];
                pb2[tg] = tb2[0];
                pout[tg] = tout[0];
                pbad[tg] = tbad[0];
            }
            """)

    /// Race certification, second pass (one thread per row): merges the row's partials; the row
    /// is certified when no score or bound was NaN, the best score and the row's hidden norm are
    /// finite, and the best beats the runner-up and every other row's upper bound by the margin.
    /// Equal best scores leave the runner-up equal to the best, so ties are never certified.
    private static let raceFinalKernel = MLXFast.metalKernel(
        name: "inkling2_certified_race_final",
        inputNames: ["pb1", "pi1", "pb2", "pout", "pbad", "margin", "norm"],
        outputNames: ["best", "certified"],
        source: """
            const uint r = thread_position_in_grid.x;
            float b1 = -INFINITY;
            uint i1 = 0xFFFFFFFFu;
            float b2 = -INFINITY;
            float out = -INFINITY;
            uint bad = 0u;
            for (uint g = 0; g < uint(G); g++) {
                const uint at = r * uint(G) + g;
                const float ob1 = pb1[at];
                const uint oi1 = pi1[at];
                const float ob2 = pb2[at];
                if (ob1 > b1 || (ob1 == b1 && oi1 < i1)) {
                    b2 = max(max(b2, b1), ob2);
                    b1 = ob1;
                    i1 = oi1;
                } else {
                    b2 = max(b2, max(ob1, ob2));
                }
                out = max(out, pout[at]);
                bad |= pbad[at];
            }
            const float m = margin[r];
            const bool ok = bad == 0u && isfinite(b1) && isfinite(norm[r])
                && b1 - b2 > m && b1 - out > m;
            best[r] = int(i1);
            certified[r] = ok ? 1 : 0;
            """)

    /// Marks the listed rows (`[N]` uint32, 1 = listed).
    private static let maskKernel = MLXFast.metalKernel(
        name: "inkling2_certified_mask",
        inputNames: ["list", "count"],
        outputNames: ["mask"],
        source: """
            const uint i = thread_position_in_grid.x;
            if (i < count[0]) {
                mask[list[i]] = 1u;
            }
            """)

    private var cachedRowIndices: MLXArray?
    private let indicesLock = NSLock()
    private var rowIndices: MLXArray {
        indicesLock.withLock {
            if let cachedRowIndices { return cachedRowIndices }
            let indices = MLXArray(Array(UInt32(0) ..< UInt32(rows)), [rows])
            cachedRowIndices = indices
            return indices
        }
    }

    /// Tokens for `hidden` (`[R, K]` bf16, 1 <= R <= 64) as the default sampler would draw them
    /// from the full product's logits `F = softcap(rowProduct(h, W))`:
    /// - greedy rows: the first maximum of F (always certified);
    /// - other rows: `argmax(softmax(F / temperature) / noise)` with the sampler's own divisor and
    ///   noise. The bounds become score bounds `F / T - ln(noise)`; the exact candidates' best
    ///   score must beat the runner-up and every other row's upper bound by a margin covering
    ///   `fast::exp` and three roundings, because the softmax normalizer is never known. Rows
    ///   that miss the margin take the sampler's own ops over exact logits of every vocabulary
    ///   row, computed in the same graph (a list of all rows when needed, empty otherwise).
    public func select(
        _ hidden: MLXArray, greedy: [Bool], temperature: MLXArray?, noise: MLXArray?
    ) -> Selection {
        precondition(greedy.count == hidden.dim(0), "one row kind per hidden row")
        guard let noise, greedy.contains(false) else { return self.greedy(hidden) }
        let r = hidden.dim(0)
        let norm = Self.upperNorm(hidden)
        let empty = MLXArray([UInt32(0)])
        let divisor = temperature ?? MLXArray.ones([r, 1], dtype: .float32)
        let greedyRows = MLXArray(greedy).reshaped([r, 1])
        // Margin in log units: fast::exp is within (3 + 2|y|) ulp with |y| <= 60 / T, plus the
        // product, the division and the score arithmetic; greedy rows need none.
        let margin = which(
            greedyRows, MLXArray(Float(0)),
            Float(1e-3) + Float(4 * 0x1p-23) * (Float(3) + Float(240) / divisor)
        ).squeezed(axis: -1)
        func scores(_ lo: MLXArray, _ hi: MLXArray) -> (MLXArray, MLXArray) {
            let out = Self.launch(
                Self.scoreKernel, "scoreKernel",
                [lo, hi, divisor, noise], template: [("N", rows)],
                grid: (r * rows, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[r, rows], [r, rows]], outputDTypes: [.float32, .float32])
            return (out[0], out[1])
        }

        // Stage 1: int6 score bounds for every row.
        let (lo6, hi6) = bounds(
            codes6, scales6, bound6, bits: 6, hidden, norm, list: empty, count: empty,
            listed: false)
        let (sLo1, sHi1) = scores(lo6, hi6)
        let tau1 = sLo1.max(axis: -1)
        let (list1, count1) = compact(sHi1, tau1 - margin)

        // Stage 2: int8 score bounds for the stage-1 rows (-inf elsewhere).
        let (lo8, hi8) = bounds(
            codes8, scales8, bound8, bits: 8, hidden, norm, list: list1, count: count1,
            listed: true)
        let (sLo2, sHi2) = scores(lo8, hi8)
        let tau2 = maximum(tau1, sLo2.max(axis: -1))
        let (list2, count2) = compact(sHi2, tau2 - margin)

        // Stage 3: exact logits of the candidates; greedy rows pick the first maximum of F.
        let exactA = exactLogits(hidden, list: list2, count: count2)
        let picked = firstMaximum(exactA, list: list2, count: count2)

        // Temperature rows: certify the race among exact candidates against every other row.
        let mask = Self.launch(
            Self.maskKernel, "maskKernel",
            [list2, count2], template: [("N", rows)],
            grid: ((rows + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows]], outputDTypes: [.uint32], initValue: 0)[0]
        let groups = Self.raceGroups
        let partial = Self.launch(
            Self.racePartialKernel, "racePartialKernel",
            [exactA, softcapTable, mask, noise, divisor, sHi1, sHi2, hi8],
            template: [("N", rows), ("G", groups)],
            grid: (r * groups * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[r * groups], [r * groups], [r * groups], [r * groups], [r * groups]],
            outputDTypes: [.float32, .uint32, .float32, .float32, .uint32])
        let verdict = Self.launch(
            Self.raceFinalKernel, "raceFinalKernel",
            partial + [margin, norm], template: [("G", groups)],
            grid: (r, 1, 1), threadGroup: (r, 1, 1),
            outputShapes: [[r], [r]], outputDTypes: [.int32, .int32])
        let best = verdict[0]
        let certified = verdict[1] .!= Int32(0)
        let resolved = logicalOr(
            greedyRows.squeezed(axis: -1), Self.forceRaceFallback ? MLXArray(false) : certified)

        // Fallback: every vocabulary row exact (only when some row is unresolved), then the
        // sampler's ops: softmax(F / T) / noise, first maximum; non-finite flags of F.
        let fallbackCount = which(
            any(logicalNot(resolved)), MLXArray([UInt32(rows)]), MLXArray([UInt32(0)]))
        // One tile per threadgroup: the full pass runs at rowProduct's parallelism, and an empty
        // list costs one early exit per threadgroup.
        let exactB = exactLogits(
            hidden, list: rowIndices, count: fallbackCount, tileGroups: (rows + 7) / 8)
        let fB = softcapTable[exactB.view(dtype: .uint16).asType(.int32)]
        let xB = temperature.map { fB / $0 } ?? fB
        let fullToken = argMax(softmax(xB, axis: -1) / noise, axis: -1).asType(.int32)
        let fullFlag = isNaN(sum(fB - fB, axes: [1])).asType(.int32)

        let race = which(Self.forceRaceFallback ? MLXArray(false) : certified, best, fullToken)
        let tokens = which(greedyRows.squeezed(axis: -1), picked.tokens, race)
        let flags = which(resolved, picked.flags, fullFlag)
        return Selection(
            tokens: tokens, nonFinite: flags,
            candidateCounts: concatenated([count1, count2, fallbackCount], axis: 0))
    }

    // MARK: Primitives for other readers of the unembedding

    /// Which in-memory proxy of the unembedding bounds the products.
    public enum Proxy: Sendable {
        /// Symmetric int6 codes per 64 columns (0.375 of the bf16 bytes); meant for every row.
        case int6
        /// Symmetric int8 codes per 64 columns (0.5 of the bf16 bytes); tighter, meant for the
        /// rows a first pass left in reach.
        case int8
    }

    /// Bounds of `envelope.table[bf16(a)]` for every listed vocabulary row j, where
    /// `a = hidden[r] · w_j` is accumulated in FP32 in any order (`rowProduct`'s MMA tile, or a
    /// GEMM with FP32 accumulation; products of bf16 values are exact) and rounded to bf16 to
    /// nearest even.
    ///
    /// - `hidden`: `[R, K]` bf16, 1 <= R <= 64. One proxy read serves all R rows.
    /// - `envelope`: nil for the head's softcap (bounds of the engine's logits);
    ///   `Envelope.identity()` for bounds of the bf16 product itself; any other `Envelope` for
    ///   bounds of that map of it.
    /// - `rows`: nil bounds every vocabulary row; otherwise only the first `count[0]` entries of
    ///   the `[N]` uint32 list, and every other entry of both outputs is -inf.
    ///
    /// Returns `[R, N]` float32 `(lower, upper)`. A hidden row that is non-finite or holds a value
    /// above 1e30 in magnitude gets the envelope's extremes (the map's minimum and maximum over
    /// every non-NaN bf16 value: -inf and +inf for the identity), so every vocabulary row stays in
    /// reach of `candidates`.
    public func proxyBounds(
        _ hidden: MLXArray, proxy: Proxy = .int6, envelope: Envelope? = nil,
        rows list: (indices: MLXArray, count: MLXArray)? = nil
    ) -> (lower: MLXArray, upper: MLXArray) {
        precondition(
            hidden.ndim == 2 && hidden.dim(1) == columns && hidden.dtype == .bfloat16
                && (1 ... 64).contains(hidden.dim(0)),
            "proxy bounds need bf16 [1...64, K]")
        let empty = MLXArray([UInt32(0)])
        let (codes, scales, bound, bits) =
            proxy == .int6 ? (codes6, scales6, bound6, 6) : (codes8, scales8, bound8, 8)
        let out = bounds(
            codes, scales, bound, bits: bits, hidden, Self.upperNorm(hidden),
            list: list?.indices ?? empty, count: list?.count ?? empty, listed: list != nil,
            envelope: envelope)
        return (out.0, out.1)
    }

    /// The vocabulary rows j with `upper[r, j] >= threshold[r]` for some row r (`[R, N]` and `[R]`
    /// float32), as a device list: `indices` `[N]` uint32 (any order; the first `count[0]` are
    /// valid) and `count` `[1]` uint32. No host round trip.
    public func candidates(upper: MLXArray, atLeast threshold: MLXArray) -> (
        indices: MLXArray, count: MLXArray
    ) {
        precondition(
            upper.ndim == 2 && upper.dim(1) == rows && threshold.shape == [upper.dim(0)],
            "candidates need [R, N] upper bounds and [R] thresholds")
        let out = compact(upper.asType(.float32), threshold.asType(.float32))
        return (out.0, out.1)
    }

    /// The bf16 products `hidden[r] · w_j` of the listed vocabulary rows (the first `count[0]`
    /// entries of `indices`, `[N]` uint32), bit-equal to
    /// `Inkling2Kernels.rowProduct(hidden, transposedWeight: weight)` at those entries: the same
    /// MMA tile and K order, reading weight rows through the list. `hidden`: `[R, K]` bf16,
    /// 1 <= R <= 64. Returns `[R, N]` bf16; entries off the list are not written (undefined).
    public func exactProducts(_ hidden: MLXArray, indices: MLXArray, count: MLXArray) -> MLXArray {
        precondition(
            hidden.ndim == 2 && hidden.dim(1) == columns && hidden.dtype == .bfloat16
                && (1 ... 64).contains(hidden.dim(0)) && indices.shape == [rows]
                && indices.dtype == .uint32 && count.shape == [1] && count.dtype == .uint32,
            "exact products need bf16 [1...64, K], a [N] uint32 list and a [1] uint32 count")
        return exactLogits(hidden, list: indices, count: count)
    }

    // MARK: Research check

    /// Compares certified tokens with the full product (`softcap(rowProduct(h, W))`, first maximum)
    /// on `hidden` (`[P, K]` bf16), `rowsPerCall` rows at a time, and checks that the softcap table
    /// and the exact-stage logits equal the full product's bits. Returns counters.
    public func selfCheck(_ hidden: MLXArray, rowsPerCall: Int, cap: Float) -> [String: Int] {
        var out = [
            "positions": 0, "token_mismatch": 0, "flag_mismatch": 0, "table_mismatch": 0,
            "exact_bit_mismatch": 0, "exact_rows_checked": 0, "stage2_rows_max": 0,
            "exact_rows_max": 0, "stage2_rows_sum": 0, "exact_rows_sum": 0,
        ]
        let p = hidden.dim(0)
        for start in stride(from: 0, to: p, by: rowsPerCall) {
            let h = hidden[start ..< min(p, start + rowsPerCall)]
            let r = h.dim(0)
            let full = Inkling2Kernels.$override.withValue(true) {
                Inkling2Kernels.rowProduct(h, transposedWeight: weight)!
            }
            let logits = Inkling2Math.softcap(full.asType(.float32), cap)
            let reference = argMax(logits, axis: -1).asType(.int32)
            let referenceFlags = CBv2NonFiniteLogitsMirror.flags(logits)
            let detailed = greedyDetailed(h)
            let viaTable = softcapTable[full.view(dtype: .uint16).asType(.int32)]
            eval(
                reference, referenceFlags, detailed.selection.tokens, detailed.selection.nonFinite,
                detailed.exact, detailed.list, detailed.count, viaTable, logits,
                detailed.selection.candidateCounts)
            out["positions"]! += r
            out["token_mismatch"]! += Int(
                sum((reference .!= detailed.selection.tokens).asType(.int32)).item(Int32.self))
            out["flag_mismatch"]! += Int(
                sum((referenceFlags .!= detailed.selection.nonFinite).asType(.int32)).item(
                    Int32.self))
            out["table_mismatch"]! += Int(
                sum((viaTable.view(dtype: .uint32) .!= logits.view(dtype: .uint32)).asType(.int32))
                    .item(Int32.self))
            let counts = detailed.selection.candidateCounts.asArray(UInt32.self).map(Int.init)
            out["stage2_rows_max"] = max(out["stage2_rows_max"]!, counts[0])
            out["exact_rows_max"] = max(out["exact_rows_max"]!, counts[1])
            out["stage2_rows_sum"]! += counts[0]
            out["exact_rows_sum"]! += counts[1]
            let listed = detailed.list[0 ..< counts[1]].asType(.int32)
            let mine = take(detailed.exact, listed, axis: 1).view(dtype: .uint16)
            let theirs = take(full, listed, axis: 1).view(dtype: .uint16)
            out["exact_bit_mismatch"]! += Int(
                sum((mine .!= theirs).asType(.int32)).item(Int32.self))
            out["exact_rows_checked"]! += counts[1] * r
        }
        return out
    }
}

extension Inkling2CertifiedHead {
    /// Research check for temperature-only rows: `select` against the default sampler's ops on the
    /// full product (`argmax(softmax(F / T) / noise)`, noise from keyed uniforms), `rowsPerCall`
    /// rows at a time. Counts token and flag mismatches and calls that needed the fallback.
    public func selfCheckRace(
        _ hidden: MLXArray, rowsPerCall: Int, cap: Float, temperature: Float, seed: UInt64
    ) -> [String: Int] {
        var out = [
            "positions": 0, "token_mismatch": 0, "flag_mismatch": 0, "fallback_calls": 0,
            "calls": 0, "exact_rows_sum": 0, "exact_rows_max": 0,
        ]
        let p = hidden.dim(0)
        for start in stride(from: 0, to: p, by: rowsPerCall) {
            let h = hidden[start ..< min(p, start + rowsPerCall)]
            let r = h.dim(0)
            let full = Inkling2Kernels.$override.withValue(true) {
                Inkling2Kernels.rowProduct(h, transposedWeight: weight)!
            }
            let f = Inkling2Math.softcap(full.asType(.float32), cap)
            let t = MLXArray.full([r, 1], values: MLXArray(temperature))
            let u = MLXRandom.uniform(
                low: Float(0), high: Float(1), [r, rows], key: MLXRandom.key(seed &+ UInt64(start)))
            let noise = maximum(-log(1 - u), MLXArray(Float(1e-20)))
            let reference = argMax(softmax(f / t, axis: -1) / noise, axis: -1).asType(.int32)
            let referenceFlags = CBv2NonFiniteLogitsMirror.flags(f)
            let s = select(
                h, greedy: Array(repeating: false, count: r), temperature: t, noise: noise)
            eval(reference, referenceFlags, s.tokens, s.nonFinite, s.candidateCounts)
            let counts = s.candidateCounts.asArray(UInt32.self).map(Int.init)
            out["positions"]! += r
            out["calls"]! += 1
            out["token_mismatch"]! += Int(
                sum((reference .!= s.tokens).asType(.int32)).item(Int32.self))
            out["flag_mismatch"]! += Int(
                sum((referenceFlags .!= s.nonFinite).asType(.int32)).item(Int32.self))
            out["fallback_calls"]! += counts[2] > 0 ? 1 : 0
            out["exact_rows_sum"]! += counts[1]
            out["exact_rows_max"] = max(out["exact_rows_max"]!, counts[1])
        }
        return out
    }
}

/// The engine's non-finite rule (`CBv2NonFiniteLogits.flags`), for the research check.
enum CBv2NonFiniteLogitsMirror {
    static func flags(_ logits: MLXArray) -> MLXArray {
        isNaN(sum(logits - logits, axes: [1])).asType(.int32)
    }
}

/// Lazily built certified head of one model (built once, under a lock).
final class Inkling2CertifiedHeadCell: @unchecked Sendable {
    private let lock = NSLock()
    private var head: Inkling2CertifiedHead?
    private var failed = false

    func get(_ make: () throws -> Inkling2CertifiedHead) -> Inkling2CertifiedHead? {
        lock.withLock {
            if let head { return head }
            if failed { return nil }
            do {
                head = try make()
            } catch {
                failed = true
            }
            return head
        }
    }
}

extension Inkling2Model: CBv2CertifiedTokenModel {
    /// `DARKBLOOM_INKLING2_CERTIFIED=0` keeps every step on the full product.
    static let certifiedEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_CERTIFIED"]?.lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Largest decode batch that takes the certified path (`DARKBLOOM_INKLING2_CERTIFIED_ROWS`).
    /// M5 Max (2026-11-21, idle GPU, decode kernels on): greedy steps save 1.19 / 0.91 / 0.83 /
    /// 0.69 / 0.57 / 0.45 / 0.36 / 0.29 ms at 1-8 rows and cost 0.62 ms at 16.
    static let certifiedRowLimit: Int = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_CERTIFIED_ROWS"]
        return raw.flatMap(Int.init).map { max(0, min($0, 64)) } ?? 8
    }()

    /// Largest step with temperature-only rows that takes the certified path
    /// (`DARKBLOOM_INKLING2_CERTIFIED_RACE_ROWS`). Same M5 Max run, fused race kernels: 0.84 / 0.67 /
    /// 0.72 / 0.65 / 0.66 / 0.63 / 0.63 / 0.62 ms saved at 1-8 rows, about break-even at 16.
    static let certifiedRaceRowLimit: Int = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_CERTIFIED_RACE_ROWS"]
        return raw.flatMap(Int.init).map { max(0, min($0, 64)) } ?? 8
    }()

    public var cbv2SupportsCertifiedTokens: Bool { Self.certifiedEnabled }

    public var cbv2CertifiedVocabularySize: Int { lmHead.weight.dim(0) }

    /// The certified head of this model, built from `lm_head` on first use.
    func certifiedHeadIfSupported() -> Inkling2CertifiedHead? {
        guard Self.certifiedEnabled, type(of: lmHead) == Linear.self, lmHead.bias == nil,
            lmHead.weight.dtype == .bfloat16
        else { return nil }
        let cap = configuration.text.finalLogitSoftcap
        return certifiedHead.get({
            try Inkling2CertifiedHead(weight: lmHead.weight) { Inkling2Math.softcap($0, cap) }
        })
    }

    /// Builds the head and compiles every kernel variant it can use for 1 ... 64 rows (decode rows,
    /// verify windows and the drafter's primitives alike), so no request pays the build or a
    /// compile.
    public func cbv2WarmCertifiedTokens() {
        certifiedHeadIfSupported()?.warm()
    }

    /// Placeholder logits when this forward's `CBv2CertifiedTokenRequest` is fulfilled from
    /// `hidden` (`[B, 1, K]`); nil when the full product must run. The full product it stands in
    /// for is `softcap(rowProduct(hidden, lm_head))`, so the offer is taken only where `logits`
    /// would use `rowProduct`.
    func certifiedLogits(_ hidden: MLXArray) -> MLXArray? {
        guard let request = CBv2CertifiedTokenRequest.current, !request.isFulfilled,
            Inkling2RowExact.active, Inkling2Kernels.active,
            hidden.ndim == 3, hidden.dim(1) == 1, hidden.dim(0) == request.rows,
            request.rows <= Self.certifiedRowLimit, hidden.dtype == .bfloat16,
            !request.greedy.contains(false) || request.rows <= Self.certifiedRaceRowLimit,
            lmHead.weight.dim(1) == hidden.dim(2),
            request.vocabularySize == lmHead.weight.dim(0),
            let head = certifiedHeadIfSupported()
        else { return nil }
        let selection = head.select(
            hidden[0..., 0, 0...], greedy: request.greedy, temperature: request.temperature,
            noise: request.noise)
        request.fulfill(
            tokens: selection.tokens, nonFinite: selection.nonFinite,
            candidateCounts: selection.candidateCounts)
        return MLXArray.zeros([hidden.dim(0), 1, 1], dtype: .float32)
    }
}
