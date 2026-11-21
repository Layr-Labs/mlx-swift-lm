// Custom Metal kernels for Inkling 2 decode and verify steps: multi-row
// projections and grouped routed experts. A row's arithmetic is fixed by the
// weight shape alone, so its bits never depend on which or how many other
// rows run in the same step.
import Foundation
import MLX

/// Row-exact kernels for steps that run under `Inkling2RowExact.active`.
///
/// Every output element is one FP32 reduction whose order depends only on the
/// weight shape: lanes split K in fixed blocks of eight, each lane keeps an
/// `fma` chain per (row, output), lanes are combined by a fixed shuffle ladder
/// and split-K simdgroups in index order. Rows run in register tiles of
/// 8, 4, 2 and 1; every tile size executes the same per-row sequence, so a row
/// alone equals the same row among any company.
public enum Inkling2Kernels {
    /// `DARKBLOOM_INKLING2_KERNELS=0` keeps the plain MLX decode path.
    public static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_KERNELS"]?.lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Overrides `enabled` for the current task (tests and benchmarks).
    @TaskLocal public static var override: Bool? = nil
    public static var active: Bool { override ?? enabled }

    /// A prepared launch: kernel, activation and weight dtypes and shape values
    /// (no string building on the decode path).
    struct LaunchKey: Hashable {
        var kernel: Int
        var dtype: DType
        var weightType: DType = .float32
        var a = 0
        var b = 0
        var c = 0
        var d = 0
        var e = 0
        var f = 0
    }

    /// Prepared launches keyed by kernel, template and shape.
    final class LaunchCache: @unchecked Sendable {
        private let lock = NSLock()
        private var launches: [LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

        func launch(_ key: LaunchKey, _ make: () -> MLXFast.MLXFastKernel.Launch)
            -> MLXFast.MLXFastKernel.Launch
        {
            lock.withLock {
                if let launch = launches[key] { return launch }
                let launch = make()
                launches[key] = launch
                return launch
            }
        }
    }

    static let launches = LaunchCache()

    /// MMA operands narrower than FP32 when every value is exact in them
    /// (same products, same FP32 accumulation, faster MMAs). Tests clear it to
    /// compare with FP32 operands bit for bit.
    nonisolated(unsafe) static var narrowOperands = true

    /// Superblock loads in flight per MMA tile (`ink2_mma_tile`'s PF): 4 for
    /// launches of fewer than `prefetchThreadgroups` threadgroups (small grids
    /// wait on DRAM latency), else 1. Bits never change (same MMAs, same order).
    /// `DARKBLOOM_INKLING2_PREFETCH`: small (default) | none | all.
    static let prefetchPolicy: String =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFETCH"] ?? "small"
    static let prefetchThreadgroups =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFETCH_TG"] ?? "") ?? 64
    /// Largest MMA row block (8-column tiles) a launch over `rows` input rows
    /// can use: sizes the threadgroup result buffer (64 rows per work item).
    static func maxRB(_ rows: Int) -> Int {
        let c = min(max(rows, 1), 64)
        return c <= 8 ? 1 : c <= 16 ? 2 : c <= 32 ? 4 : 8
    }

    static func prefetch(threadgroups: Int) -> Int {
        switch prefetchPolicy {
        case "none": return 1
        case "all": return 4
        default: return threadgroups < prefetchThreadgroups ? 4 : 1
        }
    }

    /// The MMA operand type for inputs of `x` and weights of `w` (`dense`:
    /// stored as is; quantized weights are E2M1 x E4M3 or E4M3 x 2^k values,
    /// exact in BF16).
    static func operandType(x: DType, weight: DType, dense: Bool) -> DType {
        guard narrowOperands else { return .float32 }
        if x == .bfloat16 && (!dense || weight == .bfloat16) { return .bfloat16 }
        if x == .float16 && dense && weight == .float16 { return .float16 }
        return .float32
    }

    static func supportsActivation(_ dtype: DType) -> Bool {
        dtype == .bfloat16 || dtype == .float16 || dtype == .float32
    }

    /// Helpers shared by every kernel: eight-element loads widened to FP32 and
    /// exact FP8 / FP4 decoders.
    static let header = """
        #include <metal_simdgroup_matrix>

        // Eight FP32 values.
        struct ink2_v8 {
            float4 lo;
            float4 hi;
        };

        // Eight consecutive elements widened to FP32 (16-byte aligned).
        template <typename T>
        inline ink2_v8 ink2_load8(const device T* p) {
            ink2_v8 r;
            if constexpr (metal::is_same_v<T, float>) {
                r.lo = ((const device float4*)p)[0];
                r.hi = ((const device float4*)p)[1];
            } else if constexpr (metal::is_same_v<T, half>) {
                r.lo = float4(((const device half4*)p)[0]);
                r.hi = float4(((const device half4*)p)[1]);
            } else {
                // bfloat16: the upper half of an FP32 word.
                const uint4 u = *((const device uint4*)p);
                r.lo = float4(
                    as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u),
                    as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
                r.hi = float4(
                    as_type<float>(u.z << 16), as_type<float>(u.z & 0xFFFF0000u),
                    as_type<float>(u.w << 16), as_type<float>(u.w & 0xFFFF0000u));
            }
            return r;
        }

        // E4M3 byte (low 8 bits, no NaN codes) times 2^-8, exact: the byte's
        // exponent and mantissa placed in an FP16 word.
        inline float ink2_e4m3_shifted(uint b) {
            return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8))));
        }

        // E2M1 nibble (low 4 bits) times 2^-14, exact.
        inline float ink2_e2m1_shifted(uint c) {
            return float(as_type<half>(ushort(((c & 8u) << 12) | ((c & 7u) << 9))));
        }

        // Two E2M1 nibbles in the low nibble of each 16-bit half of t, as FP32
        // pairs times 2^-14 (exact).
        inline float2 ink2_e2m1_pair(uint t) {
            const uint h = ((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9);
            return float2(as_type<half2>(h));
        }

        // Two E4M3 bytes in the low byte of each 16-bit half of t, as FP32 pairs
        // times 2^-8 (exact; no NaN codes).
        inline float2 ink2_e4m3_pair(uint t) {
            const uint h = ((t & 0x007F007Fu) << 7) | ((t & 0x00800080u) << 8);
            return float2(as_type<half2>(h));
        }

        // Block b (eight values) of one stored weight row and its block scale.
        // `row` points at the row's first weight element (dense), packed word
        // (quantized) and scale byte. FMT 0: dense WT. FMT 1: NVFP4 (E2M1 pairs,
        // low nibble first; E4M3 scale per 16; values carry 2^-14, the scale
        // 2^22). FMT 2: MXFP8 (E4M3 values carry 2^-8; E8M0 scale per 32
        // carries 2^8).
        template <typename WT, int FMT>
        inline ink2_v8 ink2_weights8(
            const device WT* row, const device uchar* srow, int b, thread float& s) {
            ink2_v8 q;
            if constexpr (FMT == 0) {
                s = 1.0f;
                q = ink2_load8<WT>(row + b * 8);
            } else if constexpr (FMT == 1) {
                const uint word = ((const device uint*)row)[b];
                s = ink2_e4m3_shifted(srow[b >> 1]) * 4194304.0f;
                const float2 p0 = ink2_e2m1_pair(word);
                const float2 p1 = ink2_e2m1_pair(word >> 4);
                const float2 p2 = ink2_e2m1_pair(word >> 8);
                const float2 p3 = ink2_e2m1_pair(word >> 12);
                q.lo = float4(p0.x, p1.x, p2.x, p3.x);
                q.hi = float4(p0.y, p1.y, p2.y, p3.y);
            } else {
                const uint2 words = ((const device uint2*)row)[b];
                s = as_type<float>((uint(srow[b >> 2]) + 8u) << 23);
                const float2 a0 = ink2_e4m3_pair(words.x);
                const float2 a1 = ink2_e4m3_pair(words.x >> 8);
                const float2 b0 = ink2_e4m3_pair(words.y);
                const float2 b1 = ink2_e4m3_pair(words.y >> 8);
                q.lo = float4(a0.x, a1.x, a0.y, a1.y);
                q.hi = float4(b0.x, b1.x, b0.y, b1.y);
            }
            return q;
        }

        // Element offset of a stored row: dense WT elements, packed words (as
        // WT = uint32) or scale bytes.
        template <int FMT, int K>
        inline size_t ink2_row_stride() {
            return FMT == 0 ? size_t(K) : FMT == 1 ? size_t(K / 8) : size_t(K / 4);
        }
        template <int FMT, int K>
        inline size_t ink2_scale_stride() {
            return FMT == 1 ? size_t(K / 16) : size_t(K / 32);
        }

        // One block's contribution to a lane accumulator. Dense: eight chained
        // FMAs in element order. Quantized: the eight-term partial (first
        // product, then seven FMAs), added as fma(scale, partial, a).
        template <int FMT>
        inline float ink2_accumulate(ink2_v8 w, float s, ink2_v8 x, float a) {
            if constexpr (FMT == 0) {
                a = fma(w.lo.x, x.lo.x, a);
                a = fma(w.lo.y, x.lo.y, a);
                a = fma(w.lo.z, x.lo.z, a);
                a = fma(w.lo.w, x.lo.w, a);
                a = fma(w.hi.x, x.hi.x, a);
                a = fma(w.hi.y, x.hi.y, a);
                a = fma(w.hi.z, x.hi.z, a);
                a = fma(w.hi.w, x.hi.w, a);
                return a;
            } else {
                float p = w.lo.x * x.lo.x;
                p = fma(w.lo.y, x.lo.y, p);
                p = fma(w.lo.z, x.lo.z, p);
                p = fma(w.lo.w, x.lo.w, p);
                p = fma(w.hi.x, x.hi.x, p);
                p = fma(w.hi.y, x.hi.y, p);
                p = fma(w.hi.z, x.hi.z, p);
                p = fma(w.hi.w, x.hi.w, p);
                return fma(s, p, a);
            }
        }

        // Stored values k0 .. k0 + 7 (k0 a multiple of 8) of one weight row
        // times their block scale: the exact dequantized values (no global scale).
        template <typename WT, int FMT, bool TBL>
        inline ink2_v8 ink2_dequant8(
            const device WT* row, const device uchar* srow, int k0, threadgroup const float2* lut) {
            if constexpr (FMT == 1 && TBL) {
                // NVFP4 through a byte -> (E2M1, E2M1) table: the same exact
                // products as the bit decode below.
                const uint word = ((const device uint*)row)[k0 / 8];
                const float s = ink2_e4m3_shifted(srow[k0 / 16]) * 256.0f;
                const float2 p0 = lut[word & 0xFFu];
                const float2 p1 = lut[(word >> 8) & 0xFFu];
                const float2 p2 = lut[(word >> 16) & 0xFFu];
                const float2 p3 = lut[word >> 24];
                ink2_v8 q;
                q.lo = float4(p0.x, p0.y, p1.x, p1.y) * s;
                q.hi = float4(p2.x, p2.y, p3.x, p3.y) * s;
                return q;
            } else {
                float s;
                ink2_v8 q = ink2_weights8<WT, FMT>(row, srow, k0 / 8, s);
                q.lo *= s;
                q.hi *= s;
                return q;
            }
        }

        // The loads of ink2_dequant8 alone (raw words and scale byte), so a
        // simdgroup can issue several superblocks' loads before it decodes any.
        struct ink2_raw8 {
            uint4 a;
            uint4 b;
            uint s;
        };
        template <typename WT, int FMT>
        inline ink2_raw8 ink2_load_raw8(const device WT* row, const device uchar* srow, int k0) {
            ink2_raw8 r = {};
            if constexpr (FMT == 0) {
                r.a = ((const device uint4*)(row + k0))[0];
                if constexpr (metal::is_same_v<WT, float>) {
                    r.b = ((const device uint4*)(row + k0))[1];
                }
            } else if constexpr (FMT == 1) {
                r.a.x = ((const device uint*)row)[k0 / 8];
                r.s = srow[k0 / 16];
            } else {
                const uint2 w = ((const device uint2*)row)[k0 / 8];
                r.a.x = w.x;
                r.a.y = w.y;
                r.s = srow[k0 / 32];
            }
            return r;
        }

        // ink2_dequant8's values from its raw loads: the same operations.
        template <typename WT, int FMT, bool TBL>
        inline ink2_v8 ink2_decode_raw8(ink2_raw8 r, threadgroup const float2* lut) {
            ink2_v8 q;
            if constexpr (FMT == 0) {
                const float s = 1.0f;
                if constexpr (metal::is_same_v<WT, float>) {
                    q.lo = as_type<float4>(r.a);
                    q.hi = as_type<float4>(r.b);
                } else if constexpr (metal::is_same_v<WT, half>) {
                    q.lo = float4(as_type<half4>(r.a.xy));
                    q.hi = float4(as_type<half4>(r.a.zw));
                } else {
                    const uint4 u = r.a;
                    q.lo = float4(
                        as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u),
                        as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
                    q.hi = float4(
                        as_type<float>(u.z << 16), as_type<float>(u.z & 0xFFFF0000u),
                        as_type<float>(u.w << 16), as_type<float>(u.w & 0xFFFF0000u));
                }
                q.lo *= s;
                q.hi *= s;
            } else if constexpr (FMT == 1 && TBL) {
                const uint word = r.a.x;
                const float s = ink2_e4m3_shifted(r.s) * 256.0f;
                const float2 p0 = lut[word & 0xFFu];
                const float2 p1 = lut[(word >> 8) & 0xFFu];
                const float2 p2 = lut[(word >> 16) & 0xFFu];
                const float2 p3 = lut[word >> 24];
                q.lo = float4(p0.x, p0.y, p1.x, p1.y) * s;
                q.hi = float4(p2.x, p2.y, p3.x, p3.y) * s;
            } else if constexpr (FMT == 1) {
                const uint word = r.a.x;
                const float s = ink2_e4m3_shifted(r.s) * 4194304.0f;
                const float2 p0 = ink2_e2m1_pair(word);
                const float2 p1 = ink2_e2m1_pair(word >> 4);
                const float2 p2 = ink2_e2m1_pair(word >> 8);
                const float2 p3 = ink2_e2m1_pair(word >> 12);
                q.lo = float4(p0.x, p1.x, p2.x, p3.x);
                q.hi = float4(p0.y, p1.y, p2.y, p3.y);
                q.lo *= s;
                q.hi *= s;
            } else {
                const uint2 words = r.a.xy;
                const float s = as_type<float>((r.s + 8u) << 23);
                const float2 a0 = ink2_e4m3_pair(words.x);
                const float2 a1 = ink2_e4m3_pair(words.x >> 8);
                const float2 b0 = ink2_e4m3_pair(words.y);
                const float2 b1 = ink2_e4m3_pair(words.y >> 8);
                q.lo = float4(a0.x, a1.x, a0.y, a1.y);
                q.hi = float4(b0.x, b1.x, b0.y, b1.y);
                q.lo *= s;
                q.hi *= s;
            }
            return q;
        }

        // The NVFP4 table: byte -> (E2M1 of the low nibble, of the high nibble).
        inline void ink2_fill_e2m1_table(threadgroup float2* lut, uint tid, uint nt) {
            for (uint i = tid; i < 256; i += nt) {
                lut[i] = float2(ink2_e2m1_shifted(i) * 16384.0f, ink2_e2m1_shifted(i >> 4) * 16384.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // OPT is the MMA operand type: bfloat or half when every input and
        // weight value is exact in it (the products and FP32 accumulation are
        // then those of FP32 operands), float otherwise.
        // One simdgroup's product of 8 weight rows (from `row0`) with up to
        // 8 * RB input rows, accumulated over this simdgroup's K superblocks
        // (32 values: s = ks, ks + SK, ...) with 8x8x8 FP32 MMAs, stored to
        // `buf` as RB row-major 8 (outputs) x 8 (rows) tiles.
        //
        // Lane (fm, fn) of the 8x8 fragment holds elements (fm, fn), (fm, fn + 1).
        // In superblock s, k-slot c of sub-block j (0..3) is the stored element
        // s * 32 + (c / 2) * 8 + 2 * j + (c & 1): each lane reads eight
        // consecutive weights and eight consecutive inputs per row. Every
        // output's sequence (superblocks, then sub-blocks, then one MMA) depends
        // on K and SK alone. Column `col` reads input row xrow(col); columns past
        // `count` repeat the last row and are never stored.
        template <
            typename T, typename WT, int FMT, int K, int SK, int RB, bool TBL = false,
            typename OPT = float, int PF = 1, typename XR>
        inline void ink2_mma_tile(
            const device T* x, XR xrow, int count, const device WT* row0,
            const device uchar* srow0, size_t rowStride, size_t scaleStride, uint lane, int ks,
            threadgroup float* buf, int rowsValid = 8, threadgroup const float2* lut = nullptr) {
            const int qid = int(lane) / 4;
            const int fm = (qid & 4) + ((int(lane) / 2) % 4);
            const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
            uint xo[2 * RB];
            #pragma unroll
            for (int t = 0; t < RB; t++) {
                xo[2 * t] = uint(xrow(min(t * 8 + fn, count - 1)) * K);
                xo[2 * t + 1] = uint(xrow(min(t * 8 + fn + 1, count - 1)) * K);
            }
            simdgroup_float8x8 acc[RB];
            #pragma unroll
            for (int t = 0; t < RB; t++) {
                acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            // Rows past `rowsValid` (a matrix of fewer than 8 rows) repeat the last
            // one; their outputs are never stored.
            const int wr = min(fm, rowsValid - 1);
            const device WT* wrow = row0 + size_t(wr) * rowStride;
            const device uchar* sr = srow0 + (FMT == 0 ? size_t(0) : size_t(wr) * scaleStride);
            const int pick = fm & 1;
            // PF > 1: PF superblocks' loads in flight before their decodes and
            // MMAs (latency hiding for small grids; every accumulator sees the
            // same MMAs in the same order as one superblock at a time).
            int s = ks;
            for (; PF > 1 && s + (PF - 1) * SK < K / 32; s += PF * SK) {
                ink2_raw8 raw[PF];
                #pragma unroll
                for (int j = 0; j < PF; j++) {
                    raw[j] = ink2_load_raw8<WT, FMT>(wrow, sr, (s + j * SK) * 32 + (fn / 2) * 8);
                }
                #pragma unroll
                for (int j = 0; j < PF; j++) {
                    const ink2_v8 a = ink2_decode_raw8<WT, FMT, TBL>(raw[j], lut);
                    simdgroup_matrix<OPT, 8, 8> A0, A1, A2, A3;
                    A0.thread_elements()[0] = OPT(a.lo.x);
                    A0.thread_elements()[1] = OPT(a.lo.y);
                    A1.thread_elements()[0] = OPT(a.lo.z);
                    A1.thread_elements()[1] = OPT(a.lo.w);
                    A2.thread_elements()[0] = OPT(a.hi.x);
                    A2.thread_elements()[1] = OPT(a.hi.y);
                    A3.thread_elements()[0] = OPT(a.hi.z);
                    A3.thread_elements()[1] = OPT(a.hi.w);
                    const int kx = (s + j * SK) * 32 + (fm / 2) * 8;
                    #pragma unroll
                    for (int t = 0; t < RB; t++) {
                        const ink2_v8 u = ink2_load8<T>(x + xo[2 * t] + kx);
                        const ink2_v8 v = ink2_load8<T>(x + xo[2 * t + 1] + kx);
                        simdgroup_matrix<OPT, 8, 8> B0, B1, B2, B3;
                        B0.thread_elements()[0] = OPT(pick ? u.lo.y : u.lo.x);
                        B0.thread_elements()[1] = OPT(pick ? v.lo.y : v.lo.x);
                        B1.thread_elements()[0] = OPT(pick ? u.lo.w : u.lo.z);
                        B1.thread_elements()[1] = OPT(pick ? v.lo.w : v.lo.z);
                        B2.thread_elements()[0] = OPT(pick ? u.hi.y : u.hi.x);
                        B2.thread_elements()[1] = OPT(pick ? v.hi.y : v.hi.x);
                        B3.thread_elements()[0] = OPT(pick ? u.hi.w : u.hi.z);
                        B3.thread_elements()[1] = OPT(pick ? v.hi.w : v.hi.z);
                        simdgroup_multiply_accumulate(acc[t], A0, B0, acc[t]);
                        simdgroup_multiply_accumulate(acc[t], A1, B1, acc[t]);
                        simdgroup_multiply_accumulate(acc[t], A2, B2, acc[t]);
                        simdgroup_multiply_accumulate(acc[t], A3, B3, acc[t]);
                    }
                }
            }
            for (; s < K / 32; s += SK) {
                const ink2_v8 a = ink2_dequant8<WT, FMT, TBL>(wrow, sr, s * 32 + (fn / 2) * 8, lut);
                simdgroup_matrix<OPT, 8, 8> A0, A1, A2, A3;
                A0.thread_elements()[0] = OPT(a.lo.x);
                A0.thread_elements()[1] = OPT(a.lo.y);
                A1.thread_elements()[0] = OPT(a.lo.z);
                A1.thread_elements()[1] = OPT(a.lo.w);
                A2.thread_elements()[0] = OPT(a.hi.x);
                A2.thread_elements()[1] = OPT(a.hi.y);
                A3.thread_elements()[0] = OPT(a.hi.z);
                A3.thread_elements()[1] = OPT(a.hi.w);
                const int kx = s * 32 + (fm / 2) * 8;
                #pragma unroll
                for (int t = 0; t < RB; t++) {
                    const ink2_v8 u = ink2_load8<T>(x + xo[2 * t] + kx);
                    const ink2_v8 v = ink2_load8<T>(x + xo[2 * t + 1] + kx);
                    simdgroup_matrix<OPT, 8, 8> B0, B1, B2, B3;
                    B0.thread_elements()[0] = OPT(pick ? u.lo.y : u.lo.x);
                    B0.thread_elements()[1] = OPT(pick ? v.lo.y : v.lo.x);
                    B1.thread_elements()[0] = OPT(pick ? u.lo.w : u.lo.z);
                    B1.thread_elements()[1] = OPT(pick ? v.lo.w : v.lo.z);
                    B2.thread_elements()[0] = OPT(pick ? u.hi.y : u.hi.x);
                    B2.thread_elements()[1] = OPT(pick ? v.hi.y : v.hi.x);
                    B3.thread_elements()[0] = OPT(pick ? u.hi.w : u.hi.z);
                    B3.thread_elements()[1] = OPT(pick ? v.hi.w : v.hi.z);
                    simdgroup_multiply_accumulate(acc[t], A0, B0, acc[t]);
                    simdgroup_multiply_accumulate(acc[t], A1, B1, acc[t]);
                    simdgroup_multiply_accumulate(acc[t], A2, B2, acc[t]);
                    simdgroup_multiply_accumulate(acc[t], A3, B3, acc[t]);
                }
            }
            #pragma unroll
            for (int t = 0; t < RB; t++) {
                simdgroup_store(acc[t], buf + t * 64, 8);
            }
        }

        // One output of ink2_mma_tile computed by one thread: the same FMA
        // sequence. A simdgroup MMA with exact (bfloat/half) operands adds its
        // eight products one at a time in k-slot order (probe: bit-equal to a
        // sequential fma chain), so split-K chain `ks` takes superblocks
        // ks, ks + SK, ..., sub-blocks j = 0..3, k-slots c = 0..7 at stored
        // element s * 32 + (c / 2) * 8 + 2 * j + (c & 1); chains add in order.
        template <typename T, typename WT, int FMT, int K, int SK, typename OPT>
        inline float ink2_row_dot(
            const device T* xr, const device WT* wrow, const device uchar* srow) {
            float acc[SK];
            #pragma unroll
            for (int i = 0; i < SK; i++) {
                acc[i] = 0.0f;
            }
            for (int s0 = 0; s0 < K / 32; s0 += SK) {
                #pragma unroll
                for (int ks = 0; ks < SK; ks++) {
                    const int s = s0 + ks;
                    if (s < K / 32) {
                        float w[32];
                        float v[32];
                        #pragma unroll
                        for (int b = 0; b < 4; b++) {
                            const ink2_v8 q = ink2_dequant8<WT, FMT, false>(wrow, srow, s * 32 + b * 8, nullptr);
                            const ink2_v8 u = ink2_load8<T>(xr + s * 32 + b * 8);
                            w[b * 8 + 0] = q.lo.x; w[b * 8 + 1] = q.lo.y; w[b * 8 + 2] = q.lo.z; w[b * 8 + 3] = q.lo.w;
                            w[b * 8 + 4] = q.hi.x; w[b * 8 + 5] = q.hi.y; w[b * 8 + 6] = q.hi.z; w[b * 8 + 7] = q.hi.w;
                            v[b * 8 + 0] = u.lo.x; v[b * 8 + 1] = u.lo.y; v[b * 8 + 2] = u.lo.z; v[b * 8 + 3] = u.lo.w;
                            v[b * 8 + 4] = u.hi.x; v[b * 8 + 5] = u.hi.y; v[b * 8 + 6] = u.hi.z; v[b * 8 + 7] = u.hi.w;
                        }
                        float a = acc[ks];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            #pragma unroll
                            for (int c = 0; c < 8; c++) {
                                const int k = (c / 2) * 8 + 2 * j + (c & 1);
                                a = fma(float(OPT(w[k])), float(OPT(v[k])), a);
                            }
                        }
                        acc[ks] = a;
                    }
                }
            }
            float total = acc[0];
            #pragma unroll
            for (int i = 1; i < SK; i++) {
                total += acc[i];
            }
            return total;
        }

        // Split-K chain `ks` of ink2_row_dot alone (superblocks ks, ks + SK, ...),
        // with the next superblock's raw loads issued before this one's FMAs.
        // Threads computing the SK chains of one output add them in order.
        // Input values of one row, eight at a time, as exact operand floats.
        template <typename T>
        struct ink2_xl_plain {
            const device T* xr;
            inline ink2_v8 load8(int k0) const { return ink2_load8<T>(xr + k0); }
        };

        template <typename WT, int FMT, int K, int SK, typename OPT, typename XL>
        inline float ink2_row_chain_x(
            XL xl, const device WT* wrow, const device uchar* srow, int ks) {
            float a = 0.0f;
            if (ks >= K / 32) {
                return a;
            }
            ink2_raw8 cw[4];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                cw[b] = ink2_load_raw8<WT, FMT>(wrow, srow, ks * 32 + b * 8);
            }
            for (int s = ks; s < K / 32; s += SK) {
                ink2_raw8 nw[4];
                const int sn = s + SK;
                if (sn < K / 32) {
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        nw[b] = ink2_load_raw8<WT, FMT>(wrow, srow, sn * 32 + b * 8);
                    }
                }
                float w[32];
                float v[32];
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    const ink2_v8 q = ink2_decode_raw8<WT, FMT, false>(cw[b], nullptr);
                    const ink2_v8 u = xl.load8(s * 32 + b * 8);
                    w[b * 8 + 0] = q.lo.x; w[b * 8 + 1] = q.lo.y; w[b * 8 + 2] = q.lo.z; w[b * 8 + 3] = q.lo.w;
                    w[b * 8 + 4] = q.hi.x; w[b * 8 + 5] = q.hi.y; w[b * 8 + 6] = q.hi.z; w[b * 8 + 7] = q.hi.w;
                    v[b * 8 + 0] = u.lo.x; v[b * 8 + 1] = u.lo.y; v[b * 8 + 2] = u.lo.z; v[b * 8 + 3] = u.lo.w;
                    v[b * 8 + 4] = u.hi.x; v[b * 8 + 5] = u.hi.y; v[b * 8 + 6] = u.hi.z; v[b * 8 + 7] = u.hi.w;
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        const int k = (c / 2) * 8 + 2 * j + (c & 1);
                        a = fma(float(OPT(w[k])), float(OPT(v[k])), a);
                    }
                }
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    cw[b] = nw[b];
                }
            }
            return a;
        }

        template <typename T, typename WT, int FMT, int K, int SK, typename OPT>
        inline float ink2_row_chain(
            const device T* xr, const device WT* wrow, const device uchar* srow, int ks) {
            return ink2_row_chain_x<WT, FMT, K, SK, OPT>(ink2_xl_plain<T>{xr}, wrow, srow, ks);
        }

        // Input row of a tile column: contiguous rows from c0, or the pair list
        // of an expert work item (gate/up reads token row pair / TOPK, down reads
        // the pair's activation row).
        struct ink2_rows_contiguous {
            int c0;
            int operator()(int c) const { return c0 + c; }
        };
        template <typename PS, int DIV>
        struct ink2_rows_pairs {
            PS p;
            int operator()(int c) const { return int(p[c]) / DIV; }
        };

        // Lane 0 receives the sum of the 32 lanes, always in this order.
        inline float ink2_simd_total(float v) {
            v += simd_shuffle_down(v, 16);
            v += simd_shuffle_down(v, 8);
            v += simd_shuffle_down(v, 4);
            v += simd_shuffle_down(v, 2);
            v += simd_shuffle_down(v, 1);
            return v;
        }
        """
}
