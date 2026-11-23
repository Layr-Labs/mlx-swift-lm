// Lossless 4-bit-exponent BF16 weights inside rowProduct's MMA tile (1 to 4 rows).
import Foundation
import MLX

/// A BF16 `[N, K]` matrix stored as one sign|mantissa byte and one 4-bit exponent code per weight
/// (75 % of the bytes), laid out row-major so `rowProduct`'s MMA tile reads each lane's eight
/// weights with one 8-byte and one 4-byte load. Code 15 marks an escape (0.02 % of Inkling 2
/// Mini's projection weights): its exponent sits in the row's sorted `(column << 8 | exponent)`
/// list, found by binary search. The decoded values are the stored BF16 values, so the MMA
/// inputs, and every output bit, equal the raw tile's. Built in memory from the loaded weight
/// and checked against it on the GPU before use.
public final class Inkling2BytesRows: @unchecked Sendable {
    public let rows: Int
    public let columns: Int
    public let base: Int
    let signMantissa: MLXArray
    let codes: MLXArray
    let escapes: MLXArray
    let rowOffsets: MLXArray

    public var byteCount: Int {
        signMantissa.nbytes + codes.nbytes + escapes.nbytes + rowOffsets.nbytes
    }

    public init(_ weight: MLXArray) throws {
        guard weight.ndim == 2, weight.dtype == .bfloat16, weight.dim(1) % 32 == 0 else {
            throw Inkling2Bytes.Failure.unsupported("coded rows need bf16 [N, K], K % 32 == 0")
        }
        let (n, k) = (weight.dim(0), weight.dim(1))
        rows = n
        columns = k
        let bits = weight.view(dtype: .uint16).asArray(UInt16.self)
        let base = bits.withUnsafeBufferPointer { Inkling2Bytes.bestBase($0, exponentBits: 4) }
        self.base = base
        var sm = [UInt8](repeating: 0, count: n * k)
        var codes = [UInt8](repeating: 0, count: n * k / 2)
        var escapes: [UInt32] = []
        var offsets = [UInt32](repeating: 0, count: n + 1)
        for row in 0 ..< n {
            offsets[row] = UInt32(escapes.count)
            for column in 0 ..< k {
                let w = bits[row * k + column]
                let e = Int((w >> 7) & 0xFF)
                sm[row * k + column] = UInt8(((w >> 8) & 0x80) | (w & 0x7F))
                var c = e - base
                if c < 0 || c >= 15 {
                    c = 15
                    escapes.append(UInt32(column) << 8 | UInt32(e))
                }
                codes[(row * k + column) / 2] |= UInt8(c) << (4 * (column & 1))
            }
        }
        offsets[n] = UInt32(escapes.count)
        if escapes.isEmpty { escapes = [0] }
        signMantissa = MLXArray(sm, [n, k])
        self.codes = MLXArray(codes, [n, k / 2])
        self.escapes = MLXArray(escapes, [escapes.count])
        rowOffsets = MLXArray(offsets, [n + 1])
        let decoded = Self.decodeKernel(
            [signMantissa, self.codes, self.escapes, rowOffsets],
            template: [("N", n), ("K", k), ("BASE", base)],
            grid: ((n * k / 8 + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[n, k]], outputDTypes: [.uint16])[0]
        guard all(decoded .== weight.view(dtype: .uint16)).item(Bool.self) else {
            throw Inkling2Bytes.Failure.notExact("coded rows [\(n), \(k)]")
        }
    }

    static let header =
        Inkling2Kernels.header + """

            // Exponent of row `row`, column `col` from its sorted escape list.
            inline uint bytes_escape(
                const device uint* esc, const device uint* offs, uint row, uint col) {
                uint lo = offs[row];
                uint hi = offs[row + 1];
                while (lo + 1 < hi) {
                    const uint mid = (lo + hi) / 2;
                    if ((esc[mid] >> 8) <= col) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return esc[lo] & 0xFFu;
            }

            // Eight stored BF16 weights of `row` from column k0 (a multiple of 8), as FP32 (exact).
            template <int K, int BASE>
            inline ink2_v8 bytes_k4_load8(
                const device uchar* sm, const device uchar* codes, const device uint* esc,
                const device uint* offs, uint row, int k0) {
                const uint2 s = *((const device uint2*)(sm + size_t(row) * K + k0));
            const uint c = *((const device uint*)(codes + size_t(row) * (K / 2) + k0 / 2));
            float v[8];
            if (((c & (c >> 1) & (c >> 2) & (c >> 3)) & 0x11111111u) == 0u) {
                // No escape: two BF16 values per 32-bit operation.
                const uint b2 = uint(BASE) * 0x10001u;
                const uint words[2] = {s.x, s.y};
                for (int h = 0; h < 2; h++) {
                    const uint w = words[h];
                    const uint cc = c >> (16 * h);
                    const uint s01 = (w & 0xFFu) | ((w & 0xFF00u) << 8);
                    const uint s23 = ((w >> 16) & 0xFFu) | ((w >> 8) & 0xFF0000u);
                    const uint e01 = ((cc & 0xFu) | ((cc & 0xF0u) << 12)) + b2;
                    const uint e23 = (((cc >> 8) & 0xFu) | ((cc << 4) & 0xF0000u)) + b2;
                    const uint p01 = ((s01 & 0x00800080u) << 8) | (s01 & 0x007F007Fu) | (e01 << 7);
                    const uint p23 = ((s23 & 0x00800080u) << 8) | (s23 & 0x007F007Fu) | (e23 << 7);
                    v[4 * h] = as_type<float>(p01 << 16);
                    v[4 * h + 1] = as_type<float>(p01 & 0xFFFF0000u);
                    v[4 * h + 2] = as_type<float>(p23 << 16);
                    v[4 * h + 3] = as_type<float>(p23 & 0xFFFF0000u);
                }
            } else {
                for (int t = 0; t < 8; t++) {
                    const uint b = ((t < 4 ? s.x : s.y) >> (8 * (t & 3))) & 0xFFu;
                    const uint code = (c >> (4 * t)) & 15u;
                    const uint e = code == 15u ? bytes_escape(esc, offs, row, uint(k0 + t))
                                               : uint(BASE) + code;
                    v[t] = as_type<float>((((b & 0x80u) << 8) | (e << 7) | (b & 0x7Fu)) << 16);
                }
            }
            ink2_v8 r;
                r.lo = float4(v[0], v[1], v[2], v[3]);
                r.hi = float4(v[4], v[5], v[6], v[7]);
                return r;
            }
            """

    private static let decodeKernel = MLXFast.metalKernel(
        name: "inkling2_bytes_rows_decode",
        inputNames: ["sm", "codes", "esc", "offs"],
        outputNames: ["out"],
        source: """
            const uint chunk = thread_position_in_grid.x;
            if (chunk >= uint(N * K / 8)) {
                return;
            }
            const uint row = chunk / (K / 8);
            const int k0 = int(chunk % (K / 8)) * 8;
            const ink2_v8 v = bytes_k4_load8<K, BASE>(sm, codes, esc, offs, row, k0);
            device ushort* dst = out + size_t(row) * K + k0;
            dst[0] = ushort(as_type<uint>(v.lo.x) >> 16);
            dst[1] = ushort(as_type<uint>(v.lo.y) >> 16);
            dst[2] = ushort(as_type<uint>(v.lo.z) >> 16);
            dst[3] = ushort(as_type<uint>(v.lo.w) >> 16);
            dst[4] = ushort(as_type<uint>(v.hi.x) >> 16);
            dst[5] = ushort(as_type<uint>(v.hi.y) >> 16);
            dst[6] = ushort(as_type<uint>(v.hi.z) >> 16);
            dst[7] = ushort(as_type<uint>(v.hi.w) >> 16);
            """,
        header: header)

    /// `rowProduct`'s tiled kernel with the coded weight load: same geometry, MMA sequence,
    /// split-K order and rounding, so `[rows, N]` equals `rowProduct(x, W)` bit for bit.
    private static let productKernel = MLXFast.metalKernel(
        name: "inkling2_bytes_rows_product",
        inputNames: ["x", "sm", "codes", "esc", "offs"],
        outputNames: ["out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int ks = sg % SK;
            const int ns = sg / SK;
            const int rows = x_shape[0];
            const int groups = (rows + 63) / 64;
            const int group = int(threadgroup_position_in_grid.x) % groups;
            const int tile = int(threadgroup_position_in_grid.x) / groups;
            const int start = group * 64;
            const int count = min(64, rows - start);
            threadgroup float buf[SK * NSG * 8 * 64];
            threadgroup float* mine = buf + (ks * NSG + ns) * 8 * 64;
            const int nb = min(tile * 8 * NSG + ns * 8, N - 8);
            const int rb = count <= 8 ? 1 : count <= 16 ? 2 : count <= 32 ? 4 : 8;
            const int qid = int(lane) / 4;
            const int fm = (qid & 4) + ((int(lane) / 2) % 4);
            const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
            const int pick = fm & 1;
            const uint wrow = uint(nb + fm);
            // Up to 8 input rows (RB 1): the only tile size the dispatcher sends here.
            uint xo[2];
            xo[0] = uint((start + min(fn, count - 1)) * K);
            xo[1] = uint((start + min(fn + 1, count - 1)) * K);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int s = ks; s < K / 32; s += SK) {
                const ink2_v8 a = bytes_k4_load8<K, BASE>(sm, codes, esc, offs, wrow,
                    s * 32 + (fn / 2) * 8);
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
                const ink2_v8 u = ink2_load8<T>(x + xo[0] + kx);
                const ink2_v8 v = ink2_load8<T>(x + xo[1] + kx);
                simdgroup_float8x8 B0, B1, B2, B3;
                B0.thread_elements()[0] = pick ? u.lo.y : u.lo.x;
                B0.thread_elements()[1] = pick ? v.lo.y : v.lo.x;
                B1.thread_elements()[0] = pick ? u.lo.w : u.lo.z;
                B1.thread_elements()[1] = pick ? v.lo.w : v.lo.z;
                B2.thread_elements()[0] = pick ? u.hi.y : u.hi.x;
                B2.thread_elements()[1] = pick ? v.hi.y : v.hi.x;
                B3.thread_elements()[0] = pick ? u.hi.w : u.hi.z;
                B3.thread_elements()[1] = pick ? v.hi.w : v.hi.z;
                simdgroup_multiply_accumulate(acc, A0, B0, acc);
                simdgroup_multiply_accumulate(acc, A1, B1, acc);
                simdgroup_multiply_accumulate(acc, A2, B2, acc);
                simdgroup_multiply_accumulate(acc, A3, B3, acc);
            }
            simdgroup_store(acc, mine, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int cols = rb * 8;
            const int total = NSG * cols * 8;
            for (int e = int(thread_index_in_threadgroup); e < total;
                 e += int(threads_per_threadgroup.x)) {
                const int nl = e % 8;
                const int c = (e / 8) % cols;
                const int sn = e / (8 * cols);
                if (c >= count) {
                    continue;
                }
                const int first = tile * 8 * NSG + sn * 8;
                const int n = min(first, N - 8) + nl;
                if (n < first) {
                    continue;
                }
                const int at = ((c / 8) * 8 + nl) * 8 + (c % 8);
                float value = buf[(0 * NSG + sn) * 8 * 64 + at];
                for (int s = 1; s < SK; s++) {
                    value += buf[(s * NSG + sn) * 8 * 64 + at];
                }
                out[size_t(start + c) * N + n] = static_cast<T>(value);
            }
            """,
        header: header)

    /// `x . W^T` for `x` `[R, K]` bf16, 1 <= R <= 8, equal to `rowProduct(x, W)` bit for bit.
    public func product(_ x: MLXArray) -> MLXArray {
        precondition(
            x.ndim == 2 && x.dim(1) == columns && x.dtype == .bfloat16
                && (1 ... 8).contains(x.dim(0)),
            "coded rows product needs bf16 [1...8, K]")
        let nsg = Inkling2Kernels.outputSimdgroups(rows)
        let tiles = (rows + 8 * nsg - 1) / (8 * nsg)
        let sk = Inkling2Kernels.splitK(columns)
        let threads = 32 * nsg * sk
        return Self.productKernel(
            [x, signMantissa, codes, escapes, rowOffsets],
            template: [
                ("T", DType.bfloat16), ("N", rows), ("K", columns), ("NSG", nsg), ("SK", sk),
                ("BASE", base),
            ],
            grid: (tiles * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[x.dim(0), rows]], outputDTypes: [.bfloat16])[0]
    }
}

/// Coded projections for row-exact steps of at most `maxRows` rows, keyed by weight identity.
/// `DARKBLOOM_INKLING2_CODED_PROJECTIONS=1` turns them on (off by default until measured).
public enum Inkling2BytesProjection {
    public static let maxRows = 4
    public static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_CODED_PROJECTIONS"]?
            .lowercased()
        return raw.map { ["1", "true", "on", "yes"].contains($0) } ?? false
    }()
    @TaskLocal public static var override: Bool? = nil
    public static var active: Bool { override ?? enabled }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var codes:
        [ObjectIdentifier: (MLXArray, Inkling2BytesRows?)] = [:]

    /// The coded product of `x` (`[R, K]`) with `w`, or nil (off, too many rows, unsupported).
    public static func product(_ x: MLXArray, _ w: MLXArray) -> MLXArray? {
        guard active, x.ndim == 2, (1 ... maxRows).contains(x.dim(0)), x.dtype == .bfloat16,
            w.dtype == .bfloat16, w.ndim == 2, w.dim(0) >= 8, w.dim(1) % 32 == 0
        else { return nil }
        return coded(w)?.product(x)
    }

    /// Code for `w`, built on first use; the weight is retained with it so its identity stays unique.
    public static func coded(_ w: MLXArray) -> Inkling2BytesRows? {
        lock.withLock {
            let id = ObjectIdentifier(w)
            if let entry = codes[id] { return entry.1 }
            let built = try? Inkling2BytesRows(w)
            codes[id] = (w, built)
            return built
        }
    }
}
