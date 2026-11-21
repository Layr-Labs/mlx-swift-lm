// Lossless fixed-width coding of BF16 weight matrices for decode-time reads (research; not wired
// into the model).
import Foundation
import MLX

/// A BF16 matrix `[N, K]` stored in fewer bytes with every bit kept.
///
/// BF16 = sign (1) | exponent (8) | mantissa (7). Sign and mantissa are stored as is (one byte per
/// weight); the exponent as a `k`-bit code (`k` = 3 or 4) relative to a per-matrix window
/// `base ..< base + 2^k - 1`; code `2^k - 1` marks an escape whose exponent sits in `escapes`.
/// Inkling 2 Mini's BF16 projections: about 2.2 % escapes at `k` = 3 (11.2 bits per weight, 70 % of
/// BF16) and 0.02 % at `k` = 4 (12 bits, 75 %).
///
/// Tiled layout (rows padded to a multiple of 4 with in-window rows that are never written), per
/// (row group `g` of 4 rows, 128-column block `b`, lane `l` in 0 ..< 32) — exactly the 16 weights one
/// thread of the row-exact gemv reads per iteration:
/// - `signMantissa`: 4 `uint32`, word `r` = the bytes of row `4g + r`, columns `128b + 4l ..< +4`;
/// - `codes`: the 16 exponent codes, row `r` at bit `12r` (`k` = 3, three `uint16`) or `16r`
///   (`k` = 4, two `uint32`), column `j` at bit `k·j` within;
/// - `escapes`: escaped exponents, row-major, column order; `rowOffsets[row]` is the row's first.
///
/// The code is derived in memory from the verified checkpoint tensor and must pass `verify` (decode
/// on the GPU, compared bit for bit with the source) before use; nothing is written to disk.
public enum Inkling2Bytes {
    public static let block = 128

    public struct Code {
        public let rows: Int
        public let columns: Int
        public let exponentBits: Int
        public let base: Int
        public let signMantissa: MLXArray
        public let codes: MLXArray
        public let escapes: MLXArray
        public let rowOffsets: MLXArray
        public let escapeCount: Int

        public var paddedRows: Int { (rows + 3) / 4 * 4 }
        public var byteCount: Int {
            signMantissa.nbytes + codes.nbytes + escapes.nbytes + rowOffsets.nbytes
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case unsupported(String)
        case notExact(String)

        public var description: String {
            switch self {
            case .unsupported(let s): "Inkling2Bytes unsupported: \(s)"
            case .notExact(let s): "Inkling2Bytes decode differs from the source: \(s)"
            }
        }
    }

    /// Window base holding the most exponents (`2^k - 1` consecutive values).
    static func bestBase(_ bits: UnsafeBufferPointer<UInt16>, exponentBits: Int) -> Int {
        var histogram = [Int](repeating: 0, count: 256)
        for b in bits {
            histogram[Int((b >> 7) & 0xFF)] += 1
        }
        let width = (1 << exponentBits) - 1
        var best = 0
        var bestCount = -1
        var inWindow = 0
        for e in 0 ..< 256 {
            inWindow += histogram[e]
            if e >= width {
                inWindow -= histogram[e - width]
            }
            if e >= width - 1, inWindow > bestCount {
                bestCount = inWindow
                best = e - width + 1
            }
        }
        return best
    }

    /// Encodes a BF16 `[N, K]` matrix (`K` a multiple of 128, `N` >= 4).
    public static func encode(_ weight: MLXArray, exponentBits: Int = 3) throws -> Code {
        guard weight.ndim == 2, weight.dtype == .bfloat16 else {
            throw Failure.unsupported(
                "need a 2-D bfloat16 matrix, got \(weight.shape) \(weight.dtype)")
        }
        let (n, k) = (weight.dim(0), weight.dim(1))
        guard k % block == 0, n >= 4, exponentBits == 3 || exponentBits == 4 else {
            throw Failure.unsupported("shape [\(n), \(k)] with \(exponentBits)-bit codes")
        }
        let bits = weight.view(dtype: .uint16).asArray(UInt16.self)
        return bits.withUnsafeBufferPointer {
            encode($0, rows: n, columns: k, exponentBits: exponentBits)
        }
    }

    static func encode(
        _ bits: UnsafeBufferPointer<UInt16>, rows n: Int, columns k: Int, exponentBits kb: Int
    ) -> Code {
        let base = bestBase(bits, exponentBits: kb)
        let n4 = (n + 3) / 4 * 4
        let lanes = n4 / 4 * (k / block) * 32
        var counts = [Int](repeating: 0, count: n4)
        var offsets = [UInt32](repeating: 0, count: n4)
        var signMantissa = [UInt32](repeating: 0, count: lanes * 4)
        var codes3 = [UInt16](repeating: 0, count: kb == 3 ? lanes * 3 : 0)
        var codes4 = [UInt32](repeating: 0, count: kb == 4 ? lanes * 2 : 0)
        var escapes = [UInt8]()
        var total = 0
        counts.withUnsafeMutableBufferPointer { counts in
            let pass = EncodePass(
                bits: bits, rows: n, columns: k, exponentBits: kb, base: base, counts: counts)
            DispatchQueue.concurrentPerform(iterations: pass.tasks) { pass.count(task: $0) }
        }
        for row in 0 ..< n4 {
            offsets[row] = UInt32(total)
            total += counts[row]
        }
        escapes = [UInt8](repeating: 0, count: max(1, total))
        counts.withUnsafeMutableBufferPointer { counts in
            offsets.withUnsafeMutableBufferPointer { offsets in
                signMantissa.withUnsafeMutableBufferPointer { sm in
                    codes3.withUnsafeMutableBufferPointer { c3 in
                        codes4.withUnsafeMutableBufferPointer { c4 in
                            escapes.withUnsafeMutableBufferPointer { esc in
                                let pass = EncodePass(
                                    bits: bits, rows: n, columns: k, exponentBits: kb, base: base,
                                    counts: counts, offsets: offsets, signMantissa: sm, codes3: c3,
                                    codes4: c4, escapes: esc)
                                DispatchQueue.concurrentPerform(iterations: pass.tasks) {
                                    pass.fill(task: $0)
                                }
                            }
                        }
                    }
                }
            }
        }
        return Code(
            rows: n, columns: k, exponentBits: kb, base: base,
            signMantissa: MLXArray(signMantissa, [lanes * 4]),
            codes: kb == 3 ? MLXArray(codes3, [lanes * 3]) : MLXArray(codes4, [lanes * 2]),
            escapes: MLXArray(escapes, [escapes.count]),
            rowOffsets: MLXArray(offsets, [n4]),
            escapeCount: total)
    }

    /// One encoding pass over disjoint row groups (each task owns its rows' outputs).
    private struct EncodePass: @unchecked Sendable {
        let bits: UnsafeBufferPointer<UInt16>
        let rows: Int
        let columns: Int
        let exponentBits: Int
        let base: Int
        var counts: UnsafeMutableBufferPointer<Int>
        var offsets = UnsafeMutableBufferPointer<UInt32>(start: nil, count: 0)
        var signMantissa = UnsafeMutableBufferPointer<UInt32>(start: nil, count: 0)
        var codes3 = UnsafeMutableBufferPointer<UInt16>(start: nil, count: 0)
        var codes4 = UnsafeMutableBufferPointer<UInt32>(start: nil, count: 0)
        var escapes = UnsafeMutableBufferPointer<UInt8>(start: nil, count: 0)

        static let groupsPerTask = 64
        var groups: Int { (rows + 3) / 4 }
        var tasks: Int { (groups + Self.groupsPerTask - 1) / Self.groupsPerTask }
        var escapeCode: Int { (1 << exponentBits) - 1 }

        func word(_ row: Int, _ column: Int) -> UInt16 {
            row < rows ? bits[row * columns + column] : UInt16(base << 7)
        }

        func count(task: Int) {
            let groupRange =
                (task * Self.groupsPerTask) ..< min(groups, (task + 1) * Self.groupsPerTask)
            for row in (groupRange.lowerBound * 4) ..< (groupRange.upperBound * 4) {
                var c = 0
                for column in 0 ..< columns {
                    let e = Int((word(row, column) >> 7) & 0xFF) - base
                    if e < 0 || e >= escapeCode {
                        c += 1
                    }
                }
                counts[row] = c
            }
        }

        func fill(task: Int) {
            let blocks = columns / Inkling2Bytes.block
            let kb = exponentBits
            for g in (task * Self.groupsPerTask) ..< min(groups, (task + 1) * Self.groupsPerTask) {
                var cursor = (0 ..< 4).map { Int(offsets[4 * g + $0]) }
                for b in 0 ..< blocks {
                    for lane in 0 ..< 32 {
                        let index = (g * blocks + b) * 32 + lane
                        var packed: UInt64 = 0
                        for r in 0 ..< 4 {
                            var smWord: UInt32 = 0
                            var chunk: UInt64 = 0
                            for j in 0 ..< 4 {
                                let w = word(4 * g + r, b * Inkling2Bytes.block + lane * 4 + j)
                                let e = Int((w >> 7) & 0xFF)
                                smWord |= UInt32(((w >> 8) & 0x80) | (w & 0x7F)) << (8 * j)
                                var c = e - base
                                if c < 0 || c >= escapeCode {
                                    c = escapeCode
                                    escapes[cursor[r]] = UInt8(e)
                                    cursor[r] += 1
                                }
                                chunk |= UInt64(c) << (kb * j)
                            }
                            signMantissa[index * 4 + r] = smWord
                            packed |= chunk << ((kb == 3 ? 12 : 16) * r)
                        }
                        if kb == 3 {
                            codes3[index * 3] = UInt16(packed & 0xFFFF)
                            codes3[index * 3 + 1] = UInt16((packed >> 16) & 0xFFFF)
                            codes3[index * 3 + 2] = UInt16((packed >> 32) & 0xFFFF)
                        } else {
                            codes4[index * 2] = UInt32(packed & 0xFFFF_FFFF)
                            codes4[index * 2 + 1] = UInt32(packed >> 32)
                        }
                    }
                }
            }
        }
    }

    /// Encodes `weight` and proves on the GPU that decoding returns every bit of it.
    public static func encodeVerified(_ weight: MLXArray, exponentBits: Int = 3) throws -> Code {
        let code = try encode(weight, exponentBits: exponentBits)
        try verify(code, against: weight)
        return code
    }

    /// Throws unless the GPU decode of `code` equals `weight` bit for bit.
    public static func verify(_ code: Code, against weight: MLXArray) throws {
        let decoded = decode(code)
        let equal = all(decoded.view(dtype: .uint16) .== weight.view(dtype: .uint16)).item(
            Bool.self)
        guard equal, decoded.shape == weight.shape else {
            throw Failure.notExact("[\(code.rows), \(code.columns)] k=\(code.exponentBits)")
        }
    }
}
