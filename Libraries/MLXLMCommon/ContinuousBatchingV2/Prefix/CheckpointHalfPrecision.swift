import Foundation

/// Lossless half-precision wire encoding for checkpoint payloads that a live
/// engine keeps in FP32 but that were produced in a half-precision activation
/// dtype (the FP32 copy only widened them). Narrowing refuses unless every
/// element is exactly representable in the wire dtype, so widening always
/// reproduces the original FP32 bits. Choosing the wire dtype from the
/// producing computation is the caller's job; this helper never rounds.
///
/// BF16 is the top half of an FP32 word, so its check is "low half is zero"
/// and every bit pattern, including NaN payloads and signed zero, survives.
/// FP16 compares the round trip bit for bit, which refuses any value FP16
/// would round, flush or overflow, and any NaN payload it cannot carry.
public enum CBv2CheckpointHalfPrecision {
    public static func isHalf(_ dtype: CBv2CheckpointDType) -> Bool {
        dtype == .bfloat16 || dtype == .float16
    }

    /// Narrows `count` FP32 elements at `source` to `wire` at `destination`.
    /// Returns false when any element is not exactly representable; the
    /// destination bytes are then unspecified and must be discarded. Neither
    /// pointer needs more than byte alignment.
    public static func narrow(
        float32 source: UnsafeRawPointer, count: Int, to wire: CBv2CheckpointDType,
        destination: UnsafeMutableRawPointer
    ) -> Bool {
        precondition(count >= 0 && isHalf(wire))
        let lanes = 8
        let vectorCount = count / lanes * lanes
        if wire == .bfloat16 {
            var dropped = SIMD8<UInt32>()
            var index = 0
            while index < vectorCount {
                let words = source.loadUnaligned(fromByteOffset: index * 4, as: SIMD8<UInt32>.self)
                dropped |= words & 0xFFFF
                destination.storeBytes(
                    of: SIMD8<UInt16>(truncatingIfNeeded: words &>> 16),
                    toByteOffset: index * 2, as: SIMD8<UInt16>.self)
                index += lanes
            }
            var tail: UInt32 = 0
            while index < count {
                let word = source.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                tail |= word & 0xFFFF
                destination.storeBytes(
                    of: UInt16(truncatingIfNeeded: word >> 16), toByteOffset: index * 2,
                    as: UInt16.self)
                index += 1
            }
            return dropped == SIMD8<UInt32>() && tail == 0
        }
        var exact = true
        var index = 0
        while index < vectorCount {
            let words = source.loadUnaligned(fromByteOffset: index * 4, as: SIMD8<UInt32>.self)
            let values = unsafeBitCast(words, to: SIMD8<Float>.self)
            let narrowed = SIMD8<Float16>(values)
            let back = unsafeBitCast(SIMD8<Float>(narrowed), to: SIMD8<UInt32>.self)
            exact = exact && back == words
            destination.storeBytes(of: narrowed, toByteOffset: index * 2, as: SIMD8<Float16>.self)
            index += lanes
        }
        while index < count {
            let word = source.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
            let narrowed = Float16(Float(bitPattern: word))
            exact = exact && Float(narrowed).bitPattern == word
            destination.storeBytes(of: narrowed, toByteOffset: index * 2, as: Float16.self)
            index += 1
        }
        return exact
    }

    /// Widens `count` `wire` elements at `source` to FP32 at `destination`.
    /// Always exact. Neither pointer needs more than byte alignment.
    public static func widen(
        _ wire: CBv2CheckpointDType, source: UnsafeRawPointer, count: Int,
        float32 destination: UnsafeMutableRawPointer
    ) {
        precondition(count >= 0 && isHalf(wire))
        let lanes = 8
        let vectorCount = count / lanes * lanes
        var index = 0
        if wire == .bfloat16 {
            while index < vectorCount {
                let halves = source.loadUnaligned(fromByteOffset: index * 2, as: SIMD8<UInt16>.self)
                destination.storeBytes(
                    of: SIMD8<UInt32>(truncatingIfNeeded: halves) &<< 16,
                    toByteOffset: index * 4, as: SIMD8<UInt32>.self)
                index += lanes
            }
            while index < count {
                let half = source.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                destination.storeBytes(
                    of: UInt32(half) << 16, toByteOffset: index * 4, as: UInt32.self)
                index += 1
            }
            return
        }
        while index < vectorCount {
            let halves = source.loadUnaligned(fromByteOffset: index * 2, as: SIMD8<Float16>.self)
            destination.storeBytes(
                of: SIMD8<Float>(halves), toByteOffset: index * 4, as: SIMD8<Float>.self)
            index += lanes
        }
        while index < count {
            let half = source.loadUnaligned(fromByteOffset: index * 2, as: Float16.self)
            destination.storeBytes(of: Float(half), toByteOffset: index * 4, as: Float.self)
            index += 1
        }
    }
}
