import Foundation

/// CPU specification of the selection networks, not a Metal execution oracle.
/// Both gradual and flushed-subnormal comparison modes are checked explicitly.
enum FinalistsReference {
    static func value(_ bits: UInt16, flush: Bool) -> Float {
        if flush && (bits & 0x7f80) == 0 { return 0 }
        return Float(bitPattern: UInt32(bits) << 16)
    }
    static func before(_ a: UInt32, _ b: UInt32, flush: Bool) -> Bool {
        let av = value(UInt16(a >> 7), flush: flush), bv = value(UInt16(b >> 7), flush: flush)
        let ab = av.isNaN || bv.isNaN ? (!av.isNaN && bv.isNaN) : av < bv
        let ba = av.isNaN || bv.isNaN ? (!bv.isNaN && av.isNaN) : bv < av
        return ab || (!ba && (a & 127) < (b & 127))
    }
    static func bitonic(_ items: [UInt32], flush: Bool) -> [UInt32] {
        var result = items, width = 2
        while width <= 32 {
            var stride = width / 2
            while stride > 0 {
                let previous = result
                for lane in 0..<32 {
                    let other = previous[lane ^ stride]
                    let otherBefore = before(other, previous[lane], flush: flush)
                    let takeMinimum = ((lane & width) == 0) == ((lane & stride) == 0)
                    if takeMinimum ? otherBefore : !otherBefore { result[lane] = other }
                }
                stride /= 2
            }
            width *= 2
        }
        return result
    }
    static func select(_ bits: [UInt16], flush: Bool) -> [Int] {
        let items = bits.enumerated().map { (UInt32($0.element) << 7) | UInt32($0.offset) }
        let finalists = (0..<4).flatMap { group in
            Array(bitonic(Array(items[group * 32 ..< (group + 1) * 32]), flush: flush).suffix(8))
        }
        return bitonic(finalists, flush: flush).suffix(8).map { Int($0 & 127) }
    }
    static func orderedKey(_ bits: UInt16, expert: Int, flush: Bool) -> UInt32 {
        let v = value(bits, flush: flush)
        let raw = UInt32(bits)
        let ordinal: UInt32 = v.isNaN ? 0xffff : (v == 0 ? 0x8000 : ((raw & 0x8000) != 0 ? ((~raw) & 0xffff) : raw ^ 0x8000))
        return (ordinal << 7) | UInt32(expert)
    }
}
