import Foundation

/// Scalar reference for format validation and quality experiments. Live
/// inference writes use the corresponding Metal kernel.
public enum PagedKVQuantizationReference {
    public struct EncodedRow: Sendable {
        public let bytes: [UInt8]
        /// Reconstructed values in the cache's basis (rotated for K).
        public let values: [Float]
    }

    public static func rotate(_ values: [Float], blockSize: Int, inverse: Bool = false) -> [Float] {
        guard blockSize != 0 else { return values }
        precondition(blockSize.nonzeroBitCount == 1 && values.count % blockSize == 0)
        var result = values
        for base in stride(from: 0, to: values.count, by: blockSize) {
            if !inverse {
                for index in 0 ..< blockSize { result[base + index] *= sign(index) }
            }
            var width = 1
            while width < blockSize {
                for start in stride(from: 0, to: blockSize, by: 2 * width) {
                    for index in 0 ..< width {
                        let lo = base + start + index
                        let hi = lo + width
                        let a = result[lo]
                        let b = result[hi]
                        result[lo] = a + b
                        result[hi] = a - b
                    }
                }
                width *= 2
            }
            let normalization = 1 / Float(blockSize).squareRoot()
            for index in 0 ..< blockSize {
                result[base + index] *= normalization * (inverse ? sign(index) : 1)
            }
        }
        return result
    }

    public static func encode(
        _ row: [Float], config: PagedKVQuantizationConfig,
        isKey: Bool
    ) throws -> EncodedRow {
        try config.validate(headDim: row.count)
        guard row.allSatisfy(\.isFinite) else {
            throw CBv2KVError.backendIneligible(reason: "non-finite quantization reference input")
        }
        let bits = isKey ? config.keyBits : config.valueBits
        let values =
            isKey
            ? rotate(row, blockSize: config.resolvedRotationBlockSize(headDim: row.count)) : row
        guard values.allSatisfy(\.isFinite) else {
            throw CBv2KVError.backendIneligible(reason: "non-finite rotated quantization input")
        }
        let groups = row.count / config.groupSize
        let dataBytes = row.count * bits / 8
        let levels = Float((1 << bits) - 1)
        var bytes = [UInt8](repeating: 0, count: dataBytes + 8 * groups)
        var reconstructed = [Float](repeating: 0, count: row.count)
        for group in 0 ..< groups {
            let start = group * config.groupSize
            let input = values[start ..< start + config.groupSize]
            let lo = input.min()!
            let hi = input.max()!
            let scale = hi / levels - lo / levels
            for (parameter, offset) in [
                (scale, dataBytes + group * 4),
                (lo, dataBytes + (groups + group) * 4),
            ] {
                let raw = parameter.bitPattern
                for byte in 0 ..< 4 {
                    bytes[offset + byte] = UInt8(truncatingIfNeeded: raw >> (byte * 8))
                }
            }
            for index in start ..< start + config.groupSize {
                let raw = scale == 0 ? 0 : (values[index] / scale - lo / scale).rounded()
                let code = UInt8(max(0, min(levels, raw)))
                if bits == 4 {
                    bytes[index / 2] |= code << ((index & 1) * 4)
                } else {
                    bytes[index] = code
                }
                reconstructed[index] = scale * Float(code) + lo
            }
        }
        return EncodedRow(bytes: bytes, values: reconstructed)
    }

    public static func roundTrip(
        _ row: [Float], config: PagedKVQuantizationConfig,
        isKey: Bool
    ) throws -> [Float] {
        let result = try encode(row, config: config, isKey: isKey).values
        return isKey
            ? rotate(
                result, blockSize: config.resolvedRotationBlockSize(headDim: row.count),
                inverse: true) : result
    }

    /// Decode one canonical affine row, returning K in the original native
    /// basis. Byte reads are explicitly little-endian and need no alignment.
    public static func decode(
        _ bytes: [UInt8], headDim: Int, config: PagedKVQuantizationConfig, isKey: Bool
    ) throws -> [Float] {
        let layout = try config.rowLayout(headDim: headDim)
        let bits = isKey ? config.keyBits : config.valueBits
        let dataBytes = isKey ? layout.keyDataBytes : layout.valueDataBytes
        guard bytes.count == (isKey ? layout.keyRowBytes : layout.valueRowBytes) else {
            throw CBv2KVError.backendIneligible(reason: "invalid affine checkpoint row length")
        }
        let groups = headDim / config.groupSize
        func coefficient(_ offset: Int) -> Float {
            let raw = (0 ..< 4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << ($1 * 8) }
            return Float(bitPattern: raw)
        }
        var values = [Float](repeating: 0, count: headDim)
        for group in 0 ..< groups {
            let scale = coefficient(dataBytes + group * 4)
            let offset = coefficient(dataBytes + (groups + group) * 4)
            guard scale.isFinite, scale >= 0, offset.isFinite else {
                throw CBv2KVError.backendIneligible(reason: "invalid affine checkpoint coefficient")
            }
            for index in group * config.groupSize ..< (group + 1) * config.groupSize {
                let code = bits == 4 ? (bytes[index / 2] >> ((index & 1) * 4)) & 15 : bytes[index]
                // Opposite finite extremes can have an overflowing product
                // even though the affine reconstruction is representable.
                values[index] = offset.addingProduct(scale, Float(code))
            }
        }
        if isKey {
            values = rotate(
                values, blockSize: config.resolvedRotationBlockSize(headDim: headDim), inverse: true
            )
        }
        guard values.allSatisfy(\.isFinite) else {
            throw CBv2KVError.backendIneligible(reason: "non-finite affine checkpoint row")
        }
        return values
    }

    private static func sign(_ index: Int) -> Float {
        var x = UInt32(index) &+ 0x9e37_79b9
        x = (x ^ (x >> 16)) &* 0x7feb_352d
        x = (x ^ (x >> 15)) &* 0x846c_a68b
        x ^= x >> 16
        return x & 1 == 0 ? 1 : -1
    }
}
