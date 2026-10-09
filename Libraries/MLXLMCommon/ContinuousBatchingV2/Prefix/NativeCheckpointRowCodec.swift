import Foundation
import MLX

/// CPU conversion is bounded by one row (D <= 512), never by prefix length.
/// This includes the reference encoder's temporary Float arrays and the
/// fragmented import row. Returned export Data belongs to the caller's IO budget.
enum CBv2NativeCheckpointRowCodec {
    static let scratchBytes = 64 << 10

    static func encode(_ data: Data, role: CBv2CheckpointPagedRoleLayout) throws -> Data {
        guard [.float16, .bfloat16, .float32].contains(role.key.dtype),
            let config = role.key.quantization, data.count == role.nativeRowBytes
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let values: [Float] = data.withUnsafeBytes { bytes in
            (0 ..< role.width).map { index in
                switch role.key.dtype {
                case .float16:
                    return Float(Float16(bitPattern: bytes.loadUnaligned(
                        fromByteOffset: index * 2, as: UInt16.self).littleEndian))
                case .bfloat16:
                    return Float(bitPattern: UInt32(bytes.loadUnaligned(
                        fromByteOffset: index * 2, as: UInt16.self).littleEndian) << 16)
                default:
                    return Float(bitPattern: bytes.loadUnaligned(
                        fromByteOffset: index * 4, as: UInt32.self).littleEndian)
                }
            }
        }
        do {
            let encoded = Data(try PagedKVQuantizationReference.encode(
                values, config: config, isKey: !role.values).bytes)
            // A valid native input can still reconstruct outside its dtype
            // after rotation/rounding. Refuse export, never persist such a row.
            _ = try decode(encoded, role: role)
            return encoded
        } catch {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
    }

    static func decode(_ data: Data, role: CBv2CheckpointPagedRoleLayout) throws -> Data {
        guard [.float16, .bfloat16, .float32].contains(role.key.dtype),
            let config = role.key.quantization
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let values: [Float]
        do {
            values = try PagedKVQuantizationReference.decode(
                Array(data), headDim: role.width, config: config, isKey: !role.values)
        } catch {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var result = Data(count: role.nativeRowBytes)
        try result.withUnsafeMutableBytes { bytes in
            for (index, value) in values.enumerated() {
                switch role.key.dtype {
                case .float16:
                    let narrowed = Float16(value)
                    guard narrowed.isFinite else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    bytes.storeBytes(of: narrowed.bitPattern.littleEndian,
                        toByteOffset: index * 2, as: UInt16.self)
                case .bfloat16:
                    let raw = value.bitPattern
                    let rounded = UInt16(truncatingIfNeeded:
                        (raw &+ 0x7fff &+ ((raw >> 16) & 1)) >> 16)
                    guard Float(bitPattern: UInt32(rounded) << 16).isFinite else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    bytes.storeBytes(of: rounded.littleEndian,
                        toByteOffset: index * 2, as: UInt16.self)
                default:
                    bytes.storeBytes(of: value.bitPattern.littleEndian,
                        toByteOffset: index * 4, as: UInt32.self)
                }
            }
        }
        return result
    }
}

/// At most one encoded/native row survives a transfer call. A partial row is
/// private until validated and reconstructed; no packed serving owner exists.
final class CBv2NativeCheckpointRowDecoder {
    private var row = Data()
    var isComplete: Bool { row.isEmpty }

    func append(
        role: CBv2CheckpointPagedRoleLayout, byteOffset: Int, data: Data,
        write: (Int, Data) throws -> Void
    ) throws {
        let bytes = try PagedKVQuantizationConfig.multiply(role.key.kvHeads, role.bytesPerHead)
        guard role.isQuantized, byteOffset >= 0, !data.isEmpty,
            byteOffset < bytes, data.count <= bytes - byteOffset
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let packedBytes = role.packedCount * role.packedRowBytes
        let tokenCount = role.position - role.tokenStart
        var consumed = 0
        while consumed < data.count {
            let offset = byteOffset + consumed
            let head = offset / role.bytesPerHead
            let inHead = offset % role.bytesPerHead
            let packed = inHead < packedBytes
            let inBand = packed ? inHead : inHead - packedBytes
            let rowBytes = packed ? role.packedRowBytes : role.nativeRowBytes
            let inRow = inBand % rowBytes
            guard row.count == inRow else { throw CBv2CompleteCheckpointError.invalidSegment }
            let count = min(data.count - consumed, rowBytes - inRow)
            let start = data.index(data.startIndex, offsetBy: consumed)
            row.append(data[start ..< data.index(start, offsetBy: count)])
            consumed += count
            if row.count == rowBytes {
                let token = inBand / rowBytes + (packed ? 0 : role.packedCount)
                let native = packed ? try CBv2NativeCheckpointRowCodec.decode(row, role: role) : row
                try write((head * tokenCount + token) * role.nativeRowBytes, native)
                row.removeAll(keepingCapacity: true)
            }
        }
    }

    func close() { row = Data() }
}
