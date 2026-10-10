import Foundation
import MLX

/// Import payload scratch is at most 44 KiB: two 16 KiB batches (including CoW),
/// a 2 KiB partial row, two 2 KiB native rows (including CoW), and 2 KiB each for
/// Float scratch, rotation factors and their construction impulse. The 64 KiB
/// charge also covers container overhead and the reference encoder's row-local
/// temporaries. Input and sink-retained output belong to the caller's IO budget;
/// neither import nor export retains history here.
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
                    return Float(
                        Float16(
                            bitPattern: bytes.loadUnaligned(
                                fromByteOffset: index * 2, as: UInt16.self
                            ).littleEndian))
                case .bfloat16:
                    return Float(
                        bitPattern: UInt32(
                            bytes.loadUnaligned(
                                fromByteOffset: index * 2, as: UInt16.self
                            ).littleEndian) << 16)
                default:
                    return Float(
                        bitPattern: bytes.loadUnaligned(
                            fromByteOffset: index * 4, as: UInt32.self
                        ).littleEndian)
                }
            }
        }
        do {
            let encoded = Data(
                try PagedKVQuantizationReference.encode(
                    values, config: config, isKey: !role.values
                ).bytes)
            // A valid native input can still reconstruct outside its dtype
            // after rotation/rounding. Refuse export, never persist such a row.
            _ = try decode(encoded, role: role)
            return encoded
        } catch {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
    }

    static func decode(_ data: Data, role: CBv2CheckpointPagedRoleLayout) throws -> Data {
        let decoder = try CBv2PreparedNativeCheckpointRowDecoder(role: role)
        return try data.withUnsafeBytes { try decoder.decode($0) }
    }
}

/// One canonical affine decoder serves both streaming restore and export's
/// representability check. Geometry, signs and row buffers are prepared once.
private final class CBv2PreparedNativeCheckpointRowDecoder {
    private let role: CBv2CheckpointPagedRoleLayout
    private let bits: Int
    private let groupSize: Int
    private let scaleOffset: Int
    private let offsetOffset: Int
    private let rotation: Int
    private let rotationFactors: [Float]
    private var scratch: [Float]
    private var native: Data

    init(role: CBv2CheckpointPagedRoleLayout) throws {
        guard [.float16, .bfloat16, .float32].contains(role.key.dtype),
            let config = role.key.quantization
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let layout: PagedKVQuantizedRowLayout
        do {
            layout = try config.rowLayout(headDim: role.width)
        } catch {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }

        self.role = role
        bits = role.values ? config.valueBits : config.keyBits
        groupSize = config.groupSize
        scaleOffset = role.values ? layout.valueScaleOffset : layout.keyScaleOffset
        offsetOffset = role.values ? layout.valueOffsetOffset : layout.keyOffsetOffset
        let rotation = role.values ? 0 : config.resolvedRotationBlockSize(headDim: role.width)
        self.rotation = rotation
        if rotation == 0 {
            rotationFactors = []
        } else {
            // An inverse-rotated impulse is the canonical normalized sign mask.
            // Cache it without maintaining another copy of the format's hash.
            var impulse = [Float](repeating: 0, count: rotation)
            impulse[0] = 1
            rotationFactors = PagedKVQuantizationReference.rotate(
                impulse, blockSize: rotation, inverse: true)
        }
        scratch = [Float](repeating: 0, count: role.width)
        native = Data(count: role.nativeRowBytes)
    }

    func matches(_ role: CBv2CheckpointPagedRoleLayout) -> Bool {
        self.role.key == role.key && self.role.values == role.values
            && self.role.position == role.position && self.role.tokenStart == role.tokenStart
            && self.role.packedCount == role.packedCount
    }

    func decode(_ bytes: UnsafeRawBufferPointer) throws -> Data {
        guard bytes.count == role.packedRowBytes else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try scratch.withUnsafeMutableBufferPointer { values in
            for group in 0 ..< role.width / groupSize {
                let scale = Float(
                    bitPattern: bytes.loadUnaligned(
                        fromByteOffset: scaleOffset + group * 4, as: UInt32.self
                    ).littleEndian)
                let offset = Float(
                    bitPattern: bytes.loadUnaligned(
                        fromByteOffset: offsetOffset + group * 4, as: UInt32.self
                    ).littleEndian)
                guard scale.isFinite, scale >= 0, offset.isFinite else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                let start = group * groupSize
                if bits == 4 {
                    for index in start ..< start + groupSize {
                        let code = (bytes[index / 2] >> ((index & 1) * 4)) & 15
                        values[index] = offset.addingProduct(scale, Float(code))
                    }
                } else {
                    for index in start ..< start + groupSize {
                        values[index] = offset.addingProduct(scale, Float(bytes[index]))
                    }
                }
            }

            if rotation != 0 {
                // Preserve the reference butterfly and normalization order.
                for base in stride(from: 0, to: role.width, by: rotation) {
                    var width = 1
                    while width < rotation {
                        for start in stride(from: 0, to: rotation, by: 2 * width) {
                            for index in 0 ..< width {
                                let lo = base + start + index
                                let hi = lo + width
                                let a = values[lo]
                                let b = values[hi]
                                values[lo] = a + b
                                values[hi] = a - b
                            }
                        }
                        width *= 2
                    }
                    for index in 0 ..< rotation {
                        values[base + index] *= rotationFactors[index]
                    }
                }
            }

            try native.withUnsafeMutableBytes { destination in
                switch role.key.dtype {
                case .float16:
                    for index in values.indices {
                        let narrowed = Float16(values[index])
                        guard narrowed.isFinite else {
                            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                        }
                        destination.storeBytes(
                            of: narrowed.bitPattern.littleEndian,
                            toByteOffset: index * 2, as: UInt16.self)
                    }
                case .bfloat16:
                    for index in values.indices {
                        let value = values[index]
                        guard value.isFinite else {
                            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                        }
                        let raw = value.bitPattern
                        let rounded = UInt16(
                            truncatingIfNeeded: (raw &+ 0x7fff &+ ((raw >> 16) & 1)) >> 16)
                        guard Float(bitPattern: UInt32(rounded) << 16).isFinite else {
                            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                        }
                        destination.storeBytes(
                            of: rounded.littleEndian,
                            toByteOffset: index * 2, as: UInt16.self)
                    }
                default:
                    for index in values.indices {
                        let value = values[index]
                        guard value.isFinite else {
                            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                        }
                        destination.storeBytes(
                            of: value.bitPattern.littleEndian,
                            toByteOffset: index * 4, as: UInt32.self)
                    }
                }
            }
        }
        return native
    }
}

/// Only a partial input row and reusable row scratch survive a transfer call.
/// Validated native rows are batched within the call, never across cancellation
/// boundaries; the sink still owns publication and native completion.
final class CBv2NativeCheckpointRowDecoder {
    private static let maximumBatchBytes = 16 << 10
    private var prepared: CBv2PreparedNativeCheckpointRowDecoder?
    private var row = Data()
    private var batch = Data()
    private var batchOffset: Int?
    private var failed = false
    var isComplete: Bool { !failed && row.isEmpty && batch.isEmpty }

    func append(
        role: CBv2CheckpointPagedRoleLayout, byteOffset: Int, data: Data,
        write: (Int, Data) throws -> Void
    ) throws {
        let bytes = try PagedKVQuantizationConfig.multiply(role.key.kvHeads, role.bytesPerHead)
        guard !failed, role.isQuantized, byteOffset >= 0, !data.isEmpty,
            byteOffset < bytes, data.count <= bytes - byteOffset
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let packedBytes = role.packedCount * role.packedRowBytes
        let inHead = byteOffset % role.bytesPerHead
        let packed = inHead < packedBytes
        let inBand = packed ? inHead : inHead - packedBytes
        let rowBytes = packed ? role.packedRowBytes : role.nativeRowBytes
        guard row.count == inBand % rowBytes else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        if prepared?.matches(role) != true {
            guard row.isEmpty else { throw CBv2CompleteCheckpointError.invalidSegment }
            // Release the previous role before allocating the next role's scratch.
            prepared = nil
            prepared = try CBv2PreparedNativeCheckpointRowDecoder(role: role)
            row.reserveCapacity(max(role.packedRowBytes, role.nativeRowBytes))
            batch.reserveCapacity(Self.maximumBatchBytes)
        }
        guard let prepared else { throw CBv2CompleteCheckpointError.invalidSegment }
        let tokenCount = role.position - role.tokenStart
        do {
            try data.withUnsafeBytes { source in
                var consumed = 0
                while consumed < source.count {
                    let offset = byteOffset + consumed
                    let head = offset / role.bytesPerHead
                    let inHead = offset % role.bytesPerHead
                    let packed = inHead < packedBytes
                    let inBand = packed ? inHead : inHead - packedBytes
                    let rowBytes = packed ? role.packedRowBytes : role.nativeRowBytes
                    let inRow = inBand % rowBytes
                    guard row.count == inRow else {
                        throw CBv2CompleteCheckpointError.invalidSegment
                    }
                    let count = min(source.count - consumed, rowBytes - inRow)
                    let fragment = UnsafeRawBufferPointer(
                        rebasing: source[consumed ..< consumed + count])
                    let token = inBand / rowBytes + (packed ? 0 : role.packedCount)
                    let nativeOffset = (head * tokenCount + token) * role.nativeRowBytes
                    if inRow == 0 && count == rowBytes {
                        try appendRow(
                            fragment, packed: packed, nativeOffset: nativeOffset,
                            nativeRowBytes: role.nativeRowBytes, decoder: prepared, write: write)
                    } else {
                        row.append(
                            fragment.baseAddress!.assumingMemoryBound(to: UInt8.self), count: count)
                        if row.count == rowBytes {
                            try row.withUnsafeBytes {
                                try appendRow(
                                    $0, packed: packed, nativeOffset: nativeOffset,
                                    nativeRowBytes: role.nativeRowBytes, decoder: prepared,
                                    write: write)
                            }
                            row.removeAll(keepingCapacity: true)
                        }
                    }
                    consumed += count
                }
            }
            try flush(write: write)
        } catch {
            failed = true
            // A rejected row never enters the batch. Preserve earlier completed
            // rows; flush clears before propagating a sink failure, so no retry.
            try flush(write: write)
            throw error
        }
    }

    private func appendRow(
        _ bytes: UnsafeRawBufferPointer, packed: Bool, nativeOffset: Int,
        nativeRowBytes: Int, decoder: CBv2PreparedNativeCheckpointRowDecoder,
        write: (Int, Data) throws -> Void
    ) throws {
        if let batchOffset,
            batchOffset + batch.count != nativeOffset
                || batch.count + nativeRowBytes > Self.maximumBatchBytes
        {
            try flush(write: write)
        }
        if packed {
            let native = try decoder.decode(bytes)
            native.withUnsafeBytes {
                batch.append($0.baseAddress!.assumingMemoryBound(to: UInt8.self), count: $0.count)
            }
        } else {
            // The recent band is opaque: preserve NaN payloads and signed zero.
            batch.append(bytes.baseAddress!.assumingMemoryBound(to: UInt8.self), count: bytes.count)
        }
        if batchOffset == nil { batchOffset = nativeOffset }
    }

    private func flush(write: (Int, Data) throws -> Void) throws {
        guard let batchOffset else { return }
        defer {
            self.batchOffset = nil
            batch.removeAll(keepingCapacity: true)
        }
        try write(batchOffset, batch)
    }

    func close() {
        prepared = nil
        row = Data()
        batch = Data()
        batchOffset = nil
        failed = false
    }
}
