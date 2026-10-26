import Foundation
import MLX

/// Import payload scratch is at most 44 KiB: two 16 KiB batches (including CoW),
/// a 2 KiB partial row, two 2 KiB native rows (including CoW), and 2 KiB each for
/// Float scratch, rotation factors and preparation/SIMD temporaries. The 64 KiB
/// charge also covers container overhead and the reference encoder's row-local
/// temporaries. Input and sink-retained output belong to the caller's IO budget;
/// neither import nor export retains history here.
enum CBv2NativeCheckpointRowCodec {
    static let scratchBytes = 64 << 10

    static func encode(_ data: Data, role: CBv2CheckpointPagedRoleLayout) throws -> Data {
        try data.withUnsafeBytes { try encode($0, role: role, decoder: nil) }
    }

    /// `decoder`, prepared for `role`, performs the representability check;
    /// nil prepares one for this row alone.
    fileprivate static func encode(
        _ bytes: UnsafeRawBufferPointer, role: CBv2CheckpointPagedRoleLayout,
        decoder: CBv2PreparedNativeCheckpointRowDecoder?
    ) throws -> Data {
        guard [.float16, .bfloat16, .float32].contains(role.key.dtype),
            let config = role.key.quantization, bytes.count == role.nativeRowBytes
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let values: [Float] = (0 ..< role.width).map { index in
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
        do {
            let encoded = Data(
                try PagedKVQuantizationReference.encode(
                    values, config: config, isKey: !role.values
                ).bytes)
            // A valid native input can still reconstruct outside its dtype
            // after rotation/rounding. Refuse export, never persist such a row.
            if let decoder {
                _ = try encoded.withUnsafeBytes { try decoder.decode($0) }
            } else {
                _ = try decode(encoded, role: role)
            }
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

/// One role's checkpoint rows for a store outside the engine's complete
/// checkpoints (the cluster's encrypted SSD prefix records): the encoder,
/// representability check and canonical decoder of
/// `CBv2NativeCheckpointRowCodec`, prepared once for a native dtype, a row
/// width, the key or value role (keys rotated as the profile says) and a
/// profile. Encoded bytes are those the complete checkpoints store for the
/// same row. Not thread-safe: one instance per concurrent caller.
@_spi(Darkbloom)
public final class CBv2CheckpointRowCodec {
    private let role: CBv2CheckpointPagedRoleLayout
    private let decoder: CBv2PreparedNativeCheckpointRowDecoder

    /// Bytes of one native row and of one encoded row.
    public var nativeRowBytes: Int { role.nativeRowBytes }
    public var encodedRowBytes: Int { role.packedRowBytes }

    /// Throws `CBv2CompleteCheckpointError` for a dtype other than FP16,
    /// BF16 or FP32, or a width or profile the format does not encode.
    public init(
        dtype: DType, width: Int, values: Bool, quantization: PagedKVQuantizationConfig
    ) throws {
        role = try CBv2CheckpointPagedRoleLayout(
            key: PagedKVGroupKey(
                kvHeads: 1, headDim: width, dtype: dtype, valueHeadDim: width,
                quantization: quantization),
            position: 1, values: values, preservePackedRecent: false)
        decoder = try CBv2PreparedNativeCheckpointRowDecoder(role: role)
    }

    /// One native row (`nativeRowBytes`, little-endian) encoded. Refuses a
    /// non-finite row and one whose reconstruction leaves the native dtype.
    public func encode(_ native: UnsafeRawBufferPointer) throws -> Data {
        try CBv2NativeCheckpointRowCodec.encode(native, role: role, decoder: decoder)
    }

    /// One encoded row (`encodedRowBytes`) as its native row: approximate,
    /// keys back in their original basis. Refuses malformed coefficients and
    /// any reconstruction outside the native dtype.
    public func decode(_ encoded: UnsafeRawBufferPointer) throws -> Data {
        try decoder.decode(encoded)
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
            let rawValues = UnsafeMutableRawBufferPointer(values)
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
                var index = group * groupSize
                let end = index + groupSize
                let scales = SIMD8<Float>(repeating: scale)
                let offsets = SIMD8<Float>(repeating: offset)
                if bits == 4 {
                    let shifts = SIMD8<UInt32>(0, 4, 8, 12, 16, 20, 24, 28)
                    while index + 8 <= end {
                        let word = bytes.loadUnaligned(
                            fromByteOffset: index / 2, as: UInt32.self
                        ).littleEndian
                        let codes = SIMD8<Float>((SIMD8<UInt32>(repeating: word) &>> shifts) & 15)
                        Self.storeVector(
                            offsets.addingProduct(scales, codes), to: rawValues, at: index * 4)
                        index += 8
                    }
                } else {
                    while index + 8 <= end {
                        let codes = SIMD8<Float>(
                            bytes.loadUnaligned(fromByteOffset: index, as: SIMD8<UInt8>.self))
                        Self.storeVector(
                            offsets.addingProduct(scales, codes), to: rawValues, at: index * 4)
                        index += 8
                    }
                }
                while index < end {
                    let code =
                        bits == 4 ? (bytes[index / 2] >> ((index & 1) * 4)) & 15 : bytes[index]
                    values[index] = offset.addingProduct(scale, Float(code))
                    index += 1
                }
            }

            if rotation != 0 { inverseRotate(rawValues) }
            try writeNative(UnsafeRawBufferPointer(rawValues))
        }
        return native
    }

    private func inverseRotate(_ values: UnsafeMutableRawBufferPointer) {
        rotationFactors.withUnsafeBufferPointer { factors in
            let rawFactors = UnsafeRawBufferPointer(factors)
            for base in stride(from: 0, to: role.width, by: rotation) {
                var width = 1
                if rotation >= 8 {
                    // The first three stages are independent within each eight-lane block.
                    // Keep those stages in registers without reassociating any butterfly.
                    for start in stride(from: 0, to: rotation, by: 8) {
                        let input = values.loadUnaligned(
                            fromByteOffset: (base + start) * 4, as: SIMD8<Float>.self)
                        let sums = input.evenHalf + input.oddHalf
                        let differences = input.evenHalf - input.oddHalf
                        let a = SIMD4(sums[0], differences[0], sums[2], differences[2])
                        let b = SIMD4(sums[1], differences[1], sums[3], differences[3])
                        let secondSums = a + b
                        let secondDifferences = a - b
                        let lo = SIMD4(
                            secondSums[0], secondSums[1], secondDifferences[0], secondDifferences[1]
                        )
                        let hi = SIMD4(
                            secondSums[2], secondSums[3], secondDifferences[2], secondDifferences[3]
                        )
                        Self.storeVector(
                            SIMD8(lowHalf: lo + hi, highHalf: lo - hi),
                            to: values, at: (base + start) * 4)
                    }
                    width = 8
                }
                while width < rotation {
                    for start in stride(from: 0, to: rotation, by: 2 * width) {
                        var index = 0
                        while index + 8 <= width {
                            let lo = (base + start + index) * 4
                            let hi = lo + width * 4
                            let a = values.loadUnaligned(fromByteOffset: lo, as: SIMD8<Float>.self)
                            let b = values.loadUnaligned(fromByteOffset: hi, as: SIMD8<Float>.self)
                            Self.storeVector(a + b, to: values, at: lo)
                            Self.storeVector(a - b, to: values, at: hi)
                            index += 8
                        }
                        while index < width {
                            let lo = (base + start + index) * 4
                            let hi = lo + width * 4
                            let a = values.loadUnaligned(fromByteOffset: lo, as: Float.self)
                            let b = values.loadUnaligned(fromByteOffset: hi, as: Float.self)
                            values.storeBytes(of: a + b, toByteOffset: lo, as: Float.self)
                            values.storeBytes(of: a - b, toByteOffset: hi, as: Float.self)
                            index += 1
                        }
                    }
                    width *= 2
                }
                var index = 0
                while index + 8 <= rotation {
                    let offset = (base + index) * 4
                    let input = values.loadUnaligned(fromByteOffset: offset, as: SIMD8<Float>.self)
                    let factor = rawFactors.loadUnaligned(
                        fromByteOffset: index * 4, as: SIMD8<Float>.self)
                    Self.storeVector(input * factor, to: values, at: offset)
                    index += 8
                }
                while index < rotation {
                    let offset = (base + index) * 4
                    let value = values.loadUnaligned(fromByteOffset: offset, as: Float.self)
                    values.storeBytes(
                        of: value * factors[index], toByteOffset: offset, as: Float.self)
                    index += 1
                }
            }
        }
    }

    private func writeNative(_ values: UnsafeRawBufferPointer) throws {
        try native.withUnsafeMutableBytes { destination in
            var index = 0
            switch role.key.dtype {
            case .float16:
                while index + 8 <= role.width {
                    let input = values.loadUnaligned(
                        fromByteOffset: index * 4, as: SIMD8<Float>.self)
                    let narrowed = SIMD8<Float16>(input)
                    var words = unsafeBitCast(narrowed, to: SIMD8<UInt16>.self)
                    guard all((words & 0x7c00) .!= 0x7c00) else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    for lane in words.indices { words[lane] = words[lane].littleEndian }
                    Self.storeVector(words, to: destination, at: index * 2)
                    index += 8
                }
                while index < role.width {
                    let narrowed = Float16(
                        values.loadUnaligned(fromByteOffset: index * 4, as: Float.self))
                    guard narrowed.isFinite else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    destination.storeBytes(
                        of: narrowed.bitPattern.littleEndian,
                        toByteOffset: index * 2, as: UInt16.self)
                    index += 1
                }
            case .bfloat16:
                while index + 8 <= role.width {
                    let input = values.loadUnaligned(
                        fromByteOffset: index * 4, as: SIMD8<Float>.self)
                    let raw = unsafeBitCast(input, to: SIMD8<UInt32>.self)
                    // Check before rounding: a NaN mantissa can carry into a finite word.
                    guard all((raw & 0x7f80_0000) .!= 0x7f80_0000) else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    let rounded = (raw &+ 0x7fff &+ ((raw &>> 16) & 1)) &>> 16
                    guard all((rounded & 0x7f80) .!= 0x7f80) else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    var words = SIMD8<UInt16>(truncatingIfNeeded: rounded)
                    for lane in words.indices { words[lane] = words[lane].littleEndian }
                    Self.storeVector(words, to: destination, at: index * 2)
                    index += 8
                }
                while index < role.width {
                    let value = values.loadUnaligned(fromByteOffset: index * 4, as: Float.self)
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
                        of: rounded.littleEndian, toByteOffset: index * 2, as: UInt16.self)
                    index += 1
                }
            default:
                while index + 8 <= role.width {
                    let input = values.loadUnaligned(
                        fromByteOffset: index * 4, as: SIMD8<Float>.self)
                    let words = unsafeBitCast(input, to: SIMD8<UInt32>.self)
                    guard all((words & 0x7f80_0000) .!= 0x7f80_0000) else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    index += 8
                }
                while index < role.width {
                    guard values.loadUnaligned(fromByteOffset: index * 4, as: Float.self).isFinite
                    else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                    index += 1
                }
                #if _endian(little)
                    destination.copyMemory(from: values)
                #else
                    for index in 0 ..< role.width {
                        let value = values.loadUnaligned(fromByteOffset: index * 4, as: Float.self)
                        destination.storeBytes(
                            of: value.bitPattern.littleEndian, toByteOffset: index * 4,
                            as: UInt32.self)
                    }
                #endif
            }
        }
    }

    @inline(__always)
    private static func storeVector<Vector: SIMD>(
        _ value: Vector, to destination: UnsafeMutableRawBufferPointer, at offset: Int
    ) {
        // A byte copy makes no SIMD-alignment or memory-binding assumptions.
        withUnsafeBytes(of: value) { source in
            UnsafeMutableRawBufferPointer(
                rebasing: destination[offset ..< offset + source.count]
            ).copyMemory(from: source)
        }
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
