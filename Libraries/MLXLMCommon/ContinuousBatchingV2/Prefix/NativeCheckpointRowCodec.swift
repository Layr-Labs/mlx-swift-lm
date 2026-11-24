import Foundation
import MLX

/// Import payload scratch is at most 44 KiB: two 16 KiB batches (including CoW),
/// a 2 KiB partial row, two 2 KiB native rows (including CoW), and 2 KiB each for
/// Float scratch, rotation factors and preparation/SIMD temporaries. The 64 KiB
/// charge also covers container overhead and encoders' row buffers: export's
/// prepared encoders (under 7 KiB each, at most `exportCodingWorkers`) or the
/// reference encoder's row-local temporaries. Input and sink-retained output
/// belong to the caller's IO budget; neither import nor export retains history
/// here. The concurrent decode's wave is charged separately
/// (`CBv2NativeCheckpointRowDecoder.scratchBytes(workers:)`).
enum CBv2NativeCheckpointRowCodec {
    static let scratchBytes = 64 << 10
    /// Native rows export reads from its source at once (whole rows, at least
    /// one), charged with their coded rows on top of `scratchBytes`.
    static let exportBatchBytes = 256 << 10
    /// Cores that code one batch: a quarter of the active ones, one to four.
    static let exportCodingWorkers = min(
        4, max(1, ProcessInfo.processInfo.activeProcessorCount / 4))

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

/// Export's encoder for one role, prepared once per segment: the bytes and
/// refusals of `CBv2NativeCheckpointRowCodec.encode`, from reused row buffers
/// and one prepared representability decoder instead of per-row allocations.
/// Not thread-safe: one instance per concurrent caller.
final class CBv2PreparedNativeCheckpointRowEncoder {
    private let role: CBv2CheckpointPagedRoleLayout
    private let decoder: CBv2PreparedNativeCheckpointRowDecoder
    private let bits: Int
    private let groupSize: Int
    private let rotation: Int
    private let signs: [Float]
    private var scratch: [Float]
    private var encoded: [UInt8]

    init(role: CBv2CheckpointPagedRoleLayout) throws {
        guard [.float16, .bfloat16, .float32].contains(role.key.dtype),
            let config = role.key.quantization
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        do {
            try config.validate(headDim: role.width)
        } catch {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let bits = role.values ? config.valueBits : config.keyBits
        guard role.width * bits / 8 + 8 * (role.width / config.groupSize) == role.packedRowBytes
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        self.role = role
        decoder = try CBv2PreparedNativeCheckpointRowDecoder(role: role)
        self.bits = bits
        groupSize = config.groupSize
        let rotation = role.values ? 0 : config.resolvedRotationBlockSize(headDim: role.width)
        // The rotation runs eight lanes at a time; a valid profile's blocks hold 32 or more.
        guard rotation % 8 == 0 else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.rotation = rotation
        if rotation == 0 {
            signs = []
        } else {
            // The inverse-rotated impulse is the normalized sign mask; keep its signs.
            var impulse = [Float](repeating: 0, count: rotation)
            impulse[0] = 1
            signs = PagedKVQuantizationReference.rotate(
                impulse, blockSize: rotation, inverse: true
            ).map { $0 < 0 ? -1 : 1 }
        }
        scratch = [Float](repeating: 0, count: role.width)
        encoded = [UInt8](repeating: 0, count: role.packedRowBytes)
    }

    func encode(_ data: Data) throws -> Data {
        var result = Data(count: role.packedRowBytes)
        try data.withUnsafeBytes { native in
            try result.withUnsafeMutableBytes { try encode(native, into: $0) }
        }
        return result
    }

    /// Same operations in the same order as `PagedKVQuantizationReference.encode`.
    func encode(
        _ native: UnsafeRawBufferPointer, into destination: UnsafeMutableRawBufferPointer
    ) throws {
        guard native.count == role.nativeRowBytes, destination.count == role.packedRowBytes
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let width = role.width
        let dtype = role.key.dtype
        try scratch.withUnsafeMutableBufferPointer { row in
            for index in 0 ..< width {
                switch dtype {
                case .float16:
                    row[index] = Float(
                        Float16(
                            bitPattern: native.loadUnaligned(
                                fromByteOffset: index * 2, as: UInt16.self
                            ).littleEndian))
                case .bfloat16:
                    row[index] = Float(
                        bitPattern: UInt32(
                            native.loadUnaligned(
                                fromByteOffset: index * 2, as: UInt16.self
                            ).littleEndian) << 16)
                default:
                    row[index] = Float(
                        bitPattern: native.loadUnaligned(
                            fromByteOffset: index * 4, as: UInt32.self
                        ).littleEndian)
                }
            }
            guard row.allSatisfy(\.isFinite) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            if rotation != 0 {
                rotate(row)
                guard row.allSatisfy(\.isFinite) else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
            }
            quantize(row)
        }
        do {
            _ = try encoded.withUnsafeBytes { try decoder.decode($0) }
        } catch {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        encoded.withUnsafeBytes { destination.copyMemory(from: $0) }
    }

    /// Eight lanes at a time, every element takes the scalar butterflies'
    /// additions in the same order; the first three stages stay in registers
    /// as in the decoder's inverse.
    private func rotate(_ row: UnsafeMutableBufferPointer<Float>) {
        let normalization = SIMD8<Float>(repeating: 1 / Float(rotation).squareRoot())
        let values = UnsafeMutableRawBufferPointer(row)
        func store(_ value: SIMD8<Float>, _ offset: Int) {
            CBv2PreparedNativeCheckpointRowDecoder.storeVector(value, to: values, at: offset)
        }
        signs.withUnsafeBytes { signs in
            for base in stride(from: 0, to: role.width, by: rotation) {
                for start in stride(from: 0, to: rotation, by: 8) {
                    let offset = (base + start) * 4
                    let input =
                        values.loadUnaligned(fromByteOffset: offset, as: SIMD8<Float>.self)
                        * signs.loadUnaligned(fromByteOffset: start * 4, as: SIMD8<Float>.self)
                    let sums = input.evenHalf + input.oddHalf
                    let differences = input.evenHalf - input.oddHalf
                    let a = SIMD4(sums[0], differences[0], sums[2], differences[2])
                    let b = SIMD4(sums[1], differences[1], sums[3], differences[3])
                    let secondSums = a + b
                    let secondDifferences = a - b
                    let lo = SIMD4(
                        secondSums[0], secondSums[1], secondDifferences[0], secondDifferences[1])
                    let hi = SIMD4(
                        secondSums[2], secondSums[3], secondDifferences[2], secondDifferences[3])
                    store(SIMD8(lowHalf: lo + hi, highHalf: lo - hi), offset)
                }
                var width = 8
                while width < rotation {
                    for start in stride(from: 0, to: rotation, by: 2 * width) {
                        for index in stride(from: 0, to: width, by: 8) {
                            let lo = (base + start + index) * 4
                            let hi = lo + width * 4
                            let a = values.loadUnaligned(fromByteOffset: lo, as: SIMD8<Float>.self)
                            let b = values.loadUnaligned(fromByteOffset: hi, as: SIMD8<Float>.self)
                            store(a + b, lo)
                            store(a - b, hi)
                        }
                    }
                    width *= 2
                }
                for start in stride(from: 0, to: rotation, by: 8) {
                    let offset = (base + start) * 4
                    let input = values.loadUnaligned(fromByteOffset: offset, as: SIMD8<Float>.self)
                    store(input * normalization, offset)
                }
            }
        }
    }

    private func quantize(_ row: UnsafeMutableBufferPointer<Float>) {
        let groups = role.width / groupSize
        let dataBytes = role.width * bits / 8
        let levels = Float((1 << bits) - 1)
        encoded.withUnsafeMutableBufferPointer { bytes in
            bytes.update(repeating: 0)
            func store(_ parameter: Float, at offset: Int) {
                let raw = parameter.bitPattern
                for byte in 0 ..< 4 {
                    bytes[offset + byte] = UInt8(truncatingIfNeeded: raw >> (byte * 8))
                }
            }
            for group in 0 ..< groups {
                let start = group * groupSize
                // `Sequence.min()`/`max()` order, so tied zeros keep their sign.
                var lo = row[start]
                var hi = row[start]
                for index in start + 1 ..< start + groupSize {
                    if row[index] < lo { lo = row[index] }
                    if hi < row[index] { hi = row[index] }
                }
                let scale = hi / levels - lo / levels
                store(scale, at: dataBytes + group * 4)
                store(lo, at: dataBytes + (groups + group) * 4)
                // A zero scale codes every element 0: the bytes are already 0.
                guard scale != 0 else { continue }
                let offset = lo / scale
                if bits == 4 {
                    for index in stride(from: start, to: start + groupSize, by: 2) {
                        let low = (row[index] / scale - offset).rounded()
                        let high = (row[index + 1] / scale - offset).rounded()
                        bytes[index / 2] =
                            UInt8(max(0, min(levels, low))) | UInt8(max(0, min(levels, high))) << 4
                    }
                } else {
                    for index in start ..< start + groupSize {
                        bytes[index] = UInt8(
                            max(0, min(levels, (row[index] / scale - offset).rounded())))
                    }
                }
            }
        }
    }
}

/// One canonical affine decoder serves both streaming restore and export's
/// representability check. Geometry, signs and row buffers are prepared once.
final class CBv2PreparedNativeCheckpointRowDecoder {
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
        try scratch.withUnsafeMutableBufferPointer { values in
            try native.withUnsafeMutableBytes { destination in
                try decode(bytes, scratch: UnsafeMutableRawBufferPointer(values), into: destination)
            }
        }
        return native
    }

    /// Reads only immutable state: concurrent calls with their own `scratch`
    /// (`width` Floats) and `destination` (`nativeRowBytes`) are safe.
    func decode(
        _ bytes: UnsafeRawBufferPointer, scratch rawValues: UnsafeMutableRawBufferPointer,
        into destination: UnsafeMutableRawBufferPointer
    ) throws {
        guard bytes.count == role.packedRowBytes, rawValues.count >= role.width * 4,
            destination.count >= role.nativeRowBytes
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        // Exactly one row of scratch and of destination: the FP32 path copies all of it.
        let rawValues = UnsafeMutableRawBufferPointer(rebasing: rawValues[0 ..< role.width * 4])
        let destination = UnsafeMutableRawBufferPointer(
            rebasing: destination[0 ..< role.nativeRowBytes])
        do {
            let values = rawValues.bindMemory(to: Float.self)
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
            try writeNative(UnsafeRawBufferPointer(rawValues), into: destination)
        }
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

    /// Adding one exponent step to an all-ones exponent (infinity or NaN) carries
    /// into the lane's top bit, so one test after the vector loop covers every vector.
    func writeNative(
        _ values: UnsafeRawBufferPointer, into destination: UnsafeMutableRawBufferPointer
    ) throws {
        do {
            var index = 0
            switch role.key.dtype {
            case .float16:
                var nonFinite = SIMD8<UInt16>()
                while index + 8 <= role.width {
                    let input = values.loadUnaligned(
                        fromByteOffset: index * 4, as: SIMD8<Float>.self)
                    let narrowed = SIMD8<Float16>(input)
                    var words = unsafeBitCast(narrowed, to: SIMD8<UInt16>.self)
                    nonFinite |= (words & 0x7c00) &+ 0x0400
                    for lane in words.indices { words[lane] = words[lane].littleEndian }
                    Self.storeVector(words, to: destination, at: index * 2)
                    index += 8
                }
                guard (nonFinite & 0x8000) == .zero else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
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
                var nonFinite = SIMD8<UInt32>()
                while index + 8 <= role.width {
                    let input = values.loadUnaligned(
                        fromByteOffset: index * 4, as: SIMD8<Float>.self)
                    let raw = unsafeBitCast(input, to: SIMD8<UInt32>.self)
                    // Check before rounding: a NaN mantissa can carry into a finite word.
                    nonFinite |= (raw & 0x7f80_0000) &+ 0x0080_0000
                    let rounded = (raw &+ 0x7fff &+ ((raw &>> 16) & 1)) &>> 16
                    nonFinite |= ((rounded & 0x7f80) &+ 0x0080) &<< 16
                    var words = SIMD8<UInt16>(truncatingIfNeeded: rounded)
                    for lane in words.indices { words[lane] = words[lane].littleEndian }
                    Self.storeVector(words, to: destination, at: index * 2)
                    index += 8
                }
                guard (nonFinite & 0x8000_0000) == .zero else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
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
                var nonFinite = SIMD8<UInt32>()
                while index + 8 <= role.width {
                    let input = values.loadUnaligned(
                        fromByteOffset: index * 4, as: SIMD8<Float>.self)
                    let words = unsafeBitCast(input, to: SIMD8<UInt32>.self)
                    nonFinite |= (words & 0x7f80_0000) &+ 0x0080_0000
                    index += 8
                }
                guard (nonFinite & 0x8000_0000) == .zero else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
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
    fileprivate static func storeVector<Vector: SIMD>(
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
///
/// With `workers` > 1, the whole coded rows of a call are decoded concurrently,
/// `waveRows` at a time. Rows whose destinations `targets` resolves are written
/// straight there; the others are batched and written in row order exactly as
/// one worker writes them: the same sink calls with the same bytes. Either way
/// the destination ends with the bytes one worker leaves, and a refused row
/// throws the same error. Each row's native bytes are a pure function of its
/// coded bytes and the role. With one worker (`DARKBLOOM_PREFIX_CACHE_PARALLEL_STAGE=0`,
/// or three active cores or fewer) every row takes the original sequential path.
/// The workers run at the caller's quality of service and take no locks.
final class CBv2NativeCheckpointRowDecoder {
    private static let maximumBatchBytes = 16 << 10
    /// Rows decoded concurrently per wave; its native rows are the decoder's
    /// largest scratch (`scratchBytes(workers:)`).
    static let defaultWaveRows = 1024
    /// Up to eight cores, at most half the active ones.
    static let defaultWorkers =
        parallelEnabled
        ? min(8, max(1, ProcessInfo.processInfo.activeProcessorCount / 2)) : 1
    /// `DARKBLOOM_PREFIX_CACHE_PARALLEL_STAGE=0` (or `false`, `no`, `off`)
    /// decodes on the calling thread alone; read once per process.
    static let parallelEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_PREFIX_CACHE_PARALLEL_STAGE"]?
            .trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "no", "off"].contains(raw)
    }()
    /// Concurrent rows of one wave at the widest supported row (512 FP32
    /// values), each worker's Float scratch and first-failure slot, and the
    /// wave's bookkeeping (pending rows, row lists and destination addresses).
    static func scratchBytes(workers: Int) -> Int {
        workers > 1 ? defaultWaveRows * (2048 + 128) + workers * (2048 + 8) : 0
    }

    /// Test-only: the decoder each import started in this task uses
    /// (`forImport`); nil uses the defaults. Task-local, so a test's decoder
    /// never reaches an import that another suite runs concurrently.
    @TaskLocal static var importDecoderForTesting: (@Sendable () -> CBv2NativeCheckpointRowDecoder)?

    static func forImport() -> CBv2NativeCheckpointRowDecoder {
        importDecoderForTesting?() ?? .init()
    }

    private let workers: Int
    private let waveRows: Int
    private let minimumRowsPerWorker: Int
    private var prepared: CBv2PreparedNativeCheckpointRowDecoder?
    private var row = Data()
    private var batch = Data()
    private var batchOffset: Int?
    private var failed = false
    /// Whole rows of the current call waiting for a concurrent decode:
    /// (offset in the call's data, coded, native offset).
    private var pending: [(source: Int, packed: Bool, nativeOffset: Int)] = []
    private var targets: ((Int, Int) -> [UnsafeMutableRawPointer]?)?
    private var arena: UnsafeMutableRawBufferPointer?
    private var workerScratch: UnsafeMutableRawBufferPointer?
    private var firstFailures: UnsafeMutableBufferPointer<Int>?
    var isComplete: Bool { !failed && row.isEmpty && batch.isEmpty }

    init(
        workers: Int = CBv2NativeCheckpointRowDecoder.defaultWorkers,
        waveRows: Int = CBv2NativeCheckpointRowDecoder.defaultWaveRows,
        minimumRowsPerWorker: Int = 32
    ) {
        self.workers = max(1, workers)
        self.waveRows = max(1, waveRows)
        self.minimumRowsPerWorker = max(1, minimumRowsPerWorker)
    }

    deinit { releaseScratch() }

    /// `targets`, when given, returns the destination address of each of a
    /// run of whole native rows (native offset, row count), or nil. With more
    /// than one worker, rows it resolves are decoded straight into those
    /// addresses; `write` receives the others, as before.
    func append(
        role: CBv2CheckpointPagedRoleLayout, byteOffset: Int, data: Data,
        targets: ((Int, Int) -> [UnsafeMutableRawPointer]?)? = nil,
        write: (Int, Data) throws -> Void
    ) throws {
        self.targets = targets
        defer { self.targets = nil }
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
        let concurrent = workers > 1
        pending.removeAll(keepingCapacity: true)
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
                        if concurrent {
                            pending.append((consumed, packed, nativeOffset))
                            if pending.count == waveRows {
                                try drainPending(
                                    source, decoder: prepared, role: role, write: write)
                            }
                        } else {
                            try appendRow(
                                fragment, packed: packed, nativeOffset: nativeOffset,
                                nativeRowBytes: role.nativeRowBytes, decoder: prepared, write: write
                            )
                        }
                    } else {
                        try drainPending(source, decoder: prepared, role: role, write: write)
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
                try drainPending(source, decoder: prepared, role: role, write: write)
            }
            try flush(write: write)
        } catch {
            pending.removeAll(keepingCapacity: true)
            failed = true
            // A rejected row never enters the batch. Preserve earlier completed
            // rows; flush clears before propagating a sink failure, so no retry.
            try flush(write: write)
            throw error
        }
    }

    /// Decodes the pending whole rows concurrently, a wave at a time, then
    /// batches them in order. A row that fails to decode throws where one
    /// worker would: after the earlier rows, before any later one.
    private func drainPending(
        _ source: UnsafeRawBufferPointer, decoder: CBv2PreparedNativeCheckpointRowDecoder,
        role: CBv2CheckpointPagedRoleLayout, write: (Int, Data) throws -> Void
    ) throws {
        guard !pending.isEmpty else { return }
        defer { pending.removeAll(keepingCapacity: true) }
        let nativeRowBytes = role.nativeRowBytes
        let (arena, scratch, firstFailures) = scratchBuffers(role: role)
        var start = 0
        while start < pending.count {
            let wave = pending[start ..< min(pending.count, start + waveRows)]
            if let pointers = resolveTargets(wave, rowBytes: nativeRowBytes) {
                // Earlier batched rows reach the sink first, as in row order.
                try flush(write: write)
                let job = CBv2RowDirectWave(
                    source: source, scratch: scratch, firstFailures: firstFailures,
                    decoder: decoder, rows: wave.map { ($0.source, $0.packed) },
                    targets: pointers, packedRowBytes: role.packedRowBytes,
                    nativeRowBytes: nativeRowBytes, scratchStride: role.width * 4)
                if let refused = job.run(
                    strips: min(workers, max(1, wave.count / minimumRowsPerWorker)))
                {
                    // Leave what one worker leaves: the staged rows before the refused
                    // one. Later rows return to the zeros they were staged with (each
                    // row is staged once per import), and so does the refused row.
                    for pointer in pointers[refused...] {
                        pointer.initializeMemory(
                            as: UInt8.self, repeating: 0, count: nativeRowBytes)
                    }
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                start += wave.count
                continue
            }
            let packedSlots = wave.indices.filter { pending[$0].packed }
            let job = CBv2RowDecodeWave(
                source: source, arena: arena, scratch: scratch, firstFailures: firstFailures,
                decoder: decoder, rows: packedSlots.map { (pending[$0].source, $0 - start) },
                packedRowBytes: role.packedRowBytes, nativeRowBytes: nativeRowBytes,
                scratchStride: role.width * 4)
            let refused =
                job.run(strips: min(workers, max(1, packedSlots.count / minimumRowsPerWorker)))
                ?? Int.max
            for index in wave.indices {
                let item = pending[index]
                let slot = index - start
                // `appendRow`'s order: flush the batch, then decode, then append.
                try flushUnlessAdjacent(
                    nativeOffset: item.nativeOffset, rowBytes: nativeRowBytes, write: write)
                // Every decode refusal is this error (`CBv2PreparedNativeCheckpointRowDecoder`).
                if slot == refused { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                let native =
                    item.packed
                    ? UnsafeRawBufferPointer(
                        rebasing: arena[slot * nativeRowBytes ..< (slot + 1) * nativeRowBytes])
                    : UnsafeRawBufferPointer(
                        rebasing: source[item.source ..< item.source + nativeRowBytes])
                batch.append(
                    native.baseAddress!.assumingMemoryBound(to: UInt8.self), count: native.count)
                if batchOffset == nil { batchOffset = item.nativeOffset }
            }
            start += wave.count
        }
    }

    /// Destination addresses for every row of `wave`, resolved per run of
    /// consecutive native offsets; nil unless every row resolves.
    private func resolveTargets(
        _ wave: ArraySlice<(source: Int, packed: Bool, nativeOffset: Int)>, rowBytes: Int
    ) -> [UnsafeMutableRawPointer]? {
        guard let targets else { return nil }
        var pointers: [UnsafeMutableRawPointer] = []
        pointers.reserveCapacity(wave.count)
        var runStart = wave.startIndex
        while runStart < wave.endIndex {
            var runEnd = runStart + 1
            while runEnd < wave.endIndex,
                wave[runEnd].nativeOffset == wave[runEnd - 1].nativeOffset + rowBytes
            {
                runEnd += 1
            }
            guard let run = targets(wave[runStart].nativeOffset, runEnd - runStart),
                run.count == runEnd - runStart
            else { return nil }
            pointers.append(contentsOf: run)
            runStart = runEnd
        }
        return pointers
    }

    private func scratchBuffers(role: CBv2CheckpointPagedRoleLayout) -> (
        UnsafeMutableRawBufferPointer, UnsafeMutableRawBufferPointer,
        UnsafeMutableBufferPointer<Int>
    ) {
        let arenaBytes = waveRows * role.nativeRowBytes
        let scratchBytes = workers * role.width * 4
        if let arena, arena.count >= arenaBytes, let workerScratch,
            workerScratch.count >= scratchBytes, let firstFailures
        {
            return (arena, workerScratch, firstFailures)
        }
        releaseScratch()
        let newArena = UnsafeMutableRawBufferPointer.allocate(byteCount: arenaBytes, alignment: 16)
        let newScratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: scratchBytes, alignment: MemoryLayout<Float>.alignment)
        let newFailures = UnsafeMutableBufferPointer<Int>.allocate(capacity: workers)
        newFailures.initialize(repeating: Int.max)
        arena = newArena
        workerScratch = newScratch
        firstFailures = newFailures
        return (newArena, newScratch, newFailures)
    }

    private func releaseScratch() {
        arena?.deallocate()
        workerScratch?.deallocate()
        firstFailures?.deallocate()
        arena = nil
        workerScratch = nil
        firstFailures = nil
    }

    private func flushUnlessAdjacent(
        nativeOffset: Int, rowBytes: Int, write: (Int, Data) throws -> Void
    ) throws {
        if let batchOffset,
            batchOffset + batch.count != nativeOffset
                || batch.count + rowBytes > Self.maximumBatchBytes
        {
            try flush(write: write)
        }
    }

    private func appendRow(
        _ bytes: UnsafeRawBufferPointer, packed: Bool, nativeOffset: Int,
        nativeRowBytes: Int, decoder: CBv2PreparedNativeCheckpointRowDecoder,
        write: (Int, Data) throws -> Void
    ) throws {
        try flushUnlessAdjacent(nativeOffset: nativeOffset, rowBytes: nativeRowBytes, write: write)
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
        pending = []
        releaseScratch()
    }
}

/// One wave of coded rows decoded on `strips` workers. Each worker decodes a
/// contiguous range of rows into its own slots of `arena` with its own Float
/// scratch, stops at its first refused row and records that row's slot in
/// `firstFailures[strip]` (`Int.max` when none).
private struct CBv2RowDecodeWave: @unchecked Sendable {
    let source: UnsafeRawBufferPointer
    let arena: UnsafeMutableRawBufferPointer
    let scratch: UnsafeMutableRawBufferPointer
    let firstFailures: UnsafeMutableBufferPointer<Int>
    let decoder: CBv2PreparedNativeCheckpointRowDecoder
    /// (offset in `source`, slot in `arena`), in slot order.
    let rows: [(source: Int, slot: Int)]
    let packedRowBytes: Int
    let nativeRowBytes: Int
    let scratchStride: Int

    /// The lowest refused slot, or nil.
    func run(strips: Int) -> Int? {
        guard !rows.isEmpty else { return nil }
        if strips <= 1 {
            decode(strip: 0, of: 1)
        } else {
            DispatchQueue.concurrentPerform(iterations: strips) { decode(strip: $0, of: strips) }
        }
        let first = firstFailures.prefix(strips).min() ?? Int.max
        return first == Int.max ? nil : first
    }

    private func decode(strip: Int, of strips: Int) {
        firstFailures[strip] = Int.max
        let values = UnsafeMutableRawBufferPointer(
            rebasing: scratch[strip * scratchStride ..< (strip + 1) * scratchStride])
        for index in rows.count * strip / strips ..< rows.count * (strip + 1) / strips {
            let item = rows[index]
            let input = UnsafeRawBufferPointer(
                rebasing: source[item.source ..< item.source + packedRowBytes])
            let output = UnsafeMutableRawBufferPointer(
                rebasing: arena[item.slot * nativeRowBytes ..< (item.slot + 1) * nativeRowBytes])
            do {
                try decoder.decode(input, scratch: values, into: output)
            } catch {
                firstFailures[strip] = item.slot
                return
            }
        }
    }
}

/// One wave of whole rows written straight to their destinations on `strips`
/// workers: coded rows decoded, recent rows copied. Rows have disjoint
/// destinations; each worker uses its own Float scratch, stops at its first
/// refused row and records its index in `firstFailures[strip]`.
private struct CBv2RowDirectWave: @unchecked Sendable {
    let source: UnsafeRawBufferPointer
    let scratch: UnsafeMutableRawBufferPointer
    let firstFailures: UnsafeMutableBufferPointer<Int>
    let decoder: CBv2PreparedNativeCheckpointRowDecoder
    /// (offset in `source`, coded) per row, in order.
    let rows: [(source: Int, packed: Bool)]
    let targets: [UnsafeMutableRawPointer]
    let packedRowBytes: Int
    let nativeRowBytes: Int
    let scratchStride: Int

    /// The lowest refused row, or nil.
    func run(strips: Int) -> Int? {
        guard !rows.isEmpty else { return nil }
        if strips <= 1 {
            write(strip: 0, of: 1)
        } else {
            DispatchQueue.concurrentPerform(iterations: strips) { write(strip: $0, of: strips) }
        }
        let first = firstFailures.prefix(strips).min() ?? Int.max
        return first == Int.max ? nil : first
    }

    private func write(strip: Int, of strips: Int) {
        firstFailures[strip] = Int.max
        let values = UnsafeMutableRawBufferPointer(
            rebasing: scratch[strip * scratchStride ..< (strip + 1) * scratchStride])
        for index in rows.count * strip / strips ..< rows.count * (strip + 1) / strips {
            let item = rows[index]
            let output = UnsafeMutableRawBufferPointer(start: targets[index], count: nativeRowBytes)
            if item.packed {
                let input = UnsafeRawBufferPointer(
                    rebasing: source[item.source ..< item.source + packedRowBytes])
                do {
                    try decoder.decode(input, scratch: values, into: output)
                } catch {
                    firstFailures[strip] = index
                    return
                }
            } else {
                // The recent band is opaque: preserve NaN payloads and signed zero.
                output.copyMemory(
                    from: UnsafeRawBufferPointer(
                        rebasing: source[item.source ..< item.source + nativeRowBytes]))
            }
        }
    }
}
