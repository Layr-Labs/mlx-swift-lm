import Foundation
import MLX
import XCTest

@_spi(Darkbloom) @testable import MLXLMCommon

final class NativeCheckpointRowCodecTests: XCTestCase {
    private let dtypes: [DType] = [.float16, .bfloat16, .float32]
    private let bitPairs = [(4, 4), (4, 8), (8, 4), (8, 8)]

    func testRowsMatchScalarReferenceAcrossDTypesWidthsAndBitWidths() throws {
        for (keyBits, valueBits) in bitPairs {
            let config = profile(keyBits: keyBits, valueBits: valueBits)
            for width in [64, 128, 192, 256, 512] {
                XCTAssertEqual(config.resolvedRotationBlockSize(headDim: width), 64)
                for dtype in dtypes {
                    for values in [false, true] {
                        let role = try role(
                            config: config, dtype: dtype, width: width, values: values)
                        let native = nativeData(sampleRow(width: width, seed: 7), dtype: dtype)
                        let input = nativeValues(native, dtype: dtype)
                        let reference = try PagedKVQuantizationReference.encode(
                            input, config: config, isKey: !values)
                        let expected = try PagedKVQuantizationReference.roundTrip(
                            input, config: config, isKey: !values)
                        let packed = try CBv2NativeCheckpointRowCodec.encode(native, role: role)

                        XCTAssertEqual(packed, Data(reference.bytes))
                        XCTAssertEqual(packed.count, role.packedRowBytes)
                        let decoded = try PagedKVQuantizationReference.decode(
                            Array(packed), headDim: width, config: config, isKey: !values)
                        XCTAssertEqual(
                            try CBv2NativeCheckpointRowCodec.decode(packed, role: role),
                            nativeData(decoded, dtype: dtype))
                        for (actual, expected) in zip(decoded, expected) {
                            XCTAssertEqual(actual, expected, accuracy: 0.00001)
                        }
                        if !values {
                            XCTAssertGreaterThan(
                                zip(decoded, reference.values).map { abs($0 - $1) }.max()!, 0.1,
                                "Key decoding must leave the rotated storage basis")
                        }
                        assertNativeRow(
                            try CBv2NativeCheckpointRowCodec.decode(packed, role: role),
                            reference: expected, dtype: dtype)
                    }
                }
            }
        }
    }

    func testPreparedRowCodecMatchesCheckpointRowsAndRefusesWhatTheyRefuse() throws {
        let profiles = [PagedKVQuantizationSelection.balanced, .k8v4, .k8v8].map {
            $0.configuration!
        }
        for config in profiles {
            for width in [64, 128, 192, 256, 512] {
                for dtype in dtypes {
                    for values in [false, true] {
                        let codec = try CBv2CheckpointRowCodec(
                            dtype: dtype, width: width, values: values, quantization: config)
                        let role = try role(
                            config: config, dtype: dtype, width: width, values: values)
                        XCTAssertEqual(codec.nativeRowBytes, role.nativeRowBytes)
                        XCTAssertEqual(codec.encodedRowBytes, role.packedRowBytes)
                        for seed in [3, 11] {
                            let native = nativeData(
                                sampleRow(width: width, seed: seed), dtype: dtype)
                            let encoded = try native.withUnsafeBytes { try codec.encode($0) }
                            XCTAssertEqual(
                                encoded, try CBv2NativeCheckpointRowCodec.encode(native, role: role)
                            )
                            XCTAssertEqual(
                                try encoded.withUnsafeBytes { try codec.decode($0) },
                                try CBv2NativeCheckpointRowCodec.decode(encoded, role: role))
                        }
                        var input = sampleRow(width: width, seed: 5)
                        input[width / 2] = .nan
                        XCTAssertThrowsError(
                            try nativeData(input, dtype: dtype).withUnsafeBytes {
                                try codec.encode($0)
                            })
                        XCTAssertThrowsError(
                            try Data(count: codec.encodedRowBytes - 1).withUnsafeBytes {
                                try codec.decode($0)
                            })
                    }
                }
            }
            for width in [0, 32, 96, 576] {
                XCTAssertThrowsError(
                    try CBv2CheckpointRowCodec(
                        dtype: .bfloat16, width: width, values: false, quantization: config))
            }
            XCTAssertThrowsError(
                try CBv2CheckpointRowCodec(
                    dtype: .int32, width: 128, values: true, quantization: config))
        }
    }

    /// Encode, decode, encode, decode: whether the second decode equals the
    /// first bit for bit (and the second bytes the first). Not for key rows,
    /// in any dtype, at any profile: a key row is rotated before it is
    /// quantized and rounded to its dtype after it is decoded, so the second
    /// encoding sees other group extremes, stores other scales and offsets,
    /// and its decode moves. Not for FP32 value rows either: the decode's
    /// unrounded affine sums give the second encoding other extremes. FP16 and
    /// BF16 value rows came back bit for bit from the same bytes on every row
    /// here (constant, all-zero, outlier, negative, tiny, mixed-binade and
    /// near-limit groups), which is not a proof for every input. A store that
    /// encodes restored rows again therefore drifts; the cluster's SSD prefix
    /// records keep a restored row's first encoding instead.
    func testReencodingDecodedRowsIsIdempotentOnlyForHalfPrecisionValues() throws {
        for selection in [PagedKVQuantizationSelection.balanced, .k8v4, .k8v8] {
            let config = selection.configuration!
            for dtype in dtypes {
                for values in [false, true] {
                    let label = "\(selection.rawValue) \(dtype) \(values ? "values" : "keys")"
                    let stable = values && dtype != .float32
                    var moved = 0
                    var compared = 0
                    for width in [64, 128, 256, 512] {
                        let codec = try CBv2CheckpointRowCodec(
                            dtype: dtype, width: width, values: values, quantization: config)
                        for (name, row) in idempotencePatterns(width: width, dtype: dtype) {
                            // A row the codec refuses (it would leave the dtype) is never stored.
                            guard
                                let first = try? nativeData(row, dtype: dtype).withUnsafeBytes({
                                    try codec.encode($0)
                                })
                            else { continue }
                            let decoded = try first.withUnsafeBytes { try codec.decode($0) }
                            let second = try? decoded.withUnsafeBytes { try codec.encode($0) }
                            let again = try second?.withUnsafeBytes { try codec.decode($0) }
                            compared += 1
                            if stable {
                                XCTAssertEqual(second, first, "\(label) width \(width) \(name)")
                                XCTAssertEqual(again, decoded, "\(label) width \(width) \(name)")
                            } else if again != decoded {
                                moved += 1
                            }
                        }
                    }
                    XCTAssertGreaterThan(compared, 0, label)
                    if !stable {
                        XCTAssertGreaterThan(moved, 0, "\(label): a decoded row encodes in place")
                    }
                }
            }
        }
    }

    func testPreparedRestoreMatchesReferenceBytesAcrossProfilesAndRoleChanges() throws {
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        for groupSize in [32, 64, 128] {
            for width in [64, 128, 192, 256, 512] where width.isMultiple(of: groupSize) {
                for (keyBits, valueBits) in bitPairs {
                    let config = PagedKVQuantizationConfig(
                        keyBits: keyBits, valueBits: valueBits, groupSize: groupSize,
                        rotationBlockSize: 512, recentTokenCount: 3)
                    for dtype in dtypes {
                        for values in [false, true] {
                            let role = try role(
                                config: config, dtype: dtype, width: width, values: values,
                                position: 9, tokenStart: 2)
                            let fixture = try streamFixture(role: role)
                            var restored = Data()
                            var retained: [(Int, Data)] = []
                            try decoder.append(role: role, byteOffset: 0, data: fixture.bytes) {
                                offset, bytes in
                                XCTAssertEqual(offset, restored.count)
                                XCTAssertLessThanOrEqual(bytes.count, 16 << 10)
                                XCTAssertTrue(bytes.count.isMultiple(of: role.nativeRowBytes))
                                restored.append(bytes)
                                retained.append((offset, bytes))
                            }
                            XCTAssertEqual(restored, fixture.restored)
                            XCTAssertTrue(decoder.isComplete)
                            for (offset, bytes) in retained {
                                XCTAssertEqual(
                                    bytes,
                                    fixture.restored.subdata(in: offset ..< offset + bytes.count))
                            }
                        }
                    }
                }
            }
        }
    }

    func testSIMDNativeRoundingMatchesScalarAtTiesAndSubnormalBoundaries() throws {
        let groupSize = 32
        for dtype: DType in [.float16, .bfloat16] {
            let step: Float = dtype == .float16 ? 1 / 1024 : 1 / 128
            let halfSubnormal: Float =
                dtype == .float16
                ? Float(sign: .plus, exponent: -25, significand: 1)
                : Float(bitPattern: 0x0000_8000)
            let coefficients: [(scale: Float, offset: Float)] = [
                (step, 1 + step / 2), (step, -1 - step / 2),
                (halfSubnormal, 0), (halfSubnormal, -halfSubnormal),
                (-Float.zero, -Float.zero), (step / 2, 2 - step / 2),
            ]
            for (keyBits, valueBits) in bitPairs {
                let config = PagedKVQuantizationConfig(
                    keyBits: keyBits, valueBits: valueBits, groupSize: groupSize,
                    rotationBlockSize: 0, recentTokenCount: 3)
                for values in [false, true] {
                    let role = try role(config: config, dtype: dtype, width: 192, values: values)
                    let packed = try affineRow(
                        role: role,
                        codes: [0, 1, 2, 3, 7, 8, 14, 15, 16, 17, 31, 63, 127, 128, 254, 255],
                        coefficients: coefficients)
                    let reference = try PagedKVQuantizationReference.decode(
                        Array(packed), headDim: role.width, config: config, isKey: !values)
                    let expected = nativeData(reference, dtype: dtype)
                    expected.withUnsafeBytes { bytes in
                        let one: UInt16 = dtype == .float16 ? 0x3c00 : 0x3f80
                        XCTAssertEqual(bytes.loadUnaligned(as: UInt16.self).littleEndian, one)
                        XCTAssertEqual(
                            bytes.loadUnaligned(fromByteOffset: 2, as: UInt16.self).littleEndian,
                            one + 2, "An odd mantissa tie rounds up to the next even mantissa")
                        XCTAssertEqual(
                            bytes.loadUnaligned(
                                fromByteOffset: (2 * groupSize + 2) * 2, as: UInt16.self
                            ).littleEndian, 1)
                        XCTAssertEqual(
                            bytes.loadUnaligned(
                                fromByteOffset: (3 * groupSize) * 2, as: UInt16.self
                            ).littleEndian, 0x8000)
                        XCTAssertEqual(
                            bytes.loadUnaligned(
                                fromByteOffset: (4 * groupSize) * 2, as: UInt16.self
                            ).littleEndian, 0x8000)
                    }
                    let envelope = Data(repeating: 0xff, count: 3) + packed + Data([0xff])
                    let payload = envelope[3 ..< 3 + packed.count]
                    XCTAssertEqual(payload.startIndex, 3)
                    XCTAssertEqual(
                        try CBv2NativeCheckpointRowCodec.decode(payload, role: role), expected)
                    for fragmentSize in [1, 257, packed.count] {
                        let decoder = CBv2NativeCheckpointRowDecoder()
                        defer { decoder.close() }
                        var restored = Data()
                        var offset = 0
                        while offset < payload.count {
                            let end = min(offset + fragmentSize, payload.count)
                            try decoder.append(
                                role: role, byteOffset: offset,
                                data: payload[
                                    (payload.startIndex + offset) ..< (payload.startIndex + end)]
                            ) { nativeOffset, bytes in
                                XCTAssertEqual(nativeOffset, 0)
                                restored.append(bytes)
                            }
                            offset = end
                        }
                        XCTAssertEqual(restored, expected)
                        XCTAssertTrue(decoder.isComplete)
                    }
                }
            }
        }
    }

    func testSIMDAffineReconstructionPreservesFMAExtremesAndFloat32Subnormals() throws {
        let tiny = Float.leastNonzeroMagnitude
        let coefficients: [(scale: Float, offset: Float)] = [
            (.greatestFiniteMagnitude / 8, -.greatestFiniteMagnitude),
            (tiny, -8 * tiny), (-Float.zero, -Float.zero),
            (1, -8), (1 / 1024, (-7.5 / Float(1024)).nextUp),
            (Float.leastNormalMagnitude, -8 * Float.leastNormalMagnitude),
        ]
        XCTAssertTrue((coefficients[0].scale * 15).isInfinite)
        for (keyBits, valueBits) in bitPairs {
            let config = PagedKVQuantizationConfig(
                keyBits: keyBits, valueBits: valueBits, groupSize: 32,
                rotationBlockSize: 0, recentTokenCount: 3)
            for values in [false, true] {
                let role = try role(config: config, width: 192, values: values)
                let packed = try affineRow(
                    role: role, codes: [0, 1, 2, 3, 7, 8, 14, 15], coefficients: coefficients)
                let reference = try PagedKVQuantizationReference.decode(
                    Array(packed), headDim: role.width, config: config, isKey: !values)
                XCTAssertTrue(reference.allSatisfy(\.isFinite))
                XCTAssertEqual(reference[32 + 4].bitPattern, 0x8000_0001)
                XCTAssertEqual(reference[32 + 6].bitPattern, 6)
                XCTAssertEqual(reference[64].bitPattern, 0x8000_0000)
                XCTAssertEqual(
                    try CBv2NativeCheckpointRowCodec.decode(packed, role: role),
                    nativeData(reference, dtype: .float32))
            }
        }
    }

    func testSIMDButterfliesPreserveCancellationAcrossRotationBlocks() throws {
        let patterns: [(scale: Float, offset: Float)] = [
            (1 / 1024, (-7.5 / Float(1024)).nextUp),
            (Float.leastNormalMagnitude, -8 * Float.leastNormalMagnitude),
            (2, -15), (-Float.zero, -Float.zero),
        ]
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        for width in [64, 128, 192, 256, 512] {
            for rotation in [32, 64, 128, 256, 512] {
                for bits in [4, 8] {
                    let config = PagedKVQuantizationConfig(
                        keyBits: bits, valueBits: 8, groupSize: 32,
                        rotationBlockSize: rotation, recentTokenCount: 3)
                    for dtype in dtypes {
                        let role = try role(config: config, dtype: dtype, width: width)
                        let coefficients = (0 ..< width / config.groupSize).map {
                            patterns[$0 % patterns.count]
                        }
                        let packed = try affineRow(
                            role: role,
                            codes: [0, 15, 8, 3, 7, 1, 14, 2, 15, 0, 4, 8, 1, 14, 7, 5],
                            coefficients: coefficients)
                        let reference = try PagedKVQuantizationReference.decode(
                            Array(packed), headDim: width, config: config, isKey: true)
                        let expected = nativeData(reference, dtype: dtype)
                        XCTAssertEqual(
                            try CBv2NativeCheckpointRowCodec.decode(packed, role: role), expected)
                        var restored = Data()
                        try decoder.append(role: role, byteOffset: 0, data: packed) {
                            offset, bytes in
                            XCTAssertEqual(offset, 0)
                            restored.append(bytes)
                        }
                        XCTAssertEqual(restored, expected)
                        XCTAssertTrue(decoder.isComplete)
                    }
                }
            }
        }
    }

    func testSIMDConversionRejectsOneOverflowLaneWithoutPublishing() throws {
        let config = PagedKVQuantizationConfig(
            keyBits: 4, valueBits: 8, groupSize: 32,
            rotationBlockSize: 0, recentTokenCount: 3)
        for dtype in dtypes {
            let scale: Float = dtype == .float16 ? 65536 : .greatestFiniteMagnitude
            for values in [false, true] {
                let role = try role(config: config, dtype: dtype, values: values)
                for lane in [0, 3, 7, 8, 31, 32, 63] {
                    var codes = [UInt8](repeating: 0, count: role.width)
                    codes[lane] = dtype == .float32 ? 2 : 1
                    let packed = try affineRow(
                        role: role, codes: codes,
                        coefficients: Array(
                            repeating: (scale: scale, offset: 0),
                            count: role.width / config.groupSize))
                    if dtype == .float32 {
                        XCTAssertThrowsError(
                            try PagedKVQuantizationReference.decode(
                                Array(packed), headDim: role.width, config: config, isKey: !values))
                    } else {
                        let reference = try PagedKVQuantizationReference.decode(
                            Array(packed), headDim: role.width, config: config, isKey: !values)
                        XCTAssertTrue(reference.allSatisfy(\.isFinite))
                    }
                    XCTAssertThrowsError(
                        try CBv2NativeCheckpointRowCodec.decode(packed, role: role))
                    let decoder = CBv2NativeCheckpointRowDecoder()
                    defer { decoder.close() }
                    var writes = 0
                    XCTAssertThrowsError(
                        try decoder.append(role: role, byteOffset: 0, data: packed) { _, _ in
                            writes += 1
                        }
                    ) {
                        XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
                    }
                    XCTAssertEqual(writes, 0)
                    XCTAssertFalse(decoder.isComplete)
                }
            }
        }
    }

    func testRowFiniteCheckMatchesReferenceForEveryLane() throws {
        // Each row has one nonzero code and a scale near its dtype's limit; key rotation
        // spreads that code over its block. The reference decides whether the row is stored.
        for dtype in dtypes {
            let scales: [Float]
            switch dtype {
            case .float16: scales = [65519.99, 65520]
            case .bfloat16:
                scales = [Float(bitPattern: 0x7f7f_7fff), Float(bitPattern: 0x7f7f_8000)]
            default: scales = [.greatestFiniteMagnitude / 2, .greatestFiniteMagnitude]
            }
            var outcomes = (stored: 0, refused: 0)
            for rotationBlockSize in [0, 32] {
                let config = PagedKVQuantizationConfig(
                    keyBits: 8, valueBits: 8, groupSize: 32,
                    rotationBlockSize: rotationBlockSize, recentTokenCount: 3)
                for width in [64, 192] {
                    for values in [false, true] {
                        let role = try role(
                            config: config, dtype: dtype, width: width, values: values)
                        for scale in scales {
                            for code: UInt8 in [1, 2, 255] {
                                for lane in 0 ..< width {
                                    var codes = [UInt8](repeating: 0, count: width)
                                    codes[lane] = code
                                    let packed = try affineRow(
                                        role: role, codes: codes,
                                        coefficients: Array(
                                            repeating: (scale: scale, offset: 0),
                                            count: width / config.groupSize))
                                    let expected = try? PagedKVQuantizationReference.decode(
                                        Array(packed), headDim: width, config: config,
                                        isKey: !values)
                                    let native = expected.map { nativeData($0, dtype: dtype) }
                                    let context =
                                        "\(dtype) rotation \(rotationBlockSize) "
                                        + "width \(width) values \(values) code \(code) lane \(lane)"
                                    var writes = Data()
                                    let decoder = CBv2NativeCheckpointRowDecoder()
                                    defer { decoder.close() }
                                    if let native,
                                        nativeValues(native, dtype: dtype).allSatisfy(\.isFinite)
                                    {
                                        outcomes.stored += 1
                                        XCTAssertEqual(
                                            try CBv2NativeCheckpointRowCodec.decode(
                                                packed, role: role),
                                            native, context)
                                        try decoder.append(role: role, byteOffset: 0, data: packed)
                                        {
                                            _, bytes in writes.append(bytes)
                                        }
                                        XCTAssertEqual(writes, native, context)
                                    } else {
                                        outcomes.refused += 1
                                        XCTAssertThrowsError(
                                            try CBv2NativeCheckpointRowCodec.decode(
                                                packed, role: role),
                                            context
                                        ) {
                                            XCTAssertEqual(
                                                $0 as? CBv2CompleteCheckpointError,
                                                .incompatibleCheckpoint, context)
                                        }
                                        XCTAssertThrowsError(
                                            try decoder.append(
                                                role: role, byteOffset: 0, data: packed
                                            ) {
                                                _, bytes in writes.append(bytes)
                                            }, context
                                        ) {
                                            XCTAssertEqual(
                                                $0 as? CBv2CompleteCheckpointError,
                                                .incompatibleCheckpoint, context)
                                        }
                                        XCTAssertTrue(writes.isEmpty, context)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            XCTAssertGreaterThan(outcomes.stored, 0, "\(dtype)")
            XCTAssertGreaterThan(outcomes.refused, 0, "\(dtype)")
        }
    }

    func testZeroRangesAndFiniteNativeExtremes() throws {
        for dtype in dtypes {
            let maximum: Float
            switch dtype {
            case .float16: maximum = Float(Float16.greatestFiniteMagnitude)
            case .bfloat16: maximum = Float(bitPattern: 0x7f7f_0000)
            default: maximum = Float.greatestFiniteMagnitude
            }
            for (keyBits, valueBits) in bitPairs {
                let config = profile(
                    keyBits: keyBits, valueBits: valueBits, rotationBlockSize: 0)
                for values in [false, true] {
                    let role = try role(config: config, dtype: dtype, values: values)
                    for constant: Float in [0, -0.0, 1.5, -2.25, maximum, -maximum] {
                        let native = nativeData(Array(repeating: constant, count: 64), dtype: dtype)
                        let packed = try CBv2NativeCheckpointRowCodec.encode(native, role: role)
                        let layout = try config.rowLayout(headDim: 64)
                        let scaleOffset = values ? layout.valueScaleOffset : layout.keyScaleOffset
                        XCTAssertEqual(
                            Array(packed[scaleOffset ..< scaleOffset + 4]), [0, 0, 0, 0],
                            "A constant native row has a zero affine range")
                        assertNativeRow(
                            try CBv2NativeCheckpointRowCodec.decode(packed, role: role),
                            reference: Array(repeating: constant, count: 64), dtype: dtype)
                    }

                    let rotatedRole = try self.role(
                        config: profile(keyBits: keyBits, valueBits: valueBits),
                        dtype: dtype, width: 192, values: values)
                    let magnitude = maximum / 256
                    let input = nativeValues(
                        nativeData(
                            (0 ..< 192).map { $0.isMultiple(of: 2) ? magnitude : -magnitude },
                            dtype: dtype), dtype: dtype)
                    let packed = try CBv2NativeCheckpointRowCodec.encode(
                        nativeData(input, dtype: dtype), role: rotatedRole)
                    let expected = try PagedKVQuantizationReference.roundTrip(
                        input, config: rotatedRole.key.quantization!, isKey: !values)
                    assertNativeRow(
                        try CBv2NativeCheckpointRowCodec.decode(packed, role: rotatedRole),
                        reference: expected, dtype: dtype)
                }
            }
        }
    }

    func testRepeatedNativeRestoreUsesTheNewRoundedInputNotOriginalBytes() throws {
        for dtype in dtypes {
            for values in [false, true] {
                let role = try role(dtype: dtype, width: 192, values: values)
                let original = nativeData(sampleRow(width: 192, seed: 19), dtype: dtype)
                var current = original
                for _ in 0 ..< 3 {
                    let reference = try PagedKVQuantizationReference.roundTrip(
                        nativeValues(current, dtype: dtype), config: role.key.quantization!,
                        isKey: !values)
                    let packed = try CBv2NativeCheckpointRowCodec.encode(current, role: role)
                    let next = try CBv2NativeCheckpointRowCodec.decode(packed, role: role)
                    assertNativeRow(next, reference: reference, dtype: dtype)
                    current = next
                }
                XCTAssertNotEqual(
                    current, original,
                    "Native restore is deliberately lossy, not the live-packed byte-preserving contract"
                )
            }
        }
    }

    func testMalformedAffineCoefficientsAndLengthsAreRejected() throws {
        for (keyBits, valueBits) in bitPairs {
            let config = profile(keyBits: keyBits, valueBits: valueBits)
            let layout = try config.rowLayout(headDim: 192)
            for values in [false, true] {
                let role = try role(config: config, width: 192, values: values)
                let packed = Data(
                    try PagedKVQuantizationReference.encode(
                        sampleRow(width: 192, seed: 3), config: config, isKey: !values
                    ).bytes)
                let scaleStart = values ? layout.valueScaleOffset : layout.keyScaleOffset
                let offsetStart = values ? layout.valueOffsetOffset : layout.keyOffsetOffset
                var malformed = [Data(), Data(packed.dropLast()), Data(packed.dropLast(4))]
                malformed.append(packed + Data([0]))
                for group in 0 ..< 3 {
                    for scale: Float in [.nan, .infinity, -.infinity, -1] {
                        malformed.append(
                            replacingFloat(packed, at: scaleStart + group * 4, with: scale))
                    }
                    for offset: Float in [.nan, .infinity, -.infinity] {
                        malformed.append(
                            replacingFloat(packed, at: offsetStart + group * 4, with: offset))
                    }
                }
                for bytes in malformed {
                    XCTAssertThrowsError(
                        try PagedKVQuantizationReference.decode(
                            Array(bytes), headDim: 192, config: config, isKey: !values))
                    XCTAssertThrowsError(try CBv2NativeCheckpointRowCodec.decode(bytes, role: role))
                    {
                        XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
                    }
                }
            }
        }
    }

    func testMixedSignedExtremeRowsNeverPublishUnrestorableBytes() throws {
        for dtype in dtypes {
            let maximum: Float
            switch dtype {
            case .float16: maximum = Float(Float16.greatestFiniteMagnitude)
            case .bfloat16: maximum = Float(bitPattern: 0x7f7f_0000)
            default: maximum = .greatestFiniteMagnitude
            }
            let native = nativeData(
                (0 ..< 64).map {
                    $0.isMultiple(of: 2) ? maximum : -maximum
                }, dtype: dtype)
            for (keyBits, valueBits) in bitPairs {
                let config = profile(keyBits: keyBits, valueBits: valueBits, rotationBlockSize: 0)
                for values in [false, true] {
                    let role = try role(config: config, dtype: dtype, values: values)
                    let encoded = try CBv2NativeCheckpointRowCodec.encode(native, role: role)
                    let decoded = try CBv2NativeCheckpointRowCodec.decode(encoded, role: role)
                    assertNativeRow(
                        decoded, reference: nativeValues(native, dtype: dtype), dtype: dtype)
                    XCTAssertTrue(
                        try PagedKVQuantizationReference.decode(
                            Array(encoded), headDim: 64, config: config, isKey: !values
                        ).allSatisfy(\.isFinite))
                }
            }
        }
    }

    func testNonFiniteNativeInputsAndIncorrectRowSizesAreRejected() throws {
        for dtype in dtypes {
            for values in [false, true] {
                let role = try role(dtype: dtype, values: values)
                let native = nativeData(sampleRow(width: 64, seed: 2), dtype: dtype)
                for bytes in [Data(), Data(native.dropLast()), native + Data([0])] {
                    XCTAssertThrowsError(try CBv2NativeCheckpointRowCodec.encode(bytes, role: role))
                    {
                        XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .invalidSegment)
                    }
                }
                for value: Float in [.nan, .infinity, -.infinity] {
                    var input = sampleRow(width: 64, seed: 2)
                    input[17] = value
                    XCTAssertThrowsError(
                        try CBv2NativeCheckpointRowCodec.encode(
                            nativeData(input, dtype: dtype), role: role)
                    ) {
                        XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
                    }
                }
            }
        }
    }

    func testFiniteCoefficientsCannotProduceNonFiniteRowsOrOverflowNativeDType() throws {
        for values in [false, true] {
            let config = profile(rotationBlockSize: 0)
            let layout = try config.rowLayout(headDim: 64)
            let scaleOffset = values ? layout.valueScaleOffset : layout.keyScaleOffset
            let offsetOffset = values ? layout.valueOffsetOffset : layout.keyOffsetOffset
            let packedRowBytes = values ? layout.valueRowBytes : layout.keyRowBytes
            var overflow = Data(repeating: 0xff, count: packedRowBytes)
            overflow = replacingFloat(overflow, at: scaleOffset, with: .greatestFiniteMagnitude)
            overflow = replacingFloat(overflow, at: offsetOffset, with: 0)
            XCTAssertThrowsError(
                try PagedKVQuantizationReference.decode(
                    Array(overflow), headDim: 64, config: config, isKey: !values))
            let float32Role = try role(config: config, values: values)
            XCTAssertThrowsError(
                try CBv2NativeCheckpointRowCodec.decode(overflow, role: float32Role)
            ) {
                XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
            }

            for dtype: DType in [.float16, .bfloat16] {
                let role = try role(config: config, dtype: dtype, values: values)
                let tooLarge: Float =
                    dtype == .float16
                    ? Float(Float16.greatestFiniteMagnitude) * 2 : .greatestFiniteMagnitude
                var packed = Data(repeating: 0, count: packedRowBytes)
                packed = replacingFloat(packed, at: offsetOffset, with: tooLarge)
                let reference = try PagedKVQuantizationReference.decode(
                    Array(packed), headDim: 64, config: config, isKey: !values)
                XCTAssertTrue(reference.allSatisfy(\.isFinite))
                XCTAssertThrowsError(try CBv2NativeCheckpointRowCodec.decode(packed, role: role)) {
                    XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
                }
            }
        }
    }

    func testInverseRotationOverflowIsRejectedBeforeAnyNativeBytesAreWritten() throws {
        let config = profile()
        let layout = try config.rowLayout(headDim: 64)
        for dtype in dtypes {
            let role = try role(config: config, dtype: dtype)
            let magnitude: Float = dtype == .float16 ? 16384 : .greatestFiniteMagnitude / 4
            let packed = replacingFloat(
                Data(count: role.packedRowBytes), at: layout.keyOffsetOffset, with: magnitude)
            if dtype == .float16 {
                let decoded = try PagedKVQuantizationReference.decode(
                    Array(packed), headDim: 64, config: config, isKey: true)
                XCTAssertTrue(decoded.allSatisfy(\.isFinite))
                XCTAssertGreaterThan(abs(decoded[0]), Float(Float16.greatestFiniteMagnitude))
            } else {
                XCTAssertThrowsError(
                    try PagedKVQuantizationReference.decode(
                        Array(packed), headDim: 64, config: config, isKey: true))
            }
            XCTAssertThrowsError(try CBv2NativeCheckpointRowCodec.decode(packed, role: role))
            let decoder = CBv2NativeCheckpointRowDecoder()
            defer { decoder.close() }
            var writes = 0
            XCTAssertThrowsError(
                try decoder.append(role: role, byteOffset: 0, data: packed) { _, _ in
                    writes += 1
                }
            ) {
                XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
            }
            XCTAssertEqual(writes, 0)
            XCTAssertFalse(decoder.isComplete)
        }
    }

    func testNativeLayoutOmitsPackedRecentMirrorAndClampsRecentRange() throws {
        for recentCount in [0, 3, 100] {
            for dtype in dtypes {
                for values in [false, true] {
                    let config = profile(recentTokenCount: recentCount)
                    let role = try role(
                        config: config, dtype: dtype, width: 192, values: values,
                        position: 19, tokenStart: 11)
                    let recent = min(recentCount, 8)
                    XCTAssertEqual(role.nativeCount, recent)
                    XCTAssertEqual(role.nativeStart, 19 - recent)
                    XCTAssertEqual(role.packedCount, 8 - recent)
                    XCTAssertEqual(role.nativeRowBytes, 192 * dtype.size)
                    XCTAssertEqual(
                        role.bytesPerHead,
                        (8 - recent) * role.packedRowBytes + recent * role.nativeRowBytes)
                }
            }
        }
        XCTAssertGreaterThanOrEqual(CBv2NativeCheckpointRowCodec.scratchBytes, 512 * 4)
        XCTAssertLessThanOrEqual(CBv2NativeCheckpointRowCodec.scratchBytes, 64 << 10)
    }

    func testOneByteAnd257ByteFragmentsPreserveNativeOffsetsAndExactRecentBits() throws {
        for (keyBits, valueBits) in [(4, 8), (8, 4)] {
            let config = profile(keyBits: keyBits, valueBits: valueBits)
            for dtype in dtypes {
                for values in [false, true] {
                    let key = PagedKVGroupKey(
                        kvHeads: 2, headDim: 192, dtype: dtype, valueHeadDim: 64,
                        quantization: config)
                    let role = try CBv2CheckpointPagedRoleLayout(
                        key: key, position: 40, tokenStart: 7, values: values,
                        preservePackedRecent: false)
                    XCTAssertEqual(role.width, values ? 64 : 192)
                    XCTAssertEqual(role.nativeRowBytes, (values ? 64 : 192) * dtype.size)
                    let fixture = try streamFixture(role: role)
                    for fragmentSize in [1, 257, fixture.bytes.count] {
                        let decoder = CBv2NativeCheckpointRowDecoder()
                        defer { decoder.close() }
                        var restored = Data()
                        var retained: [(Int, Data)] = []
                        var offset = 0
                        var splitCoefficient = false
                        var crossedRecent = false
                        var crossedHead = false
                        XCTAssertTrue(decoder.isComplete)
                        while offset < fixture.bytes.count {
                            let end = min(offset + fragmentSize, fixture.bytes.count)
                            try decoder.append(
                                role: role, byteOffset: offset,
                                data: fixture.bytes.subdata(in: offset ..< end)
                            ) { nativeOffset, bytes in
                                XCTAssertEqual(nativeOffset, restored.count)
                                XCTAssertTrue(bytes.count.isMultiple(of: role.nativeRowBytes))
                                XCTAssertLessThanOrEqual(bytes.count, 16 << 10)
                                XCTAssertEqual(
                                    bytes,
                                    fixture.restored.subdata(
                                        in: nativeOffset ..< nativeOffset + bytes.count))
                                restored.append(bytes)
                                retained.append((nativeOffset, bytes))
                            }
                            XCTAssertEqual(
                                restored.count / role.nativeRowBytes,
                                fixture.rowEnds.filter { $0 <= end }.count)
                            XCTAssertEqual(decoder.isComplete, fixture.rowEnds.contains(end))
                            splitCoefficient =
                                splitCoefficient
                                || fixture.coefficients.contains {
                                    $0.contains(end) && (end - $0.lowerBound) % 4 != 0
                                }
                            crossedRecent =
                                crossedRecent
                                || (0 ..< key.kvHeads).contains {
                                    let boundary =
                                        $0 * role.bytesPerHead
                                        + role.packedCount * role.packedRowBytes
                                    return offset < boundary && boundary < end
                                }
                            crossedHead =
                                crossedHead
                                || (offset < role.bytesPerHead && role.bytesPerHead < end)
                            offset = end
                        }
                        XCTAssertEqual(restored, fixture.restored)
                        XCTAssertTrue(decoder.isComplete)
                        for (nativeOffset, bytes) in retained {
                            XCTAssertEqual(
                                bytes,
                                fixture.restored.subdata(
                                    in: nativeOffset ..< nativeOffset + bytes.count),
                                "Retained callback Data must survive later buffer reuse")
                        }
                        if fragmentSize == fixture.bytes.count {
                            let rowsPerBatch = (16 << 10) / role.nativeRowBytes
                            XCTAssertEqual(
                                retained.count,
                                (fixture.rowEnds.count + rowsPerBatch - 1) / rowsPerBatch)
                        } else {
                            XCTAssertTrue(splitCoefficient)
                        }
                        if fragmentSize != 1 {
                            XCTAssertTrue(crossedRecent)
                            XCTAssertTrue(crossedHead)
                        }
                    }
                }
            }
        }
    }

    func testAllRecentRowsPassThroughWithoutInterpretingNaNsOrSignedZero() throws {
        for dtype in dtypes {
            for values in [false, true] {
                let role = try role(dtype: dtype, width: 192, values: values, position: 2)
                XCTAssertEqual(role.packedCount, 0)
                let fixture = try streamFixture(role: role)
                let decoder = CBv2NativeCheckpointRowDecoder()
                defer { decoder.close() }
                var restored = Data()
                let envelope = Data(repeating: 0xff, count: 7) + fixture.bytes + Data([0xff])
                let payload = envelope[7 ..< 7 + fixture.bytes.count]
                XCTAssertEqual(payload.startIndex, 7)
                try decoder.append(role: role, byteOffset: 0, data: payload) { offset, row in
                    XCTAssertEqual(offset, restored.count)
                    restored.append(row)
                }
                XCTAssertEqual(restored, fixture.bytes)
                XCTAssertTrue(decoder.isComplete)
            }
        }
    }

    func testNonconsecutiveOffsetsAndRoleChangesDoNotRetainPendingOutput() throws {
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        var retained: [(Data, Data)] = []
        for values in [false, true] {
            let role = try role(width: 192, values: values)
            let fixture = try streamFixture(role: role)
            let tokenCount = role.position - role.tokenStart
            for (offset, count, nativeOffset) in [
                (0, role.packedRowBytes, 0),
                (
                    role.packedCount * role.packedRowBytes, role.nativeRowBytes,
                    role.packedCount * role.nativeRowBytes
                ),
                (role.bytesPerHead, role.packedRowBytes, tokenCount * role.nativeRowBytes),
            ] {
                var writes = 0
                try decoder.append(
                    role: role, byteOffset: offset,
                    data: fixture.bytes.subdata(in: offset ..< offset + count)
                ) { actualOffset, bytes in
                    XCTAssertEqual(actualOffset, nativeOffset)
                    let expected = fixture.restored.subdata(
                        in: nativeOffset ..< nativeOffset + role.nativeRowBytes)
                    XCTAssertEqual(bytes, expected)
                    retained.append((bytes, expected))
                    writes += 1
                }
                XCTAssertEqual(writes, 1, "Each append flushes before returning")
                XCTAssertTrue(decoder.isComplete)
            }
        }
        decoder.close()
        for (bytes, expected) in retained {
            XCTAssertEqual(bytes, expected)
        }
    }

    func testDecoderRejectsInvalidSegmentsAndDiscardsPartialRowsOnClose() throws {
        let role = try role()
        let packed = try CBv2NativeCheckpointRowCodec.encode(
            nativeData(sampleRow(width: 64, seed: 1), dtype: role.key.dtype), role: role)
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        var writes = 0
        let write: (Int, Data) -> Void = { _, _ in writes += 1 }
        let totalBytes = role.key.kvHeads * role.bytesPerHead
        for (offset, data) in [
            (-1, Data([0])), (totalBytes, Data([0])), (totalBytes - 1, Data([0, 0])),
            (0, Data()), (1, Data([0])),
        ] {
            XCTAssertThrowsError(
                try decoder.append(role: role, byteOffset: offset, data: data, write: write)
            ) {
                XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .invalidSegment)
            }
        }
        try decoder.append(role: role, byteOffset: 0, data: Data(packed.prefix(1)), write: write)
        XCTAssertFalse(decoder.isComplete)
        XCTAssertEqual(writes, 0)
        XCTAssertThrowsError(
            try decoder.append(
                role: role, byteOffset: 2, data: Data(packed[2 ..< 3]), write: write))
        decoder.close()
        XCTAssertTrue(decoder.isComplete)
        try decoder.append(role: role, byteOffset: 0, data: packed, write: write)
        XCTAssertEqual(writes, 1)
        XCTAssertTrue(decoder.isComplete)

        let nativeRole = try CBv2CheckpointPagedRoleLayout(
            key: .init(kvHeads: 1, headDim: 64), position: 1, values: false,
            preservePackedRecent: false)
        XCTAssertThrowsError(
            try decoder.append(
                role: nativeRole, byteOffset: 0, data: Data([0]), write: write))
        XCTAssertThrowsError(
            try CBv2NativeCheckpointRowCodec.encode(
                Data(repeating: 0, count: nativeRole.nativeRowBytes), role: nativeRole))
        XCTAssertThrowsError(try CBv2NativeCheckpointRowCodec.decode(packed, role: nativeRole))
    }

    func testMalformedFragmentedRowIsNeverWritten() throws {
        let config = profile()
        let role = try role(config: config)
        let layout = try config.rowLayout(headDim: 64)
        let packed = replacingFloat(
            Data(
                try PagedKVQuantizationReference.encode(
                    sampleRow(width: 64, seed: 1), config: config, isKey: true
                ).bytes),
            at: layout.keyScaleOffset, with: .nan)
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        var writes = 0
        let write: (Int, Data) -> Void = { _, _ in writes += 1 }
        try decoder.append(
            role: role, byteOffset: 0, data: Data(packed.dropLast()), write: write)
        XCTAssertFalse(decoder.isComplete)
        XCTAssertEqual(writes, 0)
        XCTAssertThrowsError(
            try decoder.append(
                role: role, byteOffset: packed.count - 1, data: Data(packed.suffix(1)), write: write
            )
        ) {
            XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
        }
        XCTAssertEqual(writes, 0)
        XCTAssertFalse(decoder.isComplete)
        decoder.close()
        XCTAssertTrue(decoder.isComplete)
    }

    func testRejectedRowFlushesOnlyEarlierValidatedRowsAndRequiresClose() throws {
        let config = profile(rotationBlockSize: 0)
        let role = try role(config: config, dtype: .float16, width: 192)
        let layout = try config.rowLayout(headDim: role.width)
        let fixture = try streamFixture(role: role)
        // The final group overflows only after earlier groups have been cast.
        let malformed = replacingFloat(
            Data(fixture.bytes.prefix(2 * role.packedRowBytes)),
            at: role.packedRowBytes + layout.keyOffsetOffset + 2 * 4,
            with: .greatestFiniteMagnitude)
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        var retained: [(Int, Data)] = []
        let write: (Int, Data) -> Void = { retained.append(($0, $1)) }
        XCTAssertThrowsError(
            try decoder.append(role: role, byteOffset: 0, data: malformed, write: write)
        ) {
            XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .incompatibleCheckpoint)
        }
        XCTAssertEqual(retained.count, 1)
        XCTAssertEqual(retained.first?.0, 0)
        let expected = Data(fixture.restored.prefix(role.nativeRowBytes))
        XCTAssertEqual(retained.first?.1, expected)
        XCTAssertFalse(decoder.isComplete)
        XCTAssertThrowsError(
            try decoder.append(
                role: role, byteOffset: 0, data: fixture.bytes, write: write))
        XCTAssertEqual(retained.count, 1)
        decoder.close()
        XCTAssertTrue(decoder.isComplete)
        try decoder.append(role: role, byteOffset: 0, data: fixture.bytes, write: write)
        XCTAssertTrue(decoder.isComplete)
        XCTAssertEqual(retained.first?.1, expected)
    }

    func testSinkFailureIsNotRetriedOrReportedComplete() throws {
        enum Failure: Error { case rejected }
        let role = try role(width: 512, position: 12)
        let fixture = try streamFixture(role: role)
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        var retained: [Data] = []
        XCTAssertThrowsError(
            try decoder.append(role: role, byteOffset: 0, data: fixture.bytes) { offset, bytes in
                XCTAssertEqual(offset, 0)
                retained.append(bytes)
                throw Failure.rejected
            }
        ) {
            XCTAssertTrue($0 is Failure)
        }
        XCTAssertEqual(retained.count, 1)
        XCTAssertEqual(retained.first?.count, 16 << 10)
        XCTAssertFalse(decoder.isComplete)
        decoder.close()
        XCTAssertTrue(decoder.isComplete)
        XCTAssertEqual(retained.first, Data(fixture.restored.prefix(16 << 10)))
    }

    func testFragmentedDataSlicesUseTheirOwnCollectionIndices() throws {
        let role = try role()
        let packed = try CBv2NativeCheckpointRowCodec.encode(
            nativeData(sampleRow(width: 64, seed: 31), dtype: role.key.dtype), role: role)
        let envelope = Data(repeating: 0xff, count: 11) + packed + Data([0xff])
        let payload = envelope[11 ..< 11 + packed.count]
        XCTAssertEqual(payload.startIndex, 11)
        let reference = nativeData(
            try PagedKVQuantizationReference.decode(
                Array(payload), headDim: role.width, config: role.key.quantization!, isKey: true),
            dtype: role.key.dtype)
        XCTAssertEqual(try CBv2NativeCheckpointRowCodec.decode(payload, role: role), reference)
        let decoder = CBv2NativeCheckpointRowDecoder()
        defer { decoder.close() }
        var restored: [Data] = []
        try decoder.append(role: role, byteOffset: 0, data: payload.prefix(17)) { _, row in
            restored.append(row)
        }
        XCTAssertTrue(restored.isEmpty)
        try decoder.append(role: role, byteOffset: 17, data: payload.dropFirst(17)) { offset, row in
            XCTAssertEqual(offset, 0)
            restored.append(row)
        }
        XCTAssertEqual(restored, [reference])
        try decoder.append(role: role, byteOffset: role.packedRowBytes, data: payload) {
            offset, row in
            XCTAssertEqual(offset, role.nativeRowBytes)
            restored.append(row)
        }
        XCTAssertEqual(restored, [reference, reference])
        XCTAssertTrue(decoder.isComplete)
    }

    func testNativeRowCodecRejectsNonFloatingDestinationTypesBeforeConversion() throws {
        for dtype: DType in [.uint8, .int32, .int64] {
            let role = try self.role(dtype: dtype)
            XCTAssertThrowsError(
                try CBv2NativeCheckpointRowCodec.encode(
                    Data(count: role.nativeRowBytes), role: role)
            ) {
                XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .invalidSegment)
            }
            XCTAssertThrowsError(
                try CBv2NativeCheckpointRowCodec.decode(
                    Data(count: role.packedRowBytes), role: role)
            ) {
                XCTAssertEqual($0 as? CBv2CompleteCheckpointError, .invalidSegment)
            }
        }
    }

    func testPublicResolverAcceptsNativeContiguousAndSegmentedStorage() {
        let config = profile()
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 64, valueHeadDim: 192, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: .slidingWindow(32), headDim: 192, valueHeadDim: 64, kvHeads: 1,
                queryHeads: 2),
            CBv2LayerKind(attention: .full, headDim: 128, kvHeads: 1, queryHeads: 1),
        ]
        XCTAssertEqual(resolve(config, kinds: kinds, dtypes: dtypes), config)
        XCTAssertEqual(
            resolve(config, kinds: kinds, dtypes: dtypes, paged: nativePool(dtypes: dtypes)), config
        )
        let windows = [
            CBv2LayerKind(attention: .slidingWindow(3), headDim: 64, kvHeads: 1, queryHeads: 1),
            CBv2LayerKind(attention: .slidingWindow(4), headDim: 64, kvHeads: 1, queryHeads: 1),
        ]
        XCTAssertEqual(resolve(config, kinds: windows, dtypes: [.float16, .float32]), config)
    }

    func testPublicResolverRejectsIncompatibleOwnersDTypesAndLiveStorage() {
        let config = profile()
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, valueHeadDim: 64, kvHeads: 1, queryHeads: 1)
        XCTAssertNil(resolve(nil, kinds: [kind], dtypes: [.float16]))
        XCTAssertNil(resolve(config, kinds: [], dtypes: []))
        XCTAssertNil(resolve(config, kinds: [kind], dtypes: []))
        XCTAssertNil(resolve(config, kinds: [kind], dtypes: [.float16, .float32]))
        XCTAssertNil(
            CBv2CompleteCheckpointStorageQuantization.resolve(
                config, layerKinds: [kind], layerDTypes: [.float16], pagedConfig: nil,
                hasAssistantState: true))
        for dtype: DType in [.uint8, .int32] {
            XCTAssertNil(resolve(config, kinds: [kind], dtypes: [dtype]))
        }
        var qsa = kind
        qsa.qwen4IndexerCompressRatio = 4
        XCTAssertNil(resolve(config, kinds: [qsa], dtypes: [.float16]))
        for width in [0, 32, 96, 576] {
            var unsupportedKey = kind
            unsupportedKey.headDim = width
            XCTAssertNil(resolve(config, kinds: [unsupportedKey], dtypes: [.float16]))
            var unsupportedValue = kind
            unsupportedValue.valueHeadDim = width
            XCTAssertNil(resolve(config, kinds: [unsupportedValue], dtypes: [.float16]))
        }
        var invalidGeometry = kind
        invalidGeometry.kvHeads = 0
        XCTAssertNil(resolve(config, kinds: [invalidGeometry], dtypes: [.float16]))
        var smallWindow = kind
        smallWindow.attention = .slidingWindow(3)
        XCTAssertNil(resolve(config, kinds: [smallWindow], dtypes: [.float16]))
        var borrower = kind
        borrower.sharesKVWithLayer = 0
        XCTAssertNil(resolve(config, kinds: [borrower], dtypes: [.float16]))

        var pool = nativePool(dtypes: [.float16])
        pool.quantization = config
        XCTAssertNil(resolve(config, kinds: [kind], dtypes: [.float16], paged: pool))
        pool = nativePool(dtypes: [.float16])
        pool.segmentSizeBytes = nil
        XCTAssertNil(resolve(config, kinds: [kind], dtypes: [.float16], paged: pool))
        pool = nativePool(dtypes: [.float16])
        pool.layerDTypes = nil
        XCTAssertNil(resolve(config, kinds: [kind], dtypes: [.float16], paged: pool))
        pool.layerDTypes = [.bfloat16]
        XCTAssertNil(resolve(config, kinds: [kind], dtypes: [.float16], paged: pool))
        pool = nativePool(dtypes: [.float16])
        pool.nativeLayerIndices = [0]
        XCTAssertNil(resolve(config, kinds: [kind], dtypes: [.float16], paged: pool))
    }

    func testPublicResolverRejectsInvalidProfiles() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)
        let invalidVersion = try JSONDecoder().decode(
            PagedKVQuantizationConfig.self,
            from: Data(
                """
                {"keyBits":4,"valueBits":8,"groupSize":64,"rotationBlockSize":64,"version":1,"recentTokenCount":3}
                """.utf8))
        let invalidProfiles = [
            profile(keyBits: 2), profile(valueBits: 16),
            PagedKVQuantizationConfig(groupSize: 16),
            PagedKVQuantizationConfig(groupSize: 96),
            PagedKVQuantizationConfig(groupSize: 256),
            profile(rotationBlockSize: 16), profile(rotationBlockSize: 96),
            profile(rotationBlockSize: 1024), profile(recentTokenCount: -1),
            profile(recentTokenCount: 4097), invalidVersion,
        ]
        for config in invalidProfiles {
            XCTAssertNil(resolve(config, kinds: [kind], dtypes: [.float16]), config.identity)
        }
    }

    private func profile(
        keyBits: Int = 4, valueBits: Int = 8, rotationBlockSize: Int = 64,
        recentTokenCount: Int = 3
    ) -> PagedKVQuantizationConfig {
        .init(
            keyBits: keyBits, valueBits: valueBits, groupSize: 64,
            rotationBlockSize: rotationBlockSize, recentTokenCount: recentTokenCount)
    }

    private func role(
        config: PagedKVQuantizationConfig? = nil, dtype: DType = .float32,
        width: Int = 64, values: Bool = false, position: Int = 5, tokenStart: Int = 0
    ) throws -> CBv2CheckpointPagedRoleLayout {
        try .init(
            key: .init(kvHeads: 2, headDim: width, dtype: dtype, quantization: config ?? profile()),
            position: position, tokenStart: tokenStart, values: values,
            preservePackedRecent: false)
    }

    /// Rows for the re-encoding check: sampled rows and groups that are
    /// constant, zero, dominated by one outlier, wholly negative, a large
    /// negative minimum beside tiny values, spread over several binades,
    /// near the dtype's largest finite values, or near its smallest normals.
    private func idempotencePatterns(width: Int, dtype: DType) -> [(String, [Float])] {
        let limit: Float = dtype == .float16 ? 60000 : 3.0e38
        let tiny: Float = dtype == .float16 ? 6.2e-5 : 1.2e-38
        let wave = (0 ..< width).map { sin(Float($0) * 0.37) }
        var patterns = (0 ..< 8).map { seed in
            ("sample \(seed)", sampleRow(width: width, seed: seed))
        }
        patterns += [
            ("constant groups", (0 ..< width).map { Float($0 / 64) * 0.75 - 1.5 }),
            ("all zero", [Float](repeating: 0, count: width)),
            ("zero groups", (0 ..< width).map { $0 / 64 % 2 == 0 ? 0 : wave[$0] }),
            ("outliers", (0 ..< width).map { $0 % 64 == 13 ? 300 : 0.01 * wave[$0] }),
            ("negative", (0 ..< width).map { -1000 + wave[$0] }),
            (
                "negative minimum, tiny values",
                (0 ..< width).map { $0 % 64 == 0 ? -1000 : 0.001 * (1.5 + wave[$0]) }
            ),
            (
                "mixed binades",
                (0 ..< width).map { powf(10, Float($0 % 7) - 3) * ($0 % 2 == 0 ? 1 : -1) }
            ),
            (
                "near the limit",
                (0 ..< width).map { ($0 % 2 == 0 ? 1 : -1) * limit * (0.5 + 0.4 * abs(wave[$0])) }
            ),
            (
                "near the limit, positive",
                (0 ..< width).map { limit * (0.9 + 0.09 * abs(wave[$0])) }
            ),
            ("near the smallest normal", (0 ..< width).map { tiny * (1 + 3 * abs(wave[$0])) }),
            ("two values", (0 ..< width).map { $0 % 3 == 0 ? 1 : 1.5 }),
        ]
        return patterns
    }

    private func sampleRow(width: Int, seed: Int) -> [Float] {
        (0 ..< width).map {
            Float(sin(Double($0 * 7 + seed * 13) * 0.037)) * 0.75
                + Float(($0 + seed) % 11 - 5) * 0.0625
        }
    }

    private func nativeData(_ values: [Float], dtype: DType) -> Data {
        let words: [UInt32] = values.map { value in
            switch dtype {
            case .float16: return UInt32(Float16(value).bitPattern)
            case .bfloat16:
                let raw = value.bitPattern
                return (raw &+ 0x7fff &+ ((raw >> 16) & 1)) >> 16
            default: return value.bitPattern
            }
        }
        return littleEndianData(words, wordBytes: dtype.size)
    }

    private func littleEndianData(_ words: [UInt32], wordBytes: Int) -> Data {
        Data(
            words.flatMap { word in
                (0 ..< wordBytes).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
            })
    }

    private func nativeValues(_ data: Data, dtype: DType) -> [Float] {
        stride(from: 0, to: data.count, by: dtype.size).map { offset in
            let raw = (0 ..< dtype.size).reduce(UInt32(0)) {
                $0 | UInt32(data[offset + $1]) << ($1 * 8)
            }
            switch dtype {
            case .float16: return Float(Float16(bitPattern: UInt16(raw)))
            case .bfloat16: return Float(bitPattern: raw << 16)
            default: return Float(bitPattern: raw)
            }
        }
    }

    private func assertNativeRow(
        _ data: Data, reference: [Float], dtype: DType,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(data.count, reference.count * dtype.size, file: file, line: line)
        guard data.count == reference.count * dtype.size else { return }
        let tolerance: Float = dtype == .float32 ? 0.00001 : (dtype == .float16 ? 1 / 512 : 1 / 64)
        for (actual, expected) in zip(nativeValues(data, dtype: dtype), reference) {
            XCTAssertTrue(actual.isFinite, file: file, line: line)
            XCTAssertTrue(expected.isFinite, file: file, line: line)
            XCTAssertEqual(
                actual, expected, accuracy: max(1, abs(expected)) * tolerance,
                file: file, line: line)
        }
    }

    private func replacingFloat(_ data: Data, at offset: Int, with value: Float) -> Data {
        var result = data
        result.replaceSubrange(
            offset ..< offset + 4, with: littleEndianData([value.bitPattern], wordBytes: 4))
        return result
    }

    private func affineRow(
        role: CBv2CheckpointPagedRoleLayout, codes: [UInt8],
        coefficients: [(scale: Float, offset: Float)]
    ) throws -> Data {
        let config = try XCTUnwrap(role.key.quantization)
        let layout = try config.rowLayout(headDim: role.width)
        let bits = role.values ? config.valueBits : config.keyBits
        XCTAssertEqual(coefficients.count, role.width / config.groupSize)
        var bytes = Data(count: role.packedRowBytes)
        for index in 0 ..< role.width {
            let code = codes[index % codes.count] & UInt8((1 << bits) - 1)
            if bits == 4 {
                bytes[index / 2] |= code << ((index & 1) * 4)
            } else {
                bytes[index] = code
            }
        }
        for (group, coefficient) in coefficients.enumerated() {
            bytes = replacingFloat(
                bytes,
                at: (role.values ? layout.valueScaleOffset : layout.keyScaleOffset) + group * 4,
                with: coefficient.scale)
            bytes = replacingFloat(
                bytes,
                at: (role.values ? layout.valueOffsetOffset : layout.keyOffsetOffset) + group * 4,
                with: coefficient.offset)
        }
        return bytes
    }

    private func streamFixture(role: CBv2CheckpointPagedRoleLayout) throws -> (
        bytes: Data, rowEnds: [Int], coefficients: [Range<Int>], restored: Data
    ) {
        let config = try XCTUnwrap(role.key.quantization)
        let layout = try config.rowLayout(headDim: role.width)
        let dataBytes = role.values ? layout.valueDataBytes : layout.keyDataBytes
        var bytes = Data()
        var rowEnds: [Int] = []
        var coefficients: [Range<Int>] = []
        var restored = Data()
        for head in 0 ..< role.key.kvHeads {
            for token in 0 ..< role.position - role.tokenStart {
                let seed = head * 71 + token + role.tokenStart
                if token < role.packedCount {
                    let source = nativeData(
                        sampleRow(width: role.width, seed: seed), dtype: role.key.dtype)
                    let input = nativeValues(source, dtype: role.key.dtype)
                    let encoded = try PagedKVQuantizationReference.encode(
                        input, config: config, isKey: !role.values)
                    coefficients.append(
                        (bytes.count + dataBytes) ..< (bytes.count + encoded.bytes.count))
                    bytes.append(contentsOf: encoded.bytes)
                    let decoded = try PagedKVQuantizationReference.decode(
                        encoded.bytes, headDim: role.width, config: config, isKey: !role.values)
                    restored.append(nativeData(decoded, dtype: role.key.dtype))
                } else {
                    let patterns: [UInt32]
                    switch role.key.dtype {
                    case .float16: patterns = [0, 0x8000, 0x7e01, 0xfe55, 0x7c01, 0xfc03]
                    case .bfloat16: patterns = [0, 0x8000, 0x7fc1, 0xffc5, 0x7f81, 0xffa3]
                    default:
                        patterns = [
                            0, 0x8000_0000, 0x7fc0_1234, 0xffc0_5678, 0x7f80_0001, 0xff80_0123,
                        ]
                    }
                    let exact = littleEndianData(
                        (0 ..< role.width).map { patterns[($0 + seed) % patterns.count] },
                        wordBytes: role.key.dtype.size)
                    bytes.append(exact)
                    restored.append(exact)
                }
                rowEnds.append(bytes.count)
            }
        }
        XCTAssertEqual(bytes.count, role.key.kvHeads * role.bytesPerHead)
        return (bytes, rowEnds, coefficients, restored)
    }

    private func nativePool(dtypes: [DType]) -> PagedKVPoolConfig {
        // Supplying the limit avoids the default initializer's GPU device query.
        .init(
            capacityBytes: 1 << 20, maxBufferLength: 1 << 20,
            segmentSizeBytes: 64 << 10, layerDTypes: dtypes)
    }

    private func resolve(
        _ config: PagedKVQuantizationConfig?, kinds: [CBv2LayerKind], dtypes: [DType],
        paged: PagedKVPoolConfig? = nil
    ) -> PagedKVQuantizationConfig? {
        CBv2CompleteCheckpointStorageQuantization.resolve(
            config, layerKinds: kinds, layerDTypes: dtypes, pagedConfig: paged,
            hasAssistantState: false)
    }
}
