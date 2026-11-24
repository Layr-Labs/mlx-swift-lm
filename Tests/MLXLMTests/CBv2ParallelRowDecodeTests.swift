import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// The streaming checkpoint row decoder with several workers makes the same
/// sink calls, with the same bytes, as one worker, and fails at the same row.
@Suite("CBv2ParallelRowDecode")
struct CBv2ParallelRowDecodeTests {
    private struct Call: Equatable {
        let offset: Int
        let bytes: Data
    }

    private struct Outcome: Equatable {
        var calls: [Call] = []
        var error: String?
        var complete = false
    }

    private func profile(keyBits: Int, valueBits: Int) -> PagedKVQuantizationConfig {
        .init(
            keyBits: keyBits, valueBits: valueBits, groupSize: 64, rotationBlockSize: 64,
            recentTokenCount: 3)
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
        return Data(
            words.flatMap { word in
                (0 ..< dtype.size).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
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

    /// One role's stored bytes (coded rows, then recent native rows, per head)
    /// and the byte range of each coded row.
    private func stored(_ role: CBv2CheckpointPagedRoleLayout) throws -> (Data, [Range<Int>]) {
        let config = try #require(role.key.quantization)
        var bytes = Data()
        var coded: [Range<Int>] = []
        for head in 0 ..< role.key.kvHeads {
            for token in 0 ..< role.position - role.tokenStart {
                let row = nativeData(
                    sampleRow(width: role.width, seed: head * 71 + token + role.tokenStart),
                    dtype: role.key.dtype)
                if token < role.packedCount {
                    let encoded = try PagedKVQuantizationReference.encode(
                        nativeValues(row, dtype: role.key.dtype), config: config,
                        isKey: !role.values)
                    coded.append(bytes.count ..< bytes.count + encoded.bytes.count)
                    bytes.append(contentsOf: encoded.bytes)
                } else {
                    bytes.append(row)
                }
            }
        }
        #expect(bytes.count == role.key.kvHeads * role.bytesPerHead)
        return (bytes, coded)
    }

    /// Streams `bytes` in `fragment`-byte calls; `failAt` makes the sink throw on that call.
    private func run(
        _ decoder: CBv2NativeCheckpointRowDecoder, role: CBv2CheckpointPagedRoleLayout,
        bytes: Data, fragment: Int, failAt: Int? = nil
    ) -> Outcome {
        struct SinkFailure: Error {}
        var outcome = Outcome()
        var offset = 0
        do {
            while offset < bytes.count {
                let end = min(offset + fragment, bytes.count)
                try decoder.append(
                    role: role, byteOffset: offset, data: bytes.subdata(in: offset ..< end)
                ) { nativeOffset, data in
                    if outcome.calls.count == failAt { throw SinkFailure() }
                    outcome.calls.append(.init(offset: nativeOffset, bytes: data))
                }
                offset = end
            }
        } catch {
            outcome.error = "\(error)"
        }
        outcome.complete = decoder.isComplete
        decoder.close()
        return outcome
    }

    private func roles() throws -> [CBv2CheckpointPagedRoleLayout] {
        var roles: [CBv2CheckpointPagedRoleLayout] = []
        for (keyBits, valueBits) in [(4, 8), (8, 4)] {
            for dtype in [DType.float16, .bfloat16, .float32] {
                for values in [false, true] {
                    roles.append(
                        try .init(
                            key: PagedKVGroupKey(
                                kvHeads: 2, headDim: 192, dtype: dtype, valueHeadDim: 64,
                                quantization: profile(keyBits: keyBits, valueBits: valueBits)),
                            position: 300, tokenStart: 7, values: values,
                            preservePackedRecent: false))
                }
            }
        }
        return roles
    }

    private func decoders() -> [CBv2NativeCheckpointRowDecoder] {
        [
            .init(workers: 4, waveRows: 7, minimumRowsPerWorker: 2),
            .init(workers: 3, waveRows: 1024, minimumRowsPerWorker: 1),
            .init(),
        ]
    }

    @Test("several workers write what one worker writes, for every fragment size")
    func sameCalls() throws {
        for (index, role) in try roles().enumerated() {
            let (bytes, _) = try stored(role)
            // One-byte calls never form a whole row; one role is enough for them.
            for fragment in (index == 0 ? [1] : []) + [257, 4096, bytes.count] {
                let expected = run(.init(workers: 1), role: role, bytes: bytes, fragment: fragment)
                #expect(expected.error == nil && expected.complete && !expected.calls.isEmpty)
                for decoder in decoders() {
                    #expect(
                        run(decoder, role: role, bytes: bytes, fragment: fragment) == expected,
                        "\(role.key.dtype) values=\(role.values) fragment=\(fragment)")
                }
            }
        }
    }

    @Test("a row that does not decode fails at the same row with the same earlier writes")
    func sameFailure() throws {
        for role in try roles() {
            let (clean, coded) = try stored(role)
            let config = try #require(role.key.quantization)
            let layout = try config.rowLayout(headDim: role.width)
            let scale = role.values ? layout.valueScaleOffset : layout.keyScaleOffset
            for row in [0, coded.count / 2, coded.count - 1] {
                var bytes = clean
                // A NaN scale: the decoder refuses this row.
                for index in 0 ..< 4 { bytes[coded[row].lowerBound + scale + index] = 0xff }
                for fragment in [257, bytes.count] {
                    let expected = run(
                        .init(workers: 1), role: role, bytes: bytes, fragment: fragment)
                    #expect(expected.error != nil && !expected.complete)
                    for decoder in decoders() {
                        #expect(
                            run(decoder, role: role, bytes: bytes, fragment: fragment) == expected)
                    }
                }
            }
        }
    }

    @Test("a failing sink stops both at the same call")
    func sameSinkFailure() throws {
        let role = try roles()[0]
        let (bytes, _) = try stored(role)
        for failAt in [0, 1, 3] {
            for fragment in [257, bytes.count] {
                let expected = run(
                    .init(workers: 1), role: role, bytes: bytes, fragment: fragment, failAt: failAt)
                #expect(expected.error != nil && expected.calls.count == failAt)
                for decoder in decoders() {
                    #expect(
                        run(decoder, role: role, bytes: bytes, fragment: fragment, failAt: failAt)
                            == expected)
                }
            }
        }
    }

    /// Streams `bytes` with row targets into one buffer of the role's native
    /// rows; rows the targets do not take arrive through the sink, which
    /// writes into the same buffer. Returns the buffer, the error and the
    /// native ranges written.
    private func runDirect(
        _ decoder: CBv2NativeCheckpointRowDecoder, role: CBv2CheckpointPagedRoleLayout,
        bytes: Data, fragment: Int, refuse: Set<Int> = []
    ) -> (buffer: Data, error: String?, written: [Range<Int>], targeted: Int) {
        let total = role.key.kvHeads * (role.position - role.tokenStart) * role.nativeRowBytes
        let base = UnsafeMutableRawPointer.allocate(byteCount: total, alignment: 16)
        defer { base.deallocate() }
        base.initializeMemory(as: UInt8.self, repeating: 0, count: total)
        var written: [Range<Int>] = []
        var targeted = 0
        var error: String?
        var offset = 0
        do {
            while offset < bytes.count {
                let end = min(offset + fragment, bytes.count)
                try decoder.append(
                    role: role, byteOffset: offset, data: bytes.subdata(in: offset ..< end),
                    targets: { nativeOffset, rows in
                        // A refused run (by its first row) falls back to the sink.
                        guard !refuse.contains(nativeOffset / role.nativeRowBytes) else {
                            return nil
                        }
                        targeted += rows
                        return (0 ..< rows).map {
                            base.advanced(by: nativeOffset + $0 * role.nativeRowBytes)
                        }
                    }
                ) { nativeOffset, data in
                    data.withUnsafeBytes {
                        base.advanced(by: nativeOffset).copyMemory(
                            from: $0.baseAddress!, byteCount: $0.count)
                    }
                    written.append(nativeOffset ..< nativeOffset + data.count)
                }
                offset = end
            }
        } catch let failure {
            error = "\(failure)"
        }
        decoder.close()
        return (Data(bytes: base, count: total), error, written, targeted)
    }

    private func reassemble(_ outcome: Outcome, total: Int) -> (Data, [Range<Int>]) {
        var buffer = Data(count: total)
        for call in outcome.calls {
            buffer.replaceSubrange(call.offset ..< call.offset + call.bytes.count, with: call.bytes)
        }
        return (buffer, outcome.calls.map { $0.offset ..< $0.offset + $0.bytes.count })
    }

    @Test("rows decoded straight into their targets are the bytes one worker writes")
    func directTargets() throws {
        for role in try roles() {
            let (clean, coded) = try stored(role)
            let total = role.key.kvHeads * (role.position - role.tokenStart) * role.nativeRowBytes
            let config = try #require(role.key.quantization)
            let layout = try config.rowLayout(headDim: role.width)
            let scale = role.values ? layout.valueScaleOffset : layout.keyScaleOffset
            var broken = clean
            for index in 0 ..< 4 {
                broken[coded[coded.count / 2].lowerBound + scale + index] = 0xff
            }
            for (bytes, failing) in [(clean, false), (broken, true)] {
                for fragment in [257, 4096, bytes.count] {
                    let reference = run(
                        .init(workers: 1), role: role, bytes: bytes, fragment: fragment)
                    let (expected, _) = reassemble(reference, total: total)
                    for decoder in decoders() {
                        for refuse in [Set<Int>(), [0, 40, 300]] {
                            let direct = runDirect(
                                decoder, role: role, bytes: bytes, fragment: fragment,
                                refuse: refuse)
                            #expect(direct.error == reference.error, "fragment=\(fragment)")
                            // The same bytes one worker leaves, also when a row is refused.
                            #expect(
                                direct.buffer == expected,
                                "\(role.key.dtype) values=\(role.values) fragment=\(fragment) failing=\(failing)"
                            )
                        }
                    }
                    if !failing && fragment == bytes.count {
                        // Several workers took the direct path for every row.
                        let parallel = runDirect(
                            .init(workers: 4, waveRows: 7, minimumRowsPerWorker: 2), role: role,
                            bytes: bytes, fragment: fragment)
                        #expect(
                            parallel.targeted
                                == role.key.kvHeads * (role.position - role.tokenStart))
                    }
                }
            }
        }
    }

    @Test("one decoder reused across growing roles and after close writes what one worker writes")
    func reuse() throws {
        let roles = try roles()
        let sequence = [roles[1], roles[0], roles[1], roles[5], roles[4]]
        let reused = CBv2NativeCheckpointRowDecoder(
            workers: 4, waveRows: 7, minimumRowsPerWorker: 2)
        for role in sequence {
            let (bytes, _) = try stored(role)
            let expected = run(.init(workers: 1), role: role, bytes: bytes, fragment: 4096)
            // `run` closes the decoder after each role.
            #expect(run(reused, role: role, bytes: bytes, fragment: 4096) == expected)
        }
    }

    @Test("the concurrent wave is charged as import scratch")
    func scratch() {
        #expect(CBv2NativeCheckpointRowDecoder.scratchBytes(workers: 1) == 0)
        // The widest row (512 FP32 values) fills the whole charged arena.
        #expect(
            CBv2NativeCheckpointRowDecoder.scratchBytes(workers: 8)
                >= CBv2NativeCheckpointRowDecoder.defaultWaveRows * 512 * 4 + 8 * 512 * 4)
    }
}

/// `CBv2PagedCheckpointStorage.rowTargets` addresses exactly the bytes
/// `append` writes for the same rows.
@Suite("CBv2PagedCheckpointRowTargets", .serialized)
struct CBv2PagedCheckpointRowTargetsTests {
    private func kind(key: Int = 64, value: Int = 64) -> CBv2LayerKind {
        .init(attention: .full, headDim: key, valueHeadDim: value, kvHeads: 2, queryHeads: 4)
    }

    private func config(types: [DType], kind: CBv2LayerKind) -> PagedKVPoolConfig {
        let largestPage =
            kind.kvHeads * 16 * (kind.headDim + kind.valueHeadDim) * types.map(\.size).max()!
        return .init(
            capacityBytes: 16 << 20, maxPrefillChunk: 64, maxBufferLength: 16 << 20,
            segmentSizeBytes: largestPage * 3, layerDTypes: types)
    }

    private func bytes(count: Int, salt: Int) -> Data {
        Data((0 ..< count).map { UInt8(truncatingIfNeeded: ($0 * 73) ^ ($0 >> 3) ^ salt) })
    }

    @Test(
        "targets and append fill the same pages, across head, page and segment seams",
        arguments: [DType.bfloat16, .float16, .float32])
    func sameBytes(dtype: DType) throws {
        try fill(kind(), dtype: dtype)
    }

    /// The value role sits at the segment's value offset, after the key
    /// pages of the key width: a layer whose value width differs from its
    /// key width pins that offset and both row widths.
    @Test(
        "targets and append fill the same pages when the value width differs from the key width",
        arguments: [DType.bfloat16, .float32], [(key: 128, value: 64), (key: 64, value: 128)])
    func asymmetricSameBytes(dtype: DType, widths: (key: Int, value: Int)) throws {
        try fill(kind(key: widths.key, value: widths.value), dtype: dtype)
    }

    private func fill(_ kind: CBv2LayerKind, dtype: DType) throws {
        let admission = AdmissionV2(layerKinds: [kind, kind], bytesCapacity: 16 << 20)
        let plan = try CBv2PagedCheckpointStoragePlan(
            layerKinds: [kind, kind], config: config(types: [dtype, dtype], kind: kind),
            position: 65)
        let appended = try CBv2PagedCheckpointStorage(plan: plan) { array in
            try withError { eval(array) }
        }
        // Staged as an import stages it: host pages (zero-fill-on-demand
        // mappings where page-granular, copied host zeros otherwise).
        let targeted = try CBv2PagedCheckpointStorage(
            plan: plan, evaluate: { array in try withError { eval(array) } },
            hostMaterialized: true, hostZeroed: true)
        defer {
            appended.close()
            targeted.close()
        }
        let rows = 2 * 65
        for layer in plan.layers.indices {
            for values in [false, true] {
                let rowBytes = (values ? kind.valueHeadDim : kind.headDim) * dtype.size
                let expected = bytes(count: rows * rowBytes, salt: layer * 19 + (values ? 7 : 3))
                try appended.append(
                    layerIndex: layer, values: values, byteOffset: 0, data: expected)
                var row = 0
                var run = 0
                let lengths = [1, 7, 33, 16, 64]
                while row < rows {
                    let count = min(lengths[run % lengths.count], rows - row)
                    let pointers = try #require(
                        targeted.rowTargets(
                            layerIndex: layer, values: values, byteOffset: row * rowBytes,
                            rows: count,
                            rowBytes: rowBytes))
                    #expect(pointers.count == count)
                    expected.withUnsafeBytes { source in
                        for (index, pointer) in pointers.enumerated() {
                            pointer.copyMemory(
                                from: source.baseAddress!.advanced(by: (row + index) * rowBytes),
                                byteCount: rowBytes)
                        }
                    }
                    row += count
                    run += 1
                }
                for storage in [appended, targeted] {
                    let source = try CBv2PagedCheckpointTensorSource(
                        storage: storage, layerIndex: layer, values: values, admission: admission)
                    #expect(
                        try source.readSegment(byteOffset: 0, maximumBytes: expected.count)
                            == expected)
                    source.close()
                }
                // Not whole rows, past the end, a widened wire, or a bad layer: no targets.
                #expect(
                    targeted.rowTargets(
                        layerIndex: layer, values: values, byteOffset: 1, rows: 1,
                        rowBytes: rowBytes) == nil)
                #expect(
                    targeted.rowTargets(
                        layerIndex: layer, values: values, byteOffset: 0, rows: rows + 1,
                        rowBytes: rowBytes) == nil)
                #expect(
                    targeted.rowTargets(
                        layerIndex: layer, values: values, byteOffset: 0, rows: 1,
                        rowBytes: rowBytes + 1) == nil)
                #expect(
                    targeted.rowTargets(
                        layerIndex: 9, values: values, byteOffset: 0, rows: 1, rowBytes: rowBytes)
                        == nil)
                if dtype == .float32 {
                    #expect(
                        targeted.rowTargets(
                            layerIndex: layer, values: values, byteOffset: 0, rows: 1,
                            rowBytes: rowBytes, wire: .float16) == nil)
                }
            }
        }
    }
}
