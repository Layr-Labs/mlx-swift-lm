import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    /// Half-precision checkpoint wire encoding through the paged and array
    /// export sources and the FP32 import sinks they feed.
    @Suite("Half-precision checkpoint transfer", .serialized)
    struct CheckpointHalfPrecisionTransferTests {
        private var kind: CBv2LayerKind {
            .init(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        }

        private func backend(_ dtype: DType = .float32) throws -> PagedKVBackend {
            try PagedKVBackend(
                layerKinds: [kind],
                config: .init(
                    capacityBytes: 8 << 20, maxPrefillChunk: 64, segmentSizeBytes: 32 << 10,
                    layerDTypes: [dtype]))
        }

        private func boundAdmission(
            _ backend: PagedKVBackend, elementBytes: Int = 4
        ) -> AdmissionV2 {
            let admission = AdmissionV2(
                layerKinds: [kind], bytesCapacity: 8 << 20,
                config: .init(watermarkFraction: 0, elementBytes: elementBytes),
                residency: backend.kvResidency)
            backend.pool.bindAdmission(admission)
            return admission
        }

        private func storage(_ backend: PagedKVBackend, position: Int) throws
            -> CBv2PagedCheckpointStorage
        {
            let plan = try CBv2PagedCheckpointStoragePlan(
                layerKinds: [kind], config: backend.pool.config, position: position)
            return try CBv2PagedCheckpointStorage(
                plan: plan, evaluate: { array in try withError { eval(array) } })
        }

        /// Writes page contents through the native (wire nil) import path:
        /// `value` gives each element's bits, truncated to `itemSize` bytes.
        private func fill(
            _ storage: CBv2PagedCheckpointStorage, position: Int, itemSize: Int = 4,
            value: (Int) -> UInt32
        ) throws {
            for values in [false, true] {
                let elements = kind.kvHeads * position * kind.headDim
                let bytes = elements * itemSize
                var data = Data(count: bytes)
                data.withUnsafeMutableBytes { raw in
                    for index in 0 ..< elements {
                        let bits = value(index + (values ? 1 << 20 : 0))
                        if itemSize == 2 {
                            raw.storeBytes(
                                of: UInt16(truncatingIfNeeded: bits), toByteOffset: index * 2,
                                as: UInt16.self)
                        } else {
                            raw.storeBytes(of: bits, toByteOffset: index * 4, as: UInt32.self)
                        }
                    }
                }
                var offset = 0
                while offset < bytes {
                    let count = min(4 << 20, bytes - offset)
                    try storage.append(
                        layerIndex: 0, values: values, byteOffset: offset,
                        data: data.subdata(in: offset ..< offset + count))
                    offset += count
                }
            }
        }

        private func read(
            _ storage: CBv2PagedCheckpointStorage, admission: AdmissionV2, values: Bool,
            wire: CBv2CheckpointDType?, segment: Int
        ) throws -> Data {
            let source = try CBv2PagedCheckpointTensorSource(
                storage: storage, layerIndex: 0, values: values, admission: admission, wire: wire)
            defer { source.close() }
            var result = Data()
            while result.count < source.byteCount {
                result.append(
                    try source.readSegment(byteOffset: result.count, maximumBytes: segment))
            }
            return result
        }

        /// Appends `data` to `target` in `chunk`-byte segments.
        private func append(
            _ data: Data, to target: CBv2PagedCheckpointStorage, values: Bool,
            wire: CBv2CheckpointDType, chunk: Int
        ) throws {
            var offset = 0
            while offset < data.count {
                let count = min(chunk, data.count - offset)
                try target.append(
                    layerIndex: 0, values: values, byteOffset: offset,
                    data: data.subdata(in: offset ..< offset + count), wire: wire)
                offset += count
            }
        }

        @Test func pagedBFloat16WireRestoresFP32PagesExactly() throws {
            let backend = try backend()
            let admission = boundAdmission(backend)
            let position = 37
            let source = try storage(backend, position: position)
            try fill(source, position: position) { UInt32(truncatingIfNeeded: $0 &* 40503) << 16 }
            let target = try storage(backend, position: position)
            for values in [false, true] {
                // Odd segment sizes split runs mid-feature and mid-page.
                let wire = try read(
                    source, admission: admission, values: values, wire: .bfloat16, segment: 46)
                let native = try read(
                    source, admission: admission, values: values, wire: nil, segment: 4 << 20)
                #expect(wire.count * 2 == native.count)
                try append(wire, to: target, values: values, wire: .bfloat16, chunk: 30)
                let restored = try read(
                    target, admission: admission, values: values, wire: nil, segment: 4 << 20)
                #expect(restored == native)
            }
            source.close()
            target.close()
        }

        /// A wire dtype equal to the page dtype is the native layout: the
        /// export reads the native bytes (no narrowing, full byte count) and
        /// the import copies them, for half-precision pages as well as FP32.
        @Test func wireEqualToThePageDTypeIsTheNativeLayout() throws {
            for dtype in [DType.float32, .bfloat16] {
                let backend = try backend(dtype)
                let admission = boundAdmission(backend, elementBytes: dtype.size)
                let wire = try #require(CBv2CheckpointDType(dtype))
                let position = 37
                let source = try storage(backend, position: position)
                try fill(source, position: position, itemSize: dtype.size) {
                    UInt32(truncatingIfNeeded: $0 &* 2_654_435_761)
                }
                let target = try storage(backend, position: position)
                for values in [false, true] {
                    let native = try read(
                        source, admission: admission, values: values, wire: nil, segment: 4 << 20)
                    let same = try read(
                        source, admission: admission, values: values, wire: wire, segment: 46)
                    #expect(same == native)
                    try append(same, to: target, values: values, wire: wire, chunk: 28)
                    let restored = try read(
                        target, admission: admission, values: values, wire: nil, segment: 4 << 20)
                    #expect(restored == native)
                }
                source.close()
                target.close()
            }
        }

        @Test func pagedNarrowingRefusesAnInexactValue() throws {
            let backend = try backend()
            let admission = boundAdmission(backend)
            let source = try storage(backend, position: 20)
            try fill(source, position: 20) { index in
                index == 1_000 ? Float(0.1).bitPattern : UInt32(index % 512) << 16
            }
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                _ = try read(
                    source, admission: admission, values: false, wire: .bfloat16, segment: 64)
            }
            // The values role never saw the inexact element and still narrows.
            _ = try read(source, admission: admission, values: true, wire: .bfloat16, segment: 64)
            source.close()
        }

        @Test func arrayNarrowingAndWideningCopyRestoreFP32() throws {
            let shape = [1, 6, 4]
            let exact = (0 ..< 24).map { Float(bitPattern: UInt32($0 * 977) << 16) }
            let source = CBv2CompleteCheckpointTensorSource.exported(
                MLXArray(exact, shape), wire: .bfloat16)
            let descriptor = try CBv2CheckpointTensorDescriptor(
                role: .convolution, layer: 0, shape: shape, dtype: .bfloat16)
            #expect(source.matches(descriptor))
            let destination = MLXArray.zeros(shape, dtype: .float32)
            eval(destination)
            var offset = 0
            while offset < descriptor.byteCount {
                let data = try source.readSegment(
                    descriptor: descriptor, byteOffset: offset, maximumBytes: 10)
                let pointer = try #require(mlx_array_data_uint8(destination.ctx))
                try data.withUnsafeBytes { raw in
                    try CBv2CheckpointByteLayout.copy(
                        shape: shape, strides: CBv2CheckpointByteLayout.contiguousStrides(shape),
                        wire: .bfloat16, destinationDType: .float32, byteOffset: offset,
                        source: raw, destination: UnsafeMutableRawPointer(mutating: pointer))
                }
                offset += data.count
            }
            #expect(destination.asArray(Float.self).map(\.bitPattern) == exact.map(\.bitPattern))
            let inexact = CBv2CompleteCheckpointTensorSource.exported(
                MLXArray((0 ..< 24).map { Float($0) / 3 }, shape), wire: .bfloat16)
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                _ = try inexact.readSegment(
                    descriptor: descriptor, byteOffset: 0, maximumBytes: 48)
            }
        }
    }
}
