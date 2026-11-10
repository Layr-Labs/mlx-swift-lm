import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// Checkpoint export codes a batch of native rows on several cores. The
    /// bytes and the refusal are those of the reference codec row by row.
    /// Swift logic only; no arrays are created.
    @Suite("Checkpoint export codes row batches as row by row")
    struct NativeCheckpointBatchCodingTests {
        private func role(dtype: DType, width: Int, values: Bool) throws
            -> CBv2CheckpointPagedRoleLayout
        {
            try CBv2CheckpointPagedRoleLayout(
                key: PagedKVGroupKey(
                    kvHeads: 1, headDim: width, dtype: dtype, valueHeadDim: width,
                    quantization: PagedKVQuantizationSelection.balanced.configuration),
                position: 300, values: values, preservePackedRecent: false)
        }

        /// Native little-endian rows of finite values, signed zeros included.
        private func rows(_ count: Int, role: CBv2CheckpointPagedRoleLayout) -> [Data] {
            var state: UInt64 = 0x9E37_79B9_7F4A_7C15
            return (0 ..< count).map { row in
                var data = Data(count: role.nativeRowBytes)
                data.withUnsafeMutableBytes { bytes in
                    for index in 0 ..< role.width {
                        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                        var value =
                            Float(Int(state >> 40) % 4001 - 2000) / 1000 * Float(1 + row % 5)
                        if index % 37 == 0 { value = row % 2 == 0 ? 0 : -0.0 }
                        switch role.key.dtype {
                        case .float16:
                            bytes.storeBytes(
                                of: Float16(value).bitPattern.littleEndian,
                                toByteOffset: index * 2, as: UInt16.self)
                        case .bfloat16:
                            bytes.storeBytes(
                                of: UInt16(value.bitPattern >> 16).littleEndian,
                                toByteOffset: index * 2, as: UInt16.self)
                        default:
                            bytes.storeBytes(
                                of: value.bitPattern.littleEndian, toByteOffset: index * 4,
                                as: UInt32.self)
                        }
                    }
                }
                return data
            }
        }

        @Test(arguments: [1, 2, 4, 7])
        func everyWorkerCountWritesTheReferenceBytes(workers: Int) throws {
            for dtype: DType in [.bfloat16, .float16, .float32] {
                for width in [64, 256, 512] {
                    for values in [false, true] {
                        let role = try role(dtype: dtype, width: width, values: values)
                        let native = rows(101, role: role)
                        var coded = Data(count: native.count * role.packedRowBytes)
                        try CBv2NativeCheckpointBatchCoding(role: role, workers: workers)
                            .code(native.reduce(Data(), +), rows: native.count, into: &coded)
                        let reference = try native.map {
                            try CBv2NativeCheckpointRowCodec.encode($0, role: role)
                        }.reduce(Data(), +)
                        #expect(coded == reference, "\(dtype) width \(width) values \(values)")
                    }
                }
            }
        }

        @Test(arguments: [1, 4])
        func aRefusedRowRefusesTheBatch(workers: Int) throws {
            let role = try role(dtype: .float32, width: 128, values: false)
            var native = rows(64, role: role)
            native[40].withUnsafeMutableBytes {
                $0.storeBytes(of: Float.nan.bitPattern, toByteOffset: 8, as: UInt32.self)
            }
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try CBv2NativeCheckpointRowCodec.encode(native[40], role: role)
            }
            var coded = Data(count: native.count * role.packedRowBytes)
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try CBv2NativeCheckpointBatchCoding(role: role, workers: workers)
                    .code(native.reduce(Data(), +), rows: native.count, into: &coded)
            }
        }
    }
}
