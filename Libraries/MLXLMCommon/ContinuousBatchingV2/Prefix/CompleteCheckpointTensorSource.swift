import Cmlx
import Foundation
import MLX

/// One logical tensor with a bounded read operation. Paged targets never become
/// contiguous whole-prefix views; recurrent/assistant arrays retain their native
/// layout and are packed only over the requested span.
enum CBv2CompleteCheckpointTensorSource {
    case array(MLXArray)
    /// An FP32 array written at a half-precision wire dtype; the export
    /// refuses any element the wire dtype cannot hold exactly.
    case narrowed(MLXArray, CBv2CheckpointDType)
    case paged(CBv2PagedCheckpointTensorSource)
    case historicalWindow(CBv2HistoricalWindowTensorSource)
    case nativeQuantized(CBv2NativeQuantizedCheckpointTensorSource)

    /// The array itself when `wire` is its own dtype, else a narrowing source.
    static func exported(_ array: MLXArray, wire: CBv2CheckpointDType) -> Self {
        CBv2CheckpointHalfPrecision.isHalf(wire) && array.dtype == .float32
            ? .narrowed(array, wire) : .array(array)
    }

    func matches(_ descriptor: CBv2CheckpointTensorDescriptor) -> Bool {
        switch self {
        case .array(let array):
            array.shape == descriptor.shape && array.dtype == descriptor.dtype.mlxDType
        case .narrowed(let array, let wire):
            array.shape == descriptor.shape && array.dtype == .float32
                && descriptor.dtype == wire && CBv2CheckpointHalfPrecision.isHalf(wire)
        case .paged(let source): source.matches(descriptor)
        case .historicalWindow(let source): source.matches(descriptor)
        case .nativeQuantized(let source): source.matches(descriptor)
        }
    }

    func retainForNativeExport(_ work: CBv2NativeCompletePrefixWork) throws {
        switch self {
        case .array(let array), .narrowed(let array, _): try work.retain(arrays: [array])
        case .paged(let source): try source.retainForNativeExport(work)
        case .historicalWindow(let source): try source.retainForNativeExport(work)
        case .nativeQuantized(let source): try source.retainForNativeExport(work)
        }
    }

    func readSegment(
        descriptor: CBv2CheckpointTensorDescriptor, byteOffset: Int, maximumBytes: Int,
        nativeWork: CBv2NativeCompletePrefixWork? = nil
    ) throws -> Data {
        guard maximumBytes > 0, maximumBytes <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        if case .nativeQuantized(let source) = self {
            return try source.readSegment(
                byteOffset: byteOffset, maximumBytes: maximumBytes, nativeWork: nativeWork)
        }
        if case .historicalWindow(let source) = self {
            return try source.readSegment(
                byteOffset: byteOffset, maximumBytes: maximumBytes, nativeWork: nativeWork)
        }
        if case .paged(let source) = self {
            return try source.readSegment(
                byteOffset: byteOffset, maximumBytes: maximumBytes, nativeWork: nativeWork)
        }
        let array: MLXArray
        let wire: CBv2CheckpointDType?
        switch self {
        case .array(let value): (array, wire) = (value, nil)
        case .narrowed(let value, let dtype): (array, wire) = (value, dtype)
        default: throw CBv2CompleteCheckpointError.closed
        }
        let itemSize = descriptor.dtype.mlxDType.size
        guard byteOffset >= 0, byteOffset < descriptor.byteCount, byteOffset % itemSize == 0 else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let count = min(maximumBytes - maximumBytes % itemSize, descriptor.byteCount - byteOffset)
        guard count > 0 else { throw CBv2CompleteCheckpointError.invalidSegment }
        if let nativeWork {
            guard nativeWork.purpose == .publication else {
                throw CBv2NativeShutdownError.unsupportedConsumer
            }
            try nativeWork.captureCurrentStreams()
            try nativeWork.retain(arrays: [array])
            do {
                try withError { fault in
                    eval(array)
                    try fault.check()
                }
            } catch {
                nativeWork.requiredCompletionFailed()
                throw error
            }
        } else {
            try withError { eval(array) }
        }
        guard let pointer = mlx_array_data_uint8(array.ctx),
            let nativeStrides = mlx_array_strides(array.ctx)
        else {
            nativeWork?.requiredCompletionFailed()
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        let strides = descriptor.shape.indices.map { Int(nativeStrides[$0]) }
        var result = Data(count: count)
        if let wire {
            // Walk the FP32 element span and narrow each contiguous run.
            let native = array.dtype.size
            var exact = true
            result.withUnsafeMutableBytes { destination in
                CBv2CheckpointByteLayout.copy(
                    shape: descriptor.shape, strides: strides, itemSize: native,
                    byteOffset: byteOffset / itemSize * native, count: count / itemSize * native
                ) { physicalOffset, packedOffset, length in
                    exact =
                        CBv2CheckpointHalfPrecision.narrow(
                            float32: UnsafeRawPointer(pointer).advanced(by: physicalOffset),
                            count: length / native, to: wire,
                            destination: destination.baseAddress!.advanced(
                                by: packedOffset / native * itemSize)) && exact
                }
            }
            guard exact else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            return result
        }
        result.withUnsafeMutableBytes { destination in
            CBv2CheckpointByteLayout.copy(
                shape: descriptor.shape, strides: strides, itemSize: itemSize,
                byteOffset: byteOffset, count: count
            ) { physicalOffset, packedOffset, length in
                destination.baseAddress!.advanced(by: packedOffset).copyMemory(
                    from: UnsafeRawPointer(pointer).advanced(by: physicalOffset), byteCount: length)
            }
        }
        // This helper allocates provider Data only, not a native packing
        // temporary. The source array remains a whole-operation root. Do not
        // remove donor roots via retireReadbackTemporaries([]) or add a useless
        // per-segment fence; any future actual pack array must use that seam.
        return result
    }

    func close() {
        if case .historicalWindow(let source) = self { source.close() }
        if case .paged(let source) = self { source.close() }
        if case .nativeQuantized(let source) = self { source.close() }
    }
}
