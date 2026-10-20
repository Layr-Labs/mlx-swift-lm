import Foundation
import MLX

extension MLXArray {
    /// A zero-filled array that is available on creation: MLX copies host
    /// bytes into a fresh allocator buffer, so it takes no MLX eval lock,
    /// schedules no stream work and needs no completion fence. The buffer is
    /// allocated here, so an allocation fault throws here rather than at eval.
    ///
    /// The calloc-backed source is read by the copy and freed on return. It can
    /// briefly commit up to the array's size in host memory outside MLX's
    /// allocator; callers create destinations one at a time and charge the
    /// largest as staging scratch.
    static func cbv2HostZeros(_ shape: [Int], dtype: DType) throws -> MLXArray {
        let byteCount = shape.reduce(dtype.size, *)
        // Empty `Data` has no base address to copy from.
        guard byteCount > 0 else { return MLXArray.zeros(shape, dtype: dtype) }
        return try withError { MLXArray(Data(count: byteCount), shape, dtype: dtype) }
    }
}
