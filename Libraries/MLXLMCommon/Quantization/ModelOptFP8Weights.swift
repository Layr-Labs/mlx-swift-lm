import Cmlx
import MLX

/// Bounded decoder for ModelOpt per-tensor E4M3FN weights, not MXFP8.
/// Preserves the exported bytes and direct multiplier. The calibratedMatmul
/// extension is an eager-reference candidate, not a complete model loader.
public struct ModelOptFP8Weights {
    public enum ValidationError: Error {
        case invalidLayout, invalidScale, invalidRows, invalidDType, workspaceLimit, nativeConversion
        case invalidInputShape, invalidInputScale
    }
    public let bytes: MLXArray
    public let tensorScale: Float

    public init(e4m3Bytes: MLXArray, tensorScale: Float) throws {
        guard e4m3Bytes.dtype == .uint8, e4m3Bytes.ndim == 2,
            e4m3Bytes.dim(0) > 0, e4m3Bytes.dim(1) > 0 else {
            throw ValidationError.invalidLayout
        }
        guard tensorScale.isFinite, tensorScale > 0 else { throw ValidationError.invalidScale }
        self.bytes = e4m3Bytes
        self.tensorScale = tensorScale
    }

    /// Decode an explicit row slice using native MLX E4M3 conversion. Caps FP32
    /// temporary elements at 64 MiB; callers must budget concurrent operations.
    /// Scale multiplication occurs in FP32 before the requested output cast.
    public func decodedRows(_ rows: Range<Int>, dtype: DType = .bfloat16) throws -> MLXArray {
        guard !rows.isEmpty, rows.lowerBound >= 0, rows.upperBound <= bytes.dim(0) else {
            throw ValidationError.invalidRows
        }
        guard [.float16, .bfloat16, .float32].contains(dtype) else {
            throw ValidationError.invalidDType
        }
        let (count, overflow) = rows.count.multipliedReportingOverflow(by: bytes.dim(1))
        guard !overflow, count <= (64 << 20) / MemoryLayout<Float>.size else {
            throw ValidationError.workspaceLimit
        }
        let slice = bytes[rows, 0...]
        var output = mlx_array_new()
        let status = mlx_from_fp8(&output, slice.ctx, DType.float32.cmlxDtype, StreamOrDevice.default.ctx)
        guard status == 0 else {
            mlx_array_free(output)
            throw ValidationError.nativeConversion
        }
        // Native conversion decodes 0x7f/0xff as +/-480; E4M3FN reserves
        // those bit patterns for NaN. Preserve the checkpoint format explicitly.
        let invalid = logicalOr(slice .== 127, slice .== 255)
        let decoded = which(invalid, MLXArray(Float.nan), MLXArray(output))
        return (decoded * tensorScale).asType(dtype)
    }
}
