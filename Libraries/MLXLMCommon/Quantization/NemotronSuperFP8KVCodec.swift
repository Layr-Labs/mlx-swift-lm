import Cmlx
import MLX

/// Isolated unit-scale E4M3 cache codec, not a cache owner or attention backend.
/// CUDA12.8.90 cuda_fp8.hpp SHA256:
/// e0bebde335eedb623df836803fbef39be81940ba4e760b2a916686336ff6d17d
/// Constructors1556–1575 select SATFINITE;245–287 specify RN/SATFINITE and
/// canonical NaN;442–448 promotes BF16 exactly before conversion. Decode497–499
/// recognizes both NaN encodings. Independently implemented using native MLX.
/// Matches the unit-scale policy in TensorRT-LLM26927494 trtllm.py1780–1793.
/// This does not establish attention accumulation or speculative-decoding parity.
public enum NemotronSuperFP8KVCodec {
    public enum Failure: Error { case dtype, shape, workspace, conversion }
    public static let maximumElements = 1 << 20

    private static func validate(_ array: MLXArray) throws {
        guard array.ndim > 0, array.shape.allSatisfy({ $0 > 0 }) else { throw Failure.shape }
        //32bytes/element bounds explicit conversion/mask temporaries, not
        //backend scratch or simultaneous lazy invocations; caller owns admission.
        guard array.size <= maximumElements else { throw Failure.workspace }
    }

    /// RN-even, finite saturation (including infinities), signed zero retained.
    /// NaN output is canonicalized to0x7f; no promise about CUDA NaN sign/payload
    /// bits on every GPU. Unlike ModelOpt eager casting, overflow is not NaN.
    public static func encode(_ input: MLXArray) throws -> MLXArray {
        guard [.float16, .bfloat16, .float32].contains(input.dtype) else { throw Failure.dtype }
        try validate(input)
        let promoted = input.asType(.float32)
        var handle = mlx_array_new()
        let status = mlx_to_fp8(&handle, promoted.ctx, StreamOrDevice.default.ctx)
        guard status == 0 else { mlx_array_free(handle); throw Failure.conversion }
        let bytes = MLXArray(handle)
        //MLX saturates NaNs to finite max. Restore the E4M3 NaN class explicitly.
        return which(isNaN(promoted), MLXArray(UInt8(127)), bytes)
    }

    public static func decode(_ bytes: MLXArray, dtype: DType = .bfloat16) throws -> MLXArray {
        guard bytes.dtype == .uint8, [.float16, .bfloat16, .float32].contains(dtype) else { throw Failure.dtype }
        try validate(bytes)
        var handle = mlx_array_new()
        let status = mlx_from_fp8(&handle, bytes.ctx, dtype.cmlxDtype, StreamOrDevice.default.ctx)
        guard status == 0 else { mlx_array_free(handle); throw Failure.conversion }
        let decoded = MLXArray(handle)
        //MLX interprets reserved E4M3 codes as±480. CUDA treats them asNaN.
        let invalid = logicalOr(bytes .== 127, bytes .== 255)
        return which(invalid, MLXArray(Float.nan).asType(dtype), decoded)
    }
}
