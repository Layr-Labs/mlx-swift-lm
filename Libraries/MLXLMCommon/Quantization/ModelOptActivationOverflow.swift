/// What a ModelOpt activation quantizer does with a finite activation beyond
/// the E4M3 range of its calibrated input scale. Fixed per projection at load.
///
/// - ``eagerNaN`` reproduces ModelOpt's PyTorch eager cast (`_fp8_eager`, no
///   SATFINITE): an FP8 static per-tensor activation with |x * multiplier| > 464
///   (the E4M3 round-to-nearest-even overflow boundary), or an NVFP4 row whose
///   amax / (6 * inputScale) > 464, becomes NaN.
/// - ``saturating`` follows deployment conversion (TensorRT-LLM and vLLM
///   convert with SATFINITE): FP8 activations clamp to ±448, NVFP4 block scales
///   saturate at 448 and E2M1 values at 6, exactly as the shared E4M3/E2M1
///   encoders (and MLX's native `to_fp8` and NVFP4 product) already round.
///   Infinities saturate like any other overflow.
///
/// Both keep NaN inputs NaN, and both give bit-identical results for every
/// activation the eager rule leaves finite; they differ only on activations
/// the eager rule turns into NaN.
public enum ModelOptActivationOverflow: Sendable, Hashable {
    case eagerNaN
    case saturating

    /// Nemotron Super VL (language model, MTP head and its assistant). Its image
    /// embeddings can exceed the text-calibrated input scales; under the eager
    /// rule such requests produce NaN logits. A reversible numerics decision,
    /// recorded in `docs/nemotron-super.md`.
    public static let nemotronSuper = Self.saturating

    /// FP8 activations with |x * multiplier| above this bound become NaN.
    var fp8NaNBound: Float {
        switch self {
        case .eagerNaN: 464
        case .saturating: .infinity
        }
    }

    /// The NVFP4 row kernels mark a row NaN when `amax / divisor > 464`. A
    /// divisor of +infinity never satisfies that (`amax / inf` is 0, or NaN for
    /// an infinite amax), so only NaN rows stay NaN.
    func nvfp4OverflowDivisor(inputScale: Float) -> Float {
        switch self {
        case .eagerNaN: 6 * inputScale
        case .saturating: .infinity
        }
    }
}
