import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The fused one-token Super Mamba kernels and the block-boundary residual
/// norm against the op chains they replace, bit for bit, including NaN and
/// infinite inputs. Synthetic values only.
@Suite(.tags(.integration))
struct NemotronSuperMambaStepTests {
    static func random(_ shape: [Int], seed: UInt64, scale: Float = 1, dtype: DType = .bfloat16)
        -> MLXArray
    {
        (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(dtype)
    }

    /// Replaces the first few elements with NaN, +inf and -inf.
    static func poisoned(_ x: MLXArray) -> MLXArray {
        let flat = x.flattened()
        let marks = MLXArray(0 ..< flat.size)
        var result = which(marks .== 1, MLXArray(Float.nan).asType(x.dtype), flat)
        result = which(marks .== 5, MLXArray(Float.infinity).asType(x.dtype), result)
        result = which(marks .== 9, MLXArray(-Float.infinity).asType(x.dtype), result)
        return result.reshaped(x.shape)
    }

    @Test(arguments: [(1, 4, 640), (3, 4, 200), (2, 2, 96)])
    func convolveMatchesConcatConvSilu(batch: Int, taps: Int, channels: Int) throws {
        for dtype in [DType.bfloat16, .float32] {
            let conv = Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: taps,
                groups: channels, bias: true)
            try conv.update(
                parameters: .unflattened([
                    "weight": Self.random([channels, taps, 1], seed: 1, scale: 0.5, dtype: dtype),
                    "bias": Self.random([channels], seed: 2, scale: 0.2, dtype: dtype),
                ]), verify: .all)
            for poison in [false, true] {
                var state = Self.random(
                    [batch, taps - 1, channels], seed: 3, scale: 3, dtype: dtype)
                let input = Self.random([batch, 1, channels], seed: 4, scale: 3, dtype: dtype)
                if poison { state = Self.poisoned(state) }
                let padded = concatenated([state, input], axis: 1)
                let reference = silu(conv(padded))
                let retained = padded[0..., 1...] * MLXArray(1, dtype: dtype)
                let (output, next) = NemotronSuperMambaStep.convolve(
                    state: state, input: input, weight: conv.weight, bias: conv.bias!)
                eval(reference, retained, output, next)
                #expect(modelOptBitwiseEqual(reference, output))
                #expect(modelOptBitwiseEqual(retained, next))
            }
        }
    }

    @Test(arguments: [(1, 8, 32, 16, 2), (3, 4, 64, 128, 1), (2, 6, 16, 40, 3)])
    func scanMatchesComputeDtAndStepKernel(
        batch: Int, heads: Int, headDim: Int, stateSize: Int, groups: Int
    ) {
        for limits in [(Float(0), Float.infinity), (Float(0.001), Float(0.1))] {
            let hidden = Self.random([batch, 1, heads, headDim], seed: 5)
            let B = Self.random([batch, 1, groups, stateSize], seed: 6)
            let C = Self.random([batch, 1, groups, stateSize], seed: 7)
            var dt = Self.random([batch, 1, heads], seed: 8, scale: 4)
            dt = Self.poisoned(dt)
            let dtBias = Self.random([heads], seed: 9)
            let aLog = Self.random([heads], seed: 10, dtype: .float32)
            let D = Self.random([heads], seed: 11)
            let state = Self.random(
                [batch, heads, headDim, stateSize], seed: 12, dtype: .float32)
            let (referenceY, referenceState) = ssmUpdateKernel(
                hiddenStates: hidden, ALog: aLog, B: B, C: C, D: D, dt: dt, dtBias: dtBias,
                state: state, timeStepLimit: limits)
            let (y, next) = NemotronSuperMambaStep.scan(
                hiddenStates: hidden, ALog: aLog, B: B, C: C, D: D, dtRaw: dt, dtBias: dtBias,
                state: state, timeStepLimit: limits)
            eval(referenceY, referenceState, y, next)
            #expect(modelOptBitwiseEqual(referenceY, y))
            #expect(modelOptBitwiseEqual(referenceState, next))
        }
    }

    @Test(arguments: [(1, 8192, 1024), (3, 512, 128), (2, 96, 32), (1, 200, 100)])
    func gatedNormMatchesModule(rows: Int, width: Int, groupSize: Int) throws {
        for dtype in [DType.bfloat16, .float32] {
            let norm = NemotronHRMSNormGated(dimensions: width, eps: 1e-5, groupSize: groupSize)
            try norm.update(
                parameters: .unflattened([
                    "weight": Self.random([width], seed: 13, scale: 0.5, dtype: dtype) + 1
                ]), verify: .all)
            for poison in [false, true] {
                var y = Self.random([1, rows, width], seed: 14, scale: 2, dtype: dtype)
                let gate = Self.random([1, rows, width], seed: 15, scale: 2, dtype: dtype)
                if poison { y = Self.poisoned(y) }
                let reference = norm(y, gate: gate)
                let fused = NemotronSuperMambaStep.gatedNorm(
                    y, gate: gate, weight: norm.weight, groupSize: groupSize, eps: 1e-5)
                eval(reference, fused)
                #expect(reference.dtype == fused.dtype)
                #expect(modelOptBitwiseEqual(reference, fused))
            }
        }
    }

    @Test(arguments: [(1, 4096), (5, 4096), (3, 100), (2, 64)])
    func residualNormMatchesAddThenRMSNorm(rows: Int, width: Int) throws {
        for dtype in [DType.bfloat16, .float32] {
            let norm = RMSNorm(dimensions: width, eps: 1e-5)
            try norm.update(
                parameters: .unflattened([
                    "weight": Self.random([width], seed: 16, scale: 0.5, dtype: dtype) + 1
                ]), verify: .all)
            for poison in [false, true] {
                var x = Self.random([1, rows, width], seed: 17, scale: 4, dtype: dtype)
                let y = Self.random([1, rows, width], seed: 18, scale: 4, dtype: dtype)
                if poison { x = Self.poisoned(x) }
                let sum = x + y
                let reference = norm(sum)
                let fused = NemotronSuperResidualNorm.apply(x, y, weight: norm.weight, eps: 1e-5)
                eval(sum, reference, fused.sum, fused.normed)
                #expect(modelOptBitwiseEqual(sum, fused.sum))
                #expect(modelOptBitwiseEqual(reference, fused.normed))
                // A MoE boundary: x + (routed + shared).
                let shared = Self.random([1, rows, width], seed: 19, scale: 4, dtype: dtype)
                let moeSum = x + (y + shared)
                let moeReference = norm(moeSum)
                let moeFused = NemotronSuperResidualNorm.apply(
                    x, y, extra: shared, weight: norm.weight, eps: 1e-5)
                eval(moeSum, moeReference, moeFused.sum, moeFused.normed)
                #expect(modelOptBitwiseEqual(moeSum, moeFused.sum))
                #expect(modelOptBitwiseEqual(moeReference, moeFused.normed))
            }
        }
    }
}
