import Foundation
import MLX

@testable import MLXLMCommon

/// Seeded synthetic ModelOpt tensors for graph/eager parity. Not checkpoint data.
struct ModelOptGraphFixtures {
    private var state: UInt64

    init(seed: UInt64) { state = seed &* 0x9E37_79B9_7F4A_7C15 | 1 }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }

    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }

    /// Uniform packed E2M1 nibbles, including signed zero codes.
    mutating func packedNVFP4(rows: Int, columns: Int) -> MLXArray {
        MLXArray((0 ..< rows * columns / 2).map { _ in UInt8(truncatingIfNeeded: next()) })
            .reshaped(rows, columns / 2)
    }

    /// E4M3 block scales between 0.25 (0x28) and 4 (0x48).
    mutating func blockScales(rows: Int, columns: Int) -> MLXArray {
        MLXArray((0 ..< rows * columns / 16).map { _ in UInt8(0x28 + next() % 33) })
            .reshaped(rows, columns / 16)
    }

    /// Every finite E4M3FN code except the two NaN codes.
    mutating func fp8Weights(rows: Int, columns: Int) -> MLXArray {
        MLXArray(
            (0 ..< rows * columns).map { _ -> UInt8 in
                let code = UInt8(truncatingIfNeeded: next())
                return code & 0x7F == 0x7F ? code ^ 0x01 : code
            }
        ).reshaped(rows, columns)
    }

    mutating func values(_ count: Int, magnitude: Float) -> [Float] {
        (0 ..< count).map { _ in (unit() * 2 - 1) * magnitude }
    }

    mutating func expertBank(
        experts: Int, outputs: Int, inputs: Int, magnitude: Float, dtype: DType
    ) throws -> ModelOptNVFP4SwitchLinear {
        let weights = try (0 ..< experts).map { _ in
            try ModelOptNVFP4Weights(
                packedBytes: packedNVFP4(rows: outputs, columns: inputs),
                e4m3ScaleBytes: blockScales(rows: outputs, columns: inputs),
                tensorScale: 0.001 + unit() * 0.01)
        }
        // A valid row reaches |x| = magnitude; keep it below ModelOpt's 464 limit.
        let inputScales = (0 ..< experts).map { _ in magnitude / 2688 * (1 + unit()) }
        return try ModelOptNVFP4SwitchLinear(
            experts: weights, inputScales: inputScales, weightDType: dtype)
    }

    mutating func nvfp4Projection(
        outputs: Int, inputs: Int, magnitude: Float, dtype: DType
    ) throws -> ModelOptCalibratedLinear {
        try ModelOptCalibratedLinear(
            weights: ModelOptNVFP4Weights(
                packedBytes: packedNVFP4(rows: outputs, columns: inputs),
                e4m3ScaleBytes: blockScales(rows: outputs, columns: inputs),
                tensorScale: 0.001 + unit() * 0.01),
            inputScale: magnitude / 2688 * (1 + unit()), weightDType: dtype)
    }

    mutating func fp8Projection(
        outputs: Int, inputs: Int, magnitude: Float, dtype: DType
    ) throws -> ModelOptCalibratedLinear {
        try ModelOptCalibratedLinear(
            weights: ModelOptFP8Weights(
                e4m3Bytes: fp8Weights(rows: outputs, columns: inputs),
                tensorScale: 0.0005 + unit() * 0.002),
            inputScale: magnitude / 448 * (1 + unit()), weightDType: dtype)
    }
}

/// Bitwise equality that treats NaN at the same positions as equal.
func modelOptBitwiseEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.shape == b.shape, a.dtype == b.dtype else { return false }
    let nanA = isNaN(a)
    let same = logicalOr(
        logicalAnd(nanA, isNaN(b)),
        a.view(dtype: a.dtype.size == 2 ? .uint16 : .uint32)
            .== b.view(dtype: b.dtype.size == 2 ? .uint16 : .uint32))
    return same.all().item(Bool.self)
}
