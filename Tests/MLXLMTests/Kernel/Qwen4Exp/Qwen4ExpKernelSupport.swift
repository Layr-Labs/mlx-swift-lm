import Foundation
import MLX

@testable import MLXLMCommon

/// Shared helpers for the Qwen4Exp kernel tests.
///
/// The reference of each quantized kernel dequantizes the packed weight to
/// float32 and multiplies in float32. The kernels keep a float32
/// accumulator and round the output once to bfloat16 or float16, so the
/// difference is near one rounding step of the output dtype.
enum Qwen4ExpKernelSupport {

    /// A packed affine weight with seeded random values.
    ///
    /// `shape` is `[outputs, inputs]` or `[experts, outputs, inputs]`. The
    /// values are normal with a standard deviation of `1 / sqrt(inputs)`, so
    /// one output of a product with a unit normal input has a standard
    /// deviation near 1.
    static func packed(
        _ shape: [Int], bits: Int, groupSize: Int, dtype: DType = .bfloat16, seed: UInt64
    ) -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
        let inputs = shape[shape.count - 1]
        let values =
            (MLXRandom.normal(shape, key: MLXRandom.key(seed))
            * Float(1 / Double(inputs).squareRoot())).asType(dtype)
        let (weight, scales, biases) = MLX.quantized(
            values, groupSize: groupSize, bits: bits, mode: .affine)
        let offsets = biases ?? MLXArray.zeros(like: scales)
        eval(weight, scales, offsets)
        return (weight, scales, offsets)
    }

    /// The packed weight as float32 values, with float32 scales and biases.
    static func dequantizedFloat32(
        _ weight: MLXArray, scales: MLXArray, biases: MLXArray, bits: Int, groupSize: Int
    ) -> MLXArray {
        dequantized(
            weight, scales: scales.asType(.float32), biases: biases.asType(.float32),
            groupSize: groupSize, bits: bits, mode: .affine)
    }

    /// Seeded unit normal input in `dtype`.
    static func input(_ shape: [Int], dtype: DType = .bfloat16, seed: UInt64) -> MLXArray {
        let x = MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(dtype)
        eval(x)
        return x
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    /// True when `|actual - expected| <= atol + rtol * |expected|` for each
    /// element.
    static func isClose(
        _ actual: MLXArray, _ expected: MLXArray, atol: Float, rtol: Float
    ) -> Bool {
        let a = actual.asType(.float32)
        let e = expected.asType(.float32)
        return all(abs(a - e) .<= (abs(e) * rtol + atol)).item(Bool.self)
    }

    /// True when `value` is nil. The kernels return nil for inputs outside
    /// their contract.
    static func isNil<T>(_ value: T?) -> Bool {
        value == nil
    }

    static func isEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && all(a .== b).item(Bool.self)
    }

    /// Sets the process environment variables in `values` for the time of
    /// `body`, then restores the old values. `Qwen4ExpEnvironment` keeps a
    /// snapshot, so the snapshot is read again before and after `body`.
    ///
    /// CI runs the tests with `--no-parallel`. The suites that call this
    /// function are also `.serialized`.
    static func withEnvironment<T>(
        _ values: [String: String?], _ body: () throws -> T
    ) rethrows -> T {
        var previous: [String: String?] = [:]
        for (name, value) in values {
            previous.updateValue(ProcessInfo.processInfo.environment[name], forKey: name)
            if let value {
                setenv(name, value, 1)
            } else {
                unsetenv(name)
            }
        }
        Qwen4ExpEnvironment.refresh()
        defer {
            for (name, value) in previous {
                if let value {
                    setenv(name, value, 1)
                } else {
                    unsetenv(name)
                }
            }
            Qwen4ExpEnvironment.refresh()
        }
        return try body()
    }
}
