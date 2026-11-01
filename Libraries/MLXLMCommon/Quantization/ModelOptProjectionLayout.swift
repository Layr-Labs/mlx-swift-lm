import Foundation

/// Dtype-aware preflight for one bias-free projection. No tensor payload is read
/// or allocated. Expected dimensions come from the validated model architecture.
/// This is ModelOpt layout validation, not MLX affine conversion.
public struct ModelOptProjectionLayout: Sendable {
    public enum Storage: Sendable, Equatable { case bfloat16, float16, float32, fp8E4M3, nvfp4 }
    public enum Failure: Error, Equatable {
        case invalidDimensions
        case missing(String)
        case unexpected(String)
        case invalidTensor(String)
    }
    public struct Tensor: Sendable {
        public let dtype: String
        public let shape: [Int]
        public let byteCount: Int
        public init(dtype: String, shape: [Int], byteCount: Int) {
            self.dtype = dtype
            self.shape = shape
            self.byteCount = byteCount
        }
    }
    public let storage: Storage
    public let inputDimensions: Int
    public let outputDimensions: Int

    /// Keys are projection-local: weight, input_scale, weight_scale, weight_scale_2.
    /// Cache-scale or bias tensors must be handled explicitly by a different loader.
    public init(tensors: [String: Tensor], inputDimensions: Int, outputDimensions: Int) throws {
        guard inputDimensions > 0, outputDimensions > 0,
            inputDimensions <= 1_048_576, outputDimensions <= 1_048_576
        else {
            throw Failure.invalidDimensions
        }
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        guard let weight = tensors["weight"] else { throw Failure.missing("weight") }
        func tensor(_ name: String, dtype: String, shape: [Int], width: Int) throws {
            guard let value = tensors[name] else { throw Failure.missing(name) }
            var count = 1
            for dimension in shape {
                let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
                guard !overflow, dimension > 0 else { throw Failure.invalidTensor(name) }
                count = next
            }
            let (bytes, overflow) = count.multipliedReportingOverflow(by: width)
            guard !overflow, value.dtype == dtype, value.shape == shape, value.byteCount == bytes
            else {
                throw Failure.invalidTensor(name)
            }
        }
        func scalar(_ name: String) throws {
            guard let value = tensors[name] else { throw Failure.missing(name) }
            guard value.dtype == "F32", value.shape == [] || value.shape == [1],
                value.byteCount == 4
            else {
                throw Failure.invalidTensor(name)
            }
        }
        let allowed: Set<String>
        switch weight.dtype {
        case "BF16", "F16", "F32":
            storage =
                weight.dtype == "BF16" ? .bfloat16 : weight.dtype == "F16" ? .float16 : .float32
            allowed = ["weight"]
            try tensor(
                "weight", dtype: weight.dtype, shape: [outputDimensions, inputDimensions],
                width: weight.dtype == "F32" ? 4 : 2)
        case "F8_E4M3":
            storage = .fp8E4M3
            allowed = ["weight", "weight_scale", "input_scale"]
            try tensor(
                "weight", dtype: "F8_E4M3", shape: [outputDimensions, inputDimensions], width: 1)
            try scalar("weight_scale")
            try scalar("input_scale")
        case "U8":
            guard inputDimensions % 16 == 0 else { throw Failure.invalidDimensions }
            storage = .nvfp4
            allowed = ["weight", "weight_scale", "weight_scale_2", "input_scale"]
            try tensor(
                "weight", dtype: "U8", shape: [outputDimensions, inputDimensions / 2], width: 1)
            try tensor(
                "weight_scale", dtype: "F8_E4M3", shape: [outputDimensions, inputDimensions / 16],
                width: 1)
            try scalar("weight_scale_2")
            try scalar("input_scale")
        default: throw Failure.invalidTensor("weight")
        }
        for key in tensors.keys where !allowed.contains(key) { throw Failure.unexpected(key) }
    }
}
