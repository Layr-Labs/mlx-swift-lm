// Copyright © 2026 Eigen Labs.

import Foundation

/// Dtype-aware preflight for one bias-free weight-only NVFP4 projection
/// (ModelOpt W4A16): packed E2M1 codes, one E4M3 scale per 16 inputs and the
/// per-tensor `weight_scale_2`. Activations stay in their own dtype, so an
/// `input_scale` sidecar is refused; `ModelOptProjectionLayout` validates the
/// activation-quantized form. No tensor payload is read or allocated.
public struct ModelOptWeightOnlyNVFP4Layout: Sendable {
    public typealias Tensor = ModelOptProjectionLayout.Tensor
    public typealias Failure = ModelOptProjectionLayout.Failure

    public static let blockSize = 16
    public let inputDimensions: Int
    public let outputDimensions: Int

    /// Keys are projection-local: weight, weight_scale, weight_scale_2.
    public init(tensors: [String: Tensor], inputDimensions: Int, outputDimensions: Int) throws {
        guard inputDimensions > 0, outputDimensions > 0,
            inputDimensions <= 1_048_576, outputDimensions <= 1_048_576,
            inputDimensions % Self.blockSize == 0
        else { throw Failure.invalidDimensions }
        func tensor(_ name: String, dtype: String, columns: Int) throws {
            guard let value = tensors[name] else { throw Failure.missing(name) }
            let (bytes, overflow) = outputDimensions.multipliedReportingOverflow(by: columns)
            guard !overflow, value.dtype == dtype, value.shape == [outputDimensions, columns],
                value.byteCount == bytes
            else { throw Failure.invalidTensor(name) }
        }
        try tensor("weight", dtype: "U8", columns: inputDimensions / 2)
        try tensor("weight_scale", dtype: "F8_E4M3", columns: inputDimensions / Self.blockSize)
        guard let global = tensors["weight_scale_2"] else {
            throw Failure.missing("weight_scale_2")
        }
        guard global.dtype == "F32", global.shape == [] || global.shape == [1],
            global.byteCount == 4
        else { throw Failure.invalidTensor("weight_scale_2") }
        let allowed: Set<String> = ["weight", "weight_scale", "weight_scale_2"]
        for key in tensors.keys.sorted() where !allowed.contains(key) {
            throw Failure.unexpected(key)
        }
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
    }
}
