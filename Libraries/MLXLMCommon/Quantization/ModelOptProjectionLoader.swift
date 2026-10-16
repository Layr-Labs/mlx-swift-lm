import Foundation
import MLX

/// Loads one bounded calibrated projection from an already SHA256-verified shard.
/// Architecture dimensions and original computation dtype are mandatory inputs.
/// This deliberately rejects dense layouts rather than silently converting them.
public enum ModelOptProjectionLoader {
    public enum Failure: Error { case unsupportedStorage, payloadLimit, invalidScalar }
    public static func load(
        file: ModelOptTensorFile, prefix: String,
        inputDimensions: Int, outputDimensions: Int, weightDType: DType,
        maximumBytes: Int = 64 << 20
    ) throws -> ModelOptCalibratedLinear {
        guard maximumBytes > 0, maximumBytes <= 64 << 20 else { throw Failure.payloadLimit }
        let stem = prefix + "."
        var tensors: [String: ModelOptProjectionLayout.Tensor] = [:]
        var total = 0
        for (key, entry) in file.entries where key.hasPrefix(stem) {
            tensors[String(key.dropFirst(stem.count))] = entry.layout
            let (next, overflow) = total.addingReportingOverflow(entry.layout.byteCount)
            guard !overflow, next <= maximumBytes else { throw Failure.payloadLimit }
            total = next
        }
        let layout = try ModelOptProjectionLayout(
            tensors: tensors,
            inputDimensions: inputDimensions, outputDimensions: outputDimensions)
        guard layout.storage == .fp8E4M3 || layout.storage == .nvfp4 else {
            throw Failure.unsupportedStorage
        }
        func scalar(_ key: String) throws -> Float {
            let data = try file.read(stem + key)
            guard data.count == 4 else { throw Failure.invalidScalar }
            let bits = data.enumerated().reduce(UInt32(0)) {
                $0 | UInt32($1.element) << (8 * $1.offset)
            }
            let value = Float(bitPattern: bits)
            guard value.isFinite, value > 0 else { throw Failure.invalidScalar }
            return value
        }
        func array(_ key: String) throws -> MLXArray {
            let data = try file.read(stem + key)
            return MLXArray(Array(data)).reshaped(tensors[key]!.shape)
        }
        // Read and validate scalars before allocating device arrays.
        let inputScale = try scalar("input_scale")
        switch layout.storage {
        case .fp8E4M3:
            let scale = try scalar("weight_scale")
            let weights = try ModelOptFP8Weights(e4m3Bytes: array("weight"), tensorScale: scale)
            return try ModelOptCalibratedLinear(
                weights: weights, inputScale: inputScale, weightDType: weightDType)
        case .nvfp4:
            let scale = try scalar("weight_scale_2")
            let weights = try ModelOptNVFP4Weights(
                packedBytes: array("weight"),
                e4m3ScaleBytes: array("weight_scale"), tensorScale: scale)
            return try ModelOptCalibratedLinear(
                weights: weights, inputScale: inputScale, weightDType: weightDType)
        default: throw Failure.unsupportedStorage
        }
    }
}

extension ModelOptProjectionLoader {
    /// Index-bound variant, including scale sidecars placed in different shards.
    /// maximumBytes comes from an already held installation residency grant.
    public static func load(
        sources: [String: ModelOptTensorSource], inputDimensions: Int,
        outputDimensions: Int, weightDType: DType, maximumBytes: Int,
        construction: NativeConstructionScope? = nil
    ) throws -> ModelOptCalibratedLinear {
        var total = 0
        for source in sources.values {
            let (next, overflow) = total.addingReportingOverflow(source.layout.byteCount)
            guard !overflow, maximumBytes > 0, next <= maximumBytes else {
                throw Failure.payloadLimit
            }
            total = next
        }
        let layout = try ModelOptProjectionLayout(
            tensors: sources.mapValues(\.layout),
            inputDimensions: inputDimensions, outputDimensions: outputDimensions)
        let input = try sources["input_scale"]?.scalar()
        guard let input else { throw Failure.invalidScalar }
        switch layout.storage {
        case .fp8E4M3:
            let scale = try sources["weight_scale"]!.scalar()
            let bytes = try sources["weight"]!.array(
                maximumBytes: maximumBytes, construction: construction)
            return try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(e4m3Bytes: bytes, tensorScale: scale),
                inputScale: input, weightDType: weightDType)
        case .nvfp4:
            let scale = try sources["weight_scale_2"]!.scalar()
            let bytes = try sources["weight"]!.array(
                maximumBytes: maximumBytes, construction: construction)
            let blocks = try sources["weight_scale"]!.array(
                maximumBytes: maximumBytes, construction: construction)
            return try ModelOptCalibratedLinear(
                weights: ModelOptNVFP4Weights(
                    packedBytes: bytes,
                    e4m3ScaleBytes: blocks, tensorScale: scale), inputScale: input,
                weightDType: weightDType)
        default: throw Failure.unsupportedStorage
        }
    }
}
