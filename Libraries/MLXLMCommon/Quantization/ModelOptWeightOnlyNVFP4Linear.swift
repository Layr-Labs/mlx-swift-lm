// Copyright © 2026 Eigen Labs.

import MLX
import MLXNN

/// One bias-free ModelOpt weight-only NVFP4 projection (W4A16), kept in the
/// producer's layout: `W = e2m1(weight) * e4m3(weight_scale) * weight_scale_2`.
/// Element `2i` of a row is the low nibble of byte `i`, element `2i + 1` the
/// high nibble; each E4M3 scale covers 16 inputs. Activations are not quantized.
///
/// The product is MLX's `.nvfp4` quantized matmul over the packed words. That
/// kernel takes no per-tensor scale, so `weight_scale_2` multiplies the output
/// in FP32. The unscaled product is `y / weight_scale_2`, which can exceed the
/// FP16 range: callers run BF16 or FP32 activations.
///
/// Deliberately not a `Linear`: generic quantization, LoRA and fusion code must
/// not treat the packed words as a dense or affine weight.
public final class ModelOptWeightOnlyNVFP4Linear: Module, UnaryLayer {
    public enum Failure: Error, Equatable { case invalidLayout, invalidScale }

    public let inputDimensions: Int
    public let outputDimensions: Int

    /// U32 `[out, in / 8]`: a zero-copy view of the U8 `[out, in / 2]` codes.
    @ParameterInfo(key: "weight") public private(set) var weight: MLXArray
    /// U8 `[out, in / 16]`: the E4M3 block scales as stored.
    @ParameterInfo(key: "weight_scale") public private(set) var weightScale: MLXArray
    /// F32 `[1]`: ModelOpt's direct multiplier, verbatim.
    @ParameterInfo(key: "weight_scale_2") public private(set) var weightScale2: MLXArray

    /// The caller establishes provenance: `packedBytes` is the checkpoint's U8
    /// `weight` and `e4m3ScaleBytes` its F8_E4M3 `weight_scale`, both as raw bytes.
    public init(packedBytes: MLXArray, e4m3ScaleBytes: MLXArray, tensorScale: Float) throws {
        guard packedBytes.dtype == .uint8, packedBytes.ndim == 2,
            packedBytes.dim(0) > 0, packedBytes.dim(1) > 0, packedBytes.dim(1) % 8 == 0,
            e4m3ScaleBytes.dtype == .uint8,
            e4m3ScaleBytes.shape == [packedBytes.dim(0), packedBytes.dim(1) / 8]
        else { throw Failure.invalidLayout }
        try Self.validate(tensorScale)
        inputDimensions = packedBytes.dim(1) * 2
        outputDimensions = packedBytes.dim(0)
        _weight.wrappedValue = packedBytes.view(dtype: .uint32)
        _weightScale.wrappedValue = e4m3ScaleBytes
        _weightScale2.wrappedValue = MLXArray([tensorScale])
        super.init()
        freeze()
    }

    public convenience init(weights: ModelOptNVFP4Weights) throws {
        try self.init(
            packedBytes: weights.packedWeights.view(dtype: .uint8),
            e4m3ScaleBytes: weights.blockScales, tensorScale: weights.tensorScale)
    }

    /// The largest unscaled weight is 6 * 448; its scaled value must stay finite.
    static func validate(_ scale: Float) throws {
        guard scale.isFinite, scale > 0, (scale * (448 * 6)).isFinite else {
            throw Failure.invalidScale
        }
    }

    /// A parameter update is the only way the stored tensors change, so the
    /// layout and the scalar are checked again here, outside every forward.
    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        guard weight.dtype == .uint32, weight.shape == [outputDimensions, inputDimensions / 8],
            weightScale.dtype == .uint8,
            weightScale.shape == [outputDimensions, inputDimensions / 16],
            weightScale2.dtype == .float32, weightScale2.size == 1
        else { throw Failure.invalidLayout }
        try Self.validate(weightScale2.item(Float.self))
        return self
    }

    /// `x @ (e2m1 * e4m3).T` in the dtype of `x`, before `weight_scale_2`.
    public func unscaledProduct(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, weight, scales: weightScale, biases: nil, transpose: true,
            groupSize: ModelOptWeightOnlyNVFP4Layout.blockSize, bits: 4, mode: .nvfp4)
    }

    /// Applies `weight_scale_2` in FP32 and returns to the product's dtype.
    public func scaled(_ product: MLXArray) -> MLXArray {
        (product.asType(.float32) * weightScale2).asType(product.dtype)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        scaled(unscaledProduct(x))
    }

    /// Bounded reference: the dense weight in `dtype`. Allocates `out * in`
    /// elements; never a loading strategy.
    public func dequantizedWeight(dtype: DType = .float32) -> MLXArray {
        let unscaled = MLX.dequantized(
            weight, scales: weightScale, biases: nil,
            groupSize: ModelOptWeightOnlyNVFP4Layout.blockSize, bits: 4, mode: .nvfp4,
            dtype: .float32)
        return (unscaled * weightScale2).asType(dtype)
    }
}

extension ModelOptWeightOnlyNVFP4Linear {
    /// Loads one projection from index-bound, SHA-256-verified sources. The
    /// layout is validated from header metadata before any payload read.
    public static func load(
        sources: [String: ModelOptTensorSource], inputDimensions: Int, outputDimensions: Int,
        maximumBytes: Int
    ) throws -> ModelOptWeightOnlyNVFP4Linear {
        var total = 0
        for source in sources.values {
            let (next, overflow) = total.addingReportingOverflow(source.layout.byteCount)
            guard !overflow, maximumBytes > 0, next <= maximumBytes else {
                throw ModelOptProjectionLoader.Failure.payloadLimit
            }
            total = next
        }
        _ = try ModelOptWeightOnlyNVFP4Layout(
            tensors: sources.mapValues(\.layout),
            inputDimensions: inputDimensions, outputDimensions: outputDimensions)
        let scale = try sources["weight_scale_2"]!.scalar()
        return try ModelOptWeightOnlyNVFP4Linear(
            packedBytes: sources["weight"]!.array(maximumBytes: maximumBytes),
            e4m3ScaleBytes: sources["weight_scale"]!.array(maximumBytes: maximumBytes),
            tensorScale: scale)
    }
}
