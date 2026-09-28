// Copyright © 2026 Eigen Labs. HF5711 residual quantizer (Apache-2.0).
// Original source copyright 2026 Xiaomi Corporation and HuggingFace Inc.
// SPDX-License-Identifier: Apache-2.0
import Foundation
import MLX
import MLXLLM

public struct MiMoV26AudioQuantizedCodes {
    public let codes: MLXArray
    /// Must be evaluated and checked before any IDs are published.
    public let allFinite: MLXArray
    public let frameCounts: [Int]
    fileprivate init(codes: MLXArray,allFinite: MLXArray,frameCounts: [Int]) {
        self.codes = codes;self.allFinite = allFinite;self.frameCounts = frameCounts
    }
}

/// No EMA/init state and no decoder. Runtime tables hold F32 values after the
/// selected loader's BF16 roundtrip; source storage bytes remain untouched.
public final class MiMoV26ResidualVectorQuantizer {
    public let configuration: MiMoV26AudioInputConfiguration
    private let tables: [MLXArray]
    public var materializationRoots: [MLXArray] { tables }
    public init(configuration: MiMoV26AudioInputConfiguration, sourceTables: [MLXArray]) throws {
        guard sourceTables.count == configuration.quantizers,
              zip(sourceTables,configuration.codebookSizes).allSatisfy({ $0.0.shape == [$0.1,configuration.hiddenSize] && $0.0.dtype == .float32 }) else {
            throw MiMoV26AudioInputError.weights("RVQ source table shape/dtype/order")
        }
        self.configuration = configuration
        tables = sourceTables.map { $0.asType(.bfloat16).asType(.float32) }
    }

    public func quantize(features: MLXArray, frameCounts: [Int], tileFrames: Int,
                         isCancelled: () -> Bool = { false }) throws -> MiMoV26AudioQuantizedCodes {
        let frames = try frameCounts.reduce(0) { try MiMoV26AudioChecked.add($0,$1,"RVQ frame sum") }
        guard frameCounts.allSatisfy({ $0 > 0 }), tileFrames > 0, tileFrames <= Int(Int32.max),
              features.shape == [frames,configuration.hiddenSize], [.bfloat16,.float32].contains(features.dtype) else {
            throw MiMoV26AudioInputError.input("RVQ feature/length/tile contract")
        }
        if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        var valid = all(isFinite(features)), blocks: [MLXArray] = []
        for table in tables { valid = valid .&& all(isFinite(table)) }
        for start in stride(from:0,to:frames,by:tileFrames) {
            if isCancelled() { throw MiMoV26AudioInputError.cancelled }
            let end = min(frames,start+tileFrames)
            var residual = features[start..<end].asType(.float32), indices: [MLXArray] = []
            for table in tables {
                if isCancelled() { throw MiMoV26AudioInputError.cancelled }
                // Python precedence is (2*x) @ embed, followed by the two
                // additions/subtractions in this order, all F32.
                let distance = sum(residual*residual,axis:1,keepDims:true)
                    - matmul(Float(2)*residual,table.transposed())
                    + sum(table*table,axis:1).expandedDimensions(axis:0)
                valid = valid .&& all(isFinite(distance))
                let best = min(distance,axis:1,keepDims:true)
                // Explicit first-index ties, independent of arg-reduction
                // scheduling. Invalid rows use safe index0 but never publish.
                let candidates = MLX.where(distance .== best,MLXArray(0..<table.dim(0)),Int32.max)
                let first = min(candidates,axis:1)
                let selected = MLX.where(isFinite(best.squeezed(axis:1)),first,Int32(0)).asType(.int32)
                residual = residual-take(table,selected,axis:0)
                indices.append(selected)
            }
            blocks.append(stacked(indices,axis:1))
        }
        let codes = blocks.isEmpty ? MLXArray.zeros([0,configuration.quantizers],dtype:.int32)
            : blocks.count == 1 ? blocks[0] : concatenated(blocks,axis:0)
        return .init(codes:codes,allFinite:valid,frameCounts:frameCounts)
    }
}
