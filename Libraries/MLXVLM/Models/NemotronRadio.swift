// Derived from modeling_radio.py, Copyright (c) 2026 NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// http://www.apache.org/licenses/LICENSE-2.0
// Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
// See the License for the specific language governing permissions and limitations.
// Reference: nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4,
// revision 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c.

import Foundation
import MLX
import MLXNN

/// Fixed image architecture of the pinned Super VL C-RADIO tower.
/// Video projection, summary output and model registration are deliberately absent.
public final class NemotronRadio: Module {
    @ModuleInfo var embeddings: NemotronRadioEmbeddings
    @ModuleInfo var blocks: [NemotronRadioBlock]
    let dimensions: NemotronRadioDimensions

    public override convenience init() { self.init(.superVL) }

    init(_ dimensions: NemotronRadioDimensions) {
        self.dimensions = dimensions
        self.embeddings = NemotronRadioEmbeddings(dimensions)
        self.blocks = (0..<dimensions.layers).map { _ in NemotronRadioBlock(dimensions) }
    }

    /// Already CLIP-normalized NCHW pixels, as supplied by the upstream processor.
    /// The upstream wrapper calls make_preprocessor_external(); applying the
    /// serialized input_conditioner again would normalize twice.
    public func callAsFunction(normalizedPixels pixels: MLXArray) -> MLXArray {
        precondition(pixels.ndim == 4 && pixels.dim(1) == dimensions.channels)
        precondition(pixels.dim(2) > 0 && pixels.dim(3) > 0)
        precondition(pixels.dim(2) % dimensions.patch == 0 && pixels.dim(3) % dimensions.patch == 0)
        var hidden = embeddings(pixels)
        for block in blocks { hidden = block(hidden) }
        return hidden[0..., dimensions.prefix..., 0...]
    }
}

struct NemotronRadioDimensions {
    let width: Int
    let layers: Int
    let heads: Int
    let channels: Int
    let patch: Int
    let positionSide: Int
    let prefix: Int
    let mlpWidth: Int

    static let superVL = Self(width: 1280, layers: 32, heads: 16, channels: 3,
        patch: 16, positionSide: 128, prefix: 10, mlpWidth: 5120)
}

final class NemotronRadioEmbeddings: Module, UnaryLayer {
    @ModuleInfo var projection: Linear
    @ParameterInfo(key: "position_embedding") var position: MLXArray
    @ParameterInfo(key: "cls_register_token") var prefix: MLXArray
    let dimensions: NemotronRadioDimensions

    init(_ d: NemotronRadioDimensions) {
        dimensions = d
        self.projection = Linear(d.channels * d.patch * d.patch, d.width, bias: false)
        self._position.wrappedValue = MLXArray.zeros([1, d.positionSide * d.positionSide, d.width])
        self._prefix.wrappedValue = MLXArray.zeros([d.prefix, d.width])
    }

    static func imagePatches(_ pixels: MLXArray, patch: Int) -> MLXArray {
        let (b, c, h, w) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        return pixels.reshaped(b, c, h / patch, patch, w / patch, patch)
            .transposed(0, 2, 4, 1, 3, 5).reshaped(b, h / patch * (w / patch), c * patch * patch)
    }

    /// align_corners=false bilinear resize to a square, then top-left crop.
    /// Coordinate generation is host-only; all table arithmetic stays FP32 on MLX.
    static func croppedPositions(_ table: MLXArray, side: Int, rows: Int, columns: Int,
        dtype: DType) -> MLXArray {
        precondition(side > 0 && rows > 0 && columns > 0)
        let target = max(rows, columns)
        var low: [Int32] = [], high: [Int32] = [], fraction: [Float] = []
        for index in 0..<target {
            let coordinate = max(0, (Float(index) + 0.5) * Float(side) / Float(target) - 0.5)
            let lower = min(Int(coordinate), side - 1)
            low.append(Int32(lower)); high.append(Int32(min(lower + 1, side - 1)))
            fraction.append(coordinate - Float(lower))
        }
        var indices = [[Int32]](repeating: [], count: 4)
        var weights = [[Float]](repeating: [], count: 4)
        for row in 0..<rows {
            for col in 0..<columns {
                let y = fraction[row], x = fraction[col]
                indices[0].append(low[row] * Int32(side) + low[col])
                indices[1].append(low[row] * Int32(side) + high[col])
                indices[2].append(high[row] * Int32(side) + low[col])
                indices[3].append(high[row] * Int32(side) + high[col])
                weights[0].append((1-y)*(1-x)); weights[1].append((1-y)*x)
                weights[2].append(y*(1-x)); weights[3].append(y*x)
            }
        }
        let flat = table.reshaped(side * side, -1).asType(.float32)
        var output = take(flat, MLXArray(indices[0]), axis: 0) * MLXArray(weights[0]).reshaped(-1, 1)
        for corner in 1..<4 {
            output = output + take(flat, MLXArray(indices[corner]), axis: 0)
                * MLXArray(weights[corner]).reshaped(-1, 1)
        }
        return output.asType(dtype).expandedDimensions(axis: 0)
    }

    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let d = dimensions
        let patches = projection(Self.imagePatches(pixels, patch: d.patch))
        let positions = Self.croppedPositions(position, side: d.positionSide,
            rows: pixels.dim(2) / d.patch, columns: pixels.dim(3) / d.patch, dtype: patches.dtype)
        let tokens = broadcast(prefix.expandedDimensions(axis: 0), to: [pixels.dim(0), d.prefix, d.width])
        return concatenated([tokens, patches + positions], axis: 1)
    }
}

final class NemotronRadioBlock: Module, UnaryLayer {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    @ModuleInfo var output: Linear
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let dimensions: NemotronRadioDimensions

    init(_ d: NemotronRadioDimensions) {
        precondition(d.width > 0 && d.heads > 0 && d.width % d.heads == 0)
        dimensions = d
        self.norm1 = LayerNorm(dimensions: d.width, eps: 1e-6)
        self.norm2 = LayerNorm(dimensions: d.width, eps: 1e-6)
        self.query = Linear(d.width, d.width, bias: true)
        self.key = Linear(d.width, d.width, bias: true)
        self.value = Linear(d.width, d.width, bias: true)
        self.output = Linear(d.width, d.width, bias: true)
        self.fc1 = Linear(d.width, d.mlpWidth, bias: true)
        self.fc2 = Linear(d.mlpWidth, d.width, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let d = dimensions, normalized = norm1(x)
        func heads(_ linear: Linear) -> MLXArray {
            linear(normalized).reshaped(x.dim(0), x.dim(1), d.heads, d.width / d.heads).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(queries: heads(query), keys: heads(key),
            values: heads(value), scale: 1 / sqrt(Float(d.width / d.heads)), mask: nil)
            .transposed(0, 2, 1, 3).reshaped(x.shape)
        // Pinned C-RADIO has no learned LayerScale weights; reference initializes
        // its missing lambda1 parameters to one. No final tower normalization.
        let residual = x + output(attended)
        return residual + fc2(gelu(fc1(norm2(residual))))
    }
}
