// Derived from NVIDIA modeling_nemotron_h_omni.py, Copyright (c) 2026 NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// http://www.apache.org/licenses/LICENSE-2.0
// Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
// See the License for the specific language governing permissions and limitations.
import MLX
import MLXNN

/// Projector only; does not register or implement the Super VL vision tower.
/// Reference: nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
/// revision 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c,
/// modeling_nemotron_h_omni.py, NemotronH_Omni_Reasoning_V3VisionProjector.
public final class NemotronSuperVisionProjector: Module {
    public enum WeightError: Error {
        case duplicate(String)
        case missing(String)
        case unexpected(String)
        case invalidShape(String)
        case invalidDType(String)
    }
    @ModuleInfo(key: "vision_final_layernorm") var finalNorm: LayerNorm
    @ModuleInfo(key: "mlp1") var projection: NemotronSuperProjectionMLP
    let visionWidth: Int

    public override convenience init() {
        self.init(visionWidth: 1280, hiddenWidth: 20480, outputWidth: 4096)
    }

    // Small dimensions are internal to support causal component tests.
    init(visionWidth: Int, hiddenWidth: Int, outputWidth: Int) {
        precondition(visionWidth > 0 && hiddenWidth > 0 && outputWidth > 0)
        self.visionWidth = visionWidth
        self._finalNorm.wrappedValue = LayerNorm(dimensions: visionWidth, eps: 1e-6)
        self._projection.wrappedValue = NemotronSuperProjectionMLP(
            inputWidth: 4 * visionWidth, hiddenWidth: hiddenWidth, outputWidth: outputWidth)
    }

    /// Extracts only projector tensors from the pinned raw checkpoint or its
    /// canonical projector-local names. Other model tensors remain untouched.
    /// This is a full-precision loader: quantized/projector sidecars fail closed.
    public func loadProjectorWeights(from checkpoint: [String: MLXArray]) throws {
        let extracted = try validatedProjectorWeights(from: checkpoint)
        try update(parameters: ModuleParameters.unflattened(extracted), verify: [.all])
    }

    /// Pure validation, shared with atomic tower+projector loading.
    func validatedProjectorWeights(from checkpoint: [String: MLXArray]) throws -> [String: MLXArray] {
        let aliases = [
            "mlp1.0.weight": "mlp1.norm.weight",
            "mlp1.1.weight": "mlp1.linear1.weight",
            "mlp1.3.weight": "mlp1.linear2.weight",
            "vision_projector.vision_final_layernorm.weight": "vision_final_layernorm.weight",
            "vision_projector.vision_final_layernorm.bias": "vision_final_layernorm.bias",
        ]
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened())
        var extracted: [String: MLXArray] = [:]
        for (source, tensor) in checkpoint {
            let destination: String
            if let mapped = aliases[source] {
                destination = mapped
            } else if source.hasPrefix("vision_projector."), expected[String(source.dropFirst("vision_projector.".count))] != nil {
                destination = String(source.dropFirst("vision_projector.".count))
            } else if expected[source] != nil {
                destination = source
            } else {
                if source.hasPrefix("mlp1.") || source.hasPrefix("vision_projector.")
                    || source.hasPrefix("vision_final_layernorm.") {
                    throw WeightError.unexpected(source)
                }
                continue
            }
            guard extracted[destination] == nil else { throw WeightError.duplicate(destination) }
            extracted[destination] = tensor
        }
        // Validate the entire selection before mutation, including shape errors.
        for (name, parameter) in expected {
            guard let tensor = extracted[name] else { throw WeightError.missing(name) }
            guard tensor.shape == parameter.shape else { throw WeightError.invalidShape(name) }
            guard [.float16, .bfloat16, .float32].contains(tensor.dtype) else {
                throw WeightError.invalidDType(name)
            }
        }
        return extracted
    }

    /// RADIO patch features with class/register tokens already removed: [B,H*W,C].
    public func callAsFunction(_ features: MLXArray, height: Int, width: Int) -> MLXArray {
        precondition(height > 0 && width > 0 && height % 2 == 0 && width % 2 == 0)
        precondition(features.ndim == 3 && features.dim(1) == height * width)
        precondition(features.dim(2) == visionWidth)
        let normalized = finalNorm(features).reshaped(features.dim(0), height, width, visionWidth)
        let shuffled = Self.pixelShuffleV2(normalized)
        return projection(shuffled.reshaped(features.dim(0), height * width / 4, 4 * visionWidth))
    }

    /// Exact upstream v2 ordering at scale 0.5, including the final axis swap.
    static func pixelShuffleV2(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 4 && x.dim(1) % 2 == 0 && x.dim(2) % 2 == 0)
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(batch, height, width / 2, channels * 2)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, width / 2, height / 2, channels * 4)
            .transposed(0, 2, 1, 3)
    }
}

final class NemotronSuperProjectionNorm: Module, UnaryLayer {
    let weight: MLXArray

    init(width: Int) { weight = MLXArray.ones([width]) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Upstream multiplies the norm weight in FP32 before casting back.
        let promoted = x.asType(.float32)
        let variance = mean(promoted * promoted, axis: -1, keepDims: true)
        let normalized = promoted * rsqrt(variance + 1e-5)
        return (weight.asType(.float32) * normalized).asType(x.dtype)
    }
}

final class NemotronSuperProjectionMLP: Module, UnaryLayer {
    @ModuleInfo var norm: NemotronSuperProjectionNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(inputWidth: Int, hiddenWidth: Int, outputWidth: Int) {
        self.norm = NemotronSuperProjectionNorm(width: inputWidth)
        self.linear1 = Linear(inputWidth, hiddenWidth, bias: false)
        self.linear2 = Linear(hiddenWidth, outputWidth, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = relu(linear1(norm(x)))
        return linear2(activated * activated)
    }
}
