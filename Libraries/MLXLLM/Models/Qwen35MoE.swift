//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen35Configuration: Codable, Sendable {
    public var modelType: String
    public var textConfig: Qwen35TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)

        if let textConfig = try container.decodeIfPresent(
            Qwen35TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            self.textConfig = try Qwen35TextConfiguration(from: decoder)
        }
    }

    public var cbv2LayerKinds: [CBv2LayerKind] { textConfig.cbv2LayerKinds }

    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        textConfig.cbv2RecurrentStateSpec(activationDType: activationDType)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { textConfig.cbv2Capabilities }
}

/// Fuse split routed-expert projections into the fused `gate_up_proj` layout
/// that `SwitchGLU(fuseGateUp: true)` expects (one gather_qmm serving
/// gate+up).
///
/// Two checkpoint families are handled, prefix-agnostically:
/// - Raw HF exports carry one stacked tensor `<mlp>.experts.gate_up_proj`
///   (`[E, 2*ffn, hidden]`, gate rows first — exactly the order
///   `SwitchGLU` slices) plus `<mlp>.experts.down_proj`; both map directly
///   onto `<mlp>.switch_mlp.{gate_up_proj,down_proj}.weight` without the
///   historical split.
/// - MLX-converted checkpoints (including quantized ones, e.g. the
///   production W4/g64 artifact) carry split `<mlp>.switch_mlp.gate_proj.*`
///   and `up_proj.*` tensors. Concatenating along the output-row axis is
///   exact for any per-output-row grouped quantization (affine or mxfp8):
///   rows quantize independently, so `weight`, `scales`, and `biases` all
///   concatenate on axis -2 and an unquantized `bias` on axis -1. The fused
///   module path resolves its quantization through
///   `qwen35GateUpQuantizationAliases`: explicit per-layer overrides written
///   against the split paths keep applying to the fused projection, and
///   default-quantized checkpoints resolve the default as before.
///
/// `mtp.*` keys are never touched: the inline MTP head keeps split modules
/// (`Qwen35MTPDecoderLayer` passes `fuseGateUp: false`) because its
/// quantization table (`mtplx_mtp_quantization`) is keyed on the split
/// module paths.
///
/// A split checkpoint missing one half of a gate/up pair fails loudly here
/// rather than surfacing as an opaque strict-update error.
public func qwen35FuseSwitchMLPGateUp(weights: [String: MLXArray]) -> [String: MLXArray] {
    var weights = weights

    // Raw HF stacked exports: already fused, just re-keyed.
    let rawSuffix = ".experts.gate_up_proj"
    for key in Array(weights.keys)
    where key.hasSuffix(rawSuffix) && !key.contains("mtp.") {
        guard let gateUp = weights.removeValue(forKey: key) else { continue }
        let prefix = String(key.dropLast(rawSuffix.count))
        weights["\(prefix).switch_mlp.gate_up_proj.weight"] = gateUp
        if let downProj = weights.removeValue(forKey: "\(prefix).experts.down_proj") {
            weights["\(prefix).switch_mlp.down_proj.weight"] = downProj
        }
    }

    // MLX split checkpoints (float or quantized): concatenate gate then up.
    let splitMarker = ".switch_mlp.gate_proj."
    for key in Array(weights.keys)
    where key.contains(splitMarker) && !key.contains("mtp.") {
        guard let gate = weights.removeValue(forKey: key) else { continue }
        let range = key.range(of: splitMarker)!
        let base = String(key[..<range.lowerBound]) + ".switch_mlp."
        let suffix = String(key[range.upperBound...])  // weight | scales | biases | bias
        let upKey = "\(base)up_proj.\(suffix)"
        guard let up = weights.removeValue(forKey: upKey) else {
            preconditionFailure(
                "qwen35FuseSwitchMLPGateUp: \(key) has no matching \(upKey); "
                    + "refusing to load a half-split expert projection")
        }
        // Quantized tensors are [E, rows, packed-cols]; the row axis is -2.
        // An unquantized per-row bias is [E, rows]; its row axis is -1.
        let axis = suffix == "bias" ? -1 : -2
        weights["\(base)gate_up_proj.\(suffix)"] = concatenated([gate, up], axis: axis)
    }

    return weights
}

/// Per-layer quantization aliases for the fused routed-expert projection.
///
/// Mixed-precision configs are written against the checkpoint layout, so
/// their explicit overrides name the split `…switch_mlp.gate_proj` /
/// `…switch_mlp.up_proj` module paths that `qwen35FuseSwitchMLPGateUp`
/// erases. `loadWeights` consults these aliases whenever the fused path has
/// no entry of its own. Gate is listed first; the sanitizer's concatenation
/// has already forced both halves onto identical packed shapes, so whichever
/// half carries an entry describes the fused tensor.
///
/// `mtp.*` paths return no aliases: the inline MTP head keeps split modules.
public func qwen35GateUpQuantizationAliases(for path: String) -> [String] {
    let fusedSuffix = ".gate_up_proj"
    guard path.hasSuffix(fusedSuffix), !path.contains("mtp.") else { return [] }
    let base = String(path.dropLast(fusedSuffix.count))
    return ["\(base).gate_proj", "\(base).up_proj"]
}

extension Qwen35TextModel: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

// Covers `Qwen35MoEModel` through inheritance.
extension Qwen35Model: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

public class Qwen35MoEModel: Qwen35Model {

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }
            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            newWeights[key] = value
        }

        newWeights = qwen35FuseSwitchMLPGateUp(weights: newWeights)

        return languageModel.sanitize(weights: newWeights)
    }
}
