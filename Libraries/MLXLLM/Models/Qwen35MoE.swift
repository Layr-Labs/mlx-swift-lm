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
/// Fusing is a PER-LAYER decision: it is only representable when both
/// halves resolve to one quantization policy, because a single quantized
/// projection has one bits/group_size/mode. For every pair whose halves
/// differ — in resolved policy (`perLayerQuantization`, explicit entries
/// first, default fallback), in carried tensor sets (e.g. affine `biases`
/// vs mxfp without), or in packed shapes/dtypes (bits or group_size
/// mismatch, caught even without a policy table) — the split tensors are
/// kept verbatim, the reason is logged, and `unfuse` is invoked with the
/// `<mlp>.switch_mlp` path so the caller can swap that layer's `SwitchGLU`
/// for its split twin (`qwen35UnfuseSwitchGLU`). Split loading is bit-exact:
/// the tensors are untouched and each half quantizes through its own
/// explicit table entry.
///
/// `mtp.*` keys are never touched: the inline MTP head keeps split modules
/// (`Qwen35MTPDecoderLayer` passes `fuseGateUp: false`) because its
/// quantization table (`mtplx_mtp_quantization`) is keyed on the split
/// module paths.
///
/// A split checkpoint missing one half of a gate/up pair (corrupt, partial
/// download, bad conversion) is left untouched — with a warning naming the
/// layer — so the strict update reports a catchable `UpdateError` (the
/// orphaned keys are unhandled and the fused parameters stay unset), exactly
/// like other malformed checkpoints. `loadWeights` stays a throwing API;
/// nothing here terminates the host process.
public func qwen35FuseSwitchMLPGateUp(
    weights: [String: MLXArray],
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    unfuse: ((String) -> Void)? = nil
) -> [String: MLXArray] {
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

    // MLX split checkpoints (float or quantized): concatenate gate then up,
    // one gate/up pair at a time so the fuse decision is per layer.
    let splitMarker = ".switch_mlp.gate_proj."
    var bases = Set<String>()  // "<mlp path>.switch_mlp."
    for key in weights.keys where key.contains(splitMarker) && !key.contains("mtp.") {
        let range = key.range(of: splitMarker)!
        bases.insert(String(key[..<range.lowerBound]) + ".switch_mlp.")
    }

    for base in bases.sorted() {
        func suffixes(_ half: String) -> Set<String> {
            let prefix = "\(base)\(half)."
            return Set(
                weights.keys.compactMap {
                    $0.hasPrefix(prefix) ? String($0.dropFirst(prefix.count)) : nil
                })
        }
        let gateSuffixes = suffixes("gate_proj")
        let upSuffixes = suffixes("up_proj")
        guard !upSuffixes.isEmpty else {
            // Corrupt/partial checkpoint. Leave the orphaned tensors for the
            // strict update to reject with a catchable error naming them —
            // trapping here would kill the host process from inside the
            // throwing loadWeights API.
            print(
                "[WARNING] qwen35FuseSwitchMLPGateUp: \(base)gate_proj.* has no "
                    + "matching \(base)up_proj.*; leaving the half-split expert "
                    + "projection for strict verification to reject")
            continue
        }

        if let reason = qwen35GateUpFuseBlocker(
            base: base, gateSuffixes: gateSuffixes, upSuffixes: upSuffixes,
            weights: weights, perLayerQuantization: perLayerQuantization)
        {
            let modulePath = String(base.dropLast())
            print("[INFO] qwen35FuseSwitchMLPGateUp: keeping \(modulePath) split — \(reason)")
            unfuse?(modulePath)
            continue
        }

        for suffix in gateSuffixes.sorted() {
            guard let gate = weights.removeValue(forKey: "\(base)gate_proj.\(suffix)"),
                let up = weights.removeValue(forKey: "\(base)up_proj.\(suffix)")
            else { continue }
            // Quantized tensors are [E, rows, packed-cols]; the row axis is -2.
            // An unquantized per-row bias is [E, rows]; its row axis is -1.
            let axis = suffix == "bias" ? -1 : -2
            weights["\(base)gate_up_proj.\(suffix)"] = concatenated([gate, up], axis: axis)
        }
    }

    return weights
}

/// The reason the gate/up pair rooted at `base` must NOT be fused, or `nil`
/// when row-concatenation is exact.
private func qwen35GateUpFuseBlocker(
    base: String,
    gateSuffixes: Set<String>,
    upSuffixes: Set<String>,
    weights: [String: MLXArray],
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) -> String? {
    // Policy check: a fused projection has a single bits/group_size/mode, so
    // both halves must resolve to the same policy. This is the only check
    // able to catch same-shape mismatches (e.g. differing modes).
    if let table = perLayerQuantization {
        let gate = qwen35ResolvedQuantization(for: "\(base)gate_proj", in: table)
        let up = qwen35ResolvedQuantization(for: "\(base)up_proj", in: table)
        let samePolicy: Bool
        switch (gate, up) {
        case (nil, nil):
            samePolicy = true
        case (let gate?, let up?):
            samePolicy =
                gate.groupSize == up.groupSize && gate.bits == up.bits
                && gate.mode == up.mode
        default:
            samePolicy = false
        }
        if !samePolicy {
            func describe(_ quantization: BaseConfiguration.Quantization?) -> String {
                quantization.map { "\($0.bits)-bit/g\($0.groupSize)/\($0.mode)" }
                    ?? "unquantized"
            }
            return "gate resolves to \(describe(gate)) but up to \(describe(up))"
        }
    }

    // Tensor backstop: catches bits/group_size mismatches even without a
    // policy table (packed columns and scale columns depend on both).
    if gateSuffixes != upSuffixes {
        return "halves carry different tensor sets "
            + "(gate \(gateSuffixes.sorted()) vs up \(upSuffixes.sorted()))"
    }
    for suffix in gateSuffixes.sorted() {
        guard let gate = weights["\(base)gate_proj.\(suffix)"],
            let up = weights["\(base)up_proj.\(suffix)"]
        else { continue }
        if gate.shape != up.shape || gate.dtype != up.dtype {
            return "\(suffix) tensors differ: gate \(gate.shape)/\(gate.dtype) "
                + "vs up \(up.shape)/\(up.dtype)"
        }
    }
    return nil
}

/// Config-table path candidates for one checkpoint module path, bridging the
/// key spaces the sanitizers operate in: `language_model.model.*` (wrapper
/// module tree), `model.language_model.*` (raw HF VLM exports), and
/// `model.*` (text-model checkpoints).
private func qwen35PolicyPathCandidates(for path: String) -> [String] {
    var candidates = [path]
    if path.hasPrefix("language_model.model.") {
        candidates.append(
            "model.language_model." + path.dropFirst("language_model.model.".count))
    }
    if path.hasPrefix("model.language_model.") {
        candidates.append(
            "language_model.model." + path.dropFirst("model.language_model.".count))
    }
    if path.hasPrefix("language_model.") {
        candidates.append(String(path.dropFirst("language_model.".count)))
    }
    return candidates
}

/// Effective quantization policy for one module path: the first explicit
/// per-layer entry among the path's key-space candidates, else the table's
/// default. `nil` means the module loads unquantized (explicit skip, or no
/// entry and no default).
private func qwen35ResolvedQuantization(
    for path: String, in table: BaseConfiguration.PerLayerQuantization
) -> BaseConfiguration.Quantization? {
    for candidate in qwen35PolicyPathCandidates(for: path) {
        if let option = table.perLayerQuantization[candidate] {
            switch option {
            case .skip: return nil
            case .quantize(let quantization): return quantization
            }
        }
    }
    return table.quantization
}

/// Swap the fused-gate_up `SwitchGLU` at `path` (checkpoint key space) for
/// its split twin, so a heterogeneous gate/up pair loads through split
/// `gate_proj`/`up_proj` modules. Intended as the `unfuse` callback of
/// `qwen35FuseSwitchMLPGateUp`. No-op when the module is absent or already
/// split. The swap goes through `Module.update(modules:)` so the module
/// cache sees the new children.
public func qwen35UnfuseSwitchGLU(at path: String, in root: Module) {
    let candidates = qwen35PolicyPathCandidates(for: path)
    for (modulePath, module) in root.namedModules() {
        guard candidates.contains(modulePath), let glu = module as? SwitchGLU else {
            continue
        }
        if glu.hasFusedGateUp {
            root.update(
                modules: ModuleChildren.unflattened([(modulePath, glu.splittingGateUp())]))
        }
        return
    }
}

/// Per-layer quantization aliases for the routed-expert projections, across
/// every checkpoint key space.
///
/// Mixed-precision configs are written against the checkpoint layout, which
/// differs from the post-sanitize module tree in two independent ways:
/// - `qwen35FuseSwitchMLPGateUp` erases the split `…switch_mlp.gate_proj` /
///   `…switch_mlp.up_proj` names when it fuses a pair, and
/// - the wrappers remap the namespace itself (raw VLM checkpoints key
///   entries on `model.language_model.*` while the module tree is
///   `language_model.model.*`; text-only checkpoints use bare `model.*`).
///
/// `loadWeights` consults these aliases whenever the module path has no
/// entry of its own, so both remappings must be bridged: a fused
/// `gate_up_proj` module aliases to its own name in the other key spaces
/// first, then to the split halves in every key space (gate first; a pair
/// is only fused when both halves resolved to one quantization policy, so
/// whichever half carries an entry describes the fused tensor). A split
/// `gate_proj`/`up_proj` module kept by the per-layer policy decision
/// aliases to its own name in the other key spaces.
///
/// `mtp.*` paths return no aliases: the inline MTP head keeps split modules
/// with its own quantization table.
public func qwen35GateUpQuantizationAliases(for path: String) -> [String] {
    guard !path.contains("mtp.") else { return [] }

    var aliases: [String] = []
    func appendCandidates(for name: String) {
        for candidate in qwen35PolicyPathCandidates(for: name)
        where candidate != path && !aliases.contains(candidate) {
            aliases.append(candidate)
        }
    }

    let fusedSuffix = ".gate_up_proj"
    if path.hasSuffix(fusedSuffix) {
        let base = String(path.dropLast(fusedSuffix.count))
        appendCandidates(for: path)
        appendCandidates(for: "\(base).gate_proj")
        appendCandidates(for: "\(base).up_proj")
    } else if path.hasSuffix(".switch_mlp.gate_proj") || path.hasSuffix(".switch_mlp.up_proj")
        || path.hasSuffix(".switch_mlp.down_proj")
    {
        appendCandidates(for: path)
    }
    return aliases
}

extension Qwen35TextModel: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

extension Qwen35TextModel: QuantizationPolicyReceiving {}

// Covers `Qwen35MoEModel` through inheritance.
extension Qwen35Model: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

// Covers `Qwen35MoEModel` through inheritance. Storage lives on the wrapped
// text model so its own sanitizer (which the wrapper's sanitize dispatches
// into) makes the identical per-layer fuse decision.
extension Qwen35Model: QuantizationPolicyReceiving {
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization? {
        get { languageModel.checkpointPerLayerQuantization }
        set { languageModel.checkpointPerLayerQuantization = newValue }
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

        newWeights = qwen35FuseSwitchMLPGateUp(
            weights: newWeights,
            perLayerQuantization: checkpointPerLayerQuantization,
            unfuse: { qwen35UnfuseSwitchGLU(at: $0, in: self) })

        return languageModel.sanitize(weights: newWeights)
    }
}
