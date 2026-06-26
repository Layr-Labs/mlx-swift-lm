// Copyright © 2026 Eigen Labs.
//
// Gemma4PipelineShardLoader -- load ONLY this rank's weights into a
// Gemma4PipelineShard. Mirrors GPTOSSPipelineShardLoader.
//
// Targets the shipped quantized builds (gemma-4-26B-A4B-it-qat-4bit and its fp8
// sibling). The experts already ship pre-split as
// `experts.switch_glu.{gate,up,down}_proj` (no fused gate_up_proj to undo), so
// we just: filter to this rank's keys (dropping the vision/audio towers),
// remap the global layer index to local, replicate embed_tokens to a tied tail,
// and quantize the modules whose weights carry `.scales` — inferring bits +
// group_size from tensor shapes (the build mixes 4-bit and 8-bit affine).
//
// Rejected builds (would force tensors other than the hidden state across the
// pipeline cut): any config with num_kv_shared_layers > 0 (cross-layer KV) or
// hidden_size_per_layer_input > 0 (per-layer-input embeddings).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum Gemma4PipelineShardLoaderError: Error, CustomStringConvertible {
    case noSafetensors(URL)
    case missingOwnedWeights(role: String)
    case unsupportedSharedKVBuild(Int)
    case unsupportedPerLayerInputBuild(Int)

    public var description: String {
        switch self {
        case .noSafetensors(let url): return "no .safetensors files in \(url.path)"
        case .missingOwnedWeights(let role): return "shard missing expected \(role) weights"
        case .unsupportedSharedKVBuild(let n):
            return "this Gemma 4 build sets num_kv_shared_layers=\(n); cross-layer KV would cross the "
                + "pipeline cut. Clustering supports only num_kv_shared_layers=0 builds "
                + "(e.g. gemma-4-26B-A4B-it-qat-4bit / -fp8)."
        case .unsupportedPerLayerInputBuild(let n):
            return "this Gemma 4 build sets hidden_size_per_layer_input=\(n); per-layer-input "
                + "embeddings would cross the pipeline cut. Clustering supports only "
                + "hidden_size_per_layer_input=0 builds."
        }
    }
}

public enum Gemma4PipelineShardLoader {

    /// Parse config.json and load a shard for the global layer interval
    /// [start, end). Returns the shard + total layer count.
    public static func loadFromDirectory(
        _ directory: URL, start: Int, end: Int
    ) throws -> (shard: Gemma4PipelineShard, totalLayers: Int) {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        // Gemma 4 nests the text config under `text_config`; Gemma4Configuration
        // handles both the nested and flat shapes.
        let textConfig = try JSONDecoder().decode(Gemma4Configuration.self, from: data).textConfig
        try assertClusterable(textConfig)
        let range = PipelineShardRange(start: start, end: end, totalLayers: textConfig.numHiddenLayers)
        let shard = try load(directory: directory, config: textConfig, range: range)
        return (shard, textConfig.numHiddenLayers)
    }

    /// TEST HELPER: load the FULL (monolithic) Gemma 4 text model from
    /// `directory`, returning it plus the layer count. Mirrors
    /// `GPTOSSPipelineShardLoader.loadFullModel` — used by the shard smoke test
    /// to get a reference forward pass against a (tiny, synthetic) checkpoint.
    public static func loadFullModel(_ directory: URL)
        throws -> (model: Gemma4TextModel, totalLayers: Int)
    {
        let configURL = directory.appending(component: "config.json")
        let data = try Data(contentsOf: configURL)
        let textConfig = try JSONDecoder.json5().decode(Gemma4Configuration.self, from: data).textConfig
        let model = Gemma4TextModel(textConfig)

        // Read the checkpoint using the SAME on-disk convention the shard loader
        // honors: the text tower is prefixed `language_model.` (real multimodal
        // Gemma 4 checkpoints), with vision/audio towers alongside. Strip the
        // `language_model.` prefix to the model's own `model.…` / `lm_head.…`
        // keyspace, drop the non-text towers, then run the model's `sanitize`.
        var shardURLs: [URL] = []
        let en = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)!
        for case let url as URL in en where url.pathExtension == "safetensors" { shardURLs.append(url) }
        guard !shardURLs.isEmpty else { throw Gemma4PipelineShardLoaderError.noSafetensors(directory) }
        shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

        var raw = [String: MLXArray]()
        for url in shardURLs {
            let (weights, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in weights {
                if key.hasPrefix("vision_tower") || key.hasPrefix("audio_tower")
                    || key.hasPrefix("multi_modal_projector") || key.hasPrefix("embed_vision")
                    || key.hasPrefix("embed_audio") { continue }
                // "language_model.model.x" -> "model.x"; "language_model.lm_head.x" -> "lm_head.x"
                let stripped = key.hasPrefix("language_model.")
                    ? String(key.dropFirst("language_model.".count)) : key
                raw[stripped] = value
            }
        }
        let sanitized = model.sanitize(weights: raw)
        if !sanitized.isEmpty { eval(Array(sanitized.values)) }

        // Quantize modules that arrived packed, using the authoritative config
        // (NOT shape inference — see the per-shard loader for why shapes are
        // ambiguous). The full model's leaf paths are `model.…`; the config keys
        // them under `language_model.model.…`.
        let quant = try Self.quantConfig(directory)
        quantize(model: model) { localPath, _ in
            guard sanitized["\(localPath).scales"] != nil,
                  sanitized["\(localPath).weight"] != nil else { return nil }
            return quant.params(forGlobalPath: "language_model." + localPath)
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)
        return (model, textConfig.numHiddenLayers)
    }

    public static func load(
        directory: URL,
        config: Gemma4TextConfiguration,
        range: PipelineShardRange
    ) throws -> Gemma4PipelineShard {
        try assertClusterable(config)
        let shard = Gemma4PipelineShard(config, range: range)

        var shardURLs: [URL] = []
        let en = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)!
        for case let url as URL in en where url.pathExtension == "safetensors" { shardURLs.append(url) }
        guard !shardURLs.isEmpty else { throw Gemma4PipelineShardLoaderError.noSafetensors(directory) }
        shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

        var owned = [String: MLXArray]()
        for url in shardURLs {
            let (weights, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in weights {
                guard let localKey = remap(key, range: range, tied: config.tieWordEmbeddings) else { continue }
                owned[localKey] = value
            }
        }
        if !owned.isEmpty { eval(Array(owned.values)) }
        guard !owned.isEmpty else {
            throw Gemma4PipelineShardLoaderError.missingOwnedWeights(role: roleName(range))
        }

        // Quantize each module that arrived with packed weights. We CANNOT infer
        // (bits, group_size) from tensor shapes alone — the packing relation
        // `bits * group_size = 32 * weightLast / scalesLast` is satisfied by
        // multiple (bits, gs) pairs (e.g. 4-bit/gs64, 8-bit/gs32 and 2-bit/gs128
        // all give the same packed shape). Gemma 4 qat mixes 4-bit and 8-bit
        // affine per module, so a shape-only guess mis-picks the wrong bit width
        // and the QuantizedLinear allocates a buffer the checkpoint can't fill.
        // Read the authoritative (bits, group_size) from config.json's
        // `quantization` block instead, per global module path.
        let quant = try Self.quantConfig(directory)
        quantize(model: shard) { localPath, _ in
            guard owned["\(localPath).scales"] != nil,
                  owned["\(localPath).weight"] != nil else { return nil }
            let globalPath = Self.localToGlobalPath(localPath, range: range)
            return quant.params(forGlobalPath: globalPath)
        }

        let parameters = ModuleParameters.unflattened(owned)
        try shard.update(parameters: parameters, verify: [.all])
        eval(shard)
        return shard
    }

    /// Authoritative per-module quantization, read from config.json's
    /// `quantization` block. A top-level default (group_size/bits/mode) plus
    /// per-module overrides keyed by the GLOBAL module path (e.g.
    /// "language_model.model.layers.0.mlp.gate_proj"). Overrides carry only
    /// bits/group_size; they inherit the top-level mode.
    struct QuantConfig {
        let defaultGroupSize: Int
        let defaultBits: Int
        let mode: QuantizationMode
        /// globalPath -> (groupSize, bits)
        let overrides: [String: (groupSize: Int, bits: Int)]

        func params(forGlobalPath path: String) -> (groupSize: Int, bits: Int, mode: QuantizationMode) {
            if let o = overrides[path] { return (o.groupSize, o.bits, mode) }
            return (defaultGroupSize, defaultBits, mode)
        }
    }

    /// Parse the `quantization` (or `quantization_config`) block. Quant lives at
    /// the TOP level of the multimodal config (alongside `text_config`), keyed by
    /// the full `language_model.model.…` global paths.
    static func quantConfig(_ directory: URL) throws -> QuantConfig {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        let q = (obj["quantization"] as? [String: Any])
            ?? (obj["quantization_config"] as? [String: Any]) ?? [:]
        let gs = (q["group_size"] as? Int) ?? 64
        let bits = (q["bits"] as? Int) ?? 4
        let modeStr = (q["mode"] as? String) ?? "affine"
        let mode = QuantizationMode(rawValue: modeStr) ?? .affine
        var overrides = [String: (groupSize: Int, bits: Int)]()
        for (k, v) in q {
            guard let m = v as? [String: Any] else { continue }   // per-module override
            let ogs = (m["group_size"] as? Int) ?? gs
            let obits = (m["bits"] as? Int) ?? bits
            overrides[k] = (ogs, obits)
        }
        return QuantConfig(defaultGroupSize: gs, defaultBits: bits, mode: mode, overrides: overrides)
    }

    /// Map a shard-local module path to the GLOBAL config key.
    ///   "layers.{local}.…"   -> "language_model.model.layers.{local+start}.…"
    ///   "embed_tokens"/"norm" -> "language_model.model.{…}"
    static func localToGlobalPath(_ localPath: String, range: PipelineShardRange) -> String {
        let prefix = "language_model.model."
        if localPath.hasPrefix("layers.") {
            let rest = localPath.dropFirst("layers.".count)
            guard let dot = rest.firstIndex(of: "."),
                  let local = Int(rest[rest.startIndex..<dot]) else {
                return prefix + localPath
            }
            return "\(prefix)layers.\(local + range.start)\(rest[dot...])"
        }
        return prefix + localPath   // embed_tokens, norm
    }

    /// Reject builds whose architecture would require shipping tensors other
    /// than the single hidden state across the pipeline cut.
    private static func assertClusterable(_ config: Gemma4TextConfiguration) throws {
        if config.numKvSharedLayers != 0 {
            throw Gemma4PipelineShardLoaderError.unsupportedSharedKVBuild(config.numKvSharedLayers)
        }
        if config.hiddenSizePerLayerInput != 0 {
            throw Gemma4PipelineShardLoaderError.unsupportedPerLayerInputBuild(config.hiddenSizePerLayerInput)
        }
    }

    /// Map a global checkpoint key to this shard's local key, or nil to drop it.
    ///
    /// Gemma 4 weights are prefixed `language_model.model.…` (the text tower),
    /// plus `vision_tower.*` / `audio_tower.*` / `multi_modal_projector.*` /
    /// `embed_vision.*` / `embed_audio.*` which we DROP (text-only shard).
    /// The text tower mirrors the Llama layout (embed_tokens + layers + norm),
    /// with experts living UNDER `layers.{i}.experts.*` so they reindex with the
    /// same per-layer rule automatically. Gemma 4 ties embeddings (no lm_head).
    static func remap(_ key: String, range: PipelineShardRange, tied: Bool) -> String? {
        // Drop non-text towers.
        if key.hasPrefix("vision_tower") || key.hasPrefix("audio_tower")
            || key.hasPrefix("multi_modal_projector") || key.hasPrefix("embed_vision")
            || key.hasPrefix("embed_audio") {
            return nil
        }

        let prefix = "language_model.model."
        guard key.hasPrefix(prefix) else { return nil }
        let rest = key.dropFirst(prefix.count)   // e.g. "layers.5.self_attn.q_proj.weight"

        if rest.hasPrefix("embed_tokens.") {
            // Head always needs embed; a tied tail also needs a copy for the
            // output projection.
            return (range.isHead || (range.isTail && tied)) ? String(rest) : nil
        }
        if rest.hasPrefix("norm.") {
            return range.isTail ? String(rest) : nil
        }
        if rest.hasPrefix("layers.") {
            let after = rest.dropFirst("layers.".count)
            guard let dot = after.firstIndex(of: "."),
                  let g = Int(after[after.startIndex..<dot]) else { return nil }
            guard g >= range.start && g < range.end else { return nil }
            return "layers.\(g - range.start)\(after[dot...])"
        }
        return nil
    }

    private static func roleName(_ r: PipelineShardRange) -> String {
        r.isHead ? "head" : r.isTail ? "tail" : "middle"
    }
}
