// Copyright © 2026 Eigen Labs.
//
// LlamaPipelineShardLoader -- load ONLY this rank's weights into a
// `LlamaPipelineShard`, so a single machine never materializes the whole model.
//
// The monolithic loader (`MLXLMCommon.loadWeights`) reads every safetensors key
// and updates a full `LlamaModel`. For a pipeline shard we instead:
//   1. read all safetensors shards (lazily; MLX maps them),
//   2. KEEP only the keys this rank owns:
//        - `model.embed_tokens.*`         (head only)
//        - `model.layers.{i}.*` for i in [start,end)
//        - `model.norm.*`, `lm_head.*`    (tail only)
//   3. REMAP keys to the shard's flat module layout:
//        `model.embed_tokens.*`   -> `embed_tokens.*`
//        `model.layers.{i}.*`     -> `layers.{i-start}.*`
//        `model.norm.*`           -> `norm.*`
//        `lm_head.*`              -> `lm_head.*`     (unchanged)
//   4. quantize the owned modules whose weights carry `.scales` (4-bit Llama),
//   5. `update(parameters:)` the shard with verification.
//
// Memory: only the kept arrays are evaluated/resident; the rest are dropped
// before eval, so a rank's footprint ≈ its own layers' weights.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum LlamaPipelineShardLoaderError: Error, CustomStringConvertible {
    case noSafetensors(URL)
    case missingOwnedWeights(role: String)

    public var description: String {
        switch self {
        case .noSafetensors(let url): return "no .safetensors files found in \(url.path)"
        case .missingOwnedWeights(let role): return "shard is missing expected \(role) weights"
        }
    }
}

public enum LlamaPipelineShardLoader {

    /// Convenience: parse `config.json` in `directory` and load a shard for the
    /// global layer interval `[start, end)`. Reads quantization from config.
    /// Returns the shard and the model's total layer count.
    ///
    /// This top-level entry exists so callers OUTSIDE this module (ProviderCore)
    /// can build a shard without touching `LlamaConfiguration`'s internal fields.
    public static func loadFromDirectory(
        _ directory: URL, start: Int, end: Int
    ) throws -> (shard: LlamaPipelineShard, totalLayers: Int) {
        let configURL = directory.appending(component: "config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(LlamaConfiguration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let range = LlamaShardRange(start: start, end: end, totalLayers: config.hiddenLayers)
        let shard = try load(
            directory: directory, config: config, range: range,
            perLayerQuantization: base.perLayerQuantization)
        return (shard, config.hiddenLayers)
    }

    /// TEST HELPER: load the FULL (monolithic) Llama model from `directory`,
    /// returning it plus the layer count. Used by the shard smoke test to get a
    /// reference forward pass without touching internal config fields from
    /// outside the module.
    public static func loadFullModel(_ directory: URL) throws -> (model: LlamaModel, totalLayers: Int) {
        let configURL = directory.appending(component: "config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder.json5().decode(LlamaConfiguration.self, from: data)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let model = LlamaModel(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return (model, config.hiddenLayers)
    }

    /// Build and weight-load a shard for `range` from a HF-format model dir.
    ///
    /// - Parameters:
    ///   - directory: the model snapshot dir (contains *.safetensors + config.json).
    ///   - config: the parsed Llama configuration (full model's dims/layer count).
    ///   - range: this rank's owned layer interval.
    ///   - perLayerQuantization: the model's quantization (4-bit for the target
    ///     model); pass nil for fp16 checkpoints.
    public static func load(
        directory: URL,
        config: LlamaConfiguration,
        range: LlamaShardRange,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) throws -> LlamaPipelineShard {
        let shard = LlamaPipelineShard(config, range: range)

        // 1. Gather safetensors shard files.
        var shardURLs: [URL] = []
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            shardURLs.append(url)
        }
        guard !shardURLs.isEmpty else { throw LlamaPipelineShardLoaderError.noSafetensors(directory) }
        shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

        // 2+3. Read each file, keep+remap only this rank's keys. A tied-
        //      embedding tail also keeps embed_tokens (for output projection).
        let tiedTailNeedsEmbed = config.tieWordEmbeddings && range.isTail
        var owned = [String: MLXArray]()
        for url in shardURLs {
            let (weights, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in weights {
                guard let localKey = remap(key, range: range, tiedTailNeedsEmbed: tiedTailNeedsEmbed)
                else { continue }
                owned[localKey] = value
            }
        }
        // Force the read for just the kept arrays.
        if !owned.isEmpty { eval(Array(owned.values)) }

        guard !owned.isEmpty else {
            throw LlamaPipelineShardLoaderError.missingOwnedWeights(role: roleName(range))
        }

        // 4. Quantize owned modules that have packed weights (`.scales` present).
        //    A module is quantized iff its weights include a `.scales` entry;
        //    the bits/groupSize come from the per-layer config (with the default
        //    fallback), mirroring MLXLMCommon.loadWeights.
        if let perLayerQuantization {
            quantize(model: shard) { path, _ in
                guard owned["\(path).scales"] != nil else { return nil }
                return perLayerQuantization.quantization(layer: path)?.asTuple
            }
        }

        // 5. Apply weights to the shard.
        let parameters = ModuleParameters.unflattened(owned)
        try shard.update(parameters: parameters, verify: [.all])
        eval(shard)
        return shard
    }

    /// Map a global checkpoint key to this shard's local key, or nil to drop it.
    static func remap(_ key: String, range: LlamaShardRange, tiedTailNeedsEmbed: Bool) -> String? {
        // embed_tokens — head always; a tied tail also keeps it for projection.
        if key.hasPrefix("model.embed_tokens.") {
            return (range.isHead || tiedTailNeedsEmbed) ? String(key.dropFirst("model.".count)) : nil
        }
        // final norm — tail only. (Must check before the generic layer prefix.)
        if key.hasPrefix("model.norm.") {
            return range.isTail ? String(key.dropFirst("model.".count)) : nil
        }
        // lm_head — tail only; key already top-level.
        if key.hasPrefix("lm_head.") {
            return range.isTail ? key : nil
        }
        // transformer blocks: model.layers.{i}.* -> layers.{i-start}.*
        if key.hasPrefix("model.layers.") {
            let rest = key.dropFirst("model.layers.".count)
            guard let dot = rest.firstIndex(of: "."),
                  let globalIdx = Int(rest[rest.startIndex..<dot])
            else { return nil }
            guard globalIdx >= range.start && globalIdx < range.end else { return nil }
            let localIdx = globalIdx - range.start
            let suffix = rest[dot...]   // includes leading "."
            return "layers.\(localIdx)\(suffix)"
        }
        // Anything else (rotary inv_freq, etc.) is not an owned parameter.
        return nil
    }

    private static func roleName(_ range: LlamaShardRange) -> String {
        range.isHead ? "head (embed+layers)" : range.isTail ? "tail (layers+norm+lm_head)" : "middle (layers)"
    }
}
