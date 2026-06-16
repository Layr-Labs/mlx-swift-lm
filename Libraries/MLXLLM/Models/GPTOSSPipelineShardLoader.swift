// Copyright © 2026 Eigen Labs.
//
// GPTOSSPipelineShardLoader -- load ONLY this rank's weights into a
// GPTOSSPipelineShard. Mirrors LlamaPipelineShardLoader.
//
// Targets the PRE-QUANTIZED Q8 build (gpt-oss-20b-MXFP4-Q8). For that build the
// model's `sanitize` early-returns (weights already have `gate_proj.weight`), so
// no MoE-unpacking / gate_up split is needed — we just filter to this rank's
// keys, remap the global layer index to local, and quantize the modules whose
// weights carry `.scales`. (The raw MXFP4 build, which dequantizes experts to
// bf16 at load, is intentionally NOT supported here — it does not fit a small
// cluster.)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum GPTOSSPipelineShardLoaderError: Error, CustomStringConvertible {
    case noSafetensors(URL)
    case missingOwnedWeights(role: String)
    case unsupportedUnquantizedBuild

    public var description: String {
        switch self {
        case .noSafetensors(let url): return "no .safetensors files in \(url.path)"
        case .missingOwnedWeights(let role): return "shard missing expected \(role) weights"
        case .unsupportedUnquantizedBuild:
            return "this GPT-OSS build stores packed MXFP4 experts (dequantize to bf16 at load); "
                + "use the pre-quantized Q8 build (gpt-oss-20b-MXFP4-Q8) for clustering"
        }
    }
}

public enum GPTOSSPipelineShardLoader {

    /// Parse config.json and load a shard for the global layer interval
    /// [start, end). Returns the shard + total layer count.
    public static func loadFromDirectory(
        _ directory: URL, start: Int, end: Int
    ) throws -> (shard: GPTOSSPipelineShard, totalLayers: Int) {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(GPTOSSConfiguration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let range = LlamaShardRange(start: start, end: end, totalLayers: config.hiddenLayers)
        let shard = try load(directory: directory, config: config, range: range,
                             perLayerQuantization: base.perLayerQuantization)
        return (shard, config.hiddenLayers)
    }

    public static func load(
        directory: URL,
        config: GPTOSSConfiguration,
        range: LlamaShardRange,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) throws -> GPTOSSPipelineShard {
        let shard = GPTOSSPipelineShard(config, range: range)

        var shardURLs: [URL] = []
        let en = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)!
        for case let url as URL in en where url.pathExtension == "safetensors" { shardURLs.append(url) }
        guard !shardURLs.isEmpty else { throw GPTOSSPipelineShardLoaderError.noSafetensors(directory) }
        shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

        // Read all keys; if the build is the raw packed MXFP4 (has `_blocks`
        // tensors and no `gate_proj.weight`), refuse — it dequantizes to bf16
        // and won't fit a cluster.
        var owned = [String: MLXArray]()
        var sawPacked = false
        var sawUnpacked = false
        for url in shardURLs {
            let (weights, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in weights {
                if key.contains("gate_up_proj_blocks") { sawPacked = true }
                if key.contains("gate_proj.weight") { sawUnpacked = true }
                guard let localKey = remap(key, range: range) else { continue }
                owned[localKey] = value
            }
        }
        if sawPacked && !sawUnpacked {
            throw GPTOSSPipelineShardLoaderError.unsupportedUnquantizedBuild
        }
        if !owned.isEmpty { eval(Array(owned.values)) }
        guard !owned.isEmpty else {
            throw GPTOSSPipelineShardLoaderError.missingOwnedWeights(role: roleName(range))
        }

        // Quantize modules with packed weights (`.scales` present), per-layer
        // config with default fallback — same as the Llama loader.
        if let perLayerQuantization {
            quantize(model: shard) { path, _ in
                guard owned["\(path).scales"] != nil else { return nil }
                return perLayerQuantization.quantization(layer: path)?.asTuple
            }
        }

        let parameters = ModuleParameters.unflattened(owned)
        try shard.update(parameters: parameters, verify: [.all])
        eval(shard)
        return shard
    }

    /// Map a global checkpoint key to this shard's local key, or nil to drop it.
    /// Same structure as the Llama remap (embed/norm/lm_head + layer reindex);
    /// GPT-OSS's MoE expert keys live UNDER `model.layers.{i}.mlp.*`, so they
    /// reindex with the same rule automatically.
    static func remap(_ key: String, range: LlamaShardRange) -> String? {
        if key.hasPrefix("model.embed_tokens.") {
            return range.isHead ? String(key.dropFirst("model.".count)) : nil
        }
        if key.hasPrefix("model.norm.") {
            return range.isTail ? String(key.dropFirst("model.".count)) : nil
        }
        if key.hasPrefix("lm_head.") {
            return range.isTail ? key : nil
        }
        if key.hasPrefix("model.layers.") {
            let rest = key.dropFirst("model.layers.".count)
            guard let dot = rest.firstIndex(of: "."),
                  let g = Int(rest[rest.startIndex..<dot]) else { return nil }
            guard g >= range.start && g < range.end else { return nil }
            return "layers.\(g - range.start)\(rest[dot...])"
        }
        return nil
    }

    private static func roleName(_ r: LlamaShardRange) -> String {
        r.isHead ? "head" : r.isTail ? "tail" : "middle"
    }
}
