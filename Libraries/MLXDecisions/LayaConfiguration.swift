// Copyright © 2026 Eigen Labs Inc.
import Foundation

public enum LayaError: Error, LocalizedError {
    case invalidJSON(String)
    case invalidRequest(String)
    case invalidCheckpoint(String)
    case nonFiniteOutput

    public var errorDescription: String? {
        switch self {
        case .invalidJSON(let message), .invalidRequest(let message),
            .invalidCheckpoint(let message):
            message
        case .nonFiniteOutput: "Laya produced non-finite outputs"
        }
    }
}

struct LayaEncoderConfiguration: Decodable {
    var vocab_size: Int
    var hidden_size: Int
    var intermediate_size: Int
    var num_hidden_layers: Int
    var num_attention_heads: Int
    var model_type: String
    var norm_eps: Float
    var norm_bias: Bool
    var attention_bias: Bool
    var mlp_bias: Bool
    var hidden_activation: String
    var local_attention: Int
    var global_attn_every_n_layers: Int
    var max_position_embeddings: Int
    var layer_types: [String]?
    var rope_parameters: [String: RopeParameters]?
    var global_rope_theta: Float?
    var local_rope_theta: Float?

    struct RopeParameters: Decodable {
        var rope_type: String?
        var rope_theta: Float?
    }

    var headDimension: Int { hidden_size / num_attention_heads }
    func isGlobal(_ index: Int) -> Bool {
        layer_types?[index] == "full_attention"
            || (layer_types == nil && index % global_attn_every_n_layers == 0)
    }
    func ropeBase(_ index: Int) -> Float {
        let global = isGlobal(index)
        return rope_parameters?[global ? "full_attention" : "sliding_attention"]?.rope_theta
            ?? (global ? global_rope_theta ?? 160000 : local_rope_theta ?? 10000)
    }
    func validate() throws {
        guard model_type == "modernbert", hidden_activation == "gelu",
            vocab_size > 0, hidden_size > 0, intermediate_size > 0,
            num_hidden_layers > 0, num_attention_heads > 0,
            hidden_size % num_attention_heads == 0, headDimension % 2 == 0,
            global_attn_every_n_layers > 0, local_attention > 0,
            norm_eps.isFinite, norm_eps > 0,
            [global_rope_theta, local_rope_theta].compactMap({ $0 }).allSatisfy({
                $0.isFinite && $0 > 0
            }),
            layer_types == nil
                || (layer_types!.count == num_hidden_layers
                    && layer_types!.allSatisfy {
                        ["full_attention", "sliding_attention"].contains($0)
                    }),
            (rope_parameters ?? [:]).values.allSatisfy({
                ($0.rope_type ?? "default") == "default"
                    && ($0.rope_theta ?? 10000).isFinite && ($0.rope_theta ?? 10000) > 0
            })
        else { throw LayaError.invalidCheckpoint("Unsupported ModernBERT configuration") }
    }
}

struct LayaAgentConfiguration: Decodable {
    var encoder: String
    var head_layers: Int
    var max_len: Int
    var head_max_len: Int
    var act_costs: [String: Double]
    var temperature: [Float]
    var temperature_by_options: [String: Float]

    func validate(encoder: LayaEncoderConfiguration) throws {
        guard head_layers > 0, head_max_len > 4, head_max_len < max_len,
            max_len <= encoder.max_position_embeddings, temperature.count == 3,
            (temperature + Array(temperature_by_options.values)).allSatisfy({
                $0.isFinite && $0 > 0
            }),
            encoder.hidden_size % max(1, encoder.hidden_size / 64) == 0
        else { throw LayaError.invalidCheckpoint("Invalid Laya decision configuration") }
    }
}
