import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

private func qwenInlineMTPConfig(blockSize: Int = 3) -> Data {
    Data(
        """
        {
          "model_type": "qwen3_5_moe",
          "mtplx_mtp": {
            "included": true,
            "prefix": "mtp.",
            "block_size": \(blockSize)
          },
          "mtplx_mtp_quantization": {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
            "layers.0.self_attn.q_proj": {
              "group_size": 32,
              "bits": 8,
              "mode": "mxfp8"
            }
          },
          "text_config": {
            "model_type": "qwen3_5_moe",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "linear_num_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "head_dim": 8,
            "full_attention_interval": 4,
            "num_experts": 0,
            "num_experts_per_tok": 0,
            "mtp_num_hidden_layers": 1
          }
        }
        """.utf8)
}

private func withInlineMTPDirectory(
    blockSize: Int = 3,
    weights: [String: MLXArray] = ["mtp.unexpected.weight": MLXArray([Float(1)])],
    _ body: (URL) throws -> Void
) throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen-inline-mtp-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try qwenInlineMTPConfig(blockSize: blockSize)
        .write(to: directory.appendingPathComponent("config.json"))
    let shardName = "model-00001-of-00001.safetensors"
    try save(arrays: weights, url: directory.appendingPathComponent(shardName))
    let index = ["weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shardName) })]
    try JSONSerialization.data(withJSONObject: index)
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    try body(directory)
}

@Suite("Qwen inline MTP loader")
struct Qwen35InlineMTPLoaderTests {
    @Test("loader is artifact-scoped and does not mutate the legacy process flag")
    func artifactScopedFlag() throws {
        _qwen35MTPEnabled = false
        try withInlineMTPDirectory { directory in
            let config = try JSONDecoder.json5().decode(
                Qwen35TextConfiguration.self,
                from: try JSONSerialization.data(
                    withJSONObject: (try JSONSerialization.jsonObject(
                        with: qwenInlineMTPConfig()) as! [String: Any])["text_config"]!))
            let target = Qwen35TextModel(config)
            #expect(throws: (any Error).self) {
                _ = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
            }
            #expect(_qwen35MTPEnabled == false)
        }
    }

    @Test("block size is bounded before weight loading")
    func boundedBlockSize() throws {
        try withInlineMTPDirectory(blockSize: 9) { directory in
            let config = try JSONDecoder.json5().decode(
                Qwen35TextConfiguration.self,
                from: try JSONSerialization.data(
                    withJSONObject: (try JSONSerialization.jsonObject(
                        with: qwenInlineMTPConfig()) as! [String: Any])["text_config"]!))
            let target = Qwen35TextModel(config)
            #expect(throws: Qwen35InlineMTPError.self) {
                _ = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
            }
        }
    }
}
