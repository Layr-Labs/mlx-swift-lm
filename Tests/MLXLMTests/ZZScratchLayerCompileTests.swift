import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class ZZScratchLayerCompileTests: XCTestCase {
    func testLegacyGreedyTokensAndTiming() throws {
        let env = ProcessInfo.processInfo.environment
        var json = Qwen4ExpFixture.configurationJSON
        for (a, b) in [
            ("\"hidden_size\": 32", "\"hidden_size\": 128"),
            ("\"num_attention_heads\": 4", "\"num_attention_heads\": 24"),
            ("\"head_dim\": 8", "\"head_dim\": 256"),
            ("\"indexer_budget\": \(Qwen4ExpFixture.indexerBudget)", "\"indexer_budget\": 2048"),
            ("\"indexer_compress_ratio\": 2", "\"indexer_compress_ratio\": 4"),
            ("\"indexer_head_dim\": 8", "\"indexer_head_dim\": 128"),
            ("\"ple_layer_ids\": [1]", "\"ple_layer_ids\": [2]"),
            ("\"full_attention_interval\": 2", "\"full_attention_interval\": 4"),
            ("\"num_hidden_layers\": 4", "\"num_hidden_layers\": \(env["ZZ_LAYERS"] ?? "48")"),
            ("\"num_experts\": 4", "\"num_experts\": 512"),
            ("\"num_experts_per_tok\": 2", "\"num_experts_per_tok\": 10"),
            ("\"moe_intermediate_size\": 16", "\"moe_intermediate_size\": 64"),
            ("\"shared_expert_intermediate_size\": 16", "\"shared_expert_intermediate_size\": 64"),
            ("\"linear_num_key_heads\": 2", "\"linear_num_key_heads\": 16"),
            ("\"linear_num_value_heads\": 4", "\"linear_num_value_heads\": 48"),
            ("\"linear_key_head_dim\": 32", "\"linear_key_head_dim\": 128"),
            ("\"linear_value_head_dim\": 32", "\"linear_value_head_dim\": 128"),
            ("\"hc_count\": 2", "\"hc_count\": 4"),
        ] { precondition(json.contains(a), a); json = json.replacingOccurrences(of: a, with: b) }
        let config = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(7)
        let model = Qwen4ExpModel(text: config, withMTP: false)
        eval(model)
        let cast = model.parameters().flattened().map { ($0.0, $0.1.dtype == .float32 ? $0.1.asType(.bfloat16) : $0.1) }
        model.update(parameters: ModuleParameters.unflattened(cast)); eval(model)
        quantize(model: model) { _, m in
            if let l = m as? Linear, l.weight.dim(-1) % 64 == 0 { return (64, 4, .affine) }
            if let s = m as? SwitchLinear, s.weight.dim(-1) % 64 == 0 { return (64, 4, .affine) }
            return nil
        }
        eval(model)
        model.install(ngramRowSource: DeterministicNGramRowSource(rowDimensions: 2))
        let prompt: [Int] = (0 ..< 64).map { ($0 * 37 + 11) % 64 }
        let caches = model.makeCache()
        var ids = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
        var produced: [Int] = []
        var times: [Double] = []
        for step in 0 ..< 40 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let logits = model(ids, cache: caches)
            let next = argMax(logits[0..., -1, 0...], axis: -1)
            eval(next)
            if step > 0 { times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6) }
            let tok = next.item(Int.self)
            produced.append(tok)
            ids = next.reshaped([1, 1]).asType(.int32)
        }
        let steady = times.dropFirst(4).sorted()
        print("LAYERCOMPILE compile=\(qwen4ExpScratchLayerCompileEnabled) layers=\(env["ZZ_LAYERS"] ?? "48") tokens=\(produced.prefix(16)) step p50=\(String(format: "%.1f", steady[steady.count/2])) min=\(String(format: "%.1f", steady.first!)) ms")
    }
}
