import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Gemma and GPTOSS elastic-window model parity", .tags(.integration), .serialized)
struct CBv2ElasticWindowModelParityTests {
    private func caches(
        kinds: [CBv2LayerKind], elastic: Bool
    ) -> [CBv2LayerCache] {
        kinds.enumerated().map { index, kind in
            let rows: [CBv2SequenceKV] =
                kind.sharesKVWithLayer == nil
                ? (0 ..< 2).map { _ in
                    switch kind.attention {
                    case .full:
                        return CBv2FullSequenceKV(
                            promptLength: 85, maxLength: 128, kvHeads: kind.kvHeads,
                            headDim: kind.headDim, valueHeadDim: kind.valueHeadDim)
                    case .slidingWindow(let window):
                        return CBv2WindowedSequenceKV(
                            window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                            valueHeadDim: kind.valueHeadDim, elasticStorage: elastic)
                    }
                } : []
            return CBv2LayerCache(layerIndex: index, kind: kind, rows: rows)
        }
    }

    private func check(
        model: any CBv2SteppableModel, kinds: [CBv2LayerKind]
    ) {
        let fixed = caches(kinds: kinds, elastic: false)
        let elastic = caches(kinds: kinds, elastic: true)
        var offset = 0
        // Cross several growth boundaries, then use a chunk wider than W.
        // Both paths receive identical tokens; compare complete logits and
        // both tensor snapshots after every prompt chunk and decode token.
        for count in [3, 2, 10, 20, 50] + Array(repeating: 1, count: 8) {
            let tokens = MLXArray(
                (0 ..< 2 * count).map {
                    Int32((offset + $0 * 7) % 64)
                }
            ).reshaped([2, count])
            let expected = model.forward(tokens: tokens, caches: fixed)
            let actual = model.forward(tokens: tokens, caches: elastic)
            eval(expected, actual)
            eval(fixed.flatMap { $0.innerState() } + elastic.flatMap { $0.innerState() })
            #expect(arrayEqual(actual, expected).item(Bool.self))
            for (a, b) in zip(fixed, elastic) {
                for (ar, br) in zip(a.rows, b.rows) {
                    let av = ar.snapshot()
                    let bv = br.snapshot()
                    #expect(av.offset == bv.offset)
                    #expect(arrayEqual(av.keys, bv.keys).item(Bool.self))
                    #expect(arrayEqual(av.values, bv.values).item(Bool.self))
                }
            }
            offset += count
        }
    }

    @Test func gemmaBF16AcrossGrowthWrapAndDecode() throws {
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(
                """
                {"model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":2,
                 "intermediate_size":64,"num_attention_heads":2,"head_dim":16,
                 "global_head_dim":32,"num_key_value_heads":2,"num_global_key_value_heads":1,
                 "num_kv_shared_layers":0,"attention_k_eq_v":true,"sliding_window":33,
                 "layer_types":["sliding_attention","full_attention"],
                 "final_logit_softcapping":30.0,"hidden_size_per_layer_input":0,
                 "use_double_wide_mlp":false,"tie_word_embeddings":true,
                 "vocab_size":64,"vocab_size_per_layer_input":64,"rms_norm_eps":0.000001}
                """.utf8))
        let model = Gemma4TextModel(config)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { name, value in
                    let transformed =
                        name.hasSuffix("self_attn.k_norm.weight")
                        ? cos(MLXArray(0 ..< value.size).asType(.float32) * 0.17) * 0.7
                        : value
                    return (name, transformed.asType(.bfloat16))
                }))
        eval(model)
        check(model: CBv2SteppableLanguageModelAdapter(model), kinds: config.cbv2LayerKinds)
    }

    @Test func gptOSSWithTrainedSinksAcrossGrowthWrapAndDecode() throws {
        let config = try JSONDecoder().decode(
            GPTOSSConfiguration.self,
            from: Data(
                """
                {"model_type":"gpt_oss","num_hidden_layers":4,"num_local_experts":4,
                 "num_experts_per_tok":2,"vocab_size":64,"rms_norm_eps":0.00001,
                 "hidden_size":32,"intermediate_size":32,"head_dim":8,
                 "num_attention_heads":4,"num_key_value_heads":2,"sliding_window":33}
                """.utf8))
        let model = GPTOSSModel(config)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { name, value in
                    (
                        name,
                        name.hasSuffix("sinks")
                            ? MLXArray.full(value.shape, values: MLXArray(Float(0.4))) : value
                    )
                }))
        eval(model)
        check(model: CBv2SteppableLanguageModelAdapter(model), kinds: config.cbv2LayerKinds)
    }

    @Test func gemmaSharedLayersBorrowExactGrowthAndPreEvictionViews() throws {
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(
                """
                {"model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":4,
                 "intermediate_size":64,"num_attention_heads":2,"head_dim":16,
                 "global_head_dim":32,"num_key_value_heads":2,"num_global_key_value_heads":1,
                 "num_kv_shared_layers":2,"attention_k_eq_v":true,"sliding_window":33,
                 "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
                 "final_logit_softcapping":30.0,"hidden_size_per_layer_input":0,
                 "use_double_wide_mlp":false,"tie_word_embeddings":true,
                 "vocab_size":64,"vocab_size_per_layer_input":64,"rms_norm_eps":0.000001}
                """.utf8))
        let model = Gemma4TextModel(config)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { name, value in
                    let transformed =
                        name.hasSuffix("self_attn.k_norm.weight")
                        ? cos(MLXArray(0 ..< value.size).asType(.float32) * 0.23) * 0.9 : value
                    return (name, transformed.asType(.bfloat16))
                }))
        eval(model)
        #expect(config.cbv2LayerKinds[2].sharesKVWithLayer == 0)
        #expect(config.cbv2LayerKinds[3].sharesKVWithLayer == 1)
        check(model: CBv2SteppableLanguageModelAdapter(model), kinds: config.cbv2LayerKinds)
    }
}
