// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("Gemma4 assistant decoder glue admission and parity", .serialized)
struct Gemma4AssistantDecodeGlueTests {
    private let enabled = Gemma4DecodeGluePolicy(environment: [
        "DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE": "1",
        "DARKBLOOM_GEMMA4_DRAFTER_NORM_RESIDUAL_FUSE": "1",
    ])

    private func fixture(shared: Bool = true) throws -> Gemma4TextModelInner {
        let data = Data("""
        {"model_type":"gemma4_text","hidden_size":1024,"num_hidden_layers":4,
         "intermediate_size":256,"num_attention_heads":4,"head_dim":256,
         "global_head_dim":512,"num_key_value_heads":2,"num_global_key_value_heads":2,
         "num_kv_shared_layers":\(shared ? 4 : 0),"sliding_window":64,"attention_k_eq_v":false,
         "final_logit_softcapping":null,"tie_word_embeddings":true,"vocab_size":64,
         "vocab_size_per_layer_input":64,"rms_norm_eps":1e-6,
         "hidden_size_per_layer_input":0,"use_double_wide_mlp":false,
         "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"]}
        """.utf8)
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
        let model = Gemma4TextModelInner(config, forceSharedKV: shared)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        eval(model)
        return model
    }

    private func values(_ shape: [Int], salt: Int = 0) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0..<count).map { Float(($0 * 17 + salt) % 29 - 14) / 64 }, shape)
            .asType(.bfloat16)
    }

    @Test func assistantRoleAndBothOptInsAreRequired() throws {
        let model = try fixture()
        let x = values([1, 1, 1024])
        let kv = (values([1, 2, 16, 256]), values([1, 2, 16, 256], salt: 3))
        func admitted(_ policy: Gemma4DecodeGluePolicy) -> Bool {
            model.forwardAssistantLayer(at: 0, x, mask: .none, sharedKV: kv,
                positionOffset: .scalar(7), policy: policy) != nil
        }
        #expect(!admitted(enabled))
        model.markDecodeGlueAssistant(backboneHiddenSize: 1024)
        #expect(!admitted(enabled))
        model.markDecodeGlueAssistant(backboneHiddenSize: 2816)
        #expect(!admitted(Gemma4DecodeGluePolicy(environment: [:])))
        #expect(!admitted(Gemma4DecodeGluePolicy(environment: ["DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE": "1"])))
        #expect(!admitted(Gemma4DecodeGluePolicy(environment: ["DARKBLOOM_GEMMA4_DRAFTER_NORM_RESIDUAL_FUSE": "1"])))
        #expect(admitted(enabled))
    }

    @Test func scalarAndBatchedSharedKVLayersMatchOrdinaryBytes() throws {
        let model = try fixture()
        model.markDecodeGlueAssistant(backboneHiddenSize: 2816)
        for batch in [1, 2, 4, 8] {
            for (index, dimension) in [(0, 256), (3, 512)] {
                let x = values([batch, 1, 1024], salt: batch)
                let kv = (values([batch, 2, 16, dimension]), values([batch, 2, 16, dimension], salt: 3))
                eval(x, kv.0, kv.1)
                let before = [kv.0.asData(access: .copy).data, kv.1.asData(access: .copy).data]
                let actual = try #require(model.forwardAssistantLayer(at: index, x, mask: .none,
                    sharedKV: kv, positionOffset: .scalar(7), policy: enabled))
                let (expected, _, _) = model.layers[index](x, mask: .none, cache: nil,
                    perLayerInput: nil, sharedKV: kv, positionOffset: .scalar(7))
                eval(actual, expected)
                gemma4ExpectExactBytes(actual, expected, label: "assistant B=\(batch) D=\(dimension)")
                #expect(kv.0.asData(access: .copy).data == before[0])
                #expect(kv.1.asData(access: .copy).data == before[1])
            }
        }
    }

    @Test func unsupportedShapesDtypesAndRolesRetainOrdinaryDispatch() throws {
        let model = try fixture()
        model.markDecodeGlueAssistant(backboneHiddenSize: 2816)
        let kv = (values([1, 2, 16, 256]), values([1, 2, 16, 256], salt: 3))
        for x in [values([1, 2, 1024]), values([1, 1024]), values([1, 1, 512]),
                  values([1, 1, 1024]).asType(.float32)] {
            #expect(model.forwardAssistantLayer(at: 0, x, mask: .none, sharedKV: kv,
                positionOffset: .scalar(7), policy: enabled) == nil)
        }
        let ordinary = try fixture(shared: false)
        ordinary.markDecodeGlueAssistant(backboneHiddenSize: 2816)
        #expect(ordinary.forwardAssistantLayer(at: 0, values([1, 1, 1024]), mask: .none,
            sharedKV: kv, positionOffset: .scalar(7), policy: enabled) == nil)
        #expect(model.forwardAssistantLayer(at: -1, values([1, 1, 1024]), mask: .none,
            sharedKV: kv, positionOffset: .scalar(7), policy: enabled) == nil)
    }
}
