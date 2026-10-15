import Foundation
@testable import MLXVLM
import Testing

extension UnitTests {
    struct NemotronSuperImageConfigurationTests {
        private let fixture = """
        {"model_type":"nemotron_h_omni","architectures":["NemotronH_Omni_Reasoning_V3"],
         "ps_version":"v2","img_context_token":"<image>","sound_config":null,
         "downsample_ratio":0.5,"vit_hidden_size":1280,"projector_hidden_size":20480,
         "patch_size":16,"img_context_token_id":18,
         "vision_config":{"model_type":"radio","dtype":"bfloat16","hidden_act":"gelu","use_swiglu_ffn":false,
          "qkv_bias":true,"hidden_size":1280,"num_hidden_layers":32,"num_attention_heads":16,
          "num_channels":3,"patch_size":16,"max_img_size":2048,"num_cls_tokens":3,
          "num_registers":7,"mlp_ratio":4,"layer_norm_eps":0.000001,"layerscale_value":1,
          "attention_probs_dropout_prob":0,"hidden_dropout_prob":0,"drop_path_rate":0},
         "llm_config":{"model_type":"nemotron_h","hidden_size":4096,"num_nextn_predict_layers":1,
          "moe_latent_size":1024,"time_step_limit":[0,{"__float__":"Infinity"}]}}
        """

        @Test func preservesUnimplementedLanguageSemantics() throws {
            let config = try NemotronSuperImageConfiguration(data: Data(fixture.utf8))
            let language = try #require(JSONSerialization.jsonObject(with: config.languageConfigurationJSON) as? [String: Any])
            #expect(language["moe_latent_size"] as? Int == 1024)
            let limits = try #require(language["time_step_limit"] as? [Any])
            #expect((limits[1] as? [String: String])?["__float__"] == "Infinity")
            #expect(config.imageTokenID == 18)
        }

        @Test func incompatibleVisionDoesNotSilentlyDefault() {
            for altered in [fixture.replacingOccurrences(of: "\"v2\"", with: "\"v1\""),
                fixture.replacingOccurrences(of: "\"hidden_size\":1280", with: "\"hidden_size\":1024"),
                fixture.replacingOccurrences(of: "\"num_nextn_predict_layers\":1", with: "\"num_nextn_predict_layers\":0")] {
                #expect(throws: (any Error).self) {
                    try NemotronSuperImageConfiguration(data: Data(altered.utf8))
                }
            }
        }
    }
}
