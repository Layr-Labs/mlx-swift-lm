import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct LoRAConfigurationTests {

        private func decode(_ json: String) throws -> LoRAConfiguration {
            try JSONDecoder().decode(LoRAConfiguration.self, from: Data(json.utf8))
        }

        @Test
        func decodesTheTrainingOutputFormat() throws {
            let configuration = try decode(
                #"""
                {
                  "fine_tune_type": "dora",
                  "num_layers": 28,
                  "lora_parameters": {"rank": 16, "scale": 20.0, "keys": ["self_attn.q_proj"]}
                }
                """#)
            #expect(configuration.fineTuneType == .dora)
            #expect(configuration.numLayers == 28)
            #expect(configuration.loraParameters.rank == 16)
            #expect(configuration.loraParameters.scale == 20)
            #expect(configuration.loraParameters.keys == ["self_attn.q_proj"])
        }

        @Test
        func keysAreOptional() throws {
            let configuration = try decode(
                #"{"fine_tune_type": "lora", "num_layers": 4, "lora_parameters": {"rank": 2, "scale": 1.5}}"#
            )
            #expect(configuration.fineTuneType == .lora)
            #expect(configuration.loraParameters.keys == nil)
        }

        @Test
        func rejectsAnUnknownFineTuneType() {
            #expect(throws: DecodingError.self) {
                try decode(
                    #"{"fine_tune_type": "full", "num_layers": 4, "lora_parameters": {"rank": 2, "scale": 1}}"#
                )
            }
        }

        @Test
        func rejectsAMissingField() {
            #expect(throws: DecodingError.self) {
                try decode(
                    #"{"fine_tune_type": "lora", "lora_parameters": {"rank": 2, "scale": 1}}"#)
            }
        }

        @Test
        func defaultsMatchTheTrainingDefaults() {
            let configuration = LoRAConfiguration()
            #expect(configuration.numLayers == 16)
            #expect(configuration.fineTuneType == .lora)
            #expect(configuration.loraParameters.rank == 8)
            #expect(configuration.loraParameters.scale == 10)
            #expect(configuration.loraParameters.keys == nil)
        }

        @Test
        func encodingUsesTheSnakeCaseKeys() throws {
            let configuration = LoRAConfiguration(
                numLayers: 2, fineTuneType: .dora,
                loraParameters: .init(rank: 4, scale: 2, keys: ["mlp.up_proj"]))
            let data = try JSONEncoder().encode(configuration)
            let object = try #require(
                try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["num_layers"] as? Int == 2)
            #expect(object["fine_tune_type"] as? String == "dora")
            let parameters = try #require(object["lora_parameters"] as? [String: Any])
            #expect(parameters["rank"] as? Int == 4)
            #expect(parameters["keys"] as? [String] == ["mlp.up_proj"])
            let decoded = try JSONDecoder().decode(LoRAConfiguration.self, from: data)
            #expect(decoded.loraParameters.scale == 2)
        }
    }
}
