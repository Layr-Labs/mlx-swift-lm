// Copyright © 2026 Eigen Labs Inc.
import Foundation
import XCTest

@testable import MLXDecisions

final class LayaConfigurationTests: XCTestCase {
    private func encoder() throws -> LayaEncoderConfiguration {
        try JSONDecoder().decode(
            LayaEncoderConfiguration.self,
            from: Data(
                #"{"model_type":"modernbert","vocab_size":32,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,"norm_eps":0.00001,"norm_bias":false,"attention_bias":false,"mlp_bias":false,"hidden_activation":"gelu","local_attention":128,"global_attn_every_n_layers":3,"max_position_embeddings":512}"#
                    .utf8))
    }

    func testRejectsUnsupportedEncoderLayoutsAndNonFiniteConstants() throws {
        let valid = try encoder()
        XCTAssertNoThrow(try valid.validate())
        for mutate: (inout LayaEncoderConfiguration) -> Void in [
            { $0.num_attention_heads = 0 }, { $0.hidden_size = 65 },
            { $0.global_attn_every_n_layers = 0 }, { $0.norm_eps = .infinity },
            { $0.layer_types = ["full_attention"] }, { $0.hidden_activation = "relu" },
            {
                $0.rope_parameters = [
                    "full_attention": .init(rope_type: "scaled", rope_theta: 10000)
                ]
            },
            { $0.global_rope_theta = -1 }, { $0.local_rope_theta = .infinity },
        ] {
            var broken = valid
            mutate(&broken)
            XCTAssertThrowsError(try broken.validate())
        }
    }

    func testRejectsInvalidContextAndCalibrationConfiguration() throws {
        let encoder = try encoder()
        let valid = LayaAgentConfiguration(
            encoder: "modernbert", head_layers: 2,
            max_len: 512, head_max_len: 192, act_costs: ["escalate": 0.5],
            temperature: [1, 1, 1], temperature_by_options: [:])
        XCTAssertNoThrow(try valid.validate(encoder: encoder))
        for mutate: (inout LayaAgentConfiguration) -> Void in [
            { $0.max_len = 513 }, { $0.head_max_len = 512 }, { $0.head_layers = 0 },
            { $0.temperature = [1] }, { $0.temperature = [1, 0, 1] },
            { $0.temperature_by_options = ["choice:2": .nan] },
        ] {
            var broken = valid
            mutate(&broken)
            XCTAssertThrowsError(try broken.validate(encoder: encoder))
        }
    }
}
