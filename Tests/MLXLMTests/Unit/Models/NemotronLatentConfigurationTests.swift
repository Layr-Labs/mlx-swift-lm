import Foundation
import Testing
@testable import MLXLLM

extension UnitTests {
    struct NemotronLatentConfigurationTests {
        private func decode(_ latent: Any?, limits: Any? = nil) throws -> NemotronHConfiguration {
            var fields: [String: Any] = [
                "vocab_size": 32, "hidden_size": 4, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1,
                "mamba_num_heads": 1, "mamba_head_dim": 4, "ssm_state_size": 4,
                "conv_kernel": 4, "n_groups": 1, "intermediate_size": 3,
                "moe_intermediate_size": 3, "moe_shared_expert_intermediate_size": 3,
                "n_routed_experts": 2, "num_experts_per_tok": 2,
                "hybrid_override_pattern": "E", "time_step_limit_max": 100,
            ]
            if let latent { fields["moe_latent_size"] = latent }
            if let limits { fields["time_step_limit"] = limits }
            return try JSONDecoder().decode(NemotronHConfiguration.self,
                from: JSONSerialization.data(withJSONObject: fields))
        }
        @Test func optionalLatentDimensionDecodesAndRoundTrips() throws {
            #expect(try decode(nil).moeLatentSize == nil)
            #expect(try decode(NSNull()).moeLatentSize == nil)
            let configuration = try decode(2)
            #expect(configuration.moeLatentSize == 2)
            let encoded = try JSONEncoder().encode(configuration)
            #expect(try JSONDecoder().decode(NemotronHConfiguration.self,
                from: encoded).moeLatentSize == 2)
        }
        @Test func taggedInfinityExecutionLimitIsStrictlyDecoded() throws {
            let configured = try decode(2, limits: [0, ["__float__": "Infinity"]])
            #expect(configured.timeStepLimitMin == 0 && configured.timeStepLimitMax == .infinity)
            for tag in [["__float__": "NaN"], ["__float__": "Infinity", "extra": "x"]] {
                #expect(throws: (any Error).self) { try decode(2, limits: [0, tag]) }
            }
            #expect(throws: (any Error).self) { try decode(2, limits: [["__float__": "Infinity"], 1]) }
        }
        @Test func rejectsInvalidLatentDimension() {
            for invalid in [0, -1] {
                #expect(throws: (any Error).self) { try decode(invalid) }
            }
            #expect(throws: (any Error).self) { try decode("2") }
        }
    }
}
