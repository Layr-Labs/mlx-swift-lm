import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// Checkpoint precision for `nativeCheckpointModelTypes`: Nemotron 3.5
    /// Lightning keeps native checkpoint rows unless a setting chooses otherwise,
    /// so serving it without its MTP head does not move a cache hit's output.
    /// Swift logic only; no arrays are created.
    @Suite("Paged KV checkpoint policy")
    struct PagedKVCheckpointPolicyTests {
        private func checkpoint(
            _ modelType: String = "nemotron_h", global: String? = nil,
            byModel: [String: String] = [:], environment: [String: String] = [:]
        ) throws -> PagedKVQuantizationSelection {
            try PagedKVQuantizationPolicy.resolve(
                modelType: modelType, global: global, byModel: byModel, modelID: "m",
                environment: environment, purpose: .checkpoint)
        }

        @Test func lightningCheckpointsDefaultToNative() throws {
            let key = PagedKVQuantizationPolicy.checkpointEnvironmentKey
            // A saved provider.toml spells out the previous default, which is not a choice.
            for unset in [nil, "", " ", "balanced", " Balanced "] as [String?] {
                #expect(try checkpoint(global: unset) == .native)
            }
            #expect(try checkpoint(environment: [key: " "]) == .native)
            // Every explicit choice still wins, in the same order.
            #expect(try checkpoint(global: "k4v4") == .balanced)
            #expect(try checkpoint(global: "k8v4") == .k8v4)
            #expect(try checkpoint(byModel: ["m": "balanced"]) == .balanced)
            #expect(try checkpoint(environment: [key: "balanced"]) == .balanced)
            #expect(try checkpoint(global: "balanced", byModel: ["m": "k8v8"]) == .k8v8)
            #expect(
                try checkpoint(byModel: ["m": "balanced"], environment: [key: "native"]) == .native)
            // Live execution keeps its own default.
            #expect(
                try PagedKVQuantizationPolicy.resolve(
                    modelType: "nemotron_h", global: "balanced", modelID: "m", environment: [:])
                    == .balanced)
            #expect(
                try PagedKVQuantizationPolicy.resolve(
                    modelType: "nemotron_h", modelID: "m", environment: [:]) == .native)
        }

        @Test(arguments: [
            "qwen3_5", "gemma4", "gpt_oss", "glm5_next", "nemotron_h_omni", "prism_hadamard_qwen35",
            "supported",
        ])
        func otherFamiliesKeepTheirCheckpointDefault(modelType: String) throws {
            #expect(try checkpoint(modelType, global: "balanced") == .balanced)
            #expect(try checkpoint(modelType) == .balanced)
        }
    }
}
