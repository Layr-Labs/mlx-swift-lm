import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Tests of `NemotronH35MTPAssistant.load(from:target:)` with a tiny
    /// target and a synthetic artifact folder: `config.json`,
    /// `model.safetensors.index.json` and one shard with the `mtp.*`
    /// tensors.
    @Suite
    struct NemotronH35MTPLoadingTests {

        static func target(vocabularySize: Int = 128) throws -> NemotronH35Model {
            let data = Data(
                """
                {"model_type":"nemotron_h","vocab_size":\(vocabularySize),"hidden_size":64,
                 "num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,
                 "head_dim":64,"mamba_num_heads":4,"mamba_head_dim":16,"ssm_state_size":16,
                 "conv_kernel":4,"n_groups":2,"intermediate_size":128,"moe_intermediate_size":64,
                 "moe_shared_expert_intermediate_size":64,"n_routed_experts":4,
                 "num_experts_per_tok":2,"layers_block_type":["mamba","attention"],
                 "mamba_ssm_cache_dtype":"float32","num_nextn_predict_layers":1,
                 "mtp_layers_block_type":["attention","moe"]}
                """.utf8)
            return NemotronH35Model(
                try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        }

        /// A random MTP head for `target`, as module keys.
        static func headWeights(for target: NemotronH35Model) -> [String: MLXArray] {
            let head = NemotronH35MTPAssistant(target: target)
            SyntheticModel.randomize(head.module, seed: 7)
            return SyntheticModel.flatParameters(head.module)
        }

        /// Writes an artifact folder and returns its URL. `weights` uses
        /// module keys; the files use `mtp.` keys.
        static func artifact(
            config: NemotronH35Configuration, weights: [String: MLXArray],
            shard: String = "model-00001-of-00001.safetensors",
            extraIndex: [String: String] = [:]
        ) throws -> URL {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("mtp-artifact-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try JSONEncoder().encode(config).write(to: folder.appendingPathComponent("config.json"))
            var files: [String: MLXArray] = [:]
            var map: [String: String] = [:]
            for (key, value) in weights {
                files["mtp." + key] = value
                map["mtp." + key] = shard
            }
            map.merge(extraIndex) { $1 }
            try MLX.save(arrays: files, url: folder.appendingPathComponent(shard))
            let index = try JSONSerialization.data(withJSONObject: ["weight_map": map])
            try index.write(to: folder.appendingPathComponent("model.safetensors.index.json"))
            return folder
        }

        static func loadedMatches(_ weights: [String: MLXArray], expected: [String: MLXArray])
            throws -> Bool
        {
            let target = try Self.target()
            let folder = try artifact(config: target.lightningConfiguration, weights: weights)
            defer { try? FileManager.default.removeItem(at: folder) }
            let assistant = try NemotronH35MTPAssistant.load(from: folder, target: target)
            let loaded = SyntheticModel.flatParameters(assistant.module)
            guard Set(loaded.keys) == Set(expected.keys) else { return false }
            return expected.allSatisfy {
                SyntheticModel.maxAbsDifference($0.value, loaded[$0.key]!) == 0
            }
        }

        @Test func loadsTheStackedExpertLayout() throws {
            let weights = Self.headWeights(for: try Self.target())
            #expect(weights["layers.1.mixer.switch_mlp.fc1.weight"]?.shape == [4, 64, 64])
            #expect(try Self.loadedMatches(weights, expected: weights))
        }

        /// The official BF16 checkpoints store each expert on its own as
        /// `up_proj` and `down_proj`. The loader stacks them into `fc1` and
        /// `fc2`.
        @Test func stacksPerExpertWeights() throws {
            let weights = Self.headWeights(for: try Self.target())
            var perExpert = weights
            for (stacked, single) in [("fc1", "up_proj"), ("fc2", "down_proj")] {
                let key = "layers.1.mixer.switch_mlp.\(stacked).weight"
                let value = perExpert.removeValue(forKey: key)!
                for expert in 0 ..< 4 {
                    perExpert["layers.1.mixer.experts.\(expert).\(single).weight"] = value[expert]
                }
            }
            #expect(try Self.loadedMatches(perExpert, expected: weights))
        }

        @Test func rejectsAnIncompleteExpertBank() throws {
            let target = try Self.target()
            var weights = Self.headWeights(for: target)
            let value = weights.removeValue(forKey: "layers.1.mixer.switch_mlp.fc1.weight")!
            for expert in 0 ..< 3 {
                weights["layers.1.mixer.experts.\(expert).up_proj.weight"] = value[expert]
            }
            let folder = try Self.artifact(config: target.lightningConfiguration, weights: weights)
            defer { try? FileManager.default.removeItem(at: folder) }
            let error = #expect(throws: NemotronH35MTPError.self) {
                _ = try NemotronH35MTPAssistant.load(from: folder, target: target)
            }
            #expect(error?.localizedDescription.contains("incomplete expert bank") == true)
        }

        @Test func rejectsAConfigurationOfAnotherTarget() throws {
            let target = try Self.target()
            let other = try Self.target(vocabularySize: 256)
            let folder = try Self.artifact(
                config: other.lightningConfiguration, weights: Self.headWeights(for: target))
            defer { try? FileManager.default.removeItem(at: folder) }
            let error = #expect(throws: NemotronH35MTPError.self) {
                _ = try NemotronH35MTPAssistant.load(from: folder, target: target)
            }
            let message = error?.localizedDescription ?? ""
            #expect(message.contains("configuration differs from target"))
        }

        @Test func rejectsAShardPathOutsideTheFolder() throws {
            let target = try Self.target()
            let folder = try Self.artifact(
                config: target.lightningConfiguration, weights: Self.headWeights(for: target),
                extraIndex: ["mtp.extra.weight": "../other.safetensors"])
            defer { try? FileManager.default.removeItem(at: folder) }
            let error = #expect(throws: NemotronH35MTPError.self) {
                _ = try NemotronH35MTPAssistant.load(from: folder, target: target)
            }
            #expect(error?.localizedDescription.contains("invalid shard path") == true)
        }

        @Test func rejectsAnIndexedTensorThatIsMissing() throws {
            let target = try Self.target()
            let folder = try Self.artifact(
                config: target.lightningConfiguration, weights: Self.headWeights(for: target),
                extraIndex: ["mtp.layers.0.missing.weight": "model-00001-of-00001.safetensors"])
            defer { try? FileManager.default.removeItem(at: folder) }
            let error = #expect(throws: NemotronH35MTPError.self) {
                _ = try NemotronH35MTPAssistant.load(from: folder, target: target)
            }
            #expect(error?.localizedDescription.contains("indexed tensor missing") == true)
        }

        @Test func rejectsScalesWithoutAQuantizationEntry() throws {
            let target = try Self.target()
            var weights = Self.headWeights(for: target)
            weights["layers.0.eh_proj.scales"] = MLXArray.ones([64, 4])
            let folder = try Self.artifact(config: target.lightningConfiguration, weights: weights)
            defer { try? FileManager.default.removeItem(at: folder) }
            let error = #expect(throws: NemotronH35MTPError.self) {
                _ = try NemotronH35MTPAssistant.load(from: folder, target: target)
            }
            #expect(error?.localizedDescription.contains("quantization missing") == true)
        }
    }
}
