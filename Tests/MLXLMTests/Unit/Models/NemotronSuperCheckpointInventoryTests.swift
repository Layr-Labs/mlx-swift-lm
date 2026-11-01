import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    struct NemotronSuperCheckpointInventoryTests {
        private func index(_ keys: [String: String]) throws -> Data {
            try JSONSerialization.data(withJSONObject: ["weight_map": keys])
        }
        @Test func partitionsCompositeAndRetainsMTP() throws {
            let names = [
                "language_model.backbone.embeddings.weight",
                "language_model.mtp.layers.0.eh_proj.weight",
                "vision_model.radio_model.model.patch_generator.embedder.weight", "mlp1.0.weight",
            ]
            let plan = try NemotronSuperCheckpointInventory(
                indexData: index(
                    Dictionary(uniqueKeysWithValues: names.map { ($0, "model-00001.safetensors") }))
            )
            #expect(plan.entries.count == 4)
            #expect(plan.entries(for: .mtp).first?.destination == "mtp.layers.0.eh_proj.weight")
            #expect(
                plan.entries(for: .projector).first?.destination
                    == "vision_projector.mlp1.norm.weight")
            #expect(plan.shards == ["model-00001.safetensors"])
        }
        @Test func rejectsUnsafeShardAndUnknownNamespace() throws {
            for shard in [
                "../model.safetensors", "/model.safetensors", "dir/model.safetensors",
                "dir\\model.safetensors",
            ] {
                #expect(throws: (any Error).self) {
                    try NemotronSuperCheckpointInventory(
                        indexData: index(["language_model.lm_head.weight": shard]))
                }
            }
            #expect(throws: (any Error).self) {
                try NemotronSuperCheckpointInventory(
                    indexData: index(["unrecognized.weight": "model.safetensors"]))
            }
        }
        @Test func rejectsAliasedProjectorCollision() throws {
            #expect(throws: (any Error).self) {
                try NemotronSuperCheckpointInventory(
                    indexData: index([
                        "mlp1.0.weight": "model.safetensors",
                        "vision_projector.mlp1.norm.weight": "model.safetensors",
                    ]))
            }
        }
        @Test func requiresCalibrationAndRejectsAffineRelabeling() throws {
            let prefix = "language_model.backbone.layers.1.mixer.experts.0.up_proj"
            var weights = Dictionary(
                uniqueKeysWithValues: ["weight", "input_scale", "weight_scale", "weight_scale_2"]
                    .map { (prefix + "." + $0, "model.safetensors") })
            #expect(
                try NemotronSuperCheckpointInventory(indexData: index(weights)).entries.count == 4)
            weights.removeValue(forKey: prefix + ".input_scale")
            #expect(throws: (any Error).self) {
                try NemotronSuperCheckpointInventory(indexData: index(weights))
            }
            weights[prefix + ".input_scale"] = "model.safetensors"
            weights[prefix + ".scales"] = "model.safetensors"
            #expect(throws: (any Error).self) {
                try NemotronSuperCheckpointInventory(indexData: index(weights))
            }
        }
    }
}
