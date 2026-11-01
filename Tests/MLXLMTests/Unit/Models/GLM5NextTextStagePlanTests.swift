import Testing

@testable import MLXLLM

extension UnitTests {
    @Suite struct GLM5NextTextStagePlanTests {
        @Test func allFortyFourCutsConserveTextPayloadAndChargeMTPEmbedding() throws {
            let config = try GLM5NextInventoryFixture.configuration(layers: 45, q4: true)
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config,
                tensors: GLM5NextWeightInventory.expectedTensors(
                    configuration: config, storage: .vontraQ4, shard: "synthetic.safetensors"),
                storage: .vontraQ4)
            let allText = Set(inventory.entries(in: .text).map { $0.tensor.name })
            let embedding = Set(
                allText.filter { $0.hasPrefix("model.language_model.embed_tokens.") })
            let duplicateBytes = embedding.reduce(0) { $0 + inventory.entries[$1]!.bytes }
            for cut in 1 ..< 45 {
                let first = try GLM5NextTextStagePlan(
                    configuration: config.textConfig, layers: 0 ..< cut)
                let last = try GLM5NextTextStagePlan(
                    configuration: config.textConfig, layers: cut ..< 45, loadMTPEmbedding: true)
                let a = try first.tensorNames(in: inventory)
                let b = try last.tensorNames(in: inventory)
                #expect(a.intersection(b) == embedding)
                #expect(a.union(b) == allText)
                #expect(
                    try first.tensorBytes(in: inventory) + last.tensorBytes(in: inventory)
                        == inventory.bytes(in: .text) + duplicateBytes)
                #expect(first.isFirst && !first.isFinal && !last.isFirst && last.isFinal)
                #expect(!a.contains { inventory.entries[$0]!.part != .text })
                #expect(!b.contains { inventory.entries[$0]!.part != .text })
            }
        }

        @Test func refusesEmptyOutOfRangeAndNonfinalMTPPlans() throws {
            let c = try GLM5NextInventoryFixture.configuration().textConfig
            for range in [0 ..< 0, -1 ..< 2, 0 ..< 5, 4 ..< 4] {
                #expect(throws: GLM5NextLoadError.self) {
                    try GLM5NextTextStagePlan(configuration: c, layers: range)
                }
            }
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextTextStagePlan(configuration: c, layers: 0 ..< 2, loadMTPEmbedding: true)
            }
        }
    }
}
