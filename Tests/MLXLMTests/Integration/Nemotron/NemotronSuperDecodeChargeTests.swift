import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The batched decode charge of a loaded synthetic trunk.
@Suite(.tags(.integration))
struct NemotronSuperDecodeChargeTests {
    typealias Fixture = NemotronSuperBatchFixture

    @Test func loadedModelChargeUsesItsOwnShapesAndGraphQuotes() throws {
        let model = try Fixture.model(seed: 2, dtype: .bfloat16, products: .packedTiled)
        let language = model.languageModel
        let charge = try language.superDecodeCharge(kvElementBytes: 2)
        #expect(
            charge.recurrentBytesPerGeneration
                == (try model.cbv2RecurrentStateSpec.fixedBytesPerRequest()))
        #expect(charge.recurrentGenerations == CBv2RecurrentStateSpec.maximumLiveGenerations)
        // One attention layer, two KV heads of 64 for K and V, two bytes.
        #expect(charge.attentionKVBytesPerToken == 2 * 2 * 64 * 2)
        let workspace = charge.rowWorkspace
        #expect(
            workspace.modelOptLayerBytes == language.modelOptGraphPrefillWorkspaceBytes(tokens: 1))
        #expect(workspace.logitsBytes == Fixture.vocabulary * (2 + 8))
        #expect(
            workspace.totalBytes
                == workspace.modelOptLayerBytes + workspace.outputHeadBytes
                + workspace.logitsBytes + workspace.trunkBytes + workspace.backendScratchBytes)
        print("BATCH fixture decode charge: \(charge)")
        // The eager reference has no per-row workspace quote.
        language.setModelOptExecution(.eagerReference)
        #expect(throws: NemotronSuperDecodeCharge.Failure.unsupportedExecution) {
            try language.superDecodeCharge(kvElementBytes: 2)
        }
    }
}
