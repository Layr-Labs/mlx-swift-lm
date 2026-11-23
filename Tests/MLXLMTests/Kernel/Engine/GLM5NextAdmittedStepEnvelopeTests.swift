import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextAdmittedStepEnvelopeTests {
        // The tests build engines and price steps; they run no forward pass.
        private func makeModel() throws -> GLM5NextTextModel {
            let config = try GLM5NextInventoryFixture.configuration(layers: 4)
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "tiny.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
            return try GLM5NextTextModel(configuration: config.textConfig, inventory: inventory) {
                name in
                MLXArray.zeros(try #require(inventory.entries[name]).tensor.shape)
            }
        }

        /// An admission that reserves KV and step scratch for every request
        /// token holds at most the tokens its admissible bytes buy, so the
        /// worst step is priced for those tokens instead of every row at the
        /// context limit. An envelope refused under the row bound is accepted
        /// once the caller states its per-token charge, and the engine passes
        /// that charge to the check it runs when it is built.
        @Test func admittedTokensBoundTheStepEnvelope() async throws {
            let model = try makeModel()
            func backend(scratchCeiling: Int) throws -> PagedKVBackend {
                try model.makeCBv2PagedBackend(
                    kvBytesCapacity: 128 << 20, maximumBatchSize: 2, maximumQueryTokens: 64,
                    maximumContextTokens: 128, maximumScratchBytes: scratchCeiling,
                    segmentBytes: 64 << 10, maximumBufferBytes: 32 << 20)
            }
            func engine(_ backend: PagedKVBackend, charge: Int?) -> EngineV2 {
                EngineV2(
                    model: CBv2SteppableLanguageModelAdapter(model),
                    layerKinds: model.cbv2LayerKinds, backend: backend,
                    cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                    sampler: CBv2GreedySampler(),
                    schedulerConfig: .init(
                        maxConcurrentRequests: 2, maxBatchedTokensPerStep: 64,
                        prefillChunkSize: 16, maxWaiting: 2, enablePrefixCache: false),
                    admissionConfig: model.cbv2NativeAdmissionConfig,
                    processMemoryOwner: GLM5NextTestProcessBudget(),
                    glm5NextAdmittedBytesPerContextToken: charge)
            }
            let probeBackend = try backend(scratchCeiling: 256 << 20)
            let probe = engine(probeBackend, charge: nil)
            let binding = try #require(probeBackend.glm5NextModelBinding)
            let admissible = probe.admissibleKVBytesCapacity
            let setup = try binding.stepScratch(
                maximumRequests: 2, maximumStepTokens: 64, maximumRowTokens: 16
            ).setupBytes
            await probe.shutdown()
            // A charge that buys eight tokens beside the setup bytes.
            let charge = (admissible - setup) / 8
            func price(_ charge: Int?) throws -> Int {
                try binding.validateStepEnvelope(
                    maximumRequests: 2, maximumStepTokens: 64, maximumRowTokens: 16,
                    admissibleBytes: admissible, admittedBytesPerContextToken: charge)
            }
            let rows = try price(nil)
            let admitted = try price(charge)
            try #require(
                admitted < rows,
                "eight admitted tokens must price below every row at the context limit")
            #expect(admitted <= admissible)
            // No admissible bytes: no admitted-token bound.
            #expect(
                try binding.validateStepEnvelope(
                    maximumRequests: 2, maximumStepTokens: 64, maximumRowTokens: 16,
                    admissibleBytes: nil, admittedBytesPerContextToken: charge) == rows)
            // A charge below KV plus scratch counts as KV plus scratch.
            #expect(try price(1) <= rows)
            #expect(try price(1) >= admitted)

            for stated in [nil, charge] {
                let built = engine(try backend(scratchCeiling: admitted), charge: stated)
                #expect(built.glm5NextAdmittedBytesPerContextToken == stated)
                if stated == nil {
                    #expect(
                        built.pagedAttentionWorkInactiveReason?.hasPrefix(
                            "GLM worst-case step scratch exceeds the engine envelope") == true)
                } else {
                    #expect(built.pagedAttentionWorkInactiveReason == nil)
                }
                await built.shutdown()
            }
        }
    }
}
