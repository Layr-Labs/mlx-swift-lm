import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 quantized workspace scheduling", .serialized)
struct CBv2QuantizedWorkspaceEngineTests {
    private final class Model: CBv2PrefillSteppableModel {
        var widths: [Int] = []

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            widths.append(tokens.dim(1))
            var logits = broadcast(MLXArray([Float(0), Float(1)]).reshaped([1, 1, 2]),
                to: [tokens.dim(0), tokens.dim(1), 2])
            for cache in caches {
                let qkv = MLXArray.ones(
                    [tokens.dim(0), 1, tokens.dim(1), cache.kind.headDim], dtype: .float32)
                let output = cache.updateAndAttend(
                    queries: qkv, keys: qkv, values: qkv, scale: 0.125, sinks: nil)
                logits = logits + sum(output, axes: [1, 3]).expandedDimensions(axis: 2)
            }
            return logits
        }

        func prefill(tokens: MLXArray, inputEmbeddings: MLXArray?,
                     caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement) -> MLXArray {
            let output = forward(tokens: tokens, caches: caches)
            switch requirement {
            case .evaluationOnly: return output[0..., -1, 0..<1]
            case .lastPositionLogits: return output[0..., -1, 0...]
            }
        }
    }

    @Test("a final packed prefill and subsequent pure decodes run with the engine overlap allowance")
    func prefillToChainedDecode() async throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)
        let config = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 4 << 20, dtype: .float32, maxPrefillChunk: 33,
            nominalMaxSequenceLength: 64, maxBufferLength: 4 << 20,
            segmentSizeBytes: 32_768, layerDTypes: [.float32], quantization: .init())
        let backend = try PagedKVBackend(layerKinds: [kind], config: config)
        let model = Model()
        let engine = EngineV2(model: model, layerKinds: [kind], backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(), schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 33, prefillChunkSize: 33,
                maxWaiting: 2, enablePrefixCache: false))
        let expected = CBv2RequestWorkspaceProjection.quantizedPaged(
            layerKinds: [kind], config: config, maximumChunk: 33,
            maximumBatch: 1, maximumSerialDecodeCalls: 1, overlapPolicy: .engineSerialPrefill,
            sharesStepArenas: true)
        #expect(engine.routingWorkspaceBytes(forTokens: 37) == expected.bytes(forTokens: 37))
        let stream = try engine.submit(.init(id: .init(901), promptTokens: Array(repeating: 1, count: 33),
            sampling: .init(temperature: 0), maxTokens: 4, prefixCacheEnabled: false))
        let result = await cbv2SchedCollect(stream)
        #expect(result.finishReason == .length)
        await engine.shutdown()
        #expect(model.widths.first == 33)
        #expect(model.widths.dropFirst().allSatisfy { $0 == 1 })
        #expect(engine.chainedStepCount >= 2,
            "the fixture must cover both final-prefill/decode and decode/decode overlap")
        #expect(!backend.pool.writeValidation.isFaulted)
    }

    @Test("MTP marker prevents the engine from using a multi-column step as a chain base")
    func mtpStepsNeverChain() {
        let step = CBv2InFlightStep(assignments: [], participants: [], sampledRows: [],
            sampledTokens: nil, evalTargets: [], wallStartedNanos: 0)
        #expect(step.permitsChainedSuccessor)
        step.mtpRound = CBv2MTPRoundInFlight(verify: nil, seedRows: [], seedHidden: nil,
            seedPolicyTopTwoValues: nil, committedObservationRows: [])
        #expect(!step.permitsChainedSuccessor)
    }
}
