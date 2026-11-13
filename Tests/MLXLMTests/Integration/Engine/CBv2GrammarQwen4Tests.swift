// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Qwen4 grammar-aware MTP", .serialized, .tags(.integration))
struct CBv2GrammarQwen4Tests {
    private final class Grammar: CBv2TokenConstraint, Sendable {
        let mode: CBv2TokenConstraintMode = .required
        let initialState = 0
        let fallbackTokenID = 0
        let path = [31, 32, 41, 42, 51, 52, 21, 22, 11, 12, 61, 62]
        var maxTokens: Int { path.count }

        func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
            guard path.indices.contains(state), remainingTokens > 0 else { return [] }
            return [path[state]]
        }

        func nextState(state: Int, tokenID: Int) -> Int? {
            guard path.indices.contains(state), tokenID == path[state] else { return nil }
            return state + 1
        }
    }

    private func makeEngine(speculating: Bool) throws -> EngineV2 {
        var config = Qwen4ExpTextConfiguration()
        config.hiddenSize = 64
        config.hiddenLayers = 2
        config.attentionHeads = 2
        config.kvHeads = 1
        config.headDim = 64
        config.linearNumValueHeads = 2
        config.linearNumKeyHeads = 1
        config.linearKeyHeadDim = 64
        config.linearValueHeadDim = 64
        config.vocabularySize = 64
        config.maxPositionEmbeddings = 512
        config.fullAttentionInterval = 2
        config.layerTypes = ["linear_attention", "qwen_sparse_attention"]
        config.hcCount = 2
        config.hcLowrank = 8
        config.pleLayerIds = []
        config.pleEmbedDim = 64
        config.indexerNHeads = 2
        config.indexerKVHeads = 1
        config.indexerHeadDim = 32
        config.indexerBudget = 16
        config.indexerCompressRatio = 4
        config.numExperts = 1
        config.numExpertsPerTok = 1
        config.sharedExpertIntermediateSize = 32
        config.moeIntermediateSize = 32
        config.mropeSection = [2, 1, 1]
        config.partialRotaryFactor = 0.25
        config.mtpNumHiddenLayers = 1
        MLXRandom.seed(8301)
        let target = Qwen4ExpTextModel(config)
        target.update(
            parameters: ModuleParameters.unflattened(
                target.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(target)
        let adapter = CBv2SteppableLanguageModelAdapter(target)
        let kinds = target.cbv2LayerKinds
        let observed = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let chunk = max(32, CBv2AttentionV1.queryBlockSize)
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 256 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 512,
                segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
        let assistant =
            try speculating
            ? Qwen4ExpInlineMTPAssistant(
                configuration: config, blockSize: 3, target: target, verificationMode: .rectangular)
            : nil
        if let assistant { eval(assistant) }
        return EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2DefaultSampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: assistant,
            mtpConfig: .init(
                enabled: speculating, maxDraftTokens: 2, fixedDraftTokens: 2,
                verificationMode: .rectangular, constrainedSpeculation: true))
    }

    @Test func nativeQwen4AssistantMatchesActualConstrainedTargetOnlyEngine() async throws {
        let grammar = Grammar()
        let item = CBv2Request(
            id: .init(1001), promptTokens: [1, 3, 6, 10, 16],
            sampling: .init(temperature: 0), maxTokens: grammar.maxTokens, stopTokens: [0],
            tokenConstraint: grammar)
        let oracle = try makeEngine(speculating: false)
        let expected = await cbv2SchedCollect(try oracle.submit(item))
        await oracle.shutdown()
        #expect(expected.tokens == grammar.path)
        let candidate = try makeEngine(speculating: true)
        #expect(candidate.nativeShutdownExecutionContractID == nil)
        let actual = await cbv2SchedCollect(try candidate.submit(item))
        let metrics = try #require(candidate.mtpMetricsSnapshot())
        await candidate.shutdown()
        #expect(actual.tokens == expected.tokens)
        #expect(actual.finishReason == expected.finishReason)
        #expect(metrics.rounds > 0)
        #expect(metrics.rectangularVerificationRounds > 0)
    }
}
