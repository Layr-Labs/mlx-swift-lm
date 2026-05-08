// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXSpeculative

@Suite("DFlashTokenIterator")
struct DFlashTokenIteratorTests {
    private func tinyQwen3Config() throws -> Qwen3Configuration {
        let json = """
        {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128
        }
        """
        return try JSONDecoder.json5().decode(Qwen3Configuration.self, from: Data(json.utf8))
    }

    private func dflashConfig() throws -> DFlashConfiguration {
        let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 128,
            "block_size": 4,
            "num_target_layers": 3,
            "layer_types": ["full_attention", "full_attention"],
            "tie_word_embeddings": true,
            "dflash_config": {
                "target_layer_ids": [0, 1],
                "mask_token_id": 4
            }
        }
        """
        return try JSONDecoder.json5().decode(DFlashConfiguration.self, from: Data(json.utf8))
    }

    @Test func firstTokenMatchesTargetPrefillBonusAndHonorsMaxTokens() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let promptTokens = MLXArray([Int32(1), 2, 3])
            let promptBatch = promptTokens[.newAxis, .ellipsis]
            let expectedForward = try target.forwardForDFlash(
                promptBatch,
                cache: target.newCache(parameters: nil),
                targetLayerIds: drafter.config.targetLayerIds
            )
            let expectedFirst = expectedForward.logits[0..., -1, 0...]
                .asType(.float32).argMax(axis: -1)
            eval(expectedFirst)

            var iterator = try DFlashTokenIterator(
                input: LMInput(text: .init(tokens: promptTokens)),
                target: target,
                drafter: drafter,
                parameters: GenerateParameters(maxTokens: 3, temperature: 0)
            )

            #expect(iterator.next() == Int(expectedFirst.item(Int32.self)))
            #expect(iterator.next() != nil)
            #expect(iterator.next() != nil)
            #expect(iterator.next() == nil)
            #expect(iterator.tokenCount == 3)
            #expect(iterator.promptPrefillTime >= 0)
        }
    }

    @Test func rejectsNonGreedySamplingForNow() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            let promptTokens = MLXArray([Int32(1), 2, 3])

            #expect(throws: DFlashError.self) {
                _ = try DFlashTokenIterator(
                    input: LMInput(text: .init(tokens: promptTokens)),
                    target: target,
                    drafter: drafter,
                    parameters: GenerateParameters(maxTokens: 3, temperature: 0.6)
                )
            }
        }
    }

    @Test func greedyRoundTrimsTargetCacheButKeepsDraftContextCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let promptTokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            var targetCache = target.newCache(parameters: nil)
            let draftCache = try drafter.makeCache()
            let prefillOut = try target.forwardForDFlash(
                promptTokens,
                cache: targetCache,
                targetLayerIds: drafter.config.targetLayerIds
            )
            let bonus = prefillOut.logits[0..., -1, 0...]
                .asType(.float32).argMax(axis: -1)
            eval(bonus, prefillOut.targetHidden)

            let round = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: Int(bonus.item(Int32.self)),
                targetHidden: prefillOut.targetHidden,
                promptTokenCount: promptTokens.dim(1),
                generatedTokenCount: 1,
                blockSize: 4,
                maxEmitCount: 4
            )

            #expect(targetCache[0].offset == promptTokens.dim(1) + round.accepted + 1)
            #expect(draftCache[0].offset == promptTokens.dim(1))
            #expect(round.targetHidden.dim(1) == round.accepted + 1)
            #expect(round.tokens.count == round.accepted + 1)
        }
    }

    @Test func greedyRoundTrimsDraftCacheToCommittedContext() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Qwen3Model(try tinyQwen3Config())
            let drafter = DFlashDraftModel(config: try dflashConfig())
            try drafter.bind(target: target)
            eval(target, drafter)

            let promptTokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            var targetCache = target.newCache(parameters: nil)
            let draftCache = try drafter.makeCache()
            let staleKeys = MLXArray.zeros([1, 1, 2, 8])
            let staleValues = MLXArray.zeros([1, 1, 2, 8])
            for cache in draftCache {
                _ = cache.update(keys: staleKeys, values: staleValues)
            }

            let prefillOut = try target.forwardForDFlash(
                promptTokens,
                cache: targetCache,
                targetLayerIds: drafter.config.targetLayerIds
            )
            let bonus = prefillOut.logits[0..., -1, 0...]
                .asType(.float32).argMax(axis: -1)
            eval(bonus, prefillOut.targetHidden)

            _ = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: Int(bonus.item(Int32.self)),
                targetHidden: prefillOut.targetHidden,
                promptTokenCount: promptTokens.dim(1),
                generatedTokenCount: 1,
                blockSize: 4,
                maxEmitCount: 4
            )

            #expect(draftCache.allSatisfy { $0.offset == promptTokens.dim(1) })
        }
    }

    @Test func greedyRoundReplaysAcceptedPrefixForNonTrimmableTargetCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = ReplayRequiredTarget()
            let drafter = DFlashDraftModel(config: try replayDFlashConfig())
            try drafter.bind(target: target)
            eval(drafter)

            var targetCache: [KVCache] = [NonTrimmableTokenCache()]
            let draftCache = try drafter.makeCache()
            let targetHidden = MLXArray.zeros([1, 1, 4])

            let round = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: 2,
                targetHidden: targetHidden,
                promptTokenCount: 0,
                generatedTokenCount: 0,
                blockSize: 4,
                maxEmitCount: 4
            )

            let cache = try #require(targetCache[0] as? NonTrimmableTokenCache)
            #expect(round.accepted == 0)
            #expect(round.tokens == [1])
            #expect(cache.inputTokens == [2])
            #expect(cache.offset == 1)
            #expect(round.targetHidden.dim(1) == 1)
        }
    }

    @Test func greedyRoundUsesTargetRollbackHookForNonTrimmableCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = HookedRollbackTarget()
            let drafter = DFlashDraftModel(config: try replayDFlashConfig())
            try drafter.bind(target: target)
            eval(drafter)

            var targetCache: [KVCache] = [NonTrimmableTokenCache()]
            let draftCache = try drafter.makeCache()
            let targetHidden = MLXArray.zeros([1, 1, 4])

            let round = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: 2,
                targetHidden: targetHidden,
                promptTokenCount: 0,
                generatedTokenCount: 0,
                blockSize: 4,
                maxEmitCount: 4
            )

            let cache = try #require(targetCache[0] as? NonTrimmableTokenCache)
            #expect(target.forwardCalls == 1)
            #expect(target.rollbackCalls == 1)
            #expect(round.accepted == 0)
            #expect(round.tokens == [1])
            #expect(cache.inputTokens == [2])
            #expect(cache.offset == 1)
            #expect(round.targetHidden.dim(1) == 1)
        }
    }

    @Test func greedyRoundCapsAcceptedPrefixToEmitBudgetForReplayRollback() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = ReplayRequiredTarget(verifyToken: 0)
            let drafter = DFlashDraftModel(config: try replayDFlashConfig())
            try drafter.bind(target: target)
            eval(drafter)

            var targetCache: [KVCache] = [NonTrimmableTokenCache()]
            let draftCache = try drafter.makeCache()
            let targetHidden = MLXArray.zeros([1, 1, 4])

            let round = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: 2,
                targetHidden: targetHidden,
                promptTokenCount: 0,
                generatedTokenCount: 0,
                blockSize: 4,
                maxEmitCount: 2
            )

            let cache = try #require(targetCache[0] as? NonTrimmableTokenCache)
            #expect(round.accepted == 1)
            #expect(round.tokens == [0, 0])
            #expect(cache.inputTokens == [2, 0])
            #expect(cache.offset == 2)
            #expect(round.targetHidden.dim(1) == 2)
        }
    }

    private func replayDFlashConfig() throws -> DFlashConfiguration {
        let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 128,
            "block_size": 4,
            "num_target_layers": 1,
            "layer_types": ["full_attention"],
            "tie_word_embeddings": false,
            "dflash_config": {
                "target_layer_ids": [0],
                "mask_token_id": 3
            }
        }
        """
        return try JSONDecoder.json5().decode(DFlashConfiguration.self, from: Data(json.utf8))
    }
}

private final class NonTrimmableTokenCache: KVCache {
    var offset = 0
    var maxSize: Int? { nil }
    var inputTokens: [Int] = []
    var state: [MLXArray] = []
    var metaState: [String] = [""]

    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] { state }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(2)
        return (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        let copy = NonTrimmableTokenCache()
        copy.inputTokens = inputTokens
        copy.offset = offset
        copy.state = state.map { $0[.ellipsis] }
        copy.metaState = metaState
        return copy
    }
}

private final class ReplayRequiredTarget: Module, DFlashTargetModel {
    private let verifyToken: Int
    private let draftToken: Int

    let vocabularySize = 4
    let kvHeads = [1]
    let dFlashVocabularySize = 4
    let dFlashHiddenSize = 4
    let dFlashLayerCount = 1
    var loraLayers: [Module] { [] }

    init(verifyToken: Int = 1, draftToken: Int = 0) {
        self.verifyToken = verifyToken
        self.draftToken = draftToken
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(batch: inputs.dim(0), length: inputs.dim(1), token: verifyToken)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [NonTrimmableTokenCache()]
    }

    func forwardForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashTargetForward {
        if let tokenCache = cache?.first as? NonTrimmableTokenCache {
            tokenCache.inputTokens.append(
                contentsOf: inputs.squeezed(axis: 0).asArray(Int32.self).map { Int($0) })
            tokenCache.offset += inputs.dim(1)
        }

        let hidden = MLXArray.zeros([inputs.dim(0), inputs.dim(1), dFlashHiddenSize])
        return DFlashTargetForward(
            logits: makeLogits(batch: inputs.dim(0), length: inputs.dim(1), token: verifyToken),
            hiddenStates: targetLayerIds.map { _ in hidden }
        )
    }

    func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray {
        MLXArray.zeros([tokens.dim(0), tokens.dim(1), dFlashHiddenSize])
    }

    func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray {
        makeLogits(batch: hidden.dim(0), length: hidden.dim(1), token: draftToken)
    }

    private func makeLogits(batch: Int, length: Int, token: Int) -> MLXArray {
        var values = Array(repeating: Float(-100), count: batch * length * vocabularySize)
        for b in 0 ..< batch {
            for i in 0 ..< length {
                values[(b * length + i) * vocabularySize + token] = 100
            }
        }
        return MLXArray(values, [batch, length, vocabularySize])
    }
}

private struct HookRollbackState: DFlashTargetRollbackState {}

private final class HookedRollbackTarget: Module, DFlashTargetModel, DFlashTargetCacheRollbackProvider {
    private let verifyToken = 1
    private let draftToken = 0

    var forwardCalls = 0
    var rollbackCalls = 0

    let vocabularySize = 4
    let kvHeads = [1]
    let dFlashVocabularySize = 4
    let dFlashHiddenSize = 4
    let dFlashLayerCount = 1
    var loraLayers: [Module] { [] }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(batch: inputs.dim(0), length: inputs.dim(1), token: verifyToken)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [NonTrimmableTokenCache()]
    }

    func forwardForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashTargetForward {
        forwardCalls += 1
        if let tokenCache = cache?.first as? NonTrimmableTokenCache {
            tokenCache.inputTokens.append(
                contentsOf: inputs.squeezed(axis: 0).asArray(Int32.self).map { Int($0) })
            tokenCache.offset += inputs.dim(1)
        }

        let hidden = MLXArray.zeros([inputs.dim(0), inputs.dim(1), dFlashHiddenSize])
        return DFlashTargetForward(
            logits: makeLogits(batch: inputs.dim(0), length: inputs.dim(1), token: verifyToken),
            hiddenStates: targetLayerIds.map { _ in hidden }
        )
    }

    func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray {
        MLXArray.zeros([tokens.dim(0), tokens.dim(1), dFlashHiddenSize])
    }

    func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray {
        makeLogits(batch: hidden.dim(0), length: hidden.dim(1), token: draftToken)
    }

    func makeDFlashCacheRollbackState(cache: [KVCache]) -> (any DFlashTargetRollbackState)? {
        HookRollbackState()
    }

    func rollbackDFlashCache(
        _ cache: inout [KVCache],
        state: (any DFlashTargetRollbackState)?,
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray {
        rollbackCalls += 1
        let acceptedPrefix = verifyInput[0..., 0 ..< acceptedTokenCount + 1]
        if let tokenCache = cache.first as? NonTrimmableTokenCache {
            tokenCache.inputTokens = acceptedPrefix.squeezed(axis: 0)
                .asArray(Int32.self).map { Int($0) }
            tokenCache.offset = acceptedTokenCount + 1
        }
        return verifiedTargetHidden[0..., 0 ..< acceptedTokenCount + 1, 0...]
    }

    private func makeLogits(batch: Int, length: Int, token: Int) -> MLXArray {
        var values = Array(repeating: Float(-100), count: batch * length * vocabularySize)
        for b in 0 ..< batch {
            for i in 0 ..< length {
                values[(b * length + i) * vocabularySize + token] = 100
            }
        }
        return MLXArray(values, [batch, length, vocabularySize])
    }
}
