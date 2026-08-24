// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx speculative execution at the revision in NOTICE.

import MLX
import MLXLLM
import MLXLMCommon

public protocol DFlash2QwenTarget: AnyObject {
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray
    func newCache(parameters: GenerateParameters?) -> [KVCache]
    func dflashForward(
        input: LMInput.Text,
        cache: [any KVCache],
        nConfirmed: Int
    ) -> Qwen35DFlashForwardResult
    func dflashPrefillChunk(
        input: LMInput.Text,
        cache: [any KVCache]
    ) -> Qwen35DFlashForwardResult
    func dflashTargetOnlyPrefillChunk(
        input: LMInput.Text,
        cache: [any KVCache]
    ) -> MLXArray
    func dflashInputEmbedding(_ tokens: MLXArray) -> MLXArray
    func dflashLogits(_ normalizedDraftHidden: MLXArray) -> MLXArray
    func dflashCommitCachePrefix(
        cache: [any KVCache], committedRows: Int, verifyRows: Int)
}

extension Qwen35TextModel: DFlash2QwenTarget {}
extension Qwen35Model: DFlash2QwenTarget {}

public struct DFlash2CycleResult {
    public let committedTokens: MLXArray
    public let nextToken: MLXArray
    public let acceptedDraftTokens: Int
    public let physicalWidth: Int

    public init(
        committedTokens: MLXArray,
        nextToken: MLXArray,
        acceptedDraftTokens: Int,
        physicalWidth: Int
    ) {
        self.committedTokens = committedTokens
        self.nextToken = nextToken
        self.acceptedDraftTokens = acceptedDraftTokens
        self.physicalWidth = physicalWidth
    }
}

public final class DFlash2Session {
    private let target: any DFlash2QwenTarget
    private let draft: DFlash2DraftModel
    private let targetCache: [KVCache]
    private let draftCache: [DFlash2ContextKVCache]
    private let targetSampler: Qwen38TargetSampler
    private var contextPolicy: DFlash2ContextPolicy
    private var stagedToken: MLXArray?
    private var projectedDraftContext: MLXArray?

    public init(
        target: any DFlash2QwenTarget,
        draft: DFlash2DraftModel,
        promptLength: Int,
        seed: UInt64 = Qwen38TargetSampler.seed
    ) {
        self.target = target
        self.draft = draft
        targetCache = target.newCache(parameters: nil)
        draftCache = draft.makeCache()
        targetSampler = Qwen38TargetSampler(seed: seed)
        contextPolicy = DFlash2ContextPolicy(promptLength: promptLength)
    }

    /// Installs the prompt frontier. The returned token is authoritative but
    /// remains staged until the first verify cycle advances the target state.
    @discardableResult
    public func prefill(promptTokens: MLXArray) -> MLXArray {
        var finalLogits: MLXArray?
        var projectedDType: DType?
        for range in dflash2PrefillRanges(tokenCount: promptTokens.dim(1)) {
            let output = target.dflashPrefillChunk(
                input: LMInput.Text(tokens: promptTokens[0..., range]),
                cache: targetCache)
            let projected = draft.projectTargetFeatures(output.targetFeatures)
            draft.advanceProjectedContextCache(
                draftContext: projected,
                cache: draftCache)
            projectedDType = projected.dtype
            finalLogits = output.logits
            eval(
                output.logits,
                targetCache.flatMap { $0.innerState() },
                draftCache.flatMap { $0.innerState() })
        }
        let frontier = targetSampler.sample(
            logits: finalLogits![0..., -1, 0...]
        ).reshaped([-1])
        stagedToken = frontier
        projectedDraftContext = MLXArray.zeros(
            [promptTokens.dim(0), 0, draft.configuration.hiddenSize],
            dtype: projectedDType!)
        eval(frontier, projectedDraftContext!, targetCache.flatMap { $0.innerState() })
        return frontier
    }

    public func step() -> DFlash2CycleResult {
        step(physicalWidth: contextPolicy.nextPhysicalWidth)
    }

    /// Construction-only shape warm used before the measured session exists.
    /// The caller owns a throwaway session, so updating its adaptive policy has
    /// no effect on production routing.
    public func warmStep(physicalWidth: Int) -> DFlash2CycleResult {
        precondition((1 ... 8).contains(physicalWidth))
        return step(physicalWidth: physicalWidth)
    }

    @inline(__always)
    private func step(physicalWidth width: Int) -> DFlash2CycleResult {
        let stagedToken = stagedToken!
        let projectedDraftContext = projectedDraftContext!
        let drafted: MLXArray
        let verifyTokens: MLXArray
        if width == 1 {
            draft.advanceProjectedContextCache(
                draftContext: projectedDraftContext,
                cache: draftCache)
            drafted = MLXArray([Int32]()).reshaped([1, 0])
            verifyTokens = stagedToken.reshaped([1, 1])
        } else {
            let masks = MLXArray(
                [Int32](
                    repeating: Int32(draft.configuration.maskTokenID),
                    count: width - 1))
            let blockTokens = concatenated([stagedToken, masks], axis: 0)
                .reshaped([1, width])
            let noiseEmbedding = target.dflashInputEmbedding(blockTokens)
            let draftHidden = draft.forwardProjectedContext(
                noiseEmbedding: noiseEmbedding,
                draftContext: projectedDraftContext,
                cache: draftCache)
            let proposalHidden = draftHidden[0..., 1..., 0...]
            let draftLogits = target.dflashLogits(proposalHidden)
            let proposal = draft.selectProposal(
                draftHidden: proposalHidden,
                logits: draftLogits,
                anchorIDs: stagedToken,
                temperature: 0)
            drafted = proposal.tokenIDs.asType(.int32)
            verifyTokens = concatenated(
                [stagedToken.reshaped([1, 1]), drafted], axis: 1)
        }

        let verification = target.dflashForward(
            input: LMInput.Text(tokens: verifyTokens),
            cache: targetCache,
            nConfirmed: 1)
        let posterior = targetSampler.sample(logits: verification.logits[0])
        let acceptedDraftTokens: Int
        if width == 1 {
            acceptedDraftTokens = 0
        } else {
            let accepted = (drafted[0] .== posterior[0 ..< (width - 1)])
                .asType(.int32)
                .cumprod()
                .sum()
                .item(Int32.self)
            acceptedDraftTokens = Int(accepted)
        }
        let plan = DFlash2CommitPlan(
            verifyRows: width,
            acceptedDraftTokens: acceptedDraftTokens)

        target.dflashCommitCachePrefix(
            cache: targetCache,
            committedRows: plan.commitRows,
            verifyRows: width)

        let committedTokens = verifyTokens[0, 0 ..< plan.commitRows]
        let committedFeatures = verification.targetFeatures[
            0..., 0 ..< plan.commitRows, 0...]
        let nextToken = posterior[acceptedDraftTokens ..< (acceptedDraftTokens + 1)]
        self.stagedToken = nextToken
        self.projectedDraftContext = draft.projectTargetFeatures(committedFeatures)
        contextPolicy.record(
            blockLength: width,
            acceptedDraftTokens: acceptedDraftTokens)
        asyncEval(
            committedTokens,
            nextToken,
            self.projectedDraftContext!,
            targetCache.flatMap { $0.innerState() },
            draftCache.flatMap { $0.innerState() })
        return DFlash2CycleResult(
            committedTokens: committedTokens,
            nextToken: nextToken,
            acceptedDraftTokens: acceptedDraftTokens,
            physicalWidth: width)
    }
}
