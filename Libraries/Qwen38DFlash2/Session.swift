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
    func dflashCommitInnovationCachePrefix(
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

/// Cycle payload for the explicit non-measured diagnostic route. Production
/// `step()` never constructs or retains these proposal/posterior arrays.
public struct DFlash2CycleDiagnosticResult {
    public let cycle: DFlash2CycleResult
    public let proposedTokens: MLXArray
    public let posteriorTokens: MLXArray
}

/// The source engine leaves rollback arrays lazy while it submits the next
/// draft. The following target verification is their first consumer. A cycle
/// without a next-draft launch materializes cache state immediately so warm and
/// diagnostic callers still leave a settled session.
enum DFlash2CycleEvaluationPlan {
    static func explicitlyMaterializesTargetCache(
        prefetchingNextDraft: Bool
    ) -> Bool {
        !prefetchingNextDraft
    }
}

public enum DFlash2WidthPolicy: Equatable, Sendable {
    case adaptive
    case fixed(Int)

    var isValid: Bool {
        switch self {
        case .adaptive: true
        case .fixed(let width): (1 ... 8).contains(width)
        }
    }

    func resolve(adaptiveWidth: Int) -> Int {
        switch self {
        case .adaptive: adaptiveWidth
        case .fixed(let width): width
        }
    }
}

public final class DFlash2Session {
    private struct PrefetchedDraft {
        let stagedToken: MLXArray
        let drafted: MLXArray
    }

    private let target: any DFlash2QwenTarget
    private let draft: DFlash2DraftModel
    private let targetCache: [KVCache]
    private let draftCache: [DFlash2ContextKVCache]
    private let targetSampler: Qwen38TargetSampler
    private let widthPolicy: DFlash2WidthPolicy
    private var contextPolicy: DFlash2ContextPolicy
    private var stagedToken: MLXArray?
    private var projectedDraftContext: MLXArray?
    private var prefetchedDraft: PrefetchedDraft?

    public init(
        target: any DFlash2QwenTarget,
        draft: DFlash2DraftModel,
        promptLength: Int,
        widthPolicy: DFlash2WidthPolicy = .adaptive,
        seed: UInt64 = Qwen38TargetSampler.seed
    ) {
        precondition(
            widthPolicy.isValid,
            "DFlash2 fixed width must be in 1...8")
        self.target = target
        self.draft = draft
        self.widthPolicy = widthPolicy
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
        prefetchedDraft = nil
        projectedDraftContext = MLXArray.zeros(
            [promptTokens.dim(0), 0, draft.configuration.hiddenSize],
            dtype: projectedDType!)
        eval(frontier, projectedDraftContext!, targetCache.flatMap { $0.innerState() })
        return frontier
    }

    public func step(remainingOutputTokens: Int) -> DFlash2CycleResult {
        precondition(
            remainingOutputTokens > 0,
            "DFlash2 step requires a positive remaining output budget")
        let physicalWidth = widthPolicy.resolve(
            adaptiveWidth: contextPolicy.nextPhysicalWidth)
        return step(
            physicalWidth: physicalWidth,
            remainingOutputTokens: remainingOutputTokens,
            prefetchNext: true)
    }

    /// Construction-only shape warm used before the measured session exists.
    /// The caller owns a throwaway session, so updating its adaptive policy has
    /// no effect on production routing.
    public func warmStep(physicalWidth: Int) -> DFlash2CycleResult {
        precondition((1 ... 8).contains(physicalWidth))
        precondition(
            prefetchedDraft == nil,
            "DFlash2 warm step cannot follow a prefetched production step")
        return step(
            physicalWidth: physicalWidth,
            remainingOutputTokens: .max,
            prefetchNext: false)
    }

    public func diagnosticStep() -> DFlash2CycleDiagnosticResult {
        diagnosticStep(physicalWidth: contextPolicy.nextPhysicalWidth)
    }

    public func diagnosticStep(physicalWidth width: Int) -> DFlash2CycleDiagnosticResult {
        precondition((1 ... 8).contains(width))
        precondition(
            prefetchedDraft == nil,
            "DFlash2 diagnostic step cannot follow a prefetched production step")
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

        target.dflashCommitInnovationCachePrefix(
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
        return DFlash2CycleDiagnosticResult(
            cycle: DFlash2CycleResult(
                committedTokens: committedTokens,
                nextToken: nextToken,
                acceptedDraftTokens: acceptedDraftTokens,
                physicalWidth: width),
            proposedTokens: verifyTokens[0],
            posteriorTokens: posterior)
    }

    @inline(__always)
    private func prepareDraft(
        stagedToken: MLXArray,
        projectedDraftContext: MLXArray,
        physicalWidth width: Int
    ) -> MLXArray {
        if width == 1 {
            draft.advanceProjectedContextCache(
                draftContext: projectedDraftContext,
                cache: draftCache)
            return MLXArray([Int32]()).reshaped([1, 0])
        }
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
        return draft.selectProposal(
            draftHidden: proposalHidden,
            logits: draftLogits,
            anchorIDs: stagedToken,
            temperature: 0
        ).tokenIDs.asType(.int32)
    }

    @inline(__always)
    private func step(
        physicalWidth width: Int,
        remainingOutputTokens: Int,
        prefetchNext: Bool
    ) -> DFlash2CycleResult {
        let currentPrefetch = prefetchedDraft
        prefetchedDraft = nil
        let stagedToken = currentPrefetch?.stagedToken ?? stagedToken!
        let projectedDraftContext = projectedDraftContext!
        let drafted =
            currentPrefetch?.drafted
            ?? prepareDraft(
                stagedToken: stagedToken,
                projectedDraftContext: projectedDraftContext,
                physicalWidth: width)
        let verifyTokens: MLXArray
        if width == 1 {
            verifyTokens = stagedToken.reshaped([1, 1])
        } else {
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
            acceptedDraftTokens: acceptedDraftTokens
        ).capped(to: remainingOutputTokens)
        let committedDraftTokens = plan.commitRows - 1

        target.dflashCommitInnovationCachePrefix(
            cache: targetCache,
            committedRows: plan.commitRows,
            verifyRows: width)

        let committedTokens = verifyTokens[0, 0 ..< plan.commitRows]
        let committedFeatures = verification.targetFeatures[
            0..., 0 ..< plan.commitRows, 0...]
        let nextToken = posterior[committedDraftTokens ..< (committedDraftTokens + 1)]
        self.stagedToken = nextToken
        self.projectedDraftContext = draft.projectTargetFeatures(committedFeatures)
        contextPolicy.record(
            blockLength: width,
            acceptedDraftTokens: committedDraftTokens)
        let remainingAfterCommit = remainingOutputTokens - plan.commitRows
        let willPrefetchNext = prefetchNext && remainingAfterCommit > 0
        if DFlash2CycleEvaluationPlan.explicitlyMaterializesTargetCache(
            prefetchingNextDraft: willPrefetchNext)
        {
            asyncEval(
                committedTokens,
                nextToken,
                self.projectedDraftContext!,
                targetCache.flatMap { $0.innerState() },
                draftCache.flatMap { $0.innerState() })
        } else {
            asyncEval(
                committedTokens,
                nextToken,
                self.projectedDraftContext!)
        }
        if willPrefetchNext {
            let nextWidth = widthPolicy.resolve(
                adaptiveWidth: contextPolicy.nextPhysicalWidth)
            let nextDrafted = prepareDraft(
                stagedToken: nextToken,
                projectedDraftContext: self.projectedDraftContext!,
                physicalWidth: nextWidth)
            asyncEval(nextDrafted, draftCache.flatMap { $0.innerState() })
            prefetchedDraft = PrefetchedDraft(
                stagedToken: nextToken,
                drafted: nextDrafted)
        }
        return DFlash2CycleResult(
            committedTokens: committedTokens,
            nextToken: nextToken,
            acceptedDraftTokens: committedDraftTokens,
            physicalWidth: width)
    }
}
