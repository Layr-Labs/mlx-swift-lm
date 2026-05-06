// Copyright © 2026 Apple Inc.
//
// Gemma 4 Multi-Token Prediction "assistant" drafter model.
//
// Port of Google's Gemma 4 MTP drafter architecture:
//   - 4-layer trunk reusing `Gemma4TextModelInner` with every layer
//     `forceSharedKV: true` (so it consumes the target's K/V instead of
//     projecting its own).
//   - `pre_projection: Linear(2 * backbone_hidden → drafter_hidden)` —
//     combines `[target_embed(last_token), last_hidden]` per step.
//   - `post_projection: Linear(drafter_hidden → backbone_hidden)` — turns
//     the drafter's output hidden back into a backbone-compatible tensor
//     for the next step's `[target_embed, last_hidden]` input.
//   - LM head: either tied (default), explicit `lm_head` (when
//     `!tie_word_embeddings`), or `MaskedEmbedder` (centroid-routed
//     sparse head when `use_ordered_embeddings` — E2B / E4B drafters).
//
// This file contains init + bind/unbind + compatibility validation only.
// The forward pass lives in Task 17; weight sanitize in Task 18;
// load(from:using:id:) in Task 19.
//
// Reference: mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py
// in Blaizzy/mlx-vlm#1112 (merged 244f4bb).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public final class Gemma4AssistantDraftModel: Module, @unchecked Sendable {
    public let config: Gemma4AssistantConfiguration

    @ModuleInfo public var model: Gemma4TextModelInner
    @ModuleInfo(key: "pre_projection") public var preProjection: Linear
    @ModuleInfo(key: "post_projection") public var postProjection: Linear
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?
    @ModuleInfo(key: "masked_embedding") public var maskedEmbedder: MaskedEmbedder?

    // Set by bind(target:); drafter is not usable before bind().
    internal var targetEmbed: ((MLXArray) -> MLXArray)?
    internal var boundTargetID: ObjectIdentifier?

    public init(config: Gemma4AssistantConfiguration) {
        self.config = config

        let textCfg = config.textConfig
        self._model.wrappedValue =
            Gemma4TextModelInner(textCfg, forceSharedKV: true)

        // pre: [2 * backbone → drafter_hidden]
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, textCfg.hiddenSize, bias: false)
        // post: [drafter_hidden → backbone]
        self._postProjection.wrappedValue = Linear(
            textCfg.hiddenSize, config.backboneHiddenSize, bias: false)

        if !textCfg.tieWordEmbeddings {
            // Explicit LM head instantiated; MaskedEmbedder stays nil.
            self._lmHead.wrappedValue = Linear(
                textCfg.hiddenSize, textCfg.vocabSize, bias: false)
        }

        if config.useOrderedEmbeddings {
            self._maskedEmbedder.wrappedValue = MaskedEmbedder(
                hiddenSize: textCfg.hiddenSize,
                numCentroids: config.numCentroids,
                topK: config.centroidIntermediateTopK,
                vocabSize: textCfg.vocabSize
            )
        }

        super.init()
    }

    /// Bind the drafter to a Gemma 4 target. Captures a closure into the
    /// target's scaled embedding lookup + runs compatibility validation.
    /// Idempotent on the same target; throws `.rebindForbidden` if called
    /// with a different target.
    public func bind(target: Gemma4TextModel) throws {
        let newID = ObjectIdentifier(target)
        if let existing = boundTargetID {
            if existing == newID { return }  // idempotent same-target
            throw Gemma4MTPError.rebindForbidden
        }
        try validateCompatibility(with: target)
        self.targetEmbed = { [target] tokens in
            target.embedTokensForDrafter(tokens)
        }
        self.boundTargetID = newID
    }

    /// Clear the binding. Subsequent forward passes will throw
    /// `.drafterNotBound` (enforced in Task 17).
    public func unbind() {
        self.targetEmbed = nil
        self.boundTargetID = nil
    }

    // MARK: - Forward pass

    /// Drafter forward pass.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: `[B, 1, 2 * backboneHiddenSize]` — the drafter-step
    ///     input, which the round-loop constructs as
    ///     `concat([target_embed(last_token), last_hidden], axis: -1)`.
    ///   - sharedKV: K/V snapshot from the target's last non-shared
    ///     full-attention and sliding-attention layers. Every drafter layer
    ///     reads from the appropriate slot by its `layerType`.
    ///   - positionOffset: absolute position of the bonus token; held
    ///     constant across all drafter steps within a block.
    /// - Returns: `(lastHidden: [B, 1, backboneHiddenSize], logits: [B, 1, vocabSize])`.
    ///
    /// Softcap is **not** applied (drafter configs have
    /// `final_logit_softcapping: null`).
    public func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let textCfg = config.textConfig

        // Project the [target_embed, last_hidden] concat to drafter-hidden.
        var h = preProjection(inputsEmbeds)

        // Build per-layer-type masks (bidirectional; SWA may short-circuit to .none).
        let queryLen = h.dim(1)
        let queryOffset: Int
        switch positionOffset {
        case .scalar(let v): queryOffset = v
        case .batch(let arr): queryOffset = Int(arr[0].item(Int32.self))
        }

        let fullKVLen = sharedKV.fullAttention.0.dim(2)
        let slidingKVLen = sharedKV.slidingAttention.0.dim(2)
        let dtype = h.dtype

        let fullMask = DrafterMasks.bidirectionalFull(
            queryLen: queryLen, kvLen: fullKVLen, dtype: dtype)
        let slidingMask = DrafterMasks.bidirectionalSWA(
            queryLen: queryLen, queryOffset: queryOffset,
            kvLen: slidingKVLen, window: textCfg.slidingWindow, dtype: dtype)

        // Run each drafter layer with the appropriate shared-KV + mask.
        for (i, layer) in model.layers.enumerated() {
            let layerType = textCfg.layerTypes[i]
            let kv: (MLXArray, MLXArray)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch layerType {
            case "full_attention":
                kv = sharedKV.fullAttention
                mask = fullMask
            case "sliding_attention":
                kv = sharedKV.slidingAttention
                mask = slidingMask
            default:
                preconditionFailure(
                    "Gemma4AssistantDraftModel: unexpected layerType '\(layerType)' at "
                    + "layer \(i). Compat validation should have rejected this."
                )
            }
            let (out, _, _) = layer(
                h,
                mask: mask,
                cache: nil,
                perLayerInput: nil,
                sharedKV: kv,
                positionOffset: positionOffset
            )
            h = out
        }

        h = model.norm(h)
        let lastHidden = postProjection(h)
        let logits = applyLMHead(h)
        return (lastHidden, logits)
    }

    /// Dispatch the LM head: masked-centroid if `useOrderedEmbeddings`,
    /// tied otherwise (unless an explicit `lm_head` is present).
    /// No softcap.
    private func applyLMHead(_ hidden: MLXArray) -> MLXArray {
        if let maskedEmbedder {
            return maskedEmbedder(
                hiddenStates: hidden,
                lmHeadWeight: model.embedTokens.weight
            )
        }
        if let lmHead {
            return lmHead(hidden)
        }
        // Tied: project hidden through the (transposed) token embedding.
        return model.embedTokens.asLinear(hidden)
    }

    // MARK: - Compatibility validation

    /// Fail-fast on every drafter/target mismatch with the field name in
    /// the error. Called once at bind time.
    private func validateCompatibility(with target: Gemma4TextModel) throws {
        let drafterT = config.textConfig
        let targetCfg = target.configuration

        // 1. Backbone hidden size must match target hidden size (pre/post
        //    projection shapes depend on this).
        guard config.backboneHiddenSize == targetCfg.hiddenSize else {
            throw Gemma4MTPError.incompatibleDrafter(
                field: "backboneHiddenSize",
                drafter: "\(config.backboneHiddenSize)",
                target: "\(targetCfg.hiddenSize)"
            )
        }

        // 2. Vocab sizes must match (drafter's LM head produces logits over
        //    the target's vocab).
        guard drafterT.vocabSize == targetCfg.vocabSize else {
            throw Gemma4MTPError.incompatibleDrafter(
                field: "vocabSize",
                drafter: "\(drafterT.vocabSize)",
                target: "\(targetCfg.vocabSize)"
            )
        }

        // 3. Every drafter layer_type must be one of the two known types.
        for (i, lt) in drafterT.layerTypes.enumerated() {
            guard lt == "full_attention" || lt == "sliding_attention" else {
                throw Gemma4MTPError.incompatibleDrafter(
                    field: "layerTypes[\(i)]",
                    drafter: lt,
                    target: "full_attention|sliding_attention"
                )
            }
        }

        // 4. K=V compatibility: applies only when the drafter has at least
        //    one full-attention layer AND attention_k_eq_v is true.
        let drafterHasFullAttn =
            drafterT.layerTypes.contains("full_attention")
        if drafterHasFullAttn && drafterT.attentionKeqV {
            guard targetCfg.attentionKeqV else {
                throw Gemma4MTPError.incompatibleDrafter(
                    field: "attentionKeqV",
                    drafter: "true",
                    target: "false"
                )
            }
            guard drafterT.numGlobalKeyValueHeads == targetCfg.numGlobalKeyValueHeads
            else {
                throw Gemma4MTPError.incompatibleDrafter(
                    field: "numGlobalKeyValueHeads",
                    drafter: "\(drafterT.numGlobalKeyValueHeads.map(String.init) ?? "nil")",
                    target: "\(targetCfg.numGlobalKeyValueHeads.map(String.init) ?? "nil")"
                )
            }
        }

        // 5. Drafter must be fully KV-shared by construction.
        guard drafterT.numKvSharedLayers == drafterT.numHiddenLayers else {
            throw Gemma4MTPError.incompatibleDrafter(
                field: "numKvSharedLayers",
                drafter: "\(drafterT.numKvSharedLayers)",
                target: "\(drafterT.numHiddenLayers)"
            )
        }
    }
}
