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

internal struct Gemma4DrafterMasks {
    let full: MLXFast.ScaledDotProductAttentionMaskMode
    let sliding: MLXFast.ScaledDotProductAttentionMaskMode
}

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
        let h = preProjection(inputsEmbeds)
        let masks = makeMasks(
            queryLen: h.dim(1),
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            dtype: h.dtype)
        return forwardProjected(
            h,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            masks: masks)
    }

    internal func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset,
        masks: Gemma4DrafterMasks
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let h = preProjection(inputsEmbeds)
        return forwardProjected(
            h,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            masks: masks)
    }

    internal func makeMasks(
        queryLen: Int,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset,
        dtype: DType
    ) -> Gemma4DrafterMasks {
        let textCfg = config.textConfig
        let queryOffset: Int
        switch positionOffset {
        case .scalar(let v): queryOffset = v
        case .batch(let arr): queryOffset = Int(arr[0].item(Int32.self))
        }

        let fullKVLen = sharedKV.fullAttention.0.dim(2)
        let slidingKVLen = sharedKV.slidingAttention.0.dim(2)

        let fullMask = DrafterMasks.bidirectionalFull(
            queryLen: queryLen, kvLen: fullKVLen, dtype: dtype)
        let slidingMask = DrafterMasks.bidirectionalSWA(
            queryLen: queryLen, queryOffset: queryOffset,
            kvLen: slidingKVLen, window: textCfg.slidingWindow, dtype: dtype)
        return Gemma4DrafterMasks(full: fullMask, sliding: slidingMask)
    }

    private func forwardProjected(
        _ projected: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset,
        masks: Gemma4DrafterMasks
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let textCfg = config.textConfig
        var h = projected
        // Run each drafter layer with the appropriate shared-KV + mask.
        for (i, layer) in model.layers.enumerated() {
            let layerType = textCfg.layerTypes[i]
            let kv: (MLXArray, MLXArray)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch layerType {
            case "full_attention":
                kv = sharedKV.fullAttention
                mask = masks.full
            case "sliding_attention":
                kv = sharedKV.slidingAttention
                mask = masks.sliding
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

    // MARK: - Weight sanitization

    /// Sanitize a raw drafter checkpoint dictionary into the form the
    /// drafter's modules expect.
    ///
    /// Rules:
    /// - Cast `masked_embedding.token_ordering` to int32 (HF ships it as int64).
    /// - Drop `lm_head.weight` when the drafter config has
    ///   `tieWordEmbeddings == true` (the LM head is tied to `embed_tokens`;
    ///   a present `lm_head.weight` in the checkpoint is redundant).
    /// - Throw on unexpected `k_proj` / `v_proj` / `k_norm` / `v_norm`
    ///   weights: every drafter layer is kv-shared by construction, so
    ///   those modules aren't instantiated. A stray weight indicates a
    ///   checkpoint/config mismatch and should surface loudly rather than
    ///   be silently dropped.
    ///
    /// - Throws: `Gemma4MTPError.incompatibleDrafter(field:...)` with a
    ///   descriptive field name when an unexpected K/V weight is present.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        let forbiddenSubstrings = [
            ".self_attn.k_proj",
            ".self_attn.v_proj",
            ".self_attn.k_norm",
            ".self_attn.v_norm",
        ]

        for (key, value) in weights {
            // Fail loud on forbidden K/V weights.
            for substring in forbiddenSubstrings {
                if key.contains(substring) {
                    throw Gemma4MTPError.incompatibleDrafter(
                        field: key,
                        drafter: "(unexpected — drafter layers are kv-shared)",
                        target: "(should not be present)"
                    )
                }
            }

            // Drop lm_head.weight when tied.
            if key == "lm_head.weight" && config.textConfig.tieWordEmbeddings {
                continue
            }

            // Cast masked_embedding.token_ordering to int32.
            if key == "masked_embedding.token_ordering" {
                out[key] = value.asType(.int32)
                continue
            }

            out[key] = value
        }

        return out
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

    // MARK: - Loading

    /// Load a Gemma 4 MTP drafter from a local directory containing
    /// `config.json` and one or more `*.safetensors` files.
    ///
    /// - Parameter directory: local directory containing the drafter
    ///   checkpoint (e.g. pre-downloaded via `huggingface-cli`).
    /// - Returns: a ready-to-use drafter with weights loaded and evaluated.
    /// - Throws: file I/O errors, JSON decoding errors, or
    ///   `Gemma4MTPError.incompatibleDrafter` from `sanitize`.
    public static func load(from directory: URL) async throws -> Gemma4AssistantDraftModel {
        // Decode the drafter config.
        let configURL = directory.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: configData)

        // Construct the drafter (random init).
        let drafter = Gemma4AssistantDraftModel(config: config)

        // Collect all safetensors files in the directory.
        var weights = [String: MLXArray]()
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)
        else {
            throw NSError(
                domain: "Gemma4AssistantDraftModel", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Could not enumerate drafter directory: \(directory.path)"])
        }
        let urls = enumerator.allObjects.compactMap { $0 as? URL }
        for url in urls where url.pathExtension == "safetensors" {
            let (shardWeights, _) = try loadArraysAndMetadata(url: url)
            for (k, v) in shardWeights {
                weights[k] = v
            }
        }

        // Run drafter sanitize (throws on unexpected K/V weights).
        let sanitized = try drafter.sanitize(weights: weights)

        // Apply weights.
        let params = ModuleParameters.unflattened(sanitized)
        try drafter.update(parameters: params, verify: [.all])
        eval(drafter)

        return drafter
    }

    /// Load a Gemma 4 MTP drafter from a remote model ID via a `Downloader`.
    ///
    /// Downloads `config.json` and `*.safetensors` files (tokenizer files are
    /// skipped — drafters reuse the target's tokenizer at generation time).
    ///
    /// - Parameters:
    ///   - downloader: any `Downloader` (e.g. `HubClient`).
    ///   - id: the model identifier (e.g. `"mlx-community/gemma-4-E4B-it-assistant-bf16"`).
    ///   - revision: the revision to download (defaults to the downloader's
    ///     default, typically `"main"`).
    ///   - useLatest: whether to bypass the downloader's cache.
    ///   - progressHandler: optional progress callback.
    /// - Returns: a ready-to-use drafter.
    public static func load(
        from downloader: any Downloader,
        id: String,
        revision: String? = nil,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> Gemma4AssistantDraftModel {
        let directory = try await downloader.download(
            id: id,
            revision: revision,
            matching: ["*.safetensors", "config.json"],
            useLatest: useLatest,
            progressHandler: progressHandler
        )
        return try await load(from: directory)
    }
}
