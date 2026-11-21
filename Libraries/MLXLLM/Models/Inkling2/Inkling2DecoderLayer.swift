// Inkling 2 decoder layer: block attention residual inputs, attention, the
// post-attention and post-MLP short convolutions and the routed MLP; loop
// layers also take messages from the previous pass. Reference: transformers
// PR 51050 `Inkling2Loop{Sliding,Full}{,Loop}DecoderLayer` (Apache-2.0).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Message gates of a loop layer (`controllers.{attn,k,v,mlp}`).
final class Inkling2Controllers: Module {
    @ModuleInfo(key: "attn") var query: Inkling2MessageGate
    @ModuleInfo(key: "k") var key: Inkling2MessageGate
    @ModuleInfo(key: "v") var value: Inkling2MessageGate
    @ModuleInfo(key: "mlp") var mlp: Inkling2MessageGate

    init(hidden: Int, rank: Int) {
        _query.wrappedValue = Inkling2MessageGate(hidden: hidden, rank: rank)
        _key.wrappedValue = Inkling2MessageGate(hidden: hidden, rank: rank)
        _value.wrappedValue = Inkling2MessageGate(hidden: hidden, rank: rank)
        _mlp.wrappedValue = Inkling2MessageGate(hidden: hidden, rank: rank)
    }

    var gates: [Inkling2MessageGate] { [query, key, value, mlp] }
}

final class Inkling2DecoderLayer: Module {
    let index: Int
    let isFull: Bool
    let isBlockBoundary: Bool

    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2BlockAttentionResidual?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "attn") var attention: Inkling2Attention
    @ModuleInfo(key: "mlp") var mlp: Inkling2SparseMoE
    /// Sliding layers normalize one attention input; full layers normalize q, k and v separately.
    @ModuleInfo(key: "attn_norm") var inputNorm: Inkling2RMSNorm?
    @ModuleInfo(key: "q_input_norm") var queryNorm: Inkling2RMSNorm?
    @ModuleInfo(key: "k_input_norm") var keyNorm: Inkling2RMSNorm?
    @ModuleInfo(key: "v_input_norm") var valueNorm: Inkling2RMSNorm?
    @ModuleInfo(key: "mlp_norm") var mlpNorm: Inkling2RMSNorm
    @ModuleInfo(key: "attn_sconv") var postAttentionConvolution: Inkling2ShortConvolution
    @ModuleInfo(key: "mlp_sconv") var postMLPConvolution: Inkling2ShortConvolution
    @ModuleInfo(key: "controllers") var controllers: Inkling2Controllers?

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        invalidatePreparedState()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    /// `LanguageModel.prepare()` for this layer: concatenated attention
    /// projections and message gates.
    func prepareForInference() {
        attention.prepareDecodeProjections()
        for gate in controllers?.gates ?? [] { gate.prepareFusedGates() }
    }

    func invalidatePreparedState() {
        attention.invalidateDecodeProjections()
        for gate in controllers?.gates ?? [] { gate.invalidateFusedGates() }
    }

    /// Prefill chunks of fewer than 64 tokens take MLX's per-row expert
    /// kernels, longer ones the sorted tiled route, so a token's expert output
    /// would depend on its chunk's length. Zero tokens pad short prefill
    /// chunks to 64 rows: every chunk then takes the sorted route, where a
    /// token's result does not depend on the tokens beside it
    /// (Inkling2InvarianceProbeTests). Row-exact steps are unchanged.
    static let prefillExpertMinimumTokens = 64

    static func invariantPrefillExperts(_ mlp: Inkling2SparseMoE, _ x: MLXArray) -> MLXArray {
        let tokens = x.size / x.dim(-1)
        guard !Inkling2RowExact.active, Inkling2RowExact.invariantPrefill,
            tokens < prefillExpertMinimumTokens
        else { return mlp(x) }
        let flat = x.reshaped(tokens, x.dim(-1))
        let padded = concatenated(
            [
                flat,
                MLXArray.zeros(
                    [prefillExpertMinimumTokens - tokens, x.dim(-1)], dtype: x.dtype),
            ], axis: 0)
        let out = mlp(padded.expandedDimensions(axis: 0))
        return out[0, 0 ..< tokens].reshaped(x.shape)
    }

    init(_ c: Inkling2TextConfiguration, index: Int, expertFormat: Inkling2ExpertFormat) {
        self.index = index
        isFull = !c.isSliding(layer: index)
        isBlockBoundary = c.isBlockBoundary(index)
        let hidden = c.hiddenSize
        _attentionInput.wrappedValue =
            index > 0
            ? Inkling2BlockAttentionResidual(hidden: hidden, queries: isFull ? 3 : 1) : nil
        _mlpInput.wrappedValue = Inkling2BlockAttentionResidual(hidden: hidden, queries: 1)
        _attention.wrappedValue = Inkling2Attention(c, layer: index)
        _mlp.wrappedValue = Inkling2SparseMoE(c, format: expertFormat)
        if isFull {
            _queryNorm.wrappedValue = Inkling2RMSNorm(dimensions: hidden, eps: c.rmsNormEps)
            _keyNorm.wrappedValue = Inkling2RMSNorm(dimensions: hidden, eps: c.rmsNormEps)
            _valueNorm.wrappedValue = Inkling2RMSNorm(dimensions: hidden, eps: c.rmsNormEps)
        } else {
            _inputNorm.wrappedValue = Inkling2RMSNorm(dimensions: hidden, eps: c.rmsNormEps)
        }
        _mlpNorm.wrappedValue = Inkling2RMSNorm(dimensions: hidden, eps: c.rmsNormEps)
        _postAttentionConvolution.wrappedValue = Inkling2ShortConvolution(
            channels: hidden, kernel: c.convKernelSize)
        _postMLPConvolution.wrappedValue = Inkling2ShortConvolution(
            channels: hidden, kernel: c.convKernelSize)
        _controllers.wrappedValue =
            c.isLoopLayer(index) ? Inkling2Controllers(hidden: hidden, rank: c.loop.gateRank) : nil
    }

    /// `hidden`: the partial block state entering this layer (model dtype).
    /// `blocks`: completed block states. `messages`: `[B, L, 4, hidden]` FP32
    /// (q, k, v, mlp) in loop passes after the first. Returns the next partial
    /// state in the model dtype.
    func callAsFunction(
        _ hidden: MLXArray, blocks: [MLXArray], messages: MLXArray?, slot: Int,
        state: Inkling2StepState
    ) -> MLXArray {
        if Inkling2RowExact.active,
            let y = Inkling2Kernels.decoderLayer(
                self, hidden, blocks: blocks, messages: messages, slot: slot, state: state)
        {
            return y
        }
        let dtype = hidden.dtype
        let message: (Int) -> MLXArray? = { i in messages?[.ellipsis, i, 0...] }

        let queryInput: MLXArray
        let keyInput: MLXArray
        let valueInput: MLXArray
        if isFull {
            var q = hidden
            var k = hidden
            var v = hidden
            if let attentionInput {
                let mixed = attentionInput(blocks: blocks, partial: hidden)
                (q, k, v) = (
                    mixed[.ellipsis, 0, 0...], mixed[.ellipsis, 1, 0...], mixed[.ellipsis, 2, 0...]
                )
            }
            if let controllers, let mq = message(0), let mk = message(1), let mv = message(2) {
                q = controllers.query(q, message: mq)
                k = controllers.key(k, message: mk)
                v = controllers.value(v, message: mv)
            }
            queryInput = queryNorm!(q).asType(dtype)
            keyInput = keyNorm!(k).asType(dtype)
            valueInput = valueNorm!(v).asType(dtype)
        } else {
            let x =
                attentionInput.map { $0(blocks: blocks, partial: hidden)[.ellipsis, 0, 0...] }
                ?? hidden
            let norm = inputNorm!
            if let controllers, let mq = message(0), let mk = message(1), let mv = message(2) {
                queryInput = norm(controllers.query(x, message: mq)).asType(dtype)
                keyInput = norm(controllers.key(x, message: mk)).asType(dtype)
                valueInput = norm(controllers.value(x, message: mv)).asType(dtype)
            } else {
                queryInput = norm(x).asType(dtype)
                keyInput = queryInput
                valueInput = queryInput
            }
        }

        Inkling2Diagnostics.section?("inputs", [queryInput, keyInput, valueInput])
        let attended = attention(
            query: queryInput, key: keyInput, value: valueInput, slot: slot, state: state)
        Inkling2Diagnostics.section?("attention", [attended])
        let (convolved, attentionState) = postAttentionConvolution(
            attended, state: state.convState(slot: slot, .postAttention))
        state.setConvState(slot: slot, .postAttention, attentionState, input: attended)

        var residual = convolved.asType(.float32)
        var mlpBlocks = blocks
        if isBlockBoundary {
            mlpBlocks.append(hidden)
        } else {
            residual = hidden.asType(.float32) + residual
        }
        var mlpIn = mlpInput(blocks: mlpBlocks, partial: residual)[.ellipsis, 0, 0...]
        if let controllers, let mm = message(3) {
            mlpIn = controllers.mlp(mlpIn, message: mm)
        }
        let mlpNormed = mlpNorm(mlpIn).asType(dtype)
        Inkling2Diagnostics.section?("post-attention", [mlpNormed, residual])
        let routed = Self.invariantPrefillExperts(mlp, mlpNormed)
        Inkling2Diagnostics.section?("experts", [routed])
        let (mixed, mlpState) = postMLPConvolution(
            routed, state: state.convState(slot: slot, .postMLP))
        state.setConvState(slot: slot, .postMLP, mlpState, input: routed)
        let output = (residual + mixed.asType(.float32)).asType(dtype)
        Inkling2Diagnostics.section?("post-mlp", [output, mlpState])
        return output
    }
}
