// Inkling 2 (`inkling2_loop`) text model: normed embedding, 24 stored layers
// with layers 6..<18 run three times (passes 2 and 3 restart from the pre-loop
// state and take messages from the previous pass), block attention residuals,
// final residual mix, norm, unembedding and logit softcap. Reference:
// transformers PR 51050 `Inkling2LoopTextModel` (Apache-2.0).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// No-cache full-recompute state: every call attends only its own tokens and
/// keeps no K/V and no convolution state. This is the test oracle path.
public final class Inkling2FullSequenceState: Inkling2StepState {
    public let positions: MLXArray

    /// `positions`: `[B, L]` or `[1, L]`; defaults to `0 ..< L`.
    public init(length: Int, positions: MLXArray? = nil) {
        self.positions =
            positions ?? MLXArray((0 ..< length).map { Int32($0) }).reshaped(1, length)
    }

    public func convState(slot: Int, _ kind: Inkling2ConvKind) -> MLXArray? { nil }
    public func setConvState(
        slot: Int, _ kind: Inkling2ConvKind, _ state: MLXArray, input: MLXArray
    ) {}

    public func attend(slot: Int, _ inputs: Inkling2AttentionInputs) -> MLXArray {
        Inkling2ReferenceAttention.attend(
            inputs, keys: inputs.keys, values: inputs.values, keyPositions: inputs.positions)
    }
}

/// Hidden states captured for the drafter: the partial state entering each
/// requested visit (`[B, L, hidden]`, model dtype), and the final-norm output
/// under the key `visitCount`.
public typealias Inkling2CapturedHidden = [Int: MLXArray]

public final class Inkling2TextModel: Module {
    public let configuration: Inkling2TextConfiguration
    @ModuleInfo(key: "embed") var embedding: Embedding
    @ModuleInfo(key: "embed_norm") var embeddingNorm: Inkling2RMSNorm
    let layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageAttentionResidual
    @ModuleInfo(key: "final_input") var finalInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "norm") var norm: Inkling2RMSNorm

    public init(_ c: Inkling2TextConfiguration, expertFormat: Inkling2ExpertFormat = .dense) {
        configuration = c
        _embedding.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _embeddingNorm.wrappedValue = Inkling2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        layers = (0 ..< c.hiddenLayers).map {
            Inkling2DecoderLayer(c, index: $0, expertFormat: expertFormat)
        }
        _messageInput.wrappedValue = Inkling2MessageAttentionResidual(
            loopLayers: c.loopLayerCount, hidden: c.hiddenSize)
        _finalInput.wrappedValue = Inkling2BlockAttentionResidual(hidden: c.hiddenSize, queries: 1)
        _norm.wrappedValue = Inkling2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func invalidateDecodeProjections() {
        for layer in layers { layer.invalidatePreparedState() }
    }

    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        invalidateDecodeProjections()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    /// Normed token embeddings in the model dtype.
    public func embed(_ tokens: MLXArray) -> MLXArray {
        let raw = embedding(tokens)
        return embeddingNorm(raw).asType(raw.dtype)
    }

    /// Runs every visit. `inputs`: normed embeddings `[B, L, hidden]`. Returns
    /// the final-norm hidden state in the model dtype and the requested captures.
    /// Runs one visit alone: `hidden` is the partial state entering it,
    /// `blocks` the completed block states, `messages` (`[B, L, 4, hidden]`)
    /// the previous pass's messages in later loop passes. Oracle checks feed
    /// reference inputs here to test each application in isolation.
    public func runVisit(
        _ visit: Int, hidden: MLXArray, blocks: [MLXArray], messages: MLXArray?,
        state: Inkling2StepState
    ) -> MLXArray {
        let (layer, pass) = configuration.visits[visit]
        return layers[layer](
            hidden, blocks: blocks, messages: messages,
            slot: configuration.slot(layer: layer, pass: pass), state: state)
    }

    /// Messages of every loop layer, `[B, L, loop layers, 4, hidden]`, from the
    /// previous pass's block states (after the pre-loop ones) and final partial state.
    public func loopMessages(previousBlocks: [MLXArray], previousHidden: MLXArray) -> MLXArray {
        messageInput(blocks: previousBlocks, partial: previousHidden)
    }

    /// `norm(final_input(blocks, hidden))` in the model dtype of `hidden`.
    public func finalNorm(blocks: [MLXArray], hidden: MLXArray, dtype: DType) -> MLXArray {
        norm(finalInput(blocks: blocks, partial: hidden)[.ellipsis, 0, 0...]).asType(dtype)
    }

    /// `visitOutput` receives each visit's output as it is produced.
    public func callAsFunction(
        embeddings inputs: MLXArray, state: Inkling2StepState, capture: Set<Int> = [],
        visitOutput: ((Int, MLXArray) -> Void)? = nil
    ) -> (hidden: MLXArray, captured: Inkling2CapturedHidden) {
        let c = configuration
        var captured = Inkling2CapturedHidden()
        var visit = 0
        var hidden = inputs
        var blocks: [MLXArray] = []

        func run(_ layer: Int, pass: Int, messages: MLXArray?) {
            if capture.contains(visit) { captured[visit] = hidden }
            let decoder = layers[layer]
            let next = decoder(
                hidden, blocks: blocks, messages: messages,
                slot: c.slot(layer: layer, pass: pass), state: state)
            if decoder.isBlockBoundary { blocks.append(hidden) }
            hidden = next
            visitOutput?(visit, next)
            visit += 1
        }

        for layer in 0 ..< c.loop.coreStart { run(layer, pass: 0, messages: nil) }
        let preLoopHidden = hidden
        let preLoopBlocks = blocks
        for layer in c.loop.coreStart ..< c.loop.coreEnd { run(layer, pass: 0, messages: nil) }
        for pass in 1 ..< max(c.loop.passes, 1) {
            // The first loop block is the pre-loop partial state; skip it.
            let previousBlocks = Array(blocks[(preLoopBlocks.count + 1)...])
            let previousHidden = hidden
            hidden = preLoopHidden
            blocks = preLoopBlocks
            // Row-exact steps take every loop layer's messages from one kernel
            // launch per pass (decode kernels); otherwise each layer's messages when it
            // runs (bit-identical to slicing all of them).
            let precomputed =
                Inkling2RowExact.active
                ? Inkling2Kernels.loopMessages(
                    queries: messageInput.query, blocks: previousBlocks, partial: previousHidden)
                : nil
            for (i, layer) in (c.loop.coreStart ..< c.loop.coreEnd).enumerated() {
                let messages =
                    precomputed.map { Inkling2Kernels.plane($0, i) }
                    ?? messageInput(layer: i, blocks: previousBlocks, partial: previousHidden)
                run(layer, pass: pass, messages: messages)
            }
        }
        for layer in c.loop.coreEnd ..< c.hiddenLayers { run(layer, pass: 0, messages: nil) }

        let output = finalNorm(blocks: blocks, hidden: hidden, dtype: inputs.dtype)
        if capture.contains(c.visitCount) { captured[c.visitCount] = output }
        return (output, captured)
    }
}

public final class Inkling2Model: Module, LLMModel {
    public let configuration: Inkling2Configuration
    @ModuleInfo(key: "model") public var model: Inkling2TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    /// Paged attention used by the CBv2 path: the paged kernel by default;
    /// `Inkling2GatheredAttentionCore` is the plain-MLX reference.
    public var pagedAttentionCore: any Inkling2PagedAttentionCore = Inkling2KernelAttentionCore()
    /// Certified greedy unembedding, built on first use (`Inkling2BytesCertified.swift`).
    let certifiedHead = Inkling2CertifiedHeadCell()

    public var vocabularySize: Int { configuration.text.unpaddedVocabularySize }
    public var loraLayers: [Module] { model.layers }

    public init(_ configuration: Inkling2Configuration) {
        self.configuration = configuration
        let text = configuration.text
        _model.wrappedValue = Inkling2TextModel(text)
        _lmHead.wrappedValue = Linear(text.hiddenSize, text.unpaddedVocabularySize, bias: false)
    }

    public convenience init(text: Inkling2TextConfiguration) throws {
        let data = try JSONEncoder().encode(text)
        try self.init(JSONDecoder().decode(Inkling2Configuration.self, from: data))
    }

    /// Prefill products of this model's engine steps: calls of this many rows
    /// (chunk-invariant prefill: any chunking of a prompt, and a restore at
    /// any checkpoint boundary, gives the bits of the one-shot prefill), or
    /// nil for MLX's own kernel choice, which varies with the chunk length, so
    /// engines keep the fixed chunk grid and grid-aligned checkpoints
    /// (`cbv2Capabilities`). An engine reads it when it is built. Default:
    /// `DARKBLOOM_INKLING2_INVARIANT_PREFILL=1` gives
    /// `DARKBLOOM_INKLING2_PREFILL_BLOCK_ROWS` (64) rows, otherwise nil.
    public var invariantPrefillBlockRows: Int? =
        Inkling2RowExact.invariantPrefillDefault ? Inkling2RowExact.prefillBlockRowsDefault : nil

    /// FP32 softcapped logits for normed hidden states.
    public func logits(_ hidden: MLXArray) -> MLXArray {
        if let placeholder = certifiedLogits(hidden) { return placeholder }
        return Inkling2RowExact.withPrefill(blockRows: invariantPrefillBlockRows) {
            Inkling2Math.softcap(
                // Prefill needs one row here; 64-row calls keep it cheap and invariant.
                Inkling2RowExact.linear(lmHead, hidden, prefillBlockRows: 64).asType(.float32),
                configuration.text.finalLogitSoftcap)
        }
    }

    /// `embed_norm(embed(tokens))` in the model dtype: the hidden state entering layer 0.
    public func scaledInputEmbeddings(_ tokens: MLXArray) -> MLXArray {
        model.embed(tokens)
    }

    /// One forward over `tokens` or already-normed `inputEmbedding` with `state`
    /// holding K/V history and convolution states.
    public func forward(
        _ tokens: MLXArray?, inputEmbedding: MLXArray? = nil, state: Inkling2StepState,
        capture: Set<Int> = [], visitOutput: ((Int, MLXArray) -> Void)? = nil
    ) -> (logits: MLXArray, captured: Inkling2CapturedHidden) {
        let embeddings = inputEmbedding ?? model.embed(tokens!)
        let (hidden, captured) = model(
            embeddings: embeddings, state: state, capture: capture, visitOutput: visitOutput)
        return (logits(hidden), captured)
    }

    /// Full-recompute forward over `tokens` (or normed `inputEmbedding`), no
    /// cache: the test oracle path.
    public func forwardNoCache(
        _ tokens: MLXArray?, inputEmbedding: MLXArray? = nil, positions: MLXArray? = nil,
        capture: Set<Int> = [], visitOutput: ((Int, MLXArray) -> Void)? = nil
    ) -> (logits: MLXArray, captured: Inkling2CapturedHidden) {
        let length = inputEmbedding?.dim(1) ?? tokens!.dim(1)
        return forward(
            tokens, inputEmbedding: inputEmbedding,
            state: Inkling2FullSequenceState(length: length, positions: positions),
            capture: capture, visitOutput: visitOutput)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Paged KV only: the model keeps no contiguous K/V. Without a CBv2 engine
        // cache the forward recomputes the whole sequence it is given.
        precondition(
            cache?.isEmpty ?? true,
            "Inkling 2 serves through the CBv2 paged engine; contiguous KV caches are not supported"
        )
        return forwardNoCache(inputs).logits
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

    /// `LanguageModel.prepare()`: concatenated attention projections and
    /// message gates per layer (`Inkling2DecoderLayer.prepareForInference`).
    public func prepare() throws {
        for layer in model.layers { layer.prepareForInference() }
    }

    /// A parameter update anywhere below drops the concatenated projections
    /// until the next `prepare()` (the loader calls it after every update).
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        model.invalidateDecodeProjections()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    public var excludedCheckpointNamespaces: [String] { ["model.visual"] }

    // MARK: Checkpoint

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        checkpoint.weights = try sanitize(weights: checkpoint.weights)
        for (index, layer) in model.layers.enumerated() {
            let prefix = "model.layers.\(index).mlp.experts.w13"
            let format: Inkling2ExpertFormat
            if checkpoint.weights["\(prefix).global_scale"] != nil {
                format = .nvfp4
            } else if checkpoint.weights["\(prefix).scales"] != nil {
                format = .mxfp8
            } else {
                format = .dense
            }
            layer.mlp.experts.install(format: format)
        }
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try Inkling2Checkpoint.sanitize(weights, configuration: configuration.text)
    }
}

/// Checkpoint renames and layout conversion (transformers `conversion_mapping.py`
/// `inkling2_loop_text`), keeping the experts' native NVFP4 / MXFP8 bytes.
public enum Inkling2Checkpoint {
    public enum Failure: Error, CustomStringConvertible {
        case malformed(String)
        public var description: String {
            switch self {
            case .malformed(let what): "Inkling2 checkpoint: \(what)"
            }
        }
    }

    /// Keys loaded in FP32 like the reference's `_keep_in_fp32_modules_strict`.
    static func keepsFP32(_ key: String) -> Bool {
        key.hasSuffix("mlp.gate.weight") || key.hasSuffix("mlp.gate.bias")
            || key.contains("attn_sconv.") || key.contains("mlp_sconv.")
            || (key.contains("_sconv.weight") && !key.contains("weight_proj"))
            || key.hasSuffix("static_w")
            || key.hasSuffix("forget_weight") || key.hasSuffix("gain_weight")
            || key.hasSuffix(".query")
    }

    /// `keepVision`: pass `model.visual.*` through unchanged for the VLM wrapper;
    /// the text-only model drops it.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration c: Inkling2TextConfiguration,
        keepVision: Bool = false
    ) throws -> [String: MLXArray] {
        var out = [String: MLXArray]()
        var experts = [String: [String: MLXArray]]()
        var messageQueries: MLXArray?
        var messageKVQueries: MLXArray?

        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix("model.visual.") || key.hasPrefix("visual.") {
                if keepVision { out[key] = value }
                continue
            }
            if key.hasPrefix("model.llm.") {
                key = "model." + key.dropFirst("model.llm.".count)
            } else if key.hasPrefix("language_model.") {
                key = "model." + key.dropFirst("language_model.".count)
            }
            if key == "model.unembed.weight" {
                out["lm_head.weight"] = value[0 ..< c.unpaddedVocabularySize]
                continue
            }
            if key == "model.middle_cycle_cross_pass_queries" {
                messageQueries = value
                continue
            }
            if key == "model.middle_cycle_cross_pass_kv_queries" {
                messageKVQueries = value
                continue
            }
            if let range = key.range(of: "model.attn_res.attn_queries.") {
                guard let i = Int(key[range.upperBound...]) else {
                    throw Failure.malformed("attention query \(key)")
                }
                out["model.layers.\(i + 1).attention_input.query"] = value.asType(.float32)
                continue
            }
            if let range = key.range(of: "model.attn_res.mlp_queries.") {
                guard let i = Int(key[range.upperBound...]) else {
                    throw Failure.malformed("mlp query \(key)")
                }
                out["model.layers.\(i).mlp_input.query"] = value.asType(.float32)
                continue
            }
            if key == "model.attn_res.output_query" {
                out["model.final_input.query"] = value.asType(.float32)
                continue
            }
            if let range = key.range(of: ".mlp.experts.") {
                let prefix = String(key[..<range.lowerBound])
                experts[prefix, default: [:]][String(key[range.upperBound...])] = value
                continue
            }
            out[key] = keepsFP32(key) ? value.asType(.float32) : value
        }

        if let q = messageQueries, let kv = messageKVQueries {
            // [query, mlp] and [key, value] per loop layer -> [q, k, v, mlp].
            let loop = q.dim(0) / 2
            let qm = q.reshaped(loop, 2, -1)
            let kvm = kv.reshaped(loop, 2, -1)
            out["model.message_input.query"] = concatenated(
                [qm[0..., 0 ..< 1], kvm, qm[0..., 1 ..< 2]], axis: 1
            ).asType(.float32)
        } else if messageQueries != nil || messageKVQueries != nil {
            throw Failure.malformed("message queries need both query tensors")
        }

        for (prefix, tensors) in experts {
            let converted = try convertExperts(tensors, configuration: c, layer: prefix)
            for (name, value) in converted { out["\(prefix).mlp.experts.\(name)"] = value }
        }
        return out
    }

    /// Native bank tensors for `w13` / `w2`: the stored bytes viewed as `uint32`,
    /// the FP8 scale bytes as is, and stored row order (no re-layout).
    static func convertExperts(
        _ t: [String: MLXArray], configuration c: Inkling2TextConfiguration, layer: String
    ) throws -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for bank in ["w13", "w2"] {
            guard let weight = t["\(bank)_weight"] else {
                // Dense exports name the merged banks directly.
                if let dense = t["\(bank).weight"] { out["\(bank).weight"] = dense }
                continue
            }
            if let amax = t["\(bank)_weight_amax"] {
                guard let scale = t["\(bank)_weight_scale"],
                    let denominator = t["\(bank)_weight_outer_scale_denominator"],
                    weight.dtype == .uint8, scale.dtype == .uint8
                else { throw Failure.malformed("\(layer) \(bank) NVFP4 tensors") }
                out["\(bank).weight"] = weight.view(dtype: .uint32)
                out["\(bank).scales"] = scale
                // MLX applies global_scale / (448 * 6); the reference factor is
                // amax / denominator. With the stored denominator 2688 this is amax exactly.
                out["\(bank).global_scale"] =
                    amax.asType(.float32)
                    * (Float(448 * 6) / denominator.asType(.float32).reshaped([]))
            } else if let scale = t["\(bank)_weight_scale"] {
                guard weight.dtype == .uint8, scale.dtype == .uint8 else {
                    throw Failure.malformed("\(layer) \(bank) MXFP8 tensors")
                }
                out["\(bank).weight"] = weight.view(dtype: .uint32)
                out["\(bank).scales"] = scale
            } else {
                out["\(bank).weight"] = weight
            }
        }
        return out
    }
}
