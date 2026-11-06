import MLX
import MLXLMCommon
import MLXNN

/// Consecutive prompt chunks of one request built as one graph, layer by layer: every chunk's
/// mixer phase, then every chunk's MLP phase (`phaseBuilt` after each), each with its own shapes.
public enum Qwen35PromptWavefront {
    /// Whether `model` builds wavefronts: dense MLPs and a plain token embedding.
    public static func supports(_ model: Qwen35TextModel) -> Bool {
        !(model.model.embedTokens is HadamardQuantizedEmbedding)
            && model.model.layers.allSatisfy { $0.mlp is Qwen3NextMLP }
    }

    public static func supports(_ model: Qwen35Model) -> Bool { supports(model.languageModel) }

    /// `prefill` on the wrapped text model.
    public static func prefill(
        _ model: Qwen35Model, chunks: [MLXArray], finalChunk: Bool, caches: [KVCache],
        recurrentState: CBv2RecurrentStateEvaluation, phaseBuilt: () -> Void
    ) -> [MLXArray] {
        prefill(
            model.languageModel, chunks: chunks, finalChunk: finalChunk, caches: caches,
            recurrentState: recurrentState, phaseBuilt: phaseBuilt)
    }

    /// Each chunk reads the KV and recurrent state the chunk before it left at each layer and
    /// returns what its own forward returns (`cbv2RecurrentPrefill`, or the last chunk's logits).
    public static func prefill(
        _ model: Qwen35TextModel, chunks: [MLXArray], finalChunk: Bool, caches: [KVCache],
        recurrentState: CBv2RecurrentStateEvaluation, phaseBuilt: () -> Void
    ) -> [MLXArray] {
        let streams = layered(
            model, chunks: chunks, finalChunk: finalChunk, caches: caches,
            recurrentState: recurrentState, phaseBuilt: phaseBuilt)
        return chunks.indices.map { chunk in
            switch streams[chunk] {
            case .plain(let hidden):
                return hidden[0..., -1, 0 ..< 1]
            case .fused(let residual, let pending):
                return head(model, residual, pending, tokens: chunks[chunk]).logits
            }
        }
    }

    /// `prefill` on the wrapped text model, with every chunk's hidden rows.
    public static func prefillWithHidden(
        _ model: Qwen35Model, chunks: [MLXArray], finalChunk: Bool, caches: [KVCache],
        recurrentState: CBv2RecurrentStateEvaluation, phaseBuilt: () -> Void
    ) -> (outputs: [MLXArray], hidden: [MLXArray]) {
        prefillWithHidden(
            model.languageModel, chunks: chunks, finalChunk: finalChunk, caches: caches,
            recurrentState: recurrentState, phaseBuilt: phaseBuilt)
    }

    /// `prefill`'s graph, also returning each chunk's last decoder layer output (`[1, L, H]`,
    /// before the final norm: the target hidden an MTP assistant's history takes, as
    /// `Qwen35TextModel.cbv2ForwardWithHidden` returns it). The outputs are `prefill`'s.
    public static func prefillWithHidden(
        _ model: Qwen35TextModel, chunks: [MLXArray], finalChunk: Bool, caches: [KVCache],
        recurrentState: CBv2RecurrentStateEvaluation, phaseBuilt: () -> Void
    ) -> (outputs: [MLXArray], hidden: [MLXArray]) {
        let streams = layered(
            model, chunks: chunks, finalChunk: finalChunk, caches: caches,
            recurrentState: recurrentState, phaseBuilt: phaseBuilt)
        var outputs: [MLXArray] = []
        var hidden: [MLXArray] = []
        for chunk in chunks.indices {
            switch streams[chunk] {
            case .plain(let rows):
                outputs.append(rows[0..., -1, 0 ..< 1])
                hidden.append(rows)
            case .fused(let residual, let pending):
                let (rows, logits) = head(model, residual, pending, tokens: chunks[chunk])
                outputs.append(logits)
                hidden.append(rows)
            }
        }
        return (outputs, hidden)
    }

    /// The final norm and head of the prompt's last chunk: its hidden rows and logits.
    private static func head(
        _ model: Qwen35TextModel, _ residual: MLXArray, _ pending: MLXArray?, tokens: MLXArray
    ) -> (hidden: MLXArray, logits: MLXArray) {
        let inner = model.model
        let (hidden, normalized) = scoped(true, tokens) {
            Qwen35DecoderLayer.residualNorm(residual, pending, inner.norm)
        }
        let logits = Qwen35BatchDecodeProjection.withModel(tokens) {
            model.lmHead.map { Qwen35BatchDecodeProjection.linear($0, normalized) }
                ?? Qwen35BatchDecodeProjection.embedding(inner.embedTokens, normalized)
        }
        return (hidden, logits)
    }

    /// Every chunk's stream after the last decoder layer.
    private static func layered(
        _ model: Qwen35TextModel, chunks: [MLXArray], finalChunk: Bool, caches: [KVCache],
        recurrentState: CBv2RecurrentStateEvaluation, phaseBuilt: () -> Void
    ) -> [Qwen35DecoderLayer.PromptStream] {
        let inner = model.model
        precondition(
            supports(model) && !chunks.isEmpty
                && chunks.allSatisfy { $0.ndim == 2 && $0.dim(0) == 1 && $0.dim(1) > 0 },
            "Qwen35 prompt wavefront needs dense text chunks of one request")
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        precondition(
            attending.count == inner.layers.filter({ !$0.isLinear }).count,
            "Qwen35 CBv2 requires only full-attention caches")
        let last = chunks.count - 1
        let fused = { (chunk: Int) in finalChunk && chunk == last }
        var streams: [Qwen35DecoderLayer.PromptStream] = chunks.indices.map { chunk in
            let embedded = inner.embedTokens(chunks[chunk])
            return fused(chunk) ? .fused(residual: embedded, pending: nil) : .plain(embedded)
        }
        var attentionIndex = 0
        for (index, layer) in inner.layers.enumerated() {
            var cache: (any CBv2AttendingLayerCache)?
            if !layer.isLinear {
                cache = attending[attentionIndex]
                precondition(
                    cache!.kind.modelLayerIndex == nil || cache!.kind.modelLayerIndex == index,
                    "Qwen35 CBv2 attention cache mapped to the wrong model layer")
                attentionIndex += 1
            }
            var carry = layer.isLinear ? recurrentState.inputState(modelLayerIndex: index) : nil
            var mixed: [MLXArray] = []
            for chunk in chunks.indices {
                let (stream, output) = scoped(fused(chunk), chunks[chunk]) {
                    layer.promptMixer(
                        streams[chunk], modelLayerIndex: index, attentionCache: cache,
                        carry: &carry)
                }
                streams[chunk] = stream
                mixed.append(output)
                phaseBuilt()
            }
            if layer.isLinear, let carry, let conv = carry.conv, let ssm = carry.ssm {
                do {
                    try recurrentState.stage(modelLayerIndex: index, conv: conv, ssm: ssm)
                } catch {
                    preconditionFailure(
                        "Qwen35 prompt wavefront stage failed at layer \(index): \(error)")
                }
            }
            for chunk in chunks.indices {
                streams[chunk] = scoped(fused(chunk), chunks[chunk]) {
                    layer.promptFeedForward(streams[chunk], mixed: mixed[chunk])
                }
                phaseBuilt()
            }
        }
        return streams
    }

    /// The prompt's last chunk builds under `Qwen35TextModel.cbv2Forward`'s scopes.
    private static func scoped<T>(_ fused: Bool, _ tokens: MLXArray, _ body: () -> T) -> T {
        guard fused else { return body() }
        return Qwen35BatchDecodeProjection.withModel(tokens) {
            Qwen35BatchDecodeScope.$deferRowParallelCast.withValue(true, operation: body)
        }
    }
}
