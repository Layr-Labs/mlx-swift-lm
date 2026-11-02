import MLX
import MLXLMCommon
import MLXNN

/// Consecutive prompt chunks of one request built as one graph, layer by layer: every
/// chunk's mixer phase, then every chunk's MLP phase, each chunk with its own shapes.
public enum Qwen35PromptWavefront {
    public enum Phase: Sendable {
        case mixer
        case feedForward
    }

    /// Whether `model` builds wavefronts: dense MLPs and a plain token embedding.
    public static func supports(_ model: Qwen35TextModel) -> Bool {
        !(model.model.embedTokens is HadamardQuantizedEmbedding)
            && model.model.layers.allSatisfy { $0.mlp is Qwen3NextMLP }
    }

    public static func supports(_ model: Qwen35Model) -> Bool { supports(model.languageModel) }

    /// `prefill` on the wrapped text model.
    public static func prefill(
        _ model: Qwen35Model, chunks: [MLXArray], finalChunk: Bool, caches: [KVCache],
        recurrentState: CBv2RecurrentStateEvaluation,
        phaseBuilt: (_ phase: Phase, _ layer: Int, _ chunk: Int) -> Void
    ) -> [MLXArray] {
        prefill(
            model.languageModel, chunks: chunks, finalChunk: finalChunk, caches: caches,
            recurrentState: recurrentState, phaseBuilt: phaseBuilt)
    }

    /// Each chunk reads the KV and recurrent state the chunk before it left at each layer and
    /// returns what its own forward returns (`cbv2RecurrentPrefill`, or the last chunk's logits).
    public static func prefill(
        _ model: Qwen35TextModel, chunks: [MLXArray], finalChunk: Bool, caches: [KVCache],
        recurrentState: CBv2RecurrentStateEvaluation,
        phaseBuilt: (_ phase: Phase, _ layer: Int, _ chunk: Int) -> Void
    ) -> [MLXArray] {
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
                phaseBuilt(.mixer, index, chunk)
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
                phaseBuilt(.feedForward, index, chunk)
            }
        }
        return chunks.indices.map { chunk in
            switch streams[chunk] {
            case .plain(let hidden):
                return hidden[0..., -1, 0 ..< 1]
            case .fused(let residual, let pending):
                let normalized = scoped(true, chunks[chunk]) {
                    Qwen35DecoderLayer.residualNorm(residual, pending, inner.norm).1
                }
                return Qwen35BatchDecodeProjection.withModel(chunks[chunk]) {
                    model.lmHead.map { Qwen35BatchDecodeProjection.linear($0, normalized) }
                        ?? Qwen35BatchDecodeProjection.embedding(inner.embedTokens, normalized)
                }
            }
        }
    }

    /// The prompt's last chunk builds under `Qwen35TextModel.cbv2Forward`'s scopes.
    private static func scoped<T>(_ fused: Bool, _ tokens: MLXArray, _ body: () -> T) -> T {
        guard fused else { return body() }
        return Qwen35BatchDecodeProjection.withModel(tokens) {
            Qwen35BatchDecodeScope.$deferRowParallelCast.withValue(true, operation: body)
        }
    }
}
