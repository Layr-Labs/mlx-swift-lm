import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Explicit cluster-stage seam. Ordinary text, media, speculative and CBv2
/// entry points retain their own cache validation and capability contracts.
extension MiMoV26TextModel {
    /// One request, bound to this exact loaded model and its local layer order.
    /// Pages are demand-grown; this does not issue an engine paged capability.
    public func makeIndependentHistoryCaches() throws -> [MiMoV26IndependentHistoryCache] {
        guard activationDType == .bfloat16 else {
            throw MiMoV26ExecutionError.unsupportedConfiguration(
                "independent histories require BF16 precision")
        }
        let request = UUID()
        return model.layers.enumerated().map { index, layer in
            let g = layer.geometry
            return MiMoV26IndependentHistoryCache(
                modelIdentity: identity, requestIdentity: request,
                layerIndex: index, maximumContext: configuration.maxPositionEmbeddings,
                heads: g.keyValueHeads, keyWidth: g.headDim, valueWidth: g.valueHeadDim,
                dtype: activationDType, window: g.slidingWindow)
        }
    }

    /// Original scalar prefill/decode graph, with model-owned page storage.
    /// Captures remain pre-final-norm residuals, including a producer's last layer.
    public func forwardIndependentHistory(
        inputIDs: MLXArray? = nil, embeddings: MLXArray? = nil,
        caches: [MiMoV26IndependentHistoryCache], logitsStart: Int = 0,
        captureLayers: Set<Int> = []
    ) throws -> MiMoV26TextOutput {
        let dimensions = try validateIndependentInputs(
            inputIDs: inputIDs, embeddings: embeddings,
            captureLayers: captureLayers)
        guard dimensions.batch == 1, logitsStart >= 0, logitsStart < dimensions.length else {
            throw MiMoV26ExecutionError.invalidInput("scalar independent-history geometry")
        }
        try validateIndependentCaches([caches], length: dimensions.length)
        let position = caches[0].offset
        let hidden = embeddings ?? model.embedTokens(inputIDs!)
        let output = model.forward(embeddings: hidden, cache: caches, captureLayers: captureLayers)
        let projected = output.normalized[0..., logitsStart..., 0...]
        let logits = lmHead.map { $0(projected) } ?? model.embedTokens.asLinear(projected)
        return .init(
            logits: logits, normalizedHiddenStates: output.normalized,
            layerFeatures: output.features, firstPosition: position, ownerIdentity: identity)
    }

    /// One token per independent request. Dense packed projections share a
    /// broadcast M=1 operation; attention, fused norms and routed experts retain
    /// the singleton graph and physical cache order for each request.
    public func decodeIndependentHistories(
        inputIDs: MLXArray? = nil, embeddings: MLXArray? = nil,
        caches: [[MiMoV26IndependentHistoryCache]], captureLayers: Set<Int> = []
    ) throws -> [MiMoV26TextOutput] {
        let dimensions = try validateIndependentInputs(
            inputIDs: inputIDs, embeddings: embeddings,
            captureLayers: captureLayers)
        guard (1 ... 8).contains(dimensions.batch), dimensions.length == 1,
            caches.count == dimensions.batch
        else {
            throw MiMoV26ExecutionError.invalidInput(
                "independent decode requires 1...8 one-token rows")
        }
        // The batched layer loop below takes no cross-rank sums.
        guard tensorParallel == nil || dimensions.batch == 1 else {
            throw MiMoV26ExecutionError.invalidInput(
                "a tensor-parallel rank decodes one independent row at a time")
        }
        try validateIndependentCaches(caches, length: 1)
        if dimensions.batch == 1 {
            return [
                try forwardIndependentHistory(
                    inputIDs: inputIDs, embeddings: embeddings,
                    caches: caches[0], captureLayers: captureLayers)
            ]
        }
        let positions = caches.map { $0[0].offset }
        var hidden = embeddings ?? model.embedTokens(inputIDs!)
        var nextInput: MLXArray?
        var features: [Int: MLXArray] = [:]
        for (index, layer) in model.layers.enumerated() {
            let normalized =
                nextInput
                ?? MiMoV26IndependentProjection.rows(hidden, layer.inputNorm.callAsFunction)
            let attention = layer.selfAttention
            let geometry = attention.geometry
            let batch = dimensions.batch
            let queries = MiMoV26IndependentProjection.linear(attention.qProj, normalized)
                .reshaped(batch, 1, geometry.queryHeads, geometry.headDim).transposed(0, 2, 1, 3)
            let keys = MiMoV26IndependentProjection.linear(attention.kProj, normalized)
                .reshaped(batch, 1, geometry.keyValueHeads, geometry.headDim).transposed(0, 2, 1, 3)
            let values =
                MiMoV26IndependentProjection.linear(attention.vProj, normalized)
                .reshaped(batch, 1, geometry.keyValueHeads, geometry.valueHeadDim).transposed(
                    0, 2, 1, 3)
                * attention.valueScale
            let attended = (0 ..< batch).map { row -> MLXArray in
                let cache = caches[row][index]
                let mask = createAttentionMask(
                    h: normalized[row ..< row + 1], cache: cache,
                    windowSize: geometry.slidingWindow)
                let q = applyRotaryPosition(
                    attention.rope, to: queries[row ..< row + 1], cache: cache)
                let k = applyRotaryPosition(attention.rope, to: keys[row ..< row + 1], cache: cache)
                let (storedKeys, storedValues) = cache.update(
                    keys: k, values: values[row ..< row + 1])
                return MiMoV26NAXAttention.tryAttention(
                    queries: q, keys: storedKeys, values: storedValues,
                    scale: attention.scale, mask: mask, sinks: attention.attentionSinkBias)
                    ?? MLXFast.scaledDotProductAttention(
                        queries: q, keys: storedKeys, values: storedValues,
                        scale: attention.scale, mask: mask, sinks: attention.attentionSinkBias)
            }
            let joined = concatenated(attended, axis: 0).transposed(0, 2, 1, 3).reshaped(
                batch, 1, -1)
            let projected = MiMoV26IndependentProjection.linear(attention.oProj, joined)
            let nextNorm =
                index + 1 < model.layers.count ? model.layers[index + 1].inputNorm : model.norm
            var residuals: [MLXArray] = []
            var normalizedRows: [MLXArray] = []
            for row in 0 ..< batch {
                let h = hidden[row ..< row + 1]
                let a = projected[row ..< row + 1]
                if let fused = MiMoV26DecodeKernels.finishLayer(
                    h, attentionOutput: a,
                    layer: layer, nextNorm: nextNorm, enabled: model.useFusedDecodeNorms)
                {
                    residuals.append(fused.residual)
                    normalizedRows.append(fused.normalized)
                } else {
                    let residual = h + a
                    let output = residual + layer.mlp(layer.postAttentionNorm(residual))
                    residuals.append(output)
                    normalizedRows.append(nextNorm(output))
                }
            }
            hidden = concatenated(residuals, axis: 0)
            // Preserve each row's own fused carry or ordinary next norm.
            // Captured features below remain the unnormalized residual.
            nextInput = concatenated(normalizedRows, axis: 0)
            if captureLayers.contains(index) { features[index] = hidden }
        }
        let normalized =
            nextInput ?? MiMoV26IndependentProjection.rows(hidden, model.norm.callAsFunction)
        let logits =
            lmHead.map { MiMoV26IndependentProjection.linear($0, normalized) }
            ?? MiMoV26IndependentProjection.embedding(model.embedTokens, normalized)
        return (0 ..< dimensions.batch).map { row in
            .init(
                logits: logits[row ..< row + 1],
                normalizedHiddenStates: normalized[row ..< row + 1],
                layerFeatures: features.mapValues { $0[row ..< row + 1] },
                firstPosition: positions[row],
                ownerIdentity: identity)
        }
    }

    /// A `[1, 2...4]` verify window on one request's histories. Every row takes one-row
    /// decode's projections, RoPE, cache order and attention, so its logits are that step's bits.
    /// Only a tensor-parallel rank (`installTensorParallel`) takes a window; a whole model refuses.
    public func forwardVerifyWindow(
        inputIDs: MLXArray, caches: [MiMoV26IndependentHistoryCache]
    ) throws -> MiMoV26VerifyWindow {
        guard model.layers.allSatisfy({ $0.tensorParallel != nil }) else {
            throw MiMoV26ExecutionError.unsupportedConfiguration(
                "a verify window runs only on a tensor-parallel rank")
        }
        let dimensions = try validateIndependentInputs(
            inputIDs: inputIDs, embeddings: nil, captureLayers: [])
        let rows = dimensions.length
        guard dimensions.batch == 1, (2 ... MiMoV26VerifyWindow.maximumRows).contains(rows) else {
            throw MiMoV26ExecutionError.invalidInput(
                "a verify window takes 2...4 rows of one request")
        }
        try validateIndependentCaches([caches], length: rows)
        let position = caches[0].offset
        let checkpoints = caches.map { $0.checkpoint() }
        var hidden = model.embedTokens(inputIDs)
        var nextInput: MLXArray?
        var written: [[(keys: MLXArray, values: MLXArray)]] = []
        for (index, layer) in model.layers.enumerated() {
            let cache = caches[index]
            let attention = layer.selfAttention
            let g = attention.geometry
            let normalized = nextInput ?? layer.inputNorm(hidden)
            func project(_ linear: Linear, heads: Int, width: Int) -> MLXArray {
                MiMoV26RectangularDense.projection(linear, normalized, enabled: true)
                    .reshaped(1, rows, heads, width).transposed(0, 2, 1, 3)
            }
            let queries = project(attention.qProj, heads: g.queryHeads, width: g.headDim)
            let keys = project(attention.kProj, heads: g.keyValueHeads, width: g.headDim)
            let values =
                project(attention.vProj, heads: g.keyValueHeads, width: g.valueHeadDim)
                * attention.valueScale
            func row(_ x: MLXArray, _ j: Int) -> MLXArray { x[0..., 0..., j ..< (j + 1), 0...] }
            func rotated(_ x: MLXArray, _ j: Int) -> MLXArray {
                applyRotaryPosition(attention.rope, to: row(x, j), offset: .scalar(position + j))
            }
            func attend(
                _ j: Int, _ k: MLXArray, _ v: MLXArray,
                _ mask: MLXFast.ScaledDotProductAttentionMaskMode
            ) -> MLXArray {
                let q = rotated(queries, j)
                return MiMoV26NAXAttention.tryAttention(
                    queries: q, keys: k, values: v, scale: attention.scale, mask: mask,
                    sinks: attention.attentionSinkBias)
                    ?? MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v, scale: attention.scale, mask: mask,
                        sinks: attention.attentionSinkBias)
            }
            var layerRows: [(keys: MLXArray, values: MLXArray)] = []
            var attended: [MLXArray] = []
            if g.slidingWindow == nil {
                // Append-only: one append of every row, then each row reads its one-token prefix.
                layerRows = (0 ..< rows).map { (rotated(keys, $0), row(values, $0)) }
                let mask = createAttentionMask(
                    h: normalized[0..., 0 ..< 1, 0...], cache: cache, windowSize: nil)
                let (allKeys, allValues) = cache.update(
                    keys: concatenated(layerRows.map(\.keys), axis: 2),
                    values: concatenated(layerRows.map(\.values), axis: 2))
                for j in 0 ..< rows {
                    let end = position + j + 1
                    attended.append(
                        attend(
                            j, allKeys[0..., 0..., ..<end, 0...],
                            allValues[0..., 0..., ..<end, 0...],
                            mask))
                }
            } else {
                // A ring: one single-row update and one attention per row, as decode steps.
                for j in 0 ..< rows {
                    let mask = createAttentionMask(
                        h: normalized[0..., j ..< (j + 1), 0...], cache: cache,
                        windowSize: g.slidingWindow)
                    let entry = (keys: rotated(keys, j), values: row(values, j))
                    let (ringKeys, ringValues) = cache.update(
                        keys: entry.keys, values: entry.values)
                    attended.append(attend(j, ringKeys, ringValues, mask))
                    layerRows.append(entry)
                }
            }
            written.append(layerRows)
            let attendedRows = concatenated(attended, axis: 2).transposed(0, 2, 1, 3)
                .reshaped(1, rows, -1)
            let output: MLXArray
            if let tensorParallel = layer.tensorParallel {
                output = tensorParallel.sum(
                    MiMoV26RectangularDense.tensorParallelProjection(attention.oProj, attendedRows),
                    as: normalized.dtype)
            } else {
                output = MiMoV26RectangularDense.projection(
                    attention.oProj, attendedRows, enabled: true)
            }
            let nextNorm =
                index + 1 < model.layers.count ? model.layers[index + 1].inputNorm : model.norm
            if let fused = MiMoV26DecodeKernels.finishLayer(
                hidden, attentionOutput: output, layer: layer, nextNorm: nextNorm,
                enabled: model.useFusedDecodeNorms, rowLocal: true)
            {
                hidden = fused.residual
                nextInput = fused.normalized
            } else {
                let residual = hidden + output
                let x = layer.postAttentionNorm(residual)
                hidden =
                    residual
                    + (layer.tensorParallel.map {
                        layer.tensorParallelFeedForward(x, $0, rowLocal: true)
                    } ?? MiMoV26RectangularDense.mlp(layer.mlp, x, enabled: true))
                nextInput = nil
            }
        }
        let normalized = nextInput ?? model.norm(hidden)
        return MiMoV26VerifyWindow(
            logits: MiMoV26RectangularDense.readout(self, normalized, enabled: true),
            normalizedHiddenStates: normalized, firstPosition: position, caches: caches,
            checkpoints: checkpoints, written: written)
    }

    private func validateIndependentInputs(
        inputIDs: MLXArray?, embeddings: MLXArray?,
        captureLayers: Set<Int>
    ) throws -> (batch: Int, length: Int) {
        // A tensor-parallel rank's head holds its own vocabulary rows.
        guard activationDType == .bfloat16,
            hasLoadedReadoutPrecision(rows: headVocabularyRange.count),
            (inputIDs != nil) != (embeddings != nil),
            captureLayers.allSatisfy({ model.layers.indices.contains($0) })
        else {
            throw MiMoV26ExecutionError.invalidInput("independent-history input/precision/capture")
        }
        if let inputIDs {
            guard hasLoadedEmbeddingPrecision, inputIDs.ndim == 2, inputIDs.dim(0) > 0,
                inputIDs.dim(1) > 0,
                inputIDs.dtype == .int32 || inputIDs.dtype == .uint32
            else {
                throw MiMoV26ExecutionError.invalidInput("independent-history token geometry")
            }
            let minimum =
                inputIDs.dtype == .int32
                ? Int64(inputIDs.min().item(Int32.self)) : Int64(inputIDs.min().item(UInt32.self))
            let maximum =
                inputIDs.dtype == .int32
                ? Int64(inputIDs.max().item(Int32.self)) : Int64(inputIDs.max().item(UInt32.self))
            guard minimum >= 0, maximum < Int64(configuration.vocabularySize) else {
                throw MiMoV26ExecutionError.invalidInput("token ID outside vocabulary")
            }
            return (inputIDs.dim(0), inputIDs.dim(1))
        }
        let embeddings = embeddings!
        guard embeddings.ndim == 3, embeddings.dim(0) > 0, embeddings.dim(1) > 0,
            embeddings.dim(2) == configuration.hiddenSize, embeddings.dtype == activationDType
        else {
            throw MiMoV26ExecutionError.invalidInput(
                "independent-history native embedding geometry")
        }
        return (embeddings.dim(0), embeddings.dim(1))
    }

    private func validateIndependentCaches(
        _ caches: [[MiMoV26IndependentHistoryCache]], length: Int
    ) throws {
        guard length > 0, length <= configuration.maxPositionEmbeddings else {
            throw MiMoV26ExecutionError.contextExceeded
        }
        var objects = Set<ObjectIdentifier>()
        var requests = Set<UUID>()
        for row in caches {
            guard row.count == model.layers.count, let first = row.first,
                requests.insert(first.requestIdentity).inserted
            else {
                throw MiMoV26ExecutionError.invalidInput("independent cache row ownership")
            }
            for (index, cache) in row.enumerated() {
                guard objects.insert(ObjectIdentifier(cache)).inserted,
                    cache.requestIdentity == first.requestIdentity, cache.offset == first.offset
                else {
                    throw MiMoV26ExecutionError.cacheGeometry(layer: index)
                }
                try cache.validate(
                    model: identity, layer: index, geometry: model.layers[index].geometry,
                    dtype: activationDType, length: length)
            }
        }
    }
}

/// One verify window's outputs and its caches' checkpoints. `settle` keeps the
/// first rows: caches return to the checkpoint and take those rows again as one-token appends.
public final class MiMoV26VerifyWindow {
    public static let maximumRows = 4
    /// `[1, rows, V or this rank's rows]`, not evaluated.
    public let logits: MLXArray
    /// Post-final-norm rows `[1, rows, H]`, not evaluated.
    public let normalizedHiddenStates: MLXArray
    public let firstPosition: Int
    private let caches: [MiMoV26IndependentHistoryCache]
    private let checkpoints: [MiMoV26IndependentHistoryCache.Checkpoint]
    private let written: [[(keys: MLXArray, values: MLXArray)]]
    private var settled = false

    init(
        logits: MLXArray, normalizedHiddenStates: MLXArray, firstPosition: Int,
        caches: [MiMoV26IndependentHistoryCache],
        checkpoints: [MiMoV26IndependentHistoryCache.Checkpoint],
        written: [[(keys: MLXArray, values: MLXArray)]]
    ) {
        self.logits = logits
        self.normalizedHiddenStates = normalizedHiddenStates
        self.firstPosition = firstPosition
        self.caches = caches
        self.checkpoints = checkpoints
        self.written = written
    }

    public var rows: Int { logits.dim(1) }

    public func settle(keeping: Int) throws {
        guard !settled, (1 ... rows).contains(keeping) else {
            throw MiMoV26ExecutionError.invalidInput("a verify window keeps 1...\(rows) rows, once")
        }
        settled = true
        guard keeping < rows else { return }
        for (index, cache) in caches.enumerated() {
            cache.restore(checkpoints[index])
            let kept = written[index].prefix(keeping)
            if cache.maxSize == nil {
                _ = cache.update(
                    keys: concatenated(kept.map(\.keys), axis: 2),
                    values: concatenated(kept.map(\.values), axis: 2))
            } else {
                for entry in kept { _ = cache.update(keys: entry.keys, values: entry.values) }
            }
        }
    }
}

enum MiMoV26IndependentProjection {
    static func linear(_ layer: Linear, _ x: MLXArray) -> MLXArray {
        guard type(of: layer) == QuantizedLinear.self, let q = layer as? QuantizedLinear,
            supports(q.mode, bits: q.bits, groupSize: q.groupSize, input: x, scales: q.scales)
        else {
            return rows(x, layer.callAsFunction)
        }
        var output = project(x, weight: q.weight, scales: q.scales, biases: q.biases)
        if let bias = q.bias { output = output + bias }
        return output
    }
    static func embedding(_ layer: Embedding, _ x: MLXArray) -> MLXArray {
        guard type(of: layer) == QuantizedEmbedding.self, let q = layer as? QuantizedEmbedding,
            supports(q.mode, bits: q.bits, groupSize: q.groupSize, input: x, scales: q.scales)
        else {
            return rows(x, layer.asLinear)
        }
        return project(x, weight: q.weight, scales: q.scales, biases: q.biases)
    }
    private static func supports(
        _ mode: QuantizationMode, bits: Int, groupSize: Int,
        input: MLXArray, scales: MLXArray
    ) -> Bool {
        mode == .affine && bits == 8 && groupSize == 64 && input.dtype == .bfloat16
            && scales.dtype == .bfloat16 && input.ndim == 3 && (2 ... 8).contains(input.dim(0))
            && input.dim(1) == 1
    }
    private static func project(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?
    ) -> MLXArray {
        quantizedMM(
            x, weight.expandedDimensions(axis: 0), scales: scales.expandedDimensions(axis: 0),
            biases: biases.map { $0.expandedDimensions(axis: 0) }, transpose: true, groupSize: 64,
            bits: 8)
    }
    static func rows(_ x: MLXArray, _ forward: (MLXArray) -> MLXArray) -> MLXArray {
        guard x.dim(0) > 1 else { return forward(x) }
        return concatenated((0 ..< x.dim(0)).map { forward(x[$0 ..< $0 + 1]) }, axis: 0)
    }
}
