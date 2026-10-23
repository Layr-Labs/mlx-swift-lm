import MLX
import MLXLMCommon
import MLXNN

/// Lightning's cluster decode entry points keep the ordinary one-token
/// arithmetic for each independent request. This scope never changes a
/// prefill, speculative window, Nano or Super forward.
extension NemotronH35Model {
    public enum BatchedStageDecodeError: Error { case layout }

    public func cbv2BatchedStageResidual(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) throws -> MLXArray {
        try validateBatchedStageDecode(tokens, caches: caches, recurrentState: recurrentState)
        return Nemotron35BatchDecode.$active.withValue(tokens.dim(0) > 1) {
            cbv2StageResidual(tokens, inputEmbedding: nil, caches: caches, recurrentState: recurrentState)
        }
    }

    public func cbv2BatchedStageLogits(
        _ tokens: MLXArray, inputEmbedding: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) throws -> MLXArray {
        try validateBatchedStageDecode(tokens, caches: caches, recurrentState: recurrentState)
        guard inputEmbedding.shape == [tokens.dim(0), 1, configuration.hiddenSize] else {
            throw BatchedStageDecodeError.layout
        }
        return Nemotron35BatchDecode.$active.withValue(tokens.dim(0) > 1) {
            cbv2StageLogits(tokens, inputEmbedding: inputEmbedding, caches: caches,
                recurrentState: recurrentState, requirement: nil)
        }
    }

    private func validateBatchedStageDecode(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation]
    ) throws {
        guard tokens.ndim == 2, tokens.dim(1) == 1, (1...8).contains(tokens.dim(0)),
            recurrentState.count == tokens.dim(0),
            Set(recurrentState.map(ObjectIdentifier.init)).count == recurrentState.count,
            caches.count == cbv2LayerKinds.count,
            caches.allSatisfy({ ($0 as? any CBv2AttendingLayerCache)?.rows.count == tokens.dim(0) })
        else { throw BatchedStageDecodeError.layout }
    }
}

enum Nemotron35BatchDecode {
    @TaskLocal static var active = false

    static func applies(_ x: MLXArray) -> Bool {
        active && x.ndim == 3 && (2...8).contains(x.dim(0)) && x.dim(1) == 1
    }

    static func linear(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray {
        guard applies(x) else { return layer(x) }
        guard type(of: layer) == QuantizedLinear.self, let q = layer as? QuantizedLinear,
            supports(q.mode, bits: q.bits, groupSize: q.groupSize, input: x, scales: q.scales)
        else { return mapRows(x, layer.callAsFunction) }
        var output = project(x, weight: q.weight, scales: q.scales, biases: q.biases,
            groupSize: q.groupSize, bits: q.bits)
        if let bias = q.bias { output = output + bias }
        return output
    }

    static func embedding(_ layer: Embedding, _ x: MLXArray) -> MLXArray {
        guard applies(x) else { return layer.asLinear(x) }
        guard type(of: layer) == QuantizedEmbedding.self, let q = layer as? QuantizedEmbedding,
            supports(q.mode, bits: q.bits, groupSize: q.groupSize, input: x, scales: q.scales)
        else { return mapRows(x, layer.asLinear) }
        return project(x, weight: q.weight, scales: q.scales, biases: q.biases,
            groupSize: q.groupSize, bits: q.bits)
    }

    private static func supports(_ mode: QuantizationMode, bits: Int, groupSize: Int,
                                 input: MLXArray, scales: MLXArray) -> Bool {
        guard mode == .affine, [2, 3, 4, 8].contains(bits), [64, 128].contains(groupSize) else { return false }
        switch (input.dtype, scales.dtype) {
        case (.bfloat16, .bfloat16), (.float16, .float16),
             (.float32, .float16), (.float32, .bfloat16), (.float32, .float32): return true
        default: return false
        }
    }

    private static func project(_ x: MLXArray, weight: MLXArray, scales: MLXArray,
                                biases: MLXArray?, groupSize: Int, bits: Int) -> MLXArray {
        // Broadcast views prevent flattening B into matmul M. The native
        // product sees B independent M=1 rows without copying the weights.
        quantizedMM(x, weight.expandedDimensions(axis: 0), scales: scales.expandedDimensions(axis: 0),
            biases: biases.map { $0.expandedDimensions(axis: 0) }, transpose: true,
            groupSize: groupSize, bits: bits)
    }

    /// A view would keep the entire rectangular recurrent buffer alive after
    /// other lanes retire. A one-row gather owns compact backing and copies
    /// native bits without floating arithmetic or a host round trip.
    static func ownedRow(_ x: MLXArray, row: Int) -> MLXArray {
        guard active, x.dim(0) > 1 else { return x[row..<row + 1] }
        return take(x, MLXArray([Int32(row)]), axis: 0)
    }

    static func mapRows(_ x: MLXArray, _ forward: (MLXArray) -> MLXArray) -> MLXArray {
        guard applies(x) else { return forward(x) }
        return concatenated((0..<x.dim(0)).map { forward(x[$0..<$0 + 1]) }, axis: 0)
    }
}
