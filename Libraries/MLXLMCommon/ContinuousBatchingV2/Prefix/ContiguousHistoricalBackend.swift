import MLX

/// Package-only owner seam for historical full+window checkpoints. This grants
/// neither ordinary frozen/block adoption nor access to a facade's allocator.
package protocol CBv2ContiguousHistoricalBackend: CBv2KVBackend {
    func adoptContiguousHistoricalState(
        _ state: [CBv2SequenceKV?], codec: CBv2CompleteCheckpointCodec,
        layerKinds: [CBv2LayerKind], position: Int,
        requestID: CBv2RequestID, maximumSequenceLength: Int
    ) throws
}

extension CBv2ContiguousKVBackend: CBv2ContiguousHistoricalBackend {
    package func adoptContiguousHistoricalState(
        _ state: [CBv2SequenceKV?], codec: CBv2CompleteCheckpointCodec,
        layerKinds: [CBv2LayerKind], position: Int,
        requestID: CBv2RequestID, maximumSequenceLength: Int
    ) throws {
        guard codec.contiguousLayout != nil, codec.layerKinds == layerKinds else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try adoptPreparedCheckpoint(state, codec: codec, position: position,
            requestID: requestID, maximumSequenceLength: maximumSequenceLength)
    }
}
