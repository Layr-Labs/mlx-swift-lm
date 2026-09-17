import MLX
import MLXLMCommon

extension Qwen4ExpTextModel: CBv2CompleteCheckpointKVTypeProviding, CBv2Qwen4CheckpointGeometryProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? { checkpointMetadata()?.kv }
    public var cbv2Qwen4CheckpointGeometries: [CBv2Qwen4CheckpointGeometry]? { checkpointMetadata()?.qsa }

    /// Load-time graph metadata on isolated QSA rows. Runs no PLE lookup,
    /// expert, sampler, or serving-cache mutation. The full native KV probe
    /// separately checks these types against real target prefill/decode.
    private func checkpointMetadata() -> (kv: [DType], qsa: [CBv2Qwen4CheckpointGeometry])? {
        try? withError { error in
            let dtype = Qwen4ExpActivation.hiddenDType(from: model.embedTokens)
            var kv: [DType] = [], qsa: [CBv2Qwen4CheckpointGeometry] = []
            for kind in cbv2LayerKinds {
                guard let index = kind.modelLayerIndex, model.layers.indices.contains(index),
                    let attention = model.layers[index].selfAttn,
                    attention.indexer.compressRatio > 0, attention.indexer.compressRatio <= 4096
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                let count = attention.indexer.compressRatio
                let row = CBv2FullSequenceKV(promptLength: 0, maxLength: count,
                    kvHeads: kind.kvHeads, headDim: kind.headDim)
                let cache = CBv2LayerCache(layerIndex: index, kind: kind, rows: [row])
                defer { cache.setRows([]) }
                let input = MLXArray.zeros([1, count, configuration.hiddenSize], dtype: dtype)
                _ = attention.cbv2Forward(input, cache: cache)
                try error.check()
                let stored = row.snapshot()
                guard stored.offset == count, stored.keys.dtype == stored.values.dtype,
                    let raw = cache.qwen4IndexKeys, let positions = cache.qwen4IndexPositionIds,
                    let logical = Qwen4ExpIndexCapacity.logicalTokens(raw, length: count),
                    let logicalPositions = Qwen4ExpIndexCapacity.logicalPositions(positions, length: count)
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                let pooled = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
                    indexKeys: logical, indexPositionIds: logicalPositions, compressRatio: count,
                    indexKeyNorm: { attention.indexer.kLayerNorm($0) },
                    applyIndexRope: { states, positions in
                        attention.mrope.apply(queries: states, keys: states, positionIds: positions).0
                    })
                try error.check()
                kv.append(stored.keys.dtype)
                qsa.append(try .init(layer: index, headDim: attention.indexer.headDim,
                    compressRatio: count, keyDType: logical.dtype, pooledDType: pooled.dtype))
            }
            return (kv, qsa)
        }
    }
}

extension Qwen4ExpModel: CBv2CompleteCheckpointKVTypeProviding, CBv2Qwen4CheckpointGeometryProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? { languageModel.cbv2CompleteCheckpointKVDTypes }
    public var cbv2Qwen4CheckpointGeometries: [CBv2Qwen4CheckpointGeometry]? { languageModel.cbv2Qwen4CheckpointGeometries }
}
