import MLX

extension EngineLoopV2 {
    /// Preserve each Qwen4 request's positioned/unpositioned contract when
    /// media forces a rectangular positions tensor for a mixed batch. This
    /// applies to ordinary, hidden-returning and MTP target forwards alike.
    /// Other architectures and all singleton/unpositioned calls are unchanged.
    func targetForward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache], ids: [CBv2RequestID],
        positionIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        requirement: CBv2PrefillRequirement? = nil,
        phase: CBv2ForwardPhase? = nil
    ) throws -> (logits: MLXArray, recurrent: [CBv2RequestID: CBv2RecurrentStateEvaluation],
        innerState: [MLXArray])
    {
        let forward = {
            try self.targetForwardWithoutQwen4PositionScope(
                tokens: tokens, caches: caches, ids: ids, positionIds: positionIds,
                inputEmbeddings: inputEmbeddings, requirement: requirement, phase: phase)
        }
        guard positionIds != nil, ids.count > 1,
            layerKinds.contains(where: { $0.qwen4IndexerCompressRatio != nil })
        else { return try forward() }

        let explicit = ids.map { id -> Bool in
            guard let record = scheduler.record(for: id) else {
                preconditionFailure("Qwen4 position binding requires every scheduled request")
            }
            return record.request.positionState != nil
        }
        return try CBv2Qwen4PositionScope.withExplicitRows(explicit, operation: forward)
    }
}
