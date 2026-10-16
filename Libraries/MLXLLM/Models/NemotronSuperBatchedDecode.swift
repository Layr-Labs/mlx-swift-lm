import MLX
import MLXLMCommon
import MLXNN

/// Rectangular decode of B independent Super requests in which every row is
/// bit-identical to that request's one-token decode (`cbv2Forward` with one
/// row). Rows are laid out as the positions of one `[1, B, hidden]` sequence,
/// so every product runs through the row-exact helpers in the shape they are
/// proven for (`NemotronSuperMTPRowExactnessTests`): projections and the
/// router share their weight reads across groups of up to
/// `ModelOptRowExact.maximumRows` rows, routed experts keep one single-vector
/// product per row, and attention reads each row's
/// own KV. Every row's conv/SSM state is staged in its own buffer, so a
/// request that leaves the batch releases its state with it.
extension NemotronHModel {
    public enum SuperBatchedDecodeError: Error { case layout }

    public func superRowExactBatchedDecode(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        plan: NemotronSuperMTPCapturePlan
    ) throws -> MLXArray {
        let rows = tokens.dim(0)
        guard plan.target == ObjectIdentifier(self), tokens.ndim == 2, tokens.dim(1) == 1,
            rows >= 1, recurrentState.count == rows, caches.count == cbv2LayerKinds.count
        else { throw SuperBatchedDecodeError.layout }
        let attending = try caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                throw SuperBatchedDecodeError.layout
            }
            return attending
        }
        let observation =
            CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(liveBatchRows: rows, sequenceWidth: 1)
            : nil
        defer { observation?.end() }
        var hidden = mtpEmbedding(tokens).reshaped(1, rows, -1)
        var attentionIndex = 0
        for (index, layer) in plan.layers.enumerated() {
            let normalized = layer.norm(hidden)
            let output: MLXArray
            switch layer.mixer {
            case .mamba(let mamba):
                output = try superBatchedMamba(
                    mamba, normalized, modelLayerIndex: index, states: recurrentState)
            case .attention(let attention):
                output = superBatchedAttention(
                    attention, normalized, cache: attending[attentionIndex])
                attentionIndex += 1
            case .moe(let moe):
                output = superRowExactMoE(moe, normalized)
            }
            hidden = hidden + output
        }
        hidden = plan.finalNorm(hidden)
        let logits =
            lmHead.map { superRowExactRows($0, hidden) }
            ?? nemotronMTPMapRows(hidden) { self.logits($0) }
        return logits.reshaped(rows, 1, -1)
    }
}

/// `superRowExactProjection` over any row count: rows beyond one row-exact
/// call run in consecutive groups, each reading the weights once.
func superRowExactRows(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray {
    superRowExactGroups(x) { superRowExactProjection(layer, $0) }
}

/// Mirrors `NemotronHMamba2Mixer.cbv2Forward` for rows that are separate
/// requests: one depthwise convolution over every row's own history, the
/// one-step SSM kernel per row and per-row staged state.
func superBatchedMamba(
    _ mamba: NemotronHMamba2Mixer, _ x: MLXArray, modelLayerIndex: Int,
    states: [CBv2RecurrentStateEvaluation]
) throws -> MLXArray {
    let rows = x.dim(1)
    let inner = mamba.intermediateSize
    let projected = superRowExactRows(mamba.inProj, x)
    let parts = split(projected, indices: [inner, inner + mamba.convDim], axis: -1)
    let convInput = parts[1].reshaped(rows, 1, mamba.convDim)
    let history = max(0, mamba.convKernelSize - 1)
    var padded: [MLXArray] = []
    var ssm: [MLXArray] = []
    for (row, state) in states.enumerated() {
        let input = state.inputState(modelLayerIndex: modelLayerIndex)
        let conv = input?.conv ?? MLXArray.zeros([1, history, mamba.convDim], dtype: x.dtype)
        padded.append(concatenated([conv, convInput[row ..< (row + 1)]], axis: 1))
        ssm.append(
            input?.ssm
                ?? MLXArray.zeros(
                    [1, mamba.numHeads, mamba.headDim, mamba.ssmStateSize], dtype: .float32))
    }
    let convolution = silu(mamba.conv1d(rows == 1 ? padded[0] : concatenated(padded, axis: 0)))
    let groupWidth = mamba.numGroups * mamba.ssmStateSize
    let values = split(convolution, indices: [inner, inner + groupWidth], axis: -1)
    var outputs: [MLXArray] = []
    for (row, state) in states.enumerated() {
        let r = row ..< (row + 1)
        let (y, next) = ssmUpdate(
            hiddenStates: values[0][r].reshaped(1, 1, mamba.numHeads, mamba.headDim),
            ALog: mamba.aLog,
            B: values[1][r].reshaped(1, 1, mamba.numGroups, mamba.ssmStateSize),
            C: values[2][r].reshaped(1, 1, mamba.numGroups, mamba.ssmStateSize),
            D: mamba.D, dt: parts[2][0..., r].reshaped(1, 1, mamba.numHeads),
            dtBias: mamba.dtBias, state: ssm[row], timeStepLimit: mamba.timeStepLimit,
            mask: nil)
        let end = padded[row].dim(1)
        try state.stage(
            modelLayerIndex: modelLayerIndex,
            conv: padded[row][0..., (end - history) ..< end, 0...],
            ssm: next.asType(.float32))
        outputs.append(y)
    }
    let y = concatenated(outputs, axis: 1).asType(x.dtype).flattened(start: 2)
    return superRowExactRows(mamba.outProj, mamba.norm(y, gate: parts[0]))
}

/// Mirrors `NemotronHAttention.cbv2Forward` with one query per request row.
func superBatchedAttention(
    _ attention: NemotronHAttention, _ x: MLXArray, cache: any CBv2AttendingLayerCache
) -> MLXArray {
    let rows = x.dim(1)
    func heads(_ value: MLXArray, _ count: Int) -> MLXArray {
        value.reshaped(rows, 1, count, attention.headDim).transposed(0, 2, 1, 3)
    }
    let output = cache.updateAndAttend(
        queries: heads(superRowExactRows(attention.wq, x), attention.numHeads),
        keys: heads(superRowExactRows(attention.wk, x), attention.numKeyValueHeads),
        values: heads(superRowExactRows(attention.wv, x), attention.numKeyValueHeads),
        scale: attention.scale, sinks: nil)
    return superRowExactRows(
        attention.wo, output.transposed(0, 2, 1, 3).reshaped(1, rows, -1))
}
