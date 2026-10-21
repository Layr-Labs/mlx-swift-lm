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
        return try superRowExactForward(
            tokens, caches: caches, states: recurrentState, plan: plan, captured: false
        ).logits
    }
}

/// `superRowExactProjection` over any row count: rows beyond one row-exact
/// call run in consecutive groups of at most `ModelOptRowExact.maximumRows`.
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
