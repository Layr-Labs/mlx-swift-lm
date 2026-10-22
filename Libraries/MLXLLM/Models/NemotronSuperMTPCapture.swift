import MLX
import MLXLMCommon
import MLXNN

/// Exact rectangular verification for the Super target: one forward over a
/// `[requests, 1 + k]` window whose every position reproduces that request's
/// ordinary one-token decode bit for bit. Projections use only row-exact
/// batched products (`superRowExactRows`), attention is serialized per query by the
/// engine's rectangular cache contract, and every Mamba layer stages its
/// per-position conv/SSM states so finalization commits the state after the
/// last accepted token on device. Per-position capture costs one extra state
/// write per position; recomputing the accepted prefix would re-read every
/// weight in a second target pass.
public final class NemotronSuperMTPCapturePlan {
    enum Mixer {
        case mamba(NemotronHMamba2Mixer)
        case attention(NemotronHAttention)
        case moe(NemotronHMoE)
    }
    struct Layer {
        let norm: RMSNorm
        let mixer: Mixer
    }
    let layers: [Layer]
    let finalNorm: RMSNorm
    let target: ObjectIdentifier

    init(layers: [Layer], finalNorm: RMSNorm, target: ObjectIdentifier) {
        self.layers = layers
        self.finalNorm = finalNorm
        self.target = target
    }
}

extension NemotronHModel {
    public enum SuperMTPCaptureError: Error { case layout, window }

    /// Resolves the loaded blocks once. Refuses MLP blocks and any layer kind
    /// whose window path is not row-exact.
    public func superMTPCapturePlan() throws -> NemotronSuperMTPCapturePlan {
        guard type(of: self) == NemotronHModel.self else { throw SuperMTPCaptureError.layout }
        var layers: [NemotronSuperMTPCapturePlan.Layer] = []
        for block in loraLayers {
            let children = Dictionary(
                block.namedModules().filter { !$0.0.contains(".") }, uniquingKeysWith: { a, _ in a }
            )
            guard let norm = children["norm"] as? RMSNorm else {
                throw SuperMTPCaptureError.layout
            }
            switch children["mixer"] {
            case let mamba as NemotronHMamba2Mixer:
                layers.append(.init(norm: norm, mixer: .mamba(mamba)))
            case let attention as NemotronHAttention:
                layers.append(.init(norm: norm, mixer: .attention(attention)))
            case let moe as NemotronHMoE where moe.fc1LatentProj != nil && moe.fc2LatentProj != nil:
                layers.append(.init(norm: norm, mixer: .moe(moe)))
            default: throw SuperMTPCaptureError.layout
            }
        }
        guard layers.count == configuration.hybridOverridePattern.count,
            let finalNorm = namedModules().first(where: { $0.0 == "backbone.norm_f" })?.1
                as? RMSNorm
        else { throw SuperMTPCaptureError.layout }
        return NemotronSuperMTPCapturePlan(
            layers: layers, finalNorm: finalNorm, target: ObjectIdentifier(self))
    }

    /// Requests one captured window may verify together: CBv2's tested
    /// rectangular batch ceiling.
    public static let superMTPMaximumWindowRequests = CBv2MTPConfig.testedMaxSpeculativeBatch
    /// Positions of one request's window: the seed and up to seven drafts,
    /// the SSM window kernel's limit.
    public static let superMTPMaximumWindowWidth = 8

    /// One captured verification window of `[requests, 1 + k]` tokens. Every
    /// request's logits and post-final-norm hidden at every position equal its
    /// own one-token decode of the same inputs, whatever the other requests are.
    public func superMTPCapturedTextForward(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?, plan: NemotronSuperMTPCapturePlan
    ) throws -> (logits: MLXArray, lastHidden: MLXArray) {
        guard plan.target == ObjectIdentifier(self), positionIds == nil, tokens.ndim == 2,
            (1 ... Self.superMTPMaximumWindowRequests).contains(tokens.dim(0)),
            (1 ... Self.superMTPMaximumWindowWidth).contains(tokens.dim(1)),
            recurrentState.count == tokens.dim(0),
            caches.count == cbv2LayerKinds.count
        else { throw SuperMTPCaptureError.window }
        return try superRowExactForward(
            tokens, caches: caches, states: recurrentState, plan: plan, captured: true)
    }

    /// Row-exact trunk over `[requests, length]` tokens laid out as the
    /// `requests * length` positions of one sequence, so every product shares
    /// its weight reads across requests without changing any row. Each
    /// request keeps its own convolution, SSM recurrence and attention KV.
    /// `captured` stages every position's recurrent state for commit at the
    /// accepted length; otherwise `length` is one and each request stages its
    /// next state.
    func superRowExactForward(
        _ tokens: MLXArray, caches: [KVCache], states: [CBv2RecurrentStateEvaluation],
        plan: NemotronSuperMTPCapturePlan, captured: Bool
    ) throws -> (logits: MLXArray, lastHidden: MLXArray) {
        let requests = tokens.dim(0)
        let length = tokens.dim(1)
        guard captured || length == 1 else { throw SuperMTPCaptureError.window }
        let attending = try caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                throw SuperMTPCaptureError.window
            }
            return attending
        }
        let observation =
            CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(
                liveBatchRows: requests, sequenceWidth: length)
            : nil
        defer { observation?.end() }
        let fused = NemotronSuperMTPFusion.active
        let submitEvery = NemotronSuperMTPWindowSubmission.active
        var hidden = mtpEmbedding(tokens).reshaped(1, requests * length, -1)
        // As one-token decode: each block's residual add and the next norm
        // (the final one past the last block) in one kernel.
        var normed = fused && superResidualNormActive ? plan.layers.first?.norm(hidden) : nil
        var attentionIndex = 0
        for (index, layer) in plan.layers.enumerated() {
            let normalized = normed ?? layer.norm(hidden)
            let output: MLXArray
            var extra: MLXArray?
            switch layer.mixer {
            case .mamba(let mamba):
                output =
                    captured
                    ? try superCapturedMamba(
                        mamba, normalized, length: length, modelLayerIndex: index, states: states,
                        fused: fused)
                    : try superBatchedMamba(
                        mamba, normalized, modelLayerIndex: index, states: states, fused: fused)
            case .attention(let attention):
                output = superWindowAttention(
                    attention, normalized, requests: requests, cache: attending[attentionIndex])
                attentionIndex += 1
            case .moe(let moe):
                let parts = superRowExactMoE(moe, normalized, fused: fused)
                output = parts.routed
                extra = parts.shared
            }
            if normed != nil {
                let next =
                    index + 1 < plan.layers.count ? plan.layers[index + 1].norm : plan.finalNorm
                let boundary = NemotronSuperResidualNorm.boundary(
                    hidden, output, extra: extra, norm: next)
                hidden = boundary.sum
                normed = boundary.normed
            } else {
                hidden = hidden + (extra.map { output + $0 } ?? output)
            }
            // Submit the trunk in parts so the GPU runs while the rest is
            // built. Evaluation order only; values are unchanged.
            if submitEvery > 0, (index + 1) % submitEvery == 0, index + 1 < plan.layers.count {
                asyncEval([hidden] + (normed.map { [$0] } ?? []))
            }
        }
        hidden = normed ?? plan.finalNorm(hidden)
        let logits =
            lmHead.map { superRowExactRows($0, hidden) }
            ?? nemotronMTPMapRows(hidden) { self.logits($0) }
        return (
            logits.reshaped(requests, length, -1), hidden.reshaped(requests, length, -1)
        )
    }
}

/// Mirrors `NemotronHMamba2Mixer.cbv2Forward` per position for each request's
/// window of `length` consecutive rows: one causal convolution over that
/// request's own history, the canonical one-step SSM arithmetic at every
/// position, and every position's conv/SSM state staged in that request's own
/// buffers for commit. `fused` normalizes with the decode step's fused gated
/// norm, row by row.
func superCapturedMamba(
    _ mamba: NemotronHMamba2Mixer, _ x: MLXArray, length: Int, modelLayerIndex: Int,
    states: [CBv2RecurrentStateEvaluation], fused: Bool
) throws -> MLXArray {
    let history = mamba.convKernelSize - 1
    let projected = superRowExactRows(mamba.inProj, x)
    let parts = split(
        projected, indices: [mamba.intermediateSize, mamba.intermediateSize + mamba.convDim],
        axis: -1)
    var outputs: [MLXArray] = []
    for (request, state) in states.enumerated() {
        let rows = (request * length) ..< ((request + 1) * length)
        let initial = state.inputState(modelLayerIndex: modelLayerIndex)
        let convState =
            initial?.conv ?? MLXArray.zeros([1, max(0, history), mamba.convDim], dtype: x.dtype)
        let ssmState =
            initial?.ssm
            ?? MLXArray.zeros(
                [1, mamba.numHeads, mamba.headDim, mamba.ssmStateSize], dtype: .float32)
        let padded = concatenated([convState, parts[1][0..., rows, 0...]], axis: 1)
        let convolution = silu(mamba.conv1d(padded))
        let convStates = concatenated(
            (0 ..< length).map { padded[0..., ($0 + 1) ..< ($0 + 1 + history), 0...] }, axis: 0)
        let values = split(
            convolution,
            indices: [
                mamba.intermediateSize,
                mamba.intermediateSize + mamba.numGroups * mamba.ssmStateSize,
            ],
            axis: -1)
        let window = nemotronMTPSSMWindow(
            hiddenStates: values[0].reshaped(1, length, mamba.numHeads, mamba.headDim),
            ALog: mamba.aLog,
            B: values[1].reshaped(1, length, mamba.numGroups, mamba.ssmStateSize),
            C: values[2].reshaped(1, length, mamba.numGroups, mamba.ssmStateSize), D: mamba.D,
            dt: parts[2][0..., rows, 0...].reshaped(1, length, mamba.numHeads),
            dtBias: mamba.dtBias, state: ssmState, timeStepLimit: mamba.timeStepLimit)
        try state.stageCaptured(
            modelLayerIndex: modelLayerIndex, conv: convStates, ssm: window.capturedStates,
            positions: length)
        outputs.append(window.output)
    }
    let output = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1)
    let y = output.flattened(start: 2)
    let normalized =
        (fused ? mamba.superGatedNorm(y, gate: parts[0]) : nil) ?? mamba.norm(y, gate: parts[0])
    return superRowExactRows(mamba.outProj, normalized)
}

/// Mirrors `NemotronHAttention.cbv2Forward` for `requests` windows of
/// consecutive rows; each request attends its own KV and, inside a verify
/// round, the cache serializes its queries.
func superWindowAttention(
    _ attention: NemotronHAttention, _ x: MLXArray, requests: Int,
    cache: any CBv2AttendingLayerCache
) -> MLXArray {
    let length = x.dim(1) / requests
    func heads(_ value: MLXArray, _ count: Int) -> MLXArray {
        value.reshaped(requests, length, count, attention.headDim).transposed(0, 2, 1, 3)
    }
    let output = cache.updateAndAttend(
        queries: heads(superRowExactRows(attention.wq, x), attention.numHeads),
        keys: heads(superRowExactRows(attention.wk, x), attention.numKeyValueHeads),
        values: heads(superRowExactRows(attention.wv, x), attention.numKeyValueHeads),
        scale: attention.scale, sinks: nil)
    return superRowExactRows(
        attention.wo, output.transposed(0, 2, 1, 3).reshaped(1, requests * length, -1))
}

/// Mirrors `NemotronHMoE.parts` per row: row-exact router products with
/// row-local selection, row-exact latent projections, the fastpath
/// routed-expert graph and row-exact shared experts. Returns the routed and
/// shared outputs, whose sum is the layer's output. `fused` takes the decode's
/// fused expert combine (where `fusedSuperCombine` enables it) and its squared
/// ReLU fused into the shared expert's up projection.
func superRowExactMoE(_ moe: NemotronHMoE, _ x: MLXArray, fused: Bool)
    -> (routed: MLXArray, shared: MLXArray?)
{
    var indices: [MLXArray] = []
    var scores: [MLXArray] = []
    for start in stride(from: 0, to: x.dim(1), by: ModelOptRowExact.maximumRows) {
        let rows = start ..< min(x.dim(1), start + ModelOptRowExact.maximumRows)
        let (groupIndices, groupScores) = moe.gate.superRowExactForwardRows(x[0..., rows])
        indices.append(groupIndices)
        scores.append(groupScores)
    }
    let routes = indices.count == 1 ? indices[0] : concatenated(indices, axis: 1)
    let weights = scores.count == 1 ? scores[0] : concatenated(scores, axis: 1)
    let latent = superRowExactRows(moe.fc1LatentProj!, x)
    var routed: MLXArray
    if superRowExactRouting(moe.switchMLP, routedRows: routes.size) {
        routed = moe.switchMLP(latent, routes)
    } else {
        routed = concatenated(
            (0 ..< x.dim(1)).map {
                moe.switchMLP(latent[0..., $0 ..< ($0 + 1)], routes[0..., $0 ..< ($0 + 1)])
            }, axis: 1)
    }
    routed = NemotronSuperMoEKernels.weightedSum(
        routed, scores: weights, fused: fused && moe.fusedSuperCombine)
    routed = superRowExactRows(moe.fc2LatentProj!, routed)
    guard let shared = moe.sharedExperts else { return (routed, nil) }
    if fused {
        let activated = superRowExactRows(shared.upProj, x, squaredReLU: true)
        return (routed, superRowExactRows(shared.downProj, activated))
    }
    let up = maximum(superRowExactRows(shared.upProj, x), MLXArray(0))
    return (routed, superRowExactRows(shared.downProj, up * up))
}

/// The fastpath routed graph runs one single-vector product per routed row,
/// the one-token decode arithmetic, unless `.tiled` accumulation takes MLX's
/// sorted matrix route: that needs at least 16 routed rows and on average four
/// per expert (`ModelOptNVFP4SwitchLinear.graphForward`). Guarded at the
/// threshold by `NemotronSuperMTPRowExactnessTests`.
func superRowExactRouting(_ mlp: NemotronHSwitchMLP, routedRows: Int) -> Bool {
    [mlp.fc1, mlp.fc2].allSatisfy { layer in
        guard let experts = layer as? ModelOptNVFP4SwitchLinear, experts.execution == .graph
        else { return false }
        return experts.accumulation == .rowwise
            || routedRows < max(16, 4 * experts.logicalShape[0])
    }
}
