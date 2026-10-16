import MLX
import MLXLMCommon
import MLXNN

/// Exact rectangular verification for the Super target: one forward over a
/// `[1, 1 + k]` window whose every position reproduces the ordinary one-token
/// decode bit for bit. Projections use only row-exact batched products
/// (`superRowExactProjection`), attention is serialized per query by the
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

    /// One captured verification window. Logits and post-final-norm hidden
    /// for every position equal the one-token decode of the same inputs.
    public func superMTPCapturedTextForward(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?, plan: NemotronSuperMTPCapturePlan
    ) throws -> (logits: MLXArray, lastHidden: MLXArray) {
        guard plan.target == ObjectIdentifier(self), positionIds == nil, tokens.ndim == 2,
            tokens.dim(0) == 1, (1 ... 8).contains(tokens.dim(1)), recurrentState.count == 1,
            caches.count == cbv2LayerKinds.count
        else { throw SuperMTPCaptureError.window }
        let attending = try caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                throw SuperMTPCaptureError.window
            }
            return attending
        }
        let state = recurrentState[0]
        var hidden = mtpEmbedding(tokens)
        var attentionIndex = 0
        for (index, layer) in plan.layers.enumerated() {
            let normalized = layer.norm(hidden)
            let output: MLXArray
            switch layer.mixer {
            case .mamba(let mamba):
                output = try superCapturedMamba(
                    mamba, normalized, modelLayerIndex: index, state: state)
            case .attention(let attention):
                output = superWindowAttention(
                    attention, normalized, cache: attending[attentionIndex])
                attentionIndex += 1
            case .moe(let moe):
                output = superRowExactMoE(moe, normalized)
            }
            hidden = hidden + output
        }
        hidden = plan.finalNorm(hidden)
        let logits =
            lmHead.map { superRowExactProjection($0, hidden) }
            ?? nemotronMTPMapRows(hidden) { self.logits($0) }
        return (logits, hidden)
    }
}

/// Mirrors `NemotronHMamba2Mixer.cbv2Forward` per position: one causal
/// convolution over the window, the canonical one-step SSM arithmetic at every
/// position, and every position's conv/SSM state staged for commit.
func superCapturedMamba(
    _ mamba: NemotronHMamba2Mixer, _ x: MLXArray, modelLayerIndex: Int,
    state: CBv2RecurrentStateEvaluation
) throws -> MLXArray {
    let length = x.dim(1)
    let initial = state.inputState(modelLayerIndex: modelLayerIndex)
    let history = mamba.convKernelSize - 1
    let convState =
        initial?.conv ?? MLXArray.zeros([1, max(0, history), mamba.convDim], dtype: x.dtype)
    let ssmState =
        initial?.ssm
        ?? MLXArray.zeros(
            [1, mamba.numHeads, mamba.headDim, mamba.ssmStateSize], dtype: .float32)
    let projected = superRowExactProjection(mamba.inProj, x)
    let parts = split(
        projected, indices: [mamba.intermediateSize, mamba.intermediateSize + mamba.convDim],
        axis: -1)
    let padded = concatenated([convState, parts[1]], axis: 1)
    let convolution = silu(mamba.conv1d(padded))
    let convStates = concatenated(
        (0 ..< length).map { padded[0..., ($0 + 1) ..< ($0 + 1 + history), 0...] }, axis: 0)
    let values = split(
        convolution,
        indices: [
            mamba.intermediateSize, mamba.intermediateSize + mamba.numGroups * mamba.ssmStateSize,
        ],
        axis: -1)
    let window = nemotronMTPSSMWindow(
        hiddenStates: values[0].reshaped(1, length, mamba.numHeads, mamba.headDim),
        ALog: mamba.aLog,
        B: values[1].reshaped(1, length, mamba.numGroups, mamba.ssmStateSize),
        C: values[2].reshaped(1, length, mamba.numGroups, mamba.ssmStateSize), D: mamba.D,
        dt: parts[2].reshaped(1, length, mamba.numHeads), dtBias: mamba.dtBias,
        state: ssmState, timeStepLimit: mamba.timeStepLimit)
    try state.stageCaptured(
        modelLayerIndex: modelLayerIndex, conv: convStates, ssm: window.capturedStates,
        positions: length)
    let normalized = mamba.norm(window.output.flattened(start: 2), gate: parts[0])
    return superRowExactProjection(mamba.outProj, normalized)
}

/// Mirrors `NemotronHAttention.cbv2Forward`; the cache serializes queries.
func superWindowAttention(
    _ attention: NemotronHAttention, _ x: MLXArray, cache: any CBv2AttendingLayerCache
) -> MLXArray {
    let length = x.dim(1)
    func heads(_ value: MLXArray, _ count: Int) -> MLXArray {
        value.reshaped(1, length, count, attention.headDim).transposed(0, 2, 1, 3)
    }
    let output = cache.updateAndAttend(
        queries: heads(superRowExactProjection(attention.wq, x), attention.numHeads),
        keys: heads(superRowExactProjection(attention.wk, x), attention.numKeyValueHeads),
        values: heads(superRowExactProjection(attention.wv, x), attention.numKeyValueHeads),
        scale: attention.scale, sinks: nil)
    return superRowExactProjection(
        attention.wo, output.transposed(0, 2, 1, 3).reshaped(1, length, -1))
}

/// Mirrors `NemotronHMoE.callAsFunction` per row: one-row router products
/// with row-local selection, gathered latent projections, the fastpath
/// routed-expert graph and row-exact shared experts.
func superRowExactMoE(_ moe: NemotronHMoE, _ x: MLXArray) -> MLXArray {
    let (indices, scores) = moe.gate.mtpForwardRows(x)
    let latent = superRowExactProjection(moe.fc1LatentProj!, x)
    var routed: MLXArray
    if superRowExactRouting(moe.switchMLP) {
        routed = moe.switchMLP(latent, indices)
    } else {
        routed = concatenated(
            (0 ..< x.dim(1)).map {
                moe.switchMLP(latent[0..., $0 ..< ($0 + 1)], indices[0..., $0 ..< ($0 + 1)])
            }, axis: 1)
    }
    routed = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(routed.dtype)
    routed = superRowExactProjection(moe.fc2LatentProj!, routed)
    guard let shared = moe.sharedExperts else { return routed }
    let up = maximum(superRowExactProjection(shared.upProj, x), MLXArray(0))
    return routed + superRowExactProjection(shared.downProj, up * up)
}

/// The fastpath graph route is row-exact for any routed row count.
func superRowExactRouting(_ mlp: NemotronHSwitchMLP) -> Bool {
    [mlp.fc1, mlp.fc2].allSatisfy { layer in
        guard let experts = layer as? ModelOptNVFP4SwitchLinear else { return false }
        return experts.execution == .graph && experts.accumulation == .rowwise
    }
}
