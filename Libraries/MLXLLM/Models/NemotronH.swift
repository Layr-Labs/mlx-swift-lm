//
//  NemotronH.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron_h.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Block Type

private enum NemotronHBlockType {
    case mamba  // "M"
    case attention  // "*"
    case mlp  // "-"
    case moe  // "E"

    init(from char: Character) {
        switch char {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default: fatalError("Unknown NemotronH block type: \(char)")
        }
    }
}

// MARK: - Mixer Protocol

/// Protocol for all mixer types in NemotronH blocks
private protocol NemotronHMixer: Module {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

// MARK: - Activations

/// Squared ReLU activation: relu(x)^2
private func relu2(_ x: MLXArray) -> MLXArray {
    let y = MLX.maximum(x, MLXArray(0))
    return y * y
}

// MARK: - MambaRMSNormGated

class NemotronHRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int

    init(dimensions: Int, eps: Float, groupSize: Int) {
        self.eps = eps
        self.groupSize = groupSize
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray?) -> MLXArray {
        var states = x
        if let gate {
            states = states * silu(gate)
        }

        // Python: x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        // Reshape [..., hidden] -> [..., nGroups, groupSize] for per-group normalization
        let shape = states.shape
        var newShape = Array(shape.dropLast())
        newShape.append(-1)
        newShape.append(groupSize)
        let unflattened = states.reshaped(newShape)

        // Python: x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        // Apply RMS norm per group WITHOUT scaling (pass ones as weight)
        // Swift rmsNorm doesn't accept nil, so we use identity weight
        // The optional-weight Python RMSNorm preserves x.dtype. A default
        // FP32 identity silently widens BF16 activations for the rest of the
        // trunk (and disables native BF16 verification kernels).
        let identityWeight = MLXArray.ones([groupSize], dtype: unflattened.dtype)
        let normed = MLXFast.rmsNorm(unflattened, weight: identityWeight, eps: eps)

        // Python: return self.weight * x.flatten(-2)
        // Flatten back to [..., hidden] and apply learned weight
        let flattened = normed.reshaped(shape)
        return weight * flattened
    }
}

// MARK: - Mamba2Mixer

class NemotronHMamba2Mixer: Module, NemotronHMixer {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)
    let headsPerGroup: Int
    let keepsFloat32State: Bool

    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: any UnaryLayer
    @ModuleInfo(key: "out_proj") var outProj: any UnaryLayer

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var D: MLXArray

    @ModuleInfo(key: "norm") var norm: NemotronHRMSNormGated

    init(_ args: NemotronHConfiguration) {
        self.numHeads = args.mambaNumHeads
        self.hiddenSize = args.hiddenSize
        self.ssmStateSize = args.ssmStateSize
        self.convKernelSize = args.convKernel
        self.intermediateSize = args.mambaNumHeads * args.mambaHeadDim
        self.numGroups = args.nGroups
        self.headDim = args.mambaHeadDim
        self.timeStepLimit = (args.timeStepLimitMin, args.timeStepLimitMax)
        self.headsPerGroup = numHeads / numGroups
        self.keepsFloat32State = args.mambaSSMCacheDType == "float32"
        self.convDim = intermediateSize + 2 * numGroups * ssmStateSize

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.useConvBias
        )

        let projectionSize = intermediateSize + convDim + numHeads
        self._inProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: args.mambaProjBias)

        self._dtBias.wrappedValue = MLXArray.ones([numHeads])
        let headsRange = (MLXArray(0 ..< numHeads).asType(.float32) + 1)
        self._aLog.wrappedValue = MLX.log(headsRange)
        self._D.wrappedValue = MLXArray.ones([numHeads])

        let groupSize = intermediateSize / numGroups
        self._norm.wrappedValue = NemotronHRMSNormGated(
            dimensions: intermediateSize,
            eps: args.layerNormEpsilon,
            groupSize: groupSize
        )

        self._outProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: args.mambaProjBias)

        super.init()
    }

    /// Super on the graph path (ModelOpt projections) prefills through its
    /// fused SSD window kernel unless this is cleared; the eager reference,
    /// every other Nemotron-H model and all decode steps keep ssmUpdate.
    var fusedSuperSSD = true

    private func selectedSSMUpdate(
        hiddenStates: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray,
        dt: MLXArray, dtBias: MLXArray, state: MLXArray?, timeStepLimit: (Float, Float),
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        // Prompts shorter than 32 tokens keep the oracle scan, whose cost is
        // negligible there. The fused scan is bit-identical to it at any
        // length; longer prompts differ from the eager reference only in the
        // tiled matrix products (`ModelOptAccumulation.tiled`).
        if fusedSuperSSD, (inProj as? ModelOptCalibratedLinear)?.execution == .graph,
            mask == nil, hiddenStates.dim(1) >= 32
        {
            return NemotronSuperSSD.scan(
                x: hiddenStates, ALog: ALog, B: B, C: C, D: D, dt: dt, dtBias: dtBias,
                state: state, timeStepLimit: timeStepLimit)
        }
        return ssmUpdate(
            hiddenStates: hiddenStates, ALog: ALog, B: B, C: C, D: D, dt: dt, dtBias: dtBias,
            state: state, timeStepLimit: timeStepLimit, mask: mask)
    }

    /// A conv-state slice shares the whole padded chunk's buffer, which the
    /// staged state would keep alive (Super: 168 MB per layer at 8192 tokens).
    /// Super copies the few rows (`* 1` is exact, including -0 and NaN).
    private func retainedConvState(_ slice: MLXArray) -> MLXArray {
        inProj is ModelOptCalibratedLinear ? slice * MLXArray(1, dtype: slice.dtype) : slice
    }

    /// Super's graph decode step runs the conv, dt and gated-norm chains as
    /// the fused kernels of ``NemotronSuperMambaStep`` (bit-identical); prefill,
    /// the eager reference and every other model keep the op chains.
    var fusedSuperStep = true

    private var superStepActive: Bool {
        fusedSuperStep && isSuperGraph
    }

    /// Super with its ModelOpt projections on the graph path.
    var isSuperGraph: Bool {
        (inProj as? ModelOptCalibratedLinear)?.execution == .graph
    }

    /// `(silu(conv1d(concat(state, input))), retained state)` for one token per
    /// row, fused; nil where the op chain must run.
    private func superConvStep(state: MLXArray, input: MLXArray) -> (MLXArray, MLXArray)? {
        guard superStepActive, conv1d.groups == convDim, conv1d.padding == 0,
            conv1d.stride == 1, conv1d.dilation == 1, let bias = conv1d.bias,
            NemotronSuperMambaStep.supportsConvolve(
                state: state, input: input, weight: conv1d.weight, bias: bias)
        else { return nil }
        return NemotronSuperMambaStep.convolve(
            state: state, input: input, weight: conv1d.weight, bias: bias)
    }

    /// One-token SSM update with `computeDt` fused, or nil.
    private func superScanStep(
        hidden: MLXArray, B: MLXArray, C: MLXArray, dt: MLXArray, state: MLXArray?
    ) -> (MLXArray, MLXArray)? {
        guard superStepActive, hidden.dim(1) == 1, let state, dt.dtype == hidden.dtype
        else { return nil }
        return NemotronSuperMambaStep.scan(
            hiddenStates: hidden, ALog: aLog, B: B, C: C, D: D, dtRaw: dt, dtBias: dtBias,
            state: state, timeStepLimit: timeStepLimit)
    }

    /// `norm(y, gate:)`, fused for the graph decode step.
    private func gatedNorm(_ y: MLXArray, gate: MLXArray) -> MLXArray {
        if superStepActive, y.dim(1) == 1,
            NemotronSuperMambaStep.supportsGatedNorm(
                y, gate: gate, weight: norm.weight, groupSize: norm.groupSize)
        {
            return NemotronSuperMambaStep.gatedNorm(
                y, gate: gate, weight: norm.weight, groupSize: norm.groupSize, eps: norm.eps)
        }
        return norm(y, gate: gate)
    }

    private func applyConv(_ input: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        var convInput = input

        // Apply mask if present
        if let mask {
            let expandedMask = expandedDimensions(mask, axis: -1)
            convInput = MLX.where(expandedMask, convInput, MLXArray.zeros(like: convInput))
        }

        let batch = convInput.dim(0)
        let dtype = convInput.dtype
        var convState = cache?[0]

        if mask == nil, let cache, let state = convState,
            let (output, next) = superConvStep(state: state, input: convInput)
        {
            cache[0] = next
            return output
        }

        if convState == nil {
            if convKernelSize > 1 {
                convState = MLXArray.zeros([batch, convKernelSize - 1, convDim], dtype: dtype)
            } else {
                convState = MLXArray.zeros([batch, 0, convDim], dtype: dtype)
            }
        }

        let padded = concatenated([convState!, convInput], axis: 1)

        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            cache[0] = retainedConvState(padded[0..., start ..< end, 0...])
        }

        let convOutput = conv1d(padded)
        return silu(convOutput)
    }

    private func mambaForward(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let profile = inProj is ModelOptCalibratedLinear
        let projected = NemotronSuperPrefillProfile.stage(
            "Mamba in_proj", layer: self, active: profile, inputs: [hiddenStates]
        ) {
            inProj(hiddenStates)
        }
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        let convOutput = NemotronSuperPrefillProfile.stage(
            "Mamba conv", layer: self, active: profile, inputs: [convInput]
        ) {
            applyConv(convInput, mask: mask, cache: cache)
        }
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1
        )

        var hidden = convSplits[0]
        var B = convSplits[1]
        var C = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        B = B.reshaped([B.dim(0), B.dim(1), numGroups, ssmStateSize])
        C = C.reshaped([C.dim(0), C.dim(1), numGroups, ssmStateSize])

        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])

        let previousState = cache?[1]
        let scanned = NemotronSuperPrefillProfile.stages(
            "Mamba SSD", layer: self, active: profile,
            inputs: [hidden, B, C, dtArray] + (previousState.map { [$0] } ?? [])
        ) {
            if mask == nil,
                let (y, nextState) = superScanStep(
                    hidden: hidden, B: B, C: C, dt: dtArray, state: previousState)
            {
                return [y, nextState]
            }
            let (y, nextState) = selectedSSMUpdate(
                hiddenStates: hidden,
                ALog: aLog,
                B: B,
                C: C,
                D: D,
                dt: dtArray,
                dtBias: dtBias,
                state: previousState,
                timeStepLimit: timeStepLimit,
                mask: mask
            )
            return [y, nextState]
        }
        let (y, nextState) = (scanned[0], scanned[1])

        if let cache {
            cache[1] = keepsFloat32State ? nextState.asType(.float32) : nextState
        }

        let normed = NemotronSuperPrefillProfile.stage(
            "Mamba norm/gate", layer: self, active: profile, inputs: [y, gate]
        ) {
            gatedNorm(y.flattened(start: 2), gate: gate)
        }
        return NemotronSuperPrefillProfile.stage(
            "Mamba out_proj", layer: self, active: profile, inputs: [normed]
        ) {
            outProj(normed)
        }
    }

    /// CBv2 path with request-owned convolution and SSM state. The active
    /// requests are gathered into one rectangle for compute and split back
    /// into their owning recurrent transactions after the chunk.
    func cbv2Forward(
        _ hiddenStates: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        precondition(
            recurrentState.count == batch,
            "NemotronH CBv2 recurrent row count mismatch")

        let profile = inProj is ModelOptCalibratedLinear
        let projected = NemotronSuperPrefillProfile.stage(
            "Mamba in_proj", layer: self, active: profile, inputs: [hiddenStates]
        ) {
            inProj(hiddenStates)
        }
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(batch)
        ssmRows.reserveCapacity(batch)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, max(0, convKernelSize - 1), convDim],
                        dtype: hiddenStates.dtype))
            ssmRows.append(
                state?.ssm
                    ?? MLXArray.zeros(
                        [1, numHeads, headDim, ssmStateSize],
                        dtype: .float32))
        }
        let convState =
            convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState =
            ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)

        let convOutput: MLXArray
        let nextConvState: MLXArray
        if let (output, next) = superConvStep(state: convState, input: convInput) {
            (convOutput, nextConvState) = (output, next)
        } else {
            let padded = concatenated([convState, convInput], axis: 1)
            let end = padded.dim(1)
            nextConvState = retainedConvState(
                padded[0..., max(0, end - (convKernelSize - 1)) ..< end, 0...])
            convOutput = NemotronSuperPrefillProfile.stage(
                "Mamba conv", layer: self, active: profile, inputs: [padded]
            ) {
                silu(conv1d(padded))
            }
        }
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1)
        let hidden = convSplits[0].reshaped(
            batch, convOutput.dim(1), numHeads, headDim)
        let B = convSplits[1].reshaped(
            batch, convOutput.dim(1), numGroups, ssmStateSize)
        let C = convSplits[2].reshaped(
            batch, convOutput.dim(1), numGroups, ssmStateSize)
        let dtArray = dt.reshaped(batch, dt.dim(1), numHeads)
        let scanned = NemotronSuperPrefillProfile.stages(
            "Mamba SSD", layer: self, active: profile, inputs: [hidden, B, C, dtArray, ssmState]
        ) {
            if let (y, nextState) = superScanStep(
                hidden: hidden, B: B, C: C, dt: dtArray, state: ssmState)
            {
                return [y, nextState]
            }
            let (y, nextState) = selectedSSMUpdate(
                hiddenStates: hidden,
                ALog: aLog,
                B: B,
                C: C,
                D: D,
                dt: dtArray,
                dtBias: dtBias,
                state: ssmState,
                timeStepLimit: timeStepLimit,
                mask: nil)
            return [y, nextState]
        }
        let (y, nextState) = (scanned[0], scanned[1])
        let nextSSMState = nextState.asType(.float32)

        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: modelLayerIndex,
                    conv: nextConvState[row ..< row + 1],
                    ssm: nextSSMState[row ..< row + 1])
            } catch {
                preconditionFailure(
                    "NemotronH CBv2 recurrent stage failed at layer "
                        + "\(modelLayerIndex): \(error)")
            }
        }

        // The fp32 recurrent accumulator is persistent state, not an
        // activation-width promotion. mlx-lm narrows SSM output back to the
        // input dtype before gated normalization and the output projection.
        let normed = NemotronSuperPrefillProfile.stage(
            "Mamba norm/gate", layer: self, active: profile, inputs: [y, gate]
        ) {
            gatedNorm(y.asType(hiddenStates.dtype).flattened(start: 2), gate: gate)
        }
        let output = NemotronSuperPrefillProfile.stage(
            "Mamba out_proj", layer: self, active: profile, inputs: [normed]
        ) {
            outProj(normed)
        }
        // The staged states feed nothing on the residual stream, so MLX would
        // evaluate them after the whole trunk, keeping every layer's padded
        // convolution input and SSD contributions alive until the chunk ends
        // (3.2 GB at 2,048 tokens). As dependencies of this layer's output
        // they are computed, and their inputs released, within the layer.
        // Values are unchanged.
        return depends(input: output, dependencies: [nextConvState, nextSSMState])
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        mambaForward(x, mask: ssmMask, cache: cache as? MambaCache)
    }
}

// MARK: - Attention

class NemotronHAttention: Module, NemotronHMixer {
    let args: NemotronHConfiguration
    let scale: Float
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: any UnaryLayer
    @ModuleInfo(key: "k_proj") var wk: any UnaryLayer
    @ModuleInfo(key: "v_proj") var wv: any UnaryLayer
    @ModuleInfo(key: "o_proj") var wo: any UnaryLayer

    // NOTE: NemotronH attention does NOT use RoPE (unlike most transformer models)
    // The Python implementation at mlx_lm/models/nemotron_h.py lines 247-274 shows
    // direct attention without position embeddings

    init(_ args: NemotronHConfiguration) {
        self.args = args
        self.numHeads = args.numAttentionHeads
        self.numKeyValueHeads = args.numKeyValueHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        let attentionBias = args.attentionBias

        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: attentionBias)
        self._wk.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wv.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: attentionBias)

        super.init()
    }

    private func attentionForward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let profile = wq is ModelOptCalibratedLinear
        let qkv = NemotronSuperPrefillProfile.stages(
            "attention q/k/v", layer: self, active: profile, inputs: [x]
        ) {
            [wq(x), wk(x), wv(x)]
        }

        let queries = qkv[0].reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = qkv[1].reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = qkv[2].reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        // No RoPE applied - NemotronH attention uses direct attention without position embeddings

        let output = NemotronSuperPrefillProfile.stage(
            "attention KV update + attend", layer: self, active: profile,
            inputs: [queries, keys, values]
        ) {
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
        }
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return NemotronSuperPrefillProfile.stage(
            "attention o_proj", layer: self, active: profile, inputs: [output]
        ) {
            wo(output)
        }
    }

    func cbv2Forward(
        _ x: MLXArray,
        cache: any CBv2AttendingLayerCache,
        captureMTP: Bool = false
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let profile = wq is ModelOptCalibratedLinear && !captureMTP
        let qkv = NemotronSuperPrefillProfile.stages(
            "attention q/k/v", layer: self, active: profile, inputs: [x]
        ) {
            [
                captureMTP ? nemotronMTPLinearRows(x, wq) : wq(x),
                captureMTP ? nemotronMTPLinearRows(x, wk) : wk(x),
                captureMTP ? nemotronMTPLinearRows(x, wv) : wv(x),
            ]
        }
        let (q, k, v) = (qkv[0], qkv[1], qkv[2])
        let queries = q.reshaped(
            batch, length, numHeads, headDim
        ).transposed(0, 2, 1, 3)
        let keys = k.reshaped(
            batch, length, numKeyValueHeads, headDim
        ).transposed(0, 2, 1, 3)
        let values = v.reshaped(
            batch, length, numKeyValueHeads, headDim
        ).transposed(0, 2, 1, 3)

        // Keep parity with the selected MLX artifact's serial oracle: current
        // mlx-lm NemotronH attention does not apply RoPE despite rope_theta.
        let output = NemotronSuperPrefillProfile.stage(
            "attention KV update + attend", layer: self, active: profile,
            inputs: [queries, keys, values]
        ) {
            cache.updateAndAttend(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                sinks: nil)
        }
        let merged = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return NemotronSuperPrefillProfile.stage(
            "attention o_proj", layer: self, active: profile, inputs: [merged]
        ) {
            captureMTP ? nemotronMTPLinearRows(merged, wo) : wo(merged)
        }
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        attentionForward(x, mask: attentionMask, cache: cache)
    }
}

// MARK: - MLP

class NemotronHMLP: Module, UnaryLayer, NemotronHMixer {
    @ModuleInfo(key: "up_proj") var upProj: any UnaryLayer
    @ModuleInfo(key: "down_proj") var downProj: any UnaryLayer

    init(_ args: NemotronHConfiguration, intermediateSize: Int? = nil) {
        let intermediate = intermediateSize ?? args.intermediateSize
        self._upProj.wrappedValue = Linear(args.hiddenSize, intermediate, bias: args.mlpBias)
        self._downProj.wrappedValue = Linear(intermediate, args.hiddenSize, bias: args.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Super's ModelOpt projection fuses relu2 into its last kernel.
        if let up = upProj as? ModelOptCalibratedLinear { return downProj(up.squaredReLU(x)) }
        return downProj(relu2(upProj(x)))
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - MoE Gate

/// `x . weight^T` with each of 2 to 16 rows bit-identical to a one-row call, or
/// nil when the shape does not qualify (one row already is the one-row call).
func nemotronRowExactLinear(_ x: MLXArray, _ weight: MLXArray) -> MLXArray? {
    let width = x.dim(-1)
    let rows = x.size / max(1, width)
    guard (2 ... ModelOptRowExact.maximumRows).contains(rows), width > 64,
        weight.ndim == 2, weight.dim(1) == width, weight.dtype == x.dtype
    else { return nil }
    return ModelOptRowExact.matmul(x.reshaped(rows, width), weight)
        .reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
}

private func groupExpertSelect(
    gates: MLXArray,
    eSCB: MLXArray,  // e_score_correction_bias
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    let (bsz, seqLen) = (gates.dim(0), gates.dim(1))

    // Original scores using sigmoid
    let origScores = sigmoid(gates.asType(.float32))
    var scores = origScores + eSCB

    // Group-based selection if n_group > 1
    if nGroup > 1 {
        let numExperts = scores.dim(-1)
        let expertsPerGroup = numExperts / nGroup

        // Reshape to [batch, seq, n_group, experts_per_group]
        let groupScores = scores.reshaped(bsz, seqLen, nGroup, -1)

        // Get top-2 per group and sum for group scores (using sorted to get top values)
        let topKGroupScores = sorted(groupScores, axis: -1)[.ellipsis, ..<2].sum(
            axis: -1, keepDims: true)

        // Keep only top topkGroup groups (zero out the rest)
        let k = nGroup - topkGroup
        var groupIdx = argPartition(topKGroupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, expertsPerGroup])

        // Zero out scores from non-selected groups
        scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)

        // Flatten back
        scores = flattened(scores, start: -2, end: -1)
    }

    // Get top-k experts
    let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var finalScores = takeAlong(origScores, inds, axis: -1)

    // Normalize if needed
    if topK > 1 && normTopkProb {
        let denominator = finalScores.sum(axis: -1, keepDims: true) + 1e-20
        finalScores = finalScores / denominator
    }

    // Apply scaling factor
    finalScores = finalScores * routedScalingFactor

    return (inds, finalScores)
}

class NemotronHMoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eSCB: MLXArray

    init(_ args: NemotronHConfiguration) {
        self.topK = args.numExpertsPerTok
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb

        self._weight.wrappedValue = MLXArray.zeros([args.nRoutedExperts, args.hiddenSize])
        self._eSCB.wrappedValue = MLXArray.zeros([args.nRoutedExperts])

        super.init()
    }

    /// Super batched decode: router rows bit-identical to one-row calls.
    var rowExactBatches = false
    /// Super on the graph path selects experts with one fused kernel per row
    /// (bit-identical to `groupExpertSelect`); the eager reference and every
    /// other Nemotron-H model keep the op-by-op selection.
    var fusedSuperSelection = false

    private func select(_ gates: MLXArray) -> (MLXArray, MLXArray) {
        if fusedSuperSelection,
            NemotronSuperMoEKernels.supports(experts: gates.dim(-1), topK: topK, groups: nGroup)
        {
            return NemotronSuperMoEKernels.select(
                gates: gates, bias: eSCB, topK: topK, scalingFactor: routedScalingFactor,
                normalize: normTopkProb)
        }
        return groupExpertSelect(
            gates: gates,
            eSCB: eSCB,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates =
            rowExactBatches
            ? nemotronRowExactLinear(x, weight) ?? MLX.matmul(x, weight.transposed())
            : MLX.matmul(x, weight.transposed())
        return select(gates)
    }

    /// Keep each router projection at native M=1, but perform the row-local
    /// sigmoid/selection/normalization over the complete verification window.
    func mtpForwardRows(_ x: MLXArray) -> (MLXArray, MLXArray) {
        select(nemotronMTPMapRows(x) { MLX.matmul($0, weight.transposed()) })
    }

    /// Super batched decode and verify windows: 2 to 16 rows bit-identical to
    /// one-row calls, reading the router weight once per dispatch of up to
    /// eight rows; one row at a time otherwise. Selection is row-local either
    /// way.
    func superRowExactForwardRows(_ x: MLXArray) -> (MLXArray, MLXArray) {
        guard let gates = nemotronRowExactLinear(x, weight) else { return mtpForwardRows(x) }
        return select(gates)
    }
}

// MARK: - SwitchMLP for NemotronH (uses relu2 instead of silu/glu)

class NemotronHSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    /// Set for Super on the graph path when `indices` always come from the
    /// layer's own router: a top-k permutation of `0 ..< numExperts`, so the
    /// experts' device range check is skipped (always valid).
    var routerIndicesInRange = false

    /// Whether both projections take the graph route's per-row products for
    /// `rows` router-produced rows (`ModelOptNVFP4SwitchLinear.graphForward`
    /// tiles only from 16 rows and four per expert under `.tiled`).
    private func rowwiseRouterRoute(rows: Int) -> Bool {
        guard routerIndicesInRange else { return false }
        return [fc1, fc2].allSatisfy { layer in
            guard let experts = layer as? ModelOptNVFP4SwitchLinear, experts.execution == .graph
            else { return false }
            return experts.accumulation == .rowwise
                || rows < max(16, 4 * experts.logicalShape[0])
        }
    }

    /// The graph route with router-produced indices, or nil for the
    /// module's own (checked) call.
    private func routerRoute(
        _ experts: ModelOptNVFP4SwitchLinear, _ x: MLXArray, _ indices: MLXArray, sorted: Bool,
        squaredReLU: Bool
    ) -> MLXArray? {
        guard routerIndicesInRange, experts.execution == .graph else { return nil }
        do {
            return try experts.graphForward(
                x, indices, sortedIndices: sorted, squaredReLU: squaredReLU,
                indicesInRange: true
            ).value
        } catch {
            preconditionFailure("Invalid ModelOpt NVFP4 expert call: \(error)")
        }
    }

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        self._fc1.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        self._fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        // Rows only need sorting for a sorted matrix route; the graph route's
        // per-row products (below its tiled threshold) give each row the same
        // bits in any order.
        let doSort = indices.size > 64 && !rowwiseRouterRoute(rows: indices.size)

        var idx = indices
        var inverseOrder = MLXArray()
        let profile = fc1 is ModelOptNVFP4SwitchLinear

        if doSort {
            let sorted = NemotronSuperPrefillProfile.stages(
                "MoE sort/gather", layer: self, active: profile, inputs: [x, indices]
            ) {
                let (sx, si, inverse) = gatherSort(x: x, indices: indices)
                return [sx, si, inverse]
            }
            (x, idx, inverseOrder) = (sorted[0], sorted[1], sorted[2])
        }

        var y = NemotronSuperPrefillProfile.stage(
            "MoE expert fc1+relu2", layer: self, active: profile, inputs: [x, idx]
        ) {
            if let up = fc1 as? ModelOptNVFP4SwitchLinear {
                if let value = routerRoute(up, x, idx, sorted: doSort, squaredReLU: true) {
                    return value
                }
                // Super's ModelOpt experts fuse relu2 into fc1's last kernel.
                return up.squaredReLU(x, idx, sortedIndices: doSort)
            }
            return relu2(fc1(x, idx, sortedIndices: doSort))
        }
        y = NemotronSuperPrefillProfile.stage(
            "MoE expert fc2", layer: self, active: profile, inputs: [y]
        ) {
            if let down = fc2 as? ModelOptNVFP4SwitchLinear,
                let value = routerRoute(down, y, idx, sorted: doSort, squaredReLU: false)
            {
                return value
            }
            return fc2(y, idx, sortedIndices: doSort)
        }

        if doSort {
            y = NemotronSuperPrefillProfile.stage(
                "MoE unsort", layer: self, active: profile, inputs: [y]
            ) {
                scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
            }
        }

        return MLX.squeezed(y, axis: -2)
    }
}

// MARK: - MoE

class NemotronHMoE: Module, UnaryLayer, NemotronHMixer {
    let numExpertsPerTok: Int
    let hasSharedExperts: Bool
    /// Super on the graph path sums the routed outputs with one fused kernel
    /// (bit-identical to the multiply, sum and cast it replaces).
    var fusedSuperCombine = false
    let mtpCompiledRowsCache = NemotronH35MTPGraphCache()

    @ModuleInfo(key: "gate") var gate: NemotronHMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronHSwitchMLP
    @ModuleInfo(key: "fc1_latent_proj") var fc1LatentProj: Linear?
    @ModuleInfo(key: "fc2_latent_proj") var fc2LatentProj: Linear?
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHMLP?

    init(_ args: NemotronHConfiguration) {
        self.numExpertsPerTok = args.numExpertsPerTok
        self.hasSharedExperts = args.nSharedExperts != nil && args.nSharedExperts! > 0

        self._gate.wrappedValue = NemotronHMoEGate(args)
        self._switchMLP.wrappedValue = NemotronHSwitchMLP(
            inputDims: args.moeLatentSize ?? args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts
        )

        if let latentSize = args.moeLatentSize {
            precondition(latentSize > 0, "moe_latent_size must be positive")
            self._fc1LatentProj.wrappedValue = Linear(
                args.hiddenSize, latentSize, bias: args.mlpBias)
            self._fc2LatentProj.wrappedValue = Linear(
                latentSize, args.hiddenSize, bias: args.mlpBias)
        }

        if hasSharedExperts {
            self._sharedExperts.wrappedValue = NemotronHMLP(
                args,
                intermediateSize: args.moeSharedExpertIntermediateSize
            )
        }

        super.init()
    }

    /// Super batched decode: router and latent rows bit-identical to one-row calls.
    var rowExactBatches = false

    private func latent(_ projection: Linear, _ x: MLXArray) -> MLXArray {
        guard rowExactBatches, projection.bias == nil,
            let exact = nemotronRowExactLinear(x, projection.weight)
        else { return projection(x) }
        return exact
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (routed, shared) = parts(x)
        guard let shared else { return routed }
        return routed + shared
    }

    /// The routed experts' output (through the output latent projection) and
    /// the shared experts' output; the layer's output is their sum.
    func parts(_ x: MLXArray) -> (routed: MLXArray, shared: MLXArray?) {
        let profile = switchMLP.fc1 is ModelOptNVFP4SwitchLinear
        let routing = NemotronSuperPrefillProfile.stages(
            "MoE router/top-k", layer: self, active: profile, inputs: [x]
        ) {
            let (inds, scores) = gate(x)
            return [inds, scores]
        }
        let (inds, scores) = (routing[0], routing[1])
        if profile {
            NemotronSuperPrefillProfile.recordRouting(
                layer: self, indices: inds, experts: switchMLP.numExperts)
        }
        // Route in the original hidden space; only routed experts use the latent space.
        // Match the HF NemotronHMoE order: project -> experts -> weighted sum -> project.
        let routedInput = NemotronSuperPrefillProfile.stage(
            "MoE fc1 latent", layer: self, active: profile, inputs: [x]
        ) {
            fc1LatentProj.map { latent($0, x) } ?? x
        }
        var y = switchMLP(routedInput, inds)
        y = NemotronSuperPrefillProfile.stage(
            "MoE combine", layer: self, active: profile, inputs: [y, scores]
        ) {
            fusedSuperCombine && NemotronSuperMoEKernels.supportsCombine(experts: y.dim(-2))
                ? NemotronSuperMoEKernels.combine(y, scores: scores)
                : (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }
        if let fc2LatentProj {
            y = NemotronSuperPrefillProfile.stage(
                "MoE fc2 latent", layer: self, active: profile, inputs: [y]
            ) {
                latent(fc2LatentProj, y)
            }
        }

        guard let sharedExperts else { return (y, nil) }
        let shared = NemotronSuperPrefillProfile.stage(
            "MoE shared expert", layer: self, active: profile, inputs: [x]
        ) {
            sharedExperts(x)
        }
        return (y, shared)
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - Decoder Block

private class NemotronHBlock: Module {
    let blockType: NemotronHBlockType

    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Single mixer property with base Module type - concrete type assigned at init
    // This pattern is used by Jamba for feedForward: Module
    @ModuleInfo(key: "mixer") var mixer: Module

    init(_ args: NemotronHConfiguration, blockType: Character) {
        self.blockType = NemotronHBlockType(from: blockType)

        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Assign the appropriate concrete mixer type
        switch self.blockType {
        case .mamba:
            _mixer.wrappedValue = NemotronHMamba2Mixer(args)
        case .attention:
            _mixer.wrappedValue = NemotronHAttention(args)
        case .mlp:
            _mixer.wrappedValue = NemotronHMLP(args)
        case .moe:
            _mixer.wrappedValue = NemotronHMoE(args)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        x + mix(norm(x), attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
    }

    /// `mix` with a MoE layer's routed and shared outputs kept apart (their
    /// sum is the mixer output); other mixers return `(output, nil)`.
    func mixParts(
        _ hidden: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> (MLXArray, MLXArray?) {
        if let moe = mixer as? NemotronHMoE { return moe.parts(hidden) }
        return (mix(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache), nil)
    }

    /// The mixer's output for the normalized block input.
    func mix(
        _ hidden: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        // Cast to protocol and call - all mixer types conform to NemotronHMixer
        let mixerFunc = mixer as! NemotronHMixer
        return mixerFunc(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
    }

    /// Super on the graph path (ModelOpt Mamba projections).
    var isSuperGraph: Bool {
        (mixer as? NemotronHMamba2Mixer)?.isSuperGraph ?? false
    }

    func cbv2Forward(
        _ x: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        captureMTP: Bool = false
    ) -> MLXArray {
        let hidden = captureMTP ? nemotronMTPNormRows(x) { norm($0) } : norm(x)
        return x
            + cbv2Mix(
                hidden, modelLayerIndex: modelLayerIndex, attentionCache: attentionCache,
                recurrentState: recurrentState, captureMTP: captureMTP)
    }

    /// `cbv2Mix` with a MoE layer's routed and shared outputs kept apart.
    func cbv2MixParts(
        _ hidden: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> (MLXArray, MLXArray?) {
        if blockType == .moe, attentionCache == nil, let moe = mixer as? NemotronHMoE {
            return moe.parts(hidden)
        }
        return (
            cbv2Mix(
                hidden, modelLayerIndex: modelLayerIndex, attentionCache: attentionCache,
                recurrentState: recurrentState), nil
        )
    }

    /// The mixer's CBv2 output for the normalized block input.
    func cbv2Mix(
        _ hidden: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        captureMTP: Bool = false
    ) -> MLXArray {
        let output: MLXArray
        switch blockType {
        case .mamba:
            precondition(
                attentionCache == nil,
                "NemotronH recurrent layer received attention KV")
            let mamba = mixer as! NemotronHMamba2Mixer
            output =
                captureMTP
                ? mamba.cbv2ForwardCaptured(
                    hidden, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState)
                : mamba.cbv2Forward(
                    hidden, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState)
        case .attention:
            guard let attentionCache else {
                preconditionFailure("NemotronH attention layer is missing CBv2 cache")
            }
            output = (mixer as! NemotronHAttention).cbv2Forward(
                hidden, cache: attentionCache, captureMTP: captureMTP)
        case .mlp:
            precondition(
                attentionCache == nil,
                "NemotronH MLP layer received attention KV")
            let mlp = mixer as! NemotronHMLP
            output = captureMTP ? mlp.mtpForwardRows(hidden) : mlp(hidden)
        case .moe:
            precondition(
                attentionCache == nil,
                "NemotronH MoE layer received attention KV")
            let moe = mixer as! NemotronHMoE
            output = captureMTP ? moe.mtpForwardRows(hidden) : moe(hidden)
        }
        return output
    }
}

// MARK: - Backbone (matches Python's NemotronHModel which is stored as self.backbone)

private class NemotronHBackbone: Module {
    let args: NemotronHConfiguration

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronHBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    // Cache indices (into the cache list, not pattern indices)
    // Python: fa_idx counts Mamba layers before first Attention
    // Python: ssm_idx counts Attention layers before first Mamba
    let firstAttentionCacheIndex: Int?
    let firstMambaCacheIndex: Int?

    init(_ args: NemotronHConfiguration) {
        self.args = args
        precondition(args.vocabSize > 0)

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)

        let pattern = Array(args.hybridOverridePattern)
        self._layers.wrappedValue = pattern.map { NemotronHBlock(args, blockType: $0) }

        self._normF.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Calculate cache indices (only Mamba + Attention have caches)
        // fa_idx: count Mamba layers (M) before the first Attention layer (*)
        var faIdx: Int? = nil
        var mambaCount = 0
        for char in pattern {
            if char == "*" {
                faIdx = mambaCount
                break
            } else if char == "M" {
                mambaCount += 1
            }
        }
        self.firstAttentionCacheIndex = faIdx

        // ssm_idx: count Attention layers (*) before the first Mamba layer (M)
        var ssmIdx: Int? = nil
        var attnCount = 0
        for char in pattern {
            if char == "M" {
                ssmIdx = attnCount
                break
            } else if char == "*" {
                attnCount += 1
            }
        }
        self.firstMambaCacheIndex = ssmIdx

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embeddings(inputs)

        // Create attention mask using the first attention layer's cache.
        // Without a cache the mask is still causal, as in mlx-lm nemotron_h.py.
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let cacheIdx = firstAttentionCacheIndex else { return .none }
            let layerCache: KVCache? = cache.flatMap { cacheIdx < $0.count ? $0[cacheIdx] : nil }
            return createAttentionMask(h: hidden, cache: layerCache)
        }()

        // Create SSM mask using the first Mamba layer's cache
        // Python: ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        let ssmMask: MLXArray? = {
            guard let cacheIdx = firstMambaCacheIndex,
                let cache = cache,
                cacheIdx < cache.count,
                let mambaCache = cache[cacheIdx] as? MambaCache
            else { return nil }
            return mambaCache.makeMask(N: hidden.dim(1))
        }()

        // Track which cache to use for each layer
        var cacheCounter = 0
        let fused = superResidualNorm
        var normed = fused ? layers.first?.norm(hidden) : nil
        for (index, layer) in layers.enumerated() {
            let c: KVCache?
            if layer.blockType == .mamba || layer.blockType == .attention {
                c = cache?[cacheCounter]
                cacheCounter += 1
            } else {
                c = nil
            }

            if let current = normed {
                let (output, extra) = layer.mixParts(
                    current, attentionMask: attentionMask, ssmMask: ssmMask, cache: c)
                (hidden, normed) = residualNorm(hidden, output, extra, next: index + 1)
            } else {
                hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: c)
            }
        }
        if let normed { return normed }

        return normF(hidden)
    }

    func cbv2Forward(
        _ inputs: MLXArray,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        inputEmbedding: MLXArray? = nil,
        captureMTP: Bool = false
    ) -> MLXArray {
        let observation =
            CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(
                liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1))
            : nil
        defer { observation?.end() }
        var hidden = inputEmbedding ?? embeddings(inputs)
        var attentionIndex = 0
        var normed = !captureMTP && superResidualNorm ? layers.first?.norm(hidden) : nil
        for (modelLayerIndex, layer) in layers.enumerated() {
            let cache: (any CBv2AttendingLayerCache)?
            if layer.blockType == .attention {
                guard attentionIndex < caches.count else {
                    preconditionFailure(
                        "NemotronH CBv2 attention cache count is too small")
                }
                cache = caches[attentionIndex]
                attentionIndex += 1
            } else {
                cache = nil
            }
            if let current = normed {
                let (output, extra) = layer.cbv2MixParts(
                    current, modelLayerIndex: modelLayerIndex, attentionCache: cache,
                    recurrentState: recurrentState)
                (hidden, normed) = residualNorm(
                    hidden, output, extra, next: modelLayerIndex + 1)
            } else {
                hidden = layer.cbv2Forward(
                    hidden,
                    modelLayerIndex: modelLayerIndex,
                    attentionCache: cache,
                    recurrentState: recurrentState,
                    captureMTP: captureMTP)
            }
            if let fence = superPrefillFenceLayers, inputs.dim(1) > 1,
                (modelLayerIndex + 1) % fence == 0, modelLayerIndex + 1 < layers.count
            {
                eval([hidden] + (normed.map { [$0] } ?? []))
            }
        }
        precondition(
            attentionIndex == caches.count,
            "NemotronH CBv2 attention cache count is too large")
        if let normed { return normed }
        return captureMTP ? nemotronMTPNormRows(hidden) { normF($0) } : normF(hidden)
    }

    /// Cleared only to measure the op-by-op path.
    var fusedSuperResidualNorm = true

    /// Measurement only: a multi-token forward evaluates the residual stream
    /// every this many layers, so queued command buffers retain at most that
    /// many layers' intermediates. Evaluation boundaries never change results.
    var superPrefillFenceLayers: Int?

    /// Super on the graph path adds each block's output and applies the next
    /// norm (the next block's, or the final one) in one fused kernel,
    /// bit-identical to the add followed by the norm.
    var superResidualNorm: Bool {
        fusedSuperResidualNorm
            && (layers.first { $0.blockType == .mamba }?.isSuperGraph ?? false)
    }

    /// `(hidden + (output + extra), norm of that)` where the norm is layer
    /// `next`'s, or the final norm past the last layer.
    private func residualNorm(
        _ hidden: MLXArray, _ output: MLXArray, _ extra: MLXArray?, next: Int
    ) -> (MLXArray, MLXArray) {
        let norm = next < layers.count ? layers[next].norm : normF
        if NemotronSuperResidualNorm.supports(
            hidden, output, extra: extra, weight: norm.weight)
        {
            let fused = NemotronSuperResidualNorm.apply(
                hidden, output, extra: extra, weight: norm.weight, eps: norm.eps)
            return (fused.sum, fused.normed)
        }
        let mixed = extra.map { output + $0 } ?? output
        let sum = hidden + mixed
        return (sum, norm(sum))
    }
}

// MARK: - Main Model (matches Python's Model class)

public class NemotronHModel:
    Module, LLMModel, KVCacheDimensionProvider, LoRAModel
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "backbone") private var backbone: NemotronHBackbone
    let configuration: NemotronHConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: (any UnaryLayer)?

    public var loraLayers: [Module] {
        backbone.layers
    }

    public var cbv2LayerKinds: [CBv2LayerKind] {
        configuration.cbv2LayerKinds
    }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        let activationDTypes = resolvedRecurrentActivationDTypes()
        return configuration.cbv2RecurrentStateSpec(
            activationDTypes: activationDTypes,
            fallbackActivationDType:
                backbone.embeddings(MLXArray([0])).dtype)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        configuration.cbv2Capabilities
    }

    public init(_ args: NemotronHConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabSize

        // kvHeads array: non-zero for attention layers, zero for others
        let pattern = Array(args.hybridOverridePattern)
        self.kvHeads = pattern.compactMap { char -> Int? in
            let blockType = NemotronHBlockType(from: char)
            if blockType == .mamba || blockType == .attention {
                return blockType == .attention ? args.numKeyValueHeads : 0
            }
            return nil
        }

        self._backbone.wrappedValue = NemotronHBackbone(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = backbone(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        return out
    }

    /// Enables or disables every fused Super kernel of the graph path (the
    /// router, expert combine, Mamba step and block-boundary norm kernels and
    /// the router-index route), each bit-identical to the op chain it
    /// replaces; disabling is only for measuring them.
    /// Measurement only (see `NemotronHBackbone.superPrefillFenceLayers`);
    /// nil restores the single evaluation per forward.
    func setSuperPrefillFenceLayers(_ layers: Int?) {
        backbone.superPrefillFenceLayers = layers.map { max(1, $0) }
    }

    func setSuperFusedKernels(_ enabled: Bool) {
        backbone.fusedSuperResidualNorm = enabled
        for (_, module) in namedModules() {
            (module as? NemotronHMamba2Mixer)?.fusedSuperStep = enabled
        }
        if enabled {
            refreshSuperFusedSelection()
        } else {
            for (_, module) in namedModules() {
                guard let moe = module as? NemotronHMoE else { continue }
                moe.gate.fusedSuperSelection = false
                moe.fusedSuperCombine = false
                moe.switchMLP.routerIndicesInRange = false
            }
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let pattern = Array(configuration.hybridOverridePattern)
        return pattern.compactMap { char -> KVCache? in
            let blockType = NemotronHBlockType(from: char)
            switch blockType {
            case .mamba:
                return MambaCache()
            case .attention:
                return KVCacheSimple()
            case .mlp, .moe:
                return nil  // No cache needed for MLP/MoE layers
            }
        }
    }

    /// Resolve convolution-history widths from the loaded model itself.
    /// Quantized storage dtype is not the activation dtype, and retained
    /// fp32 parameters can promote later residuals. A lazy one-token graph
    /// through private caches exposes each Mamba layer's deterministic dtype
    /// without evaluating kernels or mutating request state.
    private func resolvedRecurrentActivationDTypes() -> [Int: DType] {
        let caches = newCache(parameters: nil)
        _ = backbone(
            MLXArray([0]).reshaped(1, 1),
            cache: caches)

        var result: [Int: DType] = [:]
        var cacheIndex = 0
        for (modelLayerIndex, blockType) in Array(configuration.hybridOverridePattern).enumerated()
        {
            let kind = NemotronHBlockType(from: blockType)
            guard kind == .mamba || kind == .attention else { continue }
            defer { cacheIndex += 1 }
            guard kind == .mamba,
                let cache = caches[cacheIndex] as? MambaCache,
                let conv = cache.state.first
            else { continue }
            result[modelLayerIndex] = conv.dtype
        }
        return result
    }

    /// Observe the loaded projections' native types using isolated lazy
    /// caches. This performs no eval and never installs probe state in a
    /// serving request. Packed integer embedding weights are not activation
    /// dtypes, and Mamba's fp32 path can promote later attention layers.
    func observedAttentionKVDTypes() -> [DType]? {
        let caches = newCache(parameters: nil)
        _ = backbone(MLXArray([0]).reshaped(1, 1), cache: caches)
        var result: [DType] = []
        var cacheIndex = 0
        for blockType in configuration.hybridOverridePattern {
            let kind = NemotronHBlockType(from: blockType)
            guard kind == .mamba || kind == .attention else { continue }
            defer { cacheIndex += 1 }
            guard kind == .attention else { continue }
            let state = caches[cacheIndex].state
            guard state.count == 2, state[0].dtype == state[1].dtype,
                [.float16, .bfloat16, .float32].contains(state[0].dtype)
            else { return nil }
            result.append(state[0].dtype)
        }
        return result.count == cbv2LayerKinds.count ? result : nil
    }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { storageIndex, kind in
            try makeLayerCache(kind.modelLayerIndex ?? storageIndex, kind)
        }
    }

    func cbv2Hidden(
        _ inputs: MLXArray,
        caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        inputEmbedding: MLXArray? = nil,
        captureMTP: Bool = false
    ) -> MLXArray {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("NemotronH CBv2 target received legacy KV cache")
            }
            return attending
        }
        return backbone.cbv2Forward(
            inputs, caches: attending, recurrentState: recurrentState,
            inputEmbedding: inputEmbedding, captureMTP: captureMTP)
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return backbone.embeddings.asLinear(hidden)
    }

    func mtpEmbedding(_ tokens: MLXArray) -> MLXArray { backbone.embeddings(tokens) }

    /// Unscaled lookup used to construct image-substituted prompt embeddings.
    /// This primitive does not advertise multimodal serving support.
    public func inputEmbeddings(_ tokens: MLXArray) -> MLXArray { backbone.embeddings(tokens) }

    public enum InputEmbeddingError: Error {
        case invalidTokenShape, invalidEmbeddingShape, invalidEmbeddingDType
    }

    /// Validate externally prepared embeddings before binding or mutating request
    /// state. Packed storage dtypes are not activation dtypes: probe the loaded
    /// embedding module lazily with a known valid token.
    public func validateInputEmbeddings(_ embeddings: MLXArray, tokens: MLXArray) throws {
        guard tokens.ndim == 2, tokens.dim(0) > 0, tokens.dim(1) > 0 else {
            throw InputEmbeddingError.invalidTokenShape
        }
        guard embeddings.shape == [tokens.dim(0), tokens.dim(1), configuration.hiddenSize] else {
            throw InputEmbeddingError.invalidEmbeddingShape
        }
        let nativeType = backbone.embeddings(MLXArray([Int32(0)])).dtype
        guard embeddings.dtype == nativeType else {
            throw InputEmbeddingError.invalidEmbeddingDType
        }
    }

    /// Throwing entrypoint for untrusted prepared tensors. Validation precedes
    /// graph construction and recurrent/KV staging. No media capability is enabled.
    public func prefillWithEmbeddings(
        _ inputs: MLXArray, embeddings: MLXArray, cache: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], requirement: CBv2PrefillRequirement
    ) throws -> MLXArray {
        try validateInputEmbeddings(embeddings, tokens: inputs)
        return cbv2RecurrentPrefill(
            inputs, inputEmbedding: embeddings, cache: cache,
            recurrentState: recurrentState, positionIds: nil, requirement: requirement)
    }

    /// Full-position embedding forward for the isolated Super CBv2 adapter.
    /// Validation precedes recurrent staging; callers retain request ownership.
    public func forwardWithEmbeddings(
        _ inputs: MLXArray, embeddings: MLXArray,
        cache: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation]
    ) throws -> MLXArray {
        try validateInputEmbeddings(embeddings, tokens: inputs)
        return logits(
            cbv2Hidden(
                inputs, caches: cache, recurrentState: recurrentState,
                inputEmbedding: embeddings))
    }

    func mtpShortlistLogits(_ hidden: MLXArray, ids: MLXArray) -> MLXArray {
        if let head = lmHead {
            if let quantized = head as? QuantizedLinear {
                var logits = quantizedMM(
                    hidden, quantized.weight[ids],
                    scales: quantized.scales[ids], biases: quantized.biases.map { $0[ids] },
                    transpose: true, groupSize: quantized.groupSize, bits: quantized.bits,
                    mode: quantized.mode)
                if let bias = quantized.bias { logits = logits + bias[ids] }
                return logits
            }
            if let linear = head as? Linear {
                var logits = matmul(hidden, linear.weight[ids].transposed(1, 0))
                if let bias = linear.bias { logits = logits + bias[ids] }
                return logits
            }
            // Preserve calibrated activation semantics in the serial fallback.
            // This does not qualify a learned Super MTP head.
            return take(head(hidden), ids, axis: -1)
        }
        let embedding = backbone.embeddings
        if let quantized = embedding as? QuantizedEmbedding {
            return quantizedMM(
                hidden, quantized.weight[ids], scales: quantized.scales[ids],
                biases: quantized.biases.map { $0[ids] }, transpose: true,
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
        }
        return matmul(hidden, embedding.weight[ids].transposed(1, 0))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            // Nemotron 3.5 target checkpoints may copy MTP declarations into
            // config.json without shipping the corresponding head. Keep the
            // serial target weight namespace explicit; a separately verified
            // MTP adapter owns mtp.* tensors when one is available.
            if key.hasPrefix("mtp.") {
                continue
            }
            var finalValue = value

            // Handle conv1d weight axis swap
            if key.contains("conv1d.weight"), value.dim(-1) != 1 {
                finalValue = value.swappedAxes(1, 2)
            }

            sanitized[key] = finalValue
        }

        // Stack experts: backbone.layers.{l}.mixer.experts.{e}.{proj}.weight -> backbone.layers.{l}.mixer.switch_mlp.{proj}.weight
        for l in 0 ..< configuration.numHiddenLayers {
            let prefix = "backbone.layers.\(l).mixer"
            for (m, n) in [("down_proj", "fc2"), ("up_proj", "fc1")] {
                if sanitized["\(prefix).experts.0.\(m).weight"] != nil {
                    let toJoin = (0 ..< configuration.nRoutedExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(m).weight")
                    }
                    if !toJoin.isEmpty {
                        sanitized["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitized
    }

    /// Predicate for casting: some parameters should stay in float32
    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("A_log")
        }
    }
}

extension NemotronHModel: CBv2RecurrentLanguageModelForwardable {
    public func cbv2Forward(
        _ tokens: MLXArray,
        caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        logits(
            cbv2Hidden(
                tokens, caches: caches, recurrentState: recurrentState))
    }
}

extension NemotronHModel: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { false }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        // The protocol is nonthrowing; callers at external boundaries must use
        // validateInputEmbeddings or prefillWithEmbeddings before state binding.
        if let inputEmbedding {
            precondition(
                (try? validateInputEmbeddings(inputEmbedding, tokens: inputs)) != nil,
                "NemotronH embedding prefill received an unvalidated shape or dtype")
        }
        precondition(positionIds == nil, "NemotronH serial parity omits RoPE positions")
        let hidden = cbv2Hidden(
            inputs, caches: cache ?? [], recurrentState: recurrentState,
            inputEmbedding: inputEmbedding)
        switch requirement {
        case .evaluationOnly:
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return logits(hidden[0..., -1, 0...])
        }
    }
}

// The pinned Super config represents JSON infinity as a tagged object.
// Accept only that exact representation; malformed objects must not default.
private struct NemotronExecutionLimit: Decodable {
    let value: Float
    init(from decoder: Decoder) throws {
        let scalar = try decoder.singleValueContainer()
        if let number = try? scalar.decode(Float.self) {
            value = number
            return
        }
        let tag = try scalar.decode([String: String].self)
        guard tag == ["__float__": "Infinity"] else {
            throw DecodingError.dataCorruptedError(
                in: scalar,
                debugDescription: "unsupported Nemotron execution-limit tag")
        }
        value = .infinity
    }
}

// MARK: - Configuration

public struct NemotronHConfiguration: Codable, Sendable {
    public var modelType: String = "nemotron_h"
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attentionBias: Bool
    public var mambaNumHeads: Int
    public var mambaHeadDim: Int
    public var mambaProjBias: Bool
    public var ssmStateSize: Int
    public var convKernel: Int
    public var nGroups: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var moeSharedExpertIntermediateSize: Int
    public var moeLatentSize: Int?
    public var nRoutedExperts: Int
    public var nSharedExperts: Int?
    public var numExpertsPerTok: Int
    public var hybridOverridePattern: String
    public var layerNormEpsilon: Float
    public var mlpBias: Bool
    public var useBias: Bool
    public var useConvBias: Bool
    public var tieWordEmbeddings: Bool
    public var ropeTheta: Float
    public var headDim: Int?
    public var nGroup: Int
    public var topkGroup: Int
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var timeStepLimitMin: Float
    public var timeStepLimitMax: Float
    public var chunkSize: Int
    public var mambaSSMCacheDType: String
    public var numNextnPredictLayers: Int
    public var mtpLayersBlockType: [String]

    public var cbv2LayerKinds: [CBv2LayerKind] {
        Array(hybridOverridePattern).enumerated().compactMap {
            modelLayerIndex, blockType in
            guard blockType == "*" else { return nil }
            return CBv2LayerKind(
                attention: .full,
                headDim: headDim ?? (hiddenSize / numAttentionHeads),
                kvHeads: numKeyValueHeads,
                queryHeads: numAttentionHeads,
                modelLayerIndex: modelLayerIndex)
        }
    }

    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        cbv2RecurrentStateSpec(
            activationDTypes: [:],
            fallbackActivationDType: activationDType)
    }

    func cbv2RecurrentStateSpec(
        activationDTypes: [Int: DType],
        fallbackActivationDType: DType
    ) -> CBv2RecurrentStateSpec {
        let convDim =
            mambaNumHeads * mambaHeadDim + 2 * nGroups * ssmStateSize
        let layers = Array(hybridOverridePattern).enumerated().compactMap {
            modelLayerIndex, blockType -> CBv2RecurrentLayerStateSpec? in
            guard blockType == "M" else { return nil }
            return CBv2RecurrentLayerStateSpec(
                modelLayerIndex: modelLayerIndex,
                convShape: [1, max(0, convKernel - 1), convDim],
                convDType:
                    activationDTypes[modelLayerIndex]
                    ?? fallbackActivationDType,
                ssmShape: [1, mambaNumHeads, mambaHeadDim, ssmStateSize],
                ssmDType: .float32)
        }
        return CBv2RecurrentStateSpec(layers: layers)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        // Serial first: recurrent snapshots, MTP, packed prefill, paged KV,
        // and compiled decode stay fail-closed until independently qualified.
        .initialRecurrentTarget
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mambaNumHeads = "mamba_num_heads"
        case mambaHeadDim = "mamba_head_dim"
        case mambaProjBias = "mamba_proj_bias"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case moeLatentSize = "moe_latent_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridOverridePattern = "hybrid_override_pattern"
        case layerNormEpsilon = "layer_norm_epsilon"
        case mlpBias = "mlp_bias"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
        case chunkSize = "chunk_size"
        case mambaSSMCacheDType = "mamba_ssm_cache_dtype"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case mtpLayersBlockType = "mtp_layers_block_type"
    }

    enum AliasCodingKeys: String, CodingKey {
        case layersBlockType = "layers_block_type"
        case normEpsilon = "norm_eps"
        case timeStepLimit = "time_step_limit"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let aliases = try decoder.container(keyedBy: AliasCodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_h"
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mambaNumHeads = try container.decode(Int.self, forKey: .mambaNumHeads)
        mambaHeadDim = try container.decode(Int.self, forKey: .mambaHeadDim)
        mambaProjBias = try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        ssmStateSize = try container.decode(Int.self, forKey: .ssmStateSize)
        convKernel = try container.decode(Int.self, forKey: .convKernel)
        nGroups = try container.decode(Int.self, forKey: .nGroups)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        moeSharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        moeLatentSize = try container.decodeIfPresent(Int.self, forKey: .moeLatentSize)
        if let moeLatentSize, moeLatentSize <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .moeLatentSize, in: container,
                debugDescription: "moe_latent_size must be positive when present")
        }
        nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        layerNormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon)
            ?? aliases.decodeIfPresent(Float.self, forKey: .normEpsilon)
            ?? 1e-5
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useConvBias = try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0

        // Legacy Nemotron Nano uses compact hybrid_override_pattern symbols.
        // Nemotron 3.5 uses descriptive layers_block_type entries instead.
        if let patternString = try? container.decode(String.self, forKey: .hybridOverridePattern) {
            hybridOverridePattern = patternString
        } else if let patternArray = try? container.decode(
            [String].self, forKey: .hybridOverridePattern)
        {
            hybridOverridePattern = patternArray.joined()
        } else if let blockTypes = try? aliases.decode(
            [String].self, forKey: .layersBlockType)
        {
            guard blockTypes.count == numHiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layersBlockType, in: aliases,
                    debugDescription:
                        "layers_block_type count \(blockTypes.count) does not match "
                        + "num_hidden_layers \(numHiddenLayers)")
            }
            hybridOverridePattern = try blockTypes.map { blockType in
                switch blockType {
                case "mamba": return "M"
                case "attention": return "*"
                case "mlp": return "-"
                case "moe": return "E"
                default:
                    throw DecodingError.dataCorruptedError(
                        forKey: .layersBlockType, in: aliases,
                        debugDescription: "unsupported NemotronH block type \(blockType)")
                }
            }.joined()
        } else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern, in: container,
                debugDescription:
                    "configuration requires hybrid_override_pattern or layers_block_type")
        }

        // Explicit execution limits are distinct from the published
        // timestep-initialization metadata. Reject malformed arrays before
        // indexing; an empty operator-writable config must not crash loading.
        if let limits = try aliases.decodeIfPresent(
            [NemotronExecutionLimit].self, forKey: .timeStepLimit)?.map(\.value)
        {
            guard (1 ... 2).contains(limits.count) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .timeStepLimit, in: aliases,
                    debugDescription: "time_step_limit requires one or two values")
            }
            timeStepLimitMin = limits[0]
            timeStepLimitMax = limits.count > 1 ? limits[1] : limits[0]
        } else if let limits = try? container.decode([Float].self, forKey: .timeStepLimitMin) {
            // Compatibility with an early Swift fixture that placed the
            // two-element array under time_step_limit_min.
            guard (1 ... 2).contains(limits.count) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .timeStepLimitMin, in: container,
                    debugDescription: "time_step_limit_min array requires one or two values")
            }
            timeStepLimitMin = limits[0]
            timeStepLimitMax = limits.count > 1 ? limits[1] : limits[0]
        } else {
            timeStepLimitMin =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMin)
                ?? aliases.decodeIfPresent(Float.self, forKey: .timeStepMin)
                ?? 0.0
            timeStepLimitMax =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMax)
                ?? aliases.decodeIfPresent(Float.self, forKey: .timeStepMax)
                ?? Float.infinity
        }
        guard timeStepLimitMin.isFinite, timeStepLimitMin >= 0,
            !timeStepLimitMax.isNaN, timeStepLimitMax >= timeStepLimitMin
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .timeStepLimitMin, in: container,
                debugDescription: "timestep bounds must be nonnegative and ordered")
        }
        chunkSize = try container.decodeIfPresent(Int.self, forKey: .chunkSize) ?? 128
        mambaSSMCacheDType =
            try container.decodeIfPresent(String.self, forKey: .mambaSSMCacheDType) ?? "float32"
        numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
        mtpLayersBlockType =
            try container.decodeIfPresent([String].self, forKey: .mtpLayersBlockType) ?? []
    }

    /// Memberwise initializer for testing
    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        mambaNumHeads: Int,
        mambaHeadDim: Int,
        ssmStateSize: Int,
        convKernel: Int,
        nGroups: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        moeSharedExpertIntermediateSize: Int,
        nRoutedExperts: Int,
        numExpertsPerTok: Int,
        hybridOverridePattern: String,
        layerNormEpsilon: Float = 1e-5,
        attentionBias: Bool = false,
        mambaProjBias: Bool = false,
        mlpBias: Bool = false,
        useBias: Bool = false,
        useConvBias: Bool = true,
        tieWordEmbeddings: Bool = false,
        ropeTheta: Float = 10000.0,
        headDim: Int? = nil,
        nSharedExperts: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1.0,
        timeStepLimitMin: Float = 0.0,
        timeStepLimitMax: Float = .infinity,
        chunkSize: Int = 128,
        mambaSSMCacheDType: String = "float32",
        numNextnPredictLayers: Int = 0,
        mtpLayersBlockType: [String] = [],
        moeLatentSize: Int? = nil
    ) {
        self.modelType = "nemotron_h"
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.attentionBias = attentionBias
        self.mambaNumHeads = mambaNumHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = mambaProjBias
        self.ssmStateSize = ssmStateSize
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.moeLatentSize = moeLatentSize
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.hybridOverridePattern = hybridOverridePattern
        self.layerNormEpsilon = layerNormEpsilon
        self.mlpBias = mlpBias
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.headDim = headDim
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
        self.chunkSize = chunkSize
        self.mambaSSMCacheDType = mambaSSMCacheDType
        self.numNextnPredictLayers = numNextnPredictLayers
        self.mtpLayersBlockType = mtpLayersBlockType
    }
}
