// Inkling 2 on the CBv2 paged engine. Each sequence owns 48 slots (pass 1 of
// every stored layer, then passes 2 and 3 of the loop layers): 8 full and 40
// sliding-window paged K/V layers, and per slot four short-convolution states
// carried as request-owned recurrent state. K/V lives only in pages; attention
// reads per-row gathered views of them (a compute view, not storage).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Attention over a slot's paged history. The default computes with plain MLX
/// ops over gathered pages; the paged kernel replaces it behind this seam.
public protocol Inkling2PagedAttentionCore: AnyObject {
    /// `cache` holds the slot's pages for the rows of `inputs`. Writes the
    /// step's K/V into the pages and returns `[B, L, heads * headDim]`.
    func attend(
        slot: Int, cache: any CBv2AttendingLayerCache, rowOffsets: [Int],
        _ inputs: Inkling2AttentionInputs
    ) -> MLXArray
}

/// Gathered-view attention: `updateKVAndAdvanceOffsets` writes the K/V into
/// the row's pages and returns its attendable history, then the reference math
/// runs per row (rows never share a reduction, so a row is batch-invariant).
public final class Inkling2GatheredAttentionCore: Inkling2PagedAttentionCore {
    public init() {}

    public func attend(
        slot: Int, cache: any CBv2AttendingLayerCache, rowOffsets: [Int],
        _ inputs: Inkling2AttentionInputs
    ) -> MLXArray {
        guard let gathered = cache as? any CBv2Qwen4GatheredCache else {
            preconditionFailure("Inkling2 slot \(slot): the layer cache cannot expose page views")
        }
        let (b, l) = (inputs.queries.dim(0), inputs.queries.dim(2))
        let width = inputs.queries.dim(1) * inputs.queries.dim(3)
        guard gathered.qwen4ValidateProjectedKV(keys: inputs.keys, values: inputs.values) else {
            return MLXArray.zeros([b, l, width], dtype: inputs.queries.dtype)
        }
        let views = gathered.updateKVAndAdvanceOffsets(keys: inputs.keys, values: inputs.values)
        if gathered.qwen4HasWriteFault || views.count != b {
            return MLXArray.zeros([b, l, width], dtype: inputs.queries.dtype)
        }
        var rows: [MLXArray] = []
        rows.reserveCapacity(b)
        for (row, view) in views.enumerated() {
            let t = view.keys.dim(2)
            let end = rowOffsets[row] + l
            let viewStart = end - t
            guard l > 1, Inkling2RowExact.active else {
                let keyPositions = MLXArray((viewStart ..< end).map { Int32($0) })
                rows.append(
                    Inkling2ReferenceAttention.attend(
                        inputs.row(row), keys: view.keys, values: view.values,
                        keyPositions: keyPositions))
                continue
            }
            // Verify windows: each query attends exactly the keys a decode step
            // at its position would see, with the same shapes.
            let rowInputs = inputs.row(row)
            var columns: [MLXArray] = []
            for j in 0 ..< l {
                let position = rowOffsets[row] + j
                let first = max(viewStart, inputs.window.map { position - $0 + 1 } ?? viewStart)
                let range = (first - viewStart) ..< (position + 1 - viewStart)
                columns.append(
                    Inkling2ReferenceAttention.attend(
                        rowInputs.column(j), keys: view.keys[0..., 0..., range, 0...],
                        values: view.values[0..., 0..., range, 0...],
                        keyPositions: MLXArray((first ... position).map { Int32($0) })))
            }
            rows.append(concatenated(columns, axis: 1))
        }
        return rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
    }
}

extension Inkling2AttentionInputs {
    /// The inputs of query position `j` alone (keys and values unchanged).
    public func column(_ j: Int) -> Inkling2AttentionInputs {
        Inkling2AttentionInputs(
            queries: queries[0..., 0..., j ..< j + 1], keys: keys, values: values,
            relativeLogits: relativeLogits[0..., 0..., j ..< j + 1],
            relativeStates: relativeStates[0..., j ..< j + 1],
            relativeProjection: relativeProjection,
            sinks: sinks[0..., 0..., j ..< j + 1], positions: positions[0..., j ..< j + 1],
            window: window, scale: scale, isPrefill: isPrefill)
    }

    /// The inputs of one batch row.
    public func row(_ i: Int) -> Inkling2AttentionInputs {
        func pick(_ x: MLXArray) -> MLXArray { x.dim(0) == 1 ? x : x[i ..< i + 1] }
        return Inkling2AttentionInputs(
            queries: pick(queries), keys: pick(keys), values: pick(values),
            relativeLogits: pick(relativeLogits), relativeStates: pick(relativeStates),
            relativeProjection: relativeProjection, sinks: pick(sinks),
            positions: pick(positions), window: window, scale: scale, isPrefill: isPrefill)
    }
}

/// One CBv2 step: positions from the slot caches' host offsets, conv states
/// gathered from and staged into the rows' recurrent transactions.
final class Inkling2EngineStepState: Inkling2StepState {
    let positions: MLXArray
    let caches: [any CBv2AttendingLayerCache]
    let recurrentState: [CBv2RecurrentStateEvaluation]
    let rowOffsets: [Int]
    let core: Inkling2PagedAttentionCore
    let configuration: Inkling2TextConfiguration
    let dtype: DType
    /// Verify windows stage every position's state (`stageCaptured`) so the
    /// engine can keep the accepted prefix.
    let capturesPositions: Bool
    private var inputs: [Int: [MLXArray]] = [:]
    private var outputs: [Int: [Inkling2ConvKind: (state: MLXArray, input: MLXArray)]] = [:]

    init(
        configuration: Inkling2TextConfiguration, caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], length: Int, positionIds: MLXArray?,
        core: Inkling2PagedAttentionCore, dtype: DType, capturesPositions: Bool = false
    ) {
        precondition(
            caches.count == configuration.slotLayers.count,
            "Inkling2 CBv2 expects \(configuration.slotLayers.count) slot caches, got \(caches.count)"
        )
        self.configuration = configuration
        self.caches = caches
        self.recurrentState = recurrentState
        self.core = core
        self.dtype = dtype
        self.capturesPositions = capturesPositions
        rowOffsets = caches[0].rows.map(\.absoluteOffset)
        precondition(
            recurrentState.count == rowOffsets.count,
            "Inkling2 CBv2 recurrent rows \(recurrentState.count) != batch \(rowOffsets.count)")
        if let positionIds, positionIds.ndim == 2 {
            positions = positionIds.asType(.int32)
        } else {
            let rows = rowOffsets.map { offset in
                (0 ..< length).map { Int32(offset + $0) }
            }
            positions = MLXArray(rows.flatMap { $0 }, [rowOffsets.count, length])
        }
    }

    private func channels(_ kind: Inkling2ConvKind) -> Int {
        switch kind {
        case .key, .value: configuration.kvHeads * configuration.headDim
        case .postAttention, .postMLP: configuration.hiddenSize
        }
    }

    /// Packed per slot: `conv` = key || value, `ssm` = postAttention || postMLP.
    private func gathered(slot: Int) -> [MLXArray] {
        if let cached = inputs[slot] { return cached }
        let k = configuration.convKernelSize - 1
        let kv = channels(.key)
        let hidden = channels(.postAttention)
        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: slot)
            convRows.append(state?.conv ?? MLXArray.zeros([1, k, 2 * kv], dtype: dtype))
            ssmRows.append(state?.ssm ?? MLXArray.zeros([1, k, 2 * hidden], dtype: dtype))
        }
        let conv = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssm = ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)
        let split = [
            conv[.ellipsis, ..<kv], conv[.ellipsis, kv...],
            ssm[.ellipsis, ..<hidden], ssm[.ellipsis, hidden...],
        ]
        inputs[slot] = split
        return split
    }

    func convState(slot: Int, _ kind: Inkling2ConvKind) -> MLXArray? {
        gathered(slot: slot)[kind.rawValue]
    }

    /// Stages `slot`'s next states already packed as the transactions store
    /// them (instead of the four `setConvState` calls): `conv` key || value,
    /// `ssm` postAttention || postMLP, in the activation dtype, `[B, kernel - 1, C]`;
    /// with `capturesPositions`, every window position's state,
    /// `[B * length, kernel - 1, C]` (row `b` at `b * length ..< (b + 1) * length`).
    func stagePackedConvStates(slot: Int, conv: MLXArray, ssm: MLXArray, length: Int) {
        precondition(
            outputs[slot] == nil, "Inkling2 CBv2: slot \(slot) mixes packed and per-kind stages")
        do {
            for (row, evaluation) in recurrentState.enumerated() {
                if capturesPositions {
                    let rows = (row * length) ..< ((row + 1) * length)
                    try evaluation.stageCaptured(
                        modelLayerIndex: slot, conv: conv[rows], ssm: ssm[rows], positions: length)
                } else {
                    try evaluation.stage(
                        modelLayerIndex: slot, conv: conv[row ..< row + 1],
                        ssm: ssm[row ..< row + 1])
                }
            }
        } catch {
            preconditionFailure("Inkling2 CBv2 recurrent stage failed at slot \(slot): \(error)")
        }
    }

    func setConvState(
        slot: Int, _ kind: Inkling2ConvKind, _ state: MLXArray, input: MLXArray
    ) {
        outputs[slot, default: [:]][kind] = (state, input)
        guard let all = outputs[slot], all.count == Inkling2ConvKind.allCases.count else { return }
        outputs[slot] = nil
        do {
            if capturesPositions {
                try stageCaptured(slot: slot, all)
                return
            }
            let conv = concatenated([all[.key]!.state, all[.value]!.state], axis: -1).asType(dtype)
            let ssm = concatenated(
                [all[.postAttention]!.state, all[.postMLP]!.state], axis: -1
            ).asType(dtype)
            for (row, evaluation) in recurrentState.enumerated() {
                try evaluation.stage(
                    modelLayerIndex: slot, conv: conv[row ..< row + 1], ssm: ssm[row ..< row + 1])
            }
        } catch {
            preconditionFailure("Inkling2 CBv2 recurrent stage failed at slot \(slot): \(error)")
        }
    }

    /// `[L, kernel - 1, channels]` per row: the state after each window position.
    private func stageCaptured(
        slot: Int, _ all: [Inkling2ConvKind: (state: MLXArray, input: MLXArray)]
    ) throws {
        let previous = gathered(slot: slot)
        let keep = configuration.convKernelSize - 1
        func context(_ kind: Inkling2ConvKind) -> MLXArray {
            concatenated([previous[kind.rawValue], all[kind]!.input.asType(dtype)], axis: 1)
        }
        let conv = concatenated([context(.key), context(.value)], axis: -1)
        let ssm = concatenated([context(.postAttention), context(.postMLP)], axis: -1)
        let positions = conv.dim(1) - keep
        for (row, evaluation) in recurrentState.enumerated() {
            func stack(_ x: MLXArray) -> MLXArray {
                concatenated(
                    (0 ..< positions).map { s in x[row ..< row + 1, (s + 1) ..< (s + 1 + keep)] },
                    axis: 0)
            }
            try evaluation.stageCaptured(
                modelLayerIndex: slot, conv: stack(conv), ssm: stack(ssm), positions: positions)
        }
    }

    func attend(slot: Int, _ inputs: Inkling2AttentionInputs) -> MLXArray {
        core.attend(slot: slot, cache: caches[slot], rowOffsets: rowOffsets, inputs)
    }
}

extension Inkling2TextConfiguration {
    /// One paged layer per slot; slot index = `modelLayerIndex`.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        slotLayers.enumerated().map { slot, layer in
            CBv2LayerKind(
                attention: isSliding(layer: layer) ? .slidingWindow(slidingWindow) : .full,
                headDim: headDim, kvHeads: kvHeads, queryHeads: attentionHeads,
                modelLayerIndex: slot)
        }
    }

    public func cbv2RecurrentStateSpec(activationDType: DType) -> CBv2RecurrentStateSpec {
        let k = convKernelSize - 1
        return CBv2RecurrentStateSpec(
            layers: slotLayers.indices.map { slot in
                CBv2RecurrentLayerStateSpec(
                    modelLayerIndex: slot,
                    convShape: [1, k, 2 * kvHeads * headDim], convDType: activationDType,
                    ssmShape: [1, k, 2 * hiddenSize], ssmDType: activationDType)
            })
    }
}

extension Inkling2Model {
    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.text.cbv2LayerKinds }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { slot, kind in
            try makeLayerCache(kind.modelLayerIndex ?? slot, kind)
        }
    }

    var activationDType: DType { model.embedding.weight.dtype }

    func attendingCaches(_ cache: [KVCache]?) -> [any CBv2AttendingLayerCache] {
        (cache ?? []).map { cache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Inkling2 CBv2 target received a legacy KV cache")
            }
            return attending
        }
    }

    /// Decode steps and verify windows (at most 16 positions per row) run
    /// row-exact so a row's bits never depend on its batch or window company.
    static func rowExact(_ tokens: MLXArray, _ inputEmbedding: MLXArray?) -> Bool {
        (inputEmbedding?.dim(1) ?? tokens.dim(1)) <= 16
    }

    /// The trunk over the engine's slot caches. Returns the final-norm hidden state.
    func cbv2Hidden(
        _ tokens: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        capture: Set<Int> = [], capturesPositions: Bool = false
    ) -> (hidden: MLXArray, captured: Inkling2CapturedHidden) {
        let embeddings = inputEmbedding ?? model.embed(tokens)
        Inkling2Diagnostics.step(
            self, Inkling2PrefillStep.active ? "prefill" : "forward", embeddings.dim(1))
        let state = Inkling2EngineStepState(
            configuration: configuration.text, caches: attendingCaches(cache),
            recurrentState: recurrentState, length: embeddings.dim(1), positionIds: positionIds,
            core: pagedAttentionCore, dtype: activationDType,
            capturesPositions: capturesPositions)
        return Inkling2RowExact.withPrefill(blockRows: invariantPrefillBlockRows) {
            model(embeddings: embeddings, state: state, capture: capture)
        }
    }

    /// The drafter's view of a CBv2 step: logits plus the hidden states of
    /// `captureVisits` (see `Inkling2CapturedHidden`). `capturesPositions`
    /// stages per-position recurrent stacks for an MTP verify window
    /// (`CBv2RecurrentStateEvaluation.stageCaptured`); otherwise the step
    /// stages final states as a plain forward does. Decode steps and verify
    /// windows run row-exact, as `cbv2Forward` does.
    public func cbv2ForwardCapturing(
        _ tokens: MLXArray, inputEmbedding: MLXArray? = nil, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        captureVisits: Set<Int>, capturesPositions: Bool = false
    ) -> (logits: MLXArray, captured: Inkling2CapturedHidden) {
        Inkling2RowExact.$active.withValue(Self.rowExact(tokens, inputEmbedding)) {
            let (hidden, captured) = cbv2Hidden(
                tokens, inputEmbedding: inputEmbedding, cache: caches,
                recurrentState: recurrentState, positionIds: positionIds,
                capture: captureVisits, capturesPositions: capturesPositions)
            return (logits(hidden), captured)
        }
    }
}

extension Inkling2Model: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        configuration.text.cbv2RecurrentStateSpec(activationDType: activationDType)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        // Pages plus conv state at a boundary restore a prefix exactly.
        capabilities.supportsRecurrentCheckpointReuse = true
        // The DSpark drafter (Inkling2DSparkCBv2.swift) drives MTP rounds;
        // without the flag the engine never builds the round driver.
        capabilities.supportsMTP = true
        // With fixed-row prefill products (invariantPrefillBlockRows) prefill is
        // chunk-invariant (Inkling2InvarianceProbeTests): any partition of a
        // prompt, and a restore at any checkpoint boundary, leaves the bits of
        // the one-shot prefill. With MLX's own products the bits change with
        // the chunk length, so a row's state must not depend on the chunks
        // other rows leave it: fixed grid, grid-aligned checkpoints.
        let partitionDependent = invariantPrefillBlockRows == nil
        capabilities.requiresFixedPrefillChunkGrid = partitionDependent
        capabilities.alignsRecurrentCheckpointsToPrefillChunks = partitionDependent
        return capabilities
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        Inkling2RowExact.$active.withValue(Self.rowExact(tokens, nil)) {
            logits(
                cbv2Hidden(
                    tokens, inputEmbedding: nil, cache: caches, recurrentState: recurrentState,
                    positionIds: nil
                ).hidden)
        }
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> MLXArray {
        Inkling2RowExact.$active.withValue(Self.rowExact(tokens, nil)) {
            logits(
                cbv2Hidden(
                    tokens, inputEmbedding: nil, cache: caches, recurrentState: recurrentState,
                    positionIds: positionIds
                ).hidden)
        }
    }

    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        preconditionFailure("Inkling2 embedding prefill requires request-owned recurrent state")
    }

    /// `inputEmbedding` is already normed (`scaledInputEmbeddings`); no second embed_norm.
    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        Inkling2RowExact.$active.withValue(Self.rowExact(inputs, inputEmbedding)) {
            logits(
                cbv2Hidden(
                    inputs, inputEmbedding: inputEmbedding, cache: cache,
                    recurrentState: recurrentState, positionIds: positionIds
                ).hidden)
        }
    }
}

/// Prompt chunks skip the vocabulary projection of rows nobody reads; every
/// K/V write and recurrent stage is the same as the full forward.
extension Inkling2Model: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { false }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        let hidden = Inkling2PrefillStep.$active.withValue(true) {
            cbv2Hidden(
                inputs, inputEmbedding: inputEmbedding, cache: cache,
                recurrentState: recurrentState, positionIds: positionIds
            ).hidden
        }
        Inkling2Diagnostics.step(self, "prefill-built", inputEmbedding?.dim(1) ?? inputs.dim(1))
        switch requirement {
        case .evaluationOnly:
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return logits(hidden[0..., -1, 0...])
        }
    }
}

/// Decode steps commit by operation count only (see `FlintDecodeCommandBuffers`);
/// results never depend on the partitioning.
extension Inkling2Model: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        Self.decodeCommandBufferOperations.map {
            GPU.CommandBufferLimits(operations: $0, megabytes: 1 << 20)
        }
    }

    /// Operations per decode command buffer: 400 (M5 Max: 400 and 1000 decode
    /// ~10 % faster than 100 at one and eight streams).
    /// `DARKBLOOM_INKLING2_DECODE_CB_OPS` overrides it; 0 keeps MLX's defaults.
    static let decodeCommandBufferOperations: Int? = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_DECODE_CB_OPS"]
        guard let raw, let value = Int(raw) else { return 400 }
        return value > 0 ? value : nil
    }()
}
