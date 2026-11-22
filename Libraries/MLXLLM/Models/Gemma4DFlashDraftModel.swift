import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - DFlash Configuration

/// Configuration decoded from a `DFlashDraftModel` checkpoint's `config.json`
/// (e.g. `z-lab/gemma-4-26B-A4B-it-DFlash`).
public struct Gemma4DFlashConfiguration: Codable, Sendable, Equatable {
    public struct NestedDFlashConfig: Codable, Sendable, Equatable {
        public let maskTokenId: Int
        public let targetLayerIds: [Int]
        public let blockSize: Int?
        public let finalLogitSoftcapping: Float?
        public let inputEmbeddingScale: Float?
        public let outputMultiplier: Float?

        enum CodingKeys: String, CodingKey {
            case maskTokenId = "mask_token_id"
            case targetLayerIds = "target_layer_ids"
            case blockSize = "block_size"
            case finalLogitSoftcapping = "final_logit_softcapping"
            case inputEmbeddingScale = "input_embedding_scale"
            case outputMultiplier = "output_multiplier"
        }
    }

    public let architectures: [String]?
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let numTargetLayers: Int
    public let blockSize: Int
    public let layerTypes: [String]
    public let slidingWindow: Int?
    public let finalLogitSoftcapping: Float?
    public let isCausal: Bool?
    public let hasUnsupportedRoPEScaling: Bool
    public let dflashConfig: NestedDFlashConfig

    public var maskTokenId: Int { dflashConfig.maskTokenId }
    public var targetLayerIds: [Int] { dflashConfig.targetLayerIds }
    public var effectiveBlockSize: Int { dflashConfig.blockSize ?? blockSize }
    public var inputEmbeddingScale: Float { dflashConfig.inputEmbeddingScale ?? 1.0 }
    public var outputMultiplier: Float { dflashConfig.outputMultiplier ?? 1.0 }
    public var effectiveFinalLogitSoftcapping: Float? {
        dflashConfig.finalLogitSoftcapping ?? finalLogitSoftcapping
    }

    enum CodingKeys: String, CodingKey {
        case architectures
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numTargetLayers = "num_target_layers"
        case blockSize = "block_size"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case isCausal = "is_causal"
        case dflashConfig = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / max(1, numAttentionHeads))
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        func hasNonNullValue(forKey key: CodingKeys) -> Bool {
            c.contains(key) && !((try? c.decodeNil(forKey: key)) ?? true)
        }
        hasUnsupportedRoPEScaling =
            hasNonNullValue(forKey: .ropeScaling) || hasNonNullValue(forKey: .ropeParameters)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        numTargetLayers = try c.decode(Int.self, forKey: .numTargetLayers)
        blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: numHiddenLayers)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        isCausal = try c.decodeIfPresent(Bool.self, forKey: .isCausal)
        dflashConfig = try c.decode(NestedDFlashConfig.self, forKey: .dflashConfig)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(architectures, forKey: .architectures)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(numTargetLayers, forKey: .numTargetLayers)
        try c.encode(blockSize, forKey: .blockSize)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encodeIfPresent(slidingWindow, forKey: .slidingWindow)
        try c.encodeIfPresent(finalLogitSoftcapping, forKey: .finalLogitSoftcapping)
        try c.encodeIfPresent(isCausal, forKey: .isCausal)
        try c.encode(dflashConfig, forKey: .dflashConfig)
    }

    public init(
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        intermediateSize: Int,
        vocabSize: Int,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 262_144,
        numTargetLayers: Int,
        blockSize: Int = 16,
        layerTypes: [String],
        slidingWindow: Int? = 2048,
        finalLogitSoftcapping: Float? = 30,
        isCausal: Bool? = nil,
        maskTokenId: Int = 4,
        targetLayerIds: [Int]
    ) {
        self.architectures = ["DFlashDraftModel"]
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numTargetLayers = numTargetLayers
        self.blockSize = blockSize
        self.layerTypes = layerTypes
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.isCausal = isCausal
        self.hasUnsupportedRoPEScaling = false
        self.dflashConfig = NestedDFlashConfig(
            maskTokenId: maskTokenId,
            targetLayerIds: targetLayerIds,
            blockSize: blockSize,
            finalLogitSoftcapping: finalLogitSoftcapping,
            inputEmbeddingScale: nil,
            outputMultiplier: nil)
    }
}

public enum Gemma4DFlashLoadError: Error, LocalizedError {
    case invalidConfiguration(String)
    case missingWeights(String)
    case incompatibleTarget(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let msg): "Gemma4DFlash invalid configuration: \(msg)"
        case .missingWeights(let msg): "Gemma4DFlash missing weights: \(msg)"
        case .incompatibleTarget(let msg): "Gemma4DFlash incompatible target: \(msg)"
        }
    }
}

// MARK: - Chronological Context KV Cache

/// Per-layer chronological KV cache storing verified target context K/V pairs only
/// (`[1, nKvHeads, T, headDim]`). Ephemeral proposal K/V tokens are never written
/// into this cache, so trimming/rollback of rejected drafts is unnecessary.
final class Gemma4DFlashLayerCache {
    /// Zero-copy pre-round rollback snapshot for bounded sliding layers (`maxSize != nil`).
    /// Unbounded full-attention layers (`maxSize == nil`) stage round deltas separately and
    /// return `nil` from `makeRollbackSnapshot()` so `keys`/`values` retain `use_count == 1`.
    struct RollbackSnapshot {
        let keys: MLXArray?
        let values: MLXArray?
        let used: Int
        let offset: Int
    }

    /// Committed backing storage (`[1, nKvHeads, cap, headDim]`).
    /// For unbounded full-attention layers (`maxSize == nil`), grown in `step = 256` token
    /// chunks on `commitStaged()` in `finalizeRound`, keeping `use_count == 1` for in-place
    /// Metal buffer donation while `discardStaged()` rolls back aborted rounds in O(1) without
    /// ever aliasing or mutating committed buffers.
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var used: Int = 0
    var offset: Int = 0
    private(set) var stagedKeys: MLXArray?
    private(set) var stagedValues: MLXArray?
    private(set) var stagedAdded: Int = 0
    let maxSize: Int?
    private static let growthStep = 256

    init(maxSize: Int?) {
        self.maxSize = maxSize
    }

    private func committedSlice(_ backing: MLXArray?) -> MLXArray? {
        guard let backing, used > 0 else { return nil }
        return used == backing.dim(2) ? backing : backing[0..., 0..., 0 ..< used, 0...]
    }

    private func activeTensor(committed: MLXArray?, staged: MLXArray?) -> MLXArray? {
        let base = committedSlice(committed)
        if let base, let staged { return concatenated([base, staged], axis: 2) }
        return staged ?? base
    }

    var activeKeys: MLXArray? { activeTensor(committed: keys, staged: stagedKeys) }
    var activeValues: MLXArray? { activeTensor(committed: values, staged: stagedValues) }

    func appendContext(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        let added = newK.dim(2)
        if let maxSize {
            var k: MLXArray
            var v: MLXArray
            if let existingK = keys, let existingV = values {
                k = concatenated([existingK, newK], axis: 2)
                v = concatenated([existingV, newV], axis: 2)
            } else {
                k = newK
                v = newV
            }
            if k.dim(2) > maxSize {
                let start = k.dim(2) - maxSize
                k = k[0..., 0..., start..., 0...]
                v = v[0..., 0..., start..., 0...]
            }
            self.keys = k
            self.values = v
            self.used = k.dim(2)
            self.offset += added
            return (k, v)
        }

        // Unbounded full-attention layer: stage the active round's delta separately so
        // committed `keys`/`values` are never mutated prior to `finalizeRound`.
        let sk: MLXArray
        let sv: MLXArray
        if let existingSK = stagedKeys, let existingSV = stagedValues {
            sk = concatenated([existingSK, newK], axis: 2)
            sv = concatenated([existingSV, newV], axis: 2)
        } else {
            sk = newK
            sv = newV
        }
        stagedKeys = sk
        stagedValues = sv
        stagedAdded += added
        offset += added
        let outK = committedSlice(keys).map { concatenated([$0, sk], axis: 2) } ?? sk
        let outV = committedSlice(values).map { concatenated([$0, sv], axis: 2) } ?? sv
        return (outK, outV)
    }

    /// Commit staged full-attention KV into the stepped backing buffer after a round finalizes.
    func commitStaged() {
        guard maxSize == nil, let sk = stagedKeys, let sv = stagedValues else { return }
        let added = sk.dim(2)
        guard added > 0 else {
            clearStaged()
            return
        }
        let prev = used
        let next = prev + added
        if next > (keys?.dim(2) ?? 0) {
            let b = sk.dim(0)
            let kvH = sk.dim(1)
            let d = sk.dim(3)
            let step = Self.growthStep
            let newCap = ((next + step - 1) / step) * step
            let newBufK = MLXArray.zeros([b, kvH, newCap, d], dtype: sk.dtype)
            let newBufV = MLXArray.zeros([b, kvH, newCap, d], dtype: sv.dtype)
            if let existingK = keys, let existingV = values, prev > 0 {
                newBufK[0..., 0..., 0 ..< prev, 0...] = existingK[0..., 0..., 0 ..< prev, 0...]
                newBufV[0..., 0..., 0 ..< prev, 0...] = existingV[0..., 0..., 0 ..< prev, 0...]
            }
            keys = newBufK
            values = newBufV
        }
        keys?[0..., 0..., prev ..< next, 0...] = sk
        values?[0..., 0..., prev ..< next, 0...] = sv
        used = next
        clearStaged()
    }

    /// Drop staged full-attention KV on `discardRound` without touching committed storage.
    func discardStaged() {
        guard maxSize == nil, stagedAdded > 0 || stagedKeys != nil else { return }
        offset -= stagedAdded
        clearStaged()
    }

    func makeRollbackSnapshot() -> RollbackSnapshot? {
        guard maxSize != nil else { return nil }
        return RollbackSnapshot(keys: keys, values: values, used: used, offset: offset)
    }

    func rollback(to snapshot: RollbackSnapshot?) {
        if let snapshot {
            keys = snapshot.keys
            values = snapshot.values
            used = snapshot.used
            offset = snapshot.offset
        } else {
            discardStaged()
        }
    }

    func unsharedBytes(in snapshot: RollbackSnapshot?) -> Int {
        guard let snapshot else { return 0 }
        var bytes = 0
        if let k = snapshot.keys, k !== keys { bytes += k.nbytes }
        if let v = snapshot.values, v !== values { bytes += v.nbytes }
        return bytes
    }

    func appendEvalTargets(to targets: inout [MLXArray]) {
        if let keys { targets.append(keys) }
        if let values { targets.append(values) }
        if let stagedKeys { targets.append(stagedKeys) }
        if let stagedValues { targets.append(stagedValues) }
    }

    func reset() {
        keys = nil
        values = nil
        used = 0
        offset = 0
        clearStaged()
    }

    var materializedBytes: Int {
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
            + (stagedKeys?.nbytes ?? 0) + (stagedValues?.nbytes ?? 0)
    }

    private func clearStaged() {
        stagedKeys = nil
        stagedValues = nil
        stagedAdded = 0
    }
}

// MARK: - DFlash Layers (Matches z-lab/dflash/dflash/model_mlx.py)

/// Non-`Module` holder for fused projections so `Module.buildCaches()` Mirror reflection
/// never exposes fused caches as child modules or parameters.
private final class DFlashFusedLinearSlot {
    private let watchedKeys: [String]
    private(set) var primary: Linear?
    private(set) var secondary: Linear?
    private var isPreparing: Bool = false

    init(watchedKeys: [String]) {
        self.watchedKeys = watchedKeys
    }

    func invalidate() {
        guard !isPreparing else { return }
        primary = nil
        secondary = nil
    }

    func invalidateIfNeeded(moduleKey key: String) {
        if watchedKeys.contains(key) {
            invalidate()
        }
    }

    func invalidateIfNeeded(parameterKeys keys: [String]) {
        if keys.contains(where: matchesWatchedKey) {
            invalidate()
        }
    }

    private func matchesWatchedKey(_ key: String) -> Bool {
        let head = key.firstIndex(of: ".").map { key[..<$0] } ?? key[...]
        return watchedKeys.contains { $0 == head }
    }

    /// Coalesce `projections` into a single backing `Linear`/`QuantizedLinear` and replace
    /// the named child modules on `owner` with zero-copy row slices (0 extra RAM).
    func prepare(
        on owner: Module,
        projections: [(key: String, linear: Linear, expectedRows: Int)],
        secondaryRange: Range<Int>? = nil
    ) {
        guard primary == nil else { return }
        guard
            projections.allSatisfy({
                $0.linear.weight.ndim == 2 && $0.linear.weight.dim(0) == $0.expectedRows
            }),
            let fused = Self.makeFusedLinear(projections.map(\.linear))
        else { return }

        let trainingFlags = projections.map(\.linear.training)
        let secondarySlice = secondaryRange.map {
            Self.sliceLinear(fused, range: $0, training: false)
        }
        if let secondarySlice { eval(secondarySlice) }

        var childModules: [String: Linear] = Dictionary(minimumCapacity: projections.count)
        var rowOffset = 0
        for (projection, training) in zip(projections, trainingFlags) {
            let nextOffset = rowOffset + projection.expectedRows
            childModules[projection.key] = Self.sliceLinear(
                fused, range: rowOffset ..< nextOffset, training: training)
            rowOffset = nextOffset
        }

        isPreparing = true
        defer { isPreparing = false }
        owner.update(modules: ModuleChildren.unflattened(childModules))
        primary = fused
        secondary = secondarySlice
    }

    private static func makeFusedLinear(_ linears: [Linear]) -> Linear? {
        guard let first = linears.first, linears.count >= 2 else { return nil }
        guard linears.allSatisfy({ $0.bias == nil && $0.weight.ndim == 2 }) else { return nil }
        if type(of: first) == QuantizedLinear.self, let q0 = first as? QuantizedLinear {
            return fuseQuantized(linears, first: q0)
        }
        if type(of: first) == Linear.self {
            return fuseDense(linears, first: first)
        }
        return nil
    }

    private static func fuseQuantized(
        _ linears: [Linear], first q0: QuantizedLinear
    ) -> QuantizedLinear? {
        let qLinears = linears.compactMap { $0 as? QuantizedLinear }
        guard qLinears.count == linears.count,
            q0.scales.ndim == 2,
            qLinears.allSatisfy({
                type(of: $0) == QuantizedLinear.self
                    && $0.globalScale == nil
                    && $0.groupSize == q0.groupSize && $0.bits == q0.bits && $0.mode == q0.mode
                    && $0.weight.dim(1) == q0.weight.dim(1)
                    && $0.weight.dtype == q0.weight.dtype
                    && $0.scales.ndim == 2
                    && $0.scales.dim(0) == $0.weight.dim(0)
                    && $0.scales.dim(1) == q0.scales.dim(1)
                    && $0.scales.dtype == q0.scales.dtype
                    && ($0.biases == nil) == (q0.biases == nil)
            })
        else { return nil }
        if let b0 = q0.biases {
            guard b0.ndim == 2,
                qLinears.allSatisfy({
                    guard let b = $0.biases else { return false }
                    return b.ndim == 2 && b.dim(0) == $0.weight.dim(0)
                        && b.dim(1) == b0.dim(1) && b.dtype == b0.dtype
                })
            else { return nil }
        }
        let w = concatenated(qLinears.map(\.weight), axis: 0)
        let s = concatenated(qLinears.map(\.scales), axis: 0)
        let b: MLXArray? =
            q0.biases != nil ? concatenated(qLinears.compactMap(\.biases), axis: 0) : nil
        if let b { eval(w, s, b) } else { eval(w, s) }
        let fused = QuantizedLinear(
            weight: w, bias: nil, scales: s, biases: b,
            groupSize: q0.groupSize, bits: q0.bits, mode: q0.mode)
        fused.freeze()
        fused.train(false)
        return fused
    }

    private static func fuseDense(_ linears: [Linear], first: Linear) -> Linear? {
        guard
            linears.allSatisfy({
                type(of: $0) == Linear.self
                    && $0.weight.dim(1) == first.weight.dim(1)
                    && $0.weight.dtype == first.weight.dtype
            })
        else { return nil }
        let w = concatenated(linears.map(\.weight), axis: 0)
        eval(w)
        let fused = Linear(weight: w, bias: nil)
        fused.train(false)
        return fused
    }

    private static func sliceLinear(
        _ fused: Linear, range: Range<Int>, training: Bool
    ) -> Linear {
        if let q = fused as? QuantizedLinear {
            let view = QuantizedLinear(
                weight: q.weight[range, 0...],
                bias: nil,
                scales: q.scales[range, 0...],
                biases: q.biases?[range, 0...],
                groupSize: q.groupSize,
                bits: q.bits,
                mode: q.mode)
            view.freeze()
            view.train(training)
            return view
        }
        let view = Linear(weight: fused.weight[range, 0...], bias: nil)
        view.train(training)
        return view
    }
}

final class Gemma4DFlashAttention: Module, ParameterUpdateObserving {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int?
    let isCausal: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let fusedSlot = DFlashFusedLinearSlot(watchedKeys: ["q_proj", "k_proj", "v_proj"])

    init(_ config: Gemma4DFlashConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        self.slidingWindow = isSliding ? config.slidingWindow : nil
        self.isCausal = config.isCausal ?? isSliding
        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let keys = parameters.flattened().map(\.0)
        defer { fusedSlot.invalidateIfNeeded(parameterKeys: keys) }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer { fusedSlot.invalidateIfNeeded(moduleKey: key) }
        try super.updateModule(key: key, value)
    }

    func ancestorDidUpdateParameters(keys: [String]) {
        fusedSlot.invalidateIfNeeded(parameterKeys: keys)
    }

    func prepareFusedQKV() {
        let qDim = nHeads * headDim
        let kvDim = nKvHeads * headDim
        fusedSlot.prepare(
            on: self,
            projections: [
                ("q_proj", qProj, qDim),
                ("k_proj", kProj, kvDim),
                ("v_proj", vProj, kvDim),
            ],
            secondaryRange: qDim ..< (qDim + 2 * kvDim))
    }

    private func projectProposalQKV(_ x: MLXArray) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let qDim = nHeads * headDim
        let kvDim = nKvHeads * headDim
        if let fusedQKV = fusedSlot.primary {
            let qkv = fusedQKV(x)
            return (
                qkv[0..., 0..., 0 ..< qDim],
                qkv[0..., 0..., qDim ..< (qDim + kvDim)],
                qkv[0..., 0..., (qDim + kvDim)...]
            )
        }
        return (qProj(x), kProj(x), vProj(x))
    }

    private func projectContextKV(_ ctx: MLXArray) -> (k: MLXArray, v: MLXArray) {
        let kvDim = nKvHeads * headDim
        if let fusedKV = fusedSlot.secondary {
            let kv = fusedKV(ctx)
            return (kv[0..., 0..., 0 ..< kvDim], kv[0..., 0..., kvDim...])
        }
        return (kProj(ctx), vProj(ctx))
    }

    private func trimSlidingContext(
        _ context: MLXArray, cache: Gemma4DFlashLayerCache
    ) -> MLXArray {
        guard isSliding, let window = slidingWindow else { return context }
        let keepCtx = max(1, window - 1)
        let length = context.dim(1)
        guard length > keepCtx else { return context }
        let skip = length - keepCtx
        cache.offset += skip
        return context[0..., skip..., 0...]
    }

    private func makeAttentionMask(
        queryLength: Int, contextLength: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if isSliding, let window = slidingWindow {
            let qPos =
                MLXArray.arange(queryLength, dtype: .int32).reshaped([queryLength, 1])
                + Int32(contextLength)
            let kPos = MLXArray.arange(contextLength + queryLength, dtype: .int32)
                .reshaped([1, contextLength + queryLength])
            let ctxMask = logicalAnd(
                kPos .< Int32(contextLength), (qPos - kPos) .< Int32(window))
            let blockBase = kPos .>= Int32(contextLength)
            let blockMask = isCausal ? logicalAnd(blockBase, kPos .<= qPos) : blockBase
            return .array(logicalOr(ctxMask, blockMask))
        }
        if isCausal {
            return .array(createCausalMask(n: queryLength, offset: contextLength))
        }
        return .none
    }

    func callAsFunction(
        _ x: MLXArray,
        context xCtx: MLXArray?,
        rope: RoPE,
        cache: Gemma4DFlashLayerCache
    ) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)

        let cachedK: MLXArray?
        let cachedV: MLXArray?
        if let rawCtx = xCtx, rawCtx.dim(1) > 0 {
            let ctx = trimSlidingContext(rawCtx, cache: cache)
            let s = ctx.dim(1)
            let (rawCtxK, rawCtxV) = projectContextKV(ctx)
            var ctxK = rawCtxK.reshaped(b, s, nKvHeads, headDim)
            ctxK = kNorm(ctxK).transposed(0, 2, 1, 3)
            ctxK = rope(ctxK, offset: cache.offset)
            let ctxV = rawCtxV.reshaped(b, s, nKvHeads, headDim).transposed(0, 2, 1, 3)
            (cachedK, cachedV) = cache.appendContext(keys: ctxK, values: ctxV)
        } else {
            cachedK = cache.activeKeys
            cachedV = cache.activeValues
        }

        let qOffset = cache.offset
        let (rawQ, rawPK, rawPV) = projectProposalQKV(x)
        var queries = rawQ.reshaped(b, l, nHeads, headDim)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        queries = rope(queries, offset: qOffset)

        var propK = rawPK.reshaped(b, l, nKvHeads, headDim)
        propK = kNorm(propK).transposed(0, 2, 1, 3)
        propK = rope(propK, offset: qOffset)
        let propV = rawPV.reshaped(b, l, nKvHeads, headDim).transposed(0, 2, 1, 3)

        let ctxLen = cachedK?.dim(2) ?? 0
        let keys = cachedK.map { concatenated([$0, propK], axis: 2) } ?? propK
        let values = cachedV.map { concatenated([$0, propV], axis: 2) } ?? propV
        let attn = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: makeAttentionMask(queryLength: l, contextLength: ctxLen))
        let flat = attn.transposed(0, 2, 1, 3).reshaped(b, l, nHeads * headDim)
        return oProj(flat)
    }

    /// Batched forward over `B` concurrent sequences sharing one `q_proj/k_proj/v_proj/o_proj`
    /// weight read while attending each sequence's independent chronological context cache.
    func callBatched(
        _ x: MLXArray,
        contexts: [MLXArray?],
        rope: RoPE,
        caches: [Gemma4DFlashLayerCache]
    ) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)
        if b == 1 {
            return self(x, context: contexts[0], rope: rope, cache: caches[0])
        }

        // Batched fused Q/K/V projection for all B proposal blocks (1 weight read).
        let (rawQ, rawPK, rawPV) = projectProposalQKV(x)
        let qAll = qNorm(rawQ.reshaped(b, l, nHeads, headDim)).transposed(0, 2, 1, 3)
        let pKAll = kNorm(rawPK.reshaped(b, l, nKvHeads, headDim)).transposed(0, 2, 1, 3)
        let pVAll = rawPV.reshaped(b, l, nKvHeads, headDim).transposed(0, 2, 1, 3)

        // Trim over-window pending contexts and concatenate them for 1 batched ctx K/V projection.
        var nonEmptyCtx: [MLXArray] = []
        var ctxSlices: [(row: Int, start: Int, len: Int)] = []
        var totalCtxLen = 0
        for i in 0 ..< b {
            guard let rawCtx = contexts[i], rawCtx.dim(1) > 0 else { continue }
            let ctx = trimSlidingContext(rawCtx, cache: caches[i])
            let s = ctx.dim(1)
            nonEmptyCtx.append(ctx)
            ctxSlices.append((row: i, start: totalCtxLen, len: s))
            totalCtxLen += s
        }

        var newCtxKByRow: [MLXArray?] = Array(repeating: nil, count: b)
        var newCtxVByRow: [MLXArray?] = Array(repeating: nil, count: b)
        if totalCtxLen > 0 {
            let joinedCtx =
                nonEmptyCtx.count == 1 ? nonEmptyCtx[0] : concatenated(nonEmptyCtx, axis: 1)
            let (rawCtxK, rawCtxV) = projectContextKV(joinedCtx)
            let allCtxK = kNorm(rawCtxK.reshaped(1, totalCtxLen, nKvHeads, headDim))
                .transposed(0, 2, 1, 3)
            let allCtxV = rawCtxV.reshaped(1, totalCtxLen, nKvHeads, headDim)
                .transposed(0, 2, 1, 3)
            for slice in ctxSlices {
                let lo = slice.start
                let hi = lo + slice.len
                newCtxKByRow[slice.row] = allCtxK[0..., 0..., lo ..< hi, 0...]
                newCtxVByRow[slice.row] = allCtxV[0..., 0..., lo ..< hi, 0...]
            }
        }

        var attnRows: [MLXArray] = []
        attnRows.reserveCapacity(b)
        for i in 0 ..< b {
            let cache = caches[i]
            let cachedK: MLXArray?
            let cachedV: MLXArray?
            if let rawK = newCtxKByRow[i], let rawV = newCtxVByRow[i] {
                let rotK = rope(rawK, offset: cache.offset)
                (cachedK, cachedV) = cache.appendContext(keys: rotK, values: rawV)
            } else {
                cachedK = cache.activeKeys
                cachedV = cache.activeValues
            }
            let qOffset = cache.offset
            let qi = rope(qAll[i ..< (i + 1), 0..., 0..., 0...], offset: qOffset)
            let pki = rope(pKAll[i ..< (i + 1), 0..., 0..., 0...], offset: qOffset)
            let pvi = pVAll[i ..< (i + 1), 0..., 0..., 0...]
            let ctxLen = cachedK?.dim(2) ?? 0
            let keys = cachedK.map { concatenated([$0, pki], axis: 2) } ?? pki
            let values = cachedV.map { concatenated([$0, pvi], axis: 2) } ?? pvi
            let rowAttn = MLXFast.scaledDotProductAttention(
                queries: qi, keys: keys, values: values, scale: scale,
                mask: makeAttentionMask(queryLength: l, contextLength: ctxLen))
            attnRows.append(rowAttn)
        }
        let attn = concatenated(attnRows, axis: 0)
        let flat = attn.transposed(0, 2, 1, 3).reshaped(b, l, nHeads * headDim)
        return oProj(flat)
    }
}

final class Gemma4DFlashMLP: Module, ParameterUpdateObserving {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    private let intermediateSize: Int
    private let fusedSlot = DFlashFusedLinearSlot(watchedKeys: ["gate_proj", "up_proj"])

    init(_ config: Gemma4DFlashConfiguration) {
        self.intermediateSize = config.intermediateSize
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let keys = parameters.flattened().map(\.0)
        defer { fusedSlot.invalidateIfNeeded(parameterKeys: keys) }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer { fusedSlot.invalidateIfNeeded(moduleKey: key) }
        try super.updateModule(key: key, value)
    }

    func ancestorDidUpdateParameters(keys: [String]) {
        fusedSlot.invalidateIfNeeded(parameterKeys: keys)
    }

    func prepareFusedGateUp() {
        fusedSlot.prepare(
            on: self,
            projections: [
                ("gate_proj", gateProj, intermediateSize),
                ("up_proj", upProj, intermediateSize),
            ])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let fusedGateUp = fusedSlot.primary {
            let gu = fusedGateUp(x)
            let gate = gu[0..., 0..., 0 ..< intermediateSize]
            let up = gu[0..., 0..., intermediateSize...]
            return downProj(silu(gate) * up)
        }
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class Gemma4DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: Gemma4DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: Gemma4DFlashConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = Gemma4DFlashAttention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4DFlashMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { notifyParameterUpdateObservers(of: parameters) }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(
        _ x: MLXArray,
        context xCtx: MLXArray?,
        rope: RoPE,
        cache: Gemma4DFlashLayerCache
    ) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), context: xCtx, rope: rope, cache: cache)
        return h + mlp(postAttentionLayernorm(h))
    }

    func callBatched(
        _ x: MLXArray,
        contexts: [MLXArray?],
        rope: RoPE,
        caches: [Gemma4DFlashLayerCache]
    ) -> MLXArray {
        let h =
            x
            + selfAttn.callBatched(
                inputLayernorm(x), contexts: contexts, rope: rope, caches: caches)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - DFlash Draft Model

/// Parallel block-diffusion draft model (`z-lab/gemma-4-26B-A4B-it-DFlash`).
///
/// Consumes 6 tapped target layer hidden states projected through `fc` + `hidden_norm`,
/// maintains 5 tiny cross+self-attention layers (4 sliding window 2048 + 1 full), and
/// drafts an entire block of `k` tokens in one parallel forward pass using the target's
/// shared embeddings and LM head.
public final class Gemma4DFlashDraftModel: Module, @unchecked Sendable {
    public let configuration: Gemma4DFlashConfiguration
    let rope: RoPE

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Gemma4DFlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ config: Gemma4DFlashConfiguration) throws {
        guard config.numHiddenLayers > 0,
            config.hiddenSize > 0,
            config.intermediateSize > 0,
            config.headDim > 0,
            config.vocabSize > 0,
            config.numAttentionHeads > 0,
            config.numKeyValueHeads > 0,
            config.numAttentionHeads % config.numKeyValueHeads == 0,
            config.effectiveBlockSize >= 2,
            config.layerTypes.count == config.numHiddenLayers,
            !config.targetLayerIds.isEmpty,
            Set(config.targetLayerIds).count == config.targetLayerIds.count
        else {
            throw Gemma4DFlashLoadError.invalidConfiguration(
                "invalid geometry or duplicate target_layer_ids: layers=\(config.numHiddenLayers), heads=\(config.numAttentionHeads)/\(config.numKeyValueHeads), block_size=\(config.effectiveBlockSize), target_layer_ids=\(config.targetLayerIds)"
            )
        }
        guard !config.hasUnsupportedRoPEScaling else {
            throw Gemma4DFlashLoadError.invalidConfiguration(
                "non-null rope_scaling / rope_parameters is not supported")
        }
        for kind in config.layerTypes {
            guard kind == "sliding_attention" || kind == "full_attention" else {
                throw Gemma4DFlashLoadError.invalidConfiguration(
                    "unsupported layer_type '\(kind)'")
            }
        }
        if config.layerTypes.contains("sliding_attention") {
            guard let w = config.slidingWindow, w >= 2 else {
                throw Gemma4DFlashLoadError.invalidConfiguration(
                    "sliding_attention layer requires sliding_window >= 2")
            }
        }
        self.configuration = config
        self.rope = RoPE(
            dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        let concatDim = config.targetLayerIds.count * config.hiddenSize
        self._fc.wrappedValue = Linear(concatDim, config.hiddenSize, bias: false)
        self._hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DFlashDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Returns true if `directory/config.json` declares `DFlashDraftModel`.
    public static func isDFlashDirectory(_ directory: URL) -> Bool {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
            let cfg = try? JSONDecoder().decode(Gemma4DFlashConfiguration.self, from: data)
        else { return false }
        return cfg.architectures?.contains("DFlashDraftModel") == true
    }

    /// Project concatenated tapped target hidden states `[B, S, numTapped * D]` into
    /// `[B, S, D]` context features (`hidden_norm(fc(concat))`).
    func projectTargetContext(_ tappedLayers: [MLXArray]) -> MLXArray {
        let joined = concatenated(tappedLayers, axis: -1)
        return hiddenNorm(fc(joined))
    }

    func makeCaches() -> [Gemma4DFlashLayerCache] {
        configuration.layerTypes.map { kind in
            if kind == "sliding_attention", let window = configuration.slidingWindow {
                return Gemma4DFlashLayerCache(maxSize: max(1, window - 1))
            }
            return Gemma4DFlashLayerCache(maxSize: nil)
        }
    }

    /// Run one parallel block-diffusion pass over `[B, 1 + depth]` embedded tokens
    /// attending to verified target context `contextHidden` (`[B, S, D]`, already
    /// projected through `fc` + `hidden_norm`), returning post-norm hidden states
    /// for positions `1 ..< 1 + depth` (`[B, depth, D]`).
    func forwardDraftBlock(
        embeddedBlock: MLXArray,
        projectedContext: MLXArray?,
        caches: [Gemma4DFlashLayerCache]
    ) -> MLXArray {
        var h = embeddedBlock
        for (layer, cache) in zip(layers, caches) {
            h = layer(h, context: projectedContext, rope: rope, cache: cache)
        }
        return norm(h[0..., 1..., 0...])
    }

    /// Run one batched parallel block-diffusion pass over `[B, 1 + depth]` embedded tokens
    /// across `B` concurrent sequences, sharing one weight read for all linear/MLP/LM-head ops.
    func forwardDraftBlockBatch(
        embeddedBlocks: MLXArray,
        projectedContexts: [MLXArray?],
        cachesByRow: [[Gemma4DFlashLayerCache]]
    ) -> MLXArray {
        if embeddedBlocks.dim(0) == 1 {
            return forwardDraftBlock(
                embeddedBlock: embeddedBlocks,
                projectedContext: projectedContexts[0],
                caches: cachesByRow[0])
        }
        var h = embeddedBlocks
        for (layerIdx, layer) in layers.enumerated() {
            let layerCaches = cachesByRow.map { $0[layerIdx] }
            h = layer.callBatched(
                h, contexts: projectedContexts, rope: rope, caches: layerCaches)
        }
        return norm(h[0..., 1..., 0...])
    }

    private static let quantize4BitAtLoad: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_QUANT_4BIT"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
        else { return true }
        return !(raw == "0" || raw == "false" || raw == "off" || raw == "no")
    }()

    /// Load a `DFlashDraftModel` from a local directory containing `config.json` and
    /// `*.safetensors`. By default (`DARKBLOOM_DFLASH_QUANT_4BIT`, on) quantizes BF16
    /// linear weights to 4-bit group-64 affine at load time (~1.56 GB -> ~410 MB) for
    /// faster Apple Silicon drafting with zero impact on exact target verification.
    public static func load(from directory: URL) throws -> Gemma4DFlashDraftModel {
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Gemma4DFlashConfiguration.self, from: configData)
        let model = try Gemma4DFlashDraftModel(config)

        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw Gemma4DFlashLoadError.missingWeights(directory.path)
        }
        var weights: [String: MLXArray] = [:]
        for file in files {
            let shard = try MLX.loadArrays(url: file)
            for (k, v) in shard {
                weights[k] = v
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model)

        if quantize4BitAtLoad {
            quantize(model: model) { _, module in
                guard let linear = module as? Linear,
                    !(module is QuantizedLinear),
                    linear.weight.dim(1) % 64 == 0
                else { return nil }
                return (groupSize: 64, bits: 4, mode: .affine)
            }
            model.train(false)
            eval(model)
            Memory.clearCache()
        }
        model.prepareFusions()
        eval(model)
        Memory.clearCache()
        return model
    }

    /// Zero-extra-RAM load-time fusion of `q_proj/k_proj/v_proj` and `gate_proj/up_proj`
    /// across all draft layers (replaces child projections with zero-copy row slices).
    public func prepareFusions() {
        for layer in layers {
            layer.selfAttn.prepareFusedQKV()
            layer.mlp.prepareFusedGateUp()
        }
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { notifyParameterUpdateObservers(of: parameters) }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}

// MARK: - CBv2 Request-Stateful Parallel Block Drafter

final class Gemma4DFlashRequestState: CBv2MTPRequestState, @unchecked Sendable {
    unowned let owner: Gemma4CBv2DFlashDrafter
    let caches: [Gemma4DFlashLayerCache]
    /// Projected target context vectors `[1, S_chunk, D]` queued since the last draft pass.
    var pendingContext: [MLXArray] = []
    /// True after `finalizeRound`: the final accepted token's context hidden arrives in the
    /// next round's `carry.hidden` (or in the first `observeCommittedTarget` of an ordinary step).
    var needsCarryHidden: Bool = false
    /// Total verified target positions observed (cached + queued).
    var observedCount: Int = 0
    /// Whether a speculative draft round is currently in flight.
    var roundActive: Bool = false
    /// Cached draft token arrays `[depth]` produced by `draftBlock` when called via `draftStep`.
    var bufferedBlockTokens: [MLXArray]?
    var bufferedBlockCursor: Int = 0
    /// Evaluation targets from the most recent draft pass.
    var lastEvalTargets: [MLXArray] = []
    /// Zero-copy pre-round rollback snapshots for bounded sliding layers (`maxSize != nil`).
    /// Unbounded full-attention layers (`maxSize == nil`) stage their active round separately
    /// on `Gemma4DFlashLayerCache` and leave `nil` here so `keys`/`values` retain `use_count == 1`.
    var rollbackCaches: [Gemma4DFlashLayerCache.RollbackSnapshot?]?
    var rollbackPendingContext: [MLXArray] = []
    var rollbackNeedsCarryHidden: Bool = false
    var rollbackObservedCount: Int = 0
    var isReleased: Bool = false

    init(owner: Gemma4CBv2DFlashDrafter, caches: [Gemma4DFlashLayerCache]) {
        self.owner = owner
        self.caches = caches
    }

    var committedInputCount: Int { observedCount }
    var stagedInputCount: Int { roundActive ? (bufferedBlockTokens?.count ?? 0) : 0 }
    var hasPendingPrefillForCostAccounting: Bool {
        pendingContext.reduce(0) { $0 + $1.dim(1) } > 16
    }

    var materializedBytes: Int {
        guard !isReleased else { return 0 }
        let cacheBytes = caches.reduce(0) { $0 + $1.materializedBytes }
        let pendingBytes = pendingContext.reduce(0) { $0 + $1.nbytes }
        let rollbackCacheBytes =
            if let snaps = rollbackCaches, snaps.count == caches.count {
                zip(caches, snaps).reduce(0) { $0 + $1.0.unsharedBytes(in: $1.1) }
            } else {
                0
            }
        let rollbackPendingBytes = rollbackPendingContext.reduce(0) { $0 + $1.nbytes }
        return cacheBytes + pendingBytes + rollbackCacheBytes + rollbackPendingBytes
    }

    func beginDraftRound(seedHidden: MLXArray) -> MLXArray? {
        precondition(!roundActive, "Gemma4DFlash: nested draft round")
        rollbackCaches = caches.map { $0.makeRollbackSnapshot() }
        rollbackPendingContext = pendingContext
        rollbackNeedsCarryHidden = needsCarryHidden
        rollbackObservedCount = observedCount

        if needsCarryHidden {
            pendingContext.append(seedHidden)
            observedCount += seedHidden.dim(1)
            needsCarryHidden = false
        }

        let context: MLXArray? =
            switch pendingContext.count {
            case 0: nil
            case 1: pendingContext[0]
            default: concatenated(pendingContext, axis: 1)
            }
        pendingContext.removeAll(keepingCapacity: true)
        return context
    }

    func recordDraftedBlock(rowTokens: MLXArray, depth: Int) -> [MLXArray] {
        var perPos: [MLXArray] = []
        perPos.reserveCapacity(depth)
        for i in 0 ..< depth {
            perPos.append(rowTokens[i ..< (i + 1)])
        }
        var evalTargets: [MLXArray] = [rowTokens]
        for cache in caches {
            cache.appendEvalTargets(to: &evalTargets)
        }
        lastEvalTargets = evalTargets
        roundActive = true
        bufferedBlockTokens = perPos
        bufferedBlockCursor = depth
        return perPos
    }

    func finishRound() {
        roundActive = false
        bufferedBlockTokens = nil
        bufferedBlockCursor = 0
        lastEvalTargets.removeAll(keepingCapacity: true)
        clearRollbackSnapshot()
    }

    func clearRollbackSnapshot() {
        rollbackCaches = nil
        rollbackPendingContext.removeAll(keepingCapacity: false)
        rollbackNeedsCarryHidden = false
        rollbackObservedCount = 0
    }

    func reset() {
        for c in caches { c.reset() }
        pendingContext.removeAll(keepingCapacity: false)
        needsCarryHidden = false
        observedCount = 0
        finishRound()
        lastEvalTargets.removeAll(keepingCapacity: false)
    }
}

/// CBv2 request-stateful parallel block-diffusion drafter wrapping `Gemma4DFlashDraftModel`.
///
/// Installs a 6-layer target context tap on `Gemma4TextModel`, drafts all `depth`
/// speculative tokens in a single parallel 5-layer pass via `draftBlock` / `draftBlockBatch`,
/// and shares the target's embeddings and LM head (using `Gemma4Hi4Lo4Embedding.sparseExactHeadSample`
/// when active so draft argmax stays exact without materializing `[B, L, vocab]` logits).
public final class Gemma4CBv2DFlashDrafter: CBv2MTPRequestStatefulDrafter, @unchecked Sendable {
    public let drafter: Gemma4DFlashDraftModel
    public weak var target: Gemma4TextModel?
    public var mtpTargetIdentity: ObjectIdentifier? { target.map(ObjectIdentifier.init) }

    public let requiredVerificationMode: CBv2MTPVerificationMode? = nil
    public let prefersBatchedRectangularAttention: Bool = true
    public let supportsCompactedPagedSpeculation: Bool = true
    public let supportsTargetPrefixAcceptance: Bool = true
    public let requiresCommittedObservationFence: Bool = false
    public let supportsOrdinaryDecodeHistory: Bool = true
    public let submitsEveryDraftStepEarly: Bool = false
    public let draftShortlistSize: Int? = nil
    public let maximumSpeculativeBatch: Int?

    public let maximumDraftTokens: Int?
    public let requestStateBytesPerToken: Int
    public var requestStateTokenGranularity: Int { 256 }
    public var requestStateTokenAllocationPadding: Int { 16 }

    public var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        let cfg = drafter.configuration
        let elementBytes = 2  // bfloat16/float16 KV + projected hidden states
        let (kvElements, gOv) = cfg.numKeyValueHeads.multipliedReportingOverflow(by: cfg.headDim)
        let (kvRowBytes, rOv) = kvElements.multipliedReportingOverflow(by: elementBytes)
        // K+V across all layers (including bounded sliding layers) plus transient rollback snapshot.
        let (kvBuffers, bOv) = cfg.numHiddenLayers.multipliedReportingOverflow(by: 4)
        let (hiddenRowBytes, hOv) = cfg.hiddenSize.multipliedReportingOverflow(by: elementBytes)
        guard !gOv, !rOv, !bOv, !hOv, kvRowBytes > 0, kvBuffers > 0, hiddenRowBytes > 0 else {
            return nil
        }
        return [
            .init(
                bytesPerToken: kvRowBytes,
                allocationCount: kvBuffers,
                tokenGranularity: requestStateTokenGranularity,
                tokenPadding: requestStateTokenAllocationPadding),
            // Pending context chunks and rollback copies.
            .init(
                bytesPerToken: hiddenRowBytes,
                allocationCount: 2,
                tokenPadding: requestStateTokenAllocationPadding,
                partitioned: true),
        ]
    }

    public init(
        drafter: Gemma4DFlashDraftModel, target: Gemma4TextModel,
        maxDraftTokens: Int? = nil,
        maxSpeculativeBatch: Int? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        let cfg = drafter.configuration
        let targetCfg = target.model.config
        guard targetCfg.hiddenSize == cfg.hiddenSize else {
            throw Gemma4DFlashLoadError.incompatibleTarget(
                "hidden_size target=\(targetCfg.hiddenSize) draft=\(cfg.hiddenSize)")
        }
        guard targetCfg.vocabSize == cfg.vocabSize else {
            throw Gemma4DFlashLoadError.incompatibleTarget(
                "vocab_size target=\(targetCfg.vocabSize) draft=\(cfg.vocabSize)")
        }
        guard (0 ..< targetCfg.vocabSize).contains(cfg.maskTokenId) else {
            throw Gemma4DFlashLoadError.incompatibleTarget(
                "mask_token_id \(cfg.maskTokenId) out of range 0..<\(targetCfg.vocabSize)")
        }
        guard targetCfg.numHiddenLayers == cfg.numTargetLayers else {
            throw Gemma4DFlashLoadError.incompatibleTarget(
                "num_hidden_layers target=\(targetCfg.numHiddenLayers) draft=\(cfg.numTargetLayers)"
            )
        }
        guard cfg.targetLayerIds.allSatisfy({ (0 ..< targetCfg.numHiddenLayers).contains($0) })
        else {
            throw Gemma4DFlashLoadError.incompatibleTarget(
                "target_layer_ids \(cfg.targetLayerIds) out of range 0..<\(targetCfg.numHiddenLayers)"
            )
        }
        self.drafter = drafter
        self.target = target
        let maxFromConfig = max(
            1, min(cfg.effectiveBlockSize - 1, CBv2MTPConfig.maxBlockDiffusionDraftTokens))
        if let explicit = maxDraftTokens {
            self.maximumDraftTokens = max(1, min(explicit, maxFromConfig))
        } else {
            let recommendedDefault = min(maxFromConfig, 7)
            self.maximumDraftTokens = Self.resolveMaxDraftTokens(
                defaultMax: recommendedDefault, hardCap: maxFromConfig, environment: environment)
        }
        if let explicitBatch = maxSpeculativeBatch {
            self.maximumSpeculativeBatch = max(
                1, min(explicitBatch, CBv2MTPConfig.testedMaxSpeculativeBatch))
        } else {
            // On 128-expert sparse MoE targets (Gemma 4 26B), speculative verification
            // wins strongly at B=1..2 and crosses over to pipelined chained decode at B>=3.
            self.maximumSpeculativeBatch = Self.resolveMaxSpeculativeBatch(
                defaultMax: 2, environment: environment)
        }

        // Conservative KV + projected context accounting consistent with `requestStateAllocationSpecs`.
        let kvBytesPerToken = cfg.numHiddenLayers * 4 * cfg.numKeyValueHeads * cfg.headDim * 2
        let hiddenBytesPerToken = 2 * cfg.hiddenSize * 2
        self.requestStateBytesPerToken = kvBytesPerToken + hiddenBytesPerToken

        target.installDFlashContextTap(owner: ObjectIdentifier(self), layerIDs: cfg.targetLayerIds)
        { [weak drafter] tapped in
            guard let drafter else { return tapped.last! }
            return drafter.projectTargetContext(tapped)
        }
    }

    deinit {
        target?.clearDFlashContextTap(owner: ObjectIdentifier(self))
    }

    public static func resolveMaxSpeculativeBatch(
        defaultMax: Int = 2,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        guard
            let raw = environment["DARKBLOOM_DFLASH_MAX_BATCH"]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            let parsed = Int(raw), parsed >= 1
        else { return defaultMax }
        return min(parsed, CBv2MTPConfig.testedMaxSpeculativeBatch)
    }

    public static func resolveMaxDraftTokens(
        defaultMax: Int,
        hardCap: Int? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        guard
            let raw = environment["DARKBLOOM_DFLASH_MAX_DRAFT_TOKENS"]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            let parsed = Int(raw), parsed >= 1
        else { return defaultMax }
        return min(parsed, hardCap ?? defaultMax)
    }

    private final class UnusedCapture: CBv2MTPPreparedCapture {}
    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { UnusedCapture() }
    public func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        preconditionFailure("Gemma4CBv2DFlashDrafter requires request-owned state")
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        Gemma4DFlashRequestState(owner: self, caches: drafter.makeCaches())
    }

    public func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int
    ) throws {
        _ = checked(requestState)
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        let state = checked(requestState)
        precondition(!state.roundActive, "Gemma4DFlash: observation during active draft round")
        let hidden = observation.hidden
        guard hidden.dim(1) > 0 else { return }
        state.needsCarryHidden = false
        state.pendingContext.append(hidden)
        state.observedCount += hidden.dim(1)
    }

    private func embedProposalBlocks(
        seedTokens: MLXArray, batch: Int, depth: Int, target: Gemma4TextModel
    ) -> MLXArray {
        let maskToken = Int32(drafter.configuration.maskTokenId)
        let masks = MLXArray.full([batch, depth], values: MLXArray(maskToken))
        let blockIDs = concatenated(
            [seedTokens.asType(.int32).reshaped([batch, 1]), masks], axis: 1)
        let embedded = target.embedTokensForDrafter(blockIDs)
        let extraScale = drafter.configuration.inputEmbeddingScale
        return extraScale == 1.0 ? embedded : embedded * extraScale
    }

    /// Parallel block-diffusion draft: produces all `depth` draft token `[1]` int32
    /// arrays in a single 5-layer forward pass over `[seedToken, mask, ..., mask]`.
    public func draftBlock(
        seedTokens: MLXArray,
        seedHidden: MLXArray,
        depth: Int,
        requestState: any CBv2MTPRequestState
    ) -> [MLXArray]? {
        guard depth >= 1, let target else { return nil }
        let state = checked(requestState)
        let context = state.beginDraftRound(seedHidden: seedHidden)
        let embedded = embedProposalBlocks(
            seedTokens: seedTokens, batch: 1, depth: depth, target: target)
        let draftHidden = drafter.forwardDraftBlock(
            embeddedBlock: embedded,
            projectedContext: context,
            caches: state.caches)
        let draftTokens = target.draftArgMax(draftHidden).reshaped([depth])
        return state.recordDraftedBlock(rowTokens: draftTokens, depth: depth)
    }

    /// Batched parallel block-diffusion draft across `B` concurrent rows in one
    /// forward pass through all 5 draft layers + 1 batched LM-head readout.
    public func draftBlockBatch(
        seedTokens: MLXArray,
        seedHiddens: [MLXArray],
        depth: Int,
        requestStates: [any CBv2MTPRequestState]
    ) -> [[MLXArray]]? {
        let batch = requestStates.count
        guard depth >= 1, batch >= 1, seedHiddens.count == batch, let target else { return nil }
        if batch == 1 {
            guard
                let single = draftBlock(
                    seedTokens: seedTokens,
                    seedHidden: seedHiddens[0],
                    depth: depth,
                    requestState: requestStates[0])
            else { return nil }
            return [single]
        }

        let states = requestStates.map(checked)
        let contexts = zip(states, seedHiddens).map { state, seedHidden in
            state.beginDraftRound(seedHidden: seedHidden)
        }
        let embedded = embedProposalBlocks(
            seedTokens: seedTokens, batch: batch, depth: depth, target: target)
        let draftHidden = drafter.forwardDraftBlockBatch(
            embeddedBlocks: embedded,
            projectedContexts: contexts,
            cachesByRow: states.map(\.caches))
        let draftTokens = target.draftArgMax(draftHidden).reshaped([batch, depth])
        return states.enumerated().map { b, state in
            state.recordDraftedBlock(rowTokens: draftTokens[b], depth: depth)
        }
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let state = checked(requestState)
        if !state.roundActive {
            let depth = maximumDraftTokens ?? CBv2MTPConfig.testedMaxDraftTokens
            guard
                let block = draftBlock(
                    seedTokens: tokens, seedHidden: hidden, depth: depth, requestState: state),
                let first = block.first
            else {
                preconditionFailure(
                    "Gemma4DFlash: draftStep failed (target deallocated or empty block)")
            }
            state.bufferedBlockCursor = 1
            return (first, hidden)
        }
        let cursor = state.bufferedBlockCursor
        if let block = state.bufferedBlockTokens, cursor < block.count {
            state.bufferedBlockCursor = cursor + 1
            return (block[cursor], hidden)
        }
        preconditionFailure("Gemma4DFlash: draftStep exceeded buffered block depth")
    }

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        guard let state = requestState as? Gemma4DFlashRequestState,
            state.owner === self, !state.isReleased
        else { return [] }
        return state.lastEvalTargets
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        let state = checked(requestState)
        precondition(state.roundActive, "Gemma4DFlash: finalizeRound without active round")
        let acceptedDraftCount = committedTargetHidden.dim(1)
        if acceptedDraftCount > 0 {
            state.pendingContext.append(committedTargetHidden)
            state.observedCount += acceptedDraftCount
        }
        // The final accepted position (`confirmedInputTokens - 1`) arrives in the next
        // round's `carry.hidden` (or in the first `observeCommittedTarget` on ordinary decode).
        state.needsCarryHidden = true
        state.finishRound()
        // Commit staged full-attention KV after clearing rollback state so backing storage
        // retains `use_count == 1` for in-place Metal buffer donation.
        for cache in state.caches {
            cache.commitStaged()
        }
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        let state = checked(requestState)
        if let savedCaches = state.rollbackCaches, savedCaches.count == state.caches.count {
            for (cache, snap) in zip(state.caches, savedCaches) {
                cache.rollback(to: snap)
            }
            state.pendingContext = state.rollbackPendingContext
            state.needsCarryHidden = state.rollbackNeedsCarryHidden
            state.observedCount = state.rollbackObservedCount
        }
        state.finishRound()
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? Gemma4DFlashRequestState, state.owner === self else {
            preconditionFailure("Gemma4DFlash: foreign request state")
        }
        guard !state.isReleased else { return }
        state.reset()
        state.isReleased = true
    }

    private func checked(_ requestState: any CBv2MTPRequestState) -> Gemma4DFlashRequestState {
        guard let state = requestState as? Gemma4DFlashRequestState, state.owner === self else {
            preconditionFailure("Gemma4DFlash: foreign request state")
        }
        precondition(!state.isReleased, "Gemma4DFlash: use of released request state")
        return state
    }
}
