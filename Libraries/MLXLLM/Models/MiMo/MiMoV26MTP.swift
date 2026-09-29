// Copyright © 2026 Eigen Labs.
// Native next-N predictors, not the separately shipped DFlash draft model.
// Reference: sglang/mimo_v2_nextn.py @67bb6a58d0dad4a39af80fa1b2bf86f0de0cb99b.
// That file is byte-identical to the prior @2261c2e reference (see MTP-RESULT).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum MiMoV26MTPError: Error, Equatable, Sendable {
    case unsupportedConfiguration(String)
    case incompatibleOwner
    case weightsNotLoaded
    case invalidWeights(String)
    case invalidInput(String)
    case invalidHistory(String)
    case contextExceeded
}

/// Bit-preserving device copy. The request fence must verify its fresh compact
/// backing before granting materialized credit; construction alone proves none.
func mimoV26MTPCopy(_ array: MLXArray) -> MLXArray {
    // A floating select may normalize BF16 subnormals or NaN payload bits.
    // Select an equal-width integer view instead: no floating conversion or
    // arithmetic, and even scalar/noncontiguous inputs keep their geometry.
    let bits: MLXArray
    switch array.dtype {
    case .float16, .bfloat16: bits = array.view(dtype: .uint16)
    case .float32: bits = array.view(dtype: .uint32)
    case .float64, .complex64: bits = array.view(dtype: .uint64)
    default: bits = array
    }
    return MLX.where(MLXArray(true), bits, bits).view(dtype: array.dtype)
}

/// A non-Module box deliberately hides the weak target from Module reflection.
/// The live reference, not a potentially reused ObjectIdentifier, proves ownership.
private final class MiMoV26MTPTargetOwner {
    weak var target: MiMoV26TextModel?
    init(_ target: MiMoV26TextModel) { self.target = target }
}

/// Cache-local history starts at zero; RoPE uses the absolute next-token offset.
/// Only the sealed request owner can obtain this protocol view.
final class MiMoV26MTPPositionedCache: KVCache {
    let storage: RotatingKVCache
    let firstPosition: Int
    init(window: Int, firstPosition: Int) {
        storage = RotatingKVCache(maxSize: window)
        self.firstPosition = firstPosition
    }
    var offset: Int { firstPosition + storage.offset }
    var maxSize: Int? { storage.maxSize }
    var state: [MLXArray] {
        get { storage.state }
        set { storage.state = newValue }
    }
    var metaState: [String] {
        get { storage.metaState }
        set { storage.metaState = newValue }
    }
    var isTrimmable: Bool { false }
    func trim(_ n: Int) -> Int { 0 }
    func innerState() -> [MLXArray] { storage.innerState() }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        storage.update(keys: keys, values: values)
    }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        storage.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    func copy() -> any KVCache {
        let result = MiMoV26MTPPositionedCache(
            window: storage.maxSize!, firstPosition: firstPosition)
        let snapshot = storage.copy()
        let arrays = snapshot.state
        // RotatingKVCache's setter requires exactly two arrays, even when the
        // source is fresh. Preserve its empty initial state without that setter.
        if !arrays.isEmpty { result.storage.state = arrays }
        result.storage.metaState = snapshot.metaState
        return result
    }

    // Multi-token rotating updates temporarily keep window+chunk-1 entries
    // for attention. Once that forward has captured its reads, keep a detached
    // chronological suffix; metadata preserves the absolute rotary frontier.
    func compactRetainedHistory() {
        let arrays = storage.state
        guard arrays.count == 2 else { return }
        let window = storage.maxSize!
        if arrays[0].dim(2) > window {
            storage.state = arrays.map { mimoV26MTPCopy($0[0..., 0..., (-window)..., 0...]) }
            var metadata = storage.metaState
            metadata[4] = String(window)
            storage.metaState = metadata
        } else {
            // A first multi-token update can retain transposed/shared projection
            // storage even when it fits the window. Detach the live bytes before
            // the unchanged request fence proves compact independent ownership.
            // Full rings retain their PHYSICAL order and existing ring index;
            // partial history retains its exact offset/first-position metadata.
            storage.state = arrays.map(mimoV26MTPCopy)
        }
    }

    func detachedCopy() -> MiMoV26MTPPositionedCache {
        let result = copy() as! MiMoV26MTPPositionedCache
        if !result.state.isEmpty { result.state = result.state.map(mimoV26MTPCopy) }
        return result
    }
}

/// One request owns all predictor histories; no externally mutable KV cache is
/// accepted. The assistant owns transactional copies for speculation; this
/// component remains append-only. Do not share a live cache across requests.
public final class MiMoV26MTPRequestCache: Evaluatable {
    fileprivate weak var owner: MiMoV26MTP?
    fileprivate let generation: UInt64
    fileprivate let layers: [MiMoV26MTPPositionedCache]
    fileprivate var batchSize: Int?

    fileprivate init(
        owner: MiMoV26MTP, generation: UInt64, window: Int,
        count: Int, basePosition: Int
    ) {
        self.owner = owner
        self.generation = generation
        layers = (0 ..< count).map {
            MiMoV26MTPPositionedCache(window: window, firstPosition: basePosition + $0 + 1)
        }
    }
    public var nextTokenPositions: [Int] { layers.map(\.offset) }
    public var consumedTokenCounts: [Int] { layers.map { $0.storage.offset } }
    public var stateShapesByDepth: [[[Int]]] { layers.map { $0.innerState().map(\.shape) } }
    /// Scalar-only observation for admitted-engine state tests. Each row is
    /// firstPosition, keep, window, step, localOffset, physicalRingIndex.
    /// No array/view, native read, evaluation or ownership transfer is created.
    var retainedHistoryMetadataForTesting: [[String]] {
        layers.map { [String($0.firstPosition)] + $0.metaState }
    }
    public func innerState() -> [MLXArray] { layers.flatMap { $0.innerState() } }

    private init(copying source: MiMoV26MTPRequestCache) {
        owner = source.owner
        generation = source.generation
        layers = source.layers.map { $0.detachedCopy() }
        batchSize = source.batchSize
    }

    func detachedCopy() -> MiMoV26MTPRequestCache { .init(copying: self) }
    func compactRetainedHistory(depth: Int) { layers[depth].compactRetainedHistory() }

    // Validate raw storage and metadata BEFORE RotatingKVCache.state can slice.
    // Full rings stay in their original physical order with their exact idx.
    func prefixHeads(count: Int, expectedOwner: MiMoV26MTP) -> [MiMoV26MTPPrefixHead]? {
        guard owner === expectedOwner, expectedOwner.isLoaded,
            generation == expectedOwner.loadedGeneration, batchSize == 1,
            layers.count == 3, count >= 4,
            count < expectedOwner.configuration.maxPositionEmbeddings
        else { return nil }
        let c = expectedOwner.configuration
        var result: [MiMoV26MTPPrefixHead] = []
        for (depth, layer) in layers.enumerated() {
            let raw = layer.innerState()
            let strings = layer.metaState
            guard raw.count == 2, strings.count == 5 else { return nil }
            let scalars = strings.compactMap(Int64.init)
            guard scalars.count == 5 else { return nil }
            let metadata =
                [Int64(layer.firstPosition)] + scalars
                + [Int64(min(max(0, count - depth - 1), c.slidingWindow))]
            let rawHead = MiMoV26MTPPrefixHead(keys: raw[0], values: raw[1], metadata: metadata)
            guard
                MiMoV26MTPPrefixValidation.headIsValid(
                    rawHead, depth: depth, count: count,
                    configuration: c, allowsUnusedCapacity: true)
            else { return nil }
            let live = layer.state
            let head = MiMoV26MTPPrefixHead(keys: live[0], values: live[1], metadata: metadata)
            guard
                MiMoV26MTPPrefixValidation.headIsValid(
                    head, depth: depth, count: count,
                    configuration: c)
            else { return nil }
            result.append(head)
        }
        return result
    }

    // A fresh, unobserved cache only. Preflight ALL heads before any mutation;
    // this never writes a live request or donor. The caller owns the admitted
    // off-to-the-side state until atomic target+assistant adoption completes.
    func restorePrefixHeads(
        _ heads: [MiMoV26MTPPrefixHead], count: Int,
        expectedOwner: MiMoV26MTP
    ) -> Bool {
        guard owner === expectedOwner, generation == expectedOwner.loadedGeneration,
            expectedOwner.isLoaded, batchSize == nil, layers.count == 3,
            layers.allSatisfy({ $0.innerState().isEmpty && $0.storage.offset == 0 }),
            heads.count == 3,
            heads.enumerated().allSatisfy({
                MiMoV26MTPPrefixValidation.headIsValid(
                    $0.element, depth: $0.offset,
                    count: count, configuration: expectedOwner.configuration)
            })
        else { return false }
        for (depth, head) in heads.enumerated() {
            layers[depth].state = [mimoV26MTPCopy(head.keys), mimoV26MTPCopy(head.values)]
            layers[depth].metaState = head.metadata[1 ... 5].map(String.init)
        }
        batchSize = 1
        return true
    }
}

public enum MiMoV26MTPFeatureProvenance: Equatable, Sendable {
    case targetPostNorm
    case predictorPostNorm(depth: Int)
}

/// Features describe the positions producing hidden states. The pinned MiMo
/// driver reuses the same post-final-norm TARGET feature at every predictor
/// depth; token positions advance by depth+1. Predictor readout features are
/// exposed for diagnostics, never accepted as the next predictor's input.
public struct MiMoV26MTPFeatures {
    public let normalizedHiddenStates: MLXArray
    public let firstPosition: Int
    public let provenance: MiMoV26MTPFeatureProvenance
    fileprivate let targetOwner: MiMoV26MTPTargetOwner
    fileprivate weak var predictor: MiMoV26MTP?
    fileprivate let generation: UInt64?

    /// Ownership and positions come from the target's actual forward output.
    /// The caller may select rows, but cannot relabel their producer or position.
    /// All validation precedes array indexing or creation of a sliced feature.
    public init(
        targetOutput: MiMoV26TextOutput, target: MiMoV26TextModel,
        range: Range<Int>? = nil
    ) throws {
        guard targetOutput.ownerIdentity == target.identity else {
            throw MiMoV26MTPError.incompatibleOwner
        }
        let hidden = targetOutput.normalizedHiddenStates
        guard hidden.ndim == 3, hidden.dim(0) > 0, hidden.dim(1) > 0,
            hidden.dim(2) == target.configuration.hiddenSize,
            hidden.dtype == target.activationDType
        else {
            throw MiMoV26MTPError.invalidInput("target normalized-feature geometry or dtype")
        }
        let selected = range ?? (0 ..< hidden.dim(1))
        guard selected.lowerBound >= 0, selected.upperBound > selected.lowerBound,
            selected.upperBound <= hidden.dim(1)
        else {
            throw MiMoV26MTPError.invalidInput(
                "target feature range must be nonempty and in bounds")
        }
        let limit = target.configuration.maxPositionEmbeddings
        let position = targetOutput.firstPosition.addingReportingOverflow(selected.lowerBound)
        guard targetOutput.firstPosition >= 0, targetOutput.firstPosition <= limit,
            hidden.dim(1) <= limit - targetOutput.firstPosition,
            !position.overflow, position.partialValue >= 0, position.partialValue < limit,
            selected.count <= limit - position.partialValue
        else {
            throw MiMoV26MTPError.contextExceeded
        }
        normalizedHiddenStates = hidden[0..., selected, 0...]
        firstPosition = position.partialValue
        provenance = .targetPostNorm
        targetOwner = MiMoV26MTPTargetOwner(target)
        predictor = nil
        generation = nil
    }

    fileprivate init(
        hidden: MLXArray, firstPosition: Int, depth: Int, predictor: MiMoV26MTP,
        owner: MiMoV26MTPTargetOwner, generation: UInt64
    ) {
        normalizedHiddenStates = hidden
        self.firstPosition = firstPosition
        provenance = .predictorPostNorm(depth: depth)
        targetOwner = owner
        self.predictor = predictor
        self.generation = generation
    }

    // The CBv2 assistant is the sole consumer of the target's family-specific
    // trusted hidden seam. This initializer is deliberately not public.
    init(trustedTargetHidden: MLXArray, firstPosition: Int, target: MiMoV26TextModel) {
        normalizedHiddenStates = trustedTargetHidden
        self.firstPosition = firstPosition
        provenance = .targetPostNorm
        targetOwner = MiMoV26MTPTargetOwner(target)
        predictor = nil
        generation = nil
    }
}

public struct MiMoV26MTPOutput {
    public let logits: MLXArray
    public let features: MiMoV26MTPFeatures
    public var normalizedHiddenStates: MLXArray { features.normalizedHiddenStates }
}

/// Keys retain the checkpoint's pre_mlp_layernorm spelling. SGLang renames
/// that key to post_attention_layernorm internally, not in the original file.
final class MiMoV26MTPHead: Module {
    let enorm: RMSNorm
    let hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProjection: Linear
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "pre_mlp_layernorm") var preMLPNorm: RMSNorm
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: MiMoV26Attention
    let mlp: MiMoV26DenseMLP

    init(_ config: MiMoV26Configuration) {
        let eps = Float(config.layernormEpsilon)
        enorm = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        hnorm = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        _ehProjection.wrappedValue = Linear(2 * config.hiddenSize, config.hiddenSize, bias: false)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        _preMLPNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        _attention.wrappedValue = MiMoV26Attention(config, geometry: config.slidingAttention)
        mlp = MiMoV26DenseMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    fileprivate func forward(
        hidden: MLXArray, embeddings: MLXArray,
        cache: MiMoV26MTPPositionedCache
    ) -> MLXArray {
        let projected = ehProjection(concatenated([enorm(embeddings), hnorm(hidden)], axis: -1))
        let mask = createAttentionMask(
            h: projected, cache: cache.storage, windowSize: cache.maxSize)
        let attended = projected + attention(inputNorm(projected), mask: mask, cache: cache)
        return finalNorm(attended + mlp(preMLPNorm(attended)))
    }
}

/// Native trained-head component. The target owns its embedding/readout once;
/// this module borrows those exact objects during forward and never registers
/// them in its parameter tree. Every declared head must load strictly before use.
public final class MiMoV26MTP: Module {
    public let configuration: MiMoV26Configuration
    let layers: [MiMoV26MTPHead]
    private let targetOwner: MiMoV26MTPTargetOwner
    private var weightGeneration: UInt64 = 0
    public private(set) var isLoaded = false
    public var headCount: Int { layers.count }
    var loadedGeneration: UInt64 { weightGeneration }
    func belongs(to target: MiMoV26TextModel) -> Bool { targetOwner.target === target }

    public init(target: MiMoV26TextModel, floatingProjections: Bool = false) throws {
        let config = target.configuration
        let doubled = config.hiddenSize.multipliedReportingOverflow(by: 2)
        guard config.numNextnPredictLayers == 3,
            !doubled.overflow, doubled.partialValue <= Int(Int32.max),
            config.numNextnPredictLayers < config.maxPositionEmbeddings
        else {
            throw MiMoV26MTPError.unsupportedConfiguration("next-N/concatenation dimensions")
        }
        configuration = config
        targetOwner = MiMoV26MTPTargetOwner(target)
        layers = (0 ..< config.numNextnPredictLayers).map { _ in MiMoV26MTPHead(config) }
        super.init()
        // Native converted MTP projections have explicit affine policies. Never
        // apply the trunk's MXFP4 default to an unlisted predictor projection.
        if floatingProjections {
            guard config.dtype == "bfloat16", config.quantization.nativeDefault?.mode == "affine",
                config.quantization.nativeDefault?.bits == 4,
                config.quantization.nativeDefault?.groupSize == 64,
                !config.quantization.nativeOverrides.keys.contains(where: {
                    $0.hasPrefix("mtp.") || $0.hasPrefix("language_model.mtp.")
                        || $0.hasPrefix("language_model.model.mtp.")
                })
            else {
                throw MiMoV26MTPError.unsupportedConfiguration(
                    "floating MTP must not inherit or contradict a quantization policy")
            }
        } else if config.quantization.nativeDefault != nil {
            var policies: [String: MiMoV26Quantization.Policy] = [:]
            for (path, module) in leafModules().flattened() where module is Linear {
                let short = config.quantization.nativeOverrides["mtp." + path]
                let wrapped = config.quantization.nativeOverrides["language_model.mtp." + path]
                if let short, let wrapped, short != wrapped {
                    throw MiMoV26MTPError.unsupportedConfiguration(
                        "conflicting MTP quantization aliases: \(path)")
                }
                guard let policy = short ?? wrapped else {
                    throw MiMoV26MTPError.unsupportedConfiguration(
                        "missing explicit MTP projection policy: \(path)")
                }
                switch policy {
                case .skip: break
                case .quantize(let value):
                    guard value.mode == "affine", value.bits == 4, value.groupSize == 64,
                        let linear = module as? Linear,
                        linear.weight.dim(1).isMultiple(of: value.groupSize)
                    else {
                        throw MiMoV26MTPError.unsupportedConfiguration(
                            "unsupported MTP projection policy: \(path)")
                    }
                    policies[path] = value
                }
            }
            // Head-local updates avoid sparse array traversal when a declared
            // head/projection intentionally stays in its source precision.
            for (index, head) in layers.enumerated() {
                var updates: [(String, Module)] = []
                for (path, module) in head.leafModules().flattened() {
                    guard let policy = policies["layers.\(index)." + path] else { continue }
                    guard
                        let replacement = quantizeSingle(
                            layer: module, groupSize: policy.groupSize,
                            bits: policy.bits, mode: .affine)
                    else {
                        throw MiMoV26MTPError.unsupportedConfiguration(
                            "unquantizable predictor: \(path)")
                    }
                    updates.append((path, replacement))
                }
                try head.update(modules: .unflattened(updates), verify: .noUnusedKeys)
            }
        }
    }

    /// Strict converted component load; no source FP8 expansion, fused-QKV
    /// splitting or checkpoint-wide filtering occurs here. The caller must
    /// supply exactly this component's tensors under its explicit prefix.
    public func loadConvertedWeights(_ weights: [String: MLXArray], prefix: String = "mtp.") throws
    {
        guard let target = targetOwner.target else { throw MiMoV26MTPError.incompatibleOwner }
        guard ["", "mtp.", "language_model.mtp."].contains(prefix) else {
            throw MiMoV26MTPError.invalidWeights("unsupported converted component prefix")
        }
        var normalized: [String: MLXArray] = [:]
        for (name, value) in weights {
            guard name.hasPrefix(prefix) else {
                throw MiMoV26MTPError.invalidWeights("out-of-component key: \(name)")
            }
            normalized[String(name.dropFirst(prefix.count))] = value
        }
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened())
        guard Set(normalized.keys) == Set(expected.keys) else {
            throw MiMoV26MTPError.invalidWeights("missing or extra trained predictor tensors")
        }
        let dtype = target.activationDType
        for (name, value) in normalized {
            let shape = expected[name]!
            let expectedDType: DType = shape.dtype == .uint32 ? .uint32 : dtype
            guard value.shape == shape.shape, value.dtype == expectedDType else {
                throw MiMoV26MTPError.invalidWeights("incompatible shape/dtype: \(name)")
            }
        }
        // Preflight covers every head before update can modify the first head.
        try update(parameters: .unflattened(normalized), verify: .all)
        weightGeneration &+= 1
        isLoaded = true
    }

    /// Begins an explicitly supplied history segment. A nonzero base does not
    /// reconstruct earlier context: callers must replay the required window.
    /// It is not a prefix-cache restore or speculative transaction constructor.
    public func newCache(basePosition: Int = 0) throws -> MiMoV26MTPRequestCache {
        guard basePosition >= 0, basePosition < configuration.maxPositionEmbeddings - headCount,
            targetOwner.target != nil
        else { throw MiMoV26MTPError.invalidHistory("invalid live owner/base position") }
        return MiMoV26MTPRequestCache(
            owner: self, generation: weightGeneration,
            window: configuration.slidingWindow, count: headCount,
            basePosition: basePosition)
    }

    /// Every depth consumes the same normalized target feature. MiMoV2MTP is
    /// deliberately absent from SGLang's chain_mtp_hidden_states allowlist.
    /// IDs align depth+1 positions AFTER the target feature rows. The caller
    /// owns per-depth shifted priming, scheduling and speculative rollback.
    public func forward(
        depth: Int, features: MiMoV26MTPFeatures, inputIDs: MLXArray,
        target: MiMoV26TextModel, cache: MiMoV26MTPRequestCache
    ) throws -> MiMoV26MTPOutput {
        try forward(
            depth: depth, features: features, inputIDs: inputIDs, target: target,
            cache: cache, validateTokenValues: true)
    }

    // Engine IDs already passed prompt/sampler vocabulary checks. Preserve all
    // geometry/owner/generation checks without adding a host readback per head.
    func forwardTrusted(
        depth: Int, features: MiMoV26MTPFeatures, inputIDs: MLXArray,
        target: MiMoV26TextModel, cache: MiMoV26MTPRequestCache
    ) throws -> MiMoV26MTPOutput {
        try forward(
            depth: depth, features: features, inputIDs: inputIDs, target: target,
            cache: cache, validateTokenValues: false)
    }

    func primeTrusted(
        depth: Int, features: MiMoV26MTPFeatures, inputIDs: MLXArray,
        target: MiMoV26TextModel, cache: MiMoV26MTPRequestCache
    ) throws {
        _ = try forward(
            depth: depth, features: features, inputIDs: inputIDs, target: target,
            cache: cache, validateTokenValues: false, projectLogits: false)
    }

    private func forward(
        depth: Int, features: MiMoV26MTPFeatures, inputIDs: MLXArray,
        target: MiMoV26TextModel, cache: MiMoV26MTPRequestCache,
        validateTokenValues: Bool, projectLogits: Bool = true
    ) throws -> MiMoV26MTPOutput {
        guard targetOwner.target === target, features.targetOwner.target === target,
            cache.owner === self
        else { throw MiMoV26MTPError.incompatibleOwner }
        guard isLoaded else { throw MiMoV26MTPError.weightsNotLoaded }
        guard layers.indices.contains(depth), cache.generation == weightGeneration else {
            throw MiMoV26MTPError.invalidHistory("head index or stale loaded-weight generation")
        }
        guard features.provenance == .targetPostNorm else {
            throw MiMoV26MTPError.invalidHistory(
                "MiMo predictors require target features, not a predictor chain")
        }
        let hidden = features.normalizedHiddenStates
        guard hidden.ndim == 3, hidden.dim(2) == configuration.hiddenSize,
            hidden.dtype == target.activationDType, inputIDs.ndim == 2,
            inputIDs.dtype == .int32 || inputIDs.dtype == .uint32,
            inputIDs.shape == Array(hidden.shape.prefix(2)), hidden.dim(0) > 0, hidden.dim(1) > 0,
            target.hasLoadedEmbeddingPrecision
        else {
            throw MiMoV26MTPError.invalidInput(
                "aligned IDs and loaded-precision normalized features required")
        }
        guard target.hasLoadedReadoutPrecision else {
            throw MiMoV26MTPError.invalidInput(
                "target readout must match the loaded native storage policy")
        }
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let advance = depth + 1
        guard features.firstPosition >= 0,
            features.firstPosition < configuration.maxPositionEmbeddings,
            length <= configuration.maxPositionEmbeddings - features.firstPosition - advance
        else {
            throw MiMoV26MTPError.contextExceeded
        }
        let firstTokenPosition = features.firstPosition + advance
        guard cache.layers[depth].offset == firstTokenPosition,
            cache.batchSize == nil || cache.batchSize == batch
        else {
            throw MiMoV26MTPError.invalidHistory(
                "noncontiguous shifted history or changed batch membership")
        }
        if validateTokenValues {
            let minimum =
                inputIDs.dtype == .int32
                ? Int64(inputIDs.min().item(Int32.self))
                : Int64(inputIDs.min().item(UInt32.self))
            let maximum =
                inputIDs.dtype == .int32
                ? Int64(inputIDs.max().item(Int32.self))
                : Int64(inputIDs.max().item(UInt32.self))
            guard minimum >= 0, maximum < Int64(configuration.vocabularySize) else {
                throw MiMoV26MTPError.invalidInput("token ID outside target vocabulary")
            }
        }
        // All input/ownership checks precede cache mutation and embedding lookup.
        cache.batchSize = batch
        let output = layers[depth].forward(
            hidden: hidden, embeddings: target.model.embedTokens(inputIDs),
            cache: cache.layers[depth])
        let logits =
            projectLogits
            ? (target.lmHead.map { $0(output) } ?? target.model.embedTokens.asLinear(output))
            : MLXArray.zeros([batch, 0, configuration.vocabularySize], dtype: output.dtype)
        return MiMoV26MTPOutput(
            logits: logits,
            features: .init(
                hidden: output, firstPosition: firstTokenPosition, depth: depth,
                predictor: self, owner: targetOwner, generation: weightGeneration))
    }
}
