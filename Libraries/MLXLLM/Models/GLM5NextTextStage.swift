// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon
import MLXNN

/// A contiguous range of global layer indices. Only the first stage accepts
/// tokens/embeddings; only the final stage owns the mean, final norm and head.
/// A final-stage NextN owner also needs the trained embedding, whose duplicate
/// bytes are included in `tensorNames(in:)` and `tensorBytes(in:)`.
public struct GLM5NextTextStagePlan {
    public let configuration: GLM5NextTextConfiguration
    public let layers: Range<Int>
    public let loadsMTPEmbedding: Bool
    /// Tensor parallel: this rank's part of every layer (`layers` is then the
    /// whole model). Nil for a pipeline stage or a single device.
    public let shard: GLM5NextTensorShard?
    public var isFirst: Bool { layers.lowerBound == 0 }
    public var isFinal: Bool { layers.upperBound == configuration.hiddenLayers }
    public var loadsEmbedding: Bool { isFirst || loadsMTPEmbedding }

    public init(
        configuration: GLM5NextTextConfiguration, layers: Range<Int>,
        loadMTPEmbedding: Bool = false, shard: GLM5NextTensorShard? = nil
    ) throws {
        guard !layers.isEmpty, layers.lowerBound >= 0,
            layers.upperBound <= configuration.hiddenLayers,
            !loadMTPEmbedding
                || (layers.upperBound == configuration.hiddenLayers
                    && configuration.numNextNPredictLayers == 1),
            shard == nil || (layers == 0 ..< configuration.hiddenLayers && !loadMTPEmbedding)
        else { throw GLM5NextLoadError.invalid("text stage range or MTP embedding ownership") }
        self.configuration = configuration
        self.layers = layers
        loadsMTPEmbedding = loadMTPEmbedding
        self.shard = shard
    }

    public func tensorNames(in inventory: GLM5NextWeightInventory) throws -> Set<String> {
        guard configuration == inventory.configuration.textConfig else {
            throw GLM5NextLoadError.invalid("text stage configuration/inventory mismatch")
        }
        let prefix = "model.language_model.layers."
        return Set(
            inventory.entries.values.compactMap { entry in
                guard entry.part == .text else { return nil }
                let name = entry.tensor.name
                if name.hasPrefix(prefix) {
                    let suffix = name.dropFirst(prefix.count)
                    guard let end = suffix.firstIndex(of: "."),
                        let index = Int(suffix[..<end]), layers.contains(index)
                    else { return nil }
                    return name
                }
                if loadsEmbedding && name.hasPrefix("model.language_model.embed_tokens.") {
                    return name
                }
                if isFinal
                    && (name.hasPrefix("model.language_model.norm.") || name.hasPrefix("lm_head."))
                {
                    return name
                }
                return nil
            })
    }

    public func tensorBytes(in inventory: GLM5NextWeightInventory) throws -> Int {
        try tensorNames(in: inventory).reduce(0) { total, name in
            let result = total.addingReportingOverflow(inventory.entries[name]!.bytes)
            guard !result.overflow else { throw GLM5NextLoadError.invalid("stage byte overflow") }
            return result.partialValue
        }
    }
}

private final class GLM5NextTextStageBinding {}

/// Request-local stage state. Copies preserve the stage binding and KDA rewind
/// checkpoints; arbitrary recurrent prefix truncation still requires replay.
public final class GLM5NextTextStageCache {
    fileprivate let binding: GLM5NextTextStageBinding
    fileprivate let storage: GLM5NextTextCache
    public var offset: Int { storage.offset }
    public var layers: [any KVCache] { storage.layers }

    fileprivate init(binding: GLM5NextTextStageBinding, storage: GLM5NextTextCache) {
        self.binding = binding
        self.storage = storage
    }

    public func copy() -> GLM5NextTextStageCache {
        .init(binding: binding, storage: storage.copy())
    }

    public func truncate(to count: Int) throws { try storage.truncate(to: count) }
}

/// Final-stage projection required by the caller. Intermediate stages always
/// return every residual stream, independent of this policy.
public enum GLM5NextTextStageReadout: Sendable, Equatable {
    case allTokens
    /// Project the last physical column, retaining its singleton time axis.
    /// Callers with padding must ensure that this is the intended readout row.
    case lastToken
    /// Evaluate hidden states and cache updates without constructing a head graph.
    case none
}

public enum GLM5NextTextStageOutput {
    /// The complete, unnormalized [B,T,hcMult,hidden] boundary. Never reduce
    /// the stream axis before sending it to the next stage.
    case streams(MLXArray)
    case final(GLM5NextTextOutput)
    /// Complete normalized [B,T,hidden] states without vocabulary projection.
    case hidden(MLXArray)

    fileprivate var roots: [MLXArray] {
        switch self {
        case .streams(let streams): [streams]
        case .hidden(let hidden): [hidden]
        case .final(let output): [output.hidden, output.logits]
        }
    }
}

/// Engine-thread-confined transaction. The fence must evaluate/join/fault-check
/// every root AND obtain any required downstream acknowledgement before it
/// returns. A throwing fence, stale candidate or discard leaves the cache intact.
public final class GLM5NextTextStageCandidate {
    public let output: GLM5NextTextStageOutput
    private let destination: GLM5NextTextCache
    private let original: GLM5NextTextCache.Revision
    private var working: GLM5NextTextCache?
    private enum State: Equatable { case pending, evaluating, committed, discarded }
    private var state = State.pending

    fileprivate init(
        output: GLM5NextTextStageOutput, destination: GLM5NextTextCache,
        original: GLM5NextTextCache.Revision, working: GLM5NextTextCache
    ) {
        self.output = output
        self.destination = destination
        self.original = original
        self.working = working
    }

    @discardableResult
    public func commit(afterEvaluating fence: ([MLXArray]) throws -> Void) throws
        -> GLM5NextTextStageOutput
    {
        guard state == .pending, let working else {
            throw GLM5NextLoadError.invalid("stage candidate is no longer pending")
        }
        guard destination.revision() == original else {
            discard()
            throw GLM5NextLoadError.invalid("stage cache changed before evaluation")
        }
        state = .evaluating
        do {
            try fence(output.roots + working.layers.flatMap { $0.innerState() })
            guard destination.revision() == original else {
                throw GLM5NextLoadError.invalid("stage cache changed during evaluation")
            }
            destination.adopt(working)
            self.working = nil
            state = .committed
            return output
        } catch {
            self.working = nil
            state = .discarded
            throw error
        }
    }

    public func discard() {
        guard state == .pending else { return }
        working = nil
        state = .discarded
    }
}

/// Loads only the selected stage from an already validated full inventory. It
/// never constructs and then discards layers belonging to another rank.
public final class GLM5NextTextStage: Module {
    public let plan: GLM5NextTextStagePlan
    public let inventory: GLM5NextWeightInventory
    public let executionPolicy: GLM5NextExecutionPolicy
    public let embedding: Embedding?
    public let layers: [GLM5NextTextLayer]
    public let norm: GLM5NextRMSNorm?
    public let head: Linear?
    /// Tensor parallel prefill: run each chunk as this many consecutive
    /// micro-batches so one computes while another's sum crosses the link
    /// (1: one batch). Chunks shorter than two micro-batches of
    /// `minimumMicroBatchTokens` run whole.
    public var tensorParallelMicroBatches = 1
    public static let minimumMicroBatchTokens = 256
    private let binding = GLM5NextTextStageBinding()

    public init(
        plan: GLM5NextTextStagePlan, inventory: GLM5NextWeightInventory,
        tensor: @escaping (String) throws -> MLXArray,
        materializeExpertStack: ((MLXArray) throws -> Void)? = nil,
        releaseSourceTensors: (([String]) -> Void)? = nil,
        executionPolicy: GLM5NextExecutionPolicy = .reference,
        materializeFusedProjections: (([MLXArray]) throws -> Void)? = nil
    ) throws {
        let selected = try plan.tensorNames(in: inventory)
        if inventory.storage.isPacked
            && (materializeExpertStack == nil || releaseSourceTensors == nil)
        {
            throw GLM5NextLoadError.invalid(
                "packed stage requires bounded materialization and eviction")
        }
        // A tensor-parallel rank builds every module at its local width.
        let c = try plan.shard?.localConfiguration(plan.configuration) ?? plan.configuration
        guard executionPolicy.routing == .reference || c.topkGroup == c.nGroup else {
            throw GLM5NextLoadError.invalid("native routing requires ungrouped experts")
        }
        self.plan = plan
        self.inventory = inventory
        self.executionPolicy = executionPolicy
        let loader = GLM5NextTensorLoader(
            inventory,
            { name in
                guard selected.contains(name) else { throw GLM5NextLoadError.unexpected(name) }
                return try tensor(name)
            }, releaseSourceTensors: releaseSourceTensors,
            materializeFusedProjections: materializeFusedProjections)
        loader.shard = plan.shard
        embedding =
            plan.loadsEmbedding ? try loader.embedding("model.language_model.embed_tokens") : nil
        layers = try plan.layers.map {
            try GLM5NextTextLayer(
                c, index: $0, loader: loader, materialize: materializeExpertStack,
                executionPolicy: executionPolicy)
        }
        norm =
            plan.isFinal ? try loader.norm("model.language_model.norm", epsilon: c.rmsNormEps) : nil
        head = plan.isFinal ? try loader.linear("lm_head") : nil
        guard loader.consumed == selected else {
            throw GLM5NextLoadError.invalid("unconsumed text stage tensors")
        }
        for module in layers.flatMap({ $0.modules() }) {
            if executionPolicy.normalization == .fusedKernel {
                (module as? GLM5NextRMSNorm)?.usesFusedKernel = true
            }
            if executionPolicy.routing != .reference, let router = module as? GLM5NextRouter {
                router.usesNativeKernel = true
                router.usesPrefillKernel = executionPolicy.routing == .nativeKernelPrefill
            }
        }
        norm?.usesFusedKernel = executionPolicy.normalization == .fusedKernel
    }

    /// Tensor parallel: installs the cross-rank sum every layer applies to its
    /// attention and feed-forward outputs. Only on a sharded stage, once.
    public func installTensorParallelReduction(_ reduce: @escaping (MLXArray) -> MLXArray) throws {
        guard plan.shard != nil, layers.allSatisfy({ $0.reduce == nil }) else {
            throw GLM5NextLoadError.invalid("tensor-parallel reduction requires an unreduced sharded stage")
        }
        for layer in layers { layer.reduce = reduce }
    }

    public func newCache() -> GLM5NextTextStageCache {
        let c = plan.configuration
        return .init(
            binding: binding,
            storage: GLM5NextTextCache(
                layers: plan.layers.map { index -> any KVCache in
                    c.layerTypes[index] == .linearAttention
                        ? GLM5NextKDACache() : GLM5NextDSACache()
                }))
    }

    public func prepareForward(
        tokens: MLXArray, cache: GLM5NextTextStageCache, validity: MLXArray? = nil,
        readout: GLM5NextTextStageReadout = .allTokens
    ) throws -> GLM5NextTextStageCandidate {
        guard plan.isFirst, let embedding, tokens.ndim == 2,
            tokens.dim(0) > 0, tokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(tokens.dtype)
        else {
            throw GLM5NextLoadError.invalid("only the first stage accepts integer [B,T] tokens")
        }
        return try prepareForward(
            embeddings: embedding(tokens), cache: cache, validity: validity, readout: readout)
    }

    public func prepareForward(
        embeddings: MLXArray, cache: GLM5NextTextStageCache, validity: MLXArray? = nil,
        readout: GLM5NextTextStageReadout = .allTokens
    ) throws -> GLM5NextTextStageCandidate {
        let c = plan.configuration
        guard plan.isFirst, embeddings.ndim == 3,
            embeddings.dim(0) > 0, embeddings.dim(1) > 0,
            embeddings.dim(2) == c.hiddenSize
        else {
            throw GLM5NextLoadError.invalid("only the first stage accepts [B,T,hidden] embeddings")
        }
        let streams = broadcast(
            embeddings.expandedDimensions(axis: 2),
            to: [embeddings.dim(0), embeddings.dim(1), c.hcMult, c.hiddenSize])
        return try prepare(streams: streams, cache: cache, validity: validity, readout: readout)
    }

    public func prepareForward(
        streams: MLXArray, cache: GLM5NextTextStageCache, validity: MLXArray? = nil,
        readout: GLM5NextTextStageReadout = .allTokens
    ) throws -> GLM5NextTextStageCandidate {
        guard !plan.isFirst else {
            throw GLM5NextLoadError.invalid("first stage requires tokens or embeddings")
        }
        return try prepare(streams: streams, cache: cache, validity: validity, readout: readout)
    }

    private func prepare(
        streams: MLXArray, cache: GLM5NextTextStageCache, validity: MLXArray?,
        readout: GLM5NextTextStageReadout
    ) throws -> GLM5NextTextStageCandidate {
        let c = plan.configuration
        guard cache.binding === binding, streams.ndim == 4,
            streams.dim(0) > 0, streams.dim(1) > 0,
            streams.dim(2) == c.hcMult, streams.dim(3) == c.hiddenSize,
            [.float32, .bfloat16, .float16].contains(streams.dtype),
            cache.layers.count == plan.layers.count,
            zip(plan.layers, cache.layers).allSatisfy({ index, state in
                state.offset == cache.offset
                    && (c.layerTypes[index] == .linearAttention
                        ? state is GLM5NextKDACache : state is GLM5NextDSACache)
            }), streams.dim(1) <= c.maxPositionEmbeddings - cache.offset
        else { throw GLM5NextLoadError.invalid("stage boundary geometry, cache or context") }
        let validity = validity ?? MLXArray.ones(Array(streams.shape.prefix(2)), dtype: .bool)
        guard validity.shape == Array(streams.shape.prefix(2)), validity.dtype == .bool else {
            throw GLM5NextLoadError.invalid("stage validity must be boolean [B,T]")
        }
        let original = cache.storage.revision()
        let working = cache.storage.copy()
        let parts = plan.shard == nil ? 1
            : min(tensorParallelMicroBatches, streams.dim(1) / Self.minimumMicroBatchTokens)
        if parts > 1 {
            let boundary = try microBatchedForward(streams, validity: validity, working: working, parts: parts)
            return try finish(boundary: boundary, validity: validity, readout: readout,
                cache: cache, original: original, working: working)
        }
        var next = GLM5NextTextLayer.Streams.materialized(streams)
        let early =
            executionPolicy.stepEvaluation == .earlyAsync
            && streams.dim(1) <= GLM5NextExecutionPolicy.earlyEvaluationMaximumTokens
        let stride = GLM5NextExecutionPolicy.earlyEvaluationLayerStride
        for (index, (layer, state)) in zip(layers, working.layers).enumerated() {
            next = try layer.forward(next, cache: state, validity: validity)
            if early, (index + 1) % stride == 0, index + 1 < layers.count {
                asyncEval(next.arrays)
            }
        }
        return try finish(boundary: try next.materialize(), validity: validity, readout: readout,
            cache: cache, original: original, working: working)
    }

    /// The chunk's tokens in `parts` consecutive micro-batches, layer by layer:
    /// every micro-batch's attention partial, each sum issued (`asyncEval`) as
    /// soon as it is built, then every feed-forward partial the same way. The
    /// GPU computes one micro-batch while the CPU stream seals and exchanges
    /// another's sum. Each micro-batch reads the cache the earlier ones wrote
    /// at the same layer, as consecutive chunks would.
    private func microBatchedForward(
        _ streams: MLXArray, validity: MLXArray, working: GLM5NextTextCache, parts: Int
    ) throws -> MLXArray {
        typealias Phase = GLM5NextTextLayer.Phase
        let tokens = streams.dim(1)
        let bounds = (0 ... parts).map { $0 * tokens / parts }
        var pieces: [GLM5NextTextLayer.Streams] = (0 ..< parts).map {
            .materialized(streams[0..., bounds[$0] ..< bounds[$0 + 1]])
        }
        let validities = (0 ..< parts).map { validity[0..., bounds[$0] ..< bounds[$0 + 1]] }
        for (layer, state) in zip(layers, working.layers) {
            var attention: [(phase: Phase<Phase<GLM5NextTextLayer.Streams>>, sum: MLXArray)] = []
            for index in 0 ..< parts {
                let phase = try layer.phases(pieces[index], cache: state, validity: validities[index])
                let sum = layer.summed(phase.partial)
                asyncEval(sum)
                attention.append((phase, sum))
            }
            var feedForward: [(phase: Phase<GLM5NextTextLayer.Streams>, sum: MLXArray)] = []
            for step in attention {
                let phase = try step.phase.resume(step.sum)
                let sum = layer.summed(phase.partial)
                asyncEval(sum)
                feedForward.append((phase, sum))
            }
            for (index, step) in feedForward.enumerated() { pieces[index] = try step.phase.resume(step.sum) }
        }
        return concatenated(try pieces.map { try $0.materialize() }, axis: 1)
    }

    private func finish(
        boundary: MLXArray, validity: MLXArray, readout: GLM5NextTextStageReadout,
        cache: GLM5NextTextStageCache, original: GLM5NextTextCache.Revision, working: GLM5NextTextCache
    ) throws -> GLM5NextTextStageCandidate {
        let output: GLM5NextTextStageOutput
        if let norm, let head {
            let hidden = MLX.where(
                validity.expandedDimensions(axis: -1), norm(boundary.mean(axis: 2)), 0)
            switch readout {
            case .allTokens:
                output = .final(.init(logits: head(hidden), hidden: hidden))
            case .lastToken:
                output = .final(.init(logits: head(hidden[0..., (-1)..., 0...]), hidden: hidden))
            case .none:
                output = .hidden(hidden)
            }
        } else {
            output = .streams(boundary)
        }
        guard cache.storage.revision() == original else {
            throw GLM5NextLoadError.invalid("stage cache changed while preparing candidate")
        }
        return .init(
            output: output, destination: cache.storage, original: original, working: working)
    }
}
