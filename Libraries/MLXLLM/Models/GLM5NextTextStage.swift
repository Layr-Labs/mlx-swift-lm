// Copyright © 2026 Eigen Labs.
import Foundation
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

final class GLM5NextTextStageBinding {}

/// Request-local stage state. Copies preserve the stage binding and KDA rewind
/// checkpoints; arbitrary recurrent prefix truncation still requires replay.
public final class GLM5NextTextStageCache {
    let binding: GLM5NextTextStageBinding
    let storage: GLM5NextTextCache
    public var offset: Int { storage.offset }
    public var layers: [any KVCache] { storage.layers }

    init(binding: GLM5NextTextStageBinding, storage: GLM5NextTextCache) {
        self.binding = binding
        self.storage = storage
    }

    public func copy() -> GLM5NextTextStageCache {
        .init(binding: binding, storage: storage.copy())
    }

    public func truncate(to count: Int) throws { try storage.truncate(to: count) }

    /// Paged DSA histories (`GLM5NextTextStage.dsaPagePool`): the pages every
    /// history's table holds, and their bytes (0 when contiguous). A copy
    /// shares its source's pages, so copies count them again.
    public var heldPages: Int {
        layers.reduce(0) { $0 + (($1 as? GLM5NextDSACache)?.heldPages ?? 0) }
    }
    public var pagedBytes: Int {
        layers.reduce(0) { $0 + (($1 as? GLM5NextDSACache)?.pagedBytes ?? 0) }
    }

    /// Counts the pages every paged DSA history appends from now on against
    /// `account` (a restored prefix copy joins its request's reservation).
    public func bindPages(to account: GLM5NextDSAPageBook.Account?) {
        for case let history as GLM5NextDSACache in layers where history.pagePool != nil {
            history.pageAccount = account
        }
    }
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
    /// `lastToken` projected on vocabulary rows `[part * V / parts, (part + 1) * V / parts)`
    /// only (`GLM5NextTextStage.vocabularyPart`): each output row is the
    /// whole head's bits for that row (one-row `qmv` reduces a row the same
    /// for any row count), so tensor-parallel ranks can each read half of the
    /// replicated head and agree on a greedy token through one small sum.
    case lastTokenVocabularyPart(part: Int, parts: Int)
    /// `allTokens` on the same vocabulary rows: every row of a speculative
    /// window. Each logit is the whole head's bits for its vocabulary row at
    /// any row count (`qmv`/`qmv_wide` reduce each output row alike for any
    /// number of output rows), so ranks can split a greedy window's head.
    case allTokensVocabularyPart(part: Int, parts: Int)
}

public enum GLM5NextTextStageOutput {
    /// The complete, unnormalized [B,T,hcMult,hidden] boundary. Never reduce
    /// the stream axis before sending it to the next stage.
    case streams(MLXArray)
    case final(GLM5NextTextOutput)
    /// Complete normalized [B,T,hidden] states without vocabulary projection.
    case hidden(MLXArray)

    var roots: [MLXArray] {
        switch self {
        case .streams(let streams): [streams]
        case .hidden(let hidden): [hidden]
        case .final(let output): [output.hidden, output.logits]
        }
    }
}

/// One prefill chunk's rows inside a segmented prefill
/// (`GLM5NextTextStage.prepareForward(tokens:cache:segments:readout:snapshotAfter:)`):
/// the micro-batches its rows run as, in order.
public struct GLM5NextPrefillSegment: Sendable, Equatable {
    /// Rows of each micro-batch; each at least one.
    public let microBatchRows: [Int]
    /// The micro-batches whose DSA selections these rows keep (same total),
    /// when they differ from `microBatchRows`: each row's top-k pools are
    /// chosen among the candidates its own micro-batch there would have
    /// (every pool complete through that micro-batch's last key). The
    /// selector's tie order depends on the candidate count, so this keeps a
    /// different cut's selections, and so its tokens, the product's bits
    /// (`GLM5NextSparseAttention.selectTokens(selectionHistories:)`). Prompt
    /// prefill only (every key valid). nil: each micro-batch is its own.
    public let selectionRows: [Int]?
    public var tokens: Int { microBatchRows.reduce(0, +) }

    public init(microBatchRows: [Int], selectionRows: [Int]? = nil) {
        self.microBatchRows = microBatchRows
        self.selectionRows = selectionRows == microBatchRows ? nil : selectionRows
    }

    /// The micro-batches `GLM5NextTextStage.prepareForward(tokens:cache:readout:)`
    /// runs a chunk of `tokens` rows as on a stage with `microBatches` set
    /// (`tensorParallelMicroBatches`; a tensor-parallel rank, so `sharded`):
    /// `parts` near-equal consecutive pieces, or the whole chunk.
    public static func chunk(tokens: Int, microBatches: Int, sharded: Bool = true) -> Self {
        let parts = sharded ? min(microBatches, tokens / GLM5NextTextStage.minimumMicroBatchTokens) : 1
        guard parts > 1 else { return .init(microBatchRows: [tokens]) }
        let bounds = (0 ... parts).map { $0 * tokens / parts }
        return .init(microBatchRows: (0 ..< parts).map { bounds[$0 + 1] - bounds[$0] })
    }
}

/// A segmented prefill (`GLM5NextTextStage.prepareForward(tokens:cache:segments:readout:snapshotAfter:)`,
/// or from embeddings) built in layer slices (`GLM5NextTextStage.beginForward`): `advance(layers:)`
/// builds the next layers of every micro-batch exactly as the whole call
/// does (`GLM5NextTextStage.microBatchedLayer`), and the caller may evaluate
/// `frontier` (every micro-batch's streams at that layer, the working cache
/// and the snapshot so far) and run other forwards on other caches before
/// the next slice; `finish()` reads out and returns the call's candidate.
/// Every op, its inputs and their order are the whole call's; only when the
/// GPU runs them changes, so the candidate's outputs, cache and snapshot are
/// the whole call's bits. Engine-thread-confined; the cache must not change
/// until the candidate commits or is discarded (the candidate checks).
public final class GLM5NextSlicedPrefill {
    private let stage: GLM5NextTextStage
    private let cache: GLM5NextTextStageCache
    private let segments: [GLM5NextPrefillSegment]
    private let readout: GLM5NextTextStageReadout
    private let snapshotMicroBatches: Int?
    private let original: GLM5NextTextCache.Revision
    private let start: Int
    private let working: GLM5NextTextCache
    private let validities: [MLXArray]
    /// With sequence parallelism: which micro-batches hold this rank's rows.
    private let sequenceParallel: GLM5NextTextLayer.SequenceParallel?
    private let split: [Bool]?
    /// Per micro-batch, its rows' DSA selection histories when a segment
    /// keeps other micro-batches' selections (`GLM5NextPrefillSegment.selectionRows`).
    private let selections: [GLM5NextDSAAttention.SelectionHistories?]
    private var pieces: [GLM5NextTextLayer.Streams]
    private var kept: [any KVCache] = []
    private var finished = false
    /// Layers built so far, for every micro-batch.
    public private(set) var layersBuilt = 0
    public var layerCount: Int { stage.layers.count }

    fileprivate init(
        stage: GLM5NextTextStage, cache: GLM5NextTextStageCache, segments: [GLM5NextPrefillSegment],
        readout: GLM5NextTextStageReadout, snapshotMicroBatches: Int?, pieces: [GLM5NextTextLayer.Streams],
        validities: [MLXArray], sequenceParallel: GLM5NextTextLayer.SequenceParallel?, split: [Bool]?,
        selections: [GLM5NextDSAAttention.SelectionHistories?]
    ) {
        self.stage = stage
        self.cache = cache
        self.segments = segments
        self.readout = readout
        self.snapshotMicroBatches = snapshotMicroBatches
        self.pieces = pieces
        self.validities = validities
        self.sequenceParallel = sequenceParallel
        self.split = split
        self.selections = selections
        original = cache.storage.revision()
        start = cache.offset
        working = cache.storage.copy()
    }

    /// Builds the next `count` layers of every micro-batch (unevaluated).
    public func advance(layers count: Int) throws {
        guard !finished, count > 0, layersBuilt + count <= layerCount else {
            throw GLM5NextLoadError.invalid("a sliced prefill advances past its layers or after its readout")
        }
        for index in layersBuilt ..< layersBuilt + count {
            if let sequenceParallel, let split {
                try stage.sequenceParallelLayer(
                    index, pieces: &pieces, validities: validities, split: split, working: working,
                    snapshotAfter: snapshotMicroBatches, kept: &kept, sequenceParallel: sequenceParallel,
                    selections: selections)
            } else {
                try stage.microBatchedLayer(
                    index, pieces: &pieces, validities: validities, working: working,
                    snapshotAfter: snapshotMicroBatches, kept: &kept, selections: selections)
            }
        }
        layersBuilt += count
    }

    /// Diagnostics: every micro-batch's boundary arrays as built so far, in
    /// micro-batch order (a pending mHC expand: output, residual streams,
    /// post, comb; or the materialized streams), each `[B, rows, ...]`.
    public var boundaryArrays: [[MLXArray]] { pieces.map(\.arrays) }

    /// Diagnostics: layer `index`'s working cache state as written so far.
    public func workingState(layer index: Int) -> [MLXArray] { working.layers[index].state }

    /// What a fence evaluates to run every layer built so far.
    public var frontier: [MLXArray] {
        pieces.flatMap(\.arrays) + working.layers.flatMap { $0.innerState() } + kept.flatMap { $0.innerState() }
    }

    /// The whole call's candidate, once every layer is built.
    public func finish() throws -> GLM5NextTextStageCandidate {
        guard !finished, layersBuilt == layerCount else {
            throw GLM5NextLoadError.invalid("a sliced prefill reads out before its last layer")
        }
        finished = true
        // Sequence-parallel micro-batches read out their rows and gather them
        // (a row's readout is its own); the others return their streams.
        let boundaries = try pieces.enumerated().map { part, piece -> MLXArray in
            let boundary = try piece.materialize()
            guard let sequenceParallel, split?[part] == true, let norm = stage.norm else { return boundary }
            let hidden = sequenceParallel.allGather(norm(boundary.mean(axis: 2)))
            asyncEval(hidden)
            return hidden
        }
        let (output, hidden) = try stage.segmentedReadout(boundaries, segments: segments, readout: readout)
        guard cache.storage.revision() == original else {
            throw GLM5NextLoadError.invalid("stage cache changed while preparing candidate")
        }
        return .init(
            output: output, destination: cache.storage, original: original, working: working,
            start: start, tokens: segments.reduce(0) { $0 + $1.tokens }, captured: false, segmentHidden: hidden,
            snapshot: snapshotMicroBatches == nil
                ? nil : GLM5NextTextStageCache(binding: stage.binding, storage: GLM5NextTextCache(layers: kept)))
    }
}

/// Engine-thread-confined transaction. The fence must evaluate/join/fault-check
/// every root AND obtain any required downstream acknowledgement before it
/// returns. A throwing fence, stale candidate or discard leaves the cache intact.
public final class GLM5NextTextStageCandidate {
    public let output: GLM5NextTextStageOutput
    /// A segmented prefill's normalized hidden rows, one `[B, tokens, hidden]`
    /// array per segment (each segment's readout on its own rows); empty for
    /// any other forward.
    public let segmentHidden: [MLXArray]
    /// A segmented prefill's stage cache as it stood after its
    /// `snapshotAfter` segments (a copy the fence also evaluates), for a
    /// prompt-prefix entry inside the call. Nil when none was asked for.
    public let snapshot: GLM5NextTextStageCache?
    private let destination: GLM5NextTextCache
    private let original: GLM5NextTextCache.Revision
    private var working: GLM5NextTextCache?
    /// The cache offset before this candidate's tokens, their count, and
    /// whether every KDA layer captured each token's state
    /// (`GLM5NextTextStage.prepareVerification`).
    private let start: Int
    let tokens: Int
    let captured: Bool
    private enum State: Equatable { case pending, evaluating, committed, discarded }
    private var state = State.pending

    init(
        output: GLM5NextTextStageOutput, destination: GLM5NextTextCache,
        original: GLM5NextTextCache.Revision, working: GLM5NextTextCache,
        start: Int, tokens: Int, captured: Bool, segmentHidden: [MLXArray] = [],
        snapshot: GLM5NextTextStageCache? = nil
    ) {
        self.output = output
        self.segmentHidden = segmentHidden
        self.snapshot = snapshot
        self.destination = destination
        self.original = original
        self.working = working
        self.start = start
        self.tokens = tokens
        self.captured = captured
    }

    /// Publishes every token. A verification candidate also releases its
    /// captured per-token states (as `commit(afterEvaluating:keeping:)`
    /// keeping all tokens).
    @discardableResult
    public func commit(afterEvaluating fence: ([MLXArray]) throws -> Void) throws
        -> GLM5NextTextStageOutput
    {
        try commit(afterEvaluating: fence, keep: { [tokens] _ in tokens })
    }

    /// Speculative verification: publishes only the first `keeping(output)`
    /// tokens of a `prepareVerification` window (1 through the window's
    /// length), decided after the fence from the evaluated output. Each KDA
    /// layer adopts the state it captured after the last kept token and each
    /// DSA history drops the rest, so the destination never holds a rejected
    /// token and no replay forward runs. A throwing `keeping` leaves the cache
    /// intact.
    @discardableResult
    public func commit(
        afterEvaluating fence: ([MLXArray]) throws -> Void,
        keeping: (GLM5NextTextStageOutput) throws -> Int
    ) throws -> GLM5NextTextStageOutput {
        guard captured else {
            throw GLM5NextLoadError.invalid("only a verification candidate keeps a token prefix")
        }
        return try commit(afterEvaluating: fence, keep: keeping)
    }

    /// `keep` runs only for a verification candidate.
    private func commit(
        afterEvaluating fence: ([MLXArray]) throws -> Void,
        keep: (GLM5NextTextStageOutput) throws -> Int
    ) throws -> GLM5NextTextStageOutput {
        guard state == .pending, let working else {
            throw GLM5NextLoadError.invalid("stage candidate is no longer pending")
        }
        guard destination.revision() == original else {
            discard()
            throw GLM5NextLoadError.invalid("stage cache changed before evaluation")
        }
        state = .evaluating
        do {
            try fence(
                output.roots + working.layers.flatMap { $0.innerState() }
                    + (snapshot?.layers.flatMap { $0.innerState() } ?? []))
            guard destination.revision() == original else {
                throw GLM5NextLoadError.invalid("stage cache changed during evaluation")
            }
            if captured {
                let kept = try keep(output)
                guard (1 ... tokens).contains(kept) else {
                    throw GLM5NextLoadError.invalid("verification keeps 1...\(tokens) tokens, not \(kept)")
                }
                // Also releases the captured windows when every token is kept.
                try working.truncate(to: start + kept)
            }
            destination.adopt(working)
            self.working = nil
            state = .committed
            // Every graph that read the pages this commit released was
            // submitted by the fence: they may be written again.
            destination.recyclePages()
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

    /// The roots `commit`'s fence evaluates, for an owner that evaluates
    /// several candidates in one fence (`GLM5NextTextStageRowsCandidate`).
    func pendingRoots() throws -> [MLXArray] {
        guard state == .pending, let working, destination.revision() == original else {
            throw GLM5NextLoadError.invalid("stage candidate is no longer pending")
        }
        return output.roots + working.layers.flatMap { $0.innerState() }
            + (snapshot?.layers.flatMap { $0.innerState() } ?? [])
    }

    /// `commit` after the owner's fence already evaluated `pendingRoots()`,
    /// keeping `count` tokens (a verification candidate; others keep all).
    func publishEvaluated(keeping count: Int) throws {
        try commit(afterEvaluating: { _ in }, keep: { _ in count })
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
    /// Sequence-parallel prefill (`installTensorParallelSequenceParallel`):
    /// segmented prefill micro-batches of an even `shares *
    /// sequenceParallelMinimumRows` rows or more keep the residual streams of
    /// this rank's rows only (`GLM5NextTextLayer.sequenceParallelSteps`).
    var sequenceParallel: GLM5NextTextLayer.SequenceParallel?
    public var sequenceParallelEnabled: Bool { sequenceParallel != nil }
    /// Segmented prefill shares each DSA layer's sparse-row selection between
    /// the tensor-parallel ranks by rows (`installTensorParallelSelectionSplit`).
    var selectionSplit: SelectionSplit?
    public var selectionSplitEnabled: Bool { selectionSplit != nil }

    /// This rank (`share`) of `shares`, and the cross-rank sum of the ranks'
    /// float32 selection partials (`GLM5NextSparseAttention.selectTokensSplit`).
    struct SelectionSplit {
        let share: Int
        let shares: Int
        /// A micro-batch shares its selection only when its history (the
        /// keys before its first row) holds at least this many keys.
        var minimumHistory = 0
        let exchange: (MLXArray) -> MLXArray
    }
    /// A row's mHC sites and router compute the same in a call of this many
    /// rows or more (the router's FP32 GEMM changes order below about 1,400).
    public static let sequenceParallelMinimumRows = 2048
    /// Multi-request steps (`prepareVerificationRows`, `prepareRows`) under
    /// `HyperConnection.nativeNormFused` run each mHC site once over every
    /// lane's rows and the routed experts in decode-sized chunks; false runs
    /// every layer group by group (the form of every other policy).
    public var rowsShareSites = true
    /// Qualification seam: the rows of one row group instead of the bound
    /// that keeps every row's arithmetic (`rowGroups`); a larger value makes
    /// rows differ from their own steps (the harness's negative control).
    var rowGroupRowsOverride: Int?
    let binding = GLM5NextTextStageBinding()
    /// Paged DSA storage for the request caches this stage makes
    /// (`newCache`); nil keeps every history contiguous. Set before the first
    /// cache is made.
    public var dsaPagePool: GLM5NextDSAPagePool?

    public init(
        plan: GLM5NextTextStagePlan, inventory: GLM5NextWeightInventory,
        tensor: @escaping (String) throws -> MLXArray,
        materializeExpertStack: ((MLXArray) throws -> Void)? = nil,
        releaseSourceTensors: (([String]) -> Void)? = nil,
        executionPolicy: GLM5NextExecutionPolicy = .reference,
        materializeFusedProjections: (([MLXArray]) throws -> Void)? = nil,
        prepared: GLM5NextPreparedParts? = nil
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
        loader.prepared = prepared
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

    /// Tensor parallel on two ranks, after `installTensorParallelReduction`:
    /// segmented prefill micro-batches divide their residual streams by rows
    /// between the ranks (this one `share` of `shares`); each layer gathers
    /// its attention and feed-forward inputs (`allGather`, `[1, T, ...]` ->
    /// `[1, shares * T, ...]` in rank order, every element's bits) and
    /// reduce-scatters its partials (`reduceScatter`, this rank's rows of the
    /// sum, added in rank order as `reduce` adds). The mHC sites and routers
    /// then run on this rank's rows only; every token's bits are unchanged.
    public func installTensorParallelSequenceParallel(
        share: Int, shares: Int, reduceScatter: @escaping (MLXArray) -> MLXArray,
        allGather: @escaping (MLXArray) -> MLXArray
    ) throws {
        guard plan.shard != nil, plan.isFirst, plan.isFinal, layers.allSatisfy({ $0.reduce != nil }),
            sequenceParallel == nil, shares > 1, (0 ..< shares).contains(share),
            executionPolicy.hyperConnection == .nativeNormFused
        else {
            throw GLM5NextLoadError.invalid(
                "sequence parallelism needs a reduced whole tensor-parallel stage with fused mHC sites")
        }
        sequenceParallel = .init(
            share: share, shares: shares, reduceScatter: reduceScatter, allGather: allGather)
    }

    /// Tensor parallel: segmented prefill (`beginForward(tokens:cache:segments:)`)
    /// shares every DSA layer's sparse-row top-k selection between the ranks
    /// by rows. Each rank scores and selects its `share` of each micro-batch's
    /// sparse rows (the indexer is replicated, so a rank would otherwise score
    /// every row twice over), `exchange` sums the ranks' float32 partials of
    /// chosen pool IDs (each zero outside its rows; the sealed all-sum), and
    /// every rank expands every row's selection. The layer runs every
    /// micro-batch's selection and exchange before any micro-batch's
    /// attention, so each exchange overlaps the next micro-batches' work.
    /// The same tokens and cache bits: each row is scored and selected alone
    /// (`selectTokensSplit`). Not with sequence parallelism.
    /// `minimumHistory`: only micro-batches whose history holds at least
    /// this many keys share their selection (a row's selection costs about
    /// its history; the exchange costs about its row), the others select
    /// alone as without the split. Both ranks decide alike from the cache.
    public func installTensorParallelSelectionSplit(
        share: Int, shares: Int, minimumHistory: Int = 0, exchange: @escaping (MLXArray) -> MLXArray
    ) throws {
        guard plan.shard != nil, plan.isFirst, plan.isFinal, selectionSplit == nil, sequenceParallel == nil,
            shares > 1, (0 ..< shares).contains(share), minimumHistory >= 0
        else {
            throw GLM5NextLoadError.invalid("a selection split needs a whole tensor-parallel stage, once")
        }
        selectionSplit = .init(share: share, shares: shares, minimumHistory: minimumHistory, exchange: exchange)
    }

    /// Tensor parallel: installs the cross-rank sum every layer applies to its
    /// attention and feed-forward outputs. Only on a sharded stage, once.
    public func installTensorParallelReduction(_ reduce: @escaping (MLXArray) -> MLXArray) throws {
        guard plan.shard != nil, layers.allSatisfy({ $0.reduce == nil }) else {
            throw GLM5NextLoadError.invalid("tensor-parallel reduction requires an unreduced sharded stage")
        }
        for layer in layers { layer.reduce = reduce }
    }

    public func newCache() -> GLM5NextTextStageCache { newCache(account: nil) }

    /// A request cache; with `dsaPagePool`, every DSA history is paged and
    /// its pages count against `account`.
    public func newCache(account: GLM5NextDSAPageBook.Account?) -> GLM5NextTextStageCache {
        let c = plan.configuration
        let pool = dsaPagePool
        return .init(
            binding: binding,
            storage: GLM5NextTextCache(
                layers: plan.layers.map { index -> any KVCache in
                    c.layerTypes[index] == .linearAttention
                        ? GLM5NextKDACache()
                        : pool.map { GLM5NextDSACache(pagePool: $0, account: account) }
                            ?? GLM5NextDSACache()
                }))
    }

    /// A page pool for this stage's DSA histories (and a NextN layer's,
    /// which has the same geometry): widths from the configuration, element
    /// types from the first DSA layer's projections of an `activation` input
    /// (a lazy graph, never evaluated). `activation` defaults to the
    /// embedding's output type.
    public func makeDSAPagePool(pageSize: Int = 64, activation: DType? = nil) throws
        -> GLM5NextDSAPagePool
    {
        guard
            let attention = layers.lazy.compactMap({ $0.attention as? GLM5NextDSAAttention })
                .first
        else { throw GLM5NextLoadError.invalid("paged DSA needs a DSA layer") }
        let input: DType
        if let activation {
            input = activation
        } else if let embedding {
            input = embedding(MLXArray([Int32(0)], [1, 1])).dtype
        } else {
            throw GLM5NextLoadError.invalid("paged DSA needs an activation type")
        }
        let types = try attention.historyTypes(input: input)
        let c = plan.configuration
        return try GLM5NextDSAPagePool(
            pageSize: pageSize, latentWidth: c.kvLoraRank, keyWidth: c.indexHeadDim,
            latentType: types.latents, keyType: types.keys, gateType: types.gates)
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
        if let streams = embeddedStreams(tokens) {
            return try prepare(streams: streams, cache: cache, validity: validity, readout: readout)
        }
        return try prepareForward(
            embeddings: embedding(tokens), cache: cache, validity: validity, readout: readout)
    }

    /// Decode-sized steps of a packed embedding: the embedding's broadcast
    /// residual streams `[B, T, hcMult, hidden]`, gathered, dequantized and
    /// stored in one launch (`GLM5NextPackedEmbedding.streams`), the same bits
    /// as `broadcast(embedding(tokens))`. nil keeps the separate operations.
    func embeddedStreams(_ tokens: MLXArray) -> MLXArray? {
        guard tokens.size <= GLM5NextPackedEmbedding.maximumStreamTokens,
            let packed = embedding as? GLM5NextPackedEmbedding
        else { return nil }
        return packed.streams(tokens, copies: plan.configuration.hcMult)
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

    /// The longest `prepareVerification` window: the fused KDA step captures
    /// the states of at most this many tokens.
    public static let maximumVerificationTokens = GLM5NextLinearDecodeKernel.maximumTokens

    /// A speculative verification window on a whole-model stage (a single
    /// device or a tensor-parallel rank): `prepareForward(tokens:cache:)` with
    /// the `.allTokens` readout, whose KDA layers also capture every window
    /// token's state, so `GLM5NextTextStageCandidate.commit(afterEvaluating:keeping:)`
    /// can publish any leading part of the window without a replay forward.
    /// The window's arithmetic is the ordinary `[1, T]` step's.
    /// `earlyEvaluation` starts GPU work every
    /// `GLM5NextExecutionPolicy.earlyEvaluationLayerStride` layers while the
    /// rest of the window is built (`StepEvaluation.earlyAsync` for this
    /// window only): the same graph and results, submitted sooner. Every rank
    /// submits the same layers at the same points, so its sums keep their order.
    /// `vocabularyPart` reads out only those vocabulary rows of the head
    /// (`GLM5NextTextStageReadout.allTokensVocabularyPart`).
    public func prepareVerification(
        tokens: MLXArray, cache: GLM5NextTextStageCache, earlyEvaluation: Bool = false,
        vocabularyPart: (part: Int, parts: Int)? = nil
    ) throws -> GLM5NextTextStageCandidate {
        guard plan.isFirst, plan.isFinal, let embedding, tokens.ndim == 2, tokens.dim(0) == 1,
            (1 ... Self.maximumVerificationTokens).contains(tokens.dim(1)),
            [.int32, .uint32, .int64].contains(tokens.dtype)
        else {
            throw GLM5NextLoadError.invalid(
                "verification needs a whole-model stage and [1, 1...\(Self.maximumVerificationTokens)] tokens")
        }
        let streams =
            embeddedStreams(tokens)
            ?? {
                let embeddings = embedding(tokens)
                let c = plan.configuration
                return broadcast(
                    embeddings.expandedDimensions(axis: 2),
                    to: [embeddings.dim(0), embeddings.dim(1), c.hcMult, c.hiddenSize])
            }()
        return try prepare(
            streams: streams, cache: cache, validity: nil,
            readout: vocabularyPart.map { .allTokensVocabularyPart(part: $0.part, parts: $0.parts) }
                ?? .allTokens,
            captureWindow: true, earlyEvaluation: earlyEvaluation)
    }

    private func prepare(
        streams: MLXArray, cache: GLM5NextTextStageCache, validity: MLXArray?,
        readout: GLM5NextTextStageReadout, captureWindow: Bool = false,
        earlyEvaluation: Bool = false
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
        // An omitted validity is all true: a shared constant, and the final
        // `where` is the identity.
        let allValid = validity == nil
        let rows = Array(streams.shape.prefix(2))
        let validity =
            validity ?? GLM5NextKernelConstants.shared.allTrue(rows)
            ?? MLXArray.ones(rows, dtype: .bool)
        guard validity.shape == Array(streams.shape.prefix(2)), validity.dtype == .bool else {
            throw GLM5NextLoadError.invalid("stage validity must be boolean [B,T]")
        }
        let original = cache.storage.revision()
        let start = cache.offset
        let working = cache.storage.copy()
        let parts = plan.shard == nil || captureWindow ? 1
            : min(tensorParallelMicroBatches, streams.dim(1) / Self.minimumMicroBatchTokens)
        if parts > 1 {
            let boundary = try microBatchedForward(streams, validity: validity, working: working, parts: parts)
            return try finish(boundary: boundary, validity: validity, allValid: allValid, readout: readout,
                cache: cache, original: original, working: working, start: start, captured: false)
        }
        var next = GLM5NextTextLayer.Streams.materialized(streams)
        let early =
            (executionPolicy.stepEvaluation == .earlyAsync || earlyEvaluation)
            && streams.dim(1) <= GLM5NextExecutionPolicy.earlyEvaluationMaximumTokens
        let stride = GLM5NextExecutionPolicy.earlyEvaluationLayerStride
        for (index, (layer, state)) in zip(layers, working.layers).enumerated() {
            next = try layer.forward(next, cache: state, validity: validity, captureWindow: captureWindow)
            if early, (index + 1) % stride == 0, index + 1 < layers.count {
                asyncEval(next.arrays)
            }
        }
        return try finish(boundary: try next.materialize(), validity: validity, allValid: allValid,
            readout: readout, cache: cache, original: original, working: working, start: start,
            captured: captureWindow)
    }

    /// The chunk's tokens in `parts` consecutive near-equal micro-batches
    /// (`microBatchedForward(_:validity:working:rows:snapshotAfter:)`).
    private func microBatchedForward(
        _ streams: MLXArray, validity: MLXArray, working: GLM5NextTextCache, parts: Int
    ) throws -> MLXArray {
        let tokens = streams.dim(1)
        let bounds = (0 ... parts).map { $0 * tokens / parts }
        let rows = (0 ..< parts).map { bounds[$0 + 1] - bounds[$0] }
        return concatenated(
            try microBatchedForward(streams, validity: validity, working: working, rows: rows).pieces,
            axis: 1)
    }

    /// The tokens in consecutive micro-batches of `rows` rows, layer by layer:
    /// every micro-batch's attention partial, each sum issued (`asyncEval`) as
    /// soon as it is built, then every feed-forward partial the same way. The
    /// GPU computes one micro-batch while the CPU stream seals and exchanges
    /// another's sum. Each micro-batch reads the cache the earlier ones wrote
    /// at the same layer, as consecutive chunks would. Returns each
    /// micro-batch's boundary streams and, with `snapshotAfter`, a copy of
    /// every layer's cache as that many micro-batches left it.
    private func microBatchedForward(
        _ streams: MLXArray, validity: MLXArray, working: GLM5NextTextCache, rows: [Int],
        snapshotAfter: Int? = nil
    ) throws -> (pieces: [MLXArray], snapshot: GLM5NextTextCache?) {
        let bounds = rows.reduce(into: [0]) { $0.append($0[$0.count - 1] + $1) }
        var pieces: [GLM5NextTextLayer.Streams] = (0 ..< rows.count).map {
            .materialized(streams[0..., bounds[$0] ..< bounds[$0 + 1]])
        }
        let validities = (0 ..< rows.count).map { validity[0..., bounds[$0] ..< bounds[$0 + 1]] }
        var kept: [any KVCache] = []
        for index in layers.indices {
            try microBatchedLayer(
                index, pieces: &pieces, validities: validities, working: working, snapshotAfter: snapshotAfter,
                kept: &kept)
        }
        return (
            try pieces.map { try $0.materialize() },
            snapshotAfter == nil ? nil : GLM5NextTextCache(layers: kept)
        )
    }

    /// Layer `index` of `microBatchedForward` for every micro-batch (`pieces`,
    /// replaced by the layer's output streams): every attention partial, each
    /// sum issued (`asyncEval`) as soon as it is built, then every
    /// feed-forward partial the same way. With `snapshotAfter`, appends to
    /// `kept` a copy of the layer's cache as that many micro-batches left it.
    func microBatchedLayer(
        _ index: Int, pieces: inout [GLM5NextTextLayer.Streams], validities: [MLXArray],
        working: GLM5NextTextCache, snapshotAfter: Int?, kept: inout [any KVCache],
        selections: [GLM5NextDSAAttention.SelectionHistories?]? = nil
    ) throws {
        typealias Phase = GLM5NextTextLayer.Phase
        let layer = layers[index], state = working.layers[index]
        var attention: [(phase: Phase<Phase<GLM5NextTextLayer.Streams>>, sum: MLXArray)] = []
        if let split = selectionSplit, layer.attention is GLM5NextDSAAttention {
            // Every micro-batch's share of its selection first, each exchange
            // issued as soon as it is built; then every attention.
            var exchanges: [(exchange: GLM5NextTextLayer.Exchange, sum: MLXArray?)] = []
            for part in pieces.indices {
                // Below the history bound a micro-batch selects alone (one share).
                let shared = state.offset >= split.minimumHistory
                guard
                    let exchange = try layer.exchangingPhases(
                        pieces[part], cache: state, validity: validities[part],
                        selections: selections?[part] ?? nil, share: shared ? split.share : 0,
                        shares: shared ? split.shares : 1)
                else { throw GLM5NextLoadError.invalid("a selection split on a layer without DSA") }
                // The step appended this micro-batch's rows to the layer's cache.
                if part + 1 == snapshotAfter { kept.append(state.copy()) }
                let sum = exchange.partial.map { split.exchange($0) }
                if let sum { asyncEval(sum) }
                exchanges.append((exchange, sum))
            }
            for step in exchanges {
                let phase = try step.exchange.resume(step.sum)
                let sum = layer.summed(phase.partial)
                asyncEval(sum)
                attention.append((phase, sum))
            }
        } else {
            for part in pieces.indices {
                let phase = try layer.phases(
                    pieces[part], cache: state, validity: validities[part], selections: selections?[part] ?? nil)
                // The attention phase wrote this micro-batch's rows into the
                // layer's cache; a copy shares (never mutates) its arrays.
                if part + 1 == snapshotAfter { kept.append(state.copy()) }
                let sum = layer.summed(phase.partial)
                asyncEval(sum)
                attention.append((phase, sum))
            }
        }
        var feedForward: [(phase: Phase<GLM5NextTextLayer.Streams>, sum: MLXArray)] = []
        for step in attention {
            let phase = try step.phase.resume(step.sum)
            let sum = layer.summed(phase.partial)
            asyncEval(sum)
            feedForward.append((phase, sum))
        }
        for (part, step) in feedForward.enumerated() { pieces[part] = try step.phase.resume(step.sum) }
    }

    /// Layer `index` of a segmented prefill with sequence parallelism
    /// (`installTensorParallelSequenceParallel`): micro-batches marked `split`
    /// (an even `shares * sequenceParallelMinimumRows` rows or more) hold this
    /// rank's rows of their streams (`GLM5NextTextLayer.sequenceParallelSteps`),
    /// the others all rows (`GLM5NextTextLayer.steps`). The layer runs in
    /// rounds: every micro-batch's next collectives, issued (`asyncEval`) as
    /// soon as they are built, in micro-batch order; every micro-batch builds
    /// its attention in the same round, in order, so each reads the cache the
    /// earlier ones wrote.
    func sequenceParallelLayer(
        _ index: Int, pieces: inout [GLM5NextTextLayer.Streams], validities: [MLXArray], split: [Bool],
        working: GLM5NextTextCache, snapshotAfter: Int?, kept: inout [any KVCache],
        sequenceParallel: GLM5NextTextLayer.SequenceParallel,
        selections: [GLM5NextDSAAttention.SelectionHistories?]? = nil
    ) throws {
        let layer = layers[index], state = working.layers[index], parts = pieces.count
        // Round 0 only issues each split micro-batch's first gather; every
        // attention is built in round 1.
        var steps: [GLM5NextTextLayer.Step] = try (0 ..< parts).map { part in
            let histories = selections?[part] ?? nil
            if split[part] {
                return try layer.sequenceParallelSteps(
                    pieces[part], cache: state, validity: validities[part], sequenceParallel: sequenceParallel,
                    selections: histories)
            }
            let piece = pieces[part], validity = validities[part]
            return .reduce([]) { _ in try layer.steps(piece, cache: state, validity: validity, selections: histories) }
        }
        var results: [[MLXArray]?] = Array(repeating: nil, count: parts)
        func issue(_ part: Int) throws {
            guard case .reduce(let partials, _) = steps[part] else { return }
            let values = try partials.map { try layer.collective($0.0, $0.1, sequenceParallel: sequenceParallel) }
            if !values.isEmpty { asyncEval(values) }
            results[part] = values
        }
        for part in 0 ..< parts { try issue(part) }
        var round = 1
        while results.contains(where: { $0 != nil }) {
            for part in 0 ..< parts {
                guard let values = results[part], case .reduce(_, let resume) = steps[part] else { continue }
                steps[part] = try resume(values)
                results[part] = nil
                // The attention phase wrote this micro-batch's rows into the
                // layer's cache; a copy shares (never mutates) its arrays.
                if round == 1, part + 1 == snapshotAfter { kept.append(state.copy()) }
                try issue(part)
            }
            round += 1
        }
        for part in 0 ..< parts {
            guard case .done(let next) = steps[part] else {
                throw GLM5NextLoadError.invalid("sequence-parallel layer did not finish")
            }
            pieces[part] = next
        }
    }

    /// Several consecutive prefill chunks (`segments`) in one graph on a
    /// whole-model stage (a single device or a tensor-parallel rank): every
    /// segment's micro-batches run layer by layer as one chunk's do
    /// (`microBatchedForward`), so the sums of one segment overlap another's
    /// compute and only the call's own two edges wait for the link. Each
    /// micro-batch reads the cache the earlier ones wrote at the same layer,
    /// with the same rows, positions and shapes as in a call of its own, and
    /// each segment's readout (stream mean, final norm, head) runs on that
    /// segment's rows alone. So the outputs, the cache and every segment's
    /// hidden rows (`GLM5NextTextStageCandidate.segmentHidden`) are those of
    /// one `prepareForward(tokens:cache:validity:readout:)` per segment whose
    /// chunk runs as the segment's micro-batches
    /// (`GLM5NextPrefillSegment.chunk`): only when the GPU runs them changes.
    /// The output's hidden rows are the segments' concatenated; `.lastToken`
    /// reads out the last row, `.allTokens` every segment's rows.
    /// `snapshotAfter` (1 up to the segment count less one): the candidate
    /// also carries a copy of the stage cache as it stands after that many
    /// segments (`GLM5NextTextStageCandidate.snapshot`), such as a
    /// prompt-prefix boundary inside the call.
    public func prepareForward(
        tokens: MLXArray, cache: GLM5NextTextStageCache, segments: [GLM5NextPrefillSegment],
        readout: GLM5NextTextStageReadout = .none, snapshotAfter: Int? = nil
    ) throws -> GLM5NextTextStageCandidate {
        try Self.whole(
            beginForward(
                tokens: tokens, cache: cache, segments: segments, readout: readout, snapshotAfter: snapshotAfter))
    }

    /// `prepareForward(tokens:cache:segments:readout:snapshotAfter:)` in layer
    /// slices (`GLM5NextSlicedPrefill`), so the caller may evaluate each
    /// slice and run other forwards, on other caches, between slices.
    public func beginForward(
        tokens: MLXArray, cache: GLM5NextTextStageCache, segments: [GLM5NextPrefillSegment],
        readout: GLM5NextTextStageReadout = .none, snapshotAfter: Int? = nil
    ) throws -> GLM5NextSlicedPrefill {
        let c = plan.configuration
        guard plan.isFirst, plan.isFinal, let embedding, norm != nil, head != nil, tokens.ndim == 2,
            tokens.dim(0) > 0, tokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(tokens.dtype),
            !segments.isEmpty,
            segments.allSatisfy({ !$0.microBatchRows.isEmpty && $0.microBatchRows.allSatisfy { $0 > 0 } }),
            segments.reduce(0, { $0 + $1.tokens }) == tokens.dim(1),
            snapshotAfter.map({ (1 ..< segments.count).contains($0) }) ?? true
        else {
            throw GLM5NextLoadError.invalid(
                "a segmented prefill needs a whole-model stage, [B,T] tokens and segments of T rows")
        }
        let streams =
            embeddedStreams(tokens)
            ?? {
                let embeddings = embedding(tokens)
                return broadcast(
                    embeddings.expandedDimensions(axis: 2),
                    to: [embeddings.dim(0), embeddings.dim(1), c.hcMult, c.hiddenSize])
            }()
        return try beginSegmented(
            streams: streams, cache: cache, segments: segments, readout: readout, snapshotAfter: snapshotAfter)
    }

    /// `prepareForward(tokens:cache:segments:readout:snapshotAfter:)` from
    /// input embeddings `[B, T, hidden]` (a multimodal prompt: token
    /// embeddings with media rows replaced by their features, cast to the
    /// embedding's dtype), as `prepareForward(embeddings:cache:validity:readout:)`
    /// is `prepareForward(tokens:)`'s. The embeddings become the same
    /// residual streams the token path builds from `embedding(tokens)`, and
    /// every segment and micro-batch then runs exactly as there, so merging
    /// segments stays bitwise for every row, media rows included.
    public func prepareForward(
        embeddings: MLXArray, cache: GLM5NextTextStageCache, segments: [GLM5NextPrefillSegment],
        readout: GLM5NextTextStageReadout = .none, snapshotAfter: Int? = nil
    ) throws -> GLM5NextTextStageCandidate {
        try Self.whole(
            beginForward(
                embeddings: embeddings, cache: cache, segments: segments, readout: readout,
                snapshotAfter: snapshotAfter))
    }

    /// `prepareForward(embeddings:cache:segments:readout:snapshotAfter:)` in
    /// layer slices (`GLM5NextSlicedPrefill`).
    public func beginForward(
        embeddings: MLXArray, cache: GLM5NextTextStageCache, segments: [GLM5NextPrefillSegment],
        readout: GLM5NextTextStageReadout = .none, snapshotAfter: Int? = nil
    ) throws -> GLM5NextSlicedPrefill {
        let c = plan.configuration
        guard plan.isFirst, plan.isFinal, norm != nil, head != nil, embeddings.ndim == 3,
            embeddings.dim(0) > 0, embeddings.dim(1) > 0, embeddings.dim(2) == c.hiddenSize,
            [.float32, .bfloat16, .float16].contains(embeddings.dtype),
            !segments.isEmpty,
            segments.allSatisfy({ !$0.microBatchRows.isEmpty && $0.microBatchRows.allSatisfy { $0 > 0 } }),
            segments.reduce(0, { $0 + $1.tokens }) == embeddings.dim(1),
            snapshotAfter.map({ (1 ..< segments.count).contains($0) }) ?? true
        else {
            throw GLM5NextLoadError.invalid(
                "a segmented prefill needs a whole-model stage, [B,T,hidden] embeddings and segments of T rows")
        }
        let streams = broadcast(
            embeddings.expandedDimensions(axis: 2),
            to: [embeddings.dim(0), embeddings.dim(1), c.hcMult, c.hiddenSize])
        return try beginSegmented(
            streams: streams, cache: cache, segments: segments, readout: readout, snapshotAfter: snapshotAfter)
    }

    /// A sliced prefill run whole: every layer, then the readout.
    private static func whole(_ prefill: GLM5NextSlicedPrefill) throws -> GLM5NextTextStageCandidate {
        try prefill.advance(layers: prefill.layerCount)
        return try prefill.finish()
    }

    /// The segmented prefill from its residual streams `[B, T, hcMult, hidden]`.
    private func beginSegmented(
        streams: MLXArray, cache: GLM5NextTextStageCache, segments: [GLM5NextPrefillSegment],
        readout: GLM5NextTextStageReadout, snapshotAfter: Int?
    ) throws -> GLM5NextSlicedPrefill {
        let c = plan.configuration
        guard cache.binding === binding,
            cache.layers.count == plan.layers.count,
            zip(plan.layers, cache.layers).allSatisfy({ index, state in
                state.offset == cache.offset
                    && (c.layerTypes[index] == .linearAttention
                        ? state is GLM5NextKDACache : state is GLM5NextDSACache)
            }), streams.dim(1) <= c.maxPositionEmbeddings - cache.offset
        else { throw GLM5NextLoadError.invalid("stage boundary geometry, cache or context") }
        let validity = MLXArray.ones(Array(streams.shape.prefix(2)), dtype: .bool)
        let rows = segments.flatMap(\.microBatchRows)
        let bounds = rows.reduce(into: [0]) { $0.append($0[$0.count - 1] + $1) }
        let selections = try Self.selectionHistories(segments)
        // With sequence parallelism, micro-batches of an even `shares *
        // sequenceParallelMinimumRows` rows or more keep this rank's rows.
        let split = sequenceParallel.map { sp in
            rows.map { $0 % sp.shares == 0 && $0 >= sp.shares * Self.sequenceParallelMinimumRows }
        }
        return GLM5NextSlicedPrefill(
            stage: self, cache: cache, segments: segments, readout: readout,
            snapshotMicroBatches: snapshotAfter.map { segments[..<$0].reduce(0) { $0 + $1.microBatchRows.count } },
            pieces: (0 ..< rows.count).map { part in
                let own = split?[part] == true ? sequenceParallel!.rows(rows[part]) : 0 ..< rows[part]
                return .materialized(streams[0..., (bounds[part] + own.lowerBound) ..< (bounds[part] + own.upperBound)])
            },
            validities: (0 ..< rows.count).map { validity[0..., bounds[$0] ..< bounds[$0 + 1]] },
            sequenceParallel: sequenceParallel, split: split, selections: selections)
    }

    /// Per micro-batch of `segments`, its rows' DSA selection histories
    /// relative to the micro-batch's first row and its cache offset: rows
    /// `[previous end, rowEnd)` select among the pools complete through key
    /// `keyEnd` (`GLM5NextPrefillSegment.selectionRows`'s micro-batch that
    /// holds them), which may lie past this micro-batch's own last row. nil
    /// where every row's selection micro-batch is its own.
    static func selectionHistories(_ segments: [GLM5NextPrefillSegment]) throws
        -> [GLM5NextDSAAttention.SelectionHistories?]
    {
        var result: [GLM5NextDSAAttention.SelectionHistories?] = []
        for segment in segments {
            let own = segment.microBatchRows.reduce(into: [0]) { $0.append($0[$0.count - 1] + $1) }
            guard let selection = segment.selectionRows else {
                result += Array(repeating: nil, count: segment.microBatchRows.count)
                continue
            }
            guard selection.allSatisfy({ $0 > 0 }), selection.reduce(0, +) == segment.tokens else {
                throw GLM5NextLoadError.invalid("a segment's selection micro-batches must cover its rows")
            }
            let ends = selection.reduce(into: [0]) { $0.append($0[$0.count - 1] + $1) }
            for part in 0 ..< segment.microBatchRows.count {
                let (first, last) = (own[part], own[part + 1])
                var pieces: [(rowEnd: Int, keyEnd: Int)] = []
                for index in 1 ..< ends.count where ends[index] > first && ends[index - 1] < last {
                    pieces.append((rowEnd: min(ends[index], last) - first, keyEnd: ends[index] - first))
                }
                let natural = pieces.count == 1 && pieces[0].keyEnd == last - first
                result.append(natural ? nil : .init(pieces: pieces))
            }
        }
        return result
    }

    /// The readout of a segmented prefill whose layers are all built: each
    /// segment's rows read out on their own (stream mean, final norm), and
    /// the logits `readout` asks for.
    fileprivate func segmentedReadout(
        _ pieces: [MLXArray], segments: [GLM5NextPrefillSegment], readout: GLM5NextTextStageReadout
    ) throws -> (output: GLM5NextTextStageOutput, hidden: [MLXArray]) {
        guard let norm, let head else { throw GLM5NextLoadError.invalid("a segmented prefill needs the final stage") }
        var hidden: [MLXArray] = []
        var first = 0
        for segment in segments {
            let parts = segment.microBatchRows.count
            let own = Array(pieces[first ..< first + parts])
            first += parts
            // Sequence-parallel micro-batches return their gathered readout
            // (`[1, rows, hidden]`), the others their streams; beside one, the
            // others read out on their own rows (a row's readout does not
            // depend on its call's rows on the GPU, the sequence-parallel
            // ranks' device).
            if own.contains(where: { $0.ndim == 3 }) {
                let rows = own.map { $0.ndim == 3 ? $0 : norm($0.mean(axis: 2)) }
                hidden.append(rows.count == 1 ? rows[0] : concatenated(rows, axis: 1))
                continue
            }
            let boundary = own.count == 1 ? own[0] : concatenated(own, axis: 1)
            hidden.append(norm(boundary.mean(axis: 2)))
        }
        let all = hidden.count == 1 ? hidden[0] : concatenated(hidden, axis: 1)
        switch readout {
        case .allTokens:
            let logits = hidden.map { head($0) }
            return (.final(.init(logits: logits.count == 1 ? logits[0] : concatenated(logits, axis: 1), hidden: all)), hidden)
        case .lastToken:
            return (.final(.init(logits: head(hidden[hidden.count - 1][0..., (-1)..., 0...]), hidden: all)), hidden)
        case .lastTokenVocabularyPart(let part, let parts):
            return (
                .final(
                    .init(
                        logits: try vocabularyPart(part, of: parts)(hidden[hidden.count - 1][0..., (-1)..., 0...]),
                        hidden: all)), hidden
            )
        case .allTokensVocabularyPart(let part, let parts):
            let logits = try hidden.map { try vocabularyPart(part, of: parts)($0) }
            return (.final(.init(logits: logits.count == 1 ? logits[0] : concatenated(logits, axis: 1), hidden: all)), hidden)
        case .none:
            return (.hidden(all), hidden)
        }
    }

    /// The head's vocabulary rows `[part * V / parts, (part + 1) * V / parts)`
    /// as a linear layer over row views of the head's arrays (no copy), built
    /// once per (part, parts). Quantized heads only; V must divide by
    /// `parts` and keep whole 8-row tiles (MLX's `qmv_fast` row grouping).
    public func vocabularyPart(_ part: Int, of parts: Int) throws -> Linear {
        guard let head else { throw GLM5NextLoadError.invalid("vocabulary part of this head") }
        return try vocabularyParts.part(part, of: parts) {
            try Self.vocabularyPart(of: head, part, of: parts)
        }
    }

    /// `head`'s rows `[part * V / parts, (part + 1) * V / parts)` as a
    /// quantized linear layer over row views of its arrays (`vocabularyPart`).
    static func vocabularyPart(of head: Linear, _ part: Int, of parts: Int) throws -> Linear {
        guard let head = head as? QuantizedLinear, !(head is GLM5NextNVFP4Linear), parts > 0,
            (0 ..< parts).contains(part),
            head.weight.dim(0) % (8 * parts) == 0
        else { throw GLM5NextLoadError.invalid("vocabulary part of this head") }
        let rows = head.weight.dim(0) / parts
        let range = part * rows ..< (part + 1) * rows
        let linear = QuantizedLinear(
            weight: head.weight[range], bias: head.bias?[range], scales: head.scales[range],
            biases: head.biases?[range], groupSize: head.groupSize, bits: head.bits,
            mode: head.mode)
        linear.freeze()
        try MLX.withError { error in
            eval(linear.parameters().flattened().map(\.1))
            try error.check()
        }
        return linear
    }

    private let vocabularyParts = VocabularyParts()

    private final class VocabularyParts: @unchecked Sendable {
        private let lock = NSLock()
        private var parts: [[Int]: Linear] = [:]
        func part(_ part: Int, of count: Int, _ make: () throws -> Linear) throws -> Linear {
            lock.lock()
            defer { lock.unlock() }
            if let linear = parts[[part, count]] { return linear }
            let linear = try make()
            parts[[part, count]] = linear
            return linear
        }
    }

    private func finish(
        boundary: MLXArray, validity: MLXArray, allValid: Bool, readout: GLM5NextTextStageReadout,
        cache: GLM5NextTextStageCache, original: GLM5NextTextCache.Revision, working: GLM5NextTextCache,
        start: Int, captured: Bool
    ) throws -> GLM5NextTextStageCandidate {
        let output: GLM5NextTextStageOutput
        if let norm, let head {
            let normalized = norm(boundary.mean(axis: 2))
            // `where` keeps the dtype, so with every row valid it returns these bits.
            let hidden =
                allValid
                ? normalized : MLX.where(validity.expandedDimensions(axis: -1), normalized, 0)
            switch readout {
            case .allTokens:
                output = .final(.init(logits: head(hidden), hidden: hidden))
            case .lastToken:
                output = .final(.init(logits: head(hidden[0..., (-1)..., 0...]), hidden: hidden))
            case .lastTokenVocabularyPart(let part, let parts):
                output = .final(
                    .init(
                        logits: try vocabularyPart(part, of: parts)(hidden[0..., (-1)..., 0...]),
                        hidden: hidden))
            case .allTokensVocabularyPart(let part, let parts):
                output = .final(.init(logits: try vocabularyPart(part, of: parts)(hidden), hidden: hidden))
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
            output: output, destination: cache.storage, original: original, working: working,
            start: start, tokens: boundary.dim(1), captured: captured)
    }
}
