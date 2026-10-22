// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0
// Source: Transformers f0705f991a71f05dad9569336b181119f5546f93, Glm5NextTextModel/DecoderLayer.
import MLX
import MLXLMCommon
import MLXNN

public struct GLM5NextTextOutput {
    public let logits: MLXArray
    /// Unweighted stream mean, then final RMSNorm. This is the NextN boundary.
    public let hidden: MLXArray
}

/// Engine-thread-confined candidate. Graph construction never changes the
/// destination. The required caller fence must evaluate every supplied root,
/// join device work, and throw on MLX faults before this owner can publish.
public final class GLM5NextTextCandidate {
    public let output: GLM5NextTextOutput
    private let destination: GLM5NextTextCache
    private let original: GLM5NextTextCache.Revision
    private var working: GLM5NextTextCache?
    private enum State: Equatable { case pending, evaluating, committed, discarded }
    private var state = State.pending

    fileprivate init(
        output: GLM5NextTextOutput, destination: GLM5NextTextCache,
        original: GLM5NextTextCache.Revision, working: GLM5NextTextCache
    ) {
        self.output = output
        self.destination = destination
        self.original = original
        self.working = working
    }

    @discardableResult
    public func commit(afterEvaluating fence: ([MLXArray]) throws -> Void) throws
        -> GLM5NextTextOutput
    {
        guard state == .pending, let working else {
            throw GLM5NextLoadError.invalid("candidate is no longer pending")
        }
        guard destination.revision() == original else {
            discard()
            throw GLM5NextLoadError.invalid("candidate destination changed before evaluation")
        }
        state = .evaluating
        do {
            try fence([output.logits, output.hidden] + working.layers.flatMap { $0.innerState() })
            guard destination.revision() == original else {
                throw GLM5NextLoadError.invalid("candidate destination changed during evaluation")
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

extension GLM5NextTextLayer.Phase where Next == GLM5NextTextLayer.Phase<GLM5NextTextLayer.Streams> {
    /// The whole layer: each partial summed by `sum`, then resumed.
    func run(_ sum: (MLXArray) -> MLXArray) throws -> GLM5NextTextLayer.Streams {
        let feedForward = try resume(sum(partial))
        return try feedForward.resume(sum(feedForward.partial))
    }
}

public final class GLM5NextTextLayer: Module {
    /// Tensor parallel: sums a rank's partial sublayer output across ranks.
    /// Nil on a single device. Set once after construction, before any forward.
    public var reduce: ((MLXArray) -> MLXArray)?
    @ModuleInfo(key: "self_attn") var attention: Module
    public var sparseAttention: GLM5NextDSAAttention? { attention as? GLM5NextDSAAttention }
    @ModuleInfo(key: "input_layernorm") var inputNorm: GLM5NextRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: GLM5NextRMSNorm
    let mlp: Module
    @ParameterInfo(key: "hc_attn_fn") var attentionFunction: MLXArray
    @ParameterInfo(key: "hc_attn_base") var attentionBase: MLXArray
    @ParameterInfo(key: "hc_attn_scale") var attentionScale: MLXArray
    @ParameterInfo(key: "hc_ffn_fn") var ffnFunction: MLXArray
    @ParameterInfo(key: "hc_ffn_base") var ffnBase: MLXArray
    @ParameterInfo(key: "hc_ffn_scale") var ffnScale: MLXArray
    let configuration: GLM5NextTextConfiguration
    /// The target's load-time policy, used by every forward of this layer.
    let executionPolicy: GLM5NextExecutionPolicy
    init(
        _ c: GLM5NextTextConfiguration, index: Int, loader: GLM5NextTensorLoader,
        materialize: ((MLXArray) throws -> Void)?, executionPolicy: GLM5NextExecutionPolicy
    ) throws {
        configuration = c
        self.executionPolicy = executionPolicy
        let p = "model.language_model.layers.\(index)"
        if c.layerTypes[index] == .linearAttention {
            _attention.wrappedValue = try GLM5NextKDASublayer(
                c, prefix: p + ".self_attn", loader: loader,
                linearAttention: executionPolicy.linearAttention,
                decodeStep: executionPolicy.linearDecodeStep,
                projections: executionPolicy.projections,
                prefillProjections: executionPolicy.prefillProjections)
        } else {
            _attention.wrappedValue = try GLM5NextDSAAttention(
                c, prefix: p + ".self_attn", loader: loader, executionPolicy: executionPolicy)
        }
        _inputNorm.wrappedValue = try loader.norm(p + ".input_layernorm", epsilon: c.rmsNormEps)
        _postNorm.wrappedValue = try loader.norm(
            p + ".post_attention_layernorm", epsilon: c.rmsNormEps)
        mlp =
            c.mlpLayerTypes[index] == .dense
            ? try loader.mlp(
                c, prefix: p + ".mlp", intermediate: c.intermediateSize,
                projections: executionPolicy.projections)
            : try loader.moe(
                c, prefix: p + ".mlp", materialize: materialize,
                projections: executionPolicy.projections)
        _attentionFunction.wrappedValue = try loader.array(p + ".hc_attn_fn")
        _attentionBase.wrappedValue = try loader.array(p + ".hc_attn_base")
        _attentionScale.wrappedValue = try loader.array(p + ".hc_attn_scale")
        _ffnFunction.wrappedValue = try loader.array(p + ".hc_ffn_fn")
        _ffnBase.wrappedValue = try loader.array(p + ".hc_ffn_base")
        _ffnScale.wrappedValue = try loader.array(p + ".hc_ffn_scale")
    }
    func forward(
        _ streams: MLXArray, cache: any KVCache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference
    ) throws -> MLXArray {
        try forward(
            .materialized(streams), cache: cache, validity: validity,
            recurrenceExecution: recurrenceExecution
        ).materialize()
    }
    /// `captureWindow`: a KDA layer keeps every token's state in its cache
    /// (`GLM5NextKDASublayer.forward(_:cache:validity:recurrenceExecution:captureWindow:)`).
    func forward(
        _ streams: Streams, cache: any KVCache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference,
        captureWindow: Bool = false
    ) throws -> Streams {
        try phases(
            streams, cache: cache, validity: validity, recurrenceExecution: recurrenceExecution,
            captureWindow: captureWindow
        ).run(summed)
    }

    /// `forward` stopped at its two cross-rank sums (see `Phase`).
    func phases(
        _ streams: Streams, cache: any KVCache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference,
        captureWindow: Bool = false, selections: GLM5NextDSAAttention.SelectionHistories? = nil
    ) throws -> Phase<Phase<Streams>> {
        try phases(streams) { normalized in
            if let attention = attention as? GLM5NextKDASublayer,
                let cache = cache as? GLM5NextKDACache
            {
                return try attention.forward(
                    normalized, cache: cache, validity: validity,
                    recurrenceExecution: recurrenceExecution, captureWindow: captureWindow)
            }
            if let attention = attention as? GLM5NextDSAAttention,
                let cache = cache as? GLM5NextDSACache
            {
                return try attention.forward(normalized, cache: cache, validity: validity, selections: selections)
            }
            throw GLM5NextLoadError.invalid("attention/cache kind mismatch")
        }
    }

    /// `phases` over rows of independent requests `[B,T,...]`, row b
    /// continuing `caches[b]`; everything but attention is row-wise.
    func phases(
        rows streams: Streams, caches: [any KVCache], validity: MLXArray, captureWindow: Bool
    ) throws -> Phase<Phase<Streams>> {
        try phases(streams) { normalized in
            try attendRows(
                normalized, caches: caches, validity: validity, captureWindow: captureWindow)
        }
    }

    /// Attention of rows of independent requests `[B,T,hidden]`, row b over
    /// `caches[b]` (`GLM5NextKDASublayer.forwardRows`,
    /// `GLM5NextDSAAttention.forwardRows`).
    func attendRows(
        _ normalized: MLXArray, caches: [any KVCache], validity: MLXArray, captureWindow: Bool
    ) throws -> MLXArray {
        if let attention = attention as? GLM5NextKDASublayer {
            let states = caches.compactMap { $0 as? GLM5NextKDACache }
            guard states.count == caches.count else {
                throw GLM5NextLoadError.invalid("attention/cache kind mismatch")
            }
            return try attention.forwardRows(
                normalized, caches: states, validity: validity, captureWindow: captureWindow)
        }
        if let attention = attention as? GLM5NextDSAAttention {
            let histories = caches.compactMap { $0 as? GLM5NextDSACache }
            guard histories.count == caches.count else {
                throw GLM5NextLoadError.invalid("attention/cache kind mismatch")
            }
            return try attention.forwardRows(normalized, caches: histories, validity: validity)
        }
        throw GLM5NextLoadError.invalid("attention/cache kind mismatch")
    }

    /// One layer of a multi-request step under `.nativeNormFused`, over the
    /// rows of every lane `[1, M, ...]` laid out group after group: each mHC
    /// site takes every row in one launch on `workspace` (a row's arithmetic
    /// is its own at any row count), attention and a dense MLP run per row
    /// group (`groups`: the group's rows and its `[B,T]` shape), the routed
    /// MoE in decode-sized chunks (per-token arithmetic), and each of the two
    /// sums covers every row.
    func rowsForward(
        _ streams: Streams, groups: [(rows: Range<Int>, shape: [Int])], caches: [[any KVCache]],
        validities: [MLXArray], captureWindow: Bool,
        workspace: GLM5NextHyperConnectionKernel.SiteWorkspace
    ) throws -> Streams {
        let c = configuration
        func collapse(
            _ source: GLM5NextHyperConnectionKernel.StreamsSource, fn: MLXArray, base: MLXArray,
            scale: MLXArray, norm: GLM5NextRMSNorm
        ) throws -> GLM5NextHyperConnectionKernel.NormalizedCollapse {
            try GLM5NextHyperConnectionKernel.collapseNormalized(
                source, fn: fn, base: base, scale: scale, rmsNormEpsilon: Float(c.rmsNormEps),
                epsilon: Float(c.hcEps), sinkhornIterations: c.hcSinkhornIters,
                normWeight: norm.weight, normEpsilon: norm.epsilon, workspace: workspace)
        }
        func joined(_ parts: [MLXArray]) -> MLXArray {
            parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        }
        let source: GLM5NextHyperConnectionKernel.StreamsSource
        switch streams {
        case .materialized(let streams): source = .streams(streams)
        case .pendingExpand(let output, let residual, let post, let comb):
            source = .expand(output: output, residual: residual, post: post, comb: comb)
        }
        let width = c.hiddenSize
        let attn = try collapse(
            source, fn: attentionFunction, base: attentionBase, scale: attentionScale,
            norm: inputNorm)
        let attended = try groups.indices.map { g in
            try attendRows(
                attn.normalized[0..., groups[g].rows].reshaped(groups[g].shape + [width]),
                caches: caches[g], validity: validities[g], captureWindow: captureWindow
            ).reshaped(1, -1, width)
        }
        let ffn = try collapse(
            .expand(
                output: summed(joined(attended)), residual: attn.streams, post: attn.post,
                comb: attn.comb),
            fn: ffnFunction, base: ffnBase, scale: ffnScale, norm: postNorm)
        let input = ffn.normalized
        let output: MLXArray
        if let moe = mlp as? GLM5NextMoE {
            output = joined(
                try Self.expertChunks(groups).map {
                    try moe(input[0..., $0], routedExperts: executionPolicy.routedExperts)
                })
        } else {
            output = joined(try groups.map { try feedForward(input[0..., $0.rows]) })
        }
        return .pendingExpand(
            output: summed(output), residual: ffn.streams, post: ffn.post, comb: ffn.comb)
    }

    /// The routed-expert launches of `rowsForward`: a one-token lane's row
    /// alone (a one-token step's launch), every run of consecutive
    /// multi-token groups in balanced chunks of 2 through
    /// `GLM5NextMoEDecodeKernel.maximumTokens` rows (multi-token launches).
    static func expertChunks(_ groups: [(rows: Range<Int>, shape: [Int])]) -> [Range<Int>] {
        var chunks: [Range<Int>] = []
        var run: Range<Int>?
        func flush() {
            guard let rows = run else { return }
            let pieces = (rows.count + GLM5NextMoEDecodeKernel.maximumTokens - 1)
                / GLM5NextMoEDecodeKernel.maximumTokens
            let bounds = (0 ... pieces).map { rows.lowerBound + $0 * rows.count / pieces }
            chunks += (0 ..< pieces).map { bounds[$0] ..< bounds[$0 + 1] }
            run = nil
        }
        for group in groups {
            if group.rows.count == 1 {
                flush()
                chunks.append(group.rows)
            } else {
                run = (run?.lowerBound ?? group.rows.lowerBound) ..< group.rows.upperBound
            }
        }
        flush()
        return chunks
    }

    /// Residual streams between layers: materialized, or (under
    /// `HyperConnection.nativeNormFused`) the previous layer's final expand,
    /// which the next layer computes inside its first collapse.
    enum Streams {
        case materialized(MLXArray)
        case pendingExpand(output: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray)

        var arrays: [MLXArray] {
            switch self {
            case .materialized(let streams): [streams]
            case .pendingExpand(let output, let residual, let post, let comb):
                [output, residual, post, comb]
            }
        }

        func materialize() throws -> MLXArray {
            switch self {
            case .materialized(let streams): return streams
            case .pendingExpand(let output, let residual, let post, let comb):
                return try GLM5NextHyperConnectionKernel.expand(
                    output, residual: residual, post: post, comb: comb)
            }
        }
    }

    /// Shared mHC/MLP path; the caller supplies its request-owned attention state.
    func forward(_ streams: MLXArray, attend: (MLXArray) throws -> MLXArray) throws -> MLXArray {
        try forward(.materialized(streams), attend: attend).materialize()
    }

    /// One step of a layer up to a cross-rank sum: the rank's partial
    /// sublayer output and the rest of the layer given the summed output.
    /// A layer is `phases` (attention partial), then the feed-forward partial,
    /// then the next streams; `run` with `reduce` is `forward`. A caller can
    /// sum one micro-batch's partial while another's computes.
    struct Phase<Next> {
        let partial: MLXArray
        let resume: (MLXArray) throws -> Next
    }

    /// `forward`, leaving a `nativeNormFused` layer's final expand pending so a
    /// following layer folds it into its attention collapse.
    func forward(_ streams: Streams, attend: (MLXArray) throws -> MLXArray) throws -> Streams {
        try phases(streams, attend: attend).run(summed)
    }

    func phases(_ streams: Streams, attend: (MLXArray) throws -> MLXArray) throws -> Phase<Phase<Streams>> {
        if executionPolicy.hyperConnection == .nativeNormFused {
            return try normFusedPhases(streams, attend: attend)
        }
        let streams = try streams.materialize()
        let attn = try collapse(
            streams, fn: attentionFunction, base: attentionBase, scale: attentionScale)
        let normalized = inputNorm(attn.collapsed)
        return Phase(partial: try attend(normalized)) { attentionOutput in
            let residual = try self.expand(
                attentionOutput, residual: streams, post: attn.post, comb: attn.comb)
            let ffn = try self.collapse(residual, fn: self.ffnFunction, base: self.ffnBase, scale: self.ffnScale)
            let input = self.postNorm(ffn.collapsed)
            return Phase(partial: try self.feedForward(input)) { output in
                .materialized(try self.expand(output, residual: residual, post: ffn.post, comb: ffn.comb))
            }
        }
    }

    private func normFusedPhases(
        _ streams: Streams, attend: (MLXArray) throws -> MLXArray
    ) throws -> Phase<Phase<Streams>> {
        let c = configuration
        func collapse(
            _ source: GLM5NextHyperConnectionKernel.StreamsSource, fn: MLXArray, base: MLXArray,
            scale: MLXArray, norm: GLM5NextRMSNorm
        ) throws -> GLM5NextHyperConnectionKernel.NormalizedCollapse {
            try GLM5NextHyperConnectionKernel.collapseNormalized(
                source, fn: fn, base: base, scale: scale, rmsNormEpsilon: Float(c.rmsNormEps),
                epsilon: Float(c.hcEps), sinkhornIterations: c.hcSinkhornIters,
                normWeight: norm.weight, normEpsilon: norm.epsilon)
        }
        let source: GLM5NextHyperConnectionKernel.StreamsSource
        switch streams {
        case .materialized(let streams): source = .streams(streams)
        case .pendingExpand(let output, let residual, let post, let comb):
            source = .expand(output: output, residual: residual, post: post, comb: comb)
        }
        let attn = try collapse(
            source, fn: attentionFunction, base: attentionBase, scale: attentionScale,
            norm: inputNorm)
        return Phase(partial: try attend(attn.normalized)) { attentionOutput in
            let ffn = try collapse(
                .expand(
                    output: attentionOutput, residual: attn.streams, post: attn.post, comb: attn.comb),
                fn: self.ffnFunction, base: self.ffnBase, scale: self.ffnScale, norm: self.postNorm)
            return Phase(partial: try self.feedForward(ffn.normalized)) { output in
                .pendingExpand(output: output, residual: ffn.streams, post: ffn.post, comb: ffn.comb)
            }
        }
    }

    /// A layer as a chain of cross-rank collectives: partials to reduce
    /// (each with its collective) and the rest of the layer given the
    /// results, until the next streams. `phases` is two sums; a
    /// sequence-parallel layer gathers and reduce-scatters instead.
    indirect enum Step {
        case reduce([(Collective, MLXArray)], ([MLXArray]) throws -> Step)
        case done(Streams)
    }

    enum Collective {
        /// `reduce`: the cross-rank sum.
        case sum
        /// `sequenceParallel.reduceScatter`: this rank's rows of the sum.
        case scatterSum
        /// `sequenceParallel.allGather`: every rank's rows, in rank order.
        case gather
    }

    /// Tensor-parallel prefill with the residual streams divided by rows
    /// (`GLM5NextTextStage.sequenceParallel`): this rank's part of a
    /// micro-batch's rows, and the collectives that join the parts.
    struct SequenceParallel {
        /// This rank (`share`) of `shares`.
        let share: Int
        let shares: Int
        let reduceScatter: (MLXArray) -> MLXArray
        let allGather: (MLXArray) -> MLXArray

        /// The rows this rank owns of a micro-batch of `count` rows.
        func rows(_ count: Int) -> Range<Int> {
            (share * count / shares) ..< ((share + 1) * count / shares)
        }
    }

    func collective(_ kind: Collective, _ partial: MLXArray, sequenceParallel: SequenceParallel?) throws -> MLXArray {
        switch kind {
        case .sum: return summed(partial)
        case .scatterSum, .gather:
            guard let sequenceParallel else {
                throw GLM5NextLoadError.invalid("a sequence-parallel collective without sequence parallelism")
            }
            return kind == .gather ? sequenceParallel.allGather(partial) : sequenceParallel.reduceScatter(partial)
        }
    }

    /// `phases` as steps (two sums).
    func steps(_ streams: Streams, cache: any KVCache, validity: MLXArray) throws -> Step {
        let phase = try phases(streams, cache: cache, validity: validity)
        return .reduce([(.sum, phase.partial)]) { attention in
            let feedForward = try phase.resume(attention[0])
            return .reduce([(.sum, feedForward.partial)]) { output in .done(try feedForward.resume(output[0])) }
        }
    }

    /// One layer of a sequence-parallel micro-batch (`HyperConnection.nativeNormFused`):
    /// `streams` holds this rank's rows only. The mHC sites and a MoE layer's
    /// router run on those rows; the attention and feed-forward inputs are
    /// gathered to every row (the attention and the experts run on all of the
    /// micro-batch's rows, `validity`, as without), and each partial is
    /// reduce-scattered back to this rank's rows. Every row's arithmetic is
    /// the plain layer's: an mHC site and the router compute a row the same
    /// in a call of any size from `GLM5NextTextStage.sequenceParallelMinimumRows`
    /// rows (the router's FP32 GEMM keeps one order from about 1,400 rows; the
    /// mHC kernel's tree never depends on the rows), the reduce-scatter adds
    /// each element in rank order as the sum does, and gathers are copies.
    func sequenceParallelSteps(
        _ streams: Streams, cache: any KVCache, validity: MLXArray, sequenceParallel: SequenceParallel
    ) throws -> Step {
        guard executionPolicy.hyperConnection == .nativeNormFused, validity.dim(0) == 1 else {
            throw GLM5NextLoadError.invalid("sequence-parallel layers run the fused mHC sites on one batch row")
        }
        let c = configuration
        func collapse(
            _ source: GLM5NextHyperConnectionKernel.StreamsSource, fn: MLXArray, base: MLXArray,
            scale: MLXArray, norm: GLM5NextRMSNorm
        ) throws -> GLM5NextHyperConnectionKernel.NormalizedCollapse {
            try GLM5NextHyperConnectionKernel.collapseNormalized(
                source, fn: fn, base: base, scale: scale, rmsNormEpsilon: Float(c.rmsNormEps),
                epsilon: Float(c.hcEps), sinkhornIterations: c.hcSinkhornIters,
                normWeight: norm.weight, normEpsilon: norm.epsilon)
        }
        let source: GLM5NextHyperConnectionKernel.StreamsSource
        switch streams {
        case .materialized(let streams): source = .streams(streams)
        case .pendingExpand(let output, let residual, let post, let comb):
            source = .expand(output: output, residual: residual, post: post, comb: comb)
        }
        let attn = try collapse(
            source, fn: attentionFunction, base: attentionBase, scale: attentionScale,
            norm: inputNorm)
        let rows = validity.dim(1)
        func feedForwardSteps(_ attentionOutput: MLXArray) throws -> Step {
            let ffn = try collapse(
                .expand(
                    output: attentionOutput, residual: attn.streams, post: attn.post, comb: attn.comb),
                fn: ffnFunction, base: ffnBase, scale: ffnScale, norm: postNorm)
            func finish(_ partial: MLXArray) -> Step {
                .reduce([(.scatterSum, partial)]) { output in
                    .done(.pendingExpand(output: output[0], residual: ffn.streams, post: ffn.post, comb: ffn.comb))
                }
            }
            if let moe = mlp as? GLM5NextMoE, executionPolicy.routedExperts != .reference {
                // This rank's rows routed, gathered beside the expert input.
                let routing = try moe.routingFrame(ffn.normalized)
                return .reduce([(.gather, ffn.normalized), (.gather, routing)]) { gathered in
                    finish(
                        try moe(
                            gathered[0], routedExperts: self.executionPolicy.routedExperts,
                            gatheredRouting: gathered[1].reshaped(rows, -1)))
                }
            }
            return .reduce([(.gather, ffn.normalized)]) { gathered in finish(try self.feedForward(gathered[0])) }
        }
        return .reduce([(.gather, attn.normalized)]) { gathered in
            let normalized = gathered[0]
            let partial: MLXArray
            if let attention = self.attention as? GLM5NextKDASublayer, let cache = cache as? GLM5NextKDACache {
                partial = try attention.forward(normalized, cache: cache, validity: validity)
            } else if let attention = self.attention as? GLM5NextDSAAttention,
                let cache = cache as? GLM5NextDSACache
            {
                partial = try attention.forward(normalized, cache: cache, validity: validity)
            } else {
                throw GLM5NextLoadError.invalid("attention/cache kind mismatch")
            }
            return .reduce([(.scatterSum, partial)]) { try feedForwardSteps($0[0]) }
        }
    }

    /// The cross-rank sum of a partial (the partial itself on one device).
    /// (`Phase<Phase<Streams>>.run` applies it at both sums.)
    func summed(_ partial: MLXArray) -> MLXArray { reduce?(partial) ?? partial }

    private func feedForward(_ input: MLXArray) throws -> MLXArray {
        if let dense = mlp as? GLM5NextMLP {
            return try dense(input, routedExperts: executionPolicy.routedExperts)
        }
        if let moe = mlp as? GLM5NextMoE {
            return try moe(input, routedExperts: executionPolicy.routedExperts)
        }
        throw GLM5NextLoadError.invalid("unsupported GLM feed-forward module")
    }

    private func collapse(_ streams: MLXArray, fn: MLXArray, base: MLXArray, scale: MLXArray)
        throws -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray)
    {
        let c = configuration
        switch executionPolicy.hyperConnection {
        case .reference:
            return GLM5NextHyperConnection.collapse(
                streams, fn: fn, base: base, scale: scale, rmsNormEpsilon: Float(c.rmsNormEps),
                epsilon: Float(c.hcEps), sinkhornIterations: c.hcSinkhornIters)
        case .nativeRequired, .nativeNormFused:
            return try GLM5NextHyperConnectionKernel.collapse(
                streams, fn: fn, base: base, scale: scale, rmsNormEpsilon: Float(c.rmsNormEps),
                epsilon: Float(c.hcEps), sinkhornIterations: c.hcSinkhornIters)
        }
    }

    private func expand(_ output: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray)
        throws -> MLXArray
    {
        switch executionPolicy.hyperConnection {
        case .reference:
            return GLM5NextHyperConnection.expand(
                output, residual: residual, post: post, comb: comb)
        case .nativeRequired, .nativeNormFused:
            return try GLM5NextHyperConnectionKernel.expand(
                output, residual: residual, post: post, comb: comb)
        }
    }
}

public final class GLM5NextTextBackbone: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    public let layers: [GLM5NextTextLayer]
    public let norm: GLM5NextRMSNorm
    let configuration: GLM5NextTextConfiguration
    init(
        _ c: GLM5NextTextConfiguration, loader: GLM5NextTensorLoader,
        materialize: ((MLXArray) throws -> Void)?, executionPolicy: GLM5NextExecutionPolicy
    ) throws {
        configuration = c
        _embedTokens.wrappedValue = try loader.embedding("model.language_model.embed_tokens")
        layers = try (0 ..< c.hiddenLayers).map {
            try GLM5NextTextLayer(
                c, index: $0, loader: loader, materialize: materialize,
                executionPolicy: executionPolicy)
        }
        norm = try loader.norm("model.language_model.norm", epsilon: c.rmsNormEps)
    }
    func forward(
        _ embeddings: MLXArray, cache: GLM5NextTextCache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference
    ) throws
        -> MLXArray
    {
        var streams = GLM5NextTextLayer.Streams.materialized(
            broadcast(
                embeddings.expandedDimensions(axis: 2),
                to: [
                    embeddings.dim(0), embeddings.dim(1), configuration.hcMult,
                    configuration.hiddenSize,
                ]))
        let stride = GLM5NextExecutionPolicy.earlyEvaluationLayerStride
        let early =
            layers.first?.executionPolicy.stepEvaluation == .earlyAsync
            && embeddings.dim(1) <= GLM5NextExecutionPolicy.earlyEvaluationMaximumTokens
        for (index, (layer, state)) in zip(layers, cache.layers).enumerated() {
            streams = try layer.forward(
                streams, cache: state, validity: validity, recurrenceExecution: recurrenceExecution)
            if early, (index + 1) % stride == 0, index + 1 < layers.count {
                asyncEval(streams.arrays)
            }
        }
        return norm(try streams.materialize().mean(axis: 2))
    }
}

/// Architectural target with a separate conservative native CBv2 adapter.
/// Not registered; paged/MTP/prefix capabilities require their own integration.
/// The validated inventory keeps MTP and vision separate; this owner loads only
/// the text partition. The caller must retain and load the remaining partitions.
public final class GLM5NextTextModel: Module, LLMModel {
    public let configuration: GLM5NextTextConfiguration
    public let inventory: GLM5NextWeightInventory
    public let model: GLM5NextTextBackbone
    @ModuleInfo(key: "lm_head") public var lmHead: Linear
    public var vocabularySize: Int { configuration.vocabularySize }
    public var loraLayers: [Module] { model.layers }
    /// Fixed at construction for every layer and entry point; the MTP layer
    /// must match it. No forward accepts a per-call override.
    public let executionPolicy: GLM5NextExecutionPolicy

    /// Resolver must be bounded by shard/projection and retain no duplicate whole
    /// model payload. For Q4, materializeExpertStack MUST evaluate/check each
    /// packed stack before returning so its unstacked inputs can be released.
    /// This callback belongs under the caller's MLX error/fault boundary.
    /// releaseSourceTensors must evict the named lookup aliases (and unused shard
    /// dictionary aliases), without invalidating independently owning MLXArrays.
    /// No `executionPolicy` adds resident bytes: absorbed attention uses views
    /// of kv_b and every native kernel reads the loaded weights in place.
    public init(
        configuration: GLM5NextTextConfiguration, inventory: GLM5NextWeightInventory,
        tensor: @escaping (String) throws -> MLXArray,
        materializeExpertStack: ((MLXArray) throws -> Void)? = nil,
        releaseSourceTensors: (([String]) -> Void)? = nil,
        executionPolicy: GLM5NextExecutionPolicy = .reference,
        materializeFusedProjections: (([MLXArray]) throws -> Void)? = nil
    ) throws {
        guard configuration == inventory.configuration.textConfig else {
            throw GLM5NextLoadError.invalid("target configuration/inventory mismatch")
        }
        if inventory.storage.isPacked
            && (materializeExpertStack == nil || releaseSourceTensors == nil)
        {
            throw GLM5NextLoadError.invalid(
                "Q4 target construction requires bounded stack materialization and source-cache eviction"
            )
        }
        self.configuration = configuration
        self.inventory = inventory
        self.executionPolicy = executionPolicy
        let loader = GLM5NextTensorLoader(
            inventory, tensor, releaseSourceTensors: releaseSourceTensors,
            materializeFusedProjections: materializeFusedProjections)
        model = try GLM5NextTextBackbone(
            configuration, loader: loader, materialize: materializeExpertStack,
            executionPolicy: executionPolicy)
        _lmHead.wrappedValue = try loader.linear("lm_head")
        try loader.finishText()
        if executionPolicy.routing != .reference,
            configuration.topkGroup != configuration.nGroup
        {
            throw GLM5NextLoadError.invalid("native routing requires ungrouped experts")
        }
        for module in model.modules() {
            if executionPolicy.normalization == .fusedKernel {
                (module as? GLM5NextRMSNorm)?.usesFusedKernel = true
            }
            if executionPolicy.routing != .reference, let router = module as? GLM5NextRouter {
                router.usesNativeKernel = true
                router.usesPrefillKernel = executionPolicy.routing == .nativeKernelPrefill
            }
        }
    }
    public func embed(_ tokens: MLXArray) -> MLXArray { model.embedTokens(tokens) }
    public func newTextCache() -> GLM5NextTextCache { .init(configuration: configuration) }
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] { newTextCache().layers }
    public func prepareForward(
        tokens: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil
    )
        throws -> GLM5NextTextCandidate
    {
        try prepareForward(
            tokens: tokens, cache: cache, validity: validity, recurrenceExecution: .reference)
    }
    /// Internal per-token qualification seam of the reference `linearAttention`
    /// policy; a load-time KDA choice refuses a non-reference override.
    func prepareForward(
        tokens: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextCandidate {
        guard tokens.ndim == 2, [.int32, .uint32, .int64].contains(tokens.dtype) else {
            throw GLM5NextLoadError.invalid("tokens must be integer [B,T]")
        }
        return try prepareForward(
            embeddings: embed(tokens), cache: cache, validity: validity,
            recurrenceExecution: recurrenceExecution)
    }
    public func prepareForward(
        embeddings: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil
    )
        throws -> GLM5NextTextCandidate
    {
        try prepareForward(
            embeddings: embeddings, cache: cache, validity: validity,
            recurrenceExecution: .reference)
    }
    func prepareForward(
        embeddings: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextCandidate {
        try cache.validate(configuration)
        let original = cache.revision()
        let working = cache.copy()
        let output = try forward(
            embeddings: embeddings, cache: working, validity: validity,
            recurrenceExecution: recurrenceExecution)
        guard cache.revision() == original else {
            throw GLM5NextLoadError.invalid("cache changed while preparing candidate")
        }
        return GLM5NextTextCandidate(
            output: output, destination: cache, original: original, working: working)
    }
    /// Unfenced graph/cache compatibility API for ordinary LanguageModel use.
    /// CBv2 and speculative callers MUST use prepareForward + candidate.commit,
    /// not pass their active cache here. No device-success claim is made here.
    public func forward(tokens: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil)
        throws -> GLM5NextTextOutput
    {
        try forward(
            tokens: tokens, cache: cache, validity: validity, recurrenceExecution: .reference)
    }
    func forward(
        tokens: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextOutput {
        guard tokens.ndim == 2, [.int32, .uint32, .int64].contains(tokens.dtype) else {
            throw GLM5NextLoadError.invalid("tokens must be integer [B,T]")
        }
        return try forward(
            embeddings: embed(tokens), cache: cache, validity: validity,
            recurrenceExecution: recurrenceExecution)
    }
    public func forward(embeddings: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil)
        throws -> GLM5NextTextOutput
    {
        try forward(
            embeddings: embeddings, cache: cache, validity: validity,
            recurrenceExecution: .reference)
    }
    func forward(
        embeddings: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextOutput {
        guard embeddings.ndim == 3, embeddings.dim(0) > 0, embeddings.dim(1) > 0,
            embeddings.dim(2) == configuration.hiddenSize,
            [.float32, .bfloat16, .float16].contains(embeddings.dtype)
        else {
            throw GLM5NextLoadError.invalid("embeddings must be floating [B,T,hidden]")
        }
        let allValid = validity == nil
        let rows = Array(embeddings.shape.prefix(2))
        let validity =
            validity ?? GLM5NextKernelConstants.shared.allTrue(rows)
            ?? MLXArray.ones(rows, dtype: .bool)
        guard validity.shape == Array(embeddings.shape.prefix(2)), validity.dtype == .bool else {
            throw GLM5NextLoadError.invalid("validity must be boolean [B,T]")
        }
        let destination = cache ?? newTextCache()
        try destination.validate(configuration)
        guard embeddings.dim(1) <= configuration.maxPositionEmbeddings - destination.offset else {
            throw GLM5NextLoadError.invalid("context length exceeds configuration")
        }
        let working = destination.copy()
        let hidden = try model.forward(
            embeddings, cache: working, validity: validity, recurrenceExecution: recurrenceExecution
        )
        // `where` keeps the dtype, so with every row valid it returns these bits.
        let maskedHidden =
            allValid ? hidden : MLX.where(validity.expandedDimensions(axis: -1), hidden, 0)
        let logits = lmHead(maskedHidden)
        // Compatibility-only mutation of a caller's private graph cache. This
        // is NOT evaluated publication; active serving uses prepareForward and
        // the candidate's required evaluate/join/fault-check commit fence.
        destination.adopt(working)
        return GLM5NextTextOutput(logits: logits, hidden: maskedHidden)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        do {
            return try forward(tokens: inputs, cache: cache.map { GLM5NextTextCache(layers: $0) })
                .logits
        } catch { preconditionFailure("GLM5Next forward failed: \(error)") }
    }
    public func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        precondition(state == nil, "GLM5Next has no LMOutput cross-attention state")
        do {
            return LMOutput(
                logits: try forward(
                    tokens: input.tokens,
                    cache: cache.map { GLM5NextTextCache(layers: $0) }, validity: input.mask
                ).logits)
        } catch { preconditionFailure("GLM5Next forward failed: \(error)") }
    }
}
