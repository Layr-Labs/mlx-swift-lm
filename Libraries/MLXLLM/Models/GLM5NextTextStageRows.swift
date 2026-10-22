// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon

/// One request's part of a multi-request stage step: its tokens `[1, T]`
/// after the history its own cache holds.
public struct GLM5NextTextStageLane {
    public let tokens: MLXArray
    public let cache: GLM5NextTextStageCache

    public init(tokens: MLXArray, cache: GLM5NextTextStageCache) {
        self.tokens = tokens
        self.cache = cache
    }
}

/// The lanes' candidates of one multi-request step (`GLM5NextTextStage
/// .prepareVerificationRows`, `.prepareRows`): one fence evaluates every
/// lane, then each lane publishes into its own cache exactly as its own
/// `GLM5NextTextStageCandidate` would. A throwing fence, keep decision or
/// stale lane leaves every cache intact.
public final class GLM5NextTextStageRowsCandidate {
    /// In lane order.
    public let lanes: [GLM5NextTextStageCandidate]
    public var outputs: [GLM5NextTextStageOutput] { lanes.map(\.output) }
    private var pending = true

    init(lanes: [GLM5NextTextStageCandidate]) { self.lanes = lanes }

    /// Publishes every token of every lane.
    @discardableResult
    public func commit(afterEvaluating fence: ([MLXArray]) throws -> Void) throws
        -> [GLM5NextTextStageOutput]
    {
        try commit(afterEvaluating: fence) { self.lanes.map(\.tokens) }
    }

    /// Speculative verification, per lane: lane i publishes only the first
    /// `keeping(outputs)[i]` tokens of its window (1 through its length), as
    /// `GLM5NextTextStageCandidate.commit(afterEvaluating:keeping:)` does.
    @discardableResult
    public func commit(
        afterEvaluating fence: ([MLXArray]) throws -> Void,
        keeping: ([GLM5NextTextStageOutput]) throws -> [Int]
    ) throws -> [GLM5NextTextStageOutput] {
        guard lanes.allSatisfy(\.captured) else {
            throw GLM5NextLoadError.invalid("only verification rows keep a token prefix")
        }
        return try commit(afterEvaluating: fence) { try keeping(self.outputs) }
    }

    private func commit(afterEvaluating fence: ([MLXArray]) throws -> Void, keep: () throws -> [Int])
        throws -> [GLM5NextTextStageOutput]
    {
        guard pending else {
            throw GLM5NextLoadError.invalid("stage rows candidate is no longer pending")
        }
        pending = false
        do {
            try fence(try lanes.flatMap { try $0.pendingRoots() })
            let kept = try keep()
            guard kept.count == lanes.count,
                zip(kept, lanes).allSatisfy({ (1 ... $1.tokens).contains($0) })
            else {
                throw GLM5NextLoadError.invalid("verification rows keep 1...T tokens of each lane")
            }
            // Every lane is still publishable before any lane publishes.
            for lane in lanes { _ = try lane.pendingRoots() }
            for (lane, count) in zip(lanes, kept) {
                try lane.publishEvaluated(keeping: count)
            }
            return outputs
        } catch {
            for lane in lanes { lane.discard() }
            throw error
        }
    }

    public func discard() {
        pending = false
        for lane in lanes { lane.discard() }
    }
}

extension GLM5NextTextStage {
    /// MLX's smallest quantized matrix-vector batch limit on this device
    /// (`get_qmv_batch_limit`, mlx/backend/metal/quantized.cpp, at its widest
    /// shapes; the architecture name parsed as MLX parses it). A product of
    /// fewer rows runs `qmv` for one row and `qmv_wide` for 2 or more, and
    /// both give each row the arithmetic it has at any other count in that
    /// range; from the limit on, MLX runs a tiled matrix product instead.
    static let quantizedVectorLimit: Int = {
        let name = Array(GPU.deviceInfo().architecture.utf8)
        guard name.count >= 3 else { return 6 }
        func digit(_ value: UInt8) -> Int { (48 ... 57).contains(value) ? Int(value) - 48 : 0 }
        let generation = digit(name[name.count - 3]) * 10 + digit(name[name.count - 2])
        let ultra = name[name.count - 1] == UInt8(ascii: "d")
        switch (generation, ultra) {
        case (15..., false): return 13
        case (13..., true): return 12
        case (13..., false): return 6
        case (_, true): return 12
        default: return 10
        }
    }()

    /// Rows one launch of a multi-request step takes: within every row-wise
    /// kernel's decode-sized form (fused MoE, one-launch mHC sites, packed
    /// embedding streams) and below `quantizedVectorLimit`, so each row's
    /// arithmetic is the one it has in its own request's step.
    public static var maximumGroupRows: Int {
        min(
            GLM5NextMoEDecodeKernel.maximumTokens,
            GLM5NextHyperConnectionKernel.splitCollapseMaximumRows,
            GLM5NextPackedEmbedding.maximumStreamTokens, quantizedVectorLimit - 1)
    }

    /// Rows of one group when the mHC sites and routed experts take every
    /// row at once (`rowsShareSites`): the projections' bound alone.
    public static var sharedSiteGroupRows: Int { quantizedVectorLimit - 1 }

    /// Which lanes share each row-wise launch, in lane order: lanes of one
    /// window length, at most `maximumRows` rows a group. A one-token lane
    /// stays alone: one row runs `qmv`, two or more `qmv_wide`.
    static func rowGroups(lengths: [Int], maximumRows: Int = maximumGroupRows) -> [[Int]] {
        var groups: [[Int]] = []
        var open: [Int: Int] = [:]
        for (lane, length) in lengths.enumerated() {
            let capacity = length < 2 ? 1 : max(1, maximumRows / length)
            if let index = open[length], groups[index].count < capacity {
                groups[index].append(lane)
            } else {
                open[length] = groups.count
                groups.append([lane])
            }
        }
        return groups
    }

    /// Speculative verification windows of several requests in ONE stage
    /// forward on a whole-model stage: lane i's window `[1, T_i]` (1 through
    /// `maximumVerificationTokens` tokens) after its own cache. Each lane's
    /// output and cache are bitwise those of its own `prepareVerification`:
    /// projections, dense MLPs, KDA (each group's rows on their stacked
    /// states) and the head run in `rowGroups`, which keep every row in the
    /// arithmetic of its own window; DSA appends and attends each lane over
    /// its own history; the mHC sites (one launch over every row) and the
    /// routed experts (decode-sized chunks) have per-row arithmetic at any
    /// row count (`rowsShareSites`); each layer's two sums cover every
    /// lane's rows at once (a tensor-parallel sum is elementwise).
    public func prepareVerificationRows(_ lanes: [GLM5NextTextStageLane]) throws
        -> GLM5NextTextStageRowsCandidate
    {
        try prepareRows(lanes, readout: .allTokens, captureWindow: true)
    }

    /// Ordinary decode-sized steps (1 through `maximumVerificationTokens`
    /// tokens) of several requests in one stage forward, each lane's output
    /// and cache bitwise those of its own `prepareForward(tokens:cache:readout:)`.
    /// One-token lanes do not share row-wise launches (`rowGroups`), so they
    /// save the sums and the host work of separate steps, not weight reads.
    public func prepareRows(
        _ lanes: [GLM5NextTextStageLane], readout: GLM5NextTextStageReadout = .lastToken
    ) throws -> GLM5NextTextStageRowsCandidate {
        try prepareRows(lanes, readout: readout, captureWindow: false)
    }

    private func prepareRows(
        _ lanes: [GLM5NextTextStageLane], readout: GLM5NextTextStageReadout, captureWindow: Bool
    ) throws -> GLM5NextTextStageRowsCandidate {
        let c = plan.configuration
        guard plan.isFirst, plan.isFinal, let embedding, let norm, let head, !lanes.isEmpty,
            Set(lanes.map { ObjectIdentifier($0.cache.storage) }).count == lanes.count
        else {
            throw GLM5NextLoadError.invalid(
                "stage rows need a whole-model stage and distinct lane caches")
        }
        for lane in lanes {
            let tokens = lane.tokens
            let cache = lane.cache
            guard tokens.ndim == 2, tokens.dim(0) == 1,
                (1 ... Self.maximumVerificationTokens).contains(tokens.dim(1)),
                [.int32, .uint32, .int64].contains(tokens.dtype),
                cache.binding === binding, cache.layers.count == plan.layers.count,
                zip(plan.layers, cache.layers).allSatisfy({ index, state in
                    state.offset == cache.offset
                        && (c.layerTypes[index] == .linearAttention
                            ? state is GLM5NextKDACache : state is GLM5NextDSACache)
                }), tokens.dim(1) <= c.maxPositionEmbeddings - cache.offset
            else {
                throw GLM5NextLoadError.invalid(
                    "stage row needs [1, 1...\(Self.maximumVerificationTokens)] tokens on its own bound cache"
                )
            }
        }
        let originals = lanes.map { $0.cache.storage.revision() }
        let starts = lanes.map(\.cache.offset)
        let workings = lanes.map { $0.cache.storage.copy() }
        let shareSites = rowsShareSites && executionPolicy.hyperConnection == .nativeNormFused
        let groups = Self.rowGroups(
            lengths: lanes.map { $0.tokens.dim(1) },
            maximumRows: rowGroupRowsOverride
                ?? (shareSites ? Self.sharedSiteGroupRows : Self.maximumGroupRows))
        // Each group's embedded streams `[B,T,hcMult,hidden]`; a group wider
        // than one packed-streams launch takes several (one thread per token
        // and column, so the same bits).
        let embedded = groups.map { group -> MLXArray in
            let tokens =
                group.count == 1
                ? lanes[group[0]].tokens
                : concatenated(group.map { lanes[$0].tokens.asType(.int32) }, axis: 0)
            let shape = [tokens.dim(0), tokens.dim(1), c.hcMult, c.hiddenSize]
            let launch = GLM5NextPackedEmbedding.maximumStreamTokens
            if tokens.size > launch {
                let flat = tokens.reshaped(1, -1)
                let pieces = stride(from: 0, to: tokens.size, by: launch).map {
                    embeddedStreams(flat[0..., $0 ..< min(tokens.size, $0 + launch)])
                }
                let present = pieces.compactMap { $0 }
                if present.count == pieces.count {
                    return concatenated(present, axis: 1).reshaped(shape)
                }
            }
            return embeddedStreams(tokens)
                ?? broadcast(embedding(tokens).expandedDimensions(axis: 2), to: shape)
        }
        let validities = groups.map { group -> MLXArray in
            let shape = [group.count, lanes[group[0]].tokens.dim(1)]
            return GLM5NextKernelConstants.shared.allTrue(shape)
                ?? MLXArray.ones(shape, dtype: .bool)
        }
        let caches = { (index: Int) in groups.map { $0.map { workings[$0].layers[index] } } }
        let boundaries: [MLXArray]
        if shareSites {
            boundaries = try sharedSiteRows(
                embedded, caches: caches, validities: validities, captureWindow: captureWindow)
        } else {
            boundaries = try groupedRows(
                embedded, caches: caches, validities: validities, captureWindow: captureWindow)
        }
        var outputs = [GLM5NextTextStageOutput?](repeating: nil, count: lanes.count)
        for (group, boundary) in zip(groups, boundaries) {
            // [B,T,hcMult,hidden] or the group's rows of the shared [1,M,...].
            let length = lanes[group[0]].tokens.dim(1)
            let hidden = norm(boundary.mean(axis: 2)).reshaped(1, -1, c.hiddenSize)
            let logits = readout == .allTokens ? head(hidden) : nil
            for (row, lane) in group.enumerated() {
                let rows = row * length ..< (row + 1) * length
                let laneHidden = group.count == 1 ? hidden : hidden[0..., rows]
                switch readout {
                case .allTokens:
                    outputs[lane] = .final(
                        .init(
                            logits: group.count == 1 ? logits! : logits![0..., rows],
                            hidden: laneHidden))
                case .lastToken:
                    // One row, as the lane's own step projects it.
                    outputs[lane] = .final(
                        .init(logits: head(laneHidden[0..., (-1)..., 0...]), hidden: laneHidden))
                case .none:
                    outputs[lane] = .hidden(laneHidden)
                }
            }
        }
        guard zip(lanes, originals).allSatisfy({ $0.cache.storage.revision() == $1 }) else {
            throw GLM5NextLoadError.invalid("stage cache changed while preparing rows")
        }
        return .init(
            lanes: lanes.indices.map { i in
                GLM5NextTextStageCandidate(
                    output: outputs[i]!, destination: lanes[i].cache.storage,
                    original: originals[i], working: workings[i], start: starts[i],
                    tokens: lanes[i].tokens.dim(1), captured: captureWindow)
            })
    }

    /// Every layer group by group (`GLM5NextTextLayer.phases(rows:...)`), the
    /// two sums over every row; each group's boundary `[B,T,hcMult,hidden]`.
    private func groupedRows(
        _ embedded: [MLXArray], caches: (Int) -> [[any KVCache]], validities: [MLXArray],
        captureWindow: Bool
    ) throws -> [MLXArray] {
        var streams = embedded.map { GLM5NextTextLayer.Streams.materialized($0) }
        let early = executionPolicy.stepEvaluation == .earlyAsync
        let stride = GLM5NextExecutionPolicy.earlyEvaluationLayerStride
        for (index, layer) in layers.enumerated() {
            let groupCaches = caches(index)
            let attention = try streams.indices.map { g in
                try layer.phases(
                    rows: streams[g], caches: groupCaches[g], validity: validities[g],
                    captureWindow: captureWindow)
            }
            let attentionSums = summedRows(layer, attention.map(\.partial))
            let feedForward = try zip(attention, attentionSums).map { try $0.resume($1) }
            let feedForwardSums = summedRows(layer, feedForward.map(\.partial))
            streams = try zip(feedForward, feedForwardSums).map { try $0.resume($1) }
            if early, (index + 1) % stride == 0, index + 1 < layers.count {
                asyncEval(streams.flatMap(\.arrays))
            }
        }
        return try streams.map { try $0.materialize() }
    }

    /// Every lane's rows as one `[1, M, ...]` row set, group after group
    /// (`GLM5NextTextLayer.rowsForward`): each mHC site one launch over every
    /// row, attention and dense MLPs per group, routed experts in
    /// decode-sized chunks. Each group's boundary is its rows `[1, B*T, ...]`.
    private func sharedSiteRows(
        _ embedded: [MLXArray], caches: (Int) -> [[any KVCache]], validities: [MLXArray],
        captureWindow: Bool
    ) throws -> [MLXArray] {
        let c = plan.configuration
        var groups: [(rows: Range<Int>, shape: [Int])] = []
        var total = 0
        for streams in embedded {
            let rows = streams.dim(0) * streams.dim(1)
            groups.append((total ..< total + rows, [streams.dim(0), streams.dim(1)]))
            total += rows
        }
        let flat = embedded.map { $0.reshaped(1, -1, c.hcMult, c.hiddenSize) }
        var streams = GLM5NextTextLayer.Streams.materialized(
            flat.count == 1 ? flat[0] : concatenated(flat, axis: 1))
        let workspace = GLM5NextRowSiteWorkspaces.shared.workspace(rows: total)
        let early = executionPolicy.stepEvaluation == .earlyAsync
        let stride = GLM5NextExecutionPolicy.earlyEvaluationLayerStride
        for (index, layer) in layers.enumerated() {
            streams = try layer.rowsForward(
                streams, groups: groups, caches: caches(index), validities: validities,
                captureWindow: captureWindow, workspace: workspace)
            if early, (index + 1) % stride == 0, index + 1 < layers.count {
                asyncEval(streams.arrays)
            }
        }
        let boundary = try streams.materialize()
        return groups.map { boundary[0..., $0.rows] }
    }

    /// Each group's partial summed across ranks as ONE sum over every row:
    /// the sum is elementwise, so each row gets exactly its own sum.
    private func summedRows(_ layer: GLM5NextTextLayer, _ partials: [MLXArray]) -> [MLXArray] {
        guard partials.count > 1, layer.reduce != nil else { return partials.map(layer.summed) }
        let width = partials[0].dim(-1)
        let summed = layer.summed(
            concatenated(partials.map { $0.reshaped(1, -1, width) }, axis: 1))
        var start = 0
        return partials.map { partial in
            let rows = partial.size / width
            defer { start += rows }
            return summed[0..., start ..< start + rows].reshaped(partial.shape)
        }
    }
}

/// One-launch mHC site workspaces of multi-request rows, by row capacity:
/// `SiteWorkspace.shared` up to the decode bound, else a power of two.
final class GLM5NextRowSiteWorkspaces: @unchecked Sendable {
    static let shared = GLM5NextRowSiteWorkspaces()
    private let lock = NSLock()
    private var byRows: [Int: GLM5NextHyperConnectionKernel.SiteWorkspace] = [:]

    func workspace(rows: Int) -> GLM5NextHyperConnectionKernel.SiteWorkspace {
        let bound = GLM5NextHyperConnectionKernel.splitCollapseMaximumRows
        guard rows > bound else { return .shared }
        var capacity = bound
        while capacity < rows { capacity *= 2 }
        lock.lock()
        defer { lock.unlock() }
        if let existing = byRows[capacity] { return existing }
        let created = GLM5NextHyperConnectionKernel.SiteWorkspace(rows: capacity)
        byRows[capacity] = created
        return created
    }
}
