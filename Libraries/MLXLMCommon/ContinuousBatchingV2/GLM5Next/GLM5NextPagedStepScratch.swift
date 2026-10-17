// Copyright © 2026 Eigen Labs.

import MLX

/// The GLM worst-case step price split by what it scales with. No step can
/// exceed `setupBytes + bytesPerContextToken * Σ context` over its requests:
/// every row-layer price is bounded by a context-independent part (step
/// geometry, bounded score blocks, allocator overhead) plus a rate per token
/// of that row's history (index metadata reads, pooled index planes, latent
/// reads, page tables), each counted as `GLM5NextStepBytes` counts them.
public struct GLM5NextStepScratch: Sendable, Equatable {
    /// Bytes that do not grow with any request's context: the worst step
    /// shape plus its chained decode successor.
    public let setupBytes: Int
    /// Bytes per context token of one request: its row in the largest step
    /// shape plus its row in the chained decode successor.
    public let bytesPerContextToken: Int
}

/// Step bytes by when each allocation is live:
/// `64 KiB + Σ retained + 2 × (largest layer's Σ temporaries)`. The step owner
/// retains every layer's reads, writes, tables and transports until its
/// fence, so they count for every DSA layer. A layer's temporaries (query
/// latents, scores, pooled index planes, selections, gather blocks, attended
/// latents; `transientRoles`) are counted for the largest layer only, twice
/// for overlap with a neighbour. That relies on MLX freeing an intermediate
/// once its last consumer executes — the step owner retains none of them and
/// each DSA layer consumes its own before the next one's output exists.
///
/// Calibrated on the real NVFP4 artifact (`GLM5NextRealConcurrencyTests`,
/// 2026-10-16): 1, 2, 3 and 4 rows, 512- and 2048-token steps, 32K to 261K
/// context, and MTP rectangular verification. Setup plus rate times context
/// plus FP32 sparse KV stays 1.57x to 3.6x above every measured peak
/// (`GLM5NextStepPriceTests.layerTemporariesPriceAboveMeasuredPeak`, which
/// requires 1.5x). An under-price fails as a Metal allocation mid-step, not
/// an admission refusal: raise the multiplier rather than the tolerance.
struct GLM5NextStepBytes {
    private(set) var retained = 64 << 10
    private var transient: [Int]

    init(layers: Int) { transient = Array(repeating: 0, count: max(1, layers)) }

    static let transientRoles: Set<String> = [
        "attention.queryLatents", "dense.scores", "index.validity", "index.poolMetadata",
        "index.denseEligibility", "index.cumulative", "index.decodeScores",
        "index.decodeCompletePools", "index.decodeSelector", "index.castAndPool", "index.score",
        "index.poolScores", "index.visibility", "selection.tokens", "selection.tailMatch",
        "selected.blockLatents", "selected.blockScores", "selected.attended",
        "selected.attentionScores", "selected.projectionInputs",
    ]

    /// `layer` nil marks step-wide allocations (transports), always retained.
    mutating func add(_ allocations: [CBv2PagedAttentionAllocation], layer: Int?) throws {
        for allocation in allocations {
            if let layer, Self.transientRoles.contains(allocation.role) {
                transient[layer] = try CBv2PagedWorkMath.add(
                    transient[layer], allocation.upperBoundBytes)
            } else {
                retained = try CBv2PagedWorkMath.add(retained, allocation.upperBoundBytes)
            }
        }
    }

    /// Two steps in flight: retained bytes add, as do their temporaries.
    func merged(with other: GLM5NextStepBytes) throws -> GLM5NextStepBytes {
        var result = self
        result.retained = try CBv2PagedWorkMath.add(retained, other.retained)
        for index in result.transient.indices {
            result.transient[index] = try CBv2PagedWorkMath.add(
                transient[index], other.transient[index])
        }
        return result
    }

    func total() throws -> Int {
        try CBv2PagedWorkMath.add(
            retained, CBv2PagedWorkMath.product([transient.max()!, 2]))
    }
}

extension GLM5NextPagedModelBinding {
    /// One row-layer's bound for any history `n` up to the context limit:
    /// retained `fixed + ceil(perTokenNumerator / denominator) * n`, and the
    /// layer temporaries' `transientFixed` and per-token part likewise.
    struct RowScratchBound {
        var fixed = 0
        var perTokenNumerator = 0
        var transientFixed = 0
        var transientPerTokenNumerator = 0
        let denominator: Int

        var perToken: Int { (perTokenNumerator + denominator - 1) / denominator }
        var transientPerToken: Int {
            (transientPerTokenNumerator + denominator - 1) / denominator
        }
    }

    /// One branch's contribution: retained and temporary per-token numerators
    /// and fixed bytes.
    private struct Part {
        var perToken = 0
        var fixed = 0
        var transientPerToken = 0
        var transientFixed = 0
    }

    package func stepScratch(
        maximumRequests: Int, maximumStepTokens: Int, maximumRowTokens: Int
    ) throws -> GLM5NextStepScratch {
        guard let pool = backend?.pool, let policy = Memory.allocationFootprintPolicy() else {
            throw CBv2KVError.backendIneligible(reason: "GLM step envelope lacks its bound pool")
        }
        let shapes = try stepShapes(
            maximumRequests: maximumRequests, maximumStepTokens: maximumStepTokens,
            maximumRowTokens: maximumRowTokens)
        var bounds: [Int: [RowScratchBound]] = [:]
        func rowBounds(_ q: Int) throws -> [RowScratchBound] {
            if let cached = bounds[q] { return cached }
            let value = try kinds.indices.map {
                try rowScratchBound(layer: $0, query: q, pool: pool, policy: policy)
            }
            bounds[q] = value
            return value
        }
        /// (retained, largest layer's temporaries) of one step's fixed parts.
        func setup(_ queries: [Int]) throws -> (retained: Int, transient: Int) {
            var retained = try transportAllocations(
                rows: queries.count, maximumQuery: queries.max()!,
                maximumBufferLength: nil, policy: policy
            ).reduce(64 << 10) { try CBv2PagedWorkMath.add($0, $1.upperBoundBytes) }
            var transient = Array(repeating: 0, count: kinds.count)
            for q in queries {
                for (index, bound) in try rowBounds(q).enumerated() {
                    retained = try CBv2PagedWorkMath.add(retained, bound.fixed)
                    transient[index] = try CBv2PagedWorkMath.add(
                        transient[index], bound.transientFixed)
                }
            }
            return (retained, transient.max()!)
        }
        let decode = try setup(shapes.decode)
        var setupBytes = 0
        for shape in [shapes.decode, shapes.packed, shapes.even] {
            let step = try setup(shape)
            setupBytes = max(
                setupBytes,
                try CBv2PagedWorkMath.add(
                    CBv2PagedWorkMath.add(step.retained, decode.retained),
                    CBv2PagedWorkMath.product([
                        try CBv2PagedWorkMath.add(step.transient, decode.transient), 2,
                    ])))
        }
        // A row's rate never decreases with its query count, so the largest
        // row query bounds every shape's rows.
        let rowQuery = max(shapes.packed.max()!, shapes.even.max()!)
        var perToken = 0
        var transientPerToken = 0
        for (row, decodeRow) in zip(try rowBounds(rowQuery), try rowBounds(1)) {
            perToken = try [perToken, row.perToken, decodeRow.perToken].reduce(
                0, CBv2PagedWorkMath.add)
            transientPerToken = max(
                transientPerToken,
                try CBv2PagedWorkMath.add(row.transientPerToken, decodeRow.transientPerToken))
        }
        return .init(
            setupBytes: setupBytes,
            bytesPerContextToken: try CBv2PagedWorkMath.add(
                perToken, CBv2PagedWorkMath.product([transientPerToken, 2])))
    }

    /// The step shapes `worstCaseStepBytes` prices: all-decode, as many full
    /// rows as the step admits beside decode rows, and the step split evenly.
    func stepShapes(maximumRequests: Int, maximumStepTokens: Int, maximumRowTokens: Int) throws
        -> (decode: [Int], packed: [Int], even: [Int])
    {
        guard maximumRequests > 0, maximumStepTokens > 0, maximumRowTokens > 0 else {
            throw CBv2KVError.backendIneligible(reason: "GLM step envelope lacks its bound pool")
        }
        let rows = min(maximumRequests, limits.maximumBatchSize)
        let stepTokens = min(maximumStepTokens, rows * limits.maximumQueryTokens)
        let rowQuery = min(
            maximumRowTokens, stepTokens, limits.maximumQueryTokens, limits.maximumContextTokens)
        let full = min(rows, stepTokens / rowQuery)
        let remainder = stepTokens - full * rowQuery
        var packed = Array(repeating: rowQuery, count: full)
        if remainder > 0, packed.count < rows { packed.append(remainder) }
        packed += Array(repeating: 1, count: rows - packed.count)
        let spread = min(rows, stepTokens)
        let even = Array(
            repeating: min(rowQuery, (stepTokens + spread - 1) / spread), count: spread)
        return (Array(repeating: 1, count: rows), packed, even)
    }

    /// Bounds `workAllocations` for a row of `query` tokens at any history up
    /// to the context limit and any start position, retained and temporary
    /// allocations (`GLM5NextStepBytes.transientRoles`) separately. Terms
    /// linear in the history (or in its pages and size-class segments)
    /// contribute their coefficient; bounded terms are taken at their bound;
    /// each allocation's overhead is fixed. Where the exact price takes one of
    /// several branches (dense prefix or not, dense/decode/pooled selection,
    /// history or per-query latent reads), the bound takes each part's maximum.
    func rowScratchBound(
        layer: Int, query q: Int, pool: PagedKVPool, policy: AllocationFootprintPolicy
    ) throws -> RowScratchBound {
        guard let extra = policy.maximumExtraBytes, q > 0 else {
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        let kind = kinds[layer]
        let context = limits.maximumContextTokens
        let pageSize = pool.config.pageSize
        let contextPages = (context - 1) / pageSize + 1
        let usable = max(
            1,
            pool.group(pool.groupKey(forLayer: layer)).segmentLayout?.maximumUsablePages
                ?? contextPages)
        let classes = Int.bitWidth - usable.leadingZeroBitCount
        let denominator = try CBv2PagedWorkMath.product([pageSize, usable])
        var bound = RowScratchBound(denominator: denominator)
        let add = CBv2PagedWorkMath.add
        func ub(_ factors: [Int]) throws -> Int {
            let logical = try CBv2PagedWorkMath.product(factors)
            guard let value = policy.upperBound(byteCount: logical) else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            return value
        }
        func addFixed(_ bytes: Int) throws { bound.fixed = try add(bound.fixed, bytes) }
        func fixed(_ factors: [Int], copies: Int) throws {
            try addFixed(CBv2PagedWorkMath.product([try ub(factors), copies]))
        }
        func temporary(_ factors: [Int], copies: Int) throws {
            bound.transientFixed = try add(
                bound.transientFixed, CBv2PagedWorkMath.product([try ub(factors), copies]))
        }
        /// `numerator / denominator` retained bytes per history token.
        func perToken(_ numerator: Int) throws {
            bound.perTokenNumerator = try add(bound.perTokenNumerator, numerator)
        }
        func perTokenBytes(_ bytes: Int) throws {
            try perToken(CBv2PagedWorkMath.product([bytes, denominator]))
        }
        func merge(_ parts: [Part]) throws {
            try perToken(parts.map(\.perToken).max()!)
            try addFixed(parts.map(\.fixed).max()!)
            bound.transientPerTokenNumerator = try add(
                bound.transientPerTokenNumerator, parts.map(\.transientPerToken).max()!)
            bound.transientFixed = try add(bound.transientFixed, parts.map(\.transientFixed).max()!)
        }
        // segments <= n / (pageSize * usable) + classes + 2; pages <= n / pageSize + 1.
        let segmentsFixed = classes + 2
        let writeSegments = min(q / pageSize + 2, contextPages / usable + classes + 1)
        let budget = selectionWidth - 3
        let heads = kind.queryHeads
        let latent = kind.headDim
        let pools = (context + 3) / 4

        // Write and history-read records, fences, host plans and shapes.
        try addFixed(CBv2PagedWorkMath.product([q, 12]))
        try addFixed(CBv2PagedWorkMath.product([writeSegments, add(96, extra)]))
        try perTokenBytes(12)
        try perToken(add(96, extra))
        try addFixed(CBv2PagedWorkMath.product([add(96, extra), segmentsFixed]))
        let word = try ub([4])
        try fixed([4], copies: writeSegments + 1)
        try perToken(word)
        try addFixed(CBv2PagedWorkMath.product([word, segmentsFixed + 1]))
        try fixed([q, 68], copies: 1)
        try perTokenBytes(68)
        try addFixed(extra)
        try perToken(CBv2PagedWorkMath.product([2, 456, usable]))
        try addFixed(CBv2PagedWorkMath.product([2, add(456, extra)]))
        let shape = try ub([12])
        try addFixed(CBv2PagedWorkMath.product([shape, writeSegments]))
        try perToken(shape)
        try addFixed(CBv2PagedWorkMath.product([shape, segmentsFixed]))
        try fixed([q, latent, 4], copies: writeSegments)
        try fixed([q, kind.valueHeadDim, 4], copies: writeSegments)
        // The retained index metadata read; temporary validity and pool
        // metadata planes and absorbed queries.
        try perTokenBytes(CBv2PagedWorkMath.product([kind.valueHeadDim, 4]))
        try addFixed(extra)
        try merge([
            Part(
                transientPerToken: try CBv2PagedWorkMath.product([4 * 4 + 6 * 4, denominator]),
                transientFixed: try CBv2PagedWorkMath.product([10, extra]))
        ])
        try temporary([q, heads, latent, 4], copies: 2)
        if denseHistoryPrefix {
            let dense = min(q, budget)
            let keys = min(budget, context)
            let scores = min(
                try CBv2PagedWorkMath.product([dense, heads, keys, 4]),
                max(attentionBlockBytes, try CBv2PagedWorkMath.product([heads, keys, 4])))
            try fixed([selectionWidth, latent, 4], copies: 1)
            try temporary([scores], copies: 2)
        }

        // Selection (temporary): the dense shortcut, native decode or pooled
        // scoring.
        var selection = [
            Part(
                transientFixed: try add(
                    CBv2PagedWorkMath.product([try ub([q, budget / 4]), 2]),
                    try ub([budget + 3, 4])))
        ]
        if q <= nativeSparseDecodeQueries {
            // Scores `[Q,P]`, two complete-pool masks `[P]`, the cumulative
            // visibility `[N]`, and the bounded selector rows.
            selection.append(
                Part(
                    transientPerToken: try [
                        CBv2PagedWorkMath.product([q, denominator]), (denominator + 1) / 2,
                        CBv2PagedWorkMath.product([4, denominator]),
                    ].reduce(0, add),
                    transientFixed: try [
                        CBv2PagedWorkMath.product([4, q]), extra,
                        CBv2PagedWorkMath.product([2, add(1, extra)]),
                        try ub([q, min(pools, 64 * 512 - 1), 4]), extra,
                    ].reduce(0, add)))
        } else {
            // Pooled index planes `[N,D]`, then score blocks bounded by the
            // score block bytes (or one query's row at the context limit).
            // A block holds at most `q` queries.
            let scores = min(
                try CBv2PagedWorkMath.product([q, indexHeads, pools, 4]),
                max(scoreBlockBytes, try CBv2PagedWorkMath.product([indexHeads, pools, 4])))
            let pooled = min(
                try CBv2PagedWorkMath.product([q, pools, 4]),
                max(scoreBlockBytes / indexHeads, try CBv2PagedWorkMath.product([pools, 4])))
            let visible = min(
                try CBv2PagedWorkMath.product([q, context, 5]),
                try CBv2PagedWorkMath.product([max(scoreBlockBytes / (indexHeads * 4), pools), 20]))
            selection.append(
                Part(
                    transientPerToken: try CBv2PagedWorkMath.product([
                        6, indexDimension, 4, denominator,
                    ]),
                    transientFixed: try [
                        CBv2PagedWorkMath.product([6, extra]),
                        CBv2PagedWorkMath.product([try ub([scores]), 2]),
                        CBv2PagedWorkMath.product([try ub([pooled]), 3]),
                        try ub([visible]),
                    ].reduce(0, add)))
        }
        try merge(selection)
        try temporary([q, selectionWidth, 4], copies: 4)
        try temporary([q, 3, selectionWidth], copies: 1)

        // Latent reads (retained): one history read, or per-query selection
        // rows; their attention temporaries.
        let read = try ub([selectionWidth, latent, 4])
        let historyReads = latentHistoryMinimumQueries > 0 && q >= latentHistoryMinimumQueries
        let overlap = min(q, 2)
        let scratch = try add(
            CBv2PagedWorkMath.product([try ub([heads, selectionWidth, 4]), 4 * overlap]),
            CBv2PagedWorkMath.product([read, overlap]))
        var latents: [Part] = []
        if historyReads {
            let gathers: Int
            if absorbedAttention {
                let block = max(1, min(q, attentionBlockBytes / (selectionWidth * latent * 4)))
                let blocks = min(2, (q + block - 1) / block)
                gathers = try add(
                    CBv2PagedWorkMath.product([
                        try ub([block, selectionWidth, latent, 4]), 2 * blocks,
                    ]),
                    CBv2PagedWorkMath.product([
                        try ub([block, heads, selectionWidth, 4]), 4 * blocks,
                    ]))
            } else {
                gathers = scratch
            }
            latents.append(
                Part(
                    perToken: try CBv2PagedWorkMath.product([latent, 4, denominator]),
                    fixed: extra,
                    transientFixed: try add(gathers, try ub([q, heads, latent, 4]))))
        }
        // Without its history read a row past the threshold reads fewer than
        // `n / selectionWidth` rows beyond the threshold's.
        let readsBeyondThreshold = historyReads ? latentHistoryMinimumQueries - 1 : q
        latents.append(
            Part(
                perToken: historyReads
                    ? try CBv2PagedWorkMath.product([read, denominator]) / selectionWidth + 1 : 0,
                fixed: try CBv2PagedWorkMath.product([read, readsBeyondThreshold]),
                transientFixed: scratch))
        try merge(latents)

        // Read indices and fences (reads x touched segments at the context
        // limit), the row's table and host plan, and one copy-on-write page.
        let segmentLimit = try add(contextPages / usable, segmentsFixed + 1)
        let indexRow = try ub([selectionWidth, 4])
        if historyReads {
            try perToken(
                CBv2PagedWorkMath.product([word, segmentLimit, denominator]) / selectionWidth + 1)
            try addFixed(
                CBv2PagedWorkMath.product([word, segmentLimit, latentHistoryMinimumQueries + 1]))
            try perToken(CBv2PagedWorkMath.product([indexRow, denominator]) / selectionWidth + 1)
            try addFixed(CBv2PagedWorkMath.product([indexRow, latentHistoryMinimumQueries + 1]))
        } else {
            try addFixed(CBv2PagedWorkMath.product([word, segmentLimit, q + 1]))
            try addFixed(CBv2PagedWorkMath.product([indexRow, q + 1]))
        }
        try perToken(CBv2PagedWorkMath.product([4, usable]))
        try addFixed(add(8, extra))
        try perToken(CBv2PagedWorkMath.product([128, usable]))
        try addFixed(add(128, extra))
        try fixed([pageSize, latent + kind.valueHeadDim, 4], copies: 4)
        try fixed([max(24, pageSize * 3), 4], copies: 2)
        try fixed([4], copies: 4)
        try fixed([pageSize, 128], copies: 1)
        return bound
    }
}
