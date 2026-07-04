// Streamed drop-in replacement for `SwitchGLU` (see SwitchLayers.swift) that
// never holds the full [numExperts, out, in] routed-expert weight tensors
// resident. Instead it fetches only the experts touched by the current
// batch of tokens straight off disk (via `ExpertShardStore`, cached by
// `ExpertCache`) and runs the SAME `gatherQuantizedMM` compute the resident
// `QuantizedSwitchLinear` path uses — just against a small, per-chunk
// compact expert stack instead of the full checkpoint tensor.
//
// WHY gatherSort + contiguous chunking (not full-shape masked compute):
// naively looping "for each of K candidate expert groups, run the FULL
// gatherQuantizedMM over every token and mask out the rest" multiplies
// matmul FLOPs by the number of groups — exactly the sparsity MoE routing
// exists to avoid. Instead this reuses `gatherSort` (public in
// SwitchLayers.swift, already used by the resident SwitchGLU's own sort
// fast-path) to group (token, expert) pairs into CONTIGUOUS runs by expert
// id, then buckets consecutive runs into disk-fetch chunks. Each chunk runs
// exactly one gatherQuantizedMM per projection over exactly the rows that
// need it — same FLOP count as the resident path, with I/O standing in for
// a big resident tensor.
//
// This module intentionally does NOT subclass `Module` and holds its
// MLXArray-bearing state (via `ExpertCache`/`ExpertShardStore`, not as
// direct stored properties) so `Module.parameters()` / `update(parameters:)`
// / `quantize(model:)` never see it — see `DeepseekV4MoE` in DeepseekV4.swift
// for how the resident `SwitchGLU` is swapped out for this type wholesale
// rather than nested inside the module tree.

import Foundation
import MLX

public final class StreamingQuantizedSwitchGLU: @unchecked Sendable {
    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    public let layerIndex: Int

    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode
    private let activationProduct: @Sendable (MLXArray, MLXArray) -> MLXArray

    private let cache: ExpertCache
    private let store: ExpertShardStore
    /// Optional cross-token speculative prefetch (see PrefetchCoordinator.swift).
    /// Nil disables prefetch entirely (e.g. `DSV4_STREAM_PREFETCH=0`, or
    /// callers/tests that don't want the background queue at all).
    private let prefetch: PrefetchCoordinator?
    /// Optional per-(layer, expert) usage-frequency recorder (see
    /// ExpertUsageProfile.swift), persisted to disk and consumed by
    /// `ExpertCacheWarmer` on a FUTURE process's load. Nil disables
    /// collection entirely (`DSV4_STREAM_PROFILE=0`, or tests that don't
    /// want the extra bookkeeping).
    private let usageProfile: ExpertUsageProfile?

    /// Max experts fetched/stacked into one `gatherQuantizedMM` call. Bounds
    /// peak transient memory for prefill batches that touch most/all of the
    /// routed experts in one forward pass — without this, stacking every
    /// unique expert touched into one `[256, out, in]`-shaped tensor before
    /// the matmul would briefly need a resident-sized tensor anyway, which
    /// defeats the point of streaming. Override via `DSV4_STREAM_CHUNK_EXPERTS`.
    private let maxExpertsPerChunk: Int

    /// Byte threshold above which a chunk's stacked weights are eval'd
    /// immediately even when it's the ONLY chunk in this forward call. See
    /// `shouldEvalChunk` for the full reasoning. Override via
    /// `DSV4_STREAM_EVAL_THRESHOLD_MB`.
    private let evalThresholdBytes: Int

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        layerIndex: Int,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        cache: ExpertCache,
        store: ExpertShardStore,
        activationProduct: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray,
        maxExpertsPerChunk: Int? = nil,
        evalThresholdBytes: Int? = nil,
        prefetch: PrefetchCoordinator? = nil,
        usageProfile: ExpertUsageProfile? = nil
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.layerIndex = layerIndex
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.cache = cache
        self.store = store
        self.activationProduct = activationProduct
        self.maxExpertsPerChunk = maxExpertsPerChunk ?? Self.chunkSizeFromEnv()
        self.evalThresholdBytes = evalThresholdBytes ?? Self.evalThresholdBytesFromEnv()
        self.prefetch = prefetch
        self.usageProfile = usageProfile
    }

    private static func chunkSizeFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_CHUNK_EXPERTS"],
            let n = Int(raw), n > 0
        {
            return n
        }
        return 16
    }

    /// Reads `DSV4_STREAM_EVAL_THRESHOLD_MB` (int, default 400). 400 MiB
    /// comfortably exceeds a full `maxExpertsPerChunk`-sized (default 16)
    /// mxfp4 chunk (~13.6 MB/expert x 16 = ~218 MB), which is the common
    /// decode case this threshold is meant to let through WITHOUT an eval —
    /// see `shouldEvalChunk`.
    private static func evalThresholdBytesFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_EVAL_THRESHOLD_MB"],
            let mb = Int(raw), mb > 0
        {
            return mb * 1024 * 1024
        }
        return 400 * 1024 * 1024
    }

    /// Same call surface as `SwitchGLU.callAsFunction`: `x` is the MoE
    /// input `[..., D]`, `indices` is the router's per-token top-k expert
    /// choice `[..., topK]`. Returns per-(token, expert) FFN outputs shaped
    /// `[..., topK, D]` — the caller (`DeepseekV4MoE`) applies the
    /// score-weighted sum over the topK axis itself, exactly as it does for
    /// the resident path.
    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        pinThreadDefaultStreamsToGlobal()

        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let (xSorted, idxSorted, inverseOrder) = gatherSort(x: expanded, indices: indices)

        // Pull the sorted expert-id array to the host once. It is tiny
        // (at most a few thousand int32s even for large prefill batches);
        // everything downstream — run-length grouping, chunk boundaries,
        // local-id remap — is Swift bookkeeping over this array. The actual
        // activations and weights never leave MLX/GPU memory.
        //
        // No explicit `eval(idxSorted)` here: `MLXArray.asArray` already
        // funnels through the same lazy-eval path internally (mlx-swift's
        // `eval()`/`item()`/`asArray()`/`asData()` all synchronize through
        // the same `evalLock` — see mlx-swift's MLXArray.swift). Calling
        // both was a redundant second synchronization point per forward
        // pass for no benefit.
        let sortedExpertIds = idxSorted.asArray(Int32.self)
        let n = sortedExpertIds.count

        guard n > 0 else {
            let empty = scatterUnsort(x: xSorted[0 ..< 0], invOrder: inverseOrder, shape: indices.shape)
            return MLX.squeezed(empty, axis: -2)
        }

        // Run-length encode the sorted expert ids into (expert, rowRange)
        // groups. Sortedness guarantees each group's rows are contiguous.
        var groups: [(expert: Int, range: Range<Int>)] = []
        groups.reserveCapacity(min(n, numExperts))
        var groupStart = 0
        for i in 1...n {
            if i == n || sortedExpertIds[i] != sortedExpertIds[groupStart] {
                groups.append((Int(sortedExpertIds[groupStart]), groupStart ..< i))
                groupStart = i
            }
        }

        // Record this layer's unique expert selection for the CURRENT token
        // and kick speculative background prefetch for layers ahead of us,
        // using THEIR previous-token selections (see PrefetchCoordinator.swift).
        // Placed before the fetch loop below (not after) so the background
        // reads run concurrently with this layer's own foreground fetch +
        // compute, not after it — maximizing SSD idle-time overlap.
        if let prefetch {
            prefetch.recordSelection(layer: layerIndex, experts: groups.map { $0.expert })
            prefetch.prefetchAhead(fromLayer: layerIndex)
        }
        // Cheap, host-only bookkeeping (no GPU work, no extra I/O) for the
        // persisted usage-frequency profile that a FUTURE process's
        // `ExpertCacheWarmer` will warm from — see ExpertUsageProfile.swift.
        // Reuses the SAME `groups` this call already computed for its own
        // chunk-fetch planning.
        usageProfile?.record(layer: layerIndex, groups: groups)

        // Bucket consecutive groups into fetch chunks. Because groups are
        // already in ascending-expert / ascending-row order, a chunk's
        // combined row range is exactly [first group's start, last group's
        // end) — a single contiguous slice, no gather needed.
        let totalChunks = (groups.count + maxExpertsPerChunk - 1) / maxExpertsPerChunk
        var chunkResults: [MLXArray] = []
        chunkResults.reserveCapacity(totalChunks)

        var gi = 0
        while gi < groups.count {
            let end = min(gi + maxExpertsPerChunk, groups.count)
            let chunkGroups = groups[gi ..< end]
            let rowStart = chunkGroups.first!.range.lowerBound
            let rowEnd = chunkGroups.last!.range.upperBound
            let chunkExperts = chunkGroups.map { $0.expert }

            let xChunk = xSorted[rowStart ..< rowEnd]

            // Local id per row = position of its expert within this chunk's
            // expert list (0..<chunkExperts.count). Rows stay in the same
            // (sorted) order they appear in xSorted/idxSorted, so this array
            // is itself non-decreasing — safe to pass sortedIndices: true.
            var localIds = [Int32](repeating: 0, count: rowEnd - rowStart)
            for (localExpertIdx, group) in chunkGroups.enumerated() {
                for row in group.range {
                    localIds[row - rowStart] = Int32(localExpertIdx)
                }
            }
            let localIdxChunk = MLXArray(localIds)

            let fetched = fetchChunk(experts: chunkExperts)
            let chunkBytes = chunkExperts.reduce(0) { $0 + (fetched[$1]?.byteCount ?? 0) }
            let stackedGateW = MLX.stacked(chunkExperts.map { fetched[$0]!.gateWeight })
            let stackedGateS = MLX.stacked(chunkExperts.map { fetched[$0]!.gateScales })
            let stackedUpW = MLX.stacked(chunkExperts.map { fetched[$0]!.upWeight })
            let stackedUpS = MLX.stacked(chunkExperts.map { fetched[$0]!.upScales })
            let stackedDownW = MLX.stacked(chunkExperts.map { fetched[$0]!.downWeight })
            let stackedDownS = MLX.stacked(chunkExperts.map { fetched[$0]!.downScales })
            let stackedGateB = stackedBiasesIfPresent(chunkExperts, fetched) { $0.gateBiases }
            let stackedUpB = stackedBiasesIfPresent(chunkExperts, fetched) { $0.upBiases }
            let stackedDownB = stackedBiasesIfPresent(chunkExperts, fetched) { $0.downBiases }

            let gate = MLX.gatherQuantizedMM(
                xChunk, stackedGateW, scales: stackedGateS, biases: stackedGateB,
                rhsIndices: localIdxChunk, transpose: true,
                groupSize: groupSize, bits: bits, mode: mode, sortedIndices: true)
            let up = MLX.gatherQuantizedMM(
                xChunk, stackedUpW, scales: stackedUpS, biases: stackedUpB,
                rhsIndices: localIdxChunk, transpose: true,
                groupSize: groupSize, bits: bits, mode: mode, sortedIndices: true)
            let hidden = activationProduct(gate, up)
            let down = MLX.gatherQuantizedMM(
                hidden, stackedDownW, scales: stackedDownS, biases: stackedDownB,
                rhsIndices: localIdxChunk, transpose: true,
                groupSize: groupSize, bits: bits, mode: mode, sortedIndices: true)

            // Only force this chunk's compute now if NOT doing so risks a
            // real memory blowup -- see `shouldEvalChunk`. Every eval() is a
            // full GPU sync that stalls MLX's async command-buffer
            // pipeline; at decode (1 chunk/layer x 41 layers) an
            // unconditional eval() here was 41 forced round-trip syncs per
            // generated token.
            if Self.shouldEvalChunk(
                totalChunksInThisCall: totalChunks, chunkBytes: chunkBytes,
                evalThresholdBytes: evalThresholdBytes)
            {
                eval(down)
            }
            chunkResults.append(down)

            gi = end
        }

        let ySorted = chunkResults.count == 1 ? chunkResults[0] : concatenated(chunkResults, axis: 0)
        let y = scatterUnsort(x: ySorted, invOrder: inverseOrder, shape: indices.shape)
        return MLX.squeezed(y, axis: -2)
    }

    /// Whether to force-eval a chunk's compute immediately, rather than
    /// leaving it in MLX's lazy graph until the caller's own eventual
    /// eval() on the full model output.
    ///
    /// TWO independent reasons to eval, matching the two ways skipping it
    /// can blow up memory:
    ///
    /// 1. `totalChunksInThisCall > 1` (this forward call needed MULTIPLE
    ///    chunks — the common prefill case when a big batch touches more
    ///    unique experts than `maxExpertsPerChunk`). Without evaling each
    ///    chunk, EVERY chunk's stacked weight tensors for THIS SINGLE LAYER
    ///    stay resident simultaneously until the final eval — reintroducing
    ///    a resident-sized blowup within one layer, exactly what chunking
    ///    exists to avoid. This is unconditional: always eval when there's
    ///    a next chunk waiting to reuse that memory.
    ///
    /// 2. `chunkBytes > evalThresholdBytes` even for a call with only ONE
    ///    chunk. The default threshold (400 MiB) is comfortably above a
    ///    full `maxExpertsPerChunk`-sized (16) mxfp4 chunk (~218 MB) — the
    ///    common single-chunk DECODE case — so decode's one chunk per layer
    ///    is normally left un-evaluated. Across 41 MoE layers those
    ///    un-evaluated chunks accumulate in the lazy graph until the
    ///    generate loop's own eval, which is fine at ~41 x ~80 MB (a
    ///    typical topK=6 decode chunk) ≈ 3 GB of transient memory — but a
    ///    pathological single-chunk call with an unusually large unique-
    ///    expert count (e.g. continuous batching with many concurrent
    ///    decode sequences routing to many distinct experts, all still
    ///    fitting under `maxExpertsPerChunk`) could make that 41-layer
    ///    accumulation large enough to matter. The threshold is the safety
    ///    valve for that case: it forces an eval (and therefore a free)
    ///    once a single chunk alone is already big.
    static func shouldEvalChunk(
        totalChunksInThisCall: Int, chunkBytes: Int, evalThresholdBytes: Int
    ) -> Bool {
        totalChunksInThisCall > 1 || chunkBytes > evalThresholdBytes
    }

    /// Stack a chunk's per-expert bias arrays, or return nil if this
    /// projection has no biases at all (the mxfp4 routed-expert case).
    /// A checkpoint is expected to have biases uniformly present or
    /// uniformly absent across all experts of a given projection.
    private func stackedBiasesIfPresent(
        _ experts: [Int], _ fetched: [Int: ExpertWeights],
        _ pick: (ExpertWeights) -> MLXArray?
    ) -> MLXArray? {
        let biases = experts.compactMap { fetched[$0].flatMap(pick) }
        guard biases.count == experts.count, !biases.isEmpty else { return nil }
        return MLX.stacked(biases)
    }

    /// Resolve one chunk's experts through the cache, fetching misses from
    /// disk in parallel. Streaming failures (a missing tensor / short read)
    /// mean the checkpoint doesn't match the layout we parsed at load time —
    /// unrecoverable mid-generation, so this fails loudly rather than
    /// silently producing wrong logits.
    private func fetchChunk(experts: [Int]) -> [Int: ExpertWeights] {
        do {
            return try cache.fetch(layer: layerIndex, experts: experts, from: store)
        } catch {
            fatalError(
                "StreamingQuantizedSwitchGLU: failed to read expert weights for layer "
                    + "\(layerIndex) experts \(experts): \(error)")
        }
    }
}
