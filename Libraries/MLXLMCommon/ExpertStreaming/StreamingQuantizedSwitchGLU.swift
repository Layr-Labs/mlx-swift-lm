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

import Cmlx
import Foundation
import MLX

/// Pin the calling thread's MLX default streams to the process-global
/// thread-unsafe streams (`Stream.gpu` / `Stream.cpu` in mlx-swift's
/// Stream.swift).
///
/// WHY: MLX 0.32 made default streams thread-local (mlx/stream.cpp:
/// `default_stream_storage` is `thread_local`, lazily calling `new_stream`).
/// The first `eval()` on ANY thread therefore creates a fresh Stream(gpu, N)
/// whose Metal command encoder is registered only in that thread's
/// `thread_local` encoder map. `eval_impl` (mlx/transforms.cpp) consults
/// `default_stream(default_device())` for its synchronizer/event bookkeeping,
/// so state tagged with that per-thread stream can be observed by a LATER
/// eval on a DIFFERENT thread — which aborts with "There is no Stream(gpu, N)
/// in current thread" because neither that thread's encoder map nor the
/// global one knows the stream.
///
/// The resident engine never trips this: it only evals at token boundaries
/// through code built around the process-global streams (see the Stream.gpu
/// rationale in mlx-swift's Stream.swift). Expert streaming introduces
/// mid-forward `eval()` calls on whatever thread runs the forward — the
/// ModelContainer actor thread during prefill, the generate-loop task thread
/// during decode — so each such thread must have its default streams pinned
/// to the globals before its first eval. Idempotent and cheap (two
/// thread-local writes), called unconditionally per forward.
private func pinThreadDefaultStreamsToGlobal() {
    mlx_set_default_stream(StreamOrDevice.gpu.ctx)
    mlx_set_default_stream(StreamOrDevice.cpu.ctx)
}

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

    /// Max experts fetched/stacked into one `gatherQuantizedMM` call. Bounds
    /// peak transient memory for prefill batches that touch most/all of the
    /// routed experts in one forward pass — without this, stacking every
    /// unique expert touched into one `[256, out, in]`-shaped tensor before
    /// the matmul would briefly need a resident-sized tensor anyway, which
    /// defeats the point of streaming. Override via `DSV4_STREAM_CHUNK_EXPERTS`.
    private let maxExpertsPerChunk: Int

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
        maxExpertsPerChunk: Int? = nil
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
    }

    private static func chunkSizeFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_CHUNK_EXPERTS"],
            let n = Int(raw), n > 0
        {
            return n
        }
        return 16
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
        eval(idxSorted)
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

        // Bucket consecutive groups into fetch chunks. Because groups are
        // already in ascending-expert / ascending-row order, a chunk's
        // combined row range is exactly [first group's start, last group's
        // end) — a single contiguous slice, no gather needed.
        var chunkResults: [MLXArray] = []
        chunkResults.reserveCapacity((groups.count + maxExpertsPerChunk - 1) / maxExpertsPerChunk)

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

            // Force this chunk's compute now so its (potentially large)
            // stacked weight arrays are freed before the next chunk's fetch.
            // Without this, MLX's lazy graph keeps every chunk's stacked
            // tensors alive until the single eval() the caller eventually
            // runs on the model output — reintroducing the exact resident-
            // memory blowup streaming is meant to avoid.
            eval(down)
            chunkResults.append(down)

            gi = end
        }

        let ySorted = chunkResults.count == 1 ? chunkResults[0] : concatenated(chunkResults, axis: 0)
        let y = scatterUnsort(x: ySorted, invOrder: inverseOrder, shape: indices.shape)
        return MLX.squeezed(y, axis: -2)
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
