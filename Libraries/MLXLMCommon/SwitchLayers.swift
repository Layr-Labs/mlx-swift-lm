import Foundation
import MLX
import MLXNN

/// Identity gather table for the sorted 64-assignment decode geometry.
nonisolated(unsafe) private let switchDownIdentity64 = MLXArray((0..<64).map { UInt32($0) })

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

/// Compiled SiLU-gated product (`silu(gate) * up`) for the common MoE GLU path.
/// Fusing activation + product into one compiled, shapeless kernel cuts kernel
/// dispatches and intermediates on the hot decode path. Upstream ef85ed0.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on) like the sibling `compiledSwiGLU` / `safeGeluApproximate` fusions.
/// The default SiLU `SwitchGLU` path wires this in as `activationProduct` (the
/// highest-precedence branch in `callAsFunction`) and `LFM2MoE` calls it directly,
/// so without the gate both would keep hitting compiled kernels on the very M1/M2 +
/// macOS Tahoe machines the opt-out (MLX #3329) is meant to protect. Falls back to
/// the plain uncompiled closure when off; the default (env unset) stays compiled.
public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled weighted expert-output combine (`(outputs * weights[..., None]).sum(-2)`).
/// Shared by MoE routers (e.g. Gemma 4) to fuse the scale + reduce. Upstream ef85ed0.
public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}
/// Effective-selection count for the direct sorted-expert reduction. Benchmark
/// callers arm this after warmup and snapshot it only after the engine is idle.
/// The unarmed hot path reads one plain Bool and performs no atomic operation,
/// locking, allocation, or clock access.
public struct WeightedExpertUnsortStats: Sendable, Equatable {
    public let effectiveCalls: Int
}

/// Benchmark-facing requested/effective contract for one measured scope.
public struct WeightedExpertUnsortProvenance: Sendable, Equatable {
    public let requested: Bool
    public let effectiveCalls: Int

    public var engaged: Bool { effectiveCalls > 0 }
    public var missingExpectedEngagement: Bool { requested && !engaged }
}

private final class WeightedExpertUnsortProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var effectiveCalls = 0
    // Benchmark boundaries guarantee no engine work is in flight while this
    // plain flag changes. Concurrent recorders only read it while armed.
    private var enabled = false

    @inline(__always)
    func recordEffective() {
        guard enabled else { return }
        lock.lock()
        defer { lock.unlock() }
        // Defensively close a recorder/snapshot lock handoff. The idle-boundary
        // contract prevents a concurrent unsynchronized flag mutation.
        guard enabled else { return }
        effectiveCalls += 1
    }

    func snapshot() -> WeightedExpertUnsortStats {
        lock.lock()
        enabled = false
        defer { lock.unlock() }
        return WeightedExpertUnsortStats(effectiveCalls: effectiveCalls)
    }

    func reset() {
        lock.lock()
        effectiveCalls = 0
        enabled = true
        lock.unlock()
    }
}

private let weightedExpertUnsortProbe = WeightedExpertUnsortProbe()

/// Process-wide provenance snapshot for the weighted expert unsort experiment.
public func weightedExpertUnsortStats() -> WeightedExpertUnsortStats {
    weightedExpertUnsortProbe.snapshot()
}

/// Disarm and snapshot one benchmark scope with its resolved request state.
public func weightedExpertUnsortProvenance(
    requested: Bool
) -> WeightedExpertUnsortProvenance {
    let stats = weightedExpertUnsortStats()
    return WeightedExpertUnsortProvenance(
        requested: requested,
        effectiveCalls: stats.effectiveCalls)
}

/// Reset the provenance counters before a benchmark cell.
public func resetWeightedExpertUnsortStats() {
    weightedExpertUnsortProbe.reset()
}

/// Fused inverse-permutation + weighted reduction for the sorted MoE prefill path.
///
/// `SwitchGLU` sorts expert assignments before its gathered matrix multiplies.
/// The regular path restores `[tokens, topK, hidden]` and then reduces it with
/// ``weightedExpertSum``. This kernel reads those sorted rows through the inverse
/// permutation and writes `[tokens, hidden]` directly, avoiding that full
/// `[tokens, topK, hidden]` intermediate.
private let weightedExpertUnsortKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "weighted_expert_unsort_vec8_v3",
    inputNames: ["sorted_outputs", "inverse_order", "weights"],
    outputNames: ["output"],
    source: """
        typedef vec<T, 8> T8;
        // One lane owns eight consecutive features (128-bit load/store), so the
        // grid is an eighth as wide and each row read and store are one
        // eight-wide vector. The hidden extent is `threads_per_grid.x * 8u`,
        // which is 2816.
        uint oct = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;
        const uint hidden = threads_per_grid.x * 8u;

        T8 accumulator = T8((T)0);
        const uint assignment_base = token * (uint)K;
        for (uint slot = 0; slot < (uint)K; ++slot) {
            const uint assignment = assignment_base + slot;
            const uint sorted_row = (uint)inverse_order[assignment];
            const device T8* row = reinterpret_cast<const device T8*>(
                sorted_outputs + sorted_row * hidden);
            const T8 source = row[oct];
            const float weight = (float)weights[assignment];
            // Preserve the legacy bfloat16 multiply-then-reduce rounding.
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; ++j) {
                const T weighted = (T)((float)source[j] * weight);
                accumulator[j] = accumulator[j] + weighted;
            }
        }
        reinterpret_cast<device T8*>(output + token * hidden)[oct] =
            accumulator;
    """,
    ensureRowContiguous: true
)

/// Consume production-shaped sorted Gemma 4 expert rows through their inverse
/// permutation and reduce original top-K slots into `[tokens, hidden]`.
///
/// This primitive deliberately accepts only the production logical layout:
/// bfloat16 `[tokens * 8, 2816]`, uint32 inverse order, and bfloat16
/// `[tokens, 8]`. Callers must use the legacy scatter + weighted sum for every
/// other dtype, shape, or layout.
public func weightedExpertUnsort(
    sortedOutputs: MLXArray,
    inverseOrder: MLXArray,
    weights: MLXArray
) -> MLXArray {
    let hidden = sortedOutputs.dim(1)
    precondition(
        sortedOutputs.ndim == 2 && (hidden % 64 == 0)
            && sortedOutputs.dtype == .bfloat16,
        "weightedExpertUnsort outputs must be bfloat16 [assignments, hidden] with hidden % 64 == 0")
    precondition(
        inverseOrder.ndim == 1 && inverseOrder.dtype == .uint32,
        "weightedExpertUnsort inverse order must be flat uint32")
    precondition(
        weights.ndim == 2 && weights.dim(1) == 8 && weights.size >= 64
            && weights.dtype == .bfloat16,
        "weightedExpertUnsort weights must be sorted-prefill bfloat16 [tokens, 8]")
    precondition(
        sortedOutputs.dim(0) == weights.size && inverseOrder.size == weights.size,
        "weightedExpertUnsort assignment counts must match")

    let tokens = weights.dim(0)
    weightedExpertUnsortProbe.recordEffective()
    return weightedExpertUnsortKernel(
        [sortedOutputs, inverseOrder, weights],
        template: [
            ("T", sortedOutputs.dtype),
            ("K", 8),
        ],
        // vec8 kernel: one lane owns eight consecutive features, so the
        // x extent is hidden / 8 (the kernel derives hidden back as
        // threads_per_grid.x * 8). Kept generic in `hidden` rather than the
        // engine's hard-coded 2816 / 352 so non-Gemma callers still work.
        grid: (hidden / 8, tokens, 1),
        threadGroup: (32, 4, 1),
        outputShapes: [[tokens, hidden]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Exact sorted expert rows whose ordered top-K reduction is intentionally
/// deferred to a downstream fused consumer.
///
/// Keeping this carrier explicit prevents generic callers from mistaking
/// `[assignments, hidden]` for the already-reduced `[tokens, hidden]` result.
public struct DeferredWeightedExpertRows {
    public let sortedOutputs: MLXArray
    public let inverseOrder: MLXArray
    public let weights: MLXArray

    init(sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray) {
        self.sortedOutputs = sortedOutputs
        self.inverseOrder = inverseOrder
        self.weights = weights
    }
}

/// Materialize a deferred carrier through the established reduction. Used only
/// when a downstream fused consumer declines after the producer was selected.
public func resolveDeferredWeightedExpertRows(
    _ rows: DeferredWeightedExpertRows
) -> MLXArray {
    weightedExpertUnsort(
        sortedOutputs: rows.sortedOutputs,
        inverseOrder: rows.inverseOrder,
        weights: rows.weights)
}


// MARK: - Compiled activation fusions (vMLX / osaurus-main port)

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`). The Power primitive returns zero results under the
/// macOS Tahoe Metal JIT (MLX #3329), so the explicit multiplies keep it safe
/// under `compile(shapeless: true)`. Numerically identical to
/// `MLXNN.geluApproximate`.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on); falls back to the plain closure when compiled fusions are off.
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Drop-in replacement for `MLXNN.GELU(approximation: .tanh)` that avoids the
/// Power primitive crash. Use anywhere a tanh-approx GELU unary layer is needed.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Compiled SiLU-gated GLU product (`silu(gate) * up`). Same math as
/// `compiledSiluProduct` above, but gated by `MLXHardwareInfo` so M1/M2 + macOS
/// Tahoe can opt out. Used by `SwitchGLU` when a SiLU activation is supplied via
/// the custom-activation initializer (where `activationProduct` is nil).
private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled GELU-gated GLU product (`geluApprox(gate) * up`), fusing the tanh
/// GELU and the element-wise multiply into one shapeless kernel. Uses the
/// Power-free `x * x * x` GELU so it is safe under `compile(shapeless: true)`.
private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// GELU-FUSE: the SAME body, compiled WITHOUT `shapeless`, for the routed
/// expert's pinned decode signatures only. Shapeless tracing adds broadcast
/// nodes on every binary op that a shape-specialised trace omits on equal
/// shapes; those nodes push this expression past MLX's fusion depth limit and
/// split it into two Metal kernels with a materialised intermediate. The
/// shape-specialised trace fits and emits one.
private let compiledGeGLUShaped: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(body)
    }
    return body
}()

/// GELU-FUSE-PREFILL: a bounded set of additionally admitted shapes.
///
/// GELU-FUSE left prefill on the shapeless closure for one stated reason — a
/// shape-specialised compile adds a compiler-cache entry per distinct input
/// shape, the lookup is a linear scan, and prefill row counts vary per prompt,
/// so an unbounded admission would keep growing the scan the decode hot path
/// walks. That reason is about the *number* of entries, not about prefill, so a
/// hard cap answers it directly: at most ``shapedGeluPrefillShapeCap`` distinct
/// rectangles are ever admitted, and the cap+1st falls open to the shapeless
/// closure forever after.
///
/// The decode signatures are matched before this is consulted, so the decode
/// plane never takes the lock and never sees a behaviour change.
public final class ShapedGeluPrefillShapes: @unchecked Sendable {
    private let lock = NSLock()
    private var shapes: [[Int]] = []
    private let cap: Int

    public init(cap: Int) { self.cap = cap }

    @inline(__always)
    public func admits(_ shape: [Int]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if shapes.contains(shape) { return true }
        guard shapes.count < cap else { return false }
        shapes.append(shape)
        return true
    }
}

/// Four is one more than the distinct rectangles a cohort prefill produces (the
/// full batched step, a short final chunk, and the single-stream local verb).
public let shapedGeluPrefillShapeCap = 4

/// Smallest rectangle worth a cache entry. The prefill routed-expert plane is
/// 65,536 rows; every speculative verify width is at most 256, so nothing in
/// production lands near this floor from either side.
public let shapedGeluPrefillMinElements = 1 << 20

private let switchGeluPrefillFuseEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GELU_SHAPED_FUSE_PREFILL"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let switchGeluPrefillShapes = ShapedGeluPrefillShapes(
    cap: shapedGeluPrefillShapeCap)

@inline(__always)
private func geGLUClaimsPrefill(_ gate: MLXArray, _ up: MLXArray) -> Bool {
    guard switchGeluPrefillFuseEnabled,
        gate.dtype == .bfloat16, up.dtype == .bfloat16,
        gate.shape == up.shape,
        gate.size >= shapedGeluPrefillMinElements,
        switchGeluPrefillShapes.admits(gate.shape)
    else { return false }
    CBv2EngageMark.once("gelu-shaped-prefill-experts")
    return true
}

@inline(__always)
private func geGLUProduct(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    if geGLUClaimsPrefill(gate, up) {
        return compiledGeGLUShaped(gate, up)
    }
    return compiledGeGLU(gate, up)
}

private let routeSortTile64 = 64
/// Key-space bound of the fused scatter's 256-entry counter table.
let routeCountingSortKeyBound = 256

// MARK: - PREFILL-CSORT-128 (general-geometry exact stable counting sort)

/// Exact stable counting sort for the GENERAL MoE route geometry — the
/// prefill/verification tables that ROUTE-CSORT-64 refuses (it is retiled for
/// the n = 64 eight-row decode cohort and pays an O(n) rescan per tile).
///
/// Where the census puts it: in the packed 8x1024 prefill window MLX's generic
/// `argSort` over the flattened route table (`partition_mbsort` +
/// `merge_mbsort`, ~10 dispatches per layer x 30 layers, twice — once for
/// `order`, once for the inverse) costs 392.6 ms of 5508 ms on the M4 (7.2%).
/// Sorts are latency/memory bound, so they do not shrink with the ranked box's
/// NAX GEMM speedup: the same census projects the ROUTE bucket to 19-36% of the
/// sealed 1.254 s M5 prefill window. This lane deletes the sort, not shrinks it.
///
/// Three dispatches, no comparisons:
///   1. `_hist_v1`    — one threadgroup per 256-key block builds a 256-entry
///                      threadgroup histogram (commutative integer atomics) and
///                      writes it to `H[block][key]`.
///   2. `_scan_v1`    — ONE threadgroup: thread `e` sums `H[.][e]` over blocks
///                      to get `total[e]`, a simd exclusive prefix over the 256
///                      totals gives the global bin base `base[e]`, and a second
///                      pass writes `O[block][e] = base[e] + sum_{b<block} H[b][e]`.
///   3. `_scatter_v1` — one threadgroup per block stages the block's 256 keys in
///                      threadgroup memory; thread `k` counts how many earlier
///                      keys IN ITS OWN BLOCK carry its key (`rank`) and lands at
///                      `pos = O[block][key] + rank`.
///
/// Exactness (why this is `argSort`-identical, not merely equal on tests): for
/// the key at flat index `idx` in block `b`, `O[b][key] + rank` is by
/// construction `#{keys with a smaller expert} + #{equal keys at a smaller flat
/// index}`, which is exactly the rank of `idx` under a STABLE sort by key. The
/// vendored merge argsort is stable at every stage (thread sort swaps only on
/// strictly-less, the merge prefers A on ties), so its tie order is input order
/// too — the two permutations agree for EVERY input, not just tested ones.
/// At the single write point every downstream index product is already known:
/// `idx / m` is `order.floorDivide(m)`, the key IS `indices[order]`, and `pos`
/// is the inverse-permutation entry for `idx` (`argSort(order)`), so three
/// dispatches replace `argSort` -> `floorDivide` -> take -> `argSort` with
/// byte-identical integer outputs and every consumer (the `gather_qmm`
/// `rhsIndices`/`lhsIndices`, `weightedExpertUnsort`, `scatterUnsort`) is
/// untouched.
///
/// The counter table is a fixed 256 entries wide regardless of `numExperts`, so
/// no expert count is baked into any kernel: bins above `numExperts` simply hold
/// zero and contribute nothing to the bases. Callers must still prove keys are
/// below that width via the `numExperts` guard (`routeCountingSortKeyBound`).
///
/// Every kernel indexes its inputs linearly, so all three ask MLX for
/// `ensureRowContiguous` — free for the contiguous route tables production
/// actually hands us (MLX skips the copy when the flag is already set) and a
/// hard guarantee for anything else that ever reaches this helper.
///
/// Kill switch: `DARKBLOOM_ROUTE_CSORT_PREFILL` set to `0`/`false`/`no`/`off`
/// restores the `argSort` chain. Engage mark: `route-csort-prefill`.
private let routeCsortPrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_ROUTE_CSORT_PREFILL"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// Keys per histogram/scatter block.
private let routeCsortPrefillBlock = 256
/// Counter-table width; must equal ``routeCountingSortKeyBound`` and the 256
/// threads per threadgroup the three kernels launch with.
private let routeCsortPrefillWidth = 256
/// Largest `n` accepted. Positions, block offsets and grid extents are uint32 /
/// Int32 on the Metal side; this bound keeps every one of them representable
/// with room to spare and is ~4000x the largest production route table.
private let routeCsortPrefillMaxKeys = 1 << 28

private let routeCsortPrefillHistKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort128_hist_v1",
    inputNames: ["keys"],
    outputNames: ["block_hist"],
    source: """
        constexpr uint BLOCK = \(routeCsortPrefillBlock);
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint b = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        uint n = keys_shape[0];
        threadgroup atomic_uint tg_count[WIDTH];
        atomic_store_explicit(&tg_count[k], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint idx = b * BLOCK + k;
        if (idx < n) {
            atomic_fetch_add_explicit(
                &tg_count[keys[idx]], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Integer adds commute, so the table is identical for every
        // interleaving the hardware picks.
        block_hist[b * WIDTH + k] =
            atomic_load_explicit(&tg_count[k], memory_order_relaxed);
        """,
    ensureRowContiguous: true
)

private let routeCsortPrefillScanKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort128_scan_v3",
    inputNames: ["block_hist"],
    outputNames: ["block_offset"],
    source: """
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint e = thread_position_in_threadgroup.x;
        uint simd_id = e / 32;
        uint lane = e % 32;
        uint nblocks = (uint)block_hist_shape[0];
        uint total = 0u;
        // Admission proves every key is below NE, so columns at or above it are
        // zero in every block and cannot contribute to the total. Skipping the
        // accumulation retires whole SIMD groups at once when the counter table
        // is wider than the model's expert count.
        if (e < (uint)NE) {
            for (uint b = 0; b < nblocks; ++b) {
                total += block_hist[b * WIDTH + e];
            }
        }
        // Global bin base: exclusive prefix over the 256 expert totals.
        uint lane_excl = simd_prefix_exclusive_sum(total);
        threadgroup uint simd_totals[8];
        if (lane == 31) {
            simd_totals[simd_id] = lane_excl + total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = 0u;
        for (uint s = 0; s < simd_id; ++s) {
            running += simd_totals[s];
        }
        running += lane_excl;
        // Exclusive scan over blocks for this expert, offset by the bin base.
        // Column `e` of `block_offset` is read by the scatter only as
        // `block_offset[b * WIDTH + key]` for a key that occurs in block `b`,
        // so a column whose global total is zero is never read and need not be
        // written. The counter table is 256 wide while the model routes 128
        // experts, so at minimum half the columns are unconditionally dead.
        if (total > 0u) {
            for (uint b = 0; b < nblocks; ++b) {
                block_offset[b * WIDTH + e] = running;
                running += block_hist[b * WIDTH + e];
            }
        }
        """,
    ensureRowContiguous: true
)

private let routeCsortPrefillScatterKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort128_scatter_v1",
    inputNames: ["keys", "block_offset"],
    outputNames: ["row_order", "sorted_keys", "inverse_order"],
    source: """
        constexpr uint BLOCK = \(routeCsortPrefillBlock);
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint b = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        uint n = keys_shape[0];
        uint idx = b * BLOCK + k;
        // Tail block: the sentinel is outside the proven key space (keys are
        // below the 256-wide counter table), so it can never tie a real key.
        uint key = (idx < n) ? keys[idx] : 0xffffffffu;
        threadgroup uint tg_keys[BLOCK];
        tg_keys[k] = key;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < n) {
            // Stable local rank: earlier keys in this block only. Read in
            // index order from threadgroup memory, so no write position ever
            // depends on scheduling.
            uint rank = 0u;
            for (uint j = 0; j < k; ++j) {
                rank += (tg_keys[j] == key) ? 1u : 0u;
            }
            uint pos = block_offset[b * WIDTH + key] + rank;
            row_order[pos] = idx / (uint)M;
            sorted_keys[pos] = key;
            inverse_order[idx] = pos;
        }
        """,
    ensureRowContiguous: true
)

/// Exact stable counting sort of a flat uint32 route table. Returns nil (fail
/// closed onto `argSort`) unless every precondition of the kernels holds.
private func routeCountingSortPrefill(
    _ indices: MLXArray, m: Int, numExperts: Int
) -> (rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray)? {
    let n = indices.size
    guard routeCsortPrefillEnabled,
        indices.dtype == .uint32,
        indices.ndim == 1,
        numExperts > 0,
        numExperts <= routeCsortPrefillWidth,
        numExperts <= routeCountingSortKeyBound,
        m >= 1,
        n > routeSortTile64,
        n <= routeCsortPrefillMaxKeys
    else { return nil }
    CBv2EngageMark.once("route-csort-prefill")
    let blocks = (n + routeCsortPrefillBlock - 1) / routeCsortPrefillBlock
    let width = routeCsortPrefillWidth
    let hist = routeCsortPrefillHistKernel(
        [indices],
        grid: (blocks * width, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [[blocks, width]],
        outputDTypes: [.uint32]
    )[0]
    let offsets = routeCsortPrefillScanKernel(
        [hist],
        template: [("NE", numExperts)],
        grid: (width, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [[blocks, width]],
        outputDTypes: [.uint32]
    )[0]
    let outputs = routeCsortPrefillScatterKernel(
        [indices, offsets],
        template: [("M", m)],
        grid: (blocks * width, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [[n], [n], [n]],
        outputDTypes: [.uint32, .uint32, .uint32]
    )
    return (outputs[0], outputs[1], outputs[2])
}

/// GLUE-FOLD carrier: the exact decode route table (`row_order`,
/// `sorted_keys`, `inverse_order`, each `[64]` uint32) computed upstream by a
/// producer that already holds the router scores, so `projectExperts` never
/// issues its standalone route-table dispatch. The arrays must be exactly what
/// `gatherSortIndices` would have produced for the same `[8, 8]` indices --
/// raw (untagged) sorted expert keys included -- and any shape, dtype or
/// switch-state mismatch declines the carrier and re-issues the incumbent
/// chain unchanged.
public struct SwitchRouteTable {
    public let rowOrder: MLXArray
    public let sortedKeys: MLXArray
    public let inverseOrder: MLXArray

    public init(rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray) {
        self.rowOrder = rowOrder
        self.sortedKeys = sortedKeys
        self.inverseOrder = inverseOrder
    }
}

/// `numExperts` is the exclusive upper bound of the index key space. Callers
/// that know it (SwitchGLU) pass it so PREFILL-CSORT-128 can prove its 256-entry
/// counter table covers every key; the default (`Int.max`) fails closed onto the
/// established `argSort` chain, which is what the generic MoE models that share
/// this helper (GPTOSS, NemotronH) keep getting.
public func gatherSort(
    x: MLXArray, indices: MLXArray, numExperts: Int = Int.max
) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    // PREFILL-CSORT-128: three dispatches with byte-identical outputs.
    if let fused = routeCountingSortPrefill(indices, m: m, numExperts: numExperts) {
        return (
            x.flattened(start: 0, end: -3)[fused.rowOrder],
            fused.sortedKeys,
            fused.inverseOrder
        )
    }
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

/// PRENORM-GATHER: the sort of `gatherSort` without its gather. Returns the
/// token row of every sorted position, the sorted expert keys and the inverse
/// order, derived exactly as `gatherSort` derives them (the PREFILL-CSORT-128
/// kernels when they admit, the `argSort` chain otherwise), so a producer that
/// knows the inverse order can emit the sorted plane itself and the standalone
/// gather of `x` is never issued.
public func gatherSortOrder(
    indices: MLXArray, numExperts: Int = Int.max
) -> (rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    if let fused = routeCountingSortPrefill(indices, m: m, numExperts: numExperts) {
        return (fused.rowOrder, fused.sortedKeys, fused.inverseOrder)
    }
    let order = argSort(indices)
    let inverseOrder = argSort(order)
    return (order.floorDivide(m), indices[order], inverseOrder)
}

/// PRENORM-GATHER: a producer that emits the sorted expert plane
/// `[assignments, 1, inputDims]` directly from the sort's inverse order, so
/// `SwitchGLU` never issues its standalone gather of the activations it was
/// handed. Returning `nil`, or a plane of any other shape or dtype, selects
/// that gather.
public typealias SwitchSortedPlaneProducer = (_ inverseOrder: MLXArray) -> MLXArray?

/// `numExperts` is the exclusive upper bound of the index key space; callers
/// that know it (SwitchGLU) pass it so the counting-sort fast path can prove
/// its 256-entry counter table covers every key. The default (`Int.max`)
/// fails closed onto the established `argSort` chain.
public func gatherSortIndices(
    indices: MLXArray, numExperts: Int = Int.max,
    expertPrefixBounds: Bool = false
) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    // PREFILL-CSORT-128 owns everything wider than the retiled decode cohort.
    if numExperts <= routeCountingSortKeyBound,
        let fused = routeCountingSortPrefill(indices, m: m, numExperts: numExperts)
    {
        return (fused.rowOrder, fused.sortedKeys, fused.inverseOrder)
    }
    let order = argSort(indices)
    return (order.floorDivide(m), indices[order], argSort(order))
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

private let qwenDirectExpertReductionEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["MLX_QWEN_DIRECT_EXPERT_REDUCTION"]?
        .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return raw == "1" || raw == "true" || raw == "on"
}()

// MARK: - SwitchGLU

/// Semantic profile required by the exact Gemma direct-reduction experiment.
/// Generic SwitchGLU instances never infer production eligibility from a
/// one-point activation probe.
public enum SwitchGLUWeightedReductionProfile: Sendable {
    case generic
    case gemma4ProductionGeGLU
    case qwen35ProductionSwiGLU
}


/// Safely row-concatenates split SwitchGLU gate/up checkpoint tensors.
///
/// This is the model-neutral core of the Qwen3.5 gate/up load-time fusion:
/// every tensor suffix must match in shape and dtype, and both module paths
/// must resolve to one quantization policy. An incompatible pair is left
/// byte-for-byte split and reported through `setFused`, allowing the caller to
/// install a split ``SwitchGLU`` before strict parameter verification.
public func fuseSwitchGLUGateUpWeights(
    weights: [String: MLXArray],
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    quantizationAliases: (String) -> [String] = { _ in [] },
    shouldProcess: (String) -> Bool = { _ in true },
    setFused: ((String, Bool) -> Void)? = nil
) -> [String: MLXArray] {
    var weights = weights
    let splitMarker = ".switch_mlp.gate_proj."
    var bases = Set<String>()
    for key in weights.keys where key.contains(splitMarker) {
        let range = key.range(of: splitMarker)!
        let base = String(key[..<range.lowerBound]) + ".switch_mlp."
        if shouldProcess(String(base.dropLast())) {
            bases.insert(base)
        }
    }

    func resolvedQuantization(
        for path: String,
        in table: BaseConfiguration.PerLayerQuantization
    ) -> BaseConfiguration.Quantization? {
        var candidates = [path]
        for alias in quantizationAliases(path) where !candidates.contains(alias) {
            candidates.append(alias)
        }
        for candidate in candidates {
            guard let option = table.perLayerQuantization[candidate] else { continue }
            switch option {
            case .skip:
                return nil
            case .quantize(let quantization):
                return quantization
            }
        }
        return table.quantization
    }

    func suffixes(_ half: String, base: String) -> Set<String> {
        let prefix = "\(base)\(half)."
        return Set(
            weights.keys.compactMap {
                $0.hasPrefix(prefix) ? String($0.dropFirst(prefix.count)) : nil
            })
    }

    for base in bases.sorted() {
        let modulePath = String(base.dropLast())
        let gateSuffixes = suffixes("gate_proj", base: base)
        let upSuffixes = suffixes("up_proj", base: base)
        guard !upSuffixes.isEmpty else {
            // Leave malformed half-pairs untouched so strict update reports a
            // catchable missing/unhandled-parameter error.
            continue
        }

        var blocker: String?
        if let table = perLayerQuantization {
            let gate = resolvedQuantization(for: "\(base)gate_proj", in: table)
            let up = resolvedQuantization(for: "\(base)up_proj", in: table)
            let sameEffectivePolicy: Bool
            switch (gate, up) {
            case (nil, nil):
                sameEffectivePolicy = true
            case (let gate?, let up?):
                sameEffectivePolicy =
                    gate.groupSize == up.groupSize && gate.bits == up.bits
                    && gate.mode == up.mode
            default:
                sameEffectivePolicy = false
            }
            if !sameEffectivePolicy {
                blocker = "gate and up resolve to different quantization policies"
            }
        }
        if blocker == nil, gateSuffixes != upSuffixes {
            blocker = "gate and up carry different tensor sets"
        }
        if blocker == nil {
            for suffix in gateSuffixes.sorted() {
                guard let gate = weights["\(base)gate_proj.\(suffix)"],
                    let up = weights["\(base)up_proj.\(suffix)"]
                else { continue }
                if gate.shape != up.shape || gate.dtype != up.dtype {
                    blocker = "\(suffix) tensors differ in shape or dtype"
                    break
                }
            }
        }

        if let blocker {
            print("[INFO] fuseSwitchGLUGateUpWeights: keeping \(modulePath) split — \(blocker)")
            setFused?(modulePath, false)
            continue
        }

        for suffix in gateSuffixes.sorted() {
            guard let gate = weights.removeValue(forKey: "\(base)gate_proj.\(suffix)"),
                let up = weights.removeValue(forKey: "\(base)up_proj.\(suffix)")
            else { continue }
            let axis = suffix == "bias" ? -1 : -2
            weights["\(base)gate_up_proj.\(suffix)"] = concatenated([gate, up], axis: axis)
        }
    }

    if let setFused {
        let fusedMarker = ".switch_mlp.gate_up_proj."
        var fusedPaths = Set<String>()
        for key in weights.keys where key.contains(fusedMarker) {
            let range = key.range(of: fusedMarker)!
            let path = String(key[..<range.lowerBound]) + ".switch_mlp"
            if shouldProcess(path) {
                fusedPaths.insert(path)
            }
        }
        for path in fusedPaths.sorted() {
            setFused(path, true)
        }
    }
    return weights
}

/// Replaces the SwitchGLU at a checkpoint path with its fused or split twin.
/// Aliases bridge checkpoint and module-tree namespaces.
public func setSwitchGLUGateUpFused(
    _ fused: Bool,
    at path: String,
    aliases: [String] = [],
    in root: Module
) {
    let candidates = [path] + aliases.filter { $0 != path }
    for (modulePath, module) in root.namedModules() {
        guard candidates.contains(modulePath), let glu = module as? SwitchGLU else {
            continue
        }
        if glu.hasFusedGateUp != fused {
            let twin = fused ? glu.fusingGateUp() : glu.splittingGateUp()
            root.update(modules: ModuleChildren.unflattened([(modulePath, twin)]))
        }
        return
    }
}

/// Inputs retained from the direct sorted expert reduction so a downstream
/// prefill kernel can consume the sorted rows without materializing the
/// intermediate `[tokens, hidden]` reduction.
public struct WeightedExpertUnsortCarrier {
    let sortedOutputs: MLXArray
    let inverseOrder: MLXArray
    let weights: MLXArray
}

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    /// Optional fused (activation * up) kernel. Set for the default SiLU path so
    /// the GLU product runs as one compiled op; nil when a custom activation is
    /// supplied (we then fall back to `activation(gate) * up`). Upstream ef85ed0.
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let weightedReductionProfile: SwitchGLUWeightedReductionProfile

    /// Activation-type flags detected once at init from a tiny test input (vMLX
    /// approach — no per-token check). Only consulted when `activationProduct` is
    /// nil (the custom-activation path): they let SiLU/GELU custom activations use
    /// the compiled `compiledSwiGLU` / `compiledGeGLU` fusions instead of the
    /// uncompiled `activation(gate) * up`. On any mismatch we fall back to that
    /// exact uncompiled path, so detection only ever enables a numerically
    /// equivalent fast path — it can never change results.
    let isSiluActivation: Bool
    let isGeluActivation: Bool

    /// Default SiLU GLU path -- uses the compiled fused (silu * up) kernel.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        fuseGateUp: Bool = false,
        weightedReductionProfile: SwitchGLUWeightedReductionProfile = .generic
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct
        self.weightedReductionProfile = weightedReductionProfile
        // Default path is SiLU and `activationProduct` is non-nil, so these are
        // not consulted on the hot path; set them accurately for completeness
        // (and to avoid a needless probe eval at load for every MoE layer).
        self.isSiluActivation = true
        self.isGeluActivation = false

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2,
                numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Custom-activation GLU path -- runs `activation(gate) * up` uncompiled.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false,
        fuseGateUp: Bool = false,
        weightedReductionProfile: SwitchGLUWeightedReductionProfile = .generic
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil
        self.weightedReductionProfile = weightedReductionProfile
        // Detect SiLU/GELU once via a tiny test input (vMLX approach) so the hot
        // path can select the compiled fusion without a per-token check. Exact
        // equality is intentional: a match means the supplied closure computes
        // that exact function; any non-match falls back to `activation(gate) * up`
        // in callAsFunction, so this can only ever enable an equivalent fast path.
        let probe = MLXArray([Float(1.0)])
        let probeOut = activation(probe)
        let detectedSilu = (probeOut .== MLXNN.silu(probe)).all().item(Bool.self)
        self.isSiluActivation = detectedSilu
        self.isGeluActivation =
            !detectedSilu && (probeOut .== safeGeluApproximate(probe)).all().item(Bool.self)

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2,
                numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// True while the routed gate/up projection uses the fused
    /// `gate_up_proj` layout built by `fuseGateUp: true`.
    public var hasFusedGateUp: Bool { gateUpProj != nil }

    /// Structural twin with the requested gate/up topology: same
    /// dims/bias/activation/profile, freshly initialized gate/up
    /// projection(s) (the caller's subsequent quantize + strict update
    /// supplies their tensors), `down_proj` carried over.
    ///
    /// Used because the gate/up fusion is a per-load, per-layer decision: a
    /// heterogeneous checkpoint (different gate vs up quantization policies)
    /// must load through split modules, and a later homogeneous load on the
    /// same model instance must be able to restore the fused layout.
    private init(copying other: SwitchGLU, fusedGateUp: Bool) {
        self.inputDims = other.inputDims
        self.hiddenDims = other.hiddenDims
        self.numExperts = other.numExperts
        self.activation = other.activation
        self.activationProduct = other.activationProduct
        self.weightedReductionProfile = other.weightedReductionProfile
        self.isSiluActivation = other.isSiluActivation
        self.isGeluActivation = other.isGeluActivation

        let bias = (other.gateUpProj ?? other.gateProj)?.bias != nil
        if fusedGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: other.inputDims, outputDims: other.hiddenDims * 2,
                numExperts: other.numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: other.inputDims, outputDims: other.hiddenDims,
                numExperts: other.numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: other.inputDims, outputDims: other.hiddenDims,
                numExperts: other.numExperts, bias: bias)
        }
        self._downProj.wrappedValue = other.downProj

        super.init()
    }

    /// Returns the split twin described by `init(copying:fusedGateUp:)`.
    /// Swap it in with `Module.update(modules:)`; direct property assignment
    /// would not refresh the module cache.
    public func splittingGateUp() -> SwitchGLU {
        SwitchGLU(copying: self, fusedGateUp: false)
    }

    /// Returns the fused twin described by `init(copying:fusedGateUp:)` —
    /// the inverse of ``splittingGateUp()``, restoring the fused layout when
    /// a homogeneous checkpoint is loaded onto a previously split instance.
    public func fusingGateUp() -> SwitchGLU {
        SwitchGLU(copying: self, fusedGateUp: true)
    }

    private func projectExperts(
        _ x: MLXArray, _ indices: MLXArray,
        sortedPlane: SwitchSortedPlaneProducer? = nil,
        routeTable: SwitchRouteTable? = nil
    ) -> (output: MLXArray, inverseOrder: MLXArray?, sorted: Bool) {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder: MLXArray? = nil
        let lhsIndices: MLXArray? = nil
        if doSort {
            (x, idx, inverseOrder) = gatherSort(
                x: x, indices: indices, numExperts: numExperts)
        }

        let xGate: MLXArray
        let xUp: MLXArray
        if let gateUpProj {
            let xGateUp = gateUpProj(
                x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else {
            guard let gateProj, let upProj else {
                preconditionFailure("SwitchGLU requires gate_up_proj or gate_proj/up_proj")
            }
            xUp = upProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
            xGate = gateProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
        }

        let activated: MLXArray
        if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = geGLUProduct(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        // DOWN-LHS-IDENTITY: at the sorted [64] geometry the down projection
        // gathers activation row `assignment` for assignment `assignment`;
        // hand it that identity table instead of leaving `lhsIndices` nil,
        // which otherwise materializes the same arange(64) on every call.
        let downLhs: MLXArray? =
            (doSort && idx.ndim == 1 && idx.size == 64) ? switchDownIdentity64 : nil
        x = downProj(activated, idx, lhsIndices: downLhs, sortedIndices: doSort)
        // Under `doSort` a producer above always assigned `inverseOrder`;
        // otherwise it is still nil, which is exactly what the old
        // `doSort ? inverseOrder : nil` produced.
        return (x, inverseOrder, doSort)
    }

    private func legacyWeightedReduction(
        _ projected: (output: MLXArray, inverseOrder: MLXArray?, sorted: Bool),
        indices: MLXArray,
        weights: MLXArray
    ) -> MLXArray {
        var output = projected.output
        if let inverseOrder = projected.inverseOrder {
            output = scatterUnsort(x: output, invOrder: inverseOrder, shape: indices.shape)
        }
        return weightedExpertSum(MLX.squeezed(output, axis: -2), weights)
    }

    private func supportsWeightedExpertUnsort(
        _ x: MLXArray, _ indices: MLXArray, weights: MLXArray
    ) -> Bool {
        switch weightedReductionProfile {
        case .generic:
            return false
        case .gemma4ProductionGeGLU:
            return inputDims == 2816
                && hiddenDims == 704
                && numExperts == 128
                && gateUpProj == nil
                && activationProduct == nil
                && isGeluActivation
                && x.ndim == 2
                && x.dim(1) == 2816
                && x.dtype == .bfloat16
                && indices.ndim == 2
                && indices.dim(0) == x.dim(0)
                && indices.dim(1) == 8
                && indices.dtype == .uint32
                && weights.ndim == 2
                && weights.shape == indices.shape
                && weights.dtype == .bfloat16
                && indices.size >= 64
        case .qwen35ProductionSwiGLU:
            return qwenDirectExpertReductionEnabled
                && inputDims == 2048
                && hiddenDims == 512
                && numExperts == 256
                && isSiluActivation
                && x.ndim == 2
                && x.dim(1) == 2048
                && x.dtype == .bfloat16
                && indices.ndim == 2
                && indices.dim(0) == x.dim(0)
                && indices.dim(1) == 8
                && indices.dtype == .uint32
                && weights.ndim == 2
                && weights.shape == indices.shape
                && weights.dtype == .bfloat16
                && indices.size >= 64
        }
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray,
        sortedPlane: SwitchSortedPlaneProducer? = nil
    ) -> MLXArray {
        var projected = projectExperts(x, indices, sortedPlane: sortedPlane)
        if let inverseOrder = projected.inverseOrder {
            projected.output = scatterUnsort(
                x: projected.output, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(projected.output, axis: -2)
    }

    /// Preserve the promoted gathered down projection and defer only its
    /// inverse-permutation + weighted top-K reduction to a downstream consumer.
    ///
    /// This is decode-only and exact-geometry-only. Returning nil leaves
    /// ``callAndWeightedReduce`` as the complete established fallback.
    public func callAndDeferWeightedReduce(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool,
        isProductionPrefill: Bool = true,
        routeTable: SwitchRouteTable? = nil
    ) -> DeferredWeightedExpertRows? {
        let isEightRowDecode =
            !isProductionPrefill && x.dim(0) == 8 && indices.size == 64
        guard fuseSortedReduction && isEightRowDecode,
            supportsWeightedExpertUnsort(x, indices, weights: weights)
        else { return nil }

        let projected = projectExperts(x, indices, routeTable: routeTable)
        guard projected.sorted,
            let inverseOrder = projected.inverseOrder,
            projected.output.ndim == 3,
            projected.output.dim(-2) == 1,
            projected.output.dim(-1) == 2816,
            projected.output.dtype == .bfloat16
        else { return nil }

        weightedExpertUnsortProbe.recordEffective()
        return DeferredWeightedExpertRows(
            sortedOutputs: MLX.squeezed(projected.output, axis: -2),
            inverseOrder: inverseOrder,
            weights: weights)
    }

    /// Always-called expert projection + weighted reduction entry point.
    ///
    /// When the experiment is enabled, the exact sorted production Gemma
    /// prefill contract and the exact eight-row decode cohort reduce directly
    /// to `[tokens, hidden]`. Smaller decode cohorts, rectangular speculative
    /// verification, generic/custom-activation, dtype/layout, and near-geometry
    /// calls retain scatter/unsort followed by ``weightedExpertSum``.
    public func callAndWeightedReduce(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool,
        isProductionPrefill: Bool = true
    ) -> MLXArray {
        callAndWeightedReduceWithUnsortCarrier(
            x,
            indices,
            weights: weights,
            fuseSortedReduction: fuseSortedReduction,
            isProductionPrefill: isProductionPrefill
        ).output
    }

    /// The direct reduction plus its already-sorted inputs. Generic and decode
    /// paths return no carrier and preserve the established output graph.
    public func callAndWeightedReduceWithUnsortCarrier(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool,
        isProductionPrefill: Bool = true,
        sortedPlane: SwitchSortedPlaneProducer? = nil
    ) -> (output: MLXArray, carrier: WeightedExpertUnsortCarrier?) {
        // At B=8 decode there are exactly 64 assignments (8 rows x top-k 8),
        // which is the sorting threshold and the minimum geometry accepted by
        // weightedExpertUnsort. Keep the decode gate exact so MTP rectangles
        // and smaller serving cohorts remain on their established reduction.
        let isEightRowDecode =
            !isProductionPrefill && x.dim(0) == 8 && indices.size == 64
        guard fuseSortedReduction && (isProductionPrefill || isEightRowDecode),
            supportsWeightedExpertUnsort(x, indices, weights: weights)
        else {
            return (
                weightedExpertSum(
                    callAsFunction(x, indices, sortedPlane: sortedPlane), weights),
                nil
            )
        }

        let projected = projectExperts(x, indices, sortedPlane: sortedPlane)
        guard projected.sorted,
            let inverseOrder = projected.inverseOrder,
            projected.output.ndim == 3,
            projected.output.dim(-2) == 1,
            (projected.output.dim(-1) == 2816 || projected.output.dim(-1) == inputDims),
            projected.output.dtype == .bfloat16
        else {
            return (
                legacyWeightedReduction(projected, indices: indices, weights: weights),
                nil
            )
        }

        let sortedOutputs = MLX.squeezed(projected.output, axis: -2)
        let output = weightedExpertUnsort(
            sortedOutputs: sortedOutputs,
            inverseOrder: inverseOrder,
            weights: weights)
        let carrier =
            isProductionPrefill
            ? WeightedExpertUnsortCarrier(
                sortedOutputs: sortedOutputs,
                inverseOrder: inverseOrder,
                weights: weights)
            : nil
        return (output, carrier)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, lhsIndices: MLXArray? = nil,
        sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(
            x, weightT, lhsIndices: lhsIndices, rhsIndices: indices,
            sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    /// The `sortedIndices` hint is forwarded only when `x` is index-aligned:
    /// when it already carries one row per gathered index. Otherwise it is
    /// withheld. That is a correctness constraint, not a tuning choice.
    ///
    /// THE CAUSE, in the MLX this package pins. `GatherQMM::eval_gpu` computes
    /// `M = x.size() / K` from the array it was HANDED, then passes that `M`
    /// into `gather_qmm_rhs`. `gather_qmm_rhs` broadcasts `x` up to one row per
    /// index when it is not already that shape, and never recomputes `M`; the
    /// dispatch grid and the kernel's row bound both keep using the stale
    /// value. Only the first `x.size() / K` rows of the output are written and
    /// the rest keeps whatever was in the pool. That memory is wrong from the
    /// first call, carries no NaN, and repeats exactly on reuse, so the fault
    /// reads as a stable wrong answer rather than as noise.
    ///
    /// So the fault needs a broadcast, and a broadcast is exactly what the
    /// condition below excludes. It mirrors the vendor's own test for whether
    /// the broadcast is needed.
    ///
    /// The non-quantized `gather_mm` is not exposed to any of this, and the
    /// reason is structural: `GatherMM::eval_gpu` derives its own shapes
    /// inside `gather_mm_rhs`, while only `GatherQMM::eval_gpu` carries a
    /// precomputed row count across the broadcast.
    ///
    /// WHAT IS AT STAKE EITHER WAY. The hint only reaches a different kernel
    /// when `M == 1 && B >= 16 && B / E >= 4`, and on that route it is worth
    /// worth 2.6x to 4.1x on the gather across runs, measured at the
    /// production geometry.
    /// Every caller in this package sorts through
    /// `gatherSort` before it hints, which produces one row per index, so the
    /// aligned branch is the one production takes and the fault is out of
    /// reach. Withholding the hint from that branch as well would surrender
    /// the speed, and the opt-in Gemma 4 expert-QMM tile route inside
    /// `gather_qmm_rhs` with it, for nothing.
    ///
    /// `QuantizedSwitchLinearSortedHintTests` holds both legs and the
    /// reproducer.
    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, lhsIndices: MLXArray? = nil,
        sortedIndices: Bool = false
    ) -> MLXArray {
        let indexAligned = x.size == indices.size * x.dim(-2) * x.dim(-1)
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            lhsIndices: lhsIndices,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices && indexAligned
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
