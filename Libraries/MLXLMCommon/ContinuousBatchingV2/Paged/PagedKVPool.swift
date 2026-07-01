// PagedKVPool.swift
//
// Global paged KV slab pool for the ContinuousBatchingV2 paged backend
// (WS-C). One pool per loaded model.
//
// Design
// ------
// - Layers are bucketed into GROUPS by (kvHeads, headDim). Every group owns
//   exactly TWO large MLXArray slabs (keys + values) shaped
//   `[pageCount, kvHeads, pageSize, headDim]`. Pages of a group are fungible
//   across layers and sequences, so a handful of multi-GB buffers back the
//   whole model. This is deliberate: macOS caps the Metal resource COUNT at
//   ~499k buffers (see research report 04 / mlx-lm#1332) — per-page buffers
//   are forbidden.
// - Page size is 16 tokens (`CBv2PagedDefaults.pageSize`), matching
//   vLLM/mistral.rs block sizes. Revisit after kernel benchmarks.
// - The free list is a plain stack of page indices: O(1) allocate/free.
//   Pages carry refcounts so that copy-free prefix sharing can land later
//   (refcount > 1 == shared page); today every page has refcount 0 or 1.
// - Admission is RESERVATION based: the CBv2 contract's
//   `CBv2SequenceKV.update` cannot throw, so capacity failures mid-decode
//   would be unrecoverable. Instead, `reserve` claims the worst-case page
//   count for a sequence's `maxLength` up front (mistral.rs Metal sizing
//   model) and throws `CBv2KVError.capacityExhausted` at admission time.
//   Physical pages are still allocated LAZILY as tokens are written, so
//   `bytesInUse` reports truthful actual usage; `bytesReserved` reports the
//   admission-relevant figure.
// - Writes go through MLX slice updates on the slabs. On GPU these donate
//   the input buffer (O(tokens written), not O(slab)) as long as no other
//   live MLXArray references the slab. Consumers MUST NOT retain gathered
//   views or kernel inputs across engine steps, and the engine loop must
//   asyncEval each step — otherwise the next write degrades into a full
//   slab copy (still correct, catastrophically slow).
//
// KV quantization (`CBv2KVQuantScheme`) — design TODO
// ---------------------------------------------------
// fp16 pages are the v2.0 fallback for KV-quant configs. The planned design
// for quantized pages (NOT implemented yet):
//   - Each group stores three slabs instead of two: packed uint32 codes
//     `[pageCount, kvHeads, pageSize, headDim / (32 / bits)]` plus
//     per-group-of-64 scales/biases `[pageCount, kvHeads, pageSize,
//     headDim / groupSize, 2]`, mirroring mlx-lm's QuantizedKVCache affine
//     layout so `snapshot()` stays interchange-compatible.
//   - The decode kernel grows a CACHE_T template parameter (uchar packed)
//     and dequantizes inline in `load_row` (scale/bias fetched once per
//     (token, lane-group)); mistral.rs's float8.metal shows the shape of
//     this on Metal.
//   - Writes quantize the incoming `[H, n, D]` tile with `MLX.quantized`
//     before the slice update; `update()`'s returned views dequantize
//     lazily via `MLX.dequantized` (prefill fallback path only).
//   - `hasSinks` models remain eligible: sinks are a kernel parameter, not
//     KV state, so quantization does not interact with them.
// Until then `PagedKVPool.init` throws `backendIneligible` for any scheme
// other than `.fp16` so engine build fails loudly, per the contract.

import Foundation
import MLX

/// KV storage quantization scheme for paged KV pools.
///
/// Only `.fp16` is implemented. See the design TODO in the file header for
/// the quantized-pages plan.
public enum CBv2KVQuantScheme: Sendable, Equatable {
    /// Half-precision pages (the v2.0 default and fallback).
    case fp16
    /// Affine-quantized pages (mlx-lm QuantizedKVCache layout). NOT
    /// implemented — reserved for the quantized-pages follow-up.
    case affine(groupSize: Int, bits: Int)
}

public enum CBv2PagedDefaults {
    /// Page size in tokens. Constant for now; revisit with benchmark data.
    public static let pageSize = 16
}

/// Configuration for a `PagedKVPool`.
public struct PagedKVPoolConfig: Sendable {
    /// Tokens per page. Must divide the decode kernel's expectations; keep
    /// at `CBv2PagedDefaults.pageSize` unless benchmarks say otherwise.
    public var pageSize: Int
    /// Total byte budget for all slabs (K + V, all groups).
    public var capacityBytes: Int
    /// Element type of the pages. `.float16` is the supported scheme;
    /// `.float32` is allowed for tests/parity work.
    public var dtype: DType
    /// Upper bound on tokens written to a WINDOWED layer in one
    /// `update(keys:values:)` call (i.e. the scheduler's max prefill chunk).
    /// Sizes the ring so a full chunk plus the trailing window are always
    /// resident; larger updates trap.
    public var maxPrefillChunk: Int
    /// Nominal per-sequence length used only to split `capacityBytes`
    /// across layer groups proportionally to their demand.
    public var nominalMaxSequenceLength: Int
    /// KV storage scheme. Only `.fp16` is accepted (see file header).
    public var quantScheme: CBv2KVQuantScheme

    public init(
        pageSize: Int = CBv2PagedDefaults.pageSize,
        capacityBytes: Int,
        dtype: DType = .float16,
        maxPrefillChunk: Int = 512,
        nominalMaxSequenceLength: Int = 8192,
        quantScheme: CBv2KVQuantScheme = .fp16
    ) {
        self.pageSize = pageSize
        self.capacityBytes = capacityBytes
        self.dtype = dtype
        self.maxPrefillChunk = maxPrefillChunk
        self.nominalMaxSequenceLength = nominalMaxSequenceLength
        self.quantScheme = quantScheme
    }
}

/// Identity of a slab group: layers with equal (kvHeads, headDim) share
/// pages freely.
public struct PagedKVGroupKey: Hashable, Sendable, CustomStringConvertible {
    public let kvHeads: Int
    public let headDim: Int

    public init(kvHeads: Int, headDim: Int) {
        self.kvHeads = kvHeads
        self.headDim = headDim
    }

    public init(_ kind: CBv2LayerKind) {
        self.init(kvHeads: kind.kvHeads, headDim: kind.headDim)
    }

    public var description: String { "kv\(kvHeads)xd\(headDim)" }
}

/// One slab group: two big MLXArrays plus page bookkeeping.
final class PagedKVGroup {
    let key: PagedKVGroupKey
    let pageSize: Int
    let dtype: DType
    let pageCount: Int
    /// `[pageCount, kvHeads, pageSize, headDim]`
    var kSlab: MLXArray
    var vSlab: MLXArray
    /// Stack of free page ids — O(1) alloc/free.
    var freeList: [Int32]
    /// Per-page refcount (>1 reserved for future prefix sharing).
    var refCounts: [Int]
    /// Pages currently held by sequences (refCount > 0).
    private(set) var pagesInUse: Int = 0
    /// Pages promised to admitted sequences (lazily materialized).
    var pagesReserved: Int = 0

    /// Bytes of ONE page counting both K and V slabs.
    var pageBytes: Int {
        2 * key.kvHeads * pageSize * key.headDim * dtype.size
    }

    init(key: PagedKVGroupKey, pageCount: Int, pageSize: Int, dtype: DType) {
        self.key = key
        self.pageSize = pageSize
        self.dtype = dtype
        self.pageCount = pageCount
        let shape = [pageCount, key.kvHeads, pageSize, key.headDim]
        self.kSlab = MLXArray.zeros(shape, dtype: dtype)
        self.vSlab = MLXArray.zeros(shape, dtype: dtype)
        // LIFO stack: lowest ids pop first, so fresh sequential allocations
        // tend to be physically consecutive (enables run-coalesced writes).
        self.freeList = Array((0 ..< Int32(pageCount)).reversed())
        self.refCounts = [Int](repeating: 0, count: pageCount)
    }

    func allocatePage() -> Int32 {
        precondition(
            !freeList.isEmpty,
            "[PagedKVPool] free list underflow for group \(key) — reservation accounting bug")
        let page = freeList.removeLast()
        precondition(refCounts[Int(page)] == 0)
        refCounts[Int(page)] = 1
        pagesInUse += 1
        return page
    }

    func retainPage(_ page: Int32) {
        precondition(refCounts[Int(page)] > 0, "retain of free page")
        refCounts[Int(page)] += 1
    }

    func freePage(_ page: Int32) {
        let i = Int(page)
        precondition(refCounts[i] > 0, "double free of page \(page) in group \(key)")
        refCounts[i] -= 1
        if refCounts[i] == 0 {
            freeList.append(page)
            pagesInUse -= 1
        }
    }
}

/// Global slab pool per model. Thread-affinity: all mutation must happen on
/// the engine loop thread (matching the CBv2 engine discipline); the pool
/// performs no internal locking.
public final class PagedKVPool {
    public let config: PagedKVPoolConfig
    private(set) var groups: [PagedKVGroupKey: PagedKVGroup] = [:]

    /// Groups in deterministic order (for tests/telemetry).
    public var groupKeys: [PagedKVGroupKey] {
        groups.keys.sorted { ($0.headDim, $0.kvHeads) < ($1.headDim, $1.kvHeads) }
    }

    /// Build a pool sized for `layerKinds` (one entry per model layer;
    /// KV-shared layers own no storage and contribute no demand).
    ///
    /// `capacityBytes` is split across groups proportionally to each
    /// group's worst-case per-sequence demand at
    /// `nominalMaxSequenceLength` (windowed layers capped at their ring).
    public init(layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig) throws {
        guard config.quantScheme == .fp16 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: quant scheme \(config.quantScheme) not implemented; "
                    + "use .fp16 pages (see PagedKVPool.swift design TODO)")
        }
        guard config.dtype == .float16 || config.dtype == .float32 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: unsupported page dtype \(config.dtype)")
        }
        guard config.pageSize > 0, config.capacityBytes > 0 else {
            throw CBv2KVError.backendIneligible(reason: "PagedKVPool: invalid config")
        }
        self.config = config

        // Demand-proportional capacity split.
        let owning = layerKinds.filter { $0.sharesKVWithLayer == nil }
        guard !owning.isEmpty else {
            throw CBv2KVError.backendIneligible(reason: "PagedKVPool: no storage-owning layers")
        }
        var demandTokens: [PagedKVGroupKey: Int] = [:]
        for kind in owning {
            let key = PagedKVGroupKey(kind)
            let tokens = Self.perSequenceTokenDemand(kind: kind, config: config)
            demandTokens[key, default: 0] += tokens
        }
        var demandBytes: [PagedKVGroupKey: Int] = [:]
        var totalDemand = 0
        for (key, tokens) in demandTokens {
            let bytesPerToken = 2 * key.kvHeads * key.headDim * config.dtype.size
            let bytes = tokens * bytesPerToken
            demandBytes[key] = bytes
            totalDemand += bytes
        }
        for (key, bytes) in demandBytes {
            let share = Double(bytes) / Double(totalDemand)
            let groupBytes = Int(share * Double(config.capacityBytes))
            let pageBytes = 2 * key.kvHeads * config.pageSize * key.headDim * config.dtype.size
            let pageCount = groupBytes / pageBytes
            guard pageCount > 0 else {
                throw CBv2KVError.capacityExhausted(needed: pageBytes, available: groupBytes)
            }
            groups[key] = PagedKVGroup(
                key: key, pageCount: pageCount, pageSize: config.pageSize, dtype: config.dtype)
        }
    }

    /// Worst-case tokens a single sequence can pin in one layer of `kind`.
    static func perSequenceTokenDemand(kind: CBv2LayerKind, config: PagedKVPoolConfig) -> Int {
        let maxLen = config.nominalMaxSequenceLength
        switch kind.attention {
        case .full:
            return maxLen
        case .slidingWindow(let window):
            return min(maxLen, ringPageCount(window: window, config: config) * config.pageSize)
        }
    }

    /// Ring length (in pages) for a windowed layer: holds a full prefill
    /// chunk PLUS the trailing window, with one page of wrap slack.
    static func ringPageCount(window: Int, config: PagedKVPoolConfig) -> Int {
        let tokens = window + config.maxPrefillChunk
        return (tokens + config.pageSize - 1) / config.pageSize + 1
    }

    /// Worst-case page demand for one layer of `kind` over a sequence
    /// bounded by `maxLength` tokens.
    static func pageDemand(kind: CBv2LayerKind, maxLength: Int, config: PagedKVPoolConfig) -> Int {
        let maxPages = (maxLength + config.pageSize - 1) / config.pageSize
        switch kind.attention {
        case .full:
            return maxPages
        case .slidingWindow(let window):
            return min(maxPages, ringPageCount(window: window, config: config))
        }
    }

    func group(_ key: PagedKVGroupKey) -> PagedKVGroup {
        guard let g = groups[key] else {
            fatalError("[PagedKVPool] unknown group \(key) — sequence built for another pool?")
        }
        return g
    }

    // MARK: - Reservation (admission)

    /// Atomically reserve worst-case page counts per group; throws
    /// `capacityExhausted` (in bytes) without partial effects.
    public func reserve(_ needs: [PagedKVGroupKey: Int]) throws {
        // Validate everything first — no partial reservations.
        for (key, pages) in needs {
            let g = group(key)
            let available = g.pageCount - g.pagesReserved
            if pages > available {
                throw CBv2KVError.capacityExhausted(
                    needed: pages * g.pageBytes, available: max(0, available) * g.pageBytes)
            }
        }
        for (key, pages) in needs {
            group(key).pagesReserved += pages
        }
    }

    public func unreserve(_ needs: [PagedKVGroupKey: Int]) {
        for (key, pages) in needs {
            let g = group(key)
            g.pagesReserved -= pages
            precondition(g.pagesReserved >= 0, "unreserve underflow for group \(key)")
        }
    }

    // MARK: - Page lifecycle

    func allocatePage(group key: PagedKVGroupKey) -> Int32 {
        group(key).allocatePage()
    }

    func freePages(group key: PagedKVGroupKey, pages: some Sequence<Int32>) {
        let g = group(key)
        for page in pages {
            g.freePage(page)
        }
    }

    // MARK: - Writes
    //
    // All writes are MLX slice updates on the slabs — the only in-place-able
    // write primitive (custom Metal kernels cannot donate outputs). Values
    // are converted to the pool dtype on the way in.

    /// Write `count` tokens into a single page starting at `slot`.
    /// `keys`/`values` are `[kvHeads, n, headDim]` slices with `srcOffset`
    /// selecting the segment to write.
    func write(
        group key: PagedKVGroupKey, page: Int32, slot: Int,
        keys: MLXArray, values: MLXArray, srcOffset: Int, count: Int
    ) {
        let g = group(key)
        precondition(slot + count <= g.pageSize)
        let p = Int(page)
        let k = keys[0..., srcOffset ..< (srcOffset + count), 0...].asType(g.dtype)
        let v = values[0..., srcOffset ..< (srcOffset + count), 0...].asType(g.dtype)
        g.kSlab[p, 0..., slot ..< (slot + count), 0...] = k
        g.vSlab[p, 0..., slot ..< (slot + count), 0...] = v
    }

    /// Write whole pages into a PHYSICALLY CONSECUTIVE run `[firstPage,
    /// firstPage + pageRuns)`. `keys`/`values` are `[kvHeads, n, headDim]`
    /// with `srcOffset ..< srcOffset + pageRuns*pageSize` covering the run.
    func writeFullPageRun(
        group key: PagedKVGroupKey, firstPage: Int32, pageRuns: Int,
        keys: MLXArray, values: MLXArray, srcOffset: Int
    ) {
        let g = group(key)
        let s = g.pageSize
        let h = g.key.kvHeads
        let d = g.key.headDim
        let p = Int(firstPage)
        let n = pageRuns * s
        // [H, n, D] -> [H, runs, S, D] -> [runs, H, S, D]
        let k = keys[0..., srcOffset ..< (srcOffset + n), 0...]
            .reshaped([h, pageRuns, s, d]).transposed(1, 0, 2, 3).asType(g.dtype)
        let v = values[0..., srcOffset ..< (srcOffset + n), 0...]
            .reshaped([h, pageRuns, s, d]).transposed(1, 0, 2, 3).asType(g.dtype)
        g.kSlab[p ..< (p + pageRuns), 0..., 0..., 0...] = k
        g.vSlab[p ..< (p + pageRuns), 0..., 0..., 0...] = v
    }

    // MARK: - Reads

    /// Gather `count` tokens in temporal order from `pages` (ordered oldest
    /// to newest), where the first token lives at `firstSlot` of `pages[0]`.
    /// Returns `([1, kvHeads, count, headDim]) x 2` in the pool dtype.
    ///
    /// This materializes a copy (MLX gathers always do); callers on the hot
    /// decode path must prefer the kernel, which reads the slabs in place.
    /// The returned arrays reference the CURRENT slabs — do not retain them
    /// across engine steps or slab writes stop donating.
    func gather(
        group key: PagedKVGroupKey, pages: [Int32], firstSlot: Int, count: Int
    ) -> (keys: MLXArray, values: MLXArray) {
        let g = group(key)
        let h = g.key.kvHeads
        let d = g.key.headDim
        let s = g.pageSize
        guard count > 0 else {
            let empty = MLXArray.zeros([1, h, 0, d], dtype: g.dtype)
            return (empty, empty)
        }
        precondition(firstSlot < s)
        precondition(pages.count * s >= firstSlot + count, "gather range exceeds page list")
        let idx = MLXArray(pages)
        func assemble(_ slab: MLXArray) -> MLXArray {
            // [np, H, S, D] -> [H, np, S, D] -> [H, np*S, D] -> slice -> [1, H, count, D]
            take(slab, idx, axis: 0)
                .transposed(1, 0, 2, 3)
                .reshaped([h, pages.count * s, d])[0..., firstSlot ..< (firstSlot + count), 0...]
                .expandedDimensions(axis: 0)
        }
        return (assemble(g.kSlab), assemble(g.vSlab))
    }

    // MARK: - Accounting

    /// Truthful bytes behind pages sequences have actually touched.
    public var bytesInUse: Int {
        groups.values.reduce(0) { $0 + $1.pagesInUse * $1.pageBytes }
    }

    /// Bytes promised to admitted sequences (the admission-relevant figure).
    public var bytesReserved: Int {
        groups.values.reduce(0) { $0 + $1.pagesReserved * $1.pageBytes }
    }

    public var bytesCapacity: Int {
        groups.values.reduce(0) { $0 + $1.pageCount * $1.pageBytes }
    }

    /// Force-materialize the slabs (e.g. at engine warmup, before the wired
    /// limit is measured) so first-token latency never pays the allocation.
    /// Wire-down itself is owned by the existing WiredMemory policy plumbing
    /// (`WiredSumPolicy` et al.) — slabs participate like any other resident
    /// allocation once evaluated.
    public func materializeSlabs() {
        var arrays: [MLXArray] = []
        for g in groups.values {
            arrays.append(g.kSlab)
            arrays.append(g.vSlab)
        }
        eval(arrays)
    }
}
