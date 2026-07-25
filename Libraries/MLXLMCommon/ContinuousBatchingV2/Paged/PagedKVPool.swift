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
// - Writes go through IN-PLACE Metal kernels (`PagedAttentionKernel
//   .bulkWrite` here; the decode path fuses its single-token write into
//   pass A). The slabs are STABLE buffers — never versioned through MLX
//   slice updates. Slice updates are unusable at slab scale: MLX's
//   gpu::eval retains every op's INPUT data handles until the command
//   buffer completes, so a slice-update following any kernel read of the
//   slab fails donation and degrades into a full-slab copy (~370 GiB of
//   copies per decoded token on GPT-OSS-20B at a 16 GiB pool — the
//   original ~100x real-model slowdown; docs/engine-v2/kernel-research.md
//   §3). In-place writes are invisible to MLX's hazard tracking, so each
//   bulk write emits a FENCE (chained per group) and every gather of the
//   group's slabs consumes the latest fence for scheduling order + memory
//   barriers. Decode dispatches don't consume fences: a row's decode
//   always follows its writes via host syncs or step-graph dependencies
//   (see pagedattention.metal, "In-place slab writes").
//
// Pages are fp16, always. KV quantization was retired from the product, so
// the `CBv2KVQuantScheme` hook and its refusal guard are gone; a quantized
// config is now unrepresentable rather than rejected at runtime. Should
// quantized pages come back, the v1 constraint that forced fp16 was the
// decode kernel: it reads slab rows directly, so quantized pages need a
// CACHE_T template parameter and an inline dequantize in `load_row`, plus a
// third slab per group for affine scales/biases in mlx-lm's QuantizedKVCache
// layout to keep `snapshot()` interchange-compatible. Sinks are a kernel
// parameter rather than KV state, so they never interacted with it.

import Foundation
import MLX

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
    /// Bounds one windowed update; larger updates trap. Since WS-1.2 the
    /// ring is sized from the WINDOW, not from this, so a chunk no longer
    /// has to fit inside the ring alongside the window — but it must still
    /// fit inside the ring on its own, which `checkedRingPageCount` guards.
    public var maxPrefillChunk: Int
    /// Nominal per-sequence length used only to split `capacityBytes`
    /// across layer groups proportionally to their demand.
    public var nominalMaxSequenceLength: Int
    /// Metal's maximum length for one buffer. Every K and V slab is
    /// validated against this before any MLXArray is created; exceeding it
    /// would otherwise surface as an uncatchable allocator/Metal failure.
    public var maxBufferLength: Int
    /// Prefix-cache block size this pool must be able to DONATE and ADOPT
    /// at, or `nil` for a pool that will never participate in block sharing
    /// (unit fixtures, microbenchmarks).
    ///
    /// Set this to `CBv2BlockHasher.defaultBlockSize` for any pool behind a
    /// prefix cache. It arms the WS-0.6 chunk-coverage guard below, which
    /// is not merely advisory: WS-4's windowed-sharing residency proof
    /// assumes one prefill chunk plus the frontier's partial page covers a
    /// whole block, so a pool that cannot do that must be refused rather
    /// than silently donate blocks the proof does not cover.
    ///
    /// It is opt-IN because `maxPrefillChunk` is an operator/ITL knob
    /// (`SchedulerV2` prefill chunking, and the mixed-prefill cap that
    /// bounds prompt tokens on decode-carrying steps). Making a small chunk
    /// fail engine build unconditionally would turn a latency knob into an
    /// outage for pools that never share a block.
    public var prefixSharingBlockSize: Int?
    public init(
        pageSize: Int = CBv2PagedDefaults.pageSize,
        capacityBytes: Int,
        dtype: DType = .float16,
        maxPrefillChunk: Int = 512,
        nominalMaxSequenceLength: Int = 8192,
        maxBufferLength: Int = MLX.GPU.deviceInfo().maxBufferSize,
        prefixSharingBlockSize: Int? = nil
    ) {
        self.pageSize = pageSize
        self.capacityBytes = capacityBytes
        self.dtype = dtype
        self.maxPrefillChunk = maxPrefillChunk
        self.nominalMaxSequenceLength = nominalMaxSequenceLength
        self.maxBufferLength = maxBufferLength
        self.prefixSharingBlockSize = prefixSharingBlockSize
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
    /// `[pageCount, kvHeads, pageSize, headDim]`. STABLE arrays: written
    /// in place by the write kernels, never replaced (see file header).
    let kSlab: MLXArray
    let vSlab: MLXArray
    /// Latest fence of the group's bulk-write chain (`[1]` int32). Gathers
    /// consume it so page reads order after every prior bulk write.
    var writeFence: MLXArray
    /// Stack of free page ids — O(1) alloc/free.
    var freeList: [Int32]
    /// Per-page refcount (>1 reserved for future prefix sharing).
    var refCounts: [Int]
    /// Pages currently held by sequences (refCount > 0).
    private(set) var pagesInUse: Int = 0
    /// Pages promised to admitted sequences (lazily materialized).
    var pagesReserved: Int = 0
    /// Pages queued for release by an in-flight speculative transaction
    /// (WS-3.2c). They keep `refCount > 0` until `drainDeferredFrees()`, so
    /// they cannot be recycled to another row while a round's captures
    /// still name them.
    var deferredFrees: [Int32] = []

    /// The reserved POISON page: physical page 0 of every group, permanently
    /// zeroed, never allocatable, never writable, `refCount` pinned at 1.
    ///
    /// Two call sites pad an array up to the kernel's minimum length and
    /// need a page id that is guaranteed inert: `PagedKVPool.writeTokens`
    /// (the `slots` pad) and `PagedLayerCache.deviceTables` (the block-table
    /// column pad). Both used to pad with a REAL page — a duplicated live
    /// slot and a literal `0` respectively — which are fail-OPEN: page 0 is
    /// the first page the free list hands out (`freeList` is built reversed
    /// and popped with `removeLast`), so the literal pad named whichever
    /// tenant happened to hold it.
    ///
    /// Page 0 is the poison page DELIBERATELY, rather than a high id past
    /// the tenant range: `MLXArray.zeros`, `[Int32](repeating: 0, …)` and
    /// every default-initialised int32 buffer produce 0, so reserving 0
    /// makes the entire class of "forgot to pad / uninitialised table
    /// entry" bugs read zeros instead of another sequence's live KV.
    static let poisonPage: Int32 = 0
    var poisonPage: Int32 { Self.poisonPage }

    /// Pages a sequence row can actually own — every page except the
    /// poison page. This, NOT `pageCount`, is the reservation ceiling and
    /// the honest capacity figure.
    var usablePageCount: Int { pageCount - 1 }

    /// Bytes of ONE page counting both K and V slabs.
    var pageBytes: Int {
        2 * key.kvHeads * pageSize * key.headDim * dtype.size
    }

    init(key: PagedKVGroupKey, pageCount: Int, pageSize: Int, dtype: DType) {
        precondition(
            pageCount >= 2,
            "[PagedKVPool] group \(key) needs at least one usable page beyond the poison page")
        self.key = key
        self.pageSize = pageSize
        self.dtype = dtype
        self.pageCount = pageCount
        let shape = [pageCount, key.kvHeads, pageSize, key.headDim]
        self.kSlab = MLXArray.zeros(shape, dtype: dtype)
        self.vSlab = MLXArray.zeros(shape, dtype: dtype)
        self.writeFence = MLXArray.zeros([1], dtype: .int32)
        // LIFO stack: lowest ids pop first, so fresh sequential allocations
        // tend to be physically consecutive (enables run-coalesced writes).
        // The poison page is EXCLUDED — it is never handed out, so the
        // stack starts one past it.
        self.freeList = Array((Self.poisonPage + 1 ..< Int32(pageCount)).reversed())
        // The slabs are zero-initialised and no write can ever address the
        // poison page (every slot comes from an allocated page id), so
        // pinning its refcount at 1 is the whole of "permanently zeroed":
        // it can never be allocated, retained, freed or written.
        var counts = [Int](repeating: 0, count: pageCount)
        counts[Int(Self.poisonPage)] = 1
        self.refCounts = counts
    }

    /// True when `page` is a page a sequence row can own. The poison page
    /// and out-of-range ids are not.
    func isAllocatable(_ page: Int32) -> Bool {
        page != Self.poisonPage && page >= 0 && Int(page) < pageCount
    }

    func allocatePage() -> Int32 {
        precondition(
            !freeList.isEmpty,
            "[PagedKVPool] free list underflow for group \(key) — reservation accounting bug")
        let page = freeList.removeLast()
        precondition(
            page != Self.poisonPage,
            "[PagedKVPool] poison page escaped the free list for group \(key)")
        precondition(refCounts[Int(page)] == 0)
        refCounts[Int(page)] = 1
        pagesInUse += 1
        return page
    }

    func retainPage(_ page: Int32) {
        precondition(page != Self.poisonPage, "retain of the reserved poison page")
        precondition(refCounts[Int(page)] > 0, "retain of free page")
        refCounts[Int(page)] += 1
    }

    func freePage(_ page: Int32) {
        precondition(
            page != Self.poisonPage,
            "[PagedKVPool] free of the reserved poison page in group \(key) — a row "
                + "should never have held it")
        let i = Int(page)
        precondition(refCounts[i] > 0, "double free of page \(page) in group \(key)")
        refCounts[i] -= 1
        if refCounts[i] == 0 {
            freeList.append(page)
            pagesInUse -= 1
        }
    }

    /// Queue `page` for release at the end of a speculative transaction.
    /// The page keeps its refcount until the drain.
    func deferFree(_ page: Int32) {
        precondition(
            page != Self.poisonPage,
            "[PagedKVPool] deferred free of the reserved poison page in group \(key)")
        deferredFrees.append(page)
    }

    func drainDeferredFrees() {
        for page in deferredFrees { freePage(page) }
        deferredFrees.removeAll(keepingCapacity: true)
    }
}

/// Global slab pool per model. Thread-affinity: all mutation must happen on
/// the engine loop thread (matching the CBv2 engine discipline); the pool
/// performs no internal locking.
public final class PagedKVPool {
    public let config: PagedKVPoolConfig
    /// Validated Metal source retained for the pool's lifetime. Kernel
    /// dispatch never touches Bundle.module and therefore has no
    /// request-time resource-failure path.
    let kernelSource: String
    private(set) var groups: [PagedKVGroupKey: PagedKVGroup] = [:]

    /// Monotonic identity for every `PagedSequenceKV` minted against this
    /// pool. Unlike `ObjectIdentifier` (a heap address, reusable after
    /// dealloc), serials are NEVER reused, so device block-table caches
    /// fingerprinted by serial can never confuse a finished request's rows
    /// with a new request's (see `PagedLayerCache.deviceTables`).
    private var lastRowSerial: UInt64 = 0

    func nextRowSerial() -> UInt64 {
        lastRowSerial += 1
        return lastRowSerial
    }

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
        guard config.dtype == .float16 || config.dtype == .float32 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: unsupported page dtype \(config.dtype)")
        }
        guard config.pageSize > 0, config.capacityBytes > 0,
            config.maxPrefillChunk > 0, config.nominalMaxSequenceLength > 0
        else {
            throw CBv2KVError.backendIneligible(reason: "PagedKVPool: invalid config")
        }
        guard config.maxBufferLength > 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: Metal maxBufferLength is unavailable")
        }
        // The decode kernel partitions attention into fixed
        // `PagedAttentionKernel.partitionTokens`-token slices and requires
        // page boundaries to align with them (`PTOK % pageSize == 0` is a
        // kernel-launch precondition). Reject misaligned page sizes HERE,
        // at construction, so a bad config fails engine build with a clear
        // `backendIneligible` instead of trapping on the first decode
        // (PR#62 review).
        guard PagedAttentionKernel.partitionTokens % config.pageSize == 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: pageSize \(config.pageSize) must evenly divide "
                    + "PagedAttentionKernel.partitionTokens "
                    + "(\(PagedAttentionKernel.partitionTokens)) — use a power-of-two "
                    + "divisor such as \(CBv2PagedDefaults.pageSize)")
        }
        // WS-0.6, invariant 1: a prefix-cache BLOCK must be a whole number
        // of pages.
        //
        // `CBv2BlockHasher.defaultBlockSize` (256) and
        // `CBv2PagedDefaults.pageSize` (16) are declared in two files with
        // no cross-reference, and their divisibility is what makes windowed
        // sharing a pointer swap: every matched block boundary is then also
        // a page boundary, so an adopter's post-adoption writes start at
        // slot 0 of a fresh page and `restoreWindow(keys:values:base:)`
        // never has to copy a partial page (PagedSeamContract.swift, WS-4.1
        // — which explicitly defers the assertion to this guard). Violating
        // it does not fail loudly anywhere; it silently makes adoption
        // wrong. Checked unconditionally: it is a property of two
        // constants, so no legitimate configuration can need it relaxed.
        guard CBv2BlockHasher.defaultBlockSize % config.pageSize == 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: pageSize \(config.pageSize) must evenly divide the "
                    + "prefix-cache block size \(CBv2BlockHasher.defaultBlockSize) "
                    + "(CBv2BlockHasher.defaultBlockSize) — otherwise a matched block "
                    + "boundary is not a page boundary and windowed sharing cannot adopt "
                    + "pages without a partial-page copy")
        }
        // WS-0.6, invariant 2: one prefill chunk plus the frontier's
        // partial page must cover a whole block, so a block can always be
        // completed without a second chunk straddling it.
        //
        // Armed only for pools that declare they will share blocks — see
        // `PagedKVPoolConfig.prefixSharingBlockSize` for why this is opt-in
        // rather than a blanket refusal of small prefill chunks.
        if let blockSize = config.prefixSharingBlockSize {
            guard blockSize > 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: prefixSharingBlockSize \(blockSize) must be positive")
            }
            guard blockSize % config.pageSize == 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: pageSize \(config.pageSize) must evenly divide the "
                        + "declared prefix sharing block size \(blockSize)")
            }
            // `maxPrefillChunk` is operator-influenced and may be Int.max in
            // hostile-size tests, so the sum is overflow-checked rather than
            // written inline.
            let chunkSpan = try Self.checkedAdd(
                config.maxPrefillChunk, config.pageSize, context: "prefill chunk block span")
            guard chunkSpan >= blockSize else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: maxPrefillChunk \(config.maxPrefillChunk) + pageSize "
                        + "\(config.pageSize) = \(chunkSpan) cannot cover the declared prefix "
                        + "sharing block size \(blockSize) — a chunk that cannot complete a "
                        + "block leaves WS-4's windowed-sharing residency proof unsatisfied; "
                        + "raise maxPrefillChunk, lower prefixSharingBlockSize, or set it to "
                        + "nil for a pool that never shares blocks")
            }
        }
        // Demand-proportional capacity split.
        let owning = layerKinds.filter { $0.sharesKVWithLayer == nil }
        guard !owning.isEmpty else {
            throw CBv2KVError.backendIneligible(reason: "PagedKVPool: no storage-owning layers")
        }
        for kind in owning {
            guard kind.kvHeads > 0, kind.headDim > 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: non-positive KV shape "
                        + "\(kind.kvHeads)x\(kind.headDim)")
            }
            if case .slidingWindow(let window) = kind.attention {
                guard window > 0 else {
                    throw CBv2KVError.backendIneligible(
                        reason: "PagedKVPool: invalid sliding window \(window)")
                }
                _ = try Self.checkedRingPageCount(window: window, config: config)
            }
        }

        let source: String
        do {
            source = try PagedAttentionResources.loadSourceForCurrentProcess()
        } catch {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: paged-attention runtime resource unavailable: \(error)")
        }
        self.config = config
        self.kernelSource = source

        var demandTokens: [PagedKVGroupKey: Int] = [:]
        for kind in owning {
            let key = PagedKVGroupKey(kind)
            let tokens = Self.perSequenceTokenDemand(kind: kind, config: config)
            demandTokens[key] = try Self.checkedAdd(
                demandTokens[key, default: 0],
                tokens,
                context: "group token demand")
        }
        var demandBytes: [PagedKVGroupKey: Int] = [:]
        var totalDemand = 0
        for (key, tokens) in demandTokens {
            let bytesPerToken = try Self.checkedMultiply(
                [2, key.kvHeads, key.headDim, config.dtype.size],
                context: "bytes per token")
            let bytes = try Self.checkedMultiply(
                [tokens, bytesPerToken],
                context: "group byte demand")
            demandBytes[key] = bytes
            totalDemand = try Self.checkedAdd(
                totalDemand, bytes, context: "total byte demand")
        }
        guard totalDemand > 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: zero byte demand")
        }
        for (key, bytes) in demandBytes {
            let share = Double(bytes) / Double(totalDemand)
            let groupBytes = Int(share * Double(config.capacityBytes))
            let pageBytes = try Self.checkedMultiply(
                [2, key.kvHeads, config.pageSize, key.headDim, config.dtype.size],
                context: "page bytes")
            let usablePages = groupBytes / pageBytes
            guard usablePages > 0 else {
                throw CBv2KVError.capacityExhausted(needed: pageBytes, available: groupBytes)
            }
            // One extra PHYSICAL page per group backs the reserved poison
            // page (WS-0.5). It is carved on TOP of the byte budget rather
            // than out of it, so `capacityBytes` keeps mapping to
            // tenant-usable pages exactly: a pool sized for N concurrent
            // requests still admits N. The overshoot is one page per group
            // — the same order as the floor division above already discards
            // — and it is the only allocation in the pool that is not
            // tenant storage. `bytesCapacity` reports the usable figure, so
            // no accounting surface sees the extra page.
            let pageCount = try Self.checkedAdd(usablePages, 1, context: "poison page")
            guard pageCount <= Int(Int32.max) else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: group \(key) needs \(pageCount) pages, "
                        + "over the Int32 page-table limit")
            }
            let slabBytes = try Self.checkedMultiply(
                [pageCount, key.kvHeads, config.pageSize, key.headDim, config.dtype.size],
                context: "slab bytes")
            guard slabBytes <= config.maxBufferLength else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: group \(key) slab requires \(slabBytes) B, "
                        + "over Metal maxBufferLength \(config.maxBufferLength) B")
            }
            groups[key] = PagedKVGroup(
                key: key, pageCount: pageCount, pageSize: config.pageSize, dtype: config.dtype)
        }
    }

    private static func checkedAdd(
        _ lhs: Int,
        _ rhs: Int,
        context: String
    ) throws -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: integer overflow computing \(context)")
        }
        return sum
    }

    private static func checkedMultiply(
        _ values: [Int],
        context: String
    ) throws -> Int {
        var product = 1
        for value in values {
            let (next, overflow) = product.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: integer overflow computing \(context)")
            }
            product = next
        }
        return product
    }

    /// `ringPageCount` with the WS-3.1 invariant enforced, and with every
    /// intermediate overflow-checked so an operator-supplied window, chunk
    /// or page size fails engine build instead of trapping the process.
    private static func checkedRingPageCount(
        window: Int,
        config: PagedKVPoolConfig
    ) throws -> Int {
        let span = CBv2PagedSpeculation.maxSpeculativeSpan
        // The widest range a row can expose: `PagedSequenceKV.retainedCount`
        // is `min(written, window - 1 + lastUpdateTokens)`, and a prefill
        // chunk leaves `lastUpdateTokens == n`. It is `window - 1 + chunk`,
        // NOT `window`, because the chunk's EARLIEST query must still see
        // its whole window.
        let attendable = try checkedAdd(
            window - 1, config.maxPrefillChunk, context: "windowed attendable span")
        let rounded = try checkedAdd(
            attendable, config.pageSize - 1, context: "window ring rounding")
        let pages = try checkedAdd(
            rounded / config.pageSize,
            (span + config.pageSize - 1) / config.pageSize,
            context: "window ring pages")
        let ringTokens = try checkedMultiply(
            [pages, config.pageSize], context: "window ring tokens")

        // WS-3.1: the ring must cover the whole attendable range PLUS a
        // speculative span. This is the invariant that used to hold by
        // accident — the old `+ 1` page of "wrap slack" happened to leave
        // ~528 spare tokens for gemma-4 — and it is now enforced.
        //
        // It subsumes two distinct failure modes, both silent:
        //  - under the attendable term, `gatherRange` trips "gather of
        //    evicted window range" on ordinary prefill: a process abort;
        //  - under the span term, a rolled-back speculative write aliases a
        //    live in-window entry: corrupted KV with no crash at all.
        // It also implies `maxPrefillChunk <= ringTokens`, so a chunk can
        // never lap the ring and race with itself inside one bulk dispatch.
        //
        // `PagedSequenceKV.speculativeHeadroom` is the row-side reading of
        // the same quantity; both bind to
        // `CBv2PagedSpeculation.maxSpeculativeSpan` so they cannot disagree.
        let required = try checkedAdd(attendable, span, context: "ring speculative floor")
        guard ringTokens >= required else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: windowed ring of \(pages) pages (\(ringTokens) tokens) "
                    + "cannot hold the attendable span \(attendable) (window \(window) - 1 + "
                    + "chunk \(config.maxPrefillChunk)) plus the speculative span \(span)")
        }
        return pages
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

    /// Ring length (in pages) for a windowed layer: the widest range a row
    /// can expose, plus one speculative span.
    ///
    /// WS-1.2 + WS-3.1. Derived, where the previous
    /// `ceil((window + maxPrefillChunk) / pageSize) + 1` was assembled from
    /// a term that was nearly right and a `+ 1` page of unexplained "wrap
    /// slack". The real requirement is
    ///
    ///     ring * pageSize  >=  (window - 1 + maxPrefillChunk) + maxSpeculativeSpan
    ///
    /// because `PagedSequenceKV.retainedCount` is
    /// `min(written, window - 1 + lastUpdateTokens)` and a prefill chunk
    /// leaves `lastUpdateTokens == n`: the chunk's EARLIEST query must still
    /// see its whole window, so the attendable range is `window - 1 + chunk`
    /// wide, not `window`.
    ///
    /// BE CLEAR ABOUT WHAT THIS SAVES: essentially nothing at the default
    /// chunk. gemma-4 at window 1,024 / chunk 512 is
    /// `ceil(1535/16) + 1 == 97` pages — the same 97 it was before. The
    /// change buys correctness (an enforced invariant instead of an
    /// accidental margin, and the `+ 1` explained) and it saves a page only
    /// where `window + chunk` lands just past a page boundary.
    ///
    /// The plan's headline — a 1.52x over-provision on 25 of gemma-4's 30
    /// layers — is NOT removable by resizing. 1,552 vs 1,024 tokens is the
    /// `maxPrefillChunk` term, and that term is load-bearing for as long as
    /// the layer gathers its attendable range AFTER writing the chunk. The
    /// ring can drop to `ceil(window / pageSize) + ceil(span / pageSize)`
    /// (65 pages) only together with the other half of WS-1.2: attend
    /// `gather(ring) ++ chunk` with the gather taken BEFORE the write, the
    /// way `WindowedSequenceKV.update` does with `concatenated(kParts,
    /// axis: 2)`. Then the pre-write ring supplies `[f - n - ring, f - n)`,
    /// the chunk tensor supplies the rest, and the ring only has to satisfy
    /// `ring >= window - 1`. Shrinking without that half aborts the daemon
    /// on every windowed prefill chunk over `pageSize + 1` tokens — measured,
    /// not theorised. Both halves are the item; one alone is a trap.
    ///
    /// The speculative term binds to `CBv2PagedSpeculation.maxSpeculativeSpan`
    /// rather than a literal so this sizing and
    /// `PagedSequenceKV.speculativeHeadroom` cannot disagree.
    static func ringPageCount(window: Int, config: PagedKVPoolConfig) -> Int {
        let attendable = window - 1 + config.maxPrefillChunk
        let spanPages =
            (CBv2PagedSpeculation.maxSpeculativeSpan + config.pageSize - 1) / config.pageSize
        return (attendable + config.pageSize - 1) / config.pageSize + spanPages
    }

    /// Pages one layer of `kind` must be able to hold for a sequence
    /// bounded by `maxLength` tokens. This is the ADMISSION charge, and it
    /// is also the row's hard allocation cap (`PagedSequenceKV
    /// .reservedPages`), so it may never be less than the row's true peak.
    ///
    /// EXACTNESS (WS-1.3). For a row that writes from position 0 this is
    /// not a bound, it is an identity: it equals the peak `table.count`
    /// exactly, with zero slack, for every (pageSize, window,
    /// maxPrefillChunk, maxLength) combination. That falls out of
    /// `PagedSequenceKV.ensurePage`, which grows the table to
    /// `maxSlotTouched + 1` where the slot of absolute position `p` is
    /// `(p / pageSize) % ringPages`: a row sweeping [0, maxLength) touches
    /// slots 0…min(ceil(maxLength / pageSize), ringPages) - 1 and no
    /// others. Change either side and this stops holding — the identity is
    /// pinned by `chargeEqualsPeakResidency` in CBv2PagedPoolGuardTests.
    ///
    /// CONSEQUENCE: there is no safe reduction available HERE. WS-1.3's
    /// "charge min(ctx, window)" reads as a change to this function, but the
    /// charge is already the tight bound — what makes it large is the ring.
    /// gemma-4's 1,024-token window rings at 97 pages (1,552 tokens), so
    /// every request whose `maxLength` reaches 1,537 legitimately holds all
    /// 97 of them, and charging less is a free-list underflow:
    /// `PagedKVGroup.allocatePage` traps, which is a daemon abort under
    /// load, not a rejected request.
    ///
    /// The win therefore lives in `ringPageCount`, and — see the analysis
    /// there — it is NOT a resizing change either. It needs the layer to
    /// attend `gather(ring) ++ chunk` with the gather taken before the
    /// write, which drops the ring to 65 pages; this `min` then inherits
    /// the reduction for free with no change to the line below. Until that
    /// lands, WS-1.3 has nothing to collect.
    ///
    /// Rows adopted mid-stream (`fastForward`) are the one CONSERVATIVE
    /// case: their first write lands at ring slot `(base / pageSize) %
    /// ringPages`, so they allocate that slot's prefix eagerly but never
    /// exceed the ring. The charge over-reserves them by up to `ringPages -
    /// ceil((maxLength - base) / pageSize)` pages. Tightening that needs
    /// the adoption offset, which only `PagedKVBackend` knows at admission.
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

    // MARK: - Poison page (WS-0.5)

    /// The group's reserved, permanently-zeroed, never-allocatable page.
    ///
    /// Call sites that must pad an array of page ids up to a kernel's
    /// minimum length pad with THIS, never with a real page id. See
    /// `PagedKVGroup.poisonPage` for why it is page 0.
    public func poisonPage(group key: PagedKVGroupKey) -> Int32 {
        group(key).poisonPage
    }

    /// Pages of `key` a sequence row can own — the reservation ceiling.
    /// One less than the group's physical page count.
    public func usablePageCount(group key: PagedKVGroupKey) -> Int {
        group(key).usablePageCount
    }

    /// True when `page` is a page a sequence row can own. False for the
    /// poison page and for out-of-range ids.
    public func isAllocatablePage(_ page: Int32, group key: PagedKVGroupKey) -> Bool {
        group(key).isAllocatable(page)
    }

    // MARK: - Reservation (admission)

    /// Atomically reserve worst-case page counts per group; throws
    /// `capacityExhausted` (in bytes) without partial effects.
    public func reserve(_ needs: [PagedKVGroupKey: Int]) throws {
        // Validate everything first — no partial reservations.
        for (key, pages) in needs {
            let g = group(key)
            let available = g.usablePageCount - g.pagesReserved
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

    /// Queue `pages` for release at the END of a speculative transaction
    /// (WS-3.2c). Queued pages keep `refCount > 0`, so they stay out of the
    /// free list — a page cannot be recycled to another row while a
    /// speculative round's captures still name it.
    ///
    /// The caller MUST have already removed `pages` from its own page
    /// table: the queue and any live table must stay disjoint, or the
    /// drain and the row's own release both free the same page and trip
    /// `freePage`'s double-free precondition.
    func deferFreePages(group key: PagedKVGroupKey, pages: some Sequence<Int32>) {
        let g = group(key)
        for page in pages {
            g.deferFree(page)
        }
    }

    /// Release everything queued by `deferFreePages`, across ALL groups.
    /// No-op when nothing is queued, so callers may invoke it
    /// unconditionally at commit and at release.
    ///
    /// POOL-WIDE BY CONTRACT (`PagedSeamContract.swift` freezes the no-arg
    /// signature). If two rows are mid-transaction and one commits, the
    /// other's queued pages drain too. That is safe only because every row
    /// of an MTP round commits inside the SAME finalize loop
    /// (`MTP/EngineLoopV2+MTPFinalize.swift:146`), on the engine thread,
    /// between steps — so no queued page is still named by an in-flight
    /// capture at any drain point. Committing rows across step boundaries
    /// would break this and must not be introduced without making the
    /// queue per-row.
    func drainDeferredFrees() {
        for g in groups.values {
            g.drainDeferredFrees()
        }
    }

    // MARK: - Writes
    //
    // All bulk writes go through the in-place write kernel and advance the
    // group's fence chain (see file header — slice updates on the slabs
    // are forbidden). Values are converted to the pool dtype on the way in.

    /// Scatter `slots.count` tokens into the group's slabs. `keys`/`values`
    /// are `[kvHeads, n, headDim]`; `slots[i]` is token `i`'s physical
    /// position as `page * pageSize + slot`.
    func writeTokens(
        group key: PagedKVGroupKey, slots: [Int32], keys: MLXArray, values: MLXArray
    ) {
        guard !slots.isEmpty else { return }
        let g = group(key)
        precondition(keys.dim(1) == slots.count && values.dim(1) == slots.count)
        let k = keys.dtype == g.dtype ? keys : keys.asType(g.dtype)
        let v = values.dtype == g.dtype ? values : values.asType(g.dtype)
        // Pad to >= 8 entries so the generated kernel signature keeps the
        // device address space. The pad target is the group's reserved
        // POISON page, never a real slot.
        //
        // Pad entries are provably never dereferenced: `bulkWrite`
        // dispatches grid `(headDim, kvHeads, n)` with `n` the TRUE token
        // count, so the kernel only ever indexes `slots[0 ..< n]`. This
        // pad used to repeat `slots[n - 1]`, a live physical slot of the
        // writing row, which made the safety argument fail-OPEN: it rested
        // entirely on that grid bound, and a violation would silently
        // rewrite a real token. Poison makes it fail-SAFE — a stray write
        // lands on a page no sequence can own.
        var padded = slots
        if padded.count < 8 {
            let poisonSlot = g.poisonPage * Int32(g.pageSize)
            padded.append(contentsOf: repeatElement(poisonSlot, count: 8 - padded.count))
        }
        g.writeFence = PagedAttentionKernel.bulkWrite(
            kSlab: g.kSlab, vSlab: g.vSlab,
            keys: k, values: v,
            slots: MLXArray(padded),
            prevFence: g.writeFence,
            pageSize: g.pageSize,
            kernelSource: kernelSource)
    }

    // MARK: - Reads

    /// Gather `count` tokens in temporal order from `pages` (ordered oldest
    /// to newest), where the first token lives at `firstSlot` of `pages[0]`.
    /// Returns `([1, kvHeads, count, headDim]) x 2` in the pool dtype.
    ///
    /// This materializes a copy (MLX gathers always do); callers on the hot
    /// decode path must prefer the kernel, which reads the slabs in place.
    /// The returned arrays are LAZY reads of the shared slabs — evaluate
    /// or drop them within the current engine step: the slabs are mutated
    /// in place, so a stale unevaluated gather could observe pages after
    /// they were recycled and rewritten by another row.
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
        // Consume the group's write fence: in-place writes are invisible
        // to MLX's hazard tracking, so the dependency edge (and the memory
        // barrier it induces) is what orders this gather after every prior
        // bulk write of the group.
        let idx = MLXArray(pages) + g.writeFence * 0
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

    /// Bytes a sequence can actually be given. Counts USABLE pages: the
    /// per-group poison page is physically allocated but is not tenant
    /// storage, so reporting it here would overstate what admission can
    /// hand out.
    public var bytesCapacity: Int {
        groups.values.reduce(0) { $0 + $1.usablePageCount * $1.pageBytes }
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
