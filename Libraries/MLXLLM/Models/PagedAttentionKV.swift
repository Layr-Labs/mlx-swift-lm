// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon

/// Paged keys and values for the full-attention layers of a model whose
/// attention takes a `KVCache` (`attentionWithCacheUpdate`), such as one rank
/// of a tensor-parallel Qwen 3.5 model: every request's keys and values live
/// in fixed-size pages of two shared slabs (`[capacity * pageSize,
/// kvHeads * headDim]` each), not in a per-request buffer per layer.
///
/// The page bookkeeping is the GLM DSA pool's (`GLM5NextDSAPageBook`,
/// `GLM5NextDSAPageTable`): reference-counted pages, page tables in token
/// order, copy on write at a shared partial page, released pages retired
/// until `recycle()`, reservations per request. An append writes only its
/// own rows into pages in place (one launch); attention reads the
/// `0 ..< offset` keys and values gathered from the pages in token order
/// (one launch) into `[1, kvHeads, offset, headDim]` arrays: the same values
/// in the same logical layout as `KVCacheSimple`'s, so scaled dot-product
/// attention computes the same bits.
///
/// Engine-thread confined, as the caches it serves. The slabs grow only in
/// `reserve` (between steps); a step's pages come from the reservations.
public final class PagedAttentionKVPool {
    public let book: GLM5NextDSAPageBook
    let lock = NSRecursiveLock()
    public var pageSize: Int { book.pageSize }
    public let kvHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let dtype: DType
    /// `[keys, values]`, each `[capacity * pageSize, kvHeads * headDim]`;
    /// empty at capacity 0.
    private(set) var slabs: [MLXArray] = []
    /// Smallest capacity the slabs grow to, in pages.
    public let minimumGrowthPages: Int

    public init(
        pageSize: Int = 64, kvHeads: Int, keyHeadDim: Int, valueHeadDim: Int, dtype: DType,
        minimumGrowthPages: Int = 256
    ) throws {
        guard pageSize > 0, kvHeads > 0, keyHeadDim > 0, valueHeadDim > 0, minimumGrowthPages > 0,
            [.float16, .bfloat16, .float32].contains(dtype)
        else {
            throw GLM5NextDSAPageBook.Failure.invalid("attention page geometry or element type")
        }
        book = GLM5NextDSAPageBook(pageSize: pageSize)
        self.kvHeads = kvHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.dtype = dtype
        self.minimumGrowthPages = minimumGrowthPages
    }

    var keyWidth: Int { kvHeads * keyHeadDim }
    var valueWidth: Int { kvHeads * valueHeadDim }

    /// Bytes of one page (keys and values).
    public var pageBytes: Int { pageSize * (keyWidth + valueWidth) * dtype.size }
    public var capacityPages: Int { lock.withLock { book.capacityPages } }
    public var capacityBytes: Int { capacityPages * pageBytes }
    public var allocatedPages: Int { lock.withLock { book.allocatedPages } }
    public var allocatedBytes: Int { allocatedPages * pageBytes }

    /// Pages one layer's history of `tokens` tokens holds, plus the one page a
    /// copy on write may hold beside it until its step commits.
    public func pages(tokens: Int) -> Int {
        tokens <= 0 ? 0 : (tokens + pageSize - 1) / pageSize + 1
    }

    /// Bytes `reserve(pages:)` would add to the slabs now (zero when the free
    /// capacity covers it). While they grow, the old slabs and the new ones
    /// coexist (the copy): a caller's memory gate also admits `capacityBytes`.
    public func growthBytes(reserving pages: Int) -> Int {
        lock.withLock {
            let required = book.requiredPages(adding: pages)
            guard required > book.capacityPages else { return 0 }
            return (grownCapacity(required: required) - book.capacityPages) * pageBytes
        }
    }

    private func grownCapacity(required: Int) -> Int {
        let grown = max(required, book.capacityPages + book.capacityPages / 2, minimumGrowthPages)
        return (grown + 63) / 64 * 64
    }

    /// Reserves `pages` for a request, growing the slabs if the pool cannot
    /// hold every reservation and live page. Call between steps only.
    public func reserve(pages: Int) throws -> GLM5NextDSAPageBook.Account {
        try lock.withLock {
            let required = book.requiredPages(adding: pages)
            if required > book.capacityPages {
                try growSlabs(to: grownCapacity(required: required))
            }
            return book.openAccount(reservedPages: pages)
        }
    }

    /// Ends `account`'s reservation (its live pages stay where prefix entries
    /// hold them).
    public func release(_ account: GLM5NextDSAPageBook.Account) {
        lock.withLock { book.close(account) }
    }

    /// Frees the pages released since the last call. An owner calls it once
    /// every graph built so far was submitted for evaluation.
    public func recycle() { lock.withLock { book.recycle() } }

    /// Between requests: frees the retired pages and gives the slabs back when
    /// no page is held, or shrinks them to the live pages (rounded up);
    /// returns the bytes released.
    @discardableResult
    public func trim() throws -> Int {
        try lock.withLock {
            book.recycle()
            let before = book.capacityPages
            guard before > 0, book.outstandingPages == 0 else { return 0 }
            let extent = book.usedExtent
            if extent == 0 {
                slabs = []
                try book.shrink(to: 0)
            } else {
                let target = (extent + 63) / 64 * 64
                guard target < before else { return 0 }
                let rows = target * pageSize
                // A fresh buffer each: a slice would keep the whole slab alive.
                let kept = slabs.map { MLX.where(MLXArray(true), $0[..<rows], $0[..<rows]) }
                eval(kept)
                slabs = kept
                try book.shrink(to: target)
            }
            return (before - book.capacityPages) * pageBytes
        }
    }

    private func growSlabs(to pages: Int) throws {
        let rows = pages * pageSize
        let widths = [keyWidth, valueWidth]
        let maximum = GPU.deviceInfo().maxBufferSize
        guard widths.allSatisfy({ rows * $0 * dtype.size <= maximum }) else {
            throw GLM5NextDSAPageBook.Failure.invalid("slab beyond the device buffer limit")
        }
        let grown: [MLXArray] = widths.enumerated().map { index, width in
            guard slabs.indices.contains(index) else {
                return MLXArray.zeros([rows, width], dtype: dtype)
            }
            // The copy reads the old slab after every write already submitted to it.
            return concatenated(
                [slabs[index], MLXArray.zeros([rows - slabs[index].dim(0), width], dtype: dtype)],
                axis: 0)
        }
        eval(grown)
        slabs = grown
        book.grow(to: pages)
    }

    /// Leaves `table` if no other history holds it, else replaces it with its
    /// clone. An empty `table` becomes a new one.
    func uniqueTable(_ table: inout GLM5NextDSAPageTable?) {
        if table == nil {
            table = GLM5NextDSAPageTable(book: book, lock: lock)
            return
        }
        guard !isKnownUniquelyReferenced(&table) else { return }
        lock.lock()
        let clone = table!.clone()
        lock.unlock()
        table = clone
    }

    /// Appends `[1, kvHeads, T, headDim]` keys and values to `table` (unique)
    /// after `fence`; returns the write's fence.
    func append(
        to table: GLM5NextDSAPageTable, account: GLM5NextDSAPageBook.Account?, fence: MLXArray,
        keys: MLXArray, values: MLXArray
    ) throws -> MLXArray {
        let count = keys.dim(2)
        guard keys.shape == [1, kvHeads, count, keyHeadDim],
            values.shape == [1, kvHeads, count, valueHeadDim],
            keys.dtype == dtype, values.dtype == dtype, count > 0
        else {
            throw GLM5NextDSAPageBook.Failure.invalid(
                "attention append of \(keys.shape) \(keys.dtype), \(values.shape) \(values.dtype)")
        }
        let plan = try lock.withLock { try table.planAppend(count: count, account: account) }
        return try PagedAttentionKVKernels.write(
            slabs: slabs, pageSize: pageSize, heads: kvHeads, keyHeadDim: keyHeadDim,
            valueHeadDim: valueHeadDim, slots: plan.slots, clone: plan.clone, fence: fence,
            keys: keys,
            values: values)
    }

    /// The first `count` tokens of `table` as `[1, kvHeads, count, headDim]`
    /// keys and values after `fence`, and the read's fence.
    func gather(_ table: GLM5NextDSAPageTable, count: Int, fence: MLXArray) throws -> (
        keys: MLXArray, values: MLXArray, fence: MLXArray
    ) {
        let pages = try lock.withLock { () throws -> MLXArray in
            guard table.pages.count * pageSize >= count else {
                throw GLM5NextDSAPageBook.Failure.invalid("gather past the table")
            }
            return table.deviceArray
        }
        return try PagedAttentionKVKernels.gather(
            slabs: slabs, pageSize: pageSize, heads: kvHeads, keyHeadDim: keyHeadDim,
            valueHeadDim: valueHeadDim, pages: pages, count: count, fence: fence)
    }

    func truncate(_ table: GLM5NextDSAPageTable, to count: Int) {
        lock.withLock { table.truncate(to: count) }
    }
}

/// One attention layer's keys and values for one request, in a
/// `PagedAttentionKVPool`'s pages. Behaves as `KVCacheSimple` for its
/// readers: `update` returns the `0 ..< offset` keys and values (gathered
/// from the pages), `state` the same arrays, `trim` rewinds, `copy` shares
/// the pages (the first of two copies to append takes its own table, copying
/// at most one shared partial page). Batch 1 only.
public final class PagedAttentionKVCache: KVCache, KVCacheStepAttending {
    /// Single-query decode steps attend the pages in place
    /// (`PagedAttentionDecode`, MLX's vector kernels with paged rows: the
    /// gathered path's bits). `PAGED_ATTENTION_DECODE=0` gathers instead
    /// (A/B); the tests switch it per run.
    nonisolated(unsafe) public static var decodeInPlace =
        ProcessInfo.processInfo.environment["PAGED_ATTENTION_DECODE"] != "0"
    /// The shortest history read in place: below it the gather (one launch,
    /// then MLX's own kernels) costs less host time than the paged kernels'
    /// launches; both give the same bits. `PAGED_ATTENTION_DECODE_MIN_TOKENS`
    /// overrides it (measured on Ultra A: NOTES).
    nonisolated(unsafe) public static var decodeInPlaceMinimumTokens =
        ProcessInfo.processInfo.environment["PAGED_ATTENTION_DECODE_MIN_TOKENS"].flatMap(Int.init)
        ?? 12288
    public let pool: PagedAttentionKVPool
    public private(set) var offset = 0
    public var maxSize: Int? { nil }
    /// The reservation this history's new pages count against.
    public var account: GLM5NextDSAPageBook.Account?
    private var table: GLM5NextDSAPageTable?
    /// Orders this history's page writes and reads (each launch takes and
    /// returns it).
    private var fence = PagedAttentionKVKernels.initialFence
    /// Keys and values gathered at `offset`.
    private var gathered: (offset: Int, keys: MLXArray, values: MLXArray)?

    public init(pool: PagedAttentionKVPool, account: GLM5NextDSAPageBook.Account?) {
        self.pool = pool
        self.account = account
    }

    /// Pages this history's table holds.
    public var heldPages: Int { table?.pages.count ?? 0 }
    /// Bytes of the pages this history's table holds.
    public var pagedBytes: Int { heldPages * pool.pageBytes }

    /// The fence of the last page launch: evaluating it completes every
    /// write before it.
    public func innerState() -> [MLXArray] { offset > 0 ? [fence] : [] }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        do {
            try append(keys: keys, values: values)
            return try read()
        } catch {
            preconditionFailure("paged attention KV: \(error)")
        }
    }

    private func append(keys: MLXArray, values: MLXArray) throws {
        guard keys.ndim == 4, keys.dim(0) == 1 else {
            throw GLM5NextDSAPageBook.Failure.invalid("paged attention KV holds one request")
        }
        guard keys.dim(2) > 0 else { return }
        pool.uniqueTable(&table)
        fence = try pool.append(
            to: table!, account: account, fence: fence, keys: keys, values: values)
        gathered = nil
        offset += keys.dim(2)
    }

    /// One decode step (`KVCacheStepAttending`): appends the step's row and
    /// attends `0 ..< offset` straight from the pages, where MLX would run its
    /// vector kernels (`PagedAttentionDecode.plan`); nil otherwise (nothing changed).
    public func updateAndAttendStep(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        guard Self.decodeInPlace, offset + 1 >= Self.decodeInPlaceMinimumTokens,
            Device.defaultDevice().deviceType == .gpu, case .none = mask,
            queries.ndim == 4, queries.dim(0) == 1,
            queries.dim(2) == 1, keys.dim(2) == 1, queries.dtype == pool.dtype,
            queries.dim(3) == pool.keyHeadDim,
            let plan = PagedAttentionDecode.plan(
                queryHeads: queries.dim(1), kvHeads: pool.kvHeads, headDim: pool.keyHeadDim,
                valueHeadDim: pool.valueHeadDim, N: offset + 1, dtype: pool.dtype)
        else { return nil }
        do {
            try append(keys: keys, values: values)
            let pages = try pool.lock.withLock { () throws -> MLXArray in
                guard let table, table.pages.count * pool.pageSize >= offset else {
                    throw GLM5NextDSAPageBook.Failure.invalid("decode past the table")
                }
                return table.deviceArray
            }
            let read = PagedAttentionDecode.attend(
                queries: queries, scale: scale, slabs: pool.slabs, pageSize: pool.pageSize,
                kvHeads: pool.kvHeads, headDim: pool.keyHeadDim, pages: pages, count: offset,
                fence: fence, plan: plan)
            fence = read.fence
            return read.output
        } catch {
            preconditionFailure("paged attention decode: \(error)")
        }
    }

    private func read() throws -> (MLXArray, MLXArray) {
        if let gathered, gathered.offset == offset { return (gathered.keys, gathered.values) }
        guard let table, offset > 0 else {
            throw GLM5NextDSAPageBook.Failure.invalid("read of an empty paged attention KV")
        }
        let read = try pool.gather(table, count: offset, fence: fence)
        fence = read.fence
        gathered = (offset, read.keys, read.values)
        return (read.keys, read.values)
    }

    public var state: [MLXArray] {
        get {
            guard offset > 0, let read = try? read() else { return [] }
            return [read.0, read.1]
        }
        set {
            precondition(
                newValue.isEmpty || (newValue.count == 2 && newValue[0].ndim == 4),
                "paged attention KV state must be [keys, values]")
            table = nil
            gathered = nil
            offset = 0
            guard newValue.count == 2, newValue[0].dim(2) > 0 else { return }
            do {
                try append(keys: newValue[0], values: newValue[1])
            } catch {
                preconditionFailure("paged attention KV state: \(error)")
            }
        }
    }

    public var metaState: [String] {
        get { [""] }
        set {
            precondition(
                newValue.count == 1 && newValue[0].isEmpty, "paged attention KV has no meta state")
        }
    }

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, max(0, n))
        guard trimmed > 0 else { return 0 }
        let kept = offset - trimmed
        if kept == 0 {
            table = nil
        } else {
            pool.uniqueTable(&table)
            pool.truncate(table!, to: kept)
        }
        gathered = nil
        offset = kept
        return trimmed
    }

    /// Shares this history's pages (O(1)); the first of the two to append
    /// takes its own page table.
    /// `BaseKVCache`'s (as `KVCacheSimple`): none for one token, causal
    /// otherwise, an array where asked or past a window.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    public func copy() -> any KVCache {
        let result = PagedAttentionKVCache(pool: pool, account: account)
        result.table = table
        result.fence = fence
        result.offset = offset
        return result
    }
}

/// The pool's two launches. Both take the two slabs as mutable inputs: MLX
/// then treats every launch that touches a slab as writing it, so later
/// launches (and command buffers) wait for earlier ones in submission order.
/// Data order within one history is the fence each launch takes and returns.
/// Bitwise: pure element copies.
enum PagedAttentionKVKernels {
    /// A fresh history's fence: a shared read-only `[1]` int32 leaf.
    static var initialFence: MLXArray { GLM5NextKernelConstants.shared.int32s([0]) }

    static func write(
        slabs: [MLXArray], pageSize: Int, heads: Int, keyHeadDim: Int, valueHeadDim: Int,
        slots: [Int32], clone: (source: Int32, destination: Int32, rows: Int)?, fence: MLXArray,
        keys: MLXArray, values: MLXArray
    ) throws -> MLXArray {
        guard slabs.count == 2, !slots.isEmpty else {
            throw GLM5NextDSAPageBook.Failure.invalid("attention write without slabs or tokens")
        }
        let rows = slots.count + (clone?.rows ?? 0)
        let columns = heads * (keyHeadDim + valueHeadDim)
        let constants = GLM5NextKernelConstants.shared
        let cloneArgs =
            clone.map { MLXArray([$0.source, $0.destination, Int32($0.rows)]) }
            ?? constants.int32s([0, 0, 0])
        let slotArray = slots.count == 1 ? constants.int32s(slots) : MLXArray(slots)
        return writeKernel(
            [
                keys, values, slotArray, cloneArgs, constants.int32s([Int32(slots.count)]), fence,
                slabs[0], slabs[1],
            ],
            template: [("PS", pageSize), ("H", heads), ("DK", keyHeadDim), ("DV", valueHeadDim)],
            grid: (columns, rows, 1), threadGroup: (min(256, columns), 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    static func gather(
        slabs: [MLXArray], pageSize: Int, heads: Int, keyHeadDim: Int, valueHeadDim: Int,
        pages: MLXArray, count: Int, fence: MLXArray
    ) throws -> (keys: MLXArray, values: MLXArray, fence: MLXArray) {
        guard slabs.count == 2, count > 0, pages.size * pageSize >= count else {
            throw GLM5NextDSAPageBook.Failure.invalid("attention gather without slabs or pages")
        }
        let columns = heads * (keyHeadDim + valueHeadDim)
        let outputs = gatherKernel(
            [
                slabs[0], slabs[1], pages, GLM5NextKernelConstants.shared.int32s([Int32(count)]),
                fence,
            ],
            template: [("PS", pageSize), ("H", heads), ("DK", keyHeadDim), ("DV", valueHeadDim)],
            grid: (columns, count, 1), threadGroup: (min(256, columns), 1, 1),
            outputShapes: [[1, heads, count, keyHeadDim], [1, heads, count, valueHeadDim], [1]],
            outputDTypes: [slabs[0].dtype, slabs[1].dtype, .int32])
        return (outputs[0], outputs[1], outputs[2])
    }

    /// Row r < clone rows copies source page row r to the destination page;
    /// after them, new token t = r - clone rows (input `[1, H, T, D]`, row
    /// contiguous) goes to `slots[t]`.
    private static let writeSource = """
        const uint e = thread_position_in_grid.x;
        const int r = int(thread_position_in_grid.y);
        const int cloned = clone[2];
        const bool fresh = r >= cloned;
        const int t = r - cloned;
        const int T = count[0];
        const size_t src = size_t(clone[0]) * PS + size_t(r);
        const size_t dst = fresh ? size_t(slots[t]) : size_t(clone[1]) * PS + size_t(r);
        if (e < uint(H * DK)) {
            const uint h = e / DK, d = e % DK;
            k_slab[dst * (H * DK) + e] =
                fresh ? k_in[(size_t(h) * T + t) * DK + d] : k_slab[src * (H * DK) + e];
        } else {
            const uint v = e - uint(H * DK);
            const uint h = v / DV, d = v % DV;
            v_slab[dst * (H * DV) + v] =
                fresh ? v_in[(size_t(h) * T + t) * DV + d] : v_slab[src * (H * DV) + v];
        }
        if (e == 0 && r == 0) { fence_out[0] = fence[0] ^ 1; }
        """

    /// Output token n of `[1, H, N, D]` is the slab row of token n.
    private static let gatherSource = """
        const uint e = thread_position_in_grid.x;
        const int n = int(thread_position_in_grid.y);
        const int N = count[0];
        const size_t row = size_t(pages[n / PS]) * PS + size_t(n % PS);
        if (e < uint(H * DK)) {
            const uint h = e / DK, d = e % DK;
            k_out[(size_t(h) * N + n) * DK + d] = k_slab[row * (H * DK) + e];
        } else {
            const uint v = e - uint(H * DK);
            const uint h = v / DV, d = v % DV;
            v_out[(size_t(h) * N + n) * DV + d] = v_slab[row * (H * DV) + v];
        }
        if (e == 0 && n == 0) { fence_out[0] = fence[0] ^ 1; }
        """

    private static let writeKernel = MLXFast.metalKernel(
        name: "paged_attention_kv_write",
        inputNames: ["k_in", "v_in", "slots", "clone", "count", "fence", "k_slab", "v_slab"],
        outputNames: ["fence_out"], source: writeSource, mutableInputs: ["k_slab", "v_slab"])
    private static let gatherKernel = MLXFast.metalKernel(
        name: "paged_attention_kv_gather",
        inputNames: ["k_slab", "v_slab", "pages", "count", "fence"],
        outputNames: ["k_out", "v_out", "fence_out"], source: gatherSource,
        mutableInputs: ["k_slab", "v_slab"])
}
