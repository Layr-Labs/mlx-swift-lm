// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon

/// Request-owned physical history. Pool keys are rebuilt from the retained raw
/// index/gate history, so a trim cannot expose a stale compressed pool.
///
/// Storage keeps geometric spare capacity along the token axis and appends
/// with slice updates, so a request's appends copy amortized O(1) history per
/// token instead of re-concatenating the whole history every round. MLX keeps
/// value semantics: an update writes in place only when no other array (a
/// copy, checkpoint or pending graph) still references the buffer. Readers
/// only ever see the logical `0 ..< offset` prefix.
///
/// Paged (`init(pagePool:account:)`): the history lives in the pool's pages
/// (`GLM5NextDSAPagePool`) instead. An append writes only its own rows into
/// pages in place; a reader gets the same `0 ..< offset` arrays, gathered from
/// the pages in token order (once per revision), so every attention result is
/// the contiguous storage's. Copies share pages; a copy about to append takes
/// its own page table (`GLM5NextDSAPagePool.uniqueTable`), and a page another
/// holder still reads is copied before it is written. Batch 1 only.
public final class GLM5NextDSACache: KVCache {
    /// Smallest token capacity allocated by an append that must grow.
    static let minimumCapacity = 256

    /// `.exact` keeps every append a single concatenation of the logical
    /// prefix and the new columns into a fresh, exactly sized buffer. Owners
    /// that publish each round's history as a compact measured snapshot use it
    /// so no spare columns are allocated or charged.
    enum Growth { case geometric, exact }
    let growth: Growth

    private(set) var revision: UInt64 = 0
    /// `[latents, indexKeys, poolGates, validity]` with `capacity >= offset`
    /// columns; columns past `offset` are unspecified and never read.
    private var storage: [MLXArray] = []
    public private(set) var offset = 0
    public var latents: MLXArray? { logical(0) }
    public var indexKeys: MLXArray? { logical(1) }
    public var poolGates: MLXArray? { logical(2) }
    public var validity: MLXArray? { logical(3) }
    var capacity: Int { pagePool.map { heldPages * $0.pageSize } ?? (storage.first?.dim(1) ?? 0) }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }
    /// Fitting appends on the default GPU stream run `GLM5NextDSAAppendKernel`
    /// (same bits as the slice updates); false is the slice-update oracle.
    var usesAppendKernel = true
    /// The most token columns geometric growth allocates: an owner that knows
    /// its request's longest history (prompt, outputs and any verification
    /// window) sets it, so the spare columns never exceed that request and its
    /// memory charge can be the request's own length instead of twice it. An
    /// append past it still grows to exactly what it needs. Columns past
    /// `offset` are never read, so values are identical with or without it.
    /// Nil (the default) is unbounded doubling. Copies keep it.
    public var capacityLimit: Int?
    /// The pages this history lives in; nil keeps the contiguous storage.
    public let pagePool: GLM5NextDSAPagePool?
    /// The reservation this history's new pages count against.
    public var pageAccount: GLM5NextDSAPageBook.Account?
    private var pageTable: GLM5NextDSAPageTable?
    /// Orders this history's page writes and reads (each launch takes and
    /// returns it).
    private var pageFence = GLM5NextDSAPageKernels.initialFence
    /// `[latents, keys, gates, validity]` gathered for `revision`.
    private var gathered: (revision: UInt64, arrays: [MLXArray])?
    /// `[keys, gates, validity]` alone gathered for `revision`.
    private var gatheredIndex: (revision: UInt64, arrays: [MLXArray])?
    public convenience init() { self.init(growth: .geometric) }
    init(growth: Growth) {
        self.growth = growth
        pagePool = nil
    }
    /// An empty paged history whose pages count against `account`.
    public init(pagePool: GLM5NextDSAPagePool, account: GLM5NextDSAPageBook.Account?) {
        growth = .geometric
        self.pagePool = pagePool
        pageAccount = account
    }
    /// The evaluation roots of the stored history: the arrays themselves, or
    /// for a paged history the fence of its last page launch (evaluating it
    /// completes every write before it).
    public func innerState() -> [MLXArray] {
        guard pagePool != nil else { return state }
        return offset > 0 ? [pageFence] : []
    }
    /// Pages this history's table holds (0 when contiguous).
    public var heldPages: Int { pageTable?.pages.count ?? 0 }
    /// Bytes of the pages this history's table holds; nil when contiguous.
    public var pagedBytes: Int? { pagePool.map { heldPages * $0.pageBytes } }
    /// Exactly the logical history; serialization never sees spare capacity.
    public var state: [MLXArray] {
        get {
            guard pagePool != nil else { return (0 ..< storage.count).compactMap(logical) }
            return pagedHistory() ?? []
        }
        set {
            if let pagePool {
                replacePages(newValue, pool: pagePool)
                return
            }
            precondition(
                newValue.isEmpty
                    || (newValue.count == 4 && newValue[0].ndim == 3
                        && newValue[1].ndim == 3 && newValue[2].shape == newValue[1].shape
                        && newValue[3].shape == Array(newValue[0].shape.prefix(2))
                        && newValue[3].dtype == .bool && newValue[1].dim(0) == newValue[0].dim(0)
                        && newValue[1].dim(1) == newValue[0].dim(1)),
                "invalid GLM DSA serialized state")
            replace(storage: newValue, offset: newValue.first?.dim(1) ?? 0)
        }
    }
    public var metaState: [String] {
        get { ["glm5_dsa_v1", String(offset)] }
        set { precondition(newValue == metaState, "GLM DSA metadata must match serialized arrays") }
    }
    func append(latents: MLXArray, keys: MLXArray, gates: MLXArray, validity: MLXArray) throws {
        guard latents.ndim == 3, keys.ndim == 3, gates.shape == keys.shape,
            validity.shape == Array(latents.shape.prefix(2)), validity.dtype == .bool,
            keys.dim(0) == latents.dim(0), keys.dim(1) == latents.dim(1)
        else { throw GLM5NextLoadError.invalid("DSA append geometry") }
        if let pagePool {
            try appendPages(
                latents: latents, keys: keys, gates: gates, validity: validity, pool: pagePool)
            return
        }
        let incoming = [latents, keys, gates, validity]
        if storage.isEmpty {
            state = incoming
            return
        }
        for (old, next) in zip(storage, incoming) {
            guard old.dtype == next.dtype, old.dim(0) == next.dim(0),
                Array(old.shape.dropFirst(2)) == Array(next.shape.dropFirst(2))
            else { throw GLM5NextLoadError.invalid("DSA history changed batch, dtype or width") }
        }
        let count = latents.dim(1)
        let end = offset + count
        guard end > offset else { return }
        if growth == .exact || end > capacity {
            // Copy the logical prefix once; geometric growth adds spare
            // columns that this and later appends fill in place.
            let doubled = max(2 * capacity, Self.minimumCapacity)
            let grown = growth == .exact ? end : max(end, min(doubled, capacityLimit ?? doubled))
            let next = zip(storage, incoming).map { old, new -> MLXArray in
                let prefix = offset == old.dim(1) ? old : old[0..., ..<offset]
                guard grown > end else { return concatenated([prefix, new], axis: 1) }
                var shape = old.shape
                shape[1] = grown - end
                return concatenated(
                    [prefix, new, MLXArray.zeros(shape, dtype: old.dtype)], axis: 1)
            }
            replace(storage: next, offset: end)
            return
        }
        if usesAppendKernel,
            let next = GLM5NextDSAAppendKernel.append(
                storage: storage, incoming: incoming, offset: offset)
        {
            replace(storage: next, offset: end)
            return
        }
        let next = zip(storage, incoming).map { old, new -> MLXArray in
            // Never mutate `old`: copies and snapshots share its object.
            let updated = old.sharingArray()
            updated[0..., offset ..< end] = new
            return updated
        }
        replace(storage: next, offset: end)
    }
    public func truncate(to count: Int) throws {
        guard count >= 0 && count <= offset else {
            throw GLM5NextLoadError.invalid("DSA rewind position")
        }
        if count == offset { return }
        if let pagePool {
            // Rewound pages go back to the pool; the tail page's rewound slots
            // are overwritten by the next append (or copied around it).
            if count == 0 {
                pageTable = nil
            } else {
                pagePool.uniqueTable(&pageTable)
                pagePool.truncate(pageTable!, to: count)
            }
            gathered = nil
            gatheredIndex = nil
            replace(storage: [], offset: 0)
            offset = count
            return
        }
        if count == 0 {
            state = []
            return
        }
        // Rewound columns become spare capacity; the next append overwrites them.
        replace(storage: storage, offset: count)
    }
    @discardableResult public func trim(_ n: Int) -> Int {
        guard n >= 0 else { return 0 }
        let removed = min(n, offset)
        // Bounds are established above; this does not suppress a failed rewind.
        do { try truncate(to: offset - removed) } catch { preconditionFailure("\(error)") }
        return removed
    }
    public func copy() -> any KVCache {
        if let pagePool {
            let result = GLM5NextDSACache(pagePool: pagePool, account: pageAccount)
            result.adopt(self)
            return result
        }
        let result = GLM5NextDSACache(growth: growth)
        result.usesAppendKernel = usesAppendKernel
        result.capacityLimit = capacityLimit
        result.adopt(self)
        return result
    }
    /// Shares the storage arrays: no path mutates a stored array object (an
    /// in-place append writes through a fresh wrapper), so copies stay
    /// independent while MLX may still donate an unshared buffer. A paged
    /// history shares the page table; the first of them to change it takes
    /// its own.
    func adopt(_ other: GLM5NextDSACache) {
        precondition(pagePool === other.pagePool, "DSA histories of different storage")
        guard pagePool != nil else {
            replace(storage: other.storage, offset: other.offset)
            return
        }
        pageTable = other.pageTable
        pageFence = other.pageFence
        replace(storage: [], offset: 0)
        offset = other.offset
        // Gathered arrays stay with the step that read them: a committed
        // history keeps its pages only, never a second copy of its rows.
        gathered = nil
        gatheredIndex = nil
    }
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("GLM DSA requires latent, index, gate and validity together")
    }
    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        precondition(windowSize == nil, "GLM DSA is not a sliding-window cache")
        return .causal
    }
    private func logical(_ index: Int) -> MLXArray? {
        if pagePool != nil {
            // The index planes alone, unless the whole history is gathered.
            if index == 0 || gathered?.revision == revision { return pagedHistory()?[index] }
            return pagedIndex()?[index - 1]
        }
        guard storage.indices.contains(index) else { return nil }
        let array = storage[index]
        guard offset < array.dim(1) else { return array }
        return array.ndim == 2 ? array[0..., ..<offset] : array[0..., ..<offset, 0...]
    }
    private func replace(storage next: [MLXArray], offset count: Int) {
        storage = next
        offset = next.isEmpty ? 0 : count
        precondition(revision < UInt64.max, "DSA cache revision exhausted")
        revision += 1
    }

    // MARK: - Paged storage

    /// The `0 ..< offset` history gathered from the pages, once per revision.
    private func pagedHistory() -> [MLXArray]? {
        guard let pagePool, let pageTable, offset > 0 else { return nil }
        if let gathered, gathered.revision == revision { return gathered.arrays }
        do {
            let read = try pagePool.gather(pageTable, count: offset, fence: pageFence)
            pageFence = read.fence
            gathered = (revision, read.history)
            return read.history
        } catch {
            preconditionFailure("paged DSA gather: \(error)")
        }
    }

    /// The `0 ..< offset` index planes `[keys, gates, validity]` gathered from
    /// the pages, once per revision (a decode step's selection reads them
    /// whole and its attention only the selected latents).
    private func pagedIndex() -> [MLXArray]? {
        guard let pagePool, let pageTable, offset > 0 else { return nil }
        if let gatheredIndex, gatheredIndex.revision == revision { return gatheredIndex.arrays }
        do {
            let read = try pagePool.gatherIndex(pageTable, count: offset, fence: pageFence)
            pageFence = read.fence
            gatheredIndex = (revision, read.index)
            return read.index
        } catch {
            preconditionFailure("paged DSA index gather: \(error)")
        }
    }

    /// Paged: `[latents, keys, gates, validity]` of the first `count` tokens.
    func pagedPrefix(count: Int) throws -> [MLXArray] {
        guard let pagePool, let pageTable, count > 0, count <= offset else {
            throw GLM5NextLoadError.invalid("paged DSA prefix read")
        }
        if count == offset, let history = pagedHistory() { return history }
        let read = try pagePool.gather(pageTable, count: count, fence: pageFence)
        pageFence = read.fence
        return read.history
    }

    /// Paged: the latents `selection` reads, `[1, T * W, width]` in selection
    /// order, and the selection rereading them: a consumer of both reads the
    /// same latent rows in the same order as of the whole history.
    func selectedLatents(_ selection: GLM5NextSparseAttention.Selection) throws -> (
        latents: MLXArray, selection: GLM5NextSparseAttention.Selection
    ) {
        guard let pagePool, let pageTable, selection.historyLength <= offset else {
            throw GLM5NextLoadError.invalid("paged DSA selected read")
        }
        let read = try pagePool.gatherSelected(
            pageTable, count: selection.historyLength, ids: selection.indices, fence: pageFence)
        pageFence = read.fence
        return (read.latents, selection.rereading(read.ids, historyLength: read.latents.dim(1)))
    }

    private func appendPages(
        latents: MLXArray, keys: MLXArray, gates: MLXArray, validity: MLXArray,
        pool: GLM5NextDSAPagePool
    ) throws {
        guard latents.dim(0) == 1 else {
            throw GLM5NextLoadError.invalid("paged DSA history holds one request")
        }
        guard latents.dim(1) > 0 else { return }
        pool.uniqueTable(&pageTable)
        pageFence = try pool.append(
            to: pageTable!, account: pageAccount, fence: pageFence, latents: latents, keys: keys,
            gates: gates, validity: validity)
        gathered = nil
        gatheredIndex = nil
        let end = offset + latents.dim(1)
        replace(storage: [], offset: 0)
        offset = end
    }

    /// This committed history's rows for a reader on another thread
    /// (`GLM5NextDSAHistoryRows`: its pages, held as a copy holds them, or its
    /// arrays), after its last write is evaluated; nil when empty.
    public func historyRows() -> GLM5NextDSAHistoryRows? {
        guard offset > 0 else { return nil }
        if let pagePool {
            guard let pageTable else { return nil }
            eval(pageFence)
            return GLM5NextDSAHistoryRows(pool: pagePool, table: pageTable, count: offset)
        }
        let arrays = (0 ..< storage.count).compactMap(logical)
        guard arrays.count == 4 else { return nil }
        eval(arrays)
        return GLM5NextDSAHistoryRows(arrays: arrays, count: offset)
    }

    /// Fills this empty history with serialized rows (`state`'s
    /// `[latents, keys, gates, validity]`, `[1, n, ...]`): a paged history
    /// writes them into fresh pages counted against `pageAccount`, a
    /// contiguous one keeps them. As `state`'s setter, but a geometry, element
    /// type or page refusal throws instead of trapping, and leaves it empty.
    public func restoreRows(_ arrays: [MLXArray]) throws {
        guard offset == 0, arrays.count == 4, arrays[0].ndim == 3, arrays[0].dim(0) == 1,
            arrays[0].dim(1) > 0,
            arrays[1].ndim == 3, arrays[1].dim(0) == 1, arrays[1].dim(1) == arrays[0].dim(1),
            arrays[2].shape == arrays[1].shape, arrays[3].shape == [1, arrays[0].dim(1)],
            arrays[3].dtype == .bool
        else { throw GLM5NextLoadError.invalid("DSA history rows to restore") }
        if let pagePool {
            // The pool checks widths and element types before it takes a page.
            pageTable = nil
            gathered = nil
            gatheredIndex = nil
            try appendPages(
                latents: arrays[0], keys: arrays[1], gates: arrays[2], validity: arrays[3],
                pool: pagePool)
            return
        }
        replace(storage: arrays, offset: arrays[0].dim(1))
    }

    /// Writes `arrays` (a serialized history) into fresh pages.
    private func replacePages(_ arrays: [MLXArray], pool: GLM5NextDSAPagePool) {
        precondition(
            arrays.isEmpty
                || (arrays.count == 4 && arrays[0].ndim == 3 && arrays[0].dim(0) == 1
                    && arrays[1].shape == arrays[2].shape
                    && arrays[3].shape == Array(arrays[0].shape.prefix(2))
                    && arrays[3].dtype == .bool),
            "invalid GLM DSA serialized state")
        pageTable = nil
        gathered = nil
        gatheredIndex = nil
        replace(storage: [], offset: 0)
        guard !arrays.isEmpty, arrays[0].dim(1) > 0 else { return }
        do {
            try appendPages(
                latents: arrays[0], keys: arrays[1], gates: arrays[2], validity: arrays[3],
                pool: pool)
        } catch {
            preconditionFailure("paged DSA state: \(error)")
        }
    }
}

/// Fixed-size KDA state plus one forward-start rollback checkpoint. Arbitrary
/// prefix truncation cannot be implemented by slicing this recurrent state.
/// A forward that captured its window's per-position states (an MTP
/// verification window) can also be truncated to any token of that window.
public final class GLM5NextKDACache: KVCache {
    private(set) var revision: UInt64 = 0
    var current: GLM5NextLinearAttention.State?
    private var rollback: GLM5NextLinearAttention.State?
    private var rollbackOffset = 0
    /// The last forward's per-position states when it captured them: entry t
    /// is the state after token `windowStart + t`. Dropped by the next
    /// forward, any truncation and any state replacement.
    private var window: GLM5NextKDASublayer.CapturedStates?
    private var windowStart = 0
    public private(set) var offset = 0
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { false }
    public init() {}
    public func innerState() -> [MLXArray] {
        state + (rollback.map { [$0.convolution, $0.recurrent] } ?? [])
            + (window.map { [$0.convolution, $0.recurrent] } ?? [])
    }
    public var state: [MLXArray] {
        get { current.map { [$0.convolution, $0.recurrent] } ?? [] }
        set {
            precondition(
                newValue.isEmpty
                    || (newValue.count == 2 && newValue[0].ndim == 3
                        && newValue[1].ndim == 4 && newValue[1].dtype == .float32
                        && newValue[0].dim(0) == newValue[1].dim(0)),
                "invalid GLM KDA serialized state")
            current =
                newValue.isEmpty
                ? nil : Self.compact(.init(convolution: newValue[0], recurrent: newValue[1]))
            rollback = nil
            rollbackOffset = 0
            window = nil
            if newValue.isEmpty { offset = 0 }
            changed()
        }
    }
    public var metaState: [String] {
        get { ["glm5_kda_v1", String(offset)] }
        set {
            precondition(
                newValue.count == 2 && newValue[0] == "glm5_kda_v1", "invalid KDA metadata")
            guard let count = Int(newValue[1]), count >= 0, count == 0 || current != nil else {
                preconditionFailure("invalid KDA offset")
            }
            offset = count
            changed()
        }
    }
    /// `window`: the forward's captured per-position states (`[B,count,...]`),
    /// kept until the next forward or truncation so any of its tokens can end
    /// the committed prefix.
    func advance(
        _ next: GLM5NextLinearAttention.State, count: Int,
        window captured: GLM5NextKDASublayer.CapturedStates? = nil
    ) {
        rollback = current.map { Self.clone($0) }
        rollbackOffset = offset
        current = Self.compact(next)
        window = captured
        windowStart = offset
        offset += count
        changed()
    }
    func canTruncate(to count: Int) -> Bool {
        count >= 0 && count <= offset
            && (count == 0 || count == offset || count == rollbackOffset
                || (window != nil && count > windowStart))
    }
    public func truncate(to count: Int) throws {
        guard canTruncate(to: count) else {
            throw GLM5NextLoadError.invalid(
                "KDA rewind requires a saved boundary plus accepted-token replay")
        }
        if count == offset {
            // The logical state is unchanged; only the window is released.
            window = nil
            return
        }
        if let window, count > windowStart {
            // Entry t is the state after window token t, from the same
            // recurrence that produced the window's final state.
            let index = count - windowStart - 1
            current = Self.compact(
                .init(
                    convolution: window.convolution[0..., index],
                    recurrent: window.recurrent[0..., index]))
        } else {
            current = count == 0 ? nil : rollback.map { Self.clone($0) }
        }
        offset = count
        rollback = nil
        rollbackOffset = count
        window = nil
        changed()
    }
    @discardableResult public func trim(_ n: Int) -> Int { 0 }
    public func copy() -> any KVCache {
        let result = GLM5NextKDACache()
        result.adopt(self)
        return result
    }
    func adopt(_ other: GLM5NextKDACache) {
        current = other.current.map { Self.clone($0) }
        rollback = other.rollback.map { Self.clone($0) }
        offset = other.offset
        rollbackOffset = other.rollbackOffset
        window = other.window.map {
            .init(
                convolution: $0.convolution.sharingArray(), recurrent: $0.recurrent.sharingArray())
        }
        windowStart = other.windowStart
        changed()
    }
    private func changed() {
        precondition(revision < UInt64.max, "KDA cache revision exhausted")
        revision += 1
    }
    /// Lazy ownership boundary: the owner must evaluate and join innerState()
    /// before treating this cache as compact. Pending graphs still retain inputs.
    /// In the pinned CPU/Metal core, contiguous retains at most logical bytes
    /// plus 16 KiB of backing storage (before allocator rounding).
    private static func compact(_ value: GLM5NextLinearAttention.State)
        -> GLM5NextLinearAttention.State
    {
        .init(
            convolution: value.convolution.contiguous(),
            recurrent: value.recurrent.contiguous())
    }
    private static func clone(_ value: GLM5NextLinearAttention.State)
        -> GLM5NextLinearAttention.State
    {
        // A new object for the same array (`self[.ellipsis]`'s result without
        // the indexing work), so assigning through one never touches the other.
        .init(
            convolution: value.convolution.sharingArray(), recurrent: value.recurrent.sharingArray()
        )
    }
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("KDA must stage convolution and FP32 recurrent state together")
    }
    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        precondition(windowSize == nil, "KDA is not a sliding-window cache")
        return .none
    }
}

public final class GLM5NextTextCache {
    struct Revision: Equatable {
        let identities: [ObjectIdentifier]
        let versions: [UInt64]
        let offsets: [Int]
    }
    func revision() -> Revision {
        Revision(
            identities: layers.map { ObjectIdentifier($0 as AnyObject) },
            versions: layers.map {
                if let layer = $0 as? GLM5NextKDACache { return layer.revision }
                if let layer = $0 as? GLM5NextDSACache { return layer.revision }
                return 0
            }, offsets: layers.map(\.offset))
    }
    public let layers: [any KVCache]
    public var offset: Int { layers.first?.offset ?? 0 }
    public init(configuration: GLM5NextTextConfiguration) {
        layers = configuration.layerTypes.map { kind -> any KVCache in
            kind == .linearAttention ? GLM5NextKDACache() : GLM5NextDSACache()
        }
    }
    public init(layers: [any KVCache]) { self.layers = layers }
    public func copy() -> GLM5NextTextCache { .init(layers: layers.map { $0.copy() }) }
    func validate(_ configuration: GLM5NextTextConfiguration) throws {
        guard layers.count == configuration.hiddenLayers else {
            throw GLM5NextLoadError.invalid("cache layer count")
        }
        for (index, layer) in layers.enumerated() {
            guard layer.offset == offset,
                configuration.layerTypes[index] == .linearAttention
                    ? layer is GLM5NextKDACache : layer is GLM5NextDSACache
            else { throw GLM5NextLoadError.invalid("cache layer kind/frontier \(index)") }
        }
    }
    public func truncate(to count: Int) throws {
        for layer in layers {
            if let kda = layer as? GLM5NextKDACache {
                guard kda.canTruncate(to: count) else {
                    throw GLM5NextLoadError.invalid("KDA rewind requires replay")
                }
            } else if layer is GLM5NextDSACache {
                guard count >= 0 && count <= layer.offset else {
                    throw GLM5NextLoadError.invalid("DSA rewind position")
                }
            } else {
                throw GLM5NextLoadError.invalid("foreign GLM cache")
            }
        }
        for layer in layers {
            if let kda = layer as? GLM5NextKDACache { try kda.truncate(to: count) }
            if let dsa = layer as? GLM5NextDSACache { try dsa.truncate(to: count) }
        }
    }
    /// Frees the pages paged histories released (`GLM5NextDSAPagePool.recycle`).
    func recyclePages() {
        for case let history as GLM5NextDSACache in layers {
            if let pool = history.pagePool {
                pool.recycle()
                return
            }
        }
    }
    func adopt(_ other: GLM5NextTextCache) {
        precondition(layers.count == other.layers.count)
        for (destination, source) in zip(layers, other.layers) {
            if let destination = destination as? GLM5NextKDACache,
                let source = source as? GLM5NextKDACache
            {
                destination.adopt(source)
            } else if let destination = destination as? GLM5NextDSACache,
                let source = source as? GLM5NextDSACache
            {
                destination.adopt(source)
            } else {
                preconditionFailure("foreign cache in GLM transaction")
            }
        }
    }
}

/// A fitting DSA append in two launches instead of four slice updates (eight
/// dispatches in five dependent steps): each launch writes FRESH buffers of the
/// old capacity holding the logical prefix `0 ..< offset` copied from the old
/// planes and the new columns `offset ..< end`. No input is mutated, so copies,
/// snapshots and in-flight readers of the old buffers are untouched, exactly as
/// with `SliceUpdate`'s copy. Columns past `end` are left unwritten: the cache
/// never reads them (`logical`), and every later append or growth overwrites or
/// drops them. The index planes (keys, gates, validity) the selection reads
/// next are one launch; the latents, read only by the attention, are the other.
/// Bitwise: pure copies. Default GPU stream only; nil means the slice updates.
enum GLM5NextDSAAppendKernel {
    static func append(storage: [MLXArray], incoming: [MLXArray], offset: Int) -> [MLXArray]? {
        guard storage.count == 4, incoming.count == 4, glm5NextOnDefaultGPUStream() else {
            return nil
        }
        let (batch, capacity, count) = (storage[0].dim(0), storage[0].dim(1), incoming[0].dim(1))
        let end = offset + count
        guard count > 0, end <= capacity, storage[3].ndim == 2, storage[3].dtype == .bool,
            incoming[3].dtype == .bool, storage.prefix(3).allSatisfy({ $0.ndim == 3 }),
            storage[2].dim(2) == storage[1].dim(2),
            zip(storage, incoming).allSatisfy({
                $0.dtype == $1.dtype && $0.dim(0) == batch && $1.dim(0) == batch
            }),
            zip(storage.prefix(3), incoming.prefix(3)).allSatisfy({ $0.dim(2) == $1.dim(2) }),
            batch * capacity * max(storage[0].dim(2), 2 * storage[1].dim(2) + 1) <= Int(Int32.max)
        else { return nil }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // Rows of whole 32-bit words move as words when every row plane is
        // a device buffer (MLX passes inputs under 8 elements as constant).
        let words =
            storage[1].dtype.size == storage[2].dtype.size
            && zip(storage.prefix(3), incoming.prefix(3)).allSatisfy {
                ($0.dim(2) * $0.dtype.size) % 4 == 0 && $0.size >= 8 && $1.size >= 8
            }
        // Elements per row, or 32-bit words per row.
        func width(_ plane: MLXArray) -> Int {
            words ? plane.dim(2) * plane.dtype.size / 4 : plane.dim(2)
        }
        let (latentWidth, keyWidth) = (width(storage[0]), width(storage[1]))
        let params = GLM5NextKernelConstants.shared.int32s([Int32(offset), Int32(count)])
        // One thread per (unit, column, batch row) of the copied prefix and
        // the new columns; a typed load and store moves the bits as is.
        let index = (words ? indexWordKernel : indexKernel)(
            [
                storage[1], incoming[1], storage[2], incoming[2], storage[3], incoming[3],
                params,
            ],
            template: [("KD", keyWidth)],
            grid: (2 * keyWidth + 1, end, batch),
            threadGroup: (min(256, 2 * keyWidth + 1), 1, 1),
            outputShapes: [storage[1].shape, storage[2].shape, storage[3].shape],
            outputDTypes: [storage[1].dtype, storage[2].dtype, .bool])
        let latents = (words ? latentWordKernel : latentKernel)(
            [storage[0], incoming[0], params],
            template: [("LD", latentWidth)],
            grid: (latentWidth, end, batch), threadGroup: (min(256, latentWidth), 1, 1),
            outputShapes: [storage[0].shape], outputDTypes: [storage[0].dtype])
        return [latents[0], index[0], index[1], index[2]]
        #else
        return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let indexKernel = MLXFast.metalKernel(
        name: "glm5_next_dsa_append_index",
        inputNames: ["keys", "new_keys", "gates", "new_gates", "valid", "new_valid", "params"],
        outputNames: ["keys_out", "gates_out", "valid_out"],
        source: """
            const uint e = thread_position_in_grid.x;
            const int n = int(thread_position_in_grid.y);
            const int b = int(thread_position_in_grid.z);
            const int capacity = keys_shape[1];
            const int offset = params[0];
            const int count = params[1];
            const bool fresh = n >= offset;
            const size_t old_row = size_t(b) * capacity + n;
            const size_t new_row = size_t(b) * count + (n - offset);
            if (e < uint(KD)) {
                keys_out[old_row * KD + e] =
                    fresh ? new_keys[new_row * KD + e] : keys[old_row * KD + e];
            } else if (e < uint(2 * KD)) {
                const uint v = e - uint(KD);
                gates_out[old_row * KD + v] =
                    fresh ? new_gates[new_row * KD + v] : gates[old_row * KD + v];
            } else if (e == uint(2 * KD)) {
                valid_out[old_row] = fresh ? new_valid[new_row] : valid[old_row];
            }
            """)

    /// `indexKernel` moving 32-bit words (KD words per key/gate row).
    private static let indexWordKernel = MLXFast.metalKernel(
        name: "glm5_next_dsa_append_index_words",
        inputNames: ["keys", "new_keys", "gates", "new_gates", "valid", "new_valid", "params"],
        outputNames: ["keys_out", "gates_out", "valid_out"],
        source: """
            const uint e = thread_position_in_grid.x;
            const int n = int(thread_position_in_grid.y);
            const int b = int(thread_position_in_grid.z);
            const int capacity = keys_shape[1];
            const int offset = params[0];
            const int count = params[1];
            const bool fresh = n >= offset;
            const size_t old_row = size_t(b) * capacity + n;
            const size_t new_row = size_t(b) * count + (n - offset);
            if (e < uint(KD)) {
                const device uint* src = reinterpret_cast<const device uint*>(fresh ? new_keys : keys);
                reinterpret_cast<device uint*>(keys_out)[old_row * KD + e] =
                    src[(fresh ? new_row : old_row) * KD + e];
            } else if (e < uint(2 * KD)) {
                const uint v = e - uint(KD);
                const device uint* src = reinterpret_cast<const device uint*>(fresh ? new_gates : gates);
                reinterpret_cast<device uint*>(gates_out)[old_row * KD + v] =
                    src[(fresh ? new_row : old_row) * KD + v];
            } else if (e == uint(2 * KD)) {
                valid_out[old_row] = fresh ? new_valid[new_row] : valid[old_row];
            }
            """)

    /// `latentKernel` moving 32-bit words (LD words per latent row).
    private static let latentWordKernel = MLXFast.metalKernel(
        name: "glm5_next_dsa_append_latents_words",
        inputNames: ["latents", "new_latents", "params"],
        outputNames: ["latents_out"],
        source: """
            const uint e = thread_position_in_grid.x;
            const int n = int(thread_position_in_grid.y);
            const int b = int(thread_position_in_grid.z);
            const int capacity = latents_shape[1];
            const int offset = params[0];
            const int count = params[1];
            const bool fresh = n >= offset;
            const size_t old_row = size_t(b) * capacity + n;
            const size_t new_row = size_t(b) * count + (n - offset);
            const device uint* src = reinterpret_cast<const device uint*>(fresh ? new_latents : latents);
            reinterpret_cast<device uint*>(latents_out)[old_row * LD + e] =
                src[(fresh ? new_row : old_row) * LD + e];
            """)

    private static let latentKernel = MLXFast.metalKernel(
        name: "glm5_next_dsa_append_latents",
        inputNames: ["latents", "new_latents", "params"],
        outputNames: ["latents_out"],
        source: """
            const uint e = thread_position_in_grid.x;
            const int n = int(thread_position_in_grid.y);
            const int b = int(thread_position_in_grid.z);
            const int capacity = latents_shape[1];
            const int offset = params[0];
            const int count = params[1];
            const size_t old_row = size_t(b) * capacity + n;
            latents_out[old_row * LD + e] = n >= offset
                ? new_latents[(size_t(b) * count + (n - offset)) * LD + e]
                : latents[old_row * LD + e];
            """)
    #endif
}

/// The FP32 rows a DSA layer's CBv2 step writes: latents `[B,T,C]` as keys
/// `[B,1,T,C]`, and index keys and pool gates `[B,T,KD]` with validity `[B,T]`
/// as values `[B,1,T,2*KD+1]`, in one launch instead of the validity cast,
/// the concatenation and two FP32 casts. Bitwise: 16-bit and FP32 elements
/// widen to FP32 exactly and validity becomes 1 or 0, as through the index
/// keys' dtype. Default GPU stream only; nil means the composed graph.
enum GLM5NextDSAPayloadKernel {
    static func stage(latents: MLXArray, keys: MLXArray, gates: MLXArray, validity: MLXArray)
        -> (keys: MLXArray, values: MLXArray)?
    {
        let (batch, count) = (latents.dim(0), latents.dim(1))
        let floating: [DType] = [.float16, .bfloat16, .float32]
        guard glm5NextOnDefaultGPUStream(), latents.ndim == 3, keys.ndim == 3,
            batch > 0, count > 0, latents.dim(2) > 0, keys.dim(2) > 0,
            keys.shape == [batch, count, keys.dim(2)], gates.shape == keys.shape,
            validity.shape == [batch, count], validity.dtype == .bool,
            [latents, keys, gates].allSatisfy({ floating.contains($0.dtype) }),
            batch * count * max(latents.dim(2), 2 * keys.dim(2) + 1) <= Int(Int32.max)
        else { return nil }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let (latentWidth, keyWidth) = (latents.dim(2), keys.dim(2))
        let columns = max(latentWidth, 2 * keyWidth + 1)
        let outputs = kernel(
            [latents, keys, gates, validity],
            template: [("C", latentWidth), ("KD", keyWidth)],
            grid: (columns, count, batch), threadGroup: (min(256, columns), 1, 1),
            outputShapes: [
                [batch, 1, count, latentWidth], [batch, 1, count, 2 * keyWidth + 1],
            ],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
        #else
        return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    /// One thread per (column, token, batch row); inputs are read through
    /// their strides, so views of a fused projection are not copied first.
    private static let kernel = MLXFast.metalKernel(
        name: "glm5_next_dsa_payload",
        inputNames: ["latents", "keys", "gates", "valid"],
        outputNames: ["keys_out", "values_out"],
        source: """
            const int e = int(thread_position_in_grid.x);
            const int t = int(thread_position_in_grid.y);
            const int b = int(thread_position_in_grid.z);
            const size_t row = size_t(b) * latents_shape[1] + t;
            if (e < C) {
                keys_out[row * C + e] = float(latents[
                    int64_t(b) * latents_strides[0] + int64_t(t) * latents_strides[1]
                    + int64_t(e) * latents_strides[2]]);
            }
            const size_t out = row * (2 * KD + 1) + e;
            if (e < KD) {
                values_out[out] = float(keys[
                    int64_t(b) * keys_strides[0] + int64_t(t) * keys_strides[1]
                    + int64_t(e) * keys_strides[2]]);
            } else if (e < 2 * KD) {
                values_out[out] = float(gates[
                    int64_t(b) * gates_strides[0] + int64_t(t) * gates_strides[1]
                    + int64_t(e - KD) * gates_strides[2]]);
            } else if (e == 2 * KD) {
                values_out[out] =
                    valid[int64_t(b) * valid_strides[0] + int64_t(t) * valid_strides[1]]
                    ? 1.0f : 0.0f;
            }
            """,
        ensureRowContiguous: false)
    #endif
}
