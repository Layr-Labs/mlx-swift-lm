// Copyright © 2026 Eigen Labs.
import Cmlx
import Foundation
import MLX

/// Page bookkeeping of a `GLM5NextDSAPagePool`: page reference counts, the
/// free and retired lists, write tokens and reservation accounts. Pure host
/// state (no MLX arrays), so it is testable without a device.
///
/// A page holds `pageSize` consecutive token rows of one request's DSA
/// history (one layer). A page table (`GLM5NextDSAPageTable`) lists a
/// history's pages in token order; tables share pages by reference count
/// (prefix entries, copies, transactions), never by copying rows.
///
/// Writes are in place, so a page that several tables hold may be written
/// only where no other holder can read: a page's write token names the one
/// table allowed to write it, and its floor the slots below which another
/// holder may still read. A write below the floor, or by a table without the
/// token, first copies the page's leading rows into a fresh page (copy on
/// write at the boundary). A table that clones a shared table takes over its
/// tokens and raises each floor to the shared table's extent in that page:
/// the shared table's holders keep reading there.
///
/// Released pages are retired, not freed: a lazy graph built before the
/// release may still read them. `recycle()` frees them once every graph that
/// could read them was submitted (a candidate's commit fence, or between
/// requests); evaluation order on the stream then puts every later write
/// after those reads (each pool kernel registers the slabs it touches as
/// written, so MLX orders and fences them).
public final class GLM5NextDSAPageBook {
    /// A request's (or a lane's) page budget. Pages a history of the account
    /// allocates count against it until they are released; an account closed
    /// while some of its pages live on (a prefix entry holds them) hands them
    /// to the book as orphans.
    public final class Account {
        public let id: Int
        public fileprivate(set) var reservedPages: Int
        public fileprivate(set) var allocatedPages = 0
        public fileprivate(set) var isOpen = true
        fileprivate init(id: Int, reservedPages: Int) {
            self.id = id
            self.reservedPages = reservedPages
        }
    }

    public enum Failure: Error, Equatable, CustomStringConvertible {
        /// No free page although the reservations promised one: an account
        /// wrote past its reservation and the book has no unpromised page.
        case exhausted(needed: Int, free: Int)
        case invalid(String)

        public var description: String {
            switch self {
            case .exhausted(let needed, let free):
                return "paged DSA pool exhausted: \(needed) pages needed, \(free) free"
            case .invalid(let reason): return "paged DSA pool: \(reason)"
            }
        }
    }

    public let pageSize: Int
    public private(set) var capacityPages = 0
    private var refCounts: [Int32] = []
    /// Serial of the table holding the page's write token (0: none).
    private var writer: [UInt64] = []
    /// Leading slots of the page another holder may read.
    private var floor: [Int32] = []
    /// Account that allocated the page (-1: none).
    private var owner: [Int32] = []
    /// Free pages, largest id first (allocation takes the smallest, which
    /// keeps the used pages low and the pool shrinkable).
    private var free: [Int32] = []
    private var retired: [Int32] = []
    private var accounts: [Int: Account] = [:]
    private var nextAccount = 0
    private var nextSerial: UInt64 = 0
    /// Live pages whose account closed (prefix entries hold them).
    public private(set) var orphanPages = 0

    public init(pageSize: Int) {
        precondition(pageSize > 0)
        self.pageSize = pageSize
    }

    // MARK: - Capacity and accounts

    /// Pages held by a table, or retired and not yet recycled.
    public var allocatedPages: Int { capacityPages - free.count }
    public var retiredPages: Int { retired.count }
    public var freePages: Int { free.count }
    /// Pages the open accounts may still allocate under their reservations.
    public var outstandingPages: Int {
        accounts.values.reduce(0) { $0 + max(0, $1.reservedPages - $1.allocatedPages) }
    }
    public var reservedPages: Int { accounts.values.reduce(0) { $0 + $1.reservedPages } }
    /// Pages the pool must hold for what is allocated now and every open
    /// reservation, plus `adding` more.
    public func requiredPages(adding pages: Int = 0) -> Int {
        allocatedPages + outstandingPages + max(0, pages)
    }
    /// Highest allocated page plus one: the smallest capacity that keeps
    /// every live page.
    public var usedExtent: Int {
        let free = Set(self.free)
        var extent = capacityPages
        while extent > 0, free.contains(Int32(extent - 1)) { extent -= 1 }
        return extent
    }

    /// Adds pages `capacityPages ..< pages` (the slabs grew).
    public func grow(to pages: Int) {
        guard pages > capacityPages else { return }
        let added = pages - capacityPages
        refCounts += Array(repeating: 0, count: added)
        writer += Array(repeating: 0, count: added)
        floor += Array(repeating: 0, count: added)
        owner += Array(repeating: -1, count: added)
        // Largest first: the new pages sit before the existing free ones.
        free = (capacityPages ..< pages).reversed().map(Int32.init) + free
        capacityPages = pages
    }

    /// Drops pages `pages ..< capacityPages`, which must all be free.
    public func shrink(to pages: Int) throws {
        guard pages < capacityPages else { return }
        guard pages >= usedExtent, retired.isEmpty else {
            throw Failure.invalid("shrink below a live page")
        }
        free.removeAll { $0 >= Int32(pages) }
        refCounts.removeLast(capacityPages - pages)
        writer.removeLast(capacityPages - pages)
        floor.removeLast(capacityPages - pages)
        owner.removeLast(capacityPages - pages)
        capacityPages = pages
    }

    public func openAccount(reservedPages: Int) -> Account {
        nextAccount += 1
        let account = Account(id: nextAccount, reservedPages: max(0, reservedPages))
        accounts[account.id] = account
        return account
    }

    /// Adds `pages` to an open account's reservation (the pool already holds
    /// them: `GLM5NextDSAPagePool.extend`).
    public func extend(_ account: Account, by pages: Int) {
        guard account.isOpen, pages > 0 else { return }
        account.reservedPages += pages
    }

    /// Ends `account`'s reservation; its live pages become orphans.
    public func close(_ account: Account) {
        guard account.isOpen else { return }
        account.isOpen = false
        orphanPages += account.allocatedPages
        accounts[account.id] = nil
    }

    // MARK: - Tables

    func makeSerial() -> UInt64 {
        nextSerial += 1
        return nextSerial
    }

    /// Takes the smallest free page for `account` (or no account). Refuses a
    /// page another open account was promised.
    func allocate(for account: Account?, writer serial: UInt64) throws -> Int32 {
        let promised = outstandingPages
            - (account.map { $0.isOpen && $0.allocatedPages < $0.reservedPages ? 1 : 0 } ?? 0)
        guard free.count > promised, let page = free.popLast() else {
            throw Failure.exhausted(needed: promised + 1, free: free.count)
        }
        let index = Int(page)
        refCounts[index] = 1
        writer[index] = serial
        floor[index] = 0
        if let account, account.isOpen {
            owner[index] = Int32(account.id)
            account.allocatedPages += 1
        } else {
            owner[index] = -1
            orphanPages += 1
        }
        return page
    }

    func retain<S: Sequence>(_ pages: S) where S.Element == Int32 {
        for page in pages {
            precondition(refCounts[Int(page)] > 0, "retain of a free DSA page")
            refCounts[Int(page)] += 1
        }
    }

    /// Drops one reference to each page; a page nobody holds is retired.
    func release<S: Sequence>(_ pages: S) where S.Element == Int32 {
        for page in pages {
            let index = Int(page)
            precondition(refCounts[index] > 0, "release of a free DSA page")
            refCounts[index] -= 1
            guard refCounts[index] == 0 else { continue }
            writer[index] = 0
            floor[index] = 0
            retired.append(page)
        }
    }

    /// Frees every retired page and credits its account (a retired page
    /// still counts against the account that allocated it, so the free pages
    /// always cover the open reservations).
    public func recycle() {
        guard !retired.isEmpty else { return }
        for page in retired {
            let index = Int(page)
            if let account = accounts[Int(owner[index])] {
                account.allocatedPages -= 1
            } else {
                orphanPages -= 1
            }
            owner[index] = -1
        }
        free = (free + retired).sorted(by: >)
        retired.removeAll(keepingCapacity: true)
    }

    func refCount(_ page: Int32) -> Int { Int(refCounts[Int(page)]) }

    /// Whether `serial` may write `page` from slot `slot` on in place.
    func writable(_ page: Int32, by serial: UInt64, from slot: Int) -> Bool {
        let index = Int(page)
        return writer[index] == serial && Int(floor[index]) <= slot
    }

    /// Moves `page`'s token from `from` to `to`, raising its floor to
    /// `readable` slots (what `from`'s holders read there).
    func transfer(_ page: Int32, from: UInt64, to: UInt64, readable: Int) {
        let index = Int(page)
        guard writer[index] == from else { return }
        writer[index] = to
        floor[index] = max(floor[index], Int32(min(pageSize, max(0, readable))))
    }

    func holdsToken(_ page: Int32, _ serial: UInt64) -> Bool { writer[Int(page)] == serial }

    func dropToken(_ page: Int32, of serial: UInt64) {
        if writer[Int(page)] == serial { writer[Int(page)] = 0 }
    }
}

/// One history's pages in token order. Histories that share a table read the
/// same tokens; a history about to change a shared table clones it first
/// (`GLM5NextDSAPagePool.uniqueTable`), so every holder of one table has the
/// same token count.
final class GLM5NextDSAPageTable {
    /// Strong: a history's table may outlive the pool (a cache releases its
    /// pool before its table), and the book holds no table.
    let book: GLM5NextDSAPageBook
    let lock: NSRecursiveLock
    let serial: UInt64
    private(set) var pages: [Int32] = []
    private(set) var tokenCount = 0
    /// Pages whose write token this table may hold (the last few).
    private(set) var writable: [Int32] = []
    /// `pages` as a device int32 array, built once per change of `pages`
    /// (a decode step adds a page once every `pageSize` tokens).
    private var device: MLXArray?

    /// The page ids as a device array (the caller holds the lock).
    var deviceArray: MLXArray {
        if let device { return device }
        let made = MLXArray(pages)
        device = made
        return made
    }

    init(book: GLM5NextDSAPageBook, lock: NSRecursiveLock) {
        self.book = book
        self.lock = lock
        serial = book.makeSerial()
    }

    /// A table with this one's pages and tokens (the caller holds the lock).
    func clone() -> GLM5NextDSAPageTable {
        let copy = GLM5NextDSAPageTable(book: book, lock: lock)
        book.retain(pages)
        copy.pages = pages
        copy.device = device
        copy.tokenCount = tokenCount
        for page in writable {
            guard let logical = pages.lastIndex(of: page) else { continue }
            book.transfer(page, from: serial, to: copy.serial,
                readable: tokenCount - logical * book.pageSize)
        }
        copy.writable = writable
        writable = []
        return copy
    }

    /// Plans an append of `count` tokens (the caller holds the lock): the
    /// physical slot of every new token and, when the tail page cannot be
    /// written in place, the copy of its leading rows into a fresh page.
    func planAppend(count: Int, account: GLM5NextDSAPageBook.Account?) throws -> (
        slots: [Int32], clone: (source: Int32, destination: Int32, rows: Int)?
    ) {
        let size = book.pageSize
        let start = tokenCount, end = tokenCount + count
        var clone: (source: Int32, destination: Int32, rows: Int)?
        var allocated: [Int32] = []
        do {
            if start % size != 0 {
                let logical = start / size
                let page = pages[logical]
                if !book.writable(page, by: serial, from: start % size) {
                    let fresh = try book.allocate(for: account, writer: serial)
                    allocated.append(fresh)
                    clone = (page, fresh, start % size)
                }
            }
            let needed = (end + size - 1) / size - pages.count
            for _ in 0 ..< max(0, needed) {
                allocated.append(try book.allocate(for: account, writer: serial))
            }
        } catch {
            book.release(allocated)
            throw error
        }
        if let clone {
            pages[start / size] = clone.destination
            book.dropToken(clone.source, of: serial)
            book.release([clone.source])
        }
        pages += allocated.dropFirst(clone == nil ? 0 : 1)
        if !allocated.isEmpty { device = nil }
        var slots = [Int32]()
        slots.reserveCapacity(count)
        for position in start ..< end {
            slots.append(pages[position / size] * Int32(size) + Int32(position % size))
        }
        tokenCount = end
        // Tokens for the pages a rollback can return to: the last two.
        let keep = Set(pages.suffix(2))
        for page in writable where !keep.contains(page) { book.dropToken(page, of: serial) }
        writable = pages.suffix(2).filter { book.holdsToken($0, serial) }
        return (slots, clone)
    }

    /// Drops the tokens past `count` (the caller holds the lock).
    func truncate(to count: Int) {
        precondition(count >= 0 && count <= tokenCount)
        let keep = (count + book.pageSize - 1) / book.pageSize
        if keep < pages.count {
            let dropped = Array(pages[keep...])
            for page in dropped { book.dropToken(page, of: serial) }
            writable.removeAll { dropped.contains($0) }
            pages.removeLast(pages.count - keep)
            device = nil
            book.release(dropped)
        }
        tokenCount = count
    }

    deinit {
        lock.lock()
        defer { lock.unlock() }
        for page in writable { book.dropToken(page, of: serial) }
        book.release(pages)
    }
}

/// Paged storage for the DSA histories of one stage on one rank: every DSA
/// layer's latents, index keys, pool gates and validity (and the NextN
/// layer's, which has the same geometry) live in fixed-size pages of four
/// shared slabs, `[capacity * pageSize, width]` each, instead of one growing
/// contiguous buffer per request and layer. A history (`GLM5NextDSACache`
/// with a pool) holds a page table; an append writes its rows into pages in
/// place (one launch, `GLM5NextDSAPageKernels.write`), and a read gathers
/// the history's rows in token order into the contiguous arrays the
/// attention kernels read (one launch, `GLM5NextDSAPageKernels.gather`):
/// the same values in the same layout as the contiguous cache's, so every
/// attention result is bitwise the contiguous path's. Copies share pages
/// (prefix entries, transactions), rollback drops pages, nothing is copied
/// per token beyond the tokens written.
///
/// Engine-thread confined, as the caches it serves. The slabs grow only in
/// `reserve` (they are copied into larger ones), which an owner calls between
/// steps, when no unevaluated graph writes the pool; a step never grows them:
/// its pages come from the reservations.
public final class GLM5NextDSAPagePool {
    public let book: GLM5NextDSAPageBook
    let lock = NSRecursiveLock()
    public var pageSize: Int { book.pageSize }
    public let latentWidth: Int
    public let keyWidth: Int
    public let latentType: DType
    public let keyType: DType
    public let gateType: DType
    /// `[latents, keys, gates, validity]`, each `[capacity * pageSize, width]`
    /// (validity `[capacity * pageSize]`); empty at capacity 0.
    private(set) var slabs: [MLXArray] = []
    /// Smallest capacity the slabs grow to, in pages.
    public let minimumGrowthPages: Int

    public init(
        pageSize: Int = 64, latentWidth: Int, keyWidth: Int,
        latentType: DType, keyType: DType, gateType: DType, minimumGrowthPages: Int = 256
    ) throws {
        guard pageSize > 0, latentWidth > 0, keyWidth > 0, minimumGrowthPages > 0,
            [latentType, keyType, gateType].allSatisfy({ [.float16, .bfloat16, .float32].contains($0) })
        else { throw GLM5NextDSAPageBook.Failure.invalid("page geometry or element types") }
        book = GLM5NextDSAPageBook(pageSize: pageSize)
        self.latentWidth = latentWidth
        self.keyWidth = keyWidth
        self.latentType = latentType
        self.keyType = keyType
        self.gateType = gateType
        self.minimumGrowthPages = minimumGrowthPages
    }

    /// Bytes of one page across the four planes.
    public var pageBytes: Int {
        pageSize * (latentWidth * latentType.size + keyWidth * (keyType.size + gateType.size) + 1)
    }
    public var capacityPages: Int { lock.withLock { book.capacityPages } }
    public var capacityBytes: Int { capacityPages * pageBytes }
    public var allocatedPages: Int { lock.withLock { book.allocatedPages } }
    public var allocatedBytes: Int { allocatedPages * pageBytes }

    /// Pages a history of `tokens` tokens holds, plus the one page a copy on
    /// write may hold beside it until the step's commit recycles the old one.
    public func pages(tokens: Int) -> Int {
        tokens <= 0 ? 0 : (tokens + pageSize - 1) / pageSize + 1
    }

    /// Bytes `reserve(pages:)` would add to the slabs now: zero when the free
    /// capacity covers it. While they grow, the old slabs and the new ones
    /// coexist (the copy), so a caller's memory gate should also admit
    /// `capacityBytes`.
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

    /// Extends an open account's reservation by `pages` (a request that grows
    /// on demand), growing the slabs as `reserve` does when the pool cannot
    /// hold every reservation and live page. Call between steps only.
    public func extend(_ account: GLM5NextDSAPageBook.Account, by pages: Int) throws {
        try lock.withLock {
            guard account.isOpen, pages > 0 else { return }
            let required = book.requiredPages(adding: pages)
            if required > book.capacityPages {
                try growSlabs(to: grownCapacity(required: required))
            }
            book.extend(account, by: pages)
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

    /// Bytes of slab capacity no page holds now. An idle owner keeps them
    /// for the next request rather than trimming (`trim`): a request that
    /// fits the capacity then reserves without allocating or copying a slab.
    public var unallocatedBytes: Int {
        lock.withLock { max(0, book.capacityPages - book.allocatedPages) * pageBytes }
    }

    /// Compiles this geometry's four kernels (write, gather, index gather,
    /// selected gather) before a request needs them: a process compiles each
    /// from source at its first launch, which a first request otherwise pays
    /// inside its prefill. A scratch pool of the same geometry and element
    /// types writes one row and reads it back each way; nothing of this pool
    /// is touched.
    public func prewarm() throws {
        let scratch = try GLM5NextDSAPagePool(
            pageSize: pageSize, latentWidth: latentWidth, keyWidth: keyWidth, latentType: latentType,
            keyType: keyType, gateType: gateType, minimumGrowthPages: 1)
        let account = try scratch.reserve(pages: 2)
        var table: GLM5NextDSAPageTable?
        // A table never outlives its pool's book.
        defer {
            table = nil
            scratch.release(account)
        }
        scratch.uniqueTable(&table)
        func row(_ width: Int, _ type: DType) -> MLXArray { MLXArray.zeros([1, 1, width], dtype: type) }
        let written = try scratch.append(
            to: table!, account: account, fence: GLM5NextDSAPageKernels.initialFence,
            latents: row(latentWidth, latentType), keys: row(keyWidth, keyType),
            gates: row(keyWidth, gateType), validity: MLXArray([true]).reshaped(1, 1))
        let history = try scratch.gather(table!, count: 1, fence: written)
        let index = try scratch.gatherIndex(table!, count: 1, fence: history.fence)
        let selected = try scratch.gatherSelected(
            table!, count: 1, ids: MLXArray([Int32(0)]).reshaped(1, 1, 1), fence: index.fence)
        eval(history.history + index.index + [selected.latents, selected.ids, selected.fence])
    }

    /// Between requests: frees the retired pages and gives the slabs back
    /// when no page is held, or shrinks them to the live pages (rounded up);
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
        let shapes = [[rows, latentWidth], [rows, keyWidth], [rows, keyWidth], [rows]]
        let types = [latentType, keyType, gateType, DType.bool]
        let maximum = GPU.deviceInfo().maxBufferSize
        guard zip(shapes, types).allSatisfy({ $0.reduce(1, *) * $1.size <= maximum }) else {
            throw GLM5NextDSAPageBook.Failure.invalid("slab beyond the device buffer limit")
        }
        let grown: [MLXArray] = zip(shapes, types).enumerated().map { index, entry in
            let (shape, type) = entry
            guard slabs.indices.contains(index) else { return MLXArray.zeros(shape, dtype: type) }
            var extra = shape
            extra[0] = rows - slabs[index].dim(0)
            // The copy reads the old slab after every write already submitted to it.
            return concatenated([slabs[index], MLXArray.zeros(extra, dtype: type)], axis: 0)
        }
        eval(grown)
        slabs = grown
        book.grow(to: pages)
    }

    /// Leaves `table` if no other history holds it, else replaces it with its
    /// clone (the holders keep the original; this history writes the clone).
    /// An empty `table` becomes a new one.
    func uniqueTable(_ table: inout GLM5NextDSAPageTable?) {
        if table == nil {
            table = GLM5NextDSAPageTable(book: book, lock: lock)
            return
        }
        // No other reference may be held across this check.
        guard !isKnownUniquelyReferenced(&table) else { return }
        lock.lock()
        let clone = table!.clone()
        lock.unlock()
        table = clone
    }

    /// Appends `[1, T, ...]` rows to `table` (unique) after `fence`; returns
    /// the write's fence.
    func append(
        to table: GLM5NextDSAPageTable, account: GLM5NextDSAPageBook.Account?, fence: MLXArray,
        latents: MLXArray, keys: MLXArray, gates: MLXArray, validity: MLXArray
    ) throws -> MLXArray {
        let count = latents.dim(1)
        guard latents.shape == [1, count, latentWidth], keys.shape == [1, count, keyWidth],
            gates.shape == keys.shape, validity.shape == [1, count], validity.dtype == .bool,
            latents.dtype == latentType, keys.dtype == keyType, gates.dtype == gateType
        else {
            throw GLM5NextDSAPageBook.Failure.invalid(
                "append of \(latents.shape) \(latents.dtype), \(keys.shape) \(keys.dtype), "
                    + "\(gates.shape) \(gates.dtype), \(validity.shape) \(validity.dtype)")
        }
        let plan = try lock.withLock { try table.planAppend(count: count, account: account) }
        return try GLM5NextDSAPageKernels.write(
            slabs: slabs, pageSize: pageSize, slots: plan.slots, clone: plan.clone, fence: fence,
            latents: latents, keys: keys, gates: gates, validity: validity)
    }

    /// The first `count` tokens of `table` as contiguous `[1, count, ...]`
    /// arrays after `fence`, and the read's fence.
    func gather(_ table: GLM5NextDSAPageTable, count: Int, fence: MLXArray) throws -> (
        history: [MLXArray], fence: MLXArray
    ) {
        let pages = try lock.withLock { () throws -> MLXArray in
            guard table.pages.count * pageSize >= count else {
                throw GLM5NextDSAPageBook.Failure.invalid("gather past the table")
            }
            return table.deviceArray
        }
        return try GLM5NextDSAPageKernels.gather(
            slabs: slabs, pageSize: pageSize, pages: pages, count: count, fence: fence)
    }

    /// The index planes (keys, gates, validity) of the first `count` tokens.
    func gatherIndex(_ table: GLM5NextDSAPageTable, count: Int, fence: MLXArray) throws -> (
        index: [MLXArray], fence: MLXArray
    ) {
        let pages = try lock.withLock { () throws -> MLXArray in
            guard table.pages.count * pageSize >= count else {
                throw GLM5NextDSAPageBook.Failure.invalid("gather past the table")
            }
            return table.deviceArray
        }
        return try GLM5NextDSAPageKernels.gatherIndex(
            slabs: slabs, pageSize: pageSize, pages: pages, count: count, fence: fence)
    }

    /// The latents of `ids` `[1, T, W]` (positions in the first `count`
    /// tokens; -1 or past them reads nothing) as `[1, T * W, width]` rows in
    /// selection order, and the ids rereading them (`row` or -1).
    func gatherSelected(_ table: GLM5NextDSAPageTable, count: Int, ids: MLXArray, fence: MLXArray)
        throws -> (latents: MLXArray, ids: MLXArray, fence: MLXArray)
    {
        let pages = try lock.withLock { () throws -> MLXArray in
            guard table.pages.count * pageSize >= count else {
                throw GLM5NextDSAPageBook.Failure.invalid("gather past the table")
            }
            return table.deviceArray
        }
        return try GLM5NextDSAPageKernels.gatherSelected(
            slabs: slabs, pageSize: pageSize, pages: pages, count: count, ids: ids, fence: fence)
    }

    func truncate(_ table: GLM5NextDSAPageTable, to count: Int) {
        lock.withLock { table.truncate(to: count) }
    }
}

/// The pool's two launches. Both take the four slabs as mutable inputs: MLX
/// then treats every launch that touches a slab as writing it, so later
/// launches (and command buffers) wait for earlier ones in submission order.
/// Data order within one history is the fence each launch takes and returns.
/// Bitwise: pure copies (32-bit words where every row is whole words).
enum GLM5NextDSAPageKernels {
    /// A fresh history's fence: a shared read-only `[1]` int32 leaf.
    static var initialFence: MLXArray { GLM5NextKernelConstants.shared.int32s([0]) }

    static func write(
        slabs: [MLXArray], pageSize: Int, slots: [Int32],
        clone: (source: Int32, destination: Int32, rows: Int)?, fence: MLXArray,
        latents: MLXArray, keys: MLXArray, gates: MLXArray, validity: MLXArray
    ) throws -> MLXArray {
        guard slabs.count == 4, !slots.isEmpty else {
            throw GLM5NextDSAPageBook.Failure.invalid("write without slabs or tokens")
        }
        let (words, widths) = units(slabs: slabs, rows: [latents, keys, gates])
        let rows = slots.count + (clone?.rows ?? 0)
        let columns = widths.reduce(0, +) + 1
        let constants = GLM5NextKernelConstants.shared
        let cloneArgs = clone.map { MLXArray([$0.source, $0.destination, Int32($0.rows)]) }
            ?? constants.int32s([0, 0, 0])
        let slotArray = slots.count == 1 ? constants.int32s(slots) : MLXArray(slots)
        return (words ? writeWords : writeElements)(
            [
                latents, keys, gates, validity, slotArray, cloneArgs, fence,
                slabs[0], slabs[1], slabs[2], slabs[3],
            ],
            template: [("PS", pageSize), ("LW", widths[0]), ("KW", widths[1]), ("GW", widths[2])],
            grid: (columns, rows, 1), threadGroup: (min(256, columns), 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    static func gather(
        slabs: [MLXArray], pageSize: Int, pages: MLXArray, count: Int, fence: MLXArray
    ) throws -> (history: [MLXArray], fence: MLXArray) {
        guard slabs.count == 4, count > 0, pages.size * pageSize >= count else {
            throw GLM5NextDSAPageBook.Failure.invalid("gather without slabs or pages")
        }
        let (words, widths) = units(slabs: slabs, rows: [])
        let columns = widths.reduce(0, +) + 1
        let outputs = (words ? gatherWords : gatherElements)(
            [slabs[0], slabs[1], slabs[2], slabs[3], pages, fence],
            template: [("PS", pageSize), ("LW", widths[0]), ("KW", widths[1]), ("GW", widths[2])],
            grid: (columns, count, 1), threadGroup: (min(256, columns), 1, 1),
            outputShapes: [
                [1, count, slabs[0].dim(1)], [1, count, slabs[1].dim(1)],
                [1, count, slabs[2].dim(1)], [1, count], [1],
            ],
            outputDTypes: [slabs[0].dtype, slabs[1].dtype, slabs[2].dtype, .bool, .int32])
        return (Array(outputs.prefix(4)), outputs[4])
    }

    static func gatherIndex(
        slabs: [MLXArray], pageSize: Int, pages: MLXArray, count: Int, fence: MLXArray
    ) throws -> (index: [MLXArray], fence: MLXArray) {
        guard slabs.count == 4, count > 0, pages.size * pageSize >= count else {
            throw GLM5NextDSAPageBook.Failure.invalid("index gather without slabs or pages")
        }
        let (words, widths) = units(slabs: slabs, rows: [])
        let columns = widths[1] + widths[2] + 1
        let outputs = (words ? gatherIndexWords : gatherIndexElements)(
            [slabs[1], slabs[2], slabs[3], pages, fence, slabs[0]],
            template: [("PS", pageSize), ("KW", widths[1]), ("GW", widths[2])],
            grid: (columns, count, 1), threadGroup: (min(256, columns), 1, 1),
            outputShapes: [[1, count, slabs[1].dim(1)], [1, count, slabs[2].dim(1)], [1, count], [1]],
            outputDTypes: [slabs[1].dtype, slabs[2].dtype, .bool, .int32])
        return (Array(outputs.prefix(3)), outputs[3])
    }

    static func gatherSelected(
        slabs: [MLXArray], pageSize: Int, pages: MLXArray, count: Int, ids: MLXArray, fence: MLXArray
    ) throws -> (latents: MLXArray, ids: MLXArray, fence: MLXArray) {
        guard slabs.count == 4, count > 0, pages.size * pageSize >= count, ids.ndim == 3,
            ids.dim(0) == 1, ids.dtype == .int32, ids.size > 0
        else {
            throw GLM5NextDSAPageBook.Failure.invalid("selected gather without slabs, pages or ids")
        }
        let (words, widths) = units(slabs: slabs, rows: [])
        let rows = ids.dim(1) * ids.dim(2)
        let outputs = (words ? gatherSelectedWords : gatherSelectedElements)(
            [slabs[0], pages, ids, GLM5NextKernelConstants.shared.int32(Int32(count)), fence,
             slabs[1], slabs[2], slabs[3]],
            template: [("PS", pageSize), ("LW", widths[0])],
            grid: (widths[0] + 1, rows, 1), threadGroup: (min(256, widths[0] + 1), 1, 1),
            outputShapes: [[1, rows, slabs[0].dim(1)], ids.shape, [1]],
            outputDTypes: [slabs[0].dtype, .int32, .int32])
        return (outputs[0], outputs[1], outputs[2])
    }

    /// Whether the three row planes move as 32-bit words (every row whole
    /// words, in the slabs and in `rows`, each at least eight elements so MLX
    /// passes it as a device buffer), and each plane's row width in the unit.
    private static func units(slabs: [MLXArray], rows: [MLXArray]) -> (Bool, [Int]) {
        let planes = Array(slabs.prefix(3))
        let words =
            planes.allSatisfy { ($0.dim(1) * $0.dtype.size) % 4 == 0 }
            && rows.allSatisfy { $0.size >= 8 }
        return (words, planes.map { words ? $0.dim(1) * $0.dtype.size / 4 : $0.dim(1) })
    }

    private static let writeInputs = [
        "lat_in", "key_in", "gate_in", "valid_in", "slots", "clone", "fence",
        "lat", "key", "gate", "valid",
    ]
    private static let gatherInputs = ["lat", "key", "gate", "valid", "pages", "fence"]
    private static let gatherOutputs = ["lat_out", "key_out", "gate_out", "valid_out", "fence_out"]

    /// Row r < clone rows copies source page row r to the destination page;
    /// after them, new token t = r - clone rows goes to `slots[t]`.
    private static func writeSource(_ cast: (String, Bool) -> String) -> String {
        """
        const uint e = thread_position_in_grid.x;
        const int r = int(thread_position_in_grid.y);
        const int cloned = clone[2];
        const bool fresh = r >= cloned;
        const size_t src = fresh ? size_t(r - cloned) : size_t(clone[0]) * PS + r;
        const size_t dst = fresh ? size_t(slots[r - cloned]) : size_t(clone[1]) * PS + r;
        if (e < uint(LW)) {
            \(cast("lat", true))[dst * LW + e] =
                fresh ? \(cast("lat_in", false))[src * LW + e] : \(cast("lat", true))[src * LW + e];
        } else if (e < uint(LW + KW)) {
            const uint v = e - uint(LW);
            \(cast("key", true))[dst * KW + v] =
                fresh ? \(cast("key_in", false))[src * KW + v] : \(cast("key", true))[src * KW + v];
        } else if (e < uint(LW + KW + GW)) {
            const uint v = e - uint(LW + KW);
            \(cast("gate", true))[dst * GW + v] =
                fresh ? \(cast("gate_in", false))[src * GW + v] : \(cast("gate", true))[src * GW + v];
        } else {
            valid[dst] = fresh ? valid_in[src] : valid[src];
        }
        if (e == 0 && r == 0) { fence_out[0] = fence[0] ^ 1; }
        """
    }

    private static func gatherSource(_ cast: (String, Bool) -> String) -> String {
        """
        const uint e = thread_position_in_grid.x;
        const int n = int(thread_position_in_grid.y);
        const size_t row = size_t(pages[n / PS]) * PS + size_t(n % PS);
        if (e < uint(LW)) {
            \(cast("lat_out", true))[size_t(n) * LW + e] = \(cast("lat", true))[row * LW + e];
        } else if (e < uint(LW + KW)) {
            const uint v = e - uint(LW);
            \(cast("key_out", true))[size_t(n) * KW + v] = \(cast("key", true))[row * KW + v];
        } else if (e < uint(LW + KW + GW)) {
            const uint v = e - uint(LW + KW);
            \(cast("gate_out", true))[size_t(n) * GW + v] = \(cast("gate", true))[row * GW + v];
        } else {
            valid_out[n] = valid[row];
        }
        if (e == 0 && n == 0) { fence_out[0] = fence[0] ^ 1; }
        """
    }

    private static func gatherIndexSource(_ cast: (String, Bool) -> String) -> String {
        """
        const uint e = thread_position_in_grid.x;
        const int n = int(thread_position_in_grid.y);
        const size_t row = size_t(pages[n / PS]) * PS + size_t(n % PS);
        if (e < uint(KW)) {
            \(cast("key_out", true))[size_t(n) * KW + e] = \(cast("key", true))[row * KW + e];
        } else if (e < uint(KW + GW)) {
            const uint v = e - uint(KW);
            \(cast("gate_out", true))[size_t(n) * GW + v] = \(cast("gate", true))[row * GW + v];
        } else {
            valid_out[n] = valid[row];
        }
        if (e == 0 && n == 0) { fence_out[0] = fence[0] ^ 1; }
        """
    }

    /// Row j of the output is the latent of `ids[j]` (zeros when it reads
    /// nothing); `ids_out[j]` is j, or -1 where `ids[j]` is.
    private static func gatherSelectedSource(_ cast: (String, Bool) -> String) -> String {
        """
        const uint e = thread_position_in_grid.x;
        const int j = int(thread_position_in_grid.y);
        const int id = ids[j];
        const bool read = id >= 0 && id < count;
        if (e < uint(LW)) {
            if (read) {
                const size_t row = size_t(pages[id / PS]) * PS + size_t(id % PS);
                \(cast("lat_out", true))[size_t(j) * LW + e] = \(cast("lat", true))[row * LW + e];
            } else {
                \(cast("lat_out", true))[size_t(j) * LW + e] = 0;
            }
        } else {
            ids_out[j] = read ? j : -1;
        }
        if (e == 0 && j == 0) { fence_out[0] = fence[0] ^ 1; }
        """
    }

    private static let gatherIndexInputs = ["key", "gate", "valid", "pages", "fence", "lat"]
    private static let gatherIndexOutputs = ["key_out", "gate_out", "valid_out", "fence_out"]
    private static let gatherSelectedInputs = ["lat", "pages", "ids", "count", "fence", "key", "gate", "valid"]
    private static let gatherSelectedOutputs = ["lat_out", "ids_out", "fence_out"]
    private static let slabNames = ["lat", "key", "gate", "valid"]

    private static let gatherIndexWords = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_gather_index_words", inputNames: gatherIndexInputs,
        outputNames: gatherIndexOutputs, source: gatherIndexSource(words), mutableInputs: slabNames)
    private static let gatherIndexElements = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_gather_index", inputNames: gatherIndexInputs,
        outputNames: gatherIndexOutputs, source: gatherIndexSource(elements), mutableInputs: slabNames)
    private static let gatherSelectedWords = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_gather_selected_words", inputNames: gatherSelectedInputs,
        outputNames: gatherSelectedOutputs, source: gatherSelectedSource(words), mutableInputs: slabNames)
    private static let gatherSelectedElements = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_gather_selected", inputNames: gatherSelectedInputs,
        outputNames: gatherSelectedOutputs, source: gatherSelectedSource(elements),
        mutableInputs: slabNames)

    private static func words(_ name: String, _ mutable: Bool) -> String {
        mutable
            ? "reinterpret_cast<device uint*>(\(name))"
            : "reinterpret_cast<const device uint*>(\(name))"
    }

    private static func elements(_ name: String, _ mutable: Bool) -> String { name }

    private static let writeWords = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_write_words", inputNames: writeInputs,
        outputNames: ["fence_out"], source: writeSource(words),
        mutableInputs: ["lat", "key", "gate", "valid"])
    private static let writeElements = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_write", inputNames: writeInputs,
        outputNames: ["fence_out"], source: writeSource(elements),
        mutableInputs: ["lat", "key", "gate", "valid"])
    private static let gatherWords = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_gather_words", inputNames: gatherInputs,
        outputNames: gatherOutputs, source: gatherSource(words),
        mutableInputs: ["lat", "key", "gate", "valid"])
    private static let gatherElements = MLXFast.metalKernel(
        name: "glm5_next_dsa_page_gather", inputNames: gatherInputs,
        outputNames: gatherOutputs, source: gatherSource(elements),
        mutableInputs: ["lat", "key", "gate", "valid"])
}

/// A committed DSA history's rows for a reader on another thread (a prompt
/// prefix's SSD writer), copied on the host: no launch, no array is made and
/// the history does not change (`GLM5NextDSACache.historyRows`).
///
/// A paged history's pages are held as a copy holds them: shared, and never
/// written while held (a holder that appends takes its own table first, and
/// writes past the rows held here). Each read takes the pool's slabs under its
/// lock: a growth or trim replaces them, and a held page's rows are the same
/// in every generation of slabs. A contiguous history's arrays are held as
/// they are; no path writes a stored array in place.
public final class GLM5NextDSAHistoryRows: @unchecked Sendable {
    private enum Source {
        /// The table is held, unread, so its pages stay (and stay unwritten).
        case pages(pool: GLM5NextDSAPagePool, table: GLM5NextDSAPageTable, pages: [Int32])
        case arrays([MLXArray])
    }

    /// Rows `0 ..< count` of the history.
    public let count: Int
    /// Element types of the planes: latents, index keys, pool gates, validity.
    public let types: [DType]
    /// Elements in one row of each plane (validity: 1).
    public let widths: [Int]
    private let source: Source

    /// The first `count` rows of `table` (the caller evaluated its last write).
    init(pool: GLM5NextDSAPagePool, table: GLM5NextDSAPageTable, count: Int) {
        self.count = count
        types = [pool.latentType, pool.keyType, pool.gateType, .bool]
        widths = [pool.latentWidth, pool.keyWidth, pool.keyWidth, 1]
        // One history's table changes only after it is cloned away from every
        // other holder, and this reader holds it: its pages are fixed now.
        source = .pages(pool: pool, table: table, pages: pool.lock.withLock { table.pages })
    }

    /// Evaluated `[latents, keys, gates, validity]`, `[1, count, width]` (validity `[1, count]`).
    init(arrays: [MLXArray], count: Int) {
        self.count = count
        types = arrays.map(\.dtype)
        widths = arrays.map { $0.ndim == 3 ? $0.dim(2) : 1 }
        source = .arrays(arrays)
    }

    /// Bytes of one row of `plane` (0 latents, 1 index keys, 2 pool gates, 3 validity).
    public func rowBytes(_ plane: Int) -> Int { widths[plane] * types[plane].size }

    /// Bytes this reader keeps alive: its whole pages, or its arrays.
    public var heldBytes: Int {
        switch source {
        case .pages(let pool, _, let pages): return pages.count * pool.pageBytes
        case .arrays(let arrays): return arrays.reduce(0) { $0 + $1.nbytes }
        }
    }

    /// Copies rows `rows` of `plane` to `destination`: the range's row `r` to
    /// `destination + r * stride` (`stride` at least the plane's row bytes).
    public func copy(plane: Int, rows: Range<Int>, to destination: UnsafeMutableRawPointer, stride: Int) throws {
        guard (0 ..< 4).contains(plane), rows.lowerBound >= 0, rows.upperBound <= count,
            stride >= rowBytes(plane)
        else { throw GLM5NextDSAPageBook.Failure.invalid("history rows outside the held history") }
        let bytes = rowBytes(plane)
        switch source {
        case .pages(let pool, _, let pages):
            let slab = pool.lock.withLock { pool.slabs.indices.contains(plane) ? pool.slabs[plane] : nil }
            guard let slab, let base = mlx_array_data_uint8(slab.ctx), slab.nbytes == slab.dim(0) * bytes else {
                throw GLM5NextDSAPageBook.Failure.invalid("history rows without their slab")
            }
            let size = pool.pageSize
            try withExtendedLifetime(slab) {
                var row = rows.lowerBound
                while row < rows.upperBound {
                    let slot = row % size, run = min(size - slot, rows.upperBound - row)
                    let first = Int(pages[row / size]) * size + slot
                    guard first + run <= slab.dim(0) else {
                        throw GLM5NextDSAPageBook.Failure.invalid("held page outside its slab")
                    }
                    Self.copy(from: UnsafeRawPointer(base).advanced(by: first * bytes), rows: run, bytes: bytes,
                        to: destination.advanced(by: (row - rows.lowerBound) * stride), stride: stride)
                    row += run
                }
            }
        case .arrays(let arrays):
            let array = arrays[plane]
            guard let base = mlx_array_data_uint8(array.ctx), let strides = mlx_array_strides(array.ctx),
                array.ndim >= 2, array.dim(1) >= rows.upperBound,
                array.ndim == 2 || array.dim(2) <= 1 || strides[2] == 1
            else { throw GLM5NextDSAPageBook.Failure.invalid("history rows of an unread array") }
            let rowStride = Int(strides[1]) * types[plane].size
            withExtendedLifetime(array) {
                let first = UnsafeRawPointer(base).advanced(by: rows.lowerBound * rowStride)
                if rowStride == bytes {
                    Self.copy(from: first, rows: rows.count, bytes: bytes, to: destination, stride: stride)
                } else {
                    for row in 0 ..< rows.count {
                        destination.advanced(by: row * stride).copyMemory(
                            from: first.advanced(by: row * rowStride), byteCount: bytes)
                    }
                }
            }
        }
    }

    /// `rows` consecutive rows of `bytes` at `source` to rows `stride` apart.
    private static func copy(
        from source: UnsafeRawPointer, rows: Int, bytes: Int, to destination: UnsafeMutableRawPointer,
        stride: Int
    ) {
        if stride == bytes {
            destination.copyMemory(from: source, byteCount: rows * bytes)
            return
        }
        for row in 0 ..< rows {
            destination.advanced(by: row * stride).copyMemory(
                from: source.advanced(by: row * bytes), byteCount: bytes)
        }
    }
}
