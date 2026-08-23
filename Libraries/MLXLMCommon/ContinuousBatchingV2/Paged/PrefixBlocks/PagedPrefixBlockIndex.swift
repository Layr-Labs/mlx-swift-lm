// PagedPrefixBlockIndex.swift
//
// vLLM-style automatic prefix cache over physical MLX KV pages. One entry is
// one fixed token block, keyed by the existing parent-chained SHA-256 hash.
// The index owns no page reference: zero-ref pages remain allocator-visible
// and are invalidated immediately before reuse.

import Foundation

final class PagedPrefixBlockIndex: PagedKVPageReuseObserver, @unchecked Sendable {
    private final class Entry {
        let id: UInt64
        let hash: Data
        let layerPages: [[PagedKVPageHandle]?]
        let bytes: Int

        init(
            id: UInt64, hash: Data, layerPages: [[PagedKVPageHandle]?], bytes: Int
        ) {
            self.id = id
            self.hash = hash
            self.layerPages = layerPages
            self.bytes = bytes
        }

        var allPages: [PagedKVPageHandle] { layerPages.compactMap { $0 }.flatMap { $0 } }
    }

    let config: CBv2PagedPrefixCacheConfig
    let hasher: CBv2BlockHasher
    private unowned let pool: PagedKVPool
    private let layerKinds: [CBv2LayerKind]
    private let pagesPerBlock: Int

    private let lock = NSLock()
    private var entries: [UInt64: Entry] = [:]
    /// A content hash may have multiple physical realizations. Keeping all of
    /// them matches vLLM and lets reuse of one page leave another hit alive.
    private var entryIDsByHash: [Data: [UInt64]] = [:]
    /// Complete reverse aliases. Reusing one page removes every block bundle
    /// that contains it before the generation changes.
    private var entryIDsByPage: [PagedKVPageHandle: Set<UInt64>] = [:]
    private var nextEntryID: UInt64 = 0
    private var bytesIndexed = 0
    private var hits = 0
    private var misses = 0
    private var tokensSaved = 0
    private var invalidations = 0

    init(
        pool: PagedKVPool, layerKinds: [CBv2LayerKind],
        config: CBv2PagedPrefixCacheConfig
    ) throws {
        guard config.blockSize > 0,
            UInt64(config.blockSize) <= UInt64(UInt32.max),
            config.blockSize % pool.config.pageSize == 0
        else {
            throw CBv2KVError.backendIneligible(
                reason: "paged prefix block size \(config.blockSize) must fit UInt32 and be a "
                    + "positive multiple of page size \(pool.config.pageSize)")
        }
        guard pool.config.prefixSharingBlockSize == config.blockSize else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool.prefixSharingBlockSize must equal resident cache block "
                    + "size \(config.blockSize)")
        }
        guard layerKinds.contains(where: Self.isCacheable) else {
            throw CBv2KVError.backendIneligible(
                reason: "paged prefix sharing requires a storage-owning full-attention layer")
        }
        self.pool = pool
        self.layerKinds = layerKinds
        self.config = config
        self.hasher = CBv2BlockHasher(
            blockSize: config.blockSize,
            promptContractID: config.promptContractID,
            scopeID: config.scopeID)
        self.pagesPerBlock = config.blockSize / pool.config.pageSize
        pool.installPageReuseObserver(self)
    }

    func prepareProbe(tokens: [Int], cacheSalt: String?) -> CBv2PagedPrefixProbe? {
        let requestHasher = hasher(cacheSalt: cacheSalt)
        let hashes = requestHasher.chainHashes(tokens: tokens)
        return CBv2PagedPrefixProbe(
            hasher: requestHasher,
            chainHashes: hashes,
            maxLookupBlocks: requestHasher.maxLookupBlocks(tokenCount: tokens.count))
    }

    /// Left-to-right stop-at-first-miss traversal. Unlike the snapshot cache,
    /// each physical entry represents exactly one block, so the index is
    /// downward-closed for every usable chain.
    func longestResidentPrefix(for probe: CBv2PagedPrefixProbe) -> CBv2PagedPrefixMatch? {
        residentPrefix(for: probe, recordingStats: true)
    }

    /// Thread-safe advisory lookup for provider L1→L2 tier selection. It does
    /// not pin pages or count as a real request lookup.
    func peekResidentPrefix(for probe: CBv2PagedPrefixProbe) -> CBv2PagedPrefixMatch? {
        residentPrefix(for: probe, recordingStats: false)
    }

    private func residentPrefix(
        for probe: CBv2PagedPrefixProbe, recordingStats: Bool
    ) -> CBv2PagedPrefixMatch? {
        guard probe.blockSize == hasher.blockSize, probe.maxLookupBlocks > 0 else { return nil }
        lock.lock()
        defer { lock.unlock() }
        let limit = min(probe.maxLookupBlocks, probe.chainHashes.count)
        var matchedHashes: [Data] = []
        matchedHashes.reserveCapacity(limit)
        for hash in probe.chainHashes.prefix(limit) {
            guard let ids = entryIDsByHash[hash], !ids.isEmpty else { break }
            matchedHashes.append(hash)
        }
        guard !matchedHashes.isEmpty else {
            if recordingStats { misses += 1 }
            return nil
        }
        let matched = matchedHashes.count * hasher.blockSize
        if recordingStats {
            hits += 1
            tokensSaved += matched
        }
        return CBv2PagedPrefixMatch(
            chainHashes: matchedHashes, matchedTokens: matched, blockSize: hasher.blockSize)
    }

    /// Resolve one generation-valid physical realization per hash. Called on
    /// the engine queue immediately before the pool's transactional retain.
    func resolve(_ match: CBv2PagedPrefixMatch) -> CBv2PagedSharedPrefix? {
        guard match.blockSize == hasher.blockSize,
            match.matchedTokens == match.chainHashes.count * hasher.blockSize
        else { return nil }
        lock.lock()
        defer { lock.unlock() }

        var selected: [Entry] = []
        selected.reserveCapacity(match.chainHashes.count)
        for hash in match.chainHashes {
            var valid: Entry?
            for id in entryIDsByHash[hash] ?? [] {
                guard let entry = entries[id] else { continue }
                if entry.allPages.allSatisfy({ pool.isValid($0) }) {
                    valid = entry
                    break
                }
                removeEntryLocked(id)
            }
            guard let valid else { return nil }
            selected.append(valid)
        }

        var layerPages = [[PagedKVPageHandle]?](repeating: nil, count: layerKinds.count)
        for layerIndex in layerKinds.indices where Self.isCacheable(layerKinds[layerIndex]) {
            var pages: [PagedKVPageHandle] = []
            pages.reserveCapacity(selected.count * pagesPerBlock)
            for entry in selected {
                guard let blockPages = entry.layerPages[layerIndex],
                    blockPages.count == pagesPerBlock
                else { return nil }
                pages.append(contentsOf: blockPages)
            }
            layerPages[layerIndex] = pages
        }
        return CBv2PagedSharedPrefix(
            matchedTokens: match.matchedTokens, layerPages: layerPages)
    }

    @discardableResult
    func publish(
        state: [CBv2SequenceKV?], chainHashes: [Data], blockIndices: Range<Int>
    ) -> Int {
        guard state.count == layerKinds.count,
            blockIndices.lowerBound >= 0,
            blockIndices.upperBound <= chainHashes.count
        else { return 0 }

        struct Candidate {
            let hash: Data
            let layerPages: [[PagedKVPageHandle]?]
            let bytes: Int
        }
        var candidates: [Candidate] = []
        candidates.reserveCapacity(blockIndices.count)
        for blockIndex in blockIndices {
            let tokenRange =
                (blockIndex * hasher.blockSize) ..< ((blockIndex + 1) * hasher.blockSize)
            var layerPages = [[PagedKVPageHandle]?](repeating: nil, count: layerKinds.count)
            var bytes = 0
            var sawCacheableLayer = false
            for layerIndex in layerKinds.indices where Self.isCacheable(layerKinds[layerIndex]) {
                guard let row = state[layerIndex] as? PagedSequenceKV,
                    row.pool === pool,
                    let handles = row.prefixPageHandles(tokens: tokenRange),
                    handles.count == pagesPerBlock
                else { return 0 }
                layerPages[layerIndex] = handles
                sawCacheableLayer = true
                let pageBytes = pool.group(PagedKVGroupKey(layerKinds[layerIndex])).pageBytes
                let (layerBytes, layerOverflow) = pageBytes.multipliedReportingOverflow(
                    by: handles.count)
                let (sum, sumOverflow) = bytes.addingReportingOverflow(layerBytes)
                guard !layerOverflow, !sumOverflow else { return 0 }
                bytes = sum
            }
            guard sawCacheableLayer else { return 0 }
            candidates.append(
                Candidate(
                    hash: chainHashes[blockIndex],
                    layerPages: layerPages,
                    bytes: bytes))
        }

        lock.lock()
        defer { lock.unlock() }
        var published = 0
        for candidate in candidates {
            let duplicate = (entryIDsByHash[candidate.hash] ?? []).contains { id in
                entries[id]?.layerPages == candidate.layerPages
            }
            if duplicate {
                published += 1
                continue
            }
            nextEntryID &+= 1
            precondition(nextEntryID != 0, "paged prefix entry id overflow")
            let entry = Entry(
                id: nextEntryID,
                hash: candidate.hash,
                layerPages: candidate.layerPages,
                bytes: candidate.bytes)
            entries[entry.id] = entry
            entryIDsByHash[entry.hash, default: []].append(entry.id)
            for handle in entry.allPages {
                entryIDsByPage[handle, default: []].insert(entry.id)
            }
            bytesIndexed += entry.bytes
            published += 1
        }
        return published
    }

    var stats: CBv2PagedPrefixCacheStats {
        lock.lock()
        defer { lock.unlock() }
        return CBv2PagedPrefixCacheStats(
            hits: hits, misses: misses, tokensSaved: tokensSaved,
            blockCount: entries.count, bytesIndexed: bytesIndexed,
            invalidations: invalidations)
    }

    // MARK: - PagedKVPageReuseObserver

    func pagedKVPool(_ pool: PagedKVPool, isCached handle: PagedKVPageHandle) -> Bool {
        precondition(pool === self.pool)
        lock.lock()
        defer { lock.unlock() }
        return !(entryIDsByPage[handle]?.isEmpty ?? true)
    }

    func pagedKVPool(_ pool: PagedKVPool, willReuse handle: PagedKVPageHandle) {
        precondition(pool === self.pool)
        lock.lock()
        let ids = Array(entryIDsByPage[handle] ?? [])
        for id in ids { removeEntryLocked(id) }
        lock.unlock()
    }

    // MARK: - Locked helpers

    private func removeEntryLocked(_ id: UInt64) {
        guard let entry = entries.removeValue(forKey: id) else { return }
        if var ids = entryIDsByHash[entry.hash] {
            ids.removeAll { $0 == id }
            if ids.isEmpty {
                entryIDsByHash.removeValue(forKey: entry.hash)
            } else {
                entryIDsByHash[entry.hash] = ids
            }
        }
        var orphanedPages: [PagedKVPageHandle] = []
        for handle in entry.allPages {
            guard var ids = entryIDsByPage[handle] else { continue }
            ids.remove(id)
            if ids.isEmpty {
                entryIDsByPage.removeValue(forKey: handle)
                orphanedPages.append(handle)
            } else {
                entryIDsByPage[handle] = ids
            }
        }
        bytesIndexed -= entry.bytes
        precondition(bytesIndexed >= 0)
        invalidations += 1
        // A vLLM block is one allocator unit. Here one logical block bundles
        // physical pages from every owning layer/group, so invalidating any
        // constituent must reclassify the remaining zero-ref pages too. Live
        // pages stay put and are classified by their eventual release.
        for handle in orphanedPages { pool.reclassifyAsUncached(handle) }
    }

    private func hasher(cacheSalt: String?) -> CBv2BlockHasher {
        guard let cacheSalt else { return hasher }
        return CBv2BlockHasher(
            blockSize: config.blockSize,
            promptContractID: config.promptContractID,
            scopeID: cacheSalt)
    }

    private static func isCacheable(_ kind: CBv2LayerKind) -> Bool {
        guard kind.sharesKVWithLayer == nil else { return false }
        if case .slidingWindow = kind.attention { return false }
        return true
    }
}
