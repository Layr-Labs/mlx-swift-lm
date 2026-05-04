// Copyright © 2026 Eigen Labs.
//
// SSD-backed KV cache block storage.
// Port of omlx/omlx/cache/paged_ssd_cache.py
//
// Each block is stored as a .safetensors file named by its SHA-256 hex digest
// under a one-level subdirectory keyed on the first hex character of the hash.
// Writes are dispatched to a background queue; reads are synchronous.
// Files are evicted LRU when total size exceeds the configured limit.

import Foundation
@preconcurrency import MLX

// MARK: - SSDBlockMetadata

struct SSDBlockMetadata {
    var blockHash: Data
    var filePath: URL
    var fileSize: Int
    var tokenCount: Int
    var numLayers: Int
    var lastAccess: Date

    mutating func touch() { lastAccess = .init() }
}

// MARK: - SSDCacheIndex

/// Thread-safe LRU index of SSD cache files.
final class SSDCacheIndex: @unchecked Sendable {
    private var index: [Data: SSDBlockMetadata] = [:]
    private var lruOrder: [Data] = []       // front = oldest (evict first)
    private(set) var totalSize: Int = 0
    let maxSize: Int
    private let lock = NSLock()

    init(maxSize: Int) { self.maxSize = maxSize }

    func add(_ meta: SSDBlockMetadata) {
        lock.lock(); defer { lock.unlock() }
        if let old = index[meta.blockHash] {
            totalSize -= old.fileSize
            lruOrder.removeAll { $0 == meta.blockHash }
        }
        index[meta.blockHash] = meta
        totalSize += meta.fileSize
        lruOrder.append(meta.blockHash)
    }

    func get(_ hash: Data) -> SSDBlockMetadata? {
        lock.lock(); defer { lock.unlock() }
        return index[hash]
    }

    func touch(_ hash: Data) {
        lock.lock(); defer { lock.unlock() }
        index[hash]?.touch()
        if let i = lruOrder.firstIndex(of: hash) {
            lruOrder.remove(at: i)
            lruOrder.append(hash)
        }
    }

    func contains(_ hash: Data) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return index[hash] != nil
    }

    /// Evict LRU entries until total size ≤ `targetSize`. Returns file URLs to delete.
    func evictUntil(targetSize: Int) -> [URL] {
        lock.lock(); defer { lock.unlock() }
        var evicted: [URL] = []
        while totalSize > targetSize, let oldest = lruOrder.first {
            lruOrder.removeFirst()
            if let meta = index.removeValue(forKey: oldest) {
                totalSize -= meta.fileSize
                evicted.append(meta.filePath)
            }
        }
        return evicted
    }

    func updateSize(_ hash: Data, actualSize: Int) {
        lock.lock(); defer { lock.unlock() }
        guard let old = index[hash] else { return }
        totalSize += actualSize - old.fileSize
        index[hash]?.fileSize = actualSize
    }

    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return index.count
    }
}

// MARK: - SSDCacheConfig

public struct SSDCacheConfig: Sendable {
    /// Directory for cache files.
    public var cacheDir: URL
    /// Maximum total on-disk size in bytes (default 50 GB).
    public var maxSizeBytes: Int

    public init(cacheDir: URL, maxSizeBytes: Int = 50 * 1_024 * 1_024 * 1_024) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = maxSizeBytes
    }
}

// MARK: - SSDCacheManager

/// SSD-backed storage for KV cache blocks.
///
/// Serialises `KVCacheSimple` layer state as safetensors files and reloads
/// them on demand. Writes are non-blocking (dispatched to a background queue);
/// reads block the calling thread.
///
/// ### File layout
/// ```
/// cacheDir/
///   0/  1/  ...  f/          ← first hex char of SHA-256 hash
///     <64-hex-char>.safetensors
/// ```
///
/// ### Array naming inside each file
/// ```
/// layer_0_keys, layer_0_values, layer_1_keys, layer_1_values, …
/// ```
///
/// Port of omlx/omlx/cache/paged_ssd_cache.py.
public final class SSDCacheManager: @unchecked Sendable {
    public let config: SSDCacheConfig
    private let index: SSDCacheIndex
    private let writeQueue = DispatchQueue(
        label: "com.eigen.ssd-cache-writer", qos: .background)
    private var pendingWrites: Set<Data> = []
    private let pendingLock = NSLock()

    public private(set) var saves = 0
    public private(set) var loads = 0
    public private(set) var hits = 0
    public private(set) var misses = 0
    public private(set) var evictions = 0

    public init(config: SSDCacheConfig) {
        self.config = config
        self.index = SSDCacheIndex(maxSize: config.maxSizeBytes)
        createDirectories()
        scanExisting()
    }

    // MARK: - Public API

    /// Returns true if a block with this hash is indexed on disk.
    public func hasBlock(hash: Data) -> Bool { index.contains(hash) }

    /// Synchronously load a block. Returns `nil` on miss or I/O error.
    /// Must not be called on the main thread.
    public func loadBlock(hash: Data) -> [KVCacheSimple]? {
        guard let meta = index.get(hash) else { misses += 1; return nil }
        do {
            let arrays = try loadArrays(url: meta.filePath)
            var caches: [KVCacheSimple] = []
            for i in 0 ..< meta.numLayers {
                guard let k = arrays["layer_\(i)_keys"],
                      let v = arrays["layer_\(i)_values"] else { return nil }
                let cache = KVCacheSimple()
                cache.state = [k, v]
                caches.append(cache)
            }
            guard caches.count == meta.numLayers else { return nil }
            index.touch(hash)
            hits += 1
            loads += 1
            return caches
        } catch {
            return nil
        }
    }

    /// Asynchronously save a block to disk.
    /// Safe to call from the engine loop; the write happens on a background queue.
    public func saveBlock(hash: Data, layerCaches: [KVCacheSimple], tokenCount: Int) {
        pendingLock.lock()
        guard !index.contains(hash), !pendingWrites.contains(hash) else {
            pendingLock.unlock()
            return
        }
        pendingWrites.insert(hash)
        pendingLock.unlock()

        // Evaluate arrays on the calling thread (Metal is not thread-safe).
        let pairs: [(MLXArray, MLXArray)] = layerCaches.compactMap {
            guard $0.state.count >= 2 else { return nil }
            let k = $0.state[0]; let v = $0.state[1]
            eval(k, v)
            return (k, v)
        }
        let numLayers = pairs.count
        guard numLayers > 0 else {
            pendingLock.lock(); pendingWrites.remove(hash); pendingLock.unlock()
            return
        }

        let filePath = fileURL(for: hash)
        let estimatedSize = pairs.reduce(0) { $0 + $1.0.nbytes + $1.1.nbytes }

        // Evict LRU files to stay within the size budget.
        let toDelete = index.evictUntil(targetSize: config.maxSizeBytes - estimatedSize)
        for url in toDelete {
            try? FileManager.default.removeItem(at: url)
            evictions += 1
        }

        writeQueue.async { [weak self] in
            guard let self else { return }
            defer {
                self.pendingLock.lock()
                self.pendingWrites.remove(hash)
                self.pendingLock.unlock()
            }
            do {
                var dict: [String: MLXArray] = [:]
                for (i, (k, v)) in pairs.enumerated() {
                    dict["layer_\(i)_keys"] = k
                    dict["layer_\(i)_values"] = v
                }
                try save(arrays: dict, url: filePath)
                let attrs = try FileManager.default.attributesOfItem(atPath: filePath.path)
                let fileSize = (attrs[.size] as? Int) ?? estimatedSize
                let meta = SSDBlockMetadata(
                    blockHash: hash, filePath: filePath,
                    fileSize: fileSize, tokenCount: tokenCount,
                    numLayers: numLayers, lastAccess: .init()
                )
                self.index.add(meta)
                self.saves += 1
            } catch {
                // Non-fatal: block just won't be persisted.
            }
        }
    }

    public func getStats() -> [String: Any] {
        [
            "cached_blocks": index.count,
            "total_size_bytes": index.totalSize,
            "max_size_bytes": config.maxSizeBytes,
            "saves": saves, "loads": loads,
            "hits": hits, "misses": misses,
            "evictions": evictions,
        ]
    }

    // MARK: - Private

    private static let subdirChars = "0123456789abcdef"

    private func createDirectories() {
        try? FileManager.default.createDirectory(
            at: config.cacheDir, withIntermediateDirectories: true)
        for c in Self.subdirChars {
            try? FileManager.default.createDirectory(
                at: config.cacheDir.appendingPathComponent(String(c)),
                withIntermediateDirectories: true)
        }
    }

    private func fileURL(for hash: Data) -> URL {
        let hex = hash.map { String(format: "%02x", $0) }.joined()
        return config.cacheDir
            .appendingPathComponent(String(hex.prefix(1)))
            .appendingPathComponent("\(hex).safetensors")
    }

    private func scanExisting() {
        for c in Self.subdirChars {
            let subdir = config.cacheDir.appendingPathComponent(String(c))
            guard let urls = try? FileManager.default.contentsOfDirectory(
                at: subdir, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]
            ) else { continue }
            for url in urls where url.pathExtension == "safetensors" {
                if let meta = readMetadata(url) { index.add(meta) }
            }
        }
    }

    private func readMetadata(_ url: URL) -> SSDBlockMetadata? {
        let hex = url.deletingPathExtension().lastPathComponent
        guard hex.count == 64, let hashData = Data(hexString: hex) else { return nil }
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) else { return nil }
        let fileSize = (attrs[.size] as? Int) ?? 0

        // Count layer count from tensor names without loading full arrays.
        guard let arrays = try? loadArrays(url: url) else { return nil }
        let numLayers = arrays.keys.filter { $0.hasSuffix("_keys") }.count
        guard numLayers > 0 else { return nil }

        return SSDBlockMetadata(
            blockHash: hashData, filePath: url,
            fileSize: fileSize, tokenCount: 0,
            numLayers: numLayers,
            lastAccess: (attrs[.modificationDate] as? Date) ?? .init()
        )
    }
}

// MARK: - Data hex helper

private extension Data {
    init?(hexString hex: String) {
        guard hex.count.isMultiple(of: 2) else { return nil }
        var data = Data(capacity: hex.count / 2)
        var i = hex.startIndex
        while i < hex.endIndex {
            let j = hex.index(i, offsetBy: 2)
            guard let byte = UInt8(hex[i ..< j], radix: 16) else { return nil }
            data.append(byte)
            i = j
        }
        self = data
    }
}
