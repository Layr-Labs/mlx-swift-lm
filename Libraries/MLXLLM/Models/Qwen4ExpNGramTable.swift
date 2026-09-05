//
//  Qwen4ExpNGramTable.swift
//  mlx-swift-lm
//
//  The disk-resident n-gram row source of Qwen 3.8 Flash-Next.
//
//  ORIGIN. Moved here from
//  `Layr-Labs/mlxfast-qwen38-125b-a6b-engine-dev@b0b8d28:Sources/MLXFastModel/Qwen4ExpNGramTable.swift`.
//  The file carries no dependency on that track: the geometry comes from
//  `Qwen4ExpTextConfiguration` through the model's PLE layers, the refusals
//  use this file's own error, and the safetensors header reader below is
//  private to this file.
//
//  ONE SEAM, MANY SOURCES. `Qwen4ExpNGram.swift` owns the
//  `Qwen4ExpNGramRowSource` protocol. This file holds ONE conformer, the
//  disk reader. A later conformer -- a computed function instead of a table,
//  for example -- is added beside it and reached through
//  `Qwen4ExpNGramRowSourceLoader`, so a runner never names a conformer.
//
//  CONTRACT (runner-neutral; ngram-cache-design.md section 2).
//
//  * Row identity. A row is one integer, the global row id. The table holds
//    `shard_count * rows_per_shard` rows. Global row id `g` selects shard
//    `g / rows_per_shard` and row `g % rows_per_shard` inside it. Row ids are
//    a property of the checkpoint, not of a run.
//  * Ceiling semantics. The ceiling is a byte count and it bounds the cached
//    rows only. The default is 1 gibibyte. A flag value wins over the
//    environment value. Zero turns the cache off. A ceiling below the cost of
//    one row behaves as zero. The resolved ceiling is never raised silently.
//  * Eviction. Least recently used, per row. A row a step reads moves to the
//    front. When the cache is full the row at the back is dropped and its
//    space is reused.
//  * EXACTNESS INVARIANT. The cache must never change a value the model
//    computes. It holds the RAW checkpoint bytes of a row and nothing else,
//    so a cached row and a freshly read row are the same bytes and the
//    dequantization that follows is the same arithmetic on the same input. A
//    source must not store rows in a lower precision than the forward pass
//    uses, must not approximate or fall back to a nearby row on a miss, and
//    must not change the gather order.
//

import Darwin
import Foundation
import MLX

// MARK: - Refusals

/// Refusals of the disk-resident n-gram row source.
///
/// Every case names the file, the shard, or the path that caused it, because
/// the alternative -- a silent fallback -- allocates 29.8 GiB and fails far
/// from the cause.
public enum Qwen4ExpNGramTableError: Error, CustomStringConvertible, Equatable {
    case refused(String)

    public var description: String {
        switch self {
        case .refused(let detail): return detail
        }
    }
}

// MARK: - Geometry

/// Geometry of the sharded n-gram table of a Qwen 3.8 Flash-Next checkpoint.
///
/// Every value here is READ OFF the checkpoint, never guessed: the shard count
/// and the row count come from the model configuration, and the row width and
/// the quantization come from the shard tensors themselves.
public struct Qwen4ExpNGramTableLayout: Equatable, Sendable {
    public let shardCount: Int
    public let rowsPerShard: Int
    public let rowDimensions: Int
    public let bits: Int
    public let groupSize: Int

    public init(
        shardCount: Int, rowsPerShard: Int, rowDimensions: Int, bits: Int = 4, groupSize: Int = 32
    ) {
        self.shardCount = shardCount
        self.rowsPerShard = rowsPerShard
        self.rowDimensions = rowDimensions
        self.bits = bits
        self.groupSize = groupSize
    }

    /// Packed weight words per row. Eight 4-bit values share one 32-bit word.
    public var weightWordsPerRow: Int { rowDimensions * bits / 32 }
    public var weightBytesPerRow: Int { weightWordsPerRow * 4 }
    /// One scale and one bias per quantization group.
    public var groupsPerRow: Int { rowDimensions / groupSize }
    /// Scales and biases are bfloat16.
    public var groupBytesPerRow: Int { groupsPerRow * 2 }
    /// Bytes one row costs in the cache: weights, scales and biases together.
    public var bytesPerRow: Int { weightBytesPerRow + 2 * groupBytesPerRow }
    public var rowCount: Int { shardCount * rowsPerShard }
}

/// Cache ceiling, in bytes.
///
/// Resolution order: the explicit flag value, then the environment variable,
/// then the default of one gibibyte. A value of zero turns the cache off; the
/// table then reads every row from the file. The MODEL OUTPUT IS THE SAME at
/// every ceiling -- see `Qwen4ExpNGramTable`.
public enum Qwen4ExpNGramCacheLimit {
    public static let environmentName = "MLXFAST_NGRAM_CACHE_LIMIT"
    public static let defaultBytes = 1 << 30

    public static func parse(_ raw: String, optionLabel: String = environmentName) throws -> Int {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultBytes }
        guard let value = Int(trimmed), value >= 0 else {
            throw Qwen4ExpNGramTableError.refused(
                "\(optionLabel) must be a byte count of zero or more")
        }
        return value
    }

    public static func resolve(
        flagValue: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Int {
        if let flagValue {
            return try parse(flagValue, optionLabel: "--ngram-cache-limit")
        }
        return try parse(environment[environmentName] ?? "")
    }
}

// MARK: - The table

/// Solid-state-disk resident n-gram table with a bounded row cache.
///
/// The table is 29.8 GiB for the pinned checkpoint, which is more than a
/// third of the model. It stays in its safetensors files, memory mapped. A
/// forward pass asks for the rows of the current tokens -- sixteen rows per
/// token -- and the table returns them.
///
/// EXACTNESS INVARIANT. The cache holds the raw checkpoint bytes of a row and
/// nothing else. A hit and a miss therefore return the SAME bytes, and the
/// dequantization that follows is the same arithmetic on the same input. The
/// ceiling, the eviction order and the cache state can never change a value
/// this table returns. `Qwen4ExpNGramCacheExactnessTests` pins this.
///
/// EVICTION. Least recently used, by row. A row that a forward pass touches is
/// moved to the front. When the arena is full the row at the back is dropped
/// and its slot is reused.
public final class Qwen4ExpNGramTable: Qwen4ExpNGramRowSource {

    /// One shard's byte offsets inside its mapped file.
    struct ShardLocation {
        let fileIndex: Int
        let weightOffset: Int
        let scalesOffset: Int
        let biasesOffset: Int
    }

    public let layout: Qwen4ExpNGramTableLayout
    public let ceilingBytes: Int

    private let files: [Data]
    private let shards: [ShardLocation]
    private let cache: RowCache

    public var rowDimensions: Int { layout.rowDimensions }

    /// Rows served from the cache and from the file since construction.
    public private(set) var hitCount: Int = 0
    public private(set) var missCount: Int = 0

    /// - Parameters:
    ///   - shardFiles: safetensors files that hold the n-gram shard tensors.
    ///     Files without one are ignored, so the whole shard directory can be
    ///     passed.
    ///   - tensorPrefix: prefix of the shard tensors, for example
    ///     `language_model.model.layers.1.ple.ple_embedding.ngram_embedding`.
    ///   - layout: the table geometry.
    ///   - ceilingBytes: cache ceiling. Zero turns the cache off.
    public init(
        shardFiles: [URL],
        tensorPrefix: String,
        layout: Qwen4ExpNGramTableLayout,
        ceilingBytes: Int
    ) throws {
        self.layout = layout
        self.ceilingBytes = ceilingBytes

        var mapped: [Data] = []
        var located: [Int: ShardLocation] = [:]

        for url in shardFiles {
            let header = try SafetensorsHeader.read(url)
            var used = false
            for shard in 0 ..< layout.shardCount {
                let base = "\(tensorPrefix).shard_\(shard)"
                guard let weight = header.tensors["\(base).weight"],
                    let scales = header.tensors["\(base).scales"],
                    let biases = header.tensors["\(base).biases"]
                else { continue }

                try Qwen4ExpNGramTable.validate(
                    shard: shard, weight: weight, scales: scales, biases: biases, layout: layout,
                    file: url)

                if !used {
                    // Mapped, not read: the point of this class is that the
                    // table never enters resident memory as a whole. The map is
                    // advised MADV_RANDOM so a per-token row fault does not drag
                    // in a readahead cluster -- see `mapShardFile(at:)`.
                    mapped.append(try Qwen4ExpNGramTable.mapShardFile(at: url))
                    used = true
                }
                let dataBase = header.dataBaseOffset
                located[shard] = ShardLocation(
                    fileIndex: mapped.count - 1,
                    weightOffset: dataBase + weight.dataStart,
                    scalesOffset: dataBase + scales.dataStart,
                    biasesOffset: dataBase + biases.dataStart
                )
            }
        }

        guard located.count == layout.shardCount else {
            let missing = (0 ..< layout.shardCount).filter { located[$0] == nil }
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: \(missing.count) of \(layout.shardCount) n-gram shards \
                were not found under prefix "\(tensorPrefix)". First missing: \
                shard_\(missing.first ?? -1).
                """)
        }

        self.files = mapped
        self.shards = (0 ..< layout.shardCount).map { located[$0]! }
        self.cache = RowCache(
            bytesPerRow: layout.bytesPerRow, ceilingBytes: ceilingBytes)
    }

    /// Memory-map one shard file with kernel readahead suppressed.
    ///
    /// A decode token faults in sixteen small, scattered rows through the mmap
    /// (`readRowFromFile`). Under the kernel's default readahead every random
    /// first-touch fault pulls a whole page cluster the row read never uses --
    /// hundreds of disk page-ins a token on the decode critical path, so the
    /// GPU idles waiting on I/O. `MADV_RANDOM` tells the kernel not to read
    /// ahead, so each fault brings in only the page the row needs.
    ///
    /// This is an access-pattern hint only. It changes which pages the kernel
    /// prefetches, never a byte a row read copies. The mapping stays cached
    /// (no `F_NOCACHE`, no `MADV_DONTNEED`), so the page cache and the RowCache
    /// still serve re-touched rows. The map is created here, so the advice is
    /// certain to land on the real region; the `.custom` deallocator munmaps it
    /// when the table is released.
    private static func mapShardFile(at url: URL) throws -> Data {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: cannot open shard file \(url.lastPathComponent)")
        }
        // Suppress readahead on the descriptor too; MADV_RANDOM on the mapping
        // below is the load-bearing knob.
        _ = Darwin.fcntl(fd, F_RDAHEAD, 0)

        var status = stat()
        guard fstat(fd, &status) == 0, status.st_size > 0 else {
            close(fd)
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: cannot size shard file \(url.lastPathComponent)")
        }
        let length = Int(status.st_size)

        let mapping = mmap(nil, length, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 0)
        // The mapping holds its own reference to the file; the descriptor is
        // no longer needed once the map exists.
        close(fd)
        guard let base = mapping, base != MAP_FAILED else {
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: cannot map shard file \(url.lastPathComponent)")
        }
        _ = madvise(base, length, MADV_RANDOM)

        return Data(
            bytesNoCopy: base, count: length,
            deallocator: .custom { pointer, byteCount in munmap(pointer, byteCount) })
    }

    private static func validate(
        shard: Int,
        weight: SafetensorInfo,
        scales: SafetensorInfo,
        biases: SafetensorInfo,
        layout: Qwen4ExpNGramTableLayout,
        file: URL
    ) throws {
        func check(_ condition: Bool, _ message: String) throws {
            guard condition else {
                throw Qwen4ExpNGramTableError.refused(
                    "Qwen4ExpNGramTable: \(file.lastPathComponent) shard_\(shard): \(message)")
            }
        }
        try check(
            weight.shape == [layout.rowsPerShard, layout.weightWordsPerRow],
            "weight shape \(weight.shape) is not [\(layout.rowsPerShard), \(layout.weightWordsPerRow)]"
        )
        try check(weight.dtype == "U32", "weight dtype \(weight.dtype) is not U32")
        for (name, info) in [("scales", scales), ("biases", biases)] {
            try check(
                info.shape == [layout.rowsPerShard, layout.groupsPerRow],
                "\(name) shape \(info.shape) is not [\(layout.rowsPerShard), \(layout.groupsPerRow)]"
            )
            try check(info.dtype == "BF16", "\(name) dtype \(info.dtype) is not BF16")
        }
    }

    // MARK: Row gathering

    public func rows(globalIds: MLXArray) -> MLXArray {
        let shape = globalIds.shape
        let ids = globalIds.asType(.int32).asArray(Int32.self)
        let count = ids.count

        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow
        var weights = Data(count: count * weightBytes)
        var scales = Data(count: count * groupBytes)
        var biases = Data(count: count * groupBytes)

        weights.withUnsafeMutableBytes { weightOut in
            scales.withUnsafeMutableBytes { scaleOut in
                biases.withUnsafeMutableBytes { biasOut in
                    for (position, id) in ids.enumerated() {
                        let row = readRow(Int(id))
                        row.withUnsafeBytes { source in
                            let base = source.baseAddress!
                            memcpy(
                                weightOut.baseAddress! + position * weightBytes,
                                base, weightBytes)
                            memcpy(
                                scaleOut.baseAddress! + position * groupBytes,
                                base + weightBytes, groupBytes)
                            memcpy(
                                biasOut.baseAddress! + position * groupBytes,
                                base + weightBytes + groupBytes, groupBytes)
                        }
                    }
                }
            }
        }

        let packed = MLXArray(weights, [count, layout.weightWordsPerRow], dtype: .uint32)
        let scaleArray = MLXArray(scales, [count, layout.groupsPerRow], dtype: .bfloat16)
        let biasArray = MLXArray(biases, [count, layout.groupsPerRow], dtype: .bfloat16)

        let dequantizedRows = dequantized(
            packed,
            scales: scaleArray,
            biases: biasArray,
            groupSize: layout.groupSize,
            bits: layout.bits
        )
        return dequantizedRows.reshaped(shape + [layout.rowDimensions])
    }

    /// The raw checkpoint bytes of one row: weights, then scales, then biases.
    ///
    /// This is the ONLY place a row is produced, so a cached row and a freshly
    /// read row cannot differ.
    func readRow(_ globalId: Int) -> [UInt8] {
        if let cached = cache.value(for: globalId) {
            hitCount += 1
            return cached
        }
        missCount += 1
        let row = readRowFromFile(globalId)
        cache.insert(row, for: globalId)
        return row
    }

    private func readRowFromFile(_ globalId: Int) -> [UInt8] {
        precondition(
            globalId >= 0 && globalId < layout.rowCount,
            "Qwen4ExpNGramTable: row \(globalId) is outside the table")
        let shard = shards[globalId / layout.rowsPerShard]
        let row = globalId % layout.rowsPerShard
        let file = files[shard.fileIndex]

        var out = [UInt8](repeating: 0, count: layout.bytesPerRow)
        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow
        out.withUnsafeMutableBytes { destination in
            file.withUnsafeBytes { source in
                let base = source.baseAddress!
                memcpy(
                    destination.baseAddress!,
                    base + shard.weightOffset + row * weightBytes, weightBytes)
                memcpy(
                    destination.baseAddress! + weightBytes,
                    base + shard.scalesOffset + row * groupBytes, groupBytes)
                memcpy(
                    destination.baseAddress! + weightBytes + groupBytes,
                    base + shard.biasesOffset + row * groupBytes, groupBytes)
            }
        }
        return out
    }

    // MARK: Least-recently-used row cache

    /// A fixed arena of row slots with a least-recently-used order.
    ///
    /// The arena is sized once from the ceiling, so there is no growth path and
    /// no allocation on the hot path. A ceiling below one row's cost turns the
    /// cache off.
    final class RowCache {
        private let bytesPerRow: Int
        private let capacity: Int
        private var arena: [UInt8]
        private var slotOfRow: [Int: Int] = [:]
        private var rowOfSlot: [Int]
        private var previous: [Int]
        private var next: [Int]
        private var head = -1
        private var tail = -1
        private var used = 0

        init(bytesPerRow: Int, ceilingBytes: Int) {
            self.bytesPerRow = bytesPerRow
            self.capacity = bytesPerRow > 0 ? ceilingBytes / bytesPerRow : 0
            self.arena = [UInt8](repeating: 0, count: capacity * bytesPerRow)
            self.rowOfSlot = [Int](repeating: -1, count: capacity)
            self.previous = [Int](repeating: -1, count: capacity)
            self.next = [Int](repeating: -1, count: capacity)
            self.slotOfRow.reserveCapacity(capacity)
        }

        var slotCapacity: Int { capacity }

        func value(for row: Int) -> [UInt8]? {
            guard let slot = slotOfRow[row] else { return nil }
            moveToFront(slot)
            let start = slot * bytesPerRow
            return Array(arena[start ..< (start + bytesPerRow)])
        }

        func insert(_ bytes: [UInt8], for row: Int) {
            guard capacity > 0 else { return }
            let slot: Int
            if used < capacity {
                slot = used
                used += 1
            } else {
                slot = tail
                slotOfRow.removeValue(forKey: rowOfSlot[slot])
                detach(slot)
            }
            rowOfSlot[slot] = row
            slotOfRow[row] = slot
            let start = slot * bytesPerRow
            arena.replaceSubrange(start ..< (start + bytesPerRow), with: bytes)
            pushFront(slot)
        }

        private func moveToFront(_ slot: Int) {
            guard head != slot else { return }
            detach(slot)
            pushFront(slot)
        }

        private func detach(_ slot: Int) {
            let p = previous[slot]
            let n = next[slot]
            if p != -1 { next[p] = n }
            if n != -1 { previous[n] = p }
            if head == slot { head = n }
            if tail == slot { tail = p }
            previous[slot] = -1
            next[slot] = -1
        }

        private func pushFront(_ slot: Int) {
            previous[slot] = -1
            next[slot] = head
            if head != -1 { previous[head] = slot }
            head = slot
            if tail == -1 { tail = slot }
        }
    }
}

// MARK: - Building the table from a checkpoint

extension Qwen4ExpNGramTable {

    /// Tensor-name prefixes to try for the n-gram shards of one PLE layer.
    ///
    /// The published checkpoint nests the tower under `language_model.`; a
    /// checkpoint that has already been through `sanitize(weights:)` does not.
    /// Both spellings are accepted so the table works either way.
    static func candidatePrefixes(pleLayerIndex: Int) -> [String] {
        let tail = "model.layers.\(pleLayerIndex).ple.ple_embedding.ngram_embedding"
        return ["language_model.\(tail)", tail]
    }

    /// The safetensors files of one shard directory, in name order.
    static func shardFiles(in directory: URL) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory)
        else {
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: \(directory.path) does not exist. This source \
                takes the DIRECTORY of n-gram shard files that the offline \
                transform writes.
                """)
        }
        guard isDirectory.boolValue else {
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: \(directory.path) is a file. This source takes \
                the DIRECTORY of n-gram shard files that the offline transform \
                writes, not one file inside it.
                """)
        }
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        let files = entries.filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: \(directory.path) holds no .safetensors file")
        }
        return files
    }

    /// Build the table for a model's one PLE layer from a shard directory.
    ///
    /// The geometry comes from the model, so nothing here is guessed: the
    /// shard count, the rows per shard and the row width are the values
    /// `Qwen4ExpTextConfiguration` gave the PLE layer.
    ///
    /// - Parameters:
    ///   - directory: the shard directory the offline transform writes. Files
    ///     that hold no n-gram shard are skipped.
    ///   - model: the loaded target.
    ///   - ceilingBytes: cache ceiling. Defaults to the resolved flag or
    ///     environment value, then to one gibibyte.
    public static func make(
        shardDirectory directory: URL,
        for model: Qwen4ExpModel,
        ceilingBytes: Int? = nil
    ) throws -> Qwen4ExpNGramTable {
        let embeddings = model.pleEmbeddings
        guard let embedding = embeddings.first else {
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: this model declares no PLE layer, so it has no n-gram table")
        }
        guard embeddings.count == 1 else {
            // The pinned checkpoint has exactly one PLE layer. More than one
            // would need one table each, which is a change, not a detail.
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: this model declares \(embeddings.count) PLE layers. \
                The builder serves one; extend it deliberately rather than sharing a table.
                """)
        }

        let layout = Qwen4ExpNGramTableLayout(
            shardCount: embedding.shardCount,
            rowsPerShard: embedding.rowsPerShard,
            rowDimensions: embedding.rowDimensions
        )
        let ceiling = try ceilingBytes ?? Qwen4ExpNGramCacheLimit.resolve()
        let files = try shardFiles(in: directory)
        let layerIndex = model.model.pleLayerIndices[0]

        var lastError: Error?
        for prefix in candidatePrefixes(pleLayerIndex: layerIndex) {
            do {
                return try Qwen4ExpNGramTable(
                    shardFiles: files,
                    tensorPrefix: prefix,
                    layout: layout,
                    ceilingBytes: ceiling
                )
            } catch {
                lastError = error
            }
        }
        throw lastError
            ?? Qwen4ExpNGramTableError.refused("Qwen4ExpNGramTable: no n-gram shards were found")
    }
}

// MARK: - The one construction entry point

/// Builds a `Qwen4ExpNGramRowSource` from a path resource.
///
/// A runner names THIS type and the protocol, never a conformer. The disk
/// reader is one conformer today. A later conformer -- a computed function,
/// for example -- is chosen here, so no runner changes.
///
/// RESOURCE SHAPE. One shape only: the DIRECTORY of n-gram shard files that
/// the offline transform writes. A path to a single file is refused by name.
public enum Qwen4ExpNGramRowSourceLoader {

    public static func rowSource(
        at path: URL,
        for model: Qwen4ExpModel,
        ceilingBytes: Int? = nil
    ) throws -> any Qwen4ExpNGramRowSource {
        try Qwen4ExpNGramTable.make(
            shardDirectory: path, for: model, ceilingBytes: ceilingBytes)
    }
}

// MARK: - Minimal safetensors header reader

/// One tensor's placement inside a safetensors file.
///
/// Private to this file on purpose: the table needs offsets, not arrays, and
/// the shared loader reads arrays. Keeping this here is what lets the fork
/// hold the reader without a dependency on the track.
struct SafetensorInfo: Equatable {
    let dtype: String
    let shape: [Int]
    let dataStart: Int
    let dataEnd: Int
}

struct SafetensorsHeader: Equatable {
    let headerLength: Int
    let tensors: [String: SafetensorInfo]

    /// First byte of the tensor data block: the 8-byte length word plus the
    /// header itself.
    var dataBaseOffset: Int { headerLength + 8 }

    /// Matches the safetensors reference implementation's header limit.
    static let maximumHeaderByteCount = 100_000_000

    static func read(_ path: URL) throws -> SafetensorsHeader {
        func refuse(_ message: String) -> Qwen4ExpNGramTableError {
            .refused("Qwen4ExpNGramTable: \(message): \(path.path)")
        }

        let handle = try FileHandle(forReadingFrom: path)
        defer { try? handle.close() }

        var status = stat()
        guard fstat(handle.fileDescriptor, &status) == 0 else {
            throw refuse("cannot size safetensors file")
        }
        let fileByteCount = Int(status.st_size)
        guard fileByteCount >= 8 else { throw refuse("safetensors file is too small") }

        let prefix = handle.readData(ofLength: 8)
        guard prefix.count == 8 else { throw refuse("safetensors file is too small") }
        let rawHeaderLength = prefix.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard rawHeaderLength > 0 else { throw refuse("safetensors header is empty") }
        guard rawHeaderLength <= UInt64(maximumHeaderByteCount),
            let headerLength = Int(exactly: rawHeaderLength)
        else {
            throw refuse("safetensors header exceeds \(maximumHeaderByteCount) bytes")
        }
        guard headerLength <= fileByteCount - 8 else {
            throw refuse("truncated safetensors header")
        }

        let headerData = handle.readData(ofLength: headerLength)
        guard headerData.count == headerLength else { throw refuse("truncated safetensors header") }

        guard let object = try? JSONSerialization.jsonObject(with: headerData),
            let dictionary = object as? [String: Any]
        else {
            throw refuse("safetensors header is not a JSON object")
        }

        let dataByteCount = fileByteCount - 8 - headerLength
        var tensors: [String: SafetensorInfo] = [:]
        for (name, value) in dictionary where name != "__metadata__" {
            guard let tensor = value as? [String: Any] else { continue }
            guard let dtype = tensor["dtype"] as? String,
                let shape = (tensor["shape"] as? [NSNumber])?.map({ $0.intValue }),
                let offsets = (tensor["data_offsets"] as? [NSNumber])?.map({ $0.intValue }),
                offsets.count == 2
            else {
                throw refuse("invalid tensor header for \(name)")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= dataByteCount else {
                throw refuse("invalid data_offsets for \(name)")
            }
            tensors[name] = SafetensorInfo(
                dtype: dtype, shape: shape, dataStart: offsets[0], dataEnd: offsets[1])
        }

        return SafetensorsHeader(headerLength: headerLength, tensors: tensors)
    }
}
