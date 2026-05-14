// MLXPack — a page-aligned on-disk format that enables zero-copy mmap loading
// of model weights into Metal-shared MLXArrays.
//
// On-disk layout:
//   [0..8)     magic "MLXPACK\0"
//   [8..12)    version (uint32 LE)
//   [12..20)   header_len (uint64 LE) — bytes of JSON header that follow
//   [20..hdr_end)  JSON header (utf-8)
//   [hdr_end..page_aligned) zero padding
//   [page_aligned..)  tensor data, each tensor's `offset` is a multiple of
//                     `pageSize` and its allocated extent is rounded up to
//                     the next page boundary
//
// JSON header schema:
//   { "metadata": { "key": "value", ... },
//     "tensors": [
//       { "name": "...", "dtype": "BF16", "shape": [...],
//         "offset": <int>, "size": <int> },
//       ...
//     ] }
//
// The MetalAllocator::make_buffer fast path in mlx requires page-aligned
// pointers; safetensors does not guarantee this so wrapping safetensors
// bytes via MLXArray(rawPointer:) silently falls back to malloc+memcpy.
// MLXPack stores each tensor at a page boundary so the wrap succeeds and
// the MLXArray's underlying storage *is* the mmap region — zero copy.

import Foundation
import MLX

#if canImport(Darwin)
import Darwin
#endif

public enum MLXPack {
    public static let magic: [UInt8] = Array("MLXPACK\0".utf8)
    public static let version: UInt32 = 1
    /// Apple Silicon page size. `vm_page_size` would be the runtime value;
    /// 16384 is fixed for arm64-darwin and we want files portable across
    /// machines.
    public static let pageSize: Int = 16384

    /// Suffix for converted bundles. Sits next to (not inside) the safetensors
    /// directory so a model dir can hold both formats.
    public static let suffix = ".mlxpack"

    public struct Entry: Codable, Equatable, Sendable {
        public let name: String
        public let dtype: String   // mirrors safetensors dtype names: "F32" "F16" "BF16" "U8" "I8" ...
        public let shape: [Int]
        public let offset: Int
        public let size: Int
    }

    public struct Header: Codable, Sendable {
        public let metadata: [String: String]
        public let tensors: [Entry]
    }

    // MARK: - dtype mapping

    /// Convert a safetensors-style dtype string to an MLX DType.
    public static func dtype(forSafeName name: String) -> DType? {
        switch name {
        case "BOOL": return .bool
        case "U8": return .uint8
        case "I8": return .int8
        case "U16": return .uint16
        case "I16": return .int16
        case "U32": return .uint32
        case "I32": return .int32
        case "U64": return .uint64
        case "I64": return .int64
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "F32": return .float32
        case "F64": return .float64
        default: return nil
        }
    }

    /// Inverse of `dtype(forSafeName:)`.
    public static func safeName(for dtype: DType) -> String? {
        switch dtype {
        case .bool: return "BOOL"
        case .uint8: return "U8"
        case .int8: return "I8"
        case .uint16: return "U16"
        case .int16: return "I16"
        case .uint32: return "U32"
        case .int32: return "I32"
        case .uint64: return "U64"
        case .int64: return "I64"
        case .float16: return "F16"
        case .bfloat16: return "BF16"
        case .float32: return "F32"
        case .float64: return "F64"
        default: return nil
        }
    }

    public static func itemSize(of dtype: DType) -> Int {
        switch dtype {
        case .bool, .uint8, .int8: return 1
        case .uint16, .int16, .float16, .bfloat16: return 2
        case .uint32, .int32, .float32: return 4
        case .uint64, .int64, .float64: return 8
        case .complex64: return 8
        @unknown default: return 0
        }
    }

    @inline(__always)
    public static func roundUpToPage(_ x: Int) -> Int {
        (x + pageSize - 1) & ~(pageSize - 1)
    }
}

// MARK: - Writer (safetensors → .mlxpack)

#if canImport(Darwin)

public enum MLXPackError: Error, CustomStringConvertible {
    case openFailed(String)
    case mmapFailed(String)
    case truncateFailed(String)
    case badMagic
    case unsupportedVersion(UInt32)
    case unknownDType(String)
    case malformedSafetensors(String)
    case writeFailed(String)
    case readFailed(String)

    public var description: String {
        switch self {
        case .openFailed(let s): return "open failed: \(s)"
        case .mmapFailed(let s): return "mmap failed: \(s)"
        case .truncateFailed(let s): return "ftruncate failed: \(s)"
        case .badMagic: return "not an mlxpack file (bad magic)"
        case .unsupportedVersion(let v): return "unsupported mlxpack version \(v)"
        case .unknownDType(let s): return "unknown dtype \(s)"
        case .malformedSafetensors(let s): return "malformed safetensors: \(s)"
        case .writeFailed(let s): return "write failed: \(s)"
        case .readFailed(let s): return "read failed: \(s)"
        }
    }
}

/// Convert a directory of safetensors shards into a single `.mlxpack` file
/// with page-aligned tensors. Suitable for one-time conversion at install
/// time; the loader (`MLXPackLoader.load`) then mmap's the result for
/// zero-copy reads.
public func convertSafetensorsDirectoryToMLXPack(
    directory: URL, outputFile: URL
) throws {
    let shardURLs = try collectSafetensorsShards(in: directory)
    try convertSafetensorsShardsToMLXPack(shards: shardURLs, outputFile: outputFile)
}

/// Convert a list of safetensors shards into a single `.mlxpack` file.
public func convertSafetensorsShardsToMLXPack(
    shards: [URL], outputFile: URL
) throws {
    // 1. Parse every shard's header into a unified tensor inventory.
    typealias SrcEntry = (
        name: String, dtype: String, shape: [Int],
        shardIdx: Int, srcStart: Int, srcEnd: Int
    )
    var srcEntries: [SrcEntry] = []
    var shardDataBases: [Int] = []  // byte offset of data region within each shard
    var metadata: [String: String] = [:]

    for (shardIdx, url) in shards.enumerated() {
        let (jsonHeader, dataBase) = try parseSafetensorsHeader(url: url)
        shardDataBases.append(dataBase)
        guard let dict = jsonHeader as? [String: Any] else {
            throw MLXPackError.malformedSafetensors("\(url.lastPathComponent): top-level is not a JSON object")
        }
        for (k, v) in dict {
            if k == "__metadata__" {
                if let m = v as? [String: String], metadata.isEmpty {
                    metadata = m
                }
                continue
            }
            guard let entry = v as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int],
                offsets.count == 2
            else {
                throw MLXPackError.malformedSafetensors("\(url.lastPathComponent): tensor \(k) has bad entry")
            }
            srcEntries.append(
                (
                    name: k, dtype: dtype, shape: shape,
                    shardIdx: shardIdx, srcStart: offsets[0], srcEnd: offsets[1]))
        }
    }

    // 2. Sort for deterministic file layout (stable across re-conversions).
    srcEntries.sort { $0.name < $1.name }

    // 3. First pass: build entries with placeholder offsets to size the JSON.
    var outEntries: [MLXPack.Entry] = srcEntries.map {
        MLXPack.Entry(name: $0.name, dtype: $0.dtype, shape: $0.shape, offset: 0, size: $0.srcEnd - $0.srcStart)
    }

    let encoder = JSONEncoder()
    let trialHeader = MLXPack.Header(metadata: metadata, tensors: outEntries)
    let trialJSON = try encoder.encode(trialHeader)
    // Slack covers the growth from "0" to a 12-digit offset for every tensor.
    let slack = 64 + 32 * outEntries.count
    let dataBase = MLXPack.roundUpToPage(20 + trialJSON.count + slack)

    // 4. Second pass: assign real page-aligned offsets.
    var cur = dataBase
    for i in 0..<outEntries.count {
        outEntries[i] = MLXPack.Entry(
            name: outEntries[i].name,
            dtype: outEntries[i].dtype,
            shape: outEntries[i].shape,
            offset: cur,
            size: outEntries[i].size
        )
        cur += MLXPack.roundUpToPage(outEntries[i].size)
    }
    let totalSize = cur

    // 5. Re-encode header with real offsets. Pad to dataBase - 20.
    let finalHeader = MLXPack.Header(metadata: metadata, tensors: outEntries)
    var finalJSON = try encoder.encode(finalHeader)
    let reservedJSONBytes = dataBase - 20
    precondition(finalJSON.count <= reservedJSONBytes,
        "internal: real header (\(finalJSON.count) B) exceeded reserved (\(reservedJSONBytes) B)")
    if finalJSON.count < reservedJSONBytes {
        finalJSON.append(Data(repeating: 0x20, count: reservedJSONBytes - finalJSON.count))
    }

    // 6. Open output, size it.
    let outFD = open(outputFile.path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
    guard outFD >= 0 else {
        throw MLXPackError.openFailed("\(outputFile.path): \(String(cString: strerror(errno)))")
    }
    defer { close(outFD) }
    if ftruncate(outFD, off_t(totalSize)) != 0 {
        throw MLXPackError.truncateFailed("\(outputFile.path): \(String(cString: strerror(errno)))")
    }

    // 7. Write magic + version + header_len + header.
    var prefix = Data()
    prefix.append(contentsOf: MLXPack.magic)
    var version = MLXPack.version.littleEndian
    withUnsafeBytes(of: &version) { prefix.append(contentsOf: $0) }
    var headerLen = UInt64(finalJSON.count).littleEndian
    withUnsafeBytes(of: &headerLen) { prefix.append(contentsOf: $0) }
    prefix.append(finalJSON)

    try prefix.withUnsafeBytes { raw in
        try writeAll(fd: outFD, ptr: raw.baseAddress!, count: raw.count, offset: 0)
    }

    // 8. Copy each tensor's bytes from source to dataBase+entries[i].offset.
    // Reuses one IO buffer; conversion is single-threaded but disk-bound so
    // multi-threading won't dramatically help here.
    let chunkSize = 16 * 1024 * 1024
    let buf = UnsafeMutableRawPointer.allocate(byteCount: chunkSize, alignment: MLXPack.pageSize)
    defer { buf.deallocate() }

    // Open each shard once; keep fds in array.
    var shardFDs: [Int32] = []
    for url in shards {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            for f in shardFDs { close(f) }
            throw MLXPackError.openFailed("\(url.path): \(String(cString: strerror(errno)))")
        }
        shardFDs.append(fd)
    }
    defer { for f in shardFDs { close(f) } }

    for (i, src) in srcEntries.enumerated() {
        let out = outEntries[i]
        let srcAbs = shardDataBases[src.shardIdx] + src.srcStart
        var rem = out.size
        var sOff = srcAbs
        var dOff = out.offset
        while rem > 0 {
            let n = min(rem, chunkSize)
            let r = pread(shardFDs[src.shardIdx], buf, n, off_t(sOff))
            guard r == n else {
                throw MLXPackError.readFailed("tensor \(src.name): pread short (wanted \(n) got \(r))")
            }
            try writeAll(fd: outFD, ptr: buf, count: n, offset: off_t(dOff))
            rem -= n
            sOff += n
            dOff += n
        }
    }
}

// MARK: - Loader (.mlxpack → MLXArrays via mmap, zero-copy)

/// Holds the mmap'd region alive. Each MLXArray loaded from this region
/// captures a strong reference via its finalizer closure; the last release
/// triggers `munmap`.
private final class MMapRegion: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int
    init(base: UnsafeMutableRawPointer, length: Int) {
        self.base = base
        self.length = length
    }
    deinit {
        munmap(base, length)
    }
}

public enum MLXPackLoader {
    /// mmap a `.mlxpack` and return its tensors + metadata. The returned
    /// MLXArrays share ownership of the mmap region — the region is unmapped
    /// only after every array goes out of scope.
    public static func load(url: URL) throws -> (
        weights: [String: MLXArray], metadata: [String: String]
    ) {
        let bench = ProcessInfo.processInfo.environment["BENCH_VERBOSE"] != nil
        var ts = CFAbsoluteTimeGetCurrent()
        func mark(_ label: String) {
            if bench {
                let now = CFAbsoluteTimeGetCurrent()
                FileHandle.standardError.write(Data(
                    "    [mlxpack] \(label): \(String(format: "%.1f", (now - ts) * 1000)) ms\n".utf8))
                ts = now
            }
        }

        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw MLXPackError.openFailed("\(url.path): \(String(cString: strerror(errno)))")
        }
        // Don't `defer close(fd)` — mmap keeps the file open via the page table.
        var sb = stat()
        guard fstat(fd, &sb) == 0 else {
            close(fd)
            throw MLXPackError.openFailed("fstat: \(String(cString: strerror(errno)))")
        }
        let length = Int(sb.st_size)
        mark("open")

        guard let basePtr = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0),
            basePtr != UnsafeMutableRawPointer(bitPattern: -1)
        else {
            close(fd)
            throw MLXPackError.mmapFailed(String(cString: strerror(errno)))
        }
        // We can close fd now; the mmap holds its own reference.
        close(fd)
        mark("mmap")

        let region = MMapRegion(base: basePtr, length: length)

        // Parse magic.
        let magicBytes = UnsafeRawBufferPointer(start: basePtr, count: 8)
        guard magicBytes.elementsEqual(MLXPack.magic) else {
            throw MLXPackError.badMagic
        }

        // Parse version.
        let version = basePtr.advanced(by: 8).load(fromByteOffset: 0, as: UInt32.self).littleEndian
        guard version == MLXPack.version else {
            throw MLXPackError.unsupportedVersion(version)
        }

        // Parse header_len.
        let headerLen = Int(basePtr.advanced(by: 12).load(fromByteOffset: 0, as: UInt64.self).littleEndian)
        let headerBytes = UnsafeRawBufferPointer(start: basePtr.advanced(by: 20), count: headerLen)
        // Strip trailing padding (ASCII spaces) before JSON decoding.
        let headerData = Data(headerBytes).prefix { $0 != 0x20 || true }  // keep all bytes
        // JSONDecoder tolerates trailing whitespace.
        let header = try JSONDecoder().decode(MLXPack.Header.self, from: Data(headerBytes))
        mark("parse header (\(header.tensors.count) tensors)")

        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(header.tensors.count)
        for entry in header.tensors {
            guard let dtype = MLXPack.dtype(forSafeName: entry.dtype) else {
                throw MLXPackError.unknownDType(entry.dtype)
            }
            let ptr = basePtr.advanced(by: entry.offset)
            let array = MLXArray(rawPointer: ptr, entry.shape, dtype: dtype) {
                // Capturing `region` strong-refs it; when all per-array
                // closures are destroyed, region.deinit triggers munmap.
                withExtendedLifetime(region) { }
            }
            weights[entry.name] = array
        }
        mark("construct MLXArrays")

        // Pre-touch every tensor's pages so the disk reads happen here at
        // load time, not lazily on first inference (which would regress TTFT).
        // Multi-threaded for memcpy bandwidth.
        let acc = eagerTouch(region: region, header: header)
        mark("eagerTouch (acc=\(acc))")

        // Drop the local strong reference to `region`; from here on the
        // MLXArrays' captured references are the only thing keeping it alive.
        _ = headerData
        return (weights, header.metadata)
    }

    /// Bulk-read every byte of every tensor in parallel chunks. This serves
    /// two purposes:
    ///   1. Forces every page resident in the unified buffer cache (TTFT
    ///      safety — first GPU op doesn't page-fault).
    ///   2. Uses the same memcpy-bandwidth path the safetensors `pread` uses,
    ///      so we don't regress vs that loader.
    ///
    /// Sparse "one byte per page" touch was slower in practice because the
    /// random access pattern defeats cache prefetch.
    private static func eagerTouch(region: MMapRegion, header: MLXPack.Header) -> UInt64 {
        let base = region.base

        // Determine the data region (skip header / padding).
        let sortedByOffset = header.tensors.sorted { $0.offset < $1.offset }
        guard let first = sortedByOffset.first, let last = sortedByOffset.last else {
            return 0
        }
        let dataStart = first.offset
        let dataEnd = last.offset + MLXPack.roundUpToPage(last.size)
        let dataLen = dataEnd - dataStart

        // 64 MiB chunks: large enough to amortize parallel-task overhead,
        // small enough to balance load across performance cores.
        let chunkSize = 64 * 1024 * 1024
        let numChunks = (dataLen + chunkSize - 1) / chunkSize
        let perTaskAcc = UnsafeMutablePointer<UInt64>.allocate(capacity: numChunks)
        perTaskAcc.initialize(repeating: 0, count: numChunks)
        defer {
            perTaskAcc.deinitialize(count: numChunks)
            perTaskAcc.deallocate()
        }

        // Per-thread scratch buffer for memcpy. Allocated once per thread
        // via thread-local-style indexing. memcpy to scratch is what most
        // efficiently primes the page cache → user-memory path on Darwin.
        let scratchSize = chunkSize
        let scratchPool = (0..<numChunks).map { _ in
            UnsafeMutableRawPointer.allocate(byteCount: scratchSize, alignment: 16384)
        }
        defer { for p in scratchPool { p.deallocate() } }

        DispatchQueue.concurrentPerform(iterations: numChunks) { c in
            let chunkStart = dataStart + c * chunkSize
            let chunkEnd = min(chunkStart + chunkSize, dataEnd)
            let len = chunkEnd - chunkStart
            // memcpy from mmap region to scratch. The scratch result is
            // discarded — we only care that every byte was read, forcing the
            // kernel to bring all pages into the buffer cache. memcpy here
            // is the same bulk-bandwidth path that safetensors pread uses.
            memcpy(scratchPool[c], base.advanced(by: chunkStart), len)
            // Read one byte from scratch to defeat dead-store elimination.
            perTaskAcc[c] = UInt64(scratchPool[c].load(fromByteOffset: 0, as: UInt8.self))
        }

        var total: UInt64 = 0
        for i in 0..<numChunks { total &+= perTaskAcc[i] }
        return total
    }
}

// MARK: - Helpers

private func collectSafetensorsShards(in directory: URL) throws -> [URL] {
    let enumerator = FileManager.default.enumerator(
        at: directory, includingPropertiesForKeys: nil)!
    var shards: [URL] = []
    for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        shards.append(url)
    }
    shards.sort { $0.lastPathComponent < $1.lastPathComponent }
    return shards
}

private func parseSafetensorsHeader(url: URL) throws -> (Any, Int) {
    let fd = open(url.path, O_RDONLY)
    guard fd >= 0 else {
        throw MLXPackError.openFailed("\(url.path): \(String(cString: strerror(errno)))")
    }
    defer { close(fd) }
    var headerLen: UInt64 = 0
    let n = pread(fd, &headerLen, 8, 0)
    guard n == 8 else {
        throw MLXPackError.malformedSafetensors("\(url.lastPathComponent): can't read 8-byte length")
    }
    let headerLenInt = Int(headerLen.littleEndian)
    var bytes = [UInt8](repeating: 0, count: headerLenInt)
    let r = pread(fd, &bytes, headerLenInt, 8)
    guard r == headerLenInt else {
        throw MLXPackError.malformedSafetensors("\(url.lastPathComponent): short header read")
    }
    let obj = try JSONSerialization.jsonObject(with: Data(bytes))
    return (obj, 8 + headerLenInt)
}

private func writeAll(fd: Int32, ptr: UnsafeRawPointer, count: Int, offset: off_t) throws {
    var rem = count
    var off = offset
    var cur = ptr
    while rem > 0 {
        let n = pwrite(fd, cur, rem, off)
        guard n > 0 else {
            throw MLXPackError.writeFailed("pwrite: \(String(cString: strerror(errno)))")
        }
        cur = cur.advanced(by: n)
        off += off_t(n)
        rem -= n
    }
}

#endif  // canImport(Darwin)

