import CryptoKit
import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// A verified, bounded reader for the safetensors subset used by ModelOpt.
/// Retains one file descriptor; callers must keep checkpoint files immutable.
/// No mmap, whole-shard allocation, dtype coercion or implicit dense expansion.
public final class ModelOptTensorFile {
    public enum Failure: Error {
        case invalidFile, checksum, headerLimit, malformed, unsupportedDType, payloadLimit,
            missingTensor
    }
    public struct Entry {
        public let layout: ModelOptProjectionLayout.Tensor
        public let offset: UInt64
    }
    public let entries: [String: Entry]
    public let verifiedSHA256: String
    private let handle: FileHandle
    private let verifiedSize: UInt64
    private let verifiedModification: Int64
    private let verifiedNanoseconds: Int64
    private let lock = NSLock()
    private let payloadLimit: Int

    /// SHA256 is mandatory and must come from an independently pinned manifest.
    /// Hashing uses 1 MiB chunks. Header and individual payload limits are hard
    /// capped at 16 MiB and 64 MiB respectively, even if a larger limit is passed.
    public init(
        url: URL, sha256: String, headerLimit: Int = 16 << 20,
        payloadLimit: Int = 64 << 20
    ) throws {
        guard headerLimit > 0, headerLimit <= 16 << 20,
            payloadLimit > 0, payloadLimit <= 64 << 20,
            sha256.count == 64, sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase })
        else {
            throw Failure.invalidFile
        }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw Failure.invalidFile
        }
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.invalidFile }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        var succeeded = false
        defer { if !succeeded { try? file.close() } }
        let stamp = try Self.stamp(file)
        let length = try file.seekToEnd()
        guard length >= 10 else { throw Failure.malformed }
        try file.seek(toOffset: 0)
        var hash = SHA256()
        while let chunk = try file.read(upToCount: 1 << 20), !chunk.isEmpty {
            try Task.checkCancellation()
            hash.update(data: chunk)
        }
        guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else {
            throw Failure.checksum
        }
        try file.seek(toOffset: 0)
        guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
            throw Failure.malformed
        }
        let headerBytes = prefix.enumerated().reduce(UInt64(0)) {
            $0 | UInt64($1.element) << (8 * $1.offset)
        }
        guard headerBytes <= UInt64(headerLimit) else { throw Failure.headerLimit }
        guard headerBytes <= length - 8,
            let data = try file.read(upToCount: Int(headerBytes)), data.count == Int(headerBytes)
        else { throw Failure.malformed }
        struct Tensor: Decodable {
            let dtype: String
            let shape: [Int]
            let data_offsets: [UInt64]
        }
        // Decode typed integer metadata to reject booleans, fractions and overflow.
        let object = try JSONSerialization.jsonObject(with: data)
        guard let dictionary = object as? [String: Any], !dictionary.isEmpty else {
            throw Failure.malformed
        }
        var parsed: [String: Entry] = [:]
        var ranges: [(UInt64, UInt64)] = []
        let start = 8 + headerBytes
        for (name, value) in dictionary where name != "__metadata__" {
            guard !name.isEmpty else { throw Failure.malformed }
            let tensor = try JSONDecoder().decode(
                Tensor.self, from: JSONSerialization.data(withJSONObject: value))
            let width: Int
            switch tensor.dtype {
            case "U8", "F8_E4M3": width = 1
            case "BF16", "F16": width = 2
            case "F32": width = 4
            case "I64": width = 8  // retained summary-index metadata; never coerced to float
            default: throw Failure.unsupportedDType
            }
            var bytes = width
            for dimension in tensor.shape {
                let (product, overflow) = bytes.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !overflow else { throw Failure.malformed }
                bytes = product
            }
            guard tensor.data_offsets.count == 2 else { throw Failure.malformed }
            let low = tensor.data_offsets[0]
            let high = tensor.data_offsets[1]
            guard high >= low, high - low == UInt64(bytes), high <= length - start else {
                throw Failure.malformed
            }
            parsed[name] = Entry(
                layout: .init(dtype: tensor.dtype, shape: tensor.shape, byteCount: bytes),
                offset: start + low)
            ranges.append((low, high))
        }
        // Require a complete, contiguous payload with no overlaps or hidden tail.
        var end: UInt64 = 0
        for range in ranges.sorted(by: { $0.0 < $1.0 }) {
            guard range.0 == end else { throw Failure.malformed }
            end = range.1
        }
        guard !parsed.isEmpty, end == length - start else { throw Failure.malformed }
        guard try Self.stamp(file) == stamp else { throw Failure.invalidFile }
        verifiedSHA256 = sha256
        entries = parsed
        handle = file
        self.payloadLimit = payloadLimit
        verifiedSize = stamp.0
        verifiedModification = stamp.1
        verifiedNanoseconds = stamp.2
        succeeded = true
    }

    private static func stamp(_ file: FileHandle) throws -> (UInt64, Int64, Int64) {
        var value = stat()
        guard fstat(file.fileDescriptor, &value) == 0,
            value.st_mode & S_IFMT == S_IFREG, value.st_size >= 0
        else { throw Failure.invalidFile }
        #if canImport(Darwin)
            return (
                UInt64(value.st_size), Int64(value.st_mtimespec.tv_sec),
                Int64(value.st_mtimespec.tv_nsec)
            )
        #else
            return (
                UInt64(value.st_size), Int64(value.st_mtim.tv_sec), Int64(value.st_mtim.tv_nsec)
            )
        #endif
    }

    deinit { try? handle.close() }

    /// Returns at most the configured payload cap. Large tensors must use an
    /// explicit higher-level row reader; this API never silently exceeds its cap.
    public func read(_ name: String) throws -> Data {
        guard let entry = entries[name] else { throw Failure.missingTensor }
        return try read(name, bytes: 0 ..< entry.layout.byteCount)
    }

    /// Bounded byte-range access for an admitted large tensor. Each read keeps
    /// the same descriptor identity and hard payload cap as whole-tensor reads.
    public func read(_ name: String, bytes: Range<Int>) throws -> Data {
        guard let entry = entries[name] else { throw Failure.missingTensor }
        guard bytes.lowerBound >= 0, bytes.upperBound <= entry.layout.byteCount,
            !bytes.isEmpty, bytes.count <= payloadLimit
        else { throw Failure.payloadLimit }
        lock.lock()
        defer { lock.unlock() }
        try Task.checkCancellation()
        guard try Self.stamp(handle) == (verifiedSize, verifiedModification, verifiedNanoseconds)
        else { throw Failure.invalidFile }
        try handle.seek(toOffset: entry.offset + UInt64(bytes.lowerBound))
        guard let data = try handle.read(upToCount: bytes.count), data.count == bytes.count else {
            throw Failure.malformed
        }
        guard try Self.stamp(handle) == (verifiedSize, verifiedModification, verifiedNanoseconds)
        else { throw Failure.invalidFile }
        return data
    }
}
