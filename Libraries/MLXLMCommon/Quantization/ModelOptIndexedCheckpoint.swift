// Copyright © 2026 Eigen Labs.

import CryptoKit
import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// A sharded safetensors checkpoint bound to its index: the index bytes match
/// a pinned SHA-256, every index key lives in exactly the shard the index
/// names, and no shard holds a tensor the index omits. Shards arrive as
/// `ModelOptTensorFile`s, each already verified against its own pinned hash.
/// Model-agnostic: the caller decides what each key means. No payload is read.
public struct ModelOptIndexedCheckpoint {
    public enum Failure: Error, Equatable {
        case identity
        case invalidIndex
        case unsafeShard(String)
        case shardSet
        case inventory(String)
        /// A small checkpoint file is a symlink, not a regular file, larger
        /// than its bound or unreadable.
        case unsafeFile(String)
    }

    /// The index's bound, checked before it is read.
    public static let maximumIndexBytes = 64 << 20

    /// The bytes of a small checkpoint file (an index or a JSON record),
    /// read as shards are opened: a regular file, not followed through a
    /// symlink, its size checked against `maximumBytes` before it is read.
    public static func readRegularFile(at url: URL, maximumBytes: Int) throws -> Data {
        let name = url.lastPathComponent
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard maximumBytes > 0, values?.isRegularFile == true, values?.isSymbolicLink != true
        else { throw Failure.unsafeFile(name) }
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.unsafeFile(name) }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? file.close() }
        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
            status.st_size >= 0, status.st_size <= maximumBytes
        else { throw Failure.unsafeFile(name) }
        let data: Data
        do { data = try file.read(upToCount: maximumBytes + 1) ?? Data() } catch {
            throw Failure.unsafeFile(name)
        }
        guard data.count == Int(status.st_size) else { throw Failure.unsafeFile(name) }
        return data
    }

    public let tensors: [String: ModelOptTensorSource]
    public let indexSHA256: String
    public let shardSHA256: [String: String]
    private let groups: [String: [String: ModelOptTensorSource]]

    /// The shard file names an index refers to, after its hash is checked.
    /// Hosts use it to know which files to open before binding.
    public static func shardNames(indexData: Data, indexSHA256: String) throws -> [String] {
        Set(try weightMap(indexData: indexData, indexSHA256: indexSHA256).values).sorted()
    }

    static func weightMap(indexData: Data, indexSHA256: String) throws -> [String: String] {
        guard
            SHA256.hash(data: indexData).map({ String(format: "%02x", $0) }).joined() == indexSHA256
        else { throw Failure.identity }
        guard !indexData.isEmpty, indexData.count <= maximumIndexBytes,
            let root = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let weights = root["weight_map"] as? [String: String],
            !weights.isEmpty, weights.count <= 250_000
        else { throw Failure.invalidIndex }
        for (key, shard) in weights {
            guard !key.isEmpty else { throw Failure.invalidIndex }
            guard shard.hasSuffix(".safetensors"), !shard.hasPrefix("."),
                !shard.contains("/"), !shard.contains("\\"), !shard.contains(":"),
                !shard.unicodeScalars.contains(where: { $0.value < 32 }), shard.count <= 255
            else { throw Failure.unsafeShard(shard) }
        }
        return weights
    }

    public init(indexData: Data, indexSHA256: String, files: [String: ModelOptTensorFile]) throws {
        let weights = try Self.weightMap(indexData: indexData, indexSHA256: indexSHA256)
        guard Set(files.keys) == Set(weights.values) else { throw Failure.shardSet }
        var expected: [String: Set<String>] = [:]
        var tensors: [String: ModelOptTensorSource] = [:]
        for (key, shard) in weights {
            expected[shard, default: []].insert(key)
            guard let file = files[shard], file.entries[key] != nil else {
                throw Failure.inventory(key)
            }
            tensors[key] = try ModelOptTensorSource(file: file, name: key)
        }
        for (name, file) in files {
            let extra = Set(file.entries.keys).subtracting(expected[name] ?? [])
            if let key = extra.sorted().first { throw Failure.inventory(key) }
        }
        var groups: [String: [String: ModelOptTensorSource]] = [:]
        for (name, tensor) in tensors {
            guard let separator = name.lastIndex(of: ".") else { continue }
            groups[String(name[..<separator]), default: [:]][
                String(name[name.index(after: separator)...])] = tensor
        }
        self.tensors = tensors
        self.indexSHA256 = indexSHA256
        self.shardSHA256 = files.mapValues(\.verifiedSHA256)
        self.groups = groups
    }

    /// The tensors named `prefix.<leaf>`, keyed by leaf.
    public func group(_ prefix: String) -> [String: ModelOptTensorSource] {
        groups[prefix] ?? [:]
    }
}
