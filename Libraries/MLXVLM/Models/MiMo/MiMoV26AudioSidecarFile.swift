// Copyright © 2026 Eigen Labs.
// Selected MiMo input-codec sidecar only; never a second root-model loader.

import CryptoKit
import Foundation
import MLXLLM

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

public enum MiMoV26AudioSidecarError: Error, Equatable, Sendable {
    case cancelled, unsafeFile, changedObject, shortRead, invalidHeader
    case invalidConfiguration, wrongPayload, invalidBinding, overflow
    case alreadyConsumed, insufficientReservation, unmaterializedTensor, invalidatedOwner
}

func mimoAudioDigest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func mimoAudioAdd(_ left: UInt64, _ right: UInt64) throws -> UInt64 {
    let value = left.addingReportingOverflow(right)
    guard !value.overflow else { throw MiMoV26AudioSidecarError.overflow }
    return value.partialValue
}

/// Audio-local FD owner. Children retain their opened directory ancestors;
/// payload bytes are always read from this exact FD, never reopened by path.
final class MiMoV26AudioSidecarDescriptor {
    let url: URL
    let state: MiMoV26FilesystemObjectState
    private let parent: MiMoV26AudioSidecarDescriptor?
    private let directory: Bool
    private let handle: FileHandle
    var fd: Int32 { handle.fileDescriptor }

    init(url: URL, directory: Bool, parent: MiMoV26AudioSidecarDescriptor? = nil) throws {
        self.url = url
        self.parent = parent
        self.directory = directory
        try parent?.validate()
        var before = stat()
        guard lstat(url.path, &before) == 0 else { throw MiMoV26AudioSidecarError.unsafeFile }
        let expected = try Self.object(before, directory: directory)
        let flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (directory ? O_DIRECTORY : 0)
        let opened =
            parent.map { openat($0.fd, url.lastPathComponent, flags) } ?? open(url.path, flags)
        guard opened >= 0 else { throw MiMoV26AudioSidecarError.unsafeFile }
        var actual = stat()
        do {
            guard fstat(opened, &actual) == 0,
                try Self.object(actual, directory: directory) == expected
            else {
                throw MiMoV26AudioSidecarError.changedObject
            }
            state = expected
            handle = FileHandle(fileDescriptor: opened, closeOnDealloc: true)
        } catch {
            close(opened)
            throw error
        }
        try validate()
    }

    private static func object(_ value: stat, directory: Bool) throws
        -> MiMoV26FilesystemObjectState
    {
        guard mode_t(value.st_mode) & mode_t(S_IFMT) == mode_t(directory ? S_IFDIR : S_IFREG),
            let bytes = Int(exactly: value.st_size), bytes >= 0
        else {
            throw MiMoV26AudioSidecarError.unsafeFile
        }
        #if canImport(Darwin)
            let modified = value.st_mtimespec
            let changed = value.st_ctimespec
        #else
            let modified = value.st_mtim
            let changed = value.st_ctim
        #endif
        return .init(
            device: String(value.st_dev), inode: String(value.st_ino), bytes: bytes,
            modifiedSeconds: Int64(modified.tv_sec), modifiedNanoseconds: Int64(modified.tv_nsec),
            changedSeconds: Int64(changed.tv_sec), changedNanoseconds: Int64(changed.tv_nsec))
    }

    func validate() throws {
        try parent?.validate()
        var held = stat()
        var named = stat()
        guard fstat(fd, &held) == 0, lstat(url.path, &named) == 0,
            try Self.object(held, directory: directory) == state,
            try Self.object(named, directory: directory) == state
        else {
            throw MiMoV26AudioSidecarError.changedObject
        }
    }

    func read(offset: Int, count: Int, maximum: Int, isCancelled: () -> Bool) throws -> Data {
        guard !isCancelled() else { throw MiMoV26AudioSidecarError.cancelled }
        guard !directory, offset >= 0, count >= 0, maximum > 0, count <= maximum,
            offset <= state.bytes, count <= state.bytes - offset
        else {
            throw MiMoV26AudioSidecarError.invalidHeader
        }
        try validate()
        var result = Data(count: count)
        var completed = 0
        try result.withUnsafeMutableBytes { bytes in
            while completed < count {
                guard !isCancelled() else { throw MiMoV26AudioSidecarError.cancelled }
                try validate()
                let amount = min(1 << 20, count - completed)
                let readCount = pread(
                    fd, bytes.baseAddress!.advanced(by: completed), amount,
                    off_t(offset + completed))
                if readCount < 0 && errno == EINTR { continue }
                guard readCount > 0 else { throw MiMoV26AudioSidecarError.shortRead }
                completed += readCount
            }
        }
        try validate()
        return result
    }

    /// Full-file authentication includes prefix, header and all unused payloads.
    /// At most one1-MiB read buffer is live; cancellation does not publish a hash.
    func authenticate(
        expected: String, isCancelled: () -> Bool,
        checkpoint: (Int) throws -> Void
    ) throws -> String {
        var hash = SHA256()
        var offset = 0
        guard !isCancelled() else { throw MiMoV26AudioSidecarError.cancelled }
        try validate()
        try checkpoint(0)
        while offset < state.bytes {
            let count = min(1 << 20, state.bytes - offset)
            let bytes = try read(
                offset: offset, count: count, maximum: 1 << 20, isCancelled: isCancelled)
            hash.update(data: bytes)
            offset += count
            if offset.isMultiple(of: 64 << 20) || offset == state.bytes { try checkpoint(offset) }
        }
        try validate()
        guard !isCancelled() else { throw MiMoV26AudioSidecarError.cancelled }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == expected else { throw MiMoV26AudioSidecarError.wrongPayload }
        return digest
    }
}

/// Reject duplicate object keys before Foundation's dictionary decoding can
/// collapse them. Same bounded scanner policy as MiMoV26FilesystemWeights.
func mimoAudioUniqueKeys(_ data: Data) throws {
    let bytes = [UInt8](data)
    var stack: [Set<String>?] = []
    var i = 0
    while i < bytes.count {
        if bytes[i] == 123 || bytes[i] == 91 {
            stack.append(bytes[i] == 123 ? Set<String>() : nil)
            guard stack.count <= 128 else { throw MiMoV26AudioSidecarError.invalidHeader }
        } else if bytes[i] == 125 || bytes[i] == 93 {
            guard !stack.isEmpty else { throw MiMoV26AudioSidecarError.invalidHeader }
            stack.removeLast()
        } else if bytes[i] == 34 {
            let start = i
            i += 1
            while i < bytes.count && bytes[i] != 34 {
                if bytes[i] == 92 { i += 1 }
                i += 1
            }
            guard i < bytes.count else { throw MiMoV26AudioSidecarError.invalidHeader }
            var next = i + 1
            while next < bytes.count && [9, 10, 13, 32].contains(bytes[next]) { next += 1 }
            if next < bytes.count && bytes[next] == 58 {
                guard let top = stack.indices.last, stack[top] != nil else {
                    throw MiMoV26AudioSidecarError.invalidHeader
                }
                let key = try JSONDecoder().decode(String.self, from: Data(bytes[start ... i]))
                guard stack[top]!.insert(key).inserted else {
                    throw MiMoV26AudioSidecarError.invalidHeader
                }
            }
        }
        i += 1
    }
    guard stack.isEmpty else { throw MiMoV26AudioSidecarError.invalidHeader }
}

struct MiMoV26AudioSidecarHeader: Decodable {
    struct Tensor: Decodable {
        let dtype: MiMoV26AudioTensorType
        let shape: [Int]
        let data_offsets: [Int]
    }
    struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    let tensors: [String: Tensor]
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: Key.self)
        var result: [String: Tensor] = [:]
        for key in container.allKeys {
            if key.stringValue == "__metadata__" {
                _ = try container.decode([String: String].self, forKey: key)
            } else {
                result[key.stringValue] = try container.decode(Tensor.self, forKey: key)
            }
        }
        tensors = result
    }

    static func parse(_ bytes: Data, tensorBytes: Int) throws -> Self {
        guard !bytes.isEmpty, bytes.count <= 1 << 20, tensorBytes > 0 else {
            throw MiMoV26AudioSidecarError.invalidHeader
        }
        try mimoAudioUniqueKeys(bytes)
        let result = try JSONDecoder().decode(Self.self, from: bytes)
        guard !result.tensors.isEmpty, result.tensors.count <= 828 else {
            throw MiMoV26AudioSidecarError.invalidHeader
        }
        var cursor = 0
        for (_, tensor) in result.tensors.sorted(by: {
            ($0.value.data_offsets.first ?? -1) < ($1.value.data_offsets.first ?? -1)
        }) {
            guard !tensor.shape.isEmpty, tensor.shape.count <= 8,
                tensor.shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }),
                tensor.data_offsets.count == 2, tensor.data_offsets[0] == cursor
            else {
                throw MiMoV26AudioSidecarError.invalidHeader
            }
            var count: UInt64 = tensor.dtype == .bfloat16 ? 2 : 4
            for dimension in tensor.shape {
                let next = count.multipliedReportingOverflow(by: UInt64(dimension))
                guard !next.overflow else { throw MiMoV26AudioSidecarError.overflow }
                count = next.partialValue
            }
            guard let size = Int(exactly: count), size <= tensorBytes - cursor,
                tensor.data_offsets[1] == cursor + size
            else { throw MiMoV26AudioSidecarError.invalidHeader }
            cursor += size
        }
        guard cursor == tensorBytes else { throw MiMoV26AudioSidecarError.invalidHeader }
        return result
    }
}

final class MiMoV26AudioSidecarSource {
    static let fileBytes = 1_872_618_384
    static let configurationSHA256 =
        "e0702adae37947e0c980c38bae58ffa0d48bd492d523afe814aa4d73f008c7d1"
    let root: URL
    let configurationFile, payload: MiMoV26AudioSidecarDescriptor
    let configuration: MiMoV26AudioInputConfiguration
    let plan: MiMoV26AudioSidecarPlan
    let header: MiMoV26AudioSidecarHeader
    let headerSHA256: String
    let headerBytes: Int
    var payloadOffset: Int { 8 + headerBytes }

    init(root: URL, mainConfiguration: MiMoV26Configuration, isCancelled: () -> Bool) throws {
        let canonical = root.resolvingSymlinksInPath().standardizedFileURL
        guard canonical.isFileURL else { throw MiMoV26AudioSidecarError.unsafeFile }
        let parent = try MiMoV26AudioSidecarDescriptor(url: canonical, directory: true)
        let directory = try MiMoV26AudioSidecarDescriptor(
            url: canonical.appendingPathComponent("audio_tokenizer"), directory: true,
            parent: parent)
        let configFile = try MiMoV26AudioSidecarDescriptor(
            url: directory.url.appendingPathComponent("config.json"), directory: false,
            parent: directory)
        let payloadFile = try MiMoV26AudioSidecarDescriptor(
            url: directory.url.appendingPathComponent("model.safetensors"), directory: false,
            parent: directory)
        guard configFile.state.bytes <= 1 << 20, payloadFile.state.bytes == Self.fileBytes else {
            throw MiMoV26AudioSidecarError.invalidHeader
        }
        let data = try configFile.read(
            offset: 0, count: configFile.state.bytes,
            maximum: 1 << 20, isCancelled: isCancelled)
        try mimoAudioUniqueKeys(data)
        guard mimoAudioDigest(data) == Self.configurationSHA256 else {
            throw MiMoV26AudioSidecarError.invalidConfiguration
        }
        let configuration = try MiMoV26AudioInputConfiguration(
            sidecarJSON: data, mainConfiguration: mainConfiguration)
        let prefix = try payloadFile.read(offset: 0, count: 8, maximum: 8, isCancelled: isCancelled)
        let rawLength = prefix.enumerated().reduce(UInt64(0)) {
            $0 | UInt64($1.element) << (8 * $1.offset)
        }
        guard let length = Int(exactly: rawLength), length > 0, length <= 1 << 20,
            length < payloadFile.state.bytes - 8
        else { throw MiMoV26AudioSidecarError.invalidHeader }
        let headerData = try payloadFile.read(
            offset: 8, count: length, maximum: 1 << 20, isCancelled: isCancelled)
        let header = try MiMoV26AudioSidecarHeader.parse(
            headerData, tensorBytes: payloadFile.state.bytes - 8 - length)
        var descriptors: [String: MiMoV26AudioTensorDescriptor] = [:]
        for (name, tensor) in header.tensors {
            descriptors[name] = .init(
                shape: tensor.shape, dtype: tensor.dtype,
                byteCount: tensor.data_offsets[1] - tensor.data_offsets[0])
        }
        let plan = try MiMoV26AudioTokenizerWeights.preflight(
            descriptors: descriptors,
            configuration: configuration,
            sourcePayloadSHA256: MiMoV26AudioTokenizerWeights.selectedPayloadSHA256)
        self.root = canonical
        self.configurationFile = configFile
        self.payload = payloadFile
        self.configuration = configuration
        self.header = header
        self.plan = plan
        headerBytes = length
        headerSHA256 = mimoAudioDigest(headerData)
        try validate()
    }

    func validate() throws {
        try configurationFile.validate()
        try payload.validate()
    }
}
