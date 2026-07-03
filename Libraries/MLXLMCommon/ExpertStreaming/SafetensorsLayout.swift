// Safetensors header parsing for MoE expert SSD streaming.
//
// This file is deliberately MLX-free (Foundation only) so the byte-range
// math is unit-testable without a Metal device and reusable by any future
// streaming consumer that wants raw offsets rather than materialized
// MLXArrays (e.g. a prefetcher).
//
// Format (https://github.com/huggingface/safetensors, confirmed against
// upstream mlx/io/safetensors.cpp):
//   [8 bytes little-endian u64: header length N]
//   [N bytes: JSON header]
//   [raw tensor bytes, contiguous]
// Header JSON: {"tensor.name": {"dtype": "...", "shape": [...],
//   "data_offsets": [start, end]}, ..., "__metadata__": {...}}
// `data_offsets` are relative to the START OF THE DATA SECTION (the first
// byte after the 8-byte length + N-byte JSON), NOT the start of the file.

import Foundation

/// Restricted to the dtypes MoE expert streaming actually needs to read:
/// packed quantized weights (`uint32`), MX-format scale bytes (`uint8`), and
/// unquantized/affine side tables (`bfloat16`/`float16`/`float32`). Extend as
/// needed if a future streamed format uses others.
public enum SafetensorsDType: String, Sendable {
    case uint8 = "U8"
    case uint32 = "U32"
    case bfloat16 = "BF16"
    case float16 = "F16"
    case float32 = "F32"

    /// Bytes per element, used for shape/offset validation.
    public var byteWidth: Int {
        switch self {
        case .uint8: return 1
        case .uint32: return 4
        case .bfloat16, .float16: return 2
        case .float32: return 4
        }
    }
}

/// Where one tensor's bytes live on disk: which file, what shape/dtype, and
/// the ABSOLUTE byte range within that file (header already accounted for —
/// this is directly usable with `pread`/`FileHandle.seek`, unlike the raw
/// `data_offsets` from the JSON header).
public struct SafetensorsTensorLocation: Sendable {
    public let fileURL: URL
    public let dtype: SafetensorsDType
    public let shape: [Int]
    public let byteRange: Range<Int>

    public init(fileURL: URL, dtype: SafetensorsDType, shape: [Int], byteRange: Range<Int>) {
        self.fileURL = fileURL
        self.dtype = dtype
        self.shape = shape
        self.byteRange = byteRange
    }

    public var elementCount: Int { shape.reduce(1, *) }
}

public enum SafetensorsLayoutError: Error, Equatable {
    case fileTooShort(URL)
    case invalidHeader(URL)
    case unsupportedDType(String, key: String)
    case sizeMismatch(key: String, expected: Int, got: Int)
}

/// Flat key → byte-range index over one (possibly sharded) safetensors
/// checkpoint directory. Parsing only reads shard HEADERS, never tensor
/// bytes — building the layout for a 141 GB / 33-shard checkpoint costs
/// ~33 small JSON parses, not a multi-minute weight load.
public struct SafetensorsLayout: Sendable {
    public let tensors: [String: SafetensorsTensorLocation]

    public init(tensors: [String: SafetensorsTensorLocation]) {
        self.tensors = tensors
    }

    public subscript(key: String) -> SafetensorsTensorLocation? { tensors[key] }

    /// Build a layout for a checkpoint directory. Prefers
    /// `model.safetensors.index.json` (maps tensor name → shard filename) so
    /// we know exactly which shards to open; falls back to scanning every
    /// `*.safetensors` file directly for single-shard checkpoints or ones
    /// missing an index.
    public static func load(modelDirectory: URL) throws -> SafetensorsLayout {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        let shardFiles: [String]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let data = try Data(contentsOf: indexURL)
            let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
            shardFiles = Array(Set(index.weightMap.values)).sorted()
        } else {
            let entries = try FileManager.default.contentsOfDirectory(atPath: modelDirectory.path)
            shardFiles = entries.filter { $0.hasSuffix(".safetensors") }.sorted()
        }

        var tensors: [String: SafetensorsTensorLocation] = [:]
        for name in shardFiles {
            let url = modelDirectory.appendingPathComponent(name)
            for (key, location) in try parseHeader(url: url) {
                tensors[key] = location
            }
        }
        return SafetensorsLayout(tensors: tensors)
    }

    /// Parse a single shard's header into key → absolute byte range.
    /// Exposed (not `private`) so it's independently unit-testable against
    /// a single hand-built or `MLX.save`-produced safetensors file.
    static func parseHeader(url: URL) throws -> [String: SafetensorsTensorLocation] {
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            throw SafetensorsLayoutError.fileTooShort(url)
        }
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw SafetensorsLayoutError.fileTooShort(url)
        }
        let headerLength = Int(lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard headerLength > 0,
            let headerData = try handle.read(upToCount: headerLength),
            headerData.count == headerLength
        else {
            throw SafetensorsLayoutError.fileTooShort(url)
        }

        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            throw SafetensorsLayoutError.invalidHeader(url)
        }

        // Data section starts right after the 8-byte length + JSON header.
        let dataSectionStart = 8 + headerLength
        var result: [String: SafetensorsTensorLocation] = [:]
        for (key, value) in json {
            if key == "__metadata__" { continue }
            guard let entry = value as? [String: Any],
                let dtypeString = entry["dtype"] as? String,
                let shapeAny = entry["shape"] as? [Any],
                let offsetsAny = entry["data_offsets"] as? [Any],
                offsetsAny.count == 2
            else {
                throw SafetensorsLayoutError.invalidHeader(url)
            }
            guard let dtype = SafetensorsDType(rawValue: dtypeString) else {
                throw SafetensorsLayoutError.unsupportedDType(dtypeString, key: key)
            }
            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count,
                let start = (offsetsAny[0] as? NSNumber)?.intValue,
                let end = (offsetsAny[1] as? NSNumber)?.intValue,
                start >= 0, end >= start
            else {
                throw SafetensorsLayoutError.invalidHeader(url)
            }

            let expectedBytes = shape.reduce(1, *) * dtype.byteWidth
            guard expectedBytes == end - start else {
                throw SafetensorsLayoutError.sizeMismatch(
                    key: key, expected: expectedBytes, got: end - start)
            }

            result[key] = SafetensorsTensorLocation(
                fileURL: url, dtype: dtype, shape: shape,
                byteRange: (dataSectionStart + start) ..< (dataSectionStart + end))
        }
        return result
    }
}

private struct SafetensorsIndex: Codable {
    let weightMap: [String: String]
    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
