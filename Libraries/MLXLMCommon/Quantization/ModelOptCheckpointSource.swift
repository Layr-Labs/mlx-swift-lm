import CryptoKit
import Foundation
import MLX

/// One tensor bound to its verified file. The source dtype is never guessed.
public struct ModelOptTensorSource {
    public let file: ModelOptTensorFile
    public let name: String
    public let layout: ModelOptProjectionLayout.Tensor
    public init(file: ModelOptTensorFile, name: String) throws {
        guard let entry = file.entries[name] else { throw ModelOptTensorFile.Failure.missingTensor }
        self.file = file; self.name = name; layout = entry.layout
    }
    public func scalar() throws -> Float {
        guard layout.dtype == "F32", layout.shape == [] || layout.shape == [1] else {
            throw ModelOptProjectionLoader.Failure.invalidScalar
        }
        let data = try file.read(name)
        let bits = data.enumerated().reduce(UInt32(0)) { $0 | UInt32($1.element) << (8 * $1.offset) }
        let value = Float(bitPattern: bits)
        guard value.isFinite, value > 0 else { throw ModelOptProjectionLoader.Failure.invalidScalar }
        return value
    }
    /// Caller must reserve full residency plus concatenation/copy workspace.
    /// Host reads are <=16MiB; final dense storage is the original dtype only.
    public func array(maximumBytes: Int, construction: NativeConstructionScope? = nil) throws -> MLXArray {
        guard maximumBytes > 0, layout.byteCount <= maximumBytes,
            layout.shape.allSatisfy({ $0 <= Int(Int32.max) }) else { throw ModelOptProjectionLoader.Failure.payloadLimit }
        let dtype: DType
        switch layout.dtype {
        case "F32": dtype = .float32
        case "F16": dtype = .float16
        case "BF16": dtype = .bfloat16
        case "U8", "F8_E4M3": dtype = .uint8
        default: throw ModelOptTensorFile.Failure.unsupportedDType
        }
        var pieces: [MLXArray] = []
        var offset = 0
        while offset < layout.byteCount {
            try Task.checkCancellation()
            let end = offset + min(16 << 20, layout.byteCount - offset)
            let data = try file.read(name, bytes: offset..<end)
            let piece = MLXArray(data, [data.count / dtype.size], dtype: dtype)
            try construction?.retain(piece)
            pieces.append(piece)
            offset = end
        }
        let value = (pieces.count == 1 ? pieces[0] : concatenated(pieces)).reshaped(layout.shape)
        try construction?.retain(value)
        try MLX.withError { errors in
            eval(value)
            try errors.check()
        }
        return value
    }
}

/// Binds every index key to exactly one independently verified shard. Header
/// extras, index omissions and wrong-shard bindings fail before payload reads.
public struct ModelOptCheckpointSource {
    public enum Failure: Error { case identity, inventory }
    public let inventory: NemotronSuperCheckpointInventory
    public let tensors: [String: ModelOptTensorSource]
    public let indexSHA256: String
    public let shardSHA256: [String: String]
    private let groups: [String: [String: ModelOptTensorSource]]
    public init(indexData: Data, indexSHA256: String, files: [String: ModelOptTensorFile]) throws {
        guard SHA256.hash(data: indexData).map({ String(format: "%02x", $0) }).joined() == indexSHA256 else { throw Failure.identity }
        let inventory = try NemotronSuperCheckpointInventory(indexData: indexData)
        guard Set(files.keys) == Set(inventory.shards) else { throw Failure.inventory }
        var expected: [String: Set<String>] = [:]
        var tensors: [String: ModelOptTensorSource] = [:]
        for entry in inventory.entries {
            expected[entry.shard, default: []].insert(entry.source)
            guard let file = files[entry.shard] else { throw Failure.inventory }
            tensors[entry.source] = try ModelOptTensorSource(file: file, name: entry.source)
        }
        for (name, file) in files where Set(file.entries.keys) != expected[name] { throw Failure.inventory }
        var groups: [String: [String: ModelOptTensorSource]] = [:]
        for (name, tensor) in tensors {
            guard let separator = name.lastIndex(of: ".") else { throw Failure.inventory }
            groups[String(name[..<separator]), default: [:]][String(name[name.index(after: separator)...])] = tensor
        }
        self.inventory = inventory; self.tensors = tensors; self.indexSHA256 = indexSHA256; self.groups = groups; self.shardSHA256 = files.mapValues(\.verifiedSHA256)
    }
    public func group(_ prefix: String) -> [String: ModelOptTensorSource] {
        groups[prefix] ?? [:]
    }
}
