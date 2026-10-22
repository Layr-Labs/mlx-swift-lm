import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

/// Frozen, synthetic outputs produced by the real pinned Transformers CPU model.
/// No downloads, native-generated expectations, or full-model quality claims.
struct GLM5NextReferenceFixture {
    static let environmentKey = "GLM53_REFERENCE_FIXTURE_DIR"
    static var enabled: Bool { ProcessInfo.processInfo.environment[environmentKey] != nil }
    static let sourceRevision = "f0705f991a71f05dad9569336b181119f5546f93"
    static let modelRevision = "91b69215804f2d2036a8867f6ce44848e278dcda"
    private static let manifestHash =
        "028a6a2f2aa4beb30a2a846913ed139e278b65f5006aa77099bd4baaa8671f0d"

    struct Failure: Error, CustomStringConvertible {
        let description: String
        init(_ message: String) { description = "GLM53 official reference: " + message }
    }
    struct Header {
        let shape: [Int]
        let dtype: String
        let range: Range<Int>
    }
    struct Archive {
        let data: Data
        let headers: [String: Header]
        let payloadStart: Int
        init(_ data: Data) throws {
            guard data.count >= 10 else { throw Failure("truncated safetensors") }
            let count64 = data.prefix(8).enumerated().reduce(UInt64(0)) {
                $0 | (UInt64($1.element) << (8 * $1.offset))
            }
            guard count64 <= 4 * 1024 * 1024, count64 <= UInt64(data.count - 8) else {
                throw Failure("invalid safetensors header length")
            }
            let start = 8 + Int(count64)
            let object = try GLM5NextReferenceFixture.object(data.subdata(in: 8 ..< start))
            var headers: [String: Header] = [:]
            var spans: [Range<Int>] = []
            for (name, value) in object where name != "__metadata__" {
                guard let info = value as? [String: Any], let shape = info["shape"] as? [Int],
                    let dtype = info["dtype"] as? String,
                    let offsets = info["data_offsets"] as? [Int], offsets.count == 2,
                    offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= data.count - start
                else { throw Failure("malformed tensor header \(name)") }
                let width: Int
                switch dtype {
                case "F32", "I32": width = 4
                case "I64": width = 8
                case "U8", "BOOL": width = 1
                default: throw Failure("unsupported oracle dtype \(dtype)")
                }
                var bytes = width
                for dimension in shape {
                    let next = bytes.multipliedReportingOverflow(by: dimension)
                    guard dimension >= 0, !next.overflow else {
                        throw Failure("tensor size overflow")
                    }
                    bytes = next.partialValue
                }
                guard offsets[1] - offsets[0] == bytes else {
                    throw Failure("tensor byte count \(name)")
                }
                let range = offsets[0] ..< offsets[1]
                headers[name] = Header(shape: shape, dtype: dtype, range: range)
                spans.append(range)
            }
            var end = 0
            for span in spans.sorted(by: { $0.lowerBound < $1.lowerBound }) {
                guard span.lowerBound == end else {
                    throw Failure("overlapping/gapped tensor payload")
                }
                end = span.upperBound
            }
            guard end == data.count - start else { throw Failure("unaccounted safetensors bytes") }
            self.data = data
            self.headers = headers
            payloadStart = start
        }
        func bytes(_ name: String) throws -> Data {
            guard let header = headers[name] else { throw Failure("missing tensor \(name)") }
            return data.subdata(
                in: (payloadStart + header.range.lowerBound)
                    ..< (payloadStart + header.range.upperBound))
        }
    }

    let configuration: GLM5NextConfiguration
    let inventory: GLM5NextWeightInventory
    let weights: [String: MLXArray]
    let expected: [String: MLXArray]
    private let cases: [String: Any]
    private let configurationData: Data
    private let nativeArchive: Archive

    init() throws {
        guard let raw = ProcessInfo.processInfo.environment[Self.environmentKey], !raw.isEmpty
        else {
            throw Failure("\(Self.environmentKey) was provided empty or was not provided")
        }
        let root = URL(fileURLWithPath: raw, isDirectory: true).standardizedFileURL
        let rootValues = try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard rootValues.isDirectory == true, rootValues.isSymbolicLink != true else {
            throw Failure("fixture root must be a real directory")
        }
        func read(_ name: String, maximum: Int = 16 * 1024 * 1024) throws -> Data {
            let url = root.appendingPathComponent(name)
            let values = try url.resourceValues(forKeys: [
                .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
            ])
            guard values.isRegularFile == true, values.isSymbolicLink != true,
                let size = values.fileSize, size > 0, size <= maximum
            else {
                throw Failure("missing/nonregular/oversized \(name)")
            }
            let data = try Data(contentsOf: url)
            guard data.count == size else { throw Failure("fixture changed while reading \(name)") }
            return data
        }
        let manifestData = try read("manifest.json", maximum: 2 * 1024 * 1024)
        guard Self.hash(manifestData) == Self.manifestHash else {
            throw Failure("manifest is not the frozen canonical run-2 manifest")
        }
        let manifest = try Self.object(manifestData)
        guard manifest["complete"] as? Bool == true, manifest["synthetic"] as? Bool == true,
            manifest["model_repository"] as? String == "zai-org/GLM-5.3-Flash",
            manifest["source_revision"] as? String == Self.sourceRevision,
            manifest["model_revision"] as? String == Self.modelRevision,
            manifest["device"] as? String == "cpu", manifest["dtype"] as? String == "float32",
            manifest["attention"] as? String == "eager",
            manifest["num_nextn_predict_layers"] as? Int == 0,
            let sourceHashes = manifest["source_sha256"] as? [String: String],
            sourceHashes.count == 8,
            sourceHashes["modeling_glm5_next.py"]
                == "78e23c624010751f1578254145f313db2f3266579a0f9a396a3ff6cb5779f43d"
        else { throw Failure("wrong oracle scope/source/backend") }

        func metadata(_ name: String, hashField: String) throws -> Data {
            let data = try read(name)
            guard let digest = manifest[hashField] as? String, Self.hash(data) == digest else {
                throw Failure("metadata hash \(name)")
            }
            return data
        }
        let environment = try Self.object(
            metadata("environment.json", hashField: "environment_sha256"))
        guard environment["source_sha256"] as? [String: String] == sourceHashes,
            environment["device"] as? String == "cpu", environment["hub_kernels"] as? Bool == false,
            environment["threads"] as? Int == 1,
            let origin = environment["transformers_direct_url"] as? [String: Any],
            origin["url"] as? String == "https://github.com/huggingface/transformers.git",
            let vcs = origin["vcs_info"] as? [String: Any],
            vcs["commit_id"] as? String == Self.sourceRevision
        else { throw Failure("environment/source provenance") }
        _ = try metadata("reference-manifest.json", hashField: "reference_manifest_sha256")
        _ = try metadata(
            "transformers_configuration.json", hashField: "transformers_configuration_sha256")
        guard
            Self.hash(try read("configuration-transforms.json"))
                == "1dbd298ddbe2f7c646530829512838677121c2711bd4dc06e985917fe57a4569"
        else {
            throw Failure("configuration transform receipt")
        }
        let nativeConfig = try metadata(
            "native_configuration.json", hashField: "native_configuration_sha256")
        let caseIndex = try Self.object(metadata("case-index.json", hashField: "case_index_sha256"))
        guard caseIndex["weights"] as? String == "native-weights.safetensors",
            caseIndex["expected"] as? String == "expected.safetensors",
            caseIndex["configuration"] as? String == "native_configuration.json",
            let cases = caseIndex["cases"] as? [String: Any]
        else { throw Failure("case index") }
        let mapping = try Self.object(
            metadata("weight-mapping.json", hashField: "weight_mapping_sha256"))
        guard let artifacts = manifest["artifacts"] as? [String: [String: Any]] else {
            throw Failure("artifact manifest")
        }
        func archive(_ name: String) throws -> Archive {
            let data = try read(name)
            guard let entry = artifacts[name], entry["bytes"] as? Int == data.count,
                entry["sha256"] as? String == Self.hash(data)
            else { throw Failure("artifact hash \(name)") }
            let archive = try Archive(data)
            guard entry["tensor_count"] as? Int == archive.headers.count,
                let records = entry["tensors"] as? [String: [String: Any]],
                Set(records.keys) == Set(archive.headers.keys)
            else { throw Failure("tensor inventory \(name)") }
            for (key, header) in archive.headers {
                let torchDtypes = [
                    "F32": "torch.float32", "I32": "torch.int32", "I64": "torch.int64",
                    "U8": "torch.uint8", "BOOL": "torch.bool",
                ]
                guard let record = records[key], record["shape"] as? [Int] == header.shape,
                    [header.dtype, torchDtypes[header.dtype]].compactMap({ $0 }).contains(
                        record["dtype"] as? String ?? ""),
                    record["elements"] as? Int == header.shape.reduce(1, *),
                    record["sha256"] as? String == Self.hash(try archive.bytes(key))
                else {
                    throw Failure("tensor digest/shape \(name):\(key)")
                }
            }
            return archive
        }
        let source = try archive("weights.safetensors")
        let native = try archive("native-weights.safetensors")
        let outputs = try archive("expected.safetensors")
        try Self.validateMapping(mapping, source: source, native: native)
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: nativeConfig)
        guard config.textConfig.numNextNPredictLayers == 0,
            config.textConfig.hiddenLayers == 4, config.visionConfig.depth == 2
        else {
            throw Failure("unexpected synthetic model configuration")
        }
        let descriptors = native.headers.map { name, header in
            GLM5NextWeightInventory.Tensor(
                name: name, shape: header.shape, dtype: header.dtype,
                shard: "native-weights.safetensors")
        }
        inventory = try GLM5NextWeightInventory.validate(
            configuration: config, tensors: descriptors, storage: .float32)
        configuration = config
        configurationData = nativeConfig
        nativeArchive = native
        self.cases = cases
        // Load from the verified byte snapshots, not mutable file paths.
        weights = try loadArrays(data: native.data)
        expected = try loadArrays(data: outputs.data)
        guard Set(weights.keys) == Set(native.headers.keys),
            Set(expected.keys) == Set(outputs.headers.keys)
        else {
            throw Failure("decoded tensor inventory")
        }
    }

    private static func validateMapping(_ mapping: [String: Any], source: Archive, native: Archive)
        throws
    {
        guard mapping["all_roundtrips_bitwise_equal"] as? Bool == true,
            mapping["synthetic_fixture_only"] as? Bool == true,
            (mapping["numerical_operations"] as? [Any])?.isEmpty == true,
            mapping["source_archive_sha256"] as? String == hash(source.data),
            mapping["destination_archive_sha256"] as? String == hash(native.data),
            let operations = mapping["operations"] as? [[String: Any]]
        else { throw Failure("canonical mapping provenance") }
        var sources: Set<String> = []
        var destinations: Set<String> = []
        for operation in operations {
            guard operation["bitwise_equal"] as? Bool == true,
                let name = operation["source"] as? String, sources.insert(name).inserted,
                let parts = operation["destinations"] as? [[String: Any]]
            else { throw Failure("mapping operation") }
            let original = try source.bytes(name)
            var spans: [Range<Int>] = []
            for part in parts {
                guard let destination = part["name"] as? String,
                    destinations.insert(destination).inserted,
                    let span = part["source_byte_range"] as? [Int], span.count == 2,
                    span[0] >= 0, span[1] >= span[0], span[1] <= original.count,
                    let header = native.headers[destination],
                    part["shape"] as? [Int] == header.shape,
                    part["dtype"] as? String == header.dtype
                else { throw Failure("mapping destination") }
                let range = span[0] ..< span[1]
                let bytes = try native.bytes(destination)
                guard bytes == original.subdata(in: range), part["sha256"] as? String == hash(bytes)
                else {
                    throw Failure("non-bit-preserving canonical mapping \(destination)")
                }
                spans.append(range)
            }
            var end = 0
            for span in spans.sorted(by: { $0.lowerBound < $1.lowerBound }) {
                guard span.lowerBound == end else { throw Failure("mapping gap/overlap \(name)") }
                end = span.upperBound
            }
            guard end == original.count else { throw Failure("unmapped source bytes \(name)") }
        }
        guard sources == Set(source.headers.keys), destinations == Set(native.headers.keys) else {
            throw Failure("incomplete canonical mapping")
        }
    }

    func caseObject(_ name: String) throws -> [String: Any] {
        guard let object = cases[name] as? [String: Any] else {
            throw Failure("missing case \(name)")
        }
        return object
    }
    func key(_ object: [String: Any], _ name: String) throws -> String {
        guard let result = object[name] as? String else {
            throw Failure("missing case key \(name)")
        }
        return result
    }
    func tensor(_ name: String) throws -> MLXArray {
        guard let value = expected[name] else { throw Failure("missing expected tensor \(name)") }
        return value
    }
    func textModel(executionPolicy: GLM5NextExecutionPolicy = .reference) throws
        -> GLM5NextTextModel
    {
        try GLM5NextTextModel(
            configuration: configuration.textConfig, inventory: inventory,
            tensor: { name in
                guard let value = weights[name] else { throw Failure("missing weight \(name)") }
                return value
            }, executionPolicy: executionPolicy)
    }

    /// Throws on the first named divergent layer/key, after checking finiteness.
    /// FP32 3e-5 absolute/relative follows the oracle manifest, allowing different
    /// PyTorch/MLX matmul association, not an unmeasured model-quality tolerance.
    func compare(
        _ actual: MLXArray, key: String, mask: MLXArray? = nil,
        absolute: Float = 3e-5, relative: Float = 3e-5
    ) throws {
        try compare(
            actual, expected: tensor(key), label: key, mask: mask, absolute: absolute,
            relative: relative)
    }
    func compare(
        _ actual: MLXArray, expected reference: MLXArray, label: String,
        mask: MLXArray? = nil, absolute: Float = 3e-5, relative: Float = 3e-5
    ) throws {
        guard actual.shape == reference.shape else {
            throw Failure("\(label) shape \(actual.shape), expected \(reference.shape)")
        }
        guard actual.dtype == .float32 && reference.dtype == .float32 else {
            throw Failure("\(label) must remain FP32 for this oracle")
        }
        let values = actual.asType(.float32).asArray(Float.self)
        let expected = reference.asType(.float32).asArray(Float.self)
        guard values.allSatisfy(\.isFinite), expected.allSatisfy(\.isFinite) else {
            throw Failure("\(label) contains nonfinite values")
        }
        let valid: [Bool]?
        let stride: Int
        if let mask {
            guard actual.ndim >= 2, mask.shape == Array(actual.shape.prefix(2)), mask.dtype == .bool
            else {
                throw Failure("\(label) invalid comparison mask")
            }
            valid = mask.asArray(Bool.self)
            stride = actual.size / mask.size
        } else {
            valid = nil
            stride = 1
        }
        for index in values.indices where valid?[index / stride] ?? true {
            let difference = abs(values[index] - expected[index])
            let allowance = absolute + relative * abs(expected[index])
            guard difference <= allowance else {
                throw Failure(
                    "first divergence \(label)[\(index)]: native=\(values[index]), official=\(expected[index]), abs=\(difference), allowed=\(allowance)"
                )
            }
        }
    }

    func compareSelections(_ actual: MLXArray, key: String) throws {
        let reference = try tensor(key)
        guard actual.shape == reference.shape, actual.ndim == 3 else {
            throw Failure("\(key) selection shape")
        }
        let a = actual.asType(.int64).asArray(Int64.self)
        let e = reference.asType(.int64).asArray(Int64.self)
        guard a.allSatisfy({ $0 >= -1 }), e.allSatisfy({ $0 >= -1 }) else {
            throw Failure("\(key) invalid negative index")
        }
        let width = actual.dim(-1)
        for row in 0 ..< (actual.size / width) {
            let range = row * width ..< (row + 1) * width
            let selected = a[range].filter { $0 >= 0 }
            let expected = e[range].filter { $0 >= 0 }
            guard Set(selected).count == selected.count, Set(selected) == Set(expected) else {
                throw Failure(
                    "\(key) selected set differs at row \(row): native=\(selected), official=\(expected)"
                )
            }
        }
    }

    func compareIntegers(_ actual: MLXArray, key: String) throws {
        let reference = try tensor(key)
        let integerTypes: [DType] = [.int32, .int64, .uint8, .uint32, .bool]
        guard integerTypes.contains(actual.dtype), integerTypes.contains(reference.dtype) else {
            throw Failure("\(key) must be integer or boolean")
        }
        let values = actual.asType(.int64).asArray(Int64.self)
        let expected = reference.asType(.int64).asArray(Int64.self)
        guard actual.shape == reference.shape,
            values == expected
        else {
            throw Failure(
                "\(key) integer/geometry mismatch: native \(actual.shape) \(Array(values.prefix(24))), official \(reference.shape) \(Array(expected.prefix(24)))"
            )
        }
    }

    /// Adapt the frozen canonical byte archive to the real root artifact reader;
    /// only index/container names change. All weight bytes remain identical.
    func withRootModel(_ body: (GLM5NextModel) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "glm53-reference-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: directory) }
        let index = try JSONSerialization.data(
            withJSONObject: [
                "metadata": ["total_size": inventory.entries.values.reduce(0) { $0 + $1.bytes }],
                "weight_map": Dictionary(
                    uniqueKeysWithValues: weights.keys.map { ($0, "native-weights.safetensors") }),
            ], options: [.sortedKeys])
        let files = [
            "config.json": configurationData, "model.safetensors.index.json": index,
            "native-weights.safetensors": nativeArchive.data,
        ]
        for (name, data) in files { try data.write(to: directory.appendingPathComponent(name)) }
        let identity = GLM5NextArtifactIdentity(
            repository: "synthetic/official-glm5-next-cpu",
            revision: Self.modelRevision,
            files: files.mapValues {
                GLM5NextArtifactIdentity.File(bytes: $0.count, sha256: Self.hash($0))
            })
        let artifact = try GLM5NextArtifactLoader(
            directory: directory, expectedIdentity: identity, storage: .float32)
        try body(GLM5NextModel(artifact: artifact))
    }

    private static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private static func object(_ data: Data) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Failure("JSON object required")
        }
        return value
    }
}
