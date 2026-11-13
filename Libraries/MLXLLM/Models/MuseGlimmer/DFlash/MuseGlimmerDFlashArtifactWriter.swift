// Copyright © 2026 Eigen Labs.

import CryptoKit
import Foundation
import MLX
import MLXNN

/// What `writeArtifact` produced: enough to pin and to load it.
public struct MuseGlimmerDFlashArtifactManifest: Codable, Sendable, Equatable {
    public struct File: Codable, Sendable, Equatable {
        public let bytes: Int
        public let sha256: String
    }
    public struct Projection: Codable, Sendable, Equatable {
        public let path: String
        /// Zero for a dense projection.
        public let bits: Int
        public let groupSize: Int
    }

    public static let fileName = "manifest.json"
    public let format: String
    public let files: [String: File]
    public let totalBytes: Int
    public let tensorCount: Int
    public let denseDType: String
    public let projections: [Projection]
    /// Free text naming what the artifact was converted from.
    public let source: String

    /// The pin of this artifact for `configuration`; nil without both files.
    public func pin(
        configuration: MuseGlimmerDFlashConfiguration
    ) -> MuseGlimmerDFlashArtifactPin? {
        guard let weights = files[MuseGlimmerDFlashArtifactPin.weightsFileName],
            let config = files[MuseGlimmerDFlashArtifactPin.configurationFileName]
        else { return nil }
        return MuseGlimmerDFlashArtifactPin(
            weightsSHA256: weights.sha256, configurationSHA256: config.sha256,
            configuration: configuration)
    }
}

extension MuseGlimmerDFlashDraftModel {
    /// Replaces the named dense projections with MLX affine-quantized ones in
    /// memory. A projection that is already quantized is refused: quantize
    /// from the dense source only. `plan` maps a projection path
    /// (`layers.0.mlp.up_proj`) to its format; unnamed projections stay.
    public func quantize(_ plan: [String: MuseGlimmerDFlashProjectionFormat]) throws {
        let leaves = Dictionary(uniqueKeysWithValues: leafModules().flattened())
        let known = Set(MuseGlimmerDFlashInventory(configuration).projections.map(\.path))
        var replacements: [(String, Module)] = []
        for (path, format) in plan.sorted(by: { $0.key < $1.key }) {
            guard known.contains(path), let linear = leaves[path] as? Linear else {
                throw MuseGlimmerDFlashError.inventory(path)
            }
            guard case .affine(let bits, let groupSize) = format else { continue }
            guard !(linear is QuantizedLinear), linear.bias == nil,
                MuseGlimmerDFlashInventory.affineBits.contains(bits),
                MuseGlimmerDFlashInventory.affineGroupSizes.contains(groupSize),
                linear.shape.1 % groupSize == 0, (linear.shape.1 * bits) % 32 == 0
            else { throw MuseGlimmerDFlashError.shape(path) }
            let (weight, scales, biases) = MLX.quantized(
                linear.weight, groupSize: groupSize, bits: bits, mode: .affine)
            guard let biases else { throw MuseGlimmerDFlashError.shape(path) }
            // Scales and offsets in the dense dtype, as the artifact stores them.
            let layer = QuantizedLinear(
                weight: weight, bias: nil, scales: scales.asType(activationDType),
                biases: biases.asType(activationDType), groupSize: groupSize, bits: bits,
                mode: .affine)
            layer.freeze()
            try MLX.withError { errors in
                eval(layer.parameters())
                try errors.check()
            }
            replacements.append((path, layer))
        }
        try install(projections: replacements)
    }

    /// Bytes of every tensor `writeArtifact` would store.
    public var artifactPayloadBytes: Int {
        parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    }

    /// Writes this model as a drafter artifact: `config.json`,
    /// `model.safetensors` and `manifest.json` in `directory`, which must
    /// exist and hold none of them.
    ///
    /// The bytes depend only on the weights and the configuration: tensors
    /// are stored in name order, each at the offset the previous one ends,
    /// under a header with sorted keys. Quantized weights are stored as their
    /// packed bytes (U8); the loader reads each projection's format back from
    /// the tensor shapes.
    @discardableResult
    public func writeArtifact(
        to directory: URL, source: String
    ) throws -> (manifest: MuseGlimmerDFlashArtifactManifest, pin: MuseGlimmerDFlashArtifactPin) {
        let weightsURL = directory.appendingPathComponent(
            MuseGlimmerDFlashArtifactPin.weightsFileName)
        let configurationURL = directory.appendingPathComponent(
            MuseGlimmerDFlashArtifactPin.configurationFileName)
        let manifestURL = directory.appendingPathComponent(
            MuseGlimmerDFlashArtifactManifest.fileName)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue,
            [weightsURL, configurationURL, manifestURL].allSatisfy({
                !FileManager.default.fileExists(atPath: $0.path)
            })
        else { throw MuseGlimmerDFlashError.artifact("output directory") }

        let denseName: String
        switch activationDType {
        case .bfloat16: denseName = "BF16"
        case .float16: denseName = "F16"
        case .float32: denseName = "F32"
        default: throw MuseGlimmerDFlashError.shape("dense dtype")
        }
        struct Entry {
            let name: String
            let dtype: String
            let shape: [Int]
            let array: MLXArray
        }
        let formats = projectionFormats
        var entries: [Entry] = []
        for (name, array) in parameters().flattened() {
            let owner = String(name[..<(name.lastIndex(of: ".") ?? name.startIndex)])
            if name.hasSuffix(".weight"), case .affine? = formats[owner] {
                guard array.dtype == .uint32, array.ndim == 2 else {
                    throw MuseGlimmerDFlashError.shape(name)
                }
                entries.append(
                    Entry(
                        name: name, dtype: "U8", shape: [array.dim(0), array.dim(1) * 4],
                        array: array))
            } else {
                guard array.dtype == activationDType else {
                    throw MuseGlimmerDFlashError.shape(name)
                }
                entries.append(
                    Entry(name: name, dtype: denseName, shape: array.shape, array: array))
            }
        }
        entries.sort { $0.name < $1.name }

        // Header: sorted keys, no whitespace, padded with spaces to 8 bytes.
        var offset = 0
        var fields: [String] = []
        for entry in entries {
            let end = offset + entry.array.nbytes
            let shape = entry.shape.map(String.init).joined(separator: ",")
            fields.append(
                "\"\(entry.name)\":{\"dtype\":\"\(entry.dtype)\",\"shape\":[\(shape)],"
                    + "\"data_offsets\":[\(offset),\(end)]}")
            offset = end
        }
        var header = Data(("{" + fields.joined(separator: ",") + "}").utf8)
        header.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - header.count % 8) % 8))

        guard FileManager.default.createFile(atPath: weightsURL.path, contents: nil) else {
            throw MuseGlimmerDFlashError.artifact("model.safetensors")
        }
        let handle = try FileHandle(forWritingTo: weightsURL)
        var hash = SHA256()
        var written = 0
        func write(_ data: Data) throws {
            try handle.write(contentsOf: data)
            hash.update(data: data)
            written += data.count
        }
        do {
            var length = UInt64(header.count).littleEndian
            try write(withUnsafeBytes(of: &length) { Data($0) })
            try write(header)
            for entry in entries {
                try autoreleasepool {
                    let data = entry.array.asData(access: .copy).data
                    guard data.count == entry.array.nbytes else {
                        throw MuseGlimmerDFlashError.shape(entry.name)
                    }
                    try write(data)
                }
            }
            try handle.close()
        } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: weightsURL)
            throw error
        }
        let weightsSHA256 = hash.finalize().map { String(format: "%02x", $0) }.joined()

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let configurationData = try encoder.encode(configuration)
        try configurationData.write(to: configurationURL)
        let configurationSHA256 = Self.sha256Hex(configurationData)

        let manifest = MuseGlimmerDFlashArtifactManifest(
            format: "muse-glimmer-dflash-drafter-v1",
            files: [
                MuseGlimmerDFlashArtifactPin.weightsFileName: .init(
                    bytes: written, sha256: weightsSHA256),
                MuseGlimmerDFlashArtifactPin.configurationFileName: .init(
                    bytes: configurationData.count, sha256: configurationSHA256),
            ],
            totalBytes: written + configurationData.count, tensorCount: entries.count,
            denseDType: denseName,
            projections: formats.sorted { $0.key < $1.key }.map { path, format in
                switch format {
                case .dense: .init(path: path, bits: 0, groupSize: 0)
                case .affine(let bits, let groupSize):
                    .init(path: path, bits: bits, groupSize: groupSize)
                }
            },
            source: source)
        try encoder.encode(manifest).write(to: manifestURL)
        return (
            manifest,
            MuseGlimmerDFlashArtifactPin(
                weightsSHA256: weightsSHA256, configurationSHA256: configurationSHA256,
                configuration: configuration)
        )
    }
}
