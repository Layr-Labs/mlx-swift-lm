// Copyright © 2026 Eigen Labs.

import CryptoKit
import Darwin
import Foundation
import MLX

public enum GLM5NextArtifactError: Error, Equatable {
    case invalid(String)
    case changed(String)
    case limit(String)
    case alreadyConsumed(String)
}

/// The provider's existing download/hash funnel supplies this identity from its
/// trusted, pinned manifest. This is not a download API or a `ready` assertion:
/// the local reader independently checks every expected byte count and digest.
public struct GLM5NextArtifactIdentity: Sendable {
    public struct File: Sendable {
        public let bytes: Int
        public let sha256: String
        public init(bytes: Int, sha256: String) {
            self.bytes = bytes
            self.sha256 = sha256
        }
    }
    public let repository: String
    public let revision: String
    /// Exactly config.json, model.safetensors.index.json and all indexed shards.
    public let files: [String: File]
    public init(repository: String, revision: String, files: [String: File]) {
        self.repository = repository
        self.revision = revision
        self.files = files
    }
}

/// Local parsing/IO ceilings, not replacements for OS, activation or KV admission.
public struct GLM5NextArtifactLimits: Sendable {
    /// The NVFP4 hybrid declares every routed expert matrix (~10 MiB).
    public var maximumConfigBytes = 32 * 1024 * 1024
    public var maximumIndexBytes = 32 * 1024 * 1024
    public var maximumHeaderBytes = 16 * 1024 * 1024
    public var maximumMetadataBytes = 256 * 1024 * 1024
    public var maximumTensorBytes = 2 * 1024 * 1024 * 1024
    public var maximumShards = 256
    public var maximumTensors = 200_000
    public init() {}
}

/// When `GLM5NextArtifactLoader` checks every file against its manifest SHA-256.
public enum GLM5NextArtifactVerification: Sendable, Equatable {
    /// In the initializer, before any metadata is parsed or payload read.
    case beforeLoad
    /// In `prepare(plan:includeMTP:)`: the one read of every file that hashes
    /// it also copies the plan's payload into its final buffers. The metadata
    /// is parsed first and must equal the bytes the pass then verifies; no
    /// payload is read through `tensor(_:)` and nothing is constructed until
    /// every file has matched.
    case withPreparedPayload
}

/// Conservative additional LOAD accounting. Deferred partitions remain priced.
/// The provider must add its existing OS/KV/activation guards, not substitute this
/// estimate for them. No BF16 expansion or requantization is included or allowed.
public struct GLM5NextArtifactResources: Sendable {
    public let textBytes: Int
    public let mtpBytes: Int
    public let visionBytes: Int
    public let largestTensorBytes: Int
    public let largestExpertStackBytes: Int
    public let metadataAllowanceBytes: Int
    public let additionalLoadTransientBytes: Int
    public let conservativeLoadPeakBytes: Int
}

/// Construction-lane roots, not a second parameter store. Source expert arrays
/// retire after their evaluated bank; completed stages retain the module instead
/// of parallel per-tensor aliases. Only a confirmed final/cleanup fence may clear
/// the remaining roots after an initializer or evaluation fails.
public final class GLM5NextArtifactLoadRoots {
    private var tensors: [String: MLXArray] = [:]
    private var banks: [ObjectIdentifier: MLXArray] = [:]
    private var owners: [AnyObject] = []
    private var bankEvaluated = false
    private let beforeEvaluation: (([MLXArray]) throws -> Void)?

    public init(beforeEvaluation: (([MLXArray]) throws -> Void)? = nil) {
        self.beforeEvaluation = beforeEvaluation
    }
    public var retainedArrayCount: Int { tensors.count + banks.count }
    public var retainedOwnerCount: Int { owners.count }
    public var retainedExpertSourceCount: Int {
        tensors.keys.filter { $0.contains(".mlp.experts.") }.count
    }
    fileprivate func retainTensor(_ name: String, _ array: MLXArray) { tensors[name] = array }
    public func retainOwner(_ owner: AnyObject) {
        if !owners.contains(where: { $0 === owner }) { owners.append(owner) }
    }
    public func evaluate(_ arrays: [MLXArray]) throws {
        for array in arrays { banks[ObjectIdentifier(array)] = array }
        try performEvaluation(arrays)
    }
    fileprivate func evaluateTensor(_ array: MLXArray) throws {
        try performEvaluation([array])
    }
    private func performEvaluation(_ arrays: [MLXArray]) throws {
        try MLX.withError { error in
            try beforeEvaluation?(arrays)
            try error.check()
            eval(arrays)
            try error.check()
        }
    }
    fileprivate func evaluateBank(_ bank: MLXArray) throws {
        bankEvaluated = false
        try evaluate([bank])
        bankEvaluated = true
    }
    fileprivate func releaseExpertSources(_ names: [String]) {
        guard bankEvaluated, names.allSatisfy({ $0.contains(".mlp.experts.") }) else { return }
        for name in names { tensors.removeValue(forKey: name) }
        bankEvaluated = false
    }
    fileprivate func adoptStage(_ owner: AnyObject) {
        retainOwner(owner)
        tensors.removeAll()
        banks.removeAll()
    }
    public func releaseAfterConfirmedFence() {
        tensors.removeAll()
        banks.removeAll()
        owners.removeAll()
    }
}

/// Single-load, caller-serialized local artifact owner. Headers cover ALL three
/// partitions before a payload can be read. Resolution reads one tensor range,
/// never a whole-shard dictionary; returned arrays own their bytes independently.
/// A failed/partially consumed load must be retried with a new owner.
public final class GLM5NextArtifactLoader {
    public let identity: GLM5NextArtifactIdentity
    public let configuration: GLM5NextConfiguration
    public let inventory: GLM5NextWeightInventory
    public let resources: GLM5NextArtifactResources
    public let directory: URL
    public let constructionRoots: GLM5NextArtifactLoadRoots?
    /// Fixed for the target and its MTP layer. No policy adds resident or
    /// load-transient bytes, so `resources` does not depend on this choice.
    public let executionPolicy: GLM5NextExecutionPolicy
    private let descriptorControl: (Int32, Int32, Int32) -> Int32
    private let directoryFD: Int32
    private let directoryStamp: Stamp
    private let stamps: [String: Stamp]
    private let regions: [String: Region]
    private var consumed: Set<String> = []
    private weak var loadedTextModel: GLM5NextTextModel?
    private weak var loadedTextStage: GLM5NextTextStage?
    /// `.withPreparedPayload` until `prepare` has verified every file.
    private var unverified: Bool
    /// The parsed bytes the prepared pass must find in each file it hashes:
    /// config.json and the index whole, each shard's length prefix and header.
    private let parsedPrefixes: [String: Data]
    private var preparedText: (plan: PreparedPlan, parts: GLM5NextPreparedParts)?
    private var preparedMTP: GLM5NextPreparedParts?

    private struct PreparedPlan: Equatable {
        let layers: Range<Int>
        let shard: GLM5NextTensorShard?
        let loadsMTPEmbedding: Bool
        init(_ plan: GLM5NextTextStagePlan) {
            layers = plan.layers
            shard = plan.shard
            loadsMTPEmbedding = plan.loadsMTPEmbedding
        }
    }

    private struct Region {
        let shard: String
        let offset: Int
        let bytes: Int
    }
    private struct Stamp: Equatable {
        let device: dev_t
        let inode: ino_t
        let bytes: off_t
        let modifiedSeconds: Int
        let modifiedNanos: Int
        let changedSeconds: Int
        let changedNanos: Int
        init(_ value: stat) {
            device = value.st_dev
            inode = value.st_ino
            bytes = value.st_size
            modifiedSeconds = value.st_mtimespec.tv_sec
            modifiedNanos = value.st_mtimespec.tv_nsec
            changedSeconds = value.st_ctimespec.tv_sec
            changedNanos = value.st_ctimespec.tv_nsec
        }
    }

    public convenience init(
        directory: URL, expectedIdentity: GLM5NextArtifactIdentity,
        storage: GLM5NextWeightInventory.Storage, limits: GLM5NextArtifactLimits = .init(),
        constructionRoots: GLM5NextArtifactLoadRoots? = nil,
        executionPolicy: GLM5NextExecutionPolicy = .reference,
        verification: GLM5NextArtifactVerification = .beforeLoad
    ) throws {
        try self.init(
            directory: directory, expectedIdentity: expectedIdentity, storage: storage,
            limits: limits, constructionRoots: constructionRoots, executionPolicy: executionPolicy,
            verification: verification, descriptorControl: { Darwin.fcntl($0, $1, $2) })
    }

    // Only descriptor syscalls are injectable; integrity, reads and custody stay real.
    init(
        directory: URL, expectedIdentity: GLM5NextArtifactIdentity,
        storage: GLM5NextWeightInventory.Storage, limits: GLM5NextArtifactLimits = .init(),
        constructionRoots: GLM5NextArtifactLoadRoots? = nil,
        executionPolicy: GLM5NextExecutionPolicy = .reference,
        verification: GLM5NextArtifactVerification = .beforeLoad,
        descriptorControl: @escaping (Int32, Int32, Int32) -> Int32
    ) throws {
        guard directory.isFileURL, !expectedIdentity.repository.isEmpty,
            Self.hex(expectedIdentity.revision, length: 40),
            limits.maximumConfigBytes > 0, limits.maximumIndexBytes > 0,
            limits.maximumHeaderBytes > 0, limits.maximumMetadataBytes > 0,
            limits.maximumTensorBytes > 0, limits.maximumShards > 0,
            limits.maximumTensors > 0,
            expectedIdentity.files.count - 2 <= limits.maximumShards,
            expectedIdentity.files["config.json"] != nil,
            expectedIdentity.files["model.safetensors.index.json"] != nil
        else { throw GLM5NextArtifactError.invalid("identity or reader limits") }
        for (name, file) in expectedIdentity.files {
            guard Self.safeName(name), file.bytes > 0, Self.hex(file.sha256, length: 64),
                name == "config.json" || name == "model.safetensors.index.json"
                    || name.hasSuffix(".safetensors")
            else { throw GLM5NextArtifactError.invalid("manifest file \(name)") }
        }
        guard expectedIdentity.files["config.json"]!.bytes <= limits.maximumConfigBytes,
            expectedIdentity.files["model.safetensors.index.json"]!.bytes
                <= limits.maximumIndexBytes
        else {
            throw GLM5NextArtifactError.limit("config/index size")
        }
        let fd = open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw GLM5NextArtifactError.invalid("artifact directory") }
        var transferred = false
        defer { if !transferred { close(fd) } }
        let rootStamp = try Self.stamp(fd, name: "directory", directory: true)
        try Self.checkShardNames(directory, expected: Set(expectedIdentity.files.keys))

        var fileStamps: [String: Stamp] = [:]
        // Refuse missing/preallocated-size-mismatching files before reading any
        // payload. A matching size is necessary, never sufficient: hashes follow.
        for name in expectedIdentity.files.keys.sorted() {
            let fileFD = try Self.openFile(fd, name, descriptorControl: descriptorControl)
            defer { close(fileFD) }
            let stamp = try Self.stamp(fileFD, name: name)
            guard stamp.bytes == expectedIdentity.files[name]!.bytes else {
                throw GLM5NextArtifactError.invalid("incomplete file \(name)")
            }
            fileStamps[name] = stamp
        }
        // Verification streams bounded chunks and closes each shard immediately.
        // No shard is parsed or exposed until all files have passed integrity.
        // Files are hashed concurrently by a fixed number of workers, each with
        // one page-aligned chunk, so F_NOCACHE stays effective and host memory
        // stays at `verificationHostBytes`. The first failure stops the others.
        let names = expectedIdentity.files.keys.sorted()
        let failures = GLM5NextVerificationFailures()
        try Task.checkCancellation()
        // `.withPreparedPayload` hashes in `prepare`, in the read that loads.
        let workers = verification == .beforeLoad ? Self.verificationWorkers : 0
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let chunk = UnsafeMutableRawPointer.allocate(
                byteCount: Self.hashChunkBytes,
                alignment: GLM5NextAlignedPayloadReader.alignmentBytes)
            defer { chunk.deallocate() }
            while let index = failures.next(limit: names.count) {
                let name = names[index]
                do {
                    try Self.verify(
                        directory: fd, name: name, file: expectedIdentity.files[name]!,
                        before: fileStamps[name], chunk: chunk,
                        descriptorControl: descriptorControl,
                        stopped: { failures.stopped || (worker == 0 && Task.isCancelled) })
                } catch {
                    failures.record(index: index, error)
                }
            }
        }
        if let error = failures.first { throw error }
        try Task.checkCancellation()
        func metadata(_ name: String, limit: Int) throws -> Data {
            guard let size = expectedIdentity.files[name]?.bytes, size <= limit else {
                throw GLM5NextArtifactError.limit(name)
            }
            let fileFD = try Self.openFile(fd, name, descriptorControl: descriptorControl)
            defer { close(fileFD) }
            guard try Self.stamp(fileFD, name: name) == fileStamps[name] else {
                throw GLM5NextArtifactError.changed(name)
            }
            let data = try Self.read(fileFD, count: size)
            guard try Self.stamp(fileFD, name: name) == fileStamps[name] else {
                throw GLM5NextArtifactError.changed(name)
            }
            try GLM5NextStrictJSON.validate(data)
            return data
        }
        var prefixes: [String: Data] = [:]
        let configData = try metadata("config.json", limit: limits.maximumConfigBytes)
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: configData)
        // Bound schema expansion before expectedTensors constructs its dictionaries.
        guard config.textConfig.hiddenLayers <= 512, config.textConfig.numNextNPredictLayers <= 1,
            config.visionConfig.depth <= 512,
            config.textConfig.nRoutedExperts <= 2048
        else {
            throw GLM5NextArtifactError.limit("configuration schema")
        }
        let schemaCeiling =
            (config.textConfig.hiddenLayers + config.textConfig.numNextNPredictLayers)
            * (config.textConfig.nRoutedExperts * 9 + 100) + config.visionConfig.depth * 20 + 40
        guard schemaCeiling <= limits.maximumTensors else {
            throw GLM5NextArtifactError.limit("expanded tensor schema")
        }
        let indexData = try metadata(
            "model.safetensors.index.json", limit: limits.maximumIndexBytes)
        if verification == .withPreparedPayload {
            prefixes["config.json"] = configData
            prefixes["model.safetensors.index.json"] = indexData
        }
        let index = try JSONDecoder().decode(Index.self, from: indexData)
        guard !index.weightMap.isEmpty, index.weightMap.count <= limits.maximumTensors,
            Set(index.weightMap.values).allSatisfy({
                Self.safeName($0) && $0.hasSuffix(".safetensors")
            }),
            Set(index.weightMap.values).union(["config.json", "model.safetensors.index.json"])
                == Set(expectedIdentity.files.keys)
        else { throw GLM5NextArtifactError.invalid("index/manifest shard coverage") }

        var tensors: [GLM5NextWeightInventory.Tensor] = []
        var regions: [String: Region] = [:]
        var metadataBytes = try Self.sum(configData.count, indexData.count)
        var totalPayload = 0
        for shard in Set(index.weightMap.values).sorted() {
            let fileFD = try Self.openFile(fd, shard, descriptorControl: descriptorControl)
            defer { close(fileFD) }
            guard try Self.stamp(fileFD, name: shard) == fileStamps[shard] else {
                throw GLM5NextArtifactError.changed(shard)
            }
            let prefix = try Self.read(fileFD, count: 8)
            let headerLength = prefix.enumerated().reduce(UInt64(0)) {
                $0 | UInt64($1.element) << ($1.offset * 8)
            }
            guard headerLength > 0, headerLength <= UInt64(limits.maximumHeaderBytes),
                headerLength <= UInt64(Int.max - 8),
                let fileBytes = expectedIdentity.files[shard]?.bytes,
                Int(headerLength) <= fileBytes - 8
            else {
                throw GLM5NextArtifactError.invalid("header length \(shard)")
            }
            let data = try Self.read(fileFD, count: Int(headerLength))
            if verification == .withPreparedPayload { prefixes[shard] = prefix + data }
            metadataBytes = try Self.sum(metadataBytes, data.count)
            guard metadataBytes <= limits.maximumMetadataBytes else {
                throw GLM5NextArtifactError.limit("aggregate headers")
            }
            try GLM5NextStrictJSON.validate(data)
            let header = try JSONDecoder().decode(Header.self, from: data)
            let start = Int(headerLength) + 8
            var cursor = 0
            for (name, tensor) in header.tensors.sorted(by: {
                $0.value.dataOffsets[0] < $1.value.dataOffsets[0]
            }) {
                guard index.weightMap[name] == shard, regions[name] == nil,
                    tensors.count < limits.maximumTensors,
                    tensor.dataOffsets[0] == cursor,
                    tensor.dataOffsets[1] >= cursor
                else {
                    throw GLM5NextArtifactError.invalid("index/header/offset coverage \(name)")
                }
                let bytes = try tensor.byteCount()
                guard bytes <= limits.maximumTensorBytes,
                    tensor.dataOffsets[1] - cursor == bytes,
                    tensor.dataOffsets[1] <= fileBytes - start
                else {
                    throw GLM5NextArtifactError.invalid("payload range \(name)")
                }
                regions[name] = Region(shard: shard, offset: start + cursor, bytes: bytes)
                tensors.append(
                    .init(name: name, shape: tensor.shape, dtype: tensor.dtype, shard: shard))
                cursor = tensor.dataOffsets[1]
            }
            guard cursor == fileBytes - start,
                try Self.stamp(fileFD, name: shard) == fileStamps[shard]
            else {
                throw GLM5NextArtifactError.invalid("trailing bytes or changed shard \(shard)")
            }
            totalPayload = try Self.sum(totalPayload, cursor)
        }
        guard Set(regions.keys) == Set(index.weightMap.keys),
            index.metadata.totalSize == totalPayload
        else {
            throw GLM5NextArtifactError.invalid("index tensor/total_size coverage")
        }
        let inventory = try GLM5NextWeightInventory.validate(
            configuration: config, tensors: tensors, storage: storage)
        // Allowance includes bounded raw JSON, decoded dictionaries and names.
        for tensor in tensors {
            metadataBytes = try Self.sum(
                metadataBytes, try Self.sum(1024, tensor.name.utf8.count * 4))
        }
        guard metadataBytes <= limits.maximumMetadataBytes else {
            throw GLM5NextArtifactError.limit("decoded metadata")
        }
        var banks: [String: Int] = [:]
        for entry in inventory.entries.values where entry.tensor.name.contains(".mlp.experts.") {
            banks[entry.nativeName] = try Self.sum(banks[entry.nativeName, default: 0], entry.bytes)
        }
        let largest = regions.values.map(\.bytes).max() ?? 0
        let bank = banks.values.max() ?? 0
        let vision = inventory.bytes(in: .vision)
        // Read Data + its owned MLX copy, old expert bank + evaluated destination,
        // aligned-read scratch and vision source/layout-copy/constructor allowance
        // are all explicit.
        var transient = try Self.sum(largest, largest)
        transient = try Self.sum(transient, bank)
        transient = try Self.sum(transient, try Self.sum(vision, vision))
        transient = try Self.sum(transient, limits.maximumMetadataBytes)
        transient = try Self.sum(transient, Self.verificationHostBytes)
        transient = try Self.sum(transient, Self.payloadReadScratchAllowanceBytes)
        self.descriptorControl = descriptorControl
        self.unverified = verification == .withPreparedPayload
        self.parsedPrefixes = prefixes
        self.identity = expectedIdentity
        self.constructionRoots = constructionRoots
        self.executionPolicy = executionPolicy
        self.directory = directory
        self.directoryFD = fd
        self.directoryStamp = rootStamp
        self.stamps = fileStamps
        self.configuration = config
        self.inventory = inventory
        self.regions = regions
        self.resources = .init(
            textBytes: inventory.bytes(in: .text), mtpBytes: inventory.bytes(in: .mtp),
            visionBytes: vision, largestTensorBytes: largest, largestExpertStackBytes: bank,
            metadataAllowanceBytes: limits.maximumMetadataBytes,
            additionalLoadTransientBytes: transient,
            conservativeLoadPeakBytes: try Self.sum(totalPayload, transient))
        transferred = true
        try verifyUnchanged()
    }

    deinit { close(directoryFD) }

    public func unresolved(in part: GLM5NextWeightInventory.Part) -> [String] {
        inventory.entries(in: part).map { $0.tensor.name }.filter { !consumed.contains($0) }
    }

    public func requireConsumed(_ part: GLM5NextWeightInventory.Part) throws {
        if let name = unresolved(in: part).first {
            throw GLM5NextArtifactError.invalid("unconsumed \(name)")
        }
    }

    /// Verify path identity as well as the open directory: replacing the directory
    /// or an indexed inode must not silently switch a deferred MTP load's source.
    public func verifyUnchanged() throws {
        let current = open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard current >= 0 else { throw GLM5NextArtifactError.changed("directory") }
        defer { close(current) }
        let now = try Self.stamp(current, name: "directory", directory: true)
        guard now.device == directoryStamp.device, now.inode == directoryStamp.inode else {
            throw GLM5NextArtifactError.changed("directory")
        }
        try Self.checkShardNames(directory, expected: Set(identity.files.keys))
        for name in stamps.keys { try checkFile(name) }
    }

    /// No cache of source arrays exists, so eviction never waits for a whole
    /// shard. This hook satisfies the component loader's explicit retirement API.
    public func releaseSourceTensors(_ names: [String]) {
        constructionRoots?.releaseExpertSources(names)
    }

    public func tensor(_ name: String) throws -> MLXArray {
        try Task.checkCancellation()
        guard let region = regions[name], let entry = inventory.entries[name] else {
            throw GLM5NextArtifactError.invalid("unknown tensor \(name)")
        }
        guard !consumed.contains(name) else { throw GLM5NextArtifactError.alreadyConsumed(name) }
        guard !unverified else {
            throw GLM5NextArtifactError.invalid("payload read before the artifact is verified")
        }
        try checkFile("config.json")
        try checkFile("model.safetensors.index.json")
        let fd = try Self.openFile(directoryFD, region.shard, descriptorControl: descriptorControl)
        defer { close(fd) }
        guard let expected = stamps[region.shard],
            try Self.stamp(fd, name: region.shard) == expected
        else {
            throw GLM5NextArtifactError.changed(region.shard)
        }
        // Aligned windows keep F_NOCACHE effective; see GLM5NextAlignedPayloadReader.
        let data = try GLM5NextAlignedPayloadReader.read(
            descriptor: fd, fileSize: Int(expected.bytes), offset: region.offset,
            count: region.bytes,
            checkUnchanged: {
                guard try Self.stamp(fd, name: region.shard) == expected else {
                    throw GLM5NextArtifactError.changed(region.shard)
                }
            })
        let result = try MLX.withError { error in
            guard let dtype = GLM5NextWeightInventory.storedTypes[entry.tensor.dtype]?.dtype else {
                throw GLM5NextArtifactError.invalid("tensor dtype \(name)")
            }
            let array = MLXArray(data, entry.tensor.shape, dtype: dtype)
            constructionRoots?.retainTensor(name, array)
            if let constructionRoots {
                try constructionRoots.evaluateTensor(array)
            } else {
                eval(array)
            }
            try error.check()
            return array
        }
        consumed.insert(name)
        return result
    }

    /// Each stack is evaluated and fault-checked before component loaders retire
    /// the old per-expert arrays. Never replace this with a whole-model final eval.
    public func materializeExpertStack(_ stack: MLXArray) throws {
        if let constructionRoots {
            try constructionRoots.evaluateBank(stack)
            return
        }
        try MLX.withError { error in
            eval(stack)
            try error.check()
        }
    }

    /// Fused buffers and member views must remain rooted even when admission or
    /// native evaluation fails before a containing module can be returned.
    private func materializeFusedProjections(_ arrays: [MLXArray]) throws {
        if let constructionRoots {
            try constructionRoots.evaluate(arrays)
        } else {
            try MLX.withError { error in
                eval(arrays)
                try error.check()
            }
        }
    }

    /// Retains full-artifact identity/header validation while reading payloads
    /// only for this rank. Admission runs before each selected tensor allocation.
    /// A failed load requires a fresh artifact owner, just like loadTextModel.
    public func loadTextStage(
        plan: GLM5NextTextStagePlan,
        beforeTensorRead: ((String, Int) throws -> Void)? = nil
    ) throws -> GLM5NextTextStage {
        let selected = try plan.tensorNames(in: inventory)
        guard consumed.isEmpty else {
            throw GLM5NextArtifactError.invalid("text stage requires an unconsumed artifact owner")
        }
        let prepared = try takePrepared(plan, beforeTensorRead: beforeTensorRead != nil)
        try verifyUnchanged()
        let stage = try MLX.withError { error in
            let stage = try GLM5NextTextStage(
                plan: plan, inventory: inventory,
                tensor: { name in
                    guard selected.contains(name), let entry = self.inventory.entries[name] else {
                        throw GLM5NextArtifactError.invalid("tensor outside selected stage")
                    }
                    try beforeTensorRead?(name, entry.bytes)
                    return try self.tensor(name)
                }, materializeExpertStack: materializeExpertStack,
                releaseSourceTensors: releaseSourceTensors, executionPolicy: executionPolicy,
                materializeFusedProjections: materializeFusedProjections, prepared: prepared)
            constructionRoots?.retainOwner(stage)
            try error.check()
            return stage
        }
        try consumePrepared(prepared)
        guard consumed == selected else {
            throw GLM5NextArtifactError.invalid("text stage payload ownership mismatch")
        }
        try verifyUnchanged()
        loadedTextStage = stage
        constructionRoots?.adoptStage(stage)
        return stage
    }

    public func loadTextModel() throws -> GLM5NextTextModel {
        try verifyUnchanged()
        let model = try MLX.withError { error in
            let model = try GLM5NextTextModel(
                configuration: configuration.textConfig, inventory: inventory,
                tensor: tensor, materializeExpertStack: materializeExpertStack,
                releaseSourceTensors: releaseSourceTensors, executionPolicy: executionPolicy,
                materializeFusedProjections: materializeFusedProjections)
            constructionRoots?.retainOwner(model)
            try error.check()
            return model
        }
        try requireConsumed(.text)
        try verifyUnchanged()
        loadedTextModel = model
        constructionRoots?.adoptStage(model)
        return model
    }

    /// The final stage shares only its embedding/readout with NextN. Both must
    /// come from this artifact owner; a foreign stage cannot consume its MTP.
    public func loadMTP(
        stage: GLM5NextTextStage,
        beforeTensorRead: ((String, Int) throws -> Void)? = nil
    ) throws -> GLM5NextMTP {
        guard loadedTextStage === stage, stage.plan.isFinal,
            stage.embedding != nil, stage.head != nil,
            configuration.textConfig.numNextNPredictLayers == 1
        else {
            throw GLM5NextArtifactError.invalid(
                "MTP final stage must be loaded by this artifact owner")
        }
        let prepared = preparedMTP
        preparedMTP = nil
        guard prepared == nil || beforeTensorRead == nil else {
            throw GLM5NextArtifactError.invalid("prepared payload was admitted before its pass")
        }
        try verifyUnchanged()
        let mtp = try MLX.withError { error in
            let mtp = try GLM5NextMTP(
                stage: stage,
                tensor: { name in
                    guard let entry = self.inventory.entries[name], entry.part == .mtp else {
                        throw GLM5NextArtifactError.invalid("tensor outside MTP partition")
                    }
                    try beforeTensorRead?(name, entry.bytes)
                    return try self.tensor(name)
                }, materializeExpertStack: materializeExpertStack,
                releaseSourceTensors: releaseSourceTensors,
                materializeFusedProjections: materializeFusedProjections, prepared: prepared)
            constructionRoots?.retainOwner(mtp)
            try error.check()
            return mtp
        }
        try consumePrepared(prepared)
        try requireConsumed(.mtp)
        try verifyUnchanged()
        constructionRoots?.adoptStage(mtp)
        return mtp
    }

    public func loadMTP(target: GLM5NextTextModel) throws -> GLM5NextMTP {
        guard loadedTextModel === target, configuration.textConfig.numNextNPredictLayers == 1 else {
            throw GLM5NextArtifactError.invalid("MTP target must be loaded by this artifact owner")
        }
        try verifyUnchanged()
        let mtp = try MLX.withError { error in
            let mtp = try GLM5NextMTP(
                target: target, tensor: tensor,
                materializeExpertStack: materializeExpertStack,
                releaseSourceTensors: releaseSourceTensors,
                materializeFusedProjections: materializeFusedProjections)
            constructionRoots?.retainOwner(mtp)
            try error.check()
            return mtp
        }
        try requireConsumed(.mtp)
        try verifyUnchanged()
        constructionRoots?.adoptStage(mtp)
        return mtp
    }

    private func checkFile(_ name: String) throws {
        let fd = try Self.openFile(directoryFD, name, descriptorControl: descriptorControl)
        defer { close(fd) }
        guard try Self.stamp(fd, name: name) == stamps[name] else {
            throw GLM5NextArtifactError.changed(name)
        }
    }

    private static let hashChunkBytes = 8 * 1024 * 1024
    private static let verificationWorkers = max(
        1, min(8, ProcessInfo.processInfo.activeProcessorCount / 2))
    /// Host memory held by full-artifact verification: one aligned chunk per
    /// concurrent worker. Resource quotes must include it.
    public static let verificationHostBytes = verificationWorkers * hashChunkBytes

    private static func verify(
        directory: Int32, name: String, file: GLM5NextArtifactIdentity.File, before: Stamp?,
        chunk: UnsafeMutableRawPointer, descriptorControl: (Int32, Int32, Int32) -> Int32,
        stopped: () -> Bool
    ) throws {
        let fileFD = try openFile(directory, name, descriptorControl: descriptorControl)
        defer { close(fileFD) }
        guard let before, try stamp(fileFD, name: name) == before else {
            throw GLM5NextArtifactError.changed(name)
        }
        var hash = SHA256()
        var remaining = file.bytes
        while remaining > 0 {
            if stopped() { throw CancellationError() }
            let count = min(remaining, hashChunkBytes)
            var offset = 0
            while offset < count {
                let n = Darwin.read(fileFD, chunk.advanced(by: offset), count - offset)
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw GLM5NextArtifactError.invalid("truncated read") }
                offset += n
            }
            hash.update(bufferPointer: UnsafeRawBufferPointer(start: chunk, count: count))
            remaining -= count
        }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == file.sha256 else {
            throw GLM5NextArtifactError.invalid("SHA-256 \(name)")
        }
        guard try stamp(fileFD, name: name) == before else {
            throw GLM5NextArtifactError.changed(name)
        }
    }
    /// Host scratch a payload read holds besides its returned `Data`; resource
    /// quotes must include it alongside the read and its owned MLX copy.
    public static let payloadReadScratchAllowanceBytes =
        GLM5NextAlignedPayloadReader.scratchAllowanceBytes
    private static func safeName(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".." && !name.contains("/")
            && !name.contains("\\") && !name.utf8.contains(0)
    }
    private static func hex(_ value: String, length: Int) -> Bool {
        value.utf8.count == length
            && value.utf8.allSatisfy { (48 ... 57).contains($0) || (97 ... 102).contains($0) }
    }
    private static func sum(_ a: Int, _ b: Int) throws -> Int {
        let (value, overflow) = a.addingReportingOverflow(b)
        guard a >= 0, b >= 0, !overflow else {
            throw GLM5NextArtifactError.limit("byte count overflow")
        }
        return value
    }
    private static func openFile(
        _ directory: Int32, _ name: String,
        descriptorControl: (Int32, Int32, Int32) -> Int32
    ) throws -> Int32 {
        let fd = openat(directory, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard fd >= 0 else { throw GLM5NextArtifactError.changed(name) }
        if name.hasSuffix(".safetensors") {
            // Request descriptor-local bypass before verification or range reads.
            // These OS hints do not guarantee that no bytes enter the file cache.
            guard descriptorControl(fd, F_NOCACHE, 1) == 0 else {
                close(fd)
                throw GLM5NextArtifactError.invalid("cache bypass \(name)")
            }
            guard descriptorControl(fd, F_RDAHEAD, 0) == 0 else {
                close(fd)
                throw GLM5NextArtifactError.invalid("read-ahead \(name)")
            }
        }
        return fd
    }
    private static func stamp(_ fd: Int32, name: String, directory: Bool = false) throws -> Stamp {
        var value = stat()
        guard fstat(fd, &value) == 0,
            value.st_mode & S_IFMT == (directory ? S_IFDIR : S_IFREG), value.st_size >= 0
        else {
            throw GLM5NextArtifactError.invalid(
                "not a regular \(directory ? "directory" : "file"): \(name)")
        }
        return Stamp(value)
    }
    private static func read(_ fd: Int32, count: Int) throws -> Data {
        var data = Data(count: count)
        try data.withUnsafeMutableBytes { buffer in
            var offset = 0
            while offset < count {
                let n = Darwin.read(fd, buffer.baseAddress!.advanced(by: offset), count - offset)
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw GLM5NextArtifactError.invalid("truncated read") }
                offset += n
            }
        }
        return data
    }
    private static func checkShardNames(_ directory: URL, expected: Set<String>) throws {
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        let shards = Set(names.filter { $0.hasSuffix(".safetensors") })
        guard shards == Set(expected.filter { $0.hasSuffix(".safetensors") }) else {
            throw GLM5NextArtifactError.invalid("extra or missing shard")
        }
    }

    private struct Index: Decodable {
        struct Metadata: Decodable {
            let totalSize: Int
            enum CodingKeys: String, CodingKey { case totalSize = "total_size" }
            init(from decoder: any Decoder) throws {
                let all = try decoder.container(keyedBy: Key.self)
                guard Set(all.allKeys.map(\.stringValue)) == ["total_size"] else {
                    throw GLM5NextArtifactError.invalid("index metadata fields")
                }
                let c = try decoder.container(keyedBy: CodingKeys.self)
                totalSize = try c.decode(Int.self, forKey: .totalSize)
                guard totalSize > 0 else { throw GLM5NextArtifactError.invalid("index total_size") }
            }
        }
        let metadata: Metadata
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey {
            case metadata
            case weightMap = "weight_map"
        }
        init(from decoder: any Decoder) throws {
            let all = try decoder.container(keyedBy: Key.self)
            guard Set(all.allKeys.map(\.stringValue)) == ["metadata", "weight_map"] else {
                throw GLM5NextArtifactError.invalid("index fields")
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            metadata = try c.decode(Metadata.self, forKey: .metadata)
            weightMap = try c.decode([String: String].self, forKey: .weightMap)
        }
    }
    private struct Header: Decodable {
        let tensors: [String: Tensor]
        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: Key.self)
            var tensors: [String: Tensor] = [:]
            for key in c.allKeys {
                if key.stringValue == "__metadata__" {
                    _ = try c.decode([String: String].self, forKey: key)
                } else {
                    tensors[key.stringValue] = try c.decode(Tensor.self, forKey: key)
                }
            }
            self.tensors = tensors
        }
    }
    private struct Tensor: Decodable {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey, CaseIterable {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }
        init(from decoder: any Decoder) throws {
            let all = try decoder.container(keyedBy: Key.self)
            guard Set(all.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue))
            else {
                throw GLM5NextArtifactError.invalid("tensor header fields")
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            dtype = try c.decode(String.self, forKey: .dtype)
            shape = try c.decode([Int].self, forKey: .shape)
            dataOffsets = try c.decode([Int].self, forKey: .dataOffsets)
            guard dataOffsets.count == 2, dataOffsets[0] >= 0 else {
                throw GLM5NextArtifactError.invalid("tensor offsets")
            }
        }
        func byteCount() throws -> Int {
            // Rank 0 is admitted for NVFP4 scalar global scales; the inventory
            // decides which tensors may be scalars.
            guard var bytes = GLM5NextWeightInventory.storedTypes[dtype]?.bytes,
                shape.count <= 8
            else {
                throw GLM5NextArtifactError.invalid("tensor dtype/rank")
            }
            for dimension in shape {
                let (next, overflow) = bytes.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, dimension <= Int(Int32.max), !overflow else {
                    throw GLM5NextArtifactError.invalid("tensor dimensions")
                }
                bytes = next
            }
            return bytes
        }
    }
    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
}

/// Foundation decoders silently accept duplicate object keys. Reject them,
/// including escaped aliases, before decoding the bounded metadata schema.
private struct GLM5NextStrictJSON {
    let bytes: [UInt8]
    var cursor = 0
    static func validate(_ data: Data) throws {
        var parser = Self(bytes: Array(data))
        try parser.value(depth: 0)
        parser.space()
        guard parser.cursor == parser.bytes.count else {
            throw GLM5NextArtifactError.invalid("JSON suffix")
        }
    }
    mutating func space() {
        while cursor < bytes.count && [UInt8(32), 9, 10, 13].contains(bytes[cursor]) { cursor += 1 }
    }
    mutating func take(_ byte: UInt8) -> Bool {
        space()
        if cursor < bytes.count && bytes[cursor] == byte {
            cursor += 1
            return true
        }
        return false
    }
    mutating func string() throws -> String {
        space()
        let start = cursor
        guard take(34) else { throw GLM5NextArtifactError.invalid("JSON string") }
        // A string of printable ASCII without escapes is its own bytes (the
        // decoder returns exactly them); any other goes through the decoder.
        // The 10 MB config and 35 headers hold ~250,000 strings.
        var plain = true
        while cursor < bytes.count {
            let byte = bytes[cursor]
            cursor += 1
            if byte == 34 {
                if plain {
                    return String(decoding: bytes[(start + 1) ..< (cursor - 1)], as: UTF8.self)
                }
                return try JSONDecoder().decode(String.self, from: Data(bytes[start ..< cursor]))
            }
            if byte == 92 {
                plain = false
                cursor += 1
            } else if byte < 32 || byte > 126 {
                plain = false
            }
        }
        throw GLM5NextArtifactError.invalid("unterminated JSON string")
    }
    /// `true`, `false`, `null`, or a number of at most 15 digits without an
    /// exponent: tokens Foundation's parser accepts as they are. Any other
    /// token is judged by Foundation.
    static func plainScalar(_ token: ArraySlice<UInt8>) -> Bool {
        if token.elementsEqual("true".utf8) || token.elementsEqual("false".utf8)
            || token.elementsEqual("null".utf8)
        {
            return true
        }
        var index = token.startIndex
        if index < token.endIndex && token[index] == 45 { index += 1 }
        let integer = index
        while index < token.endIndex && (48 ... 57).contains(token[index]) { index += 1 }
        let integerDigits = index - integer
        guard integerDigits > 0, integerDigits == 1 || token[integer] != 48 else { return false }
        var fractionDigits = 0
        if index < token.endIndex && token[index] == 46 {
            index += 1
            let fraction = index
            while index < token.endIndex && (48 ... 57).contains(token[index]) { index += 1 }
            fractionDigits = index - fraction
            guard fractionDigits > 0 else { return false }
        }
        return index == token.endIndex && integerDigits + fractionDigits <= 15
    }
    mutating func value(depth: Int) throws {
        guard depth <= 32 else { throw GLM5NextArtifactError.limit("JSON nesting") }
        space()
        guard cursor < bytes.count else { throw GLM5NextArtifactError.invalid("JSON value") }
        if bytes[cursor] == 34 {
            _ = try string()
            return
        }
        if take(123) {
            var keys: Set<String> = []
            if take(125) { return }
            repeat {
                let key = try string()
                guard keys.insert(key).inserted, take(58) else {
                    throw GLM5NextArtifactError.invalid("duplicate JSON key or missing colon")
                }
                try value(depth: depth + 1)
                if take(125) { return }
            } while take(44)
            throw GLM5NextArtifactError.invalid("JSON object")
        }
        if take(91) {
            if take(93) { return }
            repeat {
                try value(depth: depth + 1)
                if take(93) { return }
            } while take(44)
            throw GLM5NextArtifactError.invalid("JSON array")
        }
        let start = cursor
        while cursor < bytes.count && ![UInt8(32), 9, 10, 13, 44, 93, 125].contains(bytes[cursor]) {
            cursor += 1
        }
        guard cursor > start else { throw GLM5NextArtifactError.invalid("JSON scalar") }
        guard !Self.plainScalar(bytes[start ..< cursor]) else { return }
        _ = try JSONSerialization.jsonObject(
            with: Data(bytes[start ..< cursor]), options: .fragmentsAllowed)
    }
}

/// Shared work index and first failure (by file order) for concurrent verification.
private final class GLM5NextVerificationFailures: @unchecked Sendable {
    private let lock = NSLock()
    private var cursor = 0
    private var failure: (index: Int, error: Error)?
    var stopped: Bool { lock.withLock { failure != nil } }
    var first: Error? { lock.withLock { failure?.error } }
    func next(limit: Int) -> Int? {
        lock.withLock {
            guard failure == nil, cursor < limit else { return nil }
            defer { cursor += 1 }
            return cursor
        }
    }
    func record(index: Int, _ error: Error) {
        lock.withLock {
            // A stop caused by another file's failure never replaces that failure.
            if error is CancellationError, failure != nil { return }
            if failure == nil || index < failure!.index { failure = (index, error) }
        }
    }
}

// MARK: - Prepared payload

extension GLM5NextArtifactLoader {
    /// Bytes each worker of the prepared pass reads at a time: small enough
    /// that `verificationHostBytes` of windows lets every file of the
    /// checkpoint have its own worker (37 files, up to 64 workers on an M5
    /// Ultra). A file's SHA-256 is one sequential stream on one core (about
    /// 3 GB/s), so a pass from the file cache takes about as long as hashing
    /// its largest file only when all files are read at once; from the SSD,
    /// 37 x 1 MiB reads in flight keep it at its sequential rate (14.8 GB/s,
    /// against 9.8 GB/s for one 4 MiB reader, 2026-10-21).
    static let preparedWindowBytes = 1024 * 1024

    /// The single read of a `.withPreparedPayload` artifact: every file is
    /// read once, in page-aligned windows on as many workers as
    /// `verificationHostBytes` allows, hashed against its manifest SHA-256,
    /// compared with the metadata the initializer parsed from it, and every
    /// byte `plan` keeps (the rank's part of each split tensor, whole
    /// replicated ones; with `includeMTP`, the NextN layer whole) is copied
    /// from the window straight into its final buffer: one buffer per tensor
    /// and one per routed-expert bank, in the stacked layout the loaders
    /// build. A rank therefore reads each file once instead of twice, never
    /// holds a whole split tensor, and builds no slice or stack on the GPU;
    /// the arrays hold exactly the bytes `GLM5NextTensorShard.slice` and the
    /// bank stack give. Nothing is returned or constructed until every file
    /// has matched; `loadTextStage(plan:)` and `loadMTP(stage:)` then take the
    /// buffers. Runs once per owner.
    ///
    /// `keepFileCache` reads through the file cache, so the files stay cached
    /// (as far as memory allows) for the next load, which then reads them at
    /// memory speed (103 GB/s against 14 GB/s from the SSD, M5 Ultra); by
    /// default the pass bypasses it like every other payload read here, and
    /// either way uses pages that are already cached.
    public func prepare(plan: GLM5NextTextStagePlan, includeMTP: Bool, keepFileCache: Bool = false)
        throws
    {
        guard unverified, preparedText == nil, consumed.isEmpty else {
            throw GLM5NextArtifactError.invalid("prepare needs a fresh .withPreparedPayload owner")
        }
        guard !includeMTP || (plan.isFinal && configuration.textConfig.numNextNPredictLayers == 1)
        else {
            throw GLM5NextArtifactError.invalid("MTP is prepared only with the final stage")
        }
        try Task.checkCancellation()
        let text = try preparedUnits(plan.tensorNames(in: inventory), shard: plan.shard)
        // The NextN layer is held whole on every rank (its loader never shards).
        let mtp =
            try includeMTP
            ? preparedUnits(Set(inventory.entries(in: .mtp).map(\.tensor.name)), shard: nil) : []
        var segments: [String: [GLM5NextPreparedSegment]] = [:]
        for unit in text + mtp {
            for (index, member) in unit.members.enumerated() {
                guard let region = regions[member.name] else {
                    throw GLM5NextArtifactError.invalid("unknown tensor \(member.name)")
                }
                segments[region.shard, default: []].append(
                    GLM5NextPreparedSegment(
                        base: region.offset, runs: member.runs,
                        destination: unit.buffer.pointer + index * unit.memberBytes))
            }
        }
        for key in segments.keys { segments[key]!.sort { $0.start < $1.start } }
        try preparedPass(segments, keepFileCache: keepFileCache)
        try verifyUnchanged()
        func parts(_ units: [PreparedUnit]) throws -> GLM5NextPreparedParts {
            try MLX.withError { error in
                var tensors: [String: MLXArray] = [:]
                var banks: [[String]: MLXArray] = [:]
                for unit in units {
                    let array = unit.buffer.array(unit.shape, dtype: unit.dtype)
                    if unit.isBank {
                        banks[unit.members.map(\.name)] = array
                    } else {
                        tensors[unit.members[0].name] = array
                    }
                }
                try error.check()
                return GLM5NextPreparedParts(tensors: tensors, banks: banks)
            }
        }
        preparedText = (PreparedPlan(plan), try parts(text))
        preparedMTP = includeMTP ? try parts(mtp) : nil
        unverified = false
    }

    /// One prepared array: a tensor, or a routed-expert bank of members.
    fileprivate struct PreparedUnit {
        struct Member {
            let name: String
            let runs: GLM5NextByteRuns
        }
        let members: [Member]
        let isBank: Bool
        let shape: [Int]
        let dtype: DType
        let memberBytes: Int
        let buffer: GLM5NextPreparedBuffer
    }

    fileprivate func preparedUnits(_ names: Set<String>, shard: GLM5NextTensorShard?) throws
        -> [PreparedUnit]
    {
        func member(_ name: String) throws -> (
            PreparedUnit.Member, shape: [Int], dtype: DType, entry: GLM5NextWeightInventory.Entry
        ) {
            guard let entry = inventory.entries[name], let region = regions[name],
                region.bytes == entry.bytes,
                let dtype = GLM5NextWeightInventory.storedTypes[entry.tensor.dtype]?.dtype
            else { throw GLM5NextArtifactError.invalid("unknown tensor \(name)") }
            let shape = entry.tensor.shape
            let runs =
                try shard?.byteRuns(name, shape: shape, bytes: entry.bytes) ?? .whole(entry.bytes)
            let local = try shard?.localShape(name, shape) ?? shape
            return (.init(name: name, runs: runs), local, dtype, entry)
        }
        var units: [PreparedUnit] = []
        var banked: Set<String> = []
        for names in GLM5NextPreparedBanks.group(
            names, experts: configuration.textConfig.nRoutedExperts)
        {
            let members = try names.map(member)
            let first = members[0]
            guard
                members.allSatisfy({
                    $0.entry.tensor.shape == first.entry.tensor.shape
                        && $0.entry.tensor.dtype == first.entry.tensor.dtype
                })
            else { throw GLM5NextArtifactError.invalid("non-uniform expert bank \(names[0])") }
            let bytes = try Self.product(first.0.runs.bytes, members.count)
            units.append(
                PreparedUnit(
                    members: members.map { $0.0 }, isBank: true,
                    shape: [members.count] + first.shape,
                    dtype: first.dtype, memberBytes: first.0.runs.bytes,
                    buffer: try GLM5NextPreparedBuffer(bytes: bytes)))
            banked.formUnion(names)
        }
        for name in names.subtracting(banked).sorted() {
            let single = try member(name)
            units.append(
                PreparedUnit(
                    members: [single.0], isBank: false, shape: single.shape, dtype: single.dtype,
                    memberBytes: single.0.runs.bytes,
                    buffer: try GLM5NextPreparedBuffer(bytes: single.0.runs.bytes)))
        }
        return units
    }

    private func preparedPass(
        _ segments: [String: [GLM5NextPreparedSegment]], keepFileCache: Bool
    ) throws {
        // Largest files first, so the last ones to finish are the small ones.
        let names = identity.files.keys.sorted {
            (identity.files[$0]!.bytes, $1) > (identity.files[$1]!.bytes, $0)
        }
        let workers = max(
            1, min(names.count, Self.verificationHostBytes / Self.preparedWindowBytes))
        let failures = GLM5NextVerificationFailures()
        let (directory, files, fileStamps, prefixes, control) = (
            directoryFD, identity.files, self.stamps, parsedPrefixes, descriptorControl
        )
        try Task.checkCancellation()
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let window = UnsafeMutableRawPointer.allocate(
                byteCount: Self.preparedWindowBytes,
                alignment: GLM5NextAlignedPayloadReader.alignmentBytes)
            defer { window.deallocate() }
            while let index = failures.next(limit: names.count) {
                let name = names[index]
                do {
                    try Self.verifyAndCopy(
                        directory: directory, name: name, file: files[name]!,
                        before: fileStamps[name], window: window, parsed: prefixes[name],
                        segments: segments[name] ?? [], keepFileCache: keepFileCache,
                        descriptorControl: control,
                        stopped: { failures.stopped || (worker == 0 && Task.isCancelled) })
                } catch {
                    failures.record(index: index, error)
                }
            }
        }
        if let error = failures.first { throw error }
        try Task.checkCancellation()
    }

    /// `verify`, reading at explicit window offsets, plus: the bytes the
    /// initializer parsed must be the file's own, and each window's selected
    /// bytes are copied to their buffers.
    private static func verifyAndCopy(
        directory: Int32, name: String, file: GLM5NextArtifactIdentity.File, before: Stamp?,
        window: UnsafeMutableRawPointer, parsed: Data?, segments: [GLM5NextPreparedSegment],
        keepFileCache: Bool, descriptorControl: (Int32, Int32, Int32) -> Int32,
        stopped: () -> Bool
    ) throws {
        let fileFD = try openFile(directory, name, descriptorControl: descriptorControl)
        defer { close(fileFD) }
        if keepFileCache && name.hasSuffix(".safetensors") {
            guard descriptorControl(fileFD, F_NOCACHE, 0) == 0,
                descriptorControl(fileFD, F_RDAHEAD, 1) == 0
            else { throw GLM5NextArtifactError.invalid("file cache \(name)") }
        }
        guard let before, try stamp(fileFD, name: name) == before, let parsed,
            parsed.count <= file.bytes
        else {
            throw GLM5NextArtifactError.changed(name)
        }
        var hash = SHA256()
        var position = 0
        var next = 0
        while position < file.bytes {
            if stopped() { throw CancellationError() }
            let count = min(preparedWindowBytes, file.bytes - position)
            var filled = 0
            while filled < count {
                let n = Darwin.pread(
                    fileFD, window.advanced(by: filled), count - filled, off_t(position + filled))
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw GLM5NextArtifactError.invalid("truncated read") }
                filled += n
            }
            hash.update(bufferPointer: UnsafeRawBufferPointer(start: window, count: count))
            let upper = position + count
            if position < parsed.count {
                let length = min(upper, parsed.count) - position
                let same = parsed.withUnsafeBytes {
                    memcmp($0.baseAddress!.advanced(by: position), window, length) == 0
                }
                guard same else { throw GLM5NextArtifactError.changed(name) }
            }
            while next < segments.count && segments[next].end <= position { next += 1 }
            var index = next
            while index < segments.count && segments[index].start < upper {
                segments[index].copy(window: position ..< upper, from: window)
                index += 1
            }
            position = upper
        }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == file.sha256 else {
            throw GLM5NextArtifactError.invalid("SHA-256 \(name)")
        }
        guard try stamp(fileFD, name: name) == before else {
            throw GLM5NextArtifactError.changed(name)
        }
    }

    /// The prepared parts of exactly this plan, once; nil for an owner that
    /// verified before loading.
    private func takePrepared(_ plan: GLM5NextTextStagePlan, beforeTensorRead: Bool) throws
        -> GLM5NextPreparedParts?
    {
        guard let prepared = preparedText else {
            guard !unverified else {
                throw GLM5NextArtifactError.invalid("prepare has not verified this artifact")
            }
            return nil
        }
        guard prepared.plan == PreparedPlan(plan) else {
            throw GLM5NextArtifactError.invalid("text stage differs from the prepared plan")
        }
        guard !beforeTensorRead else {
            throw GLM5NextArtifactError.invalid("prepared payload was admitted before its pass")
        }
        preparedText = nil
        return prepared.parts
    }

    /// Records what construction took; a prepared array left over is a load
    /// that built a different model than it read.
    private func consumePrepared(_ prepared: GLM5NextPreparedParts?) throws {
        guard let prepared else { return }
        consumed.formUnion(prepared.taken)
        guard prepared.remaining == 0 else {
            throw GLM5NextArtifactError.invalid("prepared payload left unused")
        }
    }

    private static func product(_ a: Int, _ b: Int) throws -> Int {
        let (value, overflow) = a.multipliedReportingOverflow(by: b)
        guard a >= 0, b >= 0, !overflow else {
            throw GLM5NextArtifactError.limit("byte count overflow")
        }
        return value
    }
}
