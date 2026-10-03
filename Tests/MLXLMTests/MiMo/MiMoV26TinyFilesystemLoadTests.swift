import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Bounded metadata preflight, load footprint and strict serial load of the
/// tiny synthetic checkpoint that MiMoV26TinyCheckpoint writes to a temporary
/// folder. Byte totals are hand counts from the tiny geometry; file sizes are
/// read back from the file system.
final class MiMoV26TinyFilesystemLoadTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint
    private enum Probe: Error { case revoked, callback }
    private final class Reservation: MiMoV26SerialLoadReservation {
        var request: MiMoV26SerialLoadRequest
        var reservedLoadBytes: UInt64
        var revoked = false
        var validations = 0
        var onValidate: ((MiMoV26SerialLoadProgress) throws -> Void)?
        init(_ request: MiMoV26SerialLoadRequest, bytes: UInt64? = nil) {
            self.request = request
            reservedLoadBytes = bytes ?? request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {
            validations += 1
            if revoked { throw Probe.revoked }
            try onValidate?(progress)
        }
    }

    private func bundle() throws -> URL {
        let root = try Fixture.writeNativeBundle(to: Fixture.temporaryRoot("fs"))
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func size(_ url: URL) throws -> Int {
        try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber
        ).intValue
    }
    private func appendByte(_ url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data([0]))
    }
    private func mutateHeader(_ url: URL, _ mutation: (inout [String: Any]) throws -> Void) throws {
        let data = try Data(contentsOf: url)
        let length = Int(
            Array(data.prefix(8)).enumerated().reduce(UInt64(0)) {
                $0 | (UInt64($1.element) << (UInt64($1.offset) * 8))
            })
        var header = try Fixture.object(data.subdata(in: 8 ..< (8 + length)))
        try mutation(&header)
        var raw = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        raw.append(Data(repeating: 32, count: (8 - raw.count % 8) % 8))
        let n = UInt64(raw.count)
        var result = Data((0 ..< 8).map { UInt8((n >> (UInt64($0) * 8)) & 0xff) })
        result.append(raw)
        result.append(data.dropFirst(8 + length))
        try result.write(to: url)
    }

    // MARK: - Preflight

    func testPreflightReadsOnlyMetadataAndBindsEveryShard() throws {
        let root = try bundle()
        var phases: [MiMoV26FilesystemPhase] = []
        let plan = try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: Fixture.provenance, limits: Fixture.limits,
            progress: { phases.append($0.phase) })
        XCTAssertEqual(
            phases,
            [
                .configurationRead, .indexRead, .headerRead, .headerRead, .headerRead,
                .headerRead, .preflightComplete,
            ])
        XCTAssertEqual(plan.canonicalRoot.path, root.path)
        XCTAssertEqual(plan.shards.map(\.name), Fixture.nativeFiles)
        XCTAssertEqual(plan.bundlePlan.descriptors.count, 207)
        XCTAssertEqual(plan.bundlePlan.tensorBytes, 976_204)
        XCTAssertEqual(plan.shards.map(\.tensorBytes), [755_968, 67_608, 139_160, 13_468])
        let sizes = try Fixture.nativeFiles.map { try size(root.appendingPathComponent($0)) }
        XCTAssertEqual(plan.totalFileBytes, sizes.reduce(0, +))
        var metadata =
            try size(root.appendingPathComponent("config.json"))
            + size(root.appendingPathComponent("model.safetensors.index.json"))
        let specs = try Fixture.nativeSpecs()
        for (shard, bytes) in zip(plan.shards, sizes) {
            XCTAssertEqual(shard.objectState.bytes, bytes)
            XCTAssertEqual(8 + shard.headerBytes + shard.tensorBytes, bytes, shard.name)
            XCTAssertEqual(shard.headerSHA256.count, 64)
            XCTAssertNil(shard.suppliedPayloadSHA256)
            let expected = specs.filter { $0.file == shard.name }
            XCTAssertEqual(shard.tensorKeysInFileOrder, expected.map(\.name))
            XCTAssertEqual(shard.tensorKeys, Set(expected.map(\.name)))
            XCTAssertEqual(shard.tensorLocations.first?.dataOffset, 8 + shard.headerBytes)
            XCTAssertEqual(shard.tensorLocations.map(\.byteCount), expected.map(\.byteCount))
            metadata += 8 + shard.headerBytes
        }
        XCTAssertEqual(plan.metadataBytesRead, metadata)
        XCTAssertEqual(plan.configurationURL.lastPathComponent, "config.json")
        XCTAssertEqual(plan.indexURL.lastPathComponent, "model.safetensors.index.json")
        XCTAssertEqual(plan.tokenizerURL.path, root.appendingPathComponent("tokenizer.json").path)
        XCTAssertEqual(
            plan.tokenizerConfigurationURL.path,
            root.appendingPathComponent("tokenizer_config.json").path)
        XCTAssertNoThrow(try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan))
    }

    func testPreflightMatchesSuppliedEvidenceAndRefusesEachMismatch() throws {
        let root = try bundle()
        let first = try Fixture.preflight(root)
        let zeros = String(repeating: "0", count: 64)
        let files = Dictionary(
            uniqueKeysWithValues: first.shards.map {
                (
                    $0.name,
                    MiMoV26FilesystemExpectedFile(
                        objectState: $0.objectState, headerSHA256: $0.headerSHA256,
                        payloadSHA256: zeros)
                )
            })
        let exact = MiMoV26FilesystemExpectations(
            configurationSHA256: first.bundlePlan.configSHA256,
            indexSHA256: first.bundlePlan.indexSHA256, files: files)
        let bound = try Fixture.preflight(root, expectations: exact)
        // A supplied payload digest is descriptive only; preflight reads no payload.
        XCTAssertTrue(bound.shards.allSatisfy { $0.suppliedPayloadSHA256 == zeros })
        XCTAssertThrowsError(
            try Fixture.preflight(root, expectations: .init(configurationSHA256: zeros))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError, .invalidMetadata("configuration digest"))
        }
        XCTAssertThrowsError(try Fixture.preflight(root, expectations: .init(indexSHA256: zeros))) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .invalidMetadata("index digest"))
        }
        XCTAssertThrowsError(
            try Fixture.preflight(root, expectations: .init(configurationSHA256: "ABC"))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError, .limit("metadata limits or expected digest"))
        }
        XCTAssertThrowsError(
            try Fixture.preflight(
                root, expectations: .init(files: ["missing.safetensors": .init()]))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError, .invalidMetadata("root index/file inventory"))
        }
        XCTAssertThrowsError(
            try Fixture.preflight(
                root, expectations: .init(files: ["mtp.safetensors": .init(headerSHA256: "xyz")]))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError, .invalidMetadata("expected shard digest"))
        }
        XCTAssertThrowsError(
            try Fixture.preflight(
                root,
                expectations: .init(files: ["mtp.safetensors": .init(headerSHA256: zeros)]))
        ) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .changedObject("mtp.safetensors"))
        }
        let state = try XCTUnwrap(first.shards.first?.objectState)
        let moved = MiMoV26FilesystemObjectState(
            device: state.device, inode: state.inode, bytes: state.bytes + 1,
            modifiedSeconds: state.modifiedSeconds,
            modifiedNanoseconds: state.modifiedNanoseconds, changedSeconds: state.changedSeconds,
            changedNanoseconds: state.changedNanoseconds)
        XCTAssertThrowsError(
            try Fixture.preflight(
                root, expectations: .init(files: ["audio.safetensors": .init(objectState: moved)]))
        ) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .changedObject("audio.safetensors"))
        }
    }

    func testPreflightRejectsEveryHeaderDefect() throws {
        for variant in [
            "shape", "dtype", "offset", "negativeOffset", "positiveGap", "missing", "extra",
        ] {
            let root = try bundle()
            try mutateHeader(root.appendingPathComponent("target.safetensors")) { header in
                let key =
                    variant == "positiveGap"
                    ? "model.layers.0.self_attn.q_proj.weight" : "model.norm.weight"
                let original = try XCTUnwrap(header[key] as? [String: Any])
                switch variant {
                case "missing": header.removeValue(forKey: key)
                case "extra": header["unexpected.weight"] = original
                default:
                    var entry = original
                    if variant == "shape" { entry["shape"] = [1] }
                    if variant == "dtype" { entry["dtype"] = "U32" }
                    if variant == "offset" { entry["data_offsets"] = [0, 256] }
                    if variant == "negativeOffset" { entry["data_offsets"] = [-1, 127] }
                    if variant == "positiveGap" {
                        let offsets = try XCTUnwrap(original["data_offsets"] as? [Int])
                        entry["data_offsets"] = [offsets[0] + 4, offsets[1] + 4]
                    }
                    header[key] = entry
                }
            }
            XCTAssertThrowsError(try Fixture.preflight(root), variant) { error in
                switch variant {
                case "positiveGap":
                    XCTAssertEqual(
                        error as? MiMoV26FilesystemError,
                        .invalidHeader("overlap/gap: target.safetensors"))
                case "extra":
                    XCTAssertEqual(
                        error as? MiMoV26FilesystemError, .invalidHeader("unexpected.weight"))
                case "shape", "dtype", "offset", "negativeOffset":
                    XCTAssertEqual(
                        error as? MiMoV26FilesystemError, .invalidHeader("model.norm.weight"),
                        variant)
                default:
                    XCTAssertTrue(error is MiMoV26FilesystemError, variant)
                }
            }
        }
    }

    func testPreflightRejectsUnsafeFilesAndRootInventory() throws {
        let missing = try bundle()
        try FileManager.default.removeItem(at: missing.appendingPathComponent("vision.safetensors"))
        XCTAssertThrowsError(try Fixture.preflight(missing)) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError,
                .invalidMetadata("unindexed/missing root payload files"))
        }
        let extra = try bundle()
        try Data([0]).write(to: extra.appendingPathComponent("unindexed.bin"))
        XCTAssertThrowsError(try Fixture.preflight(extra)) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError,
                .invalidMetadata("unindexed/missing root payload files"))
        }
        let unsafe = try bundle()
        let indexURL = unsafe.appendingPathComponent("model.safetensors.index.json")
        var index = try Fixture.object(Data(contentsOf: indexURL))
        var map = try XCTUnwrap(index["weight_map"] as? [String: String])
        map["model.norm.weight"] = "../outside.safetensors"
        index["weight_map"] = map
        try Fixture.data(index).write(to: indexURL)
        XCTAssertThrowsError(try Fixture.preflight(unsafe)) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError, .invalidMetadata("root index/file inventory"))
        }
        let symlink = try bundle()
        let file = symlink.appendingPathComponent("target.safetensors")
        let outside = symlink.appendingPathComponent("retained.target")
        try FileManager.default.moveItem(at: file, to: outside)
        try FileManager.default.createSymbolicLink(at: file, withDestinationURL: outside)
        XCTAssertThrowsError(try Fixture.preflight(symlink)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .unsafeFile("target.safetensors"))
        }
        let truncated = try bundle()
        let handle = try FileHandle(
            forWritingTo: truncated.appendingPathComponent("target.safetensors"))
        try handle.truncate(atOffset: 16)
        try handle.close()
        XCTAssertThrowsError(try Fixture.preflight(truncated)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .invalidHeader("target.safetensors"))
        }
        let badLength = try bundle()
        let writer = try FileHandle(
            forWritingTo: badLength.appendingPathComponent("mtp.safetensors"))
        try writer.write(contentsOf: Data(repeating: 255, count: 8))
        try writer.close()
        XCTAssertThrowsError(try Fixture.preflight(badLength)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .invalidHeader("mtp.safetensors"))
        }
        let duplicate = try bundle()
        let config = duplicate.appendingPathComponent("config.json")
        var data = Data("{\"model_type\":\"mimo_v2\",".utf8)
        data.append(try Data(contentsOf: config).dropFirst())
        try data.write(to: config)
        XCTAssertThrowsError(try Fixture.preflight(duplicate)) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError, .invalidMetadata("duplicate JSON key: config.json"))
        }
        let notDirectory = try bundle().appendingPathComponent("config.json")
        XCTAssertThrowsError(try Fixture.preflight(notDirectory)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .invalidRoot)
        }
        let remote = try XCTUnwrap(URL(string: "https://example.com"))
        XCTAssertThrowsError(try Fixture.preflight(remote)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .invalidRoot)
        }
        // A separately packaged component folder is outside the root index scope.
        let external = try bundle()
        try FileManager.default.createDirectory(
            at: external.appendingPathComponent("dflash"), withIntermediateDirectories: false)
        try Data([1]).write(to: external.appendingPathComponent("dflash/model.safetensors"))
        XCTAssertEqual(try Fixture.preflight(external).shards.count, 4)
    }

    func testPreflightEnforcesEachMetadataLimit() throws {
        let root = try bundle()
        func check(_ limits: MiMoV26FilesystemLimits, _ expected: MiMoV26FilesystemError) {
            XCTAssertThrowsError(
                try MiMoV26FilesystemWeights.preflight(
                    root: root, provenance: Fixture.provenance, limits: limits)
            ) { XCTAssertEqual($0 as? MiMoV26FilesystemError, expected) }
        }
        let shard = 2 << 20
        let total = 8 << 20
        check(
            .init(maximumShardBytes: 10, maximumTotalFileBytes: total),
            .limit("audio.safetensors"))
        check(
            .init(maximumShardBytes: shard, maximumTotalFileBytes: 10),
            .limit("total declared file bytes"))
        check(
            .init(
                maximumShardBytes: shard, maximumTotalFileBytes: total, maximumConfigurationBytes: 8
            ),
            .limit("config.json"))
        check(
            .init(maximumShardBytes: shard, maximumTotalFileBytes: total, maximumIndexBytes: 8),
            .limit("model.safetensors.index.json"))
        check(
            .init(maximumShardBytes: shard, maximumTotalFileBytes: total, maximumHeaderBytes: 8),
            .invalidHeader("audio.safetensors"))
        check(
            .init(maximumShardBytes: shard, maximumTotalFileBytes: total, maximumMetadataBytes: 32),
            .limit("metadata budget"))
        check(
            .init(maximumShardBytes: shard, maximumTotalFileBytes: total, maximumShards: 1),
            .invalidMetadata("root index/file inventory"))
        check(
            .init(maximumShardBytes: shard, maximumTotalFileBytes: total, maximumTensors: 10),
            .invalidMetadata("root index/file inventory"))
        check(
            .init(maximumShardBytes: 0, maximumTotalFileBytes: total),
            .limit("metadata limits or expected digest"))
        check(
            .init(maximumShardBytes: shard, maximumTotalFileBytes: total, maximumShards: 257),
            .limit("metadata limits or expected digest"))
    }

    func testPreflightCancellationAndLaterObjectChangesAreDetected() throws {
        let root = try bundle()
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: Fixture.provenance, limits: Fixture.limits,
                isCancelled: { true })
        ) { XCTAssertEqual($0 as? MiMoV26FilesystemError, .cancelled) }
        var cancel = false
        var headers = 0
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: Fixture.provenance, limits: Fixture.limits,
                isCancelled: { cancel },
                progress: { progress in
                    if progress.phase == .headerRead {
                        headers += 1
                        cancel = true
                    }
                })
        ) { XCTAssertEqual($0 as? MiMoV26FilesystemError, .cancelled) }
        XCTAssertEqual(headers, 1)
        let plan = try Fixture.preflight(root)
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan, isCancelled: { true })
        ) { XCTAssertEqual($0 as? MiMoV26FilesystemError, .cancelled) }
        try appendByte(root.appendingPathComponent("target.safetensors"))
        XCTAssertThrowsError(try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .changedObject("target.safetensors"))
        }
        let other = try bundle()
        let otherPlan = try Fixture.preflight(other)
        try Data([1]).write(to: other.appendingPathComponent("extra.safetensors"))
        XCTAssertThrowsError(try MiMoV26FilesystemWeights.validateCurrentObjects(plan: otherPlan)) {
            XCTAssertEqual(
                $0 as? MiMoV26FilesystemError,
                .invalidMetadata("unindexed/missing root payload files"))
        }
        let third = try bundle()
        let thirdPlan = try Fixture.preflight(third)
        try appendByte(third.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try MiMoV26FilesystemWeights.validateCurrentObjects(plan: thirdPlan)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .changedObject("root metadata"))
        }
    }

    // MARK: - Native handles

    func testLoadReturnsLazyHandlesAndTheCompleteBundle() throws {
        let root = try bundle()
        let plan = try Fixture.preflight(root)
        var phases: [MiMoV26FilesystemPhase] = []
        var result = try Fixture.withScope { work in
            try MiMoV26FilesystemWeights.load(
                plan: plan, retaining: work, progress: { phases.append($0.phase) })
        }
        XCTAssertEqual(phases.filter { $0 == .willCreateShardHandles }.count, 4)
        XCTAssertEqual(phases.filter { $0 == .shardHandlesCreated }.count, 4)
        XCTAssertEqual(Array(phases.suffix(2)), [.willConstructBundle, .bundleConstructed])
        XCTAssertEqual(result.receipt.returnedHandleTensorBytes, 976_204)
        XCTAssertEqual(result.receipt.returnedHandleFileBytes, plan.totalFileBytes)
        XCTAssertEqual(result.receipt.metadataBytesRead, plan.metadataBytesRead)
        XCTAssertEqual(result.receipt.configSHA256, plan.bundlePlan.configSHA256)
        XCTAssertTrue(result.receipt.payloadDisposition.contains("lazy"))
        XCTAssertTrue(result.bundle.mtp.isLoaded)
        XCTAssertEqual(result.materializationHandleCount, 207)
        let handles = result.takeMaterializationHandles()
        XCTAssertEqual(result.materializationHandleCount, 0)
        XCTAssertTrue(result.takeMaterializationHandles().isEmpty)
        XCTAssertEqual(handles.map(\.name), plan.shards.flatMap(\.tensorKeysInFileOrder))
        XCTAssertEqual(handles.reduce(0) { $0 + $1.byteCount }, 976_204)
        for handle in handles { eval(handle.array) }
        try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan)
        let expertName = "model.layers.1.mlp.switch_mlp.gate_proj.weight"
        let target = Dictionary(uniqueKeysWithValues: result.bundle.target.parameters().flattened())
        let expert = try XCTUnwrap(target[expertName])
        XCTAssertEqual(expert.dtype, .uint32)
        XCTAssertEqual(expert.asArray(UInt32.self).first, Fixture.words(expertName, count: 1)[0])
        let scales = try XCTUnwrap(target["model.layers.1.mlp.switch_mlp.gate_proj.scales"])
        XCTAssertEqual(scales.asArray(UInt8.self).first, 120)
        let norm = try XCTUnwrap(target["model.norm.weight"])
        XCTAssertEqual(norm.dtype, .bfloat16)
        XCTAssertEqual(norm.asType(.float32).asArray(Float.self), Array(repeating: 1, count: 64))
        XCTAssertNoThrow(try JSONEncoder().encode(result.receipt))
    }

    func testLoadCancellationAndMutationPublishNoBundle() throws {
        let root = try bundle()
        let plan = try Fixture.preflight(root)
        var cancel = false
        var constructed = false
        XCTAssertThrowsError(
            try Fixture.withScope { work in
                try MiMoV26FilesystemWeights.load(
                    plan: plan, retaining: work, isCancelled: { cancel },
                    progress: { progress in
                        if progress.phase == .shardHandlesCreated { cancel = true }
                        if progress.phase == .bundleConstructed { constructed = true }
                    })
            }
        ) { XCTAssertEqual($0 as? MiMoV26FilesystemError, .cancelled) }
        XCTAssertFalse(constructed)
        let changed = try bundle()
        let changedPlan = try Fixture.preflight(changed)
        XCTAssertThrowsError(
            try Fixture.withScope { work in
                try MiMoV26FilesystemWeights.load(
                    plan: changedPlan, retaining: work,
                    progress: { progress in
                        guard progress.phase == .willCreateShardHandles,
                            let name = progress.currentFile
                        else { return }
                        try self.appendByte(changed.appendingPathComponent(name))
                    })
            }
        ) { XCTAssertEqual($0 as? MiMoV26FilesystemError, .changedObject("audio.safetensors")) }
    }

    // MARK: - Footprint

    func testFootprintPricesEveryComponentAndCopyFamily() throws {
        let root = try bundle()
        let plan = try Fixture.preflight(root)
        let estimate = try MiMoV26LoadFootprint.estimate(plan: plan)
        let audioFile = try size(root.appendingPathComponent("audio.safetensors"))
        XCTAssertEqual(estimate.contract, "mimo-v26-strict-root-serial-source-v1")
        XCTAssertEqual(estimate.residentBytes, UInt64(plan.totalFileBytes))
        XCTAssertEqual(estimate.tensorCount, 207)
        XCTAssertEqual(estimate.largestShardBytes, UInt64(audioFile))
        // audio_encoder.projection.mlp.0.weight: 1024 x 256 BF16.
        XCTAssertEqual(estimate.largestTensorBytes, 524_288)
        // Three MXFP4 projections of four experts: 3 x (4096 + 256) bytes.
        XCTAssertEqual(estimate.largestExpertBlockBytes, 13_056)
        // Vision 13468 + audio 755968 + MTP 67608.
        XCTAssertEqual(estimate.auxiliaryCopyBytes, 837_044)
        XCTAssertEqual(estimate.pageRoundingBytes, 207 * 2 * 16_384)
        XCTAssertEqual(estimate.metadataAllowanceBytes, 1 << 30)
        XCTAssertEqual(
            estimate.transientBytes,
            UInt64(audioFile) + 524_288 + 13_056 + 837_044 + 207 * 2 * 16_384 + (1 << 30))
        XCTAssertEqual(estimate.totalBytes, estimate.residentBytes + estimate.transientBytes)
        XCTAssertEqual(estimate.configSHA256, plan.bundlePlan.configSHA256)
        XCTAssertEqual(estimate.indexSHA256, plan.bundlePlan.indexSHA256)
        XCTAssertEqual(estimate.descriptorSHA256, plan.bundlePlan.descriptorSHA256)
        XCTAssertNoThrow(try JSONEncoder().encode(estimate))
        try appendByte(root.appendingPathComponent("vision.safetensors"))
        XCTAssertThrowsError(try MiMoV26LoadFootprint.estimate(plan: plan)) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .changedObject("vision.safetensors"))
        }
    }

    func testFootprintArithmeticRefusesProfileAndInventoryDrift() throws {
        let plan = try Fixture.nativeInputs().plan()
        let specs = try Fixture.nativeSpecs()
        var payload: [String: Int] = [:]
        for spec in specs { payload[spec.file, default: 0] += spec.byteCount }
        let files = Fixture.nativeFiles.map { (name: $0, bytes: payload[$0]! + 4096) }
        let estimate = try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: files, allocationPageBytes: 16_384)
        XCTAssertEqual(estimate.residentBytes, UInt64(976_204 + 4 * 4096))
        XCTAssertEqual(estimate.largestShardBytes, UInt64(755_968 + 4096))
        XCTAssertThrowsError(
            try MiMoV26LoadFootprint.estimateValidatedLayout(
                plan, files: files, allocationPageBytes: 4096)
        ) { XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .unsupportedProfile) }
        XCTAssertThrowsError(
            try MiMoV26LoadFootprint.estimateValidatedLayout(
                plan, files: Array(files.dropLast()), allocationPageBytes: 16_384)
        ) { XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .invalidInventory) }
        let renamed = files.map {
            $0.name == "mtp.safetensors" ? (name: "other", bytes: $0.bytes) : $0
        }
        XCTAssertThrowsError(
            try MiMoV26LoadFootprint.estimateValidatedLayout(
                plan, files: renamed, allocationPageBytes: 16_384)
        ) { XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .invalidInventory) }
        let headerless = Fixture.nativeFiles.map { (name: $0, bytes: payload[$0]!) }
        XCTAssertThrowsError(
            try MiMoV26LoadFootprint.estimateValidatedLayout(
                plan, files: headerless, allocationPageBytes: 16_384)
        ) { XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .invalidInventory) }
        let oversized = Fixture.nativeFiles.map { (name: $0, bytes: payload[$0]! + (8 << 20)) }
        XCTAssertThrowsError(
            try MiMoV26LoadFootprint.estimateValidatedLayout(
                plan, files: oversized, allocationPageBytes: 16_384)
        ) { XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .invalidInventory) }
        // The published MLX-VLM layout has its own affine 4/64 profile.
        let (inputs, _, mlxSpecs) = try Fixture.mlxVLMInputs()
        let mlxPlan = try inputs.plan()
        var mlxPayload: [String: Int] = [:]
        for spec in mlxSpecs { mlxPayload[spec.file, default: 0] += spec.byteCount }
        let mlxFiles = mlxPayload.keys.sorted().map { (name: $0, bytes: mlxPayload[$0]! + 512) }
        let mlx = try MiMoV26LoadFootprint.estimateValidatedLayout(
            mlxPlan, files: mlxFiles, allocationPageBytes: 16_384)
        XCTAssertEqual(mlx.tensorCount, 243)
        XCTAssertEqual(mlx.largestExpertBlockBytes, 13_056)
        XCTAssertEqual(mlx.residentBytes, UInt64(mlxPlan.tensorBytes + 4 * 512))
    }

    // MARK: - Serial load session

    func testSessionBindsOneExactRootAndUniqueLoadGeneration() throws {
        let root = try bundle()
        let plan = try Fixture.preflight(root)
        let first = try MiMoV26SerialLoadSession(plan: plan)
        let second = try MiMoV26SerialLoadSession(plan: plan)
        let binding = first.request.binding
        XCTAssertEqual(binding, second.request.binding)
        XCTAssertNotEqual(first.request.sessionID, second.request.sessionID)
        XCTAssertEqual(try binding.fingerprint(), try second.request.binding.fingerprint())
        XCTAssertEqual(try binding.fingerprint().count, 64)
        XCTAssertEqual(binding.scope, "root-bundle-only")
        XCTAssertEqual(binding.contract, MiMoV26LoadFootprint.contract)
        XCTAssertEqual(binding.canonicalRoot, root.path)
        XCTAssertEqual(binding.tensorBytes, 976_204)
        XCTAssertEqual(binding.rootFileBytes, plan.totalFileBytes)
        XCTAssertEqual(binding.metadataBytesRead, plan.metadataBytesRead)
        XCTAssertEqual(binding.shards.map(\.name), Fixture.nativeFiles)
        XCTAssertEqual(binding.sourceRepository, "XiaomiMiMo/MiMo-V2.6-Flash-RL")
        XCTAssertNil(binding.payloadVerificationReceiptSHA256)
        XCTAssertEqual(binding.estimate, try MiMoV26LoadFootprint.estimate(plan: plan))
        XCTAssertEqual(first.request.requiredLoadBytes, binding.estimate.totalBytes)
    }

    func testForeignShortRevokedAndCancelledPermitsRejectBeforeNativeWork() throws {
        let plan = try Fixture.preflight(bundle())
        let first = try MiMoV26SerialLoadSession(plan: plan)
        let second = try MiMoV26SerialLoadSession(plan: plan)
        let foreign = Reservation(second.request)
        XCTAssertThrowsError(
            try Fixture.withScope { try first.load(reservation: foreign, retaining: $0) }
        ) { XCTAssertEqual($0 as? MiMoV26SerialLoadError, .reservationMismatch) }
        XCTAssertEqual(foreign.validations, 0)
        let short = Reservation(second.request, bytes: second.request.requiredLoadBytes - 1)
        XCTAssertThrowsError(
            try Fixture.withScope { try second.load(reservation: short, retaining: $0) }
        ) { XCTAssertEqual($0 as? MiMoV26SerialLoadError, .insufficientReservation) }
        XCTAssertEqual(short.validations, 0)
        // A failed attempt still consumes the one-shot session.
        XCTAssertThrowsError(
            try Fixture.withScope {
                try second.load(reservation: Reservation(second.request), retaining: $0)
            }
        ) { XCTAssertEqual($0 as? MiMoV26SerialLoadError, .alreadyConsumed) }
        let cancelled = try MiMoV26SerialLoadSession(plan: plan)
        let permit = Reservation(cancelled.request)
        XCTAssertThrowsError(
            try Fixture.withScope {
                try cancelled.load(reservation: permit, retaining: $0, isCancelled: { true })
            }
        ) { XCTAssertEqual($0 as? MiMoV26SerialLoadError, .cancelled) }
        XCTAssertEqual(permit.validations, 0)
        let revoked = try MiMoV26SerialLoadSession(plan: plan)
        let stale = Reservation(revoked.request)
        stale.revoked = true
        XCTAssertThrowsError(
            try Fixture.withScope { try revoked.load(reservation: stale, retaining: $0) }
        ) { XCTAssertTrue($0 is Probe) }
        XCTAssertEqual(stale.validations, 1)
    }

    func testCallbacksCannotChangeFilesSwapPermitOrPublishAfterThrow() throws {
        for mutateInReservation in [false, true] {
            let root = try bundle()
            let session = try MiMoV26SerialLoadSession(plan: Fixture.preflight(root))
            let permit = Reservation(session.request)
            var mutated = false
            func mutation() throws {
                guard !mutated else { return }
                mutated = true
                try appendByte(root.appendingPathComponent("target.safetensors"))
            }
            if mutateInReservation { permit.onValidate = { _ in try mutation() } }
            XCTAssertThrowsError(
                try Fixture.withScope { work in
                    try session.load(
                        reservation: permit, retaining: work,
                        progress: { value in
                            XCTAssertEqual(value.phase, .admitted)
                            if !mutateInReservation { try mutation() }
                        })
                }
            ) { XCTAssertTrue($0 is MiMoV26FilesystemError) }
            XCTAssertTrue(mutated)
        }
        let plan = try Fixture.preflight(bundle())
        for mode in 0 ..< 3 {
            let session = try MiMoV26SerialLoadSession(plan: plan)
            let other = try MiMoV26SerialLoadSession(plan: plan)
            let permit = Reservation(session.request)
            var callbacks = 0
            XCTAssertThrowsError(
                try Fixture.withScope { work in
                    try session.load(
                        reservation: permit, retaining: work,
                        progress: { value in
                            callbacks += 1
                            XCTAssertEqual(value.phase, .admitted)
                            switch mode {
                            case 0: permit.request = other.request
                            case 1: permit.reservedLoadBytes = 0
                            default: permit.revoked = true
                            }
                        })
                }
            ) { error in
                switch mode {
                case 0:
                    XCTAssertEqual(error as? MiMoV26SerialLoadError, .reservationMismatch)
                case 1:
                    XCTAssertEqual(error as? MiMoV26SerialLoadError, .insufficientReservation)
                default: XCTAssertTrue(error is Probe)
                }
            }
            XCTAssertEqual(callbacks, 1)
        }
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let permit = Reservation(session.request)
        XCTAssertThrowsError(
            try Fixture.withScope { work in
                try session.load(
                    reservation: permit, retaining: work, progress: { _ in throw Probe.callback })
            }
        ) { XCTAssertTrue($0 is Probe) }
        XCTAssertThrowsError(
            try Fixture.withScope { try session.load(reservation: permit, retaining: $0) }
        ) { XCTAssertEqual($0 as? MiMoV26SerialLoadError, .alreadyConsumed) }
    }

    func testSerialLoadMaterializesEverySourceAndParameterInOrder() throws {
        let session = try MiMoV26SerialLoadSession(plan: Fixture.preflight(bundle()))
        var permit: Reservation? = Reservation(session.request)
        weak var witness = permit
        var events: [MiMoV26SerialLoadProgress] = []
        var loaded: MiMoV26SerialLoadResult? = try Fixture.withScope { work in
            try session.load(
                reservation: XCTUnwrap(permit), retaining: work,
                progress: { events.append($0) })
        }
        let receipt = try XCTUnwrap(loaded?.receipt)
        XCTAssertEqual(receipt.sourceTensorCount, 207)
        XCTAssertEqual(receipt.parameterCount, 207)
        XCTAssertEqual(receipt.materializedSourcePayloadBytes, 976_204)
        XCTAssertFalse(receipt.payloadHashesRecomputed)
        XCTAssertFalse(receipt.externalComponentsLoaded)
        XCTAssertEqual(receipt.sessionID, session.request.sessionID)
        XCTAssertEqual(events.first?.phase, .admitted)
        XCTAssertEqual(events.last?.phase, .complete)
        XCTAssertTrue(events.contains { $0.phase == .sourceMaterialization })
        XCTAssertTrue(events.contains { $0.phase == .parameterMaterialization })
        XCTAssertTrue(
            zip(events, events.dropFirst()).allSatisfy {
                $0.0.materializedSourcePayloadBytes <= $0.1.materializedSourcePayloadBytes
            })
        let components = try XCTUnwrap(loaded).bundle
        let modules: [Module] = [
            components.target, components.vision, components.audioPatch, components.mtp,
        ]
        for module in modules {
            for (name, array) in module.parameters().flattened() {
                XCTAssertNotNil(try array.evaluatedBufferInfo(), name)
            }
        }
        let visual = Dictionary(uniqueKeysWithValues: components.vision.parameters().flattened())
        let patch = try XCTUnwrap(visual["patch_embed.proj.weight"])
        // [out, C, T, H, W] storage is reshaped to [out, C*T*H*W]; bytes keep their order.
        XCTAssertEqual(patch.shape, [8, 24])
        XCTAssertEqual(patch.dtype, .bfloat16)
        let first = Fixture.floats("visual.patch_embed.proj.weight", count: 1)[0]
        XCTAssertEqual(patch.asType(.float32).asArray(Float.self).first, Fixture.bfloat16(first))
        XCTAssertNoThrow(try JSONEncoder().encode(receipt))
        permit = nil
        XCTAssertNotNil(witness, "the result keeps the host permit")
        loaded = nil
        XCTAssertNil(witness, "no other owner keeps the permit after the result is released")
    }

    func testSerialLoadStopsAtSafeBoundariesAndRevalidatesAtCompletion() throws {
        let plan = try Fixture.preflight(bundle())
        for atCompletion in [false, true] {
            let session = try MiMoV26SerialLoadSession(plan: plan)
            let permit = Reservation(session.request)
            var cancel = false
            var reached = false
            XCTAssertThrowsError(
                try Fixture.withScope { work in
                    try session.load(
                        reservation: permit, retaining: work, isCancelled: { cancel },
                        progress: { value in
                            if atCompletion && value.phase == .complete {
                                reached = true
                                permit.revoked = true
                            } else if !atCompletion && value.sourceTensorsCompleted > 0 {
                                reached = true
                                cancel = true
                            }
                        })
                }
            ) { error in
                if atCompletion {
                    XCTAssertTrue(error is Probe)
                } else {
                    XCTAssertEqual(error as? MiMoV26SerialLoadError, .cancelled)
                }
            }
            XCTAssertTrue(reached)
        }
    }
}
