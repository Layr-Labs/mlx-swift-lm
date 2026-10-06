import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

/// Header cases do not invoke MLX readers. Native-reader cases require root's
/// explicit opt-in and use only the complete193-tensor synthetic checkpoint.

private func withMiMoConstructionScope<Value>(
    _ body: (NativeConstructionScope) throws -> Value
) rethrows -> Value {
    let work = NativeConstructionScope()
    defer {
        // Unexpected failed completion is restart-only, including in this
        // dedicated native test process. Never deallocate its sole SDK owner.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    return try body(work)
}

final class MiMoV26FilesystemWeightsTests: XCTestCase {
    private let limits = MiMoV26FilesystemLimits(
        maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304)

    func testBoundedPreflightCloses193TensorsAndPreservesMetadata() throws {
        let root = try fixture(copy: false)
        let plan = try preflight(root, expectations: expectations(includeObjects: true))
        XCTAssertEqual(plan.bundlePlan.descriptors.count, 193)
        XCTAssertEqual(plan.bundlePlan.tensorBytes, 370672)
        XCTAssertEqual(plan.totalFileBytes, 390016)
        XCTAssertEqual(plan.metadataBytesRead, 45684)
        XCTAssertEqual(plan.shards.count, 4)
        XCTAssertTrue(
            plan.shards.allSatisfy {
                Set($0.tensorKeysInFileOrder) == $0.tensorKeys
                    && $0.tensorKeysInFileOrder.count == $0.tensorKeys.count
            })
        XCTAssertEqual(plan.shards.reduce(0) { $0 + $1.tensorBytes }, 370672)
        XCTAssertEqual(plan.bundlePlan.configuration.attentionProjectionLayout, "fused_qkv")
        XCTAssertEqual(plan.configurationURL.lastPathComponent, "config.json")
        XCTAssertEqual(plan.indexURL.lastPathComponent, "model.safetensors.index.json")
        XCTAssertEqual(plan.tokenizerURL.lastPathComponent, "tokenizer.json")
        XCTAssertEqual(plan.tokenizerConfigurationURL.lastPathComponent, "tokenizer_config.json")
        try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan)
        XCTAssertTrue(plan.shards.allSatisfy { $0.suppliedPayloadSHA256?.count == 64 })
    }

    func testCPUHeaderRejectsWrongShapeDTypeOffsetsAndUnmappedKeys() throws {
        for variant in [
            "shape", "dtype", "offset", "negativeOffset", "positiveGap", "missing", "extra",
        ] {
            let root = try fixture()
            let file = root.appendingPathComponent("target.safetensors")
            try mutateHeader(file) { header in
                let key =
                    variant == "positiveGap"
                    ? "model.layers.0.self_attn.q_proj.weight" : "model.norm.weight"
                let original = header[key] as! [String: Any]
                switch variant {
                case "missing": header.removeValue(forKey: key)
                case "extra": header["unexpected.weight"] = original
                default:
                    var entry = original
                    if variant == "shape" { entry["shape"] = [1] }
                    if variant == "dtype" { entry["dtype"] = "U32" }
                    if variant == "offset" { entry["data_offsets"] = [0, 256] }
                    if variant == "negativeOffset" { entry["data_offsets"] = [-1, 255] }
                    if variant == "positiveGap" {
                        let offsets = original["data_offsets"] as! [Int]
                        entry["data_offsets"] = [offsets[0] + 4, offsets[1] + 4]
                    }
                    header[key] = entry
                }
            }
            XCTAssertThrowsError(try preflight(root), variant) { error in
                if variant == "positiveGap" {
                    XCTAssertEqual(
                        error as? MiMoV26FilesystemError,
                        .invalidHeader("overlap/gap: target.safetensors"))
                }
            }
        }
    }

    func testCPUMissingExtraPathSymlinkAndTruncationFail() throws {
        let missing = try fixture()
        try FileManager.default.removeItem(at: missing.appendingPathComponent("vision.safetensors"))
        XCTAssertThrowsError(try preflight(missing))
        let extra = try fixture()
        try Data([0]).write(to: extra.appendingPathComponent("unindexed.bin"))
        XCTAssertThrowsError(try preflight(extra))
        let unsafe = try fixture()
        let indexURL = unsafe.appendingPathComponent("model.safetensors.index.json")
        var index = try object(Data(contentsOf: indexURL))
        var map = index["weight_map"] as! [String: String]
        map["model.norm.weight"] = "../outside.safetensors"
        index["weight_map"] = map
        try JSONSerialization.data(withJSONObject: index).write(to: indexURL)
        XCTAssertThrowsError(try preflight(unsafe))
        let symlink = try fixture()
        let file = symlink.appendingPathComponent("target.safetensors")
        let outside = symlink.appendingPathComponent("retained.target")
        try FileManager.default.moveItem(at: file, to: outside)
        try FileManager.default.createSymbolicLink(at: file, withDestinationURL: outside)
        XCTAssertThrowsError(try preflight(symlink))
        let truncated = try fixture()
        let h = try FileHandle(forWritingTo: truncated.appendingPathComponent("target.safetensors"))
        try h.truncate(atOffset: 16)
        try h.close()
        XCTAssertThrowsError(try preflight(truncated))
    }

    func testCPUHeaderLengthDuplicateJSONAndBudgetsFail() throws {
        let badLength = try fixture()
        let h = try FileHandle(forWritingTo: badLength.appendingPathComponent("target.safetensors"))
        try h.write(contentsOf: Data(repeating: 255, count: 8))
        try h.close()
        XCTAssertThrowsError(try preflight(badLength))
        let duplicate = try fixture()
        let c = duplicate.appendingPathComponent("config.json")
        let original = try Data(contentsOf: c)
        var data = Data("{\"model_type\":\"mimo_v2\",".utf8)
        data.append(original.dropFirst())
        try data.write(to: c)
        XCTAssertThrowsError(try preflight(duplicate))
        let root = try fixture(copy: false)
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(),
                limits: .init(maximumShardBytes: 10, maximumTotalFileBytes: 4_194_304)))
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(),
                limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 10)))
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(),
                limits: .init(
                    maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304,
                    maximumConfigurationBytes: 8)))
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(),
                limits: .init(
                    maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304,
                    maximumHeaderBytes: 8)))
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(),
                limits: .init(
                    maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304,
                    maximumMetadataBytes: 32)))
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(),
                limits: .init(
                    maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304, maximumShards: 1
                )))
    }

    func testCPUEvidenceAndMutationChecks() throws {
        let root = try fixture(copy: false)
        XCTAssertThrowsError(
            try preflight(
                root, expectations: .init(configurationSHA256: String(repeating: "0", count: 64))))
        XCTAssertThrowsError(
            try preflight(root, expectations: .init(files: ["missing.safetensors": .init()])))
        XCTAssertThrowsError(
            try preflight(
                root,
                expectations: .init(files: [
                    "target.safetensors": .init(headerSHA256: String(repeating: "0", count: 64))
                ])))
        // A supplied payload digest is intentionally descriptive; preflight
        // must not pretend to compare it without reading the tensor payload.
        let descriptive = try preflight(
            root,
            expectations: .init(files: [
                "target.safetensors": .init(payloadSHA256: String(repeating: "0", count: 64))
            ]))
        XCTAssertEqual(
            descriptive.shards.first { $0.name == "target.safetensors" }?.suppliedPayloadSHA256,
            String(repeating: "0", count: 64))
        let changed = try fixture()
        let plan = try preflight(changed)
        try appendByte(changed.appendingPathComponent("target.safetensors"))
        XCTAssertThrowsError(try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan))
        let laterExtra = try fixture()
        let extraPlan = try preflight(laterExtra)
        try Data([1]).write(to: laterExtra.appendingPathComponent("extra.safetensors"))
        XCTAssertThrowsError(try MiMoV26FilesystemWeights.validateCurrentObjects(plan: extraPlan))
        // A retained external component is outside root-index traversal.
        let external = try fixture()
        try FileManager.default.createDirectory(
            at: external.appendingPathComponent("dflash"), withIntermediateDirectories: false)
        try Data([1]).write(to: external.appendingPathComponent("dflash/model.safetensors"))
        XCTAssertNoThrow(try preflight(external))
    }

    func testCPUCancellationBetweenMetadataStages() throws {
        let root = try fixture(copy: false)
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(), limits: limits, isCancelled: { true })
        ) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .cancelled)
        }
        var cancel = false
        var headers = 0
        XCTAssertThrowsError(
            try MiMoV26FilesystemWeights.preflight(
                root: root, provenance: provenance(), limits: limits,
                isCancelled: { cancel },
                progress: { p in
                    if p.phase == .headerRead {
                        headers += 1
                        cancel = true
                    }
                })
        ) {
            XCTAssertEqual($0 as? MiMoV26FilesystemError, .cancelled)
        }
        XCTAssertEqual(headers, 1)
    }

    func testNativeReaderReturnsCompleteLazyBundle() throws {
        try nativeLane()
        let root = try fixture(copy: false)
        let plan = try preflight(root, expectations: expectations(includeObjects: true))
        var events: [MiMoV26FilesystemProgress] = []
        var result = try withMiMoConstructionScope { constructionWork in
            try MiMoV26FilesystemWeights.load(
                plan: plan, retaining: constructionWork, progress: { events.append($0) })
        }
        XCTAssertEqual(events.filter { $0.phase == .shardHandlesCreated }.count, 4)
        XCTAssertEqual(result.receipt.returnedHandleTensorBytes, 370672)
        XCTAssertEqual(result.receipt.returnedHandleFileBytes, 390016)
        XCTAssertTrue(result.receipt.payloadDisposition.contains("lazy"))
        XCTAssertTrue(result.bundle.mtp.isLoaded)
        // Explicit tiny-fixture materialization is a separate test action,
        // not implicit evaluation by the loader itself.
        XCTAssertEqual(result.materializationHandleCount, 193)
        let handles = result.takeMaterializationHandles()
        XCTAssertEqual(result.materializationHandleCount, 0)
        XCTAssertEqual(handles.map(\.name), plan.shards.flatMap(\.tensorKeysInFileOrder))
        XCTAssertEqual(handles.reduce(0) { $0 + $1.byteCount }, 370672)
        for handle in handles { eval(handle.array) }
        XCTAssertEqual(result.takeMaterializationHandles().count, 0)
        try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan)
        let target = Dictionary(uniqueKeysWithValues: result.bundle.target.parameters().flattened())
        let expert = try XCTUnwrap(target["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(expert.dtype, .uint32)
        XCTAssertEqual(expert.asArray(UInt32.self).first, 0x2222_2222)
        let scales = try XCTUnwrap(target["model.layers.1.mlp.switch_mlp.gate_proj.scales"])
        XCTAssertEqual(scales.dtype, .uint8)
        XCTAssertEqual(scales.asArray(UInt8.self).first, 127)
        XCTAssertNoThrow(try JSONEncoder().encode(result.receipt))
    }

    func testNativeCancellationAndMutationDoNotPublishPartialBundle() throws {
        try nativeLane()
        let root = try fixture()
        let plan = try preflight(root)
        var cancel = false
        var constructed = false
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26FilesystemWeights.load(
                    plan: plan, retaining: constructionWork, isCancelled: { cancel },
                    progress: { p in
                        if p.phase == .shardHandlesCreated { cancel = true }
                        if p.phase == .bundleConstructed { constructed = true }
                    })
            }
        ) { XCTAssertEqual($0 as? MiMoV26FilesystemError, .cancelled) }
        XCTAssertFalse(constructed)
        let changed = try fixture()
        let changedPlan = try preflight(changed)
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26FilesystemWeights.load(
                    plan: changedPlan, retaining: constructionWork,
                    progress: { p in
                        if p.phase == .willCreateShardHandles, let name = p.currentFile {
                            try self.appendByte(changed.appendingPathComponent(name))
                        }
                    })
            })
    }

    private func directory() throws -> URL {
        guard let value = ProcessInfo.processInfo.environment["MIMO_V26_FILESYSTEM_FIXTURES"] else {
            throw XCTSkip("Set MIMO_V26_FILESYSTEM_FIXTURES to owned synthetic fixture directory")
        }
        return URL(fileURLWithPath: value)
    }
    private func fixture(copy: Bool = true) throws -> URL {
        let directory = try directory()
        let original = directory.appendingPathComponent("tiny-bundle")
        if !copy { return original }
        let target = directory.appendingPathComponent("test-work-" + UUID().uuidString)
        try FileManager.default.copyItem(at: original, to: target)
        // Retain bounded unique test roots for failure inspection.
        return target
    }
    private func provenance() throws -> MiMoV26ConvertedProvenance {
        let p = try object(Data(contentsOf: directory().appendingPathComponent("provenance.json")))
        return try .init(
            artifactID: XCTUnwrap(p["artifactID"] as? String),
            sourceRepository: XCTUnwrap(p["sourceRepository"] as? String),
            sourceRevision: XCTUnwrap(p["sourceRevision"] as? String),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"] as? String))
    }
    private func expectations(includeObjects: Bool) throws -> MiMoV26FilesystemExpectations {
        let data = try Data(contentsOf: directory().appendingPathComponent("expectations.json"))
        let value = try JSONDecoder().decode(MiMoV26FilesystemExpectations.self, from: data)
        if includeObjects { return value }
        return .init(
            configurationSHA256: value.configurationSHA256, indexSHA256: value.indexSHA256,
            files: value.files.mapValues {
                .init(headerSHA256: $0.headerSHA256, payloadSHA256: $0.payloadSHA256)
            })
    }
    private func preflight(_ root: URL, expectations: MiMoV26FilesystemExpectations = .init())
        throws -> MiMoV26FilesystemLoadPlan
    {
        try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance(), limits: limits, expectations: expectations)
    }
    private func object(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
    private func appendByte(_ url: URL) throws {
        let h = try FileHandle(forWritingTo: url)
        defer { try? h.close() }
        try h.seekToEnd()
        try h.write(contentsOf: Data([0]))
    }
    private func mutateHeader(_ url: URL, _ mutation: (inout [String: Any]) throws -> Void) throws {
        let data = try Data(contentsOf: url)
        let rawLength = Array(data.prefix(8)).enumerated().reduce(UInt64(0)) {
            $0 | (UInt64($1.element) << ($1.offset * 8))
        }
        let length = Int(rawLength)
        var header = try object(data.subdata(in: 8 ..< (8 + length)))
        try mutation(&header)
        var raw = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        raw.append(Data(repeating: 32, count: (8 - raw.count % 8) % 8))
        let n = UInt64(raw.count)
        var result = Data((0 ..< 8).map { UInt8((n >> ($0 * 8)) & 255) })
        result.append(raw)
        result.append(data.dropFirst(8 + length))
        try result.write(to: url)
    }
    private func nativeLane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_FILESYSTEM_NATIVE_TESTS"] == "1" else {
            throw XCTSkip(
                "Requires coordinator-owned native lane and MIMO_V26_FILESYSTEM_NATIVE_TESTS=1")
        }
    }
}
