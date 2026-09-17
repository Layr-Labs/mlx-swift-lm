import CryptoKit
import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

/// One explicit owned artifact, using the production constructor, checkpoint
/// loader/quantization policy, PLE lease, and embedded assistant loader.
final class Qwen4RealStateFixture {
    let directory: URL
    let model: Qwen4ExpModel
    var target: Qwen4ExpTextModel { model.languageModel }
    let assistant: Qwen4ExpInlineMTPAssistant
    let dtypes: [DType]
    let chunk: Int
    let maximumLength: Int
    static let maximumOutputBudget = 9
    // Explicit test grant: the full artifact's native GDN/assistant admission
    // requires more than 1 GiB even for a one-token output. Preserve admission
    // accounting and its measured demand; this does not retune production or
    // certify the physical 128-GB hardware tier.
    let diagnosticKVCapacityBytes = 2 << 30

    init(chunk: Int = 128, maximumLength: Int = 2304) throws {
        guard (1...2048).contains(chunk), (chunk...16384).contains(maximumLength) else {
            throw Failure.invalidPrerequisite
        }
        self.chunk = chunk
        self.maximumLength = maximumLength
        let env = ProcessInfo.processInfo.environment
        let path = try XCTUnwrap(env["DARKBLOOM_QWEN4_REAL_MODEL"], "Explicit owned artifact path required")
        guard path.hasPrefix("/"), Qwen4ExpPLEResidency.useMmap,
            Qwen4ExpPLEResidency.retainCount == 0,
            env["DARKBLOOM_PREFIX_CACHE"] == "0", env["DARKBLOOM_PREFIX_CACHE_MEMORY"] == "0",
            env["DARKBLOOM_QWEN_MTP_MAX_DRAFT"] == "5" else { throw Failure.invalidPrerequisite }
        directory = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let configuration = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: configData)
        let text = configuration.textConfig
        guard base.modelType == "qwen4_exp", text.hiddenSize == 2560, text.hiddenLayers == 48,
            text.mtpNumHiddenLayers == 1, !text.pleLayerIds.isEmpty,
            text.indexerBudget == 2048, text.indexerCompressRatio == 4 else { throw Failure.invalidArtifact }
        let indexData = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        let index = try XCTUnwrap(JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let weights = try XCTUnwrap(index["weight_map"] as? [String: String])
        let profile = try Qwen4RealArtifactProfile.requested(environment: env)
        guard profile.matches(configSHA256: Self.hash(configData), indexSHA256: Self.hash(indexData),
            tensorCount: weights.count, shardCount: Set(weights.values).count,
            mtpTensorCount: weights.keys.filter { $0.hasPrefix("mtp.") }.count)
        else { throw Failure.invalidArtifact }
        for shard in Set(weights.values) {
            guard !shard.contains("/"), shard.hasSuffix(".safetensors"),
                FileManager.default.fileExists(atPath: directory.appendingPathComponent(shard).path)
            else { throw Failure.invalidArtifact }
        }
        let adoptedDirectory = directory
        guard try Qwen4ExpPLEResidency.adoptForLoadIfQwen4Exp(directory: adoptedDirectory) else { throw Failure.invalidArtifact }
        defer { Qwen4ExpPLEResidency.release(directory: adoptedDirectory) }
        let loaded = Qwen4ExpModel(configuration)
        do {
            try loadWeights(modelDirectory: directory, model: loaded,
                perLayerQuantization: base.perLayerQuantization)
            try loaded.validateExternalPLEResources()
            assistant = try Qwen4ExpInlineMTPAssistant.load(from: directory, target: loaded,
                verificationMode: .rectangular)
            model = loaded
            let adapter = CBv2SteppableLanguageModelAdapter(loaded)
            let probe = try CBv2NativeKVTypeProbe.run(model: adapter, layerKinds: loaded.cbv2LayerKinds,
                caches: loaded.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
            dtypes = probe.layerDTypes
        } catch {
            loaded.releaseExternalPLEResources()
            throw error
        }
        guard assistant.maximumDraftTokens == 5, assistant.requiredVerificationMode == .rectangular else {
            loaded.releaseExternalPLEResources()
            throw Failure.invalidPrerequisite
        }
        Self.log("loaded artifact_profile=\(profile.rawValue) config_sha256=\(Self.hash(configData)) index_sha256=\(Self.hash(indexData)) tensors=\(weights.count) shards=\(Set(weights.values).count) PLE=SSD production_default_draft_max=4 explicit_qualification_draft_max=5")
        Self.log("owned_model=Qwen3.8-Flash-Next native_qwen4 diagnostic_capacity_bytes=\(diagnosticKVCapacityBytes) host_physical_bytes=\(ProcessInfo.processInfo.physicalMemory); not a production grant or hardware-tier qualification")
    }

    func close() { model.releaseExternalPLEResources() }

    func backend(capacityBytes: Int? = nil) throws -> PagedKVBackend {
        try PagedKVBackend(layerKinds: model.cbv2LayerKinds, config: .init(
            capacityBytes: capacityBytes ?? diagnosticKVCapacityBytes,
            maxPrefillChunk: chunk, nominalMaxSequenceLength: maximumLength,
            segmentSizeBytes: 1 << 20, layerDTypes: dtypes))
    }

    func engine(mtp: Bool) async throws -> (EngineV2, PagedKVBackend) {
        // The actual engine prices overlapping captured recurrent generations
        // and assistant allocations in addition to native pages. Both arms use
        // the same bounded diagnostic grant, checked against the runtime's
        // own largest-request calculation before accepting any request.
        let backend = try backend(capacityBytes: 4 << 30)
        let engine = EngineV2(model: CBv2SteppableLanguageModelAdapter(model), layerKinds: model.cbv2LayerKinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(), schedulerConfig: .init(maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: chunk, prefillChunkSize: chunk, maxWaiting: 2, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0), mtpDrafter: mtp ? assistant : nil,
            mtpConfig: .init(enabled: mtp, maxDraftTokens: 5, maxSpeculativeBatch: 1,
                fixedDraftTokens: 5, verificationMode: .rectangular))
        let maximumRequestTokens = 37 + Self.maximumOutputBudget
        let admission = engine.admissionForTesting
        let allocated = admission.allocatedBytes(forTokens: maximumRequestTokens)
        let overhead = backend.pool.minimumSegmentedOverhead(
            tokens: maximumRequestTokens, layerKinds: model.cbv2LayerKinds) ?? Int.max
        let (needed, overflow) = allocated.addingReportingOverflow(overhead)
        Self.log("phase=engine-admission mtp=\(mtp) maximum_tokens=\(maximumRequestTokens) fixed_bytes=\(engine.resolvedFixedBytesPerRequest) allocated_bytes=\(allocated) segment_overhead_bytes=\(overhead) capacity_bytes=\(admission.admissibleBytesCapacity); diagnostic grant only")
        guard !overflow, needed <= admission.admissibleBytesCapacity,
            admission.canEverFit(promptTokens: 37, maxTokens: Self.maximumOutputBudget, additionalBackendBytes: overhead)
        else {
            await engine.shutdown()
            throw Failure.diagnosticCapacityInsufficient
        }
        return (engine, backend)
    }

    static func equal(_ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, label, file: file, line: line)
        do {
            let actualBytes = try bytes(actual, label: "\(label) actual")
            let expectedBytes = try bytes(expected, label: "\(label) expected")
            XCTAssertTrue(actualBytes == expectedBytes,
                "\(label) bytes differ: actual_sha256=\(hash(actualBytes)) expected_sha256=\(hash(expectedBytes))",
                file: file, line: line)
        } catch {
            XCTFail("\(label) cannot inspect native tensor storage: \(error)", file: file, line: line)
        }
    }

    static func bytes(_ tensor: MLXArray, label: String) throws -> Data {
        // An empty native KV snapshot has zero logical bytes and may have no
        // data pointer. MLX asDataCopy's physical-stride calculation is not a
        // valid read for that shape. Preserve shape/dtype and its exact empty
        // payload without attempting a nonexistent buffer read.
        if tensor.size == 0 {
            XCTAssertEqual(tensor.nbytes, 0, label)
            return Data()
        }
        eval(tensor)
        let metadata = try XCTUnwrap(try tensor.evaluatedBufferInfo(),
            "\(label): missing materialized storage shape=\(tensor.shape) dtype=\(tensor.dtype)")
        guard metadata.dataElements > 0 else {
            log("invalid storage \(label) shape=\(tensor.shape) dtype=\(tensor.dtype) size=\(tensor.size)")
            throw Failure.invalidTensorStorage
        }
        return tensor.asData().data
    }

    static func log(_ message: String) {
        FileHandle.standardError.write(Data("[qwen4-real-state] \(message)\n".utf8))
    }

    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    enum Failure: Error {
        case invalidArtifact, invalidPrerequisite, invalidTensorStorage, diagnosticCapacityInsufficient
    }
}

/// Request-local production paged rows plus every declared recurrent state
/// (including PLE history). The target math remains entirely in runtime code.
final class Qwen4RealPagedState {
    let fixture: Qwen4RealStateFixture
    let backend: PagedKVBackend
    let rows: [PagedSequenceKV]
    let caches: [PagedLayerCache]
    var recurrent: CBv2RecurrentRequestState
    private var closed = false

    init(_ fixture: Qwen4RealStateFixture, restoring arrays: [String: MLXArray]? = nil) throws {
        self.fixture = fixture
        backend = try fixture.backend()
        rows = try backend.makeSequenceState(layerKinds: fixture.model.cbv2LayerKinds,
            promptLength: 0, maxLength: fixture.maximumLength).map { try XCTUnwrap($0 as? PagedSequenceKV) }
        caches = backend.makeLayerCaches()
        recurrent = try CBv2RecurrentRequestState(spec: fixture.model.cbv2RecurrentStateSpec)
        if let arrays {
            for (i, row) in rows.enumerated() {
                let keys = try XCTUnwrap(arrays["kv.\(i).keys"]), values = try XCTUnwrap(arrays["kv.\(i).values"])
                _ = row.update(keys: keys, values: values)
                let indexKeys = try XCTUnwrap(arrays["qsa.\(i).keys"])
                let positions = try XCTUnwrap(arrays["qsa.\(i).positions"])
                let pooled = arrays["qsa.\(i).pooled"]
                try row.restoreQwen4Indexer(.init(tokenCount: keys.dim(2), indexKeys: indexKeys,
                    positionIds: positions, pooledIndexKeys: pooled, pooledIndexBlocks: pooled?.dim(1) ?? 0))
            }
            var layers: [Int: CBv2RecurrentLayerState] = [:]
            for spec in fixture.model.cbv2RecurrentStateSpec.layers {
                let i = spec.modelLayerIndex
                layers[i] = .init(conv: try XCTUnwrap(arrays["recurrent.\(i).conv"]),
                    ssm: try XCTUnwrap(arrays["recurrent.\(i).ssm"]))
            }
            recurrent = try CBv2RecurrentRequestState(spec: fixture.model.cbv2RecurrentStateSpec, adoptedCommitted: layers)
        }
        for (cache, row) in zip(caches, rows) { cache.setRows([row]) }
    }

    func close() throws {
        guard !closed else { return }
        for cache in caches { cache.setRows([]) }
        backend.release(rows.map { $0 as CBv2SequenceKV? })
        try recurrent.release()
        closed = true
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(backend.bytesWired, 0)
    }

    func forward(_ tokens: [Int], captured: Bool = false, keep: Int? = nil)
        throws -> (logits: MLXArray, hidden: MLXArray) {
        for (cache, row) in zip(caches, rows) { cache.setRows([row]) }
        let input = MLXArray(tokens.map(Int32.init), [1, tokens.count])
        let transaction = try recurrent.bind()
        let fill = CBv2DeferredHostFill.open()
        defer { fill.close(); fill.run() }
        if captured {
            for row in rows { row.beginSpeculativeWrite() }
        }
        for cache in caches { cache.mtpSerializesRectangularAttention = captured }
        let output = captured
            ? fixture.target.cbv2ForwardWithHiddenCaptured(input, caches: caches, recurrentState: [transaction], positionIds: nil)
            : fixture.target.cbv2ForwardWithHidden(input, caches: caches, recurrentState: [transaction], positionIds: nil)
        let roots = try transaction.evaluate()
        fill.close()
        fill.run()
        eval([output.logits, output.lastHidden] + roots + caches.flatMap { $0.innerState() })
        if captured {
            let retained = keep ?? tokens.count
            if retained == 0 { try transaction.rollback() }
            else { try transaction.commit(keepPositions: retained) }
            for (cache, row) in zip(caches, rows) {
                CBv2Qwen4IndexerBind.harvest(cache, into: row)
                row.rollback(tokens.count - retained)
                row.commitSpeculativeWrite()
                try row.trimQwen4Indexer(to: row.absoluteOffset, compressRatio: fixture.target.configuration.indexerCompressRatio)
                // Detach the rejected cache sidecar before rebinding; it must
                // not overwrite the already-trimmed row in setRows().
                _ = CBv2Qwen4IndexerBind.restore(cache, from: row)
                cache.setRows([row])
                cache.mtpSerializesRectangularAttention = false
            }
        } else { try transaction.commit() }
        return (output.logits, output.lastHidden)
    }

    func snapshot() throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (i, row) in rows.enumerated() {
            let kv = row.snapshot(), side = try row.snapshotQwen4Indexer()
            result["kv.\(i).keys"] = kv.keys; result["kv.\(i).values"] = kv.values
            result["qsa.\(i).keys"] = side.indexKeys; result["qsa.\(i).positions"] = side.positionIds
            result["qsa.\(i).pooled"] = side.pooledIndexKeys
        }
        let layers = try XCTUnwrap(recurrent.confirmedStateSnapshot())
        for (i, layer) in layers {
            result["recurrent.\(i).conv"] = layer.conv
            result["recurrent.\(i).ssm"] = layer.ssm
        }
        eval(Array(result.values))
        return result
    }
}
