import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Uses the real target, actual main pool and admission, and the ordinary
    /// bounded producer/consumer. This is not encrypted-store qualification.
    @Suite(.serialized)
    struct GLM5NextCompleteCheckpointTests {
        private static let identity = CBv2CompleteCheckpointIdentity(
            modelAggregateHash: "glm-tiny-artifact", promptContractID: "glm-tiny-template",
            buildID: "glm-tiny-native-build", numericsFingerprint: "glm-exact-fp32")

        private final class ProcessBudget: CBv2ProcessMemoryOwner, @unchecked Sendable {
            enum Failure: Error { case exceeded }
            private let lock = NSLock()
            private var charge: UInt64 = 0
            private var coverage: UInt64 = 0
            private var retired = false
            func replaceCharge(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes <= (512 << 20), bytes >= coverage,
                        !retired || bytes <= charge
                    else { throw Failure.exceeded }
                    charge = bytes
                }
            }
            func recordMaterialization(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard !retired, bytes >= coverage, bytes <= charge else {
                        throw Failure.exceeded
                    }
                    coverage = bytes
                }
            }
            func withdrawCoverage(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes <= coverage else { throw Failure.exceeded }
                    coverage -= bytes
                }
            }
            func retire() { lock.withLock { retired = true } }
        }

        private final class Counter: @unchecked Sendable {
            private let lock = NSLock()
            private var count = 0
            var value: Int { lock.withLock { count } }
            func increment() { lock.withLock { count += 1 } }
        }

        private func model(layers: Int = 4) throws -> GLM5NextTextModel {
            let encoded = try JSONEncoder().encode(
                GLM5NextInventoryFixture.configuration(layers: layers))
            var root = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            // Exercise asymmetric page-native storage at its supported minimum
            // latent width, without inflating the actual KDA/hidden geometry.
            text["kv_lora_rank"] = 64
            root["text_config"] = text
            let config = try JSONDecoder().decode(
                GLM5NextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root))
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "checkpoint-tiny.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
            return try GLM5NextTextModel(configuration: config.textConfig, inventory: inventory) {
                name in
                try Self.tensor(name, inventory: inventory)
            }
        }

        private static func tensor(_ name: String, inventory: GLM5NextWeightInventory) throws
            -> MLXArray
        {
            let shape = try #require(inventory.entries[name]?.tensor.shape)
            let count = shape.reduce(1, *)
            let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 17 }
            let values: [Float]
            if name.contains("norm.weight") || name.hasSuffix("_scale") {
                values = Array(repeating: 1, count: count)
            } else if name.hasSuffix("A_log") || name.hasSuffix("dt_bias") {
                values = Array(repeating: -0.5, count: count)
            } else if name.hasSuffix(".bias") || name.hasSuffix("_base") {
                values = Array(repeating: 0, count: count)
            } else {
                values = (0 ..< count).map { Float(($0 * 3 + salt) % 19 - 9) * 0.015 }
            }
            return MLXArray(values, shape)
        }

        private final class Runtime {
            let model: GLM5NextTextModel
            let backend: PagedKVBackend
            let engine: EngineV2
            let bank: CBv2LayerCacheBank
            let codec: CBv2CompleteCheckpointCodec
            let process: ProcessBudget
            var rows: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
            var recurrence: [CBv2RequestID: CBv2RecurrentRequestState] = [:]

            init(
                model: GLM5NextTextModel, checkpointQuantization: PagedKVQuantizationConfig? = nil
            ) throws {
                let process = ProcessBudget()
                let backend = try model.makeCBv2PagedBackend(
                    kvBytesCapacity: 512 << 20, maximumBatchSize: 2, maximumQueryTokens: 32,
                    maximumContextTokens: 128, maximumScratchBytes: 512 << 20,
                    segmentBytes: 64 << 10, maximumBufferBytes: 16 << 20)
                let bank = CBv2LayerCacheBank(caches: backend.makeLayerCaches())
                let engine = EngineV2(
                    model: CBv2SteppableLanguageModelAdapter(model),
                    layerKinds: model.cbv2LayerKinds, backend: backend, cacheProvider: bank,
                    sampler: CBv2GreedySampler(),
                    schedulerConfig: .init(
                        maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                        prefillChunkSize: 4, maxWaiting: 4, enablePrefixCache: false),
                    admissionConfig: model.cbv2NativeAdmissionConfig, processMemoryOwner: process)
                self.model = model
                self.process = process
                self.backend = backend
                self.bank = bank
                self.engine = engine
                self.codec = CBv2CompleteCheckpointCodec(
                    identity: GLM5NextCompleteCheckpointTests.identity,
                    layerKinds: model.cbv2LayerKinds, recurrentSpec: model.cbv2RecurrentStateSpec,
                    kvDTypes: Array(repeating: .float32, count: model.cbv2LayerKinds.count),
                    assistant: nil, admission: engine.admissionForTesting,
                    pagedConfig: backend.pool.config,
                    checkpointQuantization: checkpointQuantization, glm5NextBackend: backend)
            }

            func make(_ id: CBv2RequestID) throws {
                try engine.admissionForTesting.reserve(id: id, additionalTokens: 128)
                do {
                    let state = try backend.makeSequenceState(
                        layerKinds: model.cbv2LayerKinds,
                        promptLength: 0, maxLength: 128)
                    rows[id] = state
                    recurrence[id] = try CBv2RecurrentRequestState(
                        spec: model.cbv2RecurrentStateSpec)
                } catch {
                    if let state = rows.removeValue(forKey: id) { backend.release(state) }
                    engine.admissionForTesting.releaseAll(id: id)
                    throw error
                }
            }

            func forward(_ id: CBv2RequestID, tokens: [Int]) throws -> GLM5NextTextOutput {
                let state = try #require(rows[id])
                let recurrent = try #require(recurrence[id]).bind()
                let offset = try #require(state.first.flatMap { $0?.absoluteOffset })
                let preparedWork = try backend.prepareAttentionWork(
                    assignments: [(id, offset ..< (offset + tokens.count))], states: [id: state])
                let work = try #require(preparedWork)
                let boundary = CBv2PagedWriteBoundary(pool: backend.pool)
                let outcome = Result {
                    let caches = bank.layerCaches(rowStates: [state])
                    let result = try model.prepareCBv2Forward(
                        tokens: MLXArray(tokens.map(Int32.init), [1, tokens.count]),
                        caches: caches.map { $0 as! any KVCache }, recurrentState: [recurrent])
                    let roots = try recurrent.evaluate()
                    try work.seal()
                    work.publish()
                    backend.pool.endAttentionConstruction(work)
                    try withError { fault in
                        eval(roots + [result.logits, result.hidden])
                        try work.finishEvaluation()
                        try fault.check()
                    }
                    try recurrent.commit()
                    return result
                }
                if case .failure = outcome {
                    backend.pool.endAttentionConstruction(work)
                    do {
                        try withError { fault in
                            work.constructionStream.synchronize()
                            Stream.cpu.synchronize()
                            try fault.check()
                        }
                        boundary.discardFailedGraphAfterSynchronization()
                        bank.releaseBoundRows()
                        try work.discardAfterDrain()
                    } catch {
                        work.failCompletion()
                        throw error
                    }
                }
                bank.releaseBoundRows()
                work.closeGraph()
                return try outcome.get()
            }

            func close() {
                bank.releaseBoundRows()
                let unfinished = Set(
                    backend.pool.attentionWorkOwners.values
                        .filter { !$0.released }.flatMap { $0.descriptors.keys.map(\.serial) })
                #expect(unfinished.isEmpty)
                let safe = rows.filter { _, state in
                    state.compactMap { $0 as? PagedSequenceKV }.allSatisfy {
                        !unfinished.contains($0.serial)
                    }
                }
                for (id, state) in safe {
                    try? recurrence.removeValue(forKey: id)?.release()
                    backend.release(state)
                    engine.admissionForTesting.releaseAll(id: id)
                    rows.removeValue(forKey: id)
                }
            }
        }

        private func copy(
            _ source: CBv2CompleteCheckpointExport,
            to sink: CBv2CompleteCheckpointImport, segmentBytes: Int = 28
        ) throws {
            for (index, descriptor) in source.manifest.tensors.enumerated() {
                var offset = 0
                while offset < descriptor.byteCount {
                    let data = try source.readSegment(
                        tensorIndex: index, byteOffset: offset,
                        maximumBytes: segmentBytes)
                    #expect(!data.isEmpty && data.count <= segmentBytes)
                    try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: data)
                    offset += data.count
                }
            }
        }

        private func export(
            _ runtime: Runtime, id: CBv2RequestID, position: Int,
            prompt: [Int]
        ) throws
            -> (CBv2CompleteCheckpointExport, CBv2CapturedCompleteCheckpoint)
        {
            let layers = try #require(runtime.recurrence[id]?.confirmedStateSnapshot())
            // Use the actual complete-boundary capture path, not a token-count
            // object pretending to contain saved recurrent tensors.
            let capture = CBv2CompleteCheckpointCapture(
                codec: runtime.codec, store: CompleteCheckpointFixtureStore())
            let roots = capture.capture(
                requestID: id, position: position, chunkSize: position,
                layers: layers, assistantState: nil, rowStates: try #require(runtime.rows[id]))
            try withError { eval(roots) }
            let retained = try #require(capture.staged[id]?.first)
            let checkpoint = try #require(retained.checkpoint)
            return (
                try runtime.codec.export(
                    checkpoint: checkpoint,
                    state: try #require(runtime.rows[id]), tokens: prompt, cacheSalt: "tenant-a"),
                retained
            )
        }

        @Test(arguments: [3, 4, 5, 15, 16, 17])
        func realTargetRoundTripsFullStateAcrossPageAndPoolBoundaries(position: Int) async throws {
            let runtime = try Runtime(model: model(layers: position == 4 ? 45 : 4))
            let outcome = Result {
                try runtime.engine.loopForTesting.onEngineQueueSync {
                    let donor = CBv2RequestID(601)
                    let adopter = CBv2RequestID(602)
                    try runtime.make(donor)
                    let prefix = (0 ..< position).map { 4 + $0 % 20 }
                    _ = try runtime.forward(donor, tokens: prefix)
                    let (source, capture) = try export(
                        runtime, id: donor, position: position,
                        prompt: prefix + [7, 8])
                    defer {
                        source.close()
                        capture.closeAfterCompletedEvaluation()
                    }
                    let encoded = try JSONEncoder().encode(source.manifest)
                    let manifest = try JSONDecoder().decode(
                        CBv2CompleteCheckpointManifest.self, from: encoded)
                    #expect(
                        manifest.backendLayout == CBv2CompleteCheckpointManifest.glm5NextPagedLayout
                    )
                    #expect(manifest.glm5NextState?.physicalPosition == position)
                    #expect(manifest.glm5NextState?.acceptedFrontier == position)
                    #expect(
                        manifest.tensors.filter { $0.role == .convolution }.count
                            == runtime.model.cbv2RecurrentStateSpec.layers.count)
                    #expect(
                        manifest.tensors.filter { $0.role == .recurrent }.allSatisfy {
                            $0.dtype == .float32
                        })
                    let request = CBv2Request(
                        id: adopter, promptTokens: prefix + [7, 8], maxTokens: 128 - position - 2,
                        cacheSalt: "tenant-a", prefixCacheReceiptID: .init(1602))
                    let plan = try runtime.codec.plan(manifest: manifest, request: request)
                    #expect(plan.pagedStoragePlan != nil && plan.nativeAuxiliaryBytes > 0)
                    #expect(plan.nativeTargetBytes > 0 && plan.checkpointHostBytes > 0)
                    let sink = try plan.allocate(onRelease: {})
                    defer { sink.close() }
                    try copy(source, to: sink)
                    let staged = try sink.finish()
                    defer { staged.close() }
                    runtime.rows[adopter] = try runtime.codec.adoptGLM5Next(
                        staged, backend: runtime.backend,
                        requestID: adopter,
                        restore: { checkpoint in
                            runtime.recurrence[adopter] = try CBv2RecurrentRequestState(
                                spec: runtime.model.cbv2RecurrentStateSpec,
                                adoptedCommitted: checkpoint.layers)
                        })
                    #expect(
                        runtime.rows[adopter]?.allSatisfy { $0?.absoluteOffset == position } == true
                    )
                    let donorState = try #require(
                        runtime.recurrence[donor]?.confirmedStateSnapshot())
                    let adoptedState = try #require(
                        runtime.recurrence[adopter]?.confirmedStateSnapshot())
                    for index in donorState.keys {
                        #expect(
                            donorState[index]?.conv?.asArray(Float.self).map(\.bitPattern)
                                == adoptedState[index]?.conv?.asArray(Float.self).map(\.bitPattern))
                        #expect(
                            donorState[index]?.ssm?.asArray(Float.self).map(\.bitPattern)
                                == adoptedState[index]?.ssm?.asArray(Float.self).map(\.bitPattern))
                    }
                    let expected = try runtime.forward(donor, tokens: [7, 8])
                    let restored = try runtime.forward(adopter, tokens: [7, 8])
                    #expect(abs(expected.logits - restored.logits).max().item(Float.self) < 0.00001)
                    #expect(abs(expected.hidden - restored.hidden).max().item(Float.self) < 0.00001)
                    #expect(runtime.backend.pool.attentionWorkBytesReserved == 0)
                }
            }
            runtime.engine.loopForTesting.onEngineQueueSync { runtime.close() }
            await runtime.engine.shutdown()
            try outcome.get()
        }

        /// Every byte of one tensor of an export.
        private func read(_ source: CBv2CompleteCheckpointExport, tensor: Int) throws -> Data {
            var data = Data()
            while data.count < source.manifest.tensors[tensor].byteCount {
                data.append(
                    try source.readSegment(
                        tensorIndex: tensor, byteOffset: data.count, maximumBytes: 4 << 20))
            }
            return data
        }

        private func floats(_ data: Data) -> [Float] {
            data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        /// The checkpoint profile codes only the older latent rows. Index,
        /// gate and validity rows, the exact recent latent band and the KDA
        /// state restore bit for bit; older latent rows restore to the
        /// profile's reference reconstruction, and the adopter continues.
        @Test(arguments: [17, 33])
        func realTargetRoundTripsLossyLatentRowsBesideExactState(position: Int) async throws {
            let profile = PagedKVQuantizationConfig(recentTokenCount: 8)
            let runtime = try Runtime(model: model(), checkpointQuantization: profile)
            #expect(runtime.codec.checkpointQuantization == profile)
            #expect(
                runtime.codec.backendLayout
                    == CBv2CompleteCheckpointManifest.glm5NextQuantizedPagedLayout)
            let outcome = Result {
                try runtime.engine.loopForTesting.onEngineQueueSync {
                    let donor = CBv2RequestID(801)
                    try runtime.make(donor)
                    let prefix = (0 ..< position).map { 4 + $0 % 20 }
                    for start in stride(from: 0, to: position, by: 16) {
                        _ = try runtime.forward(
                            donor, tokens: Array(prefix[start ..< min(start + 16, position)]))
                    }
                    let prompt = prefix + [7, 8]
                    let (lossy, capture) = try export(
                        runtime, id: donor, position: position, prompt: prompt)
                    defer {
                        lossy.close()
                        capture.closeAfterCompletedEvaluation()
                    }
                    let exactCodec = CBv2CompleteCheckpointCodec(
                        identity: Self.identity, layerKinds: runtime.model.cbv2LayerKinds,
                        recurrentSpec: runtime.model.cbv2RecurrentStateSpec,
                        kvDTypes: runtime.codec.kvDTypes, assistant: nil,
                        admission: runtime.engine.admissionForTesting,
                        pagedConfig: runtime.backend.pool.config,
                        glm5NextBackend: runtime.backend)
                    let exact = try exactCodec.export(
                        checkpoint: try #require(capture.checkpoint),
                        state: try #require(runtime.rows[donor]), tokens: prompt,
                        cacheSalt: "tenant-a")
                    defer { exact.close() }
                    #expect(
                        exact.manifest.backendLayout
                            == CBv2CompleteCheckpointManifest.glm5NextPagedLayout)

                    let manifest = try JSONDecoder().decode(
                        CBv2CompleteCheckpointManifest.self,
                        from: JSONEncoder().encode(lossy.manifest))
                    #expect(manifest == lossy.manifest)
                    #expect(manifest.checkpointQuantization == profile)
                    #expect(manifest.checkpointNativeDTypes?.allSatisfy { $0 == .float32 } == true)
                    #expect(manifest.tensors.count == exact.manifest.tensors.count)
                    var keys: [Int] = []
                    for (index, (wire, native)) in zip(manifest.tensors, exact.manifest.tensors)
                        .enumerated()
                    {
                        #expect(wire.role == native.role && wire.layer == native.layer)
                        if wire.role == .keys {
                            keys.append(index)
                            #expect(wire.dtype == .uint8 && wire.byteCount < native.byteCount)
                        } else {
                            #expect(wire == native)
                        }
                    }
                    #expect(keys.count == runtime.model.cbv2LayerKinds.count)

                    func stage() throws -> CBv2StagedCompleteCheckpoint {
                        let request = CBv2Request(
                            id: .init(802), promptTokens: prompt,
                            maxTokens: 128 - position - 2, cacheSalt: "tenant-a",
                            prefixCacheReceiptID: .init(1802))
                        let plan = try runtime.codec.plan(manifest: manifest, request: request)
                        #expect(plan.scratchBytes >= CBv2NativeCheckpointRowCodec.scratchBytes)
                        let sink = try plan.allocate(onRelease: {})
                        defer { sink.close() }
                        try copy(lossy, to: sink, segmentBytes: 333)
                        return try sink.finish()
                    }

                    // The restored pages, read back as exact FP32 rows.
                    let restored = try stage()
                    defer { restored.close() }
                    try restored.consumePreparedState { prepared in
                        let owner = try #require(prepared.pagedFrame).consume()
                        defer { owner.close() }
                        for (layer, tensor) in keys.enumerated() {
                            for values in [false, true] {
                                let source = try CBv2PagedCheckpointTensorSource(
                                    storage: owner.storage, layerIndex: layer, values: values,
                                    admission: runtime.engine.admissionForTesting)
                                defer { source.close() }
                                var bytes = Data()
                                while bytes.count < source.byteCount {
                                    bytes.append(
                                        try source.readSegment(
                                            byteOffset: bytes.count, maximumBytes: 4 << 20))
                                }
                                let original = try read(exact, tensor: tensor + (values ? 1 : 0))
                                guard !values else {
                                    #expect(bytes == original, "layer \(layer) index rows")
                                    continue
                                }
                                let actual = floats(bytes)
                                let exactRows = floats(original)
                                let width = actual.count / position
                                for token in 0 ..< position {
                                    let row = Array(
                                        exactRows[token * width ..< (token + 1) * width])
                                    let got = Array(actual[token * width ..< (token + 1) * width])
                                    if token >= position - profile.recentTokenCount {
                                        #expect(
                                            got.map(\.bitPattern) == row.map(\.bitPattern),
                                            "layer \(layer) recent row \(token)")
                                    } else {
                                        let expected = try PagedKVQuantizationReference.roundTrip(
                                            row, config: profile, isKey: true)
                                        let error = zip(got, expected).map { abs($0 - $1) }.max()!
                                        #expect(error <= 0.00001, "layer \(layer) row \(token)")
                                    }
                                }
                            }
                        }
                    }

                    // A second restore adopts, keeps the KDA state exact and continues.
                    let adopter = CBv2RequestID(802)
                    let staged = try stage()
                    defer { staged.close() }
                    runtime.rows[adopter] = try runtime.codec.adoptGLM5Next(
                        staged, backend: runtime.backend, requestID: adopter,
                        restore: { checkpoint in
                            runtime.recurrence[adopter] = try CBv2RecurrentRequestState(
                                spec: runtime.model.cbv2RecurrentStateSpec,
                                adoptedCommitted: checkpoint.layers)
                        })
                    let donorState = try #require(
                        runtime.recurrence[donor]?.confirmedStateSnapshot())
                    let adoptedState = try #require(
                        runtime.recurrence[adopter]?.confirmedStateSnapshot())
                    for index in donorState.keys {
                        #expect(
                            donorState[index]?.ssm?.asArray(Float.self).map(\.bitPattern)
                                == adoptedState[index]?.ssm?.asArray(Float.self).map(\.bitPattern))
                        #expect(
                            donorState[index]?.conv?.asArray(Float.self).map(\.bitPattern)
                                == adoptedState[index]?.conv?.asArray(Float.self).map(\.bitPattern))
                    }
                    let expected = try runtime.forward(donor, tokens: [7, 8])
                    let continued = try runtime.forward(adopter, tokens: [7, 8])
                    let drift = abs(expected.logits - continued.logits).max().item(Float.self)
                    #expect(drift.isFinite)
                    #expect(runtime.backend.pool.attentionWorkBytesReserved == 0)
                }
            }
            runtime.engine.loopForTesting.onEngineQueueSync { runtime.close() }
            await runtime.engine.shutdown()
            try outcome.get()
        }

        @Test func genericCodecStillRejectsGLMAndForeignScopeCannotStage() async throws {
            let runtime = try Runtime(model: model())
            let outcome = Result {
                try runtime.engine.loopForTesting.onEngineQueueSync {
                    defer { runtime.close() }
                    let id = CBv2RequestID(701)
                    try runtime.make(id)
                    _ = try runtime.forward(id, tokens: [4, 5, 6, 7])
                    let (source, capture) = try export(
                        runtime, id: id, position: 4, prompt: [4, 5, 6, 7, 8])
                    defer {
                        source.close()
                        capture.closeAfterCompletedEvaluation()
                    }
                    let generic = CBv2CompleteCheckpointCodec(
                        identity: Self.identity,
                        layerKinds: runtime.model.cbv2LayerKinds,
                        recurrentSpec: runtime.model.cbv2RecurrentStateSpec,
                        kvDTypes: [.float32], assistant: nil,
                        admission: runtime.engine.admissionForTesting,
                        pagedConfig: runtime.backend.pool.config)
                    #expect(throws: CBv2CompleteCheckpointError.self) {
                        try generic.tensorDescriptors(position: 4)
                    }
                    let foreign = CBv2Request(
                        id: .init(702), promptTokens: [4, 5, 6, 7, 8],
                        maxTokens: 4, cacheSalt: "tenant-b", prefixCacheReceiptID: .init(1702))
                    #expect(throws: CBv2CompleteCheckpointError.self) {
                        try runtime.codec.plan(manifest: source.manifest, request: foreign)
                    }
                    let request = CBv2Request(
                        id: .init(703), promptTokens: [4, 5, 6, 7, 8],
                        maxTokens: 4, cacheSalt: "tenant-a", prefixCacheReceiptID: .init(1703))
                    for field in [
                        "modelAggregateHash", "promptContractID", "buildID", "numericsFingerprint",
                    ] {
                        var object = try #require(
                            JSONSerialization.jsonObject(
                                with: JSONEncoder().encode(source.manifest)) as? [String: Any])
                        var identity = try #require(object["identity"] as? [String: Any])
                        identity[field] = "different"
                        object["identity"] = identity
                        let changed = try JSONDecoder().decode(
                            CBv2CompleteCheckpointManifest.self,
                            from: JSONSerialization.data(withJSONObject: object))
                        #expect(throws: CBv2CompleteCheckpointError.self) {
                            try runtime.codec.plan(manifest: changed, request: request)
                        }
                    }
                    let plan = try runtime.codec.plan(manifest: source.manifest, request: request)
                    let before = runtime.engine.admissionForTesting.bytesReserved
                    let releases = Counter()
                    let sink = try plan.allocate(onRelease: { releases.increment() })
                    defer { sink.close() }
                    #expect(runtime.engine.admissionForTesting.bytesReserved > before)
                    let first = try source.readSegment(
                        tensorIndex: 0, byteOffset: 0, maximumBytes: 4)
                    #expect(throws: CBv2CompleteCheckpointError.self) {
                        try sink.appendSegment(tensorIndex: 1, byteOffset: 0, data: first)
                    }
                    try sink.appendSegment(tensorIndex: 0, byteOffset: 0, data: first)
                    #expect(throws: CBv2CompleteCheckpointError.self) { try sink.finish() }
                    // Dropping an incomplete stage leaves every donor row live.
                    sink.close()
                    sink.close()
                    #expect(releases.value == 1)
                    #expect(runtime.engine.admissionForTesting.bytesReserved == before)
                    #expect(runtime.rows[id]?.allSatisfy { $0?.absoluteOffset == 4 } == true)
                }
            }
            await runtime.engine.shutdown()
            try outcome.get()
        }

        @Test func checkpointRowRegistrationRejectsPartialAndDuplicateCohortsWithoutMutation()
            async throws
        {
            let runtime = try Runtime(model: model(layers: 8))
            let outcome = Result {
                try runtime.engine.loopForTesting.onEngineQueueSync {
                    let donor = CBv2RequestID(751)
                    let adopter = CBv2RequestID(752)
                    try runtime.make(donor)
                    _ = try runtime.forward(donor, tokens: [4, 5, 6, 7])
                    let (source, capture) = try export(
                        runtime, id: donor, position: 4, prompt: [4, 5, 6, 7, 8])
                    defer {
                        source.close()
                        capture.closeAfterCompletedEvaluation()
                    }
                    let request = CBv2Request(
                        id: adopter, promptTokens: [4, 5, 6, 7, 8],
                        maxTokens: 123, cacheSalt: "tenant-a", prefixCacheReceiptID: .init(1752))
                    let plan = try runtime.codec.plan(manifest: source.manifest, request: request)
                    let sink = try plan.allocate(onRelease: {})
                    defer { sink.close() }
                    try copy(source, to: sink)
                    let staged = try sink.finish()
                    defer { staged.close() }
                    let adoption = try staged.consumePreparedState { prepared in
                        let frame = try #require(prepared.pagedFrame)
                        prepared.pagedFrame = nil
                        return try runtime.backend.pool.importCheckpoint(
                            frame,
                            admission: runtime.engine.admissionForTesting, requestID: adopter,
                            layerKinds: runtime.model.cbv2LayerKinds, maximumTokens: 128)
                    }
                    defer { adoption.release() }
                    let rows: [CBv2SequenceKV?] = adoption.rows.map { $0 }
                    #expect(rows.count == 2)
                    #expect(throws: CBv2CompleteCheckpointError.self) {
                        try runtime.backend.registerGLM5NextCheckpointRows([rows[0]])
                    }
                    #expect(throws: CBv2CompleteCheckpointError.self) {
                        try runtime.backend.registerGLM5NextCheckpointRows([rows[0], rows[0]])
                    }
                    #expect(throws: CBv2CompleteCheckpointError.self) {
                        try runtime.backend.registerGLM5NextCheckpointRows(
                            try #require(runtime.rows[donor]))
                    }
                    runtime.rows[adopter] = try adoption.moveToActiveRequest { auxiliary in
                        let checkpoint = try runtime.codec.recurrentCheckpoint(
                            manifest: source.manifest, auxiliary: auxiliary)
                        let recurrent = try CBv2RecurrentRequestState(
                            spec: runtime.model.cbv2RecurrentStateSpec,
                            adoptedCommitted: checkpoint.layers)
                        try runtime.backend.registerGLM5NextCheckpointRows(rows)
                        runtime.recurrence[adopter] = recurrent
                    }
                    #expect(throws: CBv2CompleteCheckpointError.self) {
                        try runtime.backend.registerGLM5NextCheckpointRows(rows)
                    }
                    // The failed probes did not register a subset or damage the
                    // existing donor. Both complete cohorts receive real tickets.
                    let expected = try runtime.forward(donor, tokens: [8])
                    let actual = try runtime.forward(adopter, tokens: [8])
                    #expect(abs(expected.logits - actual.logits).max().item(Float.self) < 0.00001)
                }
            }
            runtime.engine.loopForTesting.onEngineQueueSync { runtime.close() }
            await runtime.engine.shutdown()
            try outcome.get()
        }

        @Test func rejectsIncompleteOrReinterpretedGLMManifestBeforeAllocation() throws {
            let state = CBv2GLM5NextCheckpointState(
                physicalPosition: 4, acceptedFrontier: 4,
                latentWidth: 64, indexWidth: 2,
                recurrentLayers: (0 ..< 3).map {
                    .init(
                        modelLayer: $0,
                        convolutionShape: [1, 6, 4], recurrentShape: [1, 1, 2, 2])
                }, storage: .float32)
            let attention = [
                CBv2CheckpointAttentionLayer(
                    modelLayer: 3, owner: 0,
                    window: nil, kvHeads: 1, headDim: 64, valueHeadDim: 5,
                    queryHeads: 1, hasSinks: false, dtype: .float32)
            ]
            let manifest = CBv2CompleteCheckpointManifest(
                identity: Self.identity, position: 4,
                chunkSize: 4, prefixTokens: [4, 5, 6, 7], cacheSalt: "tenant-a",
                assistantCodecID: nil,
                tensors: try state.targetDescriptors(attention: attention, position: 4),
                backendLayout: CBv2CompleteCheckpointManifest.glm5NextPagedLayout,
                attentionLayers: attention, glm5NextState: state)
            _ = try manifest.validateStructure()
            let original = try #require(
                JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(manifest)) as? [String: Any])
            func rejected(_ mutate: (inout [String: Any]) throws -> Void) throws {
                var object = original
                try mutate(&object)
                let changed = try JSONDecoder().decode(
                    CBv2CompleteCheckpointManifest.self,
                    from: JSONSerialization.data(withJSONObject: object))
                #expect(throws: CBv2CompleteCheckpointError.self) {
                    try changed.validateStructure()
                }
            }
            try rejected { object in
                var state = try #require(object["glm5NextState"] as? [String: Any])
                state["acceptedFrontier"] = 3
                object["glm5NextState"] = state
            }
            try rejected { object in
                var tensors = try #require(object["tensors"] as? [[String: Any]])
                tensors.removeLast()
                object["tensors"] = tensors
            }
            try rejected { object in
                object["backendLayout"] = CBv2CompleteCheckpointManifest.pagedLayout
            }
            try rejected { object in object["assistantCodecID"] = "not-real-assistant-state" }
            try rejected { object in
                var layers = try #require(object["attentionLayers"] as? [[String: Any]])
                layers[0]["valueHeadDim"] = 64
                object["attentionLayers"] = layers
            }
            let invalidValidity = Float(2).bitPattern.littleEndian
            let bytes = withUnsafeBytes(of: invalidValidity) { Data($0) }
            #expect(throws: CBv2CompleteCheckpointError.self) {
                try CBv2GLM5NextCheckpointPayload.validate(
                    manifest: manifest, tensorIndex: 1,
                    byteOffset: 4 * 4, data: bytes)
            }
            let validWord = withUnsafeBytes(of: Float(1).bitPattern.littleEndian) { Data($0) }
            try CBv2GLM5NextCheckpointPayload.validate(
                manifest: manifest, tensorIndex: 1, byteOffset: 4 * 4, data: validWord)
            // Partial or misaligned words cannot locate the validity column.
            #expect(throws: CBv2CompleteCheckpointError.self) {
                try CBv2GLM5NextCheckpointPayload.validate(
                    manifest: manifest, tensorIndex: 1, byteOffset: 4 * 4,
                    data: validWord.prefix(3))
            }
            #expect(throws: CBv2CompleteCheckpointError.self) {
                try CBv2GLM5NextCheckpointPayload.validate(
                    manifest: manifest, tensorIndex: 1, byteOffset: 4 * 4 + 2, data: validWord)
            }
        }

        @Test func actualMTPSerializationAloneDoesNotGrantProtectedPrefixOrUntrackedRegistration()
            async throws
        {
            let target = try model()
            let predictor = try GLM5NextMTP(target: target) { name in
                try Self.tensor(name, inventory: target.inventory)
            }
            let factory = try GLM5NextMTPExecution(
                target: target, predictor: predictor, maximumInputs: 64, storage: .contiguous)
            let assistant = try GLM5NextMTPAdapter(
                target: target, predictor: predictor, factory: factory)
            #expect(assistant is any CBv2MTPPrefixCheckpointCoding)
            let process = ProcessBudget()
            let limits = try CBv2PagedGatheredAttentionLimits(
                maximumBatchSize: 1, maximumQueryTokens: 32, maximumContextTokens: 64,
                maximumInFlightGraphs: 2, maximumScratchBytes: 128 << 20,
                admissionMode: .stepOwned(.pinnedMetal))
            let resources = try factory.makeNativeResources(
                assistant: assistant,
                config: .init(
                    capacityBytes: 64 << 20, dtype: .float32,
                    maxPrefillChunk: 32, nominalMaxSequenceLength: 64,
                    maxBufferLength: 16 << 20, segmentSizeBytes: 64 << 10,
                    layerDTypes: target.cbv2CompleteCheckpointKVDTypes, gatheredAttention: limits),
                processMemoryOwner: process, construction: NativeConstructionScope())
            let store = CompleteCheckpointFixtureStore()
            // This issued profile did not bind a complete-prefix store.
            // Serialization alone cannot upgrade its grant after construction.
            let engine = EngineV2(
                model: resources.model, layerKinds: target.cbv2LayerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 32, maxWaiting: 2, enablePrefixCache: true),
                admissionConfig: target.cbv2NativeAdmissionConfig, completePrefixCache: store,
                mtpDrafter: assistant,
                mtpConfig: .init(
                    enabled: true, maxDraftTokens: 1,
                    fixedDraftTokens: 1, verificationMode: .serialTarget),
                processMemoryOwner: process, nativeCompletionTracking: true,
                nativeExecutionContract: resources.contract)
            #expect(engine.completeCheckpointCodec == nil && engine.completePrefixCache == nil)
            #expect(throws: CBv2CompleteCheckpointError.self) {
                try engine.reserveCompleteCheckpointReadScratch()
            }
            #expect(throws: CBv2CompleteCheckpointError.self) {
                try resources.backend.registerGLM5NextCheckpointRows([])
            }
            #expect(resources.backend.bytesReserved == 0)
            await engine.shutdown()
        }

        /// The protected native MTP profiles compose with the complete-prefix
        /// store: a producer engine donates the target state plus the assistant
        /// history, and a fresh consumer resumes speculation from the restored
        /// state. Depth 0 is the serial full-state profile; 1 and 3 are the
        /// rectangular profile. Both the cold and the restored streams equal the
        /// target-only paged engine.
        @Test(arguments: [0, 1, 3])
        func nativeMTPProfilesResumeFromCompletePrefixExactly(depth: Int) async throws {
            let request = CBv2Request(
                id: .init(803), promptTokens: (0 ..< 73).map { 4 + $0 % 20 },
                sampling: .init(temperature: 0), maxTokens: 6,
                cacheSalt: "tenant-mtp", prefixCacheReceiptID: .init(1803))
            let reference = try model()
            let plainBackend = try reference.makeCBv2PagedBackend(
                kvBytesCapacity: 512 << 20, maximumBatchSize: 1, maximumQueryTokens: 32,
                maximumContextTokens: 128, maximumScratchBytes: 512 << 20,
                segmentBytes: 64 << 10, maximumBufferBytes: 16 << 20)
            let plain = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(reference),
                layerKinds: reference.cbv2LayerKinds, backend: plainBackend,
                cacheProvider: CBv2LayerCacheBank(caches: plainBackend.makeLayerCaches()),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 32, maxWaiting: 4, enablePrefixCache: false),
                admissionConfig: reference.cbv2NativeAdmissionConfig,
                processMemoryOwner: ProcessBudget())
            let expected = await cbv2SchedCollect(try plain.submit(request))
            await plain.shutdown()
            #expect(expected.tokens.count == 6)

            let verification: GLM5NextMTPVerification =
                depth == 0 ? .serialTarget : .rectangular(depth: depth)
            let profile =
                depth == 0
                ? "glm5_next_native_paged_full_state_serial_mtp_v1"
                : "glm5_next_native_paged_full_state_rectangular_mtp_v1"
            func engine(_ store: NativeStore) throws -> (EngineV2, PagedKVBackend) {
                let target = try model()
                let predictor = try GLM5NextMTP(target: target) { name in
                    try Self.tensor(name, inventory: target.inventory)
                }
                let factory = try GLM5NextMTPExecution(
                    target: target, predictor: predictor, maximumInputs: 128,
                    storage: .contiguous, verification: verification)
                let assistant = try GLM5NextMTPAdapter(
                    target: target, predictor: predictor, factory: factory)
                let limits = try CBv2PagedGatheredAttentionLimits(
                    maximumBatchSize: 1, maximumQueryTokens: 32, maximumContextTokens: 128,
                    maximumInFlightGraphs: 2, maximumScratchBytes: 512 << 20,
                    admissionMode: .stepOwned(.pinnedMetal))
                let process = ProcessBudget()
                let resources = try factory.makeNativeResources(
                    assistant: assistant,
                    config: .init(
                        capacityBytes: 512 << 20, dtype: .float32,
                        maxPrefillChunk: 32, nominalMaxSequenceLength: 128,
                        maxBufferLength: 16 << 20, segmentSizeBytes: 64 << 10,
                        layerDTypes: target.cbv2CompleteCheckpointKVDTypes,
                        gatheredAttention: limits),
                    processMemoryOwner: process, construction: NativeConstructionScope(),
                    completePrefixCache: store)
                #expect(resources.contract.profile == profile)
                #expect(resources.contract.supportsNativeCompletePrefix)
                #expect(resources.contract.supportsNativePagedRectangularMTP == (depth > 0))
                let engine = EngineV2(
                    model: resources.model, layerKinds: target.cbv2LayerKinds,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    sampler: CBv2GreedySampler(),
                    schedulerConfig: .init(
                        maxConcurrentRequests: 1, maxBatchedTokensPerStep: 32,
                        prefillChunkSize: 32, maxConcurrentPartialPrefills: 1, maxWaiting: 4,
                        enablePrefixCache: true),
                    admissionConfig: target.cbv2NativeAdmissionConfig,
                    completePrefixCache: store, mtpDrafter: assistant,
                    mtpConfig: .init(
                        enabled: true, maxDraftTokens: verification.depth,
                        maxSpeculativeBatch: 1, fixedDraftTokens: verification.depth,
                        verificationMode: verification.engineMode),
                    processMemoryOwner: process, nativeCompletionTracking: true,
                    nativeExecutionContract: resources.contract)
                #expect(engine.mtpInactiveReason == nil)
                #expect(engine.loopForTesting.nativeShutdownState?.isIncomplete == false)
                return (engine, resources.backend)
            }
            func expectQuiescent(_ engine: EngineV2) async -> CBv2MTPMetrics? {
                let metrics = engine.mtpMetricsSnapshot()
                guard case .quiescent = await engine.shutdownReportingNativeCompletion() else {
                    Issue.record("native MTP prefix engine did not shut down cleanly")
                    return metrics
                }
                return metrics
            }

            let producerStore = NativeStore(.init(segmentBytes: 28))
            let (producer, producerBackend) = try engine(producerStore)
            let produced = await cbv2SchedCollect(try producer.submit(request))
            let producerMetrics = await expectQuiescent(producer)
            #expect(produced.tokens == expected.tokens)
            #expect(!producerStore.base.saved.isEmpty)
            #expect(producerBackend.bytesReserved == 0)
            #expect((producerMetrics?.proposedTokens ?? 0) > 0)

            let consumerStore = NativeStore(
                .init(archives: producerStore.base.saved, segmentBytes: 28))
            let (consumer, consumerBackend) = try engine(consumerStore)
            #expect(try consumerStore.base.stage(engine: consumer, request: request))
            let resumed = await cbv2SchedCollect(try consumer.submit(request))
            let consumerMetrics = await expectQuiescent(consumer)
            #expect(resumed.finishReason == .length && resumed.tokens == expected.tokens)
            #expect((resumed.usage?.prefixCachePrefillTokensSaved ?? 0) > 0)
            #expect(resumed.usage?.prefixCacheReplayTokens == 0)
            #expect((consumerMetrics?.proposedTokens ?? 0) > 0)
            if depth > 0 {
                #expect((consumerMetrics?.rectangularVerificationRounds ?? 0) > 0)
            } else {
                #expect((consumerMetrics?.serialVerificationRounds ?? 0) > 0)
            }
            #expect(consumerBackend.bytesReserved == 0 && consumerBackend.bytesInUse == 0)
        }

        /// Native store adapter over the shared fixture store.
        private final class NativeStore: CBv2NativeCompletePrefixCache, @unchecked Sendable {
            let base: CompleteCheckpointFixtureStore
            init(_ base: CompleteCheckpointFixtureStore) { self.base = base }
            var identity: CBv2CompleteCheckpointIdentity { base.identity }
            func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
                base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
            }
            func takeStaged(
                requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
                maximumSequenceLength: Int
            ) -> CBv2StagedCompleteCheckpoint? {
                base.takeStaged(
                    requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                    maximumSequenceLength: maximumSequenceLength)
            }
            func donate(
                _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
                tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
            ) {
                base.donate(
                    source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                    completion: completion)
            }
            func close() { base.close() }
            func closeAndWait() async { base.close() }
        }

        /// Integration gate: this deliberately requires real model/engine
        /// registration. It cannot pass by substituting a declared cache hit.
        @Test func realProducerAndConsumerEngineStreamsRestoreExactTarget() async throws {
            func engine(store: CompleteCheckpointFixtureStore?) throws -> (EngineV2, PagedKVBackend)
            {
                let model = try model()
                let backend = try model.makeCBv2PagedBackend(
                    kvBytesCapacity: 64 << 20, maximumBatchSize: 1, maximumQueryTokens: 32,
                    maximumContextTokens: 128, maximumScratchBytes: 128 << 20,
                    segmentBytes: 64 << 10, maximumBufferBytes: 16 << 20)
                let engine = EngineV2(
                    model: CBv2SteppableLanguageModelAdapter(model),
                    layerKinds: model.cbv2LayerKinds, backend: backend,
                    cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                    sampler: CBv2GreedySampler(),
                    schedulerConfig: .init(
                        maxConcurrentRequests: 1, maxBatchedTokensPerStep: 32,
                        prefillChunkSize: 32, maxWaiting: 4, enablePrefixCache: store != nil),
                    admissionConfig: model.cbv2NativeAdmissionConfig,
                    completePrefixCache: store, processMemoryOwner: ProcessBudget())
                if store != nil {
                    #expect(
                        engine.completeCheckpointCodec?.backendLayout
                            == CBv2CompleteCheckpointManifest.glm5NextPagedLayout)
                    #expect(engine.completeCheckpointCodec?.glm5NextLayout?.backend === backend)
                    #expect(
                        engine.completeCheckpointCodec?.kvDTypes
                            == model.cbv2CompleteCheckpointKVDTypes)
                    #expect(model.cbv2Capabilities.supportsRecurrentCheckpointReuse)
                }
                return (engine, backend)
            }
            let store = CompleteCheckpointFixtureStore(segmentBytes: 28)
            let (producer, producerBackend) = try engine(store: store)
            let request = CBv2Request(
                id: .init(801), promptTokens: (0 ..< 73).map { 4 + $0 % 20 },
                sampling: .init(temperature: 0), maxTokens: 4,
                cacheSalt: "tenant-a", prefixCacheReceiptID: .init(1801))
            let expected = await cbv2SchedCollect(try producer.submit(request))
            await producer.shutdown()
            #expect(expected.finishReason == .length)
            #expect(!store.saved.isEmpty)
            #expect(producerBackend.bytesReserved == 0)
            let reopened = CompleteCheckpointFixtureStore(archives: store.saved, segmentBytes: 28)
            let (consumer, consumerBackend) = try engine(store: reopened)
            let staged = try reopened.stage(engine: consumer, request: request)
            #expect(staged)
            let actual = await cbv2SchedCollect(try consumer.submit(request))
            await consumer.shutdown()
            #expect(actual.finishReason == .length && actual.tokens == expected.tokens)
            #expect((actual.usage?.prefixCachePrefillTokensSaved ?? 0) > 0)
            #expect(actual.usage?.prefixCacheReplayTokens == 0)
            #expect(consumerBackend.bytesReserved == 0 && consumerBackend.bytesInUse == 0)
        }
    }
}
