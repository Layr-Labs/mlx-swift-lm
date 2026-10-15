import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextCBv2Tests {
        private func makeModel(
            layers: Int = 4, configuration: GLM5NextConfiguration? = nil,
            executionPolicy: GLM5NextExecutionPolicy = .reference
        ) throws -> GLM5NextTextModel {
            let config = try configuration ?? GLM5NextInventoryFixture.configuration(layers: layers)
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "native-tiny.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
            return try GLM5NextTextModel(
                configuration: config.textConfig, inventory: inventory,
                tensor: { name in
                let tensor = try #require(inventory.entries[name]?.tensor)
                let count = tensor.shape.reduce(1, *)
                let values: [Float]
                if name == "model.language_model.embed_tokens.weight" {
                    values = (0 ..< count).map {
                        Float((($0 / 4 + 1) * ($0 % 4 + 2)) % 11 - 5) * 0.1
                    }
                } else if name == "lm_head.weight" {
                    values = (0 ..< count).map { Float(($0 / 4 * 7 + $0 % 4 * 3) % 19 - 9) * 0.03 }
                } else if name.hasSuffix("_scale") || name.contains("norm.weight") {
                    values = Array(repeating: 1, count: count)
                } else if name.hasSuffix("A_log") {
                    values = Array(repeating: -0.7, count: count)
                } else if name.hasSuffix("dt_bias") {
                    values = Array(repeating: -0.2, count: count)
                } else if name.hasSuffix("_base") || name.hasSuffix(".bias") {
                    values = Array(repeating: 0, count: count)
                } else {
                    let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                    values = (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }
                }
                return MLXArray(values, tensor.shape)
            }, executionPolicy: executionPolicy)
        }

        private final class Harness {
            let model: GLM5NextTextModel
            let backend: CBv2ContiguousKVBackend
            let bank: CBv2LayerCacheBank
            let rows: [[CBv2SequenceKV?]]
            var recurrent: [CBv2RecurrentRequestState]
            init(model: GLM5NextTextModel, batch: Int = 1) throws {
                let backend = CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 16 << 20, kvDType: .float32))
                let rows = try (0 ..< batch).map { _ in
                    try backend.makeSequenceState(
                        layerKinds: model.cbv2LayerKinds, promptLength: 0, maxLength: 64)
                }
                self.model = model
                self.backend = backend
                self.rows = rows
                bank = CBv2LayerCacheBank(layerKinds: model.cbv2LayerKinds)
                recurrent = try (0 ..< batch).map { _ in
                    try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
                }
            }
            func forward(
                _ tokens: MLXArray, validity: MLXArray? = nil, commit: Bool = true,
                mtpPrefill: CBv2PrefillRequirement? = nil
            ) throws -> GLM5NextTextOutput {
                let evaluations = try recurrent.map { try $0.bind() }
                let caches = bank.layerCaches(rowStates: rows)
                let result: GLM5NextTextOutput
                if let mtpPrefill {
                    let forwarded = model.cbv2ForwardWithHiddenForPrefill(
                        tokens, caches: caches.map { $0 as! any KVCache },
                        recurrentState: evaluations, positionIds: nil, requirement: mtpPrefill)
                    result = .init(logits: forwarded.logits, hidden: forwarded.lastHidden)
                } else {
                    result = try model.prepareCBv2Forward(
                        tokens: tokens, caches: caches.map { $0 as! any KVCache },
                        recurrentState: evaluations, validity: validity)
                }
                let roots =
                    try evaluations.flatMap { try $0.evaluate() }
                    + caches.flatMap { ($0 as! any KVCache).innerState() }
                try withError { fault in
                    eval(roots + [result.logits, result.hidden])
                    StreamOrDevice.default.stream.synchronize()
                    try fault.check()
                }
                if commit {
                    for evaluation in evaluations { try evaluation.commit() }
                } else {
                    for evaluation in evaluations { try evaluation.rollback() }
                    for row in rows { for state in row { state?.rollback(tokens.dim(1)) } }
                    bank.invalidateBoundComposition()
                }
                return result
            }
            func checkpointClone() throws -> Harness {
                let copy = try Harness(model: model, batch: rows.count)
                for index in rows.indices {
                    for (destination, source) in zip(copy.rows[index], rows[index]) {
                        let snapshot = try #require(source).snapshot()
                        _ = destination?.update(keys: snapshot.keys, values: snapshot.values)
                    }
                    let confirmed = try #require(recurrent[index].confirmedStateSnapshot())
                    try copy.recurrent[index].release()
                    copy.recurrent[index] = try CBv2RecurrentRequestState(
                        spec: model.cbv2RecurrentStateSpec, adoptedCommitted: confirmed)
                }
                return copy
            }
            func stateBytes() -> [[Float]] {
                rows.flatMap { row in
                    row.compactMap { $0 }.flatMap { state in
                        let snapshot = state.snapshot()
                        return [
                            snapshot.keys.asArray(Float.self), snapshot.values.asArray(Float.self),
                        ]
                    }
                }
                    + recurrent.flatMap { owner in
                        (owner.confirmedStateSnapshot() ?? [:]).sorted(by: { $0.key < $1.key })
                            .flatMap { _, state in
                                [state.conv, state.ssm].compactMap { $0?.asArray(Float.self) }
                            }
                    }
            }
            deinit {
                bank.releaseBoundRows()
                for row in rows { backend.release(row) }
                for state in recurrent { try? state.release() }
            }
        }

        private func close(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape)
            #expect(
                abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                    < 0.0001)
        }

        @Test func full45LayerNativeForwardUsesAll34RecurrentAnd11SparseOwners() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel(layers: 45)
                let harness = try Harness(model: model)
                let tokens = MLXArray([Int32(4), 5, 6], [1, 3])
                let reference = try model.forward(tokens: tokens, cache: nil)
                let native = try harness.forward(tokens)
                close(native.logits, reference.logits)
                close(native.hidden, reference.hidden)
                close(native.logits, model.lmHead(native.hidden))
                #expect(model.cbv2LayerKinds.count == 11)
                #expect(model.cbv2RecurrentStateSpec.layers.count == 34)
                #expect(harness.recurrent[0].confirmedStateSnapshot()?.count == 34)
                #expect(harness.rows[0].allSatisfy { $0?.absoluteOffset == 3 })
                #expect(model.cbv2LayerKinds.allSatisfy { $0.headDim == 2 && $0.valueHeadDim == 5 })
                #expect(
                    model.cbv2Capabilities.supportsPagedKV
                        && model.cbv2Capabilities.requiresNativePagedKV)
                #expect(!model.cbv2Capabilities.supportsMTP)
                #expect(
                    model.cbv2Capabilities.supportsPrefixReuse
                        && model.cbv2Capabilities.supportsRecurrentCheckpointReuse)
            }
        }

        @Test func recurrentMTPPrefillRetainsItsPositionAxis() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let tokens = MLXArray([Int32(4), 5, 6], [1, 3])
                let reference = try model.forward(tokens: tokens, cache: nil)
                for requirement: CBv2PrefillRequirement in [.evaluationOnly, .lastPositionLogits] {
                    let harness = try Harness(model: model)
                    let output = try harness.forward(tokens, mtpPrefill: requirement)
                    let width = requirement == .evaluationOnly ? 1 : model.vocabularySize
                    #expect(output.logits.shape == [1, 1, width])
                    close(output.logits, reference.logits[0..., 2 ..< 3, 0 ..< width])
                    close(output.hidden, reference.hidden)
                }
            }
        }

        @Test func batchMaskAndRaggedPhysicalPositionsMatchIndependentTargetRows() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let harness = try Harness(model: model, batch: 2)
                let tokens = MLXArray([Int32(4), 5, 6, 0, 4, 5], [2, 3])
                let mask = MLXArray([true, true, true, false, true, true], [2, 3])
                let native = try harness.forward(tokens, validity: mask)
                let reference = try model.forward(tokens: tokens, cache: nil, validity: mask)
                close(native.logits, reference.logits)
                close(native.hidden, reference.hidden)
                let first = try model.forward(
                    tokens: MLXArray([Int32(4), 5, 6], [1, 3]), cache: nil)
                let second = try model.forward(
                    tokens: MLXArray([Int32(4), 5], [1, 2]), cache: nil)
                close(native.hidden[0 ..< 1], first.hidden)
                close(native.hidden[1 ..< 2, 1...], second.hidden)
                #expect(native.hidden[1, 0].asArray(Float.self).allSatisfy { $0 == 0 })
                let caches = harness.bank.layerCaches(rowStates: harness.rows).map {
                    $0 as! any KVCache
                }
                #expect(throws: GLM5NextLoadError.self) {
                    try model.prepareCBv2Forward(
                        tokens: MLXArray([Int32(1), 2], [2, 1]), caches: caches,
                        recurrentState: harness.recurrent.map { try $0.bind() },
                        positionIds: MLXArray.zeros([1, 2, 1], dtype: .int32))
                }
                #expect(harness.rows.allSatisfy { $0.allSatisfy { $0?.absoluteOffset == 3 } })
            }
        }

        @Test(arguments: [3, 4, 5])
        func rollbackAndExactCheckpointBoundaryPreserveContinuation(boundary: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let harness = try Harness(model: model)
                _ = try harness.forward(
                    MLXArray((0 ..< boundary).map { Int32($0 + 4) }, [1, boundary]))
                let before = harness.stateBytes()
                let copy = try harness.checkpointClone()
                let suffix = MLXArray([Int32(9), 10], [1, 2])
                let discarded = try harness.forward(suffix, commit: false)
                #expect(harness.stateBytes() == before)
                #expect(harness.rows[0].allSatisfy { $0?.absoluteOffset == boundary })
                let continued = try harness.forward(suffix)
                let restored = try copy.forward(suffix)
                close(discarded.logits, continued.logits)
                close(restored.logits, continued.logits)
                close(restored.hidden, continued.hidden)
            }
        }

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            /// The candidate DSA selection is exact by construction, so every
            /// prefill and decode output must match the reference bit for bit.
            @Test func candidateSparseSelectionMatchesReferenceBitwise() throws {
                try Device.withDefaultDevice(.gpu) {
                    let model = try makeModel()
                    let candidate = try makeModel(
                        executionPolicy: .init(
                            sparseSelection: .init(
                                selection: .parallelThreshold, deviceRanges: true)))
                    #expect(
                        candidate.model.layers.compactMap(\.sparseAttention).allSatisfy {
                            $0.executionPolicy == candidate.executionPolicy
                        })
                    func bits(_ array: MLXArray) -> [UInt32] {
                        array.asType(.float32).asArray(Float.self).map(\.bitPattern)
                    }
                    for batch in [1, 2] {
                        let reference = try Harness(model: model, batch: batch)
                        let parallel = try Harness(model: candidate, batch: batch)
                        // indexTopK 4 selects one pool, so a 23-token history
                        // runs the threshold path; repeated tokens create ties.
                        let prompt = (0 ..< 23 * batch).map { Int32($0 % 9 + 4) }
                        // With two rows, row 0 is left-padded by three tokens.
                        var steps: [(MLXArray, MLXArray?)] = [
                            (
                                MLXArray(prompt, [batch, 23]),
                                batch == 1 ? nil : MLXArray((0 ..< 46).map { $0 >= 3 }, [2, 23])
                            )
                        ]
                        for token in [7, 5, 5, 11, 4] {
                            let column = Array(repeating: Int32(token), count: batch)
                            steps.append((MLXArray(column, [batch, 1]), nil))
                        }
                        for (tokens, validity) in steps {
                            let expected = try reference.forward(tokens, validity: validity)
                            let actual = try parallel.forward(tokens, validity: validity)
                            #expect(bits(actual.logits) == bits(expected.logits))
                            #expect(bits(actual.hidden) == bits(expected.hidden))
                        }
                        #expect(
                            parallel.stateBytes().map { $0.map(\.bitPattern) }
                                == reference.stateBytes().map { $0.map(\.bitPattern) })
                    }
                }
            }

            /// Whole-step KDA through the contiguous CBv2 target: a 40-token
            /// prefill crosses one chunk, then decode and a two-token verify
            /// window. A discarded step leaves the recurrent owners unchanged,
            /// and a checkpoint clone continues like the live rows.
            @Test(arguments: [
                GLM5NextExecutionPolicy.LinearAttention.nativeRecurrence,
                .nativeRecurrenceChunkedPrefill,
            ])
            func wholeStepLinearAttentionMatchesReferenceRows(
                member: GLM5NextExecutionPolicy.LinearAttention
            ) throws {
                try Device.withDefaultDevice(.gpu) {
                    let model = try makeModel()
                    let candidate = try makeModel(executionPolicy: .init(linearAttention: member))
                    // GPU FP32 matmuls may run at TF32 on Neural-Accelerator GPUs.
                    func near(_ actual: MLXArray, _ expected: MLXArray) {
                        #expect(actual.shape == expected.shape)
                        let error = abs(actual.asType(.float32) - expected.asType(.float32))
                            .max().item(Float.self)
                        #expect(error.isFinite && error < 2e-3, "max error \(error)")
                    }
                    for batch in [1, 2] {
                        let reference = try Harness(model: model, batch: batch)
                        let native = try Harness(model: candidate, batch: batch)
                        let prompt = (0 ..< 40 * batch).map { Int32($0 % 13 + 4) }
                        var steps: [(MLXArray, MLXArray?)] = [
                            (
                                MLXArray(prompt, [batch, 40]),
                                batch == 1 ? nil : MLXArray((0 ..< 80).map { $0 >= 5 }, [2, 40])
                            )
                        ]
                        steps.append(
                            (MLXArray(Array(repeating: Int32(7), count: batch), [batch, 1]), nil))
                        steps.append(
                            (MLXArray((0 ..< 2 * batch).map { Int32($0 + 5) }, [batch, 2]), nil))
                        for (tokens, validity) in steps {
                            let expected = try reference.forward(tokens, validity: validity)
                            let actual = try native.forward(tokens, validity: validity)
                            near(actual.logits, expected.logits)
                            near(actual.hidden, expected.hidden)
                        }
                        let before = native.stateBytes()
                        let suffix = MLXArray(
                            Array(repeating: Int32(9), count: 3 * batch), [batch, 3])
                        _ = try native.forward(suffix, commit: false)
                        #expect(native.stateBytes() == before)
                        let copy = try native.checkpointClone()
                        let continued = try native.forward(suffix)
                        let restored = try copy.forward(suffix)
                        let wanted = try reference.forward(suffix)
                        near(restored.logits, continued.logits)
                        near(continued.logits, wanted.logits)
                        for (actual, expected) in zip(native.stateBytes(), reference.stateBytes()) {
                            near(MLXArray(actual), MLXArray(expected))
                        }
                    }
                }
            }
        #endif

        private func engine(_ model: GLM5NextTextModel, width: Int) -> (
            EngineV2, CBv2ContiguousKVBackend
        ) {
            let backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 16 << 20, kvDType: .float32))
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(layerKinds: model.cbv2LayerKinds),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: width, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 4, maxWaiting: 8, enablePrefixCache: false),
                admissionConfig: model.cbv2NativeAdmissionConfig)
            return (engine, backend)
        }

        private func greedy(_ model: GLM5NextTextModel, prompt: [Int], count: Int) throws -> [Int] {
            let cache = model.newTextCache()
            var input = prompt
            var tokens: [Int] = []
            for _ in 0 ..< count {
                let output = try model.forward(
                    tokens: MLXArray(input.map(Int32.init), [1, input.count]), cache: cache)
                let next = output.logits[0, -1].argMax().item(Int.self)
                tokens.append(next)
                input = [next]
            }
            return tokens
        }

        @Test func actualEnginePrefillAndDecodeMatchTargetOnlyGreedy() async throws {
            let model = try makeModel()
            let expected = try greedy(model, prompt: [4, 5, 6, 7, 8], count: 4)
            let (engine, backend) = engine(model, width: 1)
            let stream = try engine.submit(
                .init(
                    id: .init(501), promptTokens: [4, 5, 6, 7, 8],
                    sampling: .init(temperature: 0), maxTokens: 4))
            var output: [Int] = []
            var completed = false
            for await event in stream {
                if case .delta(_, let tokens, _) = event { output += tokens }
                if case .finished(let reason, _) = event, case .length = reason { completed = true }
            }
            await engine.shutdown()
            #expect(completed && output == expected)
            #expect(backend.bytesReserved == 0 && backend.bytesInUse == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
        }

        @Test func actualEngineB2MatchesBothIndependentTargets() async throws {
            let model = try makeModel()
            let expected = try greedy(model, prompt: [4, 5, 6], count: 3)
            let otherExpected = try greedy(model, prompt: [7, 8, 9, 10, 11], count: 3)
            let (engine, backend) = engine(model, width: 2)
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
            }
            let first = try engine.submit(
                .init(
                    id: .init(601), promptTokens: [4, 5, 6],
                    sampling: .init(temperature: 0), maxTokens: 3))
            let second = try engine.submit(
                .init(
                    id: .init(602), promptTokens: [7, 8, 9, 10, 11],
                    sampling: .init(temperature: 0), maxTokens: 3))
            // Registration closures run before this queue barrier: neither
            // request can finish before the other enters the same scheduler.
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            var output: [Int] = []
            for await event in first {
                if case .delta(_, let tokens, _) = event { output += tokens }
            }
            var other: [Int] = []
            for await event in second {
                if case .delta(_, let tokens, _) = event { other += tokens }
            }
            await engine.shutdown()
            #expect(output == expected && other == otherExpected)
            #expect(backend.bytesReserved == 0 && backend.bytesInUse == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
        }

        @Test func actualEngineCancellationRetiresNativeRowsAndRecurrence() async throws {
            let model = try makeModel()
            let (engine, backend) = engine(model, width: 1)
            let stream = try engine.submit(
                .init(
                    id: .init(701), promptTokens: [4, 5, 6],
                    sampling: .init(temperature: 0), maxTokens: 64))
            var cancelled = false
            var requestedCancellation = false
            for await event in stream {
                if case .delta = event, !requestedCancellation {
                    requestedCancellation = true
                    engine.cancel(.init(701))
                }
                if case .finished(let reason, _) = event, case .cancelled = reason {
                    cancelled = true
                }
            }
            await engine.shutdown()
            #expect(requestedCancellation && cancelled)
            #expect(backend.bytesReserved == 0 && backend.bytesInUse == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
        }

        private func pagedEngine(
            _ model: GLM5NextTextModel, width: Int, segmentBytes: Int = 64 << 10
        ) throws -> (EngineV2, PagedKVBackend, GLM5NextTestProcessBudget) {
            let backend = try model.makeCBv2PagedBackend(
                kvBytesCapacity: 128 << 20, maximumBatchSize: width, maximumQueryTokens: 16,
                maximumContextTokens: 128, maximumScratchBytes: 256 << 20,
                segmentBytes: segmentBytes, maximumBufferBytes: 32 << 20)
            let process = GLM5NextTestProcessBudget()
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: width, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 4, maxWaiting: 8, enablePrefixCache: false),
                admissionConfig: model.cbv2NativeAdmissionConfig, processMemoryOwner: process)
            #expect(backend.pool.memoryAdmission === engine.admissionForTesting)
            #expect(backend.pool.physicalLease != nil)
            return (engine, backend, process)
        }

        @Test(arguments: [1, 2])
        func actualUnifiedPagedEngineMatchesUntouchedContiguousTarget(width: Int) async throws {
            let model = try makeModel(layers: width == 1 ? 45 : 4)
            let expected = try greedy(model, prompt: [4, 5, 6, 7, 8], count: 4)
            let otherExpected = try greedy(model, prompt: [9, 10, 11], count: 4)
            let (engine, backend, process) = try pagedEngine(model, width: width)
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
            }
            let first = try engine.submit(
                .init(
                    id: .init(801), promptTokens: [4, 5, 6, 7, 8],
                    sampling: .init(temperature: 0), maxTokens: 4))
            let second =
                width == 2
                ? try engine.submit(
                    .init(
                        id: .init(802), promptTokens: [9, 10, 11],
                        sampling: .init(temperature: 0), maxTokens: 4)) : nil
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            var output: [Int] = []
            for await event in first {
                if case .delta(_, let tokens, _) = event { output += tokens }
            }
            var other: [Int] = []
            if let second {
                for await event in second {
                    if case .delta(_, let tokens, _) = event { other += tokens }
                }
            }
            await engine.shutdown()
            #expect(output == expected)
            if width == 2 { #expect(other == otherExpected) }
            #expect((backend.glm5NextModelBinding?.observedSelectedRows ?? 0) > 0)
            #expect((backend.glm5NextModelBinding?.observedMetadataRows ?? 0) > 0)
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
            #expect(process.bytes == 0)
        }

        @Test func unifiedPagedCancellationRetainsWorkUntilNativeFence() async throws {
            let model = try makeModel()
            let (engine, backend, process) = try pagedEngine(model, width: 1)
            let stream = try engine.submit(
                .init(
                    id: .init(901), promptTokens: [4, 5, 6, 7, 8],
                    sampling: .init(temperature: 0), maxTokens: 64))
            var sent = false
            var cancelled = false
            for await event in stream {
                if case .delta = event, !sent {
                    sent = true
                    engine.cancel(.init(901))
                }
                if case .finished(let reason, _) = event, case .cancelled = reason {
                    cancelled = true
                }
            }
            await engine.shutdown()
            #expect(sent && cancelled)
            #expect(backend.bytesReserved == 0 && backend.bytesInUse == 0)
            #expect(backend.pool.attentionWorkBytesReserved == 0)
            #expect(process.bytes == 0)
        }

        /// Register every row before the next throwing operation. A failed
        /// manual step uses the same drain/drop/retire order as the engine;
        /// neither a leaked row nor cleanup failure may hide its original error.
        private final class OwnedMainPoolRows {
            let backend: PagedKVBackend
            let caches: [PagedLayerCache]
            private var owned: [[CBv2SequenceKV?]] = []

            init(backend: PagedKVBackend) {
                self.backend = backend
                caches = backend.pool.attentionWorkCaches.keys.sorted().compactMap {
                    backend.pool.attentionWorkCaches[$0]?.value
                }
            }

            func makeState(maxLength: Int = 64) throws -> [CBv2SequenceKV?] {
                let state = try backend.makeSequenceState(
                    layerKinds: backend.layerKinds, promptLength: 0, maxLength: maxLength)
                owned.append(state)
                return state
            }

            func release(_ state: [CBv2SequenceKV?]) {
                backend.release(state)
                let serials = Set(state.compactMap { ($0 as? PagedSequenceKV)?.serial })
                owned.removeAll { candidate in
                    Set(candidate.compactMap { ($0 as? PagedSequenceKV)?.serial }) == serials
                }
            }

            func close() {
                caches.forEach { $0.setRows([]) }
                let unfinished = Set(
                    backend.pool.attentionWorkOwners.values
                        .filter { !$0.released }.flatMap { $0.descriptors.keys.map(\.serial) })
                #expect(unfinished.isEmpty)
                for state in owned {
                    // Unknown completion must remain quarantined, not recycled.
                    if state.compactMap({ $0 as? PagedSequenceKV }).allSatisfy({
                        !unfinished.contains($0.serial)
                    }) {
                        backend.release(state)
                    }
                }
                owned.removeAll()
            }

            func execute(
                _ state: [CBv2SequenceKV?], id: UInt64, range: Range<Int>, marker: Float,
                model: GLM5NextTextModel, indices: [Int32],
                values: MLXArray? = nil,
                afterPayload: (() throws -> Void)? = nil,
                beforeEvaluation: ((CBv2PagedAttentionStepOwner) throws -> Void)? = nil,
                afterEvaluation: ((GLM5NextPagedPayload) throws -> Void)? = nil,
                read: (GLM5NextPagedPayload, MLXArray) throws -> MLXArray = {
                    try $0.selectedKeys(row: 0, indices: $1)
                }
            ) throws -> [Float] {
                let rows = [CBv2RequestID(id): state]
                let preparedWork = try backend.prepareAttentionWork(
                    assignments: [(.init(id), range)], states: rows)
                let work = try #require(preparedWork)
                let boundary = CBv2PagedWriteBoundary(pool: backend.pool)
                let result = Result { () throws -> [Float] in
                    caches[0].setRows([state[0]!])
                    let keys = MLXArray(
                        range.flatMap { token in
                            [marker + Float(token), marker + Float(token) + 0.5]
                        }
                    )
                    .reshaped(1, 1, range.count, 2)
                    let values = values ?? MLXArray.ones([1, 1, range.count, 5])
                    let payload = try caches[0].beginGLM5NextPayload(
                        keys: keys, values: values, targetIdentity: ObjectIdentifier(model))
                    let selected = try read(
                        payload, MLXArray(indices).reshaped(1, indices.count))
                    payload.retainResult(selected)
                    try afterPayload?()
                    try work.seal()
                    work.publish()
                    backend.pool.endAttentionConstruction(work)
                    try beforeEvaluation?(work)
                    try work.finishEvaluation()
                    try afterEvaluation?(payload)
                    return selected.asArray(Float.self)
                }
                // All model-side handles above have unwound before either path
                // can close the step's charged roots or release its row pages.
                switch result {
                case .success:
                    caches.forEach { $0.setRows([]) }
                    work.closeGraph()
                case .failure:
                    backend.pool.endAttentionConstruction(work)
                    do {
                        try withError { fault in
                            work.constructionStream.synchronize()
                            Stream.cpu.synchronize()
                            try fault.check()
                        }
                        boundary.discardFailedGraphAfterSynchronization()
                        caches.forEach { $0.setRows([]) }
                        try work.discardAfterDrain()
                        work.closeGraph()
                        // A lazy clone may never have executed. Retire this
                        // failed branch; cursor-only rollback cannot repair it.
                        release(state)
                        backend.pool.writeValidation.clearAfterRetirement()
                    } catch {
                        work.failCompletion()
                        Issue.record("Failed-step cleanup failed: \(error)")
                    }
                }
                return try result.get()
            }
        }

        private func withMainPoolRows(
            executionPolicy: GLM5NextExecutionPolicy = .reference, segmentBytes: Int = 64 << 10,
            _ body: (GLM5NextTextModel, EngineV2, OwnedMainPoolRows) throws -> Void
        ) async throws {
            let model = try makeModel(executionPolicy: executionPolicy)
            let (engine, backend, process) = try pagedEngine(
                model, width: 2, segmentBytes: segmentBytes)
            let result = Result {
                try engine.loopForTesting.onEngineQueueSync {
                    engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
                    let rows = OwnedMainPoolRows(backend: backend)
                    defer {
                        rows.close()
                        #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
                        #expect(backend.pool.attentionWorkBytesReserved == 0)
                        engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
                    }
                    try body(model, engine, rows)
                }
            }
            await engine.shutdown()
            #expect(process.bytes == 0)
            try result.get()
        }

        @Test(arguments: [3, 4, 5, 15, 16, 17])
        func unifiedMainPoolCOWRetainsReadersAcrossTrimAndGrantShrink(boundary: Int) async throws {
            try await withMainPoolRows { model, engine, rows in
                let backend = rows.backend
                let source = try rows.makeState()
                let row = try #require(source[0] as? PagedSequenceKV)
                let ids = [Int32(0), Int32(boundary - 1), Int32(boundary), 1, 2, -1, -1]
                // The 17-token case must cross a page, not violate the real
                // backend's immutable 16-query ticket limit.
                for start in stride(from: 0, to: boundary, by: 16) {
                    _ = try rows.execute(
                        source, id: 1001, range: start ..< min(boundary, start + 16),
                        marker: 0, model: model, indices: ids)
                }
                var snapshot: PagedImmutableSnapshot? = try row.immutableSnapshot(through: boundary)
                let fork = try rows.makeState()
                try (fork[0] as! PagedSequenceKV).adoptImmutableSnapshot(try #require(snapshot))
                row.rollback(1)
                let reserved = backend.bytesReserved
                let copyBytes =
                    row.absoluteOffset % backend.pool.config.pageSize == 0
                    ? 0 : backend.pool.group(row.groupKey).pageBytes
                do {
                    defer {
                        backend.updateBytesCapacity(128 << 20)
                        engine.admissionForTesting.updateBytesCapacity(128 << 20)
                    }
                    _ = try rows.execute(
                        source, id: 1001, range: (boundary - 1) ..< boundary,
                        marker: 1000, model: model, indices: ids,
                        beforeEvaluation: { work in
                            #expect(backend.bytesReserved == reserved + copyBytes)
                            backend.updateBytesCapacity(1)
                            engine.admissionForTesting.updateBytesCapacity(1)
                            #expect(!work.completed && backend.pool.attentionWorkBytesReserved > 0)
                            #expect(throws: (any Error).self) {
                                try engine.admissionForTesting.reserveTransient(bytes: 1)
                            }
                        })
                    #expect(backend.bytesReserved == reserved)
                    #expect(throws: (any Error).self) {
                        try engine.admissionForTesting.reserveTransient(bytes: 1)
                    }
                }
                snapshot = nil
                let preserved = try rows.execute(
                    fork, id: 1002, range: boundary ..< (boundary + 1),
                    marker: 2000, model: model, indices: ids)
                #expect(preserved[0] == 0)
                #expect(preserved[2] == Float(boundary - 1))
                #expect(preserved[4] == 2000 + Float(boundary))
            }
        }

        private enum PagedStepTestError: Error, Equatable { case afterClone }

        @Test(arguments: [3, 17], [false, true])
        func unifiedMainPoolThrownCOWStepRetiresBranchAndPreservesImmutableFork(
            boundary: Int, afterPublication: Bool
        ) async throws {
            try await withMainPoolRows { model, _, rows in
                let backend = rows.backend
                let source = try rows.makeState()
                let sourceRow = try #require(source[0] as? PagedSequenceKV)
                let ids = [
                    Int32(0), Int32(boundary - 2), Int32(boundary - 1), Int32(boundary), 1, -1, -1,
                ]
                for start in stride(from: 0, to: boundary, by: 16) {
                    _ = try rows.execute(
                        source, id: 1101, range: start ..< min(boundary, start + 16),
                        marker: 0, model: model, indices: ids)
                }
                let snapshot = try sourceRow.immutableSnapshot(through: boundary)
                let fork = try rows.makeState()
                let forkRow = try #require(fork[0] as? PagedSequenceKV)
                try forkRow.adoptImmutableSnapshot(snapshot)
                // Both cases leave a shared partial frontier to clone.
                let rewind = boundary == 17 ? 2 : 1
                sourceRow.rollback(rewind)
                let logicalPage = sourceRow.table.count - 1
                let old = backend.pool.currentHandle(
                    group: sourceRow.groupKey, page: sourceRow.table[logicalPage])
                let references = try #require(backend.pool.refCount(for: old))
                let reserved = backend.bytesReserved
                let sourceBytes =
                    sourceRow.reservedPages * backend.pool.group(sourceRow.groupKey).pageBytes
                let pageBytes = backend.pool.group(sourceRow.groupKey).pageBytes
                var cloned = false
                let failAfterClone: () throws -> Void = {
                    // The real planner has prepaid the extra physical
                    // destination before the old page changes owners.
                    #expect(backend.bytesReserved == reserved + pageBytes)
                    #expect(backend.pool.refCount(for: old) == references)
                    #expect(sourceRow.table[logicalPage] != old.page)
                    cloned = true
                    throw PagedStepTestError.afterClone
                }
                #expect(throws: PagedStepTestError.afterClone) {
                    try rows.execute(
                        source, id: 1101, range: (boundary - rewind) ..< boundary,
                        marker: 1000, model: model, indices: ids,
                        afterPayload: afterPublication ? nil : failAfterClone,
                        beforeEvaluation: { work in
                            #expect(work.published && !work.completed)
                            if afterPublication { try failAfterClone() }
                        })
                }
                #expect(cloned && sourceRow.isReleased)
                #expect(!forkRow.isReleased && forkRow.absoluteOffset == boundary)
                #expect(backend.bytesReserved == reserved - sourceBytes)
                #expect(backend.pool.refCount(for: old) == references - 1)
                #expect(
                    backend.pool.activeAttentionWork == nil
                        && backend.pool.attentionWorkOwners.isEmpty)
                #expect(backend.pool.attentionWorkBytesReserved == 0)
                #expect(!backend.pool.writeValidation.isFaulted)

                // Force the failed branch's slot to be reusable, then prove
                // neither the abandoned graph nor its clone can alter the fork.
                let replacement = try rows.makeState()
                let replacementRow = try #require(replacement[0] as? PagedSequenceKV)
                #expect(replacementRow.serial != sourceRow.serial)
                _ = try rows.execute(
                    replacement, id: 1102, range: 0 ..< 1,
                    marker: 3000, model: model, indices: ids)
                let preserved = try rows.execute(
                    fork, id: 1103, range: boundary ..< (boundary + 1),
                    marker: 2000, model: model, indices: ids)
                #expect(preserved[0] == 0)
                #expect(preserved[2] == Float(boundary - 2))
                #expect(preserved[4] == Float(boundary - 1))
                #expect(preserved[6] == 2000 + Float(boundary))
                withExtendedLifetime(snapshot) {}
            }
        }

        @Test func unifiedMainPoolRejectedPlanAndForkAdoptionReleaseOwnedRows() async throws {
            try await withMainPoolRows { model, _, rows in
                let backend = rows.backend
                let source = try rows.makeState()
                let sourceRow = try #require(source[0] as? PagedSequenceKV)
                let ids: [Int32] = [0, 1, 2, 3, -1, -1, -1]
                _ = try rows.execute(
                    source, id: 1201, range: 0 ..< 3, marker: 0, model: model, indices: ids)
                let snapshot = try sourceRow.immutableSnapshot(through: 3)
                let tooShort = try rows.makeState(maxLength: 2)
                #expect(throws: PagedImmutableSnapshotError.ineligible) {
                    try (tooShort[0] as! PagedSequenceKV).adoptImmutableSnapshot(snapshot)
                }
                // The rejected candidate still owns its admitted empty row.
                // Retire it before requesting another slot in this B=2 pool.
                rows.release(tooShort)
                let fork = try rows.makeState()
                try (fork[0] as! PagedSequenceKV).adoptImmutableSnapshot(snapshot)
                let reserved = backend.bytesReserved
                #expect(throws: CBv2KVError.self) {
                    try rows.execute(
                        source, id: 1201, range: 3 ..< 20,
                        marker: 1000, model: model, indices: ids)
                }
                #expect(sourceRow.absoluteOffset == 3 && backend.bytesReserved == reserved)
                #expect(
                    backend.pool.activeAttentionWork == nil
                        && backend.pool.attentionWorkOwners.isEmpty)
                let preserved = try rows.execute(
                    fork, id: 1202, range: 3 ..< 4,
                    marker: 2000, model: model, indices: ids)
                #expect(preserved[0] == 0 && preserved[2] == 1 && preserved[4] == 2)
                #expect(preserved[6] == 2003)
                withExtendedLifetime(snapshot) {}
            }
        }

        /// Four tiny-model pages (448 bytes each): two usable pages per segment.
        private static let twoPageSegmentBytes = 4 * 16 * (2 + 5) * 4

        /// Both read forms must copy the same bytes. Each trace covers page and
        /// segment growth, then a copy-on-write fork of a shared partial page,
        /// with selected ids at, beyond and below the valid history.
        @Test(arguments: [false, true])
        func deviceTableMetadataReadsMatchHostRecords(twoPageSegments: Bool) async throws {
            let segmentBytes = twoPageSegments ? Self.twoPageSegmentBytes : 64 << 10
            var traces: [[[UInt32]]] = []
            for reads in [GLM5NextPagedMetadataReads.hostRecords, .deviceTable] {
                var trace: [[UInt32]] = []
                try await withMainPoolRows(
                    executionPolicy: .init(pagedMetadataReads: reads), segmentBytes: segmentBytes
                ) { model, _, rows in
                    #expect(rows.backend.glm5NextModelBinding?.metadataReads == reads)
                    func step(
                        _ state: [CBv2SequenceKV?], id: UInt64, range: Range<Int>, marker: Float
                    ) throws {
                        let values = MLXArray(
                            range.flatMap { token in
                                (0 ..< 5).map { marker + Float(token * 8 + $0) }
                            }
                        ).reshaped(1, 1, range.count, 5)
                        let length = Int32(range.upperBound)
                        let ids = [0, length - 1, length / 2, length, length + 17, -1, 15]
                        let selected = try rows.execute(
                            state, id: id, range: range, marker: marker, model: model,
                            indices: ids, values: values,
                            afterEvaluation: { payload in
                                trace.append(
                                    payload.values(row: 0).asArray(Float.self).map(\.bitPattern))
                            })
                        trace.append(selected.map(\.bitPattern))
                    }
                    let source = try rows.makeState(maxLength: 96)
                    for start in stride(from: 0, to: 70, by: 16) {
                        try step(source, id: 1301, range: start ..< min(70, start + 16), marker: 0)
                    }
                    let row = try #require(source[0] as? PagedSequenceKV)
                    let snapshot = try row.immutableSnapshot(through: 70)
                    let fork = try rows.makeState(maxLength: 96)
                    try (fork[0] as! PagedSequenceKV).adoptImmutableSnapshot(snapshot)
                    row.rollback(3)
                    try step(source, id: 1301, range: 67 ..< 75, marker: 1000)
                    try step(fork, id: 1302, range: 70 ..< 72, marker: 2000)
                    withExtendedLifetime(snapshot) {}
                }
                traces.append(trace)
            }
            #expect(traces.count == 2 && traces[0].count == 14)
            #expect(traces[0] == traces[1])
        }

        @Test(arguments: [1, 2])
        func deviceTableMetadataEngineMatchesContiguousTarget(width: Int) async throws {
            let model = try makeModel(executionPolicy: .init(pagedMetadataReads: .deviceTable))
            let prompts = [(0 ..< 37).map { $0 % 11 + 4 }, (0 ..< 21).map { $0 % 7 + 5 }]
                .prefix(width)
            let expected = try prompts.map { try greedy(model, prompt: $0, count: 6) }
            let (engine, backend, process) = try pagedEngine(
                model, width: width, segmentBytes: Self.twoPageSegmentBytes)
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
            }
            let streams = try prompts.enumerated().map { index, prompt in
                try engine.submit(
                    .init(
                        id: .init(UInt64(1401 + index)), promptTokens: prompt,
                        sampling: .init(temperature: 0), maxTokens: 6))
            }
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            var outputs: [[Int]] = []
            for stream in streams {
                var output: [Int] = []
                for await event in stream {
                    if case .delta(_, let tokens, _) = event { output += tokens }
                }
                outputs.append(output)
            }
            await engine.shutdown()
            #expect(outputs == expected)
            #expect(backend.glm5NextModelBinding?.metadataReads == .deviceTable)
            #expect((backend.glm5NextModelBinding?.observedMetadataRows ?? 0) > 0)
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
            #expect(process.bytes == 0)
        }

        /// The paged target serves whole-step KDA like its contiguous target.
        @Test(arguments: [1, 2])
        func wholeStepLinearAttentionPagedEngineMatchesContiguousTarget(width: Int) async throws {
            let model = try makeModel(executionPolicy: .init(linearAttention: .nativeRecurrence))
            let prompts = [(0 ..< 37).map { $0 % 11 + 4 }, (0 ..< 21).map { $0 % 7 + 5 }]
                .prefix(width)
            let expected = try prompts.map { try greedy(model, prompt: $0, count: 6) }
            let (engine, backend, process) = try pagedEngine(model, width: width)
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
            }
            let streams = try prompts.enumerated().map { index, prompt in
                try engine.submit(
                    .init(
                        id: .init(UInt64(1501 + index)), promptTokens: prompt,
                        sampling: .init(temperature: 0), maxTokens: 6))
            }
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            var outputs: [[Int]] = []
            for stream in streams {
                var output: [Int] = []
                for await event in stream {
                    if case .delta(_, let tokens, _) = event { output += tokens }
                }
                outputs.append(output)
            }
            await engine.shutdown()
            #expect(outputs == expected)
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
            #expect(process.bytes == 0)
        }

        /// The reference read is exactly one full selection row. Only an
        /// explicit four-aligned static-padding suffix of the token budget may
        /// shorten a read, by exactly its width; other widths are refused
        /// before any read is counted.
        @Test func selectedLatentReadWidthContracts() async throws {
            try await withMainPoolRows { model, _, rows in
                let source = try rows.makeState()
                // indexTopK 4: budget columns 0..<4, then three tail ids.
                let row: [Int32] = [0, 1, 2, 3, 4, 5, 6]
                func ids(_ values: [Int32]) -> MLXArray {
                    MLXArray(values).reshaped(1, values.count)
                }
                let selected = try rows.execute(
                    source, id: 1501, range: 0 ..< 7, marker: 0, model: model, indices: row,
                    read: { payload, full in
                        let tail = ids(Array(row[4...]))
                        #expect(throws: CBv2KVError.self) {
                            try payload.selectedKeys(row: 0, indices: tail)
                        }
                        for (padding, values) in [
                            (0 ..< 4, Array(row)), (2 ..< 4, Array(row[2...])),
                            (0 ..< 3, Array(row[3...])), (4 ..< 8, Array(row[..<3])),
                            (4 ..< 4, Array(row[4...])),
                        ] as [(Range<Int>, [Int32])] {
                            #expect(throws: CBv2KVError.self, "padding \(padding)") {
                                try payload.selectedKeys(
                                    row: 0, indices: ids(values), staticPadding: padding)
                            }
                        }
                        let live = try payload.selectedKeys(
                            row: 0, indices: tail, staticPadding: 0 ..< 4)
                        let unpadded = try payload.selectedKeys(
                            row: 0, indices: full, staticPadding: 4 ..< 4)
                        return concatenated(
                            [try payload.selectedKeys(row: 0, indices: full), unpadded, live],
                            axis: 1)
                    })
                // Each latent row is [marker + token, marker + token + 0.5].
                let expected = row.flatMap { [Float($0), Float($0) + 0.5] }
                #expect(Array(selected[0 ..< 14]) == expected)
                #expect(Array(selected[14 ..< 28]) == expected)
                #expect(Array(selected[28...]) == Array(expected[8...]))
                let binding = try #require(rows.backend.glm5NextModelBinding)
                #expect(binding.observedSelectedRows == 7 + 7 + 3)
            }
        }

        private typealias Assignment = (id: CBv2RequestID, range: Range<Int>)

        /// Real main-pool/EngineV2 construction, but manual step execution lets
        /// the test prove a rejected plan never reaches the target forward.
        private func withAggregatePlanner(
            batch: Int, latentWidth: Int = 512, sequenceLimit: Int = 64,
            _ body: (
                GLM5NextTextModel, PagedKVBackend, CBv2LayerCacheBank,
                [CBv2RequestID: [CBv2SequenceKV?]], [CBv2RequestID: CBv2RecurrentRequestState]
            ) throws -> Void
        ) async throws {
            let base = try GLM5NextInventoryFixture.configuration()
            var json = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
            var text = try #require(json["text_config"] as? [String: Any])
            text["kv_lora_rank"] = latentWidth
            text["index_head_dim"] = 128
            json["text_config"] = text
            let configuration = try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
            let model = try makeModel(configuration: configuration)
            // The engine prices its worst admitted step at setup, so its
            // envelope is the context these rows actually use.
            let query = min(128, sequenceLimit)
            let backend = try model.makeCBv2PagedBackend(
                kvBytesCapacity: 256 << 20, maximumBatchSize: batch, maximumQueryTokens: query,
                maximumContextTokens: sequenceLimit, maximumScratchBytes: 256 << 20,
                segmentBytes: 128 << 10, maximumBufferBytes: 128 << 10)
            let bank = CBv2LayerCacheBank(caches: backend.makeLayerCaches())
            let process = GLM5NextTestProcessBudget()
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds, backend: backend, cacheProvider: bank,
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: batch, maxBatchedTokensPerStep: query,
                    prefillChunkSize: min(64, query), maxWaiting: batch, enablePrefixCache: false),
                admissionConfig: model.cbv2NativeAdmissionConfig, processMemoryOwner: process)
            let result = Result {
                try engine.loopForTesting.onEngineQueueSync {
                    engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
                    var states: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
                    var recurrent: [CBv2RequestID: CBv2RecurrentRequestState] = [:]
                    var recurrenceLease: CBv2CheckpointReservation?
                    defer {
                        bank.releaseBoundRows()
                        let unfinished = Set(
                            backend.pool.attentionWorkOwners.values
                                .filter { !$0.released }.flatMap {
                                    $0.descriptors.keys.map(\.serial)
                                })
                        #expect(unfinished.isEmpty)
                        for state in states.values
                        where state.compactMap({ $0 as? PagedSequenceKV })
                            .allSatisfy({ !unfinished.contains($0.serial) })
                        {
                            backend.release(state)
                        }
                        if unfinished.isEmpty {
                            for state in recurrent.values {
                                state.discardPendingAfterSynchronization()
                                do { try state.release() } catch {
                                    Issue.record("Recurrent cleanup failed: \(error)")
                                }
                            }
                            recurrent.removeAll()
                            recurrenceLease?.release()
                        }
                        engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
                    }
                    let fixed = try model.cbv2RecurrentStateSpec.peakBytesPerRequest() * batch
                    recurrenceLease = try engine.admissionForTesting.reserveTransient(bytes: fixed)
                    for index in 0 ..< batch {
                        let id = CBv2RequestID(UInt64(index + 1))
                        states[id] = try backend.makeSequenceState(
                            layerKinds: model.cbv2LayerKinds, promptLength: 0,
                            maxLength: sequenceLimit)
                        recurrent[id] = try CBv2RecurrentRequestState(
                            spec: model.cbv2RecurrentStateSpec)
                    }
                    #expect(backend.pool.memoryAdmission === engine.admissionForTesting)
                    let boundary = CBv2PagedWriteBoundary(pool: backend.pool)
                    let outcome = Result { try body(model, backend, bank, states, recurrent) }
                    if case .failure = outcome {
                        // The caller's model locals have unwound. Preserve its
                        // error even if the required cleanup drain also fails.
                        do {
                            try withError { fault in
                                StreamOrDevice.default.stream.synchronize()
                                Stream.cpu.synchronize()
                                try fault.check()
                            }
                            boundary.discardFailedGraphAfterSynchronization()
                            bank.releaseBoundRows()
                            let pool = backend.pool
                            pool.endAttentionConstruction(pool.activeAttentionWork)
                            for work in Array(pool.attentionWorkOwners.values) {
                                try work.discardAfterDrain()
                                work.closeGraph()
                            }
                        } catch { Issue.record("Aggregate planner cleanup failed: \(error)") }
                    }
                    try outcome.get()
                }
            }
            await engine.shutdown()
            try result.get()
        }

        private func runPreparedPlan(
            _ work: CBv2PagedAttentionStepOwner, assignments: [Assignment], together: Bool,
            model: GLM5NextTextModel, backend: PagedKVBackend, bank: CBv2LayerCacheBank,
            states: [CBv2RequestID: [CBv2SequenceKV?]],
            recurrent: [CBv2RequestID: CBv2RecurrentRequestState]
        ) throws {
            var roots: [MLXArray] = []
            var evaluations: [CBv2RecurrentStateEvaluation] = []
            let groups = together ? [assignments] : assignments.map { [$0] }
            for group in groups {
                let width = group[0].range.count
                #expect(group.allSatisfy { $0.range.count == width })
                let current = try group.map { try recurrent[$0.id]!.bind() }
                let caches = bank.layerCaches(rowStates: group.map { states[$0.id]! })
                let tokens = MLXArray(
                    Array(repeating: Int32(4), count: group.count * width), [group.count, width])
                let output = try model.prepareCBv2Forward(
                    tokens: tokens,
                    caches: caches.map { $0 as! any KVCache }, recurrentState: current)
                roots += [output.logits, output.hidden]
                roots += try current.flatMap { try $0.evaluate() }
                evaluations += current
            }
            try work.seal()
            work.publish()
            backend.pool.endAttentionConstruction(work)
            try withError { fault in
                eval(roots + work.evaluationTargets)
                StreamOrDevice.default.stream.synchronize()
                try fault.check()
            }
            try work.finishEvaluation()
            for evaluation in evaluations { try evaluation.commit() }
            roots.removeAll()
            work.closeGraph()
        }

        @Test(arguments: [64, 65])
        func actualDecodePlannerChecksAggregateTransportBeforeModelCall(batch: Int) async throws {
            try await withAggregatePlanner(batch: batch, sequenceLimit: 2) {
                model, backend, bank, states, recurrent in
                let ids = states.keys.sorted { $0.raw < $1.raw }
                // Seed all rows through real forwards that individually fit the
                // same 128 KiB limit; the next plan is actual decode at offset 1.
                for start in stride(from: 0, to: ids.count, by: 64) {
                    let seed = ids[start ..< min(start + 64, ids.count)].map {
                        (id: $0, range: 0 ..< 1)
                    }
                    let preparedWork = try backend.prepareAttentionWork(
                        assignments: seed, states: states)
                    let work = try #require(preparedWork)
                    try runPreparedPlan(
                        work, assignments: seed, together: true,
                        model: model, backend: backend, bank: bank, states: states,
                        recurrent: recurrent)
                }
                let decode = ids.map { (id: $0, range: 1 ..< 2) }
                var modelReached = false
                if batch == 65 {
                    var rejected = false
                    do {
                        let preparedWork = try backend.prepareAttentionWork(
                            assignments: decode, states: states)
                        let work = try #require(preparedWork)
                        modelReached = true
                        // If the guard regresses, fail without attempting the
                        // known-oversized model allocation on the test machine.
                        backend.pool.endAttentionConstruction(work)
                        try work.discardAfterDrain()
                    } catch CBv2KVError.backendIneligible(let reason) {
                        #expect(
                            reason
                                == "GLM batched K transport at layer 0 exceeds native maximum buffer length"
                        )
                        rejected = true
                    }
                    #expect(rejected && !modelReached)
                    #expect(states.values.allSatisfy { $0.allSatisfy { $0?.absoluteOffset == 1 } })
                    #expect(
                        backend.pool.activeAttentionWork == nil
                            && backend.pool.attentionWorkOwners.isEmpty)
                } else {
                    let preparedWork = try backend.prepareAttentionWork(
                        assignments: decode, states: states)
                    let work = try #require(preparedWork)
                    let key = try #require(work.allocations.first { $0.role == "write.batchK.0" })
                    #expect(key.logicalBytes == 131_072 && key.allocationCount == 1)
                    #expect(
                        key.upperBoundBytes
                            == Memory.allocationFootprintPolicy()?.upperBound(
                                byteCount: key.logicalBytes))
                    #expect(
                        work.reservedBytes == (64 << 10)
                            + work.allocations.reduce(0) { $0 + $1.upperBoundBytes })
                    try runPreparedPlan(
                        work, assignments: decode, together: true,
                        model: model, backend: backend, bank: bank, states: states,
                        recurrent: recurrent)
                    #expect(states.values.allSatisfy { $0.allSatisfy { $0?.absoluteOffset == 2 } })
                }
            }
        }

        @Test(arguments: [0, 1, 2])
        func actualPrefillPlannerBoundsRectangularAndRaggedTransport(caseIndex: Int) async throws {
            let lengths = caseIndex == 0 ? [32, 32] : caseIndex == 1 ? [32, 1] : [33, 1]
            try await withAggregatePlanner(batch: 2) { model, backend, bank, states, recurrent in
                let ids = states.keys.sorted { $0.raw < $1.raw }
                let assignments = zip(ids, lengths).map { (id: $0.0, range: 0 ..< $0.1) }
                var modelReached = false
                if caseIndex == 2 {
                    var rejected = false
                    do {
                        let preparedWork = try backend.prepareAttentionWork(
                            assignments: assignments, states: states)
                        let work = try #require(preparedWork)
                        modelReached = true
                        backend.pool.endAttentionConstruction(work)
                        try work.discardAfterDrain()
                    } catch CBv2KVError.backendIneligible(let reason) {
                        #expect(
                            reason
                                == "GLM batched K transport at layer 0 exceeds native maximum buffer length"
                        )
                        rejected = true
                    }
                    #expect(rejected && !modelReached)
                    #expect(states.values.allSatisfy { $0.allSatisfy { $0?.absoluteOffset == 0 } })
                    #expect(
                        backend.pool.activeAttentionWork == nil
                            && backend.pool.attentionWorkOwners.isEmpty)
                } else {
                    let preparedWork = try backend.prepareAttentionWork(
                        assignments: assignments, states: states)
                    let work = try #require(preparedWork)
                    let keys = work.allocations.filter { $0.role == "write.batchK.0" }
                    let values = work.allocations.filter { $0.role == "write.batchV.0" }
                    #expect(keys.count == 1 && keys[0].logicalBytes == 131_072)
                    #expect(values.count == 1 && values[0].logicalBytes == 65_792)
                    #expect(keys[0].allocationCount == 1 && values[0].allocationCount == 1)
                    #expect(
                        work.reservedBytes == (64 << 10)
                            + work.allocations.reduce(0) { $0 + $1.upperBoundBytes })
                    try runPreparedPlan(
                        work, assignments: assignments, together: caseIndex == 0,
                        model: model, backend: backend, bank: bank, states: states,
                        recurrent: recurrent)
                }
            }
        }

        /// Step pricing follows the selector's actual work: no pooled keys or
        /// scores inside the exact dense range, one bounded score block beyond
        /// it, and selected-latent indices once per query (not per page).
        @Test func stepPricingFollowsDenseShortcutAndScoreBlocks() async throws {
            try await withAggregatePlanner(batch: 1, latentWidth: 32, sequenceLimit: 64) {
                model, backend, _, states, _ in
                let id = try #require(states.keys.first)
                func roles(_ range: Range<Int>) throws -> [String: CBv2PagedAttentionAllocation] {
                    let prepared = try backend.prepareAttentionWork(
                        assignments: [(id: id, range: range)], states: states)
                    let work = try #require(prepared)
                    defer {
                        backend.pool.endAttentionConstruction(work)
                        do { try work.discardAfterDrain() } catch {
                            Issue.record("unexpected discard failure: \(error)")
                        }
                    }
                    return Dictionary(
                        work.allocations.filter { $0.role.hasPrefix("index.") }
                            .map { ($0.role, $0) }, uniquingKeysWith: { first, _ in first })
                        .merging(
                            work.allocations.filter { $0.role == "selected.indices" }
                                .map { ($0.role, $0) }, uniquingKeysWith: { first, _ in first })
                }
                let c = model.configuration
                // topK 4 covers one pool: four tokens take the dense shortcut.
                let dense = try roles(0 ..< 4)
                #expect(dense["index.score"] == nil && dense["index.castAndPool"] == nil)
                #expect(dense["index.denseEligibility"] != nil)
                #expect(dense["selected.indices"]?.allocationCount == 4)
                let scored = try roles(0 ..< 48)
                let pools = 12
                let block = min(48, GLM5NextSparseAttention.scoreBlockBytes / (c.indexNHeads * pools * 4))
                let score = try #require(scored["index.score"])
                #expect(score.maximumLogicalBufferBytes == block * c.indexNHeads * pools * 4)
                #expect(score.allocationCount == 2)
                #expect(scored["index.castAndPool"]?.allocationCount == 6)
                #expect(scored["selected.indices"]?.allocationCount == 48)
                #expect(states.values.allSatisfy { $0.allSatisfy { $0?.absoluteOffset == 0 } })
            }
        }

        /// The worst admitted step is priced once when the engine is built; an
        /// envelope that cannot fit is refused before any request is accepted.
        @Test(arguments: [false, true])
        func unservableStepEnvelopeIsRefusedAtEngineSetup(fits: Bool) async throws {
            let model = try makeModel()
            let backend = try model.makeCBv2PagedBackend(
                kvBytesCapacity: 128 << 20, maximumBatchSize: 2, maximumQueryTokens: 64,
                maximumContextTokens: 128, maximumScratchBytes: fits ? 256 << 20 : 1 << 20,
                segmentBytes: 64 << 10, maximumBufferBytes: 32 << 20)
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 64,
                    prefillChunkSize: 16, maxWaiting: 2, enablePrefixCache: false),
                admissionConfig: model.cbv2NativeAdmissionConfig,
                processMemoryOwner: GLM5NextTestProcessBudget())
            let binding = try #require(backend.glm5NextModelBinding)
            if fits {
                #expect(engine.pagedAttentionWorkInactiveReason == nil)
                let required = try binding.validateStepEnvelope(
                    maximumRequests: 2, maximumStepTokens: 64, maximumRowTokens: 16,
                    admissibleBytes: nil)
                #expect(required > 0 && required <= 256 << 20)
            } else {
                #expect(
                    engine.pagedAttentionWorkInactiveReason?.hasPrefix(
                        "GLM worst-case step scratch exceeds the engine envelope") == true)
                #expect(throws: CBv2KVError.self) {
                    try binding.validateStepEnvelope(
                        maximumRequests: 2, maximumStepTokens: 64, maximumRowTokens: 16,
                        admissibleBytes: nil)
                }
                #expect(throws: CBv2KVError.self) {
                    _ = try engine.submit(
                        .init(
                            id: .init(1), promptTokens: [4, 5, 6],
                            sampling: .init(temperature: 0), maxTokens: 1))
                }
            }
            await engine.shutdown()
        }

        @Test func actualPlannerAlsoRejectsOversizedAggregateValues() async throws {
            try await withAggregatePlanner(batch: 128, latentWidth: 32, sequenceLimit: 1) {
                _, backend, _, states, _ in
                let assignments = states.keys.map { (id: $0, range: 0 ..< 1) }
                var rejected = false
                do {
                    if let work = try backend.prepareAttentionWork(
                        assignments: assignments, states: states)
                    {
                        backend.pool.endAttentionConstruction(work)
                        try work.discardAfterDrain()
                    }
                } catch CBv2KVError.backendIneligible(let reason) {
                    #expect(
                        reason
                            == "GLM batched V transport at layer 0 exceeds native maximum buffer length"
                    )
                    rejected = true
                }
                #expect(rejected)
                #expect(
                    backend.pool.activeAttentionWork == nil
                        && backend.pool.attentionWorkOwners.isEmpty)
                #expect(states.values.allSatisfy { $0.allSatisfy { $0?.absoluteOffset == 0 } })
            }
        }
    }
}
