import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Serial target-token transactions only. This does not execute an MTP head,
/// capture a multi-token recurrent window, or qualify paged/native speculation.
@Suite(.tags(.integration))
struct NemotronSuperRollbackTests {
    private func component() throws -> NemotronSuperModel {
        let configuration = NemotronHConfiguration(
            vocabSize: 32, hiddenSize: 64,
            numHiddenLayers: 2, numAttentionHeads: 2, numKeyValueHeads: 1,
            mambaNumHeads: 4, mambaHeadDim: 16, ssmStateSize: 16,
            convKernel: 4, nGroups: 2, intermediateSize: 128,
            moeIntermediateSize: 64, moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4, numExpertsPerTok: 2, hybridOverridePattern: "M*", headDim: 64)
        let dimensions = NemotronRadioDimensions(
            width: 2, layers: 1, heads: 1,
            channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)
        let images = NemotronSuperImageComposition(
            tower: NemotronRadio(dimensions),
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: 64),
            width: 64, imageTokenID: 18)
        return try NemotronSuperModel(
            languageModel: NemotronHModel(configuration), imageComposition: images)
    }

    private final class Storage {
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 16 << 20, kvDType: .float32))
        let bank: CBv2LayerCacheBank
        init(_ model: NemotronSuperModel) {
            bank = CBv2LayerCacheBank(
                caches: model.languageModel.newCacheV2 {
                    CBv2LayerCache(layerIndex: $0, kind: $1)
                })
        }
    }

    private final class Request {
        let rows: [CBv2SequenceKV?]
        let state: CBv2RecurrentRequestState
        init(_ model: NemotronSuperModel, storage: Storage, promptCount: Int) throws {
            rows = try storage.backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds,
                promptLength: promptCount, maxLength: 32)
            state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        }
    }

    private struct Snapshot {
        let offsets: [Int]
        let tensors: [MLXArray]
    }

    private func caches(_ request: Request, _ storage: Storage) throws -> [KVCache] {
        try storage.bank.layerCaches(rowStates: [request.rows]).map { try #require($0 as? KVCache) }
    }

    private func confirmed(_ request: Request) throws -> [MLXArray] {
        let snapshot = try #require(request.state.confirmedStateSnapshot())
        let arrays = try request.state.spec.layers.flatMap { spec -> [MLXArray] in
            let layer = try #require(snapshot[spec.modelLayerIndex])
            return [try #require(layer.conv) + 0, try #require(layer.ssm) + 0]
        }
        eval(arrays)
        return arrays
    }

    private func snapshot(_ request: Request) throws -> Snapshot {
        var tensors = try confirmed(request)
        var offsets: [Int] = []
        for row in request.rows.compactMap({ $0 }) {
            let kv = row.snapshot()
            offsets.append(kv.offset)
            tensors += [kv.keys + 0, kv.values + 0]
        }
        eval(tensors)
        return Snapshot(offsets: offsets, tensors: tensors)
    }

    private func exact(_ lhs: [MLXArray], _ rhs: [MLXArray]) {
        #expect(lhs.count == rhs.count)
        for (a, b) in zip(lhs, rhs) {
            #expect(a.shape == b.shape && a.dtype == b.dtype)
            #expect(arrayEqual(a, b).item(Bool.self))
        }
    }

    private func exact(_ lhs: Snapshot, _ rhs: Snapshot) {
        #expect(lhs.offsets == rhs.offsets)
        exact(lhs.tensors, rhs.tensors)
    }

    private func prefill(
        _ model: NemotronSuperModel, _ prepared: NemotronSuperModel.PreparedInput,
        request: Request, storage: Storage
    ) throws -> MLXArray {
        let cache = try caches(request, storage)
        let binding = try request.state.bind()
        let logits = NemotronSuperRecurrentAdapter(model).cbv2RecurrentPrefill(
            MLXArray(prepared.prompt).reshaped(1, prepared.prompt.count),
            inputEmbedding: prepared.embeddings, cache: cache,
            recurrentState: [binding], positionIds: nil, requirement: .lastPositionLogits)
        eval([logits] + (try binding.evaluate()) + cache.flatMap { $0.innerState() })
        try binding.commit()
        return logits
    }

    private func step(
        _ model: NemotronSuperModel, token: Int32, request: Request,
        storage: Storage, commit: Bool
    ) throws -> (MLXArray, CBv2RecurrentStateEvaluation) {
        let cache = try caches(request, storage)
        let binding = try request.state.bind()
        let logits = NemotronSuperRecurrentAdapter(model).cbv2Forward(
            MLXArray([token]).reshaped(1, 1), caches: cache, recurrentState: [binding])
        // evaluate() alone only stages metadata; execute all graph roots before finalization.
        eval([logits] + (try binding.evaluate()) + cache.flatMap { $0.innerState() })
        if commit { try binding.commit() }
        return (logits, binding)
    }

    @Test(arguments: [0, 1, 3])
    func serialAcceptedPrefixRollbackAndCorrectionPreserveHybridState(keep: Int) throws {
        let model = try component()
        let storage = Storage(model)
        // Include a visual span in the committed prefix; this does not test feature-cache reuse.
        let pixels = MLXArray((0 ..< 16).map { Float($0) / 16 }).reshaped(1, 1, 4, 4)
        let prefix = try model.prepare(prompt: [1, 18, 2], normalizedImages: [pixels])
        let unrelatedPrefix = try model.prepare(prompt: [10, 12])
        let actual = try Request(model, storage: storage, promptCount: prefix.prompt.count)
        let oracle = try Request(model, storage: storage, promptCount: prefix.prompt.count)
        let unrelated = try Request(
            model, storage: storage, promptCount: unrelatedPrefix.prompt.count)
        let unrelatedOracle = try Request(
            model, storage: storage, promptCount: unrelatedPrefix.prompt.count)
        let requests = [actual, oracle, unrelated, unrelatedOracle]
        defer {
            storage.bank.releaseBoundRows()
            for request in requests { storage.backend.release(request.rows) }
        }
        let baselineLogits = try prefill(model, prefix, request: actual, storage: storage)
        var oracleLogits = try prefill(model, prefix, request: oracle, storage: storage)
        _ = try prefill(model, unrelatedPrefix, request: unrelated, storage: storage)
        _ = try prefill(model, unrelatedPrefix, request: unrelatedOracle, storage: storage)
        exact([baselineLogits], [oracleLogits])
        let committedBefore = try confirmed(actual)
        let candidates: [Int32] = [3, 4, 5]
        for row in actual.rows.compactMap({ $0 }) { row.beginSpeculativeWrite() }
        var pending: [CBv2RecurrentStateEvaluation] = []
        var proposedLogits: [MLXArray] = []
        for token in candidates {
            let (logits, transaction) = try step(
                model, token: token, request: actual,
                storage: storage, commit: false)
            pending.append(transaction)
            proposedLogits.append(logits)
            // Pending generations must never leak into the checkpoint boundary.
            exact(try confirmed(actual), committedBefore)
        }
        let latest = try #require(actual.state.state(modelLayerIndex: 0))
        let latestConv = try #require(latest.conv)
        #expect(!arrayEqual(latestConv, committedBefore[0]).item(Bool.self))
        // Advance another request while actual has three unresolved generations.
        let unrelatedLogits = try step(
            model, token: 11, request: unrelated, storage: storage, commit: true
        ).0
        let unrelatedExpected = try step(
            model, token: 11, request: unrelatedOracle, storage: storage, commit: true
        ).0
        exact([unrelatedLogits], [unrelatedExpected])
        let unrelatedBefore = try snapshot(unrelated)
        for index in 0 ..< keep {
            oracleLogits = try step(
                model, token: candidates[index], request: oracle,
                storage: storage, commit: true
            ).0
            try pending[index].commit()
        }
        // The confirmed state is the accepted prefix even while rejected successors remain visible.
        exact(try confirmed(actual), try confirmed(oracle))
        for index in (keep ..< candidates.count).reversed() { try pending[index].rollback() }
        for row in actual.rows.compactMap({ $0 }) {
            row.rollback(candidates.count - keep)
            row.commitSpeculativeWrite()
        }
        // KV host offsets changed outside the bank's caches; discard its offset fingerprint.
        storage.bank.invalidateBoundComposition()
        exact(try snapshot(actual), try snapshot(oracle))
        exact(try snapshot(unrelated), unrelatedBefore)
        exact([keep == 0 ? baselineLogits : proposedLogits[keep - 1]], [oracleLogits])
        let correction = try step(model, token: 9, request: actual, storage: storage, commit: true)
            .0
        let correctionExpected = try step(
            model, token: 9, request: oracle, storage: storage, commit: true
        ).0
        exact([correction], [correctionExpected])
        exact(try snapshot(actual), try snapshot(oracle))
        // Rejoining the untouched request must also retain its own decode history.
        let resumed = try step(model, token: 13, request: unrelated, storage: storage, commit: true)
            .0
        let resumedExpected = try step(
            model, token: 13, request: unrelatedOracle, storage: storage, commit: true
        ).0
        exact([resumed], [resumedExpected])
        exact(try snapshot(unrelated), try snapshot(unrelatedOracle))
        for request in requests { try request.state.release() }
    }

    @Test func adapterFullAndChunkedCausalImagesMatchComposite() throws {
        let model = try component()
        let adapter = NemotronSuperRecurrentAdapter(model)
        let storage = Storage(model)
        let pixels = MLXArray((0 ..< 32).map { Float($0) / 32 }).reshaped(1, 1, 4, 8)
        let prepared = try model.prepare(prompt: [1, 18, 18, 2], normalizedImages: [pixels])
        let reference = try Request(model, storage: storage, promptCount: 4)
        let full = try Request(model, storage: storage, promptCount: 4)
        let chunked = try Request(model, storage: storage, promptCount: 4)
        defer {
            storage.bank.releaseBoundRows()
            for request in [reference, full, chunked] { storage.backend.release(request.rows) }
        }
        let referenceCache = try caches(reference, storage)
        let referenceBinding = try reference.state.bind()
        let expected = try model.prefill(
            prepared, cache: referenceCache,
            recurrentState: [referenceBinding], requirement: .lastPositionLogits)
        eval(
            [expected] + (try referenceBinding.evaluate())
                + referenceCache.flatMap { $0.innerState() })
        try referenceBinding.commit()

        let fullCache = try caches(full, storage)
        let fullBinding = try full.state.bind()
        let fullLogits = adapter.embeddingForward(
            MLXArray(prepared.prompt).reshaped(1, 4),
            inputEmbedding: prepared.embeddings, cache: fullCache,
            recurrentState: [fullBinding], positionIds: nil)
        eval([fullLogits] + (try fullBinding.evaluate()) + fullCache.flatMap { $0.innerState() })
        try fullBinding.commit()
        #expect(fullLogits.shape == [1, 4, 32])
        // Full-position GEMM and narrowed frontier GEMV have different reduction
        // orders. State is still compared bit-for-bit below; logits use FP32 tolerance.
        let frontier = fullLogits[0..., 3, 0...]
        print(
            "Super full/narrow logits max absolute delta: \(abs(frontier - expected).max().item(Float.self))"
        )
        #expect(allClose(frontier, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        exact(try snapshot(full), try snapshot(reference))

        // Split inside the visual span; each chunk advances the same causal state.
        for range in [0 ..< 2, 2 ..< 4] {
            let cache = try caches(chunked, storage)
            let binding = try chunked.state.bind()
            let logits = adapter.cbv2RecurrentPrefill(
                MLXArray(Array(prepared.prompt[range])).reshaped(1, range.count),
                inputEmbedding: prepared.embeddings[0..., range, 0...], cache: cache,
                recurrentState: [binding], positionIds: nil,
                requirement: range.upperBound == 4 ? .lastPositionLogits : .evaluationOnly)
            eval([logits] + (try binding.evaluate()) + cache.flatMap { $0.innerState() })
            try binding.commit()
            if range.upperBound == 4 {
                #expect(allClose(logits, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
            }
        }
        let actualState = try snapshot(chunked)
        let referenceState = try snapshot(reference)
        #expect(actualState.offsets == referenceState.offsets)
        for (actual, expected) in zip(actualState.tensors, referenceState.tensors) {
            #expect(allClose(actual, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
        for request in [reference, full, chunked] { try request.state.release() }
    }
}
